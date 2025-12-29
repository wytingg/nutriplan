#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_usda_align_plus.py — 食材→USDA/FNDDS 对齐（增强版）

核心改进（与此前讨论的“七点改进 + 匹配率提升策略”一致）：
1) 规范化与同义词表（B 表）统一：复数还原、修饰词剥离、别名映射（FoodOn/USDA 同义词可并入）。
2) 多路召回：倒排词典 + RapidFuzz + （可选）Sentence-Transformers 向量召回（FAISS 可选）。
3) 候选扩容与阈值自动调度：topK、min_fuzz 自适应，并为高频食材额外扩容。
4) 重排序打分器：fuzz + embed_sim + dtype_prior + 规则一致性（单位/形态），自动归一化与加权。
5) A 表（fdc_id × unit → grams）+ FNDDS household weights 融合：用于克重回填与一致性校验。
6) 动态默认克重：按品类（蔬菜/调味/液体/蛋白等）给出更合理的 default_grams 覆盖。
7) 错误感知迭代：输出 unmatched_topfreq.csv 与对齐审计 quick-report，支持下一轮 B 表增量学习。

输入：
  --ingredients_csv       解析后的配料表（长表，支持CSV或Parquet格式，至少包含 ingredient_norm / ingredient_raw；可带 quantity_* 列）
  --usda_dir              USDA FDC 解压目录（含 food.csv, food_nutrient.csv, ... 必要最少：food.csv）
  --fndds_household_csv   可选，FNDDS household weight 表
  --A_table_csv           可选，A 表（fdc_id, unit, grams_per_unit, ...）
  --B_table_csv           可选，B 表（term,synonym）或（src_term,target_term）

输出：
  --out_dir 下生成：
    aligned.parquet                    对齐结果（每条配料选出的 fdc_id + 打分细节）
    aligned_best.parquet               每条配料的最佳配对（去重聚合）
    unmatched_topfreq.csv              未匹配项的频次清单（用于补表）
    audit_summary.txt                  审计与指标概览

依赖（可选优雅退化）：
  pandas, numpy, rapidfuzz, (sentence_transformers, faiss-cpu)

用法示例：
python work/recipebench/scripts/rawdataprocess/step2_embedding_generation.py \
    --ingredients_csv work/recipebench/data/4out/ingredients_processed.parquet \
    --usda_dir work/recipebench/data/raw/usda \
    --A_table_csv work/recipebench/data/3out/household_weights_A.csv \
    --out_dir work/recipebench/data/4out/step2 \
    --embed_model sentence-transformers/all-MiniLM-L6-v2 \
    --min_fuzz 72 --topk_lex 80 --topk_embed 120 \
    --fast_precise_mode

python work/recipebench/scripts/rawdataprocess/step2_embedding_generation.py \
  --ingredients_csv work/recipebench/data/5guard/ingredients_labeled.parquet \
  --usda_dir work/recipebench/data/raw/usda \
  --A_table_csv work/recipebench/data/3out/household_weights_A1.csv \
  --out_dir work/recipebench/data/6aligned \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --min_fuzz 72 --topk_lex 80 --topk_embed 120 --fast_precise_mode \
  --filter_recipe_ids work/recipebench/data/5guard/hightrust_recipe_ids.parquet

作者：YourName (2025-09-14)
"""

from __future__ import annotations
import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# --- 依赖优雅降级 ---
try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

try:
    from sentence_transformers import SentenceTransformer
    from numpy.linalg import norm
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import faiss  # 可选
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# -------------------- 实用函数 --------------------
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
STOP_MODIFIERS = set([
    # 常见修饰词（可根据语料扩充）
    "fresh", "organic", "large", "small", "medium", "ripe", "peeled", "seeded",
    "freshly", "ground", "minced", "chopped", "sliced", "diced", "grated", "optional",
    "divided", "to", "taste", "room", "temperature", "unsalted", "salted",
])

PLURAL_RULES = [
    (re.compile(r"(.*)ies$"), r"\1y"),
    (re.compile(r"(.*)oes$"), r"\1o"),
    (re.compile(r"(.*)ses$"), r"\1s"),
    (re.compile(r"(.*)s$"), r"\1"),
]

CATEGORY_KEYWORDS = {
    # 用于 dtype_prior 与 default grams 策略
    "spice": ["spice", "seasoning", "powder", "ground"],
    "herb": ["herb", "basil", "cilantro", "parsley", "mint", "rosemary", "thyme"],
    "veg": ["vegetable", "tomato", "onion", "pepper", "carrot", "celery", "broccoli"],
    "meat": ["beef", "pork", "chicken", "turkey", "lamb", "bacon", "ham"],
    "egg": ["egg"],
    "dairy": ["milk", "cheese", "cream", "butter", "yogurt"],
    "liquid": ["sauce", "stock", "broth", "oil", "vinegar", "wine", "water"],
}

DEFAULT_GRAMS_BY_CAT = {
    "spice": 6.0,
    "herb": 5.0,
    "veg": 90.0,
    "meat": 120.0,
    "egg": 50.0,
    "dairy": 30.0,
    "liquid": 20.0,
    "other": 30.0,
}


def normalize_term(term: str) -> str:
    if pd.isna(term):
        return ""
    t = term.strip().lower()
    t = re.sub(r"\(.*?\)", " ", t)  # 去括号内容
    t = re.sub(r"\s+", " ", t)
    toks = [w for w in TOKEN_SPLIT_RE.split(t) if w]
    toks2 = []
    for w in toks:
        if w in STOP_MODIFIERS:
            continue
        # 复数还原
        w2 = w
        for pat, rep in PLURAL_RULES:
            if pat.fullmatch(w2):
                w2 = pat.sub(rep, w2)
                break
        toks2.append(w2)
    return " ".join(toks2)


def build_syn_map(B_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    syn_map = {}
    if B_df is None or B_df.empty:
        return syn_map
    cols = {c.lower(): c for c in B_df.columns}
    # 支持 (term, synonym) 或 (src_term, target_term)
    if "term" in cols and "synonym" in cols:
        for _, r in B_df.iterrows():
            a = str(r[cols["term"]]).strip().lower()
            b = str(r[cols["synonym"]]).strip().lower()
            if a and b:
                syn_map[a] = b
    elif "src_term" in cols and "target_term" in cols:
        for _, r in B_df.iterrows():
            a = str(r[cols["src_term"]]).strip().lower()
            b = str(r[cols["target_term"]]).strip().lower()
            if a and b:
                syn_map[a] = b
    return syn_map


def apply_synonym(s: str, syn_map: Dict[str, str]) -> str:
    if not s:
        return s
    if s in syn_map:
        return syn_map[s]
    # token 级别替换（粗略）
    toks = s.split()
    toks = [syn_map.get(w, w) for w in toks]
    return " ".join(toks)


# -------------------- USDA 索引 --------------------

def load_usda_foods(usda_dir: str) -> pd.DataFrame:
    food_csv = os.path.join(usda_dir, "food.csv")
    if not os.path.exists(food_csv):
        raise FileNotFoundError(f"USDA food.csv not found: {food_csv}")
    df = pd.read_csv(food_csv)
    # 兼容常见列名
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("fdc_id", None)
    desc_col = cols.get("description", None)
    cat_col = cols.get("food_category_id", None)
    if not id_col or not desc_col:
        raise ValueError("food.csv must contain fdc_id and description")
    out = df[[id_col, desc_col] + ([cat_col] if cat_col else [])].copy()
    out.columns = ["fdc_id", "description"] + (["food_category_id"] if cat_col else [])
    out["desc_norm"] = out["description"].astype(str).str.lower().map(normalize_term)
    return out


def make_inverted_index(food_df: pd.DataFrame) -> Dict[str, set]:
    inv = defaultdict(set)
    for i, r in food_df.iterrows():
        fid = int(r["fdc_id"]) if not pd.isna(r["fdc_id"]) else None
        if fid is None:
            continue
        for tok in set(r["desc_norm"].split()):
            if tok:
                inv[tok].add(fid)
    return inv


# -------------------- 向量召回（可选） --------------------
class EmbedSearcher:
    def __init__(self, model_name: str, food_df: pd.DataFrame, use_faiss: bool = True):
        if not _HAS_ST:
            raise RuntimeError("sentence_transformers not available")
        
        print(f"🔄 正在加载向量模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        self.food_df = food_df
        self.food_texts = food_df["desc_norm"].fillna("").astype(str).tolist()
        
        print(f"🔄 正在为 {len(self.food_texts)} 个食物描述生成向量嵌入...")
        print(f"   这可能需要几分钟时间，请耐心等待...")
        
        # 根据数据量动态调整batch_size
        if len(self.food_texts) > 50000:
            batch_size = 64  # 大数据集使用更小的batch
        elif len(self.food_texts) > 10000:
            batch_size = 128
        else:
            batch_size = 256
        
        print(f"   使用batch_size: {batch_size}")
        
        # 使用动态batch_size来平衡速度和内存
        self.mat = self.model.encode(
            self.food_texts, 
            batch_size=batch_size,
            show_progress_bar=True, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        print(f"✅ 向量嵌入生成完成，形状: {self.mat.shape}")
        
        self.use_faiss = use_faiss and _HAS_FAISS
        if self.use_faiss:
            print(f"🔄 正在构建FAISS索引...")
            d = self.mat.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.mat.astype(np.float32))
            print(f"✅ FAISS索引构建完成")
        else:
            self.index = None

    def search(self, query: str, topk: int = 50) -> List[Tuple[int, float]]:
        if not query:
            return []
        qv = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if self.use_faiss and self.index is not None:
            D, I = self.index.search(qv.astype(np.float32), topk)
            return [(int(ix), float(sc)) for ix, sc in zip(I[0], D[0]) if ix >= 0]
        # 退化版：全量余弦
        sims = (self.mat @ qv[0]).astype(float)
        idx = np.argpartition(-sims, min(topk, len(sims)-1))[:topk]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]


# -------------------- 重排序打分 --------------------

def safe_ratio(a: str, b: str) -> float:
    if not _HAS_RAPIDFUZZ:
        return 0.0
    return float(fuzz.token_set_ratio(a, b))


def category_prior(desc_norm: str) -> Dict[str, float]:
    s = desc_norm
    scores = {k: 0.0 for k in DEFAULT_GRAMS_BY_CAT}
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in s:
                scores[cat] += 1.0
    # 归一化
    tot = sum(scores.values())
    if tot <= 0:
        scores = {k: (1.0 if k == "other" else 0.0) for k in scores}
    else:
        scores = {k: v / tot for k, v in scores.items()}
    return scores


def choose_default_grams(prior: Dict[str, float]) -> float:
    val = 0.0
    for cat, w in prior.items():
        val += w * DEFAULT_GRAMS_BY_CAT.get(cat, DEFAULT_GRAMS_BY_CAT["other"])
    return float(val) if val > 0 else DEFAULT_GRAMS_BY_CAT["other"]


def rerank_score(fuzz_ratio: float, embed_sim: float, dtype_bonus: float, rule_bonus: float,
                 w_fuzz=0.45, w_embed=0.35, w_dtype=0.15, w_rule=0.05) -> float:
    # 所有项均为 0..100 或 0..1 需归一
    f = np.clip(fuzz_ratio / 100.0, 0, 1)
    e = np.clip(embed_sim, 0, 1)
    d = np.clip(dtype_bonus, 0, 1)
    r = np.clip(rule_bonus, 0, 1)
    return float(w_fuzz * f + w_embed * e + w_dtype * d + w_rule * r)


# -------------------- A 表 & FNDDS Household 融合 --------------------

def load_A_table(A_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if not A_csv or not os.path.exists(A_csv):
        return None
    A = pd.read_csv(A_csv)
    low = {c.lower(): c for c in A.columns}
    need = ["fdc_id", "unit", "grams_per_unit"]
    for n in need:
        if n not in low:
            raise ValueError("A_table_csv 缺少列：fdc_id, unit, grams_per_unit")
    A = A[[low["fdc_id"], low["unit"], low["grams_per_unit"]]].copy()
    A.columns = ["fdc_id", "unit", "grams_per_unit"]
    # 强制类型
    A["fdc_id"] = pd.to_numeric(A["fdc_id"], errors="coerce").astype("Int64")
    A["grams_per_unit"] = pd.to_numeric(A["grams_per_unit"], errors="coerce")
    A = A.dropna(subset=["fdc_id", "unit", "grams_per_unit"]).reset_index(drop=True)
    A["unit_norm"] = A["unit"].astype(str).str.lower().map(normalize_term)
    return A


def load_fndds_household(csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not csv_path or not os.path.exists(csv_path):
        return None
    H = pd.read_csv(csv_path)
    low = {c.lower(): c for c in H.columns}
    # 尽量兼容常见列
    cand_cols = [
        ("fdc_id", "unit", "grams"),
        ("fdc_id", "household_unit", "grams"),
        ("fdc_id", "household_measure", "gram_weight"),
    ]
    match = None
    for cols in cand_cols:
        if all(c in low for c in cols):
            match = cols
            break
    if match is None:
        # 不强制
        return None
    cols = [low[c] for c in match]
    H = H[cols].copy()
    H.columns = ["fdc_id", "unit", "grams_per_unit"]
    H["fdc_id"] = pd.to_numeric(H["fdc_id"], errors="coerce").astype("Int64")
    H["grams_per_unit"] = pd.to_numeric(H["grams_per_unit"], errors="coerce")
    H = H.dropna(subset=["fdc_id", "unit", "grams_per_unit"]).reset_index(drop=True)
    H["unit_norm"] = H["unit"].astype(str).str.lower().map(normalize_term)
    return H


# -------------------- 主流程 --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingredients_csv", required=True, help="配料表文件路径（支持CSV或Parquet格式）")
    ap.add_argument("--usda_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--A_table_csv", default=None)
    ap.add_argument("--B_table_csv", default=None)
    ap.add_argument("--fndds_household_csv", default=None)
    ap.add_argument("--embed_model", default=None, help="sentence-transformers 模型名，可选")
    ap.add_argument("--use_faiss", action="store_true", help="向量召回使用 FAISS")
    ap.add_argument("--fast_mode", action="store_true", help="快速模式：跳过向量模型，仅使用文本匹配")
    ap.add_argument("--fast_precise_mode", action="store_true", help="快速精准模式：使用轻量级向量模型，平衡速度和质量")
    ap.add_argument("--min_fuzz", type=int, default=72)
    ap.add_argument("--topk_lex", type=int, default=60)
    ap.add_argument("--topk_embed", type=int, default=120)
    ap.add_argument("--highfreq_boost", type=int, default=40, help="高频词额外扩容上限")
    ap.add_argument("--min_score", type=float, default=0.38, help="最终 rerank score 下限（0..1）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 加载配料
    # 支持CSV和Parquet格式
    if args.ingredients_csv.endswith('.parquet'):
        ing = pd.read_parquet(args.ingredients_csv)
    else:
        ing = pd.read_csv(args.ingredients_csv)
    
    ing_cols = {c.lower(): c for c in ing.columns}
    term_col = None
    for k in ["ingredient_norm", "ingredient", "term", "name"]:
        if k in ing_cols:
            term_col = ing_cols[k]
            break
    if term_col is None:
        raise ValueError("ingredients_csv 需要包含 ingredient_norm/ingredient/term/name 之一")
    ing["term_raw"] = ing[term_col].astype(str)
    ing["term_norm"] = ing["term_raw"].map(normalize_term)

    # 词频统计（用于高频扩容）
    freq = ing["term_norm"].value_counts().to_dict()

    # 2) 加载 B 表（同义词/别名映射）并应用
    B_df = pd.read_csv(args.B_table_csv) if args.B_table_csv and os.path.exists(args.B_table_csv) else None
    syn_map = build_syn_map(B_df)
    ing["term_norm"] = ing["term_norm"].map(lambda s: apply_synonym(s, syn_map))

    # 3) 加载 USDA foods 并建索引
    foods = load_usda_foods(args.usda_dir)

    # 按 B 表也对 USDA 端做一遍轻度 mapping（可选）
    foods["desc_norm_syn"] = foods["desc_norm"].map(lambda s: apply_synonym(s, syn_map))

    inv = make_inverted_index(foods.assign(desc_norm=foods["desc_norm_syn"]))
    
    # 保存原始foods的fdc_id映射，用于向量搜索结果转换
    foods_fdc_mapping = foods["fdc_id"].to_dict()

    # 4) 向量召回（可选）
    embed_search = None
    if args.fast_mode:
        print("🚀 快速模式：跳过向量模型，仅使用文本匹配")
        if args.embed_model:
            print(f"⚠️  检测到 --embed_model 参数，但快速模式下将被忽略")
    elif args.fast_precise_mode:
        print("⚡ 快速精准模式：使用轻量级向量模型")
        if not _HAS_ST:
            print("[WARN] sentence_transformers 不可用，将降级为快速模式")
        else:
            # 使用更小的模型和优化的参数
            fast_model = "sentence-transformers/all-MiniLM-L6-v2"
            try:
                print(f"🔄 加载轻量级模型: {fast_model}")
                embed_search = EmbedSearcher(fast_model, foods.assign(desc_norm=foods["desc_norm_syn"]), use_faiss=False)  # 不使用FAISS以节省内存
                print("✅ 快速精准模式准备完成")
            except Exception as e:
                print(f"[WARN] 快速精准模式加载失败: {e}")
                print("[WARN] 将降级为快速模式")
                embed_search = None
    elif args.embed_model:
        if not _HAS_ST:
            print("[WARN] sentence_transformers 不可用，跳过向量召回")
        else:
            try:
                embed_search = EmbedSearcher(args.embed_model, foods.assign(desc_norm=foods["desc_norm_syn"]), use_faiss=args.use_faiss)
            except Exception as e:
                print(f"[WARN] 向量模型加载失败: {e}")
                print("[WARN] 将跳过向量召回，继续使用其他方法")
                embed_search = None
    else:
        print("ℹ️  未指定向量模型，将仅使用文本匹配")

    # 5) A 表 + FNDDS household（融合）
    A = load_A_table(args.A_table_csv)
    H = load_fndds_household(args.fndds_household_csv)
    G = None
    if A is not None and H is not None:
        G = pd.concat([A, H], ignore_index=True)
    elif A is not None:
        G = A.copy()
    elif H is not None:
        G = H.copy()

    # 6) 匹配 - 唯一术语匹配后回填
    # 预构建便捷映射
    fdc_to_desc = foods.set_index("fdc_id")["desc_norm_syn"].to_dict()

    # 1) 只取唯一术语，并统计频次（用于自适应 topk）
    term_counts = ing["term_norm"].value_counts()
    terms_unique = term_counts.index.tolist()

    print(f"🔄 开始匹配唯一术语 {len(terms_unique)} 个（原始行 {len(ing)} ）...")

    # 2) （可选）为唯一术语一次性生成查询向量并缓存
    qvec_cache = {}
    if embed_search is not None:
        # embed_search.model 已加载；用相同 normalize 设置
        # 注意：大批量编码可分批
        from math import ceil
        BATCH = 4096
        for i in range(0, len(terms_unique), BATCH):
            batch = terms_unique[i:i+BATCH]
            vecs = embed_search.model.encode(
                batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            for t, v in zip(batch, vecs):
                qvec_cache[t] = v

    best_by_term = {}      # term_norm -> dict(fdc_id, score, fuzz, embed_sim, ...)
    detail_rows = []       # 可选：保存候选细节（用于审计）

    from tqdm import tqdm
    for term in tqdm(terms_unique, desc="匹配唯一术语"):
        # 自适应 topk（高频才扩容）
        local_topk_lex = min(args.topk_lex + (args.highfreq_boost if term_counts[term] > 100 else 0), 120)

        # ---- 词典倒排召回 ----
        toks = set(term.split())
        lex_pool = set()
        for t in toks:
            # 跳过高 DF 的"泛词"token，避免候选爆炸
            if t in {"powder","oil","sauce","ground","fresh"}:
                continue
            if t in inv:
                # 限制每个 token 带来的候选上限，避免超大 DF
                cands = list(inv[t])
                if len(cands) > 200:
                    cands = cands[:200]
                lex_pool |= set(cands)
        lex_pool = list(lex_pool)[:max(local_topk_lex, 1)]

        # ---- 向量召回（可选） ----
        embed_cands = []
        embed_sims = {}
        if embed_search is not None:
            # 用缓存的查询向量（若无则即时编码）
            qv = qvec_cache.get(term, None)
            if qv is None:
                qv = embed_search.model.encode([term], convert_to_numpy=True, normalize_embeddings=True)[0]
                qvec_cache[term] = qv
            if getattr(embed_search, "index", None) is not None:
                # FAISS 路径
                D, I = embed_search.index.search(np.asarray([qv], dtype=np.float32), args.topk_embed)
                hits = [(int(ix), float(sc)) for ix, sc in zip(I[0], D[0]) if ix >= 0]
                embed_cands = [foods_fdc_mapping[i] for (i, _sc) in hits if i in foods_fdc_mapping]
                embed_sims = {foods_fdc_mapping[i]: float(sc) for (i, sc) in hits if i in foods_fdc_mapping}
            else:
                # 退化全量（已在 embed_search.search 实现，可直接调用）
                hits = embed_search.search(term, topk=args.topk_embed)
                embed_cands = [foods_fdc_mapping[i] for (i, _sc) in hits if i in foods_fdc_mapping]
                embed_sims = {foods_fdc_mapping[i]: float(sc) for (i, sc) in hits if i in foods_fdc_mapping}

        pool = set(lex_pool) | set(embed_cands)
        if not pool:
            best_by_term[term] = None
            continue

        # ---- 打分 ----
        best = None
        for fdc in pool:
            desc = fdc_to_desc.get(fdc, "")
            if not desc:
                continue
            fr = safe_ratio(term, desc)  # 0..100
            if fr < args.min_fuzz and fdc not in embed_sims:
                continue
            es = float(embed_sims.get(fdc, 0.0))
            prior = category_prior(desc)
            dtype_bonus = max(prior.values())
            rule_bonus = 1.0 if any(k in term for k in ["egg","chicken","tomato","onion","pepper","oil"]) and \
                               any(k in desc for k in ["egg","chicken","tomato","onion","pepper","oil"]) else 0.0
            sc = rerank_score(fr, es, dtype_bonus, rule_bonus)
            if (best is None or sc > best["score"]):
                best = {"fdc_id": int(fdc), "score": sc, "fuzz": fr, "embed_sim": es,
                        "dtype_bonus": dtype_bonus, "rule_bonus": rule_bonus}
        if best and best["score"] >= args.min_score:
            best_by_term[term] = best
        else:
            best_by_term[term] = None

    # 把 term 级别的匹配结果广播回原始行
    best_df = (pd.Series(best_by_term, name="best")
                 .to_frame()
                 .reset_index().rename(columns={"index":"term_norm"}))
    # 拆开 best 字典
    def expand_best_dict(x, _np=np, _pd=pd):
        if isinstance(x, dict):
            return _pd.Series(x)
        return _pd.Series({
            "fdc_id": _np.nan, "score": _np.nan, "fuzz": _np.nan, "embed_sim": _np.nan,
            "dtype_bonus": _np.nan, "rule_bonus": _np.nan
        })
    
    best_df = pd.concat([
        best_df[["term_norm"]],
        best_df["best"].apply(expand_best_dict)
    ], axis=1)

    # 统一缺失值 & 数据类型
    best_df["fdc_id"] = pd.to_numeric(best_df["fdc_id"], errors="coerce").astype("Int64")
    num_cols = ["score","fuzz","embed_sim","dtype_bonus","rule_bonus"]
    for c in num_cols:
        best_df[c] = pd.to_numeric(best_df[c], errors="coerce")

    aligned = ing.merge(best_df, on="term_norm", how="left")

    # 8) 克重回填（A 表 / household / 动态默认）
    if G is not None:
        # 先对 unit 做规范化匹配（若配料表含 unit 列）
        unit_col = None
        for k in ["unit_std", "unit", "unit_norm", "quantity_unit", "qty_unit"]:
            if k in ing_cols:
                unit_col = ing_cols[k]
                break
        if unit_col is not None:
            aligned["unit_norm"] = aligned[unit_col].astype(str).str.lower().map(normalize_term)
            G_key = G[["fdc_id", "unit_norm", "grams_per_unit"]].dropna().copy()
            aligned = aligned.merge(G_key, on=["fdc_id", "unit_norm"], how="left", suffixes=("", "_fromG"))
        else:
            aligned["grams_per_unit"] = np.nan

    # 动态默认克重：按匹配描述类别 prior
    # 若不能从 G/A 表里拿到单位克重，则给定一个 default_grams_by_cat
    def _default_g(row, _np=np, _pd=pd, _fdc_to_desc=fdc_to_desc, _category_prior=category_prior, _choose_default_grams=choose_default_grams):
        if not _pd.isna(row.get("grams_per_unit", _np.nan)):
            return row["grams_per_unit"]
        desc = _fdc_to_desc.get(row["fdc_id"], "") if not _pd.isna(row.get("fdc_id", _np.nan)) else ""
        prior = _category_prior(desc)
        return _choose_default_grams(prior)

    aligned["grams_per_unit_fill"] = aligned.apply(_default_g, axis=1)

    # 9) 输出
    aligned_path = os.path.join(args.out_dir, "aligned.parquet")
    best_path = os.path.join(args.out_dir, "aligned_best.parquet")

    # 保留关键信息
    keep_cols = [c for c in aligned.columns if c not in {"row_id"}]
    aligned[keep_cols].to_parquet(aligned_path, index=False)

    # 对每条原始配料只保留最佳匹配行
    aligned_best = aligned.dropna(subset=["fdc_id"])\
                         .sort_values(["term_norm", "score"], ascending=[True, False])\
                         .groupby("term_norm", as_index=False).first()
    aligned_best.to_parquet(best_path, index=False)

    # 10) 未匹配项统计（error-aware）
    unmatched = aligned[aligned["fdc_id"].isna()]["term_norm"].value_counts().reset_index()
    unmatched.columns = ["term", "count"]
    unmatched_path = os.path.join(args.out_dir, "unmatched_topfreq.csv")
    unmatched.to_csv(unmatched_path, index=False)

    # 11) 审计 quick report
    tot = len(ing)
    hit = len(aligned[aligned["fdc_id"].notna()])
    hit_rate = 100.0 * hit / max(tot, 1)

    avg_fuzz = aligned["fuzz"].mean() if not aligned.empty else 0.0
    avg_embed = aligned["embed_sim"].mean() if not aligned.empty else 0.0

    audit = {
        "ingredients": tot,
        "matched": int(hit),
        "hit_rate_percent": round(hit_rate, 2),
        "min_fuzz": args.min_fuzz,
        "min_score": args.min_score,
        "avg_fuzz_in_candidates": round(float(avg_fuzz), 2),
        "avg_embed_in_candidates": round(float(avg_embed), 4),
        "use_embeddings": bool(embed_search is not None),
        "use_faiss": bool(embed_search is not None and args.use_faiss and _HAS_FAISS),
        "A_table_loaded": bool(A is not None),
        "household_loaded": bool(H is not None),
    }

    with open(os.path.join(args.out_dir, "audit_summary.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(audit, ensure_ascii=False, indent=2))

    print("==== step2 对齐完毕 ====")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"保存：{aligned_path}\n保存：{best_path}\n未匹配频次：{unmatched_path}")


if __name__ == "__main__":
    main()
