# -*- coding: utf-8 -*-
"""
step3_ingredient_alignment_plus.py — 步骤3（增强版）：配料对齐
改进点：
1) 候选集构造：倒排(token) ∪ 向量近邻 ∪ 子串/拼写近邻（可选），可去重、限幅。
2) 重排序：融合 RapidFuzz token_set_ratio、Jaccard(token) 与 nutrient_coverage / dtype_bonus，
   同时加入嵌入相似度（若可得）并采用可配置权重；支持动态阈值与置信度。
3) 可选“精确模式”：当存在高分(≥min_fuzz+δ)候选时，优先 exact-like 命中并跳过长尾重排序。
4) 并行：默认 ThreadPool + RapidFuzz（C++后端），可切换 ProcessPool；流式tqdm展示进度。
5) 产出：
   - ingredient_mapping.parquet（最终对齐）
   - ingredient_mapping_topk.parquet（每个norm的TopK候选用于审计）
   - unmatched_ingredients.csv（未命中清单）
   - 对齐快报（JSON+终端打印）

依赖：pandas, numpy, rapidfuzz>=3.0, tqdm, (可选) faiss-cpu
"""
import os
import re
import gc
import json
import math
import pickle
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from rapidfuzz import fuzz, process as rf_process
from collections import defaultdict

# =============== Common helpers ===============

def safe_norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(text: str) -> List[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    toks = [t for t in text.split() if t]
    return toks

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

# =============== 增量式处理辅助函数 ===============

def _normalize_tokens(s: str):
    return [t for t in re.sub(r"[^a-z0-9\s]", " ", str(s).lower()).split() if t]

def _jaccard(a, b):
    sa, sb = set(a), set(b)
    return (len(sa & sb) / max(1, len(sa | sb)))

def _lex_candidates(term_norm, inv_index, max_per_token=200):
    toks = set(_normalize_tokens(term_norm))
    pool = set()
    for t in toks:
        if t in {"powder","oil","sauce","ground","fresh"}:  # 降噪
            continue
        # 处理inv_index中可能是tuple、list或set的情况
        index_data = inv_index.get(t, [])
        if isinstance(index_data, (tuple, list)):
            ids = list(index_data)
        else:
            ids = list(index_data)
        if len(ids) > max_per_token:
            ids = ids[:max_per_token]
        pool.update(ids)
    return list(pool)

def _rerank(term_norm, cand_ids, food_df, min_fuzz, cov_w=0.7, qual_w=0.15, jac_w=0.1, fuzz_w=0.75):
    if not cand_ids:
        return None, []
    sub = food_df.loc[food_df["fdc_id"].isin(cand_ids),
                      ["fdc_id","description","description_norm","nutrient_coverage","dtype_bonus"]].copy()
    if sub.empty:
        return None, []
    max_cov = max(1.0, float(sub["nutrient_coverage"].max()))
    max_bonus = max(1.0, float(sub["dtype_bonus"].max()))
    t_tok = _normalize_tokens(term_norm)

    rows, best, best_sc = [], None, -1.0
    for r in sub.itertuples(index=False):
        fz = float(fuzz.token_set_ratio(term_norm, r.description_norm))  # 0..100
        if fz < min_fuzz:  # 先做硬阈值
            continue
        jac = _jaccard(t_tok, _normalize_tokens(r.description_norm))    # 0..1
        cov_sc = 100.0 * (float(r.nutrient_coverage) / max_cov)
        bon_sc = 100.0 * (float(r.dtype_bonus) / max_bonus)
        qual = cov_w * cov_sc + (1 - cov_w) * bon_sc
        total = (fuzz_w * fz) + (jac_w * (jac * 100.0)) + (qual_w * qual)
        item = {"fdc_id": int(r.fdc_id), "fdc_desc": r.description, "fuzz": fz,
                "jaccard": jac, "coverage": float(r.nutrient_coverage),
                "dtype_bonus": float(r.dtype_bonus), "score": total}
        rows.append(item)
        if total > best_sc:
            best_sc, best = total, item
    rows.sort(key=lambda x: x["score"], reverse=True)
    return best, rows

def run_step3_incremental(base_dir, min_fuzz_cli=80, audit_topk=5, max_union=300, threads=8):
    step2_dir = os.path.join(base_dir, "step2")  # step2 产物在 step2 子目录中
    step3_dir = os.path.join(base_dir, "step3")
    os.makedirs(step3_dir, exist_ok=True)
    
    print(f"[Step3+ 增量式] 输入目录: {step2_dir}")
    print(f"[Step3+ 增量式] 输出目录: {step3_dir}")

    # 1) 读 step2 命中集与审计阈值
    aligned_best_path = os.path.join(step2_dir, "aligned_best.parquet")
    print(f"[Step3+ 增量式] 读取对齐结果: {aligned_best_path}")
    aligned_best = pd.read_parquet(aligned_best_path)
    audit_path = os.path.join(step2_dir, "audit_summary.txt")
    if os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        # 自适应阈值（取较稳健者）
        min_fuzz = max(75, int(round(audit.get("avg_fuzz_in_candidates", min_fuzz_cli) - 5)))
    else:
        min_fuzz = min_fuzz_cli

    # 2) 读倒排与 food（来自 step1）
    with open(os.path.join(base_dir, "inv_index.pkl"), "rb") as f:
        inv_index = pickle.load(f)
    food = pd.read_parquet(os.path.join(base_dir, "food_processed.parquet"))

    # 3) 已命中直接收下；仅补未命中
    # 期望 aligned_best 至少有 ['term_norm','fdc_id']（参见 step2 输出说明）
    # 若列名不同，做一次小兼容
    cols = {c.lower(): c for c in aligned_best.columns}
    term_col = cols.get("term_norm") or cols.get("ingredient_norm") or list(cols.values())[0]
    out_rows, audit_rows = [], []
    seen_terms = set()

    # 3.1 已命中 - 保留step2的评分信息
    hit_df = aligned_best.dropna(subset=[cols.get("fdc_id","fdc_id")]).copy()
    for r in hit_df.itertuples(index=False):
        term = getattr(r, term_col)
        fdc  = getattr(r, cols.get("fdc_id","fdc_id"))
        
        # 尝试从step2结果中获取评分信息，如果不存在则设为None
        fdc_desc = getattr(r, cols.get("fdc_desc", "fdc_desc"), None)
        fuzz = getattr(r, cols.get("fuzz", "fuzz"), None)
        jaccard = getattr(r, cols.get("jaccard", "jaccard"), None)
        coverage = getattr(r, cols.get("coverage", "coverage"), None)
        dtype_bonus = getattr(r, cols.get("dtype_bonus", "dtype_bonus"), None)
        score = getattr(r, cols.get("score", "score"), None)
        
        # 如果fdc_desc为None，尝试从food数据中获取
        if fdc_desc is None and fdc is not None:
            food_match = food[food["fdc_id"] == fdc]
            if not food_match.empty:
                fdc_desc = food_match.iloc[0].get("description", None)
                # 如果coverage为None，也从food数据中获取
                if coverage is None:
                    coverage = food_match.iloc[0].get("nutrient_coverage", None)
                # 如果dtype_bonus为None，也从food数据中获取
                if dtype_bonus is None:
                    dtype_bonus = food_match.iloc[0].get("dtype_bonus", None)
        
        out_rows.append({"ingredient_norm": term, "fdc_id": int(fdc),
                         "fdc_desc": fdc_desc, "fuzz": fuzz, "jaccard": jaccard,
                         "coverage": coverage, "dtype_bonus": dtype_bonus, "score": score})
        seen_terms.add(term)

    # 3.2 未命中 → 倒排+模糊补救
    miss_df = aligned_best[aligned_best[cols.get("fdc_id","fdc_id")].isna()].copy()
    uniq_miss = sorted(set(miss_df[term_col].astype(str)))
    for term in tqdm(uniq_miss, desc="lex-fuzzy rescue"):
        cands = _lex_candidates(term, inv_index)
        best, rows = _rerank(term, cands[:max_union], food, min_fuzz=min_fuzz)
        if best is None:
            out_rows.append({"ingredient_norm": term, "fdc_id": None, "fdc_desc": None,
                             "fuzz": None, "jaccard": None, "coverage": 0.0,
                             "dtype_bonus": 0.0, "score": 0.0})
        else:
            out_rows.append({"ingredient_norm": term, **best})
        for k, item in enumerate(rows[:audit_topk]):
            audit_rows.append({"ingredient_norm": term, "rank": k+1, **item})

    mapping_df = pd.DataFrame(out_rows)
    audit_df = pd.DataFrame(audit_rows)

    # 4) 保存
    map_path = os.path.join(step3_dir, "ingredient_mapping.parquet")
    audit_path = os.path.join(step3_dir, "ingredient_mapping_topk.parquet")
    unmatched_path = os.path.join(step3_dir, "unmatched_ingredients.csv")
    mapping_df.to_parquet(map_path, index=False)
    audit_df.to_parquet(audit_path, index=False)
    mapping_df[mapping_df["fdc_id"].isna()][["ingredient_norm"]].drop_duplicates()\
              .to_csv(unmatched_path, index=False, encoding="utf-8")

    # 5) 快报
    tot = len(set(aligned_best[term_col].astype(str)))
    hit = int(mapping_df["fdc_id"].notna().sum())
    print(json.dumps({
        "ingredients": tot,
        "matched": hit,
        "hit_rate_percent": round(100.0*hit/max(1,tot), 2),
        "min_fuzz": min_fuzz,
        "mode": "incremental_from_step2"
    }, ensure_ascii=False, indent=2))

@dataclass
class Weights:
    w_fuzz: float = 0.6
    w_jaccard: float = 0.1
    w_embed: float = 0.15
    w_quality: float = 0.15  # nutrient_coverage & dtype_bonus
    quality_cov_ratio: float = 0.7  # within w_quality

# =============== Candidate generation ===============

def make_candidates(norm: str,
                    inv_index: Dict[str, set],
                    embed_ids: Optional[np.ndarray],
                    usda_ids: np.ndarray,
                    max_union: int = 300) -> List[int]:
    """Union of token-posting list & embedding neighbors (int fdc_id)."""
    cands: set = set()
    for t in set(tokenize(norm)):
        if len(t) >= 3:
            # 处理inv_index中可能是tuple或set的情况
            index_data = inv_index.get(t, set())
            if isinstance(index_data, (tuple, list)):
                cands.update(index_data)
            else:
                cands |= index_data
    if embed_ids is not None:
        n_ids = len(usda_ids)
        for cid in embed_ids:
            try:
                ic = int(cid)
            except Exception:
                continue
            if 0 <= ic < n_ids:
                cands.add(int(usda_ids[ic]))
            if len(cands) >= max_union:
                break
    return list(cands)

# =============== Reranking ===============

def rerank_one(norm: str,
               cand_ids: List[int],
               food_df: pd.DataFrame,
               embed_sim_map: Optional[Dict[int, float]],
               weights: Weights,
               min_fuzz: int) -> Tuple[Optional[dict], List[dict]]:
    """Return best match + full scored list (for auditing)."""
    if not cand_ids:
        return None, []

    cols = ["fdc_id", "description", "description_norm", "nutrient_coverage", "dtype_bonus"]
    sub = food_df.loc[food_df["fdc_id"].isin(cand_ids), cols]
    if sub.empty:
        return None, []

    tokens_norm = tokenize(norm)
    max_cov = max(1.0, float(sub["nutrient_coverage"].max()))
    max_bonus = max(1.0, float(sub["dtype_bonus"].max()))

    rows = []
    best = None
    best_total = -1.0

    for row in sub.itertuples(index=False):
        fz = float(fuzz.token_set_ratio(norm, row.description_norm))  # 0~100
        jac = jaccard(tokens_norm, tokenize(row.description_norm))    # 0~1
        em = float(embed_sim_map.get(row.fdc_id, 0.0)) if embed_sim_map else 0.0  # 0~1
        cov_sc = 100.0 * (float(row.nutrient_coverage) / max_cov)
        bonus_sc = 100.0 * (float(row.dtype_bonus) / max_bonus)
        quality = weights.quality_cov_ratio * cov_sc + (1 - weights.quality_cov_ratio) * bonus_sc
        total = (weights.w_fuzz * fz
                 + weights.w_jaccard * (jac * 100.0)
                 + weights.w_embed * (em * 100.0)
                 + weights.w_quality * quality)

        item = {
            "fdc_id": int(row.fdc_id),
            "fdc_desc": row.description,
            "fuzz": fz,
            "jaccard": jac,
            "embed_sim": em,
            "coverage": float(row.nutrient_coverage),
            "dtype_bonus": float(row.dtype_bonus),
            "score": total,
        }
        rows.append(item)
        if total > best_total:
            best_total = total
            best = item

    # 动态阈值：当营养覆盖很高且嵌入/模糊得分较好时，稍降min_fuzz要求
    if best is not None:
        thr = min_fuzz
        if best["coverage"] >= 8:  # 经验阈，可在CLI暴露
            thr = max(min_fuzz - 5, 60)
        if best["fuzz"] < thr:
            # 视为未匹配
            best = None

    # TopK审计：按score降序
    rows.sort(key=lambda x: x["score"], reverse=True)
    return best, rows

# =============== Parallel driver ===============

def align_batch(unique_norms: List[str],
                inv_index: Dict[str, set],
                food_df: pd.DataFrame,
                topk_ids: np.ndarray,
                topk_sims: Optional[np.ndarray],
                usda_ids: np.ndarray,
                threads: int,
                min_fuzz: int,
                max_union: int,
                weights: Weights,
                audit_topk: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    audits: List[dict] = []

    # usda_id 索引到真实 fdc_id 的映射，用于 embed 相似度字典
    def build_embed_map(idx: int) -> Dict[int, float]:
        if topk_ids is None:
            return {}
        ids = topk_ids[idx]
        sims = topk_sims[idx] if topk_sims is not None else None
        mp = {}
        for j, usda_idx in enumerate(ids):
            try:
                fid = int(usda_ids[int(usda_idx)])
            except Exception:
                continue
            sim = float(sims[j]) if sims is not None else 0.0
            mp[fid] = sim
        return mp

    with ThreadPoolExecutor(max_workers=threads) as ex:
        # 修复：使用字典存储future和对应的索引，确保顺序正确
        future_to_index = {}
        for i, norm in enumerate(unique_norms):
            embed_map = build_embed_map(i)
            cands = make_candidates(norm, inv_index, topk_ids[i] if topk_ids is not None else None, usda_ids, max_union)
            future = ex.submit(rerank_one, norm, cands, food_df, embed_map, weights, min_fuzz)
            future_to_index[future] = i

        # 修复：使用future_to_index确保正确的顺序
        for fu in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="align"):
            i = future_to_index[fu]
            norm = unique_norms[i]
            try:
                best, rows = fu.result()
            except Exception as e:
                print(f"Warning: Error processing {norm}: {e}")
                best, rows = None, []
            
            if best is None:
                results.append({
                    "ingredient_norm": norm,
                    "fdc_id": None,
                    "fdc_desc": None,
                    "fuzz": None,
                    "jaccard": None,
                    "embed_sim": None,
                    "coverage": 0.0,
                    "dtype_bonus": 0.0,
                    "score": 0.0,
                })
            else:
                out = {"ingredient_norm": norm, **best}
                results.append(out)

            # 审计TopK
            for k, item in enumerate(rows[:audit_topk]):
                audits.append({"ingredient_norm": norm, "rank": k + 1, **item})

    return pd.DataFrame(results), pd.DataFrame(audits)

# =============== Main ===============

def main():
    ap = argparse.ArgumentParser(description="步骤3（增强版）：配料对齐")
    ap.add_argument("--base_dir", required=True, help="基础目录路径（包含step1和step2输出的目录）")
    ap.add_argument("--topk_embed", type=int, default=120)
    ap.add_argument("--min_fuzz", type=int, default=80)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max_candidates_union", type=int, default=300)
    ap.add_argument("--audit_topk", type=int, default=5)
    ap.add_argument("--w_fuzz", type=float, default=0.6)
    ap.add_argument("--w_jaccard", type=float, default=0.1)
    ap.add_argument("--w_embed", type=float, default=0.15)
    ap.add_argument("--w_quality", type=float, default=0.15)
    ap.add_argument("--quality_cov_ratio", type=float, default=0.7)
    ap.add_argument("--faiss", action="store_true", help="显式启用faiss（若可用）")
    ap.add_argument("--incremental", action="store_true", help="使用增量式处理（基于step2结果）")
    args = ap.parse_args()

    base_dir = args.base_dir
    step2_dir = os.path.join(base_dir, "step2")
    step3_dir = os.path.join(base_dir, "step3")
    
    # 创建step3输出目录
    os.makedirs(step3_dir, exist_ok=True)
    print(f"[Step3+] 输入目录: {step2_dir}")
    print(f"[Step3+] 输出目录: {step3_dir}")
    
    # 如果使用增量式处理，直接调用增量函数
    if args.incremental:
        print("[Step3+] 使用增量式处理模式")
        run_step3_incremental(
            base_dir=base_dir,
            min_fuzz_cli=args.min_fuzz,
            audit_topk=args.audit_topk,
            max_union=args.max_candidates_union,
            threads=args.threads
        )
        return

    # === 加载数据 ===
    print("[Step3+] 加载数据...")
    
    # 检查必需文件是否存在
    # food_processed.parquet 从 base_dir 读取（step1输出）
    food_file = os.path.join(base_dir, "food_processed.parquet")
    if not os.path.exists(food_file):
        raise FileNotFoundError(f"Required file not found: {food_file}")
    
    # 检查step2是否生成了嵌入向量文件，如果没有则跳过嵌入检索
    embedding_files = [
        "usda_food_desc_emb.npy",
        "usda_food_desc_ids.npy", 
        "ingredient_embeddings.npy",
        "unique_ingredients.txt"
    ]
    
    has_embeddings = True
    missing_embedding_files = []
    
    for file in embedding_files:
        step2_path = os.path.join(step2_dir, file)
        base_path = os.path.join(base_dir, file)
        
        if not os.path.exists(step2_path) and not os.path.exists(base_path):
            missing_embedding_files.append(file)
            has_embeddings = False
    
    if not has_embeddings:
        print(f"[Step3+] 警告: 未找到嵌入向量文件，将跳过嵌入检索")
        print(f"  缺少的文件: {missing_embedding_files}")
        print(f"  将使用step2的对齐结果: {os.path.join(step2_dir, 'aligned.parquet')}")
    
    # 检查必需的基础文件
    required_files = ["inv_index.pkl"]
    file_locations = {}
    
    for file in required_files:
        step2_path = os.path.join(step2_dir, file)
        base_path = os.path.join(base_dir, file)
        
        if os.path.exists(step2_path):
            file_locations[file] = step2_path
        elif os.path.exists(base_path):
            file_locations[file] = base_path
        else:
            raise FileNotFoundError(f"Required file not found: {file}")
    
    print(f"[Step3+] 文件位置:")
    for file, path in file_locations.items():
        print(f"  - {file}: {path}")
    
    try:
        # 从 base_dir 加载 food 数据（step1输出）
        food = pd.read_parquet(food_file)
        print(f"  - 加载food数据: {len(food)} 条记录 (来自: {food_file})")
        
        # 从找到的位置加载倒排索引
        with open(file_locations["inv_index.pkl"], "rb") as f:
            inv_index = pickle.load(f)
        print(f"  - 加载倒排索引: {len(inv_index)} 个token (来自: {file_locations['inv_index.pkl']})")
        
        # 根据是否有嵌入向量文件决定加载策略
        if has_embeddings:
            # 加载嵌入向量文件
            usda_vec = np.load(file_locations["usda_food_desc_emb.npy"])
            usda_ids = np.load(file_locations["usda_food_desc_ids.npy"])
            print(f"  - 加载USDA向量: {usda_vec.shape}, IDs: {usda_ids.shape}")
            
            ing_vec = np.load(file_locations["ingredient_embeddings.npy"])
            print(f"  - 加载配料向量: {ing_vec.shape}")
            
            with open(file_locations["unique_ingredients.txt"], "r", encoding="utf-8") as f:
                uniq_norms = [ln.strip() for ln in f if ln.strip()]
            print(f"  - 加载唯一配料: {len(uniq_norms)} 个")
        else:
            # 使用step2的对齐结果
            aligned_file = os.path.join(step2_dir, "aligned.parquet")
            if not os.path.exists(aligned_file):
                raise FileNotFoundError(f"Step2对齐结果文件不存在: {aligned_file}")
            
            aligned_df = pd.read_parquet(aligned_file)
            print(f"  - 加载step2对齐结果: {len(aligned_df)} 条记录 (来自: {aligned_file})")
            
            # 从对齐结果中提取唯一配料
            if "term_norm" in aligned_df.columns:
                uniq_norms = aligned_df["term_norm"].dropna().unique().tolist()
            elif "ingredient_norm" in aligned_df.columns:
                uniq_norms = aligned_df["ingredient_norm"].dropna().unique().tolist()
            else:
                raise ValueError("对齐结果中找不到配料列 (term_norm 或 ingredient_norm)")
            
            print(f"  - 提取唯一配料: {len(uniq_norms)} 个")
            
            # 设置空的嵌入向量（将跳过嵌入检索）
            usda_vec = None
            usda_ids = None
            ing_vec = None
        
        # 数据一致性检查（仅在有嵌入向量时）
        if has_embeddings:
            if len(ing_vec) != len(uniq_norms):
                raise ValueError(f"配料向量数量({len(ing_vec)})与唯一配料数量({len(uniq_norms)})不匹配")
            if len(usda_vec) != len(usda_ids):
                raise ValueError(f"USDA向量数量({len(usda_vec)})与ID数量({len(usda_ids)})不匹配")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # === 嵌入检索 ===
    if has_embeddings:
        print("[Step3+] 检索嵌入近邻… N_ing=", len(ing_vec), "topk=", args.topk_embed)
        
        # 查找FAISS索引文件
        faiss_index_path = None
        if args.faiss and HAS_FAISS:
            step2_faiss_path = os.path.join(step2_dir, "usda_faiss_index.bin")
            base_faiss_path = os.path.join(base_dir, "usda_faiss_index.bin")
            
            if os.path.exists(step2_faiss_path):
                faiss_index_path = step2_faiss_path
            elif os.path.exists(base_faiss_path):
                faiss_index_path = base_faiss_path
        
        if faiss_index_path:
            print(f"  - 使用FAISS索引: {faiss_index_path}")
            index = faiss.read_index(faiss_index_path)
            sims, ids = index.search(ing_vec, args.topk_embed)  # faiss返回距离/相似度取决于index类型
            # 这里假设index基于内积；若为L2需转为相似度，可做归一化
            topk_ids = ids
            # 将sims线性归一为[0,1]
            sims_min = sims.min()
            sims_max = sims.max()
            denom = (sims_max - sims_min) if (sims_max > sims_min) else 1.0
            topk_sims = (sims - sims_min) / denom
        else:
            # numpy 余弦近邻
            # 先L2归一化
            def l2norm(x):
                n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
                return x / n
            A = l2norm(ing_vec)
            B = l2norm(usda_vec)
            sims = A @ B.T  # (n_ing, n_usda)
            topk_ids = np.argpartition(-sims, kth=min(args.topk_embed, sims.shape[1]-1), axis=1)[:, :args.topk_embed]
            # 排序
            row_indices = np.arange(sims.shape[0])[:, None]
            topk_ids = topk_ids[row_indices, np.argsort(-sims[row_indices, topk_ids])]
            topk_sims = np.take_along_axis(sims, topk_ids, axis=1)
    else:
        print("[Step3+] 跳过嵌入检索，将仅使用文本匹配")
        topk_ids = None
        topk_sims = None

    weights = Weights(args.w_fuzz, args.w_jaccard, args.w_embed, args.w_quality, args.quality_cov_ratio)

    # === 对齐 ===
    mapping_df, audit_df = align_batch(
        unique_norms=uniq_norms,
        inv_index=inv_index,
        food_df=food,
        topk_ids=topk_ids,
        topk_sims=topk_sims,
        usda_ids=usda_ids,
        threads=args.threads,
        min_fuzz=args.min_fuzz,
        max_union=args.max_candidates_union,
        weights=weights,
        audit_topk=args.audit_topk,
    )

    # === 保存 ===
    print("[Step3+] 保存结果...")
    map_path = os.path.join(step3_dir, "ingredient_mapping.parquet")
    audit_path = os.path.join(step3_dir, "ingredient_mapping_topk.parquet")
    unmatched_path = os.path.join(step3_dir, "unmatched_ingredients.csv")

    try:
        mapping_df.to_parquet(map_path, index=False)
        print(f"  - 保存对齐结果: {map_path}")
        
        audit_df.to_parquet(audit_path, index=False)
        print(f"  - 保存审计结果: {audit_path}")
        
        unmatched_df = mapping_df[mapping_df["fdc_id"].isna()][["ingredient_norm"]].drop_duplicates()
        unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8")
        print(f"  - 保存未匹配配料: {unmatched_path} ({len(unmatched_df)} 个)")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        raise

    # 快报
    total = len(uniq_norms)
    matched = int(mapping_df["fdc_id"].notna().sum())
    rate = 100.0 * matched / total if total else 0.0
    report = {
        "ingredients": total,
        "matched": matched,
        "hit_rate_percent": round(rate, 2),
        "min_fuzz": args.min_fuzz,
        "topk_embed": args.topk_embed,
        "max_union": args.max_candidates_union,
        "weights": vars(weights),
        "HAS_FAISS": bool(args.faiss and HAS_FAISS),
    }
    print("[Step3+] 对齐完成:\n", json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
