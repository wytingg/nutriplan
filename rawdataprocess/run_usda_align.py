# -*- coding: utf-8 -*-
"""
run_usda_align_v2.py — USDA 和 FNDDS 对齐流水线（Embedding 二次召回 + 并行 + 份量解析 / 全营养保留）

依赖：
  pip install pandas pyarrow tqdm rapidfuzz unidecode ftfy sentence-transformers faiss-cpu
  #（可选）如果 faiss-cpu 安装困难，会自动退化为 numpy 余弦近邻（稍慢）

用法示例（请修改你的绝对路径）：
  python run_usda_align_v2.py \
    --usda_dir "/ABS/PATH/TO/usda_csv" \
    --fndds_dir "/ABS/PATH/TO/fndds_csv" \
    --recipes "/ABS/PATH/TO/recipes.parquet" \
    --reviews "/ABS/PATH/TO/reviews.parquet" \
    --out_dir "/ABS/PATH/TO/out" \
    --threads 12 --topk_embed 120 --min_fuzz 80
"""

import os, re, json, ast, argparse, math, gc, sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from ftfy import fix_text
from unidecode import unidecode
from rapidfuzz import fuzz
import warnings

# ------------------------ 可选：Faiss 近邻 ------------------------
try:
    import faiss  # faiss-cpu
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ------------------------ 嵌入模型 ------------------------
from sentence_transformers import SentenceTransformer

# ------------------------ 解析&清洗 ------------------------
UNIT_RE = re.compile(r"\b(cups?|cup|tbsp|tablespoons?|tbsps?|tsp|teaspoons?|tsps?|grams?|gram|g|kg|kgs|ml|l|liters?|liter|oz|ounce|ounces|pounds?|lb|lbs|pinch|clove|cloves|slice|slices|stick|sticks|package|packages|can|cans|bag|bags)\b", re.I)
NUM_RE  = re.compile(r"(\d+\s+\d+/\d+|\d+/\d+|\d+(\.\d+)?)")
PAREN_RE= re.compile(r"\([^)]*\)")
NON_ASCII = re.compile(r"[^a-z0-9\s]+")

STOPWORDS = {
    "fresh","organic","large","small","medium","boneless","skinless","boned","packed",
    "finely","coarsely","light","dark","unsalted","salted","low","reduced","fat-free",
    "low-fat","low-sodium","no-salt","ripe","extra","virgin","raw","cooked","dried",
    "ground","minced","chopped","sliced","grated","shredded","peeled","seeded","thinly",
    "thick","softened","melted","warm","hot","cold","room","temperature"
}

SYNONYM_MAP = {
    "bell pepper": "capsicum",
    "green onion": "scallion",
    "spring onion": "scallion",
    "cilantro": "coriander leaves",
    "powdered sugar": "confectioners sugar",
    "corn starch": "cornstarch",
    "all purpose flour": "flour",
    "ap flour": "flour",
}

# 份量近似规则（以“克”为目标单位；尽量只覆盖“争议小”的常见物）
ITEM_GRAMS = {
    "egg": 50.0,
    "garlic clove": 3.0,
    "clove garlic": 3.0,
    "onion": 110.0,              # 中等大小 1 个
    "tomato": 120.0,             # 中等大小 1 个
    "lemon": 100.0,
    "lime": 70.0,
    "carrot": 60.0,
    "potato": 170.0,
    "banana": 120.0,

    # 片、条等
    "slice bacon": 8.0,
    "slice bread": 25.0,
    "slice cheese": 20.0,

    # stick
    "stick butter": 113.0,       # 1 条黄油（US）≈ 1/2 cup ≈ 8 tbsp ≈ 113 g
}

# 杯/勺 → 克 的近似密度（仅覆盖常见“误差较小”的食材；优先匹配规范化字符串中的关键词）
DENSITY_GRAMS = {
    "water": {"cup": 240.0, "tbsp": 15.0, "tsp": 5.0},
    "milk": {"cup": 240.0, "tbsp": 15.0, "tsp": 5.0},
    "olive oil": {"tbsp": 13.5, "tsp": 4.5, "cup": 216.0},  # 1 cup ≈ 16 tbsp
    "vegetable oil": {"tbsp": 13.5, "tsp": 4.5, "cup": 216.0},
    "soy sauce": {"tbsp": 16.0, "tsp": 5.3, "cup": 256.0},

    # 粉类
    "flour": {"cup": 120.0, "tbsp": 7.5, "tsp": 2.5},       # AP flour
    "sugar": {"cup": 200.0, "tbsp": 12.5, "tsp": 4.2},      # granulated
    "brown sugar": {"cup": 220.0, "tbsp": 13.5, "tsp": 4.5},
    "powdered sugar": {"cup": 120.0, "tbsp": 7.5, "tsp": 2.5},
    "cornstarch": {"tbsp": 8.0, "tsp": 2.7, "cup": 128.0},

    # 颗粒类
    "rice": {"cup": 180.0},                                  # uncooked
    "quinoa": {"cup": 170.0},                                # uncooked

    # 切碎蔬菜（近似）
    "onion": {"cup": 160.0},
    "tomato": {"cup": 180.0},
    "cheese": {"cup": 113.0},                                # shredded
    "butter": {"tbsp": 14.0, "tsp": 4.7, "cup": 227.0},      # 1 cup ≈ 227g
}

WEIGHT_UNITS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kgs": 1000.0,
    "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
    "lb": 453.592, "lbs": 453.592, "pound": 453.592, "pounds": 453.592,
}
VOL_UNITS = {"ml": 1.0, "l": 1000.0, "liter": 1000.0, "liters": 1000.0}

# ------------------------ 解析与标准化工具 ------------------------
def parse_vec(s):
    """
    将 recipes 中的配料/数量字段解析为 list[str]。
    支持：list/tuple/ndarray、R 风格 c("a","b")、JSON/Python 列表字面量、逗号分隔、缺失值。
    """
    import numpy as np
    if isinstance(s, (list, tuple, np.ndarray)):
        out = []
        for x in s:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            out.append(str(x).strip())
        return out
    if s is None:
        return []
    try:
        if pd.isna(s):
            return []
    except Exception:
        pass
    s = str(s).strip()
    if not s:
        return []
    try:
        if s.startswith("c("):
            return re.findall(r'"([^"]+)"', s)
        if s[0] in "[(" and s[-1] in ")]":
            try:
                arr = json.loads(s)
            except Exception:
                arr = ast.literal_eval(s)
            if isinstance(arr, (list, tuple)):
                return [str(x).strip() for x in arr]
        return [t.strip() for t in s.split(",") if t.strip()]
    except Exception:
        return [s]

def normalize_text(s):
    s = fix_text(str(s))
    s = unidecode(s).lower()
    s = PAREN_RE.sub(" ", s)
    s = re.sub(r"[-_/]", " ", s)
    s = UNIT_RE.sub(" ", s)
    s = re.sub(NUM_RE, " ", s)
    s = NON_ASCII.sub(" ", s)
    toks = [w for w in re.split(r"\s+", s) if w]
    toks = [w for w in toks if w not in STOPWORDS]
    s = " ".join(toks).strip()
    for k, v in SYNONYM_MAP.items():
        s = s.replace(k, v)
    return s

def tokenize(s):
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]

def parse_qty(raw):
    """抽取数量与单位（cup/tbsp/tsp/g/kg/ml/l/oz/lb/...）"""
    text = str(raw).lower()
    qty = None
    for m in re.finditer(NUM_RE, text):
        token = m.group(0)
        if " " in token and "/" in token:
            a, b = token.split(" ", 1)
            val = float(a) + eval(b)
        elif "/" in token:
            val = float(eval(token))
        else:
            val = float(token)
        qty = max(qty or 0.0, val)
    unit = None
    for u in list(WEIGHT_UNITS.keys()) + list(VOL_UNITS.keys()) + [
        "cup","cups","tbsp","tbsps","tablespoon","tablespoons",
        "tsp","tsps","teaspoon","teaspoons","stick","sticks",
        "clove","cloves","slice","slices"]:
        if re.search(rf"\b{u}\b", text):
            unit = u
            break
    return qty, unit

def estimate_grams(raw_str, norm_str):
    qty, unit = parse_qty(raw_str)
    if unit in WEIGHT_UNITS:
        return (qty or 1.0) * WEIGHT_UNITS[unit]
    key_ings = list(DENSITY_GRAMS.keys())
    hit = None
    for k in key_ings:
        if k in norm_str:
            hit = k
            break
    def vol_to_g(u):
        if unit in ("tbsp","tablespoon","tbsps"):  u = "tbsp"
        if unit in ("tsp","teaspoon","tsps"):      u = "tsp"
        if unit in ("cup","cups"):                   u = "cup"
        return u
    if unit in ("ml","l","liter","liters"):
        ml = (qty or 1.0) * VOL_UNITS[unit]
        return ml
    if unit in ("tbsp","tablespoon","tbsps","tsp","teaspoon","tsps","cup","cups"):
        if hit and hit in DENSITY_GRAMS:
            ug = vol_to_g(unit)
            if ug in DENSITY_GRAMS[hit]:
                return (qty or 1.0) * DENSITY_GRAMS[hit][ug]
        base = {"cup":240.0, "tbsp":15.0, "tsp":5.0}
        ug = vol_to_g(unit)
        if ug in base:
            return (qty or 1.0) * base[ug]
    for key, g in ITEM_GRAMS.items():
        if key in norm_str:
            return (qty or 1.0) * g
    return None

# ------------------------ Food.com 读取 ------------------------
def load_foodcom(recipes_path, reviews_path=None):
    recipes = pd.read_parquet(recipes_path)
    cols = set(recipes.columns)
    rename_map = {}
    if "RecipeId" in cols: rename_map["RecipeId"] = "recipe_id"
    if "Name" in cols: rename_map["Name"] = "title"
    if "RecipeIngredientParts" in cols: rename_map["RecipeIngredientParts"] = "ingredient_parts"
    if "RecipeIngredientQuantities" in cols: rename_map["RecipeIngredientQuantities"] = "ingredient_qties"
    if "RecipeInstructions" in cols: rename_map["RecipeInstructions"] = "instructions"
    recipes = recipes.rename(columns=rename_map)
    assert "recipe_id" in recipes.columns, "缺 RecipeId/recipe_id"
    assert "ingredient_parts" in recipes.columns, "缺 RecipeIngredientParts/ingredient_parts"
    recipes["ingredient_list"] = recipes["ingredient_parts"].map(parse_vec)
    if "ingredient_qties" in recipes.columns:
        recipes["qty_list"] = recipes["ingredient_qties"].map(parse_vec)
    else:
        recipes["qty_list"] = [[] for _ in range(len(recipes))]
    rows = []
    for rid, ings, qts in tqdm(zip(recipes["recipe_id"], recipes["ingredient_list"], recipes["qty_list"]),
                               total=len(recipes), desc="explode ingredients"):
        if not isinstance(ings, list):
            continue
        for idx, ing in enumerate(ings):
            qty = qts[idx] if (isinstance(qts, list) and idx < len(qts)) else None
            rows.append({"recipe_id": rid, "ingredient_raw": ing, "qty_raw": qty})
    ingr_df = pd.DataFrame(rows)
    ingr_df["ingredient_norm"] = ingr_df["ingredient_raw"].map(normalize_text)
    grams_est = []
    for raw, norm in zip(ingr_df["ingredient_raw"], ingr_df["ingredient_norm"]):
        grams_est.append(estimate_grams(raw, norm))
    ingr_df["grams"] = grams_est
    return recipes, ingr_df

# ------------------------ 嵌入 & 近邻 ------------------------
def embed_texts(model, texts, batch_size=512, normalize=True):
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encoding"):
        arr = model.encode(texts[i:i+batch_size], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=normalize)
        vecs.append(arr)
    return np.vstack(vecs)

def build_faiss_index(mat):
    d = mat.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(mat.astype(np.float32))
    return idx

def topk_by_cosine(query_vecs, base_vecs, k):
    base = base_vecs / (np.linalg.norm(base_vecs, axis=1, keepdims=True) + 1e-9)
    q = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9)
    sims = q @ base.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    return idx, np.take_along_axis(sims, idx, axis=1)

# ------------------------ 对齐（并行） ------------------------
def make_candidates_for_norm(norm, inv_index, embed_ids, food_df, usda_ids, max_union=300):
    cands = set()
    for t in set(tokenize(norm)):
        if len(t) >= 3:
            cands |= inv_index.get(t, set())
    n_ids = len(usda_ids)
    for cid in embed_ids:
        try:
            ic = int(cid)
        except Exception:
            continue
        if ic < 0 or ic >= n_ids:
            continue
        cands.add(int(usda_ids[ic]))
        if len(cands) >= max_union:
            break
    return list(cands)

def rerank_one(norm, cand_ids, food_df, alpha=0.9, beta=0.1):
    if not cand_ids:
        return None
    sub = food_df.loc[food_df["fdc_id"].isin(cand_ids), ["fdc_id","description","description_norm","nutrient_coverage","dtype_bonus"]]
    if sub.empty:
        return None
    max_cov = max(1, sub["nutrient_coverage"].max())
    max_bonus = max(1, sub["dtype_bonus"].max())
    best = None
    best_total = -1
    for row in sub.itertuples(index=False):
        fscore = fuzz.token_set_ratio(norm, row.description_norm)
        cov_sc = 100.0 * (row.nutrient_coverage / max_cov)
        bonus_sc = 100.0 * (row.dtype_bonus / max_bonus)
        total = alpha * fscore + beta * (0.7 * cov_sc + 0.3 * bonus_sc)
        if total > best_total:
            best_total = total
            best = (row.fdc_id, row.description, fscore, row.nutrient_coverage, row.dtype_bonus, total)
    return best

def parallel_align(unique_norms, inv_index, food_df, embed_topk_ids, usda_ids, threads=8, min_fuzz=80, max_union=300):
    results = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = []
        for idx, norm in enumerate(unique_norms):
            embed_ids = embed_topk_ids[idx] if embed_topk_ids is not None else []
            futs.append(ex.submit(_align_one, norm, inv_index, food_df, embed_ids, usda_ids, min_fuzz, max_union))
        for fu in tqdm(as_completed(futs), total=len(futs), desc="parallel align"):
            results.append(fu.result())
    return pd.DataFrame(results)

def _align_one(norm, inv_index, food_df, embed_ids, usda_ids, min_fuzz, max_union):
    cands = make_candidates_for_norm(norm, inv_index, embed_ids, food_df, usda_ids, max_union=max_union)
    bm = rerank_one(norm, cands, food_df)
    if bm is None:
        return {"ingredient_norm": norm, "fdc_id": None, "fdc_desc": None, "fuzz": None,
                "coverage": 0, "dtype_bonus": 0, "score": 0}
    fdc_id, desc, fz, cov, bonus, total = bm
    if fz < min_fuzz:
        fdc_id, desc = None, None
    return {"ingredient_norm": norm, "fdc_id": fdc_id, "fdc_desc": desc, "fuzz": fz,
            "coverage": cov, "dtype_bonus": bonus, "score": total}

# ------------------------ 聚合营养（全营养） ------------------------
def aggregate_recipe_nutrition(ingr_df, mapping_df, fn, nutr, out_dir, default_g=40.0, write_wide=False, wide_max_nutrients=512, agg_batch_size=200000):
    m = mapping_df[["ingredient_norm","fdc_id","fdc_desc","fuzz","score"]].drop_duplicates()
    x = ingr_df.merge(m, on="ingredient_norm", how="left")
    x["grams_eff"] = x["grams"].where(x["grams"].notna(), default_g)
    x[x["fdc_id"].isna()][["ingredient_raw","ingredient_norm"]].drop_duplicates()\
      .to_csv(os.path.join(out_dir, "unmatched_ingredients.csv"), index=False, encoding="utf-8")
    mx = x[x["fdc_id"].notna()].copy()
    if mx.empty:
        raise RuntimeError("没有匹配到任何 fdc_id，请检查阈值/路径/解析。")
    mx["fdc_id"] = mx["fdc_id"].astype(int)

    # 仅保留需要的列并尽量压缩类型
    mx = mx[["recipe_id", "fdc_id", "grams_eff"]].copy()
    mx["recipe_id"] = pd.to_numeric(mx["recipe_id"], errors="coerce")
    mx["recipe_id"] = mx["recipe_id"].astype("int64")
    mx["grams_eff"] = mx["grams_eff"].astype("float32")

    fn_small = fn[["fdc_id","nutrient_id","amount"]].copy()
    fn_small["fdc_id"] = fn_small["fdc_id"].astype("int64")
    fn_small["nutrient_id"] = fn_small["nutrient_id"].astype("int64")
    fn_small["amount"] = pd.to_numeric(fn_small["amount"], errors="coerce").astype("float32")

    nutr_small = nutr[["id","name","unit_name"]].copy()
    nutr_small["id"] = nutr_small["id"].astype("int64")

    # 分批处理，避免一次性大 join / pivot 造成 OOM
    parts = []
    n = len(mx)
    for start in tqdm(range(0, n, agg_batch_size), total=math.ceil(n/agg_batch_size), desc="aggregate batches"):
        end = min(start + agg_batch_size, n)
        sub = mx.iloc[start:end].copy()
        sub = sub.merge(fn_small, on="fdc_id", how="left")
        if sub.empty:
            continue
        sub = sub.merge(nutr_small, left_on="nutrient_id", right_on="id", how="left")
        sub = sub.rename(columns={"name":"nutrient_name","unit_name":"unit"})
        # 仅保留聚合所需列
        sub = sub[["recipe_id","nutrient_name","unit","amount","grams_eff"]]
        sub["amount_for_recipe"] = (sub["amount"].astype("float32") * (sub["grams_eff"].fillna(default_g).astype("float32") / 100.0)).astype("float32")
        sub = sub.drop(columns=["amount","grams_eff"], errors="ignore")
        # 下沉为分类，减少内存
        sub["nutrient_name"] = sub["nutrient_name"].astype("category")
        sub["unit"] = sub["unit"].astype("category")
        part = sub.groupby(["recipe_id","nutrient_name","unit"], as_index=False, observed=True)["amount_for_recipe"].sum()
        parts.append(part)
        # 周期性地合并以控制 parts 列表增长
        if len(parts) >= 8:
            tmp = pd.concat(parts, ignore_index=True)
            parts = [tmp.groupby(["recipe_id","nutrient_name","unit"], as_index=False, observed=True)["amount_for_recipe"].sum()]
            gc.collect()

    if not parts:
        raise RuntimeError("聚合阶段没有生成任何数据，可能前序映射为空。")
    grp = pd.concat(parts, ignore_index=True)
    grp = grp.groupby(["recipe_id","nutrient_name","unit"], as_index=False, observed=True)["amount_for_recipe"].sum()

    # 始终写出长表，保证大数据集不被 OOM 阻断
    grp.to_parquet(os.path.join(out_dir, "recipe_nutrients_long.parquet"), index=False)

    wide = None
    if write_wide:
        # 仅在营养种类不多时生成宽表，避免 OOM
        num_nutrients = int(grp["nutrient_name"].nunique())
        if num_nutrients <= int(wide_max_nutrients):
            wide = grp.pivot_table(index="recipe_id", columns="nutrient_name",
                                   values="amount_for_recipe", aggfunc="sum").reset_index().fillna(0.0)
            wide.columns.name = None
            wide.to_parquet(os.path.join(out_dir, "recipe_nutrients_wide.parquet"), index=False)
    def pick(cols, cand):
        for c in cand:
            if c in cols: return c
        return None
    # 构建核心营养表：优先用宽表，否则从长表快速透视仅核心项
    if wide is not None:
        cols = set(wide.columns)
        energy = pick(cols, ["Energy","Energy (Atwater General Factors)","Energy (Atwater Specific Factors)"])
        protein = pick(cols, ["Protein"])
        fat    = pick(cols, ["Total lipid (fat)"])
        carbs  = pick(cols, ["Carbohydrate, by difference"])
        core = wide[["recipe_id"]].copy()
        core["calories_kcal"] = wide[energy] if energy else 0.0
        core["protein_g"]     = wide[protein] if protein else 0.0
        core["fat_g"]         = wide[fat] if fat else 0.0
        core["carbohydrates_g"] = wide[carbs] if carbs else 0.0
    else:
        target = grp[grp["nutrient_name"].isin([
            "Energy","Energy (Atwater General Factors)","Energy (Atwater Specific Factors)",
            "Protein","Total lipid (fat)","Carbohydrate, by difference"
        ])].copy()
        core_wide = target.pivot_table(index="recipe_id", columns="nutrient_name",
                                       values="amount_for_recipe", aggfunc="sum").reset_index().fillna(0.0)
        core_wide.columns.name = None
        cols = set(core_wide.columns)
        energy = pick(cols, ["Energy","Energy (Atwater General Factors)","Energy (Atwater Specific Factors)"])
        protein = pick(cols, ["Protein"])
        fat    = pick(cols, ["Total lipid (fat)"])
        carbs  = pick(cols, ["Carbohydrate, by difference"])
        core = core_wide[["recipe_id"]].copy()
        core["calories_kcal"] = core_wide[energy] if energy else 0.0
        core["protein_g"]     = core_wide[protein] if protein else 0.0
        core["fat_g"]         = core_wide[fat] if fat else 0.0
        core["carbohydrates_g"] = core_wide[carbs] if carbs else 0.0
    core.to_csv(os.path.join(out_dir, "recipe_nutrients_core.csv"), index=False)
    return wide, core, x, mx

# ------------------------ 加载 USDA 和 FNDDS ------------------------
def load_usda_and_fndds(usda_dir, fndds_dir, filter_types=None):
    """
    同时加载 USDA 和 FNDDS 数据源，返回两个数据源的食物、营养信息。
    """
    # 以精简列和 dtype 方式加载，降低内存
    food_cols = ["fdc_id","description","data_type"]
    nutr_cols = ["id","name","unit_name"]
    fn_cols   = ["fdc_id","nutrient_id","amount"]
    food_usda = pd.read_csv(os.path.join(usda_dir, "food.csv"), usecols=food_cols,
                            dtype={"fdc_id":"int64","description":"string","data_type":"string"})
    nutr_usda = pd.read_csv(os.path.join(usda_dir, "nutrient.csv"), usecols=nutr_cols,
                            dtype={"id":"int64","name":"string","unit_name":"string"})
    fn_usda   = pd.read_csv(os.path.join(usda_dir, "food_nutrient.csv"), usecols=fn_cols,
                            dtype={"fdc_id":"int64","nutrient_id":"int64","amount":"float32"}, low_memory=False)

    # 加载 FNDDS 数据（相同列约束）
    food_fndds = pd.read_csv(os.path.join(fndds_dir, "food.csv"), usecols=food_cols,
                              dtype={"fdc_id":"int64","description":"string","data_type":"string"})
    nutr_fndds = pd.read_csv(os.path.join(fndds_dir, "nutrient.csv"), usecols=nutr_cols,
                              dtype={"id":"int64","name":"string","unit_name":"string"})
    fn_fndds   = pd.read_csv(os.path.join(fndds_dir, "food_nutrient.csv"), usecols=fn_cols,
                              dtype={"fdc_id":"int64","nutrient_id":"int64","amount":"float32"}, low_memory=False)

    # 检查列名
    assert {"fdc_id","description","data_type"}.issubset(set(food_usda.columns)), "USDA food.csv 缺少必要列"
    assert {"id","name","unit_name"}.issubset(set(nutr_usda.columns)), "USDA nutrient.csv 缺少必要列"
    assert {"fdc_id","nutrient_id","amount"}.issubset(set(fn_usda.columns)), "USDA food_nutrient.csv 缺少必要列"
    
    assert {"fdc_id","description","data_type"}.issubset(set(food_fndds.columns)), "FNDDS food.csv 缺少必要列"
    assert {"id","name","unit_name"}.issubset(set(nutr_fndds.columns)), "FNDDS nutrient.csv 缺少必要列"
    assert {"fdc_id","nutrient_id","amount"}.issubset(set(fn_fndds.columns)), "FNDDS food_nutrient.csv 缺少必要列"

    # 合并 USDA 和 FNDDS
    food = pd.concat([food_usda, food_fndds], ignore_index=True)
    nutr = pd.concat([nutr_usda, nutr_fndds], ignore_index=True).drop_duplicates(subset=["id"])
    fn   = pd.concat([fn_usda, fn_fndds], ignore_index=True).drop_duplicates(subset=["fdc_id", "nutrient_id"])

    # 数据清洗
    food["description_norm"] = food["description"].astype(str).map(normalize_text)
    food["tokens"] = food["description_norm"].map(lambda s: [t for t in tokenize(s) if len(t)>=3])

    # 统计每个 fdc_id 的“营养覆盖度”（用于重排，尽量选覆盖多的条目）
    coverage = fn.groupby("fdc_id")["nutrient_id"].nunique().rename("nutrient_coverage")
    food = food.merge(coverage, on="fdc_id", how="left")
    food["nutrient_coverage"] = food["nutrient_coverage"].fillna(0).astype(int)

    # 按 data_type 给点偏置：Foundation / SR Legacy / Survey(FNDDS) 更通用
    def dtype_bonus(x):
        x = str(x).lower()
        if "foundation" in x: return 10
        if "sr legacy" in x:  return 8
        if "survey" in x:     return 7  # FNDDS
        if "sample" in x:     return 4
        if "market" in x:     return 3
        if "branded" in x:    return 1
        return 0
    food["dtype_bonus"] = food["data_type"].map(dtype_bonus)

    # 按规范化描述去重：优先覆盖度高、dtype_bonus 高的记录，减少冗余
    if not food.empty:
        food = food.sort_values(["description_norm","nutrient_coverage","dtype_bonus"], ascending=[True, False, False])
        food = food.drop_duplicates(subset=["description_norm"], keep="first").reset_index(drop=True)

    # 构建倒排索引 token -> 候选 fdc_id（减少模糊匹配的候选集合）
    inv_index = defaultdict(set)
    for fdc_id, toks in zip(food["fdc_id"].tolist(), food["tokens"].tolist()):
        for t in toks:
            if len(t) >= 3:
                inv_index[t].add(int(fdc_id))

    return food, nutr, fn, inv_index

# ------------------------ 主流程 ------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 加载 USDA 和 FNDDS
    print(">> Loading USDA and FNDDS ...")
    food, nutr, fn, inv_index = load_usda_and_fndds(args.usda_dir, args.fndds_dir, args.filter_usda_types)

    # 2) Food.com
    print(">> Loading Food.com ...")
    recipes, ingr_df = load_foodcom(args.recipes, args.reviews)

    # 3) 嵌入（USDA / FNDDS / ingredients）
    print(">> Building embeddings ...")
    model = SentenceTransformer(args.embed_model)  # 'all-MiniLM-L6-v2'
    # USDA 描述向量（缓存到 out_dir 以复用）
    cache_vec = os.path.join(args.out_dir, "usda_food_desc_emb.npy")
    cache_ids = os.path.join(args.out_dir, "usda_food_desc_ids.npy")
    if os.path.exists(cache_vec) and os.path.exists(cache_ids):
        usda_vec = np.load(cache_vec, mmap_mode="r")
        usda_ids = np.load(cache_ids)
    else:
        usda_texts = food["description_norm"].astype(str).tolist()
        usda_vec = embed_texts(model, usda_texts, batch_size=args.embed_bs, normalize=True).astype(np.float32)
        usda_ids = food["fdc_id"].values.astype(np.int64)
        np.save(cache_vec, usda_vec); np.save(cache_ids, usda_ids)

    # Faiss 或 Numpy 近邻
    if HAS_FAISS:
        # 可选：使用 HNSW 以减少计算负担（在 faiss 可用时）
        try:
            d = usda_vec.shape[1]
            m = getattr(args, "faiss_hnsw_m", 32)
            efc = getattr(args, "faiss_hnsw_ef_construct", 100)
            ef = getattr(args, "faiss_hnsw_ef_search", 50)
            index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = efc
            index.hnsw.efSearch = ef
            index.add(usda_vec.astype(np.float32))
        except Exception:
            index = build_faiss_index(usda_vec)
    else:
        index = None
        warnings.warn("未安装 faiss-cpu，将使用 numpy 余弦近邻（更慢）。")

    uniq_norms = ingr_df["ingredient_norm"].dropna().unique().tolist()
    ing_vec = embed_texts(model, uniq_norms, batch_size=args.embed_bs, normalize=True).astype(np.float32)
    if HAS_FAISS:
        D, I = index.search(ing_vec, args.topk_embed)    # I: [n, topk]→ usda_vec 行号
    else:
        I, D = topk_by_cosine(ing_vec, usda_vec, args.topk_embed)  # I 索引，D 相似度

    # 将向量检索的“行号”映射回 USDA fdc_id 的“DataFrame 行索引位置”
    embed_topk_idx = I  # shape: [len(uniq_norms), topk]

    # 4) 并行对齐（embedding 候选 ∪ token 候选 → fuzzy+coverage 重排）
    print(">> Parallel aligning ...")
    mapping_df = parallel_align(
        unique_norms=uniq_norms,
        inv_index=inv_index,
        food_df=food,
        embed_topk_ids=embed_topk_idx,
        usda_ids=usda_ids,
        threads=args.threads,
        min_fuzz=args.min_fuzz,
        max_union=args.max_candidates_union
    )
    mapping_df.to_parquet(os.path.join(args.out_dir, "ingredient_mapping.parquet"), index=False)

    # 5) 聚合营养（保留全部营养）
    print(">> Aggregating nutrition ...")
    wide, core, exploded_ing, matched_ing = aggregate_recipe_nutrition(
        ingr_df, mapping_df, fn, nutr, args.out_dir,
        default_g=args.default_grams,
        write_wide=args.write_wide,
        wide_max_nutrients=args.wide_max_nutrients,
        agg_batch_size=args.agg_batch_size
    )

    # 6) 合并菜谱基本信息
    if "title" in recipes.columns:
        rcore = recipes[["recipe_id","title"]].drop_duplicates()
    else:
        rcore = recipes[["recipe_id"]].drop_duplicates()
    merged = rcore.merge(core, on="recipe_id", how="left")
    merged.to_parquet(os.path.join(args.out_dir, "recipe_with_nutrition.parquet"), index=False)

    # 7) 统计
    stats = {
        "n_recipes": int(recipes["recipe_id"].nunique()),
        "n_unique_ingredients_norm": int(len(uniq_norms)),
        "map_rate_unique_norm_%": float(100.0 * mapping_df["fdc_id"].notna().mean()),
        "default_grams_used_if_none": args.default_grams,
        "topk_embed": args.topk_embed,
        "min_fuzz": args.min_fuzz,
        "threads": args.threads
    }
    with open(os.path.join(args.out_dir, "logs_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(">> DONE.")
    print(f"   Unique ingredient match rate: {stats['map_rate_unique_norm_%']:.2f}%")
    print(f"   Outputs in: {args.out_dir}")
    print("   - ingredient_mapping.parquet")
    print("   - unmatched_ingredients.csv")
    print("   - recipe_nutrients_wide.parquet")
    print("   - recipe_nutrients_core.csv")
    print("   - recipe_with_nutrition.parquet")
    print("   - logs_stats.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--usda_dir", required=True, help="USDA CSV 目录（food.csv / nutrient.csv / food_nutrient.csv）")
    p.add_argument("--fndds_dir", required=True, help="FNDDS CSV 目录（food.csv / nutrient.csv / food_nutrient.csv）")
    p.add_argument("--recipes", required=True, help="Food.com recipes.parquet 路径")
    p.add_argument("--reviews", default=None, help="Food.com reviews.parquet 路径（可选）")
    p.add_argument("--out_dir", required=True, help="输出目录")
    p.add_argument("--filter_usda_types", default=None, help="可选：仅使用某些 data_type（逗号分隔，如 foundation,sr legacy,survey,branded）")
    p.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer 模型")
    p.add_argument("--embed_bs", type=int, default=512, help="嵌入 batch size")
    p.add_argument("--topk_embed", type=int, default=120, help="嵌入检索候选数")
    p.add_argument("--min_fuzz", type=int, default=80, help="模糊阈值（<此阈视为未匹配）")
    p.add_argument("--threads", type=int, default=8, help="并行线程数（RapidFuzz 释放 GIL，线程可加速）")
    p.add_argument("--max_candidates_union", type=int, default=300, help="候选集合上限（token ∪ embed）")
    p.add_argument("--default_grams", type=float, default=30.0, help="当无法估算克重时的保守默认值")
    p.add_argument("--write_wide", action="store_true", help="在营养种类不多时同时写出宽表")
    p.add_argument("--wide_max_nutrients", type=int, default=512, help="写宽表的营养种类上限阈值")
    p.add_argument("--agg_batch_size", type=int, default=200000, help="聚合阶段批大小，降低内存峰值")
    ARGS = p.parse_args()
    main(ARGS)
