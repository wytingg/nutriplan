# -*- coding: utf-8 -*-
"""
augment_recipe_nutrients.py — v2.2 (ID类型修复+对齐诊断)
- 关键改动：将营养表与元数据表的 id 强制标准化为“整数字符串”，避免 38.0 vs 38 的合并失败
- 额外输出：左右表键集合大小、交集大小、左右差集示例

用法：
python work/recipebench/scripts/rawdataprocess/augment_recipe_nutrients.py \
  --nutr-csv work/recipebench/data/out/recipe_nutrients_core_expanded.csv \
  --meta-path work/recipebench/data/raw/foodcom/recipes.parquet \
  --out-prefix work/recipebench/data/out/recipe_nutrient_full \
  --round2
"""
import argparse, json, os, sys, re
from typing import List, Any
import pandas as pd
import numpy as np

NUTR_COLS = [
    "calories_kcal", "protein_g", "fat_g", "carbohydrates_g",
    "sugar_g", "sodium_mg", "calcium_mg", "iron_mg"
]

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def choose_first(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return ""

def to_list_safe(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(t) for t in x]
    if isinstance(x, np.ndarray):
        return [str(t) for t in x.tolist()]
    if isinstance(x, pd.Series):
        return [str(t) for t in x.to_list()]
    s = str(x).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s.replace("(", "[").replace(")", "]"))
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except Exception:
            pass
    if ";" in s:
        return [t.strip() for t in s.split(";")]
    return [t.strip() for t in s.split(",")]

def clean_ingredients_list(items: List[str]) -> List[str]:
    cleaned, seen = [], set()
    for t in items:
        if not t:
            continue
        t2 = str(t)
        for l, r in [("（", "）"), ("(", ")")]:
            if l in t2 and r in t2:
                t2 = t2.split(l)[0].strip()
        for bad in ("适量", "少许"):
            t2 = t2.replace(bad, "").strip()
        if t2 and t2 not in seen:
            seen.add(t2)
            cleaned.append(t2)
    return cleaned

def coerce_ingredients(series: pd.Series) -> pd.DataFrame:
    joins, jsons = [], []
    for v in series:
        items = to_list_safe(v)
        items = clean_ingredients_list(items)
        joins.append(";".join(items))
        jsons.append(json.dumps(items, ensure_ascii=False))
    return pd.DataFrame({"ingredients": joins, "ingredients_json": jsons})

_ID_TRAIL_ZERO = re.compile(r"\.0+$")

def normalize_id_to_intstr(s: pd.Series) -> pd.Series:
    """
    将任意 id 列强制规整为“整数字符串”：
    - 38.0 / 38.000 / 3.4e+02 → "38" / "38" / "340"
    - 保留非数字的原字符串（极少出现）
    """
    # 先转字符串，去空格
    out = s.astype(str).str.strip()

    # 去掉明显的 .0/.000 尾巴（先做一次快速清洗）
    out = out.str.replace(_ID_TRAIL_ZERO, "", regex=True)

    # 尝试按数字解析（处理科学计数法、真实 float）
    num = pd.to_numeric(out, errors="coerce")
    mask = num.notna()
    if mask.any():
        # 四舍五入到整数（Food.com 的 RecipeId 应为整数主键）
        as_int = num.round(0).astype("Int64")
        out.loc[mask] = as_int.astype(str)

    # 再做一次保险性的 .0 去尾（应当已经没有）
    out = out.str.replace(_ID_TRAIL_ZERO, "", regex=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nutr-csv", required=True, help="包含 rid/recipe_id + 8项营养 的 CSV/Parquet")
    ap.add_argument("--meta-path", required=True, help="包含 RecipeId/Name/RecipeIngredientParts 的 CSV/Parquet")
    ap.add_argument("--out-prefix", required=True, help="输出前缀（不含扩展名）")
    ap.add_argument("--round2", action="store_true", help="将营养数值四舍五入到小数点后两位")
    ap.add_argument("--inspect", action="store_true", help="仅打印两表列名并退出")
    args = ap.parse_args()

    nutr = read_any(args.nutr_csv)
    meta = read_any(args.meta_path)
    nutr = _norm_cols(nutr)
    meta = _norm_cols(meta)

    if args.inspect:
        print("[营养表列名]", list(nutr.columns))
        print("[元数据列名]", list(meta.columns))
        return

    # === 列映射 ===
    nutr_id_col = choose_first(nutr, ["rid", "recipe_id", "id"])
    if nutr_id_col == "":
        raise ValueError(f"[营养表缺少 rid/recipe_id] 现有列: {list(nutr.columns)}")
    miss_nutr = [c for c in NUTR_COLS if c not in nutr.columns]
    if miss_nutr:
        raise ValueError(f"[营养表缺少营养列] {miss_nutr}；现有列: {list(nutr.columns)}")

    meta_id_col = choose_first(meta, ["recipeid", "rid", "id", "recipe_id"])
    meta_title_col = choose_first(meta, ["name", "title", "recipe_name", "title_zh", "title_en"])
    meta_ing_col = choose_first(meta, ["recipeingredientparts", "ingredients", "ingredients_text", "ingredients_raw"])

    missing_meta = []
    if meta_id_col == "":    missing_meta.append("RecipeId/rid")
    if meta_title_col == "": missing_meta.append("Name/title")
    if meta_ing_col == "":   missing_meta.append("RecipeIngredientParts/ingredients")
    if missing_meta:
        raise ValueError(f"[元数据缺列] {missing_meta}；现有列: {list(meta.columns)}")

    # === 子集与重命名 ===
    nutr_small = nutr[[nutr_id_col] + NUTR_COLS].copy()
    nutr_small = nutr_small.rename(columns={nutr_id_col: "rid"})

    meta_small = meta[[meta_id_col, meta_title_col, meta_ing_col]].copy()
    meta_small = meta_small.rename(columns={
        meta_id_col: "rid", meta_title_col: "title", meta_ing_col: "ingredients_raw"
    })

    # === 关键：ID 标准化为“整数字符串”并去重 ===
    nutr_small["rid"] = normalize_id_to_intstr(nutr_small["rid"])
    meta_small["rid"] = normalize_id_to_intstr(meta_small["rid"])

    nutr_small = nutr_small.dropna(subset=["rid"])
    meta_small = meta_small.dropna(subset=["rid"])

    nutr_small = nutr_small.drop_duplicates(subset=["rid"], keep="last")
    meta_small = meta_small.drop_duplicates(subset=["rid"], keep="last")

    # === 诊断：左右键集合与交集 ===
    left_keys = set(nutr_small["rid"].tolist())
    right_keys = set(meta_small["rid"].tolist())
    inter = left_keys & right_keys
    print("==== 键诊断 ====")
    print(f"营养键数: {len(left_keys)} | 元数据键数: {len(right_keys)} | 交集: {len(inter)}")
    if len(inter) == 0:
        # 打印各自前5个样例，辅助人工检查键格式
        print("[样例-营养键-前5]", list(sorted(list(left_keys))[:5]))
        print("[样例-元数据键-前5]", list(sorted(list(right_keys))[:5]))

    # === ingredients 处理 ===
    meta_small["title"] = meta_small["title"].astype(str).str.strip()
    ing_df = coerce_ingredients(meta_small["ingredients_raw"])
    meta_small = pd.concat([meta_small.drop(columns=["ingredients_raw"]), ing_df], axis=1)

    # === 合并 ===
    merged = pd.merge(nutr_small, meta_small, on="rid", how="left")

    # 数值列
    for c in NUTR_COLS:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        if args.round2:
            merged[c] = merged[c].round(2)

    # 统计
    unmatched = merged[merged["title"].isna() | merged["ingredients"].isna()].copy()
    total = int(len(merged))
    have_meta = int(total - len(unmatched))

    print("==== 映射摘要 ====")
    print(f"[营养表] id: {nutr_id_col} → rid")
    print(f"[元数据] id: {meta_id_col} → rid, title: {meta_title_col} → title, ingredients: {meta_ing_col} → ingredients")
    print("==== 合并统计 ====")
    print(f"总记录: {total} | 补齐元数据: {have_meta} | 缺失: {int(len(unmatched))}")

    if len(unmatched) > 0:
        # 再输出少量缺失样例的 rid，便于逆向检查
        print("[缺失样例 rid - 前10]", merged.loc[merged["title"].isna(), "rid"].head(10).tolist())
        un_path = f"{args.out_prefix}_unmatched.csv"
        unmatched.to_csv(un_path, index=False)
        print(f"[WARN] 未匹配到 title/ingredients 的 {int(len(unmatched))} 条 → {un_path}")

    # 导出
    out_cols = ["rid", "title", "ingredients", "ingredients_json"] + NUTR_COLS
    if "ingredients_json" not in merged.columns:
        merged["ingredients_json"] = "[]"
    merged = merged.reindex(columns=out_cols)

    csv_path = f"{args.out_prefix}.csv"
    pq_path = f"{args.out_prefix}.parquet"
    merged.to_csv(csv_path, index=False)
    merged.to_parquet(pq_path, index=False)
    print(f"[SAVE] {csv_path}")
    print(f"[SAVE] {pq_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
