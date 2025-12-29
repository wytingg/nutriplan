#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_parsed_ingredients_sample.py

功能：
1) 解析并对齐 RecipeIngredientParts 与 RecipeIngredientQuantities，修复 zip 为 0 的问题；
2) A：合并“包装×规格”的数量（如 `2 (14 ounce) cans diced tomatoes` → 补出 28 oz）；
3) B：允许 `to taste / optional / divided` 无数量，但不剔除该配料；
4) C：统一顶层切分（括号内逗号不切）以减少长度不匹配；
5) D：名称规范化为 `ingredient_norm`，并按该列与映射表 merge（列：ingredient_norm, fdc_id, fdc_desc, fuzz, coverage, dtype_bonus, score）；
6) 输出采样 CSV 与诊断信息。
"""

import argparse
import ast
import json
import re
import sys
from typing import Any, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# 通用工具 & 全局字典
# =========================

def smart_top_level_split(s: str) -> list[str]:
    """括号感知的顶层分隔：只在 depth=0 的逗号/分号处分割。"""
    if not isinstance(s, str):
        return [s]
    out, buf, depth = [], [], 0
    for ch in s:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in [',', ';'] and depth == 0:
            seg = "".join(buf).strip()
            if seg:
                out.append(seg)
            buf = []
        else:
            buf.append(ch)
    seg = "".join(buf).strip()
    if seg:
        out.append(seg)
    return out


def safe_to_list(x: Any) -> List[Any]:
    """把单元格值统一为 list（含顶层切分）。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (int, float, bool)):
        return [x]
    if isinstance(x, str):
        s = x.strip()
        # 1) Python/JSON 列表字面量
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            for parser in (ast.literal_eval, json.loads):
                try:
                    y = parser(s)
                    if isinstance(y, (list, tuple, np.ndarray)):
                        return list(y)
                except Exception:
                    pass
        # 2) 顶层切分（括号内逗号不切）
        parts = smart_top_level_split(s)
        return parts if parts else [s]
    return [str(x)]


def align_lengths(parts: List[Any], qtys: List[Any]) -> Tuple[List[Any], List[Any], str]:
    """对齐两列长度。"""
    lp, lq = len(parts), len(qtys)
    if lp == 0 or lq == 0:
        return [], [], "empty"
    if lp == lq:
        return parts, qtys, "ok"
    if lp == 1 and lq > 1:
        return parts * lq, qtys, "broadcast_parts"
    if lq == 1 and lp > 1:
        return parts, qtys * lp, "broadcast_qty"
    m = min(lp, lq)
    return parts[:m], qtys[:m], "truncate"


# ==== 名称规范化（D） ====

STOP_ADJ = set("""
fresh organic large small medium extra-virgin extra virgin boneless skinless ripe raw cooked finely coarsely
""".strip().split())

def normalize_ingredient_name(s: str) -> str:
    """轻量规范化：小写、去括号内容/特殊标记/修饰词、弱化复数。"""
    if not isinstance(s, str) or not s.strip():
        return ""
    x = s.lower()
    # 去掉“包装×规格”提示
    x = re.sub(r"\b\d+\s*\([\d\.]+\s*(?:oz|ounce|ounces|g|gram|grams|ml|milliliters?|l|liters?)\)\s*(?:cans?|packages?|pkgs?|tins?|bottles?|bags?)\b", " ", x)
    x = re.sub(r"\b\d+\s*(?:x|\*)\s*[\d\.]+\s*(?:oz|g|ml|l)\b", " ", x)
    # 去括号内容
    x = re.sub(r"\([^)]*\)", " ", x)
    # 去特殊标记
    x = re.sub(r"\b(to taste|divided|optional)\b", " ", x)
    # 去非字母数字
    x = re.sub(r"[^a-z0-9\s\-]", " ", x)
    # 切词、去修饰词、弱化复数
    toks = [t for t in re.split(r"\s+", x) if t]
    toks = [t for t in toks if t not in STOP_ADJ]
    def _singular(t: str) -> str:
        return t[:-1] if len(t) > 3 and t.endswith('s') else t
    toks = [_singular(t) for t in toks]
    norm = " ".join(toks)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


# ==== 单位规范化字典（基础+扩展）====
UNIT_NORM = {
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "g": "g", "gram": "g", "grams": "g",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml",
    "l": "l", "liter": "l", "liters": "l",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "cup": "cup", "cups": "cup",
    "pt": "pt", "pint": "pt", "pints": "pt",
    "qt": "qt", "quart": "qt", "quarts": "qt",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "kg": "kg",
    "mg": "mg"
}


# =========================
# 数量解析：分数、范围、前置（A 的补充）
# =========================

UNICODE_FRAC = {
    "¼": 1/4, "½": 1/2, "¾": 3/4,
    "⅐": 1/7, "⅑": 1/9, "⅒": 1/10, "⅓": 1/3, "⅔": 2/3, "⅕": 1/5, "⅖": 2/5, "⅗": 3/5, "⅘": 4/5, "⅙": 1/6,
    "⅚": 5/6, "⅛": 1/8, "⅜": 3/8, "⅝": 5/8, "⅞": 7/8
}

def parse_fraction_token(tok: str) -> Optional[float]:
    tok = tok.strip()
    if not tok:
        return None
    if tok in UNICODE_FRAC:
        return UNICODE_FRAC[tok]
    if "/" in tok:
        try:
            a, b = tok.split("/", 1)
            return float(a) / float(b)
        except Exception:
            return None
    try:
        return float(tok)
    except Exception:
        return None

def parse_mixed_number(s: str) -> Optional[float]:
    """解析 '1 1/2'、'1½'、'½'、'2.5' 到 float。"""
    s = s.strip()
    if not s:
        return None
    m = re.match(r"^\s*(\d+)\s+(\d+/\d+|[{}])\s*$".format("".join(UNICODE_FRAC.keys())), s)
    if m:
        whole = float(m.group(1))
        frac = parse_fraction_token(m.group(2))
        return whole + (frac or 0.0)
    m = re.match(r"^\s*(\d+)\s*([{}])\s*$".format("".join(UNICODE_FRAC.keys())), s)
    if m:
        whole = float(m.group(1))
        frac = UNICODE_FRAC.get(m.group(2), 0.0)
        return whole + frac
    f = parse_fraction_token(s)
    return f

def extract_leading_qty(ing: str) -> Tuple[Optional[float], Optional[str]]:
    """
    从配料行起始抽取数量（含范围），返回 (value, unit)；失败则 (None, None)
    支持：
      - "1 1/2 cups ..." / "½ cup ..." / "2 tbsp ..."
      - "1–2 tbsp" / "1-2 tbsp" / "1 to 2 tbsp"
      - "1 lb ..." / "250 g ..." / "0.5 kg ..."
    """
    if not isinstance(ing, str) or not ing.strip():
        return (None, None)
    s = ing.strip().lower()

    # 范围：1–2 / 1-2 / 1 to 2
    rng = re.match(r"""^\s*
        (?P<a>[\d\./{}]+|\d+\s+[{}])
        \s*(?:to|–|-)\s*
        (?P<b>[\d\./{}]+|\d+\s+[{}])
        \s*(?P<u>[a-z\.]+)?
    """.format("".join(UNICODE_FRAC.keys()), "".join(UNICODE_FRAC.keys()),
               "".join(UNICODE_FRAC.keys()), "".join(UNICODE_FRAC.keys())), s, re.X)
    if rng:
        a = parse_mixed_number(rng.group("a"))
        b = parse_mixed_number(rng.group("b"))
        u = rng.group("u")
        if a is not None and b is not None:
            mid = (a + b) / 2.0
            u = UNIT_NORM.get(u, u) if u else None
            return (mid, u)

    # 普通前置数值 + 单位
    m = re.match(r"""^\s*
        (?P<n1>\d+(?:\s+\d+/\d+)?|[{}]|\d+/\d+)
        \s*
        (?P<u>[a-z\.]+)?
    """.format("".join(UNICODE_FRAC.keys())), s, re.X)
    if m:
        n1 = m.group("n1")
        u  = m.group("u")
        val = parse_mixed_number(n1)
        if val is not None:
            u = UNIT_NORM.get(u, u) if u else None
            return (val, u)

    # 质量单位裸值："250 g", "1 lb"
    m = re.match(r"""^\s*
        (?P<n>\d+(?:\.\d+)?)
        \s*(?P<u>oz|g|kg|mg|lb|lbs)\b
    """, s)
    if m:
        val = float(m.group("n"))
        u = UNIT_NORM.get(m.group("u"), m.group("u"))
        return (val, u)

    return (None, None)


# =========================
# 文本数量推断（A）：包装×规格 + 前置/范围
# =========================

PKG_WORDS = r"(?:cans?|packages?|pkgs?|tins?|bottles?|bags?)"

def infer_qty_from_text(ing: str) -> Tuple[Optional[float], Optional[str]]:
    """从 ingredient_raw 文本里推测合并数量（值, 单位）。"""
    if not isinstance(ing, str) or not ing.strip():
        return (None, None)
    s = ing.lower()

    # 1) count * (size unit) pkg
    m = re.search(r"\b(?P<count>\d+)\s*\((?P<size>[\d\.]+)\s*(?P<u>oz|ounce|ounces|g|gram|grams|ml|milliliters?|l|liters?)\)\s*" + PKG_WORDS + r"\b", s)
    if m:
        count = float(m.group("count"))
        size = float(m.group("size"))
        u = UNIT_NORM.get(m.group("u"), None)
        if u:
            return (count * size, u)

    # 2) count x size unit
    m = re.search(r"\b(?P<count>\d+)\s*(?:x|\*)\s*(?P<size>[\d\.]+)\s*(?P<u>oz|g|ml|l)\b", s)
    if m:
        count = float(m.group("count"))
        size = float(m.group("size"))
        u = UNIT_NORM.get(m.group("u"), None)
        if u:
            return (count * size, u)

    # 3) 单规格："14-ounce can ..."
    m = re.search(r"\b(?P<size>[\d\.]+)\s*-\s*(?P<u>oz|ounce|ounces|g|gram|grams|ml|milliliters?|l|liters?)\s*" + PKG_WORDS + r"\b", s)
    if m:
        size = float(m.group("size"))
        u = UNIT_NORM.get(m.group("u"), None)
        if u:
            return (size, u)

    return (None, None)


def prefill_qtys_by_text(parts: List[str], qtys: List[Any]) -> List[Any]:
    """若 parts 更长：先用包装×规格补；未命中再用前置/范围数量补。"""
    q = list(qtys)
    if len(parts) > len(q):
        for i in range(len(q), len(parts)):
            val, u = infer_qty_from_text(parts[i])
            if val is None:
                val, u = extract_leading_qty(parts[i])
            if val is not None:
                q.append(f"{val:g} {u}" if u else f"{val:g}")
            else:
                q.append("")
    return q


# =========================
# 特殊标记清洗（B）
# =========================

def clean_special_flags(ingredient_raw: str, quantity_raw: Any) -> Tuple[str, str]:
    """处理 to taste / optional / divided：允许无数量，清理提示词但保留配料。"""
    ir = ingredient_raw if isinstance(ingredient_raw, str) else str(ingredient_raw)
    qr = (quantity_raw if isinstance(quantity_raw, str)
          else ("" if pd.isna(quantity_raw) else str(quantity_raw)))
    s = ir.lower()
    if "to taste" in s or "optional" in s:
        if qr.strip() == "":
            qr = ""
        ir = ir.replace("to taste", "").replace("optional", "")
    ir = ir.replace("divided", "")
    return ir.strip(), qr.strip()


# =========================
# （可选）质量估算：仅质量单位直接转克
# =========================

OZ_TO_G = 28.349523125

def qty_str_to_grams(q: str) -> Optional[float]:
    if not isinstance(q, str) or not q.strip():
        return None
    m = re.match(r"^\s*([\d\.]+)\s*(oz|g|kg|mg|lb)\s*$", q.lower())
    if not m:
        return None
    val = float(m.group(1)); u = m.group(2)
    if u == "g":   return val
    if u == "oz":  return val * OZ_TO_G
    if u == "kg":  return val * 1000.0
    if u == "mg":  return val / 1000.0
    if u == "lb":  return val * 453.59237
    return None


# =========================
# 主流程
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", required=True)
    ap.add_argument("--ingredient_mapping", required=False, default=None)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sample_n", type=int, default=8000)
    ap.add_argument("--ingredients_col", required=True)
    ap.add_argument("--quantities_col", required=True)
    ap.add_argument("--id_col", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.recipes, engine="pyarrow")
    needed = [args.id_col, args.ingredients_col, args.quantities_col]
    for col in needed:
        if col not in df.columns:
            print(f"[ERROR] Missing column in recipes: {col}", file=sys.stderr)
            sys.exit(2)

    df = df[needed].copy()
    has_any_ings = df[args.ingredients_col].notna().sum()

    # 统一转列表（含顶层切分）
    parts_list = df[args.ingredients_col].map(safe_to_list)
    qtys_list  = df[args.quantities_col].map(safe_to_list)

    # 逐行对齐（先文本推断补 qty，再对齐）
    strategy_counts = {"ok": 0, "broadcast_parts": 0, "broadcast_qty": 0, "truncate": 0, "empty": 0}
    pairs_col: List[List[Tuple[Any, Any]]] = []

    for p, q in zip(parts_list, qtys_list):
        q = prefill_qtys_by_text(p, q)
        pp, qq, strat = align_lengths(p, q)
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        pairs_col.append(list(zip(pp, qq)) if len(pp) else [])

    df_pairs = pd.DataFrame({args.id_col: df[args.id_col].values, "pairs": pairs_col})

    # 统计 zipped / exploded
    zipped_count = int(df_pairs["pairs"].map(len).sum())
    exploded = df_pairs.explode("pairs", ignore_index=True)
    if exploded["pairs"].notna().any():
        exploded[["ingredient_raw", "quantity_raw"]] = pd.DataFrame(
            exploded["pairs"].tolist(), index=exploded.index
        )
    else:
        exploded["ingredient_raw"] = None
        exploded["quantity_raw"] = None
    exploded.drop(columns=["pairs"], inplace=True)

    # B：特殊标记清洗
    if len(exploded):
        exploded[["ingredient_raw", "quantity_raw"]] = exploded.apply(
            lambda r: pd.Series(clean_special_flags(r.get("ingredient_raw"), r.get("quantity_raw"))),
            axis=1
        )

    # D：规范化名称 & 与映射表 merge（基于 ingredient_norm）
    exploded["ingredient_norm"] = exploded["ingredient_raw"].fillna("").map(normalize_ingredient_name)

    if args.ingredient_mapping:
        try:
            df_map = pd.read_parquet(args.ingredient_mapping, engine="pyarrow")
            if "ingredient_norm" in df_map.columns:
                map_small = (df_map[["ingredient_norm", "fdc_id", "fdc_desc", "fuzz", "coverage", "dtype_bonus", "score"]]
                             .dropna(subset=["ingredient_norm"])
                             .drop_duplicates("ingredient_norm"))
                exploded = exploded.merge(map_small, how="left", on="ingredient_norm")
            else:
                print("[WARN] ingredient_mapping 缺少列 'ingredient_norm'，请检查映射表列名。", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] 读取 ingredient_mapping 失败，已跳过映射。err={e}", file=sys.stderr)

    # （可选）估算质量（仅 g/oz/kg/mg/lb）
    exploded["quantity_g_est"] = exploded["quantity_raw"].map(qty_str_to_grams)

    exploded_lines = int(len(exploded))

    # 采样输出
    if exploded_lines == 0:
        out_cols = [args.id_col, "ingredient_raw", "quantity_raw", "ingredient_norm",
                    "fdc_id", "fdc_desc", "fuzz", "coverage", "dtype_bonus", "score", "quantity_g_est"]
        for c in out_cols:
            if c not in exploded.columns:
                exploded[c] = pd.Series(dtype="object")
        exploded[out_cols].head(0).to_csv(args.out_csv, index=False)
        wrote_n = 0
    else:
        sample_n = min(args.sample_n, exploded_lines)
        sample_df = exploded.sample(n=sample_n, random_state=42) if sample_n > 0 else exploded.head(0)
        sample_df.to_csv(args.out_csv, index=False)
        wrote_n = sample_n

    # 打印诊断
    print(f"[diagnostic] recipes: {len(df)}"
          f", with any ingredients: {has_any_ings}"
          f", zipped qty+parts rows: {zipped_count}"
          f", exploded lines: {exploded_lines}")
    print("[align summary] " + " | ".join([f"{k}: {v}" for k, v in strategy_counts.items()]))
    print(f"Wrote {args.out_csv} with rows: {wrote_n}")


if __name__ == "__main__":
    main()
