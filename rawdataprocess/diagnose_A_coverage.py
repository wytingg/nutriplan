

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_parsed_ingredients_sample.py  (improved)
Reconstruct a diagnostic sample with: rid, ingredient, amount, unit_raw, ingredient_norm, fdc_id
- Auto-detects columns case-insensitively (common Food.com schemas)
- Allows explicit --ingredients_col and --id_col
- Robust parsing of amounts/units; robust list/JSON/semi-colon strings

Usage:
python work/recipebench/scripts/rawdataprocess/diagnose_A_coverage.py \
  --a_table /mnt/data/household_weights_A.csv \
  --parsed_ingredients work/recipebench/data/3out/parsed_ingredients_sample.csv \
  --out_report work/recipebench/data/3out/a_coverage_report.txt \
  --out_csv work/recipebench/data/3out/parsed_with_A.csv

"""
import argparse
import ast
import json
import re
from typing import List, Tuple, Optional

import pandas as pd

VULGAR = {"¼":0.25,"½":0.5,"¾":0.75,"⅐":1/7,"⅑":1/9,"⅒":0.1,"⅓":1/3,"⅔":2/3,
          "⅕":0.2,"⅖":0.4,"⅗":0.6,"⅘":0.8,"⅙":1/6,"⅚":5/6,"⅛":0.125,"⅜":0.375,"⅝":0.625,"⅞":0.875}

UNIT_PAT = r"(cups?|c\.|tablespoons?|tbsp\.?|tbs\.?|tbl\.?|teaspoons?|tsp\.?|tsps?\.?|liters?|litres?|l|milliliters?|millilitres?|ml|fluid\s*ounces?|fl\.?\s*oz\.?|floz|ounces?|oz|pounds?|lbs?|lb|grams?|g|kilograms?|kg|slice?s?|clove?s?|sticks?|pieces?|bunch(es)?|heads?|leaf|leaves|sprigs?|stalks?|ears?|package?s?|pkg|containers?|jars?|cans?|bags?|bars?|servings?)"

UNIT_ALIAS = {"c":"cup","c.":"cup","cup.":"cup","tbsp.":"tbsp","tbs.":"tbsp","tbl.":"tbsp",
              "tsp.":"tsp","tsps.":"tsp","fl. oz.":"fl oz","fl. oz":"fl oz","floz":"fl oz"}

def to_float_amount(tokens: List[str]) -> Optional[float]:
    nums = []
    for t in tokens:
        t = t.strip()
        if not t: 
            continue
        if t in VULGAR:
            nums.append(VULGAR[t]); 
            continue
        if re.fullmatch(r"\d+/\d+", t): 
            a,b = t.split("/"); nums.append(float(a)/float(b)); 
            continue
        if re.fullmatch(r"\d+(\.\d+)?", t): 
            nums.append(float(t)); 
            continue
        if re.fullmatch(r"\d+[-–—]\d+", t):
            a,b = re.split(r"[-–—]", t); nums.append(float(b)); 
            continue
    if not nums: 
        return None
    return float(sum(nums))

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s/\.]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_unit(u: str) -> str:
    u = u.strip().lower()
    u = UNIT_ALIAS.get(u, u)
    repl = {"cups":"cup","tablespoons":"tbsp","tablespoon":"tbsp","tbs":"tbsp","tbl":"tbsp",
            "teaspoons":"tsp","teaspoon":"tsp","tsps":"tsp","liters":"l","litres":"l",
            "milliliters":"ml","millilitres":"ml","ounces":"oz","pounds":"lb","lbs":"lb",
            "grams":"g","kilograms":"kg","slices":"slice","cloves":"clove","sticks":"stick",
            "pieces":"piece","bunches":"bunch","heads":"head","sprigs":"sprig","stalks":"stalk","ears":"ear",
            "packages":"package","containers":"container","jars":"jar","cans":"can","bags":"bag","bars":"bar","servings":"serving",
            "fl oz":"fl_oz"}
    return repl.get(u, u)

def parse_line(line: str) -> Tuple[Optional[float], str, str]:
    if not isinstance(line, str): 
        return (None, "", "")
    s = normalize_text(line)
    amt_pat = r"(?P<amt>(\d+(\.\d+)?|[" + "".join(VULGAR.keys()) + r"]|\d+/\d+|\d+[-–—]\d+)(\s+\d+/\d+)?)"
    pat = re.compile(rf"^{amt_pat}\s*(?P<unit>{UNIT_PAT})?\b", re.IGNORECASE)
    m = pat.search(s)
    amount = None; unit = ""; rest = s
    if m:
        amt_tokens = m.group("amt").split()
        amount = to_float_amount(amt_tokens)
        unit = m.group("unit") or ""
        unit = normalize_unit(unit)
        rest = s[m.end():].strip()
    core = re.sub(r"\s*,.*$", "", rest).strip()
    return (amount, unit, core)

def to_list(x):
    # list column may be python-literal string, JSON, or semicolon-separated
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        for loader in (json.loads, ast.literal_eval):
            try: 
                val = loader(x)
                if isinstance(val, list): 
                    return val
            except Exception:
                pass
        return [t.strip() for t in re.split(r"[;\n]", x) if t.strip()]
    return []

def explode_ingredients(df, id_col, ing_col):
    rows = []
    for _, r in df[[id_col, ing_col]].iterrows():
        rid = r[id_col]
        lst = to_list(r[ing_col])
        for line in lst:
            if not isinstance(line, str) or not line.strip(): 
                continue
            amount, unit, core = parse_line(line)
            rows.append((rid, line, amount, unit, core))
    return pd.DataFrame(rows, columns=["rid","ingredient","amount","unit_raw","ingredient_norm"])

def autodetect_columns(columns, override_id=None, override_ing=None):
    cols_lower = {c.lower(): c for c in columns}
    # prefer explicit overrides
    id_col = override_id if override_id else None
    ing_col = override_ing if override_ing else None

    if id_col is None:
        for cand in ("recipe_id","recipeid","recipeid","rid","recipeId"):
            if cand.lower() in cols_lower:
                id_col = cols_lower[cand.lower()]
                break
    if ing_col is None:
        for cand in ("recipeingredientparts","ingredients","ingredient_list","ingredient_lines","ingredient"):
            if cand.lower() in cols_lower:
                ing_col = cols_lower[cand.lower()]
                break
    return id_col, ing_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", required=True)
    ap.add_argument("--ingredient_mapping", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sample_n", type=int, default=8000)
    ap.add_argument("--ingredients_col", default="")
    ap.add_argument("--id_col", default="")
    args = ap.parse_args()

    # read recipes (parquet or csv)
    rec = pd.read_parquet(args.recipes) if args.recipes.endswith(".parquet") else pd.read_csv(args.recipes)
    id_col, ing_col = autodetect_columns(rec.columns, args.id_col or None, args.ingredients_col or None)
    if not id_col:
        raise SystemExit(f"No recipe id column found. Columns present: {list(rec.columns)}")
    if not ing_col:
        raise SystemExit(f"No ingredients column found. Columns present: {list(rec.columns)}")

    exploded = explode_ingredients(rec, id_col, ing_col)

    # mapping
    mp = pd.read_parquet(args.ingredient_mapping) if args.ingredient_mapping.endswith(".parquet") else pd.read_csv(args.ingredient_mapping)
    if not {"ingredient_norm","fdc_id"} <= set(mp.columns):
        raise SystemExit(f"ingredient_mapping must have columns: ingredient_norm, fdc_id. Columns present: {list(mp.columns)}")
    mp2 = mp[["ingredient_norm","fdc_id"]].drop_duplicates()

    out = exploded.merge(mp2, on="ingredient_norm", how="left")
    if len(out) > args.sample_n:
        out = out.sample(args.sample_n, random_state=42)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with columns: rid, ingredient, amount, unit_raw, ingredient_norm, fdc_id")

if __name__ == "__main__":
    main()
