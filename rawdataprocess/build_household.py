#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_household_A.py — Construct the A-table (Household Weights, strong source) from USDA FoodData Central.
Inputs:
  - food_portion.csv
  - measure_unit.csv
Outputs:
  - household_weights_A.csv            (canonical per (fdc_id, unit))
  - household_weights_A_variants.csv   (all valid variants)
  - household_weights_A_report.txt     (coverage report)
Usage example:
  python build_household_A.py \
      --food_portion /mnt/data/food_portion.csv \
      --measure_unit /mnt/data/measure_unit.csv \
      --out_dir /mnt/data
"""
# python work/recipebench/scripts/rawdataprocess/build_household.py \
#   --food_portion work/recipebench/data/raw/usda/food_portion.csv \
#   --measure_unit work/recipebench/data/raw/usda/measure_unit.csv \
#   --out_dir work/recipebench/data/3out

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------
# Unit normalization
# ------------------------------
UNIT_ALIAS = {
    # volume
    "cup": "cup",
    "cups": "cup",
    "tablespoon": "tbsp",
    "table spoon": "tbsp",
    "tablespoons": "tbsp",
    "tbsp": "tbsp",
    "teaspoon": "tsp",
    "tea spoon": "tsp",
    "teaspoons": "tsp",
    "tsp": "tsp",
    "liter": "l",
    "litre": "l",
    "l": "l",
    "milliliter": "ml",
    "millilitre": "ml",
    "ml": "ml",
    "gallon": "gallon",
    "gallons": "gallon",
    "pint": "pint",
    "pints": "pint",
    "quart": "quart",
    "quarts": "quart",
    "fluid ounce": "fl_oz",
    "fluid ounces": "fl_oz",
    "fl oz": "fl_oz",
    "fl. oz": "fl_oz",
    "fl_oz": "fl_oz",
    # mass
    "ounce": "oz",
    "ounces": "oz",
    "oz": "oz",
    "pound": "lb",
    "pounds": "lb",
    "lb": "lb",
    "lbs": "lb",
    # household / piece-like
    "clove": "clove",
    "cloves": "clove",
    "slice": "slice",
    "slices": "slice",
    "stick": "stick",
    "sticks": "stick",
    "package": "package",
    "packages": "package",
    "container": "container",
    "containers": "container",
    "jar": "jar",
    "jars": "jar",
    "can": "can",
    "cans": "can",
    "bag": "bag",
    "bags": "bag",
    "bar": "bar",
    "bars": "bar",
    "serving": "serving",
    "servings": "serving",
    "piece": "piece",
    "pieces": "piece",
    "bunch": "bunch",
    "bunches": "bunch",
    "head": "head",
    "heads": "head",
    "leaf": "leaf",
    "leaves": "leaf",
    "sprig": "sprig",
    "sprigs": "sprig",
    "stalk": "stalk",
    "stalks": "stalk",
    "ear": "ear",
    "ears": "ear",
    "fillet": "fillet",
    "fillets": "fillet",
    "patty": "patty",
    "patties": "patty",
}

def _clean_text(s: str) -> str:
    s = s.strip().lower()
    # normalize weird punctuation/spaces
    s = "".join(ch if ch.isalnum() or ch in (" ", "_") else " " for ch in s)
    s = " ".join(s.split())
    # common fixes
    s = s.replace("fluid  ounce", "fluid ounce")
    s = s.replace("fl  oz", "fl oz")
    s = s.replace("fl oz ", "fl oz")  # collapse trailing space if any
    return s

def normalize_unit_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    s = _clean_text(name)
    if s in UNIT_ALIAS:
        return UNIT_ALIAS[s]
    return s.replace(" ", "_")

# ------------------------------
# Core pipeline
# ------------------------------
def build_A_table(food_portion_csv: Path, measure_unit_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    portion = pd.read_csv(food_portion_csv)
    unit = pd.read_csv(measure_unit_csv)

    # join to get human-readable unit names
    df = portion.merge(
        unit, left_on="measure_unit_id", right_on="id", how="left", suffixes=("", "_unit")
    ).rename(columns={"name": "unit_name", "id": "measure_unit_table_id"})

    # numeric casts
    for col in ("amount", "gram_weight", "data_points", "min_year_acquired"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # unit normalization
    df["unit_std"] = df["unit_name"].apply(normalize_unit_name)

    # grams per unit
    df["grams_per_unit"] = np.where(
        (df["amount"] > 0) & (df["gram_weight"] > 0),
        df["gram_weight"] / df["amount"],
        np.nan,
    )

    # validity filter (sanity bounds)
    df["is_valid"] = (
        (~df["grams_per_unit"].isna())
        & (df["grams_per_unit"] > 0)
        & (df["grams_per_unit"] < 1e6)
    )

    # quality signal: prefer more datapoints, more recent, amount near 1
    dpts = df["data_points"].fillna(1.0)
    yr_series = df["min_year_acquired"]
    yr_min = int(yr_series.dropna().min()) if yr_series.dropna().shape[0] else 2000
    yr = yr_series.fillna(yr_min)
    amt = df["amount"].replace(0, np.nan).fillna(1.0)
    df["quality_score"] = np.log1p(dpts) + 0.001 * (yr - yr.min()) + (
        1.0 - np.minimum(np.abs(amt - 1.0), 1.0)
    )

    valid = df[df["is_valid"]].copy()

    # group and aggregate to canonical per (fdc_id, unit_std)
    grp_cols = ["fdc_id", "unit_std"]
    agg = (
        valid.groupby(grp_cols)
        .agg(
            grams_per_unit_median=("grams_per_unit", "median"),
            grams_per_unit_mean=("grams_per_unit", "mean"),
            quality_score_max=("quality_score", "max"),
            data_points_max=("data_points", "max"),
            min_year_max=("min_year_acquired", "max"),
            variants=("fdc_id", "count"),
        )
        .reset_index()
    )
    agg["grams_per_unit"] = agg["grams_per_unit_median"].fillna(agg["grams_per_unit_mean"])

    canon = agg.rename(
        columns={"unit_std": "unit", "min_year_max": "min_year"}
    )[["fdc_id", "unit", "grams_per_unit", "data_points_max", "min_year", "variants"]]
    canon["source"] = "USDA_FDC_food_portion"

    # detailed variants table
    keep_cols = [
        "fdc_id",
        "unit_std",
        "amount",
        "gram_weight",
        "grams_per_unit",
        "portion_description",
        "modifier",
        "data_points",
        "min_year_acquired",
        "measure_unit_id",
        "unit_name",
    ]
    variants_cols = [c for c in keep_cols if c in valid.columns]
    variants = valid[variants_cols].rename(
        columns={"unit_std": "unit", "min_year_acquired": "min_year"}
    )

    # outputs
    out_canon = out_dir / "household_weights_A.csv"
    out_variants = out_dir / "household_weights_A_variants.csv"
    out_report = out_dir / "household_weights_A_report.txt"

    canon.to_csv(out_canon, index=False)
    variants.to_csv(out_variants, index=False)

    # coverage report
    total_pairs = (
        df[["fdc_id", "unit_std"]].dropna().drop_duplicates().shape[0]
        if {"fdc_id", "unit_std"} <= set(df.columns)
        else 0
    )
    covered_pairs = canon[["fdc_id", "unit"]].dropna().drop_duplicates().shape[0]
    coverage = covered_pairs / max(total_pairs, 1)

    report = (
        "Household Weights A-table Build Report\n"
        f"Unique (fdc_id, unit) pairs in raw: {total_pairs}\n"
        f"Covered (fdc_id, unit) pairs with valid grams_per_unit: {covered_pairs}\n"
        f"Coverage: {coverage:.2%}\n"
    )
    out_report.write_text(report, encoding="utf-8")

    return {
        "canon_path": str(out_canon),
        "variants_path": str(out_variants),
        "report_path": str(out_report),
        "coverage": coverage,
    }

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build Household Weights A-table from USDA CSVs.")
    p.add_argument("--food_portion", type=Path, required=True, help="Path to food_portion.csv")
    p.add_argument("--measure_unit", type=Path, required=True, help="Path to measure_unit.csv")
    p.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    res = build_A_table(args.food_portion, args.measure_unit, args.out_dir)
    print("== A-table build complete ==")
    print("Canonical:", res["canon_path"])
    print("Variants :", res["variants_path"])
    print("Report   :", res["report_path"])
    print(f"Coverage : {res['coverage']:.2%}")

if __name__ == "__main__":
    sys.exit(main())
