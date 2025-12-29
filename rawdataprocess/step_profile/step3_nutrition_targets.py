#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — 营养目标（带指南钳制，含“保留营养子集”）
================================================

本版针对 long 表的真实列名做了适配：
- 支持列：`recipe_id | nutrient_name | unit | amount_for_recipe`（也兼容旧版 `RecipeId | amount`）。
- 统一单位：Energy→kcal；宏量/脂肪酸→g；矿物→mg；自动处理 kJ↔kcal、mg/µg↔g、g↔mg 等。
- 多口径能量（Energy vs Atwater General/Specific）按优先级去重（General > Energy > Specific）。
 python work/recipebench/scripts/step_profile/step3_nutrition_targets.py \
  --reviews work/recipebench/data/raw/foodcom/reviews.parquet \
  --nutr-wide work/recipebench/data/4out/recipe_nutrients_main.parquet \
  --out work/recipebench/data/8step_profile/nutrition_targets.jsonl \
  --keep-set core+extended \
  --pos-thresh 4 \
  --min-liked 3 \
  --meals-per-day 2.4 \
  --energy-fallback-kcal 2000 \
  --guideline WHO \
  --sugars-limit-pct 10 \
  --fiber-guideline IOM \
  --winsor-pct 0.02 \
  --tolerance 0.05
输出仍满足：`amdr, sodium_mg_max, free_sugars_pct_max, fiber_g_min`，并写入 `audit`（含指南来源与可用性）。
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# ============================
# 指南常量（用于钳制/审计）
# ============================
AMDR = {
    "carb":   {"min": 45.0, "max": 65.0},   # IOM/DRI
    "protein":{"min": 10.0, "max": 35.0},
    "fat":    {"min": 20.0, "max": 35.0},
}

SODIUM_MAX = {
    "WHO": 2000,   # mg/day
    "US_DGA": 2300,
}

FIBER_GUIDELINE = {
    "IOM": "14g_per_1000kcal",  # adult AI
    "WHO": "25g_fixed",
}

FAT_QUALITY_LIMITS = {
    "SFA_pct_max": 10.0,  # WHO / DGA 建议水平
    "TFA_pct_max": 1.0,   # WHO 建议水平
}

# ============================
# 目标营养素规范名
# ============================
CANON_NAMES = {
    "energy_kcal": [
        "energy (atwater general factors)",
        "energy",
        "energy_kcal",
        "calories_kcal",
        "enerc_kcal",
        "energy (atwater specific factors)",
    ],
    "protein_g":   ["protein", "protein_g"],
    "fat_g":       ["total lipid (fat)", "fat", "fat_g"],
    # 兼容你的 wide 表命名：carbohydrates_g / sugars_total_g
    "carb_g":      ["carbohydrate, by difference", "carbohydrate", "carb_g", "carbohydrates_g"],
    "sugars_g":    ["sugars, total", "sugars", "sugar_g", "sugars_total_g"],
    "fiber_g":     ["fiber, total dietary", "fiber", "fibre", "fiber_g"],
    "sodium_mg":   ["sodium, na", "sodium", "na_mg", "sodium_mg"],
    # Extended
    # 把 canonical 名本身也加入候选，确保直接匹配 wide 列（potassium_mg 等）
    "potassium_mg": ["potassium_mg", "potassium, k", "potassium"],
    "calcium_mg":   ["calcium_mg", "calcium, ca", "calcium"],
    "magnesium_mg": ["magnesium_mg", "magnesium, mg", "magnesium"],
    "iron_mg":      ["iron_mg", "iron, fe", "iron"],
    "zinc_mg":      ["zinc_mg", "zinc, zn", "zinc"],
    "phosphorus_mg":["phosphorus_mg", "phosphorus, p", "phosphorus"],
    # Fatty acids totals：兼容 wide 列 saturated_fat_g / monounsaturated_fat_g / polyunsaturated_fat_g
    "sfa_g":        ["sfa_g", "fatty acids, total saturated", "sfa", "saturated_fat_g"],
    "mufa_g":       ["mufa_g", "fatty acids, total monounsaturated", "mufa", "monounsaturated_fat_g"],
    "pufa_g":       ["pufa_g", "fatty acids, total polyunsaturated", "pufa", "polyunsaturated_fat_g"],
}

LONG_TO_CANON = {
    # Core
    "Energy": "energy_kcal",
    "Energy (Atwater General Factors)": "energy_kcal",
    "Energy (Atwater Specific Factors)": "energy_kcal",  # 次选
    "Protein": "protein_g",
    "Total lipid (fat)": "fat_g",
    "Carbohydrate, by difference": "carb_g",
    "Sugars, total": "sugars_g",
    "Fiber, total dietary": "fiber_g",
    "Sodium, Na": "sodium_mg",
    # Extended — 矿物质
    "Potassium, K": "potassium_mg",
    "Calcium, Ca": "calcium_mg",
    "Magnesium, Mg": "magnesium_mg",
    "Iron, Fe": "iron_mg",
    "Zinc, Zn": "zinc_mg",
    "Phosphorus, P": "phosphorus_mg",
    # Fat quality（总量）
    "Fatty acids, total saturated": "sfa_g",
    "Fatty acids, total monounsaturated": "mufa_g",
    "Fatty acids, total polyunsaturated": "pufa_g",
    # 兼容部分 long/wide 融合场景的直写列名（稳妥起见）
    "saturated_fat_g": "sfa_g",
    "monounsaturated_fat_g": "mufa_g",
    "polyunsaturated_fat_g": "pufa_g",
    "Sugars, Total": "sugars_g",
    # Fat quality（常见代理，若无总量时用于近似）
    "SFA 14:0": "sfa14_g",
    "SFA 16:0": "sfa16_g",
    "SFA 18:0": "sfa18_g",
    "MUFA 18:1 c": "mufa181c_g",
    "PUFA 18:2": "pufa182_g",
    "PUFA 18:3": "pufa183_g",
    # Sugars components（用于回推总糖）
    "Fructose": "fructose_g",
    "Glucose": "glucose_g",
    "Sucrose": "sucrose_g",
    "Lactose": "lactose_g",
    "Maltose": "maltose_g",
    "Galactose": "galactose_g",
}

CORE_LONG_NAMES = [
    "Energy (Atwater General Factors)", "Energy", "Energy (Atwater Specific Factors)",
    "Protein", "Total lipid (fat)", "Carbohydrate, by difference",
    "Sugars, total", "Fiber, total dietary", "Sodium, Na"
]
EXT_LONG_NAMES = [
    "Potassium, K", "Calcium, Ca", "Magnesium, Mg", "Iron, Fe", "Zinc, Zn", "Phosphorus, P",
    "Fatty acids, total saturated", "Fatty acids, total monounsaturated", "Fatty acids, total polyunsaturated",
    "SFA 14:0", "SFA 16:0", "SFA 18:0", "MUFA 18:1 c", "PUFA 18:2", "PUFA 18:3",
    # Sugars components（作为扩展项）
    "Fructose", "Glucose", "Sucrose", "Lactose", "Maltose", "Galactose",
]

# ============================
# 工具函数
# ============================

def _norm_colnames(cols: List[str]) -> Dict[str, str]:
    import re
    mapping = {}
    for c in cols:
        key = re.sub(r"[^a-z0-9]+", " ", c.lower()).strip()
        mapping[key] = c
    return mapping


def _find_first_present(normmap: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        key = cand.lower()
        if key in normmap:
            return normmap[key]
    return None


def weighted_quantile(values: np.ndarray, quantiles: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    else:
        sample_weight = np.asarray(sample_weight)
    sorter = np.argsort(values)
    values, sample_weight = values[sorter], sample_weight[sorter]
    cdf = np.cumsum(sample_weight)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, values)


def winsorize_series(s: pd.Series, p: float = 0.02) -> pd.Series:
    if s.empty:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def project_to_simplex_with_box(targets: Dict[str, float], bounds: Dict[str, Tuple[float, float]], total: float = 100.0) -> Dict[str, float]:
    keys = list(targets.keys())
    for k in keys:
        lo, hi = bounds[k]
        targets[k] = clamp(targets[k], lo, hi)
    s = sum(targets.values())
    if s <= 0:
        mid = {k: (bounds[k][0] + bounds[k][1]) / 2 for k in keys}
        targets = mid
        s = sum(targets.values())
    scale = total / s
    targets = {k: targets[k] * scale for k in keys}
    for _ in range(3):
        over, under = {}, {}
        for k in keys:
            lo, hi = bounds[k]
            if targets[k] > hi: over[k] = targets[k] - hi
            if targets[k] < lo: under[k] = lo - targets[k]
        if not over and not under:
            break
        if over:
            surplus = sum(over.values())
            for k in over:
                targets[k] = bounds[k][1]
            sinks = [k for k in keys if k not in over]
            if sinks:
                dec = surplus / len(sinks)
                for k in sinks:
                    targets[k] = max(bounds[k][0], targets[k] - dec)
        if under:
            deficit = sum(under.values())
            for k in under:
                targets[k] = bounds[k][0]
            sources = [k for k in keys if k not in under]
            if sources:
                inc = deficit / len(sources)
                for k in sources:
                    targets[k] = min(bounds[k][1], targets[k] + inc)
        s = sum(targets.values())
        if s != total and s > 0:
            targets = {k: targets[k] * total / s for k in keys}
    return targets

# ============================
# 单位归一（long 表专用）
# ============================

ENERGY_NAMES = [
    "Energy (Atwater General Factors)",
    "Energy",
    "Energy (Atwater Specific Factors)",
]

MACRO_NAMES = {
    "Protein": "g",
    "Total lipid (fat)": "g",
    "Carbohydrate, by difference": "g",
    "Sugars, total": "g",
    "Fiber, total dietary": "g",
    # Sugars components 单位
    "Fructose": "g",
    "Glucose": "g",
    "Sucrose": "g",
    "Lactose": "g",
    "Maltose": "g",
    "Galactose": "g",
}

MINERAL_NAMES = {
    "Sodium, Na": "mg",
    "Potassium, K": "mg",
    "Calcium, Ca": "mg",
    "Magnesium, Mg": "mg",
    "Iron, Fe": "mg",
    "Zinc, Zn": "mg",
    "Phosphorus, P": "mg",
}

FAT_QUALITY_NAMES = {
    "Fatty acids, total saturated": "g",
    "Fatty acids, total monounsaturated": "g",
    "Fatty acids, total polyunsaturated": "g",
    "SFA 14:0": "g",
    "SFA 16:0": "g",
    "SFA 18:0": "g",
    "MUFA 18:1 c": "g",
    "PUFA 18:2": "g",
    "PUFA 18:3": "g",
}

UNIT_UP = {"KCAL":"KCAL","kcal":"KCAL","KCALS":"KCAL","KJ":"KJ","kJ":"KJ","kj":"KJ",
           "G":"G","g":"G","GRAM":"G","MG":"MG","mg":"MG","UG":"UG","µG":"UG","mcg":"UG"}


def _canon_unit_for_nutrient(nutrient_name: str) -> Optional[str]:
    if nutrient_name in ENERGY_NAMES:
        return "KCAL"
    if nutrient_name in MACRO_NAMES:
        return MACRO_NAMES[nutrient_name].upper()
    if nutrient_name in MINERAL_NAMES:
        return MINERAL_NAMES[nutrient_name].upper()
    if nutrient_name in FAT_QUALITY_NAMES:
        return FAT_QUALITY_NAMES[nutrient_name].upper()
    return None


def _convert_amount(nutrient_name: str, unit: str, amount: float) -> Tuple[float, Optional[str]]:
    """返回 (amount_in_canonical_unit, canonical_unit)。若无法识别则原样返回并保留 unit。"""
    unit_up = UNIT_UP.get(str(unit), str(unit)).upper()
    want = _canon_unit_for_nutrient(nutrient_name)
    if want is None or amount is None or not np.isfinite(amount):
        return amount, unit_up

    # Energy：kJ→kcal，其他已是 kcal
    if nutrient_name in ENERGY_NAMES:
        if unit_up == "KJ":
            return float(amount) / 4.184, "KCAL"
        return float(amount), "KCAL"

    # g / mg / µg 转换
    if want == "G":
        if unit_up == "G":
            return float(amount), "G"
        if unit_up == "MG":
            return float(amount) / 1000.0, "G"
        if unit_up == "UG":
            return float(amount) / 1e6, "G"
    if want == "MG":
        if unit_up == "MG":
            return float(amount), "MG"
        if unit_up == "G":
            return float(amount) * 1000.0, "MG"
        if unit_up == "UG":
            return float(amount) / 1000.0, "MG"

    # 其他未知组合：原样
    return float(amount), unit_up


# ============================
# 数据加载（wide / long）
# ============================

def load_reviews(path: str, pos_thresh: int, time_decay_half_life_days: Optional[float]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    col_user = next((c for c in df.columns if c.lower() in {"authorid", "user_id", "userid"}), None)
    col_recipe = next((c for c in df.columns if c.lower() in {"recipeid", "recipe_id"}), None)
    col_rating = next((c for c in df.columns if c.lower() in {"rating", "score"}), None)
    if not (col_user and col_recipe and col_rating):
        raise ValueError("reviews.parquet 缺少必要列: AuthorId/RecipeId/Rating")
    df = df[[col_user, col_recipe, col_rating] + [c for c in ["DateSubmitted", "date", "date_submitted"] if c in df.columns]].copy()
    df.rename(columns={col_user: "user_id", col_recipe: "recipe_id", col_rating: "rating"}, inplace=True)
    df = df[df["rating"] >= pos_thresh]
    df["weight"] = 1.0
    if time_decay_half_life_days is not None and any(c in df.columns for c in ["DateSubmitted", "date", "date_submitted"]):
        date_col = next((c for c in ["DateSubmitted", "date", "date_submitted"] if c in df.columns))
        dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        now = pd.Timestamp.utcnow()
        age_days = (now - dt).dt.total_seconds() / 86400.0
        hl = float(time_decay_half_life_days)
        w = np.power(0.5, np.clip(age_days, 0, None) / hl)
        w = np.where(np.isfinite(w), w, 0.0)
        df.loc[:, "weight"] = w
    return df


def load_nutrition_wide(path: str, keep_set: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    normmap = _norm_colnames(list(df.columns))
    def pick(name_key: str) -> Optional[str]:
        return _find_first_present(normmap, CANON_NAMES[name_key])

    need_keys = ["energy_kcal", "protein_g", "fat_g", "carb_g", "sugars_g", "fiber_g", "sodium_mg"]
    if keep_set == "core+extended":
        need_keys += [
            "potassium_mg", "calcium_mg", "magnesium_mg", "iron_mg", "zinc_mg", "phosphorus_mg",
            "sfa_g", "mufa_g", "pufa_g"
        ]

    cols = {"recipe_id": next((c for c in df.columns if c.lower() in {"recipeid", "recipe_id"}), None)}
    if cols["recipe_id"] is None:
        raise ValueError("nutr-wide 缺少 RecipeId 列")

    for k in need_keys:
        col = pick(k)
        if col is not None:
            cols[k] = col

    used = [cols["recipe_id"]] + [v for k, v in cols.items() if k != "recipe_id"]
    out = df[used].copy()
    out.rename(columns={v: k for k, v in cols.items() if k != "recipe_id"}, inplace=True)
    return out


def load_nutrition_long(path: str, keep_set: str) -> pd.DataFrame:
    dfl = pd.read_parquet(path)

    # 列名适配：新(old) → 统一
    id_col = "recipe_id" if "recipe_id" in dfl.columns else ("RecipeId" if "RecipeId" in dfl.columns else None)
    if id_col is None:
        raise ValueError("long 表需包含 recipe_id/RecipeId")
    amt_col = "amount_for_recipe" if "amount_for_recipe" in dfl.columns else ("amount" if "amount" in dfl.columns else None)
    if amt_col is None:
        raise ValueError("long 表需包含 amount_for_recipe/amount 列")
    unit_col = "unit" if "unit" in dfl.columns else None

    # 选择要保留的营养名
    keep = set(CORE_LONG_NAMES)
    if keep_set == "core+extended":
        keep.update(EXT_LONG_NAMES)
    sub = dfl[dfl["nutrient_name"].isin(keep)][[id_col, "nutrient_name", amt_col] + ([unit_col] if unit_col else [])].copy()
    sub.rename(columns={id_col: "recipe_id", amt_col: "amount"}, inplace=True)

    # 统一 unit（保持列名为 'unit'，避免同名 drop 误删）
    if unit_col:
        if unit_col != "unit":
            sub["unit"] = sub[unit_col].astype(str)
            sub.drop(columns=[unit_col], inplace=True)
        else:
            sub["unit"] = sub["unit"].astype(str)
        sub["unit"] = sub["unit"].fillna("")
    else:
        sub["unit"] = ""

    # 单位归一
    amt_conv = []
    canon_unit = []
    rows_to_process = list(zip(sub["nutrient_name"].tolist(), sub["unit"].tolist(), sub["amount"].tolist()))
    if tqdm is not None:
        try:
            rows_to_process = tqdm(rows_to_process, desc="[step3] 转换单位", unit="row", mininterval=1.0)
        except Exception:
            pass
    for n, u, a in rows_to_process:
        aa, uu = _convert_amount(n, u, a)
        amt_conv.append(aa)
        canon_unit.append(uu)
    sub["amount_canon"], sub["unit_canon"] = amt_conv, canon_unit

    # 能量口径优先：Energy(0) < General(1) < Specific(2)
    rank_map = {"Energy": 0, "Energy (Atwater General Factors)": 1, "Energy (Atwater Specific Factors)": 2}
    sub["_energy_rank"] = sub["nutrient_name"].map(rank_map).fillna(9)

    # 对能量项，按 recipe 选最优口径（且全部转为 kcal）
    energy_mask = sub["nutrient_name"].isin(ENERGY_NAMES)
    best_energy = (
        sub[energy_mask]
        .sort_values(["recipe_id", "_energy_rank"]) 
        .groupby("recipe_id", as_index=False)
        .first()
    )
    non_energy = sub[~energy_mask]
    sub2 = pd.concat([best_energy, non_energy], ignore_index=True)

    # 将营养名映射为规范字段名
    sub2["canon_name"] = sub2["nutrient_name"].map(LONG_TO_CANON).fillna(sub2["nutrient_name"])  # 未映射的保持原名（不会进后续逻辑）

    # 仅保留映射成功的项目
    sub2 = sub2[sub2["canon_name"].isin(LONG_TO_CANON.values())]

    # pivot：同一 recipe_id × canon_name 可能仍有多行（极少数），使用 mean 以稳健
    piv = sub2.pivot_table(index="recipe_id", columns="canon_name", values="amount_canon", aggfunc="mean").reset_index()

    return piv


def build_recipe_metrics(nut: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    df = nut.copy()
    if "recipe_id" not in df.columns:
        raise ValueError("营养表须包含 recipe_id")
    for c in df.columns:
        if c != "recipe_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    avail = {c: ((c in df.columns) and bool(df[c].notna().any())) for c in [
        "energy_kcal", "protein_g", "fat_g", "carb_g", "sugars_g", "fiber_g", "sodium_mg",
        "sfa_g", "mufa_g", "pufa_g", "sfa14_g", "sfa16_g", "sfa18_g",
        "mufa181c_g", "pufa182_g", "pufa183_g",
        "potassium_mg", "calcium_mg", "magnesium_mg", "iron_mg", "zinc_mg", "phosphorus_mg",
    ]}

    if not avail.get("sfa_g", False):
        if all(avail.get(k, False) for k in ["sfa14_g", "sfa16_g", "sfa18_g"]):
            df["sfa_g"] = df[["sfa14_g", "sfa16_g", "sfa18_g"]].sum(axis=1)
            avail["sfa_g"] = True

    # 能量兜底/校正：用宏量反推（保证为按索引对齐的 Series）
    kcal = df.get("energy_kcal", pd.Series(index=df.index, dtype=float))
    prot = df.get("protein_g", pd.Series(0.0, index=df.index))
    carb = df.get("carb_g", pd.Series(0.0, index=df.index))
    fat  = df.get("fat_g", pd.Series(0.0, index=df.index))
    prot = pd.to_numeric(prot, errors="coerce").fillna(0.0)
    carb = pd.to_numeric(carb, errors="coerce").fillna(0.0)
    fat  = pd.to_numeric(fat,  errors="coerce").fillna(0.0)
    energy_from_macros = 4.0 * prot + 4.0 * carb + 9.0 * fat
    mask = energy_from_macros.notna() & (energy_from_macros > 0)
    bad = (~np.isfinite(kcal)) | (kcal < 0.75 * energy_from_macros) | (kcal > 1.25 * energy_from_macros)
    df.loc[mask & bad, "energy_kcal"] = energy_from_macros.where(mask & bad, other=np.nan)
    kcal = df.get("energy_kcal", pd.Series(index=df.index, dtype=float))

    # 总糖缺失时，用单糖/二糖回推
    sugar_comp_cols = [c for c in [
        "fructose_g", "glucose_g", "sucrose_g", "lactose_g", "maltose_g", "galactose_g"
    ] if c in df.columns]
    if ("sugars_g" not in df.columns or df["sugars_g"].isna().all()) and sugar_comp_cols:
        df["sugars_g"] = df[sugar_comp_cols].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        protein_pct = 4.0 * prot / kcal * 100.0
        carb_pct    = 4.0 * carb / kcal * 100.0
        fat_pct     = 9.0 * fat  / kcal * 100.0
        sugar_pct   = 4.0 * df.get("sugars_g", 0.0)  / kcal * 100.0
        fiber_dens  = df.get("fiber_g", 0.0)         / kcal * 1000.0  # g/1000kcal
        sodium_dens = df.get("sodium_mg", 0.0)       / kcal * 1000.0  # mg/1000kcal
        sfa_pct     = 9.0 * df.get("sfa_g", 0.0)     / kcal * 100.0
        mufa_pct    = 9.0 * df.get("mufa_g", 0.0)    / kcal * 100.0
        pufa_pct    = 9.0 * df.get("pufa_g", 0.0)    / kcal * 100.0

    out = pd.DataFrame({
        "recipe_id": df["recipe_id"],
        "energy_kcal": kcal,
        "protein_pct": protein_pct,
        "carb_pct": carb_pct,
        "fat_pct": fat_pct,
        "sugar_pct_total": sugar_pct,
        "fiber_g_per_1000kcal": fiber_dens,
        "sodium_mg_per_1000kcal": sodium_dens,
        "sfa_pct": sfa_pct,
        "mufa_pct": mufa_pct,
        "pufa_pct": pufa_pct,
    }).replace([np.inf, -np.inf], np.nan)

    return out, avail

# ============================
# 主流程
# ============================

@dataclass
class Args:
    reviews: str
    nutr_wide: Optional[str]
    nutr_long: Optional[str]
    out: str
    keep_set: str
    pos_thresh: int
    min_liked: int
    meals_per_day: float
    energy_fallback_kcal: float
    winsor_pct: float
    time_decay_half_life_days: Optional[float]
    sodium_guideline: str
    sugars_limit_pct: float
    fiber_guideline: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--reviews', required=True)
    p.add_argument('--nutr-wide', dest='nutr_wide')
    p.add_argument('--nutr-long', dest='nutr_long')
    p.add_argument('--out', required=True)
    p.add_argument('--keep-set', choices=['core', 'core+extended'], default='core', help='保留 A层 或 A+B 层营养')

    p.add_argument('--pos-thresh', type=int, default=4)
    p.add_argument('--min-liked', type=int, default=3)
    p.add_argument('--meals-per-day', type=float, default=2.4)
    p.add_argument('--energy-fallback-kcal', type=float, default=2000)
    p.add_argument('--winsor-pct', type=float, default=0.02)
    p.add_argument('--time-decay-half_life_days', type=float, default=730, help='设为负值可禁用')

    p.add_argument('--guideline', dest='sodium_guideline', choices=['WHO', 'US_DGA'], default='WHO', help='钠的上限来源')
    p.add_argument('--sugars-limit-pct', type=float, default=10.0, help='WHO: 10 或 5（严格）')
    p.add_argument('--fiber-guideline', choices=['IOM', 'WHO'], default='IOM')
    p.add_argument('--sodium-min-floor-mg', type=float, default=500.0)
    p.add_argument('--per-meal-frac', type=float, default=0.35, help='若策略为 per_meal，按此系数折算到每份（单餐）')
    p.add_argument('--tolerance', type=float, default=0.05, help='constraints[] 的双侧容差（±百分比），默认 5%')

    a = p.parse_args()
    tdec = None if (a.time_decay_half_life_days is None or a.time_decay_half_life_days < 0) else float(a.time_decay_half_life_days)
    return Args(
        reviews=a.reviews,
        nutr_wide=a.nutr_wide,
        nutr_long=a.nutr_long,
        out=a.out,
        keep_set=a.keep_set,
        pos_thresh=a.pos_thresh,
        min_liked=a.min_liked,
        meals_per_day=a.meals_per_day,
        energy_fallback_kcal=a.energy_fallback_kcal,
        winsor_pct=a.winsor_pct,
        time_decay_half_life_days=tdec,
        sodium_guideline=a.sodium_guideline,
        sugars_limit_pct=float(a.sugars_limit_pct),
        fiber_guideline=a.fiber_guideline,
        
    )


def wmed(series: pd.Series, w: pd.Series) -> float:
    s = series.to_numpy(dtype=float)
    ww = w.to_numpy(dtype=float)
    m = np.isfinite(s) & np.isfinite(ww) & (ww > 0)
    if not m.any():
        return np.nan
    return float(weighted_quantile(s[m], np.array([0.5]), ww[m])[0])


def main():
    args = parse_args()

    print(f"[step3] 读取 reviews: {args.reviews}")
    df_like = load_reviews(args.reviews, args.pos_thresh, args.time_decay_half_life_days)

    if args.nutr_wide:
        print(f"[step3] 读取营养(宽表): {args.nutr_wide}")
        nutr = load_nutrition_wide(args.nutr_wide, args.keep_set)
    elif args.nutr_long:
        print(f"[step3] 读取营养(长表): {args.nutr_long}")
        nutr = load_nutrition_long(args.nutr_long, args.keep_set)
    else:
        raise ValueError("需提供 --nutr-wide 或 --nutr-long 其一")

    print(f"[step3] 构建食谱指标...")
    rmet, avail = build_recipe_metrics(nutr)
    print(f"[step3] 合并数据...")
    dfm = df_like.merge(rmet, how='inner', on='recipe_id')

    print(f"[step3] 处理异常值...")
    cols_to_winsor = [
        "energy_kcal", "protein_pct", "carb_pct", "fat_pct", "sugar_pct_total",
        "fiber_g_per_1000kcal", "sodium_mg_per_1000kcal", "sfa_pct", "mufa_pct", "pufa_pct"
    ]
    for c in cols_to_winsor:
        if c in dfm.columns:
            dfm[c] = winsorize_series(dfm[c].astype(float), p=args.winsor_pct)

    print(f"[step3] 聚合用户数据...")
    grp = dfm.groupby('user_id', as_index=False)
    agg = grp.apply(lambda g: pd.Series({
        'n_liked': int((g['rating'] >= args.pos_thresh).sum()),
        'energy_kcal_med': wmed(g['energy_kcal'], g['weight']),
        'protein_pct_med': wmed(g['protein_pct'], g['weight']),
        'carb_pct_med': wmed(g['carb_pct'], g['weight']),
        'fat_pct_med': wmed(g['fat_pct'], g['weight']),
        'sugar_pct_med': wmed(g['sugar_pct_total'], g['weight']),
        'fiber_dens_med': wmed(g['fiber_g_per_1000kcal'], g['weight']),
        'sodium_dens_med': wmed(g['sodium_mg_per_1000kcal'], g['weight']),
        'sfa_pct_med': wmed(g['sfa_pct'], g['weight']) if 'sfa_pct' in g else np.nan,
        'mufa_pct_med': wmed(g['mufa_pct'], g['weight']) if 'mufa_pct' in g else np.nan,
        'pufa_pct_med': wmed(g['pufa_pct'], g['weight']) if 'pufa_pct' in g else np.nan,
    }), include_groups=False).reset_index(drop=True)

    agg = agg[agg['n_liked'] >= args.min_liked].copy()
    if agg.empty:
        print("[step3] 警告：没有满足最小点赞条数的用户，输出为空。")

    print(f"[step3] 计算营养目标...")
    out_rows = []
    rows_iter = agg.itertuples(index=False)
    if tqdm is not None:
        try:
            rows_iter = tqdm(rows_iter, total=len(agg), desc="[step3] users", unit="user", mininterval=1.0)
        except Exception:
            pass
    for row in rows_iter:
        user_id = getattr(row, 'user_id')
        n_liked = int(getattr(row, 'n_liked'))
        E_med = getattr(row, 'energy_kcal_med')
        E_daily = float(args.energy_fallback_kcal)
        if math.isfinite(E_med) and E_med > 0:
            E_daily = float(np.clip(args.meals_per_day * E_med, 1200, 3500))

        pref = {
            'carb': float(getattr(row, 'carb_pct_med')) if math.isfinite(getattr(row, 'carb_pct_med')) else np.nan,
            'protein': float(getattr(row, 'protein_pct_med')) if math.isfinite(getattr(row, 'protein_pct_med')) else np.nan,
            'fat': float(getattr(row, 'fat_pct_med')) if math.isfinite(getattr(row, 'fat_pct_med')) else np.nan,
        }
        for k, b in AMDR.items():
            if not math.isfinite(pref.get(k, np.nan)):
                pref[k] = (b['min'] + b['max']) / 2.0
        bounds = {k: (b['min'], b['max']) for k, b in AMDR.items()}
        amdr_target = project_to_simplex_with_box(pref.copy(), bounds, total=100.0)

        sodium_guideline = SODIUM_MAX.get(args.sodium_guideline, 2000)
        sod_dens = getattr(row, 'sodium_dens_med')
        sod_from_density = float(sod_dens) * E_daily / 1000.0 if math.isfinite(sod_dens) and sod_dens > 0 else np.inf
        sodium_mg_max = int(min(sodium_guideline, sod_from_density)) if math.isfinite(sod_from_density) else int(sodium_guideline)

        sugar_med = getattr(row, 'sugar_pct_med')
        sugar_pref = float(sugar_med) if math.isfinite(sugar_med) and sugar_med > 0 else np.inf
        free_sugars_pct_max = float(min(args.sugars_limit_pct, sugar_pref)) if math.isfinite(sugar_pref) else float(args.sugars_limit_pct)

        if args.fiber_guideline == 'IOM':
            fiber_min_guideline = 14.0 * E_daily / 1000.0
        else:
            fiber_min_guideline = 25.0
        fib_dens = getattr(row, 'fiber_dens_med')
        fiber_from_density = float(fib_dens) * E_daily / 1000.0 if math.isfinite(fib_dens) and fib_dens > 0 else 0.0
        fiber_g_min = float(max(fiber_min_guideline, fiber_from_density))

        # 统一每份系数（优先 meals_per_day，其次 per_meal_frac 如存在）
        per_serv_frac = None
        try:
            per_serv_frac = (1.0 / float(args.meals_per_day)) if (args.meals_per_day and args.meals_per_day > 0) else None
        except Exception:
            per_serv_frac = None
        if per_serv_frac is None:
            try:
                per_serv_frac = float(getattr(args, 'per_meal_frac', 0.0)) or None
            except Exception:
                per_serv_frac = None
        if per_serv_frac is None:
            per_serv_frac = 1.0  # 兜底为整日口径

        # target_pct → g/日 → g/份
        energy_kcal_target = E_daily
        carb_g_day    = round((amdr_target['carb']    / 100.0) * energy_kcal_target / 4.0, 2)
        protein_g_day = round((amdr_target['protein'] / 100.0) * energy_kcal_target / 4.0, 2)
        fat_g_day     = round((amdr_target['fat']     / 100.0) * energy_kcal_target / 9.0, 2)
        carb_g_serv    = round(carb_g_day    * per_serv_frac, 2)
        protein_g_serv = round(protein_g_day * per_serv_frac, 2)
        fat_g_serv     = round(fat_g_day     * per_serv_frac, 2)

        # 纤维：日与每份
        fiber_g_day_min = float(round(fiber_g_min, 2))
        fiber_g_serving_min = float(round(fiber_g_day_min * per_serv_frac, 2))

        # 饱和脂肪（10% 能量 → 克）
        sat_fat_pct_max = 10.0
        sfa_g_day_max = float(round((sat_fat_pct_max / 100.0) * energy_kcal_target / 9.0, 2))
        sfa_g_serving_max = float(round(sfa_g_day_max * per_serv_frac, 2))

        # 自由糖（10% 能量 → 克）；若无 free sugars 列，用 sugars_total_g 作为保守上限
        free_sugars_pct_max_val = float(args.sugars_limit_pct) if hasattr(args, 'sugars_limit_pct') else 10.0
        sugars_g_day_max = float(round((free_sugars_pct_max_val / 100.0) * energy_kcal_target / 4.0, 2))
        sugars_g_serving_max = float(round(sugars_g_day_max * per_serv_frac, 2))
        sugars_nid = 'free_sugars_g'
        sugars_proxy = 'sugars_total_g'

        # 衍生钠：日与每份
        sodium_mg_day_max = int(sodium_mg_max)
        sodium_mg_serving_max = int(round(sodium_mg_day_max * per_serv_frac))

        out = {
            "user_id": user_id,
            "n_liked": n_liked,
            # 能量：新增 per-day 粒度，同时保留旧键（legacy）
            "energy_kcal_target_day": round(E_daily, 2),
            "energy_kcal_target": round(E_daily, 2),  # legacy
            "amdr": {
                "carb":   {"target_pct": round(amdr_target['carb'], 2),   "min_pct": AMDR['carb']['min'],   "max_pct": AMDR['carb']['max']},
                "protein":{"target_pct": round(amdr_target['protein'], 2),"min_pct": AMDR['protein']['min'],"max_pct": AMDR['protein']['max']},
                "fat":    {"target_pct": round(amdr_target['fat'], 2),    "min_pct": AMDR['fat']['min'],    "max_pct": AMDR['fat']['max']},
            },
            # 钠：日与每份
            "sodium_mg_day_max": int(sodium_mg_day_max),
            "sodium_mg_serving_max": int(sodium_mg_serving_max),
            "free_sugars_pct_max": round(float(free_sugars_pct_max), 2),
            "fiber_g_min": round(float(fiber_g_min), 2),
            # 宏量（克）：日与每份
            "macros_g_per_day": {"carb": carb_g_day, "protein": protein_g_day, "fat": fat_g_day},
            "macros_g_per_serving": {"carb": carb_g_serv, "protein": protein_g_serv, "fat": fat_g_serv},
            # 纤维/饱和脂肪/自由糖：日与每份
            "fiber_g_day_min": fiber_g_day_min,
            "fiber_g_serving_min": fiber_g_serving_min,
            "sfa_g_day_max": sfa_g_day_max,
            "sfa_g_serving_max": sfa_g_serving_max,
            "sugars_g_day_max": sugars_g_day_max,
            "sugars_g_serving_max": sugars_g_serving_max,
            # 自由糖（或其 proxy）的归一输出结构
            "sugars": {
                "nid": sugars_nid,
                "g_day_max": sugars_g_day_max,
                "g_serving_max": sugars_g_serving_max
            },
            "audit": {
                "guideline_refs": {
                    "AMDR": {"source": "IOM/DRI 2005", "ranges_pct": AMDR},
                    "WHO_free_sugars": {"source": "WHO 2015", "limit_pct": 10, "strict_pct": 5},
                    "Sodium": {"source": args.sodium_guideline, "upper_mg": SODIUM_MAX.get(args.sodium_guideline, 2000)},
                    "Fiber": {"source": args.fiber_guideline, "rule": ("14 g/1000 kcal" if args.fiber_guideline=='IOM' else "25 g/d")},
                    "FatQuality": FAT_QUALITY_LIMITS,
                },
                # 运行与兼容说明
                "per_meal_fraction_effective": float(per_serv_frac),
                "meals_per_day": float(args.meals_per_day),
                "legacy_fields": ["energy_kcal_target"],
                "free_sugars_proxy": sugars_proxy,
                "medians_from_likes": {
                    "recipe_energy_kcal": None if not math.isfinite(E_med) else round(float(E_med), 2),
                    "macro_pct": {
                        "carb": None if not math.isfinite(getattr(row, 'carb_pct_med')) else round(float(getattr(row, 'carb_pct_med')), 2),
                        "protein": None if not math.isfinite(getattr(row, 'protein_pct_med')) else round(float(getattr(row, 'protein_pct_med')), 2),
                        "fat": None if not math.isfinite(getattr(row, 'fat_pct_med')) else round(float(getattr(row, 'fat_pct_med')), 2),
                    },
                    "sugar_pct_total": None if not math.isfinite(sugar_med) else round(float(sugar_med), 2),
                    "fiber_g_per_1000kcal": None if not math.isfinite(fib_dens) else round(float(fib_dens), 2),
                    "sodium_mg_per_1000kcal": None if not math.isfinite(sod_dens) else round(float(sod_dens), 2),
                    "sfa_mufa_pufa_pct": {
                        "sfa": None if not math.isfinite(getattr(row, 'sfa_pct_med')) else round(float(getattr(row, 'sfa_pct_med')), 2),
                        "mufa": None if not math.isfinite(getattr(row, 'mufa_pct_med')) else round(float(getattr(row, 'mufa_pct_med')), 2),
                        "pufa": None if not math.isfinite(getattr(row, 'pufa_pct_med')) else round(float(getattr(row, 'pufa_pct_med')), 2),
                    },
                },
                "availability": avail,
                "assumptions": {
                    "keep_set": args.keep_set,
                    "meals_per_day": args.meals_per_day,
                    "energy_clip_kcal": [1200, 3500],
                    "winsor_pct": args.winsor_pct,
                    "pos_thresh": args.pos_thresh,
                }
            }
        }
        # === constraints[]：统一 per-serving 约束（lo/up + tolerance）===
        tol = float(getattr(args, 'tolerance', 0.05))
        constraints = []
        # 1) 钠（上限）
        constraints.append({"nid": "sodium_mg", "lo": 0, "up": int(sodium_mg_serving_max), "per": "serving", "tolerance": tol})
        # 2) 饱和脂肪（上限）
        constraints.append({"nid": "saturated_fat_g", "lo": 0, "up": sfa_g_serving_max, "per": "serving", "tolerance": tol})
        # 3) 自由糖（或其 proxy）（上限）
        sugar_con = {"nid": sugars_nid, "lo": 0, "up": sugars_g_serving_max, "per": "serving", "tolerance": tol}
        if sugars_proxy:
            sugar_con["proxy"] = sugars_proxy
        constraints.append(sugar_con)
        # 4) 纤维（下限）
        constraints.append({"nid": "fiber_g", "lo": fiber_g_serving_min, "up": None, "per": "serving", "tolerance": 0.0})
        # 5) 三大宏量（双侧围绕目标克数）
        def around(center: float):
            return (round(center * (1.0 - tol), 2), round(center * (1.0 + tol), 2))
        lo, up = around(carb_g_serv);    constraints.append({"nid": "carbohydrates_g", "lo": lo, "up": up, "per": "serving", "tolerance": tol})
        lo, up = around(protein_g_serv); constraints.append({"nid": "protein_g",       "lo": lo, "up": up, "per": "serving", "tolerance": tol})
        lo, up = around(fat_g_serv);     constraints.append({"nid": "fat_g",           "lo": lo, "up": up, "per": "serving", "tolerance": tol})
        # 6) 能量（双侧围绕目标能量/份）
        kcal_target_serv = round(energy_kcal_target * per_serv_frac, 1)
        lo_k = round(kcal_target_serv * (1.0 - tol), 1)
        up_k = round(kcal_target_serv * (1.0 + tol), 1)
        constraints.append({"nid": "calories_kcal", "lo": lo_k, "up": up_k, "per": "serving", "tolerance": tol})

        out["constraints"] = constraints

        out_rows.append(out)

    # 写出 JSONL
    print(f"[step3] 写出结果...")
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[step3] 完成: {args.out}  users={len(out_rows)}  keep_set={args.keep_set}  sodium_guideline={args.sodium_guideline}")


if __name__ == '__main__':
    main()
