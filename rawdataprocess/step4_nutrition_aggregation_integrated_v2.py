# -*- coding: utf-8 -*-
"""
step4_nutrition_aggregation.py — 步骤4：营养聚合（整合增强版）
目标：
  1) 在不丢失 USDA 营养类别的前提下，尽可能“全量”聚合食谱营养（长表永远写出）；
  2) 提供可选的单位规范化（默认仅能量 kJ→kcal，避免不安全的 IU 转换）；
  3) 明确并落盘克重来源（household / heuristic / default），便于审计与论文呈现；
  4) 宽表仅在单位一致时生成，避免“同名多单位”造成的错误合并；
  5) 加强内存友好：仅对所需 FDC 子集 join、分批聚合、阶段压缩。

输入（来自 step1–3）：
  - ingredients_processed.parquet（需含：recipe_id, ingredient_norm, ingredient_raw, qty_parsed, unit_std, grams）
  - ingredient_mapping.parquet（建议含：ingredient_norm, fdc_id, fdc_desc, fuzz, score, 以及可选 selected=1）
  - food_nutrient_processed.parquet（需含：fdc_id, nutrient_id, amount；建议 amount 已规范为“每 100 g”口径）
  - nutrient_processed.parquet（需含：id, name, unit_name）
  - 可选：recipes_processed.parquet（若含 servings，可派生每份营养）
  - 可选：household_weights_A.csv（A 表：fdc_id, unit, grams_per_unit）

输出：
  - recipe_nutrients_long.parquet（原始单位的“长表”：recipe_id, nutrient_name, unit, amount_for_recipe）
  - recipe_nutrients_long_canonical.parquet（按规则规范化后的“长表”，默认仅 Energy 统一为 kcal）
  - recipe_nutrients_wide.parquet（可选；仅在单位一致且营养种类不多时写出）
  - recipe_nutrients_core.parquet（核心字段，附带单位注释列）
  - ingredients_with_grams.parquet（带 grams_source 的行级审计表）
  - unmatched_ingredients.csv（未能映射到 FDC 的配料）
  - present_nutrients_in_recipes.csv（本次聚合实际出现的营养名-单位组合）
  - present_nutrients_in_usda.csv（USDA nutrient 表中的营养名-单位全集）
  - unit_conflicts.csv（同名多单位的营养素列表）

使用：
  python work/recipebench/scripts/rawdataprocess/step4_nutrition_aggregation_integrated_v2.py --out_dir work/recipebench/data/4out \
      --default_grams 30 --write_wide --wide_max_nutrients 256 \
      --agg_batch_size 200000 --household_weights work/recipebench/data/3out/household_weights_A.csv \
      --canonicalize_units --strict_units --per_serving


"""

import os
import argparse
import math
import warnings
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

from common_utils import (
    load_config, save_config,
    check_step_completion, mark_step_completed
)

# ---------------------------- 常量与规则 ----------------------------

ENERGY_KJ_TO_KCAL = 0.239005736

# 可扩展的单位规范化白名单（尽量保守：默认只处理能量 kJ→kcal）
CANON_RULES = {
    # 规范后的 key: {'aliases': [(name, unit), ...], 'target_unit': 'kcal', 'factor': lambda amt: ...}
    "Energy (kcal)": {
        "aliases": [
            ("Energy", "kJ"),
            ("Energy (Atwater General Factors)", "kJ"),
            ("Energy (Atwater Specific Factors)", "kJ"),
        ],
        "target_names": [
            "Energy", "Energy (Atwater General Factors)", "Energy (Atwater Specific Factors)"
        ],
        "target_unit": "kcal",
        "factor": lambda amt: amt * ENERGY_KJ_TO_KCAL,
    },
    # 可以在你确认数据语义后，安全地逐步加入 IU→质量单位 的规范化；默认关闭（见 CLI 开关）
}


# ---------------------------- 工具函数 ----------------------------

def _assert_or_note_amount_basis(fn_df: pd.DataFrame, assume_per_100g: bool = True) -> None:
    """
    检查 food_nutrient_processed.amount 的口径。
    目前多数管线在 step1 已将 amount 规范为“每 100 g”。若无法确认，允许 assume_per_100g=True 并打印提示。
    """
    basis_col = None
    for cand in ["amount_basis", "basis", "per_100g_flag"]:
        if cand in fn_df.columns:
            basis_col = cand
            break
    if basis_col is not None:
        uniq = fn_df[basis_col].dropna().astype(str).str.lower().unique().tolist()
        if not any(("100" in x) or ("per_100" in x) for x in uniq):
            warnings.warn(f"检测到 {basis_col}={uniq}，似乎不是“每100g”口径，请确认前序规范化。")
    else:
        if assume_per_100g:
            print("⚠️ 未检测到 amount 的口径标注，按“每 100 g”口径计算（可通过 --assume_per_100g 关闭）。")
        else:
            warnings.warn("未检测到 amount 的口径标注，且未启用 assume_per_100g，结果可能失真。")


def _select_best_mapping(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 ingredient_mapping 精简为单候选：优先 selected=1，否则按 score,fuzz 取 top1。
    """
    cols = ["ingredient_norm", "fdc_id", "fdc_desc", "fuzz", "score"]
    keep = [c for c in cols if c in mapping_df.columns]
    m = mapping_df[keep].dropna(subset=["ingredient_norm"]).copy()

    if "selected" in mapping_df.columns:
        # 优先 selected=1
        sel = mapping_df[mapping_df["selected"] == 1]
        if not sel.empty:
            m = sel[keep].copy()

    # 按 (ingredient_norm, score, fuzz) 排序并去重取首
    if "score" not in m.columns:
        m["score"] = np.nan
    if "fuzz" not in m.columns:
        m["fuzz"] = np.nan

    m = (m.sort_values(["ingredient_norm", "score", "fuzz"], ascending=[True, False, False])
           .drop_duplicates(subset=["ingredient_norm"], keep="first"))
    # fdc_id → Int64
    m["fdc_id"] = pd.to_numeric(m["fdc_id"], errors="coerce").astype("Int64")
    return m


def _apply_household_grams(x: pd.DataFrame, household_df: pd.DataFrame | None) -> pd.Series:
    """
    使用 A 表（household）优先计算克重：qty_parsed * grams_per_unit
    需 x 含：fdc_id, unit_std, qty_parsed
    """
    if household_df is None or household_df.empty:
        return pd.Series(np.nan, index=x.index, dtype="float32")

    try:
        hh = household_df[["fdc_id", "unit", "grams_per_unit"]].dropna().copy()
        hh["fdc_id"] = pd.to_numeric(hh["fdc_id"], errors="coerce").astype("Int64")
        hh["unit"] = hh["unit"].astype(str)
        hh["grams_per_unit"] = pd.to_numeric(hh["grams_per_unit"], errors="coerce").astype("float32")
        xx = x.merge(hh, left_on=["fdc_id", "unit_std"], right_on=["fdc_id", "unit"], how="left")
        grams_by_household = (xx["qty_parsed"].astype(float) * xx["grams_per_unit"].astype(float)).astype("float32")
        return grams_by_household.reindex(x.index)  # 对齐原索引
    except Exception as e:
        warnings.warn(f"household 克重计算失败，将回退：{e}")
        return pd.Series(np.nan, index=x.index, dtype="float32")


def _determine_grams(x: pd.DataFrame, default_g: float) -> pd.DataFrame:
    """
    选择最终克重 grams_eff，并标注 grams_source。
    优先 household，其次 heuristic（x['grams']），否则 default。
    """
    x = x.copy()
    x["grams_household"] = _apply_household_grams(x, x.attrs.get("_household_df"))
    # 先择优，再回退
    x["grams_pref"] = x["grams_household"].where(x["grams_household"].notna(), x.get("grams"))
    x["grams_eff"] = x["grams_pref"].where(x["grams_pref"].notna(), default_g).astype("float32")
    # 标注来源
    x["grams_source"] = np.where(x["grams_household"].notna(), "household",
                           np.where(x.get("grams").notna(), "heuristic", "default"))
    return x


def _canonicalize_units(grp: pd.DataFrame, enable: bool, enable_extra: bool = False) -> pd.DataFrame:
    """
    单位规范化（保守）：默认仅能量 kJ→kcal。
    enable_extra=True 时可逐步放开更多规则（需确保语义安全）。
    """
    if not enable:
        return grp

    df = grp.copy()
    # 处理能量相关
    for rule_key, rule in CANON_RULES.items():
        # 目前仅能量
        if rule_key.startswith("Energy"):
            # 将别名项（kJ）转换到 kcal，并将名称/单位替换成目标
            mask = pd.Series(False, index=df.index)
            for (alias_name, alias_unit) in rule["aliases"]:
                mask |= (df["nutrient_name"] == alias_name) & (df["unit"] == alias_unit)
            if mask.any():
                df.loc[mask, "amount_for_recipe"] = rule["factor"](df.loc[mask, "amount_for_recipe"])
                df.loc[mask, "unit"] = rule["target_unit"]
                # 名称统一到首个 target_names（保留来源信息可另加列；此处从简）
                df.loc[mask, "nutrient_name"] = rule["target_names"][0]

    # TODO：当 enable_extra=True 时，在你确认语义后逐步加入 IU→质量单位 的规则
    return df


def _safe_pivot_wide(grp: pd.DataFrame, wide_max_nutrients: int, strict_units: bool) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    尝试生成宽表；若同名多单位且 strict_units=True，则跳过并给出 unit_conflicts。
    返回 (wide_df_or_None, unit_conflicts_df_or_None)
    """
    # 检查同名多单位
    unit_cnt = grp.groupby("nutrient_name", observed=True)["unit"].nunique()
    multi = unit_cnt[unit_cnt > 1]
    unit_conflicts = None
    if not multi.empty:
        unit_conflicts = (grp[grp["nutrient_name"].isin(multi.index)]
                          .drop_duplicates(subset=["nutrient_name", "unit"])
                          .sort_values(["nutrient_name", "unit"])
                          .loc[:, ["nutrient_name", "unit"]]
                          .reset_index(drop=True))
        if strict_units:
            print(f"⚠️ 检测到 {len(multi)} 个营养素存在“同名多单位”，已跳过宽表以避免错误合并。")

    # 种类过多也跳过，以免 OOM
    num_nutrients = int(grp["nutrient_name"].nunique())
    if num_nutrients > int(wide_max_nutrients):
        print(f"⚠️ 营养种类过多 ({num_nutrients} > {wide_max_nutrients})，跳过宽表生成。")
        return None, unit_conflicts

    if strict_units and (unit_conflicts is not None):
        return None, unit_conflicts

    print("   正在生成宽表格式...")
    wide = (grp.pivot_table(index="recipe_id",
                            columns="nutrient_name",
                            values="amount_for_recipe",
                            aggfunc="sum")
               .reset_index()
               .fillna(0.0))
    wide.columns.name = None
    return wide, unit_conflicts


def _maybe_add_per_serving(wide_or_core: pd.DataFrame, recipes_df: pd.DataFrame | None, suffix: str = "_per_serving") -> pd.DataFrame:
    """
    若提供 recipes_processed.parquet 且含 servings，则派生“每份”指标。
    """
    if recipes_df is None or "servings" not in recipes_df.columns:
        return wide_or_core

    df = wide_or_core.copy()
    # 兼容两种主键列：既支持 id，也支持 recipe_id
    if "id" in recipes_df.columns and "servings" in recipes_df.columns:
        servings = recipes_df[["id", "servings"]].rename(columns={"id": "recipe_id"})
    elif "recipe_id" in recipes_df.columns and "servings" in recipes_df.columns:
        servings = recipes_df[["recipe_id", "servings"]].copy()
    else:
        return wide_or_core
    servings["servings"] = pd.to_numeric(servings["servings"], errors="coerce")
    df = df.merge(servings, on="recipe_id", how="left")

    # 对数值列做 /servings
    num_cols = [c for c in df.columns if c not in {"recipe_id", "servings"} and pd.api.types.is_numeric_dtype(df[c])]
    with np.errstate(divide="ignore", invalid="ignore"):
        for c in num_cols:
            df[c + suffix] = (df[c] / df["servings"]).astype("float32")

    df = df.drop(columns=["servings"])
    return df


# ---------------------------- 主流程 ----------------------------

def aggregate_recipe_nutrition(
    ingr_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    fn: pd.DataFrame,
    nutr: pd.DataFrame,
    out_dir: str,
    default_g: float = 30.0,
    write_wide: bool = False,
    wide_max_nutrients: int = 512,
    agg_batch_size: int = 200000,
    household_df: pd.DataFrame | None = None,
    canonicalize_units: bool = True,
    canonicalize_extra: bool = False,
    strict_units: bool = True,
    assume_per_100g: bool = True,
    recipes_df: pd.DataFrame | None = None,
    write_core: bool = True,
) -> dict:
    """
    聚合并导出多个中间/最终产物。返回一个包含关键 DataFrame 的字典，便于调试。

    重要：长表（recipe_nutrients_long.parquet）始终写出且“全量保留单位”，
          这样你在论文/审计时可以覆盖 USDA 的所有营养类别，而不会因单位冲突而丢失任何信息。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 选择单候选映射
    m = _select_best_mapping(mapping_df)

    # 2) 合并映射到配料明细
    x = ingr_df.merge(m, on="ingredient_norm", how="left")
    x["fdc_id"] = pd.to_numeric(x["fdc_id"], errors="coerce").astype("Int64")
    # 挂载 A 表以便 _determine_grams 使用
    x.attrs["_household_df"] = household_df

    # 3) 克重选择与来源标注
    x = _determine_grams(x, default_g=default_g)

    # A 表命中率体检（聚合前快速查看潜在命中率）
    if household_df is not None:
        try:
            tmp_u = (x[["fdc_id", "unit_std"]].dropna()
                        .assign(
                            fdc_id=lambda d: pd.to_numeric(d["fdc_id"], errors="coerce").astype("Int64"),
                            unit_std=lambda d: d["unit_std"].astype(str).str.strip().str.lower(),
                        ))
            hh_u = (household_df[["fdc_id", "unit"]].dropna()
                        .assign(
                            fdc_id=lambda d: pd.to_numeric(d["fdc_id"], errors="coerce").astype("Int64"),
                            unit=lambda d: d["unit"].astype(str).str.strip().str.lower(),
                        ))
            hit = tmp_u.merge(hh_u, left_on=["fdc_id", "unit_std"], right_on=["fdc_id", "unit"], how="inner")
            hit_rate = len(hit) / max(len(tmp_u), 1)
            print(f"   ▶︎ Household潜在命中率（fdc_id×unit_std 对齐后）≈ {hit_rate:.2%}  (rows: {len(hit)}/{len(tmp_u)})")
        except Exception as e:
            print(f"   ▶︎ Household命中率体检跳过：{e}")

    # 清理不可序列化的 attrs，避免 to_parquet 失败（pandas 会尝试序列化 DataFrame.attrs）
    # 此处仅用于 _determine_grams 的临时 A 表引用，写盘前可安全移除
    try:
        x.attrs.clear()
    except Exception:
        pass

    # 审计输出
    x.to_parquet(os.path.join(out_dir, "ingredients_with_grams.parquet"), index=False)
    (x[x["fdc_id"].isna()][["ingredient_raw", "ingredient_norm"]]
        .drop_duplicates()
        .to_csv(os.path.join(out_dir, "unmatched_ingredients.csv"), index=False, encoding="utf-8"))

    # 统计克重来源（行级）
    grams_stats = {
        "n_rows_with_household_grams": int((x["grams_source"] == "household").sum()),
        "n_rows_with_heuristic_grams": int((x["grams_source"] == "heuristic").sum()),
        "n_rows_with_default_grams": int((x["grams_source"] == "default").sum()),
    }
    default_used_recipes = int(x[(x["fdc_id"].notna()) & (x["grams_source"] == "default")]["recipe_id"].nunique())

    mx = x[x["fdc_id"].notna()].copy()
    if mx.empty:
        raise RuntimeError("没有匹配到任何 fdc_id，请检查对齐阈值/输入路径/解析逻辑。")

    mx["fdc_id"] = mx["fdc_id"].astype("int64")
    mx["recipe_id"] = pd.to_numeric(mx["recipe_id"], errors="coerce").astype("int64")
    mx["grams_eff"] = pd.to_numeric(mx["grams_eff"], errors="coerce").astype("float32")

    # 4) 仅取需要的 FDC 子集，减内存
    fdc_needed = mx["fdc_id"].unique()
    fn_small = fn.loc[fn["fdc_id"].isin(fdc_needed), ["fdc_id", "nutrient_id", "amount"]].copy()
    fn_small["fdc_id"] = fn_small["fdc_id"].astype("int64")
    fn_small["nutrient_id"] = pd.to_numeric(fn_small["nutrient_id"], errors="coerce").astype("int64")
    fn_small["amount"] = pd.to_numeric(fn_small["amount"], errors="coerce").astype("float32")

    nutr_small = nutr[["id", "name", "unit_name"]].copy()
    nutr_small["id"] = pd.to_numeric(nutr_small["id"], errors="coerce").astype("int64")

    # 5) 口径断言/提示
    _assert_or_note_amount_basis(fn, assume_per_100g=assume_per_100g)

    # 6) 分批聚合（确保“长表全量”）
    print(f"   正在分批处理 {len(mx)} 条记录，批次大小: {agg_batch_size}")
    parts = []
    n = len(mx)
    for start in tqdm(range(0, n, agg_batch_size), total=math.ceil(n / agg_batch_size), desc="营养聚合批次"):
        end = min(start + agg_batch_size, n)
        sub = mx.iloc[start:end].copy()
        sub = sub.merge(fn_small, on="fdc_id", how="left")
        if sub.empty:
            continue
        sub = sub.merge(nutr_small, left_on="nutrient_id", right_on="id", how="left")
        sub = sub.rename(columns={"name": "nutrient_name", "unit_name": "unit"})
        sub = sub[["recipe_id", "nutrient_name", "unit", "amount", "grams_eff"]]
        # 关键：amount 为“每 100 g” → 乘以 grams_eff/100
        sub["amount_for_recipe"] = (sub["amount"].astype("float32")
                                    * (sub["grams_eff"].fillna(default_g).astype("float32") / 100.0)).astype("float32")
        part = (sub
                .drop(columns=["amount", "grams_eff"], errors="ignore")
                .groupby(["recipe_id", "nutrient_name", "unit"], as_index=False, observed=True)["amount_for_recipe"]
                .sum())
        parts.append(part)

        # 阶段合并，控峰
        if len(parts) >= 8:
            tmp = pd.concat(parts, ignore_index=True)
            parts = [tmp.groupby(["recipe_id", "nutrient_name", "unit"], as_index=False, observed=True)["amount_for_recipe"].sum()]
            del tmp
            gc.collect()

    if not parts:
        raise RuntimeError("聚合阶段没有生成任何数据，可能前序映射为空。")

    print("   正在合并所有批次结果...")
    grp = pd.concat(parts, ignore_index=True)
    grp = grp.groupby(["recipe_id", "nutrient_name", "unit"], as_index=False, observed=True)["amount_for_recipe"].sum()

    # 7) 写出“原始单位长表”（最全、最安全）
    grp.to_parquet(os.path.join(out_dir, "recipe_nutrients_long.parquet"), index=False)

    # 8) 单位规范化（保守：默认仅能量）并写出“规范化长表”
    grp_canon = _canonicalize_units(grp, enable=canonicalize_units, enable_extra=canonicalize_extra)
    grp_canon.to_parquet(os.path.join(out_dir, "recipe_nutrients_long_canonical.parquet"), index=False)

    # 9) （可选）宽表（仅在单位一致 & 营养种类不多时生成）
    wide, unit_conflicts = _safe_pivot_wide(grp_canon, wide_max_nutrients=wide_max_nutrients, strict_units=strict_units)
    if unit_conflicts is not None and not unit_conflicts.empty:
        unit_conflicts.to_csv(os.path.join(out_dir, "unit_conflicts.csv"), index=False)
    if (wide is not None) and write_wide:
        wide.to_parquet(os.path.join(out_dir, "recipe_nutrients_wide.parquet"), index=False)

    # 9.1) 宽表（安全版）：列键为“营养名__单位”，可在单位冲突时仍产出
    try:
        wide_safe = (grp_canon
                        .assign(col=lambda d: d["nutrient_name"] + "__" + d["unit"])
                        .pivot_table(index="recipe_id", columns="col", values="amount_for_recipe", aggfunc="sum")
                        .reset_index()
                        .fillna(0.0))
        if write_wide:
            wide_safe.to_parquet(os.path.join(out_dir, "recipe_nutrients_wide_safe.parquet"), index=False)
    except Exception as e:
        print(f"   ▶︎ 生成宽表（安全版）失败：{e}")

    # 10) 核心营养（从“规范化长表”快速透视，仅四大宏量 + 能量）
    core = None
    if write_core:
        target = grp_canon[grp_canon["nutrient_name"].isin([
            "Energy", "Energy (Atwater General Factors)", "Energy (Atwater Specific Factors)",
            "Protein", "Total lipid (fat)", "Carbohydrate, by difference"
        ])].copy()
        core_wide = (target.pivot_table(index="recipe_id",
                                        columns="nutrient_name",
                                        values="amount_for_recipe",
                                        aggfunc="sum")
                            .reset_index()
                            .fillna(0.0))
        core_wide.columns.name = None
        def pick(cols, cand):
            for c in cand:
                if c in cols: return c
            return None

        cols = set(core_wide.columns)
        energy = pick(cols, ["Energy", "Energy (Atwater General Factors)", "Energy (Atwater Specific Factors)"])
        protein = pick(cols, ["Protein"])
        fat     = pick(cols, ["Total lipid (fat)"])
        carbs   = pick(cols, ["Carbohydrate, by difference"])

        core = core_wide[["recipe_id"]].copy()
        core["calories_kcal"]   = core_wide[energy] if energy else 0.0
        core["protein_g"]       = core_wide[protein] if protein else 0.0
        core["fat_g"]           = core_wide[fat] if fat else 0.0
        core["carbohydrates_g"] = core_wide[carbs] if carbs else 0.0

        # 增加单位注释列
        core["calories_unit"] = "kcal"
        core["protein_unit"] = "g"
        core["fat_unit"] = "g"
        core["carbohydrates_unit"] = "g"

        # 若有 servings，增加“每份”派生列
        core = _maybe_add_per_serving(core, recipes_df=recipes_df)

        core.to_parquet(os.path.join(out_dir, "recipe_nutrients_core.parquet"), index=False)

    # 11) 现身营养列表（本次聚合 & USDA 全集）
    try:
        present_nutr = (grp
                        .loc[:, ["nutrient_name", "unit"]]
                        .drop_duplicates()
                        .sort_values(["nutrient_name", "unit"])
                        .reset_index(drop=True))
        present_nutr.to_csv(os.path.join(out_dir, "present_nutrients_in_recipes.csv"), index=False)
    except Exception as e:
        warnings.warn(f"导出 present_nutrients_in_recipes.csv 失败: {e}")

    try:
        usda_nutr = (nutr.loc[:, ["name", "unit_name"]]
                         .drop_duplicates()
                         .rename(columns={"name": "nutrient_name", "unit_name": "unit"})
                         .sort_values(["nutrient_name", "unit"])
                         .reset_index(drop=True))
        usda_nutr.to_csv(os.path.join(out_dir, "present_nutrients_in_usda.csv"), index=False)
    except Exception as e:
        warnings.warn(f"导出 present_nutrients_in_usda.csv 失败: {e}")

    # 12) 严/松两次聚合（可选，基于外部标注 grams_conf/grams_used）
    try:
        # 读取外部标注与覆盖信息（若存在）
        lbl_path = os.path.join("work/recipebench/data/5guard", "ingredients_labeled.parquet")
        cov_path = os.path.join("work/recipebench/data/5guard", "recipe_coverage.parquet")
        if os.path.exists(lbl_path):
            ing_lbl = pd.read_parquet(lbl_path)
            # 规范列名
            rcol = "RecipeId" if "RecipeId" in ing_lbl.columns else ("recipe_id" if "recipe_id" in ing_lbl.columns else None)
            if rcol is None:
                raise KeyError("ingredients_labeled.parquet 缺少 RecipeId/recipe_id 列")

            # 构造 nutr_long 临时表：按 fdc_id × nutrient_id 提供 amount_per_100g
            nutr_long_tmp = (mx[["recipe_id", "fdc_id"]]
                             .drop_duplicates()
                             .merge(fn_small[["fdc_id", "nutrient_id", "amount"]], on="fdc_id", how="left"))
            nutr_long_tmp = nutr_long_tmp.rename(columns={"recipe_id": "recipe_id", "amount": "amount_per_100g"})

            # 合并标注（grams_conf/grams_used），支持不同大小写列名
            grams_conf_col = "grams_conf" if "grams_conf" in ing_lbl.columns else None
            grams_used_col = "grams_used" if "grams_used" in ing_lbl.columns else None
            if grams_conf_col is None or grams_used_col is None:
                raise KeyError("ingredients_labeled.parquet 缺少 grams_conf/grams_used 列")

            join_lbl = ing_lbl.rename(columns={rcol: "recipe_id"})
            if "fdc_id" not in join_lbl.columns:
                raise KeyError("ingredients_labeled.parquet 缺少 fdc_id 列")
            wide = nutr_long_tmp.merge(join_lbl[["recipe_id", "fdc_id", grams_conf_col, grams_used_col]],
                                       on=["recipe_id", "fdc_id"], how="left")

            # 尝试获取分类列用于类别中位密度；若无则退化为按 nutrient_id 的中位数
            fdc_cat_col = None
            for cand in ["fdc_cat", "food_category_id", "fdc_category"]:
                if cand in wide.columns:
                    fdc_cat_col = cand
                    break
            # strict: 仅 A/B
            strict = wide[wide[grams_conf_col].isin(["A", "B"])].copy()
            if not strict.empty:
                strict["nutrient_g"] = (pd.to_numeric(strict[grams_used_col], errors="coerce").astype("float32")
                                         * (pd.to_numeric(strict["amount_per_100g"], errors="coerce").astype("float32") / 100.0))
                nutr_strict = (strict.groupby(["recipe_id", "nutrient_id"], observed=True)["nutrient_g"]
                                     .sum().rename("nutr_strict").reset_index())
            else:
                nutr_strict = pd.DataFrame(columns=["recipe_id", "nutrient_id", "nutr_strict"])

            # relaxed: A/B 用 grams_used；C 用 类别中位密度 × 默认质量
            DEFAULT_G = float(default_g)
            rel = wide.copy()
            if fdc_cat_col is not None:
                cat_median = (rel.groupby([fdc_cat_col, "nutrient_id"], observed=True)["amount_per_100g"].median()
                                 .rename("cat_med").reset_index())
                rel = rel.merge(cat_median, on=[fdc_cat_col, "nutrient_id"], how="left")
            else:
                # 无分类列：按 nutrient_id 的中位数作为退化近似
                cat_median = (rel.groupby(["nutrient_id"], observed=True)["amount_per_100g"].median()
                                 .rename("cat_med").reset_index())
                rel = rel.merge(cat_median, on=["nutrient_id"], how="left")

            rel["grams_relaxed"] = np.where(rel[grams_conf_col].isin(["A", "B"]),
                                             pd.to_numeric(rel[grams_used_col], errors="coerce"),
                                             DEFAULT_G).astype("float32")
            rel["density_relaxed"] = np.where(rel[grams_conf_col].isin(["A", "B"]),
                                                pd.to_numeric(rel["amount_per_100g"], errors="coerce"),
                                                pd.to_numeric(rel["cat_med"], errors="coerce")).astype("float32")
            rel["nutrient_g"] = (rel["grams_relaxed"] * (rel["density_relaxed"] / 100.0)).astype("float32")

            nutr_relax = (rel.groupby(["recipe_id", "nutrient_id"], observed=True)["nutrient_g"]
                            .sum().rename("nutr_relaxed").reset_index())

            out_sr = nutr_strict.merge(nutr_relax, on=["recipe_id", "nutrient_id"], how="outer")
            sr_dir = os.path.join(out_dir, "..", "7nutr")
            os.makedirs(sr_dir, exist_ok=True)
            out_path = os.path.join(sr_dir, "recipe_nutrients_strict_relaxed.parquet")
            out_sr.to_parquet(out_path, index=False)

            # 覆盖信息另存（若存在）
            try:
                if os.path.exists(cov_path):
                    cov_df = pd.read_parquet(cov_path)
                    cov_df.to_parquet(os.path.join(sr_dir, "recipe_coverage.parquet"), index=False)
            except Exception as _:
                pass

            print("[ok] strict/relaxed nutrients written.")
        else:
            print("   ▶︎ 未找到 ingredients_labeled.parquet，跳过 strict/relaxed 聚合。")
    except Exception as e:
        warnings.warn(f"strict/relaxed 聚合失败：{e}")

    return {
        "ingredients_with_grams": x,
        "mx": mx,
        "grp_long": grp,
        "grp_long_canonical": grp_canon,
        "wide": wide,
        "core": core,
        "grams_stats": grams_stats,
        "default_used_recipes": default_used_recipes,
        "unit_conflicts": unit_conflicts,
    }


def main():
    parser = argparse.ArgumentParser(description="步骤4：营养聚合（整合增强版）")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--default_grams", type=float, default=30.0, help="当无法估算克重时的保守默认值")
    parser.add_argument("--write_wide", action="store_true", help="在营养种类不多且单位一致时，同时写出宽表")
    parser.add_argument("--wide_max_nutrients", type=int, default=512, help="写宽表的营养种类上限阈值")
    parser.add_argument("--agg_batch_size", type=int, default=200000, help="聚合阶段批大小，降低内存峰值")
    parser.add_argument("--household_weights", default=None, help="可选：A 表 CSV 路径（fdc_id, unit, grams_per_unit）")
    parser.add_argument("--canonicalize_units", action="store_true", help="启用单位规范化（默认仅能量 kJ→kcal）")
    parser.add_argument("--canonicalize_extra", action="store_true", help="在你确认语义后，放开更多规则（如 IU→质量单位）")
    parser.add_argument("--strict_units", action="store_true", help="若同名多单位则跳过宽表，避免错误合并")
    parser.add_argument("--assume_per_100g", action="store_true", help="无法确认 amount 口径时，按“每 100 g”计算（默认 False）")
    parser.add_argument("--per_serving", action="store_true", help="若 recipes 中有 servings，则输出每份营养派生列")
    parser.add_argument("--config", default=None, help="配置文件路径（可选）")

    args = parser.parse_args()

    # 步骤依赖检查
    if not check_step_completion("step1", args.out_dir):
        raise RuntimeError("步骤1未完成，请先运行 step1_data_preparation.py")
    if not check_step_completion("step2", args.out_dir):
        raise RuntimeError("步骤2未完成，请先运行 step2_embedding_generation.py")
    if not check_step_completion("step3", args.out_dir):
        raise RuntimeError("步骤3未完成，请先运行 step3_ingredient_alignment.py")

    # 加载配置
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = load_config(os.path.join(args.out_dir, "config.json"))

    # 更新配置（记录运行参数）
    config.default_grams = float(args.default_grams)
    config.write_wide = bool(args.write_wide)
    config.wide_max_nutrients = int(args.wide_max_nutrients)
    config.agg_batch_size = int(args.agg_batch_size)
    config.household_weights = args.household_weights
    config.canonicalize_units = bool(args.canonicalize_units)
    config.canonicalize_extra = bool(args.canonicalize_extra)
    config.strict_units = bool(args.strict_units)
    config.assume_per_100g = bool(args.assume_per_100g)
    config.per_serving = bool(args.per_serving)

    print(">> 步骤4：营养聚合（整合增强版）...")
    print(">> 加载数据...")
    ingr_df = pd.read_parquet(os.path.join(config.out_dir, "ingredients_processed.parquet"))
    # 兼容两种映射文件位置：优先 out_dir 根目录，其次 step3 子目录
    mapping_path_root = os.path.join(config.out_dir, "ingredient_mapping.parquet")
    mapping_path_step3 = os.path.join(config.out_dir, "step3", "ingredient_mapping.parquet")
    if os.path.exists(mapping_path_root):
        mapping_df = pd.read_parquet(mapping_path_root)
    elif os.path.exists(mapping_path_step3):
        print("   注意：从 step3 目录加载 ingredient_mapping.parquet")
        mapping_df = pd.read_parquet(mapping_path_step3)
    else:
        raise FileNotFoundError(
            f"找不到 ingredient_mapping.parquet（检查：{mapping_path_root} 或 {mapping_path_step3}）")
    fn = pd.read_parquet(os.path.join(config.out_dir, "food_nutrient_processed.parquet"))
    nutr = pd.read_parquet(os.path.join(config.out_dir, "nutrient_processed.parquet"))

    recipes_df = None
    if config.per_serving:
        rp = os.path.join(config.out_dir, "recipes_processed.parquet")
        if os.path.exists(rp):
            try:
                df_rp = pd.read_parquet(rp)
                if ("id" in df_rp.columns) and ("servings" in df_rp.columns):
                    recipes_df = df_rp[["id", "servings"]].rename(columns={"id": "recipe_id"})
                elif ("recipe_id" in df_rp.columns) and ("servings" in df_rp.columns):
                    recipes_df = df_rp[["recipe_id", "servings"]]
                else:
                    recipes_df = None
                if recipes_df is not None:
                    recipes_df["recipe_id"] = pd.to_numeric(recipes_df["recipe_id"], errors="coerce").astype("int64")
                    print(f"   载入 recipes_processed.parquet：{len(recipes_df)} 行（用于 per-serving 派生）。")
            except Exception as e:
                warnings.warn(f"读取 recipes_processed.parquet 失败，将跳过 per-serving：{e}")

    household_df = None
    if config.household_weights and os.path.exists(config.household_weights):
        try:
            household_df = pd.read_csv(config.household_weights)
            print(f"   载入 A 表（household）：{len(household_df)} 条记录")
        except Exception as e:
            warnings.warn(f"载入 household 失败，忽略：{e}")
            household_df = None

    print(">> 执行营养聚合...")
    results = aggregate_recipe_nutrition(
        ingr_df=ingr_df,
        mapping_df=mapping_df,
        fn=fn,
        nutr=nutr,
        out_dir=config.out_dir,
        default_g=config.default_grams,
        write_wide=config.write_wide,
        wide_max_nutrients=config.wide_max_nutrients,
        agg_batch_size=config.agg_batch_size,
        household_df=household_df,
        canonicalize_units=config.canonicalize_units,
        canonicalize_extra=config.canonicalize_extra,
        strict_units=config.strict_units,
        assume_per_100g=config.assume_per_100g,
        recipes_df=recipes_df,
        write_core=True,
    )

    # 保存配置
    save_config(config, os.path.join(config.out_dir, "config.json"))
    mark_step_completed("step4", config.out_dir)

    print(">> 步骤4完成！")
    print(f"   - 长表记录数（原始单位）: {len(results['grp_long'])}")
    print(f"   - 长表记录数（规范单位）: {len(results['grp_long_canonical'])}")
    print(f"   - 宽表状态: {'已写出' if (results['wide'] is not None and config.write_wide) else '未写出'}")
    print(f"   - 核心表状态: {'已写出' if (results['core'] is not None) else '未写出'}")
    print(f"   - 使用默认克重的食谱数: {results['default_used_recipes']}")
    print(f"   - Household克重行数: {results['grams_stats']['n_rows_with_household_grams']}")
    print(f"   - 启发式克重行数: {results['grams_stats']['n_rows_with_heuristic_grams']}")
    print(f"   - 默认克重行数: {results['grams_stats']['n_rows_with_default_grams']}")

if __name__ == "__main__":
    main()
