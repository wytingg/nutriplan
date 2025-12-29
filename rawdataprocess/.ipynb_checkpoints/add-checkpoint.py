# -*- coding: utf-8 -*-
"""
expand_minerals_from_macros.py
基于宏量营养（kcal, protein_g, fat_g, carbohydrates_g）推导 Na / Ca / Sugars / Fe 四列，
保证结果在正常食谱范围内（并进行上下限裁剪），且完全由已有四项推导，不使用随机数。

输入：recipe_nutrients_core_cleaned.csv 或你的核心营养CSV
输出：recipe_nutrients_core_expanded.csv（新增 sodium_mg, calcium_mg, sugar_g, iron_mg）
"""

import pandas as pd
import numpy as np
import os

# ----------- 可调参数（符合常见单道菜/单份的合理范围）-----------
# 钠：一份菜常见 200–1500mg；允许 100–2000mg 作为宽松边界
NA_MIN, NA_MAX = 200.0, 1500.0

# 钙：一份菜常见 50–400mg；允许 50–800mg 作为宽松边界
CA_MIN, CA_MAX = 50.0, 400.0

# 糖：为碳水的子集；常见 5–40g；允许 5–80g，且不超过碳水本身
SUG_MIN, SUG_MAX = 5.0, 40.0

# 铁：常见一餐 1–10mg；允许 0.3–12mg 作为宽松边界
# 保持全局阈值不变，但最终不再用全局下限夹死，而使用“动态下限”
FE_MIN, FE_MAX = 1.0, 10.0

# 钠与热量的线性基准（mg）——1000 kcal ≈ ~1800 mg 的量级
# sodium_base = NA_INTERCEPT + NA_SLOPE * calories
NA_INTERCEPT, NA_SLOPE = 300.0, 1.2  # 1000kcal -> 300 + 1200 = 1500mg（再按比例系数调整）

def safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    return np.divide(a, b)

def derive_shares(df):
    """
    由宏量计算能量占比（蛋白/脂肪/碳水的能量份额），用于“类型”判断与推导。
    """
    p_kcal = 4.0 * df["protein_g"].astype(float)
    f_kcal = 9.0 * df["fat_g"].astype(float)
    c_kcal = 4.0 * df["carbohydrates_g"].astype(float)
    kcal_from_macros = p_kcal + f_kcal + c_kcal

    # 以“从宏量算得的能量”为主来求占比；若为0，则回退到总热量
    denom = kcal_from_macros.copy()
    denom = np.where(denom <= 0, df["calories_kcal"].astype(float), denom)
    denom = np.where(denom <= 0, np.nan, denom)

    share_p = safe_div(p_kcal, denom)
    share_f = safe_div(f_kcal, denom)
    share_c = safe_div(c_kcal, denom)

    # 填缺省：若仍有空，则均分
    for s in (share_p, share_f, share_c):
        s[np.isnan(s)] = 1.0/3.0

    # 归一化确保相加约为1
    total = share_p + share_f + share_c
    total = np.where(total == 0, 1.0, total)
    share_p /= total
    share_f /= total
    share_c /= total
    return share_p, share_f, share_c

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def infer_sodium(calories_kcal, share_p, share_f, sugar_g, carbs_g):
    # 基线：与热量正相关
    base = 100.0 + 0.4 * calories_kcal.astype(float)  # 1000 kcal ≈ 500 mg
    # 调整：高蛋白高脂上调，高糖下调
    sweetness = safe_div(sugar_g.astype(float), np.maximum(carbs_g.astype(float), 1.0))
    taste = 1.0 + 0.2*share_p + 0.1*share_f - 0.3*clamp(sweetness, 0.0, 1.0)
    sodium = base * taste
    return clamp(sodium, 100.0, 1000.0)

def infer_sugar(carbs_g, share_c, share_f):
    # 糖是碳水的一部分，比例 0.1–0.3
    frac = 0.1 + 0.2 * share_c * (1 - 0.5*share_f)
    frac = clamp(frac, 0.1, 0.3)
    sugar = carbs_g.astype(float) * frac
    return clamp(sugar, 3.0, 25.0)


def infer_calcium(protein_g, share_p, share_f, share_c):
    """
    钙与“蛋白（尤其奶制品/豆制品）”相关；同时当脂肪和蛋白占比高、碳水低时更接近“奶酪/乳制品”特征：
      base = 9 * protein_g（每克蛋白 ~10mg 量级附近）
      dairy_like_boost = 200 * clamp( (share_p + 0.5*share_f - 0.5*share_c), 0, 1 )
    """
    base = 9.0 * protein_g.astype(float)
    dairy_like = (share_p + 0.5*share_f - 0.5*share_c)
    dairy_like = clamp(dairy_like, 0.0, 1.0)
    calcium = base + 200.0 * dairy_like
    calcium = clamp(calcium, CA_MIN, CA_MAX)
    return calcium

def infer_iron(protein_g, share_p, share_f, calories_kcal):
    """
    铁与“高蛋白菜肴（肉/豆）”相关；脂肪太高的“重油肉类”相对含铁密度未必更高。
    设计目标：
      - 低蛋白菜（例如 p≈5–10g）也能自然给到 ~0.6–1.0 mg，而不是被硬夹到 1.0；
      - 蛋白↑（尤其 share_p 高、share_f 不高的“瘦高蛋白/豆类”）→ 铁↑；
      - 全程无随机数，完全由宏量确定。
    公式（分三部分相加）：
      base:       0.05 * p                               （线性主项）
      curvature:  0.02 * p**0.5                          （低蛋白时补一点弯曲，避免过低）
      structure:  0.35*share_p + 0.10*(1 - share_f)      （瘦高蛋白结构有加成）
    然后套一个“动态下限”（随 p 和 share_p 微升），再做全局 clamp 到 [FE_MIN, FE_MAX]。
    """
    p = protein_g.astype(float)

    base = 0.05 * p
    curvature = 0.02 * np.sqrt(np.maximum(p, 0.0))
    structure = 0.35 * share_p + 0.10 * (1.0 - share_f)

    iron_raw = base + curvature + structure  # mg

    # 动态下限：低蛋白时下限更低，高蛋白时下限略抬高，避免整片菜肴铁值都为 1.0
    # 能量规模温和影响：高热量菜肴的动态下限略高，低热量略低
    kcal_norm = np.clip(calories_kcal.astype(float) / 800.0, 0.0, 1.0)  # 0~1 标准化（~800kcal 作为量级）
    fe_min_dyn = (
        0.35
        + 0.30 * np.clip(share_p, 0.0, 1.0)
        + 0.20 * np.clip(p, 0.0, 20.0) / 20.0
        - 0.05 * np.clip(share_f, 0.0, 1.0)
        + 0.10 * kcal_norm
    )
    # fe_min_dyn 大约落在 ~[0.35, 0.9] 区间，随菜品“结构”平滑变化
    iron = np.maximum(iron_raw, fe_min_dyn)

    # 最终仅用“动态下限”与全局上限进行裁剪：
    #  - 下限使用 fe_min_dyn（随蛋白/能量结构变化，不会整片被卡在同一值）
    #  - 上限仍保留 FE_MAX 作为兜底
    # 注意：不再用 FE_MIN 作为最终下限，保证真正的“动态下限”。
    iron = np.minimum(np.maximum(iron, fe_min_dyn), FE_MAX)
    return iron


def expand(file_in, file_out):
    df = pd.read_csv(file_in)
    # 容错：缺列则报错清晰
    required = {"calories_kcal","protein_g","fat_g","carbohydrates_g"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列：{missing}")

    # 计算能量占比
    share_p, share_f, share_c = derive_shares(df)

    # 推导四列
    sugar_g = infer_sugar(df["carbohydrates_g"], share_c, share_f)
    sodium_mg = infer_sodium(df["calories_kcal"], share_p, share_f, sugar_g, df["carbohydrates_g"])
    calcium_mg = infer_calcium(df["protein_g"], share_p, share_f, share_c)
    iron_mg = infer_iron(df["protein_g"], share_p, share_f, df["calories_kcal"])

    # ---------- 真实性保障（不改变推导公式，仅加物理/常识边界） ----------
    # 1) 糖：不超过碳水；不为负
    carbs_vec = df["carbohydrates_g"].astype(float)
    sugar_g = np.minimum(np.maximum(sugar_g, 0.0), np.maximum(0.0, carbs_vec))

    # 2) 钠：随热量规模设定合理上限（mg）
    #    常见每份 200–1500mg；极端高热量时不超过 ~2.5×kcal
    kcal_vec = df["calories_kcal"].astype(float)
    na_hi_by_cal = 2.5 * np.clip(kcal_vec, 0.0, None)
    sodium_mg = np.minimum(sodium_mg, np.minimum(NA_MAX, na_hi_by_cal))
    #    下限保持温和保守（不强行抬高）：至少 50mg
    sodium_mg = np.maximum(sodium_mg, 50.0)

    # 合并到数据框，保留小数点后两位
    out = df.copy()

    # 将已有宏量营养列统一保留两位小数（若存在）
    base_cols = ["calories_kcal","protein_g","fat_g","carbohydrates_g"]
    for c in base_cols:
        if c in out.columns:
            out[c] = np.round(out[c].astype(float), 2)
    out["sodium_mg"] = np.round(sodium_mg, 2)
    out["calcium_mg"] = np.round(calcium_mg, 2)
    out["sugar_g"] = np.round(sugar_g, 2)
    out["iron_mg"] = np.round(iron_mg, 2)

    out.to_csv(file_out, index=False, encoding="utf-8")
    print(f"[OK] Expanded CSV saved to: {os.path.abspath(file_out)}")

    # 简要统计，方便快速核查
    desc = out[["sodium_mg","calcium_mg","sugar_g","iron_mg"]].describe(percentiles=[0.5,0.9,0.95]).T
    print("\n=== New columns summary ===")
    print(desc.to_string())


if __name__ == "__main__":
    # 默认读取你前面生成的“清理后”文件；如需换源，改成你的路径
    INPUT = "work/recipebench/data/out/recipe_nutrients_core_cleaned.csv"
    OUTPUT = "work/recipebench/data/out/recipe_nutrients_core_expanded.csv"
    expand(INPUT, OUTPUT)
