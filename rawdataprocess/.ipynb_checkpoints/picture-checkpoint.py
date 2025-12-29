# -*- coding: utf-8 -*-
"""
quick_report.py — 食材频次 / 营养分布 快速报告（30秒版）
要求：
- 输入表含列：rid, title, ingredients（分号分隔）, 以及 8项营养：
  calories_kcal, protein_g, fat_g, carbohydrates_g, sugar_g, sodium_mg, calcium_mg, iron_mg
- 输出：
  - CSV：ingredients_freq_topK.csv, nutrient_stats.csv
  - PNG：ingredients_topK_bar.png, ingredients_cumcover.png（可选）, 8个营养直方图
注意：
- 仅使用 matplotlib；每张图单独绘制；不指定颜色。
"""

import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NUTR_COLS = [
    "calories_kcal","protein_g","fat_g","carbohydrates_g",
    "sugar_g","sodium_mg","calcium_mg","iron_mg"
]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def build_ingredient_freq(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    # ingredients 列：分号分隔
    ing = (
        df["ingredients"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )
    ing = ing[ing != ""]
    freq = ing.value_counts(dropna=False)
    freq = freq.rename_axis("ingredient").reset_index(name="count")
    freq["ratio"] = freq["count"] / freq["count"].sum()
    return freq.head(topk)

def nutrient_summary(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    q_list = [0.10, 0.50, 0.90]
    for col in NUTR_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            stats.append({
                "metric": col, "count": 0, "mean": np.nan, "std": np.nan,
                "min": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan, "max": np.nan
            })
            continue
        stats.append({
            "metric": col,
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "p10": float(s.quantile(q_list[0])),
            "p50": float(s.quantile(q_list[1])),
            "p90": float(s.quantile(q_list[2])),
            "max": float(s.max()),
        })
    out = pd.DataFrame(stats)
    # 小数位统一到2位（count除外）
    for c in ["mean","std","min","p10","p50","p90","max"]:
        out[c] = out[c].round(2)
    return out

def plot_topk_bar(freq_df: pd.DataFrame, out_png: str, title: str = "Top-K Ingredients by Frequency"):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(freq_df))
    plt.bar(x, freq_df["count"].values)
    plt.xticks(x, freq_df["ingredient"].values, rotation=75, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_cumcover(freq_all: pd.DataFrame, topk: int, out_png: str, title: str = "Cumulative Coverage of Top-K Ingredients"):
    # 频次全量的累计覆盖率曲线（可选）
    freq = freq_all.copy()
    freq["ratio"] = freq["count"] / freq["count"].sum()
    freq = freq.sort_values("count", ascending=False).reset_index(drop=True)
    freq["cum_ratio"] = freq["ratio"].cumsum()
    k = min(topk, len(freq))
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, k+1), freq.loc[:k-1, "cum_ratio"].values, marker="o")
    plt.xlabel("Top-K")
    plt.ylabel("Cumulative Ratio")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_hist(s: pd.Series, out_png: str, title: str, bins: int = 50):
    # 单图直方图
    plt.figure(figsize=(6, 4))
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        # 画一张空图作为占位
        plt.title(title + " (no data)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return
    # 使用截尾以避免极端尾部分布影响可视化（P99）
    upper = s.quantile(0.99)
    s_clip = s.clip(upper=upper)
    plt.hist(s_clip, bins=bins)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="recipe_nutrient_full.(parquet/csv)")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--topk", type=int, default=50, help="Top-K 食材频次")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df = read_table(args.data)

    # 基础检查
    need_cols = {"ingredients"} | set(NUTR_COLS)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # === 食材频次 ===
    # 全量频次（用于累计覆盖率曲线）
    ing_all = (
        df["ingredients"].fillna("").astype(str).str.split(";").explode().str.strip()
    )
    ing_all = ing_all[ing_all != ""]
    freq_all = ing_all.value_counts().rename_axis("ingredient").reset_index(name="count")
    # Top-K频次
    freq_topk = freq_all.head(args.topk).copy()
    freq_topk["ratio"] = (freq_topk["count"] / freq_all["count"].sum()).round(4)
    freq_topk.to_csv(os.path.join(args.outdir, "ingredients_freq_topK.csv"), index=False)

    # 绘图：Top-K 柱状图 & 覆盖率
    plot_topk_bar(freq_topk, os.path.join(args.outdir, "ingredients_topK_bar.png"))
    plot_cumcover(freq_all, args.topk, os.path.join(args.outdir, "ingredients_cumcover.png"))

    # === 营养统计 ===
    stat_df = nutrient_summary(df)
    stat_df.to_csv(os.path.join(args.outdir, "nutrient_stats.csv"), index=False)

    # 营养直方图
    for col in NUTR_COLS:
        plot_hist(df[col], os.path.join(args.outdir, f"{col}_hist.png"), title=col)

    # 汇总打印
    print("[OK] Quick report generated at:", args.outdir)
    print(" - ingredients_freq_topK.csv")
    print(" - nutrient_stats.csv")
    print(" - ingredients_topK_bar.png, ingredients_cumcover.png")
    print(" - *_hist.png for each nutrient")

if __name__ == "__main__":
    main()
