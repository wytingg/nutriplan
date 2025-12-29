# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os, argparse

def pick_first_exist(df, names):
    for n in names:
        if n in df.columns: return n
    raise KeyError(f"找不到这些列之一: {names}")

def main(a):
    lab = pd.read_parquet(a.labeled)  # 5guard/ingredients_labeled.parquet
    ali = pd.read_parquet(a.aligned)  # 6aligned/aligned_best.parquet

    # 猜测对齐表中的 fdc 列名
    fdc_col = None
    for cand in ["fdc_id", "target_fdc_id", "best_fdc_id", "fdcId", "fdc"]:
        if cand in ali.columns:
            fdc_col = cand; break
    if fdc_col is None:
        raise KeyError("aligned_best 里找不到 fdc_id 列（fdc_id/target_fdc_id/best_fdc_id 任一）。")

    # 选择 join 键
    rid_lab = pick_first_exist(lab, ["RecipeId","recipe_id","recipeid"])
    rid_ali = pick_first_exist(ali, ["RecipeId","recipe_id","recipeid"])

    # 优先使用稳定 id，如果都有行 id/uid/line_idx 就用它；否则退回 (RecipeId + ingredient_norm)
    possible_ids = ["ing_id","line_id","row_id","uid","pair_id"]
    join_id = next((c for c in possible_ids if c in lab.columns and c in ali.columns), None)

    if join_id:
        key_cols_lab = [join_id]
        key_cols_ali = [join_id]
    else:
        # 备选键
        text_key_lab = "ingredient_norm" if "ingredient_norm" in lab.columns else "ingredient_raw"
        text_key_ali = text_key_lab if text_key_lab in ali.columns else ("ingredient_norm" if "ingredient_norm" in ali.columns else None)
        if text_key_ali is None:
            raise KeyError("缺少可用于 join 的文本键（ingredient_norm/ingredient_raw）")
        key_cols_lab = [rid_lab, text_key_lab]
        key_cols_ali = [rid_ali, text_key_ali]

    # 仅保留最佳对齐 + 质量阈值（如果有分数字段就筛一下，没有就全收）
    ali_filt = ali.copy()
    if "is_best" in ali_filt.columns:
        ali_filt = ali_filt[ali_filt["is_best"]==True]
    if "fuzz" in ali_filt.columns:
        ali_filt = ali_filt[ali_filt["fuzz"]>=72]
    if "embed_score" in ali_filt.columns:
        ali_filt = ali_filt[ali_filt["embed_score"]>=0.38]

    keep_cols = key_cols_ali + [fdc_col]
    ali_small = ali_filt[keep_cols].drop_duplicates()

    # 合并
    lab2 = lab.merge(ali_small, left_on=key_cols_lab, right_on=key_cols_ali, how="left")
    lab2.rename(columns={fdc_col:"fdc_id"}, inplace=True)

    # 备份原文件并覆盖
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.out == a.labeled:
        bak = a.labeled.replace(".parquet",".bak.parquet")
        if not os.path.exists(bak):
            os.rename(a.labeled, bak)
            print(f"[bak] {a.labeled} → {bak}")
    lab2.to_parquet(a.out, index=False)
    print(lab2[["fdc_id"]].notna().mean().to_string())
    print(f"[ok] 写出: {a.out}  (总行: {len(lab2)}, fdc_id 非空行: {lab2['fdc_id'].notna().sum()})")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="work/recipebench/data/5guard/ingredients_labeled.parquet")
    ap.add_argument("--aligned", default="work/recipebench/data/6aligned/aligned_best.parquet")
    ap.add_argument("--out",     default="work/recipebench/data/5guard/ingredients_labeled.parquet")
    args = ap.parse_args()
    main(args)
