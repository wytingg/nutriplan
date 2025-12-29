#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0 — 基础统计 (idempotent, enhanced)
产出: 5out/profiles/base_stats.jsonl
字段:
  user_id, n_interactions, pos_ratio, coldstart,
  n_pos, n_neg, n_neutral,
  shrunk_signal, shrinkage_alpha, shrinkage_p0,
  pos_thresh, neg_thresh, neutral_score, coldstart_threshold, dedup_strategy,
  first_ts, last_ts, active_days,
  dup_removed, has_text_ratio

定义:
- interaction: (user_id, recipe_id) 的一次去重后的记录（按 --dedup）。
- pos/neg/neutral:
    rating >= pos_thresh → 正
    rating <= neg_thresh → 负
    rating == neutral_score → 中立
  pos_ratio = n_pos / (n_pos + n_neg)；若分母为0 → NaN
- shrunk_signal: (n_pos + α*p0) / (n_pos + n_neg + α)；默认 α=5, p0=0.6
- coldstart: n_interactions < coldstart_threshold
- dup_removed: 去重前的原始交互数(去掉 AuthorId/RecipeId 缺失) 减去去重后的 n_interactions
- has_text_ratio: 在检测到文本列的情况下，去重后样本中文本非空占比

用法示例:
python work/recipebench/scripts/step_profile/step0_base_stats.py \
  --reviews work/recipebench/data/raw/foodcom/reviews.parquet \
  --out work/recipebench/data/8step_profile/base_stats.jsonl \
  --pos-thresh 4 --neg-thresh 2 --neutral-score 3 \
  --coldstart-threshold 5 --dedup latest \
  --shrinkage-alpha 5 --shrinkage-p0 0.6
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from typing import Optional, Dict, Iterable

import numpy as np
import pandas as pd

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None

REQUIRED_COLS = {"AuthorId", "RecipeId", "Rating"}
DATE_COL_CANDIDATES = [
    "DateModified", "DateSubmitted",
    "date_modified", "date_submitted",
    "Timestamp", "timestamp", "time", "created_at", "updated_at"
]
TEXT_COL_CANDIDATES = [
    "Review", "review", "Text", "text", "Comment", "comment", "Content", "content", "Body", "body"
]

ESSENTIAL_COLS_ALIAS: Dict[str, Iterable[str]] = {
    "AuthorId": ("author_id", "user_id", "userid", "AuthorID", "UserId", "UserID"),
    "RecipeId": ("recipe_id", "RecipeID", "ItemId", "ItemID"),
    "Rating": ("rating", "stars", "score", "Score", "Stars"),
}

def _log(msg: str) -> None:
    print(msg, file=sys.stderr)

def _read_any(path: str, usecols: Optional[list] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if path.endswith(".csv") or path.endswith(".csv.gz"):
        return pd.read_csv(path, sep=",", usecols=usecols)
    if path.endswith(".tsv") or path.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", usecols=usecols)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=usecols)
    raise ValueError(f"Unsupported file extension: {path} (use CSV/TSV(.gz) or Parquet)")

def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    lower_map = {c.lower(): c for c in df.columns}
    for std_col, aliases in ESSENTIAL_COLS_ALIAS.items():
        if std_col in df.columns:
            continue
        hit: Optional[str] = None
        for a in aliases:
            if a in df.columns:
                hit = a
                break
            if a.lower() in lower_map:
                hit = lower_map[a.lower()]
                break
        if hit is not None:
            mapping[hit] = std_col
    if mapping:
        df = df.rename(columns=mapping)
    return df

def _pick_cols(df: pd.DataFrame, candidates: list) -> list:
    got = []
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            got.append(c)
        elif c.lower() in lower_map:
            got.append(lower_map[c.lower()])
    # 去重保序
    seen, out = set(), []
    for c in got:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def _coalesce_event_time(df: pd.DataFrame, date_cols: list) -> Optional[str]:
    if not date_cols:
        return None
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    df["_event_time"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for c in date_cols:
        mask = df["_event_time"].isna() & df[c].notna()
        if mask.any():
            df.loc[mask, "_event_time"] = df.loc[mask, c]
    return "_event_time"

def _ensure_required_cols(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

def _to_int_or_keep(x):
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return x

def _dumps(rec: dict) -> str:
    if orjson is not None:
        return orjson.dumps(rec, option=orjson.OPT_NON_STR_KEYS).decode("utf-8") + "\n"
    return json.dumps(rec, ensure_ascii=False) + "\n"

def _deduplicate(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "none":
        return df
    if how == "latest":
        by = ["AuthorId", "RecipeId"]
        if "_event_time" in df.columns:
            df = df.sort_values(by=by + ["_event_time"], na_position="first")
        else:
            df = df.sort_values(by=by)
        return df.drop_duplicates(subset=by, keep="last")
    if how == "max_rating":
        df["_rating_for_sort"] = df["Rating"].fillna(float("-inf"))
        df = df.sort_values(by=["AuthorId", "RecipeId", "_rating_for_sort"])
        df = df.drop_duplicates(subset=["AuthorId", "RecipeId"], keep="last")
        return df.drop(columns=["_rating_for_sort"])
    raise ValueError(f"Unknown dedup: {how}")

def build_base_stats(
    reviews_path: str,
    out_path: str = "5out/profiles/base_stats.jsonl",
    pos_thresh: float = 4.0,
    neg_thresh: float = 2.0,
    neutral_score: float = 3.0,
    coldstart_threshold: int = 5,
    dedup: str = "latest",  # {latest, max_rating, none}
    shrinkage_alpha: float = 5.0,
    shrinkage_p0: float = 0.6,
    quiet: bool = False,
) -> pd.DataFrame:

    # 轻量读取列名
    try:
        if reviews_path.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
            head = pd.read_csv(reviews_path, nrows=0)
        else:
            head = pd.read_parquet(reviews_path, rows=0) if hasattr(pd, "read_parquet") else pd.read_parquet(reviews_path)
        cols = list(head.columns)
    except Exception:
        cols = None

    # 构造最小 usecols
    minimal_usecols = None
    if cols is not None:
        need = set()
        for std_col, aliases in ESSENTIAL_COLS_ALIAS.items():
            need.add(std_col)
            for a in aliases:
                if a in cols or a.lower() in {c.lower() for c in cols}:
                    need.add(a)
        for c in DATE_COL_CANDIDATES:
            if c in cols or (c.lower() in {x.lower() for x in cols}):
                need.add(c)
        # 文本列（可选）
        for c in TEXT_COL_CANDIDATES:
            if c in cols or (c.lower() in {x.lower() for x in cols}):
                need.add(c)
        minimal_usecols = list(need) if need else None

    df_raw = _read_any(reviews_path, usecols=minimal_usecols)
    df_raw = _normalize_column_names(df_raw)
    _ensure_required_cols(df_raw)

    # 只保留必须列
    keep_cols = ["AuthorId", "RecipeId", "Rating"]
    date_cols = _pick_cols(df_raw, DATE_COL_CANDIDATES)
    text_cols = _pick_cols(df_raw, TEXT_COL_CANDIDATES)
    keep_cols.extend(date_cols + text_cols)
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df_raw.columns]))

    df = df_raw[keep_cols].copy()

    before_rows = len(df)
    df = df.dropna(subset=["AuthorId", "RecipeId"])
    after_rows = len(df)
    if not quiet:
        _log(f"[step0] loaded rows={before_rows:,}, dropna(AuthorId/RecipeId) → {after_rows:,}")

    # 保存“去重前每用户原始交互数”以计算 dup_removed
    raw_counts = df.groupby("AuthorId", observed=True, sort=False)["RecipeId"].size().rename("_raw_inter")

    # 类型标准化
    df["RecipeId"] = df["RecipeId"].map(_to_int_or_keep)
    if df["Rating"].dtype.kind not in ("i", "u", "f"):
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # 合并时间列
    event_col = _coalesce_event_time(df, date_cols)
    if not quiet:
        if event_col is None:
            _log("[step0] no time column; 'latest' will fall back to last occurrence order")
        else:
            _log(f"[step0] coalesced time: {event_col} (non-missing={df[event_col].notna().mean():.2%})")

    # 去重
    if not quiet:
        _log(f"[step0] dedup strategy: {dedup}")
    df = _deduplicate(df, dedup)

    # 评分覆盖
    rated_mask = df["Rating"].notna()
    if not quiet:
        if rated_mask.any():
            rmin = float(df.loc[rated_mask, "Rating"].min())
            rmax = float(df.loc[rated_mask, "Rating"].max())
            _log(f"[step0] rating coverage: {rated_mask.mean():.2%} (min={rmin:.3g}, max={rmax:.3g})")
        else:
            _log("[step0] WARNING: no ratings after dedup; pos/neg/neutral all zero")

    # 文本占比（可选）
    if text_cols:
        # 任一文本列非空即视为“有文本”
        text_nonempty = pd.Series(False, index=df.index)
        for tc in text_cols:
            col = df[tc].astype("string")
            text_nonempty = text_nonempty | (col.notna() & (col.str.len().fillna(0) > 0))
    else:
        text_nonempty = pd.Series(False, index=df.index)

    # 交互拆分
    rating = df["Rating"]
    n_pos_mask = rating >= pos_thresh
    n_neg_mask = rating <= neg_thresh
    n_neu_mask = rating == neutral_score

    grp = df.groupby("AuthorId", sort=False, observed=True)
    stat = grp.agg(
        n_interactions=("RecipeId", "size"),
        n_pos=("Rating", lambda s: (s >= pos_thresh).sum()),
        n_neg=("Rating", lambda s: (s <= neg_thresh).sum()),
        n_neutral=("Rating", lambda s: (s == neutral_score).sum()),
        _has_text=("Rating", lambda s: s.index.size),  # 占位，稍后替换
        first_ts=(event_col, "min") if event_col in df.columns else ("RecipeId", "size"),
        last_ts=(event_col, "max") if event_col in df.columns else ("RecipeId", "size"),
    ).reset_index(names=["user_id"])

    # 修正 has_text_ratio
    if text_cols:
        ht = grp.apply(lambda g: float(text_nonempty.loc[g.index].mean()), include_groups=False).rename("has_text_ratio").reset_index(drop=True)
    else:
        ht = pd.Series([np.nan] * len(stat), name="has_text_ratio")
    stat["_has_text"] = ht

    # 计算 pos_ratio（忽略 neutral）
    denom = (stat["n_pos"] + stat["n_neg"]).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pos_ratio = np.where(denom > 0, stat["n_pos"].to_numpy() / denom, np.nan)
    stat["pos_ratio"] = pos_ratio

    # shrunk_signal
    alpha = float(shrinkage_alpha)
    p0 = float(shrinkage_p0)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat["shrunk_signal"] = np.where(
            denom + alpha > 0,
            (stat["n_pos"].to_numpy() + alpha * p0) / (denom + alpha),
            np.nan,
        )

    # coldstart
    stat["coldstart"] = stat["n_interactions"] < int(coldstart_threshold)

    # 时间跨度（天）
    if event_col in df.columns:
        first = pd.to_datetime(stat["first_ts"], errors="coerce", utc=True)
        last = pd.to_datetime(stat["last_ts"], errors="coerce", utc=True)
        active_days = (last.dt.floor("D") - first.dt.floor("D")).dt.days
        stat["active_days"] = active_days
        # 输出 ISO 日期（仅日期部分，更稳定）
        stat["first_ts"] = first.dt.date.astype("string")
        stat["last_ts"] = last.dt.date.astype("string")
    else:
        stat["first_ts"] = np.nan
        stat["last_ts"] = np.nan
        stat["active_days"] = np.nan

    # dup_removed
    stat = stat.merge(raw_counts.rename("raw_interactions"), left_on="user_id", right_index=True, how="left")
    stat["dup_removed"] = stat["raw_interactions"].fillna(0).astype("int64") - stat["n_interactions"].astype("int64")

    # 参数回填
    stat["pos_thresh"] = float(pos_thresh)
    stat["neg_thresh"] = float(neg_thresh)
    stat["neutral_score"] = float(neutral_score)
    stat["coldstart_threshold"] = int(coldstart_threshold)
    stat["dedup_strategy"] = str(dedup)
    stat["shrinkage_alpha"] = alpha
    stat["shrinkage_p0"] = p0

    # 整理列
    cols = [
        "user_id",
        "n_interactions", "n_pos", "n_neg", "n_neutral",
        "pos_ratio", "shrunk_signal", "coldstart",
        "pos_thresh", "neg_thresh", "neutral_score",
        "coldstart_threshold", "dedup_strategy",
        "first_ts", "last_ts", "active_days",
        "dup_removed", "_has_text"
    ]
    stat = stat[cols].rename(columns={"_has_text": "has_text_ratio"})

    # 稳定排序
    stat = stat.sort_values(by="user_id", kind="mergesort")

    # 写出
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    total = len(stat)
    iterator = stat.itertuples(index=False)
    if (_tqdm is not None) and (not quiet):
        iterator = _tqdm(iterator, total=total, desc="writing base_stats.jsonl")

    with open(tmp, "w", encoding="utf-8") as f:
        for row in iterator:
            rec = {
                "user_id": row.user_id,
                "n_interactions": int(row.n_interactions),
                "n_pos": int(row.n_pos),
                "n_neg": int(row.n_neg),
                "n_neutral": int(row.n_neutral),
                "pos_ratio": (None if pd.isna(row.pos_ratio) else float(row.pos_ratio)),
                "shrunk_signal": (None if pd.isna(row.shrunk_signal) else float(row.shrunk_signal)),
                "coldstart": bool(row.coldstart),
                "pos_thresh": float(row.pos_thresh),
                "neg_thresh": float(row.neg_thresh),
                "neutral_score": float(row.neutral_score),
                "coldstart_threshold": int(row.coldstart_threshold),
                "dedup_strategy": row.dedup_strategy,
                "first_ts": (None if pd.isna(row.first_ts) else str(row.first_ts)),
                "last_ts": (None if pd.isna(row.last_ts) else str(row.last_ts)),
                "active_days": (None if pd.isna(row.active_days) else int(row.active_days)),
                "dup_removed": int(row.dup_removed) if not pd.isna(row.dup_removed) else 0,
                "has_text_ratio": (None if pd.isna(row.has_text_ratio) else float(row.has_text_ratio)),
                "shrinkage_alpha": alpha,
                "shrinkage_p0": p0,
            }
            f.write(_dumps(rec))
    os.replace(tmp, out_path)

    if not quiet:
        n_users = len(stat)
        n_cold = int(stat["coldstart"].sum())
        pr_nonnull = stat["pos_ratio"].notna().mean()
        _log(
            f"[step0] users={n_users:,}  pos_ratio_nonnull={pr_nonnull:.2%}  "
            f"coldstart_thr={coldstart_threshold}  coldstart_users={n_cold:,}"
        )
        _log(f"[step0] → {out_path}")

    return stat

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 0 — 基础统计：仅基于交互/评分（增强版）")
    p.add_argument("--reviews", required=True, help="Path to CSV/TSV(.gz) or Parquet (Food.com style)")
    p.add_argument("--out", default="5out/profiles/base_stats.jsonl", help="Output JSONL path")
    p.add_argument("--pos-thresh", type=float, default=4.0, help="Positive rating threshold (>=)")
    p.add_argument("--neg-thresh", type=float, default=2.0, help="Negative rating threshold (<=)")
    p.add_argument("--neutral-score", type=float, default=3.0, help="Neutral score (==)")
    p.add_argument("--coldstart-threshold", type=int, default=5, help="Cold-start if interactions < threshold")
    p.add_argument("--dedup", choices=["latest", "max_rating", "none"], default="latest",
                   help="How to deduplicate duplicated (AuthorId, RecipeId)")
    p.add_argument("--shrinkage-alpha", type=float, default=5.0, help="Alpha for shrunk_signal")
    p.add_argument("--shrinkage-p0", type=float, default=0.6, help="Prior p0 for shrunk_signal")
    p.add_argument("--quiet", action="store_true", help="Suppress logs")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    df = build_base_stats(
        reviews_path=args.reviews,
        out_path=args.out,
        pos_thresh=args.pos_thresh,
        neg_thresh=args.neg_thresh,
        neutral_score=args.neutral_score,
        coldstart_threshold=args.coldstart_threshold,
        dedup=args.dedup,
        shrinkage_alpha=args.shrinkage_alpha,
        shrinkage_p0=args.shrinkage_p0,
        quiet=args.quiet,
    )
    print(
        f"[step0] users={len(df)}  pos_thresh={args.pos_thresh}  neg_thresh={args.neg_thresh}  "
        f"neutral={args.neutral_score}  coldstart_thr={args.coldstart_threshold}  → {args.out}"
    )
