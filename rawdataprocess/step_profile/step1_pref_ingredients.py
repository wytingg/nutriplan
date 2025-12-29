#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_pref_ingredients_v3.py

Goal
----
Produce per-user ingredient preferences (liked/disliked) from ratings-only signals,
robustly reducing sparsity for users with few polarized ratings while keeping
scientific interpretability for a CCFA-level paper.

Key changes vs. prior versions
------------------------------
1) Ensemble scoring: informative log-odds (Monroe 2008 style) + BM25-style distinctiveness.
   - Head A: log-odds with Dirichlet prior α against a global background p_bg(i), with z-score.
   - Head B: BM25-style weight on positive vs. negative events (binary tf per recipe).
   The two heads are combined and then squashed to [-1,1].
2) Cold-start friendly fallback:
   - If a user has insufficient significant items (|z| < min_z), keep top items by combined
     score subject to weaker threshold (alt_like_thresh), or ensure a minimum K per side.
3) Smarter denoising:
   - Auto-stoplist by recipe document frequency (salt/water/oil/etc.), plus manual stop list.
   - IDF filter and length/time-decay weighting à la “weak supervision but auditable”.
4) Auditability & reproducibility:
   - Outputs include score ([-1,1]), z (from Head A), and (n_pos, n_neg) evidence counts.
   - End-of-run coverage stats (share of users with any liked/disliked) and parameter echo.

Inputs
------
- reviews.parquet: columns [ReviewId, RecipeId, AuthorId, Rating, DateSubmitted, DateModified]
- ingredients_processed.parquet: at least [RecipeId, ingredient_norm], one row per ingredient.

Output
------
- JSONL at --out, one record per user:
  {
    "user_id": int,
    "liked_ingredients": [{"name": str, "score": float, "z": float, "n_pos": int, "n_neg": int}, ...],
    "disliked_ingredients": [...]
  }

Recommended first run (reduces sparsity)
---------------------------------------
python work/recipebench/scripts/step_profile/step1_pref_ingredients.py \
  --reviews work/recipebench/data/raw/foodcom/reviews.parquet \
  --ingredients work/recipebench/data/4out/ingredients_processed.parquet \
  --out work/recipebench/data/8step_profile/pref_ingredients.jsonl \
  --pos-thresh 4 --neg-thresh 3 --dedup latest \
  --len-penalty-alpha 1.0 --time-decay-half_life_days 720 \
  --idf-alpha 1.0 --auto-stop-topk 60 --idf-min 0.15 \
  --prior-alpha 20 --min-user-samples 2 --min-count 1 \
  --min-z 1.65 --alt-like-thresh 0.08 --like-thresh 0.12 \
  --topk 80 --min-k-per-side 3 --ensemble-alpha 0.6 --k1 1.2 --b 0.75

Notes
-----
- Lowering --min-count to 1 and --min-z to 1.65 (≈ one-tailed 5%) markedly reduces “empty users”.
- If you prefer stricter outputs, raise --min-z and/or --like-thresh; coverage will drop.
- All thresholds are surfaced as CLI flags; the algorithmic ensemble itself is stable.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


# -------------------------- CLI --------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True, help="Food.com reviews.parquet")
    ap.add_argument("--ingredients", required=True, help="ingredients_processed.parquet with [RecipeId, ingredient_norm]")
    ap.add_argument("--out", required=True, help="JSONL output path")

    # Labeling
    ap.add_argument("--pos-thresh", type=int, default=4, help="Rating ≥ this → positive event")
    ap.add_argument("--neg-thresh", type=int, default=2, help="Rating ≤ this → negative event")
    ap.add_argument("--dedup", choices=["none", "latest", "earliest", "mean"], default="latest")

    # Weights
    ap.add_argument("--len-penalty-alpha", type=float, default=1.0, help="Recipe length penalty exponent (1/n_ing^alpha)")
    ap.add_argument("--time-decay-half_life_days", type=float, default=0.0, help="Half-life in days; 0 disables time decay")
    ap.add_argument("--idf-alpha", type=float, default=1.0, help="Strength of IDF in event weight (ev_w × idf^alpha)")

    # Stoplists & IDF filters
    ap.add_argument("--auto-stop-topk", type=int, default=60, help="Drop top-K ingredients by recipe df (generic items)")
    ap.add_argument("--idf-min", type=float, default=0.25, help="Keep ingredients with IDF ≥ this threshold")
    ap.add_argument(
        "--stopwords",
        type=str,
        default=(
            "salt,water,pepper,oil,sugar,flour,butter,egg,eggs,all-purpose flour,"
            "olive oil,black pepper,water,all purpose flour,all-purpose-flour"
        ),
        help="manual stopwords (comma-separated)",
    )

    # Head A: informative log-odds prior
    ap.add_argument("--prior-alpha", type=float, default=20.0, help="Dirichlet prior strength α for background p_bg(i)")

    # Head B: BM25
    ap.add_argument("--k1", type=float, default=1.2, help="BM25 k1")
    ap.add_argument("--b", type=float, default=0.75, help="BM25 b")

    # Selection thresholds
    ap.add_argument("--min-user-samples", type=int, default=3, help="min number of positive OR negative events to analyze user")
    ap.add_argument("--min-count", type=int, default=2, help="min (n_pos+n_neg) events for an ingredient for that user")
    ap.add_argument("--min-z", type=float, default=2.0, help="|z| ≥ this to pass strict significance gate")
    ap.add_argument("--alt-like-thresh", type=float, default=0.1, help="weaker |score| gate when z is small (fallback)")
    ap.add_argument("--like-thresh", type=float, default=0.15, help="final |score| ≥ this to keep")
    ap.add_argument("--topk", type=int, default=80, help="max items per side (liked/disliked)")
    ap.add_argument("--min-k-per-side", type=int, default=0, help="ensure at least K per side via fallback if possible")

    # Ensemble
    ap.add_argument("--ensemble-alpha", type=float, default=0.6, help="weight for log-odds head in combined score")

    # Misc
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--debug", action="store_true", help="print per-user debug info")

    return ap.parse_args()


# -------------------------- IO & Preprocess --------------------------

def _safe_datetime(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_reviews(path, dedup="latest"):
    usecols = ["ReviewId", "RecipeId", "AuthorId", "Rating", "DateSubmitted", "DateModified"]
    df = pd.read_parquet(path, columns=[c for c in usecols if c is not None], engine="pyarrow")
    df["AuthorId"] = pd.to_numeric(df["AuthorId"], errors="coerce").astype("Int64")
    df["RecipeId"] = pd.to_numeric(df["RecipeId"], errors="coerce").astype("Int64")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").astype("float32")

    tcol = "DateModified" if "DateModified" in df.columns else ("DateSubmitted" if "DateSubmitted" in df.columns else None)
    if tcol:
        df[tcol] = _safe_datetime(df[tcol])

    # Dedup on (AuthorId, RecipeId)
    if dedup in ("latest", "earliest") and tcol:
        asc = (dedup == "earliest")
        df = df.sort_values(tcol, ascending=asc).drop_duplicates(["AuthorId", "RecipeId"], keep="last")
    elif dedup == "mean":
        df = df.groupby(["AuthorId", "RecipeId"], as_index=False)["Rating"].mean()
    else:
        df = df.drop_duplicates(["AuthorId", "RecipeId"], keep="last")

    df = df.dropna(subset=["AuthorId", "RecipeId", "Rating"]).reset_index(drop=True)

    # Early label & filter to reduce join/explode
    pos_mask = df["Rating"].values >= pos_thresh_global
    neg_mask = df["Rating"].values <= neg_thresh_global
    df = df.loc[pos_mask | neg_mask].copy()
    df["label"] = 0
    df.loc[pos_mask, "label"] = 1
    df.loc[neg_mask, "label"] = -1
    return df


def load_recipe_ings(path, stopwords_csv, auto_stop_topk, idf_min):
    df = pd.read_parquet(path, engine="pyarrow")
    cmap = {c.lower(): c for c in df.columns}
    rid = cmap.get("recipeid") or cmap.get("recipe_id")
    ing = cmap.get("ingredient_norm") or cmap.get("ingredient")
    if not rid or not ing:
        raise ValueError("ingredients_processed.parquet must contain RecipeId and ingredient_norm")

    df = df[[rid, ing]].rename(columns={rid: "RecipeId", ing: "ingredient"})
    df["RecipeId"] = pd.to_numeric(df["RecipeId"], errors="coerce").astype("Int64")
    df["ingredient"] = df["ingredient"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["RecipeId", "ingredient"]).drop_duplicates(["RecipeId", "ingredient"])  # de-dup within recipe

    # Document frequency & IDF
    N = df["RecipeId"].nunique()
    dfreq = df.groupby("ingredient")["RecipeId"].nunique().astype("int32")
    idf = np.log1p(N / (dfreq + 1.0)).astype("float32")  # log(1 + N/(df+1))
    idf.name = "idf"

    # BM25 idf
    idf_bm25 = np.log(((N - dfreq + 0.5) / (dfreq + 0.5)).clip(lower=1e-9)).astype("float32")
    idf_bm25.name = "idf_bm25"

    # Auto-stoplist + manual stopwords
    auto_stop = set(dfreq.sort_values(ascending=False).head(auto_stop_topk).index.tolist())
    manual_stop = {s.strip().lower() for s in (stopwords_csv or "").split(",") if s.strip()}
    stop_all = auto_stop | manual_stop

    keep = (~df["ingredient"].isin(stop_all)) & (idf.loc[df["ingredient"].values].values >= idf_min)
    df = df.loc[keep].copy()

    # recompute idf on kept items
    N2 = df["RecipeId"].nunique()
    dfreq2 = df.groupby("ingredient")["RecipeId"].nunique().astype("int32")
    idf2 = np.log1p(N2 / (dfreq2 + 1.0)).astype("float32"); idf2.name = "idf"
    idf_bm25_2 = np.log(((N2 - dfreq2 + 0.5) / (dfreq2 + 0.5)).clip(lower=1e-9)).astype("float32"); idf_bm25_2.name = "idf_bm25"

    # per-recipe ingredient list & length for weights
    g = df.groupby("RecipeId")["ingredient"].agg(list).to_frame("ing_list")
    g["n_ing"] = g["ing_list"].apply(len).astype("int16")
    avg_n_ing = float(g["n_ing"].mean()) if len(g) else 1.0

    return df, idf2, idf_bm25_2, g, avg_n_ing


# -------------------------- Core computations --------------------------

def expand_events(rev_labeled: pd.DataFrame, recipe_group: pd.DataFrame,
                  len_penalty_alpha: float, time_decay_half_life_days: float) -> pd.DataFrame:
    # Merge recipe ingredient lists & lengths
    df = rev_labeled.merge(recipe_group, left_on="RecipeId", right_index=True, how="inner")
    if df.empty:
        return df

    df = df.explode("ing_list").rename(columns={"ing_list": "ingredient"})
    df = df.loc[df["n_ing"] > 0].copy()

    # Length penalty
    if len_penalty_alpha > 0:
        df["len_w"] = (1.0 / (df["n_ing"].astype("float32") ** float(len_penalty_alpha))).astype("float32")
    else:
        df["len_w"] = 1.0

    # Time decay
    tcol = "DateModified" if "DateModified" in rev_labeled.columns else ("DateSubmitted" if "DateSubmitted" in rev_labeled.columns else None)
    if time_decay_half_life_days and tcol and tcol in df.columns:
        tmax = pd.to_datetime(rev_labeled[tcol], utc=True, errors="coerce").max()
        t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        delta_days = (tmax - t).dt.total_seconds() / (3600 * 24)
        lam = math.log(2.0) / float(time_decay_half_life_days)
        df["time_w"] = np.exp(-lam * delta_days.fillna(0).astype("float32"))
    else:
        df["time_w"] = 1.0

    return df[["AuthorId", "ingredient", "label", "n_ing", "len_w", "time_w"]]


def compute_background_probs(ev: pd.DataFrame, idf: pd.Series, idf_alpha: float) -> pd.Series:
    # Ensure idf column exists and is named 'idf'
    if "idf" not in ev.columns:
        idf_df = idf.to_frame()
        merged = ev.merge(idf_df, left_on="ingredient", right_index=True, how="left")
        # Normalize merged column name to 'idf'
        col_name = idf_df.columns[0] if hasattr(idf_df, "columns") and len(idf_df.columns) else "idf"
        if col_name != "idf" and col_name in merged.columns:
            merged = merged.rename(columns={col_name: "idf"})
        ev = merged
    if "idf" not in ev.columns:
        ev["idf"] = 1.0
    ev["idf"] = ev["idf"].fillna(1.0).astype("float32")

    w = (ev["len_w"] * ev["time_w"] * (ev["idf"] ** float(idf_alpha))).astype("float64")
    ev["w"] = w
    bg_wsum = ev.groupby("ingredient")["w"].sum().astype("float64")
    total_w = float(bg_wsum.sum())
    if total_w <= 0:
        p = (bg_wsum * 0 + 1.0) / max(len(bg_wsum), 1)
    else:
        p = (bg_wsum / total_w).clip(1e-12, 1.0)
    p.name = "p_bg"
    return p


def headA_logodds(u_df: pd.DataFrame, p_bg: pd.Series, alpha: float,
                  idf: pd.Series, idf_alpha: float) -> Tuple[pd.DataFrame, float, float]:
    # Ensure idf present for event weights
    if "idf" not in u_df.columns:
        u_df = u_df.merge(idf.to_frame(), left_on="ingredient", right_index=True, how="left")
    u_df["idf"] = u_df["idf"].fillna(1.0).astype("float32")
    w = (u_df["len_w"] * u_df["time_w"] * (u_df["idf"] ** float(idf_alpha))).astype("float64")

    u_df["pos_w"] = (u_df["label"] == 1).astype("float64") * w
    u_df["neg_w"] = (u_df["label"] == -1).astype("float64") * w

    kpos = u_df.groupby("ingredient")["pos_w"].sum()
    kneg = u_df.groupby("ingredient")["neg_w"].sum()
    npos = float(u_df["pos_w"].sum())
    nneg = float(u_df["neg_w"].sum())

    total_w = float(u_df["pos_w"].sum() + u_df["neg_w"].sum())
    npos_eff = npos if npos > 0 else total_w
    nneg_eff = nneg if nneg > 0 else total_w

    idx = kpos.index.union(kneg.index)
    pi = p_bg.reindex(idx, fill_value=float(p_bg.mean()))
    kpos = kpos.reindex(idx, fill_value=0.0)
    kneg = kneg.reindex(idx, fill_value=0.0)

    eps = 1e-9
    def logit(x): return np.log((x + eps) / (1.0 - x + eps))
    p_pos = (kpos + alpha * pi) / (npos_eff + alpha)
    p_neg = (kneg + alpha * pi) / (nneg_eff + alpha)
    delta = logit(p_pos) - logit(p_neg)
    var = (1.0 / (kpos + alpha * pi + eps)) + (1.0 / (npos_eff - kpos + alpha * (1 - pi) + eps)) + \
          (1.0 / (kneg + alpha * pi + eps)) + (1.0 / (nneg_eff - kneg + alpha * (1 - pi) + eps))
    z = delta / np.sqrt(var)

    s = 2.0
    scoreA = np.tanh((delta / s).to_numpy(dtype="float64"))
    out = pd.DataFrame({
        "ingredient": np.array(idx, dtype=object),
        "scoreA": scoreA.astype("float32"),
        "z": z.to_numpy(dtype="float32"),
        "n_pos": kpos.to_numpy(dtype="float32"),
        "n_neg": kneg.to_numpy(dtype="float32"),
    })
    out.index = pd.RangeIndex(len(out))  # ensure plain RangeIndex
    return out, npos, nneg


def headB_bm25(u_df: pd.DataFrame, idf_bm25: pd.Series, k1: float, b: float, avg_n_ing: float) -> pd.DataFrame:
    u = u_df.copy()
    if "idf_bm25" not in u.columns:
        u = u.merge(idf_bm25.to_frame(), left_on="ingredient", right_index=True, how="left")
    u["idf_bm25"] = u["idf_bm25"].fillna(0.0).astype("float32")

    denom = 1.0 + k1 * (1.0 - b + b * (u["n_ing"].astype("float32") / float(avg_n_ing)))
    bm25_ev = u["idf_bm25"] * ((k1 + 1.0) / denom)
    bm25_ev = bm25_ev * u["len_w"].astype("float32") * u["time_w"].astype("float32")

    u["bm25_pos"] = (u["label"] == 1).astype("float32") * bm25_ev
    u["bm25_neg"] = (u["label"] == -1).astype("float32") * bm25_ev

    pos = u.groupby("ingredient")["bm25_pos"].sum()
    neg = u.groupby("ingredient")["bm25_neg"].sum()
    diff = pos - neg

    nz = diff[diff != 0]
    scale = float(np.median(np.abs(nz))) if len(nz) else 1.0
    scale = max(scale, 1e-6)
    scoreB = np.tanh((diff / scale).to_numpy(dtype="float64")).astype("float32")

    out = pd.DataFrame({
        "ingredient": diff.index.to_numpy(dtype=object),
        "scoreB": scoreB,
    })
    out.index = pd.RangeIndex(len(out))  # ensure no named index
    return out


def combine_heads(dfA: pd.DataFrame, dfB: pd.DataFrame, ensemble_alpha: float) -> pd.DataFrame:
    dfA = dfA.reset_index(drop=True)
    dfB = dfB.reset_index(drop=True)
    if dfA.index.name is not None: dfA.index.name = None
    if dfB.index.name is not None: dfB.index.name = None
    df = dfA.merge(dfB, on="ingredient", how="outer").fillna({"scoreA": 0.0, "scoreB": 0.0})
    wA = float(ensemble_alpha); wB = 1.0 - wA
    df["score"] = (wA * df["scoreA"].astype("float32") + wB * df["scoreB"].astype("float32")).clip(-1.0, 1.0)
    return df


# -------------------------- Fast path helpers --------------------------

def preaggregate_events_fast(ev: pd.DataFrame,
                             idf_alpha: float,
                             idf_bm25: pd.Series,
                             k1: float, b: float,
                             avg_n_ing: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 事件级权重（一次算好）
    ev = ev.copy()
    ev["ev_w"] = (ev["len_w"] * ev["time_w"] * (ev["idf"] ** float(idf_alpha))).astype("float32")

    denom = 1.0 + k1 * (1.0 - b + b * (ev["n_ing"].astype("float32") / float(avg_n_ing)))
    bm25_ev = ev["idf_bm25"] * ((k1 + 1.0) / denom)
    ev["bm25_ev"] = (bm25_ev * ev["len_w"].astype("float32") * ev["time_w"].astype("float32")).astype("float32")

    # 向量化标记
    pos = (ev["label"] == 1).astype("float32")
    neg = (ev["label"] == -1).astype("float32")

    ev["pos_w"] = pos * ev["ev_w"]
    ev["neg_w"] = neg * ev["ev_w"]
    ev["bm25_pos"] = pos * ev["bm25_ev"]
    ev["bm25_neg"] = neg * ev["bm25_ev"]

    # 关键一步：一次 groupby 聚合到 (AuthorId, ingredient)
    ev["ingredient"] = ev["ingredient"].astype("category")
    agg = (ev.groupby(["AuthorId", "ingredient"], observed=True, sort=False)
             .agg(pos_w=("pos_w","sum"),
                  neg_w=("neg_w","sum"),
                  bm25_pos=("bm25_pos","sum"),
                  bm25_neg=("bm25_neg","sum"))
             .reset_index())

    # 每个用户的总权重（Head A 需要）
    user_tot = (agg.groupby("AuthorId", sort=False)[["pos_w","neg_w"]]
                  .sum()
                  .rename(columns={"pos_w":"npos","neg_w":"nneg"})
                  .reset_index())

    return agg, user_tot


def compute_user_scores_from_agg(u_slice: pd.DataFrame,
                                 p_bg: pd.Series,
                                 prior_alpha: float,
                                 ensemble_alpha: float) -> pd.DataFrame:
    # u_slice: columns [ingredient, pos_w, neg_w, bm25_pos, bm25_neg]
    idx = u_slice["ingredient"].astype("object").to_numpy()
    kpos = u_slice["pos_w"].to_numpy(dtype="float64")
    kneg = u_slice["neg_w"].to_numpy(dtype="float64")
    # 需在外面提供 npos/nneg (此处直接由切片求和)
    npos = float(kpos.sum()); nneg = float(kneg.sum())
    total_w = npos + nneg
    npos_eff = npos if npos > 0 else total_w
    nneg_eff = nneg if nneg > 0 else total_w

    pi = p_bg.reindex(idx, fill_value=float(p_bg.mean())).to_numpy(dtype="float64")
    eps = 1e-9
    p_pos = (kpos + prior_alpha * pi) / (npos_eff + prior_alpha)
    p_neg = (kneg + prior_alpha * pi) / (nneg_eff + prior_alpha)

    delta = np.log((p_pos + eps) / (1.0 - p_pos + eps)) - np.log((p_neg + eps) / (1.0 - p_neg + eps))
    var = (1.0 / (kpos + prior_alpha * pi + eps)) + (1.0 / (npos_eff - kpos + prior_alpha * (1 - pi) + eps)) + \
          (1.0 / (kneg + prior_alpha * pi + eps)) + (1.0 / (nneg_eff - kneg + prior_alpha * (1 - pi) + eps))
    z = delta / np.sqrt(var)
    scoreA = np.tanh(delta / 2.0).astype("float32")

    # Head B
    diff = (u_slice["bm25_pos"].to_numpy(dtype="float64") -
            u_slice["bm25_neg"].to_numpy(dtype="float64"))
    nz = diff[diff != 0]
    scale = float(np.median(np.abs(nz))) if nz.size else 1.0
    scale = max(scale, 1e-6)
    scoreB = np.tanh((diff / scale)).astype("float32")

    out = pd.DataFrame({
        "ingredient": idx,
        "scoreA": scoreA,
        "z": z.astype("float32"),
        "n_pos": kpos.astype("float32"),
        "n_neg": kneg.astype("float32"),
        "scoreB": scoreB.astype("float32"),
    })
    wA = float(ensemble_alpha); wB = 1.0 - wA
    out["score"] = (wA * out["scoreA"] + wB * out["scoreB"]).clip(-1.0, 1.0)
    return out


# -------------------------- Main --------------------------

def run(args):
    np.random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load recipe ingredients + stats first (for idf/stop and avg_n_ing)
    recipe_df, idf, idf_bm25, recipe_group, avg_n_ing = load_recipe_ings(
        args.ingredients, args.stopwords, args.auto_stop_topk, args.idf_min
    )

    # Load and label reviews (filtered to pos/neg only)
    rev = load_reviews(args.reviews, dedup=args.dedup)
    if rev.empty:
        raise SystemExit("No reviews after dedup/labeling. Check thresholds or file columns.")

    # Expand to events
    ev = expand_events(
        rev, recipe_group,
        len_penalty_alpha=args.len_penalty_alpha,
        time_decay_half_life_days=args.time_decay_half_life_days,
    )
    if ev.empty:
        raise SystemExit("No positive/negative events after merging recipes. Check inputs and thresholds.")

    # Pre-attach idf/bm25 to events for efficiency in per-user loops
    ev = ev.merge(idf.to_frame(), left_on="ingredient", right_index=True, how="left")
    ev = ev.merge(idf_bm25.to_frame(), left_on="ingredient", right_index=True, how="left")
    ev["idf"] = ev["idf"].fillna(1.0).astype("float32")
    ev["idf_bm25"] = ev["idf_bm25"].fillna(0.0).astype("float32")

    # Background for Head A
    cols_bg = ["AuthorId", "ingredient", "len_w", "time_w"]
    if "idf" in ev.columns:
        cols_bg.append("idf")
    p_bg = compute_background_probs(ev[cols_bg].copy(), idf, args.idf_alpha)

    # Fast path: pre-aggregate once, then iterate users
    agg, user_tot = preaggregate_events_fast(
        ev, args.idf_alpha, idf_bm25, args.k1, args.b, avg_n_ing
    )
    tot_map = dict(zip(user_tot["AuthorId"].to_numpy(), user_tot[["npos","nneg"]].to_numpy()))

    users = agg["AuthorId"].unique().tolist()
    iterator = users if _tqdm is None else _tqdm(users, total=len(users), desc="[step1] users", unit="user")

    users_total = len(users)
    users_with_any = 0
    users_with_liked = 0
    users_with_disliked = 0

    with out_path.open("w", encoding="utf-8") as f:
        for uid in iterator:
            u_slice = agg.loc[agg["AuthorId"] == uid, ["ingredient","pos_w","neg_w","bm25_pos","bm25_neg"]]
            if u_slice.empty:
                liked_list, disliked_list = [], []
            else:
                comb = compute_user_scores_from_agg(u_slice, p_bg, args.prior_alpha, args.ensemble_alpha)
                comb["tot"] = comb["n_pos"].fillna(0) + comb["n_neg"].fillna(0)

                strict = (comb["tot"] >= args.min_count) & (np.abs(comb["z"]).values >= args.min_z) & (np.abs(comb["score"]).values >= args.like_thresh)
                soft   = (comb["tot"] >= args.min_count) & (np.abs(comb["score"]).values >= args.alt_like_thresh)

                cand = comb.loc[strict | soft].copy()
                if cand.empty:
                    liked_list, disliked_list = [], []
                else:
                    liked = cand.loc[cand["score"] > 0].sort_values(["z","score","tot"], ascending=[False, False, False]).head(args.topk)
                    disliked = cand.loc[cand["score"] < 0].sort_values(["z","score","tot"], ascending=[True, True, False]).head(args.topk)
                    liked_list = [{"name": r.ingredient, "score": float(r.score), "z": float(r.z),
                                   "n_pos": int(round(r.n_pos)), "n_neg": int(round(r.n_neg))}
                                  for r in liked.itertuples(index=False)]
                    disliked_list = [{"name": r.ingredient, "score": float(r.score), "z": float(r.z),
                                      "n_pos": int(round(r.n_pos)), "n_neg": int(round(r.n_neg))}
                                     for r in disliked.itertuples(index=False)]

                # Coldstart fallback: ensure minimal K per side if available
                if args.min_k_per_side > 0:
                    alt = comb.copy()
                    alt_like = alt.loc[alt["score"] > 0].sort_values(["score", "tot"], ascending=[False, False])
                    alt_dis = alt.loc[alt["score"] < 0].sort_values(["score", "tot"], ascending=[True, False])
                    seen_like = {x["name"] for x in liked_list}
                    for r in alt_like.itertuples(index=False):
                        if len(liked_list) >= args.min_k_per_side: break
                        if r.ingredient in seen_like: continue
                        liked_list.append({
                            "name": r.ingredient, "score": float(r.score), "z": float(0.0 if pd.isna(r.z) else r.z),
                            "n_pos": int(round(0 if pd.isna(r.n_pos) else r.n_pos)),
                            "n_neg": int(round(0 if pd.isna(r.n_neg) else r.n_neg)),
                        })
                        seen_like.add(r.ingredient)
                    seen_dis = {x["name"] for x in disliked_list}
                    for r in alt_dis.itertuples(index=False):
                        if len(disliked_list) >= args.min_k_per_side: break
                        if r.ingredient in seen_dis: continue
                        disliked_list.append({
                            "name": r.ingredient, "score": float(r.score), "z": float(0.0 if pd.isna(r.z) else r.z),
                            "n_pos": int(round(0 if pd.isna(r.n_pos) else r.n_pos)),
                            "n_neg": int(round(0 if pd.isna(r.n_neg) else r.n_neg)),
                        })
                        seen_dis.add(r.ingredient)

            # Update coverage counters
            has_liked = len(liked_list) > 0
            has_disliked = len(disliked_list) > 0
            if has_liked or has_disliked:
                users_with_any += 1
            if has_liked:
                users_with_liked += 1
            if has_disliked:
                users_with_disliked += 1

            if args.debug:
                npos_dbg, nneg_dbg = tot_map.get(uid, [0.0, 0.0])
                print(f"[u={uid}] pos={float(npos_dbg):.2f} neg={float(nneg_dbg):.2f} liked={len(liked_list)} disliked={len(disliked_list)}")

            rec = {"user_id": int(uid), "liked_ingredients": liked_list, "disliked_ingredients": disliked_list}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    p_any = 100.0 * users_with_any / max(users_total, 1)
    p_like = 100.0 * users_with_liked / max(users_total, 1)
    p_dis = 100.0 * users_with_disliked / max(users_total, 1)
    print(
        f"[done] → {out_path}\n"
        f"Users: {users_total} | any={users_with_any} ({p_any:.1f}%) | "
        f"liked={users_with_liked} ({p_like:.1f}%) | disliked={users_with_disliked} ({p_dis:.1f}%)"
    )
    print(
        f"(params) pos≥{pos_thresh_global} neg≤{neg_thresh_global}  prior_alpha={args.prior_alpha}  "
        f"min_count={args.min_count}  min_z={args.min_z}  like_thresh={args.like_thresh}  alt_like_thresh={args.alt_like_thresh}  "
        f"auto_stop_topk={args.auto_stop_topk}  idf_min={args.idf_min}  ensemble_alpha={args.ensemble_alpha}  "
        f"k1={args.k1} b={args.b}"
    )


# -------------------------- Entry --------------------------
if __name__ == "__main__":
    args = parse_args()

    # expose thresholds to loader (so we can early-filter neutrals before join)
    # NOTE: this is a small hack to avoid refactoring the signature; keep module-level read only
    global pos_thresh_global, neg_thresh_global
    pos_thresh_global = args.pos_thresh
    neg_thresh_global = args.neg_thresh

    run(args)
