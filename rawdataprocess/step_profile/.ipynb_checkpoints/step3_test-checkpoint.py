#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — Nutrition Targets from Policy-Grounded Tags
- 为 reviews 数据中的每个用户生成：标签(tag) + 营养目标(targets) + 区间(ranges) + 审计(audit)
- 标签 = 宏量配比(AMDR) × 钠(sodium cap) × 糖(sugar cap) × 饱和脂肪(SFA cap)
- 纤维最低值：IOM 规则 14 g / 1000 kcal
- 能量：按桶(low/mid/high)或混合(mix)采样（不引入敏感人群信息）
python work/recipebench/scripts/step_profile/step3_test.py \
  --reviews work/recipebench/data/raw/foodcom/reviews.parquet \
  --out work/recipebench/data/8step_profile/nutrition_targets**.jsonl \
  --energy-bucket mix \
  --persona-dist uniform \
  --sugar-mode tag \
  --verbose-audit

"""
import argparse, json, math, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# 组件（Policy Components）
# -----------------------------
AMDR_PRESETS = {
    # 皆在 IOM/DRI AMDR 成人范围内：C 45–65, P 10–35, F 20–35
    "balanced":        {"carb": (45, 65), "protein": (10, 35), "fat": (20, 35)},
    "higher_protein":  {"carb": (45, 55), "protein": (20, 30), "fat": (20, 30)},
    "lower_fat":       {"carb": (50, 60), "protein": (15, 25), "fat": (20, 25)},
    "higher_carb_endurance": {"carb": (55, 65), "protein": (10, 20), "fat": (20, 30)},
    "med_style":       {"carb": (45, 60), "protein": (15, 25), "fat": (25, 35)},
    "plant_forward":   {"carb": (50, 60), "protein": (15, 25), "fat": (20, 30)},
}

SODIUM_CAPS = {
    # 分层“严格度”：DGA(2300) → WHO(2000) → AHA(1500)
    "dga_2300": 2300.0,
    "who_2000": 2000.0,
    "aha_1500": 1500.0,
}

SUGAR_CAPS = {
    # DGA: added <10% ener; WHO: free <10% (preferably <5%)
    "added_10": {"type": "added", "pct_max": 10.0},
    "free_10":  {"type": "free",  "pct_max": 10.0},
    "free_5":   {"type": "free",  "pct_max": 5.0},
}

SFA_CAPS = {
    "sfa_10": 10.0,   # DGA
    "sfa_6":  6.0,    # AHA stricter
}

SOURCES = [
    # 论文里可引用；此处存到 audit.sources 便于审稿人核查
    "IOM/DRI AMDR (2002/2005)",
    "DGA 2020–2025 (Added sugar<10%, SFA<10%, Sodium≤2300 mg)",
    "WHO Sodium (≤2000 mg/day) & Free sugars (<10%, preferably <5%)",
    "AHA SFA <6% ideal target",
    "IOM Fiber: 14 g per 1000 kcal",
]

# -----------------------------
# 工具函数
# -----------------------------
def build_all_tags():
    tags = []
    for macro in AMDR_PRESETS.keys():
        for sod in SODIUM_CAPS.keys():
            for sug in SUGAR_CAPS.keys():
                for sfa in SFA_CAPS.keys():
                    tags.append(f"{macro}+{sod}+{sug}+{sfa}")
    return tags

def parse_pairs_to_weights(spec: str, choices: list[str]) -> np.ndarray:
    """
    解析 "name:prob,..." 到与 choices 对齐的权重向量；支持 uniform
    """
    if spec.strip() == "uniform":
        w = np.ones(len(choices), dtype=float)
        return w / w.sum()
    pairs = {}
    for seg in spec.split(","):
        if not seg.strip():
            continue
        name, prob = seg.split(":")
        pairs[name.strip()] = float(prob)
    w = np.array([pairs.get(c, 0.0) for c in choices], dtype=float)
    s = w.sum()
    if s <= 0:
        raise ValueError("persona distribution weights sum to 0")
    return w / s

def choose_energy_bucket(rng: np.random.Generator, bucket: str) -> float:
    # 成人能量参考桶（不引入个体敏感属性）
    if bucket == "low":
        lo, hi = 1600, 1800
    elif bucket == "mid":
        lo, hi = 1800, 2200
    elif bucket == "high":
        lo, hi = 2200, 2600
    elif bucket == "mix":
        # 三桶等概率选一个，再在桶内均匀采样
        bucket = rng.choice(["low","mid","high"])
        return choose_energy_bucket(rng, bucket)
    else:
        raise ValueError("energy-bucket must be low|mid|high|mix")
    return float(rng.uniform(lo, hi))

def sample_amdr(rng: np.random.Generator, ranges: dict) -> dict:
    c = rng.uniform(*ranges["carb"])
    p = rng.uniform(*ranges["protein"])
    f = rng.uniform(*ranges["fat"])
    # 归一化到 100%
    s = c + p + f
    c, p, f = 100*c/s, 100*p/s, 100*f/s
    return {"carb_pct": round(c, 2), "protein_pct": round(p, 2), "fat_pct": round(f, 2)}

def get_user_ids(path: Path, user_col_hint: str | None):
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:  # jsonl
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except Exception:
                    pass
        df = pd.DataFrame(rows)

    # 自动识别列名
    cols = [c.lower() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    cand = None
    if user_col_hint and user_col_hint in df.columns:
        cand = user_col_hint
    elif "user_id" in cols:
        cand = col_map["user_id"]
    elif "authorid" in cols:
        cand = col_map["authorid"]
    elif "author_id" in cols:
        cand = col_map["author_id"]
    else:
        raise ValueError(f"未找到 user_id 列；候选列={df.columns.tolist()[:10]}")
    s = df[cand].dropna().drop_duplicates()
    return s.tolist()

def uid_to_seed(uid) -> int:
    """
    将任意 uid 转为稳定整数种子（兼容字符串/大整型）
    """
    try:
        return int(uid) & 0xFFFFFFFF
    except Exception:
        h = hashlib.blake2b(str(uid).encode("utf-8"), digest_size=8).hexdigest()
        return int(h, 16) & 0xFFFFFFFF

def format_ranges(macro_key, sugar_key, sfa_key):
    r = {}
    macro = AMDR_PRESETS[macro_key]
    r["amdr_ranges_pct"] = {
        "carb": {"min": macro["carb"][0], "max": macro["carb"][1]},
        "protein": {"min": macro["protein"][0], "max": macro["protein"][1]},
        "fat": {"min": macro["fat"][0], "max": macro["fat"][1]},
    }
    sug = SUGAR_CAPS[sugar_key]
    r["sugars_cap"] = {"type": sug["type"], "pct_max": sug["pct_max"]}
    r["sfa_cap_pct"] = {"pct_max": SFA_CAPS[sfa_key]}
    return r

# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True, help="含 user_id/AuthorId 的 Parquet/CSV/JSONL 文件路径")
    ap.add_argument("--out", required=True, help="输出 JSONL")
    ap.add_argument("--user-col", default=None, help="可指定用户列名；默认自动识别 user_id/AuthorId/author_id")
    ap.add_argument("--persona-dist", default="uniform",
                    help='标签分布：uniform 或 "macro+sodium+sugar+sfa:prob,..."')
    ap.add_argument("--energy-bucket", default="mix", choices=["low","mid","high","mix"],
                    help="能量桶；mix=三桶随机")
    ap.add_argument("--sugar-mode", default="tag", choices=["tag", "added", "free"],
                    help="统一口径（添加糖/游离糖）；tag=跟随标签")
    ap.add_argument("--verbose-audit", action="store_true", help="输出更详细的 audit 信息")
    args = ap.parse_args()

    # 1) 标签库
    ALL_TAGS = build_all_tags()
    weights = parse_pairs_to_weights(args.persona_dist, ALL_TAGS)

    # 2) 读取用户
    user_ids = get_user_ids(Path(args.reviews), args.user_col)
    n_users = len(user_ids)

    # 3) 生成
    out_f = open(args.out, "w", encoding="utf-8")
    for uid in user_ids:
        seed = uid_to_seed(uid)
        rng = np.random.default_rng(seed)
        rnd = random.Random(seed)

        # 3.1 选标签（按给定权重）
        tag = rnd.choices(ALL_TAGS, weights=weights, k=1)[0]
        macro_key, sod_key, sug_key, sfa_key = tag.split("+")

        # 3.2 采样能量 & 纤维
        energy = choose_energy_bucket(rng, args.energy_bucket)
        fiber_min = round(14.0 * energy / 1000.0, 1)  # IOM

        # 3.3 采样宏配比（并归一化到100%）
        amdr_target = sample_amdr(rng, AMDR_PRESETS[macro_key])

        # 3.4 上限/口径
        sodium_max = float(SODIUM_CAPS[sod_key])
        sugar_policy = SUGAR_CAPS[sug_key]
        sugar_type = sugar_policy["type"] if args.sugar_mode == "tag" else args.sugar_mode
        sugar_pct_max = sugar_policy["pct_max"]
        sfa_pct_max = float(SFA_CAPS[sfa_key])

        # 3.5 组合记录（既有 target 也有 ranges）
        amdr_ranges = format_ranges(macro_key, sug_key, sfa_key)["amdr_ranges_pct"]
        record = {
            "user_id": uid,
            "tag": tag,
            "energy_kcal_target": round(energy, 0),

            # 宏配比：目标值 + 区间（min/max）
            "amdr": {
                "carb":   {"target_pct": amdr_target["carb_pct"],   "min_pct": amdr_ranges["carb"]["min"],   "max_pct": amdr_ranges["carb"]["max"]},
                "protein":{"target_pct": amdr_target["protein_pct"],"min_pct": amdr_ranges["protein"]["min"],"max_pct": amdr_ranges["protein"]["max"]},
                "fat":    {"target_pct": amdr_target["fat_pct"],    "min_pct": amdr_ranges["fat"]["min"],    "max_pct": amdr_ranges["fat"]["max"]},
            },

            # 钠/糖/SFA 目标（上限）与口径
            "sodium_mg_max": sodium_max,
            "sugars": {"type": sugar_type, "pct_max": sugar_pct_max},
            "sat_fat_pct_max": sfa_pct_max,

            # 纤维最小值（随能量）
            "fiber_g_min": fiber_min,

            # 简洁审计（sources & seed）
            "audit": {
                "sources": SOURCES,
                "seed": int(seed),
            }
        }

        if args.verbose_audit:
            record["audit"]["ranges_used"] = {
                "amdr_pct": AMDR_PRESETS[macro_key],
                "sodium_mg_max": sodium_max,
                "sugars_cap": SUGAR_CAPS[sug_key],
                "sfa_cap_pct": sfa_pct_max,
                "energy_bucket": args.energy_bucket,
            }

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"[step3] users={n_users}  tags={len(ALL_TAGS)}  → {args.out}")
    print("[done] 示例字段：user_id, tag, energy_kcal_target, amdr{carb/protein/fat target+min/max}, "
          "sodium_mg_max, sugars{type,pct_max}, sat_fat_pct_max, fiber_g_min, audit{sources,seed[,ranges_used]}")
    

if __name__ == "__main__":
    main()
