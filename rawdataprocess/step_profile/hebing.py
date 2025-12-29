#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path


def read_jsonl_by_user(path: Path) -> dict:
    data: dict = {}
    if not path.exists():
        return data
    buf = ""
    def _maybe_flush_buffer():
        nonlocal buf
        s = buf.strip()
        if not s:
            buf = ""
            return False
        # Heuristic: try parse; if fails, keep buffering
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return False
        uid = obj.get("user_id")
        if uid is not None:
            data[uid] = obj
        buf = ""
        return True

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line.strip():
                _maybe_flush_buffer()
                continue
            # Accumulate and attempt parse
            buf = (buf + line) if not buf else (buf + line)
            if not _maybe_flush_buffer():
                # If braces look balanced, attempt parse again
                open_b = buf.count("{")
                close_b = buf.count("}")
                if open_b > 0 and open_b == close_b:
                    _maybe_flush_buffer()
        # flush remainder
        _maybe_flush_buffer()
    return data


def main():
    # Default locations (adjust if needed)
    targets_path_candidates = [
        Path("work/recipebench/data/8step_profile/nutrition_targets.jsonl"),
        Path("nutrition_targets.jsonl"),
    ]
    targets_path = next((p for p in targets_path_candidates if p.exists()), targets_path_candidates[0])
    prefs_path_candidates = [
        Path("work/recipebench/data/8step_profile/pref_ingredients.jsonl"),
        Path(r"D:\PycharmProjects\recipebench\scripts\step_profile\pref_ingredients.jsonl"),
    ]
    prefs_path = next((p for p in prefs_path_candidates if p.exists()), prefs_path_candidates[0])
    out_path = Path("work/recipebench/data/8step_profile/user_profile_merged.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets = read_jsonl_by_user(targets_path)
    prefs = read_jsonl_by_user(prefs_path)

    user_ids = sorted(set(targets.keys()) | set(prefs.keys()))

    with out_path.open("w", encoding="utf-8") as w:
        for uid in user_ids:
            merged_profile = {"user_id": uid}

            # 合并偏好食材（pref_ingredients）
            if uid in prefs:
                merged_profile["liked_ingredients"] = prefs[uid].get("liked_ingredients", [])
                merged_profile["disliked_ingredients"] = prefs[uid].get("disliked_ingredients", [])
                merged_profile["__source_pref"] = "pref_ingredients"

            # 合并营养目标（nutrition_targets）
            if uid in targets:
                merged_profile["nutrition_targets"] = targets[uid]
                merged_profile["__source_target"] = "nutrition_targets"

            # 写入合并后的用户数据
            w.write(json.dumps(merged_profile, ensure_ascii=False) + "\n")

    print(f"[merge] done → {out_path} | users={len(user_ids)} | inputs=({targets_path}, {prefs_path})")


if __name__ == "__main__":
    main()
