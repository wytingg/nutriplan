# scripts/trainset/build_A_selection_from_kg.py
"""
A类训练集：候选选取/排序（Selection/Ranking）
严格按照论文标准构建正样本和三类难负样本 - 增强版错误处理
python work/recipebench/scripts/KG/build_A_selection_from_kg.py \
    --kg work/recipebench/kg/nutriplan_kg2.graphml \
    --user-h5 work/recipebench/data/9large_scale_subgraphs/user_subgraphs1.h5 \
    --recipe-h5 work/recipebench/data/9large_scale_subgraphs/recipe_subgraphs1.h5 \
    --parquet work/recipebench/data/4out/recipe_nutrients_main.parquet \
    --out work/recipebench/data/10large_scale_datasets/A_selection_v4.jsonl \
    --pref-threshold 0.3
"""
import argparse, random, json, time
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any, List, Set, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from kg_io import KGAccessor
from h5_io import SubgraphH5

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """计算Jaccard相似度"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def compute_preference_score(ingredients: List[str], likes: Set[str], dislikes: Set[str]) -> Dict[str, Any]:
    """计算偏好分数：食材命中率"""
    if not ingredients:
        return {"score": 0.0, "hit_rate": 0.0, "like_hits": 0, "dislike_hits": 0}

    ing_set = set([s.split("|")[0].strip().lower() for s in ingredients if s])
    likes_lower = set([x.lower() for x in likes if x])
    dislikes_lower = set([x.lower() for x in dislikes if x])

    # 食材命中率
    hit_rate = len(ing_set & likes_lower) / len(likes_lower) if likes_lower else 0.0

    # 禁忌食材惩罚
    dislike_hits = len(ing_set & dislikes_lower)

    # 综合偏好分数：命中率 - 禁忌惩罚(每个-0.5)
    pref_score = hit_rate - 0.5 * dislike_hits

    return {
        "score": pref_score,
        "hit_rate": hit_rate,
        "like_hits": len(ing_set & likes_lower),
        "dislike_hits": dislike_hits
    }

def check_hard_constraints(nutr: Dict[str, float], constraints: List[Dict[str, Any]],
                           epsilon: float = 0.0) -> Tuple[bool, Dict[str, Any]]:
    """检查硬约束"""
    if not nutr:
        return False, {}

    all_pass = True
    details = {}

    # 获取能量值用于百分比转换
    energy = nutr.get("energy_kcal", 2000)  # 默认2000kcal

    for c in constraints:
        nid = c["nid"]
        typ = c["type"]

        # 处理百分比约束：转换为克数
        if typ == "range_pct" or c.get("needs_conversion"):
            # AMDR三大营养素百分比范围
            if nid in ("carbohydrates_g", "protein_g", "fat_g"):
                lo_pct = c.get("lo", 0)
                up_pct = c.get("up", 100)
                # 转换公式：能量 × 百分比 / 每克热量
                cal_per_g = 4 if nid in ("carbohydrates_g", "protein_g") else 9
                lo_g = (energy * lo_pct / 100) / cal_per_g
                up_g = (energy * up_pct / 100) / cal_per_g
                # 临时修改为range类型
                c = {"nid": nid, "type": "range", "lo": lo_g, "up": up_g, "strict": True}
                typ = "range"
            # 糖分百分比最大值
            elif nid == "sugars_total_g" and typ == "max_pct":
                pct_up = c.get("pct_up", 100)
                # 糖分: 4 kcal/g
                up_g = (energy * pct_up / 100) / 4
                c = {"nid": nid, "type": "max", "up": up_g, "strict": True}
                typ = "max"
            # 饱和脂肪百分比最大值
            elif nid == "saturated_fat_g" and typ == "max_pct":
                pct_up = c.get("pct_up", 100)
                # 饱和脂肪: 9 kcal/g
                up_g = (energy * pct_up / 100) / 9
                c = {"nid": nid, "type": "max", "up": up_g, "strict": True}
                typ = "max"
            else:
                continue  # 其他类型的百分比约束暂时跳过

        val = nutr.get(nid)

        if val is None:
            all_pass = False
            details[nid] = {"pass": False, "reason": "missing"}
            continue

        # 宽容带计算
        try:
            if typ == "max":
                threshold = c["up"] * (1 + epsilon)
                passed = val <= threshold
                gap = max(0, val - c["up"])
                gap_pct = gap / c["up"] if c["up"] > 0 else 0
            elif typ == "min":
                threshold = c["lo"] * (1 - epsilon)
                passed = val >= threshold
                gap = max(0, c["lo"] - val)
                gap_pct = gap / c["lo"] if c["lo"] > 0 else 0
            elif typ == "range":
                lo_threshold = c["lo"] * (1 - epsilon)
                up_threshold = c["up"] * (1 + epsilon)
                passed = lo_threshold <= val <= up_threshold
                gap = min(abs(val - c["lo"]), abs(val - c["up"])) if not (c["lo"] <= val <= c["up"]) else 0
                gap_pct = gap / max(c["lo"], c["up"]) if max(c["lo"], c["up"]) > 0 else 0
            else:
                continue

            details[nid] = {
                "pass": passed,
                "val": val,
                "gap": gap,
                "gap_pct": gap_pct
            }

            if not passed:
                all_pass = False
        except Exception as e:
            all_pass = False
            details[nid] = {"pass": False, "reason": f"error: {e}"}

    return all_pass, details

def is_valid_nutrition(nutr: Dict[str, float]) -> bool:
    """过滤极端异常值"""
    if not nutr or not isinstance(nutr, dict):
        return False

    checks = {
        'energy_kcal': (10, 5000),
        'protein_g': (0, 300),
        'fat_g': (0, 300),
        'carbohydrates_g': (0, 500),
        'fiber_g': (0, 100),
        'sodium_mg': (0, 10000)
    }
    for nid, (min_v, max_v) in checks.items():
        if nid in nutr:
            try:
                val = float(nutr[nid])
                if val < min_v or val > max_v:
                    return False
            except (TypeError, ValueError):
                return False
    return True

def classify_sample(nutr: Dict[str, float], ings: List[str],
                   constraints: List[Dict], likes: Set[str], dislikes: Set[str],
                   pref_threshold: float = 0.3) -> Tuple[str, str]:
    """分类样本为：positive, hard_neg_boundary, hard_neg_pref, negative"""
    if not nutr or not isinstance(nutr, dict):
        return ("negative", "invalid_nutrition")

    # 检查硬约束（不含宽容带）
    strict_pass, strict_details = check_hard_constraints(nutr, constraints, epsilon=0.0)

    # 检查轻度越界（3-5%宽容带）
    loose_pass, loose_details = check_hard_constraints(nutr, constraints, epsilon=0.05)

    # 计算偏好分数
    pref = compute_preference_score(ings, likes, dislikes)

    # 正样本：严格满足硬约束 + 偏好分数≥阈值
    if strict_pass and pref["score"] >= pref_threshold:
        return ("positive", f"pass_all; pref={pref['score']:.2f}")

    # 难负样本1：轻度越界（2-5%）
    if not strict_pass and loose_pass:
        failed = [k for k, v in strict_details.items() if not v.get("pass", True)]
        return ("hard_neg_boundary", f"light_boundary: {','.join(failed[:2])}")

    # 难负样本2：营养全满足但偏好低
    if strict_pass and pref["score"] < pref_threshold:
        return ("hard_neg_pref", f"low_pref={pref['score']:.2f}")

    # 其他为普通负样本
    return ("negative", "fail_constraints")

def build_ing_inverted_index(h5: SubgraphH5, rid_list: List[str]) -> Dict[str, Set[str]]:
    inv = defaultdict(set)
    error_count = 0
    for rid in tqdm(rid_list, desc="build inverted idx"):
        try:
            ings = h5.load_recipe_ing(rid)
            if not ings:
                continue
            for token in ings:
                if not token:
                    continue
                parts = token.split("|")
                if not parts[0]:
                    continue
                name = parts[0].strip().lower()
                if name:
                    inv[name].add(rid)
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"\n[WARN] 跳过食谱 {rid}: {type(e).__name__}")
    if error_count > 0:
        print(f"\n[INFO] 共跳过 {error_count} 个食谱")
    return inv

def main(args):
    random.seed(args.seed)
    kg = KGAccessor(args.kg)

    # 载入营养表
    import pyarrow.parquet as pq
    table = pq.read_table(args.parquet)
    data_dict = {}
    for col_name in table.column_names:
        col_data = table.column(col_name).to_pylist()
        col_data_flat = []
        for val in col_data:
            if isinstance(val, (list, tuple)):
                col_data_flat.append(val[0] if len(val) > 0 else None)
            else:
                col_data_flat.append(val)
        data_dict[col_name] = col_data_flat

    nutr = pd.DataFrame(data_dict).set_index("recipe_id", drop=False)

    # 构建nutrition_per_serv
    nutrient_cols = {
        'energy_kcal': 'calories_kcal',
        'protein_g': 'protein_g',
        'fat_g': 'fat_g',
        'carbohydrates_g': 'carbohydrates_g',
        'fiber_g': 'fiber_g',
        'sugars_total_g': 'sugars_total_g',
        'saturated_fat_g': 'saturated_fat_g',
        'sodium_mg': 'sodium_mg'
    }

    def build_nutrition_dict(row):
        d = {}
        for nid, col in nutrient_cols.items():
            if col in row.index and pd.notna(row[col]):
                try:
                    d[nid] = float(row[col])
                except:
                    pass
        return d

    nutr['nutrition_per_serv'] = nutr.apply(build_nutrition_dict, axis=1)

    # 检查标题列
    title_col = None
    for name in ['Name', 'name', 'title', 'recipe_name']:
        if name in nutr.columns:
            title_col = name
            print(f"[DEBUG] 找到标题列: {title_col}")
            break

    with SubgraphH5(args.user_h5, args.recipe_h5) as h5:
        rid_list = list(h5.iter_all_recipe_ids())
        inv = build_ing_inverted_index(h5, rid_list)

        import h5py
        with h5py.File(args.user_h5, "r") as f:
            user_ids = list(f["likes"].keys())

        print(f"[DEBUG] 用户数: {len(user_ids)}, 食谱数: {len(rid_list)}")

        stats = defaultdict(int)

        with open(args.out, "w", encoding="utf-8") as fout:
            for idx, uid in enumerate(tqdm(user_ids, desc="A-build")):
                try:
                    u = h5.load_user(uid)
                except Exception as e:
                    if idx == 0:
                        print(f"[ERROR] 无法加载用户 {uid}: {e}")
                    stats["user_load_error"] += 1
                    continue

                likes = set(u["likes"]) if u["likes"] else set()
                dislikes = set(u["dislikes"]) if u["dislikes"] else set()

                # 调试第一个用户
                if idx == 0:
                    print(f"\n[DEBUG] 第一个用户 {uid}:")
                    print(f"  likes: {len(likes)}, dislikes: {len(dislikes)}")
                    print(f"  likes示例: {list(likes)[:3]}")

                # 从KG获取约束
                constraints = kg.get_user_targets_struct(uid)

                if idx == 0:
                    print(f"  constraints数量: {len(constraints)}")
                    print(f"  constraints详情:")
                    for c in constraints:
                        print(f"    - {c['nid']}: {c['type']} (strict={c.get('strict', True)})")

                # 候选召回
                cand = set()
                for l in likes:
                    if l:
                        cand |= inv.get(l.strip().lower(), set())

                if len(cand) < args.topk_cand:
                    extra = [r for r in rid_list if r not in cand]
                    random.shuffle(extra)
                    cand |= set(extra[:args.topk_cand - len(cand)])
                else:
                    cand = set(list(cand)[:args.topk_cand])

                if idx == 0:
                    print(f"  召回候选数: {len(cand)}")

                # 分类候选
                positives = []
                hard_neg_boundary = []
                hard_neg_pref = []
                hard_neg_similar = []
                recipes_context = []

                for rid in cand:
                    rid_key = rid
                    if rid not in nutr.index:
                        try:
                            rid_key = int(rid)
                            if rid_key not in nutr.index:
                                continue
                        except:
                            continue

                    row = nutr.loc[rid_key]

                    # 检查nutrition_per_serv是否存在
                    if "nutrition_per_serv" not in row or not row["nutrition_per_serv"]:
                        stats["missing_nutrition"] += 1
                        if idx == 0 and stats["missing_nutrition"] <= 2:
                            print(f"  [WARN] 跳过食谱 {rid}: 缺少 nutrition_per_serv")
                        continue

                    nutrition = row["nutrition_per_serv"]

                    # 过滤异常值
                    if not is_valid_nutrition(nutrition):
                        stats["invalid_nutrition"] += 1
                        if idx == 0 and stats["invalid_nutrition"] <= 2:
                            print(f"  [WARN] 跳过食谱 {rid}: 异常营养值")
                        continue

                    try:
                        ings = h5.load_recipe_ing(rid)
                    except Exception as e:
                        stats["ingredient_error"] += 1
                        continue

                    # 获取标题
                    title = ""
                    if title_col and title_col in row.index:
                        title = str(row[title_col]) if pd.notna(row[title_col]) else ""

                    # 构建macro字段
                    macro = {k: nutrition.get(k) for k in ['energy_kcal', 'protein_g', 'fat_g',
                                                            'carbohydrates_g', 'sodium_mg'] if k in nutrition}

                    recipe_obj = {
                        "rid": rid,
                        "title": title,
                        "macro": macro,
                        "ings": [s.split("|")[0] for s in ings if s][:20]
                    }
                    recipes_context.append(recipe_obj)

                    # 分类
                    cat, reason = classify_sample(nutrition, ings, constraints, likes, dislikes, args.pref_threshold)

                    if cat == "positive":
                        positives.append(rid)
                        stats["positives"] += 1
                    elif cat == "hard_neg_boundary":
                        hard_neg_boundary.append(rid)
                        stats["hard_neg_boundary"] += 1
                    elif cat == "hard_neg_pref":
                        hard_neg_pref.append(rid)
                        stats["hard_neg_pref"] += 1

                # 难负样本3：配料相似但营养/禁忌不符
                if positives:
                    pos_ings_sets = {}
                    for rid in positives:
                        try:
                            ings = h5.load_recipe_ing(rid)
                            if ings:
                                pos_ings_sets[rid] = set([s.split("|")[0].strip().lower() for s in ings if s])
                        except:
                            continue

                    for rid in cand:
                        if rid in positives or rid in hard_neg_boundary or rid in hard_neg_pref:
                            continue
                        try:
                            ings = h5.load_recipe_ing(rid)
                            if not ings:
                                continue
                            ing_set = set([s.split("|")[0].strip().lower() for s in ings if s])

                            if not ing_set:
                                continue

                            for pos_rid, pos_ing_set in pos_ings_sets.items():
                                if not pos_ing_set:
                                    continue
                                sim = jaccard_similarity(ing_set, pos_ing_set)
                                if sim >= 0.6:
                                    hard_neg_similar.append(rid)
                                    stats["hard_neg_similar"] += 1
                                    break
                        except:
                            continue

                if idx == 0:
                    print(f"  原始分类结果:")
                    print(f"    positives: {len(positives)}")
                    print(f"    hard_neg_boundary: {len(hard_neg_boundary)}")
                    print(f"    hard_neg_pref: {len(hard_neg_pref)}")
                    print(f"    hard_neg_similar: {len(hard_neg_similar)}")

                # 采样：正样本M条，难负2M-4M条
                M = min(len(positives), args.pos_per_user)
                positives = positives[:M]

                # 三类难负样本均衡采样
                total_hn = args.hn_per_pos * M if M > 0 else args.hn_per_pos
                hn_per_type = total_hn // 3

                sampled_hn = []
                sampled_hn.extend(hard_neg_boundary[:hn_per_type])
                sampled_hn.extend(hard_neg_pref[:hn_per_type])
                sampled_hn.extend(hard_neg_similar[:hn_per_type])

                # 如果某类不足，从其他类补充
                if len(sampled_hn) < total_hn:
                    remain = total_hn - len(sampled_hn)
                    all_hn = set(hard_neg_boundary + hard_neg_pref + hard_neg_similar) - set(sampled_hn)
                    sampled_hn.extend(list(all_hn)[:remain])

                if idx == 0:
                    print(f"  采样后:")
                    print(f"    positives: {len(positives)}")
                    print(f"    sampled_hn: {len(sampled_hn)}")

                if not positives and not sampled_hn:
                    stats["no_labels"] += 1
                    if idx < 3:
                        print(f"  [WARN] 用户{uid}跳过：无正负样本")
                    continue

                # 构建JSONL（论文格式）- 添加验证
                try:
                    rec = {
                        "uid": int(uid),
                        "constraints": constraints,
                        "pref": {
                            "likes": list(likes),
                            "dislikes": list(dislikes)
                        },
                        "context": {
                            "recipes": recipes_context
                        },
                        "labels": {
                            "positive": positives,
                            "hard_neg": sampled_hn
                        },
                        "provenance": {
                            "profile": f"{args.user_h5}#{uid}",
                            "nutr": args.parquet,
                            "ts": time.strftime("%Y-%m-%d"),
                            "pref_threshold": args.pref_threshold
                        }
                    }
                    # 验证JSON可序列化
                    json.dumps(rec, ensure_ascii=False)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                except (TypeError, ValueError) as e:
                    stats["json_error"] += 1
                    if idx < 3:
                        print(f"  [ERROR] JSON序列化失败 user {uid}: {e}")

        print(f"\n{'='*60}")
        print(f"[统计报告]")
        print(f"  生成样本数: {stats['written']}")
        print(f"  正样本总数: {stats['positives']}")
        print(f"  难负-轻越界: {stats['hard_neg_boundary']}")
        print(f"  难负-低偏好: {stats['hard_neg_pref']}")
        print(f"  难负-相似对比: {stats['hard_neg_similar']}")
        print(f"\n[跳过原因]")
        print(f"  无标签: {stats['no_labels']}")
        print(f"  缺失营养: {stats.get('missing_nutrition', 0)}")
        print(f"  异常营养: {stats['invalid_nutrition']}")
        print(f"  食材错误: {stats.get('ingredient_error', 0)}")
        print(f"  用户加载错误: {stats.get('user_load_error', 0)}")
        print(f"  JSON序列化错误: {stats.get('json_error', 0)}")
        print(f"{'='*60}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--user-h5", required=True)
    ap.add_argument("--recipe-h5", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk-cand", type=int, default=300)
    ap.add_argument("--pos-per-user", type=int, default=8)
    ap.add_argument("--hn-per-pos", type=int, default=3)
    ap.add_argument("--pref-threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
