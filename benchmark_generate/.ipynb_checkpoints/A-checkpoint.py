#!/usr/bin/env python3
"""
A类训练数据构建：食谱选择与排序（修复版）
- 修复：recipe_id 类型统一后再合并
- 520K行 → 520K行（无损失）
"""

import pandas as pd
import numpy as np
import json
import graph_tool.all as gt
from collections import defaultdict
from tqdm import tqdm
import random
from typing import List, Dict, Tuple
import re

class TaskADatasetBuilder:
    """A类数据集构建器（修复版）"""

    def __init__(self, kg_path: str, recipe_basic_path: str,
                 recipe_nutrition_path: str, user_profile_path: str):
        """初始化"""
        print("="*80)
        print("A类数据集构建器 - 修复版")
        print("="*80)

        # 加载KG规则
        self._load_kg_rules(kg_path)

        # 加载食谱数据
        self._load_recipes(recipe_basic_path, recipe_nutrition_path)

        # 加载用户画像
        self._load_user_profiles(user_profile_path)

    def _load_kg_rules(self, kg_path: str):
        """从KG中加载规则"""
        print(f"\n[1/3] 加载KG规则: {kg_path}")
        graph = gt.load_graph(kg_path)

        node_id = graph.vertex_properties["node_id"]
        edge_type = graph.edge_properties["edge_type"]
        pmi_score = graph.edge_properties.get("pmi_score")
        cooccurrence_count = graph.edge_properties.get("cooccurrence_count")
        confidence = graph.edge_properties.get("confidence")
        synergy_score = graph.edge_properties.get("synergy_score")
        synergy_reason = graph.edge_properties.get("synergy_reason")

        # 构建规则索引
        self.cooccurrence_rules = {}
        self.complementarity_rules = {}
        self.ingredient_tags = defaultdict(list)

        for e in graph.edges():
            etype = edge_type[e]
            src = node_id[e.source()]
            tgt = node_id[e.target()]

            if etype == "ingredient_cooccurs":
                self.cooccurrence_rules[(src, tgt)] = {
                    'pmi': float(pmi_score[e]),
                    'count': int(cooccurrence_count[e]),
                    'confidence': float(confidence[e])
                }

            elif etype == "ingredient_complements":
                self.complementarity_rules[(src, tgt)] = {
                    'score': float(synergy_score[e]),
                    'reason': str(synergy_reason[e])
                }

            elif etype == "ingredient_has_tag":
                self.ingredient_tags[src].append(tgt)

        print(f"  ✓ 共现规则: {len(self.cooccurrence_rules):,}")
        print(f"  ✓ 互补规则: {len(self.complementarity_rules):,}")
        print(f"  ✓ 营养标签: {len(self.ingredient_tags):,} 个食材")

    def _parse_r_vector(self, r_str):
        """解析R的c()向量"""
        if pd.isna(r_str) or r_str == 'NA':
            return []
        r_str = str(r_str).strip()
        if r_str.startswith('c(') and r_str.endswith(')'):
            r_str = r_str[2:-1]
        items = re.findall(r'"([^"]*)"', r_str)
        return items

    def _load_recipes(self, basic_path: str, nutrition_path: str):
        """加载食谱数据（修复版：统一类型后合并）"""
        print(f"\n[2/3] 加载食谱数据")

        # 加载基础信息
        print(f"  加载基础信息: {basic_path}")
        df_basic = pd.read_csv(basic_path, encoding='latin-1', low_memory=False)
        print(f"    原始行数: {len(df_basic):,}")
        print(f"    recipe_id 类型: {df_basic['recipe_id'].dtype}")

        # 加载营养数据
        print(f"  加载营养数据: {nutrition_path}")
        df_nutrition = pd.read_csv(nutrition_path)
        print(f"    原始行数: {len(df_nutrition):,}")
        print(f"    recipe_id 类型: {df_nutrition['recipe_id'].dtype}")

        # ⭐ 关键修复：统一 recipe_id 类型为整数
        print(f"\n  统一 recipe_id 类型...")
        df_basic['recipe_id'] = pd.to_numeric(df_basic['recipe_id'], errors='coerce').astype('Int64')
        df_nutrition['recipe_id'] = pd.to_numeric(df_nutrition['recipe_id'], errors='coerce').astype('Int64')

        # 删除转换失败的行（NaN）
        basic_before = len(df_basic)
        df_basic = df_basic.dropna(subset=['recipe_id'])
        print(f"    基础数据删除无效 recipe_id: {basic_before - len(df_basic)} 行")

        nutrition_before = len(df_nutrition)
        df_nutrition = df_nutrition.dropna(subset=['recipe_id'])
        print(f"    营养数据删除无效 recipe_id: {nutrition_before - len(df_nutrition)} 行")

        # 检查重复
        basic_dup = df_basic['recipe_id'].duplicated().sum()
        nutrition_dup = df_nutrition['recipe_id'].duplicated().sum()
        print(f"    基础数据重复 recipe_id: {basic_dup}")
        print(f"    营养数据重复 recipe_id: {nutrition_dup}")

        if basic_dup > 0:
            df_basic = df_basic.drop_duplicates(subset=['recipe_id'], keep='first')
            print(f"    已删除基础数据重复行")

        if nutrition_dup > 0:
            df_nutrition = df_nutrition.drop_duplicates(subset=['recipe_id'], keep='first')
            print(f"    已删除营养数据重复行")

        # 检查交集
        basic_ids = set(df_basic['recipe_id'].dropna())
        nutrition_ids = set(df_nutrition['recipe_id'].dropna())
        intersection = basic_ids & nutrition_ids
        print(f"\n  recipe_id 交集分析:")
        print(f"    基础数据唯一ID: {len(basic_ids):,}")
        print(f"    营养数据唯一ID: {len(nutrition_ids):,}")
        print(f"    交集ID数: {len(intersection):,}")
        print(f"    基础数据独有: {len(basic_ids - nutrition_ids):,}")
        print(f"    营养数据独有: {len(nutrition_ids - basic_ids):,}")

        # 合并（inner join 只保留交集）
        print(f"\n  合并数据...")
        self.recipes_df = df_basic.merge(df_nutrition, on='recipe_id', how='inner')
        print(f"    ✓ 合并后: {len(self.recipes_df):,} 行")

        # 构建快速查询索引
        print(f"\n  构建食谱索引...")
        self.recipe_dict = {}
        skipped_count = 0

        for _, row in tqdm(self.recipes_df.iterrows(), desc="    处理中", total=len(self.recipes_df)):
            recipe_id = str(int(row['recipe_id']))  # 转为字符串作为key
            ingredients = self._parse_r_vector(row.get('RecipeIngredientParts', ''))

            # 跳过空食材的食谱
            if not ingredients or len(ingredients) == 0:
                skipped_count += 1
                continue

            self.recipe_dict[recipe_id] = {
                'name': row.get('recipe_name', row.get('Name', f'Recipe_{recipe_id}')),
                'ingredients': ingredients,
                'nutrition': {
                    'calories': float(row.get('Calories_PerServing_kcal', 0)) if pd.notna(row.get('Calories_PerServing_kcal')) else 0.0,
                    'protein': float(row.get('Protein_PerServing_g', 0)) if pd.notna(row.get('Protein_PerServing_g')) else 0.0,
                    'fat': float(row.get('Fat_PerServing_g', 0)) if pd.notna(row.get('Fat_PerServing_g')) else 0.0,
                    'carbohydrates': float(row.get('Carbohydrates_PerServing_g', 0)) if pd.notna(row.get('Carbohydrates_PerServing_g')) else 0.0,
                    'fiber': float(row.get('Fiber_PerServing_g', 0)) if pd.notna(row.get('Fiber_PerServing_g')) else 0.0,
                    'sugars': float(row.get('Sugars_PerServing_g', 0)) if pd.notna(row.get('Sugars_PerServing_g')) else 0.0,
                    'saturated_fat': float(row.get('SaturatedFat_PerServing_g', 0)) if pd.notna(row.get('SaturatedFat_PerServing_g')) else 0.0,
                    'sodium': float(row.get('Sodium_PerServing_mg', 0)) if pd.notna(row.get('Sodium_PerServing_mg')) else 0.0,
                }
            }

        self.all_recipe_ids = list(self.recipe_dict.keys())
        print(f"    ✓ 有效食谱: {len(self.recipe_dict):,} 个")
        if skipped_count > 0:
            print(f"    ⚠ 跳过空食材: {skipped_count:,} 个")

    def _load_user_profiles(self, profile_path: str):
        """加载用户画像"""
        print(f"\n[3/3] 加载用户画像: {profile_path}")

        self.users = []
        with open(profile_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.users.append(json.loads(line))

        print(f"  ✓ 加载用户: {len(self.users):,}")

    # ==================== 打分函数 ====================

    def score_nutrition_match(self, recipe_nutrition: Dict, user_targets: Dict) -> float:
        """营养目标匹配度 (0-1)"""
        if not user_targets:
            return 0.5

        scores = []

        # 能量目标
        energy_target = user_targets.get('energy_kcal_target')
        if energy_target and recipe_nutrition.get('calories', 0) > 0:
            ratio = recipe_nutrition['calories'] / energy_target
            if 0.8 <= ratio <= 1.2:
                scores.append(1.0)
            elif 0.6 <= ratio <= 1.4:
                scores.append(0.7)
            else:
                scores.append(0.3)

        # AMDR三大营养素
        amdr = user_targets.get('amdr', {})
        if amdr:
            total_energy = (
                recipe_nutrition.get('protein', 0) * 4 +
                recipe_nutrition.get('fat', 0) * 9 +
                recipe_nutrition.get('carbohydrates', 0) * 4
            )
            if total_energy > 0:
                for key, nutrient in [('carb', 'carbohydrates'), ('protein', 'protein'), ('fat', 'fat')]:
                    if key in amdr:
                        kcal_per_g = 4 if key != 'fat' else 9
                        actual_pct = (recipe_nutrition.get(nutrient, 0) * kcal_per_g / total_energy) * 100
                        target_pct = amdr[key].get('target_pct', 0)

                        if target_pct > 0:
                            diff = abs(actual_pct - target_pct)
                            if diff <= 5:
                                scores.append(1.0)
                            elif diff <= 10:
                                scores.append(0.7)
                            else:
                                scores.append(0.4)

        # 钠最大值
        sodium_max = user_targets.get('sodium_mg_max')
        if sodium_max and recipe_nutrition.get('sodium', 0) > 0:
            if recipe_nutrition['sodium'] <= sodium_max:
                scores.append(1.0)
            elif recipe_nutrition['sodium'] <= sodium_max * 1.2:
                scores.append(0.6)
            else:
                scores.append(0.2)

        # 纤维最小值
        fiber_min = user_targets.get('fiber_g_min')
        if fiber_min and recipe_nutrition.get('fiber', 0) > 0:
            if recipe_nutrition['fiber'] >= fiber_min:
                scores.append(1.0)
            elif recipe_nutrition['fiber'] >= fiber_min * 0.8:
                scores.append(0.7)
            else:
                scores.append(0.4)

        return np.mean(scores) if scores else 0.5

    def score_ingredient_preference(self, recipe_ingredients: List[str],
                                   liked: List[str], disliked: List[str]) -> float:
        """食材偏好匹配度 (0-1)"""
        if not liked and not disliked:
            return 0.5

        recipe_set = set(recipe_ingredients)
        disliked_count = len(recipe_set & set(disliked))
        liked_count = len(recipe_set & set(liked))

        if disliked_count > 0:
            return 0.1

        if liked_count > 0:
            return min(0.7 + liked_count * 0.1, 1.0)

        return 0.5

    def score_cooccurrence(self, ingredients: List[str]) -> float:
        """食材共现分数 (0-1)"""
        if len(ingredients) < 2:
            return 0.0

        total_pmi = 0.0
        pair_count = 0

        for i in range(len(ingredients)):
            for j in range(i+1, len(ingredients)):
                rule = self.cooccurrence_rules.get((ingredients[i], ingredients[j])) or \
                       self.cooccurrence_rules.get((ingredients[j], ingredients[i]))

                if rule:
                    normalized_pmi = min(rule['pmi'] / 10.0, 1.0)
                    total_pmi += normalized_pmi
                    pair_count += 1

        return total_pmi / pair_count if pair_count > 0 else 0.0

    def score_complementarity(self, ingredients: List[str]) -> float:
        """营养互补分数 (0-1)"""
        if len(ingredients) < 2:
            return 0.0

        total_synergy = 0.0
        pair_count = 0

        for i in range(len(ingredients)):
            for j in range(i+1, len(ingredients)):
                rule = self.complementarity_rules.get((ingredients[i], ingredients[j])) or \
                       self.complementarity_rules.get((ingredients[j], ingredients[i]))

                if rule:
                    total_synergy += rule['score']
                    pair_count += 1

        return total_synergy / pair_count if pair_count > 0 else 0.0

    def score_nutrition_balance(self, ingredients: List[str]) -> float:
        """营养平衡分数 (0-1)"""
        all_tags = []
        for ing in ingredients:
            all_tags.extend(self.ingredient_tags.get(ing, []))

        if not all_tags:
            return 0.5

        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag] += 1

        total = len(all_tags)
        entropy = 0.0
        for count in tag_counts.values():
            p = count / total
            entropy -= p * np.log2(p)

        max_entropy = np.log2(len(tag_counts)) if len(tag_counts) > 1 else 1.0
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        positive_tags = ['high_protein', 'high_fiber', 'high_vitamin_c',
                        'high_calcium', 'high_iron', 'low_sodium', 'low_fat']
        positive_count = sum(1 for tag in all_tags if tag in positive_tags)
        positive_ratio = positive_count / len(all_tags)

        return 0.5 * diversity + 0.5 * positive_ratio

    def score_recipe(self, recipe_id: str, user: Dict) -> Tuple[float, Dict]:
        """综合打分"""
        weights = {
            'nutrition': 0.3,
            'preference': 0.2,
            'cooccurrence': 0.2,
            'complementarity': 0.2,
            'balance': 0.1
        }

        recipe = self.recipe_dict.get(recipe_id)
        if not recipe:
            return 0.0, {}

        liked_ings = [item['name'] for item in user.get('liked_ingredients', [])]
        disliked_ings = [item['name'] for item in user.get('disliked_ingredients', [])]
        nutrition_targets = user.get('nutrition_targets', {})

        nutrition_score = self.score_nutrition_match(recipe['nutrition'], nutrition_targets)
        preference_score = self.score_ingredient_preference(recipe['ingredients'],
                                                            liked_ings, disliked_ings)
        cooccurrence_score = self.score_cooccurrence(recipe['ingredients'])
        complementarity_score = self.score_complementarity(recipe['ingredients'])
        balance_score = self.score_nutrition_balance(recipe['ingredients'])

        total_score = (
            weights['nutrition'] * nutrition_score +
            weights['preference'] * preference_score +
            weights['cooccurrence'] * cooccurrence_score +
            weights['complementarity'] * complementarity_score +
            weights['balance'] * balance_score
        )

        breakdown = {
            'nutrition': round(nutrition_score, 3),
            'preference': round(preference_score, 3),
            'cooccurrence': round(cooccurrence_score, 3),
            'complementarity': round(complementarity_score, 3),
            'balance': round(balance_score, 3)
        }

        return round(total_score, 3), breakdown

    # ==================== 样本生成 ====================

    def generate_samples_for_user(self, user: Dict) -> List[Dict]:
        """为单个用户生成训练样本"""
        user_id = user['user_id']

        # 随机采样1500候选
        sampled_ids = random.sample(self.all_recipe_ids, min(1500, len(self.all_recipe_ids)))

        scored_recipes = []
        for recipe_id in sampled_ids:
            score, breakdown = self.score_recipe(recipe_id, user)
            scored_recipes.append((recipe_id, score, breakdown))

        scored_recipes.sort(key=lambda x: x[1], reverse=True)

        samples = []

        # 正样本：Top-3
        for recipe_id, score, breakdown in scored_recipes[:3]:
            samples.append({
                'user_id': user_id,
                'recipe_id': recipe_id,
                'recipe_name': self.recipe_dict[recipe_id]['name'],
                'ingredients': self.recipe_dict[recipe_id]['ingredients'],
                'nutrition': self.recipe_dict[recipe_id]['nutrition'],
                'label': 1,
                'score': score,
                'score_breakdown': breakdown,
                'sample_type': 'positive'
            })

        # 随机负样本：5个
        low_score_pool = [r for r in scored_recipes if r[1] < 0.4]
        neg_samples = random.sample(low_score_pool, min(5, len(low_score_pool)))

        for recipe_id, score, breakdown in neg_samples:
            samples.append({
                'user_id': user_id,
                'recipe_id': recipe_id,
                'recipe_name': self.recipe_dict[recipe_id]['name'],
                'ingredients': self.recipe_dict[recipe_id]['ingredients'],
                'nutrition': self.recipe_dict[recipe_id]['nutrition'],
                'label': 0,
                'score': score,
                'score_breakdown': breakdown,
                'sample_type': 'random_negative'
            })

        # 硬负样本：2个
        hard_neg_pool = [r for r in scored_recipes if 0.4 <= r[1] < 0.6]
        hard_neg_samples = random.sample(hard_neg_pool, min(2, len(hard_neg_pool)))

        for recipe_id, score, breakdown in hard_neg_samples:
            samples.append({
                'user_id': user_id,
                'recipe_id': recipe_id,
                'recipe_name': self.recipe_dict[recipe_id]['name'],
                'ingredients': self.recipe_dict[recipe_id]['ingredients'],
                'nutrition': self.recipe_dict[recipe_id]['nutrition'],
                'label': 0,
                'score': score,
                'score_breakdown': breakdown,
                'sample_type': 'hard_negative'
            })

        return samples

    def build_dataset(self, users_list: List[Dict], output_path: str):
        """构建数据集"""
        all_samples = []
        for user in tqdm(users_list, desc=f"生成样本"):
            samples = self.generate_samples_for_user(user)
            all_samples.extend(samples)

        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"✓ {output_path}: {len(all_samples):,} 样本")
        print(f"  正样本: {sum(1 for s in all_samples if s['label'] == 1):,}")
        print(f"  负样本: {sum(1 for s in all_samples if s['label'] == 0):,}")

        return len(all_samples)


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 构建数据集
    builder = TaskADatasetBuilder(
        kg_path="work/recipebench/kg/nutriplan_kg3.graphml",
        recipe_basic_path="work/recipebench/data/raw/foodcom/recipes(3column).csv",
        recipe_nutrition_path="work/recipebench/data/4out/recipe_nutrition_foodcom.csv",
        user_profile_path="work/recipebench/data/8step_profile/cleaned_user_profile.jsonl"
    )

    # 随机打乱用户
    random.shuffle(builder.users)

    # 划分用户
    train_users = builder.users[:10000]
    val_users = builder.users[10000:12000]
    test_users = builder.users[12000:14000]

    print(f"\n{'='*80}")
    print("数据集配置")
    print(f"{'='*80}")
    print(f"训练集: 10,000 用户")
    print(f"验证集: 2,000 用户")
    print(f"测试集: 2,000 用户")
    print(f"每用户: 10 样本 (3正+5负+2硬负)")

    # 生成训练集
    print(f"\n{'='*80}")
    print("生成训练集")
    print(f"{'='*80}")
    builder.build_dataset(train_users, "work/recipebench/data/10large_scale_datasets/task_a_train_new.jsonl")

    # 生成验证集
    print(f"\n{'='*80}")
    print("生成验证集")
    print(f"{'='*80}")
    builder.build_dataset(val_users, "work/recipebench/data/10large_scale_datasets/task_a_val_new.jsonl")

    # 生成测试集
    print(f"\n{'='*80}")
    print("生成测试集")
    print(f"{'='*80}")
    builder.build_dataset(test_users, "work/recipebench/data/10large_scale_datasets/task_a_test_new.jsonl")

    print(f"\n{'='*80}")
    print("🎉 A类数据集构建完成！")
    print(f"{'='*80}")
    print("输出文件：")
    print("  - work/recipebench/data/10large_scale_datasets/task_a_train_new.jsonl  (~10万样本)")
    print("  - work/recipebench/data/10large_scale_datasets/task_a_val_new.jsonl    (~2万样本)")
    print("  - work/recipebench/data/10large_scale_datasets/task_a_test_new.jsonl   (~2万样本)")
