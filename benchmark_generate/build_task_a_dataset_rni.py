#!/usr/bin/env python3
"""
Task A: Discriminative Ranking - 判别式食谱排序数据集构建
目标：训练LLM学习评估recipe suitability并进行鲁棒排序
输出：instruction + user_profile + ranked_recipes (Top-3 with detailed scoring)
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

# ============================================================================
# 10个指令模板（覆盖不同场景）
# ============================================================================
INSTRUCTION_TEMPLATES = [
    # 1. 健康状况导向
    {
        "template": "I am a {age}-year-old {gender} with {physiological_state}. Please recommend and rank recipes suitable for my health condition, prioritizing nutritional safety and disease management.",
        "type": "health_condition"
    },

    # 2. 营养目标导向
    {
        "template": "Based on my daily nutritional requirements (Energy: {energy_kcal} kcal, Protein: {protein_g}g, Fiber: {fiber_g}g), please rank recipes that best meet these targets.",
        "type": "nutrition_target"
    },

    # 3. 食材偏好导向
    {
        "template": "I enjoy {liked_ingredients} but dislike {disliked_ingredients}. Please rank recipes that match my taste preferences while ensuring nutritional balance.",
        "type": "preference"
    },

    # 4. 综合健康管理
    {
        "template": "As a {physiological_state} patient aged {age}, please rank recipes considering both my medical dietary restrictions and personal preferences.",
        "type": "comprehensive"
    },

    # 5. 特定营养素优化（仅使用食谱中存在的营养素）
    {
        "template": "I need recipes high in {key_nutrient} to meet my RNI of {nutrient_value} {nutrient_unit}. Please rank options that provide adequate amounts of this nutrient.",
        "type": "specific_nutrient"
    },

    # 6. 限制性营养素控制
    {
        "template": "Due to {physiological_state}, I must limit my {restricted_nutrient} intake to {limit_value} {limit_unit}. Please rank recipes that respect this constraint.",
        "type": "restriction"
    },

    # 7. 年龄性别特异性
    {
        "template": "As a {age}-year-old {gender}, please recommend age-appropriate and gender-specific recipes that align with my life stage nutritional needs.",
        "type": "demographic"
    },

    # 8. 能量平衡
    {
        "template": "I need meals that provide approximately {target_percentage}% of my daily energy requirement ({energy_kcal} kcal). Please rank suitable recipes.",
        "type": "energy_balance"
    },

    # 9. 宏量营养素平衡
    {
        "template": "Please rank recipes that provide a balanced ratio of protein ({protein_g}g), carbohydrates ({carb_g}g), and fat ({fat_g}g) per serving.",
        "type": "macronutrient_balance"
    },

    # 10. 多维度综合评分
    {
        "template": "Considering my complete profile (demographics, health status, preferences, and nutritional needs), please provide a comprehensive ranking of recipes with detailed scoring explanations.",
        "type": "multi_dimensional"
    }
]


class TaskADatasetBuilder:
    """Task A: Discriminative Ranking 数据集构建器"""

    def __init__(self, kg_path: str, recipe_basic_path: str,
                 recipe_nutrition_path: str, user_profile_path: str):
        """初始化"""
        print("="*80)
        print("Task A: Discriminative Ranking Dataset Builder")
        print("="*80)

        # 输出指令模板
        self._print_instruction_templates()

        # 加载KG规则
        self._load_kg_rules(kg_path)

        # 加载食谱数据
        self._load_recipes(recipe_basic_path, recipe_nutrition_path)

        # 加载用户画像
        self._load_user_profiles(user_profile_path)

    def _print_instruction_templates(self):
        """在终端输出指令模板"""
        print("\n" + "="*80)
        print("📝 指令模板（10个场景）")
        print("="*80)
        for i, template_info in enumerate(INSTRUCTION_TEMPLATES, 1):
            print(f"\n模板 {i} [{template_info['type']}]:")
            print(f"  {template_info['template']}")
        print("\n" + "="*80 + "\n")

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
        """加载食谱数据"""
        print(f"\n[2/3] 加载食谱数据")

        # 加载基础信息
        print(f"  加载基础信息: {basic_path}")
        df_basic = pd.read_csv(basic_path, encoding='latin-1', low_memory=False)
        print(f"    原始行数: {len(df_basic):,}")

        # 加载营养数据
        print(f"  加载营养数据: {nutrition_path}")
        df_nutrition = pd.read_csv(nutrition_path)
        print(f"    原始行数: {len(df_nutrition):,}")

        # 统一 recipe_id 类型
        df_basic['recipe_id'] = pd.to_numeric(df_basic['recipe_id'], errors='coerce').astype('Int64')
        df_nutrition['recipe_id'] = pd.to_numeric(df_nutrition['recipe_id'], errors='coerce').astype('Int64')

        # 删除无效行和重复
        df_basic = df_basic.dropna(subset=['recipe_id']).drop_duplicates(subset=['recipe_id'], keep='first')
        df_nutrition = df_nutrition.dropna(subset=['recipe_id']).drop_duplicates(subset=['recipe_id'], keep='first')

        # 合并
        self.recipes_df = df_basic.merge(df_nutrition, on='recipe_id', how='inner')
        print(f"    ✓ 合并后: {len(self.recipes_df):,} 行")

        # 构建食谱索引
        print(f"\n  构建食谱索引...")
        self.recipe_dict = {}
        skipped_count = 0

        for _, row in tqdm(self.recipes_df.iterrows(), desc="    处理中", total=len(self.recipes_df)):
            recipe_id = str(int(row['recipe_id']))
            ingredients = self._parse_r_vector(row.get('RecipeIngredientParts', ''))

            if not ingredients:
                skipped_count += 1
                continue

            self.recipe_dict[recipe_id] = {
                'name': row.get('recipe_name', row.get('Name', f'Recipe_{recipe_id}')),
                'ingredients': ingredients,
                'nutrition': {
                    'energy_kcal': float(row.get('Calories_PerServing_kcal', 0)) if pd.notna(row.get('Calories_PerServing_kcal')) else 0.0,
                    'protein_g': float(row.get('Protein_PerServing_g', 0)) if pd.notna(row.get('Protein_PerServing_g')) else 0.0,
                    'fat_g': float(row.get('Fat_PerServing_g', 0)) if pd.notna(row.get('Fat_PerServing_g')) else 0.0,
                    'carbohydrate_g': float(row.get('Carbohydrates_PerServing_g', 0)) if pd.notna(row.get('Carbohydrates_PerServing_g')) else 0.0,
                    'fiber_g': float(row.get('Fiber_PerServing_g', 0)) if pd.notna(row.get('Fiber_PerServing_g')) else 0.0,
                    'added_sugar_g': float(row.get('Sugars_PerServing_g', 0)) if pd.notna(row.get('Sugars_PerServing_g')) else 0.0,
                    'saturated_fat_g': float(row.get('SaturatedFat_PerServing_g', 0)) if pd.notna(row.get('SaturatedFat_PerServing_g')) else 0.0,
                    'sodium_mg': float(row.get('Sodium_PerServing_mg', 0)) if pd.notna(row.get('Sodium_PerServing_mg')) else 0.0,
                }
            }

        self.all_recipe_ids = list(self.recipe_dict.keys())
        print(f"    ✓ 有效食谱: {len(self.recipe_dict):,} 个")
        if skipped_count > 0:
            print(f"    ⚠ 跳过空食材: {skipped_count:,} 个")

    def _load_user_profiles(self, profile_path: str):
        """加载用户画像（RNI格式）"""
        print(f"\n[3/3] 加载用户画像: {profile_path}")

        self.users = []
        with open(profile_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.users.append(json.loads(line))

        print(f"  ✓ 加载用户: {len(self.users):,}")

        # 检查数据格式
        if self.users:
            sample_user = self.users[0]
            print(f"\n  数据格式检查:")
            print(f"    gender: {'✓' if 'gender' in sample_user else '✗'}")
            print(f"    age: {'✓' if 'age' in sample_user else '✗'}")
            print(f"    physiological_state: {'✓' if 'physiological_state' in sample_user else '✗'}")
            print(f"    nutrition_rni: {'✓' if 'nutrition_rni' in sample_user else '✗'}")

    # ==================== 打分函数（5个维度）====================

    def score_nutrition_match(self, recipe_nutrition: Dict, user_rni: Dict) -> float:
        """1. 营养RNI匹配度 (0-1)

        注意：仅使用食谱数据中可用的8个营养素进行评分
        缺失的微量元素（钾、钙、铁、维生素C/D、叶酸、反式脂肪）不参与评分
        """
        if not user_rni:
            return 0.5

        scores = []

        # ⚠️ 仅使用食谱数据中存在的8个营养素
        # 食谱缺失：trans_fat, potassium, calcium, iron, vitamin_c, vitamin_d, folate
        nutrient_configs = [
            ('energy_kcal', 'energy_kcal', 0.33, False),    # 单餐约1/3
            ('protein_g', 'protein_g', 0.33, False),
            ('carbohydrate_g', 'carbohydrate_g', 0.33, False),
            ('fat_g', 'fat_g', 0.33, False),
            ('fiber_g', 'fiber_g', 0.33, False),
            ('sodium_mg', 'sodium_mg', 0.30, True),         # 限制性：单餐≤30%
            ('added_sugar_g', 'added_sugar_g', 0.25, True),
            ('saturated_fat_g', 'saturated_fat_g', 0.30, True),
        ]

        for recipe_key, rni_key, target_ratio, is_restrictive in nutrient_configs:
            recipe_value = recipe_nutrition.get(recipe_key) or 0
            rni_value = user_rni.get(rni_key) or 0

            # Skip if either value is 0 (including None values converted to 0)
            if rni_value == 0 or recipe_value == 0:
                continue

            actual_ratio = recipe_value / rni_value

            if is_restrictive:
                # 限制性营养素：越低越好
                if actual_ratio <= target_ratio:
                    scores.append(1.0)
                elif actual_ratio <= target_ratio * 1.5:
                    scores.append(0.7)
                else:
                    scores.append(0.3)
            else:
                # 正向营养素：接近目标比例最好
                diff = abs(actual_ratio - target_ratio)
                if diff <= 0.1:
                    scores.append(1.0)
                elif diff <= 0.2:
                    scores.append(0.7)
                else:
                    scores.append(0.4)

        return np.mean(scores) if scores else 0.5

    def score_preference_match(self, recipe_ingredients: List[str],
                               liked: List[str], disliked: List[str]) -> float:
        """2. 食材偏好匹配度 (0-1)"""
        if not liked and not disliked:
            return 0.5

        recipe_set = set(recipe_ingredients)
        disliked_count = len(recipe_set & set(disliked))
        liked_count = len(recipe_set & set(liked))

        # 含有不喜欢的食材：严重惩罚
        if disliked_count > 0:
            return 0.1

        # 含有喜欢的食材：奖励
        if liked_count > 0:
            return min(0.7 + liked_count * 0.1, 1.0)

        return 0.5

    def score_cooccurrence(self, ingredients: List[str]) -> float:
        """3. 食材共现分数 (0-1)"""
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
        """4. 营养互补分数 (0-1)"""
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

    def score_balance(self, ingredients: List[str]) -> float:
        """5. 营养平衡分数 (0-1)"""
        all_tags = []
        for ing in ingredients:
            all_tags.extend(self.ingredient_tags.get(ing, []))

        if not all_tags:
            return 0.5

        # 多样性评分（熵）
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

        # 正向标签比例
        positive_tags = ['high_protein', 'high_fiber', 'high_vitamin_c',
                        'high_calcium', 'high_iron', 'low_sodium', 'low_fat']
        positive_count = sum(1 for tag in all_tags if tag in positive_tags)
        positive_ratio = positive_count / len(all_tags) if all_tags else 0.0

        return 0.5 * diversity + 0.5 * positive_ratio

    def score_recipe(self, recipe_id: str, user: Dict) -> Tuple[float, Dict]:
        """综合打分（5维度）"""
        weights = {
            'nutrition_match': 0.35,
            'preference_match': 0.25,
            'cooccurrence': 0.15,
            'complementarity': 0.15,
            'balance': 0.10
        }

        recipe = self.recipe_dict.get(recipe_id)
        if not recipe:
            return 0.0, {}

        liked_ings = [item['name'] for item in user.get('liked_ingredients', [])]
        disliked_ings = [item['name'] for item in user.get('disliked_ingredients', [])]
        nutrition_rni = user.get('nutrition_rni', {})

        # 计算5个维度分数
        nutrition_score = self.score_nutrition_match(recipe['nutrition'], nutrition_rni)
        preference_score = self.score_preference_match(recipe['ingredients'], liked_ings, disliked_ings)
        cooccurrence_score = self.score_cooccurrence(recipe['ingredients'])
        complementarity_score = self.score_complementarity(recipe['ingredients'])
        balance_score = self.score_balance(recipe['ingredients'])

        # 加权总分
        total_score = (
            weights['nutrition_match'] * nutrition_score +
            weights['preference_match'] * preference_score +
            weights['cooccurrence'] * cooccurrence_score +
            weights['complementarity'] * complementarity_score +
            weights['balance'] * balance_score
        )

        breakdown = {
            'nutrition_match': round(nutrition_score, 3),
            'preference_match': round(preference_score, 3),
            'cooccurrence': round(cooccurrence_score, 3),
            'complementarity': round(complementarity_score, 3),
            'balance': round(balance_score, 3)
        }

        return round(total_score, 3), breakdown

    # ==================== 推理生成 ====================

    def generate_reasoning(self, recipe: Dict, user: Dict, breakdown: Dict) -> str:
        """生成推荐理由（模拟LLM的reasoning）"""
        reasons = []

        # 营养匹配
        if breakdown['nutrition_match'] >= 0.8:
            reasons.append("excellent nutritional alignment with your RNI targets")
        elif breakdown['nutrition_match'] >= 0.6:
            reasons.append("good nutritional fit for your requirements")

        # 偏好匹配
        if breakdown['preference_match'] >= 0.8:
            liked_ings = [item['name'] for item in user.get('liked_ingredients', [])]
            liked_in_recipe = [ing for ing in recipe['ingredients'] if ing in liked_ings]
            if liked_in_recipe:
                reasons.append(f"contains your preferred ingredients ({', '.join(liked_in_recipe[:2])})")

        # 健康状况
        physio_state = user.get('physiological_state', 'healthy')
        if physio_state == 'diabetes' and breakdown['nutrition_match'] >= 0.7:
            reasons.append("suitable for diabetes management with controlled carbohydrate content")
        elif physio_state == 'hypertension' and breakdown['nutrition_match'] >= 0.7:
            reasons.append("low sodium content appropriate for hypertension control")

        # 共现和互补
        if breakdown['cooccurrence'] >= 0.7 and breakdown['complementarity'] >= 0.7:
            reasons.append("high ingredient synergy and nutritional complementarity")

        # 营养平衡
        if breakdown['balance'] >= 0.8:
            reasons.append("well-balanced nutritional profile")

        if not reasons:
            reasons.append("meets basic nutritional requirements")

        return "; ".join(reasons).capitalize()

    # ==================== 指令生成 ====================

    def generate_instruction(self, user: Dict, template_idx: int = None) -> Dict:
        """生成指令"""
        if template_idx is None:
            template_idx = random.randint(0, len(INSTRUCTION_TEMPLATES) - 1)

        template_info = INSTRUCTION_TEMPLATES[template_idx]
        template = template_info['template']

        # 提取用户信息
        gender = user.get('gender', 'unknown')
        age = user.get('age', 0)
        physiological_state = user.get('physiological_state', 'healthy').replace('_', ' ')

        liked_items = user.get('liked_ingredients', [])
        disliked_items = user.get('disliked_ingredients', [])

        liked_sample = ", ".join([item['name'] for item in liked_items[:3]]) if liked_items else "vegetables"
        disliked_sample = ", ".join([item['name'] for item in disliked_items[:3]]) if disliked_items else "none"

        nutrition_rni = user.get('nutrition_rni', {})
        # Fix: Handle None values by using 'or' operator
        energy_kcal = round(nutrition_rni.get('energy_kcal') or 2000)
        protein_g = round(nutrition_rni.get('protein_g') or 50)
        fiber_g = round(nutrition_rni.get('fiber_g') or 25)
        carb_g = round(nutrition_rni.get('carbohydrate_g') or 300)
        fat_g = round(nutrition_rni.get('fat_g') or 65)

        # 特定营养素（仅使用食谱数据中存在的营养素）
        key_nutrients = [
            ('protein', protein_g, 'g'),
            ('fiber', fiber_g, 'g'),
            ('carbohydrates', carb_g, 'g'),
            ('healthy fats', fat_g, 'g'),
        ]
        key_nutrient, nutrient_value, nutrient_unit = random.choice(key_nutrients)

        # 限制性营养素
        restricted_nutrient = "sodium"
        limit_value = round(nutrition_rni.get('sodium_mg') or 1500)
        limit_unit = "mg"

        target_percentage = random.choice([30, 33, 35])

        # 填充模板
        instruction = template.format(
            gender=gender,
            age=age,
            physiological_state=physiological_state,
            liked_ingredients=liked_sample,
            disliked_ingredients=disliked_sample,
            energy_kcal=energy_kcal,
            protein_g=protein_g,
            fiber_g=fiber_g,
            carb_g=carb_g,
            fat_g=fat_g,
            key_nutrient=key_nutrient,
            nutrient_value=nutrient_value,
            nutrient_unit=nutrient_unit,
            restricted_nutrient=restricted_nutrient,
            limit_value=limit_value,
            limit_unit=limit_unit,
            target_percentage=target_percentage
        )

        return {
            'instruction': instruction,
            'template_type': template_info['type']
        }

    # ==================== 样本生成 ====================

    def generate_sample(self, user: Dict) -> Dict:
        """为单个用户生成Task A样本（Discriminative Ranking）"""
        user_id = user['user_id']

        # 随机采样1500候选
        sampled_ids = random.sample(self.all_recipe_ids, min(1500, len(self.all_recipe_ids)))

        # 打分并排序
        scored_recipes = []
        for recipe_id in sampled_ids:
            score, breakdown = self.score_recipe(recipe_id, user)
            scored_recipes.append((recipe_id, score, breakdown))

        scored_recipes.sort(key=lambda x: x[1], reverse=True)

        # 获取Top-3
        top3 = scored_recipes[:3]

        # 生成指令
        instruction_info = self.generate_instruction(user)

        # 构建Top-3排序结果
        ranked_recipes = []
        for rank, (recipe_id, score, breakdown) in enumerate(top3, 1):
            recipe = self.recipe_dict[recipe_id]
            reasoning = self.generate_reasoning(recipe, user, breakdown)

            ranked_recipes.append({
                'rank': rank,
                'recipe_id': recipe_id,
                'recipe_name': recipe['name'],
                'overall_score': score,
                'score_breakdown': breakdown,
                'reasoning': reasoning,
                'ingredients': recipe['ingredients'],
                'nutrition_per_serving': recipe['nutrition']
            })

        # 构建完整样本
        sample = {
            'user_id': user_id,
            'instruction': instruction_info['instruction'],
            'instruction_type': instruction_info['template_type'],
            'user_profile': {
                'gender': user.get('gender', ''),
                'age': user.get('age', 0),
                'physiological_state': user.get('physiological_state', ''),
                'nutrition_rni': user.get('nutrition_rni', {}),
                'liked_ingredients_count': len(user.get('liked_ingredients', [])),
                'disliked_ingredients_count': len(user.get('disliked_ingredients', []))
            },
            'ranked_recipes': ranked_recipes
        }

        return sample

    def build_dataset(self, users_list: List[Dict], output_path: str):
        """构建数据集"""
        all_samples = []
        for user in tqdm(users_list, desc=f"生成Task A样本"):
            sample = self.generate_sample(user)
            all_samples.append(sample)

        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"✓ {output_path}: {len(all_samples):,} 样本")

        # 统计
        avg_top1_score = np.mean([s['ranked_recipes'][0]['overall_score'] for s in all_samples])
        print(f"  平均Top-1分数: {avg_top1_score:.3f}")

        # 指令类型分布
        type_counts = defaultdict(int)
        for s in all_samples:
            type_counts[s['instruction_type']] += 1
        print(f"  指令类型分布:")
        for itype, count in sorted(type_counts.items()):
            print(f"    {itype}: {count}")

        return len(all_samples)


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # ========================================================================
    # 构建数据集
    # ========================================================================
    builder = TaskADatasetBuilder(
        kg_path="work/recipebench/kg/nutriplan_kg4.graphml",
        recipe_basic_path="work/recipebench/data/raw/foodcom/recipes(3column).csv",
        recipe_nutrition_path="work/recipebench/data/4out/recipe_nutrition_foodcom.csv",
        user_profile_path="work/recipebench/data/8step_profile/update_cleaned_user_profile.jsonl"
    )

    # 随机打乱用户
    random.shuffle(builder.users)

    # 划分数据集
    train_users = builder.users[:10000]
    val_users = builder.users[10000:12000]
    test_users = builder.users[12000:14000]

    print(f"\n{'='*80}")
    print("数据集配置")
    print(f"{'='*80}")
    print(f"训练集: {len(train_users):,} 用户")
    print(f"验证集: {len(val_users):,} 用户")
    print(f"测试集: {len(test_users):,} 用户")
    print(f"每用户: 1样本 (instruction + Top-3 ranked recipes)")

    # 生成训练集
    print(f"\n{'='*80}")
    print("生成训练集")
    print(f"{'='*80}")
    builder.build_dataset(train_users, "work/recipebench/data/10large_scale_datasets/task_a_train_discriminative.jsonl")

    # 生成验证集
    print(f"\n{'='*80}")
    print("生成验证集")
    print(f"{'='*80}")
    builder.build_dataset(val_users, "work/recipebench/data/10large_scale_datasets/task_a_val_discriminative.jsonl")

    # 生成测试集
    print(f"\n{'='*80}")
    print("生成测试集")
    print(f"{'='*80}")
    builder.build_dataset(test_users, "work/recipebench/data/10large_scale_datasets/task_a_test_discriminative.jsonl")

    print(f"\n{'='*80}")
    print("🎉 Task A 数据集构建完成！")
    print(f"{'='*80}")
    print("\n📊 数据集总结:")
    print("  任务: Discriminative Ranking (判别式排序)")
    print("  训练目标: 学习评估recipe suitability并进行鲁棒排序")
    print("  输出格式: instruction + user_profile + ranked_recipes (Top-3)")
    print("  评分维度: 5维 (nutrition, preference, cooccurrence, complementarity, balance)")
    print("  指令模板: 10种场景覆盖")
    print("\n输出文件:")
    print("  - task_a_train_discriminative.jsonl  (10,000样本)")
    print("  - task_a_val_discriminative.jsonl    (2,000样本)")
    print("  - task_a_test_discriminative.jsonl   (2,000样本)")
