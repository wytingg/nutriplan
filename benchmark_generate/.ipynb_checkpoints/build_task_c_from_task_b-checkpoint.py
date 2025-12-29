#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task C Dataset Generator - Integrated with Task B (RNI Format)
创建日期：2025-10-26
状态：全新版本，与Task A/B完美衔接

任务定位：
- Task A (Discriminative Ranking): 判别式排序 - 从KG检索的候选食谱中评分和排序
- Task B (Constrained Generation): 约束生成 - 从零生成新食谱，满足用户约束
- Task C (Reflective Editing): 反思性编辑 - 自我批评和修正，诊断约束违规并执行最小化编辑

核心改进：
✅ 使用新的Task B数据源（task_b_*_from_kg.jsonl）
✅ 适配RNI格式（15个营养素直接值，而非AMDR百分比）
✅ 添加10种Task C专属指令模板（强调诊断和修正）
✅ 统一输出格式（instruction + instruction_type + user_profile）
✅ 保持8种营养违规类型完整支持
"""

import json
import random
import copy
from pathlib import Path
from calculate_recipe_nutrition import RecipeNutritionCalculator
from ingredient_parser import parse_ingredient_string, compose_ingredient_string
from tqdm import tqdm

print("="*80)
print("Task C Dataset Generator - Integrated Version")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

NUTRITION_DB = 'work/recipebench/data/11_nutrition_rule/top500_nutrition_complete.csv'

# 新的Task B数据源（RNI格式）
B_CLASS_TRAIN = 'work/recipebench/data/10large_scale_datasets/task_b_train_from_kg.jsonl'
B_CLASS_VAL = 'work/recipebench/data/10large_scale_datasets/task_b_val_from_kg.jsonl'
B_CLASS_TEST = 'work/recipebench/data/10large_scale_datasets/task_b_test_from_kg.jsonl'

OUTPUT_DIR = 'work/recipebench/data/10large_scale_datasets/'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Task C Instruction Templates (10 types)
# ============================================================================

INSTRUCTION_TEMPLATES_TASK_C = [
    {
        "template": "This recipe draft violates my nutritional constraints. As a {age}-year-old {gender} with {physiological_state}, I need you to identify the specific violations and correct them with minimal changes to preserve the recipe's quality.",
        "type": "diagnostic_correction"
    },
    {
        "template": "Please review this recipe for compliance with my requirements: {constraint_sample}. Identify any violations and provide corrected ingredient quantities that meet my targets.",
        "type": "compliance_review"
    },
    {
        "template": "I generated this recipe but it doesn't meet my dietary needs ({physiological_state}). Can you diagnose what's wrong and fix it with the smallest possible edits?",
        "type": "minimal_editing"
    },
    {
        "template": "As a {age}-year-old {gender}, I need this recipe checked against my RNI standards: {constraint_sample}. Point out violations and correct them while keeping the recipe as close to the original as possible.",
        "type": "constraint_checking"
    },
    {
        "template": "This recipe draft has nutritional issues. I need {constraint_sample}. Identify what's wrong and modify only the necessary ingredients to fix it.",
        "type": "targeted_fixing"
    },
    {
        "template": "I'm managing {physiological_state} and this recipe doesn't comply with my dietary restrictions. Please diagnose the violations and apply precision corrections.",
        "type": "precision_correction"
    },
    {
        "template": "Check this recipe against my profile: {age}yo {gender}, {physiological_state}. If there are violations of {constraint_sample}, correct them with minimal recipe changes.",
        "type": "profile_based_audit"
    },
    {
        "template": "This recipe needs refinement to match my nutritional targets: {constraint_sample}. Identify deviations and adjust ingredients precisely to meet requirements.",
        "type": "refinement_request"
    },
    {
        "template": "As someone with {physiological_state}, I need you to audit this recipe for constraint violations and fix them. My requirements: {constraint_sample}. Make the smallest edits possible.",
        "type": "audit_and_fix"
    },
    {
        "template": "Review this recipe draft for a {age}-year-old {gender} with {physiological_state}. Diagnose any violations of {constraint_sample} and provide corrected quantities.",
        "type": "diagnostic_review"
    }
]


def select_instruction_template_c(user_profile, seed=0):
    """选择Task C指令模板并填充用户信息"""
    random.seed(seed)

    template_info = random.choice(INSTRUCTION_TEMPLATES_TASK_C)
    template = template_info["template"]
    instruction_type = template_info["type"]

    # 提取用户信息
    age = user_profile.get('age', 30)
    gender = user_profile.get('gender', 'adult')
    physiological_state = user_profile.get('physiological_state', 'healthy')

    # 从RNI中选择2-3个约束作为示例
    rni = user_profile.get('nutrition_rni', {})
    constraint_samples = []

    # 优先选择关键营养素
    key_nutrients = [
        ('energy_kcal', f"{rni.get('energy_kcal', 2000)} kcal energy"),
        ('protein_g', f"{rni.get('protein_g', 60)}g protein"),
        ('sodium_mg', f"max {rni.get('sodium_mg', 2300)}mg sodium"),
        ('fiber_g', f"min {rni.get('fiber_g', 25)}g fiber"),
    ]

    # 随机选择2-3个
    selected = random.sample(key_nutrients, k=min(3, len(key_nutrients)))
    constraint_samples = [desc for _, desc in selected]
    constraint_sample = ", ".join(constraint_samples)

    # 填充模板
    instruction = template.format(
        age=age,
        gender=gender,
        physiological_state=physiological_state,
        constraint_sample=constraint_sample
    )

    return instruction, instruction_type


# ============================================================================
# RNI-to-Constraints Conversion
# ============================================================================

def extract_constraints_from_rni(user_profile):
    """
    从新的RNI格式提取约束信息

    输入: user_profile (新格式)
    {
        'nutrition_rni': {
            'energy_kcal': 2000,
            'protein_g': 60,
            'fat_g': 65,
            ...
        },
        'liked_ingredients': [...],
        'disliked_ingredients': [...]
    }

    输出: 兼容旧版violation注入函数的约束格式
    """
    rni = user_profile.get('nutrition_rni', {})

    # 提取直接值
    energy_kcal = rni.get('energy_kcal', 2000)
    protein_g = rni.get('protein_g', 60)
    fat_g = rni.get('fat_g', 65)
    carb_g = rni.get('carbohydrate_g', 260)
    fiber_g = rni.get('fiber_g', 25)
    sodium_mg = rni.get('sodium_mg', 2300)
    saturated_fat_g = rni.get('saturated_fat_g', 20)
    added_sugar_g = rni.get('added_sugar_g', 25)

    # 计算AMDR百分比（用于AMDR相关的violation检测）
    total_g = protein_g + fat_g + carb_g
    if total_g > 0:
        protein_pct = (protein_g * 4 / energy_kcal * 100) if energy_kcal > 0 else 20
        fat_pct = (fat_g * 9 / energy_kcal * 100) if energy_kcal > 0 else 30
        carb_pct = (carb_g * 4 / energy_kcal * 100) if energy_kcal > 0 else 50
    else:
        protein_pct, fat_pct, carb_pct = 20, 30, 50

    # 构建兼容旧版格式的约束字典
    constraints = {
        # 能量目标
        'energy_kcal_target': energy_kcal,

        # AMDR格式（用于百分比违规检测）
        'amdr': {
            'protein': {
                'target_pct': protein_pct,
                'min_pct': max(10, protein_pct - 5),
                'max_pct': min(35, protein_pct + 5)
            },
            'fat': {
                'target_pct': fat_pct,
                'min_pct': max(20, fat_pct - 5),
                'max_pct': min(35, fat_pct + 5)
            },
            'carb': {
                'target_pct': carb_pct,
                'min_pct': max(45, carb_pct - 5),
                'max_pct': min(65, carb_pct + 5)
            }
        },

        # 绝对值限制
        'protein_g_target': protein_g,
        'fat_g_target': fat_g,
        'carbohydrate_g_target': carb_g,
        'fiber_g_min': fiber_g,
        'sodium_mg_max': sodium_mg,
        'saturated_fat_g_max': saturated_fat_g,
        'added_sugar_g_max': added_sugar_g,

        # 糖分百分比限制（added sugar应<10% energy）
        'sugars': {
            'pct_max': 10  # FDA推荐：added sugar < 10% total energy
        },

        # 偏好约束
        'liked_ingredients': user_profile.get('liked_ingredients', []),
        'disliked_ingredients': user_profile.get('disliked_ingredients', [])
    }

    return constraints


# ============================================================================
# Load Calculator and Setup
# ============================================================================

print("\n[1/6] Loading nutrition calculator...")
calc = RecipeNutritionCalculator(NUTRITION_DB)
print(f"  [OK] Loaded {len(calc.nutrition_lookup)} ingredients")

# Build available ingredients pool
AVAILABLE_INGREDIENTS = set(calc.nutrition_lookup.keys())

# Food classification
PROTEIN_KEYWORDS = ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'tofu', 'egg', 'turkey']
CARB_KEYWORDS = ['rice', 'pasta', 'bread', 'potato', 'oat', 'quinoa', 'corn', 'noodle']
VEGETABLE_KEYWORDS = ['broccoli', 'spinach', 'kale', 'carrot', 'tomato', 'pepper', 'onion']
HIGH_FAT_KEYWORDS = ['oil', 'butter', 'cheese', 'nuts', 'avocado']
HIGH_SUGAR_KEYWORDS = ['honey', 'syrup', 'sugar', 'jam']


# ============================================================================
# Ingredient Utilities (from original code)
# ============================================================================

def is_ingredient_available(ing_name):
    """Check if ingredient has nutrition data"""
    ing_lower = ing_name.lower()
    if ing_lower in AVAILABLE_INGREDIENTS:
        return True
    for avail_ing in AVAILABLE_INGREDIENTS:
        if avail_ing in ing_lower or ing_lower in avail_ing:
            return True
    return False


def find_best_match_ingredient(ing_name):
    """Find best matching ingredient in nutrition database"""
    ing_lower = ing_name.lower()
    if ing_lower in AVAILABLE_INGREDIENTS:
        return ing_lower
    for avail_ing in AVAILABLE_INGREDIENTS:
        if avail_ing in ing_lower:
            return avail_ing
    for avail_ing in AVAILABLE_INGREDIENTS:
        if ing_lower in avail_ing:
            return avail_ing
    return None


def parse_recipe_ingredients(ingredients_list):
    """Parse recipe ingredients to structured format"""
    parsed = []
    for ing_str in ingredients_list:
        result = parse_ingredient_string(ing_str)
        if result:
            qty, name = result
            parsed.append((qty, name, ing_str))
        else:
            parsed.append((100.0, ing_str, ing_str))
    return parsed


def recalculate_nutrition(parsed_ingredients, servings=4):
    """Recalculate nutrition from parsed ingredients"""
    ingredient_strings = []
    for qty_grams, ing_name, _ in parsed_ingredients:
        matched_ing = find_best_match_ingredient(ing_name)
        if matched_ing:
            ingredient_strings.append(f"{qty_grams}g {matched_ing}")
    if not ingredient_strings:
        return None
    try:
        nutrition_result = calc.calculate_recipe_nutrition(ingredient_strings, servings)
        return nutrition_result
    except Exception as e:
        print(f"  [WARNING] Nutrition calculation failed: {e}")
        return None


# ============================================================================
# Violation Injection Module (8 NUTRITION TYPES) - From original code
# ============================================================================

def inject_sodium_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """钠超标（增加salt）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    salt_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if 'salt' in name.lower():
            salt_idx = i
            break

    if salt_idx is None:
        modified.append((2.0, 'salt', '1/2 tsp salt'))
        salt_idx = len(modified) - 1

    original_qty, name, _ = modified[salt_idx]
    new_qty = original_qty * random.uniform(2.5, 3.5)
    modified[salt_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        sodium_per_serv = new_nutrition['per_serving']['sodium_mg']
        sodium_total = sodium_per_serv * servings
        sodium_max = targets.get('sodium_mg_max', 2300)

        if sodium_total > sodium_max:
            violations.append({
                'type': 'nutrition_violation',
                'field': 'sodium_mg',
                'actual': sodium_total,
                'limit': sodium_max,
                'severity': 'critical' if sodium_total > sodium_max * 1.5 else 'major'
            })
            return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_protein_low_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """蛋白质不足（减少protein食材）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    protein_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if any(p in name.lower() for p in PROTEIN_KEYWORDS):
            protein_idx = i
            break

    if protein_idx is None:
        modified.append((200.0, 'chicken breast', '200g chicken breast'))
        protein_idx = len(modified) - 1

    original_qty, name, _ = modified[protein_idx]
    new_qty = original_qty * random.uniform(0.3, 0.5)
    modified[protein_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        protein_per_serv = new_nutrition['per_serving']['protein_g']
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        protein_kcal = protein_per_serv * 4
        if energy_per_serv > 0:
            protein_pct = (protein_kcal / energy_per_serv) * 100
            target_protein_pct = targets.get('amdr', {}).get('protein', {}).get('target_pct', 20)

            if protein_pct < target_protein_pct * 0.8:
                violations.append({
                    'type': 'nutrition_violation',
                    'field': 'protein_amdr',
                    'actual_pct': protein_pct,
                    'target_pct': target_protein_pct,
                    'severity': 'major'
                })
                return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_fat_high_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """脂肪AMDR比例过高（增加oil/butter）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    fat_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if any(f in name.lower() for f in HIGH_FAT_KEYWORDS):
            fat_idx = i
            break

    if fat_idx is None:
        modified.append((10.0, 'olive oil', '2 tsps olive oil'))
        fat_idx = len(modified) - 1

    original_qty, name, _ = modified[fat_idx]
    new_qty = original_qty * random.uniform(2.5, 4.0)
    modified[fat_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        fat_per_serv = new_nutrition['per_serving']['fat_g']
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        fat_kcal = fat_per_serv * 9
        if energy_per_serv > 0:
            fat_pct = (fat_kcal / energy_per_serv) * 100
            target_fat_pct = targets.get('amdr', {}).get('fat', {}).get('target_pct', 30)

            if fat_pct > target_fat_pct * 1.25:
                violations.append({
                    'type': 'nutrition_violation',
                    'field': 'fat_amdr',
                    'actual_pct': fat_pct,
                    'target_pct': target_fat_pct,
                    'severity': 'major'
                })
                return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_carb_low_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """碳水AMDR比例过低（减少carb食材）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    carb_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if any(c in name.lower() for c in CARB_KEYWORDS):
            carb_idx = i
            break

    if carb_idx is None:
        modified.append((370.0, 'rice', '2 cups rice'))
        carb_idx = len(modified) - 1

    original_qty, name, _ = modified[carb_idx]
    new_qty = original_qty * random.uniform(0.3, 0.5)
    modified[carb_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        carb_per_serv = new_nutrition['per_serving']['carbohydrates_g']
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        carb_kcal = carb_per_serv * 4
        if energy_per_serv > 0:
            carb_pct = (carb_kcal / energy_per_serv) * 100
            target_carb_pct = targets.get('amdr', {}).get('carb', {}).get('target_pct', 50)

            if carb_pct < target_carb_pct * 0.75:
                violations.append({
                    'type': 'nutrition_violation',
                    'field': 'carb_amdr',
                    'actual_pct': carb_pct,
                    'target_pct': target_carb_pct,
                    'severity': 'major'
                })
                return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_energy_high_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """能量超标（增加oil/carb）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    target_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if 'oil' in name.lower() or any(c in name.lower() for c in CARB_KEYWORDS):
            target_idx = i
            break

    if target_idx is None:
        modified.append((40.0, 'olive oil', '3 Tbsps olive oil'))
        target_idx = len(modified) - 1

    original_qty, name, _ = modified[target_idx]
    new_qty = original_qty * random.uniform(1.4, 1.7)
    modified[target_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']
        target_energy = targets.get('energy_kcal_target', 2000) / servings

        if energy_per_serv > target_energy * 1.25:
            violations.append({
                'type': 'nutrition_violation',
                'field': 'energy_kcal',
                'actual': energy_per_serv,
                'target': target_energy,
                'severity': 'major'
            })
            return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_fiber_low_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """纤维不足（减少vegetable）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    veg_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if any(v in name.lower() for v in VEGETABLE_KEYWORDS):
            veg_idx = i
            break

    if veg_idx is None:
        modified.append((200.0, 'broccoli', '200g broccoli'))
        veg_idx = len(modified) - 1

    original_qty, name, _ = modified[veg_idx]
    new_qty = original_qty * random.uniform(0.2, 0.4)
    modified[veg_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        fiber_per_serv = new_nutrition['per_serving']['fiber_g']
        fiber_total = fiber_per_serv * servings
        fiber_min = targets.get('fiber_g_min', 25)

        if fiber_total < fiber_min * 0.8:
            violations.append({
                'type': 'nutrition_violation',
                'field': 'fiber_g',
                'actual': fiber_total,
                'minimum': fiber_min,
                'severity': 'major'
            })
            return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_saturated_fat_high_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """饱和脂肪超标（增加butter/cheese等）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    sat_fat_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if 'butter' in name.lower() or 'cheese' in name.lower():
            sat_fat_idx = i
            break

    if sat_fat_idx is None:
        modified.append((15.0, 'butter', '1 Tbsp butter'))
        sat_fat_idx = len(modified) - 1

    original_qty, name, _ = modified[sat_fat_idx]
    new_qty = original_qty * random.uniform(2.5, 3.5)
    modified[sat_fat_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        sat_fat_per_serv = new_nutrition['per_serving'].get('saturated_fat_g') or \
                           new_nutrition['per_serving'].get('saturated_fat', 0)
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        sat_fat_kcal = sat_fat_per_serv * 9
        if energy_per_serv > 0:
            sat_fat_pct = (sat_fat_kcal / energy_per_serv) * 100

            if sat_fat_pct > 12:
                violations.append({
                    'type': 'nutrition_violation',
                    'field': 'saturated_fat_g',
                    'actual_pct': sat_fat_pct,
                    'limit_pct': 10,
                    'severity': 'major'
                })
                return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_sugars_high_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """糖分超标（增加honey/syrup等）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    sugar_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if any(s in name.lower() for s in HIGH_SUGAR_KEYWORDS):
            sugar_idx = i
            break

    if sugar_idx is None:
        modified.append((20.0, 'honey', '1 Tbsp honey'))
        sugar_idx = len(modified) - 1

    original_qty, name, _ = modified[sugar_idx]
    new_qty = original_qty * random.uniform(2.0, 3.0)
    modified[sugar_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        sugars_per_serv = new_nutrition['per_serving'].get('sugars_g') or \
                          new_nutrition['per_serving'].get('sugars', 0)
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        sugars_kcal = sugars_per_serv * 4
        if energy_per_serv > 0:
            sugars_pct = (sugars_kcal / energy_per_serv) * 100
            limit_pct = targets.get('sugars', {}).get('pct_max', 10)

            if sugars_pct > limit_pct * 1.5:
                violations.append({
                    'type': 'nutrition_violation',
                    'field': 'sugars_g',
                    'actual_pct': sugars_pct,
                    'limit_pct': limit_pct,
                    'severity': 'major'
                })
                return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_preference_violation(parsed_ingredients, liked_ingredients, disliked_ingredients):
    """偏好违规（添加disliked或删除liked）"""
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    if disliked_ingredients and random.random() < 0.6:
        disliked_names = [ing.lower() if isinstance(ing, str) else ing.get('name', '').lower()
                         for ing in disliked_ingredients]
        available_disliked = []
        for d_name in disliked_names:
            if is_ingredient_available(d_name):
                available_disliked.append(d_name)

        if available_disliked:
            bad_ing = random.choice(available_disliked)
            matched_ing = find_best_match_ingredient(bad_ing)

            if matched_ing:
                qty = random.uniform(50, 100)
                modified.append((qty, matched_ing, compose_ingredient_string(qty, matched_ing)))
                violations.append({
                    'type': 'preference_violation',
                    'subtype': 'disliked_ingredient_added',
                    'ingredient': matched_ing,
                    'severity': 'critical'
                })

    elif liked_ingredients and random.random() < 0.4:
        liked_names = [ing.lower() if isinstance(ing, str) else ing.get('name', '').lower()
                      for ing in liked_ingredients]
        for i, (qty, name, _) in enumerate(modified):
            for liked_name in liked_names:
                if liked_name in name.lower() or name.lower() in liked_name:
                    modified.pop(i)
                    violations.append({
                        'type': 'preference_violation',
                        'subtype': 'liked_ingredient_removed',
                        'ingredient': name,
                        'severity': 'major'
                    })
                    return modified, violations

    return modified, violations


def generate_violated_recipe_v2(b_class_recipe):
    """
    从B-class合法食谱生成违规初稿（适配新RNI格式）

    违规类型分布：
    - 60%: 单一营养违规（8种类型）
    - 25%: 偏好违规
    - 15%: 营养+偏好双重违规
    """
    original_recipe = b_class_recipe['output']
    user_profile = b_class_recipe['user_profile']

    # 从新格式提取约束
    targets = extract_constraints_from_rni(user_profile)
    liked = user_profile.get('liked_ingredients', [])
    disliked = user_profile.get('disliked_ingredients', [])

    parsed_ingredients = parse_recipe_ingredients(original_recipe['ingredients'])
    original_nutrition = original_recipe.get('nutrition_per_serving', {})

    rand = random.random()

    if rand < 0.6:
        # Single nutrition violation (8 types) - with fallback mechanism
        violation_types = [
            'sodium', 'protein_low', 'fat_high', 'carb_low',
            'energy_high', 'fiber_low', 'saturated_fat_high', 'sugars_high'
        ]
        random.shuffle(violation_types)

        modified_ings, violations, new_nutrition = None, [], None

        for violation_type in violation_types:
            if violation_type == 'sodium':
                modified_ings, violations, new_nutrition = inject_sodium_violation(
                    parsed_ingredients, original_nutrition, targets)
            elif violation_type == 'protein_low':
                modified_ings, violations, new_nutrition = inject_protein_low_violation(
                    parsed_ingredients, original_nutrition, targets)
            elif violation_type == 'fat_high':
                modified_ings, violations, new_nutrition = inject_fat_high_violation(
                    parsed_ingredients, original_nutrition, targets)
            elif violation_type == 'carb_low':
                modified_ings, violations, new_nutrition = inject_carb_low_violation(
                    parsed_ingredients, original_nutrition, targets)
            elif violation_type == 'energy_high':
                modified_ings, violations, new_nutrition = inject_energy_high_violation(
                    parsed_ingredients, original_nutrition, targets)
            elif violation_type == 'fiber_low':
                modified_ings, violations, new_nutrition = inject_fiber_low_violation(
                    parsed_ingredients, original_nutrition, targets)
            elif violation_type == 'saturated_fat_high':
                modified_ings, violations, new_nutrition = inject_saturated_fat_high_violation(
                    parsed_ingredients, original_nutrition, targets)
            else:  # sugars_high
                modified_ings, violations, new_nutrition = inject_sugars_high_violation(
                    parsed_ingredients, original_nutrition, targets)

            if violations:
                break

    elif rand < 0.85:
        # Preference violation only
        modified_ings, violations = inject_preference_violation(
            parsed_ingredients, liked, disliked)
        new_nutrition = recalculate_nutrition(modified_ings)

    else:
        # Combined: nutrition + preference
        violation_type = random.choice(['sodium', 'energy_high', 'fat_high'])

        if violation_type == 'sodium':
            modified_ings, violations1, new_nutrition = inject_sodium_violation(
                parsed_ingredients, original_nutrition, targets)
        elif violation_type == 'energy_high':
            modified_ings, violations1, new_nutrition = inject_energy_high_violation(
                parsed_ingredients, original_nutrition, targets)
        else:
            modified_ings, violations1, new_nutrition = inject_fat_high_violation(
                parsed_ingredients, original_nutrition, targets)

        modified_ings, violations2 = inject_preference_violation(
            modified_ings, liked, disliked)
        violations = violations1 + violations2
        new_nutrition = recalculate_nutrition(modified_ings)

    if not violations:
        return None

    violated_ingredients = [original_str for _, _, original_str in modified_ings]

    violated_recipe = copy.deepcopy(original_recipe)
    violated_recipe['ingredients'] = violated_ingredients

    if new_nutrition:
        violated_recipe['nutrition_per_serving'] = new_nutrition['per_serving']

    return {
        'violated_recipe': violated_recipe,
        'violations': violations,
        'parsed_ingredients': modified_ings,
        'nutrition_result': new_nutrition
    }


# ============================================================================
# Correction Strategy Module (FOR 8 NUTRITION TYPES) - From original code
# ============================================================================

def generate_correction_for_sodium(violation, parsed_ingredients, targets):
    """钠超标修正：减少salt"""
    corrections = []
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if 'salt' in name.lower():
            actual_sodium = violation['actual']
            limit_sodium = violation['limit']
            reduction_needed = (actual_sodium - limit_sodium) / actual_sodium
            reduction_factor = max(0.3, 1 - reduction_needed * 1.2)

            corrections.append({
                'action': 'reduce_quantity',
                'ingredient_index': i,
                'ingredient_name': name,
                'original_quantity': qty,
                'new_quantity': qty * reduction_factor,
                'reduction_factor': reduction_factor,
                'reason': 'reduce_sodium_to_meet_limit'
            })
            break
    return corrections


def generate_correction_for_protein_low(violation, parsed_ingredients, targets):
    """蛋白质不足修正：增加protein食材"""
    corrections = []
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if any(p in name.lower() for p in PROTEIN_KEYWORDS):
            increase_factor = random.uniform(1.4, 1.6)
            corrections.append({
                'action': 'increase_quantity',
                'ingredient_index': i,
                'ingredient_name': name,
                'original_quantity': qty,
                'new_quantity': qty * increase_factor,
                'increase_factor': increase_factor,
                'reason': 'increase_protein_to_meet_amdr'
            })
            break
    return corrections


def generate_correction_for_fat_high(violation, parsed_ingredients, targets):
    """脂肪过高修正：减少oil/butter"""
    corrections = []
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if any(f in name.lower() for f in HIGH_FAT_KEYWORDS):
            reduction_factor = random.uniform(0.4, 0.6)
            corrections.append({
                'action': 'reduce_quantity',
                'ingredient_index': i,
                'ingredient_name': name,
                'original_quantity': qty,
                'new_quantity': qty * reduction_factor,
                'reduction_factor': reduction_factor,
                'reason': 'reduce_fat_to_meet_amdr'
            })
            break
    return corrections


def generate_correction_for_carb_low(violation, parsed_ingredients, targets):
    """碳水过低修正：增加carb食材"""
    corrections = []
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if any(c in name.lower() for c in CARB_KEYWORDS):
            increase_factor = random.uniform(1.5, 1.8)
            corrections.append({
                'action': 'increase_quantity',
                'ingredient_index': i,
                'ingredient_name': name,
                'original_quantity': qty,
                'new_quantity': qty * increase_factor,
                'increase_factor': increase_factor,
                'reason': 'increase_carb_to_meet_amdr'
            })
            break
    return corrections


def generate_correction_for_energy_high(violation, parsed_ingredients, targets):
    """能量超标修正：减少oil或carb"""
    corrections = []
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if 'oil' in name.lower() or any(c in name.lower() for c in ['rice', 'pasta', 'bread']):
            actual_energy = violation['actual']
            target_energy = violation['target']
            reduction_needed = (actual_energy - target_energy) / actual_energy
            reduction_factor = max(0.6, 1 - reduction_needed * 1.2)

            corrections.append({
                'action': 'reduce_quantity',
                'ingredient_index': i,
                'ingredient_name': name,
                'original_quantity': qty,
                'new_quantity': qty * reduction_factor,
                'reduction_factor': reduction_factor,
                'reason': 'reduce_energy_to_meet_target'
            })
            break
    return corrections


def generate_correction_for_fiber_low(violation, parsed_ingredients, targets):
    """纤维不足修正：增加vegetable"""
    corrections = []
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if any(v in name.lower() for v in VEGETABLE_KEYWORDS):
            increase_factor = random.uniform(1.5, 2.0)
            corrections.append({
                'action': 'increase_quantity',
                'ingredient_index': i,
                'ingredient_name': name,
                'original_quantity': qty,
                'new_quantity': qty * increase_factor,
                'increase_factor': increase_factor,
                'reason': 'increase_fiber_to_meet_minimum'
            })
            break

    if not corrections:
        high_fiber_veggies = ['broccoli', 'spinach', 'kale']
        for veggie in high_fiber_veggies:
            if is_ingredient_available(veggie):
                matched = find_best_match_ingredient(veggie)
                if matched:
                    qty = 100.0
                    corrections.append({
                        'action': 'add_ingredient',
                        'ingredient_name': matched,
                        'quantity': qty,
                        'reason': 'add_high_fiber_vegetable'
                    })
                    break
    return corrections


def generate_correction_for_saturated_fat_high(violation, parsed_ingredients, targets):
    """饱和脂肪超标修正：减少butter/cheese"""
    corrections = []
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if 'butter' in name.lower() or 'cheese' in name.lower():
            actual_pct = violation.get('actual_pct', 15)
            limit_pct = violation.get('limit_pct', 10)
            reduction_needed = (actual_pct - limit_pct) / actual_pct
            reduction_factor = max(0.2, 1 - reduction_needed * 1.3)

            corrections.append({
                'action': 'reduce_quantity',
                'ingredient_index': i,
                'ingredient_name': name,
                'original_quantity': qty,
                'new_quantity': qty * reduction_factor,
                'reduction_factor': reduction_factor,
                'reason': 'reduce_saturated_fat_to_meet_limit'
            })
            break
    return corrections


def generate_correction_for_sugars_high(violation, parsed_ingredients, targets):
    """糖分超标修正：减少honey/syrup"""
    corrections = []
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if any(s in name.lower() for s in HIGH_SUGAR_KEYWORDS):
            actual_pct = violation.get('actual_pct', 15)
            limit_pct = violation.get('limit_pct', 10)
            reduction_needed = (actual_pct - limit_pct) / actual_pct
            reduction_factor = max(0.2, 1 - reduction_needed * 1.3)

            corrections.append({
                'action': 'reduce_quantity',
                'ingredient_index': i,
                'ingredient_name': name,
                'original_quantity': qty,
                'new_quantity': qty * reduction_factor,
                'reduction_factor': reduction_factor,
                'reason': 'reduce_sugars_to_meet_limit'
            })
            break
    return corrections


def generate_correction_for_preference(violation, parsed_ingredients):
    """偏好违规修正"""
    corrections = []

    if violation['subtype'] == 'disliked_ingredient_added':
        bad_ing = violation['ingredient']
        for i, (qty, name, _) in enumerate(parsed_ingredients):
            if bad_ing in name.lower() or name.lower() in bad_ing:
                corrections.append({
                    'action': 'remove_ingredient',
                    'ingredient_index': i,
                    'ingredient_name': name,
                    'reason': 'remove_disliked_ingredient'
                })
                break

    elif violation['subtype'] == 'liked_ingredient_removed':
        liked_ing = violation['ingredient']
        matched = find_best_match_ingredient(liked_ing)
        if matched:
            qty = 100.0
            corrections.append({
                'action': 'add_ingredient',
                'ingredient_name': matched,
                'quantity': qty,
                'reason': 'restore_liked_ingredient'
            })

    return corrections


def generate_corrections(violations, parsed_ingredients, targets):
    """根据违约点生成修正方案（支持8种营养类型）"""
    all_corrections = []

    for violation in violations:
        if violation['type'] == 'nutrition_violation':
            field = violation['field']

            if field == 'sodium_mg':
                corrections = generate_correction_for_sodium(violation, parsed_ingredients, targets)
            elif field == 'protein_amdr':
                corrections = generate_correction_for_protein_low(violation, parsed_ingredients, targets)
            elif field == 'fat_amdr':
                corrections = generate_correction_for_fat_high(violation, parsed_ingredients, targets)
            elif field == 'carb_amdr':
                corrections = generate_correction_for_carb_low(violation, parsed_ingredients, targets)
            elif field == 'energy_kcal':
                corrections = generate_correction_for_energy_high(violation, parsed_ingredients, targets)
            elif field == 'fiber_g':
                corrections = generate_correction_for_fiber_low(violation, parsed_ingredients, targets)
            elif field == 'saturated_fat_g':
                corrections = generate_correction_for_saturated_fat_high(violation, parsed_ingredients, targets)
            elif field == 'sugars_g':
                corrections = generate_correction_for_sugars_high(violation, parsed_ingredients, targets)
            else:
                corrections = []

            all_corrections.extend(corrections)

        elif violation['type'] == 'preference_violation':
            corrections = generate_correction_for_preference(violation, parsed_ingredients)
            all_corrections.extend(corrections)

    return all_corrections


def apply_corrections(parsed_ingredients, corrections, servings=4):
    """应用修正方案，生成修正后的食谱"""
    corrected = copy.deepcopy(parsed_ingredients)

    corrections_sorted = sorted(corrections, key=lambda x: x.get('ingredient_index', -1), reverse=True)

    for correction in corrections_sorted:
        action = correction['action']

        if action == 'reduce_quantity':
            idx = correction['ingredient_index']
            new_qty = correction['new_quantity']
            name = correction['ingredient_name']
            corrected[idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

        elif action == 'increase_quantity':
            idx = correction['ingredient_index']
            new_qty = correction['new_quantity']
            name = correction['ingredient_name']
            corrected[idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

        elif action == 'remove_ingredient':
            idx = correction['ingredient_index']
            corrected.pop(idx)

        elif action == 'add_ingredient':
            name = correction['ingredient_name']
            qty = correction['quantity']
            corrected.append((qty, name, compose_ingredient_string(qty, name)))

    corrected_nutrition = recalculate_nutrition(corrected, servings)
    corrected_ingredients = [ing_str for _, _, ing_str in corrected]

    return {
        'recipe': {
            'ingredients': corrected_ingredients,
            'nutrition_per_serving': corrected_nutrition['per_serving'] if corrected_nutrition else {}
        },
        'nutrition_result': corrected_nutrition,
        'parsed_ingredients': corrected
    }


# ============================================================================
# C-Class Sample Generation (NEW - with instruction templates)
# ============================================================================

def generate_c_class_sample_v2(b_class_recipe, seed=0):
    """从B-class食谱生成一个C-class样本（新版，带指令模板）"""
    random.seed(seed + b_class_recipe['user_id'])

    # 1. 选择指令模板
    instruction, instruction_type = select_instruction_template_c(
        b_class_recipe['user_profile'],
        seed
    )

    # 2. 提取约束
    user_profile = b_class_recipe['user_profile']
    constraints = extract_constraints_from_rni(user_profile)

    # 3. 注入违规
    violation_result = generate_violated_recipe_v2(b_class_recipe)

    if violation_result is None:
        return None

    # 4. 生成修正
    corrections = generate_corrections(
        violation_result['violations'],
        violation_result['parsed_ingredients'],
        constraints
    )

    if not corrections:
        return None

    # 5. 应用修正
    corrected_result = apply_corrections(
        violation_result['parsed_ingredients'],
        corrections,
        servings=4
    )

    if corrected_result['nutrition_result'] is None:
        return None

    # 6. 构建统一格式的输出
    c_class_sample = {
        'user_id': b_class_recipe['user_id'],
        'recipe_id': f"c_class_{b_class_recipe['user_id']}_{seed}",

        # 与Task A/B一致的instruction字段
        'instruction': instruction,
        'instruction_type': instruction_type,

        # 与Task A/B一致的user_profile字段
        'user_profile': user_profile,

        # Task C特有的输入：有问题的草稿
        'input': {
            'violated_recipe': violation_result['violated_recipe'],
            'violations': violation_result['violations']
        },

        # Task C特有的输出：诊断 + 修正
        'output': {
            'diagnosis': violation_result['violations'],  # 诊断结果
            'corrections': corrections,  # 修正方案
            'corrected_recipe': corrected_result['recipe']  # 修正后的食谱
        },

        # 元数据
        'metadata': {
            'source': 'task_b_output',
            'source_recipe_id': b_class_recipe['recipe_id'],
            'num_violations': len(violation_result['violations']),
            'num_corrections': len(corrections),
            'data_source': 'hybrid (KG + external precise data)'
        }
    }

    return c_class_sample


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f"\n[2/6] Loading B-class datasets from new paths...")

    b_train = []
    b_val = []
    b_test = []

    try:
        with open(B_CLASS_TRAIN, 'r', encoding='utf-8') as f:
            b_train = [json.loads(line) for line in f if line.strip()]
        print(f"  [OK] Loaded {len(b_train)} training samples")
    except FileNotFoundError:
        print(f"  [ERROR] File not found: {B_CLASS_TRAIN}")
        exit(1)

    try:
        with open(B_CLASS_VAL, 'r', encoding='utf-8') as f:
            b_val = [json.loads(line) for line in f if line.strip()]
        print(f"  [OK] Loaded {len(b_val)} validation samples")
    except FileNotFoundError:
        print(f"  [ERROR] File not found: {B_CLASS_VAL}")
        exit(1)

    try:
        with open(B_CLASS_TEST, 'r', encoding='utf-8') as f:
            b_test = [json.loads(line) for line in f if line.strip()]
        print(f"  [OK] Loaded {len(b_test)} test samples")
    except FileNotFoundError:
        print(f"  [ERROR] File not found: {B_CLASS_TEST}")
        exit(1)

    splits = [
        ('train', b_train, f'{OUTPUT_DIR}task_c_train_from_kg.jsonl'),
        ('val', b_val, f'{OUTPUT_DIR}task_c_val_from_kg.jsonl'),
        ('test', b_test, f'{OUTPUT_DIR}task_c_test_from_kg.jsonl'),
    ]

    for split_name, b_recipes, output_file in splits:
        print(f"\n[3/6] Generating C-class {split_name} set ({len(b_recipes)} samples)...")

        c_samples = []
        failed_count = 0

        for i, b_recipe in enumerate(tqdm(b_recipes, desc=f"  Generating {split_name}")):
            c_sample = generate_c_class_sample_v2(b_recipe, seed=i)
            if c_sample:
                c_samples.append(c_sample)
            else:
                failed_count += 1

        print(f"\n[4/6] Saving {split_name} set to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in c_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        success_rate = (len(c_samples) / len(b_recipes) * 100) if b_recipes else 0
        print(f"  [OK] Saved {len(c_samples)} C-class samples")
        print(f"  Success rate: {success_rate:.1f}% ({failed_count} failed)")

    print(f"\n[6/6] Done!")
    print("="*80)
    print(f"Generated Task C datasets (integrated with Task B):")
    print(f"  Train: {OUTPUT_DIR}task_c_train_from_kg.jsonl")
    print(f"  Val:   {OUTPUT_DIR}task_c_val_from_kg.jsonl")
    print(f"  Test:  {OUTPUT_DIR}task_c_test_from_kg.jsonl")
    print("="*80)
