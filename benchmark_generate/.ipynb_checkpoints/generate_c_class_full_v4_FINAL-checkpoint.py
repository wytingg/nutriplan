#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-Class Dataset Generator v4 - FINAL VERSION (8 Nutrition Violation Types)
创建日期：2025-10-09
状态：已修复KeyError，可正式使用

版本更新：
v4.1 (2025-10-09 - KeyError修复):
- ✅ 修复saturated_fat_g和sugars_g字段兼容性问题
- ✅ 使用.get()方法支持多种字段名

v4 (2025-10-09 - FINAL):
- ✅ 添加fallback机制：如果一种violation失败，自动尝试其他类型
- ✅ 放宽violation阈值（0.65→0.8, 1.4→1.25）提高触发成功率
- ✅ 加强modification力度（更激进的增减）
- ✅ 精确计算correction factor（基于actual和limit的精确比例）
- ✅ 预期成功率：95%+

v3 (2025-10-09):
- 修复了注入函数在找不到所需食材时直接返回失败的问题
- 现在会自动添加默认食材（鸡胸肉、米饭、橄榄油、西兰花等）

完整实现：
1. 真实解析ingredient字符串并修改quantity
2. 重新计算营养值（使用RecipeNutritionCalculator）
3. 8种营养违规类型 + 2种偏好违规
4. 严谨的违规注入和修正策略
5. Fallback机制确保高成功率
6. 字段名兼容性处理
"""

import json
import random
import copy
from pathlib import Path
from calculate_recipe_nutrition import RecipeNutritionCalculator
from ingredient_parser import parse_ingredient_string, compose_ingredient_string
from tqdm import tqdm

print("="*80)
print("C-Class Dataset Generator v4.1 - FINAL (8 Nutrition Types)")
print("="*80)

# Configuration
NUTRITION_DB = 'work/recipebench/data/11_nutrition_rule/top500_nutrition_complete.csv'
B_CLASS_TRAIN = 'work/recipebench/data/10large_scale_datasets/task_b_train_large.jsonl'
B_CLASS_VAL = 'work/recipebench/data/10large_scale_datasets/task_b_val_large.jsonl'
B_CLASS_TEST = 'work/recipebench/data/10large_scale_datasets/task_b_test_large.jsonl'

OUTPUT_DIR = 'work/recipebench/data/10large_scale_datasets/'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load calculator
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
# Violation Injection Module (8 NUTRITION TYPES)
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
        # Add chicken breast as default protein source
        modified.append((200.0, 'chicken breast', '200g chicken breast'))
        protein_idx = len(modified) - 1

    original_qty, name, _ = modified[protein_idx]
    new_qty = original_qty * random.uniform(0.3, 0.5)  # 更激进：减少50-70%
    modified[protein_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        protein_per_serv = new_nutrition['per_serving']['protein_g']
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        protein_kcal = protein_per_serv * 4
        if energy_per_serv > 0:
            protein_pct = (protein_kcal / energy_per_serv) * 100
            target_protein_pct = targets.get('amdr', {}).get('protein', {}).get('target_pct', 20)

            if protein_pct < target_protein_pct * 0.8:  # 放宽阈值：0.65 → 0.8
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
        # Add oil if not present
        modified.append((10.0, 'olive oil', '2 tsps olive oil'))
        fat_idx = len(modified) - 1

    original_qty, name, _ = modified[fat_idx]
    new_qty = original_qty * random.uniform(2.5, 4.0)  # 更激进：增加150-300%
    modified[fat_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        fat_per_serv = new_nutrition['per_serving']['fat_g']
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        fat_kcal = fat_per_serv * 9
        if energy_per_serv > 0:
            fat_pct = (fat_kcal / energy_per_serv) * 100
            target_fat_pct = targets.get('amdr', {}).get('fat', {}).get('target_pct', 30)

            if fat_pct > target_fat_pct * 1.25:  # 放宽阈值：1.4 → 1.25
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
        # Add rice as default carb source
        modified.append((370.0, 'rice', '2 cups rice'))
        carb_idx = len(modified) - 1

    original_qty, name, _ = modified[carb_idx]
    new_qty = original_qty * random.uniform(0.3, 0.5)  # 更激进：减少50-70%
    modified[carb_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        carb_per_serv = new_nutrition['per_serving']['carbohydrates_g']
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        carb_kcal = carb_per_serv * 4
        if energy_per_serv > 0:
            carb_pct = (carb_kcal / energy_per_serv) * 100
            target_carb_pct = targets.get('amdr', {}).get('carb', {}).get('target_pct', 50)

            if carb_pct < target_carb_pct * 0.75:  # 放宽阈值：0.6 → 0.75
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
        # Add olive oil as default energy source
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
        # Add broccoli as default vegetable
        modified.append((200.0, 'broccoli', '200g broccoli'))
        veg_idx = len(modified) - 1

    original_qty, name, _ = modified[veg_idx]
    new_qty = original_qty * random.uniform(0.2, 0.4)  # 更激进：减少60-80%
    modified[veg_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        fiber_per_serv = new_nutrition['per_serving']['fiber_g']
        fiber_total = fiber_per_serv * servings
        fiber_min = targets.get('fiber_g_min', 25)

        if fiber_total < fiber_min * 0.8:  # 放宽阈值：0.65 → 0.8
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

    # Find butter or high saturated fat ingredient
    sat_fat_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if 'butter' in name.lower() or 'cheese' in name.lower():
            sat_fat_idx = i
            break

    if sat_fat_idx is None:
        # Add butter
        modified.append((15.0, 'butter', '1 Tbsp butter'))
        sat_fat_idx = len(modified) - 1

    original_qty, name, _ = modified[sat_fat_idx]
    new_qty = original_qty * random.uniform(2.5, 3.5)
    modified[sat_fat_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        # 尝试不同的字段名 - 兼容性处理
        sat_fat_per_serv = new_nutrition['per_serving'].get('saturated_fat_g') or \
                           new_nutrition['per_serving'].get('saturated_fat', 0)
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        sat_fat_kcal = sat_fat_per_serv * 9
        if energy_per_serv > 0:
            sat_fat_pct = (sat_fat_kcal / energy_per_serv) * 100

            # Saturated fat should be <10% of total energy
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

    # Find or add sugar source
    sugar_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if any(s in name.lower() for s in HIGH_SUGAR_KEYWORDS):
            sugar_idx = i
            break

    if sugar_idx is None:
        # Add honey
        modified.append((20.0, 'honey', '1 Tbsp honey'))
        sugar_idx = len(modified) - 1

    original_qty, name, _ = modified[sugar_idx]
    new_qty = original_qty * random.uniform(2.0, 3.0)
    modified[sugar_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    new_nutrition = recalculate_nutrition(modified, servings)
    if new_nutrition:
        # 尝试不同的字段名 - 兼容性处理
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
        disliked_names = [ing['name'].lower() for ing in disliked_ingredients]
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
        liked_names = [ing['name'].lower() for ing in liked_ingredients]
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


def generate_violated_recipe(b_class_recipe):
    """
    从B-class合法食谱生成违规初稿

    违规类型分布：
    - 60%: 单一营养违规（8种类型）
    - 25%: 偏好违规
    - 15%: 营养+偏好双重违规
    """
    original_recipe = b_class_recipe['output']
    targets = b_class_recipe['input']['nutrition_targets']
    liked = b_class_recipe['input']['liked_ingredients']
    disliked = b_class_recipe['input']['disliked_ingredients']

    parsed_ingredients = parse_recipe_ingredients(original_recipe['ingredients'])
    original_nutrition = original_recipe['nutrition_per_serv']

    rand = random.random()

    if rand < 0.6:
        # Single nutrition violation (8 types) - with fallback mechanism
        violation_types = [
            'sodium', 'protein_low', 'fat_high', 'carb_low',
            'energy_high', 'fiber_low', 'saturated_fat_high', 'sugars_high'
        ]
        random.shuffle(violation_types)  # 随机顺序

        modified_ings, violations, new_nutrition = None, [], None

        # 尝试多种violation类型，直到成功
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

            # 如果成功注入violation，跳出循环
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
        violated_recipe['nutrition_per_serv'] = new_nutrition['per_serving']

    return {
        'violated_recipe': violated_recipe,
        'violations': violations,
        'parsed_ingredients': modified_ings,
        'nutrition_result': new_nutrition
    }


# ============================================================================
# Correction Strategy Module (FOR 8 NUTRITION TYPES)
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
            # 精确计算需要减少的量
            actual_pct = violation.get('actual_pct', 15)
            limit_pct = violation.get('limit_pct', 10)
            reduction_needed = (actual_pct - limit_pct) / actual_pct
            reduction_factor = max(0.2, 1 - reduction_needed * 1.3)  # 多减少30%作为缓冲

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
            # 精确计算需要减少的量
            actual_pct = violation.get('actual_pct', 15)
            limit_pct = violation.get('limit_pct', 10)
            reduction_needed = (actual_pct - limit_pct) / actual_pct
            reduction_factor = max(0.2, 1 - reduction_needed * 1.3)  # 多减少30%作为缓冲

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

    return corrected_ingredients, corrected_nutrition, corrected


# ============================================================================
# C-Class Sample Generation
# ============================================================================

def generate_c_class_sample(b_class_recipe, seed=0):
    """从B-class食谱生成一个C-class样本"""
    random.seed(seed)

    violation_result = generate_violated_recipe(b_class_recipe)
    if violation_result is None:
        return None

    violated_recipe = violation_result['violated_recipe']
    violations = violation_result['violations']
    parsed_ingredients = violation_result['parsed_ingredients']

    targets = b_class_recipe['input']['nutrition_targets']
    corrections = generate_corrections(violations, parsed_ingredients, targets)

    if not corrections:
        return None

    corrected_ingredients, corrected_nutrition, corrected_parsed = apply_corrections(
        parsed_ingredients, corrections, servings=4)

    if corrected_nutrition is None:
        return None

    corrected_recipe = copy.deepcopy(violated_recipe)
    corrected_recipe['ingredients'] = corrected_ingredients
    corrected_recipe['nutrition_per_serv'] = corrected_nutrition['per_serving']

    c_class_sample = {
        'user_id': b_class_recipe['user_id'],
        'recipe_id': f"c_class_{b_class_recipe['user_id']}_{seed}",
        'input': {
            'violated_recipe': violated_recipe,
            'violations': violations,
            'nutrition_targets': b_class_recipe['input']['nutrition_targets'],
            'liked_ingredients': b_class_recipe['input']['liked_ingredients'],
            'disliked_ingredients': b_class_recipe['input']['disliked_ingredients']
        },
        'output': {
            'corrected_recipe': corrected_recipe,
            'corrections': corrections
        },
        'metadata': {
            'source_recipe_id': b_class_recipe['recipe_id'],
            'num_violations': len(violations),
            'num_corrections': len(corrections)
        }
    }

    return c_class_sample


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f"\n[2/6] Loading B-class datasets...")

    b_train = []
    b_val = []
    b_test = []

    with open(B_CLASS_TRAIN, 'r', encoding='utf-8') as f:
        b_train = [json.loads(line) for line in f if line.strip()]

    with open(B_CLASS_VAL, 'r', encoding='utf-8') as f:
        b_val = [json.loads(line) for line in f if line.strip()]

    with open(B_CLASS_TEST, 'r', encoding='utf-8') as f:
        b_test = [json.loads(line) for line in f if line.strip()]

    print(f"  [OK] Loaded {len(b_train)} train / {len(b_val)} val / {len(b_test)} test B-class recipes")

    splits = [
        ('train', b_train, f'{OUTPUT_DIR}task_c_train_large.jsonl'),
        ('val', b_val, f'{OUTPUT_DIR}task_c_val_large.jsonl'),
        ('test', b_test, f'{OUTPUT_DIR}task_c_test_large.jsonl'),
    ]

    for split_name, b_recipes, output_file in splits:
        print(f"\n[3/6] Generating C-class {split_name} set ({len(b_recipes)} samples)...")

        c_samples = []
        failed_count = 0

        for i, b_recipe in enumerate(tqdm(b_recipes, desc=f"  Generating {split_name}")):
            c_sample = generate_c_class_sample(b_recipe, seed=i)
            if c_sample:
                c_samples.append(c_sample)
            else:
                failed_count += 1

        print(f"\n[4/6] Saving {split_name} set to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in c_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"  [OK] Saved {len(c_samples)} C-class samples ({failed_count} failed)")

    print(f"\n[6/6] Done!")
    print("="*80)
    print(f"Generated C-class datasets:")
    print(f"  Train: {OUTPUT_DIR}task_c_train_large.jsonl")
    print(f"  Val:   {OUTPUT_DIR}task_c_val_large.jsonl")
    print(f"  Test:  {OUTPUT_DIR}task_c_test_large.jsonl")
    print("="*80)
