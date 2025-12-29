#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-Class Dataset Generator v2 - COMPLETE & RIGOROUS VERSION

完整实现：
1. 真实解析ingredient字符串并修改quantity
2. 重新计算营养值（使用RecipeNutritionCalculator）
3. 只使用有营养数据的食材池（避免85%覆盖率问题）
4. 严谨的违规注入和修正策略
"""

import json
import random
import copy
from pathlib import Path
from calculate_recipe_nutrition import RecipeNutritionCalculator
from ingredient_parser import parse_ingredient_string, compose_ingredient_string
from tqdm import tqdm

print("="*80)
print("C-Class Dataset Generator v2 (Complete & Rigorous)")
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

# Build available ingredients pool (only those with nutrition data)
AVAILABLE_INGREDIENTS = set(calc.nutrition_lookup.keys())

# Food classification
PROTEIN_KEYWORDS = ['chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna', 'tofu', 'egg', 'turkey']
CARB_KEYWORDS = ['rice', 'pasta', 'bread', 'potato', 'oat', 'quinoa', 'corn', 'noodle']
VEGETABLE_KEYWORDS = ['broccoli', 'spinach', 'kale', 'carrot', 'tomato', 'pepper', 'onion']
HIGH_SODIUM_KEYWORDS = ['salt', 'soy sauce', 'bacon', 'ham', 'cheese']


def is_ingredient_available(ing_name):
    """Check if ingredient has nutrition data"""
    # Exact match or fuzzy match
    ing_lower = ing_name.lower()
    if ing_lower in AVAILABLE_INGREDIENTS:
        return True

    # Fuzzy match: check if any available ingredient is substring
    for avail_ing in AVAILABLE_INGREDIENTS:
        if avail_ing in ing_lower or ing_lower in avail_ing:
            return True

    return False


def find_best_match_ingredient(ing_name):
    """Find best matching ingredient in nutrition database"""
    ing_lower = ing_name.lower()

    # Exact match
    if ing_lower in AVAILABLE_INGREDIENTS:
        return ing_lower

    # Fuzzy match
    for avail_ing in AVAILABLE_INGREDIENTS:
        if avail_ing in ing_lower:
            return avail_ing

    # Reverse fuzzy match
    for avail_ing in AVAILABLE_INGREDIENTS:
        if ing_lower in avail_ing:
            return avail_ing

    return None


def parse_recipe_ingredients(ingredients_list):
    """
    Parse recipe ingredients to structured format

    Returns: [(qty_grams, ingredient_name, original_string), ...]
    """
    parsed = []
    for ing_str in ingredients_list:
        result = parse_ingredient_string(ing_str)
        if result:
            qty, name = result
            parsed.append((qty, name, ing_str))
        else:
            # Parsing failed, keep as-is with default 100g
            parsed.append((100.0, ing_str, ing_str))

    return parsed


def recalculate_nutrition(parsed_ingredients, servings=4):
    """
    Recalculate nutrition from parsed ingredients

    Returns: nutrition_result dict or None if failed
    """
    ingredient_strings = []

    for qty_grams, ing_name, _ in parsed_ingredients:
        # Find matching ingredient in database
        matched_ing = find_best_match_ingredient(ing_name)
        if matched_ing:
            ingredient_strings.append(f"{qty_grams}g {matched_ing}")
        else:
            # Skip ingredients without nutrition data
            continue

    if not ingredient_strings:
        return None

    try:
        nutrition_result = calc.calculate_recipe_nutrition(ingredient_strings, servings)
        return nutrition_result
    except Exception as e:
        print(f"  [WARNING] Nutrition calculation failed: {e}")
        return None


# ============================================================================
# Violation Injection Module (RIGOROUS VERSION)
# ============================================================================

def inject_sodium_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """
    注入钠超标违规（通过增加salt quantity）

    Returns: (modified_ingredients, violations, new_nutrition)
    """
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    # Find salt in ingredients
    salt_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if 'salt' in name.lower():
            salt_idx = i
            break

    if salt_idx is None:
        # No salt found, add it
        modified.append((2.0, 'salt', '1/2 tsp salt'))
        salt_idx = len(modified) - 1

    # Increase salt by 150-200% to cause sodium violation
    original_qty, name, _ = modified[salt_idx]
    new_qty = original_qty * random.uniform(2.5, 3.0)
    modified[salt_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    # Recalculate nutrition
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
                'severity': 'critical' if sodium_total > sodium_max * 1.5 else 'major',
                'per_serving_actual': sodium_per_serv,
                'per_serving_limit': sodium_max / servings
            })

            return modified, violations, new_nutrition

    # Fallback: return original if calculation failed
    return parsed_ingredients, [], original_nutrition


def inject_protein_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """
    注入蛋白质不足违规（通过减少protein食材quantity）
    """
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    # Find protein source
    protein_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if any(p in name.lower() for p in PROTEIN_KEYWORDS):
            protein_idx = i
            break

    if protein_idx is None:
        return parsed_ingredients, [], original_nutrition

    # Reduce protein by 30-40%
    original_qty, name, _ = modified[protein_idx]
    new_qty = original_qty * random.uniform(0.6, 0.7)
    modified[protein_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    # Recalculate nutrition
    new_nutrition = recalculate_nutrition(modified, servings)

    if new_nutrition:
        protein_per_serv = new_nutrition['per_serving']['protein_g']
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']

        # Check AMDR protein percentage
        protein_kcal = protein_per_serv * 4
        if energy_per_serv > 0:
            protein_pct = (protein_kcal / energy_per_serv) * 100
            target_protein_pct = targets.get('amdr', {}).get('protein', {}).get('target_pct', 20)

            if protein_pct < target_protein_pct * 0.7:  # 30% below target
                violations.append({
                    'type': 'nutrition_violation',
                    'field': 'protein_g',
                    'actual': protein_per_serv,
                    'actual_pct': protein_pct,
                    'target_pct': target_protein_pct,
                    'severity': 'major'
                })

                return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_energy_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """
    注入能量超标违规（通过增加oil/carb quantity）
    """
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    # Find oil or carb
    target_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if 'oil' in name.lower() or any(c in name.lower() for c in CARB_KEYWORDS):
            target_idx = i
            break

    if target_idx is None:
        return parsed_ingredients, [], original_nutrition

    # Increase by 30-50%
    original_qty, name, _ = modified[target_idx]
    new_qty = original_qty * random.uniform(1.3, 1.5)
    modified[target_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    # Recalculate nutrition
    new_nutrition = recalculate_nutrition(modified, servings)

    if new_nutrition:
        energy_per_serv = new_nutrition['per_serving']['energy_kcal']
        target_energy = targets.get('energy_kcal_target', 2000) / servings

        if energy_per_serv > target_energy * 1.2:
            violations.append({
                'type': 'nutrition_violation',
                'field': 'energy_kcal',
                'actual': energy_per_serv,
                'target': target_energy,
                'severity': 'major'
            })

            return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_fiber_violation(parsed_ingredients, original_nutrition, targets, servings=4):
    """
    注入纤维不足违规（通过减少vegetable quantity）
    """
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    # Find vegetable
    veg_idx = None
    for i, (qty, name, _) in enumerate(modified):
        if any(v in name.lower() for v in VEGETABLE_KEYWORDS):
            veg_idx = i
            break

    if veg_idx is None:
        return parsed_ingredients, [], original_nutrition

    # Reduce by 40-50%
    original_qty, name, _ = modified[veg_idx]
    new_qty = original_qty * random.uniform(0.5, 0.6)
    modified[veg_idx] = (new_qty, name, compose_ingredient_string(new_qty, name))

    # Recalculate nutrition
    new_nutrition = recalculate_nutrition(modified, servings)

    if new_nutrition:
        fiber_per_serv = new_nutrition['per_serving']['fiber_g']
        fiber_total = fiber_per_serv * servings
        fiber_min = targets.get('fiber_g_min', 25)

        if fiber_total < fiber_min * 0.7:
            violations.append({
                'type': 'nutrition_violation',
                'field': 'fiber_g',
                'actual': fiber_total,
                'minimum': fiber_min,
                'severity': 'major'
            })

            return modified, violations, new_nutrition

    return parsed_ingredients, [], original_nutrition


def inject_preference_violation(parsed_ingredients, liked_ingredients, disliked_ingredients):
    """
    注入偏好违规（添加disliked或删除liked食材）

    只使用有营养数据的食材
    """
    modified = copy.deepcopy(parsed_ingredients)
    violations = []

    # 60% chance: add disliked ingredient
    if disliked_ingredients and random.random() < 0.6:
        disliked_names = [ing['name'].lower() for ing in disliked_ingredients]

        # Find one that has nutrition data
        available_disliked = []
        for d_name in disliked_names:
            if is_ingredient_available(d_name):
                available_disliked.append(d_name)

        if available_disliked:
            bad_ing = random.choice(available_disliked)
            matched_ing = find_best_match_ingredient(bad_ing)

            if matched_ing:
                # Add with reasonable quantity (50-100g)
                qty = random.uniform(50, 100)
                modified.append((qty, matched_ing, compose_ingredient_string(qty, matched_ing)))

                violations.append({
                    'type': 'preference_violation',
                    'subtype': 'disliked_ingredient_added',
                    'ingredient': matched_ing,
                    'severity': 'critical'
                })

    # 40% chance: remove liked ingredient
    elif liked_ingredients and random.random() < 0.4:
        liked_names = [ing['name'].lower() for ing in liked_ingredients]

        # Find liked ingredient in current recipe
        for i, (qty, name, _) in enumerate(modified):
            for liked_name in liked_names:
                if liked_name in name.lower() or name.lower() in liked_name:
                    # Remove this ingredient
                    removed_ing = modified.pop(i)

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
    从B-class合法食谱生成违规初稿（完整版）

    违规类型分布：
    - 50%: 单一营养违规
    - 30%: 偏好违规
    - 20%: 营养+偏好双重违规
    """
    original_recipe = b_class_recipe['output']
    targets = b_class_recipe['input']['nutrition_targets']
    liked = b_class_recipe['input']['liked_ingredients']
    disliked = b_class_recipe['input']['disliked_ingredients']

    # Parse ingredients
    parsed_ingredients = parse_recipe_ingredients(original_recipe['ingredients'])
    original_nutrition = original_recipe['nutrition_per_serv']

    rand = random.random()

    if rand < 0.5:
        # Single nutrition violation
        violation_type = random.choice(['sodium', 'protein', 'energy', 'fiber'])

        if violation_type == 'sodium':
            modified_ings, violations, new_nutrition = inject_sodium_violation(
                parsed_ingredients, original_nutrition, targets)
        elif violation_type == 'protein':
            modified_ings, violations, new_nutrition = inject_protein_violation(
                parsed_ingredients, original_nutrition, targets)
        elif violation_type == 'energy':
            modified_ings, violations, new_nutrition = inject_energy_violation(
                parsed_ingredients, original_nutrition, targets)
        else:  # fiber
            modified_ings, violations, new_nutrition = inject_fiber_violation(
                parsed_ingredients, original_nutrition, targets)

    elif rand < 0.8:
        # Preference violation only
        modified_ings, violations = inject_preference_violation(
            parsed_ingredients, liked, disliked)
        new_nutrition = recalculate_nutrition(modified_ings)

    else:
        # Combined: nutrition + preference
        violation_type = random.choice(['sodium', 'energy'])

        if violation_type == 'sodium':
            modified_ings, violations1, new_nutrition = inject_sodium_violation(
                parsed_ingredients, original_nutrition, targets)
        else:
            modified_ings, violations1, new_nutrition = inject_energy_violation(
                parsed_ingredients, original_nutrition, targets)

        modified_ings, violations2 = inject_preference_violation(
            modified_ings, liked, disliked)
        violations = violations1 + violations2

        # Recalculate after both violations
        new_nutrition = recalculate_nutrition(modified_ings)

    # If no violations were successfully injected, return None
    if not violations:
        return None

    # Construct violated recipe
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
# Correction Strategy Module (RIGOROUS VERSION)
# ============================================================================

def generate_correction_for_sodium(violation, parsed_ingredients, targets):
    """生成钠超标的修正方案"""
    corrections = []

    # Find salt and reduce quantity
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if 'salt' in name.lower():
            # Calculate reduction needed
            actual_sodium = violation['actual']
            limit_sodium = violation['limit']
            reduction_needed = (actual_sodium - limit_sodium) / actual_sodium

            # Reduce salt by at least reduction_needed, with some margin
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


def generate_correction_for_protein(violation, parsed_ingredients, targets):
    """生成蛋白质不足的修正方案"""
    corrections = []

    # Find protein source and increase quantity
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if any(p in name.lower() for p in PROTEIN_KEYWORDS):
            # Increase by 30-50%
            increase_factor = random.uniform(1.3, 1.5)

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


def generate_correction_for_energy(violation, parsed_ingredients, targets):
    """生成能量超标的修正方案"""
    corrections = []

    # Find oil or high-energy carb and reduce
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if 'oil' in name.lower() or any(c in name.lower() for c in ['rice', 'pasta', 'bread']):
            # Calculate reduction needed
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


def generate_correction_for_fiber(violation, parsed_ingredients, targets):
    """生成纤维不足的修正方案"""
    corrections = []

    # Find vegetable and increase quantity
    for i, (qty, name, _) in enumerate(parsed_ingredients):
        if any(v in name.lower() for v in VEGETABLE_KEYWORDS):
            # Increase by 40-60%
            increase_factor = random.uniform(1.4, 1.6)

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

    # If no vegetable found, add one
    if not corrections:
        # Add a high-fiber vegetable
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


def generate_correction_for_preference(violation, parsed_ingredients):
    """生成偏好违规的修正方案"""
    corrections = []

    if violation['subtype'] == 'disliked_ingredient_added':
        # Remove the disliked ingredient
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
        # Add back the liked ingredient
        liked_ing = violation['ingredient']

        # Find matching ingredient in database
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
    """根据违约点生成修正方案"""
    all_corrections = []

    for violation in violations:
        if violation['type'] == 'nutrition_violation':
            field = violation['field']

            if field == 'sodium_mg':
                corrections = generate_correction_for_sodium(violation, parsed_ingredients, targets)
            elif field == 'protein_g':
                corrections = generate_correction_for_protein(violation, parsed_ingredients, targets)
            elif field == 'energy_kcal':
                corrections = generate_correction_for_energy(violation, parsed_ingredients, targets)
            elif field == 'fiber_g':
                corrections = generate_correction_for_fiber(violation, parsed_ingredients, targets)
            else:
                corrections = []

            all_corrections.extend(corrections)

        elif violation['type'] == 'preference_violation':
            corrections = generate_correction_for_preference(violation, parsed_ingredients)
            all_corrections.extend(corrections)

    return all_corrections


def apply_corrections(parsed_ingredients, corrections, servings=4):
    """
    应用修正方案，生成修正后的食谱

    Returns: (corrected_ingredients, corrected_nutrition)
    """
    corrected = copy.deepcopy(parsed_ingredients)

    # Sort corrections by index (descending) to handle removals correctly
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

    # Recalculate nutrition
    corrected_nutrition = recalculate_nutrition(corrected, servings)

    # Compose ingredient strings
    corrected_ingredients = [ing_str for _, _, ing_str in corrected]

    return corrected_ingredients, corrected_nutrition, corrected


# ============================================================================
# C-Class Sample Generation
# ============================================================================

def generate_c_class_sample(b_class_recipe, seed=0):
    """
    从B-class食谱生成一个C-class样本（完整版）
    """
    random.seed(seed)

    # Step 1: Generate violated recipe
    violation_result = generate_violated_recipe(b_class_recipe)

    if violation_result is None:
        # Failed to inject violations
        return None

    violated_recipe = violation_result['violated_recipe']
    violations = violation_result['violations']
    parsed_ingredients = violation_result['parsed_ingredients']

    # Step 2: Generate corrections
    targets = b_class_recipe['input']['nutrition_targets']
    corrections = generate_corrections(violations, parsed_ingredients, targets)

    if not corrections:
        # No corrections generated
        return None

    # Step 3: Apply corrections
    corrected_ingredients, corrected_nutrition, corrected_parsed = apply_corrections(
        parsed_ingredients, corrections, servings=4)

    if corrected_nutrition is None:
        # Failed to calculate corrected nutrition
        return None

    # Step 4: Construct corrected recipe
    corrected_recipe = copy.deepcopy(violated_recipe)
    corrected_recipe['ingredients'] = corrected_ingredients
    corrected_recipe['nutrition_per_serv'] = corrected_nutrition['per_serving']

    # Step 5: Build C-class sample
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

    # Generate C-class datasets
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
