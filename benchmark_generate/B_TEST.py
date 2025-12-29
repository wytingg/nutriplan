#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-class Dataset Builder v3: CRITICAL FIXES

Fixes:
1. Expand protein keywords to catch "snapper", "fillet", etc.
2. Filter high-sodium ingredients (>500mg/100g)
3. Force include high-fiber vegetables
4. Set maximum quantities for seasonings (garlic, onion)
5. Smarter recipe step generation
"""

import pandas as pd
import json
import sys
import random
from pathlib import Path
from collections import defaultdict
from calculate_recipe_nutrition import RecipeNutritionCalculator
from ingredient_normalizer import normalize_ingredient, fuzzy_match_ingredient

print("="*80)
print("B-Class Dataset Builder v3: CRITICAL FIXES")
print("="*80)

# Configuration
NUTRITION_DB = 'work/recipebench/data/11_nutrition_rule/top500_nutrition_complete.csv'
USER_PROFILES = 'work/recipebench/data/8step_profile/cleaned_user_profile.jsonl'

# KG Components
COOCCURRENCE_FILE = 'work/recipebench/data/11_nutrition_rule/ingredient_cooccurrence.csv'
NUTRIENT_TAGS_FILE = 'work/recipebench/data/11_nutrition_rule/ingredient_nutrient_tags.csv'
COMPLEMENTARITY_FILE = 'work/recipebench/data/11_nutrition_rule/nutrition_complementarity_pairs.csv'

OUTPUT_DIR = 'work/recipebench/data/10large_scale_datasets/'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# FIX 1: Expanded protein keywords (catches "snapper", "fillet", etc.)
PROTEIN_KEYWORDS = [
    'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck',
    'fish', 'snapper', 'fillet', 'salmon', 'tuna', 'cod', 'tilapia', 'halibut',
    'shrimp', 'prawn', 'crab', 'lobster', 'scallop',
    'egg', 'tofu', 'tempeh', 'seitan',
    'bean', 'lentil', 'chickpea'
]

CARB_KEYWORDS = ['rice', 'pasta', 'noodle', 'spaghetti', 'bread', 'potato',
                 'quinoa', 'oat', 'corn', 'cornmeal', 'barley', 'couscous']

# FIX 2: High-fiber vegetables (must include at least 1)
HIGH_FIBER_VEGETABLES = ['broccoli', 'spinach', 'kale', 'brussels sprout',
                         'carrot', 'cauliflower', 'sweet potato', 'bean']

# ============================================================================
# Load KG
# ============================================================================

print("\n[1/5] Loading KG...")
calc = RecipeNutritionCalculator(NUTRITION_DB)

df_cooccur = pd.read_csv(COOCCURRENCE_FILE)
cooccur_dict = defaultdict(list)
for _, row in df_cooccur.iterrows():
    ing1 = row['ingredient_1'].lower()
    ing2 = row['ingredient_2'].lower()
    score = row['pmi_score'] if 'pmi_score' in row else row['confidence']
    cooccur_dict[ing1].append({'ingredient': ing2, 'score': score})
    cooccur_dict[ing2].append({'ingredient': ing1, 'score': score})

df_tags = pd.read_csv(NUTRIENT_TAGS_FILE)
tag_to_ingredients = defaultdict(list)
for _, row in df_tags.iterrows():
    tag_to_ingredients[row['nutrient_tag']].append(row['ingredient'].lower())

df_compl = pd.read_csv(COMPLEMENTARITY_FILE)

print(f"  ✓ Loaded KG")

# ============================================================================
# Helper Functions
# ============================================================================

def is_protein_source(ing):
    return any(keyword in ing for keyword in PROTEIN_KEYWORDS)

def is_carb_source(ing):
    return any(keyword in ing for keyword in CARB_KEYWORDS)

def is_high_fiber(ing):
    return any(veg in ing for veg in HIGH_FIBER_VEGETABLES)

def get_sodium_per_100g(ing):
    """Get sodium content from database"""
    nutrition = calc.find_ingredient_in_db(ing)
    if nutrition:
        return nutrition.get('sodium_mg', 0)
    return 0

def is_high_sodium(ing):
    """FIX 2: Check if ingredient has high sodium (>500mg/100g)"""
    return get_sodium_per_100g(ing) > 500

# ============================================================================
# Ingredient Selection v3
# ============================================================================

def match_user_preferences_to_db(user_ingredients, database_ingredients):
    """
    Match user preference ingredients to database using normalization

    Args:
        user_ingredients: list of ingredient names from user preferences
        database_ingredients: list of ingredient names in database

    Returns:
        matched_names: list of matched database ingredient names
    """
    matched = []
    for user_ing in user_ingredients:
        # Try fuzzy match with database
        match, score = fuzzy_match_ingredient(user_ing, database_ingredients, threshold=0.6)
        if match:
            matched.append(match.lower())
            print(f"        [Matched] '{user_ing}' -> '{match}' (score: {score:.2f})")
        else:
            # Fall back to original if no match found
            matched.append(user_ing.lower())
            print(f"        [No match] '{user_ing}' (keeping original)")

    return matched

def select_ingredients_v3(liked, disliked, nutrition_targets, seed=0):
    """v3: Better liked matching + sodium filter + fiber requirement + ingredient normalization"""
    random.seed(seed)

    # Get database ingredients list
    db_ingredients = list(calc.nutrition_lookup.keys())

    # Match user preferences to database using normalization
    print("    [Matching user preferences to database...]")
    raw_liked_names = [ing['name'] for ing in liked]
    raw_disliked_names = [ing['name'] for ing in disliked]

    liked_names = match_user_preferences_to_db(raw_liked_names, db_ingredients)
    disliked_names = match_user_preferences_to_db(raw_disliked_names, db_ingredients)

    sugars_info = nutrition_targets.get('sugars', {})
    restrict_sugar = sugars_info and sugars_info.get('pct_max', 100) <= 10

    # FIX 2: Check sodium restriction
    sodium_max = nutrition_targets.get('sodium_mg_max', 2300)
    restrict_sodium = sodium_max <= 1500

    selected = []

    def is_valid(ing):
        if ing in disliked_names or any(d in ing for d in disliked_names):
            return False
        if restrict_sugar and 'sugar' in ing:
            return False
        # FIX 2: Filter high-sodium ingredients
        if restrict_sodium and is_high_sodium(ing):
            print(f"        [Filtered] {ing} (high sodium: {get_sodium_per_100g(ing):.0f}mg/100g)")
            return False
        if not calc.find_ingredient_in_db(ing):
            return False
        return True

    # STEP 1: Find liked protein (with expanded keywords)
    print("      [Step 1] Looking for liked protein...")
    liked_proteins = [ing for ing in liked_names
                     if is_protein_source(ing) and is_valid(ing)]

    if liked_proteins:
        selected.append(random.choice(liked_proteins))
        print(f"        ✓ Liked protein: {selected[-1]}")
    else:
        # Fallback to database
        available = [ing for ing in calc.nutrition_lookup.keys()
                    if is_protein_source(ing) and is_valid(ing)]
        if available:
            selected.append(random.choice(available[:20]))
            print(f"        ✓ Protein (fallback): {selected[-1]}")
        else:
            selected.append('chicken')

    # STEP 2: Find liked carb
    print("      [Step 2] Looking for liked carb...")
    liked_carbs = [ing for ing in liked_names
                  if is_carb_source(ing) and is_valid(ing)]

    if liked_carbs:
        selected.append(random.choice(liked_carbs))
        print(f"        ✓ Liked carb: {selected[-1]}")
    else:
        available = [ing for ing in calc.nutrition_lookup.keys()
                    if is_carb_source(ing) and is_valid(ing)]
        if available:
            selected.append(random.choice(available[:20]))
            print(f"        ✓ Carb (fallback): {selected[-1]}")
        else:
            selected.append('rice')

    # FIX 3: MUST include at least 1 high-fiber vegetable
    print("      [Step 3] Adding high-fiber vegetable...")
    fiber_min = nutrition_targets.get('fiber_g_min', 25)

    if fiber_min >= 25:
        available_fiber = [ing for ing in calc.nutrition_lookup.keys()
                          if is_high_fiber(ing) and is_valid(ing)]
        if available_fiber:
            selected.append(random.choice(available_fiber[:10]))
            print(f"        ✓ High-fiber veg: {selected[-1]}")

    # STEP 4: Add 2-3 more vegetables
    print("      [Step 4] Adding vegetables...")
    for seed_ing in selected[:2]:
        if len(selected) >= 6:
            break

        cooccur_list = cooccur_dict.get(seed_ing, [])
        cooccur_list = sorted(cooccur_list, key=lambda x: x['score'], reverse=True)

        for item in cooccur_list[:20]:
            co_ing = item['ingredient']
            if (co_ing not in selected and is_valid(co_ing)
                and not is_protein_source(co_ing) and not is_carb_source(co_ing)):
                selected.append(co_ing)
                print(f"        ✓ Vegetable: {co_ing}")
                break

    # STEP 5: Add oil + seasonings (but not too much)
    essentials = ['olive oil', 'salt', 'pepper']
    for ing in essentials:
        if len(selected) >= 7:
            break
        if ing not in selected and is_valid(ing):
            selected.append(ing)

    return selected[:7]

# ============================================================================
# Quantity Optimization v3
# ============================================================================

def optimize_quantities_v3(ingredients, nutrition_targets, servings=4):
    """v3: Set max limits for seasonings"""

    target_kcal_per_serving = nutrition_targets.get('energy_kcal_target', 2000) / 3
    print(f"      [Optimization] Target: {target_kcal_per_serving:.0f} kcal/serving")

    # Base quantities
    base_qty = {}
    for ing in ingredients:
        if is_protein_source(ing):
            base_qty[ing] = 150  # 150g per serving
        elif is_carb_source(ing):
            base_qty[ing] = 100
        elif 'oil' in ing or 'butter' in ing:
            base_qty[ing] = 10
        elif ing in ['salt', 'pepper']:
            base_qty[ing] = 2
        # FIX 4: Limit garlic/onion
        elif 'garlic' in ing:
            base_qty[ing] = 10   # Max 10g per serving
        elif 'onion' in ing:
            base_qty[ing] = 50   # Max 50g per serving
        else:
            base_qty[ing] = 80

    # Scale to total
    total_qty = {ing: qty * servings for ing, qty in base_qty.items()}

    # Calculate nutrition
    ingredient_strings = [f"{qty}g {ing}" for ing, qty in total_qty.items()]
    nutrition_result = calc.calculate_recipe_nutrition(ingredient_strings, servings)

    actual_kcal = nutrition_result['per_serving']['energy_kcal']
    print(f"        Base: {actual_kcal:.0f} kcal/serving")

    # Scale
    if actual_kcal > 0:
        scale_factor = target_kcal_per_serving / actual_kcal
        scale_factor = max(0.5, min(2.0, scale_factor))
        print(f"        Scale: {scale_factor:.2f}")
        scaled_qty = {ing: qty * scale_factor for ing, qty in total_qty.items()}
    else:
        scaled_qty = total_qty

    return [(ing, qty) for ing, qty in scaled_qty.items()]

# ============================================================================
# Recipe Steps v3
# ============================================================================

def generate_steps_v3(ingredients_with_qty):
    """FIX 5: Smarter step generation"""
    steps = []

    proteins = []
    carbs = []
    vegetables = []
    fats = []
    seasonings = []

    for ing, qty in ingredients_with_qty:
        if is_protein_source(ing):
            proteins.append(ing)
        elif is_carb_source(ing):
            carbs.append(ing)
        elif 'oil' in ing or 'butter' in ing:
            fats.append(ing)
        elif ing in ['salt', 'pepper']:
            seasonings.append(ing)
        else:
            vegetables.append(ing)

    step_num = 1

    # Prep vegetables (but not oil)
    prep_veg = [v for v in vegetables if 'oil' not in v]
    if prep_veg:
        steps.append(f"{step_num}. Wash and chop {', '.join(prep_veg)}.")
        step_num += 1

    # Cook carbs (FIX 5: Skip if already cooked)
    if carbs:
        raw_carbs = [c for c in carbs if 'cooked' not in c]
        if raw_carbs:
            steps.append(f"{step_num}. Cook {', '.join(raw_carbs)} until tender.")
            step_num += 1

    # Cook protein
    if proteins:
        steps.append(f"{step_num}. Cook {', '.join(proteins)} until done.")
        step_num += 1

    # Sauté vegetables (if oil available)
    if vegetables and fats:
        steps.append(f"{step_num}. Heat {fats[0]}, sauté vegetables until softened.")
        step_num += 1

    # Combine
    steps.append(f"{step_num}. Combine all ingredients, season with {', '.join(seasonings) if seasonings else 'salt and pepper'}, and serve.")

    return steps

# ============================================================================
# Main Loop
# ============================================================================

print("\n[2/5] Loading user profiles...")
user_profiles = []
with open(USER_PROFILES, 'r') as f:
    for line in f:
        user_profiles.append(json.loads(line))
print(f"  Loaded {len(user_profiles)} profiles")

print("\n[3/5] Generating samples...")
b_class_samples = []

for user_idx, user_profile in enumerate(user_profiles[:10]):
    print(f"\n  User {user_idx+1}/10 (ID={user_profile['user_id']})...")

    user_id = user_profile['user_id']
    liked = user_profile.get('liked_ingredients', [])
    disliked = user_profile.get('disliked_ingredients', [])
    nutrition_targets = user_profile.get('nutrition_targets', {})

    for recipe_idx in range(2):
        print(f"    Recipe {recipe_idx+1}/2...")

        seed = user_id * 100 + recipe_idx

        selected_ingredients = select_ingredients_v3(
            liked, disliked, nutrition_targets, seed
        )

        print(f"      Selected: {selected_ingredients}")

        ingredients_with_qty = optimize_quantities_v3(
            selected_ingredients, nutrition_targets, servings=4
        )

        steps = generate_steps_v3(ingredients_with_qty)

        ingredient_strings = [f"{qty:.0f}g {ing}" for ing, qty in ingredients_with_qty]
        nutrition_result = calc.calculate_recipe_nutrition(ingredient_strings, servings=4)

        title = f"{selected_ingredients[0].title()} with {selected_ingredients[1].title()}"

        sample = {
            'user_id': user_id,
            'recipe_id': f"generated_{user_id}_{recipe_idx}",
            'input': {
                'liked_ingredients': liked,
                'disliked_ingredients': disliked,
                'nutrition_targets': nutrition_targets,
            },
            'output': {
                'title': title,
                'ingredients': [f"{qty:.0f}g {ing}" for ing, qty in ingredients_with_qty],
                'steps': steps,
                'servings': 4,
                'nutrition_per_serv': nutrition_result['per_serving'],
            },
            'metadata': {
                'selected_ingredients': selected_ingredients,
                'nutrition_coverage': nutrition_result['coverage'],
            }
        }

        b_class_samples.append(sample)

        nutr = nutrition_result['per_serving']
        print(f"      ✓ {title}")
        print(f"      Nutrition/serving: {nutr['energy_kcal']:.0f} kcal, "
              f"{nutr['fiber_g']:.1f}g fiber, {nutr['sodium_mg']:.0f}mg sodium")

# Save
output_file = OUTPUT_DIR + 'b_class_v3.jsonl'
with open(output_file, 'w') as f:
    for sample in b_class_samples:
        f.write(json.dumps(sample) + '\n')

print(f"\n[4/5] Saved {len(b_class_samples)} samples to {output_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total samples: {len(b_class_samples)}")
if b_class_samples:
    sample = b_class_samples[0]
    print(f"\nSample:")
    print(f"  Title: {sample['output']['title']}")
    print(f"  Ingredients: {', '.join(sample['output']['ingredients'][:3])}...")
    print(f"  Kcal: {sample['output']['nutrition_per_serv']['energy_kcal']:.0f}")
    print(f"  Fiber: {sample['output']['nutrition_per_serv']['fiber_g']:.1f}g")
    print(f"  Sodium: {sample['output']['nutrition_per_serv']['sodium_mg']:.0f}mg")
print("="*80)
