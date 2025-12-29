#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-Class Full Dataset Generator v7 - FINAL VERSION

Generate 10k train / 2k val / 2k test samples

v7 Fixes:
- MANDATORY protein selection (3-layer fallback)
- Expanded EXCLUDE_KEYWORDS (flour, syrup, condiments)
- Limited vegetable portions (150g max per serving)
- Conservative fiber vegetable selection
- Deduplication check for ingredients
"""

import pandas as pd
import json
import sys
import random
from pathlib import Path
from collections import defaultdict
from calculate_recipe_nutrition import RecipeNutritionCalculator
from ingredient_normalizer import normalize_ingredient, fuzzy_match_ingredient
from household_units_converter import convert_to_household_unit
from tqdm import tqdm
import time

print("="*80)
print("B-Class Full Dataset Generator v7 (10k/2k/2k)")
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

# Dataset sizes
TRAIN_SIZE = 10000
VAL_SIZE = 2000
TEST_SIZE = 2000
TOTAL_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

# Keywords
PROTEIN_KEYWORDS = [
    'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck',
    'fish', 'snapper', 'fillet', 'salmon', 'tuna', 'cod', 'tilapia', 'halibut',
    'shrimp', 'prawn', 'crab', 'lobster', 'scallop',
    'egg', 'tofu', 'tempeh', 'seitan',
    'bean', 'lentil', 'chickpea'
]

CARB_KEYWORDS = [
    'rice', 'pasta', 'noodle', 'spaghetti', 'linguine', 'penne', 'fettuccine',
    'macaroni', 'rigatoni', 'orzo', 'lasagna', 'ravioli', 'tortellini',
    'bread', 'baguette', 'roll', 'bun', 'tortilla', 'pita',
    'potato', 'sweet potato', 'yam',
    'quinoa', 'oat', 'oatmeal', 'corn', 'cornmeal', 'barley', 'couscous',
    'bulgur', 'millet', 'farro'
]

HERB_KEYWORDS = [
    'cilantro', 'coriander', 'parsley', 'basil', 'mint', 'dill', 'chive',
    'rosemary', 'thyme', 'oregano', 'sage', 'tarragon', 'marjoram'
]

HIGH_FIBER_VEGETABLES = ['broccoli', 'spinach', 'kale', 'brussels sprout',
                         'carrot', 'cauliflower', 'sweet potato', 'bean']

# Exclude non-vegetable items (seasonings, cheese, sauces, bouillon, bones, flour, syrups, condiments, etc.)
EXCLUDE_KEYWORDS = [
    'cheese', 'cheddar', 'parmesan', 'mozzarella', 'ricotta', 'feta', 'pecorino',
    'seasoning', 'spice', 'powder', 'onion powder', 'garlic powder',
    'bouillon', 'cubes', 'bouillon cubes', 'stock', 'broth',
    'sauce', 'paste', 'extract', 'vinegar', 'wine', 'liquor', 'beer',
    'cream', 'milk', 'yogurt', 'butter', 'margarine',
    'gelatin', 'yeast', 'baking powder', 'baking soda',
    'sugar', 'honey', 'syrup', 'molasses', 'corn syrup', 'light corn syrup', 'dark corn syrup',
    'flour', 'bread flour', 'gluten flour', 'wheat flour', 'all-purpose flour',
    'wheat germ', 'wheat bran', 'cornstarch',
    'ham bone', 'beef bone', 'pork bone', 'bone',
    'mayonnaise', 'miracle whip', 'whip', 'ketchup', 'mustard', 'relish',
    'jam', 'jelly', 'preserves', 'marmalade'
]

# ============================================================================
# Load KG
# ============================================================================

print("\n[1/6] Loading KG and nutrition database...")
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

print(f"  [OK] Loaded KG with {len(calc.nutrition_lookup)} ingredients")

# ============================================================================
# Helper Functions
# ============================================================================

def is_protein_source(ing):
    return any(keyword in ing for keyword in PROTEIN_KEYWORDS)

def is_carb_source(ing):
    return any(keyword in ing for keyword in CARB_KEYWORDS)

def is_herb(ing):
    """Check if ingredient is herb/cilantro/parsley"""
    return any(herb in ing for herb in HERB_KEYWORDS)

def is_high_fiber(ing):
    return any(veg in ing for veg in HIGH_FIBER_VEGETABLES)

def should_exclude(ing):
    """Check if ingredient should be excluded (cheese, seasoning, sauce, etc.)"""
    return any(keyword in ing for keyword in EXCLUDE_KEYWORDS)

def is_vegetable(ing):
    """True vegetable check - not protein, carb, or excluded items"""
    return (not is_protein_source(ing) and
            not is_carb_source(ing) and
            not should_exclude(ing))

def get_sodium_per_100g(ing):
    nutrition = calc.find_ingredient_in_db(ing)
    if nutrition:
        return nutrition.get('sodium_mg', 0)
    return 0

def is_high_sodium(ing):
    return get_sodium_per_100g(ing) > 500

def is_catastrophic_sodium(ing):
    """Filter out ingredients with >8000mg sodium per 100g (bouillon, etc.)"""
    return get_sodium_per_100g(ing) > 8000

def get_fiber_per_100g(ing):
    """Get fiber content from database"""
    nutrition = calc.find_ingredient_in_db(ing)
    if nutrition:
        return nutrition.get('fiber_g', 0)
    return 0

def simplify_ingredient_name(ing):
    """Simplify ingredient names by removing modifiers"""
    modifiers_to_remove = [
        'boneless', 'skinless', 'bone-in', 'skin-on',
        'fresh', 'frozen', 'dried', 'canned',
        'raw', 'cooked', 'uncooked',
        'large', 'medium', 'small', 'extra-large',
        'whole', 'halved', 'quartered', 'chopped', 'diced', 'sliced',
        'low-sodium', 'low-fat', 'fat-free', 'reduced-fat'
    ]

    words = ing.split()
    simplified = []

    for word in words:
        if word.lower() not in modifiers_to_remove:
            simplified.append(word)

    result = ' '.join(simplified)

    if 'chicken breast halves' in result:
        result = result.replace('chicken breast halves', 'chicken breast')

    return result if result else ing

def normalize_for_dedup(ing):
    """Normalize ingredient name for deduplication (handle plurals, modifiers, etc.)"""
    normalized = ing.lower().strip()

    modifiers = [
        'fresh', 'frozen', 'dried', 'canned', 'raw', 'cooked',
        'chopped', 'diced', 'sliced', 'minced', 'crushed',
        'large', 'medium', 'small', 'baby',
        'italian-style', 'greek-style', 'mexican-style'
    ]
    for mod in modifiers:
        normalized = normalized.replace(mod + ' ', '').replace(' ' + mod, '')

    # Handle plurals: carrots -> carrot
    if normalized.endswith('ies'):
        normalized = normalized[:-3] + 'y'
    elif normalized.endswith('es') and not normalized.endswith('ses'):
        normalized = normalized[:-2]
    elif normalized.endswith('s') and len(normalized) > 3:
        normalized = normalized[:-1]

    return normalized.strip()

def is_duplicate_ingredient(new_ing, existing_ingredients):
    """Check if new ingredient is a duplicate of any existing ingredient"""
    new_norm = normalize_for_dedup(new_ing)
    for existing in existing_ingredients:
        existing_norm = normalize_for_dedup(existing)
        if new_norm == existing_norm or new_norm in existing_norm or existing_norm in new_norm:
            return True
    return False

# ============================================================================
# Ingredient Selection
# ============================================================================

def match_user_preferences_to_db(user_ingredients, database_ingredients):
    matched = []
    match_details = []

    for user_ing in user_ingredients:
        match, score = fuzzy_match_ingredient(user_ing, database_ingredients, threshold=0.6)
        if match:
            matched.append(match.lower())
            match_details.append({
                'user_input': user_ing,
                'matched': match,
                'score': score
            })
        else:
            matched.append(user_ing.lower())
            match_details.append({
                'user_input': user_ing,
                'matched': user_ing,
                'score': 0
            })

    return matched, match_details

def select_ingredients(liked, disliked, nutrition_targets, seed=0):
    random.seed(seed)

    db_ingredients = list(calc.nutrition_lookup.keys())

    raw_liked_names = [ing['name'] for ing in liked]
    raw_disliked_names = [ing['name'] for ing in disliked]

    liked_names, liked_matches = match_user_preferences_to_db(raw_liked_names, db_ingredients)
    disliked_names, _ = match_user_preferences_to_db(raw_disliked_names, db_ingredients)

    sugars_info = nutrition_targets.get('sugars', {})
    restrict_sugar = sugars_info and sugars_info.get('pct_max', 100) <= 10

    sodium_max = nutrition_targets.get('sodium_mg_max', 2300)
    restrict_sodium = sodium_max <= 1500

    selected = []

    def is_valid(ing):
        if ing in disliked_names or any(d in ing for d in disliked_names):
            return False
        # Always exclude forbidden items (flour, bouillon, syrup, etc.)
        if should_exclude(ing):
            return False
        if restrict_sugar and 'sugar' in ing:
            return False
        if restrict_sodium and is_high_sodium(ing):
            return False
        # Always filter catastrophic sodium (bouillon, etc.)
        if is_catastrophic_sodium(ing):
            return False
        if not calc.find_ingredient_in_db(ing):
            return False
        return True

    # STEP 1: Find liked protein (MANDATORY - every recipe needs protein)
    liked_proteins = [ing for ing in liked_names
                     if is_protein_source(ing) and is_valid(ing)]

    protein_selected = None
    if liked_proteins:
        protein_selected = random.choice(liked_proteins)
    else:
        # Fallback: select from common proteins in database
        common_proteins = ['chicken breast', 'chicken', 'beef', 'pork', 'salmon',
                          'tilapia', 'cod', 'tofu', 'eggs', 'turkey']
        for protein in common_proteins:
            if is_valid(protein) and calc.find_ingredient_in_db(protein):
                protein_selected = protein
                break

        # Final fallback: any protein from database
        if not protein_selected:
            available = [ing for ing in calc.nutrition_lookup.keys()
                        if is_protein_source(ing) and is_valid(ing)]
            if available:
                protein_selected = random.choice(available[:20])
            else:
                protein_selected = 'chicken'  # Last resort

    selected.append(protein_selected)

    # STEP 2: Find liked carb
    liked_carbs = [ing for ing in liked_names
                  if is_carb_source(ing) and is_valid(ing)]

    if liked_carbs:
        selected.append(random.choice(liked_carbs))
    else:
        available = [ing for ing in calc.nutrition_lookup.keys()
                    if is_carb_source(ing) and is_valid(ing)]
        if available:
            selected.append(random.choice(available[:20]))
        else:
            selected.append('rice')

    # STEP 3: Add high-fiber vegetables (0-2 items based on fiber target)
    fiber_min = nutrition_targets.get('fiber_g_min', 25)

    # Sort high-fiber vegetables by fiber content
    available_fiber = [(ing, get_fiber_per_100g(ing))
                      for ing in calc.nutrition_lookup.keys()
                      if is_high_fiber(ing) and is_valid(ing) and is_vegetable(ing)]
    available_fiber.sort(key=lambda x: x[1], reverse=True)

    # Conservative fiber vegetable selection based on target
    # fiber >= 40g: select 2 high-fiber veggies
    # fiber 25-40g: select 1 high-fiber veggie
    # fiber < 25g: select 0 (will add normal veggies later)
    if fiber_min >= 40:
        target_count = 2
    elif fiber_min >= 25:
        target_count = 1
    else:
        target_count = 0

    count = 0
    for ing, fiber in available_fiber[:8]:
        if count < target_count and ing not in selected and not is_duplicate_ingredient(ing, selected):
            selected.append(ing)
            count += 1

    # STEP 4: Add 1-2 more vegetables based on fiber target (true vegetables only, exclude herbs)
    # Conservative approach: limit total vegetables to prevent fiber overflow
    # fiber_min < 30g: add only 1 more vegetable
    # fiber_min >= 30g: add up to 2 more vegetables
    max_additional_veggies = 2 if fiber_min >= 30 else 1
    veggies_added = 0

    for seed_ing in selected[:2]:
        if veggies_added >= max_additional_veggies:
            break

        cooccur_list = cooccur_dict.get(seed_ing, [])
        cooccur_list = sorted(cooccur_list, key=lambda x: x['score'], reverse=True)

        for item in cooccur_list[:30]:
            co_ing = item['ingredient']
            if (co_ing not in selected and is_valid(co_ing) and
                is_vegetable(co_ing) and not is_herb(co_ing) and
                not is_duplicate_ingredient(co_ing, selected)):
                selected.append(co_ing)
                veggies_added += 1
                break

    # STEP 5: Add essentials
    essentials = ['olive oil', 'salt', 'pepper']
    for ing in essentials:
        if len(selected) >= 7:
            break
        if ing not in selected and is_valid(ing):
            selected.append(ing)

    return selected[:7], liked_matches

# ============================================================================
# Quantity Optimization
# ============================================================================

def optimize_quantities(ingredients, nutrition_targets, servings=4, max_iterations=5):
    target_kcal_total = nutrition_targets.get('energy_kcal_target', 2000)
    target_kcal_per_serving = target_kcal_total / servings

    amdr = nutrition_targets.get('amdr', {})
    target_protein_pct = amdr.get('protein', {}).get('target_pct', 20)
    target_fat_pct = amdr.get('fat', {}).get('target_pct', 30)
    target_carb_pct = amdr.get('carb', {}).get('target_pct', 50)

    base_qty = {}
    for ing in ingredients:
        if is_protein_source(ing):
            base_qty[ing] = 120
        elif is_carb_source(ing):
            base_qty[ing] = 80
        elif 'oil' in ing or 'butter' in ing:
            base_qty[ing] = 8
        elif ing == 'salt':
            base_qty[ing] = 0.5
        elif ing == 'pepper':
            base_qty[ing] = 0.25
        elif 'garlic' in ing:
            base_qty[ing] = 8
        elif 'onion' in ing:
            base_qty[ing] = 40
        elif is_herb(ing):
            base_qty[ing] = 5
        else:
            base_qty[ing] = 70

    for iteration in range(max_iterations):
        total_qty = {ing: qty * servings for ing, qty in base_qty.items()}

        ingredient_strings = [f"{qty}g {ing}" for ing, qty in total_qty.items()]
        nutrition_result = calc.calculate_recipe_nutrition(ingredient_strings, servings)

        per_serv = nutrition_result['per_serving']
        actual_kcal = per_serv['energy_kcal']
        actual_protein_g = per_serv['protein_g']
        actual_fat_g = per_serv['fat_g']
        actual_carb_g = per_serv['carbohydrates_g']

        protein_kcal = actual_protein_g * 4
        fat_kcal = actual_fat_g * 9
        carb_kcal = actual_carb_g * 4
        total_macro_kcal = protein_kcal + fat_kcal + carb_kcal

        if total_macro_kcal > 0:
            actual_protein_pct = (protein_kcal / total_macro_kcal) * 100
            actual_fat_pct = (fat_kcal / total_macro_kcal) * 100
            actual_carb_pct = (carb_kcal / total_macro_kcal) * 100
        else:
            actual_protein_pct = actual_fat_pct = actual_carb_pct = 0

        energy_error = abs(actual_kcal - target_kcal_per_serving) / target_kcal_per_serving
        protein_error = abs(actual_protein_pct - target_protein_pct)
        fat_error = abs(actual_fat_pct - target_fat_pct)

        if energy_error < 0.1 and protein_error < 3 and fat_error < 3:
            break

        if actual_kcal > 0:
            energy_scale = target_kcal_per_serving / actual_kcal
            energy_scale = max(0.3, min(3.0, energy_scale))
        else:
            energy_scale = 1.0

        protein_adjustment = 1.0
        fat_adjustment = 1.0
        carb_adjustment = 1.0

        if actual_protein_pct < target_protein_pct - 2:
            protein_adjustment = 1.2
        elif actual_protein_pct > target_protein_pct + 2:
            protein_adjustment = 0.85

        if actual_fat_pct > target_fat_pct + 2:
            fat_adjustment = 0.8
        elif actual_fat_pct < target_fat_pct - 2:
            fat_adjustment = 1.15

        if actual_carb_pct < target_carb_pct - 2:
            carb_adjustment = 1.15
        elif actual_carb_pct > target_carb_pct + 2:
            carb_adjustment = 0.85

        for ing in base_qty:
            base_qty[ing] *= energy_scale

            if is_protein_source(ing):
                base_qty[ing] *= protein_adjustment
            elif 'oil' in ing or 'butter' in ing:
                base_qty[ing] *= fat_adjustment
            elif is_carb_source(ing):
                base_qty[ing] *= carb_adjustment

        for ing in base_qty:
            if ing == 'salt':
                base_qty[ing] = max(0.3, min(0.8, base_qty[ing]))
            elif ing == 'pepper':
                base_qty[ing] = max(0.15, min(0.5, base_qty[ing]))
            elif 'garlic' in ing:
                base_qty[ing] = max(2, min(15, base_qty[ing]))
            elif 'oil' in ing or 'butter' in ing:
                base_qty[ing] = max(3, min(20, base_qty[ing]))
            elif is_herb(ing):
                base_qty[ing] = max(2, min(8, base_qty[ing]))
            elif is_protein_source(ing):
                base_qty[ing] = max(30, min(200, base_qty[ing]))
            elif is_carb_source(ing):
                base_qty[ing] = max(20, min(150, base_qty[ing]))
            else:
                # Vegetables: limit to 150g per serving to prevent fiber overflow
                base_qty[ing] = max(10, min(150, base_qty[ing]))

    final_qty = {ing: qty * servings for ing, qty in base_qty.items()}
    return [(ing, qty) for ing, qty in final_qty.items()]

# ============================================================================
# Recipe Steps
# ============================================================================

def generate_steps(ingredients_with_qty):
    steps = []

    proteins = []
    carbs = []
    vegetables = []
    herbs = []
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
        elif is_herb(ing):
            herbs.append(ing)
        else:
            vegetables.append(ing)

    step_num = 1

    # Prep vegetables (not herbs)
    prep_veg = [v for v in vegetables if 'oil' not in v]
    if prep_veg:
        steps.append(f"{step_num}. Wash and chop {', '.join(prep_veg)}.")
        step_num += 1

    # Prep herbs separately
    if herbs:
        steps.append(f"{step_num}. Rinse and chop {', '.join(herbs)}.")
        step_num += 1

    # Cook carbs
    if carbs:
        raw_carbs = [c for c in carbs if 'cooked' not in c]
        if raw_carbs:
            steps.append(f"{step_num}. Cook {', '.join(raw_carbs)} until tender.")
            step_num += 1

    # Cook protein
    if proteins:
        steps.append(f"{step_num}. Cook {', '.join(proteins)} until done.")
        step_num += 1

    # Saute vegetables
    if vegetables and fats:
        steps.append(f"{step_num}. Heat {fats[0]}, saute vegetables until softened.")
        step_num += 1

    # Combine with herbs and seasonings
    garnish_items = []
    if herbs:
        garnish_items.extend(herbs)
    if seasonings:
        garnish_items.extend(seasonings)
    elif not seasonings:
        garnish_items.append('salt and pepper')

    steps.append(f"{step_num}. Combine all ingredients, garnish with {', '.join(garnish_items)}, and serve.")

    return steps

# ============================================================================
# Recipe Generation
# ============================================================================

def generate_recipe(user_profile, seed=0):
    """Generate single recipe"""
    try:
        user_id = user_profile['user_id']
        liked = user_profile['liked_ingredients']
        disliked = user_profile['disliked_ingredients']
        nutrition_targets = user_profile['nutrition_targets']

        ingredients, liked_matches = select_ingredients(liked, disliked, nutrition_targets, seed)

        servings = 4
        ingredients_with_qty = optimize_quantities(ingredients, nutrition_targets, servings)

        steps = generate_steps(ingredients_with_qty)

        ingredient_strings = [f"{qty}g {ing}" for ing, qty in ingredients_with_qty]
        nutrition_result = calc.calculate_recipe_nutrition(ingredient_strings, servings)

        ingredients_natural_units = []
        for ing, qty_grams in ingredients_with_qty:
            natural_unit = convert_to_household_unit(ing, qty_grams, servings)
            ingredients_natural_units.append(natural_unit)

        # Create title with simplified ingredient names
        protein_ings = [ing for ing in ingredients if is_protein_source(ing)]
        carb_ings = [ing for ing in ingredients if is_carb_source(ing)]
        veg_ings = [ing for ing in ingredients if is_vegetable(ing) and not is_herb(ing)]

        title_parts = []
        if protein_ings:
            title_parts.append(simplify_ingredient_name(protein_ings[0]))

        # 60% chance to use carb, 40% chance to use vegetable (for variety)
        if carb_ings and (not veg_ings or random.random() < 0.6):
            title_parts.append(simplify_ingredient_name(carb_ings[0]))
        elif veg_ings:
            title_parts.append(simplify_ingredient_name(veg_ings[0]))

        title = " with ".join([ing.title() for ing in title_parts]) if len(title_parts) >= 2 else (title_parts[0].title() if title_parts else "Mixed Dish")

        recipe = {
            'user_id': user_id,
            'recipe_id': f'generated_{user_id}_{seed}',
            'input': {
                'liked_ingredients': liked,
                'disliked_ingredients': disliked,
                'nutrition_targets': nutrition_targets
            },
            'output': {
                'title': title,
                'ingredients': ingredients_natural_units,
                'steps': steps,
                'servings': servings,
                'nutrition_per_serv': nutrition_result['per_serving']
            },
            'metadata': {
                'selected_ingredients': ingredients,
                'liked_matches': liked_matches,
                'nutrition_coverage': 100.0
            }
        }

        return recipe

    except Exception as e:
        print(f"\n[ERROR] Failed to generate recipe for user {user_profile.get('user_id', 'unknown')}: {e}")
        return None

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f"\n[2/6] Loading user profiles...")
    with open(USER_PROFILES, 'r', encoding='utf-8') as f:
        all_users = [json.loads(line) for line in f]

    print(f"  [OK] Loaded {len(all_users)} users")

    print(f"\n[3/6] Sampling {TOTAL_SIZE} users for dataset...")
    random.seed(42)

    if len(all_users) < TOTAL_SIZE:
        print(f"  [WARNING] Only {len(all_users)} users available, using all")
        selected_users = all_users
    else:
        selected_users = random.sample(all_users, TOTAL_SIZE)

    # Split into train/val/test
    train_users = selected_users[:TRAIN_SIZE]
    val_users = selected_users[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
    test_users = selected_users[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE]

    print(f"  [OK] Split: {len(train_users)} train / {len(val_users)} val / {len(test_users)} test")

    # Generate datasets
    splits = [
        ('train', train_users, f'{OUTPUT_DIR}task_b_train_large.jsonl'),
        ('val', val_users, f'{OUTPUT_DIR}task_b_val_large.jsonl'),
        ('test', test_users, f'{OUTPUT_DIR}task_b_test_large.jsonl'),
    ]

    for split_name, users, output_file in splits:
        print(f"\n[4/6] Generating {split_name} set ({len(users)} samples)...")

        recipes = []
        failed_count = 0

        for user in tqdm(users, desc=f"  Generating {split_name}"):
            recipe = generate_recipe(user, seed=0)
            if recipe:
                recipes.append(recipe)
            else:
                failed_count += 1

        print(f"\n[5/6] Saving {split_name} set to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for recipe in recipes:
                f.write(json.dumps(recipe, ensure_ascii=False) + '\n')

        print(f"  [OK] Saved {len(recipes)} recipes ({failed_count} failed)")

    print(f"\n[6/6] Done!")
    print("="*80)
    print(f"Generated datasets:")
    print(f"  Train: {OUTPUT_DIR}task_b_train_large.jsonl")
    print(f"  Val:   {OUTPUT_DIR}task_b_val_large.jsonl")
    print(f"  Test:  {OUTPUT_DIR}task_b_test_large.jsonl")
    print("="*80)
