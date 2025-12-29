#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test C-Class v3 Generation (Enhanced & Fixed Version)
"""

import json
from generate_c_class_full_v3_ENHANCED_FIXED import (
    generate_c_class_sample,
    parse_recipe_ingredients,
    recalculate_nutrition
)

print("="*80)
print("C-Class v3 Dataset Generation Test (3 Samples)")
print("="*80)

# Load first 3 B-class recipes
B_CLASS_TRAIN = 'work/recipebench/data/10large_scale_datasets/task_b_train_large.jsonl'

print("\n[1/4] Loading B-class samples...")
with open(B_CLASS_TRAIN, 'r', encoding='utf-8') as f:
    b_recipes = [json.loads(line) for i, line in enumerate(f) if i < 3 and line.strip()]

print(f"  Loaded {len(b_recipes)} B-class recipes")

# Test ingredient parsing first
print("\n[2/4] Testing ingredient parser...")
sample_recipe = b_recipes[0]
parsed = parse_recipe_ingredients(sample_recipe['output']['ingredients'])
print(f"  Original ingredients: {len(sample_recipe['output']['ingredients'])}")
print(f"  Parsed ingredients: {len(parsed)}")
for qty, name, orig in parsed[:3]:
    print(f"    {orig} → {qty:.1f}g {name}")

# Test nutrition recalculation
print("\n[3/4] Testing nutrition recalculation...")
recalc_nutrition = recalculate_nutrition(parsed, servings=4)
if recalc_nutrition:
    print(f"  ✓ Nutrition recalculated successfully")
    orig_energy = sample_recipe['output']['nutrition_per_serv']['energy_kcal']
    recalc_energy = recalc_nutrition['per_serving']['energy_kcal']
    print(f"  Original energy: {orig_energy:.0f} kcal")
    print(f"  Recalculated energy: {recalc_energy:.0f} kcal (difference: {abs(orig_energy-recalc_energy):.0f})")
else:
    print(f"  ✗ Nutrition recalculation failed")

# Generate C-class samples
print("\n[4/4] Generating C-class samples...")
success_count = 0

for i, b_recipe in enumerate(b_recipes, 1):
    print(f"\n{'='*70}")
    print(f"Sample {i}: User {b_recipe['user_id']}")
    print(f"{'='*70}")

    # Generate C-class sample
    c_sample = generate_c_class_sample(b_recipe, seed=i*100)

    if c_sample is None:
        print(f"  ✗ Failed to generate C-class sample")
        continue

    success_count += 1

    print(f"\n► Original B-class Recipe:")
    print(f"   Title: {b_recipe['output']['title']}")
    print(f"   Ingredients: {len(b_recipe['output']['ingredients'])} items")
    orig_nutrition = b_recipe['output']['nutrition_per_serv']
    print(f"   Nutrition:")
    print(f"     Energy: {orig_nutrition['energy_kcal']:.0f} kcal")
    print(f"     Protein: {orig_nutrition['protein_g']:.1f}g")
    print(f"     Sodium: {orig_nutrition['sodium_mg']:.0f}mg")
    print(f"     Fiber: {orig_nutrition['fiber_g']:.1f}g")

    print(f"\n► Violated Recipe (Input):")
    violated = c_sample['input']['violated_recipe']
    print(f"   Title: {violated['title']}")
    print(f"   Ingredients: {len(violated['ingredients'])} items")
    violated_nutrition = violated['nutrition_per_serv']
    print(f"   Nutrition:")
    print(f"     Energy: {violated_nutrition['energy_kcal']:.0f} kcal")
    print(f"     Protein: {violated_nutrition['protein_g']:.1f}g")
    print(f"     Sodium: {violated_nutrition['sodium_mg']:.0f}mg")
    print(f"     Fiber: {violated_nutrition['fiber_g']:.1f}g")

    print(f"\n► Violations Detected: {len(c_sample['input']['violations'])}")
    for v in c_sample['input']['violations']:
        if v['type'] == 'nutrition_violation':
            field = v['field']
            if field == 'sodium_mg':
                print(f"     • Sodium: {v['actual']:.0f}mg > {v['limit']:.0f}mg limit ({v['severity']})")
            elif field == 'protein_amdr':
                print(f"     • Protein AMDR: {v.get('actual_pct', 0):.1f}% < {v.get('target_pct', 20):.0f}% target ({v['severity']})")
            elif field == 'fat_amdr':
                print(f"     • Fat AMDR: {v.get('actual_pct', 0):.1f}% > {v.get('target_pct', 30):.0f}% target ({v['severity']})")
            elif field == 'carb_amdr':
                print(f"     • Carb AMDR: {v.get('actual_pct', 0):.1f}% < {v.get('target_pct', 50):.0f}% target ({v['severity']})")
            elif field == 'energy_kcal':
                print(f"     • Energy: {v['actual']:.0f} kcal > {v['target']:.0f} target ({v['severity']})")
            elif field == 'fiber_g':
                print(f"     • Fiber: {v['actual']:.1f}g < {v['minimum']:.0f}g minimum ({v['severity']})")
            elif field == 'saturated_fat_g':
                print(f"     • Saturated Fat: {v.get('actual_pct', 0):.1f}% > {v.get('limit_pct', 10):.0f}% limit ({v['severity']})")
            elif field == 'sugars_g':
                print(f"     • Sugars: {v.get('actual_pct', 0):.1f}% > {v.get('limit_pct', 10):.0f}% limit ({v['severity']})")
        else:
            print(f"     • Preference: {v['subtype']} - {v['ingredient']} ({v['severity']})")

    print(f"\n► Corrections Applied: {len(c_sample['output']['corrections'])}")
    for j, corr in enumerate(c_sample['output']['corrections'], 1):
        action = corr['action']
        if action == 'reduce_quantity':
            print(f"     {j}. Reduce {corr['ingredient_name']}: {corr['original_quantity']:.1f}g → {corr['new_quantity']:.1f}g ({corr['reason']})")
        elif action == 'increase_quantity':
            print(f"     {j}. Increase {corr['ingredient_name']}: {corr['original_quantity']:.1f}g → {corr['new_quantity']:.1f}g ({corr['reason']})")
        elif action == 'remove_ingredient':
            print(f"     {j}. Remove {corr['ingredient_name']} ({corr['reason']})")
        elif action == 'add_ingredient':
            print(f"     {j}. Add {corr['quantity']:.0f}g {corr['ingredient_name']} ({corr['reason']})")

    print(f"\n► Corrected Recipe (Output):")
    corrected = c_sample['output']['corrected_recipe']
    print(f"   Title: {corrected['title']}")
    print(f"   Ingredients: {len(corrected['ingredients'])} items")
    for ing in corrected['ingredients'][:5]:
        print(f"     - {ing}")
    if len(corrected['ingredients']) > 5:
        print(f"     ... ({len(corrected['ingredients']) - 5} more)")

    corrected_nutrition = corrected['nutrition_per_serv']
    print(f"   Nutrition:")
    print(f"     Energy: {corrected_nutrition['energy_kcal']:.0f} kcal")
    print(f"     Protein: {corrected_nutrition['protein_g']:.1f}g")
    print(f"     Sodium: {corrected_nutrition['sodium_mg']:.0f}mg")
    print(f"     Fiber: {corrected_nutrition['fiber_g']:.1f}g")

    # Validate correction effectiveness
    print(f"\n► Validation:")
    targets = c_sample['input']['nutrition_targets']

    violations_fixed = []
    for v in c_sample['input']['violations']:
        if v['type'] == 'nutrition_violation':
            field = v['field']
            if field == 'sodium_mg':
                corrected_sodium = corrected_nutrition['sodium_mg'] * 4
                limit = targets['sodium_mg_max']
                if corrected_sodium <= limit:
                    violations_fixed.append(f"✓ Sodium fixed: {corrected_sodium:.0f}mg ≤ {limit:.0f}mg")
                else:
                    violations_fixed.append(f"✗ Sodium still over: {corrected_sodium:.0f}mg > {limit:.0f}mg")

            elif field == 'protein_amdr':
                protein_kcal = corrected_nutrition['protein_g'] * 4
                energy_kcal = corrected_nutrition['energy_kcal']
                if energy_kcal > 0:
                    protein_pct = (protein_kcal / energy_kcal) * 100
                    target_pct = targets.get('amdr', {}).get('protein', {}).get('target_pct', 20)
                    if protein_pct >= target_pct * 0.9:
                        violations_fixed.append(f"✓ Protein AMDR fixed: {protein_pct:.1f}% ≥ {target_pct:.0f}%")
                    else:
                        violations_fixed.append(f"✗ Protein AMDR still low: {protein_pct:.1f}% < {target_pct:.0f}%")

            elif field == 'fat_amdr':
                fat_kcal = corrected_nutrition['fat_g'] * 9
                energy_kcal = corrected_nutrition['energy_kcal']
                if energy_kcal > 0:
                    fat_pct = (fat_kcal / energy_kcal) * 100
                    target_pct = targets.get('amdr', {}).get('fat', {}).get('target_pct', 30)
                    if fat_pct <= target_pct * 1.2:
                        violations_fixed.append(f"✓ Fat AMDR fixed: {fat_pct:.1f}% ≤ {target_pct:.0f}%")
                    else:
                        violations_fixed.append(f"✗ Fat AMDR still high: {fat_pct:.1f}% > {target_pct:.0f}%")

            elif field == 'carb_amdr':
                carb_kcal = corrected_nutrition['carbohydrates_g'] * 4
                energy_kcal = corrected_nutrition['energy_kcal']
                if energy_kcal > 0:
                    carb_pct = (carb_kcal / energy_kcal) * 100
                    target_pct = targets.get('amdr', {}).get('carb', {}).get('target_pct', 50)
                    if carb_pct >= target_pct * 0.9:
                        violations_fixed.append(f"✓ Carb AMDR fixed: {carb_pct:.1f}% ≥ {target_pct:.0f}%")
                    else:
                        violations_fixed.append(f"✗ Carb AMDR still low: {carb_pct:.1f}% < {target_pct:.0f}%")

            elif field == 'energy_kcal':
                corrected_energy = corrected_nutrition['energy_kcal']
                target = targets.get('energy_kcal_target', 2000) / 4
                if abs(corrected_energy - target) / target < 0.15:
                    violations_fixed.append(f"✓ Energy fixed: {corrected_energy:.0f} ≈ {target:.0f} kcal")
                else:
                    violations_fixed.append(f"✗ Energy still off: {corrected_energy:.0f} vs {target:.0f} kcal")

            elif field == 'fiber_g':
                corrected_fiber = corrected_nutrition['fiber_g'] * 4
                minimum = targets['fiber_g_min']
                if corrected_fiber >= minimum * 0.9:
                    violations_fixed.append(f"✓ Fiber fixed: {corrected_fiber:.1f}g ≥ {minimum:.0f}g")
                else:
                    violations_fixed.append(f"✗ Fiber still low: {corrected_fiber:.1f}g < {minimum:.0f}g")

            elif field == 'saturated_fat_g':
                sat_fat_kcal = corrected_nutrition['saturated_fat_g'] * 9
                energy_kcal = corrected_nutrition['energy_kcal']
                if energy_kcal > 0:
                    sat_fat_pct = (sat_fat_kcal / energy_kcal) * 100
                    if sat_fat_pct <= 10:
                        violations_fixed.append(f"✓ Saturated Fat fixed: {sat_fat_pct:.1f}% ≤ 10%")
                    else:
                        violations_fixed.append(f"✗ Saturated Fat still high: {sat_fat_pct:.1f}% > 10%")

            elif field == 'sugars_g':
                sugars_kcal = corrected_nutrition['sugars_g'] * 4
                energy_kcal = corrected_nutrition['energy_kcal']
                if energy_kcal > 0:
                    sugars_pct = (sugars_kcal / energy_kcal) * 100
                    limit_pct = targets.get('sugars', {}).get('pct_max', 10)
                    if sugars_pct <= limit_pct:
                        violations_fixed.append(f"✓ Sugars fixed: {sugars_pct:.1f}% ≤ {limit_pct:.0f}%")
                    else:
                        violations_fixed.append(f"✗ Sugars still high: {sugars_pct:.1f}% > {limit_pct:.0f}%")

    for fix_msg in violations_fixed:
        print(f"     {fix_msg}")

print(f"\n{'='*80}")
print(f"Test Complete: {success_count}/{len(b_recipes)} samples generated successfully")
print(f"Success Rate: {success_count/len(b_recipes)*100:.1f}%")
print(f"{'='*80}")

if success_count == len(b_recipes):
    print(f"\n✓ All tests passed! Ready for full dataset generation.")
    print(f"\nNext step: Run full generation with:")
    print(f"  python generate_c_class_full_v3_ENHANCED_FIXED.py")
elif success_count >= len(b_recipes) * 0.9:
    print(f"\n✓ Good success rate (≥90%)! Ready for full dataset generation.")
    print(f"\nNext step: Run full generation with:")
    print(f"  python generate_c_class_full_v3_ENHANCED_FIXED.py")
else:
    print(f"\n⚠ Success rate below 90%. Please review the output above.")
