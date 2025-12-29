#!/usr/bin/env python3
"""
Extract Food.com recipe data into two CSV files:
1. recipe_basic.csv: recipe_id, recipe_name, ingredients, instructions
2. recipe_nutrition.csv: recipe_id, recipe_name, nutrition (per-serving + per-recipe), servings
"""

import pandas as pd
import ast
import re
from pathlib import Path

def parse_r_vector(r_str):
    """Parse R's c() vector syntax to Python list"""
    if pd.isna(r_str) or r_str == 'NA':
        return []

    # Remove c( and trailing )
    r_str = str(r_str).strip()
    if r_str.startswith('c(') and r_str.endswith(')'):
        r_str = r_str[2:-1]

    # Parse quoted strings
    items = []
    in_quote = False
    current = ""
    escape = False

    for char in r_str:
        if escape:
            current += char
            escape = False
            continue

        if char == '\\':
            escape = True
            continue

        if char == '"':
            in_quote = not in_quote
            continue

        if char == ',' and not in_quote:
            if current.strip():
                items.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        items.append(current.strip())

    return items

def main():
    input_file = r"work/recipebench/data/raw/foodcom/recipes.csv"
    output_dir = Path(r"work/recipebench/data/4out")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Loaded {len(df)} recipes")

    # ===== File 1: Basic Recipe Info =====
    print("\nProcessing basic recipe info...")

    # Parse ingredients (combine quantities + parts)
    df['ingredients_list'] = df.apply(
        lambda row: [
            f"{qty} {part}".strip() if qty and part else part
            for qty, part in zip(
                parse_r_vector(row['RecipeIngredientQuantities']),
                parse_r_vector(row['RecipeIngredientParts'])
            )
        ] if pd.notna(row['RecipeIngredientParts']) else [],
        axis=1
    )

    # Join as semicolon-separated string for CSV compatibility
    df['ingredients'] = df['ingredients_list'].apply(lambda x: '; '.join(x) if x else '')

    # Parse instructions
    df['instructions'] = df['RecipeInstructions'].apply(
        lambda x: '; '.join(parse_r_vector(x)) if pd.notna(x) else ''
    )

    basic_df = df[['RecipeId', 'Name', 'ingredients', 'instructions']].copy()
    basic_df.columns = ['recipe_id', 'recipe_name', 'ingredients', 'instructions']

    basic_output = output_dir / 'recipe_basic_foodcom.csv'
    basic_df.to_csv(basic_output, index=False, encoding='utf-8')
    print(f"✓ Saved {len(basic_df)} recipes to {basic_output}")

    # ===== File 2: Nutrition Info =====
    print("\nProcessing nutrition info...")

    # Map Food.com columns to standard names
    nutr_cols = {
        'Calories': 'Calories_kcal',
        'FatContent': 'Fat_g',
        'SaturatedFatContent': 'SaturatedFat_g',
        'CholesterolContent': 'Cholesterol_mg',
        'SodiumContent': 'Sodium_mg',
        'CarbohydrateContent': 'Carbohydrates_g',
        'FiberContent': 'Fiber_g',
        'SugarContent': 'Sugars_g',
        'ProteinContent': 'Protein_g'
    }

    # Ensure numeric and handle servings=0
    df['RecipeServings'] = pd.to_numeric(df['RecipeServings'], errors='coerce').fillna(1)
    df.loc[df['RecipeServings'] == 0, 'RecipeServings'] = 1

    nutrition_df = pd.DataFrame()
    nutrition_df['recipe_id'] = df['RecipeId']
    nutrition_df['recipe_name'] = df['Name']

    # Per-serving nutrition (Food.com values are TOTAL per recipe, need to divide)
    # Based on Food.com documentation, these are total recipe values
    for old_col, new_col in nutr_cols.items():
        per_recipe_col = f'{new_col.split("_")[0]}_PerRecipe_{new_col.split("_")[1]}'
        per_serving_col = f'{new_col.split("_")[0]}_PerServing_{new_col.split("_")[1]}'

        # Total values (as-is from Food.com)
        nutrition_df[per_recipe_col] = pd.to_numeric(df[old_col], errors='coerce')

        # Per-serving values (divide by servings)
        nutrition_df[per_serving_col] = (
            pd.to_numeric(df[old_col], errors='coerce') / df['RecipeServings']
        )

    nutrition_df['RecipeServings'] = df['RecipeServings']

    # Reorder columns: recipe_id, recipe_name, per-serving nutrients, per-recipe nutrients, servings
    per_serving_cols = [col for col in nutrition_df.columns if 'PerServing' in col]
    per_recipe_cols = [col for col in nutrition_df.columns if 'PerRecipe' in col]

    final_cols = ['recipe_id', 'recipe_name'] + per_serving_cols + per_recipe_cols + ['RecipeServings']
    nutrition_df = nutrition_df[final_cols]

    nutrition_output = output_dir / 'recipe_nutrition_foodcom.csv'
    nutrition_df.to_csv(nutrition_output, index=False, encoding='utf-8')
    print(f"✓ Saved {len(nutrition_df)} recipes to {nutrition_output}")
    print(f"\nColumns in nutrition file ({len(final_cols)}): {final_cols}")

    # Show sample
    print("\n=== Sample Output ===")
    print(nutrition_df.head(2))
    print(f"\n✓ Done! Files saved to {output_dir}")

if __name__ == '__main__':
    main()
