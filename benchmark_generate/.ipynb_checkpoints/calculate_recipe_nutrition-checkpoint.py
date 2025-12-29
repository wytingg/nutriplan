#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate nutrition information for recipes
"""

import pandas as pd
import json
import re
from unit_converter import convert_to_grams

class RecipeNutritionCalculator:
    def __init__(self, nutrition_db_path):
        """
        Initialize calculator with nutrition database

        Args:
            nutrition_db_path: path to nutrition_top100_verified.csv
        """
        self.nutrition_db = pd.read_csv(nutrition_db_path)
        print(f"[*] Loaded {len(self.nutrition_db)} ingredients from nutrition database")

        # Create lookup dictionary for fast access
        self.nutrition_lookup = {}
        for _, row in self.nutrition_db.iterrows():
            self.nutrition_lookup[row['ingredient'].lower()] = {
                'energy_kcal': row['energy_kcal'],
                'protein_g': row['protein_g'],
                'fat_g': row['fat_g'],
                'carbohydrates_g': row['carbohydrates_g'],
                'fiber_g': row['fiber_g'],
                'sodium_mg': row['sodium_mg'],
                'sugars_g': row['sugars_g'],
            }

    def parse_ingredient_line(self, ingredient_text):
        """
        Parse ingredient line to extract quantity, unit, and name

        Examples:
            "1 cup butter" -> (1.0, 'cup', 'butter')
            "2 tablespoons olive oil" -> (2.0, 'tablespoons', 'olive oil')
            "1/2 teaspoon salt" -> (0.5, 'teaspoon', 'salt')
            "eggs" -> (1, 'item', 'eggs')

        Returns:
            tuple: (quantity, unit, ingredient_name) or None if parsing fails
        """
        ingredient_text = ingredient_text.strip().lower()

        # Pattern: [quantity] [unit] [ingredient_name]
        # Quantity: number, fraction, or mixed (e.g., "1 1/2")
        pattern = r'^([\d\s/\.]+)\s+([\w]+)\s+(.+)$'
        match = re.match(pattern, ingredient_text)

        if match:
            qty_str, unit, name = match.groups()

            # Parse quantity (handle fractions like "1/2", "1 1/2")
            qty = self._parse_quantity(qty_str)

            return (qty, unit.strip(), name.strip())

        # No quantity/unit specified - assume 1 item
        # Common for "eggs", "onion", etc.
        return (1, 'item', ingredient_text.strip())

    def _parse_quantity(self, qty_str):
        """Parse quantity string (handles fractions like 1/2, 1 1/2)"""
        qty_str = qty_str.strip()

        # Handle mixed fractions like "1 1/2"
        if ' ' in qty_str:
            parts = qty_str.split()
            if len(parts) == 2 and '/' in parts[1]:
                whole = float(parts[0])
                frac = self._parse_fraction(parts[1])
                return whole + frac

        # Handle simple fractions like "1/2"
        if '/' in qty_str:
            return self._parse_fraction(qty_str)

        # Handle decimals like "1.5"
        try:
            return float(qty_str)
        except:
            return 1.0  # Default

    def _parse_fraction(self, frac_str):
        """Parse fraction string like '1/2' -> 0.5"""
        try:
            parts = frac_str.split('/')
            return float(parts[0]) / float(parts[1])
        except:
            return 1.0

    def find_ingredient_in_db(self, ingredient_name):
        """
        Find ingredient in nutrition database (fuzzy matching)

        Returns:
            dict: nutrition info or None if not found
        """
        ingredient_lower = ingredient_name.lower().strip()

        # Exact match
        if ingredient_lower in self.nutrition_lookup:
            return self.nutrition_lookup[ingredient_lower]

        # Fuzzy matching - check if any DB ingredient is substring of query
        for db_ingredient, nutrition in self.nutrition_lookup.items():
            if db_ingredient in ingredient_lower or ingredient_lower in db_ingredient:
                return nutrition

        # Not found
        return None

    def calculate_ingredient_nutrition(self, quantity, unit, ingredient_name):
        """
        Calculate nutrition for a single ingredient

        Args:
            quantity: numerical amount
            unit: unit string
            ingredient_name: ingredient name

        Returns:
            dict: nutrition info in grams or None if ingredient not found
        """
        # Find ingredient in database
        nutrition_per_100g = self.find_ingredient_in_db(ingredient_name)

        if nutrition_per_100g is None:
            return None

        # Convert to grams
        grams = convert_to_grams(quantity, unit, ingredient_name)

        # Scale nutrition from per 100g to actual amount
        scale_factor = grams / 100.0

        nutrition = {}
        for nutrient, value_per_100g in nutrition_per_100g.items():
            nutrition[nutrient] = value_per_100g * scale_factor

        nutrition['grams'] = grams
        return nutrition

    def calculate_recipe_nutrition(self, ingredients_list, servings=1):
        """
        Calculate total nutrition for a recipe

        Args:
            ingredients_list: list of ingredient strings
            servings: number of servings (default 1)

        Returns:
            dict: {
                'total': total nutrition for entire recipe,
                'per_serving': nutrition per serving,
                'coverage': percentage of ingredients with nutrition data,
                'missing_ingredients': list of ingredients not found
            }
        """
        total_nutrition = {
            'energy_kcal': 0,
            'protein_g': 0,
            'fat_g': 0,
            'carbohydrates_g': 0,
            'fiber_g': 0,
            'sodium_mg': 0,
            'sugars_g': 0,
            'grams': 0,
        }

        found_count = 0
        missing_ingredients = []

        for ingredient_text in ingredients_list:
            # Parse ingredient
            parsed = self.parse_ingredient_line(ingredient_text)

            if parsed:
                qty, unit, name = parsed

                # Calculate nutrition
                nutrition = self.calculate_ingredient_nutrition(qty, unit, name)

                if nutrition:
                    # Add to total
                    for nutrient in total_nutrition:
                        total_nutrition[nutrient] += nutrition[nutrient]
                    found_count += 1
                else:
                    missing_ingredients.append(name)

        # Calculate per serving
        per_serving = {}
        for nutrient, total_value in total_nutrition.items():
            per_serving[nutrient] = total_value / servings

        # Calculate coverage
        coverage = found_count / len(ingredients_list) * 100 if len(ingredients_list) > 0 else 0

        return {
            'total': total_nutrition,
            'per_serving': per_serving,
            'coverage': coverage,
            'missing_ingredients': missing_ingredients,
            'servings': servings,
        }

def test_calculator():
    """Test recipe nutrition calculation"""
    print("="*80)
    print("Testing Recipe Nutrition Calculator")
    print("="*80)

    # Initialize calculator
    calc = RecipeNutritionCalculator('work/recipebench/data/raw/foodcom/nutrition_top100_manual.csv')

    # Test ingredient parsing
    print("\n[*] Testing ingredient parsing:")
    test_ingredients = [
        "1 cup butter",
        "2 tablespoons olive oil",
        "1/2 teaspoon salt",
        "3 eggs",
        "1 1/2 cups flour",
    ]

    for ing in test_ingredients:
        parsed = calc.parse_ingredient_line(ing)
        print(f"  '{ing}' -> {parsed}")

    # Test recipe calculation
    print("\n[*] Testing recipe nutrition calculation:")
    recipe_ingredients = [
        "1 cup butter",
        "1 cup sugar",
        "2 eggs",
        "2 cups flour",
        "1 teaspoon salt",
    ]

    result = calc.calculate_recipe_nutrition(recipe_ingredients, servings=12)

    print(f"\n  Total nutrition:")
    print(f"    Calories: {result['total']['energy_kcal']:.0f} kcal")
    print(f"    Protein: {result['total']['protein_g']:.1f} g")
    print(f"    Fat: {result['total']['fat_g']:.1f} g")
    print(f"    Carbs: {result['total']['carbohydrates_g']:.1f} g")

    print(f"\n  Per serving (servings={result['servings']}):")
    print(f"    Calories: {result['per_serving']['energy_kcal']:.0f} kcal")
    print(f"    Protein: {result['per_serving']['protein_g']:.1f} g")
    print(f"    Fat: {result['per_serving']['fat_g']:.1f} g")
    print(f"    Carbs: {result['per_serving']['carbohydrates_g']:.1f} g")

    print(f"\n  Coverage: {result['coverage']:.1f}%")
    if result['missing_ingredients']:
        print(f"  Missing: {', '.join(result['missing_ingredients'])}")

    print("\n" + "="*80)

if __name__ == '__main__':
    test_calculator()
