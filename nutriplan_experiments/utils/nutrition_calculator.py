"""
Nutrition Calculator for NutriPlan
Calculates nutrition values for recipes based on ingredients
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import re


class NutritionCalculator:
    """Calculate nutrition values for recipes"""

    def __init__(self, nutrition_db_path: str = None):
        """
        Args:
            nutrition_db_path: Path to nutrition database JSON file
                              Expected format: {ingredient_name: {nutrient: value_per_100g}}
        """
        self.nutrition_db = {}

        if nutrition_db_path and Path(nutrition_db_path).exists():
            with open(nutrition_db_path, 'r', encoding='utf-8') as f:
                self.nutrition_db = json.load(f)
            print(f"Loaded nutrition database with {len(self.nutrition_db)} ingredients")
        else:
            print("Warning: Nutrition database not found. Using default values.")
            self._init_default_db()

    def _init_default_db(self):
        """Initialize with some default nutrition values"""
        # Common ingredients (values per 100g)
        self.nutrition_db = {
            "chicken breast": {
                "calories": 165, "protein": 31, "fat": 3.6, "carbs": 0,
                "sodium": 74, "fiber": 0, "sugar": 0
            },
            "rice": {
                "calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28,
                "sodium": 1, "fiber": 0.4, "sugar": 0.1
            },
            "broccoli": {
                "calories": 34, "protein": 2.8, "fat": 0.4, "carbs": 7,
                "sodium": 33, "fiber": 2.6, "sugar": 1.7
            },
            "olive oil": {
                "calories": 884, "protein": 0, "fat": 100, "carbs": 0,
                "sodium": 2, "fiber": 0, "sugar": 0
            },
            "lettuce": {
                "calories": 15, "protein": 1.4, "fat": 0.2, "carbs": 2.9,
                "sodium": 28, "fiber": 1.3, "sugar": 0.8
            },
            "tomato": {
                "calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9,
                "sodium": 5, "fiber": 1.2, "sugar": 2.6
            },
            "egg": {
                "calories": 155, "protein": 13, "fat": 11, "carbs": 1.1,
                "sodium": 124, "fiber": 0, "sugar": 1.1
            },
            "salmon": {
                "calories": 208, "protein": 20, "fat": 13, "carbs": 0,
                "sodium": 59, "fiber": 0, "sugar": 0
            }
        }

    def normalize_ingredient_name(self, ingredient: str) -> str:
        """Normalize ingredient name for lookup"""
        # Remove quantities and units
        ingredient = re.sub(r'\d+', '', ingredient)
        ingredient = re.sub(r'\b(cup|cups|tbsp|tsp|oz|lb|g|kg|ml|l|tablespoon|teaspoon)\b', '', ingredient, flags=re.IGNORECASE)
        # Remove extra spaces
        ingredient = ' '.join(ingredient.split())
        return ingredient.lower().strip()

    def parse_quantity(self, ingredient_entry: Any) -> float:
        """
        Parse ingredient quantity to grams

        Args:
            ingredient_entry: Can be string or dict with 'quantity' and 'unit'

        Returns:
            Quantity in grams
        """
        if isinstance(ingredient_entry, dict):
            quantity = ingredient_entry.get('quantity', 1)
            unit = ingredient_entry.get('unit', 'g').lower()
        else:
            # Try to extract from string
            match = re.search(r'(\d+\.?\d*)\s*(\w+)', str(ingredient_entry))
            if match:
                quantity = float(match.group(1))
                unit = match.group(2).lower()
            else:
                return 100  # Default 100g

        # Convert to grams
        conversion = {
            'g': 1,
            'kg': 1000,
            'mg': 0.001,
            'oz': 28.35,
            'lb': 453.6,
            'cup': 240,  # Approximate for liquids
            'tbsp': 15,
            'tablespoon': 15,
            'tsp': 5,
            'teaspoon': 5,
            'ml': 1,  # Approximate (1ml ≈ 1g for water)
            'l': 1000
        }

        return quantity * conversion.get(unit, 1)

    def calculate_ingredient_nutrition(
        self,
        ingredient: Any
    ) -> Dict[str, float]:
        """
        Calculate nutrition for a single ingredient

        Args:
            ingredient: Ingredient (string or dict)

        Returns:
            Nutrition values
        """
        # Get ingredient name and quantity
        if isinstance(ingredient, dict):
            name = ingredient.get('name', '')
        else:
            name = str(ingredient)

        normalized_name = self.normalize_ingredient_name(name)
        quantity_grams = self.parse_quantity(ingredient)

        # Lookup in database
        nutrition_per_100g = self.nutrition_db.get(normalized_name)

        if not nutrition_per_100g:
            # Try fuzzy matching
            for db_ingredient in self.nutrition_db.keys():
                if normalized_name in db_ingredient or db_ingredient in normalized_name:
                    nutrition_per_100g = self.nutrition_db[db_ingredient]
                    break

        if not nutrition_per_100g:
            # Return zeros if not found
            return {
                "calories": 0, "protein": 0, "fat": 0, "carbs": 0,
                "sodium": 0, "fiber": 0, "sugar": 0
            }

        # Scale by quantity
        scale_factor = quantity_grams / 100
        return {
            nutrient: value * scale_factor
            for nutrient, value in nutrition_per_100g.items()
        }

    def calculate_recipe_nutrition(
        self,
        ingredients: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate total nutrition for a recipe

        Args:
            ingredients: List of ingredients

        Returns:
            Total nutrition values
        """
        total_nutrition = {
            "calories": 0, "protein": 0, "fat": 0, "carbs": 0,
            "sodium": 0, "fiber": 0, "sugar": 0
        }

        for ingredient in ingredients:
            ingredient_nutrition = self.calculate_ingredient_nutrition(ingredient)
            for nutrient in total_nutrition:
                total_nutrition[nutrient] += ingredient_nutrition.get(nutrient, 0)

        # Round values
        return {k: round(v, 2) for k, v in total_nutrition.items()}

    def check_constraints(
        self,
        recipe_nutrition: Dict[str, float],
        constraints: Dict[str, Any],
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        Check if recipe nutrition satisfies constraints

        Args:
            recipe_nutrition: Recipe nutrition values
            constraints: Nutrition constraints
            tolerance: Tolerance for constraint satisfaction (10%)

        Returns:
            Dict with constraint check results
        """
        results = {
            'satisfied': True,
            'violations': []
        }

        nutrition_targets = constraints.get('nutrition_targets', {})

        for nutrient, target in nutrition_targets.items():
            if nutrient not in recipe_nutrition:
                continue

            actual = recipe_nutrition[nutrient]
            target_value = target.get('value', target) if isinstance(target, dict) else target
            constraint_type = target.get('type', 'max') if isinstance(target, dict) else 'max'

            violated = False
            violation_msg = ""

            if constraint_type == 'max':
                if actual > target_value * (1 + tolerance):
                    violated = True
                    violation_msg = f"{nutrient}: {actual:.1f} exceeds max {target_value * (1 + tolerance):.1f}"
            elif constraint_type == 'min':
                if actual < target_value * (1 - tolerance):
                    violated = True
                    violation_msg = f"{nutrient}: {actual:.1f} below min {target_value * (1 - tolerance):.1f}"
            elif constraint_type == 'range':
                min_val = target.get('min', 0)
                max_val = target.get('max', float('inf'))
                if not (min_val * (1 - tolerance) <= actual <= max_val * (1 + tolerance)):
                    violated = True
                    violation_msg = f"{nutrient}: {actual:.1f} outside range [{min_val}, {max_val}]"

            if violated:
                results['satisfied'] = False
                results['violations'].append({
                    'nutrient': nutrient,
                    'actual': actual,
                    'target': target_value,
                    'type': constraint_type,
                    'message': violation_msg
                })

        return results


if __name__ == "__main__":
    # Test nutrition calculator
    calculator = NutritionCalculator()

    # Test recipe
    test_recipe = {
        "ingredients": [
            {"name": "chicken breast", "quantity": 200, "unit": "g"},
            {"name": "rice", "quantity": 150, "unit": "g"},
            {"name": "broccoli", "quantity": 100, "unit": "g"},
            {"name": "olive oil", "quantity": 10, "unit": "g"}
        ]
    }

    nutrition = calculator.calculate_recipe_nutrition(test_recipe["ingredients"])
    print("Recipe Nutrition:")
    for nutrient, value in nutrition.items():
        print(f"  {nutrient}: {value:.2f}")

    # Test constraints
    constraints = {
        "nutrition_targets": {
            "calories": {"value": 600, "type": "max"},
            "sodium": {"value": 500, "type": "max"}
        }
    }

    check_result = calculator.check_constraints(nutrition, constraints)
    print(f"\nConstraints satisfied: {check_result['satisfied']}")
    if check_result['violations']:
        print("Violations:")
        for v in check_result['violations']:
            print(f"  {v['message']}")
