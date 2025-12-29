#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit conversion system for ingredients
Converts cups, tablespoons, teaspoons, ounces, etc. to grams
"""

# Standard conversions (approximate, based on common ingredient densities)
UNIT_CONVERSIONS = {
    # Volume to grams (assuming water density for liquids, varies by ingredient type)
    'cup': {
        'default': 236,  # 1 cup water ≈ 236g
        'flour': 125,    # 1 cup all-purpose flour ≈ 125g
        'sugar': 200,    # 1 cup granulated sugar ≈ 200g
        'brown sugar': 220,
        'butter': 227,   # 1 cup butter ≈ 227g
        'oil': 218,      # 1 cup oil ≈ 218g
        'milk': 244,
        'water': 236,
        'rice': 185,
        'cheese': 113,   # shredded
        'nuts': 125,
        'liquid': 236,   # generic liquid
        'solid': 150,    # generic solid
    },
    'tablespoon': {
        'default': 15,   # 1 tbsp ≈ 15g (liquid)
        'butter': 14,
        'oil': 13,
        'sugar': 12,
        'flour': 8,
        'salt': 18,
        'liquid': 15,
        'solid': 12,
    },
    'teaspoon': {
        'default': 5,    # 1 tsp ≈ 5g (liquid)
        'salt': 6,
        'sugar': 4,
        'oil': 4.5,
        'liquid': 5,
        'solid': 4,
    },
    'ounce': 28.35,      # 1 oz = 28.35g
    'pound': 453.592,    # 1 lb = 453.592g
    'gram': 1,
    'g': 1,
    'kg': 1000,
    'kilogram': 1000,
    'ml': 1,             # 1 ml ≈ 1g for water-like liquids
    'milliliter': 1,
    'liter': 1000,
    'l': 1000,
    'pint': 473,         # 1 US pint ≈ 473g
    'quart': 946,        # 1 US quart ≈ 946g
    'gallon': 3785,      # 1 US gallon ≈ 3785g
}

# Ingredient category mapping (for better conversion accuracy)
INGREDIENT_CATEGORIES = {
    'flour': ['flour', 'all-purpose flour', 'whole wheat flour', 'bread flour'],
    'sugar': ['sugar', 'granulated sugar', 'white sugar', 'caster sugar'],
    'brown sugar': ['brown sugar', 'light brown sugar', 'dark brown sugar'],
    'butter': ['butter', 'unsalted butter', 'salted butter'],
    'oil': ['oil', 'olive oil', 'vegetable oil', 'canola oil', 'coconut oil'],
    'liquid': ['water', 'milk', 'cream', 'broth', 'stock', 'juice', 'wine', 'sauce'],
    'nuts': ['nuts', 'almonds', 'walnuts', 'pecans', 'peanuts', 'cashews'],
    'cheese': ['cheese', 'cheddar', 'mozzarella', 'parmesan'],
    'rice': ['rice', 'white rice', 'brown rice', 'basmati rice'],
}

def identify_category(ingredient_name):
    """Identify ingredient category for better conversion accuracy"""
    ingredient_lower = ingredient_name.lower()

    for category, keywords in INGREDIENT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in ingredient_lower:
                return category

    # Default fallback
    if any(liquid in ingredient_lower for liquid in ['water', 'milk', 'juice', 'broth', 'stock', 'sauce']):
        return 'liquid'

    return 'default'

def convert_to_grams(quantity, unit, ingredient_name=''):
    """
    Convert ingredient quantity to grams

    Args:
        quantity: numerical amount (float)
        unit: unit string (e.g., 'cup', 'tablespoon', 'oz')
        ingredient_name: ingredient name for category-based conversion

    Returns:
        float: quantity in grams
    """
    if quantity == 0:
        return 0.0

    unit_lower = unit.lower().strip()

    # Handle plural forms
    unit_lower = unit_lower.rstrip('s')

    # Common abbreviations
    abbrev_map = {
        'c': 'cup',
        'tbsp': 'tablespoon',
        'tbs': 'tablespoon',
        'tb': 'tablespoon',
        'tsp': 'teaspoon',
        'ts': 'teaspoon',
        'oz': 'ounce',
        'lb': 'pound',
        'lbs': 'pound',
        'pt': 'pint',
        'qt': 'quart',
        'gal': 'gallon',
    }

    unit_lower = abbrev_map.get(unit_lower, unit_lower)

    # Handle 'item' unit (when no quantity/unit specified, e.g., "eggs")
    if unit_lower == 'item':
        # Assume 100g per item (reasonable for most ingredients)
        return quantity * 100

    # Direct conversions (no category needed)
    simple_units = ['ounce', 'pound', 'gram', 'g', 'kg', 'kilogram',
                    'ml', 'milliliter', 'liter', 'l', 'pint', 'quart', 'gallon']

    if unit_lower in simple_units:
        conversion_factor = UNIT_CONVERSIONS[unit_lower]
        return quantity * conversion_factor

    # Category-based conversions (cup, tablespoon, teaspoon)
    if unit_lower in ['cup', 'tablespoon', 'teaspoon']:
        category = identify_category(ingredient_name)
        conversion_table = UNIT_CONVERSIONS[unit_lower]

        if isinstance(conversion_table, dict):
            conversion_factor = conversion_table.get(category, conversion_table['default'])
        else:
            conversion_factor = conversion_table

        return quantity * conversion_factor

    # Unknown unit - assume it's already in grams or use default
    print(f"[WARNING] Unknown unit '{unit}' for ingredient '{ingredient_name}', assuming 100g")
    return quantity * 100  # Fallback

def test_conversions():
    """Test unit conversions"""
    test_cases = [
        (1, 'cup', 'flour', 125),
        (1, 'cup', 'sugar', 200),
        (1, 'cup', 'butter', 227),
        (1, 'tablespoon', 'butter', 14),
        (1, 'teaspoon', 'salt', 6),
        (1, 'oz', '', 28.35),
        (1, 'pound', '', 453.592),
        (100, 'gram', '', 100),
    ]

    print("Testing unit conversions:")
    print("-" * 60)

    for qty, unit, ingredient, expected in test_cases:
        result = convert_to_grams(qty, unit, ingredient)
        status = "✓" if abs(result - expected) < 0.1 else "✗"
        print(f"{status} {qty} {unit} {ingredient:15s} = {result:7.2f}g (expected {expected}g)")

if __name__ == '__main__':
    test_conversions()
