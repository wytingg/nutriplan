#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingredient String Parser & Composer

用于解析和重组ingredient字符串，支持各种单位
"""

import re

# Unit conversion to grams
UNIT_CONVERSIONS = {
    # Spoons
    'tbsp': {'salt': 18, 'pepper': 6.9, 'oil': 13.5, 'default': 15},
    'tablespoon': {'salt': 18, 'pepper': 6.9, 'oil': 13.5, 'default': 15},
    'tsp': {'salt': 6, 'pepper': 2.3, 'oil': 4.5, 'default': 5},
    'teaspoon': {'salt': 6, 'pepper': 2.3, 'oil': 4.5, 'default': 5},

    # Cups
    'cup': {'rice': 185, 'flour': 125, 'oats': 80, 'default': 240},
    'cups': {'rice': 185, 'flour': 125, 'oats': 80, 'default': 240},

    # Items
    'egg': 50,
    'eggs': 50,
    'onion': 150,
    'onions': 150,
    'carrot': 61,
    'carrots': 61,
    'tomato': 123,
    'tomatoes': 123,
}


def parse_fraction(frac_str):
    """Parse fraction string like '1/2' or '1 1/2' to float"""
    frac_str = frac_str.strip()

    # Handle mixed fractions like "1 1/2"
    if ' ' in frac_str:
        parts = frac_str.split()
        whole = float(parts[0])
        frac = parts[1]
        num, denom = frac.split('/')
        return whole + float(num) / float(denom)

    # Handle simple fractions like "1/2"
    if '/' in frac_str:
        num, denom = frac_str.split('/')
        return float(num) / float(denom)

    # Handle whole numbers
    return float(frac_str)


def parse_ingredient_string(ing_str):
    """
    Parse ingredient string to (quantity_in_grams, ingredient_name)

    Examples:
        "500g chicken breast" → (500.0, "chicken breast")
        "2 cups rice" → (370.0, "rice")
        "1/2 tsp salt" → (3.0, "salt")
        "3 large eggs" → (150.0, "eggs")

    Returns:
        (quantity_grams, ingredient_name) or None if parsing fails
    """
    ing_str = ing_str.strip()

    # Pattern: quantity unit ingredient
    # e.g., "2 cups rice", "1/2 tsp salt", "500g chicken"
    pattern = r'^([\d\s/\.]+)\s*([a-zA-Z]+)?\s+(.+)$'
    match = re.match(pattern, ing_str)

    if not match:
        # Try pattern without unit: "500g chicken"
        pattern2 = r'^([\d\s/\.]+)g\s+(.+)$'
        match2 = re.match(pattern2, ing_str)
        if match2:
            qty_str, ing_name = match2.groups()
            qty = parse_fraction(qty_str)
            return (qty, ing_name.strip())
        return None

    qty_str, unit, ing_name = match.groups()
    qty = parse_fraction(qty_str)
    ing_name = ing_name.strip()

    # If no unit or unit is 'g', return as-is
    if not unit or unit.lower() == 'g':
        return (qty, ing_name)

    unit = unit.lower()

    # Convert to grams
    if unit in UNIT_CONVERSIONS:
        conversion = UNIT_CONVERSIONS[unit]

        # Handle dict-based conversions (spoons, cups)
        if isinstance(conversion, dict):
            # Find matching ingredient key
            ing_lower = ing_name.lower()
            for key in conversion:
                if key in ing_lower:
                    grams = qty * conversion[key]
                    return (grams, ing_name)
            # Use default
            grams = qty * conversion['default']
            return (grams, ing_name)
        else:
            # Handle simple conversions (items)
            grams = qty * conversion
            return (grams, ing_name)

    # Unknown unit - assume grams
    return (qty, ing_name)


def compose_ingredient_string(qty_grams, ing_name):
    """
    Compose ingredient string from quantity and name

    Examples:
        (500.0, "chicken breast") → "500g chicken breast"
        (3.0, "salt") → "1/2 tsp salt"
        (370.0, "rice") → "2 cups rice"
    """
    # For small quantities, use appropriate units
    ing_lower = ing_name.lower()

    # Check if it's a spice/seasoning
    if any(spice in ing_lower for spice in ['salt', 'pepper', 'garlic powder', 'onion powder']):
        # Use tsp/tbsp
        if 'salt' in ing_lower:
            g_per_tsp = 6
        elif 'pepper' in ing_lower:
            g_per_tsp = 2.3
        else:
            g_per_tsp = 5

        tsp = qty_grams / g_per_tsp

        if tsp < 0.3:
            return f"1/4 tsp {ing_name}"
        elif tsp < 0.6:
            return f"1/2 tsp {ing_name}"
        elif tsp < 0.9:
            return f"3/4 tsp {ing_name}"
        elif tsp < 1.4:
            return f"1 tsp {ing_name}"
        elif tsp < 2.5:
            return f"2 tsp {ing_name}"
        else:
            tbsp = tsp / 3
            if tbsp < 1.4:
                return f"1 Tbsp {ing_name}"
            elif tbsp < 2.5:
                return f"2 Tbsps {ing_name}"
            else:
                return f"{qty_grams:.0f}g {ing_name}"

    # Check if it's oil/butter
    if 'oil' in ing_lower or 'butter' in ing_lower:
        tbsp = qty_grams / 13.5
        if tbsp < 1.4:
            return f"1 Tbsp {ing_name}"
        elif tbsp < 2.5:
            return f"2 Tbsps {ing_name}"
        elif tbsp < 3.5:
            return f"3 Tbsps {ing_name}"
        elif tbsp < 5:
            return f"1/4 cup {ing_name}"
        else:
            return f"{qty_grams:.0f}g {ing_name}"

    # Check if it's a cup-measurable ingredient
    if any(item in ing_lower for item in ['rice', 'flour', 'oats', 'quinoa', 'cornmeal']):
        if 'rice' in ing_lower:
            g_per_cup = 185
        elif 'flour' in ing_lower:
            g_per_cup = 125
        elif 'oats' in ing_lower:
            g_per_cup = 80
        else:
            g_per_cup = 200

        cups = qty_grams / g_per_cup

        if cups < 0.4:
            return f"1/4 cup {ing_name}"
        elif cups < 0.6:
            return f"1/2 cup {ing_name}"
        elif cups < 0.9:
            return f"3/4 cup {ing_name}"
        elif cups < 1.4:
            return f"1 cup {ing_name}"
        elif cups < 1.75:
            return f"1 1/2 cups {ing_name}"
        elif cups < 2.25:
            return f"2 cups {ing_name}"
        else:
            return f"{int(cups)} cups {ing_name}"

    # Check if it's an item-countable ingredient
    if any(item in ing_lower for item in ['egg', 'onion', 'carrot', 'tomato']):
        if 'egg' in ing_lower:
            g_per_item = 50
            unit = 'large egg'
        elif 'onion' in ing_lower:
            g_per_item = 150
            unit = 'medium onion'
        elif 'carrot' in ing_lower:
            g_per_item = 61
            unit = 'medium carrot'
        elif 'tomato' in ing_lower:
            g_per_item = 123
            unit = 'medium tomato'
        else:
            g_per_item = 100
            unit = 'item'

        count = qty_grams / g_per_item

        if count < 1.4:
            return f"1 {unit} of {ing_name}" if 'of' not in ing_name else f"1 {unit}"
        elif count < 2.5:
            return f"2 {unit}s"
        elif count > 8:
            # Too many items, use grams
            return f"{qty_grams:.0f}g {ing_name}"
        else:
            return f"{int(round(count))} {unit}s"

    # Default: use grams
    return f"{qty_grams:.0f}g {ing_name}"


def test_parser():
    """Test the parser with various ingredient strings"""
    test_cases = [
        "500g chicken breast",
        "2 cups rice",
        "1/2 tsp salt",
        "1 1/2 Tbsps olive oil",
        "3 large eggs",
        "245g kale",
        "1 3/4 cups cornmeal"
    ]

    print("Testing Ingredient Parser:")
    print("="*60)

    for ing_str in test_cases:
        result = parse_ingredient_string(ing_str)
        if result:
            qty, name = result
            reconstructed = compose_ingredient_string(qty, name)
            print(f"Original:  {ing_str}")
            print(f"Parsed:    {qty:.1f}g {name}")
            print(f"Composed:  {reconstructed}")
            print()


if __name__ == '__main__':
    test_parser()
