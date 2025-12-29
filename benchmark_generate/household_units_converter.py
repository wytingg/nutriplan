#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Household Units Converter - 家用单位转换器

Convert gram quantities to natural household units for recipe display
"""

# ============================================================================
# Standard Volume Conversions (基础容量单位)
# ============================================================================
# 1 cup = 240ml
# 1 tablespoon (Tbsp) = 15ml
# 1 teaspoon (tsp) = 5ml

# ============================================================================
# Item Counts (按"个"计量的食材)
# ============================================================================
ITEM_UNITS = {
    # Eggs - 鸡蛋
    'egg': {'grams_per_item': 50, 'unit': 'large egg', 'unit_cn': '个大鸡蛋'},
    'eggs': {'grams_per_item': 50, 'unit': 'large egg', 'unit_cn': '个大鸡蛋'},

    # Common produce items
    'onion': {'grams_per_item': 150, 'unit': 'medium onion', 'unit_cn': '个中等洋葱'},
    'onions': {'grams_per_item': 150, 'unit': 'medium onion', 'unit_cn': '个中等洋葱'},
    'potato': {'grams_per_item': 150, 'unit': 'medium potato', 'unit_cn': '个中等土豆'},
    'potatoes': {'grams_per_item': 150, 'unit': 'medium potato', 'unit_cn': '个中等土豆'},
    'tomato': {'grams_per_item': 123, 'unit': 'medium tomato', 'unit_cn': '个中等番茄'},
    'tomatoes': {'grams_per_item': 123, 'unit': 'medium tomato', 'unit_cn': '个中等番茄'},
    'carrot': {'grams_per_item': 61, 'unit': 'medium carrot', 'unit_cn': '根中等胡萝卜'},
    'carrots': {'grams_per_item': 61, 'unit': 'medium carrot', 'unit_cn': '根中等胡萝卜'},
    'lemon': {'grams_per_item': 58, 'unit': 'medium lemon', 'unit_cn': '个中等柠檬'},
    'lemons': {'grams_per_item': 58, 'unit': 'medium lemon', 'unit_cn': '个中等柠檬'},
    'lime': {'grams_per_item': 44, 'unit': 'medium lime', 'unit_cn': '个中等青柠'},
    'limes': {'grams_per_item': 44, 'unit': 'medium lime', 'unit_cn': '个中等青柠'},
}

# ============================================================================
# Spoon Measures (勺子单位 - 调料/香料)
# ============================================================================
# Format: ingredient -> grams_per_tablespoon
SPOON_UNITS = {
    # Salt & Pepper
    'salt': {'g_per_tbsp': 18, 'g_per_tsp': 6, 'prefer': 'tsp'},
    'pepper': {'g_per_tbsp': 6.9, 'g_per_tsp': 2.3, 'prefer': 'tsp'},
    'black pepper': {'g_per_tbsp': 6.9, 'g_per_tsp': 2.3, 'prefer': 'tsp'},

    # Dried herbs & spices
    'garlic powder': {'g_per_tbsp': 9.4, 'g_per_tsp': 3.1, 'prefer': 'tsp'},
    'onion powder': {'g_per_tbsp': 6.8, 'g_per_tsp': 2.3, 'prefer': 'tsp'},
    'paprika': {'g_per_tbsp': 6.9, 'g_per_tsp': 2.3, 'prefer': 'tsp'},
    'cumin': {'g_per_tbsp': 6, 'g_per_tsp': 2, 'prefer': 'tsp'},
    'oregano': {'g_per_tbsp': 2.7, 'g_per_tsp': 0.9, 'prefer': 'tsp'},
    'dried oregano': {'g_per_tbsp': 2.7, 'g_per_tsp': 0.9, 'prefer': 'tsp'},
    'basil': {'g_per_tbsp': 1.8, 'g_per_tsp': 0.6, 'prefer': 'tsp'},
    'dried basil': {'g_per_tbsp': 1.8, 'g_per_tsp': 0.6, 'prefer': 'tsp'},
    'thyme': {'g_per_tbsp': 2.7, 'g_per_tsp': 0.9, 'prefer': 'tsp'},
    'dried thyme': {'g_per_tbsp': 2.7, 'g_per_tsp': 0.9, 'prefer': 'tsp'},
    'rosemary': {'g_per_tbsp': 1.8, 'g_per_tsp': 0.6, 'prefer': 'tsp'},
    'dried rosemary': {'g_per_tbsp': 1.8, 'g_per_tsp': 0.6, 'prefer': 'tsp'},
    'cinnamon': {'g_per_tbsp': 7.8, 'g_per_tsp': 2.6, 'prefer': 'tsp'},
    'ginger': {'g_per_tbsp': 5.4, 'g_per_tsp': 1.8, 'prefer': 'tsp'},
    'ground ginger': {'g_per_tbsp': 5.4, 'g_per_tsp': 1.8, 'prefer': 'tsp'},
    'chili powder': {'g_per_tbsp': 7.5, 'g_per_tsp': 2.5, 'prefer': 'tsp'},
    'cayenne pepper': {'g_per_tbsp': 5.3, 'g_per_tsp': 1.8, 'prefer': 'tsp'},

    # Fresh herbs (larger quantities, prefer tablespoon)
    'fresh parsley': {'g_per_tbsp': 3.8, 'g_per_tsp': 1.3, 'prefer': 'tbsp'},
    'parsley': {'g_per_tbsp': 3.8, 'g_per_tsp': 1.3, 'prefer': 'tbsp'},
    'fresh basil': {'g_per_tbsp': 2.5, 'g_per_tsp': 0.8, 'prefer': 'tbsp'},
    'fresh cilantro': {'g_per_tbsp': 1, 'g_per_tsp': 0.3, 'prefer': 'tbsp'},
    'cilantro': {'g_per_tbsp': 1, 'g_per_tsp': 0.3, 'prefer': 'tbsp'},

    # Minced aromatics
    'garlic': {'g_per_tbsp': 8.5, 'g_per_tsp': 2.8, 'prefer': 'tsp', 'note': 'minced'},
    'fresh ginger': {'g_per_tbsp': 8, 'g_per_tsp': 2.7, 'prefer': 'tsp', 'note': 'minced'},
    'gingerroot': {'g_per_tbsp': 8, 'g_per_tsp': 2.7, 'prefer': 'tsp', 'note': 'minced'},
}

# ============================================================================
# Cup Measures (杯子单位 - 干货/液体)
# ============================================================================
CUP_UNITS = {
    # Grains & Flours
    'flour': {'g_per_cup': 125, 'prefer': 'cup'},
    'all-purpose flour': {'g_per_cup': 125, 'prefer': 'cup'},
    'bread flour': {'g_per_cup': 127, 'prefer': 'cup'},
    'whole wheat flour': {'g_per_cup': 120, 'prefer': 'cup'},
    'rice': {'g_per_cup': 185, 'prefer': 'cup'},
    'white rice': {'g_per_cup': 185, 'prefer': 'cup'},
    'brown rice': {'g_per_cup': 195, 'prefer': 'cup'},
    'quinoa': {'g_per_cup': 170, 'prefer': 'cup'},
    'oats': {'g_per_cup': 80, 'prefer': 'cup'},
    'rolled oats': {'g_per_cup': 80, 'prefer': 'cup'},
    'cornmeal': {'g_per_cup': 138, 'prefer': 'cup'},
    'pasta': {'g_per_cup': 100, 'prefer': 'cup', 'note': 'uncooked'},

    # Sugars
    'sugar': {'g_per_cup': 200, 'prefer': 'cup'},
    'granulated sugar': {'g_per_cup': 200, 'prefer': 'cup'},
    'brown sugar': {'g_per_cup': 220, 'prefer': 'cup', 'note': 'packed'},
    'powdered sugar': {'g_per_cup': 120, 'prefer': 'cup'},
    'icing sugar': {'g_per_cup': 120, 'prefer': 'cup'},

    # Liquids (1 cup = 240ml = 240g for water-based)
    'water': {'g_per_cup': 240, 'prefer': 'cup'},
    'milk': {'g_per_cup': 245, 'prefer': 'cup'},
    'chicken broth': {'g_per_cup': 240, 'prefer': 'cup'},
    'beef broth': {'g_per_cup': 240, 'prefer': 'cup'},
    'vegetable broth': {'g_per_cup': 240, 'prefer': 'cup'},
    'stock': {'g_per_cup': 240, 'prefer': 'cup'},

    # Oils & Fats
    'olive oil': {'g_per_cup': 216, 'prefer': 'tbsp'},  # Usually measured in tbsp
    'vegetable oil': {'g_per_cup': 218, 'prefer': 'tbsp'},
    'butter': {'g_per_cup': 227, 'prefer': 'tbsp'},

    # Nuts & Seeds
    'almonds': {'g_per_cup': 143, 'prefer': 'cup'},
    'walnuts': {'g_per_cup': 117, 'prefer': 'cup'},
    'peanuts': {'g_per_cup': 146, 'prefer': 'cup'},

    # Dried fruits
    'raisins': {'g_per_cup': 165, 'prefer': 'cup'},
    'dried cranberries': {'g_per_cup': 120, 'prefer': 'cup'},
}

# ============================================================================
# Liquid Measures (液体单位 - 使用tablespoon)
# ============================================================================
LIQUID_TBSP = {
    'olive oil': {'g_per_tbsp': 13.5},
    'vegetable oil': {'g_per_tbsp': 13.6},
    'canola oil': {'g_per_tbsp': 14},
    'sesame oil': {'g_per_tbsp': 13.6},
    'butter': {'g_per_tbsp': 14.2},
    'soy sauce': {'g_per_tbsp': 16},
    'vinegar': {'g_per_tbsp': 15},
    'rice vinegar': {'g_per_tbsp': 15},
    'lemon juice': {'g_per_tbsp': 15},
    'lime juice': {'g_per_tbsp': 15},
    'honey': {'g_per_tbsp': 21},
}

def convert_to_household_unit(ingredient_name, grams_total, servings=1):
    """
    Convert grams to natural household units

    Args:
        ingredient_name: ingredient name (lowercase)
        grams_total: total grams for recipe
        servings: number of servings (for display purposes)

    Returns:
        formatted_string: e.g., "4 large eggs", "2 tsp salt", "1 1/2 cups flour"
    """

    ingredient = ingredient_name.lower().strip()
    grams_per_serving = grams_total / servings

    # 1. Check if it's a countable item
    if ingredient in ITEM_UNITS:
        info = ITEM_UNITS[ingredient]
        count = grams_total / info['grams_per_item']

        # If count is too high (>8), use grams instead
        if count > 8:
            grams_rounded = round(grams_total / 10) * 10  # Round to nearest 10g
            return f"{grams_rounded}g {ingredient}"

        if count >= 0.75:
            count_rounded = round(count * 2) / 2  # Round to nearest 0.5
            if count_rounded == int(count_rounded):
                return f"{int(count_rounded)} {info['unit']}{'s' if count_rounded > 1 else ''}"
            else:
                return f"{count_rounded} {info['unit']}s"

    # 2. Check if it's a spoon-measured ingredient
    if ingredient in SPOON_UNITS:
        info = SPOON_UNITS[ingredient]
        prefer = info['prefer']

        if prefer == 'tsp':
            tsp_count = grams_total / info['g_per_tsp']
            if tsp_count < 3:
                # Use teaspoons
                tsp_rounded = round(tsp_count * 4) / 4  # Round to nearest 1/4
                return format_fraction(tsp_rounded, 'tsp', info.get('note')) + f" {ingredient}"
            else:
                # Convert to tablespoons
                tbsp_count = grams_total / info['g_per_tbsp']
                tbsp_rounded = round(tbsp_count * 4) / 4
                return format_fraction(tbsp_rounded, 'Tbsp', info.get('note')) + f" {ingredient}"
        else:
            # Prefer tablespoon
            tbsp_count = grams_total / info['g_per_tbsp']
            tbsp_rounded = round(tbsp_count * 4) / 4
            return format_fraction(tbsp_rounded, 'Tbsp', info.get('note')) + f" {ingredient}"

    # 3. Check if it's liquid measured in tablespoons
    if ingredient in LIQUID_TBSP:
        info = LIQUID_TBSP[ingredient]
        tbsp_count = grams_total / info['g_per_tbsp']

        if tbsp_count < 4:
            # Use tablespoons
            tbsp_rounded = round(tbsp_count * 2) / 2
            return format_fraction(tbsp_rounded, 'Tbsp') + f" {ingredient}"
        else:
            # Convert to cups (16 Tbsp = 1 cup)
            cup_count = tbsp_count / 16
            cup_rounded = round(cup_count * 4) / 4
            return format_fraction(cup_rounded, 'cup') + f" {ingredient}"

    # 4. Check if it's cup-measured
    if ingredient in CUP_UNITS:
        info = CUP_UNITS[ingredient]
        cup_count = grams_total / info['g_per_cup']
        cup_rounded = round(cup_count * 4) / 4  # Round to nearest 1/4 cup

        return format_fraction(cup_rounded, 'cup', info.get('note')) + f" {ingredient}"

    # 5. Default: use grams for solid foods > 50g, otherwise tablespoons
    if grams_total >= 50:
        grams_rounded = round(grams_total / 5) * 5  # Round to nearest 5g
        return f"{grams_rounded}g {ingredient}"
    else:
        # Small amount, try tablespoon estimate (assume density ~15g/tbsp)
        tbsp_count = grams_total / 15
        if tbsp_count >= 0.5:
            tbsp_rounded = round(tbsp_count * 2) / 2
            return format_fraction(tbsp_rounded, 'Tbsp') + f" {ingredient}"
        else:
            return f"{grams_total:.0f}g {ingredient}"

def format_fraction(value, unit, note=None):
    """
    Format decimal to fraction string

    Args:
        value: decimal number (e.g., 1.5)
        unit: unit string (e.g., 'cup', 'tsp')
        note: optional note (e.g., 'minced')

    Returns:
        formatted string (e.g., "1 1/2 cups", "1/4 tsp minced garlic")
    """

    if value == 0:
        return f"0 {unit}"

    whole = int(value)
    frac = value - whole

    # Common fractions
    frac_map = {
        0.25: '1/4',
        0.33: '1/3',
        0.5: '1/2',
        0.67: '2/3',
        0.75: '3/4',
    }

    result = ""
    if whole > 0:
        result = f"{whole}"

    if frac > 0.1:
        # Find closest fraction
        closest = min(frac_map.keys(), key=lambda x: abs(x - frac))
        if abs(closest - frac) < 0.1:
            if result:
                result += f" {frac_map[closest]}"
            else:
                result = frac_map[closest]
        else:
            # Use decimal
            result = f"{value:.1f}"

    # Pluralize unit
    if value > 1 and unit in ['cup', 'tsp', 'Tbsp']:
        unit = unit + 's'

    if note:
        return f"{result} {unit} {note}"
    else:
        return f"{result} {unit}"

def test_converter():
    """Test household unit conversion"""

    test_cases = [
        ('egg', 200, 4),  # 4 eggs for recipe
        ('salt', 6, 4),  # 6g salt total
        ('pepper', 2, 4),  # 2g pepper
        ('garlic', 20, 4),  # 20g garlic (minced)
        ('olive oil', 40, 4),  # 40g oil
        ('flour', 500, 4),  # 500g flour
        ('rice', 370, 4),  # 370g rice
        ('butter', 56, 4),  # 56g butter
        ('sugar', 100, 4),  # 100g sugar
        ('milk', 480, 4),  # 480g milk
    ]

    print("Testing Household Unit Converter")
    print("="*80)

    for ingredient, grams, servings in test_cases:
        result = convert_to_household_unit(ingredient, grams, servings)
        print(f"{grams}g {ingredient} ({servings} servings) -> {result}")

if __name__ == '__main__':
    test_converter()
