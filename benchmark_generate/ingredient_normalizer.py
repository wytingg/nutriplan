#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingredient Name Normalizer

Removes common modifiers to improve matching between:
- User preferences (with modifiers)
- Nutrition database (standardized names)
"""

import re

# Modifiers to remove (in order of application)
MODIFIERS = {
    # Colors
    'color': ['red', 'green', 'yellow', 'white', 'black', 'orange', 'purple',
              'brown', 'pink', 'golden'],

    # Preparation/Processing
    'preparation': ['fresh', 'frozen', 'dried', 'canned', 'cooked', 'raw',
                   'roasted', 'grilled', 'baked', 'fried', 'steamed', 'boiled',
                   'smoked', 'pickled', 'marinated'],

    # Cuts/Parts (especially for meat/fish)
    'cuts': ['boneless', 'skinless', 'bone-in', 'skin-on', 'fillet', 'filet',
            'breast', 'breasts', 'thigh', 'thighs', 'leg', 'legs', 'wing', 'wings',
            'chunk', 'chunks', 'piece', 'pieces', 'slice', 'slices',
            'strip', 'strips', 'cube', 'cubes'],

    # Texture/Form
    'texture': ['whole', 'half', 'halved', 'quartered', 'chopped', 'diced',
               'minced', 'sliced', 'shredded', 'grated', 'ground', 'crushed',
               'mashed', 'pureed', 'peeled', 'unpeeled'],

    # Quality/Grade
    'quality': ['extra', 'premium', 'organic', 'free-range', 'grass-fed',
               'wild-caught', 'farm-raised', 'imported', 'domestic'],

    # Size
    'size': ['large', 'medium', 'small', 'jumbo', 'baby', 'mini', 'giant',
            'extra-large', 'extra-small'],

    # Descriptive
    'descriptive': ['plain', 'simple', 'basic', 'regular', 'standard', 'classic',
                   'traditional', 'authentic', 'homemade', 'store-bought'],

    # Brand/Style (common)
    'style': ['italian', 'chinese', 'mexican', 'french', 'japanese', 'thai',
             'indian', 'greek', 'spanish', 'korean', 'american'],
}

# All modifiers in flat list
ALL_MODIFIERS = []
for category in MODIFIERS.values():
    ALL_MODIFIERS.extend(category)

def normalize_ingredient(ingredient_name):
    """
    Normalize ingredient name by removing modifiers

    Args:
        ingredient_name: raw ingredient name (e.g., "red snapper fillet")

    Returns:
        normalized name (e.g., "snapper")

    Examples:
        "red snapper fillet" -> "snapper"
        "boneless skinless chicken breasts" -> "chicken"
        "fresh ground black pepper" -> "pepper"
        "extra virgin olive oil" -> "olive oil"
    """

    normalized = ingredient_name.lower().strip()

    # Remove special characters (but keep hyphens for compound words)
    normalized = re.sub(r'[^\w\s-]', ' ', normalized)

    # Split into words
    words = normalized.split()

    # Remove modifiers
    filtered_words = []
    for word in words:
        # Strip trailing 's' for plural matching (optional, be careful)
        # word_singular = word.rstrip('s') if len(word) > 3 else word

        # Keep word if not a modifier
        if word not in ALL_MODIFIERS:
            filtered_words.append(word)

    # Rejoin
    normalized = ' '.join(filtered_words)

    # Clean up extra spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized

def fuzzy_match_ingredient(user_ingredient, database_ingredients, threshold=0.6):
    """
    Find best match for user ingredient in database

    Args:
        user_ingredient: ingredient from user preference
        database_ingredients: list of ingredients in database
        threshold: minimum similarity score (0-1)

    Returns:
        best_match: matched ingredient name or None
        score: similarity score
    """

    # Normalize user input
    user_norm = normalize_ingredient(user_ingredient)

    best_match = None
    best_score = 0

    for db_ing in database_ingredients:
        db_norm = normalize_ingredient(db_ing)

        # Exact match
        if user_norm == db_norm:
            return db_ing, 1.0

        # Substring match (either direction)
        if user_norm in db_norm or db_norm in user_norm:
            score = 0.9
            if score > best_score:
                best_score = score
                best_match = db_ing

        # Word overlap
        user_words = set(user_norm.split())
        db_words = set(db_norm.split())

        if user_words and db_words:
            overlap = len(user_words & db_words) / len(user_words | db_words)
            if overlap > best_score:
                best_score = overlap
                best_match = db_ing

    if best_score >= threshold:
        return best_match, best_score

    return None, 0

def test_normalizer():
    """Test ingredient normalization"""

    test_cases = [
        ("red snapper fillet", "snapper"),
        ("boneless skinless chicken breasts", "chicken"),
        ("fresh ground black pepper", "pepper"),
        ("plain cornmeal", "cornmeal"),
        ("extra virgin olive oil", "olive oil"),
        ("cooked white rice", "rice"),
        ("frozen chopped spinach", "spinach"),
        ("large eggs", "eggs"),
        ("organic grass-fed beef", "beef"),
        ("wild-caught salmon", "salmon"),
    ]

    print("Testing Ingredient Normalizer")
    print("="*80)

    for raw, expected in test_cases:
        normalized = normalize_ingredient(raw)
        status = "✓" if normalized == expected else "✗"
        print(f"{status} '{raw}' -> '{normalized}' (expected: '{expected}')")

    print("\n" + "="*80)
    print("Testing Fuzzy Matching")
    print("="*80)

    # Simulate database
    database = [
        'chicken', 'chicken breast', 'beef', 'pork', 'salmon', 'snapper',
        'rice', 'pasta', 'olive oil', 'cornmeal', 'eggs', 'spinach'
    ]

    user_inputs = [
        "red snapper fillet",
        "boneless skinless chicken breasts",
        "plain cornmeal",
        "extra virgin olive oil",
        "large organic eggs"
    ]

    for user_input in user_inputs:
        match, score = fuzzy_match_ingredient(user_input, database)
        print(f"'{user_input}' -> '{match}' (score: {score:.2f})")

if __name__ == '__main__':
    test_normalizer()
