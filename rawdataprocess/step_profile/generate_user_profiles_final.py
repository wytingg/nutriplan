"""
User Profile Generator with RNI Data - Version 2
Generates user profiles with 15 core nutrients based on realistic demographic distributions
"""

import pandas as pd
import json
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("Starting user profile generation (15 Core Nutrients)...")
print("=" * 60)

# ============================================================================
# Configuration
# ============================================================================

# File paths - UPDATE THESE TO MATCH YOUR SERVER PATHS
INPUT_USER_FILE = r'work/recipebench/data/8step_profile/cleaned_user_profile.jsonl'
INPUT_RNI_FILE = r'work/recipebench/data/8step_profile/所有人群RNI.xlsx'
OUTPUT_FILE = r'work/recipebench/data/8step_profile/update_cleaned_user_profile.jsonl'

# ============================================================================
# Load Data
# ============================================================================

print("\n1. Loading RNI data...")
rni_df = pd.read_excel(INPUT_RNI_FILE, engine='openpyxl')
rni_df = rni_df.dropna(subset=['年龄组'])
print(f"   Loaded {len(rni_df)} RNI records")

print("\n2. Loading existing user profiles...")
user_profiles = []
with open(INPUT_USER_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        user_profiles.append(json.loads(line))
print(f"   Loaded {len(user_profiles)} user profiles")

# ============================================================================
# Define Distributions
# ============================================================================

# Age groups with weights based on realistic population distribution
age_groups = {
    '18-29': {'range': (18, 29), 'weight': 0.25, 'rni_key': '18岁~29岁'},
    '30-50': {'range': (30, 50), 'weight': 0.35, 'rni_key': '30岁~50岁'},
    '50-64': {'range': (50, 64), 'weight': 0.25, 'rni_key': '50岁~64岁'},
    '65-74': {'range': (65, 74), 'weight': 0.10, 'rni_key': '65岁~74岁'},
    '75+': {'range': (75, 90), 'weight': 0.05, 'rni_key': '75岁~'}
}

# Mapping English to Chinese
state_mapping = {
    'healthy': '健康',
    'diabetes': '糖尿病',
    'hypertension': '高血压',
    'hyperlipidemia': '高血脂',
    'gout': '痛风',
    'pregnancy_early': '孕早期',
    'pregnancy_mid': '孕中期',
    'pregnancy_late': '孕晚期',
    'lactation': '哺乳期'
}

gender_mapping = {
    'male': '男',
    'female': '女'
}

# ============================================================================
# 15 Core Nutrients (Tier 1)
# ============================================================================
# Mapping: output_field_name -> Excel column name
rni_column_mapping = {
    # Macronutrients (5)
    'energy_kcal': 'energy',
    'protein_g': 'protein',
    'carbohydrate_g': 'carbohydrate',
    'fat_g': 'fat',
    'fiber_g': 'fiber',

    # Nutrients to limit (4)
    'added_sugar_g': 'add sugar',           # ⚠️ Important for diabetes
    'saturated_fat_g': 'standfat(SAF)',     # Cardiovascular health
    'trans_fat_g': 'tranfat',               # Cardiovascular health
    'sodium_mg': 'Sodium',                  # Hypertension

    # Key micronutrients (6)
    'potassium_mg': 'Potassium',            # Cardiovascular health
    'calcium_mg': 'calcium',                # Bone health
    'iron_mg': 'iron',                      # Anemia prevention
    'vitamin_c_mg': 'vitamin C',            # Immune system
    'vitamin_d_ug': 'vitamin D',            # Bone health, elderly
    'folate_ug': 'Folate, DFE(B9) (total)' # Pregnancy
}

print("\n3. Selected 15 core nutrients:")
for i, (eng_name, _) in enumerate(rni_column_mapping.items(), 1):
    print(f"   {i:2d}. {eng_name}")

# ============================================================================
# Helper Functions
# ============================================================================

def assign_physiological_state(age, gender):
    """Assign physiological state based on age and gender with realistic probabilities"""

    # Women of childbearing age (20-40) can be pregnant or lactating
    if gender == 'female' and 20 <= age <= 40:
        states = ['healthy', 'diabetes', 'hypertension', 'hyperlipidemia',
                 'pregnancy_early', 'pregnancy_mid', 'pregnancy_late', 'lactation']

        # Adjust weights for younger women - lower disease rates
        if age < 30:
            weights = [0.70, 0.05, 0.10, 0.05, 0.03, 0.02, 0.02, 0.03]
        else:
            weights = [0.55, 0.12, 0.18, 0.08, 0.02, 0.015, 0.015, 0.02]

    # Older adults have higher disease prevalence
    elif age >= 65:
        states = ['healthy', 'diabetes', 'hypertension', 'hyperlipidemia', 'gout']
        if gender == 'male':
            weights = [0.20, 0.22, 0.40, 0.13, 0.05]  # Higher hypertension and gout
        else:
            weights = [0.25, 0.20, 0.38, 0.15, 0.02]  # Lower gout in women

    # Middle-aged adults
    elif 40 <= age < 65:
        states = ['healthy', 'diabetes', 'hypertension', 'hyperlipidemia', 'gout']
        if gender == 'male':
            weights = [0.35, 0.18, 0.30, 0.12, 0.05]
        else:
            weights = [0.40, 0.15, 0.28, 0.15, 0.02]

    # Young adults
    else:
        states = ['healthy', 'diabetes', 'hypertension', 'hyperlipidemia', 'gout']
        if gender == 'male':
            weights = [0.65, 0.08, 0.15, 0.10, 0.02]
        else:
            weights = [0.70, 0.06, 0.12, 0.10, 0.02]

    return random.choices(states, weights=weights, k=1)[0]


def assign_age_gender_state():
    """Assign age, gender, and physiological state based on realistic probabilities"""

    # Select age group
    age_group_keys = list(age_groups.keys())
    age_group_weights = [age_groups[k]['weight'] for k in age_group_keys]
    selected_age_group = random.choices(age_group_keys, weights=age_group_weights, k=1)[0]

    # Generate specific age within range
    age_range = age_groups[selected_age_group]['range']
    age = random.randint(age_range[0], age_range[1])
    rni_age_key = age_groups[selected_age_group]['rni_key']

    # Assign gender (roughly equal)
    gender = random.choice(['male', 'female'])

    # Assign physiological state based on age and gender
    state = assign_physiological_state(age, gender)

    return age, gender, state, rni_age_key


def get_rni_values(age, gender, state, rni_age_key):
    """Get RNI values for given demographics"""

    # Handle pregnancy and lactation - they only have data for 30-50岁 women
    if state in ['pregnancy_early', 'pregnancy_mid', 'pregnancy_late', 'lactation']:
        rni_age_key = '30岁~50岁'
        gender = 'female'

    # Map to Chinese
    gender_cn = gender_mapping[gender]
    state_cn = state_mapping[state]

    # Find matching row in RNI data
    matching_rows = rni_df[
        (rni_df['年龄组'].str.strip() == rni_age_key) &
        (rni_df['性别'] == gender_cn) &
        (rni_df['生理状态'] == state_cn)
    ]

    if matching_rows.empty:
        # Fallback to healthy state if specific condition not found
        state_cn = '健康'
        matching_rows = rni_df[
            (rni_df['年龄组'].str.strip() == rni_age_key) &
            (rni_df['性别'] == gender_cn) &
            (rni_df['生理状态'] == state_cn)
        ]

    if matching_rows.empty:
        return None

    # Extract nutrient values
    row = matching_rows.iloc[0]
    nutrient_values = {}

    for eng_name, rni_col in rni_column_mapping.items():
        value = row[rni_col]
        # Handle missing values
        if pd.isna(value) or value == '-':
            nutrient_values[eng_name] = None
        else:
            # Convert to appropriate type
            try:
                nutrient_values[eng_name] = float(value)
            except:
                nutrient_values[eng_name] = None

    return nutrient_values

# ============================================================================
# Generate Profiles
# ============================================================================

print("\n4. Generating new user profiles...")
print("   This may take a few minutes for large datasets...")

new_profiles = []
skipped_count = 0

for i, user in enumerate(user_profiles):
    user_id = user['user_id']
    liked = user.get('liked_ingredients', [])
    disliked = user.get('disliked_ingredients', [])

    # Generate demographics
    age, gender, state, rni_age_key = assign_age_gender_state()

    # Get RNI values
    rni_values = get_rni_values(age, gender, state, rni_age_key)

    if rni_values is None:
        skipped_count += 1
        continue

    # Create new profile
    new_profile = {
        'user_id': user_id,
        'gender': gender,
        'age': age,
        'physiological_state': state,
        'liked_ingredients': liked,
        'disliked_ingredients': disliked,
        'nutrition_rni': rni_values
    }

    new_profiles.append(new_profile)

    # Progress update every 10000 users
    if (i + 1) % 10000 == 0:
        print(f"   Processed {i + 1}/{len(user_profiles)} users ({(i+1)/len(user_profiles)*100:.1f}%)")

print(f"\n5. Generation complete!")
print(f"   Successfully generated: {len(new_profiles)} profiles")
print(f"   Skipped (missing RNI data): {skipped_count} profiles")

# ============================================================================
# Save Results
# ============================================================================

print(f"\n6. Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for profile in new_profiles:
        f.write(json.dumps(profile, ensure_ascii=False) + '\n')
print("   Saved successfully!")

# ============================================================================
# Statistics
# ============================================================================

print("\n" + "=" * 60)
print("STATISTICS")
print("=" * 60)

# Gender distribution
print(f"\nGender distribution:")
gender_counts = {}
for p in new_profiles:
    gender_counts[p['gender']] = gender_counts.get(p['gender'], 0) + 1
for g, c in sorted(gender_counts.items()):
    print(f"  {g}: {c:,} ({c/len(new_profiles)*100:.1f}%)")

# Physiological state distribution
print(f"\nPhysiological state distribution:")
state_counts = {}
for p in new_profiles:
    state_counts[p['physiological_state']] = state_counts.get(p['physiological_state'], 0) + 1
for s, c in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {s}: {c:,} ({c/len(new_profiles)*100:.1f}%)")

# Age distribution
print(f"\nAge distribution:")
age_ranges = {'18-29': 0, '30-49': 0, '50-64': 0, '65-74': 0, '75+': 0}
for p in new_profiles:
    age = p['age']
    if age < 30:
        age_ranges['18-29'] += 1
    elif age < 50:
        age_ranges['30-49'] += 1
    elif age < 65:
        age_ranges['50-64'] += 1
    elif age < 75:
        age_ranges['65-74'] += 1
    else:
        age_ranges['75+'] += 1
for r, c in age_ranges.items():
    print(f"  {r} years: {c:,} ({c/len(new_profiles)*100:.1f}%)")

# Nutrient availability check
print(f"\nNutrient data completeness:")
nutrient_counts = {k: 0 for k in rni_column_mapping.keys()}
for p in new_profiles:
    for nutrient in rni_column_mapping.keys():
        if p['nutrition_rni'].get(nutrient) is not None:
            nutrient_counts[nutrient] += 1

for nutrient, count in nutrient_counts.items():
    pct = count / len(new_profiles) * 100
    status = "✓" if pct > 95 else "⚠"
    print(f"  {status} {nutrient}: {count:,} ({pct:.1f}%)")

# Sample profiles
print("\n" + "=" * 60)
print("SAMPLE PROFILES")
print("=" * 60)

for i in range(min(3, len(new_profiles))):
    print(f"\nSample {i+1}:")
    sample = new_profiles[i]
    print(f"  User ID: {sample['user_id']}")
    print(f"  Demographics: {sample['gender']}, {sample['age']} years, {sample['physiological_state']}")
    print(f"  Liked ingredients: {len(sample['liked_ingredients'])} items")
    print(f"  Disliked ingredients: {len(sample['disliked_ingredients'])} items")
    print(f"  Nutrition RNI (15 nutrients):")
    for nutrient, value in sample['nutrition_rni'].items():
        if value is not None:
            print(f"    {nutrient}: {value}")
        else:
            print(f"    {nutrient}: N/A")

print("\n" + "=" * 60)
print("DONE! 15 Core Nutrients Included")
print("=" * 60)
print("\nCore nutrients included:")
print("  Macros: energy, protein, carbohydrate, fat, fiber")
print("  Limits: added_sugar ⚠️, saturated_fat, trans_fat, sodium")
print("  Micros: potassium, calcium, iron, vitamin_c, vitamin_d, folate")
