#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build hybrid nutrition database: Manual Top 100 + API for remaining
"""

import pandas as pd
import sys

print("="*80)
print("Building Hybrid Nutrition Database")
print("="*80)

# Load priority ingredients list
print("\n[*] Loading ingredient priority list...")
df_priority = pd.read_csv('work/recipebench/data/raw/foodcom/ingredients_top500_priority.csv')
print(f"  + Total ingredients to process: {len(df_priority):,}")

# Load manual mappings (Top 100)
print("\n[*] Loading manual nutrition mappings...")
df_manual = pd.read_csv('work/recipebench/data/raw/foodcom/top100_manual_nutrition.csv')
print(f"  + Manual mappings available: {len(df_manual)}")

# Merge manual mappings
print("\n[*] Applying manual mappings...")

# Select columns that exist
merge_cols = ['ingredient', 'energy_kcal', 'protein_g', 'fat_g', 'carbohydrates_g', 'fiber_g', 'sodium_mg', 'sugars_g']

# Add optional columns if they exist
if 'source' in df_manual.columns:
    merge_cols.append('source')
if 'notes' in df_manual.columns:
    merge_cols.append('notes')

df_result = df_priority.merge(
    df_manual[merge_cols],
    on='ingredient',
    how='left'
)

manual_count = df_result['energy_kcal'].notna().sum()
print(f"  + Matched from manual: {manual_count}")

# Check which ingredients still need API
remaining = df_result[df_result['energy_kcal'].isna()]
print(f"  + Remaining for API: {len(remaining)}")

# Calculate coverage
total_freq = df_priority['frequency'].sum()
manual_freq = df_result[df_result['energy_kcal'].notna()]['frequency'].sum()
coverage = manual_freq / total_freq * 100

print(f"\n[*] Coverage Analysis:")
print(f"  + Manual mappings cover: {coverage:.1f}% of all ingredient usage")
print(f"  + Top 100 frequency coverage: {df_manual['frequency'].sum() / total_freq * 100:.1f}%")

# For now, save what we have
output_manual_only = 'work/recipebench/data/raw/foodcom/nutrition_top100_manual.csv'
df_manual_result = df_result[df_result['energy_kcal'].notna()].copy()

# Add required columns for consistency
df_manual_result.loc[:, 'fdc_id'] = 0  # Manual entries don't have fdc_id

if 'notes' in df_manual_result.columns:
    df_manual_result.loc[:, 'usda_description'] = df_manual_result['notes']
else:
    df_manual_result.loc[:, 'usda_description'] = 'Manual entry'

if 'source' in df_manual_result.columns:
    df_manual_result.loc[:, 'data_type'] = df_manual_result['source']
else:
    df_manual_result.loc[:, 'data_type'] = 'Manual'

# Reorder columns to match API output
column_order = ['ingredient', 'frequency', 'fdc_id', 'usda_description', 'data_type',
                'energy_kcal', 'protein_g', 'fat_g', 'carbohydrates_g',
                'fiber_g', 'sodium_mg', 'sugars_g']

df_manual_result = df_manual_result[column_order]
df_manual_result.to_csv(output_manual_only, index=False)

print(f"\n[*] Saved manual mappings to: {output_manual_only}")

# Save remaining list for API query
if len(remaining) > 0:
    output_remaining = 'work/recipebench/data/raw/foodcom/ingredients_remaining_for_api.csv'
    remaining[['ingredient', 'frequency']].to_csv(output_remaining, index=False)
    print(f"[*] Saved remaining ingredients to: {output_remaining}")
    print(f"    ({len(remaining)} ingredients, {len(remaining)/len(df_priority)*100:.1f}% of total)")

# Show statistics
print(f"\n{'='*80}")
print("STATISTICS")
print("="*80)
print(f"Manual nutrition data: {len(df_manual_result):,} ingredients")
print(f"Coverage: {coverage:.1f}% of ingredient usage")
print(f"\nTop 20 ingredients:")
print(f"{'Rank':<5} {'Ingredient':<30} {'Calories':<10} {'Status'}")
print("-"*80)

for i, (_, row) in enumerate(df_result.head(20).iterrows(), 1):
    status = "✓ Manual" if pd.notna(row['energy_kcal']) else "○ Need API"
    cal_str = f"{row['energy_kcal']:.0f}" if pd.notna(row['energy_kcal']) else "N/A"
    print(f"{i:<5} {row['ingredient'][:30]:<30} {cal_str:<10} {status}")

print(f"\n{'='*80}")
print("Next Steps:")
print("="*80)

if len(remaining) > 0:
    print(f"1. Run API query for remaining {len(remaining)} ingredients:")
    print(f"   python query_nutrition_api.py ingredients_remaining_for_api.csv nutrition_remaining.csv {min(len(remaining), 500)}")
    print(f"\n2. Merge results:")
    print(f"   python merge_nutrition_sources.py")
else:
    print("✓ All ingredients have manual mappings!")

print("="*80)
