#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test v5 fixes with 3 samples

这个脚本会：
1. 从generate_b_class_full.py导入generate_recipe函数（实际生成食谱的函数）
2. 读取3个用户配置文件
3. 为每个用户生成1个食谱
4. 检查生成的食谱是否有P0/P1级别的问题
5. 输出质量评估结果
"""

import json
import sys

# 导入生成器模块中的核心函数
from generate_b_class_full import generate_recipe

# ============================================================================
# 辅助函数：用于检测重复食材
# ============================================================================
def normalize_for_dedup(ing):
    """Normalize ingredient name for deduplication (handle plurals, modifiers, etc.)"""
    normalized = ing.lower().strip()

    # Remove common modifiers/descriptors
    modifiers = [
        'fresh', 'frozen', 'dried', 'canned', 'raw', 'cooked',
        'chopped', 'diced', 'sliced', 'minced', 'crushed',
        'large', 'medium', 'small', 'baby',
        'italian-style', 'greek-style', 'mexican-style'
    ]
    for mod in modifiers:
        normalized = normalized.replace(mod + ' ', '').replace(' ' + mod, '')

    # Handle plurals: carrots -> carrot
    if normalized.endswith('ies'):
        normalized = normalized[:-3] + 'y'  # berries -> berry
    elif normalized.endswith('es') and not normalized.endswith('ses'):
        normalized = normalized[:-2]  # tomatoes -> tomato
    elif normalized.endswith('s') and len(normalized) > 3:
        normalized = normalized[:-1]  # carrots -> carrot

    return normalized.strip()

print("="*80)
print("B-Class Dataset v5 Quality Test (3 Samples)")
print("="*80)

# ============================================================================
# 步骤1: 加载用户配置文件
# ============================================================================
print("\n[Step 1] Loading user profiles...")
with open('work/recipebench/data/8step_profile/cleaned_user_profile.jsonl', 'r', encoding='utf-8') as f:
    all_profiles = [json.loads(line) for line in f if line.strip()]

# 选择前3个用户进行测试
test_users = all_profiles[:3]
print(f"  Loaded {len(test_users)} users for testing")

# ============================================================================
# 步骤2: 为每个用户生成食谱并检查质量
# ============================================================================
print("\n[Step 2] Generating recipes and checking quality...")

all_issues = []  # 收集所有问题

for i, user_profile in enumerate(test_users, 1):
    user_id = user_profile['user_id']
    print(f"\n{'='*70}")
    print(f"Sample {i}: Generating recipe for User {user_id}")
    print(f"{'='*70}")

    # ========================================================================
    # 调用generate_recipe函数生成食谱
    # 输入: user_profile (包含liked_ingredients, disliked_ingredients, nutrition_targets)
    # 输出: recipe字典 (包含title, ingredients, steps, nutrition等)
    # ========================================================================
    recipe = generate_recipe(user_profile, seed=i*100)

    if not recipe:
        print(f"  ✗ ERROR: Failed to generate recipe")
        all_issues.append(f"Sample {i}: Generation failed")
        continue

    # ========================================================================
    # 显示生成结果
    # ========================================================================
    print(f"\n► Title: {recipe['output']['title']}")

    print(f"\n► Selected Ingredients (from database):")
    for ing in recipe['metadata']['selected_ingredients']:
        print(f"    - {ing}")

    print(f"\n► Recipe Ingredients (with household units):")
    for ing in recipe['output']['ingredients']:
        print(f"    - {ing}")

    # ========================================================================
    # 营养指标检查
    # ========================================================================
    nutrition = recipe['output']['nutrition_per_serv']
    targets = recipe['input']['nutrition_targets']

    # 能量检查
    target_energy = targets['energy_kcal_target'] / 4  # 除以4份
    actual_energy = nutrition['energy_kcal']
    energy_error = abs(actual_energy - target_energy) / target_energy * 100

    # 钠检查
    sodium_total = nutrition['sodium_mg'] * 4  # 4份总量
    sodium_max = targets['sodium_mg_max']
    sodium_ok = sodium_total <= sodium_max

    # 纤维检查
    fiber_total = nutrition['fiber_g'] * 4  # 4份总量
    fiber_min = targets['fiber_g_min']
    fiber_ratio = fiber_total / fiber_min

    print(f"\n► Nutrition Quality Check:")
    print(f"    Energy:  {actual_energy:.0f} kcal (target: {target_energy:.0f}, error: {energy_error:.1f}%)")
    print(f"    Sodium:  {sodium_total:.0f} mg / {sodium_max:.0f} mg max [{('OK' if sodium_ok else 'OVER')}]")
    print(f"    Fiber:   {fiber_total:.1f} g / {fiber_min:.0f} g min ({fiber_ratio*100:.0f}%)")

    # ========================================================================
    # 质量问题检测
    # ========================================================================
    issues = []

    # P0检查: 钠超标2倍以上
    if sodium_total > sodium_max * 2:
        issues.append(f"P0: Sodium catastrophic - {sodium_total:.0f}mg ({sodium_total/sodium_max:.1f}x over limit)")

    # P0检查: 禁止的食材（bouillon、flour、bone、syrup、condiments等）
    selected_ingredients = recipe['metadata']['selected_ingredients']
    forbidden_keywords = [
        'bouillon', 'cubes', 'flour', 'wheat germ', 'bone', 'broth',
        'syrup', 'corn syrup', 'mayonnaise', 'miracle whip', 'whip',
        'ketchup', 'mustard', 'jam', 'jelly'
    ]
    for ingredient in selected_ingredients:
        for forbidden in forbidden_keywords:
            if forbidden in ingredient and 'olive oil' not in ingredient:
                issues.append(f"P0: Forbidden ingredient '{ingredient}' (contains '{forbidden}')")
                break

    # P1检查: 重复食材
    normalized_names = [normalize_for_dedup(ing) for ing in selected_ingredients]
    if len(normalized_names) != len(set(normalized_names)):
        duplicates = [name for name in normalized_names if normalized_names.count(name) > 1]
        issues.append(f"P1: Duplicate ingredients detected: {set(duplicates)}")

    # P1检查: 纤维偏离目标过多
    if fiber_ratio < 0.8:
        issues.append(f"P1: Fiber too low - {fiber_total:.1f}g vs target {fiber_min:.0f}g ({fiber_ratio*100:.0f}%)")
    elif fiber_ratio > 1.4:
        issues.append(f"P1: Fiber too high - {fiber_total:.1f}g vs target {fiber_min:.0f}g ({fiber_ratio*100:.0f}%)")

    # P1检查: 能量偏离目标>20%
    if energy_error > 20:
        issues.append(f"P1: Energy error too large - {energy_error:.0f}% off target")

    # 显示问题
    if issues:
        print(f"\n► Issues Found:")
        for issue in issues:
            print(f"    ✗ {issue}")
        all_issues.extend(issues)
    else:
        print(f"\n► Quality Status: ✓ PASS - No critical issues detected")

# ============================================================================
# 步骤3: 汇总评估
# ============================================================================
print(f"\n{'='*80}")
print("FINAL QUALITY ASSESSMENT")
print(f"{'='*80}")

p0_issues = [i for i in all_issues if i.startswith('P0')]
p1_issues = [i for i in all_issues if i.startswith('P1')]

print(f"\nTotal Issues: {len(all_issues)}")
print(f"  - P0 (Catastrophic): {len(p0_issues)}")
print(f"  - P1 (Critical):     {len(p1_issues)}")

if len(p0_issues) == 0 and len(p1_issues) <= 2:
    verdict = "✓ PASS - Ready for full dataset generation (10k/2k/2k)"
    grade = "EXCELLENT"
elif len(p0_issues) == 0:
    verdict = "⚠ ACCEPTABLE - Can proceed with minor issues"
    grade = "GOOD"
else:
    verdict = "✗ FAIL - Must fix P0 issues before full generation"
    grade = "FAIL"

print(f"\nGrade: {grade}")
print(f"Verdict: {verdict}")
print(f"\n{'='*80}")
