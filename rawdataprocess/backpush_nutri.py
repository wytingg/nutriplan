#!/usr/bin/env python3
"""
方案A：从Food.com食谱营养反推食材营养
为未匹配的高频食材估算营养值
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

def parse_r_vector(r_str):
    """解析R的c()向量"""
    if pd.isna(r_str) or r_str == 'NA':
        return []
    r_str = str(r_str).strip()
    if r_str.startswith('c(') and r_str.endswith(')'):
        r_str = r_str[2:-1]
    items = []
    in_quote = False
    current = ""
    for char in r_str:
        if char == '"':
            in_quote = not in_quote
        elif char == ',' and not in_quote:
            if current.strip():
                items.append(current.strip().strip('"'))
            current = ""
        else:
            current += char
    if current.strip():
        items.append(current.strip().strip('"'))
    return items

def main():
    # 路径
    recipes_file = r"work/recipebench/data/raw/foodcom/recipes.csv"
    matched_file = r"work/recipebench/data/4out/1005ingredient_nutri_mapping/foodcom_ingredient_nutrition.csv"
    output_dir = Path(r"work/recipebench/data/4out/1005ingredient_nutri_mapping")

    print("=" * 80)
    print("从Food.com食谱反推食材营养")
    print("=" * 80)

    # 1. 加载已匹配的食材
    df_matched = pd.read_csv(matched_file)
    unmatched = df_matched[df_matched['match_method'] == 'unmatched'].copy()

    # 只处理Top 300高频未匹配食材
    unmatched_top = unmatched.nlargest(300, 'frequency')
    target_ingredients = set(unmatched_top['foodcom_ingredient'].str.lower())

    print(f"\n目标：为 {len(target_ingredients)} 个高频未匹配食材估算营养")
    print(f"示例: {list(target_ingredients)[:10]}")

    # 2. 加载Food.com食谱
    print(f"\n加载Food.com食谱...")
    df_recipes = pd.read_csv(recipes_file, low_memory=False)
    print(f"  加载 {len(df_recipes):,} 个食谱")

    # 3. 为每个目标食材统计包含它的食谱营养
    print(f"\n反推食材营养...")
    ingredient_recipes = defaultdict(list)

    for idx, row in tqdm(df_recipes.iterrows(), total=len(df_recipes), desc="  分析中"):
        # 获取食材列表
        ingredients_raw = parse_r_vector(row.get('RecipeIngredientParts', ''))
        if not ingredients_raw:
            continue

        ingredients_normalized = [ing.lower().strip() for ing in ingredients_raw]

        # 获取食谱营养
        nutrition = {
            'energy': row.get('Calories', np.nan),
            'protein': row.get('ProteinContent', np.nan),
            'fat': row.get('FatContent', np.nan),
            'carb': row.get('CarbohydrateContent', np.nan),
            'fiber': row.get('FiberContent', np.nan),
            'sugar': row.get('SugarContent', np.nan),
            'saturated_fat': row.get('SaturatedFatContent', np.nan),
            'sodium': row.get('SodiumContent', np.nan),
            'servings': row.get('RecipeServings', 1)
        }

        # 检查是否有效营养数据
        if pd.notna(nutrition['energy']) and nutrition['energy'] > 0:
            # 检查哪些目标食材在这个食谱中
            for target_ing in target_ingredients:
                if any(target_ing in ing_norm for ing_norm in ingredients_normalized):
                    ingredient_recipes[target_ing].append({
                        'recipe_id': row.get('RecipeId'),
                        'num_ingredients': len(ingredients_normalized),
                        **nutrition
                    })

    # 4. 计算每个食材的估算营养值
    print(f"\n计算估算营养值...")
    estimated_nutrition = []

    for ingredient, recipes in ingredient_recipes.items():
        if len(recipes) < 5:  # 至少5个食谱才估算
            continue

        # 计算中位数（更稳定，不受极端值影响）
        # 假设食材贡献 = 食谱营养 / 食材数量
        contributions = []
        for recipe in recipes:
            servings = recipe['servings'] if recipe['servings'] > 0 else 1
            num_ing = recipe['num_ingredients'] if recipe['num_ingredients'] > 0 else 5

            # 简化估算：per-serving营养 / 食材数
            contribution = {
                'energy_kcal': recipe['energy'] / servings / num_ing if pd.notna(recipe['energy']) else np.nan,
                'protein_g': recipe['protein'] / servings / num_ing if pd.notna(recipe['protein']) else np.nan,
                'fat_g': recipe['fat'] / servings / num_ing if pd.notna(recipe['fat']) else np.nan,
                'carbohydrates_g': recipe['carb'] / servings / num_ing if pd.notna(recipe['carb']) else np.nan,
                'fiber_g': recipe['fiber'] / servings / num_ing if pd.notna(recipe['fiber']) else np.nan,
                'sugars_total_g': recipe['sugar'] / servings / num_ing if pd.notna(recipe['sugar']) else np.nan,
                'saturated_fat_g': recipe['saturated_fat'] / servings / num_ing if pd.notna(recipe['saturated_fat']) else np.nan,
                'sodium_mg': recipe['sodium'] / servings / num_ing if pd.notna(recipe['sodium']) else np.nan,
            }
            contributions.append(contribution)

        # 计算中位数
        df_contrib = pd.DataFrame(contributions)
        estimated = {
            'foodcom_ingredient': ingredient,
            'method': 'foodcom_reverse',
            'num_recipes_used': len(recipes),
            **{col: df_contrib[col].median() for col in df_contrib.columns}
        }

        estimated_nutrition.append(estimated)

    df_estimated = pd.DataFrame(estimated_nutrition)

    print(f"  ✓ 估算完成: {len(df_estimated)} 个食材")

    # 5. 合并到原始结果
    print(f"\n合并结果...")

    # 更新未匹配的食材
    for idx, row in df_matched.iterrows():
        if row['match_method'] == 'unmatched':
            ing = row['foodcom_ingredient']
            estimated_row = df_estimated[df_estimated['foodcom_ingredient'] == ing]

            if len(estimated_row) > 0:
                estimated_data = estimated_row.iloc[0]
                # 更新
                df_matched.at[idx, 'match_method'] = 'foodcom_reverse'
                df_matched.at[idx, 'match_score'] = 0.5  # 估算值，分数0.5
                df_matched.at[idx, 'usda_food_name'] = f"[估算自{int(estimated_data['num_recipes_used'])}个食谱]"

                # 更新营养值
                for col in ['energy_kcal', 'protein_g', 'fat_g', 'carbohydrates_g',
                           'fiber_g', 'sugars_total_g', 'saturated_fat_g', 'sodium_mg']:
                    if col in estimated_data:
                        df_matched.at[idx, col] = estimated_data[col]

    # 6. 保存
    output_file = output_dir / 'foodcom_ingredient_nutrition_补充后.csv'
    df_matched.to_csv(output_file, index=False, encoding='utf-8')
    print(f"  ✓ 保存到: {output_file}")

    # 统计
    print(f"\n{'='*80}")
    print("统计报告")
    print(f"{'='*80}")

    total = len(df_matched)
    matched_now = len(df_matched[df_matched['match_method'] != 'unmatched'])
    match_rate = matched_now / total * 100

    print(f"总食材数: {total:,}")
    print(f"匹配+估算成功: {matched_now:,} ({match_rate:.1f}%)")
    print(f"仍未覆盖: {total - matched_now:,} ({100 - match_rate:.1f}%)")

    print(f"\n方法分布:")
    print(df_matched['match_method'].value_counts())

    # Top 50高频食材覆盖率
    top_50 = df_matched.nlargest(50, 'frequency')
    top_covered = len(top_50[top_50['match_method'] != 'unmatched'])
    print(f"\nTop 50高频食材覆盖率: {top_covered}/50 ({top_covered/50*100:.1f}%)")

    print(f"\n✓ 完成！提升后匹配率: {match_rate:.1f}%")

if __name__ == '__main__':
    main()
