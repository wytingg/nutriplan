#!/usr/bin/env python3
"""
简化版：只使用USDA数据源，放宽过滤条件
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    usda_path = Path(r"work/recipebench/data/raw/usda")
    output_dir = Path(r"work/recipebench/data/4out/1005ingredient_nutri_mapping")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("USDA营养成分汇总表构建（仅USDA源）")
    print("=" * 80)

    # 加载USDA数据
    print(f"\n[1/4] 加载USDA数据...")
    df_food = pd.read_csv(usda_path / 'food.csv', low_memory=False)
    df_nutrient = pd.read_csv(usda_path / 'nutrient.csv', low_memory=False)
    df_food_nutrient = pd.read_csv(usda_path / 'food_nutrient.csv', low_memory=False)

    print(f"  食材数: {len(df_food):,}")
    print(f"  营养素数: {len(df_nutrient):,}")
    print(f"  关联记录数: {len(df_food_nutrient):,}")

    # 核心营养素映射（精确匹配）
    print(f"\n[2/4] 映射核心营养素...")

    core_nutrients_mapping = {
        'Energy (Atwater General Factors)': 'energy_kcal',
        'Energy (Atwater Specific Factors)': 'energy_kcal_specific',
        'Energy': 'energy_kcal_basic',
        'Protein': 'protein_g',
        'Total lipid (fat)': 'fat_g',
        'Carbohydrate, by difference': 'carbohydrates_g',
        'Fiber, total dietary': 'fiber_g',
        'Sugars, total including NLEA': 'sugars_total_g',
        'Sugars, Total': 'sugars_total_g_alt',
        'Fatty acids, total saturated': 'saturated_fat_g',
        'Sodium, Na': 'sodium_mg',
        'Iron, Fe': 'iron_mg',
        'Calcium, Ca': 'calcium_mg',
        'Vitamin C, total ascorbic acid': 'vitamin_c_mg',
        'Vitamin D (D2 + D3), International Units': 'vitamin_d_iu',
        'Vitamin D (D2 + D3)': 'vitamin_d_mcg',
        'Vitamin A, IU': 'vitamin_a_iu'
    }

    # 构建营养素ID到短名称的映射
    nutrient_id_to_short = {}
    for _, row in df_nutrient.iterrows():
        nutrient_name = str(row['name']).strip()
        nutrient_id = row['id']

        # 精确匹配或开头匹配
        for long_name, short_name in core_nutrients_mapping.items():
            if nutrient_name == long_name or nutrient_name.startswith(long_name):
                if nutrient_id not in nutrient_id_to_short:  # 避免重复
                    nutrient_id_to_short[nutrient_id] = short_name
                break

    print(f"  匹配到 {len(set(nutrient_id_to_short.values()))} 种核心营养素")
    print(f"  示例: {list(set(nutrient_id_to_short.values()))[:10]}")

    # 构建宽表
    print(f"\n[3/4] 构建宽表（pivot）...")

    # 只保留映射的营养素
    df_fn_filtered = df_food_nutrient[
        df_food_nutrient['nutrient_id'].isin(nutrient_id_to_short.keys())
    ].copy()

    print(f"  过滤后记录: {len(df_fn_filtered):,} / {len(df_food_nutrient):,}")

    # 映射到短名称
    df_fn_filtered['nutrient_short'] = df_fn_filtered['nutrient_id'].map(nutrient_id_to_short)

    # 去重（同一食材+同一营养素，取第一个）
    df_fn_filtered = df_fn_filtered.drop_duplicates(subset=['fdc_id', 'nutrient_short'], keep='first')

    # Pivot
    df_wide = df_fn_filtered.pivot_table(
        index='fdc_id',
        columns='nutrient_short',
        values='amount',
        aggfunc='first'
    ).reset_index()

    print(f"  Pivot后: {len(df_wide):,} 食材")

    # 合并食材名称
    df_wide = df_wide.merge(
        df_food[['fdc_id', 'description']],
        on='fdc_id',
        how='left'
    )

    df_wide = df_wide.rename(columns={
        'fdc_id': 'food_id',
        'description': 'food_name'
    })

    # 合并多个能量字段（优先级：Atwater General > Atwater Specific > Basic）
    energy_cols = ['energy_kcal', 'energy_kcal_specific', 'energy_kcal_basic']
    for col in energy_cols:
        if col in df_wide.columns and col != 'energy_kcal':
            df_wide['energy_kcal'] = df_wide['energy_kcal'].fillna(df_wide[col])
            df_wide = df_wide.drop(columns=[col])

    # 合并多个糖分字段
    if 'sugars_total_g_alt' in df_wide.columns:
        if 'sugars_total_g' not in df_wide.columns:
            df_wide['sugars_total_g'] = df_wide['sugars_total_g_alt']
        else:
            df_wide['sugars_total_g'] = df_wide['sugars_total_g'].fillna(df_wide['sugars_total_g_alt'])
        df_wide = df_wide.drop(columns=['sugars_total_g_alt'])

    # 放宽过滤条件：至少有能量 OR 至少有2个宏量营养素
    print(f"\n[4/4] 过滤低质量食材...")

    has_energy = df_wide.get('energy_kcal', pd.Series()).notna()

    macro_count = 0
    if 'protein_g' in df_wide.columns:
        macro_count += df_wide['protein_g'].notna().astype(int)
    if 'fat_g' in df_wide.columns:
        macro_count += df_wide['fat_g'].notna().astype(int)
    if 'carbohydrates_g' in df_wide.columns:
        macro_count += df_wide['carbohydrates_g'].notna().astype(int)

    # 保留：有能量 OR 至少2个宏量营养素
    valid_mask = has_energy | (macro_count >= 2)
    df_final = df_wide[valid_mask].copy()

    print(f"  保留食材: {len(df_final):,} / {len(df_wide):,}")

    # 添加数据源标识
    df_final['source'] = 'USDA'

    # 重新排列列
    base_cols = ['food_id', 'food_name', 'source']
    nutrient_cols = [col for col in df_final.columns if col not in base_cols]
    df_final = df_final[base_cols + nutrient_cols]

    # 保存
    output_file = output_dir / 'nutrient_catalog_usda_fndds.csv'
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ 保存到: {output_file}")

    # 统计报告
    print(f"\n{'=' * 80}")
    print("统计报告")
    print(f"{'=' * 80}")
    print(f"总食材数: {len(df_final):,}")

    print(f"\n营养素覆盖率:")
    for col in nutrient_cols:
        if col in df_final.columns:
            coverage = df_final[col].notna().sum() / len(df_final) * 100
            mean_val = df_final[col].mean()
            print(f"  {col:25s}: {coverage:5.1f}%  (均值: {mean_val:.2f})")

    # 完整数据示例
    complete_mask = (
        df_final.get('energy_kcal', pd.Series()).notna() &
        df_final.get('protein_g', pd.Series()).notna() &
        df_final.get('fat_g', pd.Series()).notna() &
        df_final.get('carbohydrates_g', pd.Series()).notna()
    )
    df_complete = df_final[complete_mask]

    print(f"\n有完整核心营养数据的食材: {len(df_complete):,} / {len(df_final):,}")
    print(f"\n示例食材:")
    display_cols = ['food_name', 'energy_kcal', 'protein_g', 'fat_g', 'carbohydrates_g']
    display_cols = [c for c in display_cols if c in df_complete.columns]
    print(df_complete[display_cols].head(15).to_string(index=False))

    print(f"\n✓ 完成！")

if __name__ == '__main__':
    main()
