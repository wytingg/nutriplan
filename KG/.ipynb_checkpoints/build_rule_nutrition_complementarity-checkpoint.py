#!/usr/bin/env python3
"""
最终版：基于Food.com食材营养数据构建互补规则库
使用已经匹配好的1,580个食材（Top 100覆盖率100%）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def main():
    # 输入文件
    ingredient_file = r"work/recipebench/data/4out/1005ingredient_nutri_mapping/foodcom_ingredient_nutrition_补充后.csv"
    output_dir = Path(r"work/recipebench/data/11_nutrition_rule")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("营养互补规则库构建（基于Food.com食材营养数据）")
    print("=" * 80)

    # ========== 步骤1：加载食材营养数据 ==========
    print(f"\n[1/4] 加载食材营养数据...")
    df = pd.read_csv(ingredient_file)

    # 只使用已匹配的食材
    df_matched = df[df['match_method'] != 'unmatched'].copy()
    print(f"  总食材数: {len(df):,}")
    print(f"  有营养数据: {len(df_matched):,}")

    # ========== 步骤2：计算营养素百分位数 ==========
    print(f"\n[2/4] 计算营养素阈值...")

    nutrient_cols = ['energy_kcal', 'protein_g', 'fat_g', 'carbohydrates_g',
                    'fiber_g', 'sodium_mg', 'iron_mg', 'calcium_mg', 'vitamin_c_mg']

    thresholds = {}
    for col in nutrient_cols:
        if col in df_matched.columns:
            valid_values = df_matched[col].dropna()
            if len(valid_values) > 0:
                thresholds[col] = {
                    'p75': valid_values.quantile(0.75),
                    'p50': valid_values.quantile(0.50),
                    'p25': valid_values.quantile(0.25)
                }
                print(f"  {col:20s}: 低<{thresholds[col]['p25']:.1f}  中={thresholds[col]['p50']:.1f}  高>{thresholds[col]['p75']:.1f}")

    # ========== 步骤3：为每个食材打标签 ==========
    print(f"\n[3/4] 为食材打营养标签...")

    ingredient_tags = {}

    for _, row in df_matched.iterrows():
        ing = row['foodcom_ingredient']
        tags = []

        # 高蛋白
        if pd.notna(row.get('protein_g')) and row['protein_g'] >= thresholds.get('protein_g', {}).get('p75', 999):
            tags.append('high_protein')

        # 高纤维
        if pd.notna(row.get('fiber_g')) and row['fiber_g'] >= thresholds.get('fiber_g', {}).get('p75', 999):
            tags.append('high_fiber')

        # 高铁
        if pd.notna(row.get('iron_mg')) and row['iron_mg'] >= thresholds.get('iron_mg', {}).get('p75', 999):
            tags.append('high_iron')

        # 高维生素C
        if pd.notna(row.get('vitamin_c_mg')) and row['vitamin_c_mg'] >= thresholds.get('vitamin_c_mg', {}).get('p75', 999):
            tags.append('high_vitamin_c')

        # 高钙
        if pd.notna(row.get('calcium_mg')) and row['calcium_mg'] >= thresholds.get('calcium_mg', {}).get('p75', 999):
            tags.append('high_calcium')

        # 高碳水
        if pd.notna(row.get('carbohydrates_g')) and row['carbohydrates_g'] >= thresholds.get('carbohydrates_g', {}).get('p75', 999):
            tags.append('high_carb')

        # 高脂肪
        if pd.notna(row.get('fat_g')) and row['fat_g'] >= thresholds.get('fat_g', {}).get('p75', 999):
            tags.append('high_fat')

        # 低钠
        if pd.notna(row.get('sodium_mg')) and row['sodium_mg'] <= thresholds.get('sodium_mg', {}).get('p25', 0):
            tags.append('low_sodium')

        # 低脂
        if pd.notna(row.get('fat_g')) and row['fat_g'] <= thresholds.get('fat_g', {}).get('p25', 0):
            tags.append('low_fat')

        if tags:
            ingredient_tags[ing] = tags

    print(f"  ✓ 标注完成: {len(ingredient_tags):,} 食材")

    # 示例
    print(f"\n  标签分布:")
    tag_counts = defaultdict(int)
    for tags in ingredient_tags.values():
        for tag in tags:
            tag_counts[tag] += 1

    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {tag:20s}: {count:4d} 食材")

    # ========== 步骤4：生成互补规则 ==========
    print(f"\n[4/4] 生成营养互补规则...")

    # 定义互补逻辑（营养学原理）
    complementarity_rules = {
        ('high_iron', 'high_vitamin_c'): {
            'score': 1.0,
            'reason': '维生素C促进铁吸收'
        },
        ('high_calcium', 'high_vitamin_d'): {
            'score': 0.9,
            'reason': '维生素D促进钙吸收'
        },
        ('high_protein', 'high_fiber'): {
            'score': 0.85,
            'reason': '蛋白质+纤维提升饱腹感'
        },
        ('high_carb', 'high_protein'): {
            'score': 0.8,
            'reason': '碳水+蛋白质能量平衡'
        },
        ('high_fat', 'low_sodium'): {
            'score': 0.7,
            'reason': '健康脂肪+低钠（心血管友好）'
        },
        ('low_fat', 'high_fiber'): {
            'score': 0.75,
            'reason': '低脂+高纤维（减重友好）'
        }
    }

    # 按标签分组食材
    tag_to_ingredients = defaultdict(list)
    for ing, tags in ingredient_tags.items():
        for tag in tags:
            # 只保留高频食材（Top 1000）
            ing_row = df_matched[df_matched['foodcom_ingredient'] == ing]
            if len(ing_row) > 0:
                freq = ing_row.iloc[0]['frequency']
                tag_to_ingredients[tag].append((ing, freq))

    # 对每类按频次排序
    for tag in tag_to_ingredients:
        tag_to_ingredients[tag].sort(key=lambda x: x[1], reverse=True)

    # 生成互补对
    complementary_pairs = []

    for (tag1, tag2), rule in complementarity_rules.items():
        ings1 = [ing for ing, freq in tag_to_ingredients.get(tag1, [])[:100]]  # Top 100
        ings2 = [ing for ing, freq in tag_to_ingredients.get(tag2, [])[:100]]

        for ing1 in ings1:
            for ing2 in ings2:
                if ing1 != ing2:
                    complementary_pairs.append({
                        'ingredient_1': ing1,
                        'ingredient_2': ing2,
                        'tag_1': tag1,
                        'tag_2': tag2,
                        'synergy_score': rule['score'],
                        'reason': rule['reason']
                    })

    print(f"  ✓ 生成 {len(complementary_pairs):,} 条互补规则")

    # ========== 保存结果 ==========
    print(f"\n[保存] 写入规则文件...")

    # 1. 互补对CSV
    df_pairs = pd.DataFrame(complementary_pairs)
    pairs_output = output_dir / 'nutrition_complementarity_pairs.csv'
    df_pairs.to_csv(pairs_output, index=False, encoding='utf-8')
    print(f"  ✓ {pairs_output} ({len(df_pairs):,} 条)")

    # 2. 食材营养标签CSV
    tag_records = []
    for ing, tags in ingredient_tags.items():
        for tag in tags:
            tag_records.append({'ingredient': ing, 'nutrient_tag': tag})

    df_tags = pd.DataFrame(tag_records)
    tags_output = output_dir / 'ingredient_nutrient_tags.csv'
    df_tags.to_csv(tags_output, index=False, encoding='utf-8')
    print(f"  ✓ {tags_output} ({len(df_tags):,} 条)")

    # 3. JSON快速查询格式
    complementarity_dict = defaultdict(dict)
    for _, row in df_pairs.iterrows():
        ing1, ing2 = row['ingredient_1'], row['ingredient_2']
        score = row['synergy_score']
        complementarity_dict[ing1][ing2] = float(score)

    json_output = output_dir / 'nutrition_complementarity.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(dict(complementarity_dict), f, ensure_ascii=False, indent=2)
    print(f"  ✓ {json_output}")

    # ========== 统计报告 ==========
    print(f"\n{'='*80}")
    print("最终统计")
    print(f"{'='*80}")
    print(f"有营养数据的食材: {len(df_matched):,}")
    print(f"打标签的食材: {len(ingredient_tags):,}")
    print(f"互补规则对数: {len(complementary_pairs):,}")

    print(f"\nTop 10高频食材示例:")
    top10 = df_matched.nlargest(10, 'frequency')
    for _, row in top10.iterrows():
        ing = row['foodcom_ingredient']
        tags = ingredient_tags.get(ing, [])
        print(f"  {ing:25s}  freq={row['frequency']:6d}  tags={tags}")

    print(f"\n✓ 完成！规则库已保存到 {output_dir}")

if __name__ == '__main__':
    main()
