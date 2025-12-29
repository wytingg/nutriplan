#!/usr/bin/env python3
"""
方案B：为Top 100高频未匹配食材手工定义常识营养值
快速解决方案（基于营养学常识）
"""

import pandas as pd

# 高频食材的常识营养值（per 100g）
COMMON_NUTRITION = {
    # 调味料（几乎无营养，主要是钠）
    'salt': {'energy_kcal': 0, 'protein_g': 0, 'fat_g': 0, 'carbohydrates_g': 0, 'sodium_mg': 38000},
    'soy sauce': {'energy_kcal': 53, 'protein_g': 5.6, 'fat_g': 0.1, 'carbohydrates_g': 4.9, 'sodium_mg': 5600},
    'worcestershire sauce': {'energy_kcal': 78, 'protein_g': 0, 'fat_g': 0, 'carbohydrates_g': 19, 'sodium_mg': 1100},

    # 糖类
    'granulated sugar': {'energy_kcal': 387, 'protein_g': 0, 'fat_g': 0, 'carbohydrates_g': 100, 'sodium_mg': 1},
    'brown sugar': {'energy_kcal': 380, 'protein_g': 0, 'fat_g': 0, 'carbohydrates_g': 98, 'sodium_mg': 28},
    'honey': {'energy_kcal': 304, 'protein_g': 0.3, 'fat_g': 0, 'carbohydrates_g': 82, 'sodium_mg': 4},

    # 果汁/柠檬汁
    'lemon juice': {'energy_kcal': 22, 'protein_g': 0.4, 'fat_g': 0.2, 'carbohydrates_g': 6.9, 'sodium_mg': 1, 'vitamin_c_mg': 39},
    'lime juice': {'energy_kcal': 25, 'protein_g': 0.4, 'fat_g': 0.1, 'carbohydrates_g': 8.4, 'sodium_mg': 1, 'vitamin_c_mg': 30},
    'orange juice': {'energy_kcal': 45, 'protein_g': 0.7, 'fat_g': 0.2, 'carbohydrates_g': 10.4, 'sodium_mg': 1, 'vitamin_c_mg': 50},

    # 香料（极少量，营养可忽略）
    'cilantro': {'energy_kcal': 23, 'protein_g': 2.1, 'fat_g': 0.5, 'carbohydrates_g': 3.7, 'sodium_mg': 46, 'vitamin_c_mg': 27},
    'mint leaf': {'energy_kcal': 70, 'protein_g': 3.8, 'fat_g': 0.9, 'carbohydrates_g': 14.9, 'sodium_mg': 31},
    'basil': {'energy_kcal': 23, 'protein_g': 3.2, 'fat_g': 0.6, 'carbohydrates_g': 2.7, 'sodium_mg': 4},
    'parsley': {'energy_kcal': 36, 'protein_g': 3, 'fat_g': 0.8, 'carbohydrates_g': 6.3, 'sodium_mg': 56},

    # 香料种子
    'cumin seed': {'energy_kcal': 375, 'protein_g': 18, 'fat_g': 22, 'carbohydrates_g': 44, 'sodium_mg': 168},
    'coriander seed': {'energy_kcal': 298, 'protein_g': 12.4, 'fat_g': 17.8, 'carbohydrates_g': 55, 'sodium_mg': 35},
    'cardamom seed': {'energy_kcal': 311, 'protein_g': 11, 'fat_g': 6.7, 'carbohydrates_g': 68, 'sodium_mg': 18},
    'poppy seed': {'energy_kcal': 525, 'protein_g': 18, 'fat_g': 42, 'carbohydrates_g': 28, 'sodium_mg': 26},
    'sesame seed': {'energy_kcal': 573, 'protein_g': 18, 'fat_g': 50, 'carbohydrates_g': 23, 'sodium_mg': 11},

    # 香料粉末
    'clove': {'energy_kcal': 274, 'protein_g': 6, 'fat_g': 13, 'carbohydrates_g': 65, 'sodium_mg': 277},
    'peppercorn': {'energy_kcal': 251, 'protein_g': 10.4, 'fat_g': 3.3, 'carbohydrates_g': 64, 'sodium_mg': 20},
    'saffron': {'energy_kcal': 310, 'protein_g': 11.4, 'fat_g': 5.9, 'carbohydrates_g': 65, 'sodium_mg': 148},
    'mace': {'energy_kcal': 475, 'protein_g': 6.7, 'fat_g': 32.4, 'carbohydrates_g': 50, 'sodium_mg': 80},

    # 其他
    'vanilla extract': {'energy_kcal': 288, 'protein_g': 0.1, 'fat_g': 0.1, 'carbohydrates_g': 12.7, 'sodium_mg': 9},
    'baking powder': {'energy_kcal': 53, 'protein_g': 0, 'fat_g': 0, 'carbohydrates_g': 27.7, 'sodium_mg': 10600},
    'baking soda': {'energy_kcal': 0, 'protein_g': 0, 'fat_g': 0, 'carbohydrates_g': 0, 'sodium_mg': 27360},
}

def main():
    matched_file = r"work/recipebench/data/4out/1005ingredient_nutri_mapping/foodcom_ingredient_nutrition_补充后.csv"
    output_dir = Path(r"work/recipebench/data/4out/1005ingredient_nutri_mapping")

    df = pd.read_csv(matched_file)

    # 应用手工规则
    for idx, row in df.iterrows():
        if row['match_method'] == 'unmatched':
            ing = row['foodcom_ingredient']
            if ing in COMMON_NUTRITION:
                df.at[idx, 'match_method'] = 'manual_rule'
                df.at[idx, 'match_score'] = 0.9  # 手工规则，高可信度
                df.at[idx, 'usda_food_name'] = '[手工规则]'

                for col, val in COMMON_NUTRITION[ing].items():
                    df.at[idx, col] = val

    # 保存
    output_file = output_dir / 'foodcom_ingredient_nutrition_手工规则后.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')

    # 统计
    matched = len(df[df['match_method'] != 'unmatched'])
    print(f"匹配率: {matched / len(df) * 100:.1f}%")

if __name__ == '__main__':
    from pathlib import Path
    main()
