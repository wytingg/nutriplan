#!/usr/bin/env python3
"""
完整流程：为Food.com的6,921个食材构建营养表
步骤1：构建USDA完整营养表（74,175食材）
步骤2：Food.com食材匹配USDA营养数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import re
from difflib import SequenceMatcher
import multiprocessing as mp

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
    escape = False

    for char in r_str:
        if escape:
            current += char
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if char == ',' and not in_quote:
            if current.strip():
                items.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        items.append(current.strip())
    return items

def normalize_for_matching(text):
    """标准化用于匹配"""
    text = text.lower().strip()

    # 移除括号内容
    text = re.sub(r'\([^)]*\)', '', text)

    # 移除标点符号（关键修复！）
    text = re.sub(r'[,;:.]', ' ', text)

    # 移除修饰词
    modifiers = ['fresh', 'dried', 'frozen', 'canned', 'raw', 'cooked',
                'chopped', 'minced', 'sliced', 'diced', 'ground', 'whole',
                'organic', 'unsalted', 'salted', 'plain', 'boneless', 'skinless']
    for mod in modifiers:
        text = re.sub(r'\b' + mod + r'\b\s*', '', text)

    # 处理复数（s/es结尾）
    words = text.split()
    normalized_words = []
    for word in words:
        if len(word) > 4:
            if word.endswith('ies'):
                word = word[:-3] + 'y'  # berries → berry
            elif word.endswith('es') and not word.endswith('oes'):
                word = word[:-2]  # tomatoes保留，但dishes → dish
            elif word.endswith('s') and not word.endswith('ss'):
                word = word[:-1]  # onions → onion
        normalized_words.append(word)

    text = ' '.join(normalized_words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 同义词映射
SYNONYM_MAP = {
    'cilantro': 'coriander', 'scallion': 'green onion', 'scallions': 'green onion',
    'zucchini': 'courgette', 'eggplant': 'aubergine',
    'bell pepper': 'sweet pepper', 'green pepper': 'sweet pepper',
    'roma tomato': 'tomato', 'plum tomato': 'tomato', 'cherry tomato': 'tomato',
    'heavy cream': 'cream', 'whipping cream': 'cream',
    'cheddar cheese': 'cheese', 'mozzarella': 'cheese', 'parmesan': 'cheese',
    'ground beef': 'beef', 'chicken breast': 'chicken', 'pork chop': 'pork',
    'chili powder': 'chili', 'garlic powder': 'garlic', 'onion powder': 'onion',
    'olive oil': 'oil', 'vegetable oil': 'oil', 'canola oil': 'oil',
    'all-purpose flour': 'flour', 'brown sugar': 'sugar', 'white sugar': 'sugar',
}

def get_synonyms(ingredient):
    """获取同义词"""
    synonyms = [ingredient]
    for key, value in SYNONYM_MAP.items():
        if key in ingredient:
            synonyms.append(ingredient.replace(key, value))
    return list(set(synonyms))

def string_similarity(s1, s2):
    """计算字符串相似度"""
    return SequenceMatcher(None, s1, s2).ratio()

def match_ingredient_to_usda(foodcom_ing, usda_dict, usda_list_for_fuzzy):
    """为单个Food.com食材匹配USDA（优化版：使用字典索引）"""
    synonyms = get_synonyms(foodcom_ing)

    best_match = None
    best_score = 0
    best_method = None

    # 策略1: 精确匹配（O(1)查询）
    for syn in synonyms:
        if syn in usda_dict:
            return usda_dict[syn], 1.0, 'exact'

    # 策略2: 包含匹配（遍历字典键）
    for syn in synonyms:
        # USDA名包含Food.com名
        candidates = []
        for usda_name, usda_data in usda_dict.items():
            if syn in usda_name:
                score = len(syn) / len(usda_name)
                candidates.append((usda_data, score))

        if candidates:
            # 选择最高分
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_candidate = candidates[0]
            if best_candidate[1] > best_score:
                best_match = best_candidate[0]
                best_score = best_candidate[1]
                best_method = 'contains_usda'

        # Food.com名包含USDA名（词拆分）
        words = syn.split()
        for word in words:
            if len(word) > 3 and word in usda_dict:
                score = 0.7
                if score > best_score:
                    best_match = usda_dict[word]
                    best_score = score
                    best_method = 'word_match'

    # 策略3: 模糊匹配（限制数量，避免性能问题）
    if best_score < 0.85:
        for usda_name, usda_data in usda_list_for_fuzzy[:2000]:  # 只对前2000个
            for syn in synonyms:
                sim = string_similarity(syn, usda_name)
                if sim > best_score and sim >= 0.75:
                    best_match = usda_data
                    best_score = sim
                    best_method = 'fuzzy'

    if best_match is not None:
        return best_match, best_score, best_method

    return None, 0.0, 'unmatched'

def match_batch(args):
    """批量匹配（用于多进程）"""
    batch_ingredients, usda_dict, usda_list_for_fuzzy = args
    results = []

    for foodcom_ing, freq in batch_ingredients:
        match_data, score, method = match_ingredient_to_usda(foodcom_ing, usda_dict, usda_list_for_fuzzy)

        result = {
            'foodcom_ingredient': foodcom_ing,
            'frequency': freq,
            'match_score': score,
            'match_method': method
        }

        if match_data is not None:
            result.update(match_data)
        else:
            result['usda_food_name'] = None

        results.append(result)

    return results

def main():
    # 路径配置
    recipes_file = r"work/recipebench/data/raw/foodcom/recipes.csv"
    usda_path = Path(r"work/recipebench/data/raw/usda")
    output_dir = Path(r"work/recipebench/data/4out/1005ingredient_nutri_mapping")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Food.com食材营养表构建（完整流程）")
    print("=" * 80)

    # ========== 步骤1：构建USDA完整营养表 ==========
    print(f"\n{'='*80}")
    print("步骤1：构建USDA完整营养表")
    print(f"{'='*80}")

    print(f"\n[1.1] 加载USDA原始数据...")
    df_food = pd.read_csv(usda_path / 'food.csv', low_memory=False)
    df_nutrient = pd.read_csv(usda_path / 'nutrient.csv', low_memory=False)
    df_food_nutrient = pd.read_csv(usda_path / 'food_nutrient.csv', low_memory=False)

    print(f"  食材数: {len(df_food):,}")
    print(f"  营养素数: {len(df_nutrient):,}")
    print(f"  关联记录数: {len(df_food_nutrient):,}")

    print(f"\n[1.2] 映射核心营养素...")
    core_nutrients = {
        'Energy (Atwater General Factors)': 'energy_kcal',
        'Energy (Atwater Specific Factors)': 'energy_kcal_alt',
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
    }

    nutrient_id_to_short = {}
    for _, row in df_nutrient.iterrows():
        nutrient_name = str(row['name']).strip()
        for long_name, short_name in core_nutrients.items():
            if nutrient_name == long_name or nutrient_name.startswith(long_name):
                if row['id'] not in nutrient_id_to_short:
                    nutrient_id_to_short[row['id']] = short_name
                break

    print(f"  匹配到核心营养素: {len(set(nutrient_id_to_short.values()))}")

    print(f"\n[1.3] 构建宽表（pivot）...")
    df_fn_filtered = df_food_nutrient[
        df_food_nutrient['nutrient_id'].isin(nutrient_id_to_short.keys())
    ].copy()

    df_fn_filtered['nutrient_short'] = df_fn_filtered['nutrient_id'].map(nutrient_id_to_short)
    df_fn_filtered = df_fn_filtered.drop_duplicates(subset=['fdc_id', 'nutrient_short'], keep='first')

    df_wide = df_fn_filtered.pivot_table(
        index='fdc_id',
        columns='nutrient_short',
        values='amount',
        aggfunc='first'
    ).reset_index()

    print(f"  Pivot后: {len(df_wide):,} 食材")

    # 合并食材名称
    df_wide = df_wide.merge(df_food[['fdc_id', 'description']], on='fdc_id', how='left')
    df_wide = df_wide.rename(columns={'fdc_id': 'food_id', 'description': 'food_name'})

    # 合并多个能量字段
    energy_cols = ['energy_kcal', 'energy_kcal_alt', 'energy_kcal_basic']
    for col in energy_cols:
        if col in df_wide.columns and col != 'energy_kcal':
            if 'energy_kcal' not in df_wide.columns:
                df_wide['energy_kcal'] = df_wide[col]
            else:
                df_wide['energy_kcal'] = df_wide['energy_kcal'].fillna(df_wide[col])
            df_wide = df_wide.drop(columns=[col])

    # 合并糖分字段
    if 'sugars_total_g_alt' in df_wide.columns:
        if 'sugars_total_g' not in df_wide.columns:
            df_wide['sugars_total_g'] = df_wide['sugars_total_g_alt']
        else:
            df_wide['sugars_total_g'] = df_wide['sugars_total_g'].fillna(df_wide['sugars_total_g_alt'])
        df_wide = df_wide.drop(columns=['sugars_total_g_alt'])

    # 添加标准化名称
    df_wide['food_name_normalized'] = df_wide['food_name'].apply(normalize_for_matching)

    df_usda = df_wide.copy()
    print(f"  ✓ USDA营养表构建完成: {len(df_usda):,} 食材")

    # ========== 步骤2：提取Food.com食材 ==========
    print(f"\n{'='*80}")
    print("步骤2：提取Food.com食材")
    print(f"{'='*80}")

    print(f"\n[2.1] 加载Food.com食谱...")
    df_recipes = pd.read_csv(recipes_file, low_memory=False)
    print(f"  加载 {len(df_recipes):,} 个食谱")

    print(f"\n[2.2] 统计食材...")
    ingredient_counter = Counter()
    for idx, row in tqdm(df_recipes.iterrows(), total=len(df_recipes), desc="  统计中"):
        parts = parse_r_vector(row['RecipeIngredientParts'])
        for part in parts:
            if part:
                normalized = normalize_for_matching(part)
                if normalized and len(normalized) > 2:
                    ingredient_counter[normalized] += 1

    print(f"  ✓ 统计到 {len(ingredient_counter):,} 个独立食材")

    # ========== 步骤3：匹配（多进程优化） ==========
    print(f"\n{'='*80}")
    print("步骤3：Food.com食材匹配USDA营养数据（多进程加速）")
    print(f"{'='*80}")

    print(f"\n[3.1] 构建USDA字典索引...")
    # 将DataFrame转为字典（O(1)查询）
    usda_dict = {}
    for _, row in df_usda.iterrows():
        usda_name_norm = row['food_name_normalized']
        # 转为字典
        row_dict = {
            'usda_food_name': row['food_name'],
            **{col: row.get(col) for col in df_usda.columns
               if col not in ['food_id', 'food_name', 'food_name_normalized']}
        }
        usda_dict[usda_name_norm] = row_dict

    # 为模糊匹配准备列表（按频率排序，高频的优先）
    usda_list_for_fuzzy = list(usda_dict.items())

    print(f"  ✓ 索引完成: {len(usda_dict):,} 食材")

    print(f"\n[3.2] 开始多进程匹配...")

    # 准备批次
    ingredients_list = list(ingredient_counter.items())
    n_cores = max(1, mp.cpu_count() - 2)
    batch_size = max(100, len(ingredients_list) // (n_cores * 4))

    batches = []
    for i in range(0, len(ingredients_list), batch_size):
        batch = ingredients_list[i:i+batch_size]
        batches.append((batch, usda_dict, usda_list_for_fuzzy))

    print(f"  使用 {n_cores} 核心，分 {len(batches)} 批处理")

    # 多进程匹配
    results = []
    with mp.Pool(n_cores) as pool:
        batch_results = list(tqdm(
            pool.imap(match_batch, batches),
            total=len(batches),
            desc="  匹配中"
        ))

        for batch_result in batch_results:
            results.extend(batch_result)

    df_result = pd.DataFrame(results)

    # ========== 步骤4：保存 ==========
    print(f"\n{'='*80}")
    print("步骤4：保存结果")
    print(f"{'='*80}")

    # 完整版
    output_file = output_dir / 'foodcom_ingredient_nutrition.csv'
    df_result.to_csv(output_file, index=False, encoding='utf-8')
    print(f"  ✓ 完整版: {output_file}")

    # 高质量版（匹配分数>=0.75）
    df_hq = df_result[df_result['match_score'] >= 0.75].copy()
    hq_file = output_dir / 'foodcom_ingredient_nutrition_hq.csv'
    df_hq.to_csv(hq_file, index=False, encoding='utf-8')
    print(f"  ✓ 高质量版: {hq_file} ({len(df_hq):,} 条)")

    # 未匹配列表
    df_unmatched = df_result[df_result['match_method'] == 'unmatched'].copy()
    unmatched_file = output_dir / 'foodcom_ingredient_unmatched.csv'
    df_unmatched[['foodcom_ingredient', 'frequency']].to_csv(unmatched_file, index=False, encoding='utf-8')
    print(f"  ✓ 未匹配列表: {unmatched_file} ({len(df_unmatched):,} 条)")

    # ========== 统计报告 ==========
    print(f"\n{'='*80}")
    print("最终统计报告")
    print(f"{'='*80}")

    total = len(df_result)
    matched = len(df_result[df_result['match_method'] != 'unmatched'])
    match_rate = matched / total * 100

    print(f"\n总食材数: {total:,}")
    print(f"匹配成功: {matched:,} ({match_rate:.1f}%)")
    print(f"未匹配: {total - matched:,} ({100 - match_rate:.1f}%)")

    print(f"\n匹配方法分布:")
    print(df_result['match_method'].value_counts())

    print(f"\n匹配分数分布:")
    print(df_result[df_result['match_score'] > 0]['match_score'].describe())

    # Top 1000高频食材匹配率
    top_1000 = df_result.head(1000)
    top_matched = len(top_1000[top_1000['match_method'] != 'unmatched'])
    print(f"\nTop 1000高频食材匹配率: {top_matched / 1000 * 100:.1f}%")

    # Top 20示例
    print(f"\nTop 20高频食材匹配示例:")
    print("=" * 100)
    for _, row in df_result.head(20).iterrows():
        usda_name = str(row.get('usda_food_name', 'N/A'))[:40]
        print(f"{row['foodcom_ingredient']:25s} → {usda_name:40s}  "
              f"[{row['match_method']:15s} {row['match_score']:.2f}]  freq={row['frequency']:6d}")

    # 营养覆盖率（仅匹配成功的）
    df_matched = df_result[df_result['match_method'] != 'unmatched']
    if len(df_matched) > 0:
        print(f"\n营养素覆盖率（匹配成功的{len(df_matched):,}个食材）:")
        nutrient_cols = [col for col in df_matched.columns
                        if col.endswith(('_g', '_mg', '_kcal', '_mcg', '_iu'))]
        for col in nutrient_cols:
            if col in df_matched.columns:
                coverage = df_matched[col].notna().sum() / len(df_matched) * 100
                print(f"  {col:25s}: {coverage:5.1f}%")

    print(f"\n{'='*80}")
    print(f"✓ 完成！总匹配率: {match_rate:.1f}%")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
