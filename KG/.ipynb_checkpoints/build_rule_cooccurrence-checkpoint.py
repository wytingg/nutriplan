#!/usr/bin/env python3
"""
规则1：食材共现频率规则库构建
从Food.com食谱中统计食材pair共现，计算PMI分数
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
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

def normalize_ingredient(ing):
    """标准化食材名称"""
    ing = ing.lower().strip()
    # 移除常见量词
    ing = ing.replace('fresh ', '').replace('dried ', '').replace('frozen ', '')
    ing = ing.replace('chopped ', '').replace('minced ', '').replace('sliced ', '')
    return ing

def extract_ingredients_from_row(row):
    """从单行提取标准化食材列表"""
    try:
        parts = parse_r_vector(row['RecipeIngredientParts'])
        if not parts:
            return []
        ings = [normalize_ingredient(p) for p in parts if p]
        return list(set(ings))  # 去重
    except:
        return []

def process_chunk(chunk_data):
    """处理数据块：统计共现"""
    chunk_df, chunk_id = chunk_data

    local_ing_count = Counter()
    local_pair_count = defaultdict(int)

    for _, row in chunk_df.iterrows():
        ings = extract_ingredients_from_row(row)
        if len(ings) < 2:
            continue

        # 统计单个食材
        for ing in ings:
            local_ing_count[ing] += 1

        # 统计食材对（按字母序排序确保唯一性）
        for ing1, ing2 in combinations(sorted(ings), 2):
            local_pair_count[(ing1, ing2)] += 1

    return local_ing_count, local_pair_count

def main():
    input_file = r"work/recipebench/data/raw/foodcom/recipes.csv"
    output_dir = Path(r"work/recipebench/data/nutrition_rule")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("食材共现规则库构建")
    print("=" * 60)

    # 读取数据
    print(f"\n[1/5] 加载数据: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    total_recipes = len(df)
    print(f"  ✓ 加载 {total_recipes:,} 个食谱")

    # 分块处理（多进程加速）
    print(f"\n[2/5] 统计食材共现（多进程）...")
    n_cores = max(1, mp.cpu_count() - 1)
    chunk_size = max(1000, total_recipes // (n_cores * 4))
    chunks = [(df[i:i+chunk_size], i//chunk_size)
              for i in range(0, len(df), chunk_size)]

    print(f"  使用 {n_cores} 核心，分 {len(chunks)} 块处理")

    with mp.Pool(n_cores) as pool:
        results = list(tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="  处理中"
        ))

    # 合并结果
    print(f"\n[3/5] 合并统计结果...")
    ing_count = Counter()
    pair_count = defaultdict(int)

    for local_ing, local_pair in results:
        ing_count.update(local_ing)
        for pair, count in local_pair.items():
            pair_count[pair] += count

    print(f"  ✓ 统计到 {len(ing_count):,} 个独立食材")
    print(f"  ✓ 统计到 {len(pair_count):,} 个食材对")

    # 计算PMI和置信度
    print(f"\n[4/5] 计算PMI分数...")
    N = total_recipes

    rules = []
    for (ing1, ing2), count_ab in tqdm(pair_count.items(), desc="  计算中"):
        count_a = ing_count[ing1]
        count_b = ing_count[ing2]

        # PMI(A,B) = log(P(A,B) / (P(A)*P(B)))
        #          = log((count_ab/N) / ((count_a/N)*(count_b/N)))
        #          = log((count_ab * N) / (count_a * count_b))
        try:
            pmi = np.log((count_ab * N) / (count_a * count_b))
        except:
            pmi = 0.0

        # 置信度：P(B|A) = count_ab / count_a
        confidence = count_ab / count_a

        # 支持度：P(A,B)
        support = count_ab / N

        rules.append({
            'ingredient_1': ing1,
            'ingredient_2': ing2,
            'cooccurrence_count': count_ab,
            'pmi_score': round(pmi, 4),
            'confidence': round(confidence, 4),
            'support': round(support, 6),
            'freq_1': count_a,
            'freq_2': count_b
        })

    # 按PMI降序排序
    rules.sort(key=lambda x: x['pmi_score'], reverse=True)

    # 保存完整规则库
    print(f"\n[5/5] 保存规则库...")

    # CSV格式（完整版）
    df_rules = pd.DataFrame(rules)
    full_output = output_dir / 'ingredient_cooccurrence_full.csv'
    df_rules.to_csv(full_output, index=False, encoding='utf-8')
    print(f"  ✓ 完整版: {full_output} ({len(df_rules):,} 条规则)")

    # 高质量筛选版（PMI>2.0 且 count>10）
    df_filtered = df_rules[
        (df_rules['pmi_score'] > 2.0) &
        (df_rules['cooccurrence_count'] >= 10)
    ].copy()
    filtered_output = output_dir / 'ingredient_cooccurrence.csv'
    df_filtered.to_csv(filtered_output, index=False, encoding='utf-8')
    print(f"  ✓ 高质量版: {filtered_output} ({len(df_filtered):,} 条规则)")

    # JSON格式（快速查询）
    cooccur_dict = {}
    for _, row in df_filtered.iterrows():
        ing1, ing2 = row['ingredient_1'], row['ingredient_2']
        score = row['pmi_score']

        if ing1 not in cooccur_dict:
            cooccur_dict[ing1] = {}
        if ing2 not in cooccur_dict:
            cooccur_dict[ing2] = {}

        cooccur_dict[ing1][ing2] = float(score)
        cooccur_dict[ing2][ing1] = float(score)

    json_output = output_dir / 'ingredient_cooccurrence.json'
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(cooccur_dict, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON版: {json_output}")

    # 统计报告
    print(f"\n{'='*60}")
    print("统计报告")
    print(f"{'='*60}")
    print(f"总食谱数: {total_recipes:,}")
    print(f"独立食材数: {len(ing_count):,}")
    print(f"食材对总数: {len(pair_count):,}")
    print(f"高质量规则数 (PMI>2, count>=10): {len(df_filtered):,}")
    print(f"\nPMI分布:")
    print(f"  最高: {df_rules['pmi_score'].max():.2f}")
    print(f"  中位数: {df_rules['pmi_score'].median():.2f}")
    print(f"  平均: {df_rules['pmi_score'].mean():.2f}")

    print(f"\nTop 10 最强共现对:")
    for i, row in df_rules.head(10).iterrows():
        print(f"  {row['ingredient_1']:20s} + {row['ingredient_2']:20s}  "
              f"PMI={row['pmi_score']:6.2f}  count={row['cooccurrence_count']:5d}")

    print(f"\n✓ 完成！规则库已保存到 {output_dir}")

# if __name__ == '__main__':
#     main()
     #!/usr/bin/env python3
#!/usr/bin/env python3
# import pandas as pd

# # 读取高质量版
# df = pd.read_csv('work/recipebench/data/nutrition_rule/ingredient_cooccurrence.csv')

# print("高质量规则库（PMI>2, count>=10）Top 20:")
# print("="*80)
# for i, row in df.head(20).iterrows():
#     print(f"{row['ingredient_1']:25s} + {row['ingredient_2']:25s}  "
#           f"PMI={row['pmi_score']:5.2f}  count={row['cooccurrence_count']:5d}")

# print("\n按共现次数排序 Top 20:")
# print("="*80)
# df_by_count = df.sort_values('cooccurrence_count', ascending=False)
# for i, row in df_by_count.head(20).iterrows():
#     print(f"{row['ingredient_1']:25s} + {row['ingredient_2']:25s}  "
#           f"count={row['cooccurrence_count']:6d}  PMI={row['pmi_score']:5.2f}")

# # 统计
# print(f"\n统计:")
# print(f"  平均PMI: {df['pmi_score'].mean():.2f}")
# print(f"  平均共现次数: {df['cooccurrence_count'].mean():.1f}")
# print(f"  共现次数>100的规则: {len(df[df['cooccurrence_count']>100]):,} 条")
