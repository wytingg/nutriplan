#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task B 数据过滤脚本 - 只保留100%完美匹配样本
创建日期: 2025-10-30
目标: 确保LLM不会学到错误行为（忽略用户要求的食材）
"""

import json
import sys


def check_ingredient_matching(user_profile, output_ingredients):
    """
    检查用户要求的食材是否100%出现在输出中

    返回: (match_rate, matched_list, missing_list)
    """
    liked_ingredients = [ing['name'].lower() for ing in user_profile.get('liked_ingredients', [])]

    if not liked_ingredients:
        # 没有liked_ingredients，默认通过
        return 1.0, [], []

    output_text = ' '.join([ing.lower() for ing in output_ingredients])

    matched = []
    missing = []

    for liked_ing in liked_ingredients:
        # 提取关键词
        key_words = liked_ing.split()
        is_matched = False

        for key_word in key_words:
            if len(key_word) > 2 and key_word in output_text:
                is_matched = True
                break

        if is_matched:
            matched.append(liked_ing)
        else:
            missing.append(liked_ing)

    match_rate = len(matched) / len(liked_ingredients) if liked_ingredients else 1.0

    return match_rate, matched, missing


def filter_perfect_match(input_file, output_file):
    """
    过滤Task B数据，只保留100%完美匹配的样本
    """
    print(f"Loading data from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    print(f"Total samples: {len(samples)}")
    print()

    # 统计
    stats = {
        'total': len(samples),
        'perfect_match': 0,
        'partial_match': 0,
        'poor_match': 0,
        'no_liked': 0
    }

    perfect_samples = []

    for sample in samples:
        user_profile = sample.get('user_profile', {})
        output = sample.get('output', {})
        output_ingredients = output.get('ingredients', [])

        match_rate, matched, missing = check_ingredient_matching(user_profile, output_ingredients)

        if not user_profile.get('liked_ingredients'):
            stats['no_liked'] += 1
            # 没有liked_ingredients的样本也保留
            perfect_samples.append(sample)
        elif match_rate == 1.0:
            stats['perfect_match'] += 1
            perfect_samples.append(sample)
        elif match_rate >= 0.5:
            stats['partial_match'] += 1
        else:
            stats['poor_match'] += 1

    # 保存过滤后的数据
    print(f"Saving filtered data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in perfect_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 打印统计
    print()
    print('=' * 80)
    print('FILTERING RESULTS')
    print('=' * 80)
    print()
    print(f"Total samples: {stats['total']}")
    print()
    print(f"Perfect match (100%): {stats['perfect_match']} ({stats['perfect_match']/stats['total']*100:.1f}%)")
    print(f"Partial match (50-99%): {stats['partial_match']} ({stats['partial_match']/stats['total']*100:.1f}%)")
    print(f"Poor match (<50%): {stats['poor_match']} ({stats['poor_match']/stats['total']*100:.1f}%)")
    print(f"No liked ingredients: {stats['no_liked']} ({stats['no_liked']/stats['total']*100:.1f}%)")
    print()
    print(f"Kept samples: {len(perfect_samples)} ({len(perfect_samples)/stats['total']*100:.1f}%)")
    print(f"Removed samples: {stats['total'] - len(perfect_samples)} ({(stats['total'] - len(perfect_samples))/stats['total']*100:.1f}%)")
    print()
    print(f"Output saved to: {output_file}")
    print()
    print('=' * 80)

    return stats, len(perfect_samples)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python filter_task_b_perfect_match.py <input_file> <output_file>")
        print()
        print("Example:")
        print("  python filter_task_b_perfect_match.py task_b_train_from_kg.jsonl task_b_train_PERFECT.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    filter_perfect_match(input_file, output_file)
