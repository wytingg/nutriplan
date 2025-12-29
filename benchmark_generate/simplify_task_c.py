#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task C 数据简化脚本
将完整的Task C数据转换为只包含 instruction + input + output 的格式
"""

import json
import sys


def format_violated_recipe(violated_recipe):
    """格式化违规食谱（input）"""
    lines = []

    lines.append(f"**{violated_recipe['title']}**")
    lines.append(f"Servings: {violated_recipe['servings']}")
    lines.append("")

    lines.append("Ingredients:")
    for ing in violated_recipe['ingredients']:
        lines.append(f"- {ing}")
    lines.append("")

    lines.append("Instructions:")
    for step in violated_recipe['steps']:
        lines.append(step)
    lines.append("")

    # 添加营养信息
    nutrition = violated_recipe['nutrition_per_serving']
    servings = violated_recipe['servings']

    total_energy = nutrition['energy_kcal'] * servings
    total_protein = nutrition['protein_g'] * servings
    total_carbs = nutrition['carbohydrates_g'] * servings
    total_fat = nutrition['fat_g'] * servings
    total_fiber = nutrition['fiber_g'] * servings
    total_sodium = nutrition['sodium_mg'] * servings

    lines.append(f"Nutrition (total for {servings} servings):")
    lines.append(f"{total_energy:.0f} kcal, {total_protein:.0f}g protein, {total_carbs:.0f}g carbs, {total_fat:.0f}g fat, {total_fiber:.0f}g fiber, {total_sodium:.0f}mg sodium")

    return "\n".join(lines)


def format_diagnosis(violations):
    """格式化诊断结果"""
    lines = []

    lines.append("**Diagnosis:**")
    lines.append("")

    for i, v in enumerate(violations, 1):
        if v['type'] == 'nutrition_violation':
            field = v['field']

            if field == 'sodium_mg':
                lines.append(f"{i}. Sodium violation:")
                lines.append(f"   - Actual: {v['actual']:.0f}mg")
                lines.append(f"   - Limit: {v['limit']:.0f}mg")
                lines.append(f"   - Severity: {v['severity']}")

            elif field == 'protein_amdr':
                lines.append(f"{i}. Protein AMDR violation:")
                lines.append(f"   - Actual: {v['actual_pct']:.1f}%")
                lines.append(f"   - Target: {v['target_pct']:.1f}%")
                lines.append(f"   - Severity: {v['severity']}")

            elif field == 'fat_amdr':
                lines.append(f"{i}. Fat AMDR violation:")
                lines.append(f"   - Actual: {v['actual_pct']:.1f}%")
                lines.append(f"   - Target: {v['target_pct']:.1f}%")
                lines.append(f"   - Severity: {v['severity']}")

            elif field == 'carb_amdr':
                lines.append(f"{i}. Carbohydrate AMDR violation:")
                lines.append(f"   - Actual: {v['actual_pct']:.1f}%")
                lines.append(f"   - Target: {v['target_pct']:.1f}%")
                lines.append(f"   - Severity: {v['severity']}")

            elif field == 'energy_kcal':
                lines.append(f"{i}. Energy violation:")
                lines.append(f"   - Actual: {v['actual']:.0f} kcal/serving")
                lines.append(f"   - Target: {v['target']:.0f} kcal/serving")
                lines.append(f"   - Severity: {v['severity']}")

            elif field == 'fiber_g':
                lines.append(f"{i}. Fiber violation:")
                lines.append(f"   - Actual: {v['actual']:.0f}g total")
                lines.append(f"   - Minimum: {v['minimum']:.0f}g")
                lines.append(f"   - Severity: {v['severity']}")

            elif field == 'saturated_fat_g':
                lines.append(f"{i}. Saturated fat violation:")
                lines.append(f"   - Actual: {v['actual_pct']:.1f}% of energy")
                lines.append(f"   - Limit: {v['limit_pct']:.1f}%")
                lines.append(f"   - Severity: {v['severity']}")

            elif field == 'sugars_g':
                lines.append(f"{i}. Sugars violation:")
                lines.append(f"   - Actual: {v['actual_pct']:.1f}% of energy")
                lines.append(f"   - Limit: {v['limit_pct']:.1f}%")
                lines.append(f"   - Severity: {v['severity']}")

        elif v['type'] == 'preference_violation':
            if v['subtype'] == 'disliked_ingredient_added':
                lines.append(f"{i}. Preference violation:")
                lines.append(f"   - Disliked ingredient added: {v['ingredient']}")
                lines.append(f"   - Severity: {v['severity']}")
            elif v['subtype'] == 'liked_ingredient_removed':
                lines.append(f"{i}. Preference violation:")
                lines.append(f"   - Liked ingredient removed: {v['ingredient']}")
                lines.append(f"   - Severity: {v['severity']}")

        lines.append("")

    return "\n".join(lines)


def format_corrections(corrections):
    """格式化修正方案"""
    lines = []

    lines.append("**Corrections:**")
    lines.append("")

    for i, c in enumerate(corrections, 1):
        action = c['action']

        if action == 'reduce_quantity':
            lines.append(f"{i}. Reduce {c['ingredient_name']}:")
            lines.append(f"   - From: {c['original_quantity']:.1f}g")
            lines.append(f"   - To: {c['new_quantity']:.1f}g")
            lines.append(f"   - Reason: {c['reason']}")

        elif action == 'increase_quantity':
            lines.append(f"{i}. Increase {c['ingredient_name']}:")
            lines.append(f"   - From: {c['original_quantity']:.1f}g")
            lines.append(f"   - To: {c['new_quantity']:.1f}g")
            lines.append(f"   - Reason: {c['reason']}")

        elif action == 'remove_ingredient':
            lines.append(f"{i}. Remove {c['ingredient_name']}:")
            lines.append(f"   - Reason: {c['reason']}")

        elif action == 'add_ingredient':
            lines.append(f"{i}. Add {c['ingredient_name']}:")
            lines.append(f"   - Quantity: {c['quantity']:.1f}g")
            lines.append(f"   - Reason: {c['reason']}")

        lines.append("")

    return "\n".join(lines)


def format_corrected_recipe(corrected_recipe):
    """格式化修正后的食谱"""
    lines = []

    lines.append("**Corrected Recipe:**")
    lines.append("")

    lines.append("Ingredients:")
    for ing in corrected_recipe['ingredients']:
        lines.append(f"- {ing}")
    lines.append("")

    # 添加营养信息
    nutrition = corrected_recipe['nutrition_per_serving']

    # 假设4人份
    servings = 4
    total_energy = nutrition['energy_kcal'] * servings
    total_protein = nutrition['protein_g'] * servings
    total_carbs = nutrition['carbohydrates_g'] * servings
    total_fat = nutrition['fat_g'] * servings
    total_fiber = nutrition['fiber_g'] * servings
    total_sodium = nutrition['sodium_mg'] * servings

    lines.append(f"Nutrition (total for {servings} servings):")
    lines.append(f"{total_energy:.0f} kcal, {total_protein:.0f}g protein, {total_carbs:.0f}g carbs, {total_fat:.0f}g fat, {total_fiber:.0f}g fiber, {total_sodium:.0f}mg sodium")

    return "\n".join(lines)


def simplify_task_c(input_file, output_file):
    """
    简化Task C数据

    格式: instruction + input (violated_recipe) + output (diagnosis + corrections + corrected_recipe)
    """
    print(f"Loading Task C data from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    print(f"Total samples: {len(samples)}")
    print()

    simplified_samples = []

    for sample in samples:
        # 提取字段
        instruction = sample['instruction']
        violated_recipe = sample['input']['violated_recipe']
        violations = sample['output']['diagnosis']
        corrections = sample['output']['corrections']
        corrected_recipe = sample['output']['corrected_recipe']

        # 格式化
        input_text = format_violated_recipe(violated_recipe)

        output_parts = []
        output_parts.append(format_diagnosis(violations))
        output_parts.append("")
        output_parts.append(format_corrections(corrections))
        output_parts.append("")
        output_parts.append(format_corrected_recipe(corrected_recipe))

        output_text = "\n".join(output_parts)

        # 创建简化样本
        simplified = {
            'instruction': instruction,
            'input': input_text,
            'output': output_text
        }

        simplified_samples.append(simplified)

    # 保存
    print(f"Saving simplified data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in simplified_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Saved {len(simplified_samples)} simplified samples")
    print()
    print("Sample preview:")
    if simplified_samples:
        sample = simplified_samples[0]
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Input: {sample['input'][:100]}...")
        print(f"Output: {sample['output'][:100]}...")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python simplify_task_c.py <input_file> <output_file>")
        print()
        print("Example:")
        print("  python simplify_task_c.py task_c_val_from_kg.jsonl task_c_val_simplified.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    simplify_task_c(input_file, output_file)
