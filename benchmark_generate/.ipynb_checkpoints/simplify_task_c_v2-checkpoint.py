"""
Simplify Task C: Recipe Editing (Version 2)
改进版本：
1. 保留原始 instruction 多样性
2. 保留完整的 15 种营养素信息
3. 提供 role-based 格式选项
"""
import json
from pathlib import Path
import argparse

def format_nutrition_rni(rni_dict):
    """格式化完整的 RNI 信息（15 种营养素）"""
    nutrients = []
    nutrients.append(f"Energy: {int(rni_dict['energy_kcal'])} kcal")
    nutrients.append(f"Protein: {int(rni_dict['protein_g'])}g")
    nutrients.append(f"Carbohydrates: {int(rni_dict['carbohydrate_g'])}g")
    nutrients.append(f"Fat: {int(rni_dict['fat_g'])}g")
    nutrients.append(f"Fiber: {int(rni_dict['fiber_g'])}g")
    nutrients.append(f"Added Sugar: {int(rni_dict['added_sugar_g'])}g")
    nutrients.append(f"Saturated Fat: {int(rni_dict['saturated_fat_g'])}g")
    nutrients.append(f"Trans Fat: {round(rni_dict['trans_fat_g'], 1)}g")
    nutrients.append(f"Sodium: {int(rni_dict['sodium_mg'])}mg")
    nutrients.append(f"Potassium: {int(rni_dict['potassium_mg'])}mg")
    nutrients.append(f"Calcium: {int(rni_dict['calcium_mg'])}mg")
    nutrients.append(f"Iron: {int(rni_dict['iron_mg'])}mg")
    nutrients.append(f"Vitamin C: {int(rni_dict['vitamin_c_mg'])}mg")
    nutrients.append(f"Vitamin D: {int(rni_dict['vitamin_d_ug'])}μg")
    nutrients.append(f"Folate: {int(rni_dict['folate_ug'])}μg")

    return " | ".join(nutrients)

def format_recipe_nutrition(nutr_dict):
    """格式化食谱营养信息（完整版）"""
    nutrients = []

    # 主要营养素
    if 'energy_kcal' in nutr_dict:
        nutrients.append(f"Energy: {int(nutr_dict['energy_kcal'])} kcal")
    if 'protein_g' in nutr_dict:
        nutrients.append(f"Protein: {int(nutr_dict['protein_g'])}g")
    if 'carbohydrates_g' in nutr_dict or 'carbohydrate_g' in nutr_dict:
        carbs = nutr_dict.get('carbohydrates_g', nutr_dict.get('carbohydrate_g', 0))
        nutrients.append(f"Carbs: {int(carbs)}g")
    if 'fat_g' in nutr_dict:
        nutrients.append(f"Fat: {int(nutr_dict['fat_g'])}g")
    if 'fiber_g' in nutr_dict:
        nutrients.append(f"Fiber: {int(nutr_dict['fiber_g'])}g")

    # 其他营养素
    if 'added_sugar_g' in nutr_dict:
        nutrients.append(f"Sugar: {int(nutr_dict['added_sugar_g'])}g")
    if 'saturated_fat_g' in nutr_dict:
        nutrients.append(f"Sat.Fat: {int(nutr_dict['saturated_fat_g'])}g")
    if 'trans_fat_g' in nutr_dict:
        nutrients.append(f"Trans.Fat: {round(nutr_dict['trans_fat_g'], 1)}g")
    if 'sodium_mg' in nutr_dict:
        nutrients.append(f"Sodium: {int(nutr_dict['sodium_mg'])}mg")
    if 'potassium_mg' in nutr_dict:
        nutrients.append(f"Potassium: {int(nutr_dict['potassium_mg'])}mg")
    if 'calcium_mg' in nutr_dict:
        nutrients.append(f"Calcium: {int(nutr_dict['calcium_mg'])}mg")
    if 'iron_mg' in nutr_dict:
        nutrients.append(f"Iron: {int(nutr_dict['iron_mg'])}mg")
    if 'vitamin_c_mg' in nutr_dict:
        nutrients.append(f"Vitamin C: {int(nutr_dict['vitamin_c_mg'])}mg")
    if 'vitamin_d_ug' in nutr_dict:
        nutrients.append(f"Vitamin D: {int(nutr_dict['vitamin_d_ug'])}μg")
    if 'folate_ug' in nutr_dict:
        nutrients.append(f"Folate: {int(nutr_dict['folate_ug'])}μg")

    return " | ".join(nutrients) if nutrients else "Nutrition data unavailable"

def simplify_task_c_standard(sample):
    """
    标准格式：instruction-output
    保留原始 instruction，完整的营养信息
    """
    try:
        # 1. 保留原始 instruction（不修改）
        instruction = sample.get('instruction', '')

        # 2. 如果 instruction 为空，构建基础版本
        if not instruction:
            user = sample['user_profile']
            rni = user['nutrition_rni']

            input_data = sample['input']
            violated_recipe = input_data['violated_recipe']
            violations = input_data.get('violations', [])

            instruction = (
                f"Correct this recipe for a {user['age']}-year-old {user['gender']} "
                f"({user['physiological_state']}) with daily nutritional needs:\n"
                f"{format_nutrition_rni(rni)}\n\n"
            )

            # 违规信息
            if violations:
                violation_types = list(set([v.get('type', 'unknown') for v in violations]))
                instruction += f"**Issues:** {', '.join(violation_types)}\n\n"

            # 原始食谱信息
            instruction += f"**Original Recipe: {violated_recipe['title']}**\n"
            instruction += f"Ingredients: {', '.join(violated_recipe['ingredients'][:7])}"
            if len(violated_recipe['ingredients']) > 7:
                instruction += f"... ({len(violated_recipe['ingredients'])} total)"

            # 原始营养信息
            if 'nutrition_per_serving' in violated_recipe:
                orig_nutr = violated_recipe['nutrition_per_serving']
                instruction += f"\n\nOriginal Nutrition (per serving):\n"
                instruction += format_recipe_nutrition(orig_nutr)

        # 3. 构建 output（保留完整信息）
        output_data = sample['output']
        output = ""

        # 诊断信息
        if 'diagnosis' in output_data and output_data['diagnosis']:
            diagnosis = output_data['diagnosis']
            output += "**Diagnosis:**\n"
            for diag in diagnosis[:3]:  # 前3个最重要的诊断
                field = diag.get('field', 'unknown')
                actual = diag.get('actual', 'N/A')
                target = diag.get('target', 'N/A')

                # 格式化数值
                if isinstance(actual, (int, float)):
                    actual = int(actual) if field.endswith('_g') or field.endswith('_mg') or field.endswith('_ug') or field.endswith('_kcal') else round(actual, 1)
                if isinstance(target, (int, float)):
                    target = int(target) if field.endswith('_g') or field.endswith('_mg') or field.endswith('_ug') or field.endswith('_kcal') else round(target, 1)

                output += f"- {field}: actual {actual}, target {target}\n"
            output += "\n"

        # 修正措施
        if 'corrections' in output_data and output_data['corrections']:
            corrections = output_data['corrections']
            output += "**Corrections Applied:**\n"
            for corr in corrections[:5]:  # 前5个修正
                action = corr.get('action', 'modify')
                ingredient = corr.get('ingredient_name', 'ingredient')
                reason = corr.get('reason', 'nutritional adjustment')

                line = f"- {action}: {ingredient}"

                if 'original_quantity' in corr and 'new_quantity' in corr:
                    orig_qty = corr['original_quantity']
                    new_qty = corr['new_quantity']

                    # 格式化数量
                    if isinstance(orig_qty, (int, float)):
                        orig_qty = int(orig_qty) if orig_qty > 10 else round(orig_qty, 1)
                    if isinstance(new_qty, (int, float)):
                        new_qty = int(new_qty) if new_qty > 10 else round(new_qty, 1)

                    line += f" from {orig_qty}g to {new_qty}g"

                line += f" ({reason})"
                output += line + "\n"
            output += "\n"

        # 修正后的食谱
        corrected = output_data['corrected_recipe']
        output += f"**Corrected Recipe: {corrected['title']}**\n\n"

        output += f"**Servings:** {corrected.get('servings', 'N/A')}\n\n"

        output += "**Ingredients:**\n"
        for ing in corrected['ingredients']:
            output += f"- {ing}\n"

        output += "\n**Instructions:**\n"
        for i, step in enumerate(corrected['steps'], 1):
            step_clean = step.strip()
            if step_clean and step_clean[0].isdigit() and '.' in step_clean[:3]:
                output += f"{step_clean}\n"
            else:
                output += f"{i}. {step_clean}\n"

        # 修正后的营养信息（完整）
        if 'nutrition_per_serving' in corrected:
            nutr = corrected['nutrition_per_serving']
            output += f"\n**Corrected Nutrition (per serving):**\n"
            output += format_recipe_nutrition(nutr)

        return {
            "instruction": instruction,
            "output": output
        }

    except Exception as e:
        print(f"Error: {e}")
        return None

def simplify_task_c_chat(sample):
    """
    Chat 格式：role-based messages
    """
    try:
        standard = simplify_task_c_standard(sample)
        if not standard:
            return None

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional nutritionist and chef AI assistant. Analyze recipe violations and provide corrected recipes that meet users' nutritional requirements and ingredient preferences."
                },
                {
                    "role": "user",
                    "content": standard["instruction"]
                },
                {
                    "role": "assistant",
                    "content": standard["output"]
                }
            ]
        }

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', choices=['standard', 'chat'], default='standard',
                        help='Output format: standard (instruction-output) or chat (role-based)')
    args = parser.parse_args()

    input_file = Path(r"work/recipebench/data/10large_scale_datasets/task_c_val_from_kg.jsonl")

    if args.format == 'standard':
        output_file = Path(r"work/recipebench/data/10large_scale_datasets/task_c_val__simplified.jsonl")
        simplify_fn = simplify_task_c_standard
    else:
        output_file = Path(r"work/recipebench/data/10large_scale_datasets/task_c_val_chat.jsonl")
        simplify_fn = simplify_task_c_chat

    print(f"Processing: {input_file}")
    print(f"Format: {args.format}")
    print(f"Output: {output_file}")

    success_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line_num, line in enumerate(f_in, 1):
                if not line.strip():
                    continue

                try:
                    sample = json.loads(line)
                    simplified = simplify_fn(sample)

                    if simplified:
                        f_out.write(json.dumps(simplified, ensure_ascii=False) + '\n')
                        success_count += 1
                    else:
                        error_count += 1

                    if line_num % 1000 == 0:
                        print(f"Processed {line_num} lines...")

                except Exception as e:
                    print(f"Error at line {line_num}: {e}")
                    error_count += 1

    print(f"\n✅ Task C Simplification Complete!")
    print(f"   Success: {success_count} samples")
    print(f"   Errors: {error_count} samples")
    print(f"   Output: {output_file}")

    # 显示示例
    print(f"\n📄 First example:")
    with open(output_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        if line:
            sample = json.loads(line)
            if args.format == 'standard':
                print(f"Instruction: {sample['instruction'][:200]}...")
                print(f"\nOutput: {sample['output'][:300]}...")
            else:
                print(f"System: {sample['messages'][0]['content']}")
                print(f"\nUser: {sample['messages'][1]['content'][:200]}...")
                print(f"\nAssistant: {sample['messages'][2]['content'][:300]}...")

if __name__ == "__main__":
    main()
