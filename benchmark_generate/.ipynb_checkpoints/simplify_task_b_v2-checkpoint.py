"""
Simplify Task B: Recipe Generation (Version 2)
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

    # 微量营养素
    if 'sodium_mg' in nutr_dict:
        nutrients.append(f"Sodium: {int(nutr_dict['sodium_mg'])}mg")
    if 'potassium_mg' in nutr_dict:
        nutrients.append(f"Potassium: {int(nutr_dict.get('potassium_mg', 0))}mg")
    if 'calcium_mg' in nutr_dict:
        nutrients.append(f"Calcium: {int(nutr_dict.get('calcium_mg', 0))}mg")
    if 'iron_mg' in nutr_dict:
        nutrients.append(f"Iron: {int(nutr_dict.get('iron_mg', 0))}mg")
    if 'vitamin_c_mg' in nutr_dict:
        nutrients.append(f"Vitamin C: {int(nutr_dict.get('vitamin_c_mg', 0))}mg")

    return " | ".join(nutrients) if nutrients else "Nutrition data unavailable"

def simplify_task_b_standard(sample):
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

            instruction = (
                f"Generate a recipe for a {user['age']}-year-old {user['gender']} "
                f"({user['physiological_state']}) with the following nutritional needs:\n"
                f"{format_nutrition_rni(rni)}"
            )

            # 添加食材偏好
            if 'liked_ingredients' in user and user['liked_ingredients']:
                liked = [ing.get('name', ing) if isinstance(ing, dict) else ing
                        for ing in user['liked_ingredients']]
                instruction += f"\n\nPreferred ingredients: {', '.join(liked[:5])}"

            if 'disliked_ingredients' in user and user['disliked_ingredients']:
                disliked = [ing.get('name', ing) if isinstance(ing, dict) else ing
                           for ing in user['disliked_ingredients']]
                instruction += f"\nAvoid: {', '.join(disliked[:5])}"

        # 3. 构建 output（保留完整信息，但格式清晰）
        recipe = sample['output']

        output = f"**{recipe['title']}**\n\n"
        output += f"**Servings:** {recipe['servings']}\n\n"

        # 食材
        output += "**Ingredients:**\n"
        for ing in recipe['ingredients']:
            output += f"- {ing}\n"

        # 步骤
        output += "\n**Instructions:**\n"
        for i, step in enumerate(recipe['steps'], 1):
            # 如果步骤已有序号，保留原样，否则添加
            step_clean = step.strip()
            if step_clean[0].isdigit() and '.' in step_clean[:3]:
                output += f"{step_clean}\n"
            else:
                output += f"{i}. {step_clean}\n"

        # 完整营养信息（保留所有有效字段）
        if 'nutrition_per_serving' in recipe:
            nutr = recipe['nutrition_per_serving']
            output += f"\n**Nutrition per Serving:**\n"
            output += format_recipe_nutrition(nutr)

        return {
            "instruction": instruction,
            "output": output
        }

    except Exception as e:
        print(f"Error: {e}")
        return None

def simplify_task_b_chat(sample):
    """
    Chat 格式：role-based messages
    """
    try:
        standard = simplify_task_b_standard(sample)
        if not standard:
            return None

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional nutritionist and chef AI assistant. Generate recipes that precisely meet users' nutritional requirements and ingredient preferences."
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

    input_file = Path(r"work/recipebench/data/10large_scale_datasets/task_b_test_from_kg.jsonl")

    if args.format == 'standard':
        output_file = Path(r"work/recipebench/data/10large_scale_datasets/task_b_test_simplified.jsonl")
        simplify_fn = simplify_task_b_standard
    else:
        output_file = Path(r"work/recipebench/data/10large_scale_datasets/task_b_test_chat.jsonl")
        simplify_fn = simplify_task_b_chat

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

    print(f"\n✅ Task B Simplification Complete!")
    print(f"   Success: {success_count} samples")
    print(f"   Errors: {error_count} samples")
    print(f"   Output: {output_file}")

    # 显示示例
    print(f"\n📄 First example:")
    with open(output_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        sample = json.loads(line)
        if args.format == 'standard':
            print(f"Instruction: {sample['instruction'][:150]}...")
            print(f"\nOutput: {sample['output'][:300]}...")
        else:
            print(f"System: {sample['messages'][0]['content']}")
            print(f"\nUser: {sample['messages'][1]['content'][:150]}...")
            print(f"\nAssistant: {sample['messages'][2]['content'][:300]}...")

if __name__ == "__main__":
    main()
