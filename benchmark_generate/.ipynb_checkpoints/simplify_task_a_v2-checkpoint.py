"""
Simplify Task A: Recipe Ranking (Version 2)
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
    return (
        f"Energy: {int(rni_dict['energy_kcal'])} kcal, "
        f"Protein: {int(rni_dict['protein_g'])}g, "
        f"Carbs: {int(rni_dict['carbohydrate_g'])}g, "
        f"Fat: {int(rni_dict['fat_g'])}g, "
        f"Fiber: {int(rni_dict['fiber_g'])}g, "
        f"Sugar: {int(rni_dict['added_sugar_g'])}g, "
        f"Sat.Fat: {int(rni_dict['saturated_fat_g'])}g, "
        f"Trans.Fat: {round(rni_dict['trans_fat_g'], 1)}g, "
        f"Sodium: {int(rni_dict['sodium_mg'])}mg, "
        f"Potassium: {int(rni_dict['potassium_mg'])}mg, "
        f"Calcium: {int(rni_dict['calcium_mg'])}mg, "
        f"Iron: {int(rni_dict['iron_mg'])}mg, "
        f"Vitamin C: {int(rni_dict['vitamin_c_mg'])}mg, "
        f"Vitamin D: {int(rni_dict['vitamin_d_ug'])}μg, "
        f"Folate: {int(rni_dict['folate_ug'])}μg"
    )

def format_nutrition_per_serving(nutr_dict):
    """格式化食谱营养信息（只保留有的字段，四舍五入）"""
    parts = []
    if 'energy_kcal' in nutr_dict:
        parts.append(f"{int(nutr_dict['energy_kcal'])} kcal")
    if 'protein_g' in nutr_dict:
        parts.append(f"{int(nutr_dict['protein_g'])}g protein")
    if 'carbohydrate_g' in nutr_dict or 'carbohydrates_g' in nutr_dict:
        carbs = nutr_dict.get('carbohydrate_g', nutr_dict.get('carbohydrates_g', 0))
        parts.append(f"{int(carbs)}g carbs")
    if 'fat_g' in nutr_dict:
        parts.append(f"{int(nutr_dict['fat_g'])}g fat")
    if 'fiber_g' in nutr_dict:
        parts.append(f"{int(nutr_dict['fiber_g'])}g fiber")
    if 'sodium_mg' in nutr_dict:
        parts.append(f"{int(nutr_dict['sodium_mg'])}mg sodium")

    return ", ".join(parts) if parts else "nutrition info unavailable"

def simplify_task_a_standard(sample):
    """
    标准格式：instruction-output
    保留原始 instruction，只简化 output
    """
    try:
        # 1. 保留原始 instruction（不修改，保持多样性）
        instruction = sample.get('instruction', '')

        # 2. 如果 instruction 为空，从 user_profile 构建
        if not instruction:
            user = sample['user_profile']
            rni = user['nutrition_rni']
            instruction = (
                f"Based on nutritional requirements ({format_nutrition_rni(rni)}), "
                f"rank recipes for a {user['age']}-year-old {user['gender']} ({user['physiological_state']})."
            )

        # 3. 构建简化的 output（保留完整营养信息）
        ranked_recipes = sample.get('ranked_recipes', [])

        output_lines = []
        for i, recipe in enumerate(ranked_recipes[:5], 1):  # 保留 top 5
            # 重新计算总分数：四类分数均等加权（删除 cooccurrence）
            if 'score_breakdown' in recipe:
                scores = recipe['score_breakdown']
                nutrition_score = scores.get('nutrition_match', 0)
                preference_score = scores.get('preference_match', 0)
                complementarity_score = scores.get('complementarity', 0)
                balance_score = scores.get('balance', 0)

                # 四类分数均等加权求和
                recalculated_score = (nutrition_score + preference_score + complementarity_score + balance_score) / 4.0
            else:
                recalculated_score = recipe.get('overall_score', 0)
                nutrition_score = 0
                preference_score = 0
                complementarity_score = 0
                balance_score = 0

            line = f"{i}. **{recipe['recipe_name']}** (Score: {recalculated_score:.3f})"

            # 保留完整的营养信息
            if 'nutrition_per_serving' in recipe:
                nutr = recipe['nutrition_per_serving']
                line += f"\n   Nutrition: {format_nutrition_per_serving(nutr)}"

            # 保留四类评分维度（删除 cooccurrence）
            if 'score_breakdown' in recipe:
                line += f"\n   Scores: nutrition={nutrition_score:.3f}, "
                line += f"preference={preference_score:.3f}, "
                line += f"complementarity={complementarity_score:.3f}, "
                line += f"balance={balance_score:.3f}"

            output_lines.append(line)

        output = "\n\n".join(output_lines)

        return {
            "instruction": instruction,
            "output": output
        }

    except Exception as e:
        print(f"Error: {e}")
        return None

def simplify_task_a_chat(sample):
    """
    Chat 格式：role-based messages
    适用于 chat 模型（如 Llama-3-Instruct, Qwen-Chat）
    """
    try:
        standard = simplify_task_a_standard(sample)
        if not standard:
            return None

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional nutritionist AI assistant. Rank recipes based on user's nutritional requirements and preferences."
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

    input_file = Path(r"work/recipebench/data/10large_scale_datasets/task_a_test_discriminative.jsonl")

    if args.format == 'standard':
        output_file = Path(r"work/recipebench/data/10large_scale_datasets/task_a_test_simplified.jsonl")
        simplify_fn = simplify_task_a_standard
    else:
        output_file = Path(r"work/recipebench/data/10large_scale_datasets/task_a_test_chat.jsonl")
        simplify_fn = simplify_task_a_chat

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

    print(f"\n✅ Task A Simplification Complete!")
    print(f"   Success: {success_count} samples")
    print(f"   Errors: {error_count} samples")
    print(f"   Output: {output_file}")

    # 显示示例
    print(f"\n📄 First 2 examples:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > 2:
                break
            sample = json.loads(line)
            print(f"\n--- Example {i} ---")
            if args.format == 'standard':
                print(f"Instruction: {sample['instruction'][:120]}...")
                print(f"Output: {sample['output'][:200]}...")
            else:
                print(f"System: {sample['messages'][0]['content']}")
                print(f"User: {sample['messages'][1]['content'][:120]}...")
                print(f"Assistant: {sample['messages'][2]['content'][:200]}...")

if __name__ == "__main__":
    main()
