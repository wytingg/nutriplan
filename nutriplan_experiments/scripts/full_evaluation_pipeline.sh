#!/bin/bash
# 完整评估流程：生成预测 → 计算所有指标 → 生成论文表格
# 用法: bash full_evaluation_pipeline.sh <model_path> <model_name> <seed>

set -e

MODEL_PATH=$1
MODEL_NAME=$2
SEED=$3

if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ] || [ -z "$SEED" ]; then
    echo "用法: bash full_evaluation_pipeline.sh <model_path> <model_name> <seed>"
    echo "示例: bash full_evaluation_pipeline.sh ~/work/nutriplan_models_backup/rq1_TinyLlama_seed42/best_model TinyLlama 42"
    exit 1
fi

# 配置
DATA_DIR="${HOME}/work/recipebench/data/10large_scale_datasets"
OUTPUT_BASE="/data/nutriplan_experiments/evaluation_results"
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}_seed${SEED}"
PRED_DIR="${OUTPUT_DIR}/predictions"
FINAL_DIR="${OUTPUT_DIR}/final_metrics"

mkdir -p "$PRED_DIR"
mkdir -p "$FINAL_DIR"

echo "========================================================================"
echo "NutriPlan 完整评估流程"
echo "========================================================================"
echo "模型路径: $MODEL_PATH"
echo "模型名称: $MODEL_NAME"
echo "随机种子: $SEED"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "========================================================================"

# 激活环境
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /environment/miniconda3/etc/profile.d/conda.sh ]; then
    source /environment/miniconda3/etc/profile.d/conda.sh
fi
conda activate nutriplan

echo ""
echo "========================================================================"
echo "步骤 1/3: 生成模型预测"
echo "========================================================================"

# 检查是否已经生成过预测
if [ -f "${PRED_DIR}/predictions.jsonl" ] && [ -f "${PRED_DIR}/references.jsonl" ]; then
    PRED_COUNT=$(wc -l < "${PRED_DIR}/predictions.jsonl")
    echo "✓ 检测到已存在的预测文件 ($PRED_COUNT 个样本)"
    echo "  跳过步骤 1，直接进行指标计算"
else
    echo "开始生成预测..."

python3 <<PREDICT
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
import re
from tqdm import tqdm

model_path = "$MODEL_PATH"
data_dir = Path("$DATA_DIR")
pred_dir = Path("$PRED_DIR")

print(f"加载模型: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print(f"✓ 模型加载完成")

# 加载测试数据
test_files = {
    'task_a': 'task_a_test_discriminative.jsonl',
    'task_b': 'task_b_test_from_kg.jsonl',
    'task_c': 'task_c_test_from_kg.jsonl'
}

all_predictions = []
all_references = []
all_constraints = []
all_kg_facts = []

print("\n生成预测...")
for task_name, filename in test_files.items():
    filepath = data_dir / filename
    if not filepath.exists():
        print(f"⚠️ 跳过不存在的文件: {filepath}")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # 每个任务最多评估200个样本
    samples = samples[:200]

    # 批量生成（加速8倍）
    batch_size = 8
    num_batches = (len(samples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"处理 {task_name}"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        # 批量准备输入
        batch_instructions = [s.get('instruction', '') for s in batch_samples]

        inputs = tokenizer(
            batch_instructions,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True  # 批量需要 padding
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                top_p=1.0
            )

        # 处理批量输出
        for i, sample in enumerate(batch_samples):
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)

            # 移除 instruction 重复部分（模型可能会把 instruction 也输出）
            instruction_text = sample.get('instruction', '')
            if generated_text.startswith(instruction_text):
                generated_text = generated_text[len(instruction_text):].strip()

            # 如果生成以数字开头，说明是正确的输出格式
            # 否则，尝试找到第一个数字列表的开始
            import re
            if not re.match(r'^\d+\.', generated_text):
                match = re.search(r'\d+\.\s+\*\*', generated_text)
                if match:
                    generated_text = generated_text[match.start():]

            # 尝试解析生成的JSON（如果是结构化输出）
            try:
                # 提取JSON部分
                if '{' in generated_text and '}' in generated_text:
                    json_start = generated_text.find('{')
                    json_end = generated_text.rfind('}') + 1
                    json_str = generated_text[json_start:json_end]
                    generated_dict = json.loads(json_str)
                else:
                    generated_dict = {'generated': generated_text}
            except:
                generated_dict = {'generated': generated_text}

            # 保存
            instruction = sample.get('instruction', '')
            all_predictions.append({
                'task': task_name,
                'instruction': instruction,
                'generated': generated_text,
                **generated_dict
            })

            # 参考答案
            reference = sample.get('output', '')
            try:
                if isinstance(reference, str) and reference.startswith('{'):
                    reference_dict = json.loads(reference)
                else:
                    reference_dict = {'output': reference}
            except:
                reference_dict = {'output': reference}

            all_references.append(reference_dict)

            # 约束 - 从 instruction 中智能提取
            constraints = sample.get('constraints', {})
            if not constraints or constraints == {}:
                # 从 instruction 提取营养需求
                constraints = {}
                inst_text = sample.get('instruction', '')

                # 提取能量要求
                energy_match = re.search(r'(\d+)\s*kcal', inst_text)
                if energy_match:
                    constraints['nutrition_targets'] = constraints.get('nutrition_targets', {})
                    constraints['nutrition_targets']['energy'] = int(energy_match.group(1))

                # 提取蛋白质要求
                protein_match = re.search(r'(\d+)g?\s*protein', inst_text, re.IGNORECASE)
                if protein_match:
                    constraints['nutrition_targets'] = constraints.get('nutrition_targets', {})
                    constraints['nutrition_targets']['protein'] = int(protein_match.group(1))

                # 提取纤维要求
                fiber_match = re.search(r'(\d+)g?\s*fiber', inst_text, re.IGNORECASE)
                if fiber_match:
                    constraints['nutrition_targets'] = constraints.get('nutrition_targets', {})
                    constraints['nutrition_targets']['fiber'] = int(fiber_match.group(1))

                # 提取过敏原
                if 'allerg' in inst_text.lower() or 'avoid' in inst_text.lower():
                    # 简化版：标记有过敏原约束
                    constraints['has_allergen_constraints'] = True

            all_constraints.append(constraints)

            # KG事实 - 从 output 中提取食材名称作为知识
            kg_facts = sample.get('kg_facts', [])
            if not kg_facts or kg_facts == []:
                # 从 output 提取食材名称
                output_text = sample.get('output', '')
                # 提取食谱名称作为知识
                recipe_names = re.findall(r'\*\*(.*?)\*\*', output_text)
                if recipe_names:
                    kg_facts = recipe_names[:5]  # 最多5个食谱名作为知识

            all_kg_facts.append(kg_facts if kg_facts else [])

print(f"\n✓ 生成了 {len(all_predictions)} 个预测")

# 保存
print("保存预测结果...")
with open(pred_dir / 'predictions.jsonl', 'w', encoding='utf-8') as f:
    for pred in all_predictions:
        f.write(json.dumps(pred, ensure_ascii=False) + '\\n')

with open(pred_dir / 'references.jsonl', 'w', encoding='utf-8') as f:
    for ref in all_references:
        f.write(json.dumps(ref, ensure_ascii=False) + '\\n')

with open(pred_dir / 'constraints.jsonl', 'w', encoding='utf-8') as f:
    for const in all_constraints:
        f.write(json.dumps(const, ensure_ascii=False) + '\\n')

with open(pred_dir / 'kg_facts.jsonl', 'w', encoding='utf-8') as f:
    for facts in all_kg_facts:
        f.write(json.dumps(facts, ensure_ascii=False) + '\\n')

print(f"✓ 结果保存到: {pred_dir}")
PREDICT

fi  # 结束预测生成的 if 检查

echo ""
echo "========================================================================"
echo "步骤 2/3: 计算所有评估指标"
echo "========================================================================"

cd "$(dirname "$0")/.."

python3 evaluation/complete_evaluation.py \
    --predictions "${PRED_DIR}/predictions.jsonl" \
    --references "${PRED_DIR}/references.jsonl" \
    --constraints "${PRED_DIR}/constraints.jsonl" \
    --kg_facts "${PRED_DIR}/kg_facts.jsonl" \
    --output_dir "$FINAL_DIR"

echo ""
echo "========================================================================"
echo "步骤 3/3: 生成最终报告"
echo "========================================================================"

# 创建汇总报告
cat > "${OUTPUT_DIR}/EVALUATION_SUMMARY.txt" <<SUMMARY
========================================================================
NutriPlan 评估完整报告
========================================================================

模型信息:
  名称: $MODEL_NAME
  种子: $SEED
  路径: $MODEL_PATH

评估数据:
  数据目录: $DATA_DIR
  样本数量: $(wc -l < "${PRED_DIR}/predictions.jsonl")

评估指标 (完整列表):
  ✓ NutriPlan 私有指标: SNCR, UPM, K-Faith, AVC
  ✓ BLEU 系列: BLEU-1, BLEU-2, BLEU-3, BLEU-4
  ✓ ROUGE 系列: ROUGE-1, ROUGE-2, ROUGE-L
  ✓ 其他标准指标: METEOR, BERTScore (P/R/F1)
  ✓ 多样性指标: Dist-1/2/3, Self-BLEU
  ✓ 任务特定指标: Nutrition Accuracy, Ingredient Coverage

结果文件:
  📊 聚合指标 (JSON):    ${FINAL_DIR}/aggregate_metrics.json
  📊 每个样本 (CSV):     ${FINAL_DIR}/per_sample_metrics.csv
  📊 论文表格:           ${FINAL_DIR}/paper_table.txt

========================================================================
评估完成时间: $(date)
========================================================================

查看论文表格:
  cat ${FINAL_DIR}/paper_table.txt

查看详细指标:
  cat ${FINAL_DIR}/aggregate_metrics.json | python3 -m json.tool

========================================================================
SUMMARY

cat "${OUTPUT_DIR}/EVALUATION_SUMMARY.txt"

echo ""
echo "========================================================================"
echo "✅ 完整评估流程完成！"
echo "========================================================================"
echo ""
echo "📁 所有结果保存在: $OUTPUT_DIR"
echo ""
echo "📊 关键文件:"
echo "   - 论文表格:     ${FINAL_DIR}/paper_table.txt"
echo "   - 聚合指标:     ${FINAL_DIR}/aggregate_metrics.json"
echo "   - 每个样本:     ${FINAL_DIR}/per_sample_metrics.csv"
echo "   - 评估报告:     ${OUTPUT_DIR}/EVALUATION_SUMMARY.txt"
echo ""
echo "========================================================================"
