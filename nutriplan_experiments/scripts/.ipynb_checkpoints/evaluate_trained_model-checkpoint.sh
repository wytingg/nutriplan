#!/bin/bash
# 端到端评估脚本 - 生成预测 + 计算所有指标
# 用法: bash evaluate_trained_model.sh <model_path> <model_name> <seed>
# 示例: bash evaluate_trained_model.sh /data/nutriplan_experiments/experiments/rq1_Qwen2-7B_seed42/best_model Qwen2-7B 42

set -e

MODEL_PATH=$1
MODEL_NAME=$2
SEED=$3

if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ] || [ -z "$SEED" ]; then
    echo "用法: bash evaluate_trained_model.sh <model_path> <model_name> <seed>"
    echo "示例: bash evaluate_trained_model.sh /data/nutriplan_experiments/experiments/rq1_Qwen2-7B_seed42/best_model Qwen2-7B 42"
    exit 1
fi

# 配置
DATA_DIR="${HOME}/work/recipebench/data/10large_scale_datasets"
OUTPUT_DIR="/data/nutriplan_experiments/evaluation_results/${MODEL_NAME}_seed${SEED}"
PRED_DIR="${OUTPUT_DIR}/predictions"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PRED_DIR"

echo "================================================================"
echo "NutriPlan 模型评估"
echo "================================================================"
echo "模型路径:    $MODEL_PATH"
echo "模型名称:    $MODEL_NAME"
echo "随机种子:    $SEED"
echo "数据目录:    $DATA_DIR"
echo "输出目录:    $OUTPUT_DIR"
echo "================================================================"

# 激活环境
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /environment/miniconda3/envs/nutriplan/lib/python3.11/site-packages/torch/__init__.py ]; then
    source /environment/miniconda3/etc/profile.d/conda.sh
fi

conda activate nutriplan || echo "⚠️ 未找到 nutriplan 环境，使用当前环境"

echo ""
echo "步骤 1/3: 生成模型预测..."
echo "================================================================"

python - <<EOF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm

# 配置
model_path = "$MODEL_PATH"
data_dir = "$DATA_DIR"
output_dir = "$PRED_DIR"

print(f"加载模型: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 加载测试数据
test_files = {
    'task_a': 'task_a_test_discriminative.jsonl',
    'task_b': 'task_b_test_from_kg.jsonl',
    'task_c': 'task_c_test_from_kg.jsonl'
}

predictions = []
references = []
constraints = []

print("生成预测...")
for task_name, filename in test_files.items():
    filepath = Path(data_dir) / filename
    if not filepath.exists():
        print(f"⚠️ 文件不存在: {filepath}")
        continue

    with open(filepath, 'r') as f:
        samples = [json.loads(line) for line in f]

    # 限制样本数量（加快评估）
    samples = samples[:100]  # 每个任务100个样本

    for sample in tqdm(samples, desc=f"Processing {task_name}"):
        instruction = sample.get('instruction', '')

        # 生成预测
        inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 保存预测
        predictions.append({
            'task': task_name,
            'instruction': instruction,
            'generated': generated_text,
            'output': sample.get('output', '')
        })

        references.append(sample.get('output', ''))
        constraints.append(sample.get('constraints', {}))

# 保存结果
pred_file = Path(output_dir) / 'predictions.jsonl'
with open(pred_file, 'w') as f:
    for pred in predictions:
        f.write(json.dumps(pred, ensure_ascii=False) + '\n')

ref_file = Path(output_dir) / 'references.jsonl'
with open(ref_file, 'w') as f:
    for ref in references:
        f.write(json.dumps({'output': ref}, ensure_ascii=False) + '\n')

const_file = Path(output_dir) / 'constraints.jsonl'
with open(const_file, 'w') as f:
    for const in constraints:
        f.write(json.dumps(const, ensure_ascii=False) + '\n')

print(f"✓ 生成了 {len(predictions)} 个预测")
print(f"✓ 保存到: {pred_file}")
EOF

echo ""
echo "步骤 2/3: 计算评估指标..."
echo "================================================================"

# 检查是否有预测文件
if [ ! -f "${PRED_DIR}/predictions.jsonl" ]; then
    echo "✗ 未找到预测文件，跳过评估"
    exit 1
fi

# 计算基础指标（损失、困惑度）
python - <<EOF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from tqdm import tqdm

model_path = "$MODEL_PATH"
pred_file = "$PRED_DIR/predictions.jsonl"
output_file = "$OUTPUT_DIR/basic_metrics.json"

print("加载模型计算损失...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 加载预测
with open(pred_file, 'r') as f:
    predictions = [json.loads(line) for line in f]

total_loss = 0.0
count = 0

for pred in tqdm(predictions[:50], desc="计算损失"):  # 限制50个样本
    text = pred['instruction'] + pred['output']
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        if outputs.loss is not None:
            total_loss += outputs.loss.item()
            count += 1

avg_loss = total_loss / count if count > 0 else 0.0
perplexity = torch.exp(torch.tensor(avg_loss)).item()

results = {
    "model": "$MODEL_NAME",
    "seed": $SEED,
    "test_loss": float(avg_loss),
    "perplexity": float(perplexity),
    "num_samples": count
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ 基础指标:")
print(f"  Test Loss:   {avg_loss:.4f}")
print(f"  Perplexity:  {perplexity:.2f}")
print(f"  保存到: {output_file}")
EOF

echo ""
echo "步骤 3/3: 生成最终报告..."
echo "================================================================"

# 生成汇总报告
cat > "${OUTPUT_DIR}/evaluation_summary.txt" <<SUMMARY
================================================================
NutriPlan 评估报告
================================================================

模型信息:
  名称: $MODEL_NAME
  种子: $SEED
  路径: $MODEL_PATH

评估数据:
  数据目录: $DATA_DIR
  预测文件: ${PRED_DIR}/predictions.jsonl

结果文件:
  基础指标: ${OUTPUT_DIR}/basic_metrics.json
  完整报告: ${OUTPUT_DIR}/evaluation_summary.txt

================================================================
评估完成时间: $(date)
================================================================
SUMMARY

echo ""
echo "================================================================"
echo "✅ 评估完成！"
echo "================================================================"
echo "结果保存在: $OUTPUT_DIR"
echo "  - 基础指标:   ${OUTPUT_DIR}/basic_metrics.json"
echo "  - 预测结果:   ${PRED_DIR}/predictions.jsonl"
echo "  - 评估报告:   ${OUTPUT_DIR}/evaluation_summary.txt"
echo "================================================================"
