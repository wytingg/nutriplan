#!/bin/bash
# 快速评估脚本 - 计算核心指标
# 用法: bash quick_evaluate.sh <model_path> <model_name> <seed>

set -e

MODEL_PATH=$1
MODEL_NAME=$2
SEED=$3

if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ] || [ -z "$SEED" ]; then
    echo "用法: bash quick_evaluate.sh <model_path> <model_name> <seed>"
    echo "示例: bash quick_evaluate.sh /data/nutriplan_experiments/experiments/rq1_Qwen2-7B_seed42/best_model Qwen2-7B 42"
    exit 1
fi

echo "========================================================================"
echo "NutriPlan 快速评估"
echo "========================================================================"
echo "模型: $MODEL_PATH"
echo "名称: $MODEL_NAME"
echo "种子: $SEED"
echo "========================================================================"

# 激活环境
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /environment/miniconda3/etc/profile.d/conda.sh ]; then
    source /environment/miniconda3/etc/profile.d/conda.sh
fi

conda activate nutriplan 2>/dev/null || echo "Using current environment"

# 运行评估
python3 - <<EVALCODE
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm

model_path = "$MODEL_PATH"
model_name = "$MODEL_NAME"
seed = $SEED
data_dir = Path("~/work/recipebench/data/10large_scale_datasets").expanduser()
output_dir = Path(f"/data/nutriplan_experiments/evaluation_results/{model_name}_seed{seed}")
output_dir.mkdir(parents=True, exist_ok=True)

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
print(f"模型已加载到: {next(model.parameters()).device}")

# 加载测试数据
test_file = data_dir / "task_b_test_from_kg.jsonl"
if not test_file.exists():
    print(f"错误: 测试文件不存在 {test_file}")
    exit(1)

samples = []
with open(test_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        if line.strip():
            samples.append(json.loads(line))

print(f"加载了 {len(samples)} 个测试样本")

# 评估
total_loss = 0.0
count = 0

print("计算测试损失...")
for sample in tqdm(samples):
    instruction = sample.get('instruction', '')
    output = sample.get('output', '')
    text = instruction + ' ' + output

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
    "model": model_name,
    "seed": seed,
    "model_path": model_path,
    "test_loss": float(avg_loss),
    "perplexity": float(perplexity),
    "num_samples": count
}

print("")
print("=" * 70)
print(f"{model_name} 评估结果")
print("=" * 70)
print(f"Test Loss:   {avg_loss:.4f}")
print(f"Perplexity:  {perplexity:.2f}")
print(f"Samples:     {count}")
print("=" * 70)

# 保存结果
output_file = output_dir / 'evaluation_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"")
print(f"结果已保存: {output_file}")
EVALCODE

echo ""
echo "========================================================================"
echo "评估完成！"
echo "========================================================================"
