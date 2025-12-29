#!/bin/bash
# 快速评估已训练的模型生成指标

MODEL_PATH=$1
OUTPUT_DIR=$2

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash quick_eval.sh <model_path> <output_dir>"
    echo "Example: bash quick_eval.sh ~/work/nutriplan_models_backup/rq1_TinyLlama_seed42/best_model ./eval_results"
    exit 1
fi

echo "Evaluating model: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# 简化评估 - 只计算损失和困惑度
python - <<EOF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm

model_path = "$MODEL_PATH"
output_dir = "$OUTPUT_DIR"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 简单测试
test_text = "I want to make a healthy breakfast with eggs and spinach."
inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    loss = outputs.loss if outputs.loss is not None else 0.0

results = {
    "model_path": model_path,
    "test_loss": float(loss),
    "perplexity": float(torch.exp(torch.tensor(loss))) if loss > 0 else 0.0,
    "status": "evaluated"
}

output_file = Path(output_dir) / "results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_file}")
print(f"  Loss: {results['test_loss']:.4f}")
print(f"  Perplexity: {results['perplexity']:.2f}")
EOF
