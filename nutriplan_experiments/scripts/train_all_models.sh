#!/bin/bash
# 批量训练所有模型（自动跳过已完成的）
# 用法: bash train_all_models.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型列表
MODELS=(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "Qwen/Qwen2-7B"
    "mistralai/Mistral-7B-v0.1"
    "google/gemma-2-9b"
    "microsoft/Phi-3.5-mini-instruct"
)

# 随机种子列表
SEEDS=(42 123 2024)

echo "========================================================================"
echo "NutriPlan 批量训练脚本"
echo "========================================================================"
echo "模型数量: ${#MODELS[@]}"
echo "种子数量: ${#SEEDS[@]}"
echo "总实验数: $((${#MODELS[@]} * ${#SEEDS[@]}))"
echo "========================================================================"
echo ""

SKIPPED=0
TRAINED=0
FAILED=0

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(basename "$MODEL")

    for SEED in "${SEEDS[@]}"; do
        echo "========================================================================"
        echo "模型: $MODEL_SHORT | 种子: $SEED"
        echo "========================================================================"

        # 检查是否已训练
        OUTPUT_DIR="/data/nutriplan_experiments/experiments/rq1_${MODEL_SHORT}_seed${SEED}"
        if [ -d "${OUTPUT_DIR}/best_model" ]; then
            echo "✅ 已完成训练，跳过: ${MODEL_SHORT}_seed${SEED}"
            echo "   模型位置: ${OUTPUT_DIR}/best_model"
            SKIPPED=$((SKIPPED + 1))
            echo ""
            continue
        fi

        # 开始训练
        echo "🚀 开始训练: ${MODEL_SHORT}_seed${SEED}"
        if bash "${SCRIPT_DIR}/train_single_model.sh" "$MODEL" "$SEED"; then
            echo "✅ 训练成功: ${MODEL_SHORT}_seed${SEED}"
            TRAINED=$((TRAINED + 1))
        else
            echo "❌ 训练失败: ${MODEL_SHORT}_seed${SEED}"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done
done

echo "========================================================================"
echo "批量训练完成"
echo "========================================================================"
echo "跳过（已完成）: $SKIPPED"
echo "新训练成功:     $TRAINED"
echo "训练失败:       $FAILED"
echo "========================================================================"
