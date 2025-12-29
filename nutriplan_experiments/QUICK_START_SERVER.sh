#!/bin/bash
# NutriPlan 实验快速启动脚本 - 服务器版
# 用法：bash QUICK_START_SERVER.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "NutriPlan 实验快速启动检查"
echo "=========================================="
echo ""

# 检查当前目录
echo "[1/6] 检查当前目录..."
CURRENT_DIR=$(pwd)
echo "✓ 当前目录: $CURRENT_DIR"
echo ""

# 检查数据文件
echo "[2/6] 检查数据文件..."
DATA_DIR="data"
REQUIRED_FILES=(
    "task_a_train_discriminative.jsonl"
    "task_a_val_discriminative.jsonl"
    "task_a_test_discriminative.jsonl"
    "task_b_train_from_kg.jsonl"
    "task_b_val_from_kg.jsonl"
    "task_b_test_from_kg.jsonl"
    "task_c_train_from_kg.jsonl"
    "task_c_val_from_kg.jsonl"
    "task_c_test_from_kg.jsonl"
)

MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATA_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (缺失)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "⚠️  警告: 缺少 $MISSING_FILES 个数据文件"
    echo "请确保所有数据文件都在 $DATA_DIR/ 目录中"
    echo ""
else
    echo "✓ 所有数据文件完整"
    echo ""
fi

# 检查 Python 环境
echo "[3/6] 检查 Python 环境..."
python --version
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA 可用"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠️  未检测到 CUDA"
fi
echo ""

# 检查必要的 Python 包
echo "[4/6] 检查 Python 依赖..."
python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import transformers; print('✓ Transformers:', transformers.__version__)"
python -c "import pandas; print('✓ Pandas:', pandas.__version__)"
echo ""

# 检查脚本权限
echo "[5/6] 设置脚本执行权限..."
chmod +x scripts/*.sh
echo "✓ 脚本权限已设置"
echo ""

# 创建必要的目录
echo "[6/6] 创建输出目录..."
mkdir -p experiments
mkdir -p results
mkdir -p logs
echo "✓ 输出目录已创建"
echo ""

echo "=========================================="
echo "✅ 环境检查完成！"
echo "=========================================="
echo ""
echo "接下来的步骤："
echo ""
echo "📌 阶段 II：训练 15 个基础模型实验"
echo "   运行命令："
echo "   bash scripts/train_all_llms_PLAN_A.sh"
echo ""
echo "   预计时间：3-7 天"
echo "   GPU 需求：24GB+ VRAM"
echo ""
echo "📌 监控训练进度："
echo "   find experiments -name 'training_complete.txt' | wc -l"
echo ""
echo "📌 训练完成后，聚合结果："
echo "   python scripts/aggregate_rq1_results.py \\"
echo "       --experiments_dir experiments \\"
echo "       --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/Phi-3-mini-4k-instruct Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \\"
echo "       --seeds 42 123 2024 \\"
echo "       --output_file results/table_x.txt"
echo ""
echo "详细说明请查看: SERVER_EXECUTION_GUIDE.md"
echo "=========================================="
