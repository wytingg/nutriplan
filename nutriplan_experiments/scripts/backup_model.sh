#!/bin/bash
# 安全备份训练好的模型
# 用法: bash backup_model.sh <model_dir> <model_name> <seed>

MODEL_DIR=$1
MODEL_NAME=$2
SEED=$3

if [ -z "$MODEL_DIR" ] || [ -z "$MODEL_NAME" ] || [ -z "$SEED" ]; then
    echo "用法: bash backup_model.sh <model_dir> <model_name> <seed>"
    echo "示例: bash backup_model.sh /data/nutriplan_experiments/experiments/rq1_Qwen2-7B_seed42 Qwen2-7B 42"
    exit 1
fi

BACKUP_DIR="${HOME}/work/nutriplan_models_backup"
mkdir -p "$BACKUP_DIR"

echo "================================================================"
echo "备份模型"
echo "================================================================"
echo "源目录: $MODEL_DIR"
echo "模型:   $MODEL_NAME"
echo "种子:   $SEED"
echo "备份到: $BACKUP_DIR"
echo "================================================================"

# 检查源目录
if [ ! -d "$MODEL_DIR/best_model" ]; then
    echo "✗ 错误: 未找到 best_model 目录"
    exit 1
fi

# 压缩（使用更好的压缩方式）
BACKUP_FILE="${BACKUP_DIR}/${MODEL_NAME}_seed${SEED}_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "开始压缩..."
cd "$(dirname "$MODEL_DIR")"
tar -czf "$BACKUP_FILE" "$(basename "$MODEL_DIR")/best_model"

# 验证
echo "验证备份文件..."
tar -tzf "$BACKUP_FILE" | head -10

if [ $? -eq 0 ]; then
    echo "================================================================"
    echo "✅ 备份成功！"
    echo "================================================================"
    echo "备份文件: $BACKUP_FILE"
    echo "文件大小: $(du -h "$BACKUP_FILE" | cut -f1)"
    echo "================================================================"
else
    echo "✗ 备份验证失败！"
    exit 1
fi
