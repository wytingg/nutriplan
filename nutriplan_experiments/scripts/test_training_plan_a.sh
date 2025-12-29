#!/bin/bash
# Test Training Script - Plan A
# Runs 1 epoch with TinyLlama to verify environment is working

# Configuration
DATA_DIR="$HOME/work/recipebench/data/10large_scale_datasets"
OUTPUT_DIR="experiments/test_tinyllama_1epoch"
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Hyperparameters (reduced for memory constraints + NaN protection)
LEARNING_RATE=3e-5  # Reduced from 5e-5 to 3e-5 to prevent NaN
BATCH_SIZE=2  # Reduced from 8 to 2 to fit in GPU memory
NUM_EPOCHS=1  # Just 1 epoch for testing
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=0.5  # Reduced from 1.0 to 0.5 for more aggressive gradient clipping

# Task ratios (Generation-focused)
TASK_A_RATIO=0.3  # Discriminative (auxiliary)
TASK_B_RATIO=0.5  # Generation (PRIMARY)
TASK_C_RATIO=0.2  # Editing (auxiliary)

SEED=42

echo "=========================================="
echo "Test Training - Plan A (1 Epoch)"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Task Ratios: A=$TASK_A_RATIO (aux), B=$TASK_B_RATIO (PRIMARY), C=$TASK_C_RATIO (aux)"
echo "Epochs: $NUM_EPOCHS (test mode)"
echo "=========================================="
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Check if model is accessible
echo "Testing model access..."
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True); print('✓ Model accessible')"

if [ $? -ne 0 ]; then
    echo "❌ Error: Cannot access model $MODEL_NAME"
    exit 1
fi

echo ""
echo "Starting test training..."
echo ""

# Run training
python training/run_nutriplan.py \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --num_epochs $NUM_EPOCHS \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --task_a_ratio $TASK_A_RATIO \
    --task_b_ratio $TASK_B_RATIO \
    --task_c_ratio $TASK_C_RATIO \
    --seed $SEED \
    --fp16 \
    --logging_steps 10

# Check training result
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Test Training Completed Successfully!"
    echo "=========================================="
    echo ""
    echo "Output saved to: $OUTPUT_DIR"
    echo ""
    echo "Next step: Run evaluation test"
    echo "Command:"
    echo "  python evaluation/run_evaluation.py \\"
    echo "      --model_path $OUTPUT_DIR/best_model \\"
    echo "      --data_dir $DATA_DIR \\"
    echo "      --output_dir $OUTPUT_DIR/eval \\"
    echo "      --split test"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Test Training Failed"
    echo "=========================================="
    echo "Please check the error messages above"
    exit 1
fi
