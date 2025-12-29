#!/bin/bash
# Server 3: Train Qwen2-7B with 3 seeds
# 预计时间: ~60 hours (7B model)

# Configuration
DATA_DIR="$HOME/work/recipebench/data/10large_scale_datasets"
EXPERIMENTS_BASE_DIR="experiments"
SEEDS=(42 123 2024)

# Model for Server 3
MODEL="mistralai/Mistral-7B-v0.3"

# Hyperparameters
LEARNING_RATE=5e-5
BATCH_SIZE=2
GRADIENT_ACCUM_STEPS=4
NUM_EPOCHS=5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Task ratios (Generation-focused)
TASK_A_RATIO=0.3
TASK_B_RATIO=0.5
TASK_C_RATIO=0.2

echo "========================================"
echo "SERVER 3: Training Qwen2-7B"
echo "========================================"
echo "Model: $MODEL"
echo "Seeds: ${SEEDS[@]}"
echo "Total experiments: ${#SEEDS[@]}"
echo ""

model_clean=$(echo "$MODEL" | tr '/' '_')

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Seed: $seed"
    echo "----------------------------------------"

    OUTPUT_DIR="${EXPERIMENTS_BASE_DIR}/rq1_${model_clean}_seed_${seed}"

    # Check if already trained
    if [ -f "$OUTPUT_DIR/training_complete.txt" ]; then
        echo "✓ Already trained, skipping: $OUTPUT_DIR"
        continue
    fi

    # Train
    echo "Starting training..."
    python training/run_nutriplan.py \
        --model_name "$MODEL" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
        --num_epochs $NUM_EPOCHS \
        --warmup_ratio $WARMUP_RATIO \
        --weight_decay $WEIGHT_DECAY \
        --max_grad_norm $MAX_GRAD_NORM \
        --task_a_ratio $TASK_A_RATIO \
        --task_b_ratio $TASK_B_RATIO \
        --task_c_ratio $TASK_C_RATIO \
        --seed $seed \
        --fp16 \
        --logging_steps 50

    if [ $? -eq 0 ]; then
        echo "✓ Training completed: $OUTPUT_DIR"

        # Evaluate
        echo "Starting evaluation..."
        python evaluation/run_evaluation.py \
            --model_path "$OUTPUT_DIR/best_model" \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR/eval" \
            --split "test"

        if [ $? -eq 0 ]; then
            echo "✓ Evaluation completed"
            echo "Training and evaluation completed at $(date)" > "$OUTPUT_DIR/training_complete.txt"
        else
            echo "✗ Evaluation failed"
        fi
    else
        echo "✗ Training failed"
    fi
done

echo ""
echo "========================================"
echo "SERVER 3 COMPLETED!"
echo "========================================"
echo "Qwen2-7B training finished at $(date)"
