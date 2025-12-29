#!/bin/bash
# Batch training script for RQ1 (Stage II) - PLAN A
# Trains 5 base LLMs with multiple seeds
# Updated for: TinyLlama + Phi-3 + Qwen2 + Mistral + Gemma-2

# Configuration
DATA_DIR="$HOME/work/recipebench/data/10large_scale_datasets"
EXPERIMENTS_BASE_DIR="experiments"
SEEDS=(42 123 2024)

# List of base LLMs to evaluate (RQ1) - PLAN A
BASE_LLMS=(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "microsoft/Phi-3-mini-4k-instruct"
    "Qwen/Qwen2-7B"
    "mistralai/Mistral-7B-v0.3"
    "google/gemma-2-9b"
)

# Hyperparameters (FP32 + 4090 24GB 优化配置)
LEARNING_RATE=2e-5
BATCH_SIZE=2  # 512 序列长度可以用 batch_size=2
GRADIENT_ACCUM_STEPS=4  # 有效批大小 = 8
NUM_EPOCHS=3
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Task ratios - UPDATED: A=0.3, B=0.5, C=0.2
TASK_A_RATIO=0.3
TASK_B_RATIO=0.5
TASK_C_RATIO=0.2

echo "========================================"
echo "RQ1: Base LLM Selection (Stage II)"
echo "PLAN A: TinyLlama + Phi-3 + Qwen2 + Mistral + Gemma-2"
echo "========================================"
echo "Training ${#BASE_LLMS[@]} models with ${#SEEDS[@]} seeds each"
echo "Total experiments: $((${#BASE_LLMS[@]} * ${#SEEDS[@]}))"
echo ""

# Train each model with each seed
for model in "${BASE_LLMS[@]}"; do
    # Clean model name for directory
    model_clean=$(echo "$model" | tr '/' '_')

    echo ""
    echo "========================================"
    echo "Training Model: $model"
    echo "========================================"

    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Seed: $seed"
        echo "----------------------------------------"

        # Output directory
        OUTPUT_DIR="${EXPERIMENTS_BASE_DIR}/rq1_${model_clean}_seed_${seed}"

        # Check if already trained
        if [ -f "$OUTPUT_DIR/training_complete.txt" ]; then
            echo "✓ Already trained, skipping: $OUTPUT_DIR"
            continue
        fi

        # 为不同大小的模型动态设置 batch size（避免 OOM）
        echo "Configuring batch size for $model..."
        if [[ "$model" == *"TinyLlama"* ]]; then
            CURRENT_BATCH_SIZE=2
            CURRENT_GRAD_ACCUM=4
            echo "  TinyLlama: batch_size=2, grad_accum=4 (effective=8)"
        elif [[ "$model" == *"Phi-3"* ]] || [[ "$model" == *"phi-3"* ]]; then
            CURRENT_BATCH_SIZE=1
            CURRENT_GRAD_ACCUM=8
            echo "  Phi-3: batch_size=1, grad_accum=8 (effective=8)"
        elif [[ "$model" == *"Qwen2"* ]]; then
            CURRENT_BATCH_SIZE=1
            CURRENT_GRAD_ACCUM=8
            echo "  Qwen2-7B: batch_size=1, grad_accum=8 (effective=8)"
        elif [[ "$model" == *"Mistral"* ]]; then
            CURRENT_BATCH_SIZE=1
            CURRENT_GRAD_ACCUM=8
            echo "  Mistral-7B: batch_size=1, grad_accum=8 (effective=8)"
        elif [[ "$model" == *"gemma"* ]]; then
            CURRENT_BATCH_SIZE=1
            CURRENT_GRAD_ACCUM=8
            echo "  Gemma-2-9b: batch_size=1, grad_accum=8 (effective=8)"
        else
            CURRENT_BATCH_SIZE=$BATCH_SIZE
            CURRENT_GRAD_ACCUM=$GRADIENT_ACCUM_STEPS
            echo "  Default: batch_size=$BATCH_SIZE, grad_accum=$GRADIENT_ACCUM_STEPS"
        fi

        # Train NutriPlan
        echo "Starting training..."
        python training/run_nutriplan.py \
            --model_name "$model" \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --learning_rate $LEARNING_RATE \
            --batch_size $CURRENT_BATCH_SIZE \
            --gradient_accumulation_steps $CURRENT_GRAD_ACCUM \
            --num_epochs $NUM_EPOCHS \
            --warmup_ratio $WARMUP_RATIO \
            --weight_decay $WEIGHT_DECAY \
            --max_grad_norm $MAX_GRAD_NORM \
            --task_a_ratio $TASK_A_RATIO \
            --task_b_ratio $TASK_B_RATIO \
            --task_c_ratio $TASK_C_RATIO \
            --seed $seed \
            --logging_steps 50

        # Check if training succeeded
        if [ $? -eq 0 ]; then
            echo "✓ Training completed: $OUTPUT_DIR"

            # Run evaluation
            echo "Starting evaluation..."
            python evaluation/run_evaluation.py \
                --model_path "$OUTPUT_DIR/best_model" \
                --data_dir "$DATA_DIR" \
                --output_dir "$OUTPUT_DIR/eval" \
                --split "test"

            if [ $? -eq 0 ]; then
                echo "✓ Evaluation completed"
                # Mark as complete
                echo "Training and evaluation completed at $(date)" > "$OUTPUT_DIR/training_complete.txt"
            else
                echo "✗ Evaluation failed"
            fi
        else
            echo "✗ Training failed for $model with seed $seed"
        fi

        echo ""
    done
done

echo ""
echo "========================================"
echo "All RQ1 experiments completed!"
echo "========================================"
echo ""
echo "Next step: Aggregate results"
echo ""
echo "Run:"
echo "  python scripts/aggregate_rq1_results.py \\"
echo "      --experiments_dir $EXPERIMENTS_BASE_DIR \\"
echo "      --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/Phi-3-mini-4k-instruct Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \\"
echo "      --seeds 42 123 2024 \\"
echo "      --output_file results/table_x.txt"
echo ""
