#!/bin/bash
# Batch training script for RQ1 (Stage II)
# Trains multiple base LLMs with multiple seeds

# Configuration
DATA_DIR="D:/Downloads"
EXPERIMENTS_BASE_DIR="experiments"
SEEDS=(42 123 2024)

# List of base LLMs to evaluate (RQ1)
BASE_LLMS=(
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3-8B"
    "Qwen/Qwen2-7B"
    "mistralai/Mistral-7B-v0.3"
    "google/gemma-2-9b"
)

# Hyperparameters (use best config from Stage I.5)
LEARNING_RATE=5e-5
BATCH_SIZE=8
NUM_EPOCHS=5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Task ratios
TASK_A_RATIO=0.5
TASK_B_RATIO=0.3
TASK_C_RATIO=0.2

echo "========================================"
echo "RQ1: Base LLM Selection (Stage II)"
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
            echo "[SKIP] Already trained: $OUTPUT_DIR"
            continue
        fi

        # Train NutriPlan
        python training/run_nutriplan.py \
            --model_name "$model" \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --learning_rate $LEARNING_RATE \
            --batch_size $BATCH_SIZE \
            --num_epochs $NUM_EPOCHS \
            --warmup_ratio $WARMUP_RATIO \
            --weight_decay $WEIGHT_DECAY \
            --max_grad_norm $MAX_GRAD_NORM \
            --task_a_ratio $TASK_A_RATIO \
            --task_b_ratio $TASK_B_RATIO \
            --task_c_ratio $TASK_C_RATIO \
            --seed $seed \
            --fp16 \
            --use_wandb \
            --run_name "rq1_${model_clean}_seed_${seed}" \
            --logging_steps 50

        # Check if training succeeded
        if [ $? -eq 0 ]; then
            echo "[OK] Training completed: $OUTPUT_DIR"

            # Run evaluation
            echo "[INFO] Running evaluation..."
            python evaluation/run_evaluation.py \
                --model_path "$OUTPUT_DIR/best_model" \
                --data_dir "$DATA_DIR" \
                --output_dir "$OUTPUT_DIR/eval" \
                --split "test"

            if [ $? -eq 0 ]; then
                echo "[OK] Evaluation completed"
                # Mark as complete
                echo "Training and evaluation completed at $(date)" > "$OUTPUT_DIR/training_complete.txt"
            else
                echo "[ERROR] Evaluation failed"
            fi
        else
            echo "[ERROR] Training failed for $model with seed $seed"
        fi

        echo ""
    done
done

echo ""
echo "========================================"
echo "All RQ1 experiments completed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Aggregate results:"
echo "   python scripts/aggregate_rq1_results.py \\"
echo "       --experiments_dir $EXPERIMENTS_BASE_DIR \\"
echo "       --models ${BASE_LLMS[@]} \\"
echo "       --seeds ${SEEDS[@]} \\"
echo "       --output_file results/table_x.txt"
echo ""
echo "2. Review Table X to select best base LLM"
echo ""
