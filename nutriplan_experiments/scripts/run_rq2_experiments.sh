#!/bin/bash
# Batch script for RQ2 (Stage III)
# Compares NutriPlan against all baselines

# Configuration
DATA_DIR="D:/Downloads"
EXPERIMENTS_BASE_DIR="experiments"
KG_PATH="work/recipebench/kg/nutriplan_kg4.graphml"

# Best model from RQ1 (update after Stage II)
BEST_BASE_LLM="meta-llama/Llama-3-8B"  # Update based on Table X results
BEST_BASE_LLM_CLEAN=$(echo "$BEST_BASE_LLM" | tr '/' '_')
BEST_SEED=42  # Use first seed for comparisons

# NutriPlan model path
NUTRIPLAN_MODEL_PATH="${EXPERIMENTS_BASE_DIR}/rq1_${BEST_BASE_LLM_CLEAN}_seed_${BEST_SEED}/best_model"

echo "========================================"
echo "RQ2: Overall Performance (Stage III)"
echo "========================================"
echo "Using best base LLM: $BEST_BASE_LLM"
echo "NutriPlan model: $NUTRIPLAN_MODEL_PATH"
echo ""

# ========================================
# Baseline 1: Retrieval (BM25)
# ========================================
echo ""
echo "========================================"
echo "Running Baseline: Retrieval (BM25)"
echo "========================================"

RETRIEVAL_OUTPUT="${EXPERIMENTS_BASE_DIR}/rq2_retrieval"

if [ ! -f "$RETRIEVAL_OUTPUT/eval/aggregate_metrics.json" ]; then
    python baselines/retrieval.py \
        --data_dir "$DATA_DIR" \
        --kg_path "$KG_PATH" \
        --output_dir "$RETRIEVAL_OUTPUT" \
        --top_k 5 \
        --split "test"

    if [ $? -eq 0 ]; then
        echo "[OK] Retrieval baseline completed"
    else
        echo "[ERROR] Retrieval baseline failed"
    fi
else
    echo "[SKIP] Retrieval results already exist"
fi

# ========================================
# Baseline 2: RAG
# ========================================
echo ""
echo "========================================"
echo "Running Baseline: RAG"
echo "========================================"

RAG_OUTPUT="${EXPERIMENTS_BASE_DIR}/rq2_rag"

if [ ! -f "$RAG_OUTPUT/eval/aggregate_metrics.json" ]; then
    python baselines/rag.py \
        --model_name "$BEST_BASE_LLM" \
        --data_dir "$DATA_DIR" \
        --kg_path "$KG_PATH" \
        --output_dir "$RAG_OUTPUT" \
        --retrieval_top_k 5 \
        --split "test"

    if [ $? -eq 0 ]; then
        echo "[OK] RAG baseline completed"
    else
        echo "[ERROR] RAG baseline failed"
    fi
else
    echo "[SKIP] RAG results already exist"
fi

# ========================================
# Baseline 3: SFT (Task B only)
# ========================================
echo ""
echo "========================================"
echo "Running Baseline: SFT (Task B only)"
echo "========================================"

SFT_OUTPUT="${EXPERIMENTS_BASE_DIR}/rq2_sft"

if [ ! -f "$SFT_OUTPUT/training_complete.txt" ]; then
    # Train SFT baseline
    python training/train_sft.py \
        --model_name "$BEST_BASE_LLM" \
        --data_dir "$DATA_DIR" \
        --output_dir "$SFT_OUTPUT" \
        --learning_rate 5e-5 \
        --batch_size 8 \
        --num_epochs 5 \
        --seed 42 \
        --fp16

    if [ $? -eq 0 ]; then
        echo "[OK] SFT training completed"

        # Evaluate
        python evaluation/run_evaluation.py \
            --model_path "$SFT_OUTPUT/best_model" \
            --data_dir "$DATA_DIR" \
            --output_dir "$SFT_OUTPUT/eval" \
            --split "test" \
            --task_b_only

        if [ $? -eq 0 ]; then
            echo "[OK] SFT evaluation completed"
            echo "Completed at $(date)" > "$SFT_OUTPUT/training_complete.txt"
        else
            echo "[ERROR] SFT evaluation failed"
        fi
    else
        echo "[ERROR] SFT training failed"
    fi
else
    echo "[SKIP] SFT results already exist"
fi

# ========================================
# Baseline 4: Zero-shot LLM
# ========================================
echo ""
echo "========================================"
echo "Running Baseline: Zero-shot LLM"
echo "========================================"

ZEROSHOT_OUTPUT="${EXPERIMENTS_BASE_DIR}/rq2_zeroshot"

# Task B
if [ ! -f "$ZEROSHOT_OUTPUT/task_b_predictions.jsonl" ]; then
    python baselines/zero_shot.py \
        --test_file "${DATA_DIR}/task_b_test_from_kg.jsonl" \
        --model_name "$BEST_BASE_LLM" \
        --output_file "$ZEROSHOT_OUTPUT/task_b_predictions.jsonl" \
        --task "b"

    if [ $? -eq 0 ]; then
        echo "[OK] Zero-shot Task B completed"
    else
        echo "[ERROR] Zero-shot Task B failed"
    fi
else
    echo "[SKIP] Zero-shot Task B already exists"
fi

# Task C
if [ ! -f "$ZEROSHOT_OUTPUT/task_c_predictions.jsonl" ]; then
    python baselines/zero_shot.py \
        --test_file "${DATA_DIR}/task_c_test_from_kg.jsonl" \
        --model_name "$BEST_BASE_LLM" \
        --output_file "$ZEROSHOT_OUTPUT/task_c_predictions.jsonl" \
        --task "c"

    if [ $? -eq 0 ]; then
        echo "[OK] Zero-shot Task C completed"
    else
        echo "[ERROR] Zero-shot Task C failed"
    fi
else
    echo "[SKIP] Zero-shot Task C already exists"
fi

# Evaluate zero-shot
if [ ! -f "$ZEROSHOT_OUTPUT/eval/aggregate_metrics.json" ]; then
    python evaluation/run_evaluation.py \
        --predictions_dir "$ZEROSHOT_OUTPUT" \
        --data_dir "$DATA_DIR" \
        --output_dir "$ZEROSHOT_OUTPUT/eval" \
        --split "test"

    if [ $? -eq 0 ]; then
        echo "[OK] Zero-shot evaluation completed"
    else
        echo "[ERROR] Zero-shot evaluation failed"
    fi
fi

# ========================================
# NutriPlan Evaluation (if not done)
# ========================================
echo ""
echo "========================================"
echo "Evaluating NutriPlan"
echo "========================================"

NUTRIPLAN_EVAL_DIR="${EXPERIMENTS_BASE_DIR}/rq1_${BEST_BASE_LLM_CLEAN}_seed_${BEST_SEED}/eval"

if [ ! -f "$NUTRIPLAN_EVAL_DIR/aggregate_metrics.json" ]; then
    python evaluation/run_evaluation.py \
        --model_path "$NUTRIPLAN_MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$NUTRIPLAN_EVAL_DIR" \
        --split "test"

    if [ $? -eq 0 ]; then
        echo "[OK] NutriPlan evaluation completed"
    else
        echo "[ERROR] NutriPlan evaluation failed"
    fi
else
    echo "[SKIP] NutriPlan evaluation already exists"
fi

# ========================================
# Generate Table Y
# ========================================
echo ""
echo "========================================"
echo "Generating Table Y (RQ2 Comparison)"
echo "========================================"

# Create baseline configuration JSON
BASELINE_CONFIG="configs/rq2_baseline_config.json"
cat > "$BASELINE_CONFIG" <<EOF
{
    "NutriPlan": {
        "path": "rq1_${BEST_BASE_LLM_CLEAN}_seed_${BEST_SEED}/eval",
        "category": "main"
    },
    "Retrieval (BM25)": {
        "path": "rq2_retrieval/eval",
        "category": "baseline"
    },
    "RAG": {
        "path": "rq2_rag/eval",
        "category": "baseline"
    },
    "SFT (Task B)": {
        "path": "rq2_sft/eval",
        "category": "baseline"
    },
    "Zero-shot LLM": {
        "path": "rq2_zeroshot/eval",
        "category": "baseline"
    }
}
EOF

echo "[INFO] Created baseline config: $BASELINE_CONFIG"

# Generate Table Y
python scripts/generate_table_y.py \
    --experiments_dir "$EXPERIMENTS_BASE_DIR" \
    --baseline_config "$BASELINE_CONFIG" \
    --output_file "results/table_y.txt" \
    --nutriplan_name "NutriPlan"

if [ $? -eq 0 ]; then
    echo "[OK] Table Y generated successfully"
else
    echo "[ERROR] Table Y generation failed"
fi

echo ""
echo "========================================"
echo "RQ2 experiments completed!"
echo "========================================"
echo ""
echo "Results summary:"
echo "  - Retrieval: $RETRIEVAL_OUTPUT/eval"
echo "  - RAG: $RAG_OUTPUT/eval"
echo "  - SFT: $SFT_OUTPUT/eval"
echo "  - Zero-shot: $ZEROSHOT_OUTPUT/eval"
echo "  - NutriPlan: $NUTRIPLAN_EVAL_DIR"
echo ""
echo "Table Y saved to: results/table_y.txt"
echo ""
