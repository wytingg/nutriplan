#!/bin/bash

###############################################################################
# NutriPlan Single Model Training Script (Improved)
# Usage: bash train_single_model.sh <model_name> <seed> [optional_args]
# Example: bash train_single_model.sh "Qwen/Qwen2-7B" 42
###############################################################################

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_name> <seed> [optional_args]"
    echo "Example: $0 'Qwen/Qwen2-7B' 42"
    exit 1
fi

MODEL_NAME="$1"
SEED="$2"
shift 2  # Remove first two arguments, keep remaining as optional args

# Extract model short name for directory naming
MODEL_SHORT=$(basename "$MODEL_NAME")

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# Paths (relative to /data/nutriplan_experiments)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRAINING_SCRIPT="${PROJECT_ROOT}/training/run_nutriplan.py"
DATA_DIR="${HOME}/work/recipebench/data/10large_scale_datasets"
OUTPUT_DIR="/data/nutriplan_experiments/experiments/rq1_${MODEL_SHORT}_seed${SEED}"

echo "================================================================"
echo "NutriPlan Training - Single Model"
echo "================================================================"
echo "Model:        $MODEL_NAME"
echo "Seed:         $SEED"
echo "Output:       $OUTPUT_DIR"
echo "Data:         $DATA_DIR"
echo "Training:     $TRAINING_SCRIPT"
echo "================================================================"

# Verify paths exist
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "❌ Error: Training script not found at $TRAINING_SCRIPT"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found at $DATA_DIR"
    exit 1
fi

# Activate conda environment
echo "🔧 Activating conda environment..."
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /environment/miniconda3/etc/profile.d/conda.sh ]; then
    source /environment/miniconda3/etc/profile.d/conda.sh
else
    echo "❌ Error: Cannot find conda.sh"
    exit 1
fi

conda activate nutriplan || {
    echo "❌ Error: Failed to activate nutriplan environment"
    exit 1
}

# Clear GPU cache
echo "🧹 Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache(); print('✓ GPU cache cleared')"

# Determine if this is a large model (7B+)
IS_LARGE=false
if [[ "$MODEL_NAME" =~ "7B" ]] || [[ "$MODEL_NAME" =~ "9b" ]] || [[ "$MODEL_NAME" =~ "Qwen2" ]] || [[ "$MODEL_NAME" =~ "Mistral" ]] || [[ "$MODEL_NAME" =~ "gemma" ]]; then
    IS_LARGE=true
fi

# Set gradient accumulation based on model size
if [ "$IS_LARGE" = true ]; then
    GRAD_ACCUM=4
    BATCH_SIZE=2
    echo "📊 Large model detected - Using gradient_accumulation=4, batch_size=2"
else
    GRAD_ACCUM=8
    BATCH_SIZE=4
    echo "📊 Small model detected - Using gradient_accumulation=8, batch_size=4"
fi

# Start training
echo "🚀 Starting training..."
echo "================================================================"

python "$TRAINING_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --task_a_ratio 0.3 \
    --task_b_ratio 0.5 \
    --task_c_ratio 0.2 \
    --num_epochs 5 \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --num_workers 0 \
    --patience 3 \
    --logging_steps 10 \
    --seed $SEED \
    "$@"  # Pass any additional arguments

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================================"
    echo "✅ Training completed successfully!"
    echo "================================================================"
    echo "Model saved to: ${OUTPUT_DIR}/best_model"
    echo ""
    echo "📦 Backup recommendation:"
    echo "  cd /data/nutriplan_experiments/experiments"
    echo "  tar -czf ${MODEL_SHORT}_seed${SEED}.tar.gz rq1_${MODEL_SHORT}_seed${SEED}/best_model"
    echo "  cp ${MODEL_SHORT}_seed${SEED}.tar.gz ~/work/nutriplan_models_backup/"
    echo "================================================================"
else
    echo "================================================================"
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "================================================================"
    exit $EXIT_CODE
fi
