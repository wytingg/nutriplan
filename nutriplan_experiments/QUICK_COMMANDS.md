# Quick Command Reference - NutriPlan Experiments

Copy-paste commands for running the complete experimental pipeline.

---

## Environment Setup (One-time)

```bash
# Create environment
conda create -n nutriplan python=3.10 -y
conda activate nutriplan

# Install dependencies
pip install torch transformers datasets pandas numpy scipy scikit-learn nltk rouge-score tqdm pyyaml

# Install graph-tool
conda install -c conda-forge graph-tool -y

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

---

## Stage I.5: Hyperparameter Search

```bash
# Random search (recommended - faster)
python scripts/hyperparameter_search.py \
    --model_name meta-llama/Llama-3.2-3B \
    --search_space configs/search_space.yaml \
    --data_dir D:/Downloads \
    --output_dir experiments/hyperparam_search \
    --search_type random

# Grid search (comprehensive but slower)
python scripts/hyperparameter_search.py \
    --model_name meta-llama/Llama-3.2-3B \
    --search_space configs/search_space.yaml \
    --data_dir D:/Downloads \
    --output_dir experiments/hyperparam_search \
    --search_type grid
```

**Output:** `experiments/hyperparam_search/best_config.json`

---

## Stage II: Base LLM Selection (RQ1)

### Train All Models (Batch)

```bash
# Make script executable
chmod +x scripts/train_all_llms.sh

# Run batch training (5 models × 3 seeds = 15 experiments)
bash scripts/train_all_llms.sh
```

### OR Train Individual Model (Manual)

```bash
# Example: Train Llama-3-8B with seed 42
python training/run_nutriplan.py \
    --model_name meta-llama/Llama-3-8B \
    --data_dir D:/Downloads \
    --output_dir experiments/rq1_meta-llama_Llama-3-8B_seed_42 \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --num_epochs 5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --task_a_ratio 0.5 \
    --task_b_ratio 0.3 \
    --task_c_ratio 0.2 \
    --seed 42 \
    --fp16 \
    --use_wandb \
    --run_name rq1_Llama-3-8B_seed_42 \
    --logging_steps 50
```

### Aggregate RQ1 Results (Generate Table X)

```bash
python scripts/aggregate_rq1_results.py \
    --experiments_dir experiments \
    --models meta-llama/Llama-3.2-3B meta-llama/Llama-3-8B Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \
    --seeds 42 123 2024 \
    --output_file results/table_x.txt
```

**Output:** `results/table_x.txt`, `.csv`, `.tex`

---

## Stage III: Overall Performance (RQ2)

### Run All Baselines (Batch)

```bash
# 1. First update BEST_BASE_LLM in run_rq2_experiments.sh based on Table X
nano scripts/run_rq2_experiments.sh  # Edit line 11

# 2. Make script executable
chmod +x scripts/run_rq2_experiments.sh

# 3. Run all RQ2 experiments
bash scripts/run_rq2_experiments.sh
```

### OR Run Individual Baselines (Manual)

#### Baseline 1: Retrieval (BM25)

```bash
python baselines/retrieval.py \
    --data_dir D:/Downloads \
    --kg_path work/recipebench/kg/nutriplan_kg4.graphml \
    --output_dir experiments/rq2_retrieval \
    --top_k 5 \
    --split test
```

#### Baseline 2: RAG

```bash
python baselines/rag.py \
    --model_name meta-llama/Llama-3-8B \
    --data_dir D:/Downloads \
    --kg_path work/recipebench/kg/nutriplan_kg4.graphml \
    --output_dir experiments/rq2_rag \
    --retrieval_top_k 5 \
    --split test
```

#### Baseline 3: SFT (Task B only)

```bash
# Train
python training/train_sft.py \
    --model_name meta-llama/Llama-3-8B \
    --data_dir D:/Downloads \
    --output_dir experiments/rq2_sft \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --num_epochs 5 \
    --seed 42 \
    --fp16

# Evaluate
python evaluation/run_evaluation.py \
    --model_path experiments/rq2_sft/best_model \
    --data_dir D:/Downloads \
    --output_dir experiments/rq2_sft/eval \
    --split test \
    --task_b_only
```

#### Baseline 4: Zero-shot LLM

```bash
# Task B
python baselines/zero_shot.py \
    --test_file D:/Downloads/task_b_test_from_kg.jsonl \
    --model_name meta-llama/Llama-3-8B \
    --output_file experiments/rq2_zeroshot/task_b_predictions.jsonl \
    --task b

# Task C
python baselines/zero_shot.py \
    --test_file D:/Downloads/task_c_test_from_kg.jsonl \
    --model_name meta-llama/Llama-3-8B \
    --output_file experiments/rq2_zeroshot/task_c_predictions.jsonl \
    --task c

# Evaluate
python evaluation/run_evaluation.py \
    --predictions_dir experiments/rq2_zeroshot \
    --data_dir D:/Downloads \
    --output_dir experiments/rq2_zeroshot/eval \
    --split test
```

### Generate Table Y (RQ2 Comparison)

```bash
# Create baseline config (if not auto-generated)
cat > configs/rq2_baseline_config.json <<'EOF'
{
    "NutriPlan": {
        "path": "rq1_meta-llama_Llama-3-8B_seed_42/eval",
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

# Generate Table Y
python scripts/generate_table_y.py \
    --experiments_dir experiments \
    --baseline_config configs/rq2_baseline_config.json \
    --output_file results/table_y.txt \
    --nutriplan_name NutriPlan
```

**Output:** `results/table_y.txt`, `.csv`, `.tex`

---

## Monitoring and Debugging

### Check Training Progress

```bash
# List completed experiments
find experiments -name "training_complete.txt"

# Count completed experiments
find experiments -name "training_complete.txt" | wc -l

# Monitor training log
tail -f experiments/rq1_meta-llama_Llama-3-8B_seed_42/logs/train.log

# Check GPU usage
nvidia-smi

# Check GPU usage continuously
watch -n 1 nvidia-smi
```

### View Results

```bash
# View Table X (RQ1)
cat results/table_x.txt

# View Table Y (RQ2)
cat results/table_y.txt

# View specific model metrics
cat experiments/rq1_meta-llama_Llama-3-8B_seed_42/eval/aggregate_metrics.json | python -m json.tool
```

### Test KG Loading

```bash
# Test KG loading
python -c "from utils.kg_utils import NutriPlanKG; kg = NutriPlanKG(); print('KG loaded:', kg.num_nodes, 'nodes')"

# Test statistical functions
python -c "from utils.statistical_tests import SignificanceTests; print('Statistics module loaded successfully')"
```

### Verify Data

```bash
# Check data files
ls -lh D:/Downloads/task_*.jsonl

# Count samples in each file
wc -l D:/Downloads/task_a_train_discriminative.jsonl
wc -l D:/Downloads/task_b_train_from_kg.jsonl
wc -l D:/Downloads/task_c_train_from_kg.jsonl

# View sample data
head -n 1 D:/Downloads/task_a_train_discriminative.jsonl | python -m json.tool
```

---

## Complete Pipeline (All Stages)

```bash
# Stage I.5: Hyperparameter Search (~4-8 hours)
python scripts/hyperparameter_search.py \
    --model_name meta-llama/Llama-3.2-3B \
    --search_space configs/search_space.yaml \
    --data_dir D:/Downloads \
    --output_dir experiments/hyperparam_search \
    --search_type random

# Review best config
cat experiments/hyperparam_search/best_config.json

# Stage II: Train all models (~3-7 days)
bash scripts/train_all_llms.sh

# Aggregate RQ1 results (~1-2 minutes)
python scripts/aggregate_rq1_results.py \
    --experiments_dir experiments \
    --models meta-llama/Llama-3.2-3B meta-llama/Llama-3-8B Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \
    --seeds 42 123 2024 \
    --output_file results/table_x.txt

# Review Table X
cat results/table_x.txt

# Update BEST_BASE_LLM in run_rq2_experiments.sh
nano scripts/run_rq2_experiments.sh

# Stage III: Run all baselines (~1-2 days)
bash scripts/run_rq2_experiments.sh

# Review Table Y
cat results/table_y.txt
```

---

## Troubleshooting Commands

### GPU Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Reduce batch size (edit train_all_llms.sh line 17)
BATCH_SIZE=4  # Instead of 8

# Or use smaller model
--model_name meta-llama/Llama-3.2-3B
```

### Installation Issues

```bash
# Reinstall graph-tool
conda install -c conda-forge graph-tool -y

# Install from pip (alternative)
pip install graph-tool

# Check Python version
python --version  # Should be 3.10

# Check CUDA version
nvcc --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Permission Issues

```bash
# Make all scripts executable
chmod +x scripts/*.sh

# Or run with bash explicitly
bash scripts/train_all_llms.sh
```

### WandB Issues

```bash
# Login to WandB
wandb login

# Or disable WandB (remove --use_wandb from scripts)
# Edit train_all_llms.sh and remove line 48: --use_wandb
```

---

## Quick File Transfer to Server

```bash
# From local machine, transfer entire directory to server
scp -r C:/Users/wyt03/nutriplan_kg/nutriplan_experiments/ username@server:/path/to/destination/

# Or using rsync (better for large transfers)
rsync -avz -e ssh C:/Users/wyt03/nutriplan_kg/nutriplan_experiments/ username@server:/path/to/destination/

# Verify transfer
ssh username@server "ls -la /path/to/destination/nutriplan_experiments/"
```

---

## Time Estimates

| Stage | Command | Time | GPU |
|-------|---------|------|-----|
| I.5 | `hyperparameter_search.py` | 4-8h | 16GB+ |
| II | `train_all_llms.sh` | 3-7d | 24GB+ |
| II | `aggregate_rq1_results.py` | 1-2m | No |
| III | `run_rq2_experiments.sh` | 1-2d | 24GB+ |
| **Total** | **Full Pipeline** | **~1-2 weeks** | **24GB+** |

---

## Expected Output Summary

```
results/
├── table_x.txt      # RQ1: Best base LLM
├── table_x.csv
├── table_x.tex
├── table_y.txt      # RQ2: Overall comparison
├── table_y.csv
└── table_y.tex

experiments/
├── hyperparam_search/best_config.json
├── rq1_{model}_seed_{seed}/best_model/  (15 models)
├── rq2_retrieval/eval/
├── rq2_rag/eval/
├── rq2_sft/eval/
└── rq2_zeroshot/eval/
```

---

**Quick Reference Card**
- For detailed instructions: See `EXECUTION_GUIDE.md`
- For file descriptions: See `COMPLETE_FILE_LIST.md`
- For project overview: See `README.md`

**Ready to transfer to server and run!**
