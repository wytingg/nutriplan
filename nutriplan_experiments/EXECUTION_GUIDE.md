# NutriPlan Experiments Execution Guide

Complete step-by-step guide for running Stages I-III of NutriPlan experiments.

## Prerequisites

### Environment Setup
```bash
# Create conda environment
conda create -n nutriplan python=3.10
conda activate nutriplan

# Install dependencies
pip install torch transformers datasets
pip install pandas numpy scipy scikit-learn
pip install graph-tool  # For KG access
pip install nltk rouge-score
pip install tensorboard wandb  # Optional: for logging
pip install tqdm pyyaml

# Download NLTK data (for BLEU)
python -c "import nltk; nltk.download('punkt')"
```

### Data Preparation
Ensure your data is at: `D:/Downloads/`
- `task_a_train_discriminative.jsonl`, `task_a_val_discriminative.jsonl`, `task_a_test_discriminative.jsonl`
- `task_b_train_from_kg.jsonl`, `task_b_val_from_kg.jsonl`, `task_b_test_from_kg.jsonl`
- `task_c_train_from_kg.jsonl`, `task_c_val_from_kg.jsonl`, `task_c_test_from_kg.jsonl`

### Knowledge Graph
Ensure KG is available at: `work/recipebench/kg/nutriplan_kg4.graphml`

---

## Stage I: Infrastructure & Baseline Setup

### Step 1: Verify Data Statistics
```bash
python data/data_statistics_version1.py \
    --data_dir D:/Downloads \
    --output_dir results/data_stats
```

**Expected Output:**
- Total samples: ~38,160 (Train: 27,992, Val: 5,593, Test: 5,575)
- Task A: 14,000 samples
- Task B: 14,000 samples
- Task C: 11,160 samples
- Statistics saved to: `results/data_stats/`

**Time Estimate:** 2-3 minutes

---

## Stage I.5: Hyperparameter Search

### Step 2: Run Hyperparameter Search

**Option A: Grid Search (Comprehensive but slow)**
```bash
python scripts/hyperparameter_search.py \
    --model_name meta-llama/Llama-3.2-3B \
    --search_space configs/search_space.yaml \
    --data_dir D:/Downloads \
    --output_dir experiments/hyperparam_search \
    --search_type grid
```

**Option B: Random Search (Recommended - faster)**
```bash
python scripts/hyperparameter_search.py \
    --model_name meta-llama/Llama-3.2-3B \
    --search_space configs/search_space.yaml \
    --data_dir D:/Downloads \
    --output_dir experiments/hyperparam_search \
    --search_type random
```

**Expected Output:**
- Best configuration saved to: `experiments/hyperparam_search/best_config.json`
- All trial results: `experiments/hyperparam_search/search_results.json`

**Time Estimate:**
- Grid search: 12-24 hours (depends on search space size)
- Random search: 4-8 hours (20 trials with 3 epochs each)

**Next Step:** Review `best_config.json` and update hyperparameters in `train_all_llms.sh` if needed.

---

## Stage II: Base LLM Selection (RQ1)

### Step 3: Train Multiple Base LLMs with Multiple Seeds

**Method 1: Using Batch Script (Recommended)**
```bash
# Make script executable (Linux/Mac)
chmod +x scripts/train_all_llms.sh

# Run batch training
bash scripts/train_all_llms.sh
```

**Method 2: Manual Training (Individual models)**

For each model and seed combination:
```bash
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

**Models to Train (5 models × 3 seeds = 15 experiments):**
1. `meta-llama/Llama-3.2-3B`
2. `meta-llama/Llama-3-8B`
3. `Qwen/Qwen2-7B`
4. `mistralai/Mistral-7B-v0.3`
5. `google/gemma-2-9b`

**Seeds:** 42, 123, 2024

**Expected Output per Experiment:**
- Best model: `experiments/rq1_{model}_seed_{seed}/best_model/`
- Training logs: `experiments/rq1_{model}_seed_{seed}/logs/`
- Checkpoints: `experiments/rq1_{model}_seed_{seed}/checkpoints/`
- Evaluation results: `experiments/rq1_{model}_seed_{seed}/eval/aggregate_metrics.json`

**Time Estimate per Model:** 6-12 hours (depends on GPU)
**Total Time for Stage II:** 3-7 days (if run sequentially)

**GPU Memory Requirements:**
- 3B models: ~16GB VRAM
- 7-9B models: ~24GB VRAM (use gradient checkpointing if needed)

---

### Step 4: Aggregate RQ1 Results and Generate Table X

After all 15 experiments complete:

```bash
python scripts/aggregate_rq1_results.py \
    --experiments_dir experiments \
    --models meta-llama/Llama-3.2-3B meta-llama/Llama-3-8B Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \
    --seeds 42 123 2024 \
    --output_file results/table_x.txt
```

**Expected Output:**
- Text table: `results/table_x.txt`
- CSV: `results/table_x.csv`
- LaTeX: `results/table_x.tex`
- Console output showing best model by SNCR

**Table X Contents:**
- Mean ± Std for each metric across 3 seeds
- Bold values for best performance
- Statistical significance markers

**Time Estimate:** 1-2 minutes

**Next Step:**
1. Review Table X
2. Identify best base LLM
3. Update `BEST_BASE_LLM` variable in `run_rq2_experiments.sh`

---

## Stage III: Overall Performance Comparison (RQ2)

### Step 5: Run All Baseline Comparisons

**Method 1: Using Batch Script (Recommended)**
```bash
# Update BEST_BASE_LLM in run_rq2_experiments.sh first!
# Edit line 11 based on Table X results

# Make script executable
chmod +x scripts/run_rq2_experiments.sh

# Run all RQ2 experiments
bash scripts/run_rq2_experiments.sh
```

**Method 2: Manual Baseline Execution**

**Baseline 1: Retrieval (BM25)**
```bash
python baselines/retrieval.py \
    --data_dir D:/Downloads \
    --kg_path work/recipebench/kg/nutriplan_kg4.graphml \
    --output_dir experiments/rq2_retrieval \
    --top_k 5 \
    --split test
```

**Baseline 2: RAG**
```bash
python baselines/rag.py \
    --model_name meta-llama/Llama-3-8B \
    --data_dir D:/Downloads \
    --kg_path work/recipebench/kg/nutriplan_kg4.graphml \
    --output_dir experiments/rq2_rag \
    --retrieval_top_k 5 \
    --split test
```

**Baseline 3: SFT (Task B only)**
```bash
# Train SFT
python training/train_sft.py \
    --model_name meta-llama/Llama-3-8B \
    --data_dir D:/Downloads \
    --output_dir experiments/rq2_sft \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --num_epochs 5 \
    --seed 42 \
    --fp16

# Evaluate SFT
python evaluation/run_evaluation.py \
    --model_path experiments/rq2_sft/best_model \
    --data_dir D:/Downloads \
    --output_dir experiments/rq2_sft/eval \
    --split test \
    --task_b_only
```

**Baseline 4: Zero-shot LLM**
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

**Time Estimates:**
- Retrieval: 30-60 minutes
- RAG: 2-4 hours
- SFT training: 4-8 hours
- Zero-shot: 3-6 hours
- **Total:** ~1-2 days

---

### Step 6: Generate Table Y (RQ2 Comparison)

```bash
python scripts/generate_table_y.py \
    --experiments_dir experiments \
    --baseline_config configs/rq2_baseline_config.json \
    --output_file results/table_y.txt \
    --nutriplan_name NutriPlan
```

**Note:** The baseline config JSON is auto-generated by `run_rq2_experiments.sh`, or create manually:

```json
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
```

**Expected Output:**
- Text table: `results/table_y.txt`
- CSV: `results/table_y.csv`
- LaTeX: `results/table_y.tex`
- Full metric comparison across all baselines

**Time Estimate:** 1-2 minutes

---

## Complete Execution Timeline

### Quick Reference: Full Pipeline

```bash
# Stage I.5: Hyperparameter Search (4-8 hours)
python scripts/hyperparameter_search.py \
    --model_name meta-llama/Llama-3.2-3B \
    --search_space configs/search_space.yaml \
    --data_dir D:/Downloads \
    --output_dir experiments/hyperparam_search \
    --search_type random

# Stage II: Train all models (3-7 days)
bash scripts/train_all_llms.sh

# Stage II: Aggregate RQ1 results (1-2 minutes)
python scripts/aggregate_rq1_results.py \
    --experiments_dir experiments \
    --models meta-llama/Llama-3.2-3B meta-llama/Llama-3-8B Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \
    --seeds 42 123 2024 \
    --output_file results/table_x.txt

# Stage III: Run all baselines (1-2 days)
# First: Update BEST_BASE_LLM in run_rq2_experiments.sh
bash scripts/run_rq2_experiments.sh

# Stage III: Generate RQ2 comparison (1-2 minutes)
python scripts/generate_table_y.py \
    --experiments_dir experiments \
    --baseline_config configs/rq2_baseline_config.json \
    --output_file results/table_y.txt \
    --nutriplan_name NutriPlan
```

**Total Time:** ~1-2 weeks (mostly GPU training time)

---

## Output Files Summary

### Stage I.5 Outputs
- `experiments/hyperparam_search/best_config.json` - Best hyperparameters
- `experiments/hyperparam_search/search_results.json` - All trial results

### Stage II (RQ1) Outputs
- `experiments/rq1_{model}_seed_{seed}/best_model/` - Trained models (15 total)
- `experiments/rq1_{model}_seed_{seed}/eval/aggregate_metrics.json` - Evaluation results
- `results/table_x.txt` / `.csv` / `.tex` - RQ1 comparison table

### Stage III (RQ2) Outputs
- `experiments/rq2_retrieval/eval/` - Retrieval baseline results
- `experiments/rq2_rag/eval/` - RAG baseline results
- `experiments/rq2_sft/eval/` - SFT baseline results
- `experiments/rq2_zeroshot/eval/` - Zero-shot baseline results
- `results/table_y.txt` / `.csv` / `.tex` - RQ2 comparison table

---

## Troubleshooting

### GPU Out of Memory
```bash
# Option 1: Reduce batch size
--batch_size 4

# Option 2: Enable gradient checkpointing (add to training scripts)
--gradient_checkpointing

# Option 3: Use smaller model first
--model_name meta-llama/Llama-3.2-3B
```

### KG Loading Issues
```bash
# Verify KG file exists
ls work/recipebench/kg/nutriplan_kg4.graphml

# Test KG loading
python -c "from utils.kg_utils import NutriPlanKG; kg = NutriPlanKG(); print('KG loaded successfully')"
```

### Missing Dependencies
```bash
# graph-tool installation (Ubuntu/Debian)
sudo apt-get install python3-graph-tool

# graph-tool installation (conda - recommended)
conda install -c conda-forge graph-tool

# If graph-tool fails, use NetworkX alternative (slower)
pip install networkx
```

### WandB Authentication
```bash
# Login to WandB (if using --use_wandb)
wandb login

# Or disable WandB
# Remove --use_wandb flag from training commands
```

### Data Loading Errors
```bash
# Verify data format
python -c "
import json
with open('D:/Downloads/task_a_train_discriminative.jsonl', 'r') as f:
    sample = json.loads(f.readline())
    print('Sample keys:', sample.keys())
"
```

---

## Performance Benchmarks

Expected performance ranges (based on validation set):

### Core Metrics (Primary)
- **SNCR (Strict Nutrition Constraint Recall)**: 0.60-0.85
  - Higher is better
  - Measures exact constraint satisfaction

- **UPM (User Preference Matching)**: 0.55-0.80
  - Higher is better
  - Measures dietary/allergy compliance

- **K-Faith (KG Faithfulness)**: 0.50-0.75
  - Higher is better
  - Measures KG alignment

- **AVC (Average Violation Count)**: 0.5-2.0
  - Lower is better
  - Number of constraint violations

### Generation Quality Metrics (Secondary)
- **Dist-2**: 0.40-0.70 (higher = more diverse)
- **BLEU**: 0.15-0.35
- **ROUGE-L**: 0.25-0.45
- **Nutrition Accuracy**: 0.60-0.85

---

## Next Steps After Stage III

1. **Analyze Results**
   - Review Table X for best base LLM (RQ1)
   - Review Table Y for overall comparison (RQ2)
   - Identify failure cases

2. **Ablation Studies (Stage IV)**
   - Task-specific performance
   - Component ablations
   - KG impact analysis

3. **Generalization Testing (Stage V)**
   - Cross-domain evaluation
   - Zero-shot capability
   - Few-shot adaptation

4. **Human Evaluation (Stage VI)**
   - Recipe quality assessment
   - Constraint satisfaction verification
   - User preference validation

---

## Citation

If you use this code, please cite:

```bibtex
@article{nutriplan2025,
  title={NutriPlan: A Knowledge Graph-Enhanced Benchmark and Reflective Framework for Individualized Nutrition-Constrained Recipe Generation},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2025}
}
```

---

## Contact

For issues or questions:
- Check troubleshooting section above
- Review code comments in source files
- Consult paper for methodology details

**Last Updated:** 2025-01-XX
