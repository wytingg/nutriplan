# NutriPlan Experiments

Complete experimental infrastructure for the NutriPlan paper: "NutriPlan: A Knowledge Graph-Enhanced Benchmark and Reflective Framework for Individualized Nutrition-Constrained Recipe Generation"

## Project Structure

```
nutriplan_experiments/
├── configs/                           # Configuration files
│   ├── search_space.yaml             # Hyperparameter search space (NEW)
│   ├── rq2_baseline_config_template.json  # Baseline comparison config (NEW)
│   ├── nutriplan_config.yaml         # NutriPlan configuration
│   └── baseline_configs.yaml         # Baseline configurations
│
├── data/                              # Data loading and statistics
│   ├── dataset.py                    # Multi-task dataset loader
│   ├── data_statistics.py            # Dataset statistics (original)
│   └── data_statistics_version1.py   # Dataset statistics (fixed version)
│
├── training/                          # Training scripts
│   ├── run_nutriplan.py              # NutriPlan multi-task training
│   └── train_sft.py                  # SFT baseline (Task B only)
│
├── evaluation/                        # Evaluation utilities
│   ├── metrics.py                    # 8 evaluation metrics
│   ├── evaluation.py                 # Main evaluator class
│   └── run_evaluation.py             # Evaluation runner
│
├── baselines/                         # Baseline implementations
│   ├── retrieval.py                  # BM25 + User Similarity
│   ├── rag.py                        # RAG baseline
│   └── zero_shot.py                  # Zero-shot LLM baseline (NEW)
│
├── utils/                             # Utility functions
│   ├── kg_utils.py                   # Knowledge graph accessor (NEW)
│   ├── statistical_tests.py         # Significance testing (NEW)
│   ├── logger.py                     # Logging utilities
│   └── nutrition_calculator.py       # Nutrition calculation
│
├── scripts/                           # Experiment automation (NEW)
│   ├── hyperparameter_search.py      # Stage I.5: Hyperparameter search
│   ├── train_all_llms.sh             # Stage II: Batch training script
│   ├── aggregate_rq1_results.py      # Stage II: RQ1 result aggregation
│   ├── run_rq2_experiments.sh        # Stage III: RQ2 batch experiments
│   └── generate_table_y.py           # Stage III: RQ2 result aggregation
│
├── EXECUTION_GUIDE.md                 # Detailed execution instructions (NEW)
├── README.md                          # This file
├── logs/                              # Training logs
├── checkpoints/                       # Model checkpoints
└── results/                           # Evaluation results
```

**NEW files** indicate code added to support Stages I.5, II (RQ1), and III (RQ2).

## Quick Start

**For complete step-by-step instructions, see [`EXECUTION_GUIDE.md`](EXECUTION_GUIDE.md)**

### 1. Environment Setup

```bash
# Create environment
conda create -n nutriplan python=3.10
conda activate nutriplan

# Install dependencies
pip install torch transformers datasets
pip install pandas numpy scipy scikit-learn
pip install graph-tool  # For KG access
pip install nltk rouge-score tqdm pyyaml
```

### 2. Data Preparation

Ensure your data is at `D:/Downloads/`:
- Task A: `task_a_{train,val,test}_discriminative.jsonl`
- Task B: `task_b_{train,val,test}_from_kg.jsonl`
- Task C: `task_c_{train,val,test}_from_kg.jsonl`

Ensure KG is at: `work/recipebench/kg/nutriplan_kg4.graphml`

### 3. Run Complete Pipeline

```bash
# Stage I.5: Hyperparameter Search (4-8 hours)
python scripts/hyperparameter_search.py \
    --model_name meta-llama/Llama-3.2-3B \
    --search_space configs/search_space.yaml \
    --data_dir D:/Downloads \
    --output_dir experiments/hyperparam_search \
    --search_type random

# Stage II: Train all base LLMs (3-7 days)
bash scripts/train_all_llms.sh

# Aggregate RQ1 results
python scripts/aggregate_rq1_results.py \
    --experiments_dir experiments \
    --models meta-llama/Llama-3.2-3B meta-llama/Llama-3-8B Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \
    --seeds 42 123 2024 \
    --output_file results/table_x.txt

# Stage III: Run all baselines (1-2 days)
bash scripts/run_rq2_experiments.sh
```

**See [`EXECUTION_GUIDE.md`](EXECUTION_GUIDE.md) for detailed instructions, troubleshooting, and time estimates.**

### 3. Training NutriPlan (Multi-Task)

```bash
python training/run_nutriplan.py \
    --model_name meta-llama/Llama-3.2-3B \
    --data_dir "D:\Downloads" \
    --output_dir checkpoints/nutriplan \
    --task_a_ratio 0.5 \
    --task_b_ratio 0.3 \
    --task_c_ratio 0.2 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --fp16 \
    --use_wandb \
    --seed 42
```

**For multi-seed experiments (required for Stage II):**
```bash
for seed in 42 123 2024; do
    python training/run_nutriplan.py \
        --model_name meta-llama/Llama-3.2-3B \
        --output_dir checkpoints/nutriplan_seed_${seed} \
        --seed ${seed}
done
```

### 4. Training Baselines

#### SFT (Task B Only)
```bash
python training/train_sft.py \
    --model_name meta-llama/Llama-3.2-3B \
    --data_dir "D:\Downloads" \
    --output_dir checkpoints/sft_task_b \
    --num_epochs 5 \
    --seed 42
```

#### Retrieval Baseline
```bash
python baselines/retrieval.py \
    --test_file "D:\Downloads\task_a_test_discriminative.jsonl" \
    --recipe_corpus data/recipe_corpus.jsonl \
    --output_file results/retrieval_predictions.jsonl
```

#### RAG Baseline
```bash
python baselines/rag.py \
    --test_file "D:\Downloads\task_b_test_from_kg.jsonl" \
    --recipe_corpus data/recipe_corpus.jsonl \
    --model_name meta-llama/Llama-3.2-3B \
    --output_file results/rag_predictions.jsonl \
    --task b
```

### 5. Evaluation

After generating predictions, run evaluation:

```bash
python evaluation/evaluation.py \
    --predictions results/nutriplan_predictions.jsonl \
    --references "D:\Downloads\task_b_test_from_kg.jsonl" \
    --constraints results/constraints_test.jsonl \
    --kg_facts results/kg_facts_test.jsonl \
    --output_dir results/nutriplan_eval
```

This generates:
- `aggregate_metrics.json` - Mean ± std for all 8 metrics
- `per_sample_metrics.csv` - Detailed per-sample results
- `summary_table.txt` - LaTeX-ready table
- Visualization plots (distribution, boxplot, correlation)

## 📊 Evaluation Metrics

The evaluation script computes **8 comprehensive metrics**:

### Core Metrics
1. **SNCR** (Strict Nutrition Constraint Recall): % of nutrition constraints satisfied
2. **UPM** (User Preference Matching): How well recipe matches user preferences
3. **K-Faith** (Knowledge Graph Faithfulness): Adherence to KG knowledge

### Auxiliary Metrics
4. **AVC** (Average Violation Count): Number of constraint violations (↓ lower is better)
5. **Dist-2** (Diversity): Unique bigram ratio for generation diversity
6. **BLEU**: Text quality (4-gram)
7. **ROUGE-L**: Longest common subsequence F1
8. **Nutrition Accuracy**: Accuracy of nutrition calculation (±15% tolerance)

## 🔬 Stage I Checklist

- [x] Data statistics analysis
- [x] Dataset loaders for Tasks A, B, C
- [x] 8 evaluation metrics implementation
- [x] Main evaluation script with visualization
- [x] NutriPlan multi-task training script
- [x] Baseline implementations:
  - [x] Retrieval (BM25 + User Similarity)
  - [x] RAG (Retrieval + LLM)
  - [x] SFT (Task B only)
- [x] Configuration files
- [x] Logging and experiment tracking
- [x] Utility tools (nutrition calculator, etc.)

## 🎯 Next Steps (Stage II: RQ1)

1. **Train 5 base LLMs** with NutriPlan framework:
   - Llama-3-8B
   - Qwen-2-7B
   - Mistral-7B
   - Gemma-2-9B
   - Yi-1.5-9B

2. **Multi-seed runs**: Train each model with seeds [42, 123, 2024]

3. **Evaluation**: Compute SNCR, UPM, K-Faith with mean ± std

4. **Select best base LLM** for subsequent experiments

## 📝 Configuration

Edit `configs/nutriplan_config.yaml` to customize:
- Model architecture
- Task mixing ratios
- Hyperparameters
- Hardware settings

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Enable `gradient_checkpointing`
- Use `fp16` or `bf16`

### Slow Training
- Enable `multi_gpu` mode
- Increase `num_workers` for data loading
- Use `accelerate` for distributed training

### Evaluation Errors
- Ensure predictions are in correct JSONL format
- Check that all required fields are present
- Verify nutrition values are numeric

## 📚 Citation

```bibtex
@inproceedings{nutriplan2024,
  title={NutriPlan: A Knowledge Graph-Enhanced Benchmark and Reflective Framework for Individualized Nutrition-Constrained Recipe Generation},
  author={Your Name et al.},
  booktitle={Conference Name},
  year={2024}
}
```

## 📧 Contact

For questions or issues, please contact: [your-email@example.com]

## 🔒 License

[Your License Here]
