# NutriPlan Usage Examples

Complete examples for running NutriPlan experiments.

---

## Example 1: Complete Stage I Pipeline

Run the entire Stage I setup in one command:

```bash
# Install dependencies first
pip install -r requirements.txt

# Run complete pipeline
python run_all_experiments.py \
    --run_all \
    --data_dir "D:\Downloads" \
    --output_base experiments/stage1 \
    --model_name meta-llama/Llama-3.2-3B \
    --recipe_corpus data/recipe_corpus.jsonl \
    --seeds 42 123 2024
```

**This will:**
1. ✅ Generate data statistics
2. ✅ Train NutriPlan with 3 seeds
3. ✅ Train SFT baseline
4. ✅ Run Retrieval baseline
5. ✅ Run RAG baseline
6. ✅ Generate comprehensive report

**Expected time:** 24-48 hours on 8×A100 GPUs

---

## Example 2: Individual Components

### Step-by-Step Execution

#### 1. Data Analysis
```bash
# Analyze dataset statistics
python data/data_statistics.py

# Output: results/data_statistics_report.json
```

**Example output:**
```json
{
  "dataset_overview": {
    "train": 45000,
    "val": 5000,
    "test": 10000
  },
  "task_statistics": {
    "task_a": {
      "train": {
        "num_samples": 15000,
        "avg_candidates": 8.5,
        "avg_constraints_per_user": 4.2
      }
    }
  }
}
```

#### 2. Train NutriPlan (Single Seed)
```bash
python training/run_nutriplan.py \
    --model_name meta-llama/Llama-3.2-3B \
    --data_dir "D:\Downloads" \
    --output_dir checkpoints/nutriplan_seed_42 \
    --task_a_ratio 0.5 \
    --task_b_ratio 0.3 \
    --task_c_ratio 0.2 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --fp16 \
    --multi_gpu \
    --use_wandb \
    --wandb_project nutriplan \
    --run_name nutriplan_llama3_seed42 \
    --seed 42
```

**Expected output:**
```
================================================================================
Starting NutriPlan Multi-Task Training
================================================================================
Model: meta-llama/Llama-3.2-3B
Task Ratios: A=0.5, B=0.3, C=0.2
Epochs: 5
Batch Size: 8
Learning Rate: 5e-05
Device: cuda
================================================================================

Loaded Task A train: 15000 samples
Loaded Task B train: 9000 samples
Loaded Task C train: 6000 samples
Created mixed dataset: 30000 samples

Epoch 1 - Train Loss: 1.2345
Epoch 1 - Val Loss: 1.1234
✅ Best model saved to: checkpoints/nutriplan_seed_42/best_model

...

✅ Training completed!
```

#### 3. Train SFT Baseline
```bash
python training/train_sft.py \
    --model_name meta-llama/Llama-3.2-3B \
    --data_dir "D:\Downloads" \
    --output_dir checkpoints/sft_task_b_seed_42 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --fp16 \
    --seed 42
```

#### 4. Run Retrieval Baseline
```bash
python baselines/retrieval.py \
    --test_file "D:\Downloads\task_a_test_discriminative.jsonl" \
    --recipe_corpus data/recipe_corpus.jsonl \
    --output_file results/retrieval_predictions.jsonl \
    --top_k 10
```

**Example output:**
```
Loaded 50000 recipes from corpus
Building BM25 index...
Evaluating retrieval: 100%|████████| 1000/1000 [00:15<00:00, 65.43it/s]
✅ Retrieval predictions saved to: results/retrieval_predictions.jsonl
```

#### 5. Run RAG Baseline
```bash
python baselines/rag.py \
    --test_file "D:\Downloads\task_b_test_from_kg.jsonl" \
    --recipe_corpus data/recipe_corpus.jsonl \
    --model_name meta-llama/Llama-3.2-3B \
    --output_file results/rag_task_b_predictions.jsonl \
    --task b
```

#### 6. Evaluate Model
```bash
python evaluation/evaluation.py \
    --predictions results/nutriplan_predictions.jsonl \
    --references "D:\Downloads\task_b_test_from_kg.jsonl" \
    --constraints "D:\Downloads\task_b_test_from_kg.jsonl" \
    --output_dir results/nutriplan_eval \
    --failure_threshold 0.5
```

**Example output:**
```
================================================================================
NutriPlan Evaluation Results Summary
================================================================================

Metric                    Mean         Std          Median
--------------------------------------------------------------------------------
SNCR                      0.8523       0.1234       0.8650
UPM                       0.7845       0.0987       0.7920
K-FAITH                   0.8156       0.1045       0.8240
AVC (↓)                   0.4231       0.2156       0.3800
DIST-2                    0.6723       0.0845       0.6810
BLEU                      0.5634       0.1123       0.5720
ROUGE-L                   0.6234       0.0956       0.6310
NUTRITION-ACCURACY        0.8945       0.0623       0.9020
--------------------------------------------------------------------------------

✅ Evaluation completed successfully!
```

---

## Example 3: Multi-Seed Experiment (RQ1)

For statistical robustness, train with 3 seeds:

```bash
#!/bin/bash
# train_multiseed.sh

MODEL_NAME="meta-llama/Llama-3.2-3B"
DATA_DIR="D:\Downloads"

for SEED in 42 123 2024; do
    echo "Training with seed $SEED..."

    python training/run_nutriplan.py \
        --model_name $MODEL_NAME \
        --data_dir $DATA_DIR \
        --output_dir "checkpoints/nutriplan_seed_${SEED}" \
        --num_epochs 5 \
        --batch_size 8 \
        --learning_rate 5e-5 \
        --fp16 \
        --seed $SEED \
        --run_name "nutriplan_seed_${SEED}"

    echo "✅ Completed seed $SEED"
done

echo "✅ All seeds completed!"
```

**Then aggregate results:**

```python
# aggregate_results.py
import json
import numpy as np
from pathlib import Path

seeds = [42, 123, 2024]
all_metrics = []

for seed in seeds:
    eval_file = Path(f"results/nutriplan_seed_{seed}_eval/aggregate_metrics.json")
    with open(eval_file) as f:
        metrics = json.load(f)
        all_metrics.append(metrics)

# Compute mean ± std across seeds
for metric in ['sncr', 'upm', 'k_faith', 'avc']:
    values = [m[metric]['mean'] for m in all_metrics]
    mean = np.mean(values)
    std = np.std(values)
    print(f"{metric.upper()}: {mean:.4f} ± {std:.4f}")
```

**Output:**
```
SNCR: 0.8523 ± 0.0123
UPM: 0.7845 ± 0.0087
K-FAITH: 0.8156 ± 0.0145
AVC: 0.4231 ± 0.0234
```

---

## Example 4: Ablation Study (RQ4)

### Train without Task C (No Reflective Editing)

```bash
python training/run_nutriplan.py \
    --output_dir checkpoints/ablation_no_task_c \
    --task_a_ratio 0.625 \
    --task_b_ratio 0.375 \
    --task_c_ratio 0.0 \
    --run_name "ablation_no_task_c"
```

### Train without Task A (No Discriminative Ranking)

```bash
python training/run_nutriplan.py \
    --output_dir checkpoints/ablation_no_task_a \
    --task_a_ratio 0.0 \
    --task_b_ratio 0.6 \
    --task_c_ratio 0.4 \
    --run_name "ablation_no_task_a"
```

### Compare Results

```python
# compare_ablations.py
import json
import pandas as pd

models = {
    'Full (A+B+C)': 'results/nutriplan_full_eval/aggregate_metrics.json',
    'w/o Task C (A+B)': 'results/ablation_no_task_c_eval/aggregate_metrics.json',
    'w/o Task A (B+C)': 'results/ablation_no_task_a_eval/aggregate_metrics.json',
    'w/o KG (SFT)': 'results/sft_eval/aggregate_metrics.json'
}

results = []
for name, path in models.items():
    with open(path) as f:
        metrics = json.load(f)
        results.append({
            'Model': name,
            'SNCR': metrics['sncr']['mean'],
            'UPM': metrics['upm']['mean'],
            'K-Faith': metrics['k_faith']['mean'],
            'AVC': metrics['avc']['mean']
        })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

**Output:**
```
| Model           |   SNCR |    UPM | K-Faith |    AVC |
|:----------------|-------:|-------:|--------:|-------:|
| Full (A+B+C)    | 0.8523 | 0.7845 |  0.8156 | 0.4231 |
| w/o Task C (A+B)| 0.7234 | 0.7756 |  0.8045 | 1.2345 |
| w/o Task A (B+C)| 0.8412 | 0.6923 |  0.7534 | 0.5123 |
| w/o KG (SFT)    | 0.7123 | 0.7012 |  0.6845 | 1.3456 |
```

**Key observation:** Removing Task C causes SNCR to drop by 12.9% and AVC to increase by 2.9×, proving its importance for constraint satisfaction!

---

## Example 5: Testing Different LLMs (RQ1)

```bash
#!/bin/bash
# test_different_llms.sh

MODELS=(
    "meta-llama/Llama-3.2-3B"
    "Qwen/Qwen2-7B"
    "mistralai/Mistral-7B-v0.3"
    "google/gemma-2-9b"
    "01-ai/Yi-1.5-9B"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo $MODEL | tr '/' '_')

    echo "Training $MODEL_NAME..."

    python training/run_nutriplan.py \
        --model_name $MODEL \
        --output_dir "checkpoints/rq1_${MODEL_NAME}" \
        --num_epochs 5 \
        --seed 42 \
        --run_name "rq1_${MODEL_NAME}"

    # Evaluate
    python evaluation/evaluation.py \
        --predictions "results/${MODEL_NAME}_predictions.jsonl" \
        --references "D:\Downloads\task_b_test_from_kg.jsonl" \
        --constraints "D:\Downloads\task_b_test_from_kg.jsonl" \
        --output_dir "results/rq1_${MODEL_NAME}_eval"
done

# Aggregate results for Table X
python scripts/generate_table_x.py
```

---

## Example 6: Using with W&B (Weights & Biases)

```bash
# Login to W&B first
wandb login

# Train with W&B tracking
python training/run_nutriplan.py \
    --use_wandb \
    --wandb_project nutriplan \
    --run_name "experiment_v1" \
    --model_name meta-llama/Llama-3.2-3B \
    --num_epochs 5 \
    --seed 42
```

**View results at:** https://wandb.ai/your-username/nutriplan

**Key metrics tracked:**
- `train/loss` - Training loss per step
- `train/learning_rate` - Learning rate schedule
- `val/loss` - Validation loss per epoch
- `test/loss` - Final test loss

---

## Example 7: Custom Data Format

If you have custom data, format it as JSONL:

### Task B (Generation) Format
```json
{
  "constraints": {
    "nutrition_targets": {
      "calories": {"value": 500, "type": "max"},
      "sodium": {"value": 500, "type": "max"},
      "protein": {"value": 30, "type": "min"}
    },
    "dietary_restrictions": ["vegetarian", "low-sodium"],
    "allergies": ["nuts", "dairy"],
    "cuisine": "Mediterranean",
    "max_time": 30
  },
  "recipe": {
    "title": "Mediterranean Quinoa Bowl",
    "ingredients": [
      {"name": "quinoa", "quantity": 100, "unit": "g"},
      {"name": "chickpeas", "quantity": 150, "unit": "g"},
      {"name": "cucumber", "quantity": 50, "unit": "g"},
      {"name": "tomato", "quantity": 80, "unit": "g"},
      {"name": "olive oil", "quantity": 10, "unit": "ml"},
      {"name": "lemon juice", "quantity": 15, "unit": "ml"}
    ],
    "steps": [
      "Cook quinoa according to package instructions",
      "Drain and rinse chickpeas",
      "Chop cucumber and tomato",
      "Mix all ingredients in a bowl",
      "Drizzle with olive oil and lemon juice",
      "Season with herbs and serve"
    ],
    "nutrition": {
      "calories": 480,
      "protein": 18,
      "fat": 12,
      "carbs": 72,
      "sodium": 420,
      "fiber": 14,
      "sugar": 8
    },
    "tags": ["vegetarian", "vegan", "low-sodium", "mediterranean"]
  },
  "kg_context": {
    "recommended_ingredients": ["quinoa", "chickpeas", "olive oil", "lemon"],
    "cooking_rules": ["Rinse quinoa before cooking", "Use fresh lemon juice"]
  }
}
```

---

## Example 8: Error Debugging

### Check Training Logs
```bash
# View real-time logs
tail -f logs/nutriplan_20241028_143052.log

# Search for errors
grep -i "error" logs/*.log

# Check GPU usage
nvidia-smi -l 1
```

### Validate Data Format
```python
# validate_data.py
import json

def validate_task_b_sample(sample):
    required_fields = ['constraints', 'recipe']
    for field in required_fields:
        assert field in sample, f"Missing field: {field}"

    # Check constraints
    constraints = sample['constraints']
    assert 'nutrition_targets' in constraints

    # Check recipe
    recipe = sample['recipe']
    assert 'ingredients' in recipe
    assert 'steps' in recipe
    assert 'nutrition' in recipe

    print("✅ Sample is valid")

# Test
with open('D:\Downloads\task_b_train_from_kg.jsonl') as f:
    sample = json.loads(f.readline())
    validate_task_b_sample(sample)
```

---

## Summary of Key Commands

```bash
# 1. Data analysis
python data/data_statistics.py

# 2. Train NutriPlan
python training/run_nutriplan.py --model_name MODEL --seed SEED

# 3. Train SFT baseline
python training/train_sft.py --model_name MODEL --seed SEED

# 4. Run Retrieval
python baselines/retrieval.py --test_file FILE --recipe_corpus CORPUS

# 5. Run RAG
python baselines/rag.py --test_file FILE --model_name MODEL

# 6. Evaluate
python evaluation/evaluation.py --predictions PREDS --references REFS

# 7. Full pipeline
python run_all_experiments.py --run_all
```

**For more help:** See `README.md` and `QUICKSTART.md`
