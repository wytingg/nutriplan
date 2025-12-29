# NutriPlan Quick Start Guide

## ⚡ 5-Minute Setup

### 1. Install Dependencies

```bash
cd nutriplan_experiments
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### 2. Verify Data

Your data should be in `D:\Downloads\`:
```
✅ task_a_train_discriminative.jsonl
✅ task_a_val_discriminative.jsonl
✅ task_a_test_discriminative.jsonl
✅ task_b_train_from_kg.jsonl
✅ task_b_val_from_kg.jsonl
✅ task_b_test_from_kg.jsonl
✅ task_c_train_from_kg.jsonl
✅ task_c_val_from_kg.jsonl
✅ task_c_test_from_kg.jsonl
```

### 3. Run Data Statistics

```bash
python data/data_statistics.py
```

**Output:** `results/data_statistics_report.json`

### 4. Train Your First Model

**Option A: Train NutriPlan (Full Multi-Task)**
```bash
python training/run_nutriplan.py \
    --model_name meta-llama/Llama-3.2-3B \
    --data_dir "D:\Downloads" \
    --output_dir checkpoints/nutriplan_test \
    --num_epochs 3 \
    --batch_size 4 \
    --fp16 \
    --seed 42
```

**Option B: Train SFT Baseline**
```bash
python training/train_sft.py \
    --model_name meta-llama/Llama-3.2-3B \
    --data_dir "D:\Downloads" \
    --output_dir checkpoints/sft_test \
    --num_epochs 3 \
    --batch_size 4 \
    --seed 42
```

### 5. Generate Predictions

After training, generate predictions on test set:

```python
# Example prediction script (you'll need to implement inference)
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_path = "checkpoints/nutriplan_test/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Load test data and generate predictions
# (See evaluation/evaluation.py for format)
```

### 6. Evaluate Results

```bash
python evaluation/evaluation.py \
    --predictions results/predictions.jsonl \
    --references "D:\Downloads\task_b_test_from_kg.jsonl" \
    --constraints "D:\Downloads\task_b_test_from_kg.jsonl" \
    --output_dir results/evaluation
```

**Output:**
- `results/evaluation/aggregate_metrics.json` - Summary statistics
- `results/evaluation/per_sample_metrics.csv` - Detailed results
- `results/evaluation/metrics_distribution.png` - Visualizations

---

## 🎯 Common Workflows

### Workflow 1: Train Multiple Seeds (for RQ1)

```bash
# Automated multi-seed training
python run_all_experiments.py \
    --train_nutriplan_only \
    --model_name meta-llama/Llama-3.2-3B \
    --seeds 42 123 2024
```

### Workflow 2: Compare Baselines

```bash
# 1. Train NutriPlan
python training/run_nutriplan.py --output_dir checkpoints/nutriplan

# 2. Train SFT
python training/train_sft.py --output_dir checkpoints/sft

# 3. Run Retrieval (no training needed)
python baselines/retrieval.py \
    --test_file "D:\Downloads\task_a_test_discriminative.jsonl" \
    --recipe_corpus data/recipe_corpus.jsonl

# 4. Run RAG
python baselines/rag.py \
    --test_file "D:\Downloads\task_b_test_from_kg.jsonl" \
    --recipe_corpus data/recipe_corpus.jsonl

# 5. Evaluate all
python evaluation/evaluation.py --predictions ... (for each model)
```

### Workflow 3: Ablation Study (for RQ4)

```bash
# Train w/o Task A (B+C only)
python training/run_nutriplan.py \
    --task_a_ratio 0.0 \
    --task_b_ratio 0.6 \
    --task_c_ratio 0.4 \
    --output_dir checkpoints/ablation_no_task_a

# Train w/o Task C (A+B only)
python training/run_nutriplan.py \
    --task_a_ratio 0.6 \
    --task_b_ratio 0.4 \
    --task_c_ratio 0.0 \
    --output_dir checkpoints/ablation_no_task_c

# Compare with full model
```

---

## 🔧 Customization

### Change Model Architecture

Edit `configs/nutriplan_config.yaml`:
```yaml
model:
  name: "Qwen/Qwen2-7B"  # or "mistralai/Mistral-7B-v0.3"
```

Or use command line:
```bash
python training/run_nutriplan.py --model_name Qwen/Qwen2-7B
```

### Adjust Task Ratios

```bash
python training/run_nutriplan.py \
    --task_a_ratio 0.4 \
    --task_b_ratio 0.4 \
    --task_c_ratio 0.2
```

### Change Hyperparameters

```bash
python training/run_nutriplan.py \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --num_epochs 10 \
    --warmup_ratio 0.2
```

---

## 📊 Understanding the Output

### Training Output

```
checkpoints/nutriplan/
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
├── best_model/              ← Use this for inference
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
└── training_state.pt
```

### Evaluation Output

```
results/evaluation/
├── aggregate_metrics.json   ← Mean ± Std for all metrics
├── per_sample_metrics.csv   ← Detailed per-sample results
├── failure_cases.jsonl      ← SNCR < 0.5 samples
├── summary_table.txt        ← LaTeX-ready table
├── metrics_distribution.png ← Histograms
├── metrics_boxplot.png      ← Box plots
└── metrics_correlation.png  ← Correlation heatmap
```

### Key Metrics to Check

1. **SNCR** (↑ higher is better): Constraint satisfaction rate
   - Target: >0.85 for NutriPlan
2. **AVC** (↓ lower is better): Average violations per recipe
   - Target: <0.5 for NutriPlan
3. **UPM** (↑ higher is better): User preference matching
   - Target: >0.75
4. **K-Faith** (↑ higher is better): KG knowledge adherence
   - Target: >0.80

---

## 🐛 Troubleshooting

### Problem: CUDA Out of Memory

**Solution 1:** Reduce batch size
```bash
python training/run_nutriplan.py --batch_size 2
```

**Solution 2:** Enable gradient checkpointing
```bash
python training/run_nutriplan.py --gradient_checkpointing
```

**Solution 3:** Use 8-bit quantization
```python
# Add to training script
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### Problem: Training is Too Slow

**Solution:** Use multiple GPUs
```bash
# Method 1: Multi-GPU with DataParallel
python training/run_nutriplan.py --multi_gpu

# Method 2: Distributed training with accelerate
accelerate launch training/run_nutriplan.py
```

### Problem: Evaluation Script Fails

**Common causes:**
1. Missing fields in prediction file
2. Incorrect JSON format
3. Mismatched sample counts

**Debug:**
```python
import json

# Check prediction format
with open('results/predictions.jsonl') as f:
    sample = json.loads(f.readline())
    print(sample.keys())  # Should include 'recipe', 'nutrition', etc.
```

---

## 📞 Getting Help

1. Check `README.md` for detailed documentation
2. Review example scripts in `examples/`
3. Check logs in `logs/` directory
4. Contact: [your-email@example.com]

---

## 🚀 Next Steps

After completing Stage I:

1. ✅ Verify all 7 Stage I tasks are complete (see README)
2. 📊 Analyze data statistics report
3. 🏃 Move to **Stage II: RQ1 (Base LLM Selection)**
   - Train 5 different LLMs
   - Run multi-seed experiments
   - Compare SNCR/UPM/K-Faith
4. 📝 Start drafting Section 7.2 of your paper

**Good luck with your experiments!** 🎉
