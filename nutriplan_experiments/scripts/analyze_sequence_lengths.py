"""
Analyze sequence lengths to determine optimal max_length
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
import json
from tqdm import tqdm
import numpy as np

# Configuration
DATA_DIR = "/home/featurize/work/recipebench/data/10large_scale_datasets"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("="*80)
print("Sequence Length Analysis")
print("="*80)

# Initialize tokenizer
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Analyze each task
tasks = ['a', 'b', 'c']
splits = ['train', 'val', 'test']

for task in tasks:
    print(f"\n{'='*80}")
    print(f"Task {task.upper()}")
    print('='*80)

    for split in splits:
        if task == 'a':
            filepath = Path(DATA_DIR) / f"task_a_{split}_discriminative.jsonl"
        else:
            filepath = Path(DATA_DIR) / f"task_{task}_{split}_from_kg.jsonl"

        if not filepath.exists():
            print(f"  {split}: File not found")
            continue

        lengths = []
        prompt_lengths = []
        target_lengths = []

        # Sample first 100 examples for speed
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 100:  # Sample first 100
                    break

                sample = json.loads(line.strip())

                # Reconstruct prompt and target based on task
                if task == 'a':
                    # Simplified approximation
                    user_profile = sample.get('user_profile', {})
                    candidates = sample.get('candidates', [])
                    ranking = sample.get('ranking', [])

                    prompt_text = f"Rank recipes based on: {json.dumps(user_profile)}\nCandidates: {len(candidates)}"
                    target_text = json.dumps(ranking)

                elif task == 'b':
                    constraints = sample.get('constraints', {})
                    recipe = sample.get('recipe', {})
                    kg_context = sample.get('kg_context', {})

                    prompt_text = f"Generate recipe with constraints: {json.dumps(constraints)}"
                    target_text = json.dumps(recipe, ensure_ascii=False)

                else:  # task == 'c'
                    original_recipe = sample.get('original_recipe', {})
                    constraints = sample.get('constraints', {})
                    violations = sample.get('violations', [])
                    edited_recipe = sample.get('edited_recipe', {})

                    prompt_text = f"Edit recipe: {json.dumps(original_recipe)}\nConstraints: {json.dumps(constraints)}\nViolations: {len(violations)}"
                    target_text = json.dumps({'edited_recipe': edited_recipe, 'changes_made': violations}, ensure_ascii=False)

                # Tokenize
                full_text = prompt_text + target_text
                prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
                target_tokens = tokenizer(target_text, add_special_tokens=False)['input_ids']
                full_tokens = tokenizer(full_text, add_special_tokens=True)['input_ids']

                prompt_lengths.append(len(prompt_tokens))
                target_lengths.append(len(target_tokens))
                lengths.append(len(full_tokens))

        if lengths:
            print(f"\n  {split.upper()} (sampled {len(lengths)} examples):")
            print(f"    Full sequence length:")
            print(f"      Mean: {np.mean(lengths):.0f} tokens")
            print(f"      Median: {np.median(lengths):.0f} tokens")
            print(f"      95th percentile: {np.percentile(lengths, 95):.0f} tokens")
            print(f"      Max: {np.max(lengths):.0f} tokens")
            print(f"    Prompt length:")
            print(f"      Mean: {np.mean(prompt_lengths):.0f} tokens")
            print(f"      Median: {np.median(prompt_lengths):.0f} tokens")
            print(f"    Target length:")
            print(f"      Mean: {np.mean(target_lengths):.0f} tokens")
            print(f"      Median: {np.median(target_lengths):.0f} tokens")

            # Calculate truncation percentage
            truncated_at_2048 = sum(1 for l in lengths if l > 2048)
            truncation_pct = (truncated_at_2048 / len(lengths)) * 100
            print(f"    Truncation at 2048: {truncation_pct:.1f}% ({truncated_at_2048}/{len(lengths)} samples)")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)

# Read actual data to give better recommendations
print("\nBased on this analysis:")
print("- If truncation > 10%: Consider increasing max_length or using a different approach")
print("- If truncation < 5%: Current max_length=2048 is acceptable")
print("- Alternative: Use prompt-only masking (only compute loss on target tokens)")
print("="*80)
