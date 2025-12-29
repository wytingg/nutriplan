"""
Debug script to test dataset loading
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from data.dataset import NutriPlanDataset
from torch.utils.data import DataLoader

# Configuration
DATA_DIR = "/home/featurize/work/recipebench/data/10large_scale_datasets"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("="*80)
print("Dataset Debug Test")
print("="*80)

# Initialize tokenizer
print(f"\n1. Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

# Create dataset
print(f"\n2. Creating dataset from: {DATA_DIR}")
dataset = NutriPlanDataset(
    data_dir=DATA_DIR,
    split='train',
    tasks=['a', 'b', 'c'],
    task_ratios={'a': 0.3, 'b': 0.5, 'c': 0.2},
    tokenizer=tokenizer
)
print(f"✓ Dataset created with {len(dataset)} samples")

# Test single sample
print("\n3. Testing single sample access")
try:
    sample = dataset[0]
    print(f"✓ Sample keys: {sample.keys()}")
    print(f"  - input_ids shape: {sample['input_ids'].shape}")
    print(f"  - attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  - labels shape: {sample['labels'].shape}")
    print(f"  - input_ids type: {type(sample['input_ids'])}")
    print(f"  - input_ids dtype: {sample['input_ids'].dtype}")
except Exception as e:
    print(f"✗ Error accessing sample: {e}")
    sys.exit(1)

# Test DataLoader with num_workers=0
print("\n4. Testing DataLoader (num_workers=0)")
try:
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0  # Single process for debugging
    )

    batch = next(iter(loader))
    print(f"✓ Batch loaded successfully")
    print(f"  - Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  - Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  - Batch labels shape: {batch['labels'].shape}")
except Exception as e:
    print(f"✗ Error with num_workers=0: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test DataLoader with num_workers=1
print("\n5. Testing DataLoader (num_workers=1)")
try:
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1  # Multi-process
    )

    batch = next(iter(loader))
    print(f"✓ Batch loaded successfully with multiprocessing")
    print(f"  - Batch input_ids shape: {batch['input_ids'].shape}")
except Exception as e:
    print(f"✗ Error with num_workers=1: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  Multiprocessing fails - need to fix tensor sharing issue")
    sys.exit(1)

print("\n" + "="*80)
print("✅ All tests passed!")
print("="*80)
