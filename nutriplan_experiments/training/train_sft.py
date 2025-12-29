"""
SFT (Supervised Fine-Tuning) Baseline for NutriPlan
Trains LLM ONLY on Task B (Constrained Generation)
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class TaskBDataset(Dataset):
    """Dataset for Task B (Generation) only"""

    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_length: int = 2048,
        split: str = 'train'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Load Task B data
        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        print(f"Loaded {len(self.samples)} samples for SFT ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build prompt
        prompt = self._build_prompt(sample)
        # Build target
        target = self._build_target(sample)

        # Concatenate for causal LM training
        full_text = prompt + target

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Create labels (mask prompt tokens)
        labels = encoding['input_ids'].clone()
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[0, :prompt_length] = -100  # Ignore prompt in loss

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

    def _build_prompt(self, sample: dict) -> str:
        """Build Task B prompt"""
        constraints = sample.get('constraints', {})
        kg_context = sample.get('kg_context', {})

        prompt = "### Task: Generate a recipe that satisfies the following constraints.\n\n"

        # Constraints
        prompt += "### Constraints:\n"
        prompt += f"Nutrition Targets: {json.dumps(constraints.get('nutrition_targets', {}))}\n"
        prompt += f"Dietary Restrictions: {constraints.get('dietary_restrictions', [])}\n"
        prompt += f"Allergies: {constraints.get('allergies', [])}\n"
        prompt += f"Cuisine Preference: {constraints.get('cuisine', 'Any')}\n"
        prompt += f"Cooking Time: {constraints.get('max_time', 'Flexible')}\n\n"

        # KG context (if available)
        if kg_context:
            prompt += "### Knowledge Graph Guidance:\n"
            if 'recommended_ingredients' in kg_context:
                prompt += f"Recommended Ingredients: {', '.join(kg_context['recommended_ingredients'][:10])}\n"
            if 'cooking_rules' in kg_context:
                prompt += f"Cooking Tips: {'; '.join(kg_context['cooking_rules'][:3])}\n"
            prompt += "\n"

        prompt += "### Instruction: Generate a complete recipe in JSON format.\n"
        prompt += "### Recipe:\n"

        return prompt

    def _build_target(self, sample: dict) -> str:
        """Build Task B target"""
        recipe = sample.get('recipe', {})
        return json.dumps(recipe, ensure_ascii=False)


def train_sft_baseline(args):
    """Train SFT baseline on Task B only"""

    print("\n" + "="*80)
    print("Training SFT Baseline (Task B Only)")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto" if args.multi_gpu else None
    )

    # Create datasets
    train_file = Path(args.data_dir) / "task_b_train_from_kg.jsonl"
    val_file = Path(args.data_dir) / "task_b_val_from_kg.jsonl"

    train_dataset = TaskBDataset(
        data_file=str(train_file),
        tokenizer=tokenizer,
        split='train'
    )

    val_dataset = TaskBDataset(
        data_file=str(val_file),
        tokenizer=tokenizer,
        split='val'
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type='linear',
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        report_to='wandb' if args.use_wandb else 'none',
        run_name=args.run_name,
        seed=args.seed
    )

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train
    print("\n🚀 Starting training...")
    trainer.train()

    # Save final model
    print("\n💾 Saving model...")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    # Save best model separately
    best_model_path = f"{args.output_dir}/best_model"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    print(f"\n✅ Training completed!")
    print(f"Best model saved to: {best_model_path}")

    if args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train SFT Baseline (Task B Only)")

    # Model arguments
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Pretrained model name')
    parser.add_argument('--data_dir', type=str, default=r'D:\Downloads',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints/sft_task_b',
                        help='Output directory')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm')

    # Hardware
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use multiple GPUs')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing')

    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use W&B')
    parser.add_argument('--wandb_project', type=str, default='nutriplan',
                        help='W&B project')
    parser.add_argument('--run_name', type=str, default='sft_task_b',
                        help='Run name')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Logging steps')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(Path(args.output_dir) / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Train
    train_sft_baseline(args)


if __name__ == "__main__":
    main()
