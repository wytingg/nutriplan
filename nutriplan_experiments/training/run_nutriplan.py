"""
Main Training Script for NutriPlan Multi-Task Learning
Trains a single LLM on Tasks A, B, C with configurable task ratios
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data.dataset import NutriPlanDataset, create_dataloaders
# from evaluation.evaluation import NutriPlanEvaluator  # 训练时不需要


class NutriPlanTrainer:
    """Multi-task trainer for NutriPlan"""

    def __init__(self, args):
        self.args = args
        # 使用 device_map="auto" 时，不应该手动指定 device
        # 模型会自动分配到多个设备，数据也需要根据输入层的设备来发送
        self.device = None  # 稍后从模型获取

        # 清空 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Set random seeds
        self._set_seed(args.seed)

        # Initialize tokenizer
        print(f"Loading tokenizer: {args.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True  # Required for Phi-3, TinyLlama
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model (使用 FP32 保证数值稳定)
        print(f"Loading model: {args.model_name}")

        # 为不同模型配置特殊参数
        # 大模型（7B+）使用 FP16 节省显存，小模型用 FP32 保证稳定性
        is_large_model = any(name in args.model_name for name in ["Qwen2-7B", "Qwen2", "Mistral", "gemma", "Phi-3.5"])

        if is_large_model:
            print(f"✓ Using FP16 for large model (saves 50% memory)")
            dtype = torch.float16
        else:
            print(f"✓ Using FP32 for small model (more stable)")
            dtype = torch.float32

        # 对于单 GPU 训练，不使用 device_map，直接加载后手动移到 GPU
        # device_map="auto" 会导致模型分散到 CPU+GPU，引起设备不匹配
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True  # 降低 CPU 内存使用
        }

        # 只在真正的多GPU场景才使用 device_map
        if args.multi_gpu:
            model_kwargs["device_map"] = "auto"

        # Phi-3 和 Phi-3.5 需要 eager attention 避免缓存错误
        if "Phi-3" in args.model_name or "phi-3" in args.model_name or "Phi-3.5" in args.model_name:
            model_kwargs["attn_implementation"] = "eager"
            print("✓ Using eager attention for Phi-3/Phi-3.5")

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs
        )

        # 设置设备并移动模型
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            # 如果没有使用 device_map，需要手动移动模型到 GPU
            if not args.multi_gpu:
                self.model = self.model.to(self.device)
            print(f"✓ Model loaded on GPU: cuda:0")
        else:
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
            print("⚠️  No GPU available, using CPU (will be very slow!)")

        # 禁用梯度检查点（与 use_cache 冲突导致训练不稳定）
        # if is_large_model and hasattr(self.model, 'gradient_checkpointing_enable'):
        #     self.model.gradient_checkpointing_enable()
        #     print("✓ Gradient checkpointing enabled (saves 40% memory)")

        # Initialize datasets
        task_ratios = {
            'a': args.task_a_ratio,
            'b': args.task_b_ratio,
            'c': args.task_c_ratio
        }

        print(f"Creating dataloaders with task ratios: {task_ratios}")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=args.data_dir,
            tokenizer=self.tokenizer,
            batch_size=args.batch_size,
            tasks=['a', 'b', 'c'],
            task_ratios=task_ratios,
            num_workers=args.num_workers
        )

        # Initialize optimizer (使用 8-bit Adam 节省显存)
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            print("✓ Using 8-bit AdamW (saves 75% optimizer memory)")
        except ImportError:
            print("⚠️  bitsandbytes not available, using standard AdamW")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )

        # Initialize scheduler
        total_steps = len(self.train_loader) * args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 不使用 GradScaler，因为模型已经是 FP16
        self.scaler = None

        # Initialize wandb
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args)
            )

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        task_losses = {'a': [], 'b': [], 'c': []}

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.args.num_epochs}")

        # Gradient accumulation
        accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        step_in_accumulation = 0

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device (使用非阻塞传输)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            # Forward pass with autocast for FP16
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            step_in_accumulation += 1

            # Only update weights every accumulation_steps
            if step_in_accumulation >= accumulation_steps or batch_idx == len(self.train_loader) - 1:
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                step_in_accumulation = 0
                self.global_step += 1

            # Track metrics (unscaled loss)
            total_loss += loss.item() * accumulation_steps

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Log to wandb
            if self.args.use_wandb and self.global_step % self.args.logging_steps == 0:
                wandb.log({
                    'train/loss': loss.item() * accumulation_steps,
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/global_step': self.global_step
                })

        avg_loss = total_loss / len(self.train_loader)
        return {'loss': avg_loss}

    def evaluate(self, dataloader: DataLoader, split: str = 'val') -> Dict[str, float]:
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(dataloader)
        return {'loss': avg_loss}

    def save_checkpoint(self, checkpoint_path: Path, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(state, checkpoint_path / 'training_state.pt')

        if is_best:
            best_path = checkpoint_path.parent / 'best_model'
            self.model.save_pretrained(best_path)
            self.tokenizer.save_pretrained(best_path)
            print(f"✅ Best model saved to: {best_path}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting NutriPlan Multi-Task Training")
        print("="*80)
        print(f"Model: {self.args.model_name}")
        print(f"Task Ratios: A={self.args.task_a_ratio}, B={self.args.task_b_ratio}, C={self.args.task_c_ratio}")
        print(f"Epochs: {self.args.num_epochs}")
        print(f"Batch Size: {self.args.batch_size}")
        print(f"Learning Rate: {self.args.learning_rate}")
        print(f"Device: {self.device}")
        print("="*80 + "\n")

        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}")

            # Validate
            val_metrics = self.evaluate(self.val_loader, split='val')
            print(f"Epoch {epoch+1} - Val Loss: {val_metrics['loss']:.4f}")

            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss']
                })

            # Early stopping 和模型保存
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0

                # 保存最佳模型
                best_path = Path(self.args.output_dir) / 'best_model'
                best_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(best_path)
                self.tokenizer.save_pretrained(best_path)
                print(f"✅ Best model saved (val_loss={val_metrics['loss']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.args.patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    break

        # Final test evaluation
        print("\n" + "="*80)
        print("Running final test evaluation")
        print("="*80)
        test_metrics = self.evaluate(self.test_loader, split='test')
        print(f"Test Loss: {test_metrics['loss']:.4f}")

        if self.args.use_wandb:
            wandb.log({'test/loss': test_metrics['loss']})
            wandb.finish()

        print("\n✅ Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train NutriPlan Multi-Task Model")

    # Model arguments
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Pretrained model name or path')
    parser.add_argument('--data_dir', type=str, default=r'D:\Downloads',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints/nutriplan',
                        help='Output directory for checkpoints')

    # Task ratio arguments
    parser.add_argument('--task_a_ratio', type=float, default=0.5,
                        help='Task A (Discriminative) sampling ratio')
    parser.add_argument('--task_b_ratio', type=float, default=0.3,
                        help='Task B (Generation) sampling ratio')
    parser.add_argument('--task_c_ratio', type=float, default=0.2,
                        help='Task C (Editing) sampling ratio')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps (effective_batch_size = batch_size * accumulation_steps)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')

    # Hardware arguments
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use multiple GPUs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')

    # Early stopping
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')

    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='nutriplan',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Logging frequency')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Validate task ratios
    total_ratio = args.task_a_ratio + args.task_b_ratio + args.task_c_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        print(f"⚠️  Warning: Task ratios sum to {total_ratio}, normalizing to 1.0")
        args.task_a_ratio /= total_ratio
        args.task_b_ratio /= total_ratio
        args.task_c_ratio /= total_ratio

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = Path(args.output_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Train
    trainer = NutriPlanTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
