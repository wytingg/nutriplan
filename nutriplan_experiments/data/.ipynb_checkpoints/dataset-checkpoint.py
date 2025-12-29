"""
Dataset with prompt masking - only compute loss on target tokens
This ensures the model learns to generate complete outputs even with long prompts
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


class NutriPlanDataset(Dataset):
    """Multi-task dataset with prompt masking for better training"""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        tasks: List[str] = ['a', 'b', 'c'],
        task_ratios: Optional[Dict[str, float]] = None,
        tokenizer=None,
        max_length: int = 2048
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = 256  # 降到 512 适配 FP32 + 4090 24GB
        self.task_ratios = task_ratios or {'a': 0.5, 'b': 0.3, 'c': 0.2}

        # Load all task data
        self.task_data = {}
        self.samples = []

        for task in tasks:
            data = self._load_task_data(task, split)
            self.task_data[task] = data
            print(f"Loaded Task {task.upper()} {split}: {len(data)} samples")

        # Create mixed dataset according to ratios
        self._create_mixed_dataset()

    def _load_task_data(self, task: str, split: str) -> List[Dict]:
        """Load data for a specific task"""
        if task == 'a':
            filepath = self.data_dir / f"task_a_{split}_discriminative.jsonl"
        else:
            filepath = self.data_dir / f"task_{task}_{split}_from_kg.jsonl"

        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample['task'] = task
                data.append(sample)

        return data

    def _create_mixed_dataset(self):
        """Create mixed dataset according to task ratios"""
        total_samples = sum(len(data) for data in self.task_data.values())

        for task, ratio in self.task_ratios.items():
            if task not in self.tasks:
                continue

            task_samples = self.task_data[task]
            target_count = int(total_samples * ratio)

            if len(task_samples) >= target_count:
                sampled = random.sample(task_samples, target_count)
            else:
                sampled = random.choices(task_samples, k=target_count)

            self.samples.extend(sampled)

        random.shuffle(self.samples)
        print(f"Created mixed dataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample - 直接使用 instruction + output 格式"""
        sample = self.samples[idx]

        # 获取 instruction 和 output（你的数据格式）
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')  # Task C 有 input 字段
        output = sample.get('output', '')

        # 构建完整的 prompt
        if input_text:
            # Task C: instruction + input + output
            prompt = f"{instruction}\n\n{input_text}\n\n### Response:\n"
        else:
            # Task A, B: instruction + output
            prompt = f"{instruction}\n\n### Response:\n"

        target = output

        return self._format_with_prompt_masking(prompt, target)

    def _format_with_prompt_masking(self, prompt: str, target: str) -> Dict[str, torch.Tensor]:
        """
        Format sample with prompt masking:
        - Tokenize prompt and target separately
        - Concatenate them
        - Set labels to -100 for prompt tokens (no loss computed)
        - Set labels to actual token ids for target tokens (loss computed)
        """
        # Tokenize prompt (no loss on these tokens)
        prompt_encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,  # 截断过长的 prompt
            max_length=self.max_length // 2,  # prompt 最多占一半
            padding=False,
            return_tensors='pt'
        )
        prompt_ids = prompt_encoding['input_ids'].squeeze(0)

        # Tokenize target (loss computed on these tokens)
        remaining_length = self.max_length - len(prompt_ids)
        target_encoding = self.tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,  # 截断过长的 target
            max_length=remaining_length,
            padding=False,
            return_tensors='pt'
        )
        target_ids = target_encoding['input_ids'].squeeze(0)

        # Concatenate prompt + target
        full_ids = torch.cat([prompt_ids, target_ids], dim=0)

        # Create labels: -100 for prompt, actual ids for target
        labels = torch.full_like(full_ids, -100)
        prompt_len = len(prompt_ids)
        labels[prompt_len:] = full_ids[prompt_len:]  # 只对 target 计算 loss

        # Create attention mask
        attention_mask = torch.ones_like(full_ids)

        # Pad to max_length (动态填充，不是固定填充)
        current_len = len(full_ids)
        if current_len < self.max_length:
            padding_len = self.max_length - current_len
            full_ids = torch.cat([full_ids, torch.full((padding_len,), self.tokenizer.pad_token_id)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_len)])
            labels = torch.cat([labels, torch.full((padding_len,), -100)])

        return {
            'input_ids': full_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }



def create_dataloaders(
    data_dir: str,
    tokenizer,
    batch_size: int = 8,
    tasks: List[str] = ['a', 'b', 'c'],
    task_ratios: Optional[Dict[str, float]] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""

    train_dataset = NutriPlanDataset(
        data_dir=data_dir,
        split='train',
        tasks=tasks,
        task_ratios=task_ratios,
        tokenizer=tokenizer
    )

    val_dataset = NutriPlanDataset(
        data_dir=data_dir,
        split='val',
        tasks=tasks,
        task_ratios=task_ratios,
        tokenizer=tokenizer
    )

    test_dataset = NutriPlanDataset(
        data_dir=data_dir,
        split='test',
        tasks=tasks,
        task_ratios=task_ratios,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 设为 0 避免多进程问题
        pin_memory=False  # 大模型时禁用 pin_memory 避免 OOM
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader
