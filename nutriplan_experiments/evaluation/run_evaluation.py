#!/usr/bin/env python3
"""
快速评估脚本 - 计算测试集上的损失和困惑度
适用于 DDL 紧急情况
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from data.dataset import NutriPlanDataset, create_dataloaders


def evaluate_model(model_path, data_dir, output_dir, batch_size=4):
    """评估模型并生成指标"""

    print(f"=" * 80)
    print(f"Evaluating model: {model_path}")
    print(f"=" * 80)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型和tokenizer
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"✓ Model loaded on: {device}")

    # 加载测试数据
    print("\nLoading test data...")
    try:
        _, _, test_loader = create_dataloaders(
            data_dir=data_dir,
            tokenizer=tokenizer,
            batch_size=batch_size,
            tasks=['a', 'b', 'c'],
            task_ratios={'a': 0.3, 'b': 0.5, 'c': 0.2},
            num_workers=0
        )
        print(f"✓ Test dataset loaded: {len(test_loader)} batches")
    except Exception as e:
        print(f"✗ Failed to load test data: {e}")
        print("Using simple validation instead...")
        test_loader = None

    # 评估
    results = {}

    if test_loader:
        # 完整评估
        print("\n" + "=" * 80)
        print("Running evaluation on test set...")
        print("=" * 80)

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss.item()
                total_loss += loss * input_ids.size(0)
                total_samples += input_ids.size(0)

        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        results = {
            "model_path": str(model_path),
            "test_loss": float(avg_loss),
            "perplexity": float(perplexity),
            "num_samples": int(total_samples),
            "status": "completed"
        }
    else:
        # 简单验证
        print("\n" + "=" * 80)
        print("Running simple validation...")
        print("=" * 80)

        test_texts = [
            "I want to make a healthy breakfast with eggs and spinach.",
            "What are the nutritional benefits of salmon?",
            "Can you suggest a low-carb dinner recipe?"
        ]

        total_loss = 0.0
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(test_texts)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        results = {
            "model_path": str(model_path),
            "test_loss": float(avg_loss),
            "perplexity": float(perplexity),
            "num_samples": len(test_texts),
            "status": "simple_validation"
        }

    # 保存结果
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Evaluation completed!")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"\nMetrics:")
    print(f"  Test Loss:   {results['test_loss']:.4f}")
    print(f"  Perplexity:  {results['perplexity']:.2f}")
    print(f"  Samples:     {results['num_samples']}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained NutriPlan model")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str,
                        default='~/work/recipebench/data/10large_scale_datasets',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')

    args = parser.parse_args()

    # Expand paths
    args.model_path = str(Path(args.model_path).expanduser())
    args.data_dir = str(Path(args.data_dir).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())

    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
