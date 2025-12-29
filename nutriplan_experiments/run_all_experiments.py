"""
Master script to run all Stage I experiments
Automates the complete experimental pipeline
"""

import subprocess
import argparse
from pathlib import Path
import json
import time
from datetime import datetime


class ExperimentRunner:
    """Master experiment runner"""

    def __init__(self, data_dir: str, output_base: str, seeds: list = [42, 123, 2024]):
        self.data_dir = Path(data_dir)
        self.output_base = Path(output_base)
        self.seeds = seeds
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Log file
        self.log_file = self.output_base / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

    def run_command(self, command: list, description: str):
        """Run shell command with logging"""
        self.log(f"Starting: {description}")
        self.log(f"Command: {' '.join(command)}")

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            elapsed = time.time() - start_time
            self.log(f"✅ Completed in {elapsed:.2f}s: {description}")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed: {description}")
            self.log(f"Error: {e.stderr}")
            return False

    def run_data_statistics(self):
        """Step 1: Run data statistics"""
        self.log("="*80)
        self.log("STEP 1: Data Statistics Analysis")
        self.log("="*80)

        command = ["python", "data/data_statistics.py"]
        return self.run_command(command, "Data statistics analysis")

    def train_nutriplan_multiseed(self, model_name: str = "meta-llama/Llama-3.2-3B"):
        """Step 2: Train NutriPlan with multiple seeds"""
        self.log("="*80)
        self.log("STEP 2: Train NutriPlan (Multi-Seed)")
        self.log("="*80)

        results = []
        for seed in self.seeds:
            output_dir = self.output_base / f"nutriplan_seed_{seed}"

            command = [
                "python", "training/run_nutriplan.py",
                "--model_name", model_name,
                "--data_dir", str(self.data_dir),
                "--output_dir", str(output_dir),
                "--task_a_ratio", "0.5",
                "--task_b_ratio", "0.3",
                "--task_c_ratio", "0.2",
                "--num_epochs", "5",
                "--batch_size", "8",
                "--learning_rate", "5e-5",
                "--fp16",
                "--seed", str(seed)
            ]

            success = self.run_command(
                command,
                f"NutriPlan training (seed={seed})"
            )
            results.append((seed, success))

        return results

    def train_sft_baseline(self, model_name: str = "meta-llama/Llama-3.2-3B"):
        """Step 3: Train SFT baseline"""
        self.log("="*80)
        self.log("STEP 3: Train SFT Baseline (Task B Only)")
        self.log("="*80)

        output_dir = self.output_base / "sft_task_b"

        command = [
            "python", "training/train_sft.py",
            "--model_name", model_name,
            "--data_dir", str(self.data_dir),
            "--output_dir", str(output_dir),
            "--num_epochs", "5",
            "--batch_size", "8",
            "--seed", "42"
        ]

        return self.run_command(command, "SFT baseline training")

    def run_retrieval_baseline(self, recipe_corpus: str):
        """Step 4: Run retrieval baseline"""
        self.log("="*80)
        self.log("STEP 4: Run Retrieval Baseline")
        self.log("="*80)

        test_file = self.data_dir / "task_a_test_discriminative.jsonl"
        output_file = self.output_base / "results" / "retrieval_predictions.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "python", "baselines/retrieval.py",
            "--test_file", str(test_file),
            "--recipe_corpus", recipe_corpus,
            "--output_file", str(output_file),
            "--top_k", "10"
        ]

        return self.run_command(command, "Retrieval baseline evaluation")

    def run_rag_baseline(self, recipe_corpus: str, model_name: str = "meta-llama/Llama-3.2-3B"):
        """Step 5: Run RAG baseline"""
        self.log("="*80)
        self.log("STEP 5: Run RAG Baseline")
        self.log("="*80)

        test_file = self.data_dir / "task_b_test_from_kg.jsonl"
        output_file = self.output_base / "results" / "rag_predictions.jsonl"

        command = [
            "python", "baselines/rag.py",
            "--test_file", str(test_file),
            "--recipe_corpus", recipe_corpus,
            "--model_name", model_name,
            "--output_file", str(output_file),
            "--task", "b"
        ]

        return self.run_command(command, "RAG baseline evaluation")

    def run_evaluation(self, model_name: str, predictions_file: str):
        """Step 6: Run evaluation"""
        self.log("="*80)
        self.log(f"STEP 6: Evaluate {model_name}")
        self.log("="*80)

        references_file = self.data_dir / "task_b_test_from_kg.jsonl"
        output_dir = self.output_base / "results" / f"{model_name}_eval"

        # Note: You'll need to prepare constraints and kg_facts files
        command = [
            "python", "evaluation/evaluation.py",
            "--predictions", predictions_file,
            "--references", str(references_file),
            "--constraints", str(references_file),  # Assuming constraints are in the same file
            "--output_dir", str(output_dir)
        ]

        return self.run_command(command, f"Evaluation for {model_name}")

    def generate_report(self):
        """Generate final experiment report"""
        self.log("="*80)
        self.log("Generating Final Report")
        self.log("="*80)

        report = {
            "experiment_date": datetime.now().isoformat(),
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_base),
            "seeds": self.seeds,
            "log_file": str(self.log_file)
        }

        report_file = self.output_base / "experiment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.log(f"✅ Report saved to: {report_file}")

    def run_all(self, model_name: str, recipe_corpus: str):
        """Run complete experimental pipeline"""
        self.log("="*80)
        self.log("STARTING COMPLETE EXPERIMENTAL PIPELINE")
        self.log("="*80)

        pipeline_start = time.time()

        # Step 1: Data statistics
        self.run_data_statistics()

        # Step 2: Train NutriPlan (multi-seed)
        self.train_nutriplan_multiseed(model_name)

        # Step 3: Train SFT baseline
        self.train_sft_baseline(model_name)

        # Step 4: Retrieval baseline
        self.run_retrieval_baseline(recipe_corpus)

        # Step 5: RAG baseline
        self.run_rag_baseline(recipe_corpus, model_name)

        # Step 6: Generate report
        self.generate_report()

        total_time = time.time() - pipeline_start
        self.log("="*80)
        self.log(f"✅ PIPELINE COMPLETED in {total_time/3600:.2f} hours")
        self.log("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run all NutriPlan experiments")

    parser.add_argument('--data_dir', type=str, default=r'D:\Downloads',
                        help='Data directory')
    parser.add_argument('--output_base', type=str, default='experiments',
                        help='Base output directory')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Model name')
    parser.add_argument('--recipe_corpus', type=str, default='data/recipe_corpus.jsonl',
                        help='Recipe corpus path')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024],
                        help='Random seeds')

    # Experiment selection
    parser.add_argument('--run_all', action='store_true',
                        help='Run complete pipeline')
    parser.add_argument('--data_stats_only', action='store_true',
                        help='Run data statistics only')
    parser.add_argument('--train_nutriplan_only', action='store_true',
                        help='Train NutriPlan only')

    args = parser.parse_args()

    runner = ExperimentRunner(args.data_dir, args.output_base, args.seeds)

    if args.run_all:
        runner.run_all(args.model_name, args.recipe_corpus)
    elif args.data_stats_only:
        runner.run_data_statistics()
    elif args.train_nutriplan_only:
        runner.train_nutriplan_multiseed(args.model_name)
    else:
        print("Please specify --run_all or a specific experiment flag")
        print("Use --help for more options")


if __name__ == "__main__":
    main()
