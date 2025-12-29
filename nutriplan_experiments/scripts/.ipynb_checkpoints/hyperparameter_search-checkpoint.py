"""
Hyperparameter Search for NutriPlan (Stage I.5)
Uses validation set to find optimal configuration
python work/recipebench/scripts/nutriplan_experiments/scripts/hyperparameter_search.py \
    --model_name meta-llama/Llama-3.2-3B \
    --search_space configs/search_space.yaml \
    --data_dir work/recipebench/data/10large_scale_datasets \
    --output_dir work/recipebench/scripts/nutriplan_experiments/results/hyperparam_search \
    --search_type random

"""

import argparse
import json
import yaml
import sys
from pathlib import Path
import numpy as np
import itertools
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.run_nutriplan import NutriPlanTrainer


class HyperparameterSearch:
    """Grid search or random search for hyperparameters"""

    def __init__(
        self,
        model_name: str,
        search_space_config: str,
        data_dir: str,
        output_dir: str,
        search_type: str = "grid"
    ):
        """
        Args:
            model_name: Base model to use
            search_space_config: Path to search space YAML
            data_dir: Data directory
            output_dir: Output directory
            search_type: "grid" or "random"
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.search_type = search_type

        # Load search space
        with open(search_space_config, 'r') as f:
            self.search_space = yaml.safe_load(f)

        self.results = []

    def generate_configurations(self):
        """Generate all configurations to try"""
        if self.search_type == "grid":
            return self._grid_search_configs()
        elif self.search_type == "random":
            return self._random_search_configs()
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

    def _grid_search_configs(self):
        """Generate grid search configurations"""
        param_names = list(self.search_space.keys())
        param_values = [self.search_space[name] for name in param_names]

        # Generate all combinations
        configs = []
        for values in itertools.product(*param_values):
            config = dict(zip(param_names, values))
            configs.append(config)

        print(f"Generated {len(configs)} configurations for grid search")
        return configs

    def _random_search_configs(self, n_trials: int = 20):
        """Generate random search configurations"""
        configs = []
        for _ in range(n_trials):
            config = {}
            for param, values in self.search_space.items():
                config[param] = np.random.choice(values)
            configs.append(config)

        print(f"Generated {n_trials} configurations for random search")
        return configs

    def run_trial(self, trial_id: int, config: dict):
        """
        Run a single trial with given configuration

        Args:
            trial_id: Trial ID
            config: Configuration dict

        Returns:
            Validation metrics
        """
        print(f"\n{'='*80}")
        print(f"Trial {trial_id}")
        print(f"{'='*80}")
        print(f"Configuration: {config}")

        # Create args object
        class Args:
            pass

        args = Args()
        args.model_name = self.model_name
        args.data_dir = self.data_dir
        args.output_dir = str(self.output_dir / f"trial_{trial_id}")

        # Set hyperparameters from config
        args.learning_rate = config.get('learning_rate', 5e-5)
        args.batch_size = config.get('batch_size', 8)
        args.num_epochs = config.get('num_epochs', 3)  # Use fewer epochs for search
        args.warmup_ratio = config.get('warmup_ratio', 0.1)
        args.weight_decay = config.get('weight_decay', 0.01)
        args.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Task ratios
        args.task_a_ratio = config.get('task_a_ratio', 0.5)
        args.task_b_ratio = config.get('task_b_ratio', 0.3)
        args.task_c_ratio = config.get('task_c_ratio', 0.2)

        # Fixed args
        args.fp16 = True
        args.multi_gpu = False
        args.num_workers = 4
        args.patience = 3
        args.use_wandb = False
        args.run_name = f"hyperparam_search_trial_{trial_id}"
        args.logging_steps = 10
        args.seed = 42

        # Train
        try:
            trainer = NutriPlanTrainer(args)
            trainer.train()

            # Get validation metrics
            val_metrics = trainer.evaluate(trainer.val_loader, split='val')

            result = {
                'trial_id': trial_id,
                'config': config,
                'val_loss': val_metrics['loss'],
                'status': 'success'
            }

        except Exception as e:
            print(f"❌ Trial {trial_id} failed: {e}")
            result = {
                'trial_id': trial_id,
                'config': config,
                'val_loss': float('inf'),
                'status': 'failed',
                'error': str(e)
            }

        return result

    def search(self):
        """Run hyperparameter search"""
        print("\n" + "="*80)
        print("Starting Hyperparameter Search")
        print("="*80)
        print(f"Search type: {self.search_type}")
        print(f"Model: {self.model_name}")
        print(f"Search space: {self.search_space}")

        # Generate configurations
        configs = self.generate_configurations()

        # Run trials
        for trial_id, config in enumerate(configs):
            result = self.run_trial(trial_id, config)
            self.results.append(result)

            # Save intermediate results
            self._save_results()

        # Find best configuration
        best_result = min(self.results, key=lambda x: x['val_loss'])

        print("\n" + "="*80)
        print("Hyperparameter Search Completed")
        print("="*80)
        print(f"Best configuration:")
        print(json.dumps(best_result['config'], indent=2))
        print(f"Best validation loss: {best_result['val_loss']:.4f}")

        # Save best config
        best_config_path = self.output_dir / "best_config.json"
        with open(best_config_path, 'w') as f:
            json.dump(best_result['config'], f, indent=2)

        print(f"\n✅ Best configuration saved to: {best_config_path}")

        return best_result

    def _save_results(self):
        """Save all results"""
        results_path = self.output_dir / "search_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search for NutriPlan")

    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Model to use for search')
    parser.add_argument('--search_space', type=str, required=True,
                        help='Path to search space YAML config')
    parser.add_argument('--data_dir', type=str, default=r'D:\Downloads',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/hyperparam_search',
                        help='Output directory')
    parser.add_argument('--search_type', type=str, choices=['grid', 'random'], default='grid',
                        help='Search type')

    args = parser.parse_args()

    # Run search
    searcher = HyperparameterSearch(
        model_name=args.model_name,
        search_space_config=args.search_space,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        search_type=args.search_type
    )

    best_result = searcher.search()

    print("\n🎉 Hyperparameter search completed!")


if __name__ == "__main__":
    main()
