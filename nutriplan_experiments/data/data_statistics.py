"""
Data Statistics Analysis for NutriPlan-Bench
Analyzes train/val/test splits for Task A, B, C
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import numpy as np


class DataStatistics:
    """Comprehensive statistics for NutriPlan-Bench dataset"""

    def __init__(self, data_dir: str = r"D:\Downloads"):
        self.data_dir = Path(data_dir)
        self.tasks = ['a', 'b', 'c']
        self.splits = ['train', 'val', 'test']

    def load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def get_filepath(self, task: str, split: str) -> Path:
        """Get filepath for specific task and split"""
        if task == 'a':
            return self.data_dir / f"task_a_{split}_discriminative.jsonl"
        else:
            return self.data_dir / f"task_{task}_{split}_from_kg.jsonl"

    def analyze_task_a(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze Task A (Discriminative Ranking) data"""
        stats = {
            'num_samples': len(data),
            'avg_ranked_recipes': [],
            'physiological_states': Counter(),
            'gender_distribution': Counter(),
            'age_distribution': [],
            'instruction_types': Counter(),
            'avg_nutrition_constraints': 0
        }

        for sample in data:
            # Ranked recipes
            ranked_recipes = sample.get('ranked_recipes', [])
            stats['avg_ranked_recipes'].append(len(ranked_recipes))

            # User profile
            user_profile = sample.get('user_profile', {})
            stats['physiological_states'][user_profile.get('physiological_state', 'unknown')] += 1
            stats['gender_distribution'][user_profile.get('gender', 'unknown')] += 1

            age = user_profile.get('age')
            if age:
                stats['age_distribution'].append(age)

            # Nutrition RNI (constraints)
            nutrition_rni = user_profile.get('nutrition_rni', {})
            if nutrition_rni:
                stats['avg_nutrition_constraints'] = len(nutrition_rni)

            # Instruction types
            instruction_type = sample.get('instruction_type', 'unknown')
            stats['instruction_types'][instruction_type] += 1

        stats['avg_ranked_recipes'] = np.mean(stats['avg_ranked_recipes']) if stats['avg_ranked_recipes'] else 0
        stats['avg_age'] = np.mean(stats['age_distribution']) if stats['age_distribution'] else 0

        return stats

    def analyze_task_b(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze Task B (Constrained Generation) data"""
        stats = {
            'num_samples': len(data),
            'avg_ingredients': [],
            'avg_steps': [],
            'physiological_states': Counter(),
            'gender_distribution': Counter(),
            'age_distribution': [],
            'instruction_types': Counter(),
            'avg_liked_ingredients': [],
            'avg_disliked_ingredients': [],
            'avg_servings': [],
            'nutrition_rni_fields': set()
        }

        for sample in data:
            # Output recipe
            output = sample.get('output', {})
            ingredients = output.get('ingredients', [])
            steps = output.get('steps', [])
            servings = output.get('servings', 1)

            stats['avg_ingredients'].append(len(ingredients))
            stats['avg_steps'].append(len(steps))
            stats['avg_servings'].append(servings)

            # User profile
            user_profile = sample.get('user_profile', {})
            stats['physiological_states'][user_profile.get('physiological_state', 'unknown')] += 1
            stats['gender_distribution'][user_profile.get('gender', 'unknown')] += 1

            age = user_profile.get('age')
            if age:
                stats['age_distribution'].append(age)

            # Liked/disliked ingredients
            liked = user_profile.get('liked_ingredients', [])
            disliked = user_profile.get('disliked_ingredients', [])
            stats['avg_liked_ingredients'].append(len(liked))
            stats['avg_disliked_ingredients'].append(len(disliked))

            # Nutrition RNI fields
            nutrition_rni = user_profile.get('nutrition_rni', {})
            stats['nutrition_rni_fields'].update(nutrition_rni.keys())

            # Instruction types
            instruction_type = sample.get('instruction_type', 'unknown')
            stats['instruction_types'][instruction_type] += 1

        stats['avg_ingredients'] = np.mean(stats['avg_ingredients']) if stats['avg_ingredients'] else 0
        stats['avg_steps'] = np.mean(stats['avg_steps']) if stats['avg_steps'] else 0
        stats['avg_servings'] = np.mean(stats['avg_servings']) if stats['avg_servings'] else 0
        stats['avg_liked_ingredients'] = np.mean(stats['avg_liked_ingredients']) if stats['avg_liked_ingredients'] else 0
        stats['avg_disliked_ingredients'] = np.mean(stats['avg_disliked_ingredients']) if stats['avg_disliked_ingredients'] else 0
        stats['avg_age'] = np.mean(stats['age_distribution']) if stats['age_distribution'] else 0
        stats['nutrition_rni_fields'] = list(stats['nutrition_rni_fields'])

        return stats

    def analyze_task_c(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze Task C (Reflective Editing) data"""
        stats = {
            'num_samples': len(data),
            'violation_types': Counter(),
            'violation_severity': Counter(),
            'avg_violations_per_sample': [],
            'avg_corrections_per_sample': [],
            'correction_actions': Counter(),
            'physiological_states': Counter(),
            'gender_distribution': Counter(),
            'age_distribution': [],
            'instruction_types': Counter(),
            'avg_ingredients_before': [],
            'avg_ingredients_after': []
        }

        for sample in data:
            # Input violations
            input_data = sample.get('input', {})
            violations = input_data.get('violations', [])
            stats['avg_violations_per_sample'].append(len(violations))

            for violation in violations:
                vtype = violation.get('type', 'unknown')
                severity = violation.get('severity', 'unknown')
                stats['violation_types'][vtype] += 1
                stats['violation_severity'][severity] += 1

            # Output corrections
            output = sample.get('output', {})
            corrections = output.get('corrections', [])
            stats['avg_corrections_per_sample'].append(len(corrections))

            for correction in corrections:
                action = correction.get('action', 'unknown')
                stats['correction_actions'][action] += 1

            # Recipe comparison
            violated_recipe = input_data.get('violated_recipe', {})
            corrected_recipe = output.get('corrected_recipe', {})

            violated_ingredients = violated_recipe.get('ingredients', [])
            corrected_ingredients = corrected_recipe.get('ingredients', [])

            stats['avg_ingredients_before'].append(len(violated_ingredients))
            stats['avg_ingredients_after'].append(len(corrected_ingredients))

            # User profile
            user_profile = sample.get('user_profile', {})
            stats['physiological_states'][user_profile.get('physiological_state', 'unknown')] += 1
            stats['gender_distribution'][user_profile.get('gender', 'unknown')] += 1

            age = user_profile.get('age')
            if age:
                stats['age_distribution'].append(age)

            # Instruction types
            instruction_type = sample.get('instruction_type', 'unknown')
            stats['instruction_types'][instruction_type] += 1

        stats['avg_violations_per_sample'] = np.mean(stats['avg_violations_per_sample']) if stats['avg_violations_per_sample'] else 0
        stats['avg_corrections_per_sample'] = np.mean(stats['avg_corrections_per_sample']) if stats['avg_corrections_per_sample'] else 0
        stats['avg_ingredients_before'] = np.mean(stats['avg_ingredients_before']) if stats['avg_ingredients_before'] else 0
        stats['avg_ingredients_after'] = np.mean(stats['avg_ingredients_after']) if stats['avg_ingredients_after'] else 0
        stats['avg_age'] = np.mean(stats['age_distribution']) if stats['age_distribution'] else 0

        return stats

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistics report"""
        report = {
            'dataset_overview': {},
            'task_statistics': {}
        }

        # Overall statistics
        total_samples = {'train': 0, 'val': 0, 'test': 0}

        for task in self.tasks:
            report['task_statistics'][f'task_{task}'] = {}

            for split in self.splits:
                filepath = self.get_filepath(task, split)

                if not filepath.exists():
                    print(f"Warning: {filepath} not found")
                    continue

                data = self.load_jsonl(filepath)
                total_samples[split] += len(data)

                # Task-specific analysis
                if task == 'a':
                    stats = self.analyze_task_a(data)
                elif task == 'b':
                    stats = self.analyze_task_b(data)
                else:  # task == 'c'
                    stats = self.analyze_task_c(data)

                report['task_statistics'][f'task_{task}'][split] = stats

        report['dataset_overview'] = total_samples
        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted statistics report"""
        print("=" * 80)
        print("NutriPlan-Bench Dataset Statistics Report")
        print("=" * 80)

        # Overview
        print("\n[Dataset Overview]")
        print("-" * 80)
        overview = report['dataset_overview']
        print(f"{'Split':<15} {'Total Samples':<20}")
        print("-" * 80)
        for split, count in overview.items():
            print(f"{split.capitalize():<15} {count:<20,}")

        # Task-specific statistics
        for task_name, task_stats in report['task_statistics'].items():
            print(f"\n\n{'=' * 80}")
            print(f"[{task_name.upper()} Statistics]")
            print("=" * 80)

            for split, stats in task_stats.items():
                print(f"\n{split.capitalize()} Split:")
                print("-" * 80)
                print(f"  Samples: {stats.get('num_samples', 0):,}")

                # Task A specific
                if 'avg_ranked_recipes' in stats:
                    print(f"  Avg Ranked Recipes per Query: {stats['avg_ranked_recipes']:.2f}")
                    print(f"  Avg Age: {stats.get('avg_age', 0):.1f}")
                    print(f"  Nutrition Constraints: {stats['avg_nutrition_constraints']} fields")
                    print(f"  Physiological States: {dict(stats['physiological_states'])}")
                    print(f"  Gender Distribution: {dict(stats['gender_distribution'])}")
                    print(f"  Instruction Types: {dict(stats['instruction_types'])}")

                # Task B specific
                if 'avg_ingredients' in stats:
                    print(f"  Avg Ingredients per Recipe: {stats['avg_ingredients']:.2f}")
                    print(f"  Avg Steps per Recipe: {stats['avg_steps']:.2f}")
                    print(f"  Avg Servings: {stats['avg_servings']:.1f}")
                    print(f"  Avg Liked Ingredients: {stats['avg_liked_ingredients']:.2f}")
                    print(f"  Avg Disliked Ingredients: {stats['avg_disliked_ingredients']:.2f}")
                    print(f"  Avg Age: {stats.get('avg_age', 0):.1f}")
                    print(f"  Nutrition RNI Fields: {len(stats.get('nutrition_rni_fields', []))} fields")
                    print(f"  Physiological States: {dict(stats['physiological_states'])}")
                    print(f"  Gender Distribution: {dict(stats['gender_distribution'])}")
                    print(f"  Instruction Types: {dict(stats['instruction_types'])}")

                # Task C specific
                if 'avg_violations_per_sample' in stats:
                    print(f"  Avg Violations per Sample: {stats['avg_violations_per_sample']:.2f}")
                    print(f"  Avg Corrections per Sample: {stats['avg_corrections_per_sample']:.2f}")
                    print(f"  Avg Ingredients Before: {stats['avg_ingredients_before']:.2f}")
                    print(f"  Avg Ingredients After: {stats['avg_ingredients_after']:.2f}")
                    print(f"  Avg Age: {stats.get('avg_age', 0):.1f}")
                    print(f"  Violation Types: {dict(stats['violation_types'])}")
                    print(f"  Violation Severity: {dict(stats['violation_severity'])}")
                    print(f"  Correction Actions: {dict(stats['correction_actions'])}")
                    print(f"  Physiological States: {dict(stats['physiological_states'])}")
                    print(f"  Gender Distribution: {dict(stats['gender_distribution'])}")
                    print(f"  Instruction Types: {dict(stats['instruction_types'])}")

        print("\n" + "=" * 80)

    def save_report(self, report: Dict[str, Any], output_path: str = "data_statistics_report.json"):
        """Save report to JSON file"""
        # Convert Counter and defaultdict to regular dicts for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, defaultdict):
                return dict(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_report = convert_to_serializable(report)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Report saved to: {output_path}")


def main():
    """Main execution"""
    stats = DataStatistics()
    report = stats.generate_report()
    stats.print_report(report)
    stats.save_report(report, "nutriplan_experiments/results/data_statistics_report.json")


if __name__ == "__main__":
    main()
