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


class NutriPlanDatasetV2(Dataset):
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
        self.max_length = max_length
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
        """Get a single sample"""
        sample = self.samples[idx]
        task = sample['task']

        if task == 'a':
            return self._format_task_a(sample)
        elif task == 'b':
            return self._format_task_b(sample)
        else:
            return self._format_task_c(sample)

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
            truncation=False,  # Don't truncate yet
            padding=False,
            return_tensors='pt'
        )
        prompt_ids = prompt_encoding['input_ids'].squeeze(0)

        # Tokenize target (loss computed on these tokens)
        target_encoding = self.tokenizer(
            target,
            add_special_tokens=False,  # No special tokens between prompt and target
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        target_ids = target_encoding['input_ids'].squeeze(0)

        # Concatenate prompt + target
        full_ids = torch.cat([prompt_ids, target_ids], dim=0)

        # Truncate if needed
        if len(full_ids) > self.max_length:
            # Truncate from the end (preserve prompt, truncate target if needed)
            full_ids = full_ids[:self.max_length]

        # Create labels: -100 for prompt, actual ids for target
        labels = torch.full_like(full_ids, -100)
        # Only set labels for target tokens (after prompt)
        prompt_len = min(len(prompt_ids), self.max_length)
        target_start = prompt_len
        target_end = min(len(full_ids), prompt_len + len(target_ids))
        labels[target_start:target_end] = full_ids[target_start:target_end]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(full_ids)

        # Pad to max_length
        current_len = len(full_ids)
        if current_len < self.max_length:
            padding_len = self.max_length - current_len
            full_ids = torch.cat([full_ids, torch.full((padding_len,), self.tokenizer.pad_token_id)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_len)])
            labels = torch.cat([labels, torch.full((padding_len,), -100)])

        return {
            'input_ids': full_ids.clone(),
            'attention_mask': attention_mask.clone(),
            'labels': labels.clone()
        }

    def _format_task_a(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Format Task A (Discriminative Ranking) sample"""
        user_profile = sample.get('user_profile', {})
        candidates = sample.get('candidates', [])
        ranking = sample.get('ranking', [])

        prompt = self._build_task_a_prompt(user_profile, candidates)
        target = self._build_task_a_target(ranking)

        return self._format_with_prompt_masking(prompt, target)

    def _format_task_b(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Format Task B (Constrained Generation) sample"""
        constraints = sample.get('constraints', {})
        recipe = sample.get('recipe', {})
        kg_context = sample.get('kg_context', {})

        prompt = self._build_task_b_prompt(constraints, kg_context)
        target = self._build_task_b_target(recipe)

        return self._format_with_prompt_masking(prompt, target)

    def _format_task_c(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Format Task C (Reflective Editing) sample"""
        original_recipe = sample.get('original_recipe', {})
        constraints = sample.get('constraints', {})
        violations = sample.get('violations', [])
        edited_recipe = sample.get('edited_recipe', {})

        prompt = self._build_task_c_prompt(original_recipe, constraints, violations)
        target = self._build_task_c_target(edited_recipe, violations)

        return self._format_with_prompt_masking(prompt, target)

    # Build prompt/target methods (same as original)
    def _build_task_a_prompt(self, user_profile: Dict, candidates: List[Dict]) -> str:
        prompt = "### Task: Rank the following recipes based on user preferences and constraints.\n\n"
        prompt += "### User Profile:\n"
        constraints = user_profile.get('constraints', {})
        prompt += f"Nutrition Goals: {json.dumps(constraints.get('nutrition_targets', {}))}\n"
        prompt += f"Dietary Restrictions: {constraints.get('dietary_restrictions', [])}\n"
        prompt += f"Allergies: {constraints.get('allergies', [])}\n"
        prompt += f"Preferences: {constraints.get('preferences', {})}\n\n"
        prompt += "### Candidate Recipes:\n"
        for i, candidate in enumerate(candidates):
            prompt += f"{i+1}. {candidate.get('title', 'Untitled')}\n"
            prompt += f"   Ingredients: {', '.join(candidate.get('ingredients', [])[:5])}...\n"
            prompt += f"   Nutrition: {json.dumps(candidate.get('nutrition', {}))}\n\n"
        prompt += "### Instruction: Rank these recipes from most to least suitable. Output format: [recipe_id_1, recipe_id_2, ...]\n"
        prompt += "### Ranking:\n"
        return prompt

    def _build_task_a_target(self, ranking: List[int]) -> str:
        return json.dumps(ranking)

    def _build_task_b_prompt(self, constraints: Dict, kg_context: Dict) -> str:
        prompt = "### Task: Generate a recipe that satisfies the following constraints.\n\n"
        prompt += "### Constraints:\n"
        prompt += f"Nutrition Targets: {json.dumps(constraints.get('nutrition_targets', {}))}\n"
        prompt += f"Dietary Restrictions: {constraints.get('dietary_restrictions', [])}\n"
        prompt += f"Allergies: {constraints.get('allergies', [])}\n"
        prompt += f"Cuisine Preference: {constraints.get('cuisine', 'Any')}\n"
        prompt += f"Cooking Time: {constraints.get('max_time', 'Flexible')}\n\n"
        if kg_context:
            prompt += "### Knowledge Graph Guidance:\n"
            if 'recommended_ingredients' in kg_context:
                prompt += f"Recommended Ingredients: {', '.join(kg_context['recommended_ingredients'][:10])}\n"
            if 'cooking_rules' in kg_context:
                prompt += f"Cooking Tips: {'; '.join(kg_context['cooking_rules'][:3])}\n"
            prompt += "\n"
        prompt += "### Instruction: Generate a complete recipe in JSON format with fields: title, ingredients (list of {name, quantity, unit}), steps (list), nutrition (dict).\n"
        prompt += "### Recipe:\n"
        return prompt

    def _build_task_b_target(self, recipe: Dict) -> str:
        return json.dumps(recipe, ensure_ascii=False)

    def _build_task_c_prompt(self, original_recipe: Dict, constraints: Dict, violations: List[Dict]) -> str:
        prompt = "### Task: Edit the following recipe to fix constraint violations while minimizing changes.\n\n"
        prompt += "### Original Recipe:\n"
        prompt += f"Title: {original_recipe.get('title', 'Untitled')}\n"
        prompt += f"Ingredients: {json.dumps(original_recipe.get('ingredients', []))}\n"
        prompt += f"Steps: {json.dumps(original_recipe.get('steps', []))}\n"
        prompt += f"Nutrition: {json.dumps(original_recipe.get('nutrition', {}))}\n\n"
        prompt += "### Required Constraints:\n"
        prompt += f"Nutrition Targets: {json.dumps(constraints.get('nutrition_targets', {}))}\n"
        prompt += f"Dietary Restrictions: {constraints.get('dietary_restrictions', [])}\n"
        prompt += f"Allergies: {constraints.get('allergies', [])}\n\n"
        prompt += "### Detected Violations:\n"
        for i, violation in enumerate(violations):
            prompt += f"{i+1}. {violation.get('type', 'Unknown')}: {violation.get('description', '')}\n"
        prompt += "\n"
        prompt += "### Instruction: Edit the recipe to fix all violations. Output format: {\"edited_recipe\": {...}, \"changes_made\": [...]}\n"
        prompt += "### Edited Recipe:\n"
        return prompt

    def _build_task_c_target(self, edited_recipe: Dict, violations: List[Dict]) -> str:
        output = {
            "edited_recipe": edited_recipe,
            "changes_made": [v.get('fix_description', '') for v in violations]
        }
        return json.dumps(output, ensure_ascii=False)
