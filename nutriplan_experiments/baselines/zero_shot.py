"""
Zero-shot LLM Baseline for NutriPlan
Directly prompts LLM without fine-tuning
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse


class ZeroShotBaseline:
    """Zero-shot baseline using prompt engineering"""

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 2048
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
            max_length: Maximum generation length
        """
        self.device = device
        self.max_length = max_length

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def build_prompt(
        self,
        task: str,
        constraints: Dict[str, Any],
        original_recipe: Dict[str, Any] = None
    ) -> str:
        """
        Build zero-shot prompt

        Args:
            task: Task type ('b' for generation, 'c' for editing)
            constraints: User constraints
            original_recipe: Original recipe (for Task C)

        Returns:
            Formatted prompt string
        """
        if task == 'b':
            return self._build_generation_prompt(constraints)
        elif task == 'c':
            return self._build_editing_prompt(constraints, original_recipe)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _build_generation_prompt(self, constraints: Dict[str, Any]) -> str:
        """Build prompt for Task B (generation)"""
        prompt = "You are an expert nutritionist and chef. Generate a healthy recipe that satisfies the following requirements.\n\n"

        # User constraints
        prompt += "## Requirements:\n"

        # Nutrition targets
        nutrition_targets = constraints.get('nutrition_targets', {})
        if nutrition_targets:
            prompt += "### Nutrition Targets (per serving):\n"
            for nutrient, target in nutrition_targets.items():
                if isinstance(target, dict):
                    value = target.get('value')
                    constraint_type = target.get('type', 'max')
                    prompt += f"- {nutrient}: {constraint_type} {value}\n"
                else:
                    prompt += f"- {nutrient}: max {target}\n"

        # Dietary restrictions
        dietary_restrictions = constraints.get('dietary_restrictions', [])
        if dietary_restrictions:
            prompt += f"\n### Dietary Restrictions: {', '.join(dietary_restrictions)}\n"

        # Allergies
        allergies = constraints.get('allergies', [])
        if allergies:
            prompt += f"\n### Allergies (must avoid): {', '.join(allergies)}\n"

        # Cuisine preference
        cuisine = constraints.get('cuisine')
        if cuisine:
            prompt += f"\n### Cuisine: {cuisine}\n"

        # Cooking time
        max_time = constraints.get('max_time')
        if max_time:
            prompt += f"\n### Maximum Cooking Time: {max_time} minutes\n"

        # Output format
        prompt += "\n## Output Format:\n"
        prompt += "Please generate a recipe in JSON format with the following structure:\n"
        prompt += "```json\n"
        prompt += "{\n"
        prompt += '  "title": "Recipe Name",\n'
        prompt += '  "ingredients": [\n'
        prompt += '    {"name": "ingredient1", "quantity": "amount", "unit": "unit"},\n'
        prompt += '    ...\n'
        prompt += '  ],\n'
        prompt += '  "steps": ["step 1", "step 2", ...],\n'
        prompt += '  "servings": 4,\n'
        prompt += '  "nutrition": {\n'
        prompt += '    "calories": X,\n'
        prompt += '    "protein": Y,\n'
        prompt += '    "sodium": Z,\n'
        prompt += '    ...\n'
        prompt += '  }\n'
        prompt += "}\n"
        prompt += "```\n\n"
        prompt += "## Recipe:\n"

        return prompt

    def _build_editing_prompt(
        self,
        constraints: Dict[str, Any],
        original_recipe: Dict[str, Any]
    ) -> str:
        """Build prompt for Task C (editing)"""
        prompt = "You are an expert nutritionist and chef. Edit the following recipe to satisfy the given constraints while making minimal changes to preserve the original flavor.\n\n"

        # Original recipe
        prompt += "## Original Recipe:\n"
        prompt += f"**Title:** {original_recipe.get('title', 'Untitled')}\n\n"

        prompt += "**Ingredients:**\n"
        for ing in original_recipe.get('ingredients', []):
            if isinstance(ing, dict):
                prompt += f"- {ing.get('quantity', '')} {ing.get('unit', '')} {ing.get('name', '')}\n"
            else:
                prompt += f"- {ing}\n"

        prompt += "\n**Steps:**\n"
        for i, step in enumerate(original_recipe.get('steps', []), 1):
            prompt += f"{i}. {step}\n"

        prompt += f"\n**Servings:** {original_recipe.get('servings', 4)}\n"

        # Current nutrition
        nutrition = original_recipe.get('nutrition', {})
        if nutrition:
            prompt += "\n**Current Nutrition (per serving):**\n"
            for nutrient, value in nutrition.items():
                prompt += f"- {nutrient}: {value}\n"

        # Required constraints
        prompt += "\n## Required Constraints:\n"

        nutrition_targets = constraints.get('nutrition_targets', {})
        if nutrition_targets:
            prompt += "### Nutrition Targets (per serving):\n"
            for nutrient, target in nutrition_targets.items():
                if isinstance(target, dict):
                    value = target.get('value')
                    constraint_type = target.get('type', 'max')
                    prompt += f"- {nutrient}: {constraint_type} {value}\n"
                else:
                    prompt += f"- {nutrient}: max {target}\n"

        dietary_restrictions = constraints.get('dietary_restrictions', [])
        if dietary_restrictions:
            prompt += f"\n### Dietary Restrictions: {', '.join(dietary_restrictions)}\n"

        allergies = constraints.get('allergies', [])
        if allergies:
            prompt += f"\n### Allergies (must avoid): {', '.join(allergies)}\n"

        # Instructions
        prompt += "\n## Task:\n"
        prompt += "1. Identify which constraints are violated\n"
        prompt += "2. Make MINIMAL edits to fix violations\n"
        prompt += "3. Preserve the original flavor as much as possible\n\n"

        prompt += "## Output Format:\n"
        prompt += "Please output in JSON format:\n"
        prompt += "```json\n"
        prompt += "{\n"
        prompt += '  "violations": ["violation 1", "violation 2", ...],\n'
        prompt += '  "edited_recipe": {\n'
        prompt += '    "title": "...",\n'
        prompt += '    "ingredients": [...],\n'
        prompt += '    "steps": [...],\n'
        prompt += '    "servings": 4,\n'
        prompt += '    "nutrition": {...}\n'
        prompt += '  },\n'
        prompt += '  "changes_made": ["change 1", "change 2", ...]\n'
        prompt += "}\n"
        prompt += "```\n\n"
        prompt += "## Edited Recipe:\n"

        return prompt

    def generate(
        self,
        task: str,
        constraints: Dict[str, Any],
        original_recipe: Dict[str, Any] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate recipe using zero-shot prompting

        Args:
            task: Task type ('b' or 'c')
            constraints: User constraints
            original_recipe: Original recipe (for Task C)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated recipe
        """
        # Build prompt
        prompt = self.build_prompt(task, constraints, original_recipe)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Parse JSON output
        try:
            # Try to extract JSON from generated text
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                generated_recipe = json.loads(json_str)
            else:
                # Fallback: return raw text
                generated_recipe = {'raw_output': generated_text}
        except json.JSONDecodeError:
            generated_recipe = {'raw_output': generated_text}

        return generated_recipe


def evaluate_zero_shot_baseline(
    test_file: str,
    model_name: str,
    output_file: str,
    task: str = 'b'
):
    """
    Evaluate zero-shot baseline on test set

    Args:
        test_file: Test set JSONL file
        model_name: LLM model name
        output_file: Output predictions file
        task: Task type ('b' or 'c')
    """
    # Load zero-shot system
    zero_shot = ZeroShotBaseline(model_name=model_name)

    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    # Run generation
    predictions = []
    for sample in tqdm(test_data, desc=f"Running Zero-shot (Task {task.upper()})"):
        constraints = sample.get('constraints', {})
        original_recipe = sample.get('original_recipe') if task == 'c' else None

        generated = zero_shot.generate(
            task=task,
            constraints=constraints,
            original_recipe=original_recipe
        )

        predictions.append({
            'constraints': constraints,
            'original_recipe': original_recipe,
            'generated_recipe': generated
        })

    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"✅ Zero-shot predictions saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot Baseline for NutriPlan")
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test file (Task B or C)')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B',
                        help='LLM model name')
    parser.add_argument('--output_file', type=str, default='zero_shot_predictions.jsonl',
                        help='Output predictions file')
    parser.add_argument('--task', type=str, choices=['b', 'c'], default='b',
                        help='Task type (b: generation, c: editing)')

    args = parser.parse_args()

    evaluate_zero_shot_baseline(
        test_file=args.test_file,
        model_name=args.model_name,
        output_file=args.output_file,
        task=args.task
    )
