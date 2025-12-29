"""
RAG (Retrieval-Augmented Generation) Baseline for NutriPlan
Combines retrieval with LLM generation
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys

# Import retrieval baseline
from retrieval import RetrievalBaseline


class RAGBaseline:
    """RAG baseline combining retrieval and generation"""

    def __init__(
        self,
        model_name: str,
        recipe_corpus_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 2048,
        top_k_retrieval: int = 3
    ):
        """
        Args:
            model_name: Pretrained LLM name
            recipe_corpus_path: Recipe corpus for retrieval
            device: Device to run model on
            max_length: Maximum generation length
            top_k_retrieval: Number of recipes to retrieve for context
        """
        self.device = device
        self.max_length = max_length
        self.top_k_retrieval = top_k_retrieval

        # Load retrieval system
        print("Loading retrieval system...")
        self.retriever = RetrievalBaseline(recipe_corpus_path)

        # Load LLM
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

    def build_rag_prompt(
        self,
        task: str,
        constraints: Dict[str, Any],
        retrieved_recipes: List[Dict[str, Any]],
        original_recipe: Dict[str, Any] = None
    ) -> str:
        """
        Build RAG prompt with retrieved context

        Args:
            task: Task type ('b' for generation, 'c' for editing)
            constraints: User constraints
            retrieved_recipes: Retrieved similar recipes
            original_recipe: Original recipe (for Task C)

        Returns:
            Formatted prompt string
        """
        if task == 'b':
            return self._build_generation_prompt(constraints, retrieved_recipes)
        elif task == 'c':
            return self._build_editing_prompt(constraints, original_recipe, retrieved_recipes)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _build_generation_prompt(
        self,
        constraints: Dict[str, Any],
        retrieved_recipes: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Task B (generation)"""
        prompt = "### Task: Generate a recipe that satisfies the user's constraints.\n\n"

        # User constraints
        prompt += "### User Constraints:\n"
        prompt += f"Nutrition Targets: {json.dumps(constraints.get('nutrition_targets', {}))}\n"
        prompt += f"Dietary Restrictions: {constraints.get('dietary_restrictions', [])}\n"
        prompt += f"Allergies: {constraints.get('allergies', [])}\n"
        prompt += f"Cuisine Preference: {constraints.get('cuisine', 'Any')}\n"
        prompt += f"Cooking Time: {constraints.get('max_time', 'Flexible')}\n\n"

        # Retrieved recipes as context
        prompt += "### Similar Recipes (for inspiration):\n"
        for i, recipe in enumerate(retrieved_recipes):
            prompt += f"\n**Example {i+1}:**\n"
            prompt += f"Title: {recipe.get('title', 'Untitled')}\n"
            prompt += f"Ingredients: {json.dumps(recipe.get('ingredients', [])[:8])}\n"
            prompt += f"Nutrition: {json.dumps(recipe.get('nutrition', {}))}\n"
            prompt += f"Tags: {recipe.get('tags', [])}\n"

        prompt += "\n### Instruction:\n"
        prompt += "Based on the user constraints and inspired by the similar recipes above, "
        prompt += "generate a NEW recipe in JSON format with the following structure:\n"
        prompt += "{\n"
        prompt += '  "title": "Recipe Name",\n'
        prompt += '  "ingredients": [{"name": "ingredient", "quantity": "amount", "unit": "unit"}, ...],\n'
        prompt += '  "steps": ["step 1", "step 2", ...],\n'
        prompt += '  "nutrition": {"calories": X, "protein": Y, "sodium": Z, ...},\n'
        prompt += '  "tags": ["tag1", "tag2", ...]\n'
        prompt += "}\n\n"
        prompt += "### Generated Recipe:\n"

        return prompt

    def _build_editing_prompt(
        self,
        constraints: Dict[str, Any],
        original_recipe: Dict[str, Any],
        retrieved_recipes: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for Task C (editing)"""
        prompt = "### Task: Edit the recipe to satisfy constraints while minimizing changes.\n\n"

        # Original recipe
        prompt += "### Original Recipe:\n"
        prompt += f"Title: {original_recipe.get('title', 'Untitled')}\n"
        prompt += f"Ingredients: {json.dumps(original_recipe.get('ingredients', []))}\n"
        prompt += f"Steps: {json.dumps(original_recipe.get('steps', []))}\n"
        prompt += f"Nutrition: {json.dumps(original_recipe.get('nutrition', {}))}\n\n"

        # Constraints
        prompt += "### Required Constraints:\n"
        prompt += f"Nutrition Targets: {json.dumps(constraints.get('nutrition_targets', {}))}\n"
        prompt += f"Dietary Restrictions: {constraints.get('dietary_restrictions', [])}\n"
        prompt += f"Allergies: {constraints.get('allergies', [])}\n\n"

        # Similar compliant recipes
        prompt += "### Similar Compliant Recipes (for guidance):\n"
        for i, recipe in enumerate(retrieved_recipes[:2]):  # Use fewer examples for editing
            prompt += f"\n**Example {i+1}:**\n"
            prompt += f"Title: {recipe.get('title', 'Untitled')}\n"
            prompt += f"Ingredients: {json.dumps(recipe.get('ingredients', [])[:5])}\n"
            prompt += f"Nutrition: {json.dumps(recipe.get('nutrition', {}))}\n"

        prompt += "\n### Instruction:\n"
        prompt += "Edit the original recipe to satisfy all constraints. Make MINIMAL changes to preserve the original flavor.\n"
        prompt += "Output format: JSON with structure:\n"
        prompt += "{\n"
        prompt += '  "edited_recipe": {...},\n'
        prompt += '  "changes_made": ["change 1", "change 2", ...]\n'
        prompt += "}\n\n"
        prompt += "### Edited Recipe:\n"

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
        Generate recipe using RAG

        Args:
            task: Task type ('b' or 'c')
            constraints: User constraints
            original_recipe: Original recipe (for Task C)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated recipe
        """
        # Retrieve similar recipes
        user_profile = constraints.copy()
        retrieved_recipes = self.retriever.retrieve(
            user_profile,
            top_k=self.top_k_retrieval
        )

        # Build prompt
        prompt = self.build_rag_prompt(task, constraints, retrieved_recipes, original_recipe)

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


def evaluate_rag_baseline(
    test_file: str,
    recipe_corpus_path: str,
    model_name: str,
    output_file: str,
    task: str = 'b'
):
    """
    Evaluate RAG baseline on test set

    Args:
        test_file: Test set JSONL file
        recipe_corpus_path: Recipe corpus path
        model_name: LLM model name
        output_file: Output predictions file
        task: Task type ('b' or 'c')
    """
    # Load RAG system
    rag = RAGBaseline(
        model_name=model_name,
        recipe_corpus_path=recipe_corpus_path
    )

    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    # Run generation
    predictions = []
    for sample in tqdm(test_data, desc=f"Running RAG (Task {task.upper()})"):
        constraints = sample.get('constraints', {})
        original_recipe = sample.get('original_recipe') if task == 'c' else None

        generated = rag.generate(
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

    print(f"✅ RAG predictions saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Baseline for NutriPlan")
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test file (Task B or C)')
    parser.add_argument('--recipe_corpus', type=str, required=True,
                        help='Recipe corpus JSONL file')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B',
                        help='LLM model name')
    parser.add_argument('--output_file', type=str, default='rag_predictions.jsonl',
                        help='Output predictions file')
    parser.add_argument('--task', type=str, choices=['b', 'c'], default='b',
                        help='Task type (b: generation, c: editing)')

    args = parser.parse_args()

    evaluate_rag_baseline(
        test_file=args.test_file,
        recipe_corpus_path=args.recipe_corpus,
        model_name=args.model_name,
        output_file=args.output_file,
        task=args.task
    )
