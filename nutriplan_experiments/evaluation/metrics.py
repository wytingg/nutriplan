"""
Evaluation Metrics for NutriPlan
Implements all 8 metrics: SNCR, UPM, K-Faith, AVC, Dist-2, BLEU, ROUGE-L, Nutrition Accuracy
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
import json
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


class NutriPlanMetrics:
    """Comprehensive metrics for NutriPlan evaluation"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()

    # ==================== Core Metrics ====================

    def compute_sncr(
        self,
        generated_recipe: Dict[str, Any],
        constraints: Dict[str, Any],
        tolerance: float = 0.1
    ) -> float:
        """
        Strict Nutrition Constraint Recall (SNCR)
        Measures the percentage of nutrition constraints satisfied

        Args:
            generated_recipe: Generated recipe with 'nutrition' field
            constraints: User constraints with 'nutrition_targets' field
            tolerance: Tolerance for constraint satisfaction (default 10%)

        Returns:
            SNCR score [0, 1]
        """
        nutrition_targets = constraints.get('nutrition_targets', {})
        if not nutrition_targets:
            return 1.0  # No constraints to satisfy

        generated_nutrition = generated_recipe.get('nutrition', {})
        satisfied_count = 0
        total_constraints = 0

        for nutrient, target in nutrition_targets.items():
            total_constraints += 1

            if nutrient not in generated_nutrition:
                continue

            actual = generated_nutrition[nutrient]
            target_value = target.get('value', target) if isinstance(target, dict) else target
            constraint_type = target.get('type', 'max') if isinstance(target, dict) else 'max'

            # Check constraint satisfaction
            if constraint_type == 'max':
                if actual <= target_value * (1 + tolerance):
                    satisfied_count += 1
            elif constraint_type == 'min':
                if actual >= target_value * (1 - tolerance):
                    satisfied_count += 1
            elif constraint_type == 'range':
                min_val = target.get('min', 0)
                max_val = target.get('max', float('inf'))
                if min_val * (1 - tolerance) <= actual <= max_val * (1 + tolerance):
                    satisfied_count += 1

        return satisfied_count / total_constraints if total_constraints > 0 else 1.0

    def compute_upm(
        self,
        generated_recipe: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> float:
        """
        User Preference Matching (UPM)
        Measures how well the recipe matches user preferences

        Args:
            generated_recipe: Generated recipe
            user_preferences: User preferences (cuisine, ingredients, cooking_time, etc.)

        Returns:
            UPM score [0, 1]
        """
        match_scores = []

        # Cuisine preference
        if 'cuisine' in user_preferences:
            preferred_cuisine = user_preferences['cuisine'].lower()
            recipe_cuisine = generated_recipe.get('cuisine', '').lower()
            match_scores.append(1.0 if preferred_cuisine in recipe_cuisine else 0.0)

        # Ingredient preferences
        if 'liked_ingredients' in user_preferences:
            liked = set(user_preferences['liked_ingredients'])
            recipe_ingredients = set([ing.get('name', ing) if isinstance(ing, dict) else ing
                                     for ing in generated_recipe.get('ingredients', [])])
            overlap = len(liked & recipe_ingredients)
            match_scores.append(overlap / len(liked) if liked else 1.0)

        # Disliked ingredients (penalty)
        if 'disliked_ingredients' in user_preferences:
            disliked = set(user_preferences['disliked_ingredients'])
            recipe_ingredients = set([ing.get('name', ing) if isinstance(ing, dict) else ing
                                     for ing in generated_recipe.get('ingredients', [])])
            violations = len(disliked & recipe_ingredients)
            match_scores.append(1.0 - (violations / len(recipe_ingredients)) if recipe_ingredients else 1.0)

        # Cooking time preference
        if 'max_cooking_time' in user_preferences:
            max_time = user_preferences['max_cooking_time']
            recipe_time = generated_recipe.get('cooking_time', 0)
            match_scores.append(1.0 if recipe_time <= max_time else max(0, 1 - (recipe_time - max_time) / max_time))

        # Dietary restrictions
        if 'dietary_restrictions' in user_preferences:
            restrictions = set(user_preferences['dietary_restrictions'])
            recipe_tags = set(generated_recipe.get('tags', []))
            violations = restrictions - recipe_tags
            match_scores.append(1.0 - len(violations) / len(restrictions) if restrictions else 1.0)

        return np.mean(match_scores) if match_scores else 0.5

    def compute_k_faith(
        self,
        generated_recipe: Dict[str, Any],
        kg_facts: List[Dict[str, Any]]
    ) -> float:
        """
        Knowledge Graph Faithfulness (K-Faith)
        Measures how well the recipe adheres to KG knowledge

        Args:
            generated_recipe: Generated recipe
            kg_facts: Relevant KG facts (ingredient compatibility, cooking rules, etc.)

        Returns:
            K-Faith score [0, 1]
        """
        if not kg_facts:
            return 1.0

        faithfulness_scores = []
        recipe_ingredients = set([
            self._normalize_ingredient(ing.get('name', ing) if isinstance(ing, dict) else ing)
            for ing in generated_recipe.get('ingredients', [])
        ])

        for fact in kg_facts:
            fact_type = fact.get('type', 'unknown')

            if fact_type == 'ingredient_compatibility':
                # Check if compatible ingredients are used together
                compatible_pair = (
                    self._normalize_ingredient(fact['ingredient1']),
                    self._normalize_ingredient(fact['ingredient2'])
                )
                if compatible_pair[0] in recipe_ingredients and compatible_pair[1] in recipe_ingredients:
                    faithfulness_scores.append(1.0)
                elif compatible_pair[0] in recipe_ingredients or compatible_pair[1] in recipe_ingredients:
                    faithfulness_scores.append(0.5)

            elif fact_type == 'cooking_rule':
                # Check if cooking rule is followed
                rule = fact.get('rule', '')
                steps = ' '.join(generated_recipe.get('steps', []))
                if any(keyword in steps.lower() for keyword in rule.lower().split()):
                    faithfulness_scores.append(1.0)
                else:
                    faithfulness_scores.append(0.0)

            elif fact_type == 'nutrition_rule':
                # Check if nutrition combination rule is followed
                nutrient1 = fact.get('nutrient1')
                nutrient2 = fact.get('nutrient2')
                relation = fact.get('relation', 'synergy')
                nutrition = generated_recipe.get('nutrition', {})

                if nutrient1 in nutrition and nutrient2 in nutrition:
                    if relation == 'synergy' and nutrition[nutrient1] > 0 and nutrition[nutrient2] > 0:
                        faithfulness_scores.append(1.0)
                    else:
                        faithfulness_scores.append(0.5)

        return np.mean(faithfulness_scores) if faithfulness_scores else 1.0

    def compute_avc(
        self,
        generated_recipe: Dict[str, Any],
        constraints: Dict[str, Any],
        tolerance: float = 0.1
    ) -> float:
        """
        Average Violation Count (AVC)
        Counts the average number of constraint violations per recipe

        Args:
            generated_recipe: Generated recipe
            constraints: User constraints
            tolerance: Tolerance for constraint satisfaction

        Returns:
            Average number of violations
        """
        violations = 0

        # Nutrition violations
        nutrition_targets = constraints.get('nutrition_targets', {})
        generated_nutrition = generated_recipe.get('nutrition', {})

        for nutrient, target in nutrition_targets.items():
            if nutrient not in generated_nutrition:
                violations += 1
                continue

            actual = generated_nutrition[nutrient]
            target_value = target.get('value', target) if isinstance(target, dict) else target
            constraint_type = target.get('type', 'max') if isinstance(target, dict) else 'max'

            if constraint_type == 'max' and actual > target_value * (1 + tolerance):
                violations += 1
            elif constraint_type == 'min' and actual < target_value * (1 - tolerance):
                violations += 1
            elif constraint_type == 'range':
                min_val = target.get('min', 0)
                max_val = target.get('max', float('inf'))
                if not (min_val * (1 - tolerance) <= actual <= max_val * (1 + tolerance)):
                    violations += 1

        # Allergy violations
        allergies = set(constraints.get('allergies', []))
        recipe_ingredients = set([
            self._normalize_ingredient(ing.get('name', ing) if isinstance(ing, dict) else ing)
            for ing in generated_recipe.get('ingredients', [])
        ])

        for allergy in allergies:
            if self._normalize_ingredient(allergy) in recipe_ingredients:
                violations += 1

        # Dietary restriction violations
        dietary_restrictions = set(constraints.get('dietary_restrictions', []))
        recipe_tags = set(generated_recipe.get('tags', []))

        for restriction in dietary_restrictions:
            if restriction.lower() not in [tag.lower() for tag in recipe_tags]:
                violations += 1

        return violations

    def compute_dist_2(self, generated_texts: List[str]) -> float:
        """
        Distinct-2 (Dist-2)
        Measures generation diversity using unique bigram ratio

        Args:
            generated_texts: List of generated text (e.g., recipe titles or descriptions)

        Returns:
            Dist-2 score [0, 1]
        """
        all_bigrams = []

        for text in generated_texts:
            tokens = text.lower().split()
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            all_bigrams.extend(bigrams)

        if not all_bigrams:
            return 0.0

        unique_bigrams = len(set(all_bigrams))
        total_bigrams = len(all_bigrams)

        return unique_bigrams / total_bigrams

    def compute_bleu(
        self,
        generated_text: str,
        reference_text: str,
        max_n: int = 4
    ) -> float:
        """
        BLEU Score (average of BLEU-1 to BLEU-4)
        Measures text quality against reference

        Args:
            generated_text: Generated text
            reference_text: Reference text
            max_n: Maximum n-gram (default 4)

        Returns:
            BLEU score [0, 1]
        """
        generated_tokens = generated_text.lower().split()
        reference_tokens = reference_text.lower().split()

        weights = tuple([1.0/max_n] * max_n)

        try:
            score = sentence_bleu(
                [reference_tokens],
                generated_tokens,
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
        except:
            score = 0.0

        return score

    def compute_bleu_n(
        self,
        generated_text: str,
        reference_text: str
    ) -> Dict[str, float]:
        """
        Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 separately

        Args:
            generated_text: Generated text
            reference_text: Reference text

        Returns:
            Dictionary with bleu_1, bleu_2, bleu_3, bleu_4 scores
        """
        generated_tokens = generated_text.lower().split()
        reference_tokens = reference_text.lower().split()

        bleu_scores = {}

        for n in range(1, 5):
            # weights for BLEU-n: only the n-th position is 1.0
            weights = tuple([1.0 if i == n-1 else 0.0 for i in range(4)])

            try:
                score = sentence_bleu(
                    [reference_tokens],
                    generated_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
            except:
                score = 0.0

            bleu_scores[f'bleu_{n}'] = score

        return bleu_scores

    def compute_rouge_l(
        self,
        generated_text: str,
        reference_text: str
    ) -> float:
        """
        ROUGE-L Score
        Measures longest common subsequence

        Args:
            generated_text: Generated text
            reference_text: Reference text

        Returns:
            ROUGE-L F1 score [0, 1]
        """
        scores = self.rouge_scorer.score(reference_text, generated_text)
        return scores['rougeL'].fmeasure

    def compute_nutrition_accuracy(
        self,
        generated_nutrition: Dict[str, float],
        reference_nutrition: Dict[str, float],
        tolerance: float = 0.15
    ) -> float:
        """
        Nutrition Accuracy
        Measures how accurately nutrition values are calculated

        Args:
            generated_nutrition: Generated nutrition values
            reference_nutrition: Reference nutrition values
            tolerance: Acceptable error margin (default 15%)

        Returns:
            Accuracy score [0, 1]
        """
        if not reference_nutrition:
            return 1.0

        accurate_count = 0
        total_nutrients = 0

        for nutrient, ref_value in reference_nutrition.items():
            total_nutrients += 1

            if nutrient not in generated_nutrition:
                continue

            gen_value = generated_nutrition[nutrient]

            # Check if within tolerance
            if ref_value == 0:
                if gen_value == 0:
                    accurate_count += 1
            else:
                relative_error = abs(gen_value - ref_value) / ref_value
                if relative_error <= tolerance:
                    accurate_count += 1

        return accurate_count / total_nutrients if total_nutrients > 0 else 1.0

    # ==================== Helper Methods ====================

    def _normalize_ingredient(self, ingredient: str) -> str:
        """Normalize ingredient name for comparison"""
        # Remove quantities, units, and extra spaces
        ingredient = re.sub(r'\d+', '', ingredient)
        ingredient = re.sub(r'\b(cup|tbsp|tsp|oz|lb|g|kg|ml|l)\b', '', ingredient, flags=re.IGNORECASE)
        ingredient = ' '.join(ingredient.split())
        return ingredient.lower().strip()

    def compute_all_metrics(
        self,
        generated_recipe: Dict[str, Any],
        reference_recipe: Dict[str, Any],
        constraints: Dict[str, Any],
        kg_facts: List[Dict[str, Any]],
        all_generated_texts: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single recipe

        Returns:
            Dictionary with all metric scores
        """
        metrics = {}

        # Core metrics
        metrics['sncr'] = self.compute_sncr(generated_recipe, constraints)
        metrics['upm'] = self.compute_upm(generated_recipe, constraints)
        metrics['k_faith'] = self.compute_k_faith(generated_recipe, kg_facts)
        metrics['avc'] = self.compute_avc(generated_recipe, constraints)

        # Text quality metrics
        gen_text = json.dumps(generated_recipe, ensure_ascii=False)
        ref_text = json.dumps(reference_recipe, ensure_ascii=False)

        # BLEU-N scores (individual BLEU-1, BLEU-2, BLEU-3, BLEU-4)
        bleu_n_scores = self.compute_bleu_n(gen_text, ref_text)
        metrics.update(bleu_n_scores)  # Adds bleu_1, bleu_2, bleu_3, bleu_4

        # Average BLEU (for backward compatibility)
        metrics['bleu'] = self.compute_bleu(gen_text, ref_text)

        metrics['rouge_l'] = self.compute_rouge_l(gen_text, ref_text)

        # Nutrition accuracy
        metrics['nutrition_accuracy'] = self.compute_nutrition_accuracy(
            generated_recipe.get('nutrition', {}),
            reference_recipe.get('nutrition', {})
        )

        # Dist-2 (computed across all generated texts)
        if all_generated_texts:
            metrics['dist_2'] = self.compute_dist_2(all_generated_texts)
        else:
            metrics['dist_2'] = 0.0

        return metrics


# ==================== Batch Evaluation ====================

def evaluate_batch(
    generated_recipes: List[Dict[str, Any]],
    reference_recipes: List[Dict[str, Any]],
    constraints_list: List[Dict[str, Any]],
    kg_facts_list: List[List[Dict[str, Any]]]
) -> Dict[str, float]:
    """
    Evaluate a batch of generated recipes

    Returns:
        Dictionary with average metric scores
    """
    metrics_calculator = NutriPlanMetrics()

    all_metrics = []
    all_generated_texts = [json.dumps(recipe, ensure_ascii=False) for recipe in generated_recipes]

    for i, (gen_recipe, ref_recipe, constraints, kg_facts) in enumerate(
        zip(generated_recipes, reference_recipes, constraints_list, kg_facts_list)
    ):
        metrics = metrics_calculator.compute_all_metrics(
            gen_recipe,
            ref_recipe,
            constraints,
            kg_facts,
            all_generated_texts
        )
        all_metrics.append(metrics)

    # Compute averages
    avg_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        avg_metrics[f'{metric_name}_mean'] = np.mean(values)
        avg_metrics[f'{metric_name}_std'] = np.std(values)

    return avg_metrics


if __name__ == "__main__":
    # Test metrics
    metrics = NutriPlanMetrics()

    # Example data
    generated = {
        "title": "Grilled Chicken Salad",
        "ingredients": ["chicken breast", "lettuce", "tomato", "olive oil"],
        "nutrition": {"calories": 350, "protein": 35, "sodium": 450},
        "steps": ["Grill chicken", "Chop vegetables", "Mix with olive oil"],
        "tags": ["low-carb", "high-protein"]
    }

    constraints = {
        "nutrition_targets": {"calories": {"value": 400, "type": "max"}, "sodium": {"value": 500, "type": "max"}},
        "allergies": [],
        "dietary_restrictions": ["low-carb"]
    }

    kg_facts = [
        {"type": "ingredient_compatibility", "ingredient1": "chicken", "ingredient2": "lettuce"},
        {"type": "cooking_rule", "rule": "grill meat for better flavor"}
    ]

    # Compute all metrics
    all_metrics = metrics.compute_all_metrics(generated, generated, constraints, kg_facts)

    print("Test Metrics:")
    for metric, value in all_metrics.items():
        print(f"  {metric}: {value:.4f}")
