"""
Retrieval Baseline for NutriPlan
BM25 + User Profile Similarity for Recipe Ranking
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import math
from tqdm import tqdm


class BM25Retriever:
    """BM25 algorithm for recipe retrieval"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = Counter()
        self.idf = {}
        self.N = 0

    def fit(self, documents: List[str]):
        """Fit BM25 on corpus"""
        self.corpus = documents
        self.N = len(documents)

        # Compute document lengths
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / self.N

        # Compute document frequencies
        for doc in documents:
            tokens = set(doc.lower().split())
            for token in tokens:
                self.doc_freqs[token] += 1

        # Compute IDF
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for query and document"""
        query_tokens = query.lower().split()
        doc = self.corpus[doc_idx]
        doc_tokens = doc.lower().split()
        doc_length = self.doc_lengths[doc_idx]

        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)

        score = 0
        for token in query_tokens:
            if token not in self.idf:
                continue

            tf = doc_term_freqs.get(token, 0)
            idf = self.idf[token]

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def rank(self, query: str, top_k: int = 10) -> List[int]:
        """Rank documents by BM25 score"""
        scores = [(idx, self.score(query, idx)) for idx in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:top_k]]


class UserProfileSimilarity:
    """Compute similarity between recipe and user profile"""

    def __init__(self):
        pass

    def compute_nutrition_similarity(
        self,
        recipe_nutrition: Dict[str, float],
        target_nutrition: Dict[str, float]
    ) -> float:
        """Compute nutrition similarity"""
        if not target_nutrition:
            return 1.0

        similarities = []
        for nutrient, target in target_nutrition.items():
            if nutrient not in recipe_nutrition:
                similarities.append(0.0)
                continue

            actual = recipe_nutrition[nutrient]
            target_value = target.get('value', target) if isinstance(target, dict) else target

            # Normalize by target value
            if target_value == 0:
                sim = 1.0 if actual == 0 else 0.0
            else:
                error = abs(actual - target_value) / target_value
                sim = max(0, 1 - error)

            similarities.append(sim)

        return np.mean(similarities) if similarities else 1.0

    def compute_preference_similarity(
        self,
        recipe: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> float:
        """Compute preference similarity"""
        scores = []

        # Cuisine match
        if 'cuisine' in user_preferences:
            preferred = user_preferences['cuisine'].lower()
            recipe_cuisine = recipe.get('cuisine', '').lower()
            scores.append(1.0 if preferred in recipe_cuisine else 0.0)

        # Liked ingredients
        if 'liked_ingredients' in user_preferences:
            liked = set(user_preferences['liked_ingredients'])
            recipe_ingredients = set([
                ing.get('name', ing) if isinstance(ing, dict) else ing
                for ing in recipe.get('ingredients', [])
            ])
            overlap = len(liked & recipe_ingredients)
            scores.append(overlap / len(liked) if liked else 1.0)

        # Dietary restrictions
        if 'dietary_restrictions' in user_preferences:
            restrictions = set(user_preferences['dietary_restrictions'])
            recipe_tags = set(recipe.get('tags', []))
            satisfied = len(restrictions & recipe_tags)
            scores.append(satisfied / len(restrictions) if restrictions else 1.0)

        return np.mean(scores) if scores else 0.5

    def compute_similarity(
        self,
        recipe: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> float:
        """Compute overall similarity"""
        nutrition_sim = self.compute_nutrition_similarity(
            recipe.get('nutrition', {}),
            user_profile.get('nutrition_targets', {})
        )

        preference_sim = self.compute_preference_similarity(
            recipe,
            user_profile
        )

        # Weighted combination
        return 0.6 * nutrition_sim + 0.4 * preference_sim


class RetrievalBaseline:
    """Complete retrieval baseline combining BM25 and user similarity"""

    def __init__(self, recipe_corpus_path: str):
        """
        Args:
            recipe_corpus_path: Path to recipe corpus JSONL file
        """
        self.recipes = self._load_recipes(recipe_corpus_path)
        self.bm25 = BM25Retriever()
        self.user_sim = UserProfileSimilarity()

        # Build BM25 index
        print("Building BM25 index...")
        documents = self._build_documents()
        self.bm25.fit(documents)

    def _load_recipes(self, path: str) -> List[Dict]:
        """Load recipe corpus"""
        recipes = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                recipes.append(json.loads(line.strip()))
        print(f"Loaded {len(recipes)} recipes from corpus")
        return recipes

    def _build_documents(self) -> List[str]:
        """Build text documents for BM25"""
        documents = []
        for recipe in self.recipes:
            # Concatenate title, ingredients, and tags
            text = recipe.get('title', '')
            text += ' ' + ' '.join(recipe.get('ingredients', []))
            text += ' ' + ' '.join(recipe.get('tags', []))
            documents.append(text)
        return documents

    def retrieve(
        self,
        user_profile: Dict[str, Any],
        top_k: int = 10,
        bm25_weight: float = 0.4,
        user_sim_weight: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recipes for user profile

        Args:
            user_profile: User profile with constraints and preferences
            top_k: Number of recipes to retrieve
            bm25_weight: Weight for BM25 score
            user_sim_weight: Weight for user similarity score

        Returns:
            List of top-k recipes
        """
        # Build query from user profile
        query = self._build_query(user_profile)

        # Get BM25 scores
        bm25_scores = {}
        for idx in range(len(self.recipes)):
            bm25_scores[idx] = self.bm25.score(query, idx)

        # Normalize BM25 scores
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        if max_bm25 > 0:
            bm25_scores = {k: v/max_bm25 for k, v in bm25_scores.items()}

        # Compute user similarity scores
        user_scores = {}
        for idx, recipe in enumerate(self.recipes):
            user_scores[idx] = self.user_sim.compute_similarity(recipe, user_profile)

        # Combine scores
        final_scores = {}
        for idx in range(len(self.recipes)):
            final_scores[idx] = (
                bm25_weight * bm25_scores[idx] +
                user_sim_weight * user_scores[idx]
            )

        # Rank and return top-k
        ranked_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        top_indices = ranked_indices[:top_k]

        results = []
        for idx in top_indices:
            result = self.recipes[idx].copy()
            result['retrieval_score'] = final_scores[idx]
            result['bm25_score'] = bm25_scores[idx]
            result['user_sim_score'] = user_scores[idx]
            results.append(result)

        return results

    def _build_query(self, user_profile: Dict[str, Any]) -> str:
        """Build query string from user profile"""
        query_parts = []

        # Add cuisine preference
        if 'cuisine' in user_profile:
            query_parts.append(user_profile['cuisine'])

        # Add liked ingredients
        if 'liked_ingredients' in user_profile:
            query_parts.extend(user_profile['liked_ingredients'])

        # Add dietary restrictions
        if 'dietary_restrictions' in user_profile:
            query_parts.extend(user_profile['dietary_restrictions'])

        return ' '.join(query_parts)


def evaluate_retrieval_baseline(
    test_file: str,
    recipe_corpus_path: str,
    output_file: str,
    top_k: int = 10
):
    """
    Evaluate retrieval baseline on test set

    Args:
        test_file: Test set JSONL file (Task A)
        recipe_corpus_path: Recipe corpus path
        output_file: Output predictions file
        top_k: Number of recipes to retrieve
    """
    # Load retrieval system
    retriever = RetrievalBaseline(recipe_corpus_path)

    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    # Run retrieval
    predictions = []
    for sample in tqdm(test_data, desc="Running retrieval"):
        user_profile = sample.get('user_profile', {})
        retrieved = retriever.retrieve(user_profile, top_k=top_k)

        predictions.append({
            'user_profile': user_profile,
            'retrieved_recipes': retrieved,
            'ranking': [r.get('recipe_id', i) for i, r in enumerate(retrieved)]
        })

    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"✅ Retrieval predictions saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval Baseline for NutriPlan")
    parser.add_argument('--test_file', type=str, required=True,
                        help='Test file (Task A)')
    parser.add_argument('--recipe_corpus', type=str, required=True,
                        help='Recipe corpus JSONL file')
    parser.add_argument('--output_file', type=str, default='retrieval_predictions.jsonl',
                        help='Output predictions file')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of recipes to retrieve')

    args = parser.parse_args()

    evaluate_retrieval_baseline(
        test_file=args.test_file,
        recipe_corpus_path=args.recipe_corpus,
        output_file=args.output_file,
        top_k=args.top_k
    )
