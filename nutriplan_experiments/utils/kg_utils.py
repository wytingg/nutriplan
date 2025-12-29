"""
Knowledge Graph Utilities for NutriPlan
Provides access to nutriplan_kg4.graphml for baseline models
"""

import graph_tool.all as gt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class NutriPlanKG:
    """Knowledge Graph accessor for NutriPlan experiments"""

    def __init__(self, kg_path: str = "work/recipebench/kg/nutriplan_kg4.graphml"):
        """
        Load NutriPlan Knowledge Graph

        Args:
            kg_path: Path to KG file (nutriplan_kg4.graphml)
        """
        self.kg_path = Path(kg_path)
        print(f"Loading Knowledge Graph from: {self.kg_path}")

        if not self.kg_path.exists():
            raise FileNotFoundError(f"KG file not found: {self.kg_path}")

        # Load graph
        self.graph = gt.load_graph(str(self.kg_path))

        # Access properties
        self.node_type = self.graph.vp["node_type"]
        self.node_id = self.graph.vp["node_id"]
        self.node_name = self.graph.vp["node_name"]
        self.node_unit = self.graph.vp.get("node_unit")

        # User properties
        self.user_gender = self.graph.vp.get("user_gender")
        self.user_age = self.graph.vp.get("user_age")
        self.user_physio_state = self.graph.vp.get("user_physio_state")

        # Edge properties
        self.edge_type = self.graph.ep["edge_type"]
        self.qty_raw = self.graph.ep.get("qty_raw")
        self.unit_raw = self.graph.ep.get("unit_raw")
        self.amount_raw = self.graph.ep.get("amount_raw")
        self.amount_unit = self.graph.ep.get("amount_unit")
        self.sign = self.graph.ep.get("sign")
        self.rni_value = self.graph.ep.get("rni_value")
        self.rni_unit = self.graph.ep.get("rni_unit")
        self.pmi_score = self.graph.ep.get("pmi_score")
        self.cooccurrence_count = self.graph.ep.get("cooccurrence_count")
        self.confidence = self.graph.ep.get("confidence")
        self.synergy_score = self.graph.ep.get("synergy_score")
        self.synergy_reason = self.graph.ep.get("synergy_reason")

        # Build indices
        self._build_indices()

        print(f"✅ KG loaded: {self.graph.num_vertices()} nodes, {self.graph.num_edges()} edges")

    def _build_indices(self):
        """Build lookup indices for fast access"""
        self.vertices_by_type = {}  # node_type -> [vertices]
        self.vertices_by_id = {}    # (node_type, node_id) -> vertex

        for v in self.graph.vertices():
            ntype = self.node_type[v]
            nid = self.node_id[v]

            # Type index
            if ntype not in self.vertices_by_type:
                self.vertices_by_type[ntype] = []
            self.vertices_by_type[ntype].append(v)

            # ID index
            self.vertices_by_id[(ntype, nid)] = v

    def get_recipe_by_id(self, recipe_id: str) -> Optional[gt.Vertex]:
        """Get recipe vertex by ID"""
        return self.vertices_by_id.get(("recipe", str(recipe_id)))

    def get_user_by_id(self, user_id: str) -> Optional[gt.Vertex]:
        """Get user vertex by ID"""
        return self.vertices_by_id.get(("user", str(user_id)))

    def get_ingredient_by_name(self, ingredient_name: str) -> Optional[gt.Vertex]:
        """Get ingredient vertex by name"""
        return self.vertices_by_id.get(("ingredient", ingredient_name))

    def get_recipe_ingredients(self, recipe_id: str) -> List[Dict[str, Any]]:
        """
        Get all ingredients for a recipe

        Returns:
            List of {"name": str, "quantity": str, "unit": str}
        """
        recipe_v = self.get_recipe_by_id(recipe_id)
        if not recipe_v:
            return []

        ingredients = []
        for edge in recipe_v.out_edges():
            if self.edge_type[edge] == "recipe_to_ingredient":
                target_v = edge.target()
                ingredients.append({
                    "name": self.node_name[target_v],
                    "quantity": self.qty_raw[edge] if self.qty_raw else "",
                    "unit": self.unit_raw[edge] if self.unit_raw else ""
                })

        return ingredients

    def get_recipe_nutrition(self, recipe_id: str) -> Dict[str, float]:
        """
        Get nutrition values for a recipe

        Returns:
            Dict of {nutrient_name: value}
        """
        recipe_v = self.get_recipe_by_id(recipe_id)
        if not recipe_v:
            return {}

        nutrition = {}
        for edge in recipe_v.out_edges():
            if self.edge_type[edge] == "recipe_to_nutrient":
                target_v = edge.target()
                nutrient_name = self.node_name[target_v]
                amount = self.amount_raw[edge] if self.amount_raw else "0"
                try:
                    nutrition[nutrient_name] = float(amount)
                except ValueError:
                    nutrition[nutrient_name] = 0.0

        return nutrition

    def get_user_rni_targets(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get user's RNI nutrition targets

        Returns:
            Dict of {nutrient_name: {"value": float, "unit": str}}
        """
        user_v = self.get_user_by_id(user_id)
        if not user_v:
            return {}

        rni_targets = {}
        for edge in user_v.out_edges():
            if self.edge_type[edge] == "user_to_nutrient_rni":
                target_v = edge.target()
                nutrient_name = self.node_name[target_v]
                rni_targets[nutrient_name] = {
                    "value": float(self.rni_value[edge]) if self.rni_value else 0.0,
                    "unit": self.rni_unit[edge] if self.rni_unit else ""
                }

        return rni_targets

    def get_user_ingredient_preferences(self, user_id: str) -> Tuple[List[str], List[str]]:
        """
        Get user's liked and disliked ingredients

        Returns:
            (liked_ingredients, disliked_ingredients)
        """
        user_v = self.get_user_by_id(user_id)
        if not user_v:
            return [], []

        liked = []
        disliked = []

        for edge in user_v.out_edges():
            if self.edge_type[edge] == "user_to_ingredient":
                target_v = edge.target()
                ingredient_name = self.node_name[target_v]
                sign_val = int(self.sign[edge]) if self.sign else 0

                if sign_val == 1:
                    liked.append(ingredient_name)
                elif sign_val == -1:
                    disliked.append(ingredient_name)

        return liked, disliked

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get complete user profile

        Returns:
            Dict with gender, age, physiological_state, nutrition_rni, liked_ingredients, disliked_ingredients
        """
        user_v = self.get_user_by_id(user_id)
        if not user_v:
            return {}

        liked, disliked = self.get_user_ingredient_preferences(user_id)
        rni_targets = self.get_user_rni_targets(user_id)

        return {
            "user_id": user_id,
            "gender": self.user_gender[user_v] if self.user_gender else "",
            "age": int(self.user_age[user_v]) if self.user_age else 0,
            "physiological_state": self.user_physio_state[user_v] if self.user_physio_state else "",
            "nutrition_rni": rni_targets,
            "liked_ingredients": liked,
            "disliked_ingredients": disliked
        }

    def get_ingredient_cooccurrence(self, ingredient1: str, ingredient2: str) -> Optional[Dict[str, float]]:
        """
        Get cooccurrence scores between two ingredients

        Returns:
            {"pmi_score": float, "cooccurrence_count": int, "confidence": float} or None
        """
        ing1_v = self.get_ingredient_by_name(ingredient1)
        ing2_v = self.get_ingredient_by_name(ingredient2)

        if not ing1_v or not ing2_v:
            return None

        # Check edge from ing1 to ing2
        for edge in ing1_v.out_edges():
            if edge.target() == ing2_v and self.edge_type[edge] == "ingredient_cooccurs":
                return {
                    "pmi_score": float(self.pmi_score[edge]) if self.pmi_score else 0.0,
                    "cooccurrence_count": int(self.cooccurrence_count[edge]) if self.cooccurrence_count else 0,
                    "confidence": float(self.confidence[edge]) if self.confidence else 0.0
                }

        return None

    def get_ingredient_complementarity(self, ingredient1: str, ingredient2: str) -> Optional[Dict[str, Any]]:
        """
        Get complementarity scores between two ingredients

        Returns:
            {"synergy_score": float, "reason": str} or None
        """
        ing1_v = self.get_ingredient_by_name(ingredient1)
        ing2_v = self.get_ingredient_by_name(ingredient2)

        if not ing1_v or not ing2_v:
            return None

        # Check edge from ing1 to ing2
        for edge in ing1_v.out_edges():
            if edge.target() == ing2_v and self.edge_type[edge] == "ingredient_complements":
                return {
                    "synergy_score": float(self.synergy_score[edge]) if self.synergy_score else 0.0,
                    "reason": self.synergy_reason[edge] if self.synergy_reason else ""
                }

        return None

    def get_recommended_ingredients_for_user(
        self,
        user_id: str,
        top_k: int = 20,
        use_cooccurrence: bool = True
    ) -> List[str]:
        """
        Get recommended ingredients based on user's liked ingredients and cooccurrence

        Args:
            user_id: User ID
            top_k: Number of recommendations
            use_cooccurrence: Use cooccurrence rules for recommendation

        Returns:
            List of recommended ingredient names
        """
        liked, disliked = self.get_user_ingredient_preferences(user_id)

        if not liked or not use_cooccurrence:
            # Fallback: return random ingredients
            all_ingredients = [self.node_name[v] for v in self.vertices_by_type.get("ingredient", [])]
            return list(set(all_ingredients) - set(disliked))[:top_k]

        # Score ingredients based on cooccurrence with liked ingredients
        ingredient_scores = {}

        for liked_ing in liked:
            liked_v = self.get_ingredient_by_name(liked_ing)
            if not liked_v:
                continue

            # Find cooccurring ingredients
            for edge in liked_v.out_edges():
                if self.edge_type[edge] == "ingredient_cooccurs":
                    target_v = edge.target()
                    target_name = self.node_name[target_v]

                    # Skip disliked ingredients
                    if target_name in disliked:
                        continue

                    # Accumulate score
                    pmi = float(self.pmi_score[edge]) if self.pmi_score else 0.0
                    conf = float(self.confidence[edge]) if self.confidence else 0.0
                    score = pmi * conf  # Combined score

                    ingredient_scores[target_name] = ingredient_scores.get(target_name, 0) + score

        # Sort by score
        ranked = sorted(ingredient_scores.items(), key=lambda x: x[1], reverse=True)
        return [ing for ing, _ in ranked[:top_k]]

    def get_all_recipes(self, limit: Optional[int] = None) -> List[str]:
        """
        Get all recipe IDs

        Args:
            limit: Maximum number of recipes to return

        Returns:
            List of recipe IDs
        """
        recipe_vertices = self.vertices_by_type.get("recipe", [])
        recipe_ids = [self.node_id[v] for v in recipe_vertices]

        if limit:
            return recipe_ids[:limit]
        return recipe_ids

    def get_recipe_metadata(self, recipe_id: str) -> Dict[str, Any]:
        """
        Get complete recipe metadata

        Returns:
            Dict with title, ingredients, nutrition
        """
        recipe_v = self.get_recipe_by_id(recipe_id)
        if not recipe_v:
            return {}

        return {
            "recipe_id": recipe_id,
            "title": self.node_name[recipe_v],
            "ingredients": self.get_recipe_ingredients(recipe_id),
            "nutrition": self.get_recipe_nutrition(recipe_id)
        }


if __name__ == "__main__":
    # Test KG loading
    kg = NutriPlanKG("work/recipebench/kg/nutriplan_kg4.graphml")

    # Test recipe access
    all_recipes = kg.get_all_recipes(limit=5)
    print(f"\nFirst 5 recipes: {all_recipes}")

    if all_recipes:
        recipe_id = all_recipes[0]
        recipe_data = kg.get_recipe_metadata(recipe_id)
        print(f"\nRecipe {recipe_id}:")
        print(f"  Title: {recipe_data['title']}")
        print(f"  Ingredients: {len(recipe_data['ingredients'])}")
        print(f"  Nutrition: {recipe_data['nutrition']}")

    # Test user access
    all_users = kg.get_all_recipes(limit=5)  # Just for testing
    print(f"\nKG statistics:")
    print(f"  Recipes: {len(kg.vertices_by_type.get('recipe', []))}")
    print(f"  Users: {len(kg.vertices_by_type.get('user', []))}")
    print(f"  Ingredients: {len(kg.vertices_by_type.get('ingredient', []))}")
    print(f"  Nutrients: {len(kg.vertices_by_type.get('nutrient', []))}")
