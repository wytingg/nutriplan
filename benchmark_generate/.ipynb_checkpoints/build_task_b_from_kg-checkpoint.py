#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task B Dataset Generator - Hybrid Approach (KG + External Data)

设计理念：
1. 主要从KG提取：用户数据、食材共现关系、食材互补关系、可用食材列表
2. 从外部文件补充KG缺失的精确数据：
   - top500_nutrition_complete.csv: 精确的食材营养数据库
   - calculate_recipe_nutrition: 精确的营养计算
   - household_units_converter: 家庭单位转换
   - ingredient_normalizer: 食材名称匹配

优势：
- 利用KG的图结构优势（关系查询、用户数据统一管理）
- 利用专业数据库的精确性（营养计算、单位转换）
- 生成最优秀的训练数据
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import graph_tool.all as gt

# 导入外部工具（KG中没有的精确工具）
from calculate_recipe_nutrition import RecipeNutritionCalculator
from ingredient_normalizer import fuzzy_match_ingredient
from household_units_converter import convert_to_household_unit

print("="*80)
print("Task B Dataset Generator - Hybrid Approach (KG + External Data)")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

# KG数据源（主要）
KG_PATH = "work/recipebench/kg/nutriplan_kg4.graphml"

# 外部数据源（补充KG缺失的精确数据）
NUTRITION_DB = "work/recipebench/data/11_nutrition_rule/top500_nutrition_complete.csv"

OUTPUT_DIR = 'work/recipebench/data/10large_scale_datasets'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# In build_task_b_from_kg.py (lines 49-53)
TRAIN_SIZE = 10000  # 改回10000
VAL_SIZE = 2000     # 改回2000
TEST_SIZE = 2000    # 改回2000
TOTAL_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

# ============================================================================
# Instruction Templates (10 diverse templates)
# ============================================================================

INSTRUCTION_TEMPLATES = [
    {
        "template": "I am a {age}-year-old {gender} with {physiological_state}. Based on my health condition and nutritional needs, please create a recipe that meets my RNI requirements. I like {liked_sample} but dislike {disliked_sample}. Please generate a complete recipe with ingredients, quantities, and cooking instructions.",
        "type": "health_condition"
    },
    {
        "template": "Please design a recipe for me based on my daily nutritional requirements: {rni_sample}. I prefer dishes with {liked_sample}, and I cannot eat {disliked_sample}. Make sure the recipe is balanced and meets my RNI targets.",
        "type": "nutrition_target"
    },
    {
        "template": "As a {age}-year-old {gender}, I need help creating a meal plan. My physiological state is {physiological_state}, which requires specific nutrition management. Generate a recipe using {liked_sample}, avoiding {disliked_sample}, that aligns with my RNI standards.",
        "type": "meal_planning"
    },
    {
        "template": "I'm looking for a recipe that incorporates {liked_sample} while excluding {disliked_sample}. My nutritional requirements are: {rni_sample}. Please create a dish that satisfies these constraints and tastes good.",
        "type": "preference_focused"
    },
    {
        "template": "Generate a {physiological_state}-friendly recipe for a {gender} aged {age}. The recipe should use ingredients I enjoy ({liked_sample}), avoid what I dislike ({disliked_sample}), and meet my specific RNI nutritional targets.",
        "type": "condition_specific"
    },
    {
        "template": "Based on my RNI requirements ({rni_sample}), create a nutritionally balanced recipe. I have preferences for {liked_sample} and restrictions against {disliked_sample}. Please ensure the recipe is practical and delicious.",
        "type": "rni_constraint"
    },
    {
        "template": "I need a customized recipe as a {age}-year-old {gender} managing {physiological_state}. Please use {liked_sample} in the recipe, avoid {disliked_sample}, and make sure it meets my daily RNI nutritional needs.",
        "type": "customized_request"
    },
    {
        "template": "Create a recipe that matches my nutritional profile: {rni_sample}. I want to include {liked_sample} and exclude {disliked_sample}. The recipe should be suitable for someone with {physiological_state}.",
        "type": "profile_matching"
    },
    {
        "template": "As someone with {physiological_state}, I need recipes carefully designed for my condition. I'm a {age}-year-old {gender}. Please create a dish using {liked_sample}, avoiding {disliked_sample}, that meets my RNI standards.",
        "type": "health_aware"
    },
    {
        "template": "Design a recipe for me that achieves these nutritional targets: {rni_sample}. My ingredient preferences include {liked_sample}, and I need to avoid {disliked_sample}. Make it both nutritious and appetizing.",
        "type": "target_achievement"
    }
]

# ============================================================================
# Keywords for Ingredient Classification
# ============================================================================

PROTEIN_KEYWORDS = [
    'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck',
    'fish', 'snapper', 'fillet', 'salmon', 'tuna', 'cod', 'tilapia', 'halibut',
    'shrimp', 'prawn', 'crab', 'lobster', 'scallop',
    'egg', 'tofu', 'tempeh', 'seitan',
    'bean', 'lentil', 'chickpea'
]

CARB_KEYWORDS = [
    'rice', 'pasta', 'noodle', 'spaghetti', 'linguine', 'penne', 'fettuccine',
    'macaroni', 'rigatoni', 'orzo', 'lasagna', 'ravioli', 'tortellini',
    'bread', 'baguette', 'roll', 'bun', 'tortilla', 'pita',
    'potato', 'sweet potato', 'yam',
    'quinoa', 'oat', 'oatmeal', 'corn', 'cornmeal', 'barley', 'couscous',
    'bulgur', 'millet', 'farro'
]

HERB_KEYWORDS = [
    'cilantro', 'coriander', 'parsley', 'basil', 'mint', 'dill', 'chive',
    'rosemary', 'thyme', 'oregano', 'sage', 'tarragon', 'marjoram'
]

HIGH_FIBER_VEGETABLES = ['broccoli', 'spinach', 'kale', 'brussels sprout',
                         'carrot', 'cauliflower', 'sweet potato', 'bean']

EXCLUDE_KEYWORDS = [
    'cheese', 'cheddar', 'parmesan', 'mozzarella', 'ricotta', 'feta', 'pecorino',
    'seasoning', 'spice', 'powder', 'onion powder', 'garlic powder',
    'bouillon', 'cubes', 'bouillon cubes', 'stock', 'broth',
    'sauce', 'paste', 'extract', 'vinegar', 'wine', 'liquor', 'beer',
    'cream', 'milk', 'yogurt', 'butter', 'margarine',
    'gelatin', 'yeast', 'baking powder', 'baking soda',
    'sugar', 'honey', 'syrup', 'molasses', 'corn syrup',
    'flour', 'bread flour', 'gluten flour', 'wheat flour', 'all-purpose flour',
    'wheat germ', 'wheat bran', 'cornstarch', 'starch',  # Added 'starch'
    'ham bone', 'beef bone', 'pork bone', 'bone',
    'mayonnaise', 'miracle whip', 'whip', 'ketchup', 'mustard', 'relish',
    'jam', 'jelly', 'preserves', 'marmalade',
    'juice',  # Added 'juice' - exclude fruit/vegetable juices
    'dip', 'guacamole', 'salsa', 'hummus'  # Added 'dip' and related items
]

# ============================================================================
# Load Knowledge Graph (主要数据源)
# ============================================================================

print(f"\n[1/8] Loading Knowledge Graph from {KG_PATH}...")
try:
    graph = gt.load_graph(KG_PATH)
    print(f"  [OK] Loaded KG with {graph.num_vertices():,} nodes, {graph.num_edges():,} edges")
except Exception as e:
    print(f"  [ERROR] Failed to load KG: {e}")
    exit(1)

# Extract graph properties
node_type = graph.vp.node_type
node_id = graph.vp.node_id
node_name = graph.vp.node_name
user_gender = graph.vp.user_gender
user_age = graph.vp.user_age
user_physio_state = graph.vp.user_physio_state

edge_type = graph.ep.edge_type
sign = graph.ep.sign
rni_value = graph.ep.rni_value
rni_unit = graph.ep.rni_unit
pmi_score = graph.ep.pmi_score
confidence = graph.ep.confidence
synergy_score = graph.ep.synergy_score

print(f"  [OK] Loaded graph properties")

# ============================================================================
# Load External Precise Data (补充数据源)
# ============================================================================

print(f"\n[2/8] Loading external precise nutrition database from {NUTRITION_DB}...")
try:
    calc = RecipeNutritionCalculator(NUTRITION_DB)
    print(f"  [OK] Loaded nutrition calculator with {len(calc.nutrition_lookup)} ingredients")
except Exception as e:
    print(f"  [ERROR] Failed to load nutrition database: {e}")
    exit(1)

# ============================================================================
# Build Indices from KG
# ============================================================================

print("\n[3/8] Building indices from KG...")

# Index: user_id -> user data (从KG提取)
users_from_kg = {}
for v in graph.vertices():
    if node_type[v] == "user":
        user_id = node_id[v]
        users_from_kg[user_id] = {
            'user_id': int(user_id) if user_id.isdigit() else user_id,
            'gender': user_gender[v],
            'age': user_age[v],
            'physiological_state': user_physio_state[v],
            'liked_ingredients': [],
            'disliked_ingredients': [],
            'nutrition_rni': {}
        }

print(f"  [OK] Found {len(users_from_kg)} users in KG")

# Extract user preferences and RNI from KG edges
for v in graph.vertices():
    if node_type[v] == "user":
        user_id = node_id[v]

        for e in v.out_edges():
            target = e.target()
            etype = edge_type[e]

            # User -> Ingredient (preferences from KG)
            if etype == "user_to_ingredient":
                ing_name = node_name[target]
                preference_sign = sign[e]

                if preference_sign == 1:
                    users_from_kg[user_id]['liked_ingredients'].append({'name': ing_name})
                elif preference_sign == -1:
                    users_from_kg[user_id]['disliked_ingredients'].append({'name': ing_name})

            # User -> Nutrient (RNI from KG)
            elif etype == "user_to_nutrient_rni":
                nutrient_name = node_name[target]
                rni_val = rni_value[e]

                # Map nutrient names to standard keys
                nutrient_key_map = {
                    'Energy': 'energy_kcal',
                    'Protein': 'protein_g',
                    'Carbohydrate': 'carbohydrate_g',
                    'Fat': 'fat_g',
                    'Fiber': 'fiber_g',
                    'Added Sugar': 'added_sugar_g',
                    'Saturated Fat': 'saturated_fat_g',
                    'Trans Fat': 'trans_fat_g',
                    'Sodium': 'sodium_mg',
                    'Potassium': 'potassium_mg',
                    'Calcium': 'calcium_mg',
                    'Iron': 'iron_mg',
                    'Vitamin C': 'vitamin_c_mg',
                    'Vitamin D': 'vitamin_d_ug',
                    'Folate': 'folate_ug',
                }

                if nutrient_name in nutrient_key_map:
                    key = nutrient_key_map[nutrient_name]
                    users_from_kg[user_id]['nutrition_rni'][key] = rni_val

print(f"  [OK] Extracted user preferences and RNI from KG")

# Index: Ingredient co-occurrence (从KG提取)
cooccur_dict = defaultdict(list)
for e in graph.edges():
    if edge_type[e] == "ingredient_cooccurs":
        ing1 = node_name[e.source()]
        ing2 = node_name[e.target()]
        score = pmi_score[e] if pmi_score[e] else confidence[e]
        cooccur_dict[ing1].append({'ingredient': ing2, 'score': score})

print(f"  [OK] Built co-occurrence index from KG ({len(cooccur_dict)} ingredients)")

# Index: Ingredient complementarity (从KG提取)
complement_dict = defaultdict(list)
for e in graph.edges():
    if edge_type[e] == "ingredient_complements":
        ing1 = node_name[e.source()]
        ing2 = node_name[e.target()]
        score = synergy_score[e]
        complement_dict[ing1].append({'ingredient': ing2, 'score': score})

print(f"  [OK] Built complementarity index from KG ({len(complement_dict)} ingredients)")

# Index: All available ingredients (KG提供列表，但营养数据从外部数据库获取)
all_ingredients_kg = set()
for v in graph.vertices():
    if node_type[v] == "ingredient":
        all_ingredients_kg.add(node_name[v])

print(f"  [OK] Found {len(all_ingredients_kg)} ingredients in KG")

# ============================================================================
# Helper Functions
# ============================================================================

def is_protein_source(ing):
    return any(keyword in ing.lower() for keyword in PROTEIN_KEYWORDS)

def is_carb_source(ing):
    return any(keyword in ing.lower() for keyword in CARB_KEYWORDS)

def is_herb(ing):
    return any(herb in ing.lower() for herb in HERB_KEYWORDS)

def is_high_fiber(ing):
    return any(veg in ing.lower() for veg in HIGH_FIBER_VEGETABLES)

def should_exclude(ing):
    return any(keyword in ing.lower() for keyword in EXCLUDE_KEYWORDS)

def is_vegetable(ing):
    return (not is_protein_source(ing) and
            not is_carb_source(ing) and
            not should_exclude(ing))

def get_sodium_per_100g(ing):
    """使用外部精确营养数据库"""
    nutrition = calc.find_ingredient_in_db(ing)
    if nutrition:
        return nutrition.get('sodium_mg', 0)
    return 0

def is_high_sodium(ing):
    return get_sodium_per_100g(ing) > 500

def is_catastrophic_sodium(ing):
    return get_sodium_per_100g(ing) > 8000

def get_fiber_per_100g(ing):
    """使用外部精确营养数据库"""
    nutrition = calc.find_ingredient_in_db(ing)
    if nutrition:
        return nutrition.get('fiber_g', 0)
    return 0

def simplify_ingredient_name(ing):
    modifiers_to_remove = [
        'boneless', 'skinless', 'bone-in', 'skin-on',
        'fresh', 'frozen', 'dried', 'canned',
        'raw', 'cooked', 'uncooked',
        'large', 'medium', 'small', 'extra-large',
        'whole', 'halved', 'quartered', 'chopped', 'diced', 'sliced',
        'low-sodium', 'low-fat', 'fat-free', 'reduced-fat'
    ]

    words = ing.split()
    simplified = [word for word in words if word.lower() not in modifiers_to_remove]
    result = ' '.join(simplified)

    if 'chicken breast halves' in result:
        result = result.replace('chicken breast halves', 'chicken breast')

    return result if result else ing

def normalize_for_dedup(ing):
    normalized = ing.lower().strip()

    modifiers = [
        'fresh', 'frozen', 'dried', 'canned', 'raw', 'cooked',
        'chopped', 'diced', 'sliced', 'minced', 'crushed',
        'large', 'medium', 'small', 'baby',
        'italian-style', 'greek-style', 'mexican-style'
    ]
    for mod in modifiers:
        normalized = normalized.replace(mod + ' ', '').replace(' ' + mod, '')

    # Handle plurals
    if normalized.endswith('ies'):
        normalized = normalized[:-3] + 'y'
    elif normalized.endswith('es') and not normalized.endswith('ses'):
        normalized = normalized[:-2]
    elif normalized.endswith('s') and len(normalized) > 3:
        normalized = normalized[:-1]

    return normalized.strip()

def is_duplicate_ingredient(new_ing, existing_ingredients):
    new_norm = normalize_for_dedup(new_ing)
    for existing in existing_ingredients:
        existing_norm = normalize_for_dedup(existing)
        if new_norm == existing_norm or new_norm in existing_norm or existing_norm in new_norm:
            return True
    return False

# ============================================================================
# Instruction Template Selection
# ============================================================================

def select_instruction_template(user_profile, seed=0):
    """Select and fill instruction template"""
    random.seed(seed + user_profile['user_id'])

    template_obj = random.choice(INSTRUCTION_TEMPLATES)
    template = template_obj['template']
    template_type = template_obj['type']

    # Sample ingredients for template
    liked_sample = ', '.join([ing['name'] for ing in random.sample(
        user_profile['liked_ingredients'],
        min(3, len(user_profile['liked_ingredients']))
    )]) if user_profile['liked_ingredients'] else 'various ingredients'

    disliked_sample = ', '.join([ing['name'] for ing in random.sample(
        user_profile['disliked_ingredients'],
        min(2, len(user_profile['disliked_ingredients']))
    )]) if user_profile['disliked_ingredients'] else 'none'

    # Sample RNI values for template
    rni = user_profile['nutrition_rni']
    rni_items = []

    if rni.get('energy_kcal'):
        rni_items.append(f"{rni['energy_kcal']:.0f} kcal energy")
    if rni.get('protein_g'):
        rni_items.append(f"{rni['protein_g']:.0f}g protein")
    if rni.get('fiber_g'):
        rni_items.append(f"{rni['fiber_g']:.0f}g fiber")
    if rni.get('sodium_mg'):
        rni_items.append(f"{rni['sodium_mg']:.0f}mg sodium")

    rni_sample = ', '.join(rni_items[:3]) if rni_items else "balanced nutrition"

    # Fill template
    instruction = template.format(
        age=user_profile['age'],
        gender=user_profile['gender'],
        physiological_state=user_profile['physiological_state'],
        liked_sample=liked_sample,
        disliked_sample=disliked_sample,
        rni_sample=rni_sample
    )

    return instruction, template_type

# ============================================================================
# Ingredient Selection (使用KG的共现关系 + 外部营养数据库)
# ============================================================================

def select_ingredients_hybrid(user_profile, seed=0):
    """
    混合方法选择食材：
    - 从KG获取：用户偏好、共现关系、互补关系
    - 从外部数据库获取：精确营养数据用于过滤
    """
    random.seed(seed)

    liked = user_profile['liked_ingredients']
    disliked = user_profile['disliked_ingredients']
    nutrition_rni = user_profile['nutrition_rni']

    liked_names = [ing['name'].lower() for ing in liked]
    disliked_names = [ing['name'].lower() for ing in disliked]

    # 使用外部工具匹配食材名称到数据库
    db_ingredients = list(calc.nutrition_lookup.keys())
    matched_liked = []
    for liked_ing in liked_names:
        match, score = fuzzy_match_ingredient(liked_ing, db_ingredients, threshold=0.6)
        if match:
            matched_liked.append(match.lower())
        else:
            matched_liked.append(liked_ing)

    matched_disliked = []
    for disliked_ing in disliked_names:
        match, score = fuzzy_match_ingredient(disliked_ing, db_ingredients, threshold=0.6)
        if match:
            matched_disliked.append(match.lower())
        else:
            matched_disliked.append(disliked_ing)

    # RNI constraints
    added_sugar_limit = nutrition_rni.get('added_sugar_g', 50)
    restrict_sugar = added_sugar_limit <= 25

    sodium_limit = nutrition_rni.get('sodium_mg', 2300)
    restrict_sodium = sodium_limit <= 1500

    fiber_target = nutrition_rni.get('fiber_g', 25)

    selected = []

    def is_valid(ing):
        """严格验证（用于一般食材选择）"""
        if ing in matched_disliked or any(d in ing for d in matched_disliked):
            return False
        if should_exclude(ing):
            return False
        if restrict_sugar and 'sugar' in ing:
            return False
        if restrict_sodium and is_high_sodium(ing):
            return False
        if is_catastrophic_sodium(ing):
            return False
        # 确保食材在外部数据库中存在（用于精确营养计算）
        if not calc.find_ingredient_in_db(ing):
            return False
        return True

    def is_valid_for_liked(ing):
        """宽松验证（用于用户喜欢的食材）"""
        # 只检查最基本的条件
        if ing in matched_disliked or any(d in ing for d in matched_disliked):
            return False
        if is_catastrophic_sodium(ing):  # 只排除极端高钠食材
            return False
        if not calc.find_ingredient_in_db(ing):
            return False
        return True

    # STEP 1: Protein (MANDATORY)
    liked_proteins = [ing for ing in matched_liked if is_protein_source(ing) and is_valid(ing)]

    if liked_proteins:
        protein_selected = random.choice(liked_proteins)
    else:
        # 增加多样性：从更大的蛋白质池中随机选择
        common_proteins = ['chicken breast', 'chicken', 'beef', 'pork', 'salmon',
                          'tilapia', 'cod', 'tofu', 'eggs', 'turkey', 'shrimp',
                          'ground beef', 'lamb', 'duck', 'tuna']

        # 随机打乱顺序，避免总是选择第一个
        random.shuffle(common_proteins)

        protein_selected = None
        for protein in common_proteins:
            if is_valid(protein):
                protein_selected = protein
                break

        if not protein_selected:
            available = [ing for ing in calc.nutrition_lookup.keys()
                        if is_protein_source(ing) and is_valid(ing)]
            if available:
                # 从更大的池中随机选择（前50个而不是前20个）
                protein_selected = random.choice(available[:min(50, len(available))])
            else:
                protein_selected = 'chicken'

    selected.append(protein_selected)

    # STEP 2: Carb
    liked_carbs = [ing for ing in matched_liked if is_carb_source(ing) and is_valid(ing)]

    if liked_carbs:
        selected.append(random.choice(liked_carbs))
    else:
        # 增加多样性：扩大碳水化合物选择范围
        available = [ing for ing in calc.nutrition_lookup.keys()
                    if is_carb_source(ing) and is_valid(ing)]
        if available:
            # 从更大的池中随机选择（前50个）
            selected.append(random.choice(available[:min(50, len(available))]))
        else:
            # 提供多个备选，随机选择
            fallback_carbs = ['rice', 'pasta', 'potato', 'bread']
            for carb in fallback_carbs:
                if is_valid(carb):
                    selected.append(carb)
                    break
            else:
                selected.append('rice')

    # STEP 2.5: 强制添加用户喜欢的食材 (NEW! - 解决用户偏好被忽略的问题)
    if matched_liked:
        # 找出还没被选中的liked ingredients
        remaining_liked = [ing for ing in matched_liked
                          if ing not in selected and not is_duplicate_ingredient(ing, selected)]

        # 使用宽松验证，尝试添加1-2个用户喜欢的食材
        added_liked = []
        for liked_ing in remaining_liked:
            if is_valid_for_liked(liked_ing):
                added_liked.append(liked_ing)
                if len(added_liked) >= 2:  # 最多添加2个
                    break

        # 随机选择1-2个添加到selected中
        if added_liked:
            num_to_add = min(len(added_liked), random.randint(1, 2))
            for ing in random.sample(added_liked, num_to_add):
                if len(selected) < 5:  # 为essentials留出空间
                    selected.append(ing)

    # STEP 3: High-fiber vegetables (使用外部数据库的纤维数据)
    available_fiber = [(ing, get_fiber_per_100g(ing))
                      for ing in calc.nutrition_lookup.keys()
                      if is_high_fiber(ing) and is_valid(ing) and is_vegetable(ing)]
    available_fiber.sort(key=lambda x: x[1], reverse=True)

    if fiber_target >= 40:
        target_count = 2
    elif fiber_target >= 25:
        target_count = 1
    else:
        target_count = 0

    count = 0
    # 增加多样性：从前20个高纤维蔬菜中随机选择，而不是总选前几个
    candidate_fiber_vegs = available_fiber[:20]
    random.shuffle(candidate_fiber_vegs)

    for ing, fiber in candidate_fiber_vegs:
        if count < target_count and ing not in selected and not is_duplicate_ingredient(ing, selected):
            selected.append(ing)
            count += 1
            if count >= target_count:
                break

    # STEP 4: Add vegetables using co-occurrence from KG
    max_additional_veggies = 2 if fiber_target >= 30 else 1
    veggies_added = 0

    for seed_ing in selected[:2]:
        if veggies_added >= max_additional_veggies:
            break

        # 使用KG的共现关系
        cooccur_list = cooccur_dict.get(seed_ing, [])
        cooccur_list = sorted(cooccur_list, key=lambda x: x['score'], reverse=True)

        for item in cooccur_list[:30]:
            co_ing = item['ingredient']
            if (co_ing not in selected and is_valid(co_ing) and
                is_vegetable(co_ing) and not is_herb(co_ing) and
                not is_duplicate_ingredient(co_ing, selected)):
                selected.append(co_ing)
                veggies_added += 1
                break

    # STEP 5: Essentials
    essentials = ['olive oil', 'salt', 'pepper']
    for ing in essentials:
        if len(selected) >= 7:
            break
        if ing not in selected and is_valid(ing):
            selected.append(ing)

    return selected[:7]

# ============================================================================
# Quantity Optimization (使用外部精确营养计算)
# ============================================================================

def optimize_quantities_rni_hybrid(ingredients, nutrition_rni, servings=4, max_iterations=5):
    """
    使用外部RecipeNutritionCalculator进行精确营养计算
    """
    # Extract RNI targets
    target_energy_daily = nutrition_rni.get('energy_kcal', 2000)
    target_protein_daily = nutrition_rni.get('protein_g', 60)
    target_fat_daily = nutrition_rni.get('fat_g', 65)
    target_carb_daily = nutrition_rni.get('carbohydrate_g', 260)

    # Per-serving targets
    target_energy = target_energy_daily / servings
    target_protein = target_protein_daily / servings
    target_fat = target_fat_daily / servings
    target_carb = target_carb_daily / servings

    # Calculate target percentages
    total_macro_kcal = (target_protein * 4) + (target_fat * 9) + (target_carb * 4)
    if total_macro_kcal > 0:
        target_protein_pct = (target_protein * 4 / total_macro_kcal) * 100
        target_fat_pct = (target_fat * 9 / total_macro_kcal) * 100
        target_carb_pct = (target_carb * 4 / total_macro_kcal) * 100
    else:
        target_protein_pct = 20
        target_fat_pct = 30
        target_carb_pct = 50

    # Initialize base quantities
    base_qty = {}
    for ing in ingredients:
        if is_protein_source(ing):
            base_qty[ing] = 120
        elif is_carb_source(ing):
            base_qty[ing] = 80
        elif 'oil' in ing or 'butter' in ing:
            base_qty[ing] = 8
        elif ing == 'salt':
            base_qty[ing] = 0.5
        elif ing == 'pepper':
            base_qty[ing] = 0.25
        elif 'garlic' in ing:
            base_qty[ing] = 8
        elif 'onion' in ing:
            base_qty[ing] = 40
        elif is_herb(ing):
            base_qty[ing] = 5
        else:
            base_qty[ing] = 70

    # Iterative optimization
    for iteration in range(max_iterations):
        total_qty = {ing: qty * servings for ing, qty in base_qty.items()}

        # 使用外部RecipeNutritionCalculator进行精确计算
        ingredient_strings = [f"{qty}g {ing}" for ing, qty in total_qty.items()]
        nutrition_result = calc.calculate_recipe_nutrition(ingredient_strings, servings)

        per_serv = nutrition_result['per_serving']
        actual_energy = per_serv['energy_kcal']
        actual_protein = per_serv['protein_g']
        actual_fat = per_serv['fat_g']
        actual_carb = per_serv['carbohydrates_g']

        # Calculate actual percentages
        protein_kcal = actual_protein * 4
        fat_kcal = actual_fat * 9
        carb_kcal = actual_carb * 4
        total_macro_kcal = protein_kcal + fat_kcal + carb_kcal

        if total_macro_kcal > 0:
            actual_protein_pct = (protein_kcal / total_macro_kcal) * 100
            actual_fat_pct = (fat_kcal / total_macro_kcal) * 100
            actual_carb_pct = (carb_kcal / total_macro_kcal) * 100
        else:
            actual_protein_pct = actual_fat_pct = actual_carb_pct = 0

        # Check convergence
        energy_error = abs(actual_energy - target_energy) / target_energy if target_energy > 0 else 0
        protein_error = abs(actual_protein_pct - target_protein_pct)
        fat_error = abs(actual_fat_pct - target_fat_pct)

        if energy_error < 0.1 and protein_error < 3 and fat_error < 3:
            break

        # Calculate adjustments
        if actual_energy > 0:
            energy_scale = target_energy / actual_energy
            energy_scale = max(0.3, min(3.0, energy_scale))
        else:
            energy_scale = 1.0

        protein_adjustment = 1.0
        fat_adjustment = 1.0
        carb_adjustment = 1.0

        if actual_protein_pct < target_protein_pct - 2:
            protein_adjustment = 1.2
        elif actual_protein_pct > target_protein_pct + 2:
            protein_adjustment = 0.85

        if actual_fat_pct > target_fat_pct + 2:
            fat_adjustment = 0.8
        elif actual_fat_pct < target_fat_pct - 2:
            fat_adjustment = 1.15

        if actual_carb_pct < target_carb_pct - 2:
            carb_adjustment = 1.15
        elif actual_carb_pct > target_carb_pct + 2:
            carb_adjustment = 0.85

        # Apply adjustments
        for ing in base_qty:
            base_qty[ing] *= energy_scale

            if is_protein_source(ing):
                base_qty[ing] *= protein_adjustment
            elif 'oil' in ing or 'butter' in ing:
                base_qty[ing] *= fat_adjustment
            elif is_carb_source(ing):
                base_qty[ing] *= carb_adjustment

        # Apply bounds
        for ing in base_qty:
            if ing == 'salt':
                base_qty[ing] = max(0.3, min(0.8, base_qty[ing]))
            elif ing == 'pepper':
                base_qty[ing] = max(0.15, min(0.5, base_qty[ing]))
            elif 'garlic' in ing:
                base_qty[ing] = max(2, min(15, base_qty[ing]))
            elif 'oil' in ing or 'butter' in ing:
                base_qty[ing] = max(3, min(20, base_qty[ing]))
            elif is_herb(ing):
                base_qty[ing] = max(2, min(8, base_qty[ing]))
            elif is_protein_source(ing):
                base_qty[ing] = max(30, min(200, base_qty[ing]))
            elif is_carb_source(ing):
                base_qty[ing] = max(20, min(150, base_qty[ing]))
            else:
                base_qty[ing] = max(10, min(150, base_qty[ing]))

    final_qty = {ing: qty * servings for ing, qty in base_qty.items()}
    return [(ing, qty) for ing, qty in final_qty.items()]

# ============================================================================
# Recipe Steps Generation
# ============================================================================

def generate_steps(ingredients_with_qty):
    """Generate cooking steps (improved to handle different ingredient types)"""
    steps = []

    proteins = []
    carbs = []
    vegetables = []
    herbs = []
    fats = []
    seasonings = []

    # 不应该被"wash and chop"的关键词
    skip_prep_keywords = ['starch', 'flour', 'powder', 'juice', 'broth', 'stock',
                          'sauce', 'oil', 'vinegar', 'wine', 'extract']

    for ing, qty in ingredients_with_qty:
        if is_protein_source(ing):
            proteins.append(ing)
        elif is_carb_source(ing):
            carbs.append(ing)
        elif 'oil' in ing or 'butter' in ing:
            fats.append(ing)
        elif ing in ['salt', 'pepper']:
            seasonings.append(ing)
        elif is_herb(ing):
            herbs.append(ing)
        else:
            vegetables.append(ing)

    step_num = 1

    # Prep vegetables (只处理真正的固体蔬菜)
    prep_veg = [v for v in vegetables
                if not any(keyword in v.lower() for keyword in skip_prep_keywords)]
    if prep_veg:
        steps.append(f"{step_num}. Wash and chop {', '.join(prep_veg)}.")
        step_num += 1

    # Prep herbs
    if herbs:
        steps.append(f"{step_num}. Rinse and chop {', '.join(herbs)}.")
        step_num += 1

    # Cook carbs
    if carbs:
        raw_carbs = [c for c in carbs if 'cooked' not in c]
        if raw_carbs:
            steps.append(f"{step_num}. Cook {', '.join(raw_carbs)} until tender.")
            step_num += 1

    # Cook protein
    if proteins:
        steps.append(f"{step_num}. Cook {', '.join(proteins)} until done.")
        step_num += 1

    # Saute vegetables
    if vegetables and fats:
        steps.append(f"{step_num}. Heat {fats[0]}, saute vegetables until softened.")
        step_num += 1

    # Combine
    garnish_items = []
    if herbs:
        garnish_items.extend(herbs)
    if seasonings:
        garnish_items.extend(seasonings)
    elif not seasonings:
        garnish_items.append('salt and pepper')

    steps.append(f"{step_num}. Combine all ingredients, garnish with {', '.join(garnish_items)}, and serve.")

    return steps

# ============================================================================
# Recipe Generation (混合方法)
# ============================================================================

def generate_recipe_hybrid(user_profile, seed=0):
    """
    混合方法生成食谱：
    - 从KG获取：用户数据、食材关系
    - 从外部工具获取：精确营养计算、单位转换
    """
    try:
        user_id = user_profile['user_id']
        nutrition_rni = user_profile['nutrition_rni']

        # Select instruction template
        instruction, instruction_type = select_instruction_template(user_profile, seed)

        # Select ingredients (KG关系 + 外部营养数据)
        ingredients = select_ingredients_hybrid(user_profile, seed)

        # Optimize quantities (外部精确计算)
        servings = 4
        ingredients_with_qty = optimize_quantities_rni_hybrid(ingredients, nutrition_rni, servings)

        # Generate steps
        steps = generate_steps(ingredients_with_qty)

        # Calculate nutrition (外部精确计算)
        ingredient_strings = [f"{qty}g {ing}" for ing, qty in ingredients_with_qty]
        nutrition_result = calc.calculate_recipe_nutrition(ingredient_strings, servings)

        # Convert to household units (外部工具)
        ingredients_natural_units = []
        for ing, qty_grams in ingredients_with_qty:
            natural_unit = convert_to_household_unit(ing, qty_grams, servings)
            ingredients_natural_units.append(natural_unit)

        # Create title
        protein_ings = [ing for ing in ingredients if is_protein_source(ing)]
        carb_ings = [ing for ing in ingredients if is_carb_source(ing)]
        veg_ings = [ing for ing in ingredients if is_vegetable(ing) and not is_herb(ing)]

        title_parts = []
        if protein_ings:
            title_parts.append(simplify_ingredient_name(protein_ings[0]))

        if carb_ings and (not veg_ings or random.random() < 0.6):
            title_parts.append(simplify_ingredient_name(carb_ings[0]))
        elif veg_ings:
            title_parts.append(simplify_ingredient_name(veg_ings[0]))

        title = " with ".join([ing.title() for ing in title_parts]) if len(title_parts) >= 2 else (title_parts[0].title() if title_parts else "Mixed Dish")

        # Build recipe output
        recipe = {
            'user_id': user_id,
            'recipe_id': f'generated_{user_id}_{seed}',
            'instruction': instruction,
            'instruction_type': instruction_type,
            'user_profile': {
                'gender': user_profile['gender'],
                'age': user_profile['age'],
                'physiological_state': user_profile['physiological_state'],
                'liked_ingredients': user_profile['liked_ingredients'],
                'disliked_ingredients': user_profile['disliked_ingredients'],
                'nutrition_rni': nutrition_rni
            },
            'output': {
                'title': title,
                'ingredients': ingredients_natural_units,
                'steps': steps,
                'servings': servings,
                'nutrition_per_serving': nutrition_result['per_serving']
            },
            'metadata': {
                'selected_ingredients': ingredients,
                'data_source': 'hybrid (KG + external precise data)',
                'kg_source': 'user data, co-occurrence, complementarity',
                'external_source': 'nutrition calculation, unit conversion',
                'nutrient_coverage': '8/15 nutrients (recipe data limitation)',
                'rni_optimization': 'Direct RNI values from KG'
            }
        }

        return recipe

    except Exception as e:
        print(f"\n[ERROR] Failed to generate recipe for user {user_profile.get('user_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print(f"\n[4/8] Preparing user data from KG...")
    all_users = list(users_from_kg.values())
    print(f"  [OK] {len(all_users)} users available from KG")

    print(f"\n[5/8] Sampling {TOTAL_SIZE} users for dataset...")
    random.seed(42)

    if len(all_users) < TOTAL_SIZE:
        print(f"  [WARNING] Only {len(all_users)} users available, using all")
        selected_users = all_users
    else:
        selected_users = random.sample(all_users, TOTAL_SIZE)

    # Split into train/val/test
    train_users = selected_users[:TRAIN_SIZE]
    val_users = selected_users[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
    test_users = selected_users[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE]

    print(f"  [OK] Split: {len(train_users)} train / {len(val_users)} val / {len(test_users)} test")

    # Generate datasets (TEST MODE: only train)
    splits = [
        ('train', train_users, f'{OUTPUT_DIR}/task_b_train_from_kg.jsonl'),
        ('val', val_users, f'{OUTPUT_DIR}/task_b_val_from_kg.jsonl'),
        ('test', test_users, f'{OUTPUT_DIR}/task_b_test_from_kg.jsonl'),
    ]

    for split_name, users, output_file in splits:
        print(f"\n[6/8] Generating {split_name} set ({len(users)} samples)...")

        recipes = []
        failed_count = 0

        for user in tqdm(users, desc=f"  Generating {split_name}"):
            recipe = generate_recipe_hybrid(user, seed=0)
            if recipe:
                recipes.append(recipe)
            else:
                failed_count += 1

        print(f"\n[7/8] Saving {split_name} set to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for recipe in recipes:
                f.write(json.dumps(recipe, ensure_ascii=False) + '\n')

        print(f"  [OK] Saved {len(recipes)} recipes ({failed_count} failed)")

    print(f"\n[8/8] Sample output (first recipe from train set):")
    if recipes:
        sample = recipes[0]
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:1500] + "...")

    print(f"\n✅ Done!")
    print("="*80)
    print(f"Generated datasets (Hybrid: KG + External Precise Data):")
    print(f"  📊 From KG: User data, preferences, RNI, co-occurrence, complementarity")
    print(f"  🔬 From External: Precise nutrition calculation, unit conversion")
    print(f"")
    print(f"  Train: {OUTPUT_DIR}task_b_train_from_kg.jsonl")
    print(f"  Val:   {OUTPUT_DIR}task_b_val_from_kg.jsonl")
    print(f"  Test:  {OUTPUT_DIR}task_b_test_from_kg.jsonl")
    print("="*80)
