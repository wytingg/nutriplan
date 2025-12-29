#!/usr/bin/env python3
"""
改进版KG构建器 - 基于真实数据结构重构 + 规则库集成
数据源:
- recipes(3column).csv: recipe_id, Name, RecipeIngredientQuantities, RecipeIngredientParts
- recipe_nutrition_foodcom.csv: Food.com原始营养数据(PerServing列)
- cleaned_user_profile.jsonl: user_id, liked/disliked_ingredients, nutrition_targets
- ingredient_cooccurrence.csv: 食材共现规则(12,619条)
- nutrition_complementarity_pairs.csv: 营养互补规则(45,928条)
- ingredient_nutrient_tags.csv: 食材营养标签(1,055个食材)
"""

import pandas as pd
import json
import ast
import graph_tool.all as gt
import os
from tqdm import tqdm
from pathlib import Path
import re

def build_improved_kg():
    """构建改进版知识图谱"""
    print("🔧 构建改进版NutriPlan知识图谱（含规则库）...")

    # 创建图
    graph = gt.Graph(directed=True)

    # 节点属性设计
    node_type = graph.new_vertex_property("string")    # 节点类型: user/recipe/ingredient/nutrient/nutrient_tag
    node_id = graph.new_vertex_property("string")      # 节点ID
    node_name = graph.new_vertex_property("string")    # 节点名称
    node_unit = graph.new_vertex_property("string")    # 营养素单位(仅Nutrient节点)

    # 边属性设计(所有属性带默认值)
    edge_type = graph.new_edge_property("string", val="")      # 边类型

    # Recipe->Ingredient边属性
    qty_raw = graph.new_edge_property("string", val="")        # 原始数量
    unit_raw = graph.new_edge_property("string", val="")       # 原始单位

    # Recipe->Nutrient边属性
    amount_raw = graph.new_edge_property("string", val="")     # 营养素原始值
    amount_unit = graph.new_edge_property("string", val="")    # 营养素单位

    # User->Ingredient边属性
    sign = graph.new_edge_property("int", val=0)              # 偏好符号: +1喜欢, -1不喜欢

    # User->Nutrient边属性(营养目标)
    target_raw = graph.new_edge_property("string", val="")     # 目标原始值
    target_unit = graph.new_edge_property("string", val="")    # 目标单位
    target_pct = graph.new_edge_property("string", val="")     # 百分比目标
    target_grams = graph.new_edge_property("string", val="")   # 克数目标
    target_type = graph.new_edge_property("string", val="")    # 目标类型: min/max/range

    # Ingredient->Ingredient边属性(共现和互补规则)
    pmi_score = graph.new_edge_property("double", val=0.0)     # PMI分数(共现)
    cooccurrence_count = graph.new_edge_property("int", val=0) # 共现次数
    confidence = graph.new_edge_property("double", val=0.0)    # 置信度(共现)
    synergy_score = graph.new_edge_property("double", val=0.0) # 协同分数(互补)
    synergy_reason = graph.new_edge_property("string", val="") # 互补原因

    # 节点索引
    vertices = {}  # (node_type, node_id) -> vertex

    def add_node(node_type_val, node_id_val, node_name_val=None, unit_val=""):
        """添加节点（避免重复）"""
        key = (node_type_val, node_id_val)
        if key not in vertices:
            vertex = graph.add_vertex()
            node_type[vertex] = node_type_val
            node_id[vertex] = node_id_val
            node_name[vertex] = node_name_val or node_id_val
            node_unit[vertex] = unit_val
            vertices[key] = vertex
        return vertices[key]

    def parse_r_vector(r_string):
        """解析R语言c()向量字符串"""
        if pd.isna(r_string):
            return []
        r_string = str(r_string).strip()
        if not r_string.startswith('c('):
            return []
        # 提取c()内的内容
        content = r_string[2:-1]
        # 使用正则表达式匹配引号内的字符串
        items = re.findall(r'"([^"]*)"', content)
        return items

    # 1️⃣ 加载食谱数据 - 从recipes(3column).csv
    print("1️⃣ 处理食谱数据...")
    try:
        recipes_df = pd.read_csv('work/recipebench/data/raw/foodcom/recipes(3column).csv', encoding='latin-1')
        print(f"✓ 加载食谱数据: {len(recipes_df)} 条记录")

        for _, row in tqdm(recipes_df.iterrows(), desc="添加食谱和食材", total=len(recipes_df)):
            recipe_id = str(row['recipe_id'])
            recipe_name_val = str(row['Name']) if pd.notna(row['Name']) else f"Recipe_{recipe_id}"

            # 添加食谱节点
            recipe_vertex = add_node("recipe", recipe_id, recipe_name_val)

            # 解析食材数量和名称
            quantities = parse_r_vector(row['RecipeIngredientQuantities'])
            ingredients = parse_r_vector(row['RecipeIngredientParts'])

            # 处理食谱-食材边
            for qty, ing_name in zip(quantities, ingredients):
                if not ing_name:
                    continue

                # 添加食材节点(使用食材名称作为ID,因为数据中没有ingredient_id)
                ingredient_vertex = add_node("ingredient", ing_name, ing_name)

                # 添加食谱->食材边
                edge = graph.add_edge(recipe_vertex, ingredient_vertex)
                edge_type[edge] = "recipe_to_ingredient"
                qty_raw[edge] = qty if qty else ""
                unit_raw[edge] = ""  # 数量字符串中已包含单位

    except Exception as e:
        print(f"❌ 食谱数据处理失败: {e}")

    # 2️⃣ 加载营养数据 - 从recipe_nutrition_foodcom.csv
    print("2️⃣ 处理营养数据...")
    try:
        nutrients_df = pd.read_csv('recipe_nutrition_foodcom.csv')
        print(f"✓ 加载营养数据: {len(nutrients_df)} 条记录")

        # 定义营养素映射: Food.com列名 -> (营养素显示名, 单位)
        nutrient_mapping = {
            'Calories_PerServing_kcal': ('Calories', 'kcal'),
            'Protein_PerServing_g': ('Protein', 'g'),
            'Fat_PerServing_g': ('Fat', 'g'),
            'Carbohydrates_PerServing_g': ('Carbohydrates', 'g'),
            'Fiber_PerServing_g': ('Fiber', 'g'),
            'Sugars_PerServing_g': ('Sugars', 'g'),
            'SaturatedFat_PerServing_g': ('Saturated Fat', 'g'),
            'Sodium_PerServing_mg': ('Sodium', 'mg'),
            'Cholesterol_PerServing_mg': ('Cholesterol', 'mg'),
        }

        for _, row in tqdm(nutrients_df.iterrows(), desc="添加营养边", total=len(nutrients_df)):
            recipe_id = str(row['recipe_id'])

            # 确保食谱节点存在
            recipe_key = ("recipe", recipe_id)
            if recipe_key not in vertices:
                continue

            recipe_vertex = vertices[recipe_key]

            # 处理Food.com营养数据(仅PerServing列)
            for foodcom_col, (nutrient_name, unit_value) in nutrient_mapping.items():
                if foodcom_col not in nutrients_df.columns:
                    continue

                nutrient_value = row[foodcom_col]

                if pd.notna(nutrient_value) and float(nutrient_value) > 0:
                    # 添加营养素节点
                    nutrient_vertex = add_node("nutrient", nutrient_name, nutrient_name, unit_value)

                    # 添加食谱->营养素边
                    edge = graph.add_edge(recipe_vertex, nutrient_vertex)
                    edge_type[edge] = "recipe_to_nutrient"
                    amount_raw[edge] = str(nutrient_value)
                    amount_unit[edge] = unit_value

    except Exception as e:
        import traceback
        print(f"❌ 营养数据处理失败: {e}")
        print(traceback.format_exc())

    # 3️⃣ 加载用户数据 - 从cleaned_user_profile.jsonl
    print("3️⃣ 处理用户数据...")
    try:
        users_data = []
        with open('work/recipebench/data/8step_profile/cleaned_user_profile.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                users_data.append(json.loads(line))
        print(f"✓ 加载用户数据: {len(users_data)} 条记录")

        # 8维核心营养素映射到Nutrient节点
        core_nutrients_map = {
            'energy_kcal': 'Calories',
            'carb': 'Carbohydrates',
            'protein': 'Protein',
            'fat': 'Fat',
            'sodium_mg': 'Sodium',
            'sugars': 'Sugars',
            'sat_fat': 'Saturated Fat',
            'fiber_g': 'Fiber'
        }

        for user_data in tqdm(users_data, desc="添加用户节点"):
            user_id = str(user_data['user_id'])

            # 添加用户节点
            user_vertex = add_node("user", user_id, f"User_{user_id}")

            # 处理喜欢的食材
            for ing_item in user_data.get('liked_ingredients', []):
                ing_name = ing_item.get('name')
                if ing_name:
                    # 使用食材名称作为key查找
                    ingredient_key = ("ingredient", ing_name)
                    if ingredient_key in vertices:
                        ingredient_vertex = vertices[ingredient_key]
                        edge = graph.add_edge(user_vertex, ingredient_vertex)
                        edge_type[edge] = "user_to_ingredient"
                        sign[edge] = 1  # 喜欢

            # 处理不喜欢的食材
            for ing_item in user_data.get('disliked_ingredients', []):
                ing_name = ing_item.get('name')
                if ing_name:
                    ingredient_key = ("ingredient", ing_name)
                    if ingredient_key in vertices:
                        ingredient_vertex = vertices[ingredient_key]
                        edge = graph.add_edge(user_vertex, ingredient_vertex)
                        edge_type[edge] = "user_to_ingredient"
                        sign[edge] = -1  # 不喜欢

            # 处理营养目标 - 仅连接到8维核心营养素
            nutrition_targets = user_data.get('nutrition_targets', {})

            # 能量目标
            energy_target = nutrition_targets.get('energy_kcal_target')
            if energy_target:
                nutrient_key = ("nutrient", "Calories")
                if nutrient_key in vertices:
                    edge = graph.add_edge(user_vertex, vertices[nutrient_key])
                    edge_type[edge] = "user_to_nutrient_target"
                    target_raw[edge] = str(energy_target)
                    target_unit[edge] = "kcal"
                    target_type[edge] = "target"

            # AMDR三大营养素百分比目标
            amdr = nutrition_targets.get('amdr', {})
            for key, nutrient_name in [('carb', 'Carbohydrates'), ('protein', 'Protein'), ('fat', 'Fat')]:
                if key in amdr:
                    nutrient_key = ("nutrient", nutrient_name)
                    if nutrient_key in vertices:
                        edge = graph.add_edge(user_vertex, vertices[nutrient_key])
                        edge_type[edge] = "user_to_nutrient_target"
                        target_pct[edge] = str(amdr[key].get('target_pct', ''))

                        # 计算克数目标
                        if energy_target and 'target_pct' in amdr[key]:
                            pct = amdr[key]['target_pct']
                            if key == 'carb':
                                grams = (energy_target * pct / 100) / 4
                            elif key == 'protein':
                                grams = (energy_target * pct / 100) / 4
                            else:  # fat
                                grams = (energy_target * pct / 100) / 9
                            target_grams[edge] = str(round(grams, 1))

                        # 目标类型(范围)
                        if 'min_pct' in amdr[key] and 'max_pct' in amdr[key]:
                            target_type[edge] = "range"
                            target_raw[edge] = f"{amdr[key]['min_pct']}-{amdr[key]['max_pct']}"
                            target_unit[edge] = "%"

            # 钠最大值
            sodium_max = nutrition_targets.get('sodium_mg_max')
            if sodium_max:
                nutrient_key = ("nutrient", "Sodium")
                if nutrient_key in vertices:
                    edge = graph.add_edge(user_vertex, vertices[nutrient_key])
                    edge_type[edge] = "user_to_nutrient_target"
                    target_raw[edge] = str(sodium_max)
                    target_unit[edge] = "mg"
                    target_type[edge] = "max"

            # 糖分百分比最大值
            sugars = nutrition_targets.get('sugars', {})
            if 'pct_max' in sugars:
                nutrient_key = ("nutrient", "Sugars")
                if nutrient_key in vertices:
                    edge = graph.add_edge(user_vertex, vertices[nutrient_key])
                    edge_type[edge] = "user_to_nutrient_target"
                    target_raw[edge] = str(sugars['pct_max'])
                    target_unit[edge] = "%"
                    target_type[edge] = "max"

            # 饱和脂肪百分比最大值
            sat_fat_max = nutrition_targets.get('sat_fat_pct_max')
            if sat_fat_max:
                nutrient_key = ("nutrient", "Saturated Fat")
                if nutrient_key in vertices:
                    edge = graph.add_edge(user_vertex, vertices[nutrient_key])
                    edge_type[edge] = "user_to_nutrient_target"
                    target_raw[edge] = str(sat_fat_max)
                    target_unit[edge] = "%"
                    target_type[edge] = "max"

            # 纤维最小值
            fiber_min = nutrition_targets.get('fiber_g_min')
            if fiber_min:
                nutrient_key = ("nutrient", "Fiber")
                if nutrient_key in vertices:
                    edge = graph.add_edge(user_vertex, vertices[nutrient_key])
                    edge_type[edge] = "user_to_nutrient_target"
                    target_raw[edge] = str(fiber_min)
                    target_unit[edge] = "g"
                    target_type[edge] = "min"

    except Exception as e:
        print(f"❌ 用户数据处理失败: {e}")

    # 4️⃣ 加载食材共现规则
    print("4️⃣ 处理食材共现规则...")
    try:
        cooccurrence_df = pd.read_csv('ingredient_cooccurrence.csv')
        print(f"✓ 加载共现规则: {len(cooccurrence_df)} 条记录")

        cooccurrence_added = 0
        for _, row in tqdm(cooccurrence_df.iterrows(), desc="添加共现规则", total=len(cooccurrence_df)):
            ing1_name = row['ingredient_1']
            ing2_name = row['ingredient_2']

            # 检查两个食材节点是否存在
            ing1_key = ("ingredient", ing1_name)
            ing2_key = ("ingredient", ing2_name)

            if ing1_key in vertices and ing2_key in vertices:
                ing1_vertex = vertices[ing1_key]
                ing2_vertex = vertices[ing2_key]

                # 添加双向边(共现关系是对称的)
                edge1 = graph.add_edge(ing1_vertex, ing2_vertex)
                edge_type[edge1] = "ingredient_cooccurs"
                pmi_score[edge1] = float(row['pmi_score'])
                cooccurrence_count[edge1] = int(row['cooccurrence_count'])
                confidence[edge1] = float(row['confidence'])

                edge2 = graph.add_edge(ing2_vertex, ing1_vertex)
                edge_type[edge2] = "ingredient_cooccurs"
                pmi_score[edge2] = float(row['pmi_score'])
                cooccurrence_count[edge2] = int(row['cooccurrence_count'])
                confidence[edge2] = float(row['confidence'])

                cooccurrence_added += 2

        print(f"  ✓ 添加共现边: {cooccurrence_added} 条")

    except Exception as e:
        print(f"❌ 共现规则处理失败: {e}")

    # 5️⃣ 加载营养互补规则
    print("5️⃣ 处理营养互补规则...")
    try:
        complementarity_df = pd.read_csv('nutrition_complementarity_pairs.csv')
        print(f"✓ 加载互补规则: {len(complementarity_df)} 条记录")

        complementarity_added = 0
        for _, row in tqdm(complementarity_df.iterrows(), desc="添加互补规则", total=len(complementarity_df)):
            ing1_name = row['ingredient_1']
            ing2_name = row['ingredient_2']

            # 检查两个食材节点是否存在
            ing1_key = ("ingredient", ing1_name)
            ing2_key = ("ingredient", ing2_name)

            if ing1_key in vertices and ing2_key in vertices:
                ing1_vertex = vertices[ing1_key]
                ing2_vertex = vertices[ing2_key]

                # 添加双向边(互补关系是对称的)
                edge1 = graph.add_edge(ing1_vertex, ing2_vertex)
                edge_type[edge1] = "ingredient_complements"
                synergy_score[edge1] = float(row['synergy_score'])
                synergy_reason[edge1] = str(row['reason'])

                edge2 = graph.add_edge(ing2_vertex, ing1_vertex)
                edge_type[edge2] = "ingredient_complements"
                synergy_score[edge2] = float(row['synergy_score'])
                synergy_reason[edge2] = str(row['reason'])

                complementarity_added += 2

        print(f"  ✓ 添加互补边: {complementarity_added} 条")

    except Exception as e:
        print(f"❌ 互补规则处理失败: {e}")

    # 6️⃣ 加载食材营养标签
    print("6️⃣ 处理食材营养标签...")
    try:
        tags_df = pd.read_csv('ingredient_nutrient_tags.csv')
        print(f"✓ 加载营养标签: {len(tags_df)} 条记录")

        tags_added = 0
        for _, row in tqdm(tags_df.iterrows(), desc="添加营养标签", total=len(tags_df)):
            ing_name = row['ingredient']
            tag_name = row['nutrient_tag']

            # 检查食材节点是否存在
            ing_key = ("ingredient", ing_name)

            if ing_key in vertices:
                ing_vertex = vertices[ing_key]

                # 添加营养标签节点
                tag_vertex = add_node("nutrient_tag", tag_name, tag_name)

                # 添加食材->标签边
                edge = graph.add_edge(ing_vertex, tag_vertex)
                edge_type[edge] = "ingredient_has_tag"
                tags_added += 1

        print(f"  ✓ 添加标签边: {tags_added} 条")

    except Exception as e:
        print(f"❌ 营养标签处理失败: {e}")

    # 7️⃣ 设置图属性
    graph.vertex_properties["node_type"] = node_type
    graph.vertex_properties["node_id"] = node_id
    graph.vertex_properties["node_name"] = node_name
    graph.vertex_properties["node_unit"] = node_unit

    graph.edge_properties["edge_type"] = edge_type
    graph.edge_properties["qty_raw"] = qty_raw
    graph.edge_properties["unit_raw"] = unit_raw
    graph.edge_properties["amount_raw"] = amount_raw
    graph.edge_properties["amount_unit"] = amount_unit
    graph.edge_properties["sign"] = sign
    graph.edge_properties["target_raw"] = target_raw
    graph.edge_properties["target_unit"] = target_unit
    graph.edge_properties["target_pct"] = target_pct
    graph.edge_properties["target_grams"] = target_grams
    graph.edge_properties["target_type"] = target_type
    graph.edge_properties["pmi_score"] = pmi_score
    graph.edge_properties["cooccurrence_count"] = cooccurrence_count
    graph.edge_properties["confidence"] = confidence
    graph.edge_properties["synergy_score"] = synergy_score
    graph.edge_properties["synergy_reason"] = synergy_reason

    # 8️⃣ 统计信息
    print("\n📊 改进版KG统计:")
    print(f"总节点数: {graph.num_vertices()}")
    print(f"总边数: {graph.num_edges()}")

    # 节点类型统计
    node_type_counts = {}
    for v in graph.vertices():
        ntype = node_type[v]
        node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1

    print("节点类型分布:")
    for ntype, count in node_type_counts.items():
        print(f"  {ntype}: {count}")

    # 边类型统计
    edge_type_counts = {}
    for e in graph.edges():
        etype = edge_type[e]
        edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

    print("边类型分布:")
    for etype, count in edge_type_counts.items():
        print(f"  {etype}: {count}")

    # 9️⃣ 保存图
    output_path = "fixed_knowledge_graph/nutriplan_kg_with_rules.graphml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    graph.save(output_path)
    print(f"\n✅ 改进版KG已保存: {output_path}")

    return graph, output_path

if __name__ == "__main__":
    graph, output_path = build_improved_kg()
    print(f"🎉 改进版KG构建完成: {output_path}")
