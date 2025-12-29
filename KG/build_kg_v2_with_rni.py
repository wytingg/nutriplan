#!/usr/bin/env python3
"""

数据源:
- recipes(3column).csv: recipe_id, Name, RecipeIngredientQuantities, RecipeIngredientParts
- recipe_nutrition_foodcom.csv: Food.com原始营养数据(PerServing列)
- updated_user_profile_15nutrients.jsonl: 新的用户数据(含15个RNI营养素)
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

    print("NutriPlan知识图谱")

    # 创建图
    graph = gt.Graph(directed=True)

    # ========================================================================
    # 节点属性设计
    # ========================================================================
    node_type = graph.new_vertex_property("string")    # 节点类型: user/recipe/ingredient/nutrient/nutrient_tag
    node_id = graph.new_vertex_property("string")      # 节点ID
    node_name = graph.new_vertex_property("string")    # 节点名称
    node_unit = graph.new_vertex_property("string")    # 营养素单位(仅Nutrient节点)

    # 用户节点专属属性 (新增)
    user_gender = graph.new_vertex_property("string", val="")          # 性别: male/female
    user_age = graph.new_vertex_property("int", val=0)                 # 年龄
    user_physio_state = graph.new_vertex_property("string", val="")    # 生理状态

    # ========================================================================
    # 边属性设计
    # ========================================================================
    edge_type = graph.new_edge_property("string", val="")      # 边类型

    # Recipe->Ingredient边属性
    qty_raw = graph.new_edge_property("string", val="")        # 原始数量
    unit_raw = graph.new_edge_property("string", val="")       # 原始单位

    # Recipe->Nutrient边属性
    amount_raw = graph.new_edge_property("string", val="")     # 营养素原始值
    amount_unit = graph.new_edge_property("string", val="")    # 营养素单位

    # User->Ingredient边属性
    sign = graph.new_edge_property("int", val=0)              # 偏好符号: +1喜欢, -1不喜欢

    # User->Nutrient边属性(RNI推荐值 - 简化版)
    rni_value = graph.new_edge_property("double", val=0.0)     # RNI推荐值
    rni_unit = graph.new_edge_property("string", val="")       # RNI单位

    # Ingredient->Ingredient边属性(共现和互补规则)
    pmi_score = graph.new_edge_property("double", val=0.0)     # PMI分数(共现)
    cooccurrence_count = graph.new_edge_property("int", val=0) # 共现次数
    confidence = graph.new_edge_property("double", val=0.0)    # 置信度(共现)
    synergy_score = graph.new_edge_property("double", val=0.0) # 协同分数(互补)
    synergy_reason = graph.new_edge_property("string", val="") # 互补原因

    # 节点索引
    vertices = {}  # (node_type, node_id) -> vertex

    def add_node(node_type_val, node_id_val, node_name_val=None, unit_val="", **kwargs):
        """添加节点（避免重复）"""
        key = (node_type_val, node_id_val)
        if key not in vertices:
            vertex = graph.add_vertex()
            node_type[vertex] = node_type_val
            node_id[vertex] = node_id_val
            node_name[vertex] = node_name_val or node_id_val
            node_unit[vertex] = unit_val

            # 处理用户节点的额外属性
            if node_type_val == "user":
                user_gender[vertex] = kwargs.get('gender', '')
                user_age[vertex] = kwargs.get('age', 0)
                user_physio_state[vertex] = kwargs.get('physiological_state', '')

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

    # ========================================================================
    # 1️ 加载食谱数据 - 从recipes(3column).csv
    # ========================================================================
    print("1️ 处理食谱数据...")
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

                # 添加食材节点(使用食材名称作为ID)
                ingredient_vertex = add_node("ingredient", ing_name, ing_name)

                # 添加食谱->食材边
                edge = graph.add_edge(recipe_vertex, ingredient_vertex)
                edge_type[edge] = "recipe_to_ingredient"
                qty_raw[edge] = qty if qty else ""
                unit_raw[edge] = ""  # 数量字符串中已包含单位

    except Exception as e:
        print(f" 食谱数据处理失败: {e}")

    # ========================================================================
    # 2️ 加载营养数据 - 从recipe_nutrition_foodcom.csv
    # ========================================================================
    print(" 2处理营养数据...")
    try:
        nutrients_df = pd.read_csv('work/recipebench/data/4out/recipe_nutrition_foodcom.csv')
        print(f"✓ 加载营养数据: {len(nutrients_df)} 条记录")

        # 定义营养素映射: Food.com列名 -> (营养素显示名, 单位)
        nutrient_mapping = {
            'Calories_PerServing_kcal': ('Energy', 'kcal'),
            'Protein_PerServing_g': ('Protein', 'g'),
            'Fat_PerServing_g': ('Fat', 'g'),
            'Carbohydrates_PerServing_g': ('Carbohydrate', 'g'),
            'Fiber_PerServing_g': ('Fiber', 'g'),
            'Sugars_PerServing_g': ('Added Sugar', 'g'),
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
        print(f" 营养数据处理失败: {e}")
        print(traceback.format_exc())

    # ========================================================================
    # 3️ 加载用户数据 - 从updated_user_profile_15nutrients.jsonl (新格式)
    # ========================================================================
    print("3 处理用户数据（新RNI格式）...")
    try:
        users_data = []
        # 修改为新的用户数据文件路径
        with open('work/recipebench/data/8step_profile/update_cleaned_user_profile.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                users_data.append(json.loads(line))
        print(f"✓ 加载用户数据: {len(users_data)} 条记录")

        # 15个核心营养素映射: 用户数据字段名 -> (Nutrient节点名称, 单位)
        nutrient_rni_mapping = {
            # 宏量营养素
            'energy_kcal': ('Energy', 'kcal'),
            'protein_g': ('Protein', 'g'),
            'carbohydrate_g': ('Carbohydrate', 'g'),
            'fat_g': ('Fat', 'g'),
            'fiber_g': ('Fiber', 'g'),

            # 限制性营养素
            'added_sugar_g': ('Added Sugar', 'g'),
            'saturated_fat_g': ('Saturated Fat', 'g'),
            'trans_fat_g': ('Trans Fat', 'g'),
            'sodium_mg': ('Sodium', 'mg'),

            # 微量营养素
            'potassium_mg': ('Potassium', 'mg'),
            'calcium_mg': ('Calcium', 'mg'),
            'iron_mg': ('Iron', 'mg'),
            'vitamin_c_mg': ('Vitamin C', 'mg'),
            'vitamin_d_ug': ('Vitamin D', 'ug'),
            'folate_ug': ('Folate', 'ug'),
        }

        for user_data in tqdm(users_data, desc="添加用户节点", total=len(users_data)):
            user_id = str(user_data['user_id'])
            gender = user_data.get('gender', '')
            age = user_data.get('age', 0)
            physio_state = user_data.get('physiological_state', '')

            # 添加用户节点（含新属性）
            user_vertex = add_node(
                "user",
                user_id,
                f"User_{user_id}",
                gender=gender,
                age=age,
                physiological_state=physio_state
            )

            # 处理喜欢的食材
            for ing_item in user_data.get('liked_ingredients', []):
                ing_name = ing_item.get('name')
                if ing_name:
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

            # 处理营养RNI目标 - 直接从nutrition_rni读取
            nutrition_rni = user_data.get('nutrition_rni', {})

            for rni_field, (nutrient_name, unit) in nutrient_rni_mapping.items():
                rni_val = nutrition_rni.get(rni_field)

                # 跳过None值
                if rni_val is None:
                    continue

                # 确保营养素节点存在
                nutrient_key = ("nutrient", nutrient_name)
                if nutrient_key not in vertices:
                    # 如果营养素节点不存在，创建它
                    add_node("nutrient", nutrient_name, nutrient_name, unit)

                # 添加User->Nutrient边（RNI推荐值）
                nutrient_vertex = vertices[nutrient_key]
                edge = graph.add_edge(user_vertex, nutrient_vertex)
                edge_type[edge] = "user_to_nutrient_rni"
                rni_value[edge] = float(rni_val)
                rni_unit[edge] = unit

    except FileNotFoundError:
        print(f" 未找到新格式用户数据文件，跳过用户数据处理")
        print(f"   请确保文件路径正确: updated_user_profile_15nutrients.jsonl")
    except Exception as e:
        import traceback
        print(f" 用户数据处理失败: {e}")
        print(traceback.format_exc())

    # ========================================================================
    #  加载食材共现规则
    # ========================================================================
    print("4 处理食材共现规则...")
    try:
        cooccurrence_df = pd.read_csv('work/recipebench/data/11_nutrition_rule/ingredient_cooccurrence_full.csv')
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

    except FileNotFoundError:
        print(f"  未找到共现规则文件，跳过")
    except Exception as e:
        print(f" 共现规则处理失败: {e}")

    # ========================================================================
    #  加载营养互补规则
    # ========================================================================
    print("5 处理营养互补规则...")
    try:
        complementarity_df = pd.read_csv('work/recipebench/data/11_nutrition_rule/nutrition_complementarity_pairs.csv')
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

    except FileNotFoundError:
        print(f"  未找到互补规则文件，跳过")
    except Exception as e:
        print(f" 互补规则处理失败: {e}")

    # ========================================================================
    #  加载食材营养标签
    # ========================================================================
    print(" 处理食材营养标签...")
    try:
        tags_df = pd.read_csv('work/recipebench/data/11_nutrition_rule/ingredient_nutrient_tags.csv')
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

    except FileNotFoundError:
        print(f"  未找到营养标签文件，跳过")
    except Exception as e:
        print(f" 营养标签处理失败: {e}")

    # ========================================================================
    #  设置图属性
    # ========================================================================
    graph.vertex_properties["node_type"] = node_type
    graph.vertex_properties["node_id"] = node_id
    graph.vertex_properties["node_name"] = node_name
    graph.vertex_properties["node_unit"] = node_unit
    graph.vertex_properties["user_gender"] = user_gender
    graph.vertex_properties["user_age"] = user_age
    graph.vertex_properties["user_physio_state"] = user_physio_state

    graph.edge_properties["edge_type"] = edge_type
    graph.edge_properties["qty_raw"] = qty_raw
    graph.edge_properties["unit_raw"] = unit_raw
    graph.edge_properties["amount_raw"] = amount_raw
    graph.edge_properties["amount_unit"] = amount_unit
    graph.edge_properties["sign"] = sign
    graph.edge_properties["rni_value"] = rni_value
    graph.edge_properties["rni_unit"] = rni_unit
    graph.edge_properties["pmi_score"] = pmi_score
    graph.edge_properties["cooccurrence_count"] = cooccurrence_count
    graph.edge_properties["confidence"] = confidence
    graph.edge_properties["synergy_score"] = synergy_score
    graph.edge_properties["synergy_reason"] = synergy_reason

    # ========================================================================
    #  统计信息
    # ========================================================================
    print("\n" + "="*60)
    print(" KG统计 ")
    print("="*60)
    print(f"总节点数: {graph.num_vertices():,}")
    print(f"总边数: {graph.num_edges():,}")

    # 节点类型统计
    node_type_counts = {}
    for v in graph.vertices():
        ntype = node_type[v]
        node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1

    print("\n节点类型分布:")
    for ntype in sorted(node_type_counts.keys()):
        count = node_type_counts[ntype]
        print(f"  {ntype:15s}: {count:,}")

    # 边类型统计
    edge_type_counts = {}
    for e in graph.edges():
        etype = edge_type[e]
        edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

    print("\n边类型分布:")
    for etype in sorted(edge_type_counts.keys()):
        count = edge_type_counts[etype]
        print(f"  {etype:30s}: {count:,}")

    # 用户属性统计
    if node_type_counts.get("user", 0) > 0:
        print("\n用户属性统计:")

        # 性别分布
        gender_counts = {}
        age_list = []
        physio_counts = {}

        for v in graph.vertices():
            if node_type[v] == "user":
                gender = user_gender[v]
                age_val = user_age[v]
                physio = user_physio_state[v]

                if gender:
                    gender_counts[gender] = gender_counts.get(gender, 0) + 1
                if age_val > 0:
                    age_list.append(age_val)
                if physio:
                    physio_counts[physio] = physio_counts.get(physio, 0) + 1

        if gender_counts:
            print("  性别分布:")
            for gender, count in sorted(gender_counts.items()):
                print(f"    {gender}: {count:,} ({count/sum(gender_counts.values())*100:.1f}%)")

        if age_list:
            print(f"  年龄统计:")
            print(f"    平均年龄: {sum(age_list)/len(age_list):.1f} 岁")
            print(f"    年龄范围: {min(age_list)} - {max(age_list)} 岁")

        if physio_counts:
            print("  生理状态分布 (Top 5):")
            for physio, count in sorted(physio_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {physio:20s}: {count:,} ({count/sum(physio_counts.values())*100:.1f}%)")

    # ========================================================================
    #  保存图
    # ========================================================================
    output_path = "work/recipebench/kg/nutriplan_kg4.graphml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    graph.save(output_path)
    print(f"\n KG已保存: {output_path}")
    print("="*60)

    return graph, output_path

if __name__ == "__main__":
    print(" 开始构建知识图谱...")
    print("="*60)
    graph, output_path = build_improved_kg()
    print(f"\n KG构建完成: {output_path}")
