#!/usr/bin/env python3
"""
Fixed NutriPlan Batch Subgraph Generator for  specific KG structure
Adapted to work with user_node, recipe_node properties
"""

import pandas as pd
import numpy as np
import json
import pickle
import graph_tool.all as gt
import os
import h5py
from pathlib import Path
from tqdm import tqdm
import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from dataclasses import dataclass
import h5py
import gc
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

@dataclass
class BatchConfig:
    """批量处理配置"""
    # 批次大小配置
    user_batch_size: int = 10000       # 每批处理的用户数
    recipe_batch_size: int = 20000    # 每批处理的食谱数

    # 并行处理配置
    num_workers: int = 4              # 并行工作进程数
    use_threading: bool = False       # 是否使用线程池(vs进程池)

    # 内存管理配置
    max_memory_gb: float = 8.0        # 最大内存使用(GB)
    enable_gc: bool = True            # 是否启用垃圾回收

    # 缓存配置
    enable_caching: bool = True       # 是否启用缓存
    cache_dir: str = "cache/"         # 缓存目录

class NutriPlanKGProcessor:
    """针对你的KG结构的专用处理器"""

    def __init__(self, graph_path: str, config: BatchConfig):
        self.graph_path = graph_path
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load graph
        self.logger.info(f"加载KG从: {graph_path}")
        self.graph = gt.load_graph(graph_path)
        self.logger.info(f"✓ KG加载完成: {self.graph.num_vertices()}个节点, {self.graph.num_edges()}条边")

        # Build custom indices for your KG structure
        self._build_custom_indices()
    
    def _build_custom_indices(self):
        """为KG结构构建专用索引 - 适配新KG结构"""
        self.logger.info("构建专用查询索引...")

        # Initialize indices
        self.user_ids = []
        self.recipe_ids = []
        self.ingredient_ids = []
        self.nutrient_ids = []

        # 反向索引
        self.id_to_vertex = {}  # (entity_type, entity_id) -> vertex

        # 获取新KG的属性
        node_type_prop = self.graph.vertex_properties.get('node_type')
        node_id_prop = self.graph.vertex_properties.get('node_id')
        node_name_prop = self.graph.vertex_properties.get('node_name')

        if not node_type_prop or not node_id_prop:
            raise ValueError("KG缺少必要属性: node_type或node_id")

        self.logger.info("扫描节点类型...")

        # 遍历节点构建索引
        for v in tqdm(self.graph.vertices(), desc="分析节点"):
            ntype = node_type_prop[v]
            nid = str(node_id_prop[v])

            if ntype == "user":
                self.user_ids.append(nid)
                self.id_to_vertex[('user', nid)] = v
            elif ntype == "recipe":
                self.recipe_ids.append(nid)
                self.id_to_vertex[('recipe', nid)] = v
            elif ntype == "ingredient":
                self.ingredient_ids.append(nid)
                self.id_to_vertex[('ingredient', nid)] = v
            elif ntype == "nutrient":
                self.nutrient_ids.append(nid)
                self.id_to_vertex[('nutrient', nid)] = v

        # ✅ ADD SAMPLING HERE (optional for smaller dataset)
        MAX_USERS = 10000
        MAX_RECIPES = 200000

        if len(self.user_ids) > MAX_USERS:
            import random
            random.seed(42)
            self.user_ids = random.sample(self.user_ids, MAX_USERS)
            self.logger.info(f"📊 用户采样: {MAX_USERS:,}")

        if len(self.recipe_ids) > MAX_RECIPES:
            import random
            random.seed(42)
            self.recipe_ids = random.sample(self.recipe_ids, MAX_RECIPES)
            self.logger.info(f"📊 食谱采样: {MAX_RECIPES:,}")

        self.logger.info("✓ 索引构建完成")
        self.logger.info(f"  - 用户: {len(self.user_ids)}")
        self.logger.info(f"  - 食谱: {len(self.recipe_ids)}")

    def get_user_subgraph(self, user_id: str) -> Dict:
        """获取用户子图 - 适配新KG结构"""
        user_vertex = self.id_to_vertex.get(('user', user_id))

        if not user_vertex:
            return {'user_id': user_id, 'likes': [], 'dislikes': [], 'targets': []}

        # 获取新KG的属性
        node_type_prop = self.graph.vertex_properties.get('node_type')
        node_name_prop = self.graph.vertex_properties.get('node_name')
        edge_type_prop = self.graph.edge_properties.get('edge_type')
        sign_prop = self.graph.edge_properties.get('sign')
        target_raw_prop = self.graph.edge_properties.get('target_raw')

        likes = []
        dislikes = []
        targets = []

        for e in user_vertex.out_edges():
            target = e.target()
            etype = edge_type_prop[e] if edge_type_prop else ""
            target_ntype = node_type_prop[target] if node_type_prop else ""

            # 处理用户-食材边
            if etype == "user_to_ingredient" and target_ntype == "ingredient":
                ingredient_name = node_name_prop[target]
                sign_val = sign_prop[e] if sign_prop else 0

                if sign_val > 0:
                    likes.append(ingredient_name)
                elif sign_val < 0:
                    dislikes.append(ingredient_name)

            # 处理用户-营养目标边
            elif etype == "user_to_nutrient_target" and target_ntype == "nutrient":
                nutrient_name = node_name_prop[target]
                # 可选: 添加目标值信息
                target_val = target_raw_prop[e] if target_raw_prop else ""
                if target_val:
                    targets.append(f"{nutrient_name}:{target_val}")
                else:
                    targets.append(nutrient_name)

        return {
            'user_id': user_id,
            'likes': likes,
            'dislikes': dislikes,
            'targets': targets
        }
    
    def get_recipe_subgraph(self, recipe_id: str) -> Dict:
        """获取食谱子图 - 适配新KG结构"""
        recipe_vertex = self.id_to_vertex.get(('recipe', recipe_id))

        if not recipe_vertex:
            return {'recipe_id': recipe_id, 'ingredients': [], 'nutrients': []}

        # 获取新KG的属性
        node_type_prop = self.graph.vertex_properties.get('node_type')
        node_name_prop = self.graph.vertex_properties.get('node_name')
        edge_type_prop = self.graph.edge_properties.get('edge_type')
        amount_raw_prop = self.graph.edge_properties.get('amount_raw')

        ingredients = []
        nutrients = []

        for e in recipe_vertex.out_edges():
            target = e.target()
            etype = edge_type_prop[e] if edge_type_prop else ""
            target_ntype = node_type_prop[target] if node_type_prop else ""

            # 处理食谱-食材边
            if etype == "recipe_to_ingredient" and target_ntype == "ingredient":
                ingredient_name = node_name_prop[target]
                ingredients.append(ingredient_name)

            # 处理食谱-营养素边
            elif etype == "recipe_to_nutrient" and target_ntype == "nutrient":
                nutrient_name = node_name_prop[target]
                # 可选: 添加营养值信息
                amount_val = amount_raw_prop[e] if amount_raw_prop else ""
                if amount_val:
                    nutrients.append(f"{nutrient_name}:{amount_val}")
                else:
                    nutrients.append(nutrient_name)

        return {
            'recipe_id': recipe_id,
            'ingredients': ingredients,
            'nutrients': nutrients
        }

    def generate_recipe_subgraphs(self, output_path: str):
        """批量生成食谱子图"""
        self.logger.info(f"开始生成食谱子图: {len(self.recipe_ids)}个食谱")

        with h5py.File(output_path, 'w') as f:
            # Create groups
            ingredients_group = f.create_group('ingredients')
            nutrients_group = f.create_group('nutrients')

            # Process recipes in batches
            batch_size = self.config.recipe_batch_size
            num_batches = (len(self.recipe_ids) + batch_size - 1) // batch_size

            for i in tqdm(range(0, len(self.recipe_ids), batch_size),
                         total=num_batches, desc="食谱批次"):
                batch_recipe_ids = self.recipe_ids[i:i+batch_size]

                for recipe_id in batch_recipe_ids:
                    subgraph = self.get_recipe_subgraph(recipe_id)

                    # 🔧 FIX: Handle Unicode encoding properly
                    try:
                        # Clean and encode ingredients
                        clean_ingredients = []
                        for ing in subgraph['ingredients']:
                            if ing:  # Skip empty strings
                                # Encode to UTF-8 bytes, handling unicode characters
                                clean_ingredients.append(str(ing).encode('utf-8', errors='ignore'))

                        # Clean and encode nutrients
                        clean_nutrients = []
                        for nut in subgraph['nutrients']:
                            if nut:  # Skip empty strings
                                clean_nutrients.append(str(nut).encode('utf-8', errors='ignore'))

                        # Save to HDF5 with proper encoding
                        if clean_ingredients:
                            ingredients_group.create_dataset(recipe_id, data=np.array(clean_ingredients, dtype='S200'))
                        else:
                            ingredients_group.create_dataset(recipe_id, data=np.array([], dtype='S200'))

                        if clean_nutrients:
                            nutrients_group.create_dataset(recipe_id, data=np.array(clean_nutrients, dtype='S200'))
                        else:
                            nutrients_group.create_dataset(recipe_id, data=np.array([], dtype='S200'))

                    except Exception as e:
                        self.logger.warning(f"跳过食谱 {recipe_id}: {e}")
                        # Create empty datasets for failed recipes
                        ingredients_group.create_dataset(recipe_id, data=np.array([], dtype='S200'))
                        nutrients_group.create_dataset(recipe_id, data=np.array([], dtype='S200'))

        self.logger.info(f"✓ 食谱子图生成完成: {output_path}")

    def generate_user_subgraphs(self, output_path: str):
        """批量生成用户子图"""
        self.logger.info(f"开始生成用户子图: {len(self.user_ids)}个用户")

        with h5py.File(output_path, 'w') as f:
            # Create groups
            likes_group = f.create_group('likes')
            dislikes_group = f.create_group('dislikes')
            targets_group = f.create_group('targets')

            # Process users in batches
            batch_size = self.config.user_batch_size
            num_batches = (len(self.user_ids) + batch_size - 1) // batch_size

            for i in tqdm(range(0, len(self.user_ids), batch_size),
                         total=num_batches, desc="用户批次"):
                batch_user_ids = self.user_ids[i:i+batch_size]

                for user_id in batch_user_ids:
                    subgraph = self.get_user_subgraph(user_id)

                    # 🔧 FIX: Handle Unicode encoding properly
                    try:
                        # Clean and encode data
                        clean_likes = [str(item).encode('utf-8', errors='ignore') for item in subgraph['likes'] if item]
                        clean_dislikes = [str(item).encode('utf-8', errors='ignore') for item in subgraph['dislikes'] if item]
                        clean_targets = [str(item).encode('utf-8', errors='ignore') for item in subgraph['targets'] if item]

                        # Save to HDF5
                        likes_group.create_dataset(user_id, data=np.array(clean_likes, dtype='S200'))
                        dislikes_group.create_dataset(user_id, data=np.array(clean_dislikes, dtype='S200'))
                        targets_group.create_dataset(user_id, data=np.array(clean_targets, dtype='S200'))

                    except Exception as e:
                        self.logger.warning(f"跳过用户 {user_id}: {e}")
                        # Create empty datasets for failed users
                        likes_group.create_dataset(user_id, data=np.array([], dtype='S200'))
                        dislikes_group.create_dataset(user_id, data=np.array([], dtype='S200'))
                        targets_group.create_dataset(user_id, data=np.array([], dtype='S200'))

        self.logger.info(f"✓ 用户子图生成完成: {output_path}")

def main():
    """主函数"""
    # Configuration
    config = BatchConfig(
        user_batch_size=5000,
        recipe_batch_size=10000,
        num_workers=4
    )

    # Paths - 使用新生成的KG
    kg_path = "work/recipebench/kg/nutriplan_kg2.graphml"
    output_dir = Path("work/recipebench/data/9large_scale_subgraphs/")
    output_dir.mkdir(parents=True, exist_ok=True)

    user_output = output_dir / "user_subgraphs1.h5"
    recipe_output = output_dir / "recipe_subgraphs1.h5"

    # Process
    processor = NutriPlanKGProcessor(kg_path, config)

    # Generate subgraphs
    processor.generate_user_subgraphs(str(user_output))
    processor.generate_recipe_subgraphs(str(recipe_output))

    print(f"✓ 子图生成完成:")
    print(f"  - 用户子图: {user_output}")
    print(f"  - 食谱子图: {recipe_output}")
    print(f"  - 处理用户数: {len(processor.user_ids)}")
    print(f"  - 处理食谱数: {len(processor.recipe_ids)}")

if __name__ == "__main__":
    main()

class SubgraphLoader:
    """子图加载器，用于加载Step2生成的H5文件"""

    def __init__(self, user_subgraphs_path: str, recipe_subgraphs_path: str):
        self.user_subgraphs_path = user_subgraphs_path
        self.recipe_subgraphs_path = recipe_subgraphs_path
        self.logger = logging.getLogger(__name__)

    def load_user_subgraph(self, user_id: str) -> Optional[Dict]:
        """加载指定用户的子图数据"""
        try:
            with h5py.File(self.user_subgraphs_path, 'r') as f:
                if 'likes' in f and user_id in f['likes']:
                    # 解码字节数据到字符串
                    likes = [item.decode('utf-8') for item in f['likes'][user_id][:]]
                    dislikes = [item.decode('utf-8') for item in f['dislikes'][user_id][:]] if 'dislikes' in f and user_id in f['dislikes'] else []
                    targets = [item.decode('utf-8') for item in f['targets'][user_id][:]] if 'targets' in f and user_id in f['targets'] else []

                    return {
                        'likes': likes,
                        'dislikes': dislikes,
                        'targets': targets
                    }
        except Exception as e:
            self.logger.error(f"加载用户子图失败 {user_id}: {e}")
        return None

    def load_recipe_subgraph(self, recipe_id: str) -> Optional[Dict]:
        """加载指定食谱的子图数据"""
        try:
            with h5py.File(self.recipe_subgraphs_path, 'r') as f:
                if 'ingredients' in f and recipe_id in f['ingredients']:
                    # 解码字节数据到字符串
                    ingredients = [item.decode('utf-8') for item in f['ingredients'][recipe_id][:]]
                    nutrients = [item.decode('utf-8') for item in f['nutrients'][recipe_id][:]] if 'nutrients' in f and recipe_id in f['nutrients'] else []

                    return {
                        'ingredients': ingredients,
                        'nutrients': nutrients
                    }
        except Exception as e:
            self.logger.error(f"加载食谱子图失败 {recipe_id}: {e}")
        return None