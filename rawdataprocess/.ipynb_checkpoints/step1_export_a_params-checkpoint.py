# -*- coding: utf-8 -*-
"""
export_a_params.py — A表参数导出脚本

功能概述：
    将A表的外键优势转为可复用的lookup表，为后续Step实现ingredient和fdc_id对齐时
    提供精准的克重换算参数。本脚本独立于Step1主流程，不污染核心处理逻辑。

核心功能：
    1. fdc级体积密度（g/ml）：按fdc_id聚合体积单位的密度数据
    2. fdc×unit级件数克重（g/件）：按(fdc_id, unit)聚合件数单位的重量数据
    3. 全局中位数（兜底）：按单位聚合的全局转换系数
    4. 类别/Token聚合（次优回退）：按food_category_id和token聚合的转换参数

输出文件：
    - A_fdc_volume_density.parquet: fdc级体积密度表
    - A_fdc_piece_weight.parquet: fdc×unit级件数克重表
    - A_unit_global_median.parquet: 全局中位数表
    - A_cat_token_aggregates.parquet: 类别/Token聚合表

使用方法：
    python work/recipebench/scripts/rawdataprocess/step1_export_a_params.py \
      --a_table_path work/recipebench/data/3out/household_weights_A.csv \
      --food_path /path/to/food_processed.parquet \
      --out_dir /path/to/output/params

设计考虑：
    - 使用winsorize去极值处理，提高数据质量
    - 支持体积→密度和件数→重量的分层映射策略
    - 提供fdc_id、category、token三个层级的回退机制
    - 输出parquet格式，便于后续高效查询
"""

import os
import argparse
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from common_utils import tokenize

# =============================================================================
# 单位映射配置
# =============================================================================

# 体积单位到毫升的转换系数
ML_PER = {
    'tsp': 4.92892,        # 茶匙到毫升
    'tbsp': 14.7868,       # 汤匙到毫升
    'cup': 236.588,        # 杯到毫升
    'fl_oz': 29.5735,      # 液体盎司到毫升
    'pt': 473.176,         # 品脱到毫升
    'qt': 946.353,         # 夸脱到毫升
    'l': 1000,             # 升到毫升
    'ml': 1,               # 毫升到毫升（基准）
    'gallon': 3785.41,     # 加仑到毫升
    'pint': 473.176,       # 品脱到毫升
    'quart': 946.353,      # 夸脱到毫升
    'liter': 1000,         # 升到毫升
    'milliliter': 1,       # 毫升到毫升
    'fluid_ounce': 29.5735, # 液体盎司到毫升
    'teaspoon': 4.92892,   # 茶匙到毫升
    'tablespoon': 14.7868, # 汤匙到毫升
}

# 件数单位集合
PIECE_UNITS = {
    'piece', 'slice', 'sheet', 'clove', 'stick', 'can', 'package', 'packet',
    'serving', 'head', 'jar', 'bag', 'bunch', 'sprig', 'egg', 'drumstick', 
    'thigh', 'steak', 'stalk', 'link', 'banana', 'spear', 'bottle', 'box',
    'carton', 'container', 'bar', 'fillet', 'breast', 'wing', 'rib'
}

# =============================================================================
# 导出函数实现
# =============================================================================

def export_a_fdc_volume_density(a_df, ml_map, out_path):
    """
    导出fdc级体积密度（g/ml）
    
    功能说明：
        1. 筛选体积单位记录
        2. 计算密度 = grams_per_unit / ml_per_unit
        3. 按fdc_id进行winsorize去极值处理
        4. 取中位数作为最终密度值
        5. 记录样本数量用于质量评估
    
    参数：
        a_df (pd.DataFrame): A表数据
        ml_map (dict): 体积单位到毫升的转换映射
        out_path (str): 输出文件路径
    
    输出列：
        - fdc_id: 食物ID
        - density_g_per_ml: 密度（克/毫升）
        - n: 样本数量
    """
    print("   正在导出fdc级体积密度...")
    
    # 仅取体积单位记录
    v = a_df[a_df["unit_std"].isin(ml_map.keys())].copy()
    if v.empty:
        print("   ⚠️  没有体积单位数据，跳过导出")
        return
    
    # 计算毫升转换系数
    v["ml_per_unit"] = v["unit_std"].map(ml_map).astype("float64")
    v = v[v["ml_per_unit"] > 0]
    
    # 计算密度
    v["density_g_per_ml"] = v["grams_per_unit"].astype("float64") / v["ml_per_unit"]
    
    # 过滤异常值（密度范围：0.1-3.0 g/ml）
    v = v[(v["density_g_per_ml"] >= 0.1) & (v["density_g_per_ml"] <= 3.0)]
    
    if v.empty:
        print("   ⚠️  过滤后没有有效密度数据，跳过导出")
        return
    
    # winsorize per fdc_id（2.5%-97.5%分位数）
    def _winsorize_median(x):
        if len(x) < 3:
            return x.median()
        q025, q975 = x.quantile([0.025, 0.975])
        return x.clip(lower=q025, upper=q975).median()
    
    g = (v.groupby("fdc_id")["density_g_per_ml"]
           .apply(_winsorize_median)
           .reset_index(name="density_g_per_ml"))
    
    # 记录样本数量
    n = v.groupby("fdc_id").size().reset_index(name="n")
    out = g.merge(n, on="fdc_id", how="left")
    
    # 保存结果
    out.to_parquet(out_path, index=False)
    print(f"   ✅ 导出完成: {len(out)} 个fdc_id，保存到 {out_path}")
    
    # 统计信息
    print(f"   📊 密度统计: min={out['density_g_per_ml'].min():.3f}, "
          f"median={out['density_g_per_ml'].median():.3f}, "
          f"max={out['density_g_per_ml'].max():.3f}")

def export_a_fdc_piece_weight(a_df, piece_units, out_path):
    """
    导出fdc×unit级件数克重（g/件）
    
    功能说明：
        1. 筛选件数单位记录
        2. 按(fdc_id, unit)进行winsorize去极值处理
        3. 取中位数作为最终件重值
        4. 记录样本数量用于质量评估
    
    参数：
        a_df (pd.DataFrame): A表数据
        piece_units (set): 件数单位集合
        out_path (str): 输出文件路径
    
    输出列：
        - fdc_id: 食物ID
        - unit_std: 标准化单位
        - grams_per_unit_clean: 每件克重（去极值后）
        - n: 样本数量
    """
    print("   正在导出fdc×unit级件数克重...")
    
    # 筛选件数单位记录
    p = a_df[a_df["unit_std"].isin(piece_units)].copy()
    if p.empty:
        print("   ⚠️  没有件数单位数据，跳过导出")
        return
    
    # 过滤异常值（件重范围：0.1-1000g）
    p = p[(p["grams_per_unit"] >= 0.1) & (p["grams_per_unit"] <= 1000)]
    
    if p.empty:
        print("   ⚠️  过滤后没有有效件重数据，跳过导出")
        return
    
    # winsorize per (fdc_id, unit)
    def _winsorize_median(x):
        if len(x) < 3:
            return x.median()
        q025, q975 = x.quantile([0.025, 0.975])
        return x.clip(lower=q025, upper=q975).median()
    
    g = (p.groupby(["fdc_id", "unit_std"])["grams_per_unit"]
           .apply(_winsorize_median)
           .reset_index(name="grams_per_unit_clean"))
    
    # 记录样本数量
    n = p.groupby(["fdc_id", "unit_std"]).size().reset_index(name="n")
    out = g.merge(n, on=["fdc_id", "unit_std"], how="left")
    
    # 保存结果
    out.to_parquet(out_path, index=False)
    print(f"   ✅ 导出完成: {len(out)} 个(fdc_id, unit)对，保存到 {out_path}")
    
    # 统计信息
    print(f"   📊 件重统计: min={out['grams_per_unit_clean'].min():.1f}g, "
          f"median={out['grams_per_unit_clean'].median():.1f}g, "
          f"max={out['grams_per_unit_clean'].max():.1f}g")

def export_a_unit_global_median(a_df, out_path):
    """
    导出全局中位数（兜底）
    
    功能说明：
        1. 按单位聚合所有记录
        2. 计算每个单位的中位数转换系数
        3. 记录样本数量用于质量评估
        4. 作为最终兜底方案使用
    
    参数：
        a_df (pd.DataFrame): A表数据
        out_path (str): 输出文件路径
    
    输出列：
        - unit_std: 标准化单位
        - grams_per_unit_global_median: 全局中位数转换系数
        - n: 样本数量
    """
    print("   正在导出全局中位数...")
    
    # 过滤异常值
    a_clean = a_df[(a_df["grams_per_unit"] >= 0.1) & (a_df["grams_per_unit"] <= 10000)]
    
    if a_clean.empty:
        print("   ⚠️  没有有效数据，跳过导出")
        return
    
    # 按单位聚合中位数
    g = (a_clean.groupby("unit_std")["grams_per_unit"]
           .median().reset_index()
           .rename(columns={"grams_per_unit": "grams_per_unit_global_median"}))
    
    # 记录样本数量
    n = a_clean.groupby("unit_std").size().reset_index(name="n")
    out = g.merge(n, on="unit_std", how="left")
    
    # 保存结果
    out.to_parquet(out_path, index=False)
    print(f"   ✅ 导出完成: {len(out)} 个单位，保存到 {out_path}")
    
    # 统计信息
    print(f"   📊 单位覆盖: {len(out)} 种单位，总样本数 {out['n'].sum():,}")

def export_a_cat_token_aggregates(a_df, food_df, ml_map, piece_units, out_path):
    """
    导出类别/Token聚合（次优回退）
    
    功能说明：
        1. 将A表连接到USDA food表获取category和description信息
        2. 按food_category_id聚合体积密度和件数重量
        3. 从description_norm提取token，按token聚合
        4. 统一输出格式，支持多层级回退查询
    
    参数：
        a_df (pd.DataFrame): A表数据
        food_df (pd.DataFrame): 食物信息表
        ml_map (dict): 体积单位到毫升的转换映射
        piece_units (set): 件数单位集合
        out_path (str): 输出文件路径
    
    输出列：
        - key: 聚合键（category_id或token）
        - key_type: 键类型（'cat'或'token'）
        - unit_std: 标准化单位
        - value: 转换值（密度或重量）
        - n: 样本数量
    """
    print("   正在导出类别/Token聚合...")
    
    # 连接A表和food表
    m = a_df.merge(
        food_df[["fdc_id", "food_category_id", "description_norm"]], 
        on="fdc_id", how="inner"
    ).dropna(subset=["food_category_id"])
    
    if m.empty:
        print("   ⚠️  A表与food表连接后无数据，跳过导出")
        return
    
    results = []
    
    # 1. 体积单位：转密度
    v = m[m["unit_std"].isin(ml_map.keys())].copy()
    if not v.empty:
        v["density_g_per_ml"] = v["grams_per_unit"].astype("float64") / v["unit_std"].map(ml_map)
        v = v[(v["density_g_per_ml"] >= 0.1) & (v["density_g_per_ml"] <= 3.0)]
        
        if not v.empty:
            # 按category聚合
            v_cat = (v.groupby(["food_category_id", "unit_std"])["density_g_per_ml"]
                       .median().reset_index())
            v_cat["key_type"] = "cat"
            v_cat["key"] = v_cat["food_category_id"]
            v_cat["value"] = v_cat["density_g_per_ml"]
            v_cat = v_cat[["key", "key_type", "unit_std", "value"]]
            
            # 记录样本数量
            v_cat_n = v.groupby(["food_category_id", "unit_std"]).size().reset_index(name="n")
            v_cat = v_cat.merge(v_cat_n, left_on=["key", "unit_std"], 
                               right_on=["food_category_id", "unit_std"], how="left")
            v_cat = v_cat[["key", "key_type", "unit_std", "value", "n"]]
            
            results.append(v_cat)
            print(f"   📊 体积单位category聚合: {len(v_cat)} 个(category, unit)对")
    
    # 2. 件数单位：直接克重
    p = m[m["unit_std"].isin(piece_units)].copy()
    if not p.empty:
        p = p[(p["grams_per_unit"] >= 0.1) & (p["grams_per_unit"] <= 1000)]
        
        if not p.empty:
            # 按category聚合
            p_cat = (p.groupby(["food_category_id", "unit_std"])["grams_per_unit"]
                       .median().reset_index())
            p_cat["key_type"] = "cat"
            p_cat["key"] = p_cat["food_category_id"]
            p_cat["value"] = p_cat["grams_per_unit"]
            p_cat = p_cat[["key", "key_type", "unit_std", "value"]]
            
            # 记录样本数量
            p_cat_n = p.groupby(["food_category_id", "unit_std"]).size().reset_index(name="n")
            p_cat = p_cat.merge(p_cat_n, left_on=["key", "unit_std"], 
                               right_on=["food_category_id", "unit_std"], how="left")
            p_cat = p_cat[["key", "key_type", "unit_std", "value", "n"]]
            
            results.append(p_cat)
            print(f"   📊 件数单位category聚合: {len(p_cat)} 个(category, unit)对")
    
    # 3. Token聚合
    print("   正在提取token并聚合...")
    
    # 从description_norm提取token
    def extract_tokens(s):
        if not isinstance(s, str):
            return []
        return [t for t in tokenize(s) if len(t) >= 3]
    
    m["tokens"] = m["description_norm"].map(extract_tokens)
    m_expanded = m.explode("tokens").dropna(subset=["tokens"])
    
    if not m_expanded.empty:
        # 体积单位token聚合
        if not v.empty:
            v_tok = v.merge(m_expanded[["fdc_id", "tokens"]], on="fdc_id", how="inner")
            if not v_tok.empty:
                v_tok_agg = (v_tok.groupby(["tokens", "unit_std"])["density_g_per_ml"]
                               .median().reset_index())
                v_tok_agg["key_type"] = "token"
                v_tok_agg["key"] = v_tok_agg["tokens"]
                v_tok_agg["value"] = v_tok_agg["density_g_per_ml"]
                v_tok_agg = v_tok_agg[["key", "key_type", "unit_std", "value"]]
                
                # 记录样本数量
                v_tok_n = v_tok.groupby(["tokens", "unit_std"]).size().reset_index(name="n")
                v_tok_agg = v_tok_agg.merge(v_tok_n, left_on=["key", "unit_std"], 
                                           right_on=["tokens", "unit_std"], how="left")
                v_tok_agg = v_tok_agg[["key", "key_type", "unit_std", "value", "n"]]
                
                results.append(v_tok_agg)
                print(f"   📊 体积单位token聚合: {len(v_tok_agg)} 个(token, unit)对")
        
        # 件数单位token聚合
        if not p.empty:
            p_tok = p.merge(m_expanded[["fdc_id", "tokens"]], on="fdc_id", how="inner")
            if not p_tok.empty:
                p_tok_agg = (p_tok.groupby(["tokens", "unit_std"])["grams_per_unit"]
                               .median().reset_index())
                p_tok_agg["key_type"] = "token"
                p_tok_agg["key"] = p_tok_agg["tokens"]
                p_tok_agg["value"] = p_tok_agg["grams_per_unit"]
                p_tok_agg = p_tok_agg[["key", "key_type", "unit_std", "value"]]
                
                # 记录样本数量
                p_tok_n = p_tok.groupby(["tokens", "unit_std"]).size().reset_index(name="n")
                p_tok_agg = p_tok_agg.merge(p_tok_n, left_on=["key", "unit_std"], 
                                           right_on=["tokens", "unit_std"], how="left")
                p_tok_agg = p_tok_agg[["key", "key_type", "unit_std", "value", "n"]]
                
                results.append(p_tok_agg)
                print(f"   📊 件数单位token聚合: {len(p_tok_agg)} 个(token, unit)对")
    
    # 合并所有结果
    if results:
        out = pd.concat(results, ignore_index=True)
        out.to_parquet(out_path, index=False)
        print(f"   ✅ 导出完成: {len(out)} 个聚合记录，保存到 {out_path}")
        
        # 统计信息
        cat_count = (out["key_type"] == "cat").sum()
        token_count = (out["key_type"] == "token").sum()
        print(f"   📊 聚合统计: category={cat_count}, token={token_count}")
    else:
        print("   ⚠️  没有有效的聚合数据，跳过导出")

# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主函数：执行A表参数导出流程
    
    执行流程：
        1. 解析命令行参数
        2. 加载A表和food表数据
        3. 创建输出目录
        4. 执行四个导出函数
        5. 生成统计报告
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="A表参数导出脚本")
    parser.add_argument("--a_table_path", required=True, help="A表parquet文件路径")
    parser.add_argument("--food_path", required=True, help="food_processed.parquet文件路径")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(">> A表参数导出开始...")
    print(f"   A表路径: {args.a_table_path}")
    print(f"   Food表路径: {args.food_path}")
    print(f"   输出目录: {args.out_dir}")
    
    # 检查输入文件
    if not os.path.exists(args.a_table_path):
        raise FileNotFoundError(f"A表文件不存在: {args.a_table_path}")
    if not os.path.exists(args.food_path):
        raise FileNotFoundError(f"Food表文件不存在: {args.food_path}")
    
    # 加载数据
    print("\n>> 加载数据...")
    with tqdm(total=2, desc="加载数据") as pbar:
        a_df = pd.read_parquet(args.a_table_path)
        print(f"   A表: {len(a_df)} 条记录")
        pbar.update(1)
        
        food_df = pd.read_parquet(args.food_path)
        print(f"   Food表: {len(food_df)} 条记录")
        pbar.update(1)
    
    # 检查必要的列
    required_a_cols = ["fdc_id", "unit_std", "grams_per_unit"]
    missing_a_cols = [col for col in required_a_cols if col not in a_df.columns]
    if missing_a_cols:
        raise ValueError(f"A表缺少必要列: {missing_a_cols}")
    
    required_food_cols = ["fdc_id", "food_category_id", "description_norm"]
    missing_food_cols = [col for col in required_food_cols if col not in food_df.columns]
    if missing_food_cols:
        raise ValueError(f"Food表缺少必要列: {missing_food_cols}")
    
    # 执行导出
    print("\n>> 执行导出...")
    
    # 1. fdc级体积密度
    out_path1 = os.path.join(args.out_dir, "A_fdc_volume_density.parquet")
    export_a_fdc_volume_density(a_df, ML_PER, out_path1)
    
    # 2. fdc×unit级件数克重
    out_path2 = os.path.join(args.out_dir, "A_fdc_piece_weight.parquet")
    export_a_fdc_piece_weight(a_df, PIECE_UNITS, out_path2)
    
    # 3. 全局中位数
    out_path3 = os.path.join(args.out_dir, "A_unit_global_median.parquet")
    export_a_unit_global_median(a_df, out_path3)
    
    # 4. 类别/Token聚合
    out_path4 = os.path.join(args.out_dir, "A_cat_token_aggregates.parquet")
    export_a_cat_token_aggregates(a_df, food_df, ML_PER, PIECE_UNITS, out_path4)
    
    # 生成统计报告
    print("\n>> 导出完成！统计报告")
    print("=" * 60)
    
    output_files = [
        ("A_fdc_volume_density.parquet", "fdc级体积密度"),
        ("A_fdc_piece_weight.parquet", "fdc×unit级件数克重"),
        ("A_unit_global_median.parquet", "全局中位数"),
        ("A_cat_token_aggregates.parquet", "类别/Token聚合")
    ]
    
    for filename, description in output_files:
        filepath = os.path.join(args.out_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            print(f"✅ {description}: {len(df)} 条记录")
        else:
            print(f"❌ {description}: 文件未生成")
    
    print(f"\n📁 输出目录: {args.out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
