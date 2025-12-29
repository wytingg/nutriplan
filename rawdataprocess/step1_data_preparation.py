# -*- coding: utf-8 -*-
"""
step1_data_preparation.py — 步骤1：数据准备和预处理

功能概述：
    本脚本是食谱基准测试数据预处理流程的第一步，主要负责：
    1. 加载和合并USDA、FNDDS营养数据库
    2. 加载和解析Food.com食谱数据
    3. 单位标准化和克重估算
    4. 数据清洗和列精简
    5. 构建倒排索引用于后续匹配

核心特性：
    - 列精简模式：支持 safe_min 和 ultra_min 两种模式，通过 MODE 常量控制
    - 分层映射：基于A表构建体积→密度、件数→重量的分层映射策略
    - 单位识别：支持多种单位格式，包括Unicode分数、贴连数字等
    - PyArrow兼容：使用PyArrow兼容的正则表达式，避免引擎冲突
    - 向量化处理：大量使用pandas向量化操作提升性能

使用方法：
    python step1_data_preparation.py \
      --usda_dir /path/to/usda/csv \
      --fdnn_dir /path/to/fdnn/csv \
      --recipes /path/to/recipes.parquet \
      --out_dir /path/to/output \
      --filter_usda_types "Foundation Foods,SR Legacy"

输出文件：
    - food_processed.parquet: 处理后的食物信息
    - nutrient_processed.parquet: 营养素信息
    - food_nutrient_processed.parquet: 食物-营养素关系
    - recipes_processed.parquet: 处理后的食谱信息
    - ingredients_processed.parquet: 处理后的配料信息
    - inv_index.pkl: 倒排索引文件
"""

# =============================================================================
# 导入依赖库
# =============================================================================
import os
import argparse
import pandas as pd
import warnings
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
import numpy as np
import gc
import json
from common_utils import *

# =============================================================================
# 全局配置参数
# =============================================================================

# 列精简模式开关
# safe_min: 保留核心列，适合大多数用途
# ultra_min: 仅保留最必要的列，最小化存储空间
MODE = "safe_min"  # 可选: "safe_min" 或 "ultra_min"

# 性能优化开关
# 模糊匹配功能开关，用于处理单位名称的拼写错误
# 当前关闭以提升性能，后续可针对小残差集开启
ENABLE_FUZZY = False  # 先整体关闭模糊匹配；第二阶段只对小残差集开启

# =============================================================================
# A表配置和单位同义词映射
# =============================================================================

# A表路径配置
# A表包含家庭常用单位的重量转换数据，用于提升单位识别和克重估算精度
# 支持跨平台路径，按优先级查找
A_TABLE_PATH = None
for possible_path in [
    "/mnt/data/household_weights_A.csv",  # 用户指定路径
    "work/recipebench/data/3out/household_weights_A1.csv",  # 服务器绝对路径
    "data/household_weights_A.csv",  # 相对路径
    "household_weights_A.csv",  # 当前目录
]:
    if os.path.exists(possible_path):
        A_TABLE_PATH = possible_path
        break

if A_TABLE_PATH is None:
    A_TABLE_PATH = "work/recipebench/data/3out/household_weights_A.csv"  # 默认路径

# 本地单位同义词映射
# 作为 normalize_unit_name 的兜底方案，处理常见的单位名称变体
# 包括Unicode分数、复数形式、缩写等
UNIT_SYNONYMS_LOCAL = {
    # Unicode 分数处理 - 将Unicode分数字符转换为ASCII分数
    "½": "1/2", "¼": "1/4", "¾": "3/4",
    
    # 体积单位 - 茶匙、汤匙、杯等
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp", "tsps": "tsp",
    "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp", "tbsps": "tbsp",
    "cup": "cup", "cups": "cup",
    "fl oz": "fl_oz", "fluid ounce": "fl_oz", "fluid ounces": "fl_oz", "fl. oz": "fl_oz",
    "pint": "pint", "pints": "pint", "pt": "pint",
    "quart": "quart", "quarts": "quart", "qt": "quart",
    "gallon": "gallon", "gallons": "gallon", "gal": "gallon",
    "ml": "ml", "milliliter": "ml", "millilitre": "ml", "milliliters": "ml", "millilitres": "ml",
    "l": "l", "liter": "l", "litre": "l", "liters": "l", "litres": "l",
    
    # 质量单位 - 克、千克、盎司、磅等
    "g": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "lb": "lb", "pound": "lb", "pounds": "lb", "lbs": "lb",
    
    # 家用单位 - 件数单位，如瓣、片、根等
    "clove": "clove", "cloves": "clove",        # 蒜瓣
    "slice": "slice", "slices": "slice",        # 片
    "sheet": "sheet", "sheets": "sheet",        # 张
    "stick": "stick", "sticks": "stick",        # 根、条
    "bunch": "bunch", "bunches": "bunch",       # 束
    "sprig": "sprig", "sprigs": "sprig",        # 小枝
    "can": "can", "cans": "can",                # 罐
    "bottle": "bottle", "bottles": "bottle",    # 瓶
    "box": "box", "boxes": "box",               # 盒
    "carton": "carton", "cartons": "carton",    # 纸盒
    "package": "package", "packages": "package", "pkg": "package", "pkgs": "package",  # 包
    "packet": "packet", "packets": "packet",    # 小包
    "jar": "jar", "jars": "jar",                # 瓶
    "bag": "bag", "bags": "bag",                # 袋
    "head": "head", "heads": "head",            # 头（如卷心菜）
    "leaf": "leaf", "leaves": "leaf",           # 叶
    "stalk": "stalk", "stalks": "stalk",        # 茎
    "ear": "ear", "ears": "ear",                # 穗（如玉米）
    "container": "container", "containers": "container",  # 容器
    "bar": "bar", "bars": "bar",                # 条
    "piece": "piece", "pieces": "piece",        # 块、片
    "serving": "serving", "servings": "serving", # 份
    "egg": "egg", "eggs": "egg",                # 蛋
    "fillet": "fillet", "fillets": "fillet",    # 鱼片
    "breast": "breast", "breasts": "breast",    # 胸肉
    "wing": "wing", "wings": "wing",            # 翅膀
    "rib": "rib", "ribs": "rib",                # 肋骨
}

# =============================================================================
# 单位正则表达式模式配置
# =============================================================================

# 单位正则模式（按最长别名优先排序）
# 用于从文本中识别和提取单位名称
# 按长度降序排列，确保长模式优先匹配（如"tablespoons"优先于"tbsp"）
UNIT_PATTERNS = [
    # 体积单位（按长度排序，从长到短）
    "tablespoons?", "tbsps?", "tbsp",           # 汤匙
    "teaspoons?", "tsps?", "tsp",               # 茶匙
    "fluid ounces?", "fl\\.?\\s*oz", "fl oz",   # 液体盎司
    "milliliters?", "millilitres?", "ml",       # 毫升
    "liters?", "litres?", "l",                  # 升
    "gallons?", "gal",                          # 加仑
    "quarts?", "qt",                            # 夸脱
    "pints?", "pt",                             # 品脱
    "cups?", "cup",                             # 杯
    
    # 质量单位（按长度排序，从长到短）
    "kilograms?", "kgs?",                       # 千克
    "pounds?", "lbs?",                          # 磅
    "ounces?", "oz",                            # 盎司
    "grams?", "g",                              # 克
    
    # 家用单位（按长度排序，从长到短）
    "containers?", "container",                 # 容器
    "packages?", "pkgs?", "pkg", "packets?", "packet",  # 包
    "cartons?", "carton",                       # 纸盒
    "bottles?", "bottle",                       # 瓶
    "boxes?", "box",                            # 盒
    "bunches?", "bunch",                        # 束
    "sprigs?", "sprig",                         # 小枝
    "stalks?", "stalk",                         # 茎
    "fillets?", "fillet",                       # 鱼片
    "breasts?", "breast",                       # 胸肉
    "wings?", "wing",                           # 翅膀
    "ribs?", "rib",                             # 肋骨
    "slices?", "slice",                         # 片
    "sheets?", "sheet",                         # 张
    "cloves?", "clove",                         # 瓣
    "sticks?", "stick",                         # 根、条
    "pieces?", "piece",                         # 块、片
    "servings?", "serving",                     # 份
    "heads?", "head",                           # 头
    "leaves?", "leaf",                          # 叶
    "jars?", "jar",                             # 瓶
    "bags?", "bag",                             # 袋
    "cans?", "can",                             # 罐
    "bars?", "bar",                             # 条
    "eggs?", "egg",                             # 蛋
    "ears?", "ear",                             # 穗
]

# 编译单位正则模式
# 使用负向断言确保单位名称的边界匹配，避免误匹配
import re
UNIT_REGEX = re.compile(
    r'(?<![A-Za-z])(' + '|'.join(sorted(UNIT_PATTERNS, key=len, reverse=True)) + r')(?![A-Za-z])',
    re.IGNORECASE
)

# PyArrow兼容的单位正则模式（初始占位，稍后基于量具+名词动态重建）
UNIT_REGEX_PYARROW = re.compile(
    r'\b(?:' + '|'.join(sorted(UNIT_PATTERNS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

# 非捕获组版本，用于str.contains避免警告
# 使用非捕获组(?:...)避免捕获组警告
UNIT_REGEX_NC = re.compile(
    r'(?<![A-Za-z])(?:' + '|'.join(sorted(UNIT_PATTERNS, key=len, reverse=True)) + r')(?![A-Za-z])',
    re.IGNORECASE
)

# 通用缺失判断
def _is_missing(x) -> bool:
    # 对 pd.NA / NaN / None / 空串 都按缺失处理
    try:
        if x is None:
            return True
        if pd.isna(x):
            return True
        if isinstance(x, str) and x.strip() == "":
            return True
        return False
    except Exception:
        return False

# 性能优化：预编译常用正则表达式
# 避免重复编译，提升处理大量数据时的性能
FRACTION_PATTERN = re.compile(r'^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$')  # 分数模式
DIGIT_UNIT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)')          # 数字+单位模式
UNICODE_FRACTION_PATTERN = re.compile(r'[¼½¾⅓⅔⅛⅜⅝⅞]')                    # Unicode分数模式

def singularize(w: str) -> str:
    """
    简易单复数规约（含常见不规则）
    
    功能说明：
        将复数形式转换为单数形式，用于名词单位标准化
        处理常见的不规则复数变化和规则复数变化
    
    参数：
        w (str): 需要转换的单词
    
    返回：
        str: 转换后的单数形式
    """
    if _is_missing(w) or not isinstance(w, str):
        return w
    wl = w.lower()
    if wl in IRREG_SING:
        return IRREG_SING[wl]
    if not re.fullmatch(r"[A-Za-z\-]+", wl):
        return None
    if wl.endswith("ies") and len(wl) > 3:  # berries->berry
        return wl[:-3] + "y"
    if wl.endswith("oes") and len(wl) > 3:
        return wl[:-2]                    # heroes->hero
    if wl.endswith("es") and len(wl) > 2:
        return wl[:-2]                    # wedges->wedge
    if wl.endswith("s") and len(wl) > 1:
        return wl[:-1]                    # lemons->lemon
    return wl

def extract_noun_units_from_A(a_table):
    """
    从A表构造"名词单位"词表 + 每件克重映射
    
    功能说明：
        1. 从A表中提取纯字母单位（如tomato、lemon等）
        2. 将复数形式转换为单数形式
        3. 计算每个名词单位的中位数克重
        4. 过滤掉样本量太少的条目
    
    参数：
        a_table (pd.DataFrame): A表数据
    
    返回：
        dict: 名词单位 -> 每件克重的映射
    """
    global NOUN_PIECE_WEIGHT
    
    if a_table is None or a_table.empty:
        print("   [A-noun] A表为空，跳过名词单位提取")
        return {}
    
    # 确定列名
    unit_col_candidates = ["unit_std", "unit", "units", "Unit", "UnitStd"]
    grams_col_candidates = ["grams_per_unit", "g_per_unit", "grams", "gram_per_unit", "g"]
    
    def pick(colnames, df):
        for c in colnames:
            if c in df.columns:
                return c
        return None
    
    unit_col = pick(unit_col_candidates, a_table)
    grams_col = pick(grams_col_candidates, a_table)
    
    if not unit_col or not grams_col:
        print(f"   [A-noun] 未找到合适的列名，跳过名词单位提取")
        print(f"   [A-noun] 候选单位列: {unit_col_candidates}")
        print(f"   [A-noun] 候选克重列: {grams_col_candidates}")
        return {}
    
    print(f"   [A-noun] 使用列: {unit_col} -> {grams_col}")
    
    # === 从 A 表构建 "名词单位 → 每件克重" ===
    assert unit_col in a_table.columns, "A表缺 unit 列"
    # 允许 unit 是食材名（tomatoes/egg/onion），不要过滤掉
    A_MIN_SUPPORT = 1   # 先放宽，后面审计再收紧

    # 单复数规约（足够用的轻量规则）
    IRREG_SING = {"tomatoes":"tomato","potatoes":"potato","cherries":"cherry",
                  "leaves":"leaf","knives":"knife","loaves":"loaf","children":"child"}
    def singularize(w: str) -> str:
        if not isinstance(w,str) or not w: return w
        wl = w.strip().lower()
        if wl in IRREG_SING: return IRREG_SING[wl]
        if re.fullmatch(r"[A-Za-z\-]+", wl) is None: return None
        if wl.endswith("ies") and len(wl)>3: return wl[:-3]+"y"
        if wl.endswith("oes") and len(wl)>3: return wl[:-2]
        if wl.endswith("es")  and len(wl)>2: return wl[:-2]
        if wl.endswith("s")   and len(wl)>1: return wl[:-1]
        return wl

    tmp = a_table.dropna(subset=[unit_col]).copy()
    tmp["noun"] = tmp[unit_col].astype(str).map(singularize)

    # grams_per_unit 列名按你的 A 表实际修改（支持 grams_per_unit 或 grams）
    val_col = grams_col
    assert val_col is not None, "A表缺 grams_per_unit/grams 列"

    # 统计中位数 + 支持度
    agg = (tmp.dropna(subset=["noun", val_col])
              .groupby("noun")[val_col]
              .agg(["median","count"])
              .reset_index())
    noun_piece_weight = {r["noun"]: float(r["median"]) for _,r in agg.iterrows() if r["count"]>=A_MIN_SUPPORT}

    print(f"[A-noun] 名词单位权重就绪: {len(noun_piece_weight)} 条；示例: {list(noun_piece_weight.items())[:8]}")
    
    # 更新全局变量
    NOUN_PIECE_WEIGHT = noun_piece_weight
    
    return noun_piece_weight

def rebuild_unit_regex_from_A():
    """
    从A表动态扩展单位词表并更新UNIT_STD_MAP（仅调用一次）
    
    功能说明：
        1. 从A表中提取所有单位名称，扩展UNIT_STD_MAP
        2. 更新正则表达式模式，包含A表中的单位
        3. 扩展同义词映射，提升单位识别覆盖率
        4. 执行烟雾测试，验证正则表达式工作正常
    
    设计考虑：
        - 使用REBUILD_ONCE_FLAG确保只执行一次，避免重复计算
        - 与新的强化单位正则系统兼容
        - 提供详细的调试信息，便于问题排查
    
    Returns:
        None
    """
    global UNIT_REGEX, UNIT_STD_MAP, REBUILD_ONCE_FLAG
    
    # 防止重复调用
    if REBUILD_ONCE_FLAG:
        print("   ⚠️  rebuild_unit_regex_from_A 已调用过，跳过重复调用")
        return
        
    # 检查A表是否已加载
    if A_TABLE is None or A_TABLE.empty:
        return
    
    # 从A表中提取所有单位名称
    units_from_A = set(A_TABLE["unit_std"].dropna().unique())
    print(f"   A表单位数量: {len(units_from_A)}")
    print(f"   A表单位示例: {list(units_from_A)[:10]}")
    
    # 扩展UNIT_STD_MAP，将A表中的单位添加到映射中
    for unit in units_from_A:
        if unit not in UNIT_STD_MAP:
            UNIT_STD_MAP[unit] = unit
    
    # ===== 基于“量具单位 + 名词单位”动态重建 =====
    print("   正在基于量具+名词单位重建 UNIT 正则 ...")

    # 量具单位（静态白名单，可按需扩展）
    MEASURE_UNITS = [
        "teaspoons?", "tbsps?", "tbsp", "tablespoons?", "tsp",
        "cups?", "pints?", "pt", "quarts?", "qt", "gallons?", "gal",
        "milliliters?", "millilitres?", "ml", "liters?", "litres?", "l",
        "grams?", "g", "kilograms?", "kg", "ounces?", "oz", "pounds?", "lbs?", "lb",
        "fl\.?\s*oz",
        "stick", "sticks", "can", "cans", "jar", "jars", "package", "packages", "pkg", "container", "containers", "bottle", "bottles",
    ]
    MEASURE_ALTS = r'(?:' + r'|'.join(MEASURE_UNITS) + r')'

    # 从 A 表构建名词单位（件数语义），自动扩展复数
    IRREG_PLURALS = {
        "leaf": "leaves", "loaf": "loaves", "tomato": "tomatoes",
        "potato": "potatoes", "cherry": "cherries"
    }
    def pluralize(u: str):
        u = u.strip().lower()
        if not u: return []
        if u in IRREG_PLURALS: return [u, IRREG_PLURALS[u]]
        if u.endswith('y') and (len(u) >= 2 and u[-2] not in 'aeiou'):
            return [u, u[:-1] + 'ies']
        if u.endswith(('s','x','z','ch','sh')):
            return [u, u + 'es']
        return [u, u + 's']

    def build_noun_units_from_A(a_units):
        piece_like = []
        skip = {"ml","g","kg","tsp","tbsp","cup","oz","lb","l","fl oz","pt","qt","gal"}
        for u in a_units:
            if _is_missing(u):
                continue
            token = str(u).strip().lower()
            if token in skip:
                continue
            piece_like.extend(pluralize(token))
        return set(piece_like)

    # 误匹配保护（如 egg != eggplant, lemon != lemongrass）
    NEG_GUARDS = {
        "egg": r'(?!plant\w*)',
        "eggs": r'(?!plant\w*)',
        "lemon": r'(?!grass\w*)',
        "lemons": r'(?!grass\w*)',
    }
    def guarded(u: str) -> str:
        g = NEG_GUARDS.get(u, '')
        return fr'(?:{re.escape(u)}){g}'

    a_units = set(A_TABLE["unit_std"].dropna().astype(str).unique()) if (A_TABLE is not None and not A_TABLE.empty and "unit_std" in A_TABLE.columns) else set()
    NOUN_UNITS = build_noun_units_from_A(a_units)
    NOUN_ALTS = r'(?:' + r'|'.join(guarded(u) for u in sorted(NOUN_UNITS, key=len, reverse=True)) + r')'

    # 最终 PyArrow 兼容正则（全部使用非捕获组）
    UNIT_REGEX_PYARROW = re.compile(fr'(?i)\b(?:{MEASURE_ALTS}|{NOUN_ALTS})\b')
    # 同步通用正则（不使用lookbehind，避免后端差异）
    UNIT_REGEX = UNIT_REGEX_PYARROW

    # 暴露量具/名词子正则供来源判别
    globals()["MEASURE_REGEX"] = re.compile(fr'(?i)\b{MEASURE_ALTS}\b')
    globals()["NOUN_REGEX"] = re.compile(fr'(?i)\b{NOUN_ALTS}\b')

    print(f"   量具单位样本: {MEASURE_UNITS[:8]} ... 共 {len(MEASURE_UNITS)} 种模式")
    print(f"   名词单位数量: {len(NOUN_UNITS)}（来自A表）")
    print(f"   UNIT_REGEX_PYARROW 预览: {UNIT_REGEX_PYARROW.pattern[:120]}")
    
    # 烟雾测试：验证正则表达式能正确识别常见单位
    test_text = "1/2 cup milk; 2tbsp oil; 8oz chicken (boneless); 2 eggs; 3 lemons; eggplant; lemongrass"
    matches = re.findall(UNIT_REGEX.pattern, test_text, re.I)
    print(f"   Smoke test: re.findall(UNIT_REGEX, \"{test_text}\") = {matches}")
    
    # 设置标志，防止重复调用
    REBUILD_ONCE_FLAG = True

# =============================================================================
# 全局变量和分层映射数据结构
# =============================================================================

# A表相关全局变量
A_TABLE = None                    # A表数据框，包含单位转换数据
A_UNIT_MEDIAN = {}               # A表中每个单位的中位数转换值
UNIT_GLOBAL = {}                 # A表全局单位中位数字典，作为兜底方案
REBUILD_ONCE_FLAG = False        # 确保只调用一次rebuild_unit_regex_from_A

# 单位标准化映射
UNIT_STD_MAP = {
    # 重量
    "g":"g","gs":"g","gram":"g","grams":"g","gr":"g","gm":"g",
    "kg":"kg","kgs":"kg","kilogram":"kg","kilograms":"kg",
    "oz":"oz","ounce":"oz","ounces":"oz",
    "lb":"lb","lbs":"lb","pound":"lb","pounds":"lb","#":"lb",
    # 体积
    "ml":"ml","mL":"ml","milliliter":"ml","milliliters":"ml","cc":"ml",
    "l":"l","L":"l","liter":"l","liters":"l",
    "tsp":"tsp","tsps":"tsp","teaspoon":"tsp","teaspoons":"tsp","t":"tsp",
    "tbsp":"tbsp","tbsps":"tbsp","tbl":"tbsp","tbls":"tbsp","tablespoon":"tbsp","tablespoons":"tbsp","T":"tbsp",
    "cup":"cup","cups":"cup",
    # 件数
    "piece":"piece","pieces":"piece","pc":"piece","pcs":"piece",
    "clove":"clove","cloves":"clove",
    "slice":"slice","slices":"slice",
    "sprig":"sprig","sprigs":"sprig",
    "bunch":"bunch","bunches":"bunch",
    "can":"can","cans":"can",
    "jar":"jar","jars":"jar",
    "package":"package","packages":"package","packet":"package","packets":"package",
    "stick":"stick","sticks":"stick",
    "head":"head","heads":"head",
    "fillet":"fillet","fillets":"fillet",
    "strip":"strip","strips":"strip",
    "ear":"ear","ears":"ear",
    "leaf":"leaf","leaves":"leaf",
    "sheet":"sheet","sheets":"sheet",
    "block":"block","blocks":"block",
    "breast":"breast","breasts":"breast",
}

# 名词单位相关全局变量
NOUN_PIECE_WEIGHT = {}           # A表名词单位 -> 每件克重映射
IRREG_SING = {                   # 不规则复数变单数映射
    "tomatoes":"tomato","potatoes":"potato","cherries":"cherry",
    "leaves":"leaf","knives":"knife","loaves":"loaf"
}

# 单位转换系数（体积单位到毫升的转换）
# 用于将体积单位转换为毫升，进而计算密度
ML_PER = {
    'tsp': 4.92892,    # 茶匙到毫升
    'tbsp': 14.7868,   # 汤匙到毫升
    'cup': 236.588,    # 杯到毫升
    'fl_oz': 29.5735,  # 液体盎司到毫升
    'pt': 473.176,     # 品脱到毫升
    'qt': 946.353,     # 夸脱到毫升
    'l': 1000,         # 升到毫升
    'ml': 1            # 毫升到毫升（基准）
}

# 单位分类集合
VOLUME_UNITS = {'tsp', 'tbsp', 'cup', 'ml', 'l', 'fl_oz', 'pt', 'qt'}  # 体积单位
PIECE_UNITS = {                                                         # 件数单位
    'piece', 'slice', 'sheet', 'clove', 'stick', 'can', 'package', 'packet',
    'serving', 'head', 'jar', 'bag', 'bunch', 'sprig', 'egg', 'drumstick', 
    'thigh', 'steak', 'stalk', 'link', 'banana', 'spear', 'bottle', 'box',
    'carton', 'container', 'bar', 'fillet', 'breast', 'wing', 'rib'
}
WEIGHT_UNITS = {'g', 'kg', 'oz', 'lb'}  # 重量单位（仅用于校验）

# 分层映射结果字典
# 用于存储不同层级的单位转换映射，提升克重估算精度
density_by_fdc = {}              # {fdc_id: density} - 按食物ID的密度映射
density_by_cat = {}              # {category_id: density} - 按食物类别的密度映射
density_by_token = {}            # {token: density} - 按关键词的密度映射
piece_weight_fdc = {}            # {(fdc_id, unit): weight} - 按食物ID和单位的重量映射
piece_weight_by_cat = {}         # {(unit, category): weight} - 按单位和类别的重量映射
piece_weight_by_token = {}       # {(unit, token): weight} - 按单位和关键词的重量映射

def _winsorize_iqr(series, factor=1.5):
    """
    使用IQR方法进行Winsorize去极值处理
    
    功能说明：
        通过四分位数间距(IQR)方法识别和限制异常值，将超出合理范围的值
        限制到边界值，而不是直接删除，保持数据完整性。
    
    算法原理：
        1. 计算第一四分位数(Q1)和第三四分位数(Q3)
        2. 计算四分位数间距 IQR = Q3 - Q1
        3. 设置下界 = Q1 - factor * IQR
        4. 设置上界 = Q3 + factor * IQR
        5. 将超出边界的值限制到边界值
    
    参数：
        series (pd.Series): 需要处理的数值序列
        factor (float): IQR倍数因子，默认1.5（标准值）
    
    返回：
        pd.Series: 去极值后的序列
    
    注意：
        - 当序列长度小于4时，直接返回原序列
        - 使用clip方法限制值，保持数据分布形状
    """
    # 数据量太少时直接返回
    if len(series) < 4:
        return series
    
    # 计算四分位数
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    
    # 计算边界值
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    # 限制极值到边界
    return series.clip(lower_bound, upper_bound)

def _build_layered_mappings(a_table, grams_col):
    """
    构建分层映射：体积→密度，件数→每件克重
    
    功能说明：
        基于A表数据构建分层映射策略，用于提升单位转换精度：
        1. 体积单位：计算密度（g/ml），用于体积→重量转换
        2. 件数单位：计算每件重量（g/piece），用于件数→重量转换
        3. 使用IQR去极值和中位数聚合，提高数据质量
    
    分层策略：
        - 第一层：按fdc_id（食物ID）精确匹配
        - 第二层：按category（食物类别）匹配
        - 第三层：按token（关键词）匹配
        - 兜底层：全局中位数
    
    参数：
        a_table (pd.DataFrame): A表数据，包含单位转换信息
        grams_col (str): 克重列名
    
    返回：
        None (更新全局映射字典)
    """
    global density_by_fdc, density_by_cat, density_by_token
    global piece_weight_fdc, piece_weight_by_cat, piece_weight_by_token
    
    print("     【A】A表标准化与过滤...")
    
    # 过滤不可转移单位，仅保留体积、件数、重量单位
    # 排除无法转换的单位，如"serving"等
    valid_units = VOLUME_UNITS | PIECE_UNITS | WEIGHT_UNITS
    a_filtered = a_table[a_table["unit_std"].isin(valid_units)].copy()
    print(f"     过滤后保留 {len(a_filtered)} 行（原 {len(a_table)} 行）")
    
    # 对每个(fdc_id, unit)组做winsorize去极值后取中位数
    # 去除异常值，提高数据质量
    print("     正在winsorize去极值...")
    a_clean = a_filtered.groupby(["fdc_id", "unit_std"])[grams_col].apply(
        lambda x: _winsorize_iqr(x).median() if len(x) > 1 else x.iloc[0]
    ).reset_index()
    a_clean.columns = ["fdc_id", "unit_std", "grams_per_unit_clean"]
    
    print("     【B】体积→密度映射...")
    # 体积单位处理：计算密度（g/ml）
    volume_data = a_clean[a_clean["unit_std"].isin(VOLUME_UNITS)].copy()
    if not volume_data.empty:
        # 计算密度：density = grams_per_unit / ML_PER[unit]
        # 将不同体积单位统一转换为密度（g/ml）
        volume_data["density"] = volume_data.apply(
            lambda row: row["grams_per_unit_clean"] / ML_PER.get(row["unit_std"], 1), 
            axis=1
        )
        
        # 按fdc_id取中位数得到density_by_fdc
        # 为每个食物ID建立密度映射
        density_by_fdc.update(
            volume_data.groupby("fdc_id")["density"].median().to_dict()
        )
        print(f"     体积单位数据: {len(volume_data)} 行，{len(density_by_fdc)} 个fdc_id")
    
    print("     【C】件数→每件克重映射...")
    # 件数单位处理：计算每件重量（g/piece）
    piece_data = a_clean[a_clean["unit_std"].isin(PIECE_UNITS)].copy()
    if not piece_data.empty:
        # 按(fdc_id, unit)求中位grams_per_unit
        # 为每个食物ID和单位组合建立重量映射
        piece_weight_fdc.update(
            piece_data.set_index(["fdc_id", "unit_std"])["grams_per_unit_clean"].to_dict()
        )
        print(f"     件数单位数据: {len(piece_data)} 行，{len(piece_weight_fdc)} 个(fdc_id, unit)对")
    
    print("     分层映射基础数据构建完成")

def estimate_grams_enhanced(row, return_path=False):
    """
    增强版克重估算：实施分层映射查找优先级（全局版本）
    
    功能说明：
        基于分层映射策略进行克重估算，按优先级从高到低查找：
        1. 精确匹配：按fdc_id（食物ID）匹配
        2. 关键词匹配：按ingredient_norm（标准化配料名）匹配
        3. 类别匹配：按food_category_id（食物类别）匹配
        4. 全局兜底：使用UNIT_GLOBAL全局中位数
    
    分层策略：
        体积单位：数量 × 体积系数 × 密度
        件数单位：数量 × 每件重量
        其他单位：数量 × 全局转换系数
    
    参数：
        row (pd.Series): 包含配料信息的行数据
            - qty_parsed: 解析后的数量
            - unit_std: 标准化单位
            - ingredient_norm: 标准化配料名
            - fdc_id: 食物ID（可选）
            - food_category_id: 食物类别ID（可选）
        return_path (bool): 是否返回使用的路径信息
    
    返回：
        float or tuple: 估算的克重，失败时返回None
                       如果return_path=True，返回(grams, path_used)
    """
    qty_parsed = row.get("qty_parsed")
    unit_std = row.get("unit_std")
    ingredient_norm = row.get("ingredient_norm", "")
    
    # 基础参数检查和类型转换
    if pd.isna(qty_parsed) or pd.isna(unit_std):
        return (None, "no_data") if return_path else None
    
    # 确保 qty_parsed 是数值类型
    try:
        qty_parsed = float(qty_parsed)
    except (ValueError, TypeError):
        return (None, "invalid_qty") if return_path else None
    
    # 体积单位：密度策略
    if unit_std in VOLUME_UNITS:
        # 1. 精确匹配：按fdc_id匹配密度
        # 公式：grams = qty_parsed * ML_PER[unit] * density_by_fdc[fdc_id]
        fdc_id = row.get("fdc_id")
        if pd.notna(fdc_id) and fdc_id in density_by_fdc:
            grams = qty_parsed * ML_PER.get(unit_std, 1) * density_by_fdc[fdc_id]
            return (grams, "fdc_density") if return_path else grams
        
        # 2. 关键词匹配：按ingredient_norm匹配密度
        # 公式：grams = qty_parsed * ML_PER[unit] * density_by_token[token]
        if ingredient_norm in density_by_token:
            grams = qty_parsed * ML_PER.get(unit_std, 1) * density_by_token[ingredient_norm]
            return (grams, "token_density") if return_path else grams
        
        # 3. 类别匹配：按food_category_id匹配密度
        # 公式：grams = qty_parsed * ML_PER[unit] * density_by_cat[category]
        category_id = row.get("food_category_id")
        if pd.notna(category_id) and category_id in density_by_cat:
            grams = qty_parsed * ML_PER.get(unit_std, 1) * density_by_cat[category_id]
            return (grams, "cat_density") if return_path else grams
        
        # 4. 全局兜底：使用全局中位数
        if unit_std in UNIT_GLOBAL:
            grams = qty_parsed * UNIT_GLOBAL[unit_std]
            return (grams, "global_volume") if return_path else grams
    
    # 件数单位：每件克重策略
    elif unit_std in PIECE_UNITS:
        # 1. 精确匹配：按(fdc_id, unit)匹配每件重量
        # 公式：grams = qty_parsed * piece_weight_fdc[(fdc_id, unit)]
        fdc_id = row.get("fdc_id")
        if pd.notna(fdc_id) and (fdc_id, unit_std) in piece_weight_fdc:
            grams = qty_parsed * piece_weight_fdc[(fdc_id, unit_std)]
            return (grams, "fdc_piece") if return_path else grams
        
        # 2. 关键词匹配：按(unit, token)匹配每件重量
        # 公式：grams = qty_parsed * piece_weight_by_token[(unit, token)]
        if (unit_std, ingredient_norm) in piece_weight_by_token:
            grams = qty_parsed * piece_weight_by_token[(unit_std, ingredient_norm)]
            return (grams, "token_piece") if return_path else grams
        
        # 3. 类别匹配：按(unit, category)匹配每件重量
        # 公式：grams = qty_parsed * piece_weight_by_cat[(unit, category)]
        category_id = row.get("food_category_id")
        if pd.notna(category_id) and (unit_std, category_id) in piece_weight_by_cat:
            grams = qty_parsed * piece_weight_by_cat[(unit_std, category_id)]
            return (grams, "cat_piece") if return_path else grams
        
        # 4. 全局兜底：使用全局中位数
        if unit_std in UNIT_GLOBAL:
            grams = qty_parsed * UNIT_GLOBAL[unit_std]
            return (grams, "global_piece") if return_path else grams
    
    # 其他单位或回退：使用全局转换系数
    if unit_std in UNIT_GLOBAL:
        grams = qty_parsed * UNIT_GLOBAL[unit_std]
        return (grams, "global_other") if return_path else grams
    
    return (None, "no_match") if return_path else None

# >>> [ADD-LANES] 四车道决策与兜底工具 <<<

def _expand_packsize(qty_raw: str):
    """
    识别"包装×规格"：如 '2 (14 ounce) cans' → 返回 (mult=2.0, size=14.0, unit='oz')
    """
    if _is_missing(qty_raw):
        return None
    s = str(qty_raw)
    # 2 (14 ounce) cans / 3(400 g) jars / 4 (8oz) packages
    m = re.search(r'(\d+(?:\.\d+)?)\s*\(\s*([\d\.]+)\s*([A-Za-z]+)\s*\)', s)
    if not m:
        return None
    try:
        mult = float(m.group(1))
        size = float(m.group(2))
        unit = _normalize_unit_local(m.group(3))
        return (mult, size, unit)
    except Exception:
        return None

# ======== [ADD] 强化数量解析：parse_qty_v2 + parse_qty_and_unit_vectorized_v2 ========

# 配置：范围策略（"upper" 取上界；"mean" 取均值）
RANGE_POLICY = "upper"

_WORD_NUM = {
    # 基础词到数值
    "a": 1.0, "an": 1.0,
    "half": 0.5, "quarter": 0.25,
    "dozen": 12.0, "couple": 2.0,
}
# 软性数量词可选（如"few"、"several"）；默认不启用以保证严谨性
_ENABLE_SOFT_WORDS = False
if _ENABLE_SOFT_WORDS:
    _WORD_NUM.update({"few": 3.0, "several": 4.0})

# 统一把 unicode 分数替换为 ASCII 的简单函数（已存在 _normalize_numeric_text / trans_tbl，这里兜底单串版本）
def _norm_num_text_local(s):
    if _is_missing(s): return ""
    s = str(s)
    return _normalize_numeric_text(s)

_MIXED_FRAC = re.compile(r'^\s*(\d+)\s+(\d+/\d+)\s*$')  # '1 1/2'
_RANGE = re.compile(r'^\s*([^\s]+(?:\s+\d+/\d+)?)\s*(?:-|–|to|or|~|〜)\s*([^\s]+(?:\s+\d+/\d+)?)\s*$',
                    re.IGNORECASE)
_SIMPLE = re.compile(r'^\s*\d+(?:\.\d+)?\s*$')          # '2' or '2.5'
_FRAC_ONLY = re.compile(r'^\s*\d+\s*/\s*\d+\s*$')       # '1/2'
_WORDS = re.compile(r'^\s*(a|an|half|quarter|dozen|couple|few|several)\s*$', re.IGNORECASE)

def _to_float_num(token: str):
    """把 token 转为 float，支持纯小数、分数"""
    token = token.strip()
    if _SIMPLE.match(token):
        return float(token)
    if _FRAC_ONLY.match(token):
        return _parse_fraction_to_float(token)
    # '1 1/2'
    m = _MIXED_FRAC.match(token)
    if m:
        a = float(m.group(1))
        b = _parse_fraction_to_float(m.group(2))
        return a + (b or 0.0)
    # 词汇数字
    m2 = _WORDS.match(token)
    if m2:
        return float(_WORD_NUM.get(m2.group(1).lower(), None)) if m2.group(1).lower() in _WORD_NUM else None
    # 最后尝试直接 float
    try:
        return float(token)
    except:
        return None

def parse_qty_v2(qty_text: str):
    """
    强化版数量解析：
      - 支持范围：'1-2', '1 to 2', '2 or 3'（RANGE_POLICY 控制取值）
      - 支持混合分数：'1 1/2'
      - 支持单分数：'1/2'
      - 支持 unicode 分数
      - 支持词汇数量：a/an/half/quarter/dozen/couple（可选 few/several）
    返回 float 或 None
    """
    if _is_missing(qty_text):
        return None
    s = _norm_num_text_local(qty_text)

    # 包装×规格已由 _expand_packsize 处理；这里处理单一 qty 字段
    # 1) 范围
    m = _RANGE.match(s)
    if m:
        x1 = _to_float_num(m.group(1))
        x2 = _to_float_num(m.group(2))
        if x1 is not None and x2 is not None:
            if RANGE_POLICY == "mean":
                return (x1 + x2) / 2.0
            return max(x1, x2)
        # 半解析失败则继续尝试后续规则

    # 2) 纯数 或 纯分数 或 混合分数
    if _SIMPLE.match(s) or _FRAC_ONLY.match(s) or _MIXED_FRAC.match(s):
        v = _to_float_num(s)
        return v

    # 3) 单词
    m2 = _WORDS.match(s)
    if m2:
        return float(_WORD_NUM.get(m2.group(1).lower(), None)) if m2.group(1).lower() in _WORD_NUM else None

    # 4) '2x' '2×' 这类乘号落网（理论上 qty_raw 里较少，通常是包装×规格）
    s2 = re.sub(r'[x×]', ' ', s)
    if s2 != s:
        return parse_qty_v2(s2)

    # 无法识别
    return None

def parse_qty_and_unit_vectorized_v2(parts_series: pd.Series, qty_series: pd.Series):
    """
    向量化解析：综合 ingredient_parts 与 ingredient_qties 两列，返回 DataFrame[qty_parsed, unit_std, audit_flags]
    规则：
      - 先从 qty_series 解析数量（parse_qty_v2）
      - 若 qty 缺失/无效，尝试从 parts_series 嵌入的 '2 onions' 等模式抽取（名词单位兜底）
      - 单位部分：优先从 qty_series 提供的单位字段；否则从 parts_series 中用 UNIT_REGEX / A表扩展词表捕获
      - 返回 audit_flags（JSON字符串），标注使用了哪些兜底或遇到了哪些文本标记（如 to taste / optional / divided）
    """
    # 预清洗：断开 8oz → 8 oz，去括号/连字符 等
    parts_clean = _normalize_for_unit(parts_series.astype("string[pyarrow]").fillna(""))
    qty_clean   = _normalize_for_unit(qty_series.astype("string[pyarrow]").fillna(""))

    # 1) 解析数量（qty_series 优先）
    qty_parsed = qty_clean.map(parse_qty_v2)

    # 2) 数量缺失：尝试从 parts 中抽取起始"数量+名词"(e.g., '2 onions', '3 cloves garlic')
    #    仅在 qty_parsed 为空时触发
    need_from_parts = qty_parsed.isna()
    if bool(need_from_parts.any()):
        # 抽取开头的数量（混合/分数/小数）+ 名词
        # 例： "1 1/2 cups milk", "2 onions", "3 cloves garlic"
        pat = re.compile(r'^\s*([0-9]+(?:\s+[0-9]+\/\/[0-9]+)?|[0-9]+(?:\.[0-9]+)?|[¼½¾⅓⅔⅛⅜⅝⅞]|a|an|half|quarter|dozen|couple)\b', re.I)
        sub = parts_clean[need_from_parts].str.extract(pat)
        qty_guess = sub[0].map(parse_qty_v2)
        qty_parsed.loc[need_from_parts] = qty_guess

    # 3) 单位识别：优先 qty_clean 中的单位令牌，其次 parts_clean
    #    （你已有 UNIT_REGEX / rebuild_unit_regex_from_A，可直接复用）
    def _pick_unit(s):
        if _is_missing(s):
            return None
        s = str(s)
        m = re.search(UNIT_REGEX, s)
        if m:
            return _normalize_unit_local(m.group(0))
        # PyArrow兼容的弱匹配
        m2 = re.search(UNIT_REGEX_PYARROW, s)
        if m2:
            return _normalize_unit_local(m2.group(0))
        return None

    unit_from_qty   = qty_clean.map(_pick_unit)
    unit_from_parts = parts_clean.map(_pick_unit)
    unit_std = unit_from_qty.fillna(unit_from_parts)

    # 单位来源标注（审计）：regex_measure_hit / regex_noun_hit
    def _hit_kind(s):
        try:
            txt = str(s)
            if 'MEASURE_REGEX' in globals() and MEASURE_REGEX.search(txt):
                return 'regex_measure_hit'
            if 'NOUN_REGEX' in globals() and NOUN_REGEX.search(txt):
                return 'regex_noun_hit'
        except Exception:
            pass
        return None
    src_from_qty = qty_clean.map(_hit_kind)
    src_from_parts = parts_clean.map(_hit_kind)
    unit_src = src_from_qty.where(src_from_qty.notna(), src_from_parts)

    # 4) 审计 flags
    def _flags_for_row(qs, us, ptext, src):
        f = []
        if pd.isna(qs): f.append("no_qty")
        if pd.isna(us): f.append("no_unit")
        t = str(ptext).lower()
        for kw in ["to taste", "optional", "divided"]:
            if kw in t: f.append(kw.replace(" ", "_"))
        if src:
            f.append(src)
        return json.dumps(f) if f else json.dumps([])
    audit_flags = [ _flags_for_row(q, u, p, s) for q,u,p,s in zip(qty_parsed, unit_std, parts_series, unit_src) ]

    out = pd.DataFrame({
        "qty_parsed": qty_parsed.astype("Float64"),
        "unit_std": unit_std.astype("string[pyarrow]"),
        "audit_flags": audit_flags
    })
    return out
# ======== [END ADD] ========

def _infer_unit_by_ingredient(ingredient_norm: str):
    """
    简易单位先验：仅用于无单位/失败残差；返回 unit_std 或 None
    """
    if not ingredient_norm:
        return None
    x = ingredient_norm.lower()
    # 液体/乳制：tbsp/cup；粉类：cup；整果蔬：piece；香料瓣：clove
    if any(k in x for k in ["oil","milk","cream","vinegar","sauce"]):
        return "tbsp"
    if any(k in x for k in ["flour","sugar","powder","starch","cornmeal"]):
        return "cup"
    if any(k in x for k in ["garlic"]):
        return "clove"
    if any(k in x for k in ["onion","tomato","apple","banana","potato","egg"]):
        return "piece"
    return None

def _safe_alias_unit(unit_raw: str):
    """
    安全别名修正：仅在 ENABLE_FUZZY=True 且残差样本上启用
    """
    if _is_missing(unit_raw):
        return None
    cand = _nearest_unit_alias(str(unit_raw))
    return cand

def _lane_conf_from_path(path_used: str, n: int = 0):
    """
    根据路径推断 lane 与基础置信度。n 可传映射样本量提升细化（未知则置0）。
    """
    # lane
    if path_used in ("fdc_density","token_density","cat_density","global_volume"):
        lane = "volume"
    elif path_used in ("fdc_piece","token_piece","cat_piece","global_piece","a_table_piece"):
        lane = "piece"
    elif path_used in ("global_other",):
        lane = "weight"  # 其他量纲但有全局换算（如 'oz','lb','kg','g'）
    elif path_used in ("impute_packsize","infer_unit","alias_fix","global_default"):
        lane = "fallback"
    else:
        lane = "fallback"
    # conf
    base = {
        "fdc_density":0.95, "token_density":0.85, "cat_density":0.80, "global_volume":0.70,
        "fdc_piece":0.93,   "token_piece":0.85,   "cat_piece":0.80,   "global_piece":0.65, "a_table_piece":0.92,
        "global_other":0.90,
        "impute_packsize":0.80, "infer_unit":0.60, "alias_fix":0.55, "global_default":0.50,
        "no_match":0.0
    }.get(path_used, 0.5)
    # 样本量轻微加成（上限+0.1）
    if n and isinstance(n, (int,float)) and n>0:
        base = min(1.0, base + 0.02 * min(5, np.log1p(n)))
    return lane, float(base)

def decide_lane_and_grams(row, impute_units=True):
    """
    统一入口：先走你的 estimate_grams_enhanced(return_path=True)，失败则兜底。
    返回: grams, lane, path_used, conf, unit_imputed_flag, audit_flags(list[str])
    """
    flags = []
    # 先尝试主通路
    grams, path = estimate_grams_enhanced(row, return_path=True)
    if grams is not None:
        lane, conf = _lane_conf_from_path(path)
        return grams, lane, path, conf, False, flags

    qty_parsed = row.get("qty_parsed")
    unit_std   = row.get("unit_std")
    qty_raw    = row.get("qty_raw")
    ingr_norm  = row.get("ingredient_norm","")

    # 基础保护和类型转换
    if pd.isna(qty_parsed):
        flags.append("no_qty")
        return None, "fallback", "no_match", 0.0, False, flags
    
    # 确保 qty_parsed 是数值类型
    try:
        qty_parsed = float(qty_parsed)
        if qty_parsed <= 0:
            flags.append("non_positive")
            return None, "fallback", "no_match", 0.0, False, flags
    except (ValueError, TypeError):
        flags.append("invalid_qty")
        return None, "fallback", "no_match", 0.0, False, flags

    # [ORDER] 先尝试包装×规格展开，再进入单位识别/名词单位
    pk = _expand_packsize(qty_raw)
    if pk:
        mult, size, u = pk
        tmp = row.copy()
        tmp["qty_parsed"] = float(mult * size)
        tmp["unit_std"]   = u
        g2, p2 = estimate_grams_enhanced(tmp, return_path=True)
        if g2 is not None:
            lane, conf = _lane_conf_from_path("impute_packsize")
            return g2, lane, "impute_packsize", conf, False, flags

    # 子属性歧义：出现 juice|zest|peel|rind|seed(s)?|skin 等时，禁用名词单位件重
    SUBATTR = re.compile(r"(?i)\b(juice|zest|peel|rind|seeds?|skin|fillet|breast|thigh|drumstick|wing)\b")
    text_ctx = " ".join([str(row.get("ingredient_raw","")), str(row.get("ingredient_norm",""))]).lower()
    has_subattr = bool(SUBATTR.search(text_ctx))

    # [PATCH-FALLBACK] 名词单位识别（仅当无子属性歧义）
    if not has_subattr and ((unit_std is None or pd.isna(unit_std)) and NOUN_PIECE_WEIGHT):
        m = re.search(r'(?i)^\s*(\d+(?:\.\d+)?(?:\s+\d+/\d+)?)\s+([A-Za-z]+)\b', str(row.get("ingredient_raw","")))
        if m:
            noun = singularize(m.group(2))
            if noun in NOUN_PIECE_WEIGHT:
                qty = float(qty_parsed or m.group(1))
                grams = qty * float(NOUN_PIECE_WEIGHT[noun])
                return grams, "piece", "a_table_piece", 0.92, False, flags

    # 注意：包装×规格已在前面优先展开

    # 兜底2：单位推断（仅对失败样本且允许impute）
    imputed = False
    valid_units = (VOLUME_UNITS | PIECE_UNITS | WEIGHT_UNITS)
    if impute_units and (_is_missing(unit_std) or str(unit_std) not in valid_units):
        guess = _infer_unit_by_ingredient(ingr_norm)
        if guess:
            imputed = True
            tmp = row.copy()
            tmp["unit_std"] = guess
            # 确保 qty_parsed 是数值类型
            if pd.notna(tmp.get("qty_parsed")):
                tmp["qty_parsed"] = float(tmp["qty_parsed"])
            g3, p3 = estimate_grams_enhanced(tmp, return_path=True)
            if g3 is not None:
                lane, conf = _lane_conf_from_path("infer_unit")
                return g3, lane, "infer_unit", conf, True, flags
            else:
                flags.append("infer_unit_fail")

    # 兜底3：别名修正（需启 fuzzy）
    valid_units = (VOLUME_UNITS | PIECE_UNITS | WEIGHT_UNITS)
    if impute_units and (_is_missing(unit_std) or str(unit_std) not in valid_units):
        alias = _safe_alias_unit(unit_std)
        if alias and alias != unit_std:
            tmp = row.copy()
            tmp["unit_std"] = alias
            # 确保 qty_parsed 是数值类型
            if pd.notna(tmp.get("qty_parsed")):
                tmp["qty_parsed"] = float(tmp["qty_parsed"])
            g4, p4 = estimate_grams_enhanced(tmp, return_path=True)
            if g4 is not None:
                lane, conf = _lane_conf_from_path("alias_fix")
                return g4, lane, "alias_fix", conf, True, flags

    # 兜底4：全局默认（UNIT_GLOBAL）
    if unit_std in UNIT_GLOBAL:
        g5 = float(qty_parsed) * float(UNIT_GLOBAL[unit_std])
        lane, conf = _lane_conf_from_path("global_default")
        return g5, lane, "global_default", conf, imputed, flags

    flags.append("no_match")
    return None, "fallback", "no_match", 0.0, imputed, flags

def _build_usda_linked_mappings(food_df, a_table, grams_col):
    """联表USDA构建density_by_cat和density_by_token"""
    global density_by_cat, density_by_token, piece_weight_by_cat, piece_weight_by_token
    
    print("     【D】USDA联表构建category和token映射...")
    
    # 获取A表中的fdc_id和对应的密度/重量数据
    a_clean = a_table.groupby(["fdc_id", "unit_std"])[grams_col].apply(
        lambda x: _winsorize_iqr(x).median() if len(x) > 1 else x.iloc[0]
    ).reset_index()
    a_clean.columns = ["fdc_id", "unit_std", "grams_per_unit_clean"]
    
    # 体积单位：构建density_by_cat和density_by_token
    volume_data = a_clean[a_clean["unit_std"].isin(VOLUME_UNITS)].copy()
    if not volume_data.empty:
        volume_data["density"] = volume_data.apply(
            lambda row: row["grams_per_unit_clean"] / ML_PER.get(row["unit_std"], 1), 
            axis=1
        )
        
        # 联表food_df获取category信息
        volume_with_cat = volume_data.merge(
            food_df[["fdc_id", "food_category_id", "description_norm"]], 
            on="fdc_id", how="inner"
        )
        
        # 按category聚合密度
        if "food_category_id" in volume_with_cat.columns:
            cat_density = volume_with_cat.groupby("food_category_id")["density"].median()
            density_by_cat.update(cat_density.to_dict())
            print(f"     体积单位category映射: {len(density_by_cat)} 个category")
        
        # 从description_norm提取关键词构建token映射
        keywords = ['milk', 'oil', 'flour', 'syrup', 'cheese', 'bread', 'garlic', 
                   'onion', 'banana', 'tomato', 'butter', 'cream', 'sugar', 'salt']
        
        for keyword in keywords:
            keyword_data = volume_with_cat[
                volume_with_cat["description_norm"].str.contains(keyword, case=False, na=False)
            ]
            if len(keyword_data) >= 5:  # 频次≥5才保留
                density_by_token[keyword] = keyword_data["density"].median()
        
        print(f"     体积单位token映射: {len(density_by_token)} 个token")
        
        # 断言检测density_by_token的值不是函数对象
        for key, value in density_by_token.items():
            assert not callable(value), f"density_by_token[{key}] 是函数对象: {value}"
            assert isinstance(value, (int, float)), f"density_by_token[{key}] 不是数值: {type(value)}"
    
    # 件数单位：构建piece_weight_by_cat和piece_weight_by_token
    piece_data = a_clean[a_clean["unit_std"].isin(PIECE_UNITS)].copy()
    if not piece_data.empty:
        # 联表food_df获取category信息
        piece_with_cat = piece_data.merge(
            food_df[["fdc_id", "food_category_id", "description_norm"]], 
            on="fdc_id", how="inner"
        )
        
        # 按(unit, category)聚合重量
        if "food_category_id" in piece_with_cat.columns:
            cat_weight = piece_with_cat.groupby(["unit_std", "food_category_id"])["grams_per_unit_clean"].median()
            piece_weight_by_cat.update(cat_weight.to_dict())
            print(f"     件数单位category映射: {len(piece_weight_by_cat)} 个(unit, category)对")
        
        # 从description_norm提取关键词构建token映射
        for keyword in keywords:
            keyword_data = piece_with_cat[
                piece_with_cat["description_norm"].str.contains(keyword, case=False, na=False)
            ]
            if len(keyword_data) >= 5:  # 频次≥5才保留
                for unit in keyword_data["unit_std"].unique():
                    unit_keyword_data = keyword_data[keyword_data["unit_std"] == unit]
                    if len(unit_keyword_data) >= 3:  # 每个unit至少3个样本
                        piece_weight_by_token[(unit, keyword)] = unit_keyword_data["grams_per_unit_clean"].median()
        
        print(f"     件数单位token映射: {len(piece_weight_by_token)} 个(unit, token)对")

def load_a_table():
    """
    加载 A 表并计算单位中位数，实施分层映射改造
    
    功能说明：
        1. 加载A表（家庭常用单位重量转换数据）
        2. 计算每个单位的中位数转换值
        3. 构建分层映射策略
        4. 动态扩展单位词表
    
    A表作用：
        - 提供家庭常用单位的重量转换数据
        - 支持体积→密度、件数→重量的转换
        - 提升单位识别和克重估算精度
    
    返回：
        bool: 加载成功返回True，失败返回False
    """
    global A_TABLE, A_UNIT_MEDIAN, UNIT_GLOBAL
    global density_by_fdc, density_by_cat, density_by_token
    global piece_weight_fdc, piece_weight_by_cat, piece_weight_by_token
    
    print(f"🔍 正在查找 A 表文件...")
    print(f"   目标路径: {A_TABLE_PATH}")
    print(f"   文件存在: {os.path.exists(A_TABLE_PATH)}")
    
    # 检查A表文件是否存在
    if not os.path.exists(A_TABLE_PATH):
        warnings.warn(f"A 表文件不存在: {A_TABLE_PATH}，将降级运行")
        print(f"   请确保 A 表文件存在于以下位置之一:")
        print(f"   - data/household_weights_A.csv")
        print(f"   - household_weights_A.csv")
        print(f"   - /mnt/data/household_weights_A.csv")
        print(f"   - D:/data/household_weights_A.csv")
        return False
    
    try:
        print(f"📂 正在加载 A 表: {A_TABLE_PATH}")
        print(f"   命中路径: {A_TABLE_PATH}")
        A_TABLE = pd.read_csv(A_TABLE_PATH)
        print(f"✅ 成功加载 A 表: {len(A_TABLE)} 条记录")
        
        # 检查A表列名
        print(f"   A 表列名: {list(A_TABLE.columns)}")
        
        # 允许多种列名写法
        unit_col_candidates = ["unit_std", "unit", "units", "Unit", "UnitStd"]
        grams_col_candidates = ["grams_per_unit", "g_per_unit", "grams", "gram_per_unit", "g"]

        def pick(colnames, df):
            for c in colnames:
                if c in df.columns:
                    return c
            raise KeyError(f"未找到需要的列名，候选：{colnames}")

        unit_col  = pick(unit_col_candidates,  A_TABLE)
        grams_col = pick(grams_col_candidates, A_TABLE)
        
        print(f"   使用单位列: {unit_col}")
        print(f"   使用克重列: {grams_col}")

        # 归一化单位
        A_TABLE["unit_std"] = A_TABLE[unit_col].map(_normalize_unit_local)

        # 中位数兜底（按 unit_std 聚合）
        A_UNIT_MEDIAN = (A_TABLE.dropna(subset=["unit_std", grams_col])
                                  .groupby("unit_std")[grams_col]
                                  .median().to_dict())
        UNIT_GLOBAL = A_UNIT_MEDIAN

        print(f"✅ 成功加载 A 表：{len(A_TABLE)} 行，单位种类 {len(UNIT_GLOBAL)}")
        print(f"   单位示例: {list(A_UNIT_MEDIAN.keys())[:5]}")
        print(f"   使用列名: {unit_col} -> {grams_col}")
        
        # 【A】A表标准化与过滤
        print("   正在实施A表分层映射改造...")
        _build_layered_mappings(A_TABLE, grams_col)
        
        # 提取名词单位
        print("   正在从A表提取名词单位...")
        extract_noun_units_from_A(A_TABLE)
        
        # 动态扩展单位词表
        print("   正在从A表扩展单位词表...")
        rebuild_unit_regex_from_A()
        
        return True
    except Exception as e:
        warnings.warn(f"加载 A 表失败：{e}，将降级运行")
        print(f"   错误详情: {str(e)}")
        return False

# 预处理：统一分数字符 + 断开"数字-单位"贴连 + 去括号/连字符
trans_tbl = str.maketrans({"¼":"1/4","½":"1/2","¾":"3/4","⅓":"1/3","⅔":"2/3","⅛":"1/8","⅜":"3/8","⅝":"5/8","⅞":"7/8","⁄":"/"})

def _norm_num(s: pd.Series) -> pd.Series:
    """向量化数值文本标准化"""
    s = s.fillna("").astype("string[pyarrow]").str.translate(trans_tbl)
    return s

def _normalize_numeric_text(text):
    """数值文本标准化：处理 Unicode 分数和特殊字符（优化版）"""
    if _is_missing(text):
        return ""
    
    s = str(text).strip()
    
    # 使用预编译的正则表达式处理Unicode分数
    if UNICODE_FRACTION_PATTERN.search(s):
        # Unicode 分数映射
        unicode_fractions = {
            "¼": "1/4", "½": "1/2", "¾": "3/4",
            "⅓": "1/3", "⅔": "2/3",
            "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8"
        }
        
        for unicode_char, ascii_equiv in unicode_fractions.items():
            s = s.replace(unicode_char, ascii_equiv)
    
    # 替换 U+2044 '⁄' 为 '/'
    s = s.replace("⁄", "/")
    
    return s

@lru_cache(maxsize=100000)
def _parse_fraction_to_float(fraction_str):
    """将分数字符串转换为浮点数（优化版）"""
    if _is_missing(fraction_str):
        return None
    
    try:
        s = str(fraction_str).strip()
        
        # 使用预编译的正则表达式处理分数
        match = FRACTION_PATTERN.match(s)
        if match:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator != 0:
                return numerator / denominator
        
        # 处理小数形式
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None

def parse_qty_number_only(s):
    """
    只返回数值的parse_qty包装器
    
    功能说明：
        调用parse_qty函数，如果返回元组或列表，则只取第一个元素（数值部分）
        确保返回值是纯数值，避免tuple类型错误
    
    参数：
        s (str): 需要解析的数量字符串
    
    返回：
        float or None: 解析后的数值，失败时返回None
    """
    out = parse_qty(s)
    if isinstance(out, (tuple, list)):
        out = out[0] if out else None
    return out

def _normalize_for_unit(s: pd.Series) -> pd.Series:
    """
    预归一化：把数字与字母贴合处断开、去掉连字符影响、统一括号空格
    
    功能说明：
        对pandas Series进行预归一化处理，确保单位识别能正确匹配：
        1. 断开数字与字母的贴连：8oz → 8 oz
        2. 去除括号：(14 ounce) → 14 ounce
        3. 处理连字符：12-ounce → 12 ounce
        4. 统一空格：多个空格合并为一个
    
    参数：
        s (pd.Series): 需要预归一化的字符串Series
    
    返回：
        pd.Series: 预归一化后的字符串Series
    """
    return (s
        .str.replace(r'(\d)([a-zA-Z])', r'\1 \2', regex=True)   # 8oz→8 oz
        .str.replace(r'[\(\)\[\]{}]', ' ', regex=True)          # 去括号
        .str.replace(r'[-–—]', ' ', regex=True)                 # 连字符→空格
        .str.replace(r'\s+', ' ', regex=True)                   # 多空格
        .str.strip()
        .astype("string")
    )

def _nearest_unit_alias(token):
    """模糊别名兜底：对长度≥3的token，若与UNIT_SYNONYMS_LOCAL的key编辑距离≤1，则用其映射"""
    # 性能优化：先检查开关
    if not ENABLE_FUZZY:
        return None
        
    if not token or len(token) < 3:
        return None
    
    token_lower = token.lower()
    
    # 计算编辑距离的简单实现
    def edit_distance(s1, s2):
        if len(s1) < len(s2):
            return edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # 寻找编辑距离≤1的匹配
    for key in UNIT_SYNONYMS_LOCAL.keys():
        if edit_distance(token_lower, key) <= 1:
            return UNIT_SYNONYMS_LOCAL[key]
    
    return None

def _normalize_unit_local(unit_str):
    if _is_missing(unit_str):
        return None
    s = str(unit_str).strip().lower()
    s = s.replace("⁄", "/").replace("–", "-").replace("—", "-")
    s = s.replace(".", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if s in UNIT_SYNONYMS_LOCAL:
        return UNIT_SYNONYMS_LOCAL[s]
    if re.search(r"(cups?|tablespoons?|teaspoons?|cloves?|slices?|sticks?|bunches?|sprigs?|cans?|packages?|jars?|bags?|heads?|leaves?|grams?|kilograms?|ounces?|pounds?|milliliters?|liters?|pints?|quarts?|gallons?|servings?)", s):
        s = re.sub(r"s$", "", s)
        s = s.replace("millilitres", "milliliter").replace("milliliters", "milliliter")
        s = s.replace("litres", "liter")
    if s in UNIT_SYNONYMS_LOCAL:
        return UNIT_SYNONYMS_LOCAL[s]
    s = s.replace("fl oz", "fl_oz")
    s = s.replace(" ", "_")
    return s or None

# =============================================================================
# 列精简配置
# =============================================================================

# 各输出文件的保留列定义
# 支持两种模式：safe_min（保留核心列）和ultra_min（仅保留必要列）
KEEP_COLS = {
    "food_processed.parquet": {
        # 食物信息表
        "safe_min": [
            "fdc_id",              # 食物ID
            "data_type",           # 数据类型
            "description_norm",    # 标准化描述
            "nutrient_coverage",   # 营养覆盖度
            "dtype_bonus",         # 数据类型奖励分
            "publication_date",    # 发布日期
            "food_category_id",    # 食物类别ID
            "source"               # 数据源
        ],
        "ultra_min": [
            "fdc_id", "data_type", "description_norm", 
            "nutrient_coverage", "dtype_bonus", "source"
        ]
    },
    "nutrient_processed.parquet": {
        # 营养素信息表
        "safe_min": [
            "id",           # 营养素ID
            "name",         # 营养素名称
            "unit_name",    # 单位名称
            "rank",         # 排序权重
            "nutrient_nbr"  # 营养素编号
        ],
        "ultra_min": ["id", "name", "unit_name", "rank"]
    },
    "food_nutrient_processed.parquet": {
        # 食物-营养素关系表
        "safe_min": [
            "fdc_id",         # 食物ID
            "nutrient_id",    # 营养素ID
            "amount_robust",  # 鲁棒营养含量
            "amount",         # 原始营养含量
            "median",         # 中位数
            "min",            # 最小值
            "max"             # 最大值
        ],
        "ultra_min": ["fdc_id", "nutrient_id", "amount_robust"]
    },
    "recipes_processed.parquet": {
        # 食谱信息表
        "safe_min": [
            "recipe_id",           # 食谱ID
            "title",               # 食谱标题
            "ingredient_parts",    # 配料部分
            "ingredient_qties",    # 配料数量
            "servings",            # 份数
            "DatePublished"        # 发布日期
        ],
        "ultra_min": ["recipe_id", "ingredient_parts", "ingredient_qties", "servings"]
    },
    "ingredients_processed.parquet": {
        # 配料信息表
        "safe_min": [
            "recipe_id",           # 食谱ID
            "ingredient_raw",      # 原始配料文本
            "ingredient_norm",     # 标准化配料名
            "qty_raw",             # 原始数量文本
            "qty_parsed",          # 解析后数量
            "unit_std",            # 标准化单位
            "grams",               # 估算克重
            "servings",            # 份数
            "unit_imputed_flag",   # 单位回填标志
            "path_used",           # 克重估算路径
            # >>> [PATCH-COLS] 追加三列 <<<
            "lane",               # 四车道标记: weight/volume/piece/fallback
            "conf",               # 置信度(0~1)
            "audit_flags"         # JSON字符串数组，记录异常/兜底原因
        ],
        "ultra_min": [
            "recipe_id", "ingredient_norm", "qty_parsed", 
            "unit_std", "grams", "servings", "unit_imputed_flag", "path_used",
            "lane", "conf", "audit_flags"
        ]
    }
}

def _apply_keep(df, keep_cols):
    """
    筛选DataFrame保留指定列，并调整数值类型
    
    功能说明：
        1. 根据列精简配置筛选保留的列
        2. 调整数值列的数据类型，优化存储空间
        3. 处理缺失列的情况，提供警告信息
    
    数据类型优化：
        - int32: 用于nutrient_coverage等中等范围整数
        - int16: 用于dtype_bonus等小范围整数
        - float32: 用于qty_parsed、grams、servings等浮点数
    
    参数：
        df (pd.DataFrame): 需要处理的DataFrame
        keep_cols (list): 需要保留的列名列表
    
    返回：
        pd.DataFrame: 筛选和优化后的DataFrame
    """
    # 空数据检查
    if df is None or df.empty:
        return df
    
    # 筛选存在的列
    existing_cols = [col for col in keep_cols if col in df.columns]
    missing_cols = [col for col in keep_cols if col not in df.columns]
    
    # 处理缺失列
    if missing_cols:
        warnings.warn(f"缺少列: {missing_cols}")
    
    if not existing_cols:
        warnings.warn("没有可保留的列")
        return df
    
    # 筛选列
    result_df = df[existing_cols].copy()
    
    # 调整数值类型，优化存储空间
    if "nutrient_coverage" in result_df.columns:
        result_df["nutrient_coverage"] = result_df["nutrient_coverage"].astype("int32")
    if "dtype_bonus" in result_df.columns:
        result_df["dtype_bonus"] = result_df["dtype_bonus"].astype("int16")
    if "qty_parsed" in result_df.columns:
        result_df["qty_parsed"] = result_df["qty_parsed"].astype("float32")
    if "grams" in result_df.columns:
        result_df["grams"] = result_df["grams"].astype("float32")
    if "servings" in result_df.columns:
        result_df["servings"] = result_df["servings"].astype("float32")
    # >>> [PATCH-DTYPE] conf类型 <<<
    if "conf" in result_df.columns:
        result_df["conf"] = result_df["conf"].astype("float32")
    
    return result_df

def load_usda_and_fdnn(usda_dir, fdnn_dir, filter_types=None):
    """
    同时加载 USDA 和 FNDDS 数据源，返回两个数据源的食物、营养信息
    
    功能说明：
        1. 并行加载USDA和FNDDS营养数据库
        2. 合并两个数据源，去重处理
        3. 数据清洗和标准化
        4. 构建倒排索引用于后续匹配
        5. 应用data_type过滤（可选）
    
    数据源说明：
        - USDA: 美国农业部营养数据库，包含基础食物信息
        - FNDDS: 食物和营养素数据库系统，包含加工食品信息
    
    参数：
        usda_dir (str): USDA CSV文件目录路径
        fdnn_dir (str): FNDDS CSV文件目录路径
        filter_types (str, optional): 过滤的data_type，逗号分隔
    
    返回：
        tuple: (food, nutr, fn, inv_index, nutr_usda_raw, nutr_fdnn_raw)
            - food: 合并后的食物信息DataFrame
            - nutr: 合并后的营养素信息DataFrame
            - fn: 合并后的食物-营养素关系DataFrame
            - inv_index: 倒排索引字典
            - nutr_usda_raw: 原始USDA营养素信息
            - nutr_fdnn_raw: 原始FNDDS营养素信息
    """
    # 设置pandas选项以优化字符串存储
    pd.set_option("mode.string_storage", "pyarrow")
    
    # 完整的列名定义（严格按真实列名）
    food_cols = ["fdc_id", "data_type", "description", "food_category_id", "publication_date"]
    nutr_cols = ["id", "name", "unit_name", "nutrient_nbr", "rank"]
    fn_cols = ["id", "fdc_id", "nutrient_id", "amount", "data_points", "derivation_id", 
               "min", "max", "median", "footnote", "min_year_acquired"]
    
    print("   正在加载 USDA 数据...")
    with tqdm(total=3, desc="加载USDA数据") as pbar:
        # 使用 pyarrow 引擎和优化的数据类型
        food_usda = pd.read_csv(
            os.path.join(usda_dir, "food.csv"), usecols=food_cols,
            dtype={"fdc_id":"int64","data_type":"string[pyarrow]","description":"string[pyarrow]",
                   "food_category_id":"Int64","publication_date":"string[pyarrow]"}
        )
        pbar.update(1)
        
        nutr_usda = pd.read_csv(
            os.path.join(usda_dir, "nutrient.csv"), usecols=nutr_cols,
            dtype={"id":"int64","name":"string[pyarrow]","unit_name":"string[pyarrow]",
                   "nutrient_nbr":"string[pyarrow]","rank":"float32"}
        )
        pbar.update(1)
        
        fn_usda = pd.read_csv(
            os.path.join(usda_dir, "food_nutrient.csv"), usecols=fn_cols,
            dtype={"id":"Int64","fdc_id":"int64","nutrient_id":"int64","amount":"float32",
                   "data_points":"Int64","derivation_id":"Int64","min":"float32","max":"float32",
                   "median":"float32","footnote":"string[pyarrow]","min_year_acquired":"Int64"},
            low_memory=False
        )
        pbar.update(1)

    print("   正在加载 FNDDS 数据...")
    with tqdm(total=3, desc="加载FNDDS数据") as pbar:
        # 加载 FNDDS 数据（相同列约束）
        food_fdnn = pd.read_csv(
            os.path.join(fdnn_dir, "food.csv"), usecols=food_cols,
            dtype={"fdc_id":"int64","data_type":"string[pyarrow]","description":"string[pyarrow]",
                   "food_category_id":"Int64","publication_date":"string[pyarrow]"}
        )
        pbar.update(1)
        
        nutr_fdnn = pd.read_csv(
            os.path.join(fdnn_dir, "nutrient.csv"), usecols=nutr_cols,
            dtype={"id":"int64","name":"string[pyarrow]","unit_name":"string[pyarrow]",
                   "nutrient_nbr":"string[pyarrow]","rank":"float32"}
        )
        pbar.update(1)
        
        fn_fdnn = pd.read_csv(
            os.path.join(fdnn_dir, "food_nutrient.csv"), usecols=fn_cols,
            dtype={"id":"Int64","fdc_id":"int64","nutrient_id":"int64","amount":"float32",
                   "data_points":"Int64","derivation_id":"Int64","min":"float32","max":"float32",
                   "median":"float32","footnote":"string[pyarrow]","min_year_acquired":"Int64"},
            low_memory=False
        )
        pbar.update(1)

    # 检查列名
    assert {"fdc_id","data_type","description","food_category_id","publication_date"}.issubset(set(food_usda.columns)), "USDA food.csv 缺少必要列"
    assert {"id","name","unit_name","nutrient_nbr","rank"}.issubset(set(nutr_usda.columns)), "USDA nutrient.csv 缺少必要列"
    assert {"fdc_id","nutrient_id","amount","min","max","median","footnote","min_year_acquired"}.issubset(set(fn_usda.columns)), "USDA food_nutrient.csv 缺少必要列"
    
    assert {"fdc_id","data_type","description","food_category_id","publication_date"}.issubset(set(food_fdnn.columns)), "FDNN food.csv 缺少必要列"
    assert {"id","name","unit_name","nutrient_nbr","rank"}.issubset(set(nutr_fdnn.columns)), "FDNN nutrient.csv 缺少必要列"
    assert {"fdc_id","nutrient_id","amount","min","max","median","footnote","min_year_acquired"}.issubset(set(fn_fdnn.columns)), "FDNN food_nutrient.csv 缺少必要列"

    # 解析 publication_date 为日期
    food_usda["publication_date"] = pd.to_datetime(food_usda["publication_date"], errors="coerce")
    food_fdnn["publication_date"] = pd.to_datetime(food_fdnn["publication_date"], errors="coerce")
    
    # 读完再打来源标签
    food_usda["source"] = "USDA"
    food_fdnn["source"] = "FNDDS"
    
    # 合并 USDA 和 FNDDS
    food = pd.concat([food_usda, food_fdnn], ignore_index=True)
    nutr = pd.concat([nutr_usda, nutr_fdnn], ignore_index=True).drop_duplicates(subset=["id"])
    fn = pd.concat([fn_usda, fn_fdnn], ignore_index=True).drop_duplicates(subset=["fdc_id", "nutrient_id"])  # 保留所有营养种类

    # 数据清洗和预处理
    print("   正在标准化描述文本...")
    # 使用向量化操作提高性能，同时保持精度
    with tqdm(total=1, desc="标准化描述文本") as pbar:
        food["description_norm"] = food["description"].astype(str).map(normalize_text)
        pbar.update(1)
    
    print("   正在生成tokens...")
    # 使用向量化操作提高性能，同时保持精度
    with tqdm(total=1, desc="生成tokens") as pbar:
        food["tokens"] = food["description_norm"].map(lambda s: [t for t in tokenize(s) if len(t)>=3])
        pbar.update(1)
    
    # 替换 amount_robust 计算
    fn["amount_robust"] = fn["amount"]
    
    # 用 median 补缺
    mask1 = fn["amount_robust"].isna() & fn["median"].notna()
    fn.loc[mask1, "amount_robust"] = fn.loc[mask1, "median"]
    
    # 用 (min+max)/2 再补
    mask2 = fn["amount_robust"].isna() & fn["min"].notna() & fn["max"].notna()
    fn.loc[mask2, "amount_robust"] = (fn.loc[mask2, "min"] + fn.loc[mask2, "max"]) / 2.0
    
    # 计算营养覆盖度：使用去重后的 (fdc_id, nutrient_id)
    fn_cov = fn[["fdc_id", "nutrient_id"]].drop_duplicates()
    coverage = fn_cov.groupby("fdc_id")["nutrient_id"].nunique().rename("nutrient_coverage")
    food = food.merge(coverage, on="fdc_id", how="left")
    food["nutrient_coverage"] = food["nutrient_coverage"].fillna(0).astype("int32")
    
    # 应用 data_type 过滤（在合并完成后）
    if filter_types:
        # 1) 在过滤前先打印 data_type 分布
        print("   合并 USDA/FNDDS 后的 data_type 分布 (前20):")
        data_type_counts = food['data_type'].str.lower().value_counts().head(20)
        for dtype, count in data_type_counts.items():
            print(f"     {dtype}: {count:,}")
        
        # 2) 人类可读名映射到真实 data_type
        dtype_mapping = {
            "foundation foods": "foundation_food",
            "foundation": "foundation_food", 
            "sr legacy": "sr_legacy_food",
            "legacy": "sr_legacy_food",
            "fndds": "survey_fndds_food",
            "survey": "survey_fndds_food",
            "sample": "sample_food",
            "sub sample": "sub_sample_food",
            "market": "market_acquisition",
            "market acquisition": "market_acquisition",
            "branded": "branded"
        }
        
        # 标准化过滤条件
        wanted = []
        for t in filter_types.split(","):
            t_clean = t.strip().lower()
            if t_clean in dtype_mapping:
                wanted.append(dtype_mapping[t_clean])
            else:
                # 对未命中的词用 .replace(" ", "_") 兜底
                wanted.append(t_clean.replace(" ", "_"))
        
        # 3) 应用标准化后的 wanted 列表进行过滤
        print(f"   过滤条件标准化为: {wanted}")
        food_filtered = food[food["data_type"].str.lower().isin(wanted)].reset_index(drop=True)
        
        # 4) 若过滤后记录数 < 10000，自动回退为"不过滤"
        if len(food_filtered) < 10000:
            print(f"   ⚠️  过滤后记录数 {len(food_filtered)} < 10000，自动回退为不过滤")
            print(f"   保持原始记录数: {len(food)} 条")
        else:
            food = food_filtered
            print(f"   过滤后保留 {len(food)} 条食物记录")
    else:
        print("   未设置过滤条件，保持所有 data_type")


    # 按 data_type 给点偏置：Foundation / SR Legacy / Survey(FNDDS) 更通用
    def dtype_bonus(x):
        x = str(x).lower()
        if "foundation" in x: return 10
        if "sr legacy" in x:  return 8
        if "survey" in x:     return 7  # FNDDS
        if "sample" in x:     return 4
        if "market" in x:     return 3
        if "branded" in x:    return 1
        return 0
    food["dtype_bonus"] = food["data_type"].map(dtype_bonus)
    
    # 添加 data_type_norm 用于去重
    food["data_type_norm"] = food["data_type"].str.lower()


    # 按规范化描述去重：优先覆盖度高、dtype_bonus 高的记录，减少冗余
    # 使用 description_norm+data_type_norm 作为去重key，防止cooked/raw被误删
    if not food.empty:
        food = food.sort_values(
            ["description_norm", "nutrient_coverage", "dtype_bonus", "publication_date"],
            ascending=[True, False, False, False],
            kind="mergesort"  # 稳定排序，兼容更广
        )

        food = food.drop_duplicates(subset=["description_norm", "data_type_norm"], keep="first").reset_index(drop=True)

    # 构建倒排索引 token -> 候选 fdc_id（减少模糊匹配的候选集合）
    def _ok(t): 
        return len(t) >= 3 and not t.isdigit()
    
    print("   正在构建倒排索引...")
    inv_index = defaultdict(set)
    
    # 使用向量化操作优化倒排索引构建，保持精度
    fdc_ids = food["fdc_id"].values
    tokens_list = food["tokens"].values
    
    # 保持原有的处理逻辑，确保精度
    for fdc_id, toks in tqdm(zip(fdc_ids, tokens_list), 
                            total=len(food), desc="构建倒排索引"):
        for t in toks:
            if _ok(t):
                inv_index[t].add(int(fdc_id))
    
    # 转换为tuple去重，避免环境差异
    print("   正在转换倒排索引...")
    inv_index = {k: tuple(sorted(v)) for k, v in tqdm(inv_index.items(), desc="转换索引")}
    
    # 清理内存
    del fdc_ids, tokens_list
    gc.collect()
    
    return food, nutr, fn, inv_index, nutr_usda, nutr_fdnn

def load_foodcom(recipes_path, reviews_path=None, enable_four_lanes=True, impute_units=True):
    """
    加载和解析Food.com食谱数据
    
    功能说明：
        1. 加载Food.com食谱parquet文件
        2. 解析配料和数量列表
        3. 标准化配料文本
        4. 单位识别和克重估算
        5. 先验回填缺失的单位信息
    
    数据处理流程：
        1. 解析配料和数量列表（支持JSON格式）
        2. 对齐配料和数量（处理长度不一致的情况）
        3. 展开配料为单独行
        4. 标准化配料文本
        5. 识别和标准化单位
        6. 解析数量（支持分数、小数等）
        7. 估算克重（使用分层映射策略）
        8. 先验回填（基于历史数据回填缺失单位）
    
    参数：
        recipes_path (str): 食谱parquet文件路径
        reviews_path (str, optional): 评论parquet文件路径（暂未使用）
    
    返回：
        tuple: (recipes, ingr_df)
            - recipes: 处理后的食谱DataFrame
            - ingr_df: 展开后的配料DataFrame
    """
    # 设置pandas选项以优化字符串存储
    pd.set_option("mode.string_storage", "pyarrow")
    
    # >>> [OPTIONAL] 再次尝试扩展单位词表（幂等）
    try:
        rebuild_unit_regex_from_A()
    except Exception as _:
        pass
    
    # 定义所有Food.com列名（按真实列名）
    expected_cols = [
        "RecipeId", "Name", "AuthorId", "AuthorName", "CookTime", "PrepTime", "TotalTime",
        "DatePublished", "Description", "Images", "RecipeCategory", "Keywords",
        "RecipeIngredientQuantities", "RecipeIngredientParts", "AggregatedRating",
        "ReviewCount", "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
        "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent",
        "ProteinContent", "RecipeServings", "RecipeYield", "RecipeInstructions"
    ]
    
    print("   正在加载 Food.com 数据...")
    with tqdm(total=1, desc="加载recipes") as pbar:
        recipes = pd.read_parquet(recipes_path)
        pbar.update(1)
    
    # 检查并保留所有原始列
    missing_cols = [col for col in expected_cols if col not in recipes.columns]
    if missing_cols:
        warnings.warn(f"Food.com recipes 缺少列: {missing_cols}")
    
    # 重命名标准列
    rename_map = {
        "RecipeId": "recipe_id",
        "Name": "title", 
        "RecipeIngredientParts": "ingredient_parts",
        "RecipeIngredientQuantities": "ingredient_qties",
        "RecipeInstructions": "instructions",
        "RecipeServings": "servings"
    }
    
    # 只重命名存在的列
    existing_rename_map = {k: v for k, v in rename_map.items() if k in recipes.columns}
    recipes = recipes.rename(columns=existing_rename_map)
    
    assert "recipe_id" in recipes.columns, "缺 RecipeId/recipe_id"
    assert "ingredient_parts" in recipes.columns, "缺 RecipeIngredientParts/ingredient_parts"
    
    # 解析配料和数量列表
    print("   正在解析配料和数量列表...")
    with tqdm(total=2, desc="解析配料数据") as pbar:
        recipes["ingredient_list"] = recipes["ingredient_parts"].map(parse_vec)
        pbar.update(1)
        
        if "ingredient_qties" in recipes.columns:
            recipes["qty_list"] = recipes["ingredient_qties"].map(parse_vec)
        else:
            recipes["qty_list"] = [[] for _ in range(len(recipes))]
        pbar.update(1)
    
    # 替换原先 align_lists + apply 那段
    def _safe_list(x):
        return x if isinstance(x, list) else []
    
    recipes["ingredient_list"] = recipes["ingredient_list"].map(_safe_list)
    recipes["qty_list"] = recipes["qty_list"].map(_safe_list)
    
    # 三种情形：
    # A) 两列都有：按最短长度对齐
    # B) 只有配料无数量：保留全部配料，qty 用等长的 None 列
    # C) 配料为空：直接空
    def align_lists(ing_list, qty_list):
        if not ing_list:  # C
            return [], []
        if qty_list:      # A
            m = min(len(ing_list), len(qty_list))
            return ing_list[:m], qty_list[:m]
        else:             # B
            return ing_list, [None] * len(ing_list)
    
    recipes[["ingredient_list", "qty_list"]] = recipes.apply(
        lambda row: pd.Series(align_lists(row["ingredient_list"], row["qty_list"])),
        axis=1
    )
    
    # 使用 DataFrame.explode 矢量化操作（高性能、安全对齐）
    print("   使用矢量化操作展开配料...")
    
    # 准备展开数据，包含servings用于后续营养换算
    explode_cols = ["recipe_id", "ingredient_list", "qty_list"]
    if "servings" in recipes.columns:
        explode_cols.append("servings")
    
    with tqdm(total=3, desc="展开配料数据") as pbar:
        ingr_df = recipes[explode_cols].copy()
        pbar.update(1)
        
        ingr_df = ingr_df.explode(["ingredient_list", "qty_list"], ignore_index=True)
        ingr_df = ingr_df.rename(columns={"ingredient_list": "ingredient_raw", "qty_list": "qty_raw"})
        pbar.update(1)
        
        # 过滤掉空的配料
        ingr_df = ingr_df[ingr_df["ingredient_raw"].notna()].reset_index(drop=True)
        pbar.update(1)
    
    # 文本标准化和克重估算（使用缓存）
    print("   正在标准化配料文本...")
    # 使用向量化操作提高性能，同时保持精度
    with tqdm(total=1, desc="标准化配料文本") as pbar:
        ingr_df["ingredient_norm"] = ingr_df["ingredient_raw"].map(normalize_text)
        pbar.update(1)
    
    # >>> [ADD] 先用强化版数量与单位解析补齐 qty_parsed / unit_std（向量化） <<<
    try:
        parsed_df = parse_qty_and_unit_vectorized_v2(
            parts_series=ingr_df["ingredient_raw"],
            qty_series=ingr_df["qty_raw"],
        )
        ingr_df["qty_parsed"] = parsed_df["qty_parsed"].astype("Float64")
        ingr_df["unit_std"] = parsed_df["unit_std"].astype("string[pyarrow]")
        ingr_df["audit_flags"] = parsed_df["audit_flags"]
    except Exception as e:
        warnings.warn(f"强化解析器失败，回退旧逻辑：{e}")

    # >>> [REPLACE-EST] 统一用四车道决策输出 grams/lane/path/conf/audit_flags <<<
    print("   正在增强单位识别和克重估算...")
    
    # 数量解析（使用包装器确保只返回数值）
    ingr_df["qty_norm"]  = _norm_num(ingr_df["qty_raw"].astype("string[pyarrow]"))
    # 仅在缺失时使用旧包装器补齐，避免覆盖强化解析结果
    _old_qty = ingr_df.get("qty_parsed")
    _qty_from_old = ingr_df["qty_norm"].map(parse_qty_number_only)
    if _old_qty is None:
        ingr_df["qty_parsed"] = _qty_from_old
    else:
        ingr_df["qty_parsed"] = pd.to_numeric(_old_qty, errors="coerce").fillna(_qty_from_old)
    
    # [PATCH-UNIT-NORM-1] 预清洗：拆开连写、统一分数、清理括号
    print("   正在预清洗配料文本...")
    
    # 预清洗：'8oz'→'8 oz'、'1½ cup'→'1 1/2 cup'、去掉多余括号噪声
    UNICODE_FRAC = {
        "¼": "1/4", "½": "1/2", "¾": "3/4",
        "⅐":"1/7","⅑":"1/9","⅒":"1/10","⅓":"1/3","⅔":"2/3","⅕":"1/5","⅖":"2/5","⅗":"3/5","⅘":"4/5","⅙":"1/6","⅚":"5/6","⅛":"1/8","⅜":"3/8","⅝":"5/8","⅞":"7/8"
    }
    def _normalize_pre(s: str) -> str:
        if not isinstance(s, str): return s
        # 1) 替换 Unicode 分数
        for k,v in UNICODE_FRAC.items():
            s = s.replace(k, " " + v + " ")
        # 2) 数字+字母连写拆开（8oz→8 oz，14ounce→14 ounce）
        s = re.sub(r'(?i)(\d)([a-zA-Z])', r'\1 \2', s)
        # 3) 去掉"单位后面的点"（tsp.→tsp）
        s = re.sub(r'(?i)\b(oz|tsp|tbsp|t|T|tsps?|tbsp?s?|tblsp?s?|lb|lbs|g|kg|ml|l)\.\b', r'\1', s)
        # 4) 规整括号（保留 packsize，但去掉噪声括号空格）
        s = re.sub(r'\s*\(\s*', ' (', s)
        s = re.sub(r'\s*\)\s*', ') ', s)
        return s

    ingr_df["ingredient_raw"] = ingr_df["ingredient_raw"].astype(str).map(_normalize_pre)
    
    # [PATCH-UNIT-NORM-2] 强化单位正则（覆盖重量/体积/件数别名、大小写、复数）
    print("   正在强化单位正则...")
    
    # 先尝试从A表扩展单位词表（幂等操作）
    try:
        rebuild_unit_regex_from_A()
    except Exception as e:
        print(f"   A表扩展单位词表失败: {e}")
    
    # 替换你原来的 UNIT 正则与标准化表
    # 构建包含所有单位的单一捕获组正则表达式
    all_units = list(UNIT_STD_MAP.keys())
    all_units = sorted(all_units, key=len, reverse=True)
    UNIT_REGEX_STR = rf'(?i)\b({"|".join(re.escape(u) for u in all_units)})\b'


    UNIT_REGEX = re.compile(UNIT_REGEX_STR)

    def _extract_unit_fast(s: str):
        if not isinstance(s, str): return None
        m = UNIT_REGEX.search(s)
        if not m: return None
        u = m.group(0).lower()
        return UNIT_STD_MAP.get(u, u)
    
    # [PATCH-UNIT-NORM-3] 向量化提取单位（禁用逐行正则）
    print("   正在向量化提取单位...")
    
    # 向量化提取单位 + 回填 A表扩展的单位（若你有 rebuild_unit_regex_from_A，可放在这之前调用）
    # 仅在缺失时从上下文抽取单位，避免覆盖强化解析结果
    _existing_unit = ingr_df.get("unit_std")
    _extracted_unit = pd.Series(
        ingr_df["ingredient_raw"].astype(str).str.extract(UNIT_REGEX, expand=False)
    ).str.lower().map(UNIT_STD_MAP).where(lambda s: s.notna(), None)
    if _existing_unit is None:
        ingr_df["unit_std"] = _extracted_unit
    else:
        ingr_df["unit_std"] = _existing_unit.fillna(_extracted_unit)

    # ——额外：对 (14 ounce) 或 (400 g) 这种"括号内规格"的第二通道再识别一次
    PACKSIZE_UNIT = re.compile(r'(?i)\(\s*[\d\.]+\s*([a-zA-Z]+)\s*\)')
    u2 = ingr_df["ingredient_raw"].astype(str).str.extract(PACKSIZE_UNIT, expand=False).str.lower().map(UNIT_STD_MAP)
    ingr_df["unit_std"] = ingr_df["unit_std"].fillna(u2)
    
    # [PATCH-UNIT-EXTRACT-NOUN] 名词单位通道（仅在仍然缺单位时触发）
    print("   正在识别名词单位...")
    if NOUN_PIECE_WEIGHT:
        # 仅对主通道没识别到标准单位的行做名词单位识别
        mask_no_std = ingr_df["unit_std"].isna()

        # (A) 有数字的： "2 large tomatoes", "3 onions"
        PAT_QTY_NOUN = re.compile(
            r'(?i)^\s*(\d+(?:\.\d+)?(?:\s+\d+/\d+)?)\s*(?:x\s*)?'      # 2 / 1 1/2 / 3.5 / 2x
            r'(?:extra[-\s]?large|extra[-\s]?small|very[-\s]?large|very[-\s]?small|large|medium|small|lg|md|sm)?\s*'
            r'([A-Za-z][A-Za-z\-]+)\b'
        )
        m_qty = ingr_df.loc[mask_no_std, "ingredient_raw"].str.extract(PAT_QTY_NOUN, expand=True)
        # 安全处理数量提取，避免tuple错误
        def safe_parse_qty(s):
            if pd.isna(s) or not isinstance(s, str):
                return None
            try:
                # 处理 "1 1/2" 格式
                if ' ' in s and '/' in s:
                    parts = s.split()
                    if len(parts) == 2 and '/' in parts[1]:
                        whole = float(parts[0])
                        frac_parts = parts[1].split('/')
                        if len(frac_parts) == 2:
                            frac = float(frac_parts[0]) / float(frac_parts[1])
                            return whole + frac
                # 处理普通数字
                return float(s)
            except (ValueError, TypeError):
                return None
        
        qty_num = m_qty[0].map(safe_parse_qty)
        noun1   = m_qty[1].map(singularize)

        hit_qty = noun1.notna() & noun1.isin(NOUN_PIECE_WEIGHT.keys())

        # (B) 无数字的：默认 qty=1，如 "onion", "tomatoes", "egg"
        PAT_NOUN_ONLY = re.compile(
            r'(?i)^\s*(?:extra[-\s]?large|extra[-\s]?small|large|medium|small|lg|md|sm)?\s*([A-Za-z][A-Za-z\-]+)\b'
        )
        noun2 = ingr_df.loc[mask_no_std & ~hit_qty, "ingredient_raw"].str.extract(PAT_NOUN_ONLY, expand=False).map(singularize)
        hit_noun = noun2.notna() & noun2.isin(NOUN_PIECE_WEIGHT.keys())

        # 写回 unit_std：命中的走名词单位
        unit_std = ingr_df["unit_std"].copy()
        unit_std.loc[mask_no_std & hit_qty]  = noun1[hit_qty]
        unit_std.loc[mask_no_std & hit_noun] = noun2[hit_noun]
        ingr_df["unit_std"] = unit_std

        # 写回 qty_parsed：无数字 → 设为 1
        qty_parsed = ingr_df.get("qty_parsed")
        if qty_parsed is None:
            ingr_df["qty_parsed"] = np.nan
        qty_parsed = ingr_df["qty_parsed"].copy()
        
        # 安全地处理数量赋值
        if hit_qty.any():
            qty_values = qty_num[hit_qty]
            # 确保所有值都是数值类型，过滤掉None值
            qty_values = pd.to_numeric(qty_values, errors='coerce')
            # 只更新非空值
            valid_mask = qty_values.notna()
            if valid_mask.any():
                qty_parsed.loc[mask_no_std & hit_qty & valid_mask] = qty_values[valid_mask]
        
        if hit_noun.any():
            qty_parsed.loc[mask_no_std & hit_noun] = 1.0
            
        # 确保qty_parsed是float32类型，处理任何异常值
        ingr_df["qty_parsed"] = pd.to_numeric(qty_parsed, errors='coerce').astype("float32")

        n_noun = int((mask_no_std & (hit_qty|hit_noun)).sum())
        print(f"     名词单位识别: {n_noun} 条")
    else:
        print("     无A表名词单位数据，跳过名词单位识别")
    
    # 调试信息：单位识别覆盖率
    unit_coverage = ingr_df["unit_std"].notna().mean() * 100
    print(f"   单位识别覆盖率: {unit_coverage:.2f}%")
    
    # 显示单位分布统计 - 修正统计口径
    if unit_coverage > 0:
        dist = (ingr_df["unit_std"].dropna().astype(str).value_counts().head(10))
        print("   单位分布Top10:")
        for k, v in dist.items():
            print(f"     {k}: {v} 次")
    
    # 最后保护：把不在"有效集合"的清掉
    VALID_UNITS = WEIGHT_UNITS | VOLUME_UNITS | PIECE_UNITS | set(NOUN_PIECE_WEIGHT.keys())
    invalid_mask = ~ingr_df["unit_std"].isin(VALID_UNITS)
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        ingr_df.loc[invalid_mask, "unit_std"] = None
        print(f"   清理无效单位: {invalid_count:,} 条")
    
    # 显示预清洗效果
    sample_before = ingr_df["ingredient_raw"].iloc[0] if len(ingr_df) > 0 else "N/A"
    print(f"   预清洗示例: {sample_before}")
    
    # [PATCH-LANES-VEC-1] 准备单位换算表（向量化）
    print("   正在准备单位换算表...")

    # ==== [NEW] 单位判别与解析：最小改动版 ======================================
    from pathlib import Path

    CFG_DIR = Path("work/recipebench/data/config")

    # --- 1) 加载三表
    def _load_units_tables():
        wl = pd.read_csv(CFG_DIR / "units_whitelist.csv")
        bl = pd.read_csv(CFG_DIR / "units_blacklist.csv")
        cu = pd.read_csv(CFG_DIR / "units_conditional.csv")
        # 归一化
        wl["unit"] = wl["unit"].str.strip().str.lower()
        bl["token"] = bl["token"].str.strip().str.lower()
        cu["unit"] = cu["unit"].str.strip().str.lower()
        cu["trigger_lemmas"] = cu["trigger_lemmas"].str.strip().str.lower()
        cu["size_key"] = cu["size_key"].fillna("").str.lower()
        return wl, bl, cu

    UNITS_WL, UNITS_BL, COND_UNITS = _load_units_tables()
    UNIT_WL_SET = set(UNITS_WL["unit"].tolist())
    UNIT_WL_ALIAS = {
        row["unit"]: re.compile(r"\b(?:" + row.get("alias", row["unit"]).replace("|", "|") + r")\b", re.I)
        for _, row in UNITS_WL.iterrows()
    }
    UNIT_BL_PAT = re.compile(r"\b(?:" + "|".join(map(re.escape, UNITS_BL["token"].tolist())) + r")\b", re.I)

    # 条件单位索引：unit -> list of rules
    from collections import defaultdict
    COND_IDX = defaultdict(list)
    for _, r in COND_UNITS.iterrows():
        COND_IDX[r["unit"]].append(r.to_dict())

    # 覆盖条件件数：若 A 表提供 (fdc_id, unit[, size_key]) -> grams_per_unit，则以 A 表为准
    try:
        if 'A_TABLE' in globals() and A_TABLE is not None and not A_TABLE.empty:
            unit_col = "unit_std" if "unit_std" in A_TABLE.columns else ("unit" if "unit" in A_TABLE.columns else None)
            grams_col = "grams_per_unit" if "grams_per_unit" in A_TABLE.columns else None
            if unit_col and grams_col and "fdc_id" in A_TABLE.columns:
                A_map = {}
                for _, row in A_TABLE.iterrows():
                    try:
                        key = (int(row["fdc_id"]), str(row[unit_col]).lower().strip(), "")
                        A_map[key] = float(row[grams_col])
                    except Exception:
                        continue
                for unit, rules in COND_IDX.items():
                    for r in rules:
                        try:
                            key = (int(r.get("fdc_id")), unit, (r.get("size_key", "") or "").lower())
                            if key in A_map:
                                r["grams_per_unit"] = A_map[key]
                        except Exception:
                            continue
    except Exception as _:
        pass

    # --- 2) 常用正则（数量/包装×规格）
    QTY_NUM = r"(?P<qty>\d+(?:\.\d+)?)"
    QTY_FRAC = r"(?P<qty_frac>\d+\s*/\s*\d+)"
    QTY = rf"(?:{QTY_NUM}|{QTY_FRAC})"

    PAT_WEIGHT = re.compile(rf"\b{QTY}\s*(?P<unit>g|grams?|kg|kilograms?|lb|pounds?|oz|ounces?)\b", re.I)
    PAT_VOLUME = re.compile(rf"\b{QTY}\s*(?P<unit>ml|millilit(?:er|re)s?|l|lit(?:er|re)s?|cups?|tsp|teaspoons?|tbsp|tablespoons?|fl\.\?\s*oz)\b", re.I)
    PAT_PACK = re.compile(
        rf"\b(?P<npack>\d+)\s*(?:x|\()\s*(?P<spec_qty>\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*(?P<spec_unit>oz|g|ml|l|lb)\)?\s*(?P<pack_unit>cans?|bottles?|cartons?|packages?)\b",
        re.I
    )

    # 反模式（强降噪）
    BLOCK_PATTERNS = [
        re.compile(r"\bolive\s+oil\b", re.I),
        re.compile(r"\b(tomato|chili|garlic)\s+(paste|sauce)\b", re.I),
        re.compile(r"\b(chicken|turkey)\s+breasts?\b", re.I),
    ]

    def _is_blocked(text: str) -> bool:
        return any(p.search(text) for p in BLOCK_PATTERNS) or UNIT_BL_PAT.search(text or "") is not None

    # --- 3) 解析工具
    def _to_float(q):
        if q is None: 
            return None
        s = str(q).strip()
        if "/" in s and " " in s:
            a, b = s.split(" ", 1)
            num, den = b.split("/")
            return float(a) + float(num) / float(den)
        if "/" in s:
            num, den = s.split("/")
            return float(num) / float(den)
        try:
            return float(s)
        except:
            return None

    def _convert_to_grams(qty, unit):
        unit = unit.lower().strip().replace("grams", "g").replace("gram", "g")
        unit = unit.replace("kilograms", "kg").replace("kilogram", "kg")
        unit = unit.replace("pounds", "lb").replace("pound", "lb")
        unit = unit.replace("ounces", "oz").replace("ounce", "oz")
        unit = unit.replace("milliliters", "ml").replace("millilitres", "ml").replace("milliliter", "ml").replace("millilitre", "ml")
        unit = unit.replace("liters", "l").replace("litres", "l").replace("liter", "l").replace("litre", "l")
        if unit == "g":   return qty
        if unit == "kg":  return qty * 1000
        if unit == "lb":  return qty * 453.59237
        if unit == "oz":  return qty * 28.349523125
        return None

    # --- 4) 条件单位触发（件数→克）
    def _conditional_piece_to_grams(row):
        ctx = (row.get("ctx") or row.get("ingredient_raw") or "").lower()
        if _is_blocked(ctx):
            return (None, None)

        head = (row.get("ingredient_head") or row.get("ingredient_norm") or "").split(",")[0].split(" ")[0].lower()
        size_hint = ""
        if re.search(r"\blarge\b", ctx):  size_hint = "large"
        elif re.search(r"\bmedium\b", ctx): size_hint = "medium"
        elif re.search(r"\bsmall\b", ctx):  size_hint = "small"

        for unit, rules in COND_IDX.items():
            if not re.search(rf"\b{QTY}\s*{unit}s?\b", ctx):
                continue
            if not any(head == r["trigger_lemmas"] for r in rules):
                continue
            m = re.search(rf"\b{QTY}\s*{unit}s?\b", ctx)
            qty = _to_float(m.group("qty") if m and m.groupdict().get("qty") else m.group(0).split()[0])
            if qty is None: 
                continue
            cand = None
            for r in rules:
                if size_hint and r.get("size_key","") == size_hint:
                    cand = r; break
            if cand is None:
                for r in rules:
                    if not r.get("size_key",""):
                        cand = r; break
            if cand is None:
                continue
            g_per = float(cand["grams_per_unit"])
            grams = qty * g_per
            return (grams, unit)
        return (None, None)

    # --- 5) 主入口：判别→解析（重量→体积→包装→条件件数）
    def parse_units_block(df: pd.DataFrame) -> pd.DataFrame:
        texts = df["ingredient_raw"].fillna("").astype(str)

        def _weight_g(text):
            m = PAT_WEIGHT.search(text)
            if not m: return None
            q = _to_float(m.group("qty"))
            return _convert_to_grams(q, m.group("unit"))

        df["g_by_weight"] = texts.apply(_weight_g)
        df["vol_hit"] = texts.str.contains(PAT_VOLUME)

        def _pack_g(text):
            m = PAT_PACK.search(text)
            if not m: return None
            npack = float(m.group("npack"))
            sq = _to_float(m.group("spec_qty"))
            su = m.group("spec_unit")
            g = _convert_to_grams(sq, su)
            return npack * g if g is not None else None

        df["g_by_pack"] = texts.apply(_pack_g)

        df["g_by_piece"] = None
        mask_candidate = ~texts.apply(_is_blocked)
        if "ingredient_head" not in df.columns:
            df["ingredient_head"] = df.get("ingredient_norm", df["ingredient_raw"]).str.extract(r"^([a-zA-Z\-]+)")
        for idx in df[mask_candidate].index:
            grams, used_unit = _conditional_piece_to_grams(df.loc[idx].to_dict())
            if grams is not None:
                df.at[idx, "g_by_piece"] = grams

        df["quantity_g_est_new"] = df["g_by_weight"].fillna(df["g_by_pack"]).fillna(df["g_by_piece"])
        df["lane_hit_weight"] = df["g_by_weight"].notna()
        df["lane_hit_pack"]   = df["g_by_pack"].notna()
        df["lane_hit_piece"]  = df["g_by_piece"].notna()
        df["lane_hit_volume"] = df["vol_hit"].fillna(False)

        # —— 体积→克：仅在 quantity_g_est_new 仍为空且命中体积时，按密度映射换算
        def _volume_to_grams_row(r):
            try:
                text = str(r.get("ingredient_raw", ""))
                m = PAT_VOLUME.search(text)
                if not m:
                    return None
                # 数量（可能是纯数或分数）
                q = m.group("qty") if m.groupdict().get("qty") else None
                if q is None and m.groupdict().get("qty_frac"):
                    q = m.group("qty_frac")
                qty = _to_float(q)
                if qty is None:
                    return None
                unit = m.group("unit")
                # 转毫升
                unit_norm = unit.lower().strip().replace("milliliters", "ml").replace("millilitres", "ml").replace("milliliter", "ml").replace("millilitre", "ml")
                unit_norm = unit_norm.replace("liters", "l").replace("litres", "l").replace("liter", "l").replace("litre", "l")
                unit_norm = unit_norm.replace("teaspoons", "tsp").replace("tablespoons", "tbsp").replace("cups", "cup").replace("fl. oz", "fl oz")
                ml_per = {
                    "ml": 1.0, "l": 1000.0, "tsp": 4.92892159375, "tbsp": 14.78676478125, "cup": 236.5882365, "fl oz": 29.5735,
                }.get(unit_norm)
                if ml_per is None:
                    return None
                ml = qty * ml_per
                # 密度优先：fdc → token → 默认1.0
                dens = None
                fdc_id = r.get("fdc_id")
                try:
                    if pd.notna(fdc_id) and fdc_id in density_by_fdc:
                        dens = float(density_by_fdc[fdc_id])
                except Exception:
                    pass
                if dens is None:
                    ing = str(r.get("ingredient_norm", "")).lower()
                    for tok, val in density_by_token.items():
                        try:
                            if tok and tok in ing:
                                dens = float(val); break
                        except Exception:
                            continue
                if dens is None:
                    dens = 1.0
                return ml * dens
            except Exception:
                return None

        vol_mask = df["lane_hit_volume"] & df["quantity_g_est_new"].isna()
        if bool(vol_mask.any()):
            df.loc[vol_mask, "quantity_g_est_new"] = df.loc[vol_mask].apply(_volume_to_grams_row, axis=1)

        return df
    # ==== [END NEW] ================================================================

    WEIGHT_TO_G = {"g":1.0, "kg":1000.0, "oz":28.349523125, "lb":453.59237}
    VOL_TO_ML    = {"ml":1.0, "l":1000.0, "tsp":4.92892159375, "tbsp":14.78676478125, "cup":236.5882365}
    # PIECE_UNITS 已经在全局作用域定义，直接使用
    
    # [PATCH-LANES-VEC-2] 解析数量（向量化，把分数/范围转成数值）
    # 接入：在数量解析前运行单位判别与解析块（最小改动版）
    try:
        ingr_df = parse_units_block(ingr_df)
        # 若新估计到克重，可作为权重优先路径供后续参考（不直接覆盖四车道结果）
    except Exception as e:
        warnings.warn(f"parse_units_block 运行失败，跳过：{e}")

    print("   正在向量化解析数量...")
    # 量：先提取主数字片段（含分数/连字符范围），再转小数
    NUM_TOKEN = re.compile(r'(?:(\d+(?:\.\d+)?)\s+(\d+/\d+))|(\d+/\d+)|(\d+(?:\.\d+)?)')
    def _parse_qty_one(s: str):
        if not isinstance(s, str): return np.nan
        m = NUM_TOKEN.search(s)
        if not m: return np.nan
        if m.group(1) and m.group(2):  # '1 1/2'
            a = float(m.group(1)); b = m.group(2); num,den = b.split('/'); return a + float(num)/float(den)
        if m.group(3):  # '1/2'
            num,den = m.group(3).split('/'); return float(num)/float(den)
        return float(m.group(4))  # '1.5' or '2'
    
    # 重新解析数量（覆盖之前的解析）
    # 仅在缺失时用弱解析补齐，保留强化解析结果
    _qty_existing = pd.to_numeric(ingr_df.get("qty_parsed"), errors="coerce")
    _qty_guess = ingr_df["ingredient_raw"].astype(str).map(_parse_qty_one).astype("float32")
    ingr_df["qty_parsed"] = _qty_existing.fillna(_qty_guess).astype("float32")
    
    # 初始化输出列
    ingr_df["grams"] = np.nan
    ingr_df["lane"] = "fallback"
    ingr_df["path_used"] = "no_match"
    ingr_df["conf"] = 0.0
    ingr_df["unit_imputed_flag"] = False
    ingr_df["audit_flags"] = "[]"
    
    # [PATCH-LANES-VEC-3] 向量化三条主车道
    print("   正在执行向量化三车道...")
    
    # 1) 重量直达
    mask_weight = ingr_df["unit_std"].isin(WEIGHT_TO_G.keys())
    weight_count = mask_weight.sum()
    if weight_count > 0:
        print(f"     重量车道: {weight_count:,} 条")
        ingr_df.loc[mask_weight, "grams"] = (ingr_df.loc[mask_weight, "qty_parsed"].astype(float)
                                             * ingr_df.loc[mask_weight, "unit_std"].map(WEIGHT_TO_G).astype(float))
        ingr_df.loc[mask_weight, "lane"] = "weight"
        ingr_df.loc[mask_weight, "path_used"] = "global_other"
        ingr_df.loc[mask_weight, "conf"] = 0.90
    
    # 2) 体积→密度（优先 fdc_density，其次 token_density/category_density，再到 global_volume）
    mask_volume = ingr_df["unit_std"].isin(VOL_TO_ML.keys())
    volume_count = mask_volume.sum()
    if volume_count > 0:
        print(f"     体积车道: {volume_count:,} 条")
        # 体积毫升
        ml = ingr_df.loc[mask_volume, "qty_parsed"].astype(float) * ingr_df.loc[mask_volume, "unit_std"].map(VOL_TO_ML).astype(float)
        
        # 合并密度（按优先级拼一列 density_g_per_ml）
        # 注意：在step1阶段，ingr_df还没有fdc_id列，只使用ingredient_norm进行匹配
        vol_df = ingr_df.loc[mask_volume, ["ingredient_norm"]].copy()
        vol_df["dens_tok"] = vol_df["ingredient_norm"].map(density_by_token)
        vol_df["dens_cat"] = vol_df["ingredient_norm"].map(density_by_cat)   # 或 category_id
        vol_df["density"]  = vol_df["dens_tok"].fillna(vol_df["dens_cat"]).fillna(1.0)  # 默认密度1.0
        
        ingr_df.loc[mask_volume, "grams"] = ml.values * vol_df["density"].astype(float).values
        
        # path & conf
        path_series = np.where(vol_df["dens_tok"].notna(), "token_density",
                        np.where(vol_df["dens_cat"].notna(), "cat_density", "global_volume"))
        ingr_df.loc[mask_volume, "path_used"] = path_series
        ingr_df.loc[mask_volume, "lane"] = "volume"
        conf_map = {"token_density":0.85,"cat_density":0.80,"global_volume":0.70}
        ingr_df.loc[mask_volume, "conf"] = pd.Series(path_series).map(conf_map).values
    
    # 3) 件数→每件克重（支持A表名词单位）
    PIECE_UNITS_DYNAMIC = PIECE_UNITS | set(NOUN_PIECE_WEIGHT.keys())
    mask_piece = ingr_df["unit_std"].isin(PIECE_UNITS_DYNAMIC)
    piece_count = mask_piece.sum()
    if piece_count > 0:
        print(f"     件数车道: {piece_count:,} 条")
        piece_df = ingr_df.loc[mask_piece, ["unit_std","ingredient_norm","qty_parsed"]].copy()
        piece_df["w_a_noun"] = piece_df["unit_std"].map(NOUN_PIECE_WEIGHT)

        # 你原来就有的映射（fdc/token/cat/global），保持不变；A 表优先
        # piece_df["w_fdc"] = ...
        # piece_df["w_tok"] = ...
        # piece_df["w_cat"] = ...
        UNIT_GLOBAL_PIECE_DEFAULT = 50.0  # 默认50g
        piece_df["w_final"] = (piece_df["w_a_noun"]
                                # .fillna(piece_df["w_fdc"])
                                # .fillna(piece_df["w_tok"])
                                # .fillna(piece_df["w_cat"])
                                .fillna(UNIT_GLOBAL_PIECE_DEFAULT))

        ingr_df.loc[mask_piece, "grams"] = piece_df["qty_parsed"].astype(float) * piece_df["w_final"].astype(float)
        ingr_df.loc[mask_piece, "lane"] = "piece"
        ingr_df.loc[mask_piece, "path_used"] = np.where(piece_df["w_a_noun"].notna(), "a_table_piece", ingr_df.loc[mask_piece, "path_used"].fillna("global_piece"))
        conf_map = {"a_table_piece":0.92,"fdc_piece":0.93,"token_piece":0.85,"cat_piece":0.80,"global_piece":0.65}
        ingr_df.loc[mask_piece, "conf"] = ingr_df.loc[mask_piece, "path_used"].map(conf_map).fillna(0.70)
        
        # 统计A表名词单位的使用情况
        a_noun_count = (piece_df["w_a_noun"].notna()).sum()
        if a_noun_count > 0:
            print(f"     A表名词单位: {a_noun_count:,} 条")
    
    # [PATCH-LANES-VEC-4] 残差集（~20%）再逐行兜底 + 正确显示进度
    print("   正在处理残差集...")
    residual_mask = ingr_df["grams"].isna()
    residual_count = residual_mask.sum()
    
    if residual_count > 0:
        print(f"     残差集: {residual_count:,} 条 ({residual_count/len(ingr_df)*100:.1f}%)")
        residual = ingr_df.loc[residual_mask].copy()
        
        def _fallback_row(row):
            g, lane, path, conf, imputed, flags = decide_lane_and_grams(row, impute_units=True)
            row["grams"] = g; row["lane"] = lane; row["path_used"] = path; row["conf"] = conf
            row["unit_imputed_flag"] = bool(imputed)
            row["audit_flags"] = json.dumps(flags) if flags else "[]"
            return row
        
        # 使用tqdm显示进度
        tqdm_auto.pandas(desc="四车道兜底(残差)")
        residual = residual.progress_apply(_fallback_row, axis=1)
        ingr_df.loc[residual.index, ["grams","lane","path_used","conf","unit_imputed_flag","audit_flags"]] = \
            residual[["grams","lane","path_used","conf","unit_imputed_flag","audit_flags"]].values
    else:
        print("     无残差集，向量化处理完成")
    
    # 统计处理结果
    processed_count = ingr_df["grams"].notna().sum()
    print(f"   四车道处理完成: {processed_count:,}/{len(ingr_df):,} 条 ({processed_count/len(ingr_df)*100:.1f}%)")
    
    # 快速自检（一定要看这三行）
    print("[check] noun_piece_weight:", len(NOUN_PIECE_WEIGHT))
    print("[check] unit_std coverage:", ingr_df["unit_std"].notna().mean())
    print("[check] a_table_piece hits:", int((ingr_df["path_used"]=="a_table_piece").sum()))
    
    return recipes, ingr_df

# >>> [ADD-AUDIT] 审计统计与写盘 <<<
def write_step1_audit(ingr_df, out_dir, audit_dir=None):
    if audit_dir is None:
        audit_dir = os.path.join(out_dir, "audit", "step1")
    os.makedirs(audit_dir, exist_ok=True)

    # lane/path 分布
    ingr_df["__one"] = 1
    lane_counts = ingr_df.groupby("lane")["__one"].sum().rename("count").reset_index()
    path_counts = ingr_df.groupby("path_used")["__one"].sum().rename("count").reset_index()
    lane_counts.to_csv(os.path.join(audit_dir, "lane_counts.csv"), index=False)
    path_counts.to_csv(os.path.join(audit_dir, "path_counts.csv"), index=False)

    # 单位识别命中率
    unit_hit = pd.DataFrame({
        "recognized": [int((ingr_df["unit_std"].notna()).sum())],
        "total": [len(ingr_df)]
    })
    unit_hit["hit_rate"] = unit_hit["recognized"] / unit_hit["total"].clip(lower=1)
    unit_hit.to_csv(os.path.join(audit_dir, "unit_hit_rate.csv"), index=False)

    # 克重分布
    grams_ok = ingr_df["grams"].dropna().astype(float)
    if len(grams_ok) > 0:
        stats = pd.DataFrame({
            "metric":["p1","p5","p50","p95","p99","mean","std","min","max"],
            "value":[np.percentile(grams_ok,1), np.percentile(grams_ok,5), np.percentile(grams_ok,50),
                     np.percentile(grams_ok,95), np.percentile(grams_ok,99),
                     grams_ok.mean(), grams_ok.std(), grams_ok.min(), grams_ok.max()]
        })
        stats.to_csv(os.path.join(audit_dir, "grams_dist.csv"), index=False)

    # 异常样本
    def _has(flag):
        return ingr_df["audit_flags"].astype(str).str.contains(flag, regex=False, na=False)
    bad_mask = _has("grams_outlier") | _has("ambiguous_unit") | _has("non_positive") | _has("no_match")
    bad_cases = ingr_df[bad_mask].copy()
    if len(bad_cases) > 0:
        bad_cases.sample(min(1000, len(bad_cases))).to_parquet(os.path.join(audit_dir, "audit_bad_cases.parquet"))

    # 红线规则提示（简单打印）
    n = len(ingr_df)
    share = lambda cond: float((ingr_df[cond]["__one"].sum())) / max(1,n)
    fallback_share = share(ingr_df["lane"]=="fallback")
    ambiguous_share = share(_has("ambiguous_unit"))
    outlier_share = share(_has("grams_outlier"))
    print(f"[Audit] lane: weight={share(ingr_df['lane']=='weight'):.1%}, volume={share(ingr_df['lane']=='volume'):.1%}, piece={share(ingr_df['lane']=='piece'):.1%}, fallback={fallback_share:.1%}")
    print(f"[Audit] path(top5):\n{path_counts.sort_values('count', ascending=False).head(5)}")
    print(f"[Audit] bad_cases: grams_outlier={outlier_share:.1%}, ambiguous_unit={ambiguous_share:.1%}")

    # 清理辅助列
    del ingr_df["__one"]

def main():
    """
    主函数：执行数据准备和预处理流程
    
    执行流程：
        1. 解析命令行参数
        2. 加载A表（单位转换数据）
        3. 加载USDA和FNDDS营养数据库
        4. 加载Food.com食谱数据
        5. 应用列精简和保存处理后的数据
        6. 生成统计报告和审计信息
    
    输出文件：
        - food_processed.parquet: 处理后的食物信息
        - nutrient_processed.parquet: 营养素信息
        - food_nutrient_processed.parquet: 食物-营养素关系
        - recipes_processed.parquet: 处理后的食谱信息
        - ingredients_processed.parquet: 处理后的配料信息
        - inv_index.pkl: 倒排索引文件
        - config.json: 配置文件
        - 各种营养素目录CSV文件
    
    统计报告：
        - 基础统计信息
        - 单位识别审计报告
        - A表贡献统计
        - 分层映射统计
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="步骤1：数据准备和预处理")
    parser.add_argument("--usda_dir", required=True, help="USDA CSV 目录")
    parser.add_argument("--fdnn_dir", required=True, help="FDNN CSV 目录")
    parser.add_argument("--recipes", required=True, help="Food.com recipes.parquet 路径")
    parser.add_argument("--reviews", default=None, help="Food.com reviews.parquet 路径（可选）")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--filter_usda_types", default=None, help="可选：仅使用某些 data_type")
    parser.add_argument("--config", default=None, help="配置文件路径（可选）")
    
    # >>> [PATCH-CLI] 四车道/审计 开关 <<<
    parser.add_argument("--enable_four_lanes", action="store_true", default=True,
                        help="启用四车道决策器（重量直达/体积→密度/件数→每件/兜底）")
    parser.add_argument("--impute_units", action="store_true", default=True,
                        help="仅对失败残差样本进行单位推断与别名兜底")
    parser.add_argument("--audit_dir", type=str, default=None,
                        help="审计输出目录，如不设则默认写到 out_dir/audit/step1")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设置审计目录
    if args.audit_dir:
        audit_dir = args.audit_dir
    else:
        audit_dir = os.path.join(args.out_dir, "audit", "step1")
    os.makedirs(audit_dir, exist_ok=True)
    
    # 加载配置（如果提供）
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = Config(
            usda_dir=args.usda_dir,
            fdnn_dir=args.fdnn_dir,
            recipes=args.recipes,
            reviews=args.reviews,
            out_dir=args.out_dir,
            filter_usda_types=args.filter_usda_types
        )
    
    print(">> 步骤1：加载和预处理数据...")
    
    # 默认白名单（可被 --filter_usda_types 覆盖）
    if not getattr(config, "filter_usda_types", None):
        config.filter_usda_types = "Foundation Foods,SR Legacy,FNDDS"
    print(f"[CFG] Using USDA data_type whitelist: {config.filter_usda_types}")
    
    # 一次性验证（必须打印）
    print("\n🔍 一次性验证:")
    test_text = "1/2 cup milk; 2tbsp oil; 8oz chicken (boneless)"
    matches = re.findall(UNIT_REGEX.pattern, test_text)
    matches_pyarrow = re.findall(UNIT_REGEX_PYARROW.pattern, test_text)
    print(f"   re.findall(UNIT_REGEX, \"{test_text}\"): {matches}")
    print(f"   re.findall(UNIT_REGEX_PYARROW, \"{test_text}\"): {matches_pyarrow}")
    print(f"   当前 UNIT_REGEX 前 100 个字符: {str(UNIT_REGEX.pattern)[:100]}")
    print(f"   当前 UNIT_REGEX_PYARROW 前 100 个字符: {str(UNIT_REGEX_PYARROW.pattern)[:100]}")
    print(f"   动态扩展前模式数量: {len(UNIT_PATTERNS)}")
    print(f"   动态扩展前同义词数量: {len(UNIT_SYNONYMS_LOCAL)}")
    
    # 0) 加载 A 表
    print(">> 加载 A 表...")
    a_table_loaded = load_a_table()
    
    # 1) 加载 USDA 和 FNDDS
    print(">> Loading USDA and FDNN ...")
    print(f"   USDA目录: {config.usda_dir}")
    print(f"   FNDDS目录: {config.fdnn_dir}")
    print(f"   过滤类型: {config.filter_usda_types}")
    
    food, nutr, fn, inv_index, nutr_usda_raw, nutr_fdnn_raw = load_usda_and_fdnn(
        config.usda_dir, config.fdnn_dir, config.filter_usda_types
    )
    
    # 如果A表加载成功，进行USDA联表构建分层映射
    if a_table_loaded and A_TABLE is not None:
        print(">> 构建USDA联表分层映射...")
        grams_col = None
        for col in ["grams_per_unit", "g_per_unit", "grams", "gram_per_unit", "g"]:
            if col in A_TABLE.columns:
                grams_col = col
                break
        
        if grams_col:
            _build_usda_linked_mappings(food, A_TABLE, grams_col)
        else:
            print("   警告：未找到A表中的克重列，跳过USDA联表")
    
    # 调试信息：检查source和data_type分布
    print(f"   source 分布:")
    source_counts = food["source"].value_counts()
    for source, count in source_counts.items():
        print(f"     {source}: {count:,}")
    
    print(f"   data_type 分布:")
    data_type_counts = food["data_type"].value_counts()
    for dtype, count in data_type_counts.head(10).items():
        print(f"     {dtype}: {count:,}")
    
    # 保存原始数据条数用于统计
    # 正确口径（依赖 load_usda_and_fdnn 中给 source 赋值）
    food_usda_count = (food.get("source") == "USDA").sum()
    food_fdnn_count = (food.get("source") == "FNDDS").sum()
    
    print(f"   USDA数据条数: {food_usda_count:,}")
    print(f"   FNDDS数据条数: {food_fdnn_count:,}")
    print(f"   合并后总条数: {len(food):,}") 
    # 2) 加载 Food.com
    print(">> Loading Food.com ...")
    recipes, ingr_df = load_foodcom(config.recipes, config.reviews, 
                                   enable_four_lanes=args.enable_four_lanes, 
                                   impute_units=args.impute_units)
    
    # 3) 列精简和保存处理后的数据
    print(f">> 保存处理后的数据 (模式: {MODE})...")
    
    # 应用列精简逻辑
    print(">> 应用列精简逻辑...")
    
    with tqdm(total=6, desc="保存处理后的数据") as pbar:
        # food_processed.parquet
        food_keep_cols = KEEP_COLS["food_processed.parquet"][MODE]
        food_reduced = _apply_keep(food, food_keep_cols)
        print(f"   food_processed.parquet 保留列: {list(food_reduced.columns)}")
        food_reduced.to_parquet(os.path.join(config.out_dir, "food_processed.parquet"), 
                               index=False, engine="pyarrow", compression="zstd")
        pbar.update(1)
        
        # nutrient_processed.parquet
        nutr_keep_cols = KEEP_COLS["nutrient_processed.parquet"][MODE]
        nutr_reduced = _apply_keep(nutr, nutr_keep_cols)
        print(f"   nutrient_processed.parquet 保留列: {list(nutr_reduced.columns)}")
        nutr_reduced.to_parquet(os.path.join(config.out_dir, "nutrient_processed.parquet"), 
                               index=False, engine="pyarrow", compression="zstd")
        pbar.update(1)
        
        # food_nutrient_processed.parquet
        fn_keep_cols = KEEP_COLS["food_nutrient_processed.parquet"][MODE]
        fn_reduced = _apply_keep(fn, fn_keep_cols)
        print(f"   food_nutrient_processed.parquet 保留列: {list(fn_reduced.columns)}")
        fn_reduced.to_parquet(os.path.join(config.out_dir, "food_nutrient_processed.parquet"), 
                             index=False, engine="pyarrow", compression="zstd")
        pbar.update(1)
        
        # 保存倒排索引
        import pickle
        with open(os.path.join(config.out_dir, "inv_index.pkl"), 'wb') as f:
            pickle.dump(dict(inv_index), f)
        pbar.update(1)
        
        # recipes_processed.parquet
        recipes_keep_cols = KEEP_COLS["recipes_processed.parquet"][MODE]
        recipes_reduced = _apply_keep(recipes, recipes_keep_cols)
        print(f"   recipes_processed.parquet 保留列: {list(recipes_reduced.columns)}")
        recipes_reduced.to_parquet(os.path.join(config.out_dir, "recipes_processed.parquet"), 
                                  index=False, engine="pyarrow", compression="zstd")
        pbar.update(1)
        
        # —— 在写 ingredients_processed.parquet 之前调用：
        audit_out_dir = args.audit_dir if args.audit_dir else os.path.join(args.out_dir, "audit", "step1")
        write_step1_audit(ingr_df, args.out_dir, audit_out_dir)
        
        # ingredients_processed.parquet
        ingr_keep_cols = KEEP_COLS["ingredients_processed.parquet"][MODE]
        ingr_reduced = _apply_keep(ingr_df, ingr_keep_cols)
        
        # 优化字符串类型，减少体积
        for c in ["ingredient_norm", "unit_std"]:
            if c in ingr_reduced.columns:
                ingr_reduced[c] = ingr_reduced[c].astype("string[pyarrow]")
        
        print(f"   ingredients_processed.parquet 保留列: {list(ingr_reduced.columns)}")
        ingr_reduced.to_parquet(os.path.join(config.out_dir, "ingredients_processed.parquet"), 
                               index=False, engine="pyarrow", compression="zstd")
        pbar.update(1)
    
    # === 新增：单位命中与新估克重审计 ===
    try:
        df = ingr_df if 'ingr_df' in locals() else None
        if df is not None and len(df) > 0:
            tot = len(df)
            hits_w = int(df.get("lane_hit_weight", pd.Series([False]*tot)).sum())
            hits_p = int(df.get("lane_hit_pack", pd.Series([False]*tot)).sum())
            hits_pc = int(df.get("lane_hit_piece", pd.Series([False]*tot)).sum())
            hits_v = int(df.get("lane_hit_volume", pd.Series([False]*tot)).sum())
            qcol = df.get("quantity_g_est_new")
            filled = int(qcol.notna().sum()) if qcol is not None else 0
            residual = tot - filled
            print(f"✅ 三车道(改)命中: 重量={hits_w}, 包装×规格={hits_p}, 条件件数={hits_pc}, 体积标记={hits_v}")
            print(f"✅ 新估克重填充: {filled} / {tot} ({(filled/tot):.1%})")
            print(f"⚠️ 残差(未转克): {residual} ({(residual/tot):.1%})")

            sample_mask = (df.get("lane_hit_piece", False) | df.get("lane_hit_weight", False) |
                           df.get("lane_hit_volume", False) | df.get("lane_hit_pack", False))
            print("样本检查（前10行）:")
            cols = [c for c in ["ingredient_raw","quantity_g_est_new"] if c in df.columns]
            if cols:
                print(df.loc[sample_mask, cols].head(10).to_string(index=False))
    except Exception as e:
        warnings.warn(f"新估克重审计输出失败：{e}")

    # 导出营养素目录 - 处理NaN rank并稳定排序（使用fillna(1e9)）
    try:
        nutr_catalog = nutr[["id","name","unit_name","nutrient_nbr","rank"]].copy()
        nutr_catalog["rank"] = nutr_catalog["rank"].fillna(1e9)  # 填大值
        nutr_catalog = nutr_catalog.drop_duplicates().sort_values(
            ["name","unit_name","rank"], kind="stable"
        ).reset_index(drop=True)
        nutr_catalog.to_csv(os.path.join(config.out_dir, "nutrient_catalog_usda_fndds.csv"), 
                           index=False, encoding="utf-8")
    except Exception as e:
        warnings.warn(f"导出 nutrient_catalog_usda_fndds.csv 失败: {e}")
    
    try:
        usda_catalog = nutr_usda_raw[["id","name","unit_name","nutrient_nbr","rank"]].copy()
        usda_catalog["rank"] = usda_catalog["rank"].fillna(1e9)  # 填大值
        usda_catalog = usda_catalog.drop_duplicates().sort_values(
            ["name","unit_name","rank"], kind="stable"
        ).reset_index(drop=True)
        usda_catalog.to_csv(os.path.join(config.out_dir, "nutrient_catalog_usda.csv"), 
                           index=False, encoding="utf-8")
    except Exception as e:
        warnings.warn(f"导出 nutrient_catalog_usda.csv 失败: {e}")
    
    try:
        fdnn_catalog = nutr_fdnn_raw[["id","name","unit_name","nutrient_nbr","rank"]].copy()
        fdnn_catalog["rank"] = fdnn_catalog["rank"].fillna(1e9)  # 填大值
        fdnn_catalog = fdnn_catalog.drop_duplicates().sort_values(
            ["name","unit_name","rank"], kind="stable"
        ).reset_index(drop=True)
        fdnn_catalog.to_csv(os.path.join(config.out_dir, "nutrient_catalog_fdnn.csv"), 
                           index=False, encoding="utf-8")
    except Exception as e:
        warnings.warn(f"导出 nutrient_catalog_fdnn.csv 失败: {e}")
    
    # 保存配置
    save_config(config, os.path.join(config.out_dir, "config.json"))
    
    # 标记步骤完成
    mark_step_completed("step1", config.out_dir)
    
    # 程序结束时的验收和日志（必须打印）
    print("\n" + "="*60)
    print(">> 步骤1完成！验收报告")
    print("="*60)
    
    # 列精简模式信息
    print(f"\n🔧 列精简模式: {MODE}")
    print(f"   - food_processed.parquet: {len(food_reduced.columns)} 列")
    print(f"   - nutrient_processed.parquet: {len(nutr_reduced.columns)} 列")
    print(f"   - food_nutrient_processed.parquet: {len(fn_reduced.columns)} 列")
    print(f"   - recipes_processed.parquet: {len(recipes_reduced.columns)} 列")
    print(f"   - ingredients_processed.parquet: {len(ingr_reduced.columns)} 列")
    
    # 基础统计
    print(f"\n📊 基础统计:")
    print(f"   - food 条目数: {len(food_reduced):,}")
    print(f"   - recipes 条目数: {len(recipes_reduced):,}")
    print(f"   - ingredients 行数: {len(ingr_reduced):,}")
    print(f"   - 倒排索引 term 数量: {len(inv_index):,}")
    
    # food 的 nutrient_coverage 分布
    if "nutrient_coverage" in food_reduced.columns:
        coverage_stats = food_reduced["nutrient_coverage"].describe()
        print(f"\n📈 food 的 nutrient_coverage 分布:")
        print(f"   - min: {coverage_stats['min']:.0f}")
        print(f"   - median: {coverage_stats['50%']:.0f}")
        print(f"   - max: {coverage_stats['max']:.0f}")
    
    # data_type Top5
    if "data_type" in food_reduced.columns:
        data_type_counts = food_reduced["data_type"].value_counts().head(5)
        print(f"\n🏷️  data_type Top5:")
        for dtype, count in data_type_counts.items():
            print(f"   - {dtype}: {count:,}")
    
    # amount_robust 缺失率统计
    if "amount_robust" in fn_reduced.columns:
        amount_missing_rate = fn["amount"].isna().mean() * 100
        amount_robust_missing_rate = fn["amount_robust"].isna().mean() * 100
        print(f"\n🔍 food_nutrient_processed.parquet 缺失率对比:")
        print(f"   - amount 缺失率: {amount_missing_rate:.2f}%")
        print(f"   - amount_robust 缺失率: {amount_robust_missing_rate:.2f}%")
        print(f"   - 改善程度: {amount_missing_rate - amount_robust_missing_rate:.2f} 百分点")
    
    # 随机抽样 3 条 ingredients_processed.parquet 记录
    if len(ingr_reduced) > 0:
        print(f"\n🍽️  随机抽样 3 条 ingredients_processed.parquet 记录:")
        sample_cols = ["ingredient_raw", "qty_raw", "qty_parsed", "unit_raw", "unit_std", "grams", "servings", "unit_imputed_flag"]
        available_cols = [col for col in sample_cols if col in ingr_reduced.columns]
        if available_cols:
            sample_records = ingr_reduced[available_cols].sample(min(3, len(ingr_reduced)), random_state=42)
            for i, (_, record) in enumerate(sample_records.iterrows(), 1):
                print(f"\n   记录 {i}:")
                for col in available_cols:
                    value = record[col]
                    if pd.isna(value):
                        value = "NaN"
                    print(f"     {col}: {value}")
    
    # 新增审计报告
    print(f"\n🔍 单位识别审计报告:")
    
    if len(ingr_reduced) > 0:
        # 1. 文本含单位率（ctx中是否存在任意单位命中）
        # 重新构造 ctx 进行审计
        qty_norm = _norm_num(ingr_reduced["qty_raw"])
        ing_norm = _norm_num(ingr_reduced["ingredient_raw"])
        ctx_series = (qty_norm + " " + ing_norm).str.lower()
        ctx_series = (ctx_series
               .str.replace(r'(\d)([a-zA-Z])', r'\1 \2', regex=True)
               .str.replace(r'[-–—]', ' ', regex=True)
               .str.replace(r'[()\[\]]', ' ', regex=True)
               .str.replace(r'\s+', ' ', regex=True)
               .str.strip())
        
        has_unit_pattern = ctx_series.str.contains(UNIT_REGEX_PYARROW.pattern, regex=True, na=False)
        text_has_unit_rate = has_unit_pattern.mean() * 100
        print(f"   - 文本含单位率: {text_has_unit_rate:.2f}%")
        
        # 2. 真正parse失败率（文本含单位但unit_std为空）
        has_unit_but_failed = has_unit_pattern & ingr_reduced["unit_std"].isna()
        unit_parse_failure_rate = has_unit_but_failed.mean() * 100
        print(f"   - 真正parse失败率: {unit_parse_failure_rate:.2f}%")
        
        # 3. 打印当前 UNIT_REGEX_PYARROW 前 100 个字符
        print(f"   - 当前 UNIT_REGEX_PYARROW 前 100 个字符: {str(UNIT_REGEX_PYARROW.pattern)[:100]}")
        
        # 4. 打印 ctx 示例与 ctx.contains(UNIT_REGEX_PYARROW) 的 head(5)
        print(f"   - ctx 示例 (前5个):")
        for i, ctx_val in enumerate(ctx_series.head(5)):
            print(f"     {i+1}. {ctx_val}")
        print(f"   - ctx.contains(UNIT_REGEX_PYARROW) 结果 (前5个):")
        for i, (ctx_val, has_unit) in enumerate(zip(ctx_series.head(5), has_unit_pattern.head(5))):
            print(f"     {i+1}. {ctx_val} → {has_unit}")
        
        # 3. 数字-单位贴连占比（归一化前后对比）
        digit_unit_attached = ingr_reduced["ingredient_raw"].astype(str).str.contains(
            r'(?i)\d(?:oz|tsp|tbsp|ml|g|kg|lb)', 
            na=False
        ).mean() * 100
        print(f"   - 数字-单位贴连占比: {digit_unit_attached:.2f}%")
        
        # 4. 连字符/括号包裹的单位占比（归一化前后对比）
        hyphen_bracket_units = ingr_reduced["ingredient_raw"].astype(str).str.contains(
            r'(?i)[-–—\(\)\[\]]\s*(?:cup|tsp|tbsp|oz|ml|g|kg|lb|pound|ounce|gram|liter|pint|quart|gallon)\s*[-–—\(\)\[\]]', 
            na=False
        ).mean() * 100
        print(f"   - 连字符/括号包裹的单位占比: {hyphen_bracket_units:.2f}%")
        
        # 5. 先验回填统计
        if "unit_imputed_flag" in ingr_reduced.columns:
            imputed_count = (ingr_reduced["unit_imputed_flag"] == 1).sum()
            imputed_rate = imputed_count / len(ingr_reduced) * 100
            print(f"   - 先验回填记录数: {imputed_count:,} ({imputed_rate:.2f}%)")
        
        # 6. 动态扩展前后对比
        print(f"   - 动态扩展后单位模式数量: {len(UNIT_PATTERNS)}")
        print(f"   - 动态扩展后同义词映射数量: {len(UNIT_SYNONYMS_LOCAL)}")
        
        # 7. 断言：检查危险版本
        try:
            # 检查是否存在危险版本
            import inspect
            source = inspect.getsource(_normalize_unit_local)
            if '.replace("s", "")' in source:
                print("   ⚠️  警告：发现危险版本单位归一化代码！")
            else:
                print("   ✅ 安全版本单位归一化验证通过")
        except:
            print("   ℹ️  无法验证单位归一化版本")
    
    # 详细统计
    print(f"\n📋 详细统计:")
    print(f"   - USDA食物条目: {food_usda_count:,}")
    print(f"   - FNDDS食物条目: {food_fdnn_count:,}")
    print(f"   - 合并后食物条目: {len(food):,}")
    print(f"   - 营养素种类: {len(nutr):,}")
    print(f"   - 食物营养素关系: {len(fn):,}")
    print(f"   - 倒排索引token数: {len(inv_index):,}")
    if config.filter_usda_types:
        print(f"   - 过滤条件: {config.filter_usda_types}")
    
    # === A表贡献统计：unit_std 命中率与 grams 由A表计算的比例 ===
    try:
        # 重新加载刚写出的 ingredients_processed 以防上游筛列
        ingr_chk = pd.read_parquet(os.path.join(config.out_dir, "ingredients_processed.parquet"))
        total = len(ingr_chk)
        unit_cov = ingr_chk["unit_std"].notna().mean() * 100 if "unit_std" in ingr_chk.columns else float('nan')
        grams_cov = ingr_chk["grams"].notna().mean() * 100 if "grams" in ingr_chk.columns else float('nan')

        # 粗略估算：由 A表计算的 grams（即：qty_parsed 非空 & unit_std ∈ A表单位 且 grams 非空）
        # 注意：这里只是告知贡献比例，不会与旧逻辑严格可分。用于论文"引入A表后的改善"报告
        from math import isnan
        a_units = set(UNIT_GLOBAL.keys()) if UNIT_GLOBAL else set()
        if "unit_std" in ingr_chk.columns and "qty_parsed" in ingr_chk.columns and "grams" in ingr_chk.columns:
            via_A = ingr_chk["unit_std"].isin(a_units) & ingr_chk["qty_parsed"].notna() & ingr_chk["grams"].notna()
            via_A_ratio = via_A.mean() * 100
        else:
            via_A_ratio = float('nan')

        print(f"\n🧮 A表接入后的覆盖评估：")
        print(f"   - A表加载状态：{'✅ 已加载' if a_table_loaded else '❌ 未加载'}")
        if a_table_loaded:
            print(f"   - A表单位数量：{len(A_UNIT_MEDIAN)}")
        print(f"   - unit_std 非缺失率：{unit_cov:.2f}%")
        print(f"   - grams 非缺失率（总体）：{grams_cov:.2f}%")
        print(f"   - 由 A表计算的 grams 占比（改造后）：{via_A_ratio:.2f}%")
        
        # 改造前后对比（需要重新计算）
        if "unit_imputed_flag" in ingr_chk.columns:
            imputed_via_A = ingr_chk["unit_imputed_flag"] == 1
            imputed_via_A_ratio = imputed_via_A.mean() * 100
            print(f"   - 先验回填中A表命中占比：{imputed_via_A_ratio:.2f}%")
        
        # 分层映射统计报告
        if a_table_loaded:
            print(f"\n🎯 分层映射统计报告：")
            print(f"   - 体积单位密度映射：")
            print(f"     * density_by_fdc: {len(density_by_fdc)} 个fdc_id")
            print(f"     * density_by_cat: {len(density_by_cat)} 个category")
            print(f"     * density_by_token: {len(density_by_token)} 个token")
            print(f"   - 件数单位重量映射：")
            print(f"     * piece_weight_fdc: {len(piece_weight_fdc)} 个(fdc_id, unit)对")
            print(f"     * piece_weight_by_cat: {len(piece_weight_by_cat)} 个(unit, category)对")
            print(f"     * piece_weight_by_token: {len(piece_weight_by_token)} 个(unit, token)对")
            
            # 计算分层映射策略的占比
            if len(ingr_chk) > 0:
                volume_mask = ingr_chk["unit_std"].isin(VOLUME_UNITS)
                piece_mask = ingr_chk["unit_std"].isin(PIECE_UNITS)
                
                volume_count = volume_mask.sum()
                piece_count = piece_mask.sum()
                
                print(f"   - 体积单位记录数：{volume_count:,} ({volume_count/len(ingr_chk)*100:.1f}%)")
                print(f"   - 件数单位记录数：{piece_count:,} ({piece_count/len(ingr_chk)*100:.1f}%)")
                
                # 抽样误差对比（如果有足够数据）
                if len(ingr_chk) >= 1000:
                    sample_size = min(5000, len(ingr_chk))
                    sample_df = ingr_chk.sample(sample_size, random_state=42)
                    
                    # 计算分层映射vs全局中位数的误差
                    errors = []
                    for _, row in sample_df.iterrows():
                        if pd.notna(row.get("qty_parsed")) and pd.notna(row.get("unit_std")):
                            # 分层映射结果
                            layered_result = estimate_grams_enhanced(row)
                            # 全局中位数结果
                            global_result = row["qty_parsed"] * UNIT_GLOBAL.get(row["unit_std"], 0) if row["unit_std"] in UNIT_GLOBAL else 0
                            
                            if layered_result and global_result and layered_result > 0 and global_result > 0:
                                error = abs(layered_result - global_result) / global_result
                                errors.append(error)
                    
                    if errors:
                        mae = np.mean(errors) * 100
                        medae = np.median(errors) * 100
                        print(f"   - 分层映射vs全局中位数误差（抽样{sample_size}条）：")
                        print(f"     * 平均绝对误差(MAE)：{mae:.2f}%")
                        print(f"     * 中位数绝对误差(MedAE)：{medae:.2f}%")
    except Exception as e:
        warnings.warn(f'A表覆盖统计失败：{e}')
    
    # === 科学审计与路径贡献统计 ===
    try:
        print(f"\n🔬 科学审计与路径贡献统计:")
        
        # 重新加载ingredients数据以获取path_used列
        ingr_full = pd.read_parquet(os.path.join(config.out_dir, "ingredients_processed.parquet"))
        
        if "path_used" in ingr_full.columns:
            # 1. 路径贡献分布统计
            path_counts = ingr_full["path_used"].value_counts()
            path_ratios = ingr_full["path_used"].value_counts(normalize=True) * 100
            
            print(f"   📊 A表驱动路径分布:")
            for path, count in path_counts.items():
                ratio = path_ratios[path]
                print(f"     - {path}: {count:,} 条 ({ratio:.2f}%)")
            
            # 2. 各路径的grams覆盖率
            print(f"   📈 各路径grams覆盖率:")
            for path in path_counts.index:
                path_data = ingr_full[ingr_full["path_used"] == path]
                if len(path_data) > 0:
                    grams_coverage = path_data["grams"].notna().mean() * 100
                    print(f"     - {path}: {grams_coverage:.2f}%")
            
            # 3. 抽样展示路径使用情况
            print(f"   🎯 路径使用抽样展示（10条）:")
            sample_data = ingr_full[["ingredient_raw", "qty_parsed", "unit_std", "grams", "path_used"]].sample(
                min(10, len(ingr_full)), random_state=42
            )
            
            for i, (_, row) in enumerate(sample_data.iterrows(), 1):
                ingredient = row["ingredient_raw"] if pd.notna(row["ingredient_raw"]) else "NaN"
                qty = row["qty_parsed"] if pd.notna(row["qty_parsed"]) else "NaN"
                unit = row["unit_std"] if pd.notna(row["unit_std"]) else "NaN"
                grams = row["grams"] if pd.notna(row["grams"]) else "NaN"
                path = row["path_used"] if pd.notna(row["path_used"]) else "NaN"
                
                print(f"     {i:2d}. {ingredient} | {qty} {unit} → {grams}g | {path}")
            
            # 4. 路径效果对比（A表路径 vs 全局路径）
            a_table_paths = ["fdc_density", "fdc_piece", "token_density", "token_piece", "cat_density", "cat_piece"]
            global_paths = ["global_volume", "global_piece", "global_other"]
            
            a_table_count = ingr_full["path_used"].isin(a_table_paths).sum()
            global_count = ingr_full["path_used"].isin(global_paths).sum()
            total_count = len(ingr_full)
            
            print(f"   🎯 路径效果对比:")
            print(f"     - A表驱动路径: {a_table_count:,} 条 ({a_table_count/total_count*100:.2f}%)")
            print(f"     - 全局兜底路径: {global_count:,} 条 ({global_count/total_count*100:.2f}%)")
            
        else:
            print("   ⚠️  未找到path_used列，跳过路径贡献统计")
            
    except Exception as e:
        warnings.warn(f'科学审计统计失败：{e}')
    
    print(f"\n✅ 输出目录: {config.out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
