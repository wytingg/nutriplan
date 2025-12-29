# -*- coding: utf-8 -*-
"""
common_utils.py — 公共工具模块
包含所有步骤共享的函数、常量和配置
"""

import os, re, json, ast, math, gc
from collections import defaultdict, Counter
from functools import lru_cache
import numpy as np
import pandas as pd
from tqdm import tqdm
from ftfy import fix_text
from unidecode import unidecode
from rapidfuzz import fuzz
import warnings

# ------------------------ 可选：Faiss 近邻 ------------------------
try:
    import faiss  # faiss-cpu
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ------------------------ 嵌入模型 ------------------------
from sentence_transformers import SentenceTransformer

# ------------------------ 解析&清洗 ------------------------
UNIT_RE = re.compile(r"\b(cups?|cup|tbsp|tablespoons?|tbsps?|tsp|teaspoons?|tsps?|grams?|gram|g|kg|kgs|ml|l|liters?|liter|oz|ounce|ounces|pounds?|lb|lbs|pinch|clove|cloves|slice|slices|stick|sticks|package|packages|can|cans|bag|bags|servings?|serving|pieces?|piece|bunches?|bunch|heads?|head|leaves?|leaf|sprigs?|sprig|stalks?|stalk|ears?|ear|jars?|jar|containers?|container|bars?|bar)\b", re.I)
NUM_RE  = re.compile(r"(\d+\s+\d+/\d+|\d+/\d+|\d+(\.\d+)?)")
PAREN_RE= re.compile(r"\([^)]*\)")
NON_ASCII = re.compile(r"[^a-z0-9\s]+")

STOPWORDS = {
    "fresh","organic","large","small","medium","boneless","skinless","boned","packed",
    "finely","coarsely","light","dark","unsalted","salted","low","reduced","fat-free",
    "low-fat","low-sodium","no-salt","ripe","extra","virgin","raw","cooked","dried",
    "ground","minced","chopped","sliced","grated","shredded","peeled","seeded","thinly",
    "thick","softened","melted","warm","hot","cold","room","temperature"
}

SYNONYM_MAP = {
    "bell pepper": "capsicum",
    "green onion": "scallion",
    "spring onion": "scallion",
    "cilantro": "coriander leaves",
    "powdered sugar": "confectioners sugar",
    "corn starch": "cornstarch",
    "all purpose flour": "flour",
    "ap flour": "flour",
}

# 份量近似规则（以"克"为目标单位；尽量只覆盖"争议小"的常见物）
ITEM_GRAMS = {
    "egg": 50.0,
    "garlic clove": 3.0,
    "clove garlic": 3.0,
    "onion": 110.0,              # 中等大小 1 个
    "tomato": 120.0,             # 中等大小 1 个
    "lemon": 100.0,
    "lime": 70.0,
    "carrot": 60.0,
    "potato": 170.0,
    "banana": 120.0,

    # 片、条等
    "slice bacon": 8.0,
    "slice bread": 25.0,
    "slice cheese": 20.0,

    # stick
    "stick butter": 113.0,       # 1 条黄油（US）≈ 1/2 cup ≈ 8 tbsp ≈ 113 g
}

# 杯/勺 → 克 的近似密度（仅覆盖常见"误差较小"的食材；优先匹配规范化字符串中的关键词）
DENSITY_GRAMS = {
    "water": {"cup": 240.0, "tbsp": 15.0, "tsp": 5.0},
    "milk": {"cup": 240.0, "tbsp": 15.0, "tsp": 5.0},
    "olive oil": {"tbsp": 13.5, "tsp": 4.5, "cup": 216.0},  # 1 cup ≈ 16 tbsp
    "vegetable oil": {"tbsp": 13.5, "tsp": 4.5, "cup": 216.0},
    "soy sauce": {"tbsp": 16.0, "tsp": 5.3, "cup": 256.0},

    # 粉类
    "flour": {"cup": 120.0, "tbsp": 7.5, "tsp": 2.5},       # AP flour
    "sugar": {"cup": 200.0, "tbsp": 12.5, "tsp": 4.2},      # granulated
    "brown sugar": {"cup": 220.0, "tbsp": 13.5, "tsp": 4.5},
    "powdered sugar": {"cup": 120.0, "tbsp": 7.5, "tsp": 2.5},
    "cornstarch": {"tbsp": 8.0, "tsp": 2.7, "cup": 128.0},

    # 颗粒类
    "rice": {"cup": 180.0},                                  # uncooked
    "quinoa": {"cup": 170.0},                                # uncooked

    # 切碎蔬菜（近似）
    "onion": {"cup": 160.0},
    "tomato": {"cup": 180.0},
    "cheese": {"cup": 113.0},                                # shredded
    "butter": {"tbsp": 14.0, "tsp": 4.7, "cup": 227.0},      # 1 cup ≈ 227g
}

WEIGHT_UNITS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kgs": 1000.0,
    "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
    "lb": 453.592, "lbs": 453.592, "pound": 453.592, "pounds": 453.592,
}
VOL_UNITS = {"ml": 1.0, "l": 1000.0, "liter": 1000.0, "liters": 1000.0}

# 与 build_household.py 保持一致的单位标准化（子集，够用即可）
UNIT_ALIAS = {
    # volume
    "cup": "cup", "cups": "cup",
    "tablespoon": "tbsp", "table spoon": "tbsp", "tablespoons": "tbsp", "tbsp": "tbsp",
    "teaspoon": "tsp", "tea spoon": "tsp", "teaspoons": "tsp", "tsp": "tsp",
    "liter": "l", "litre": "l", "l": "l",
    "milliliter": "ml", "millilitre": "ml", "ml": "ml",
    # volume special
    "fluid ounce": "fl_oz", "fluid ounces": "fl_oz", "fl oz": "fl_oz", "fl. oz": "fl_oz", "fl_oz": "fl_oz",
    # mass
    "ounce": "oz", "ounces": "oz", "oz": "oz",
    "pound": "lb", "pounds": "lb", "lb": "lb", "lbs": "lb",
    # household / piece-like
    "clove": "clove", "cloves": "clove",
    "slice": "slice", "slices": "slice",
    "stick": "stick", "sticks": "stick",
    "package": "package", "packages": "package",
    "can": "can", "cans": "can",
    "bag": "bag", "bags": "bag",
}

def _clean_text_unit(s: str):
    s = str(s).strip().lower()
    s = "".join(ch if ch.isalnum() or ch in (" ", "_") else " " for ch in s)
    s = " ".join(s.split())
    s = s.replace("fluid  ounce", "fluid ounce").replace("fl  oz", "fl oz").replace("fl oz ", "fl oz")
    return s

def normalize_unit_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    s = _clean_text_unit(name)
    if s in UNIT_ALIAS:
        return UNIT_ALIAS[s]
    return s.replace(" ", "_")

# ------------------------ 解析与标准化工具 ------------------------
def parse_vec(s):
    """
    将 recipes 中的配料/数量字段解析为 list[str]。
    支持：list/tuple/ndarray、R 风格 c("a","b")、JSON/Python 列表字面量、逗号分隔、缺失值。
    """
    import numpy as np
    if isinstance(s, (list, tuple, np.ndarray)):
        out = []
        for x in s:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            out.append(str(x).strip())
        return out
    if s is None:
        return []
    try:
        if pd.isna(s):
            return []
    except Exception:
        pass
    s = str(s).strip()
    if not s:
        return []
    try:
        if s.startswith("c("):
            return re.findall(r'"([^"]+)"', s)
        if s[0] in "[(" and s[-1] in ")]":
            try:
                arr = json.loads(s)
            except Exception:
                arr = ast.literal_eval(s)
            if isinstance(arr, (list, tuple)):
                return [str(x).strip() for x in arr]
        return [t.strip() for t in s.split(",") if t.strip()]
    except Exception:
        return [s]

def normalize_text(s):
    s = fix_text(str(s))
    s = unidecode(s).lower()
    s = PAREN_RE.sub(" ", s)
    s = re.sub(r"[-_/]", " ", s)
    s = UNIT_RE.sub(" ", s)
    s = re.sub(NUM_RE, " ", s)
    s = NON_ASCII.sub(" ", s)
    toks = [w for w in re.split(r"\s+", s) if w]
    toks = [w for w in toks if w not in STOPWORDS]
    s = " ".join(toks).strip()
    for k, v in SYNONYM_MAP.items():
        s = s.replace(k, v)
    return s

def tokenize(s):
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]

@lru_cache(maxsize=10000)
def parse_qty(raw):
    """抽取数量与单位（cup/tbsp/tsp/g/kg/ml/l/oz/lb/...）"""
    text = str(raw).lower()
    qty = None
    for m in re.finditer(NUM_RE, text):
        token = m.group(0)
        if " " in token and "/" in token:
            a, b = token.split(" ", 1)
            val = float(a) + eval(b)
        elif "/" in token:
            val = float(eval(token))
        else:
            val = float(token)
        qty = max(qty or 0.0, val)
    unit = None
    unit_candidates = list(WEIGHT_UNITS.keys()) + list(VOL_UNITS.keys()) + [
        "cup","cups","tbsp","tbsps","tablespoon","tablespoons",
        "tsp","tsps","teaspoon","teaspoons","stick","sticks",
        "clove","cloves","slice","slices","serving","servings",
        "piece","pieces","bunch","bunches","head","heads",
        "leaf","leaves","sprig","sprigs","stalk","stalks",
        "ear","ears","jar","jars","container","containers",
        "bar","bars","package","packages","can","cans","bag","bags",
        "fl oz","fl. oz","fluid ounce","fluid ounces"
    ]
    for u in unit_candidates:
        if re.search(rf"\b{u}\b", text):
            unit = u
            break
    return qty, unit

@lru_cache(maxsize=10000)
def estimate_grams(raw_str, norm_str):
    qty, unit = parse_qty(raw_str)
    if unit in WEIGHT_UNITS:
        return (qty or 1.0) * WEIGHT_UNITS[unit]
    key_ings = list(DENSITY_GRAMS.keys())
    hit = None
    for k in key_ings:
        if k in norm_str:
            hit = k
            break
    def vol_to_g(u):
        if unit in ("tbsp","tablespoon","tbsps"):  u = "tbsp"
        if unit in ("tsp","teaspoon","tsps"):      u = "tsp"
        if unit in ("cup","cups"):                   u = "cup"
        return u
    if unit in ("ml","l","liter","liters"):
        ml = (qty or 1.0) * VOL_UNITS[unit]
        return ml
    if unit in ("tbsp","tablespoon","tbsps","tsp","teaspoon","tsps","cup","cups"):
        if hit and hit in DENSITY_GRAMS:
            ug = vol_to_g(unit)
            if ug in DENSITY_GRAMS[hit]:
                return (qty or 1.0) * DENSITY_GRAMS[hit][ug]
        base = {"cup":240.0, "tbsp":15.0, "tsp":5.0}
        ug = vol_to_g(unit)
        if ug in base:
            return (qty or 1.0) * base[ug]
    for key, g in ITEM_GRAMS.items():
        if key in norm_str:
            return (qty or 1.0) * g
    return None

# ------------------------ 嵌入 & 近邻 ------------------------
def embed_texts(model, texts, batch_size=512, normalize=True):
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encoding"):
        arr = model.encode(texts[i:i+batch_size], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=normalize)
        vecs.append(arr)
    return np.vstack(vecs)

def build_faiss_index(mat):
    d = mat.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(mat.astype(np.float32))
    return idx

def topk_by_cosine(query_vecs, base_vecs, k):
    base = base_vecs / (np.linalg.norm(base_vecs, axis=1, keepdims=True) + 1e-9)
    q = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9)
    sims = q @ base.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    return idx, np.take_along_axis(sims, idx, axis=1)

# ------------------------ 配置管理 ------------------------
class Config:
    def __init__(self, **kwargs):
        # 默认配置
        self.usda_dir = kwargs.get('usda_dir', '')
        self.fdnn_dir = kwargs.get('fdnn_dir', '')
        self.recipes = kwargs.get('recipes', '')
        self.reviews = kwargs.get('reviews', None)
        self.out_dir = kwargs.get('out_dir', '')
        self.filter_usda_types = kwargs.get('filter_usda_types', None)
        self.embed_model = kwargs.get('embed_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embed_bs = kwargs.get('embed_bs', 512)
        self.topk_embed = kwargs.get('topk_embed', 120)
        self.min_fuzz = kwargs.get('min_fuzz', 80)
        self.threads = kwargs.get('threads', 8)
        self.max_candidates_union = kwargs.get('max_candidates_union', 300)
        self.default_grams = kwargs.get('default_grams', 30.0)
        self.write_wide = kwargs.get('write_wide', False)
        self.wide_max_nutrients = kwargs.get('wide_max_nutrients', 512)
        self.agg_batch_size = kwargs.get('agg_batch_size', 200000)
        self.household_weights = kwargs.get('household_weights', None)
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

def save_config(config, filepath):
    """保存配置到JSON文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

def load_config(filepath):
    """从JSON文件加载配置"""
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return Config.from_dict(config_dict)

def check_step_completion(step_name, out_dir):
    """检查步骤是否已完成"""
    completion_file = os.path.join(out_dir, f"{step_name}_completed.flag")
    return os.path.exists(completion_file)

def mark_step_completed(step_name, out_dir):
    """标记步骤为已完成"""
    completion_file = os.path.join(out_dir, f"{step_name}_completed.flag")
    with open(completion_file, 'w') as f:
        f.write(f"Step {step_name} completed at {pd.Timestamp.now()}")
