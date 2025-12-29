# C-Class Dataset v2 - Complete & Rigorous Implementation

## 🎯 核心改进（v2 vs v1）

### v1版本的问题
❌ **简化处理**：直接在nutrition字典上乘系数
❌ **假营养计算**：修正后的营养值是估算的
❌ **字符串标注**：append "(reduced by 50%)"到ingredient字符串
❌ **忽略覆盖率**：可能选到没有营养数据的食材（失败率高）

### v2版本的严谨实现
✅ **真实修改**：解析ingredient字符串→修改quantity→重新计算营养
✅ **精确计算**：调用RecipeNutritionCalculator重新计算所有营养值
✅ **智能重组**：用household_units重新组装ingredient字符串
✅ **覆盖率保障**：只使用有营养数据的食材池（避免15%缺失问题）

---

## 📐 架构设计

### 1. Ingredient Parser Module (`ingredient_parser.py`)

**功能**：解析和重组ingredient字符串

```python
# 解析
parse_ingredient_string("2 cups rice")
→ (370.0, "rice")

parse_ingredient_string("1/2 tsp salt")
→ (3.0, "salt")

# 重组
compose_ingredient_string(370.0, "rice")
→ "2 cups rice"

compose_ingredient_string(3.0, "salt")
→ "1/2 tsp salt"
```

**单位转换表**：
- Spoons: tsp (salt: 6g, pepper: 2.3g), Tbsp (salt: 18g, oil: 13.5g)
- Cups: rice (185g/cup), flour (125g/cup), oats (80g/cup)
- Items: egg (50g), onion (150g), carrot (61g)

### 2. Violation Injection Module

**严谨流程**：

```python
# 1. 解析ingredients
parsed = parse_recipe_ingredients(["500g chicken", "2 cups rice", "1 tsp salt"])
# → [(500.0, "chicken", "500g chicken"), (370.0, "rice", "2 cups rice"), (6.0, "salt", "1 tsp salt")]

# 2. 修改quantity（例如：钠超标 → 增加salt）
modified = copy.deepcopy(parsed)
salt_qty = 6.0 * 2.5  # 增加150%
modified[2] = (salt_qty, "salt", compose_ingredient_string(salt_qty, "salt"))

# 3. 重新计算营养
ingredient_strings = ["500g chicken", "370g rice", "15g salt"]
new_nutrition = calc.calculate_recipe_nutrition(ingredient_strings, servings=4)

# 4. 验证违规
actual_sodium = new_nutrition['per_serving']['sodium_mg'] * 4
if actual_sodium > sodium_limit:
    violations.append({...})
```

**违规类型实现**：

| 违规类型 | 注入策略 | 营养重计算 |
|---------|---------|-----------|
| **sodium** | 增加salt quantity 150-200% | ✅ recalculate |
| **protein** | 减少protein食材30-40% | ✅ recalculate |
| **energy** | 增加oil/carb 30-50% | ✅ recalculate |
| **fiber** | 减少vegetable 40-50% | ✅ recalculate |
| **preference** | 添加disliked食材（只用有营养数据的） | ✅ recalculate |

### 3. Correction Strategy Module

**严谨修正流程**：

```python
# 1. 生成修正方案
correction = {
    'action': 'reduce_quantity',
    'ingredient_index': 2,
    'ingredient_name': 'salt',
    'original_quantity': 15.0,
    'new_quantity': 6.0,  # 精确计算减少量
    'reduction_factor': 0.4,
    'reason': 'reduce_sodium_to_meet_limit'
}

# 2. 应用修正
corrected_parsed[2] = (6.0, "salt", compose_ingredient_string(6.0, "salt"))

# 3. 重新计算营养
corrected_nutrition = recalculate_nutrition(corrected_parsed, servings=4)

# 4. 验证修正效果
corrected_sodium = corrected_nutrition['per_serving']['sodium_mg'] * 4
assert corrected_sodium <= sodium_limit  # 确保满足约束
```

**修正动作类型**：

```python
{
    'reduce_quantity': {
        'ingredient_index': 2,
        'original_quantity': 15.0,
        'new_quantity': 6.0,
        'reduction_factor': 0.4
    },
    'increase_quantity': {
        'ingredient_index': 0,
        'original_quantity': 500.0,
        'new_quantity': 650.0,
        'increase_factor': 1.3
    },
    'remove_ingredient': {
        'ingredient_index': 3,
        'ingredient_name': 'bacon'
    },
    'add_ingredient': {
        'ingredient_name': 'broccoli',
        'quantity': 100.0
    }
}
```

---

## 🔍 营养覆盖率问题的解决

### 问题
- 营养数据库：500个食材（覆盖85%）
- 潜在风险：修正时添加的食材可能不在数据库中

### 解决方案

```python
# 1. 维护可用食材池
AVAILABLE_INGREDIENTS = set(calc.nutrition_lookup.keys())

# 2. 检查食材是否可用
def is_ingredient_available(ing_name):
    ing_lower = ing_name.lower()
    # Exact match
    if ing_lower in AVAILABLE_INGREDIENTS:
        return True
    # Fuzzy match
    for avail_ing in AVAILABLE_INGREDIENTS:
        if avail_ing in ing_lower or ing_lower in avail_ing:
            return True
    return False

# 3. 只使用可用食材
def inject_preference_violation(...):
    disliked_names = [ing['name'].lower() for ing in disliked_ingredients]

    # Filter to only use ingredients with nutrition data
    available_disliked = []
    for d_name in disliked_names:
        if is_ingredient_available(d_name):
            available_disliked.append(d_name)

    # Only add if we found available disliked ingredients
    if available_disliked:
        bad_ing = random.choice(available_disliked)
        matched_ing = find_best_match_ingredient(bad_ing)
        ...
```

**保障机制**：
- ✅ 添加食材前检查is_ingredient_available
- ✅ 使用find_best_match_ingredient进行模糊匹配
- ✅ 失败时返回None（样本生成失败，而非产生错误数据）

---

## 📊 数据格式示例

### Input（违规初稿）

```json
{
  "violated_recipe": {
    "title": "Chicken Breast with Rice",
    "ingredients": [
      "500g chicken breast",
      "2 cups rice",
      "2 1/2 tsps salt"  // 违规：钠超标
    ],
    "nutrition_per_serv": {
      "sodium_mg": 1250  // 超标！
    }
  },
  "violations": [
    {
      "type": "nutrition_violation",
      "field": "sodium_mg",
      "actual": 5000,        // 总量（4份）
      "limit": 2000,         // 用户限制
      "severity": "critical",
      "per_serving_actual": 1250,
      "per_serving_limit": 500
    }
  ]
}
```

### Output（修正后食谱）

```json
{
  "corrected_recipe": {
    "title": "Chicken Breast with Rice",
    "ingredients": [
      "500g chicken breast",
      "2 cups rice",
      "1 tsp salt"  // 修正：减少40%
    ],
    "nutrition_per_serv": {
      "sodium_mg": 480  // 达标✓
    }
  },
  "corrections": [
    {
      "action": "reduce_quantity",
      "ingredient_index": 2,
      "ingredient_name": "salt",
      "original_quantity": 15.0,  // 2.5 tsp = 15g
      "new_quantity": 6.0,        // 1 tsp = 6g
      "reduction_factor": 0.4,
      "reason": "reduce_sodium_to_meet_limit"
    }
  ]
}
```

---

## 🧪 质量保障

### 验证点

1. **解析准确性**
   ```python
   # Test: "2 cups rice" → 370g rice → "2 cups rice"
   qty, name = parse_ingredient_string("2 cups rice")
   assert qty == 370.0
   assert name == "rice"
   reconstructed = compose_ingredient_string(qty, name)
   assert "2 cups" in reconstructed
   ```

2. **营养计算一致性**
   ```python
   # Original nutrition
   orig_nutrition = b_recipe['output']['nutrition_per_serv']

   # Parse & recalculate
   parsed = parse_recipe_ingredients(b_recipe['output']['ingredients'])
   recalc_nutrition = recalculate_nutrition(parsed)

   # Should match within 5%
   energy_diff = abs(orig_nutrition['energy_kcal'] - recalc_nutrition['per_serving']['energy_kcal'])
   assert energy_diff / orig_nutrition['energy_kcal'] < 0.05
   ```

3. **修正有效性**
   ```python
   # After correction
   corrected_sodium = corrected_nutrition['per_serving']['sodium_mg'] * 4
   sodium_limit = targets['sodium_mg_max']

   # Must satisfy constraint
   assert corrected_sodium <= sodium_limit
   ```

### 失败处理

```python
def generate_c_class_sample(b_class_recipe, seed=0):
    # Step 1: Generate violated recipe
    violation_result = generate_violated_recipe(b_class_recipe)
    if violation_result is None:
        return None  # 注入失败，跳过此样本

    # Step 2: Generate corrections
    corrections = generate_corrections(...)
    if not corrections:
        return None  # 无法生成修正，跳过

    # Step 3: Apply corrections
    corrected_nutrition = recalculate_nutrition(...)
    if corrected_nutrition is None:
        return None  # 营养计算失败，跳过

    return c_class_sample
```

**预期失败率**：5-10%（由于营养覆盖率85%和复杂修正场景）

---

## 📈 违规类型分布

| 类型 | 比例 | 子类型 | 实现方式 |
|------|------|--------|---------|
| **营养违规** | 50% | sodium (25%) | 增加salt 150-200% + 重算 |
|  |  | protein (10%) | 减少protein 30-40% + 重算 |
|  |  | energy (10%) | 增加oil/carb 30-50% + 重算 |
|  |  | fiber (5%) | 减少veggie 40-50% + 重算 |
| **偏好违规** | 30% | disliked_added (18%) | 添加disliked食材（仅可用） |
|  |  | liked_removed (12%) | 删除liked食材 |
| **双重违规** | 20% | sodium+preference (10%) | 组合注入 |
|  |  | energy+preference (10%) | 组合注入 |

---

## 🚀 使用方法

### 1. 测试ingredient parser

```bash
python ingredient_parser.py
```

### 2. 测试C-class生成（3个样本）

```bash
python test_c_class_v2.py
```

**预期输出**：
```
Sample 1: User 1234
► Original B-class Recipe:
   Nutrition: Energy=450 kcal, Sodium=500mg

► Violated Recipe:
   Nutrition: Energy=450 kcal, Sodium=1250mg

► Violations Detected: 1
     • Sodium: 5000mg > 2000mg limit (critical)

► Corrections Applied: 1
     1. Reduce salt: 15.0g → 6.0g (reduce_sodium_to_meet_limit)

► Corrected Recipe:
   Nutrition: Energy=450 kcal, Sodium=480mg

► Validation:
     ✓ Sodium fixed: 1920mg ≤ 2000mg
```

### 3. 生成完整数据集（10k/2k/2k）

```bash
python generate_c_class_full_v2.py
```

---

## 📋 文件清单

| 文件 | 功能 | 代码行数 |
|------|------|---------|
| `ingredient_parser.py` | 解析和重组ingredient字符串 | ~250 |
| `generate_c_class_full_v2.py` | 完整严谨的C-class生成器 | ~650 |
| `test_c_class_v2.py` | 测试脚本（3样本） | ~180 |
| `C_CLASS_V2_DESIGN.md` | 设计文档 | - |

---

## ⚡ 性能优化建议

### 当前性能
- **生成速度**: ~0.5秒/样本（包含营养重计算）
- **成功率**: ~90-95%（5-10%失败由于营养覆盖率）
- **预计时间**: 10k样本 ≈ 1.5小时

### 优化方向
1. **缓存营养计算**：相同ingredient组合缓存结果
2. **批量处理**：一次计算多个样本的营养（如果API支持）
3. **预过滤食材池**：提前过滤出有营养数据的disliked/liked食材

---

## ✅ 检查清单

上传到服务器前：
- [ ] `ingredient_parser.py` - 解析器模块
- [ ] `generate_c_class_full_v2.py` - 完整生成器
- [ ] `test_c_class_v2.py` - 测试脚本

运行顺序：
1. [ ] 测试ingredient parser: `python ingredient_parser.py`
2. [ ] 测试C-class生成: `python test_c_class_v2.py`
3. [ ] 生成完整数据集: `python generate_c_class_full_v2.py`

预期结果：
- [ ] 3个测试样本全部成功
- [ ] 营养计算误差<5%
- [ ] 修正后满足约束
- [ ] 完整数据集成功率>90%
