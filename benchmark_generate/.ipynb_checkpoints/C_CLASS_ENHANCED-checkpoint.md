# C-Class Dataset - Enhanced Version (8 Nutrition Violation Types)

## 🎯 增强版本特性

### 营养违规类型（8种）

| # | 违规类型 | 注入策略 | 修正策略 | 比例 |
|---|---------|---------|---------|------|
| 1 | **sodium_mg** | 增加salt 150-250% | 减少salt到满足限制 | 15% |
| 2 | **protein_amdr** | 减少protein食材50-70% | 增加protein食材40-60% | 8% |
| 3 | **fat_amdr** | 增加oil/butter 100-200% | 减少oil/butter 40-60% | 8% |
| 4 | **carb_amdr** | 减少carb食材40-60% | 增加carb食材50-80% | 8% |
| 5 | **energy_kcal** | 增加oil/carb 40-70% | 减少oil/carb到目标 | 8% |
| 6 | **fiber_g** | 减少vegetable 50-70% | 增加vegetable 50-100% | 8% |
| 7 | **saturated_fat_g** | 增加butter/cheese 150-250% | 减少butter/cheese 50-70% | 3% |
| 8 | **sugars_g** | 增加honey/syrup 100-200% | 减少honey/syrup 50-70% | 2% |

**总计营养违规**: 60%

### 偏好违规类型（2种）

| # | 违规类型 | 注入策略 | 修正策略 | 比例 |
|---|---------|---------|---------|------|
| 1 | **disliked_added** | 添加disliked食材50-100g | 删除该食材 | 15% |
| 2 | **liked_removed** | 删除1个liked食材 | 重新添加100g | 10% |

**总计偏好违规**: 25%

### 双重违规

**组合**: 营养违规 + 偏好违规
**比例**: 15%

---

## 📊 完整违规类型说明

### 1. sodium_mg - 钠超标 ⚠️

**违规注入**:
```python
# 找到salt食材，增加150-250%
salt_qty = original_qty * random.uniform(2.5, 3.5)

# 示例: 1 tsp salt (6g) → 2.5 tsp salt (15g)
```

**违规条件**:
```python
sodium_total = nutrition['sodium_mg'] * 4
sodium_max = targets['sodium_mg_max']  # 例如2000mg

if sodium_total > sodium_max:
    violation = {
        'field': 'sodium_mg',
        'actual': 5000,
        'limit': 2000,
        'severity': 'critical'  # >1.5x limit
    }
```

**修正策略**:
```python
reduction_needed = (5000 - 2000) / 5000 = 0.6
reduction_factor = 1 - 0.6 * 1.2 = 0.28

# 减少salt到原来的28%
new_salt_qty = 15g * 0.28 = 4.2g → "3/4 tsp salt"
```

---

### 2. protein_amdr - 蛋白质AMDR比例过低 ⚠️

**AMDR (Acceptable Macronutrient Distribution Range)**:
- 蛋白质: 15-25% of total energy
- 脂肪: 20-35% of total energy
- 碳水化合物: 45-65% of total energy

**违规注入**:
```python
# 减少protein食材50-70%
protein_qty = original_qty * random.uniform(0.5, 0.7)

# 示例: 500g chicken → 300g chicken
```

**违规条件**:
```python
protein_kcal = protein_g * 4
protein_pct = (protein_kcal / energy_kcal) * 100
target_protein_pct = targets['amdr']['protein']['target_pct']  # 例如20%

if protein_pct < target_protein_pct * 0.65:  # <13%
    violation = {
        'field': 'protein_amdr',
        'actual_pct': 12.5,
        'target_pct': 20.0,
        'severity': 'major'
    }
```

**修正策略**:
```python
# 增加protein食材40-60%
increase_factor = random.uniform(1.4, 1.6)
new_protein_qty = 300g * 1.5 = 450g
```

---

### 3. fat_amdr - 脂肪AMDR比例过高 ⚠️

**违规注入**:
```python
# 增加oil/butter 100-200%
fat_qty = original_qty * random.uniform(2.0, 3.0)

# 示例: 2 Tbsp oil (27g) → 5 Tbsp oil (68g)
```

**违规条件**:
```python
fat_kcal = fat_g * 9
fat_pct = (fat_kcal / energy_kcal) * 100
target_fat_pct = targets['amdr']['fat']['target_pct']  # 例如30%

if fat_pct > target_fat_pct * 1.4:  # >42%
    violation = {
        'field': 'fat_amdr',
        'actual_pct': 45.0,
        'target_pct': 30.0
    }
```

**修正策略**:
```python
# 减少oil/butter 40-60%
reduction_factor = random.uniform(0.4, 0.6)
new_fat_qty = 68g * 0.5 = 34g → "2.5 Tbsp oil"
```

---

### 4. carb_amdr - 碳水AMDR比例过低 ⚠️

**违规注入**:
```python
# 减少carb食材40-60%
carb_qty = original_qty * random.uniform(0.4, 0.6)

# 示例: 2 cups rice (370g) → 0.9 cups rice (167g)
```

**违规条件**:
```python
carb_pct = (carb_g * 4 / energy_kcal) * 100
target_carb_pct = targets['amdr']['carb']['target_pct']  # 例如50%

if carb_pct < target_carb_pct * 0.6:  # <30%
    violation = {
        'field': 'carb_amdr',
        'actual_pct': 28.0,
        'target_pct': 50.0
    }
```

**修正策略**:
```python
# 增加carb食材50-80%
increase_factor = random.uniform(1.5, 1.8)
new_carb_qty = 167g * 1.6 = 267g → "1.4 cups rice"
```

---

### 5. energy_kcal - 能量超标 ⚠️

**违规注入**:
```python
# 增加oil或carb 40-70%
energy_source_qty = original_qty * random.uniform(1.4, 1.7)

# 示例: 2 cups rice → 3.2 cups rice
```

**违规条件**:
```python
actual_energy = nutrition['energy_kcal']
target_energy = targets['energy_kcal_target'] / 4  # per serving

if actual_energy > target_energy * 1.25:  # >25%超标
    violation = {
        'field': 'energy_kcal',
        'actual': 625,
        'target': 500
    }
```

**修正策略**:
```python
reduction_needed = (625 - 500) / 625 = 0.2
reduction_factor = 1 - 0.2 * 1.2 = 0.76

# 减少到原来的76%
```

---

### 6. fiber_g - 纤维不足 ⚠️

**违规注入**:
```python
# 减少vegetable 50-70%
veggie_qty = original_qty * random.uniform(0.3, 0.5)

# 示例: 200g broccoli → 80g broccoli
```

**违规条件**:
```python
fiber_total = nutrition['fiber_g'] * 4
fiber_min = targets['fiber_g_min']  # 例如28g

if fiber_total < fiber_min * 0.65:  # <18g
    violation = {
        'field': 'fiber_g',
        'actual': 16.8,
        'minimum': 28.0
    }
```

**修正策略**:
```python
# 增加vegetable 50-100%
increase_factor = random.uniform(1.5, 2.0)
new_veggie_qty = 80g * 1.8 = 144g

# 或者添加高纤维蔬菜
add_ingredient('broccoli', 100g)
```

---

### 7. saturated_fat_g - 饱和脂肪超标 ⚠️

**健康标准**: 饱和脂肪应<10% of total energy

**违规注入**:
```python
# 增加butter/cheese 150-250%
sat_fat_source_qty = original_qty * random.uniform(2.5, 3.5)

# 示例: 1 Tbsp butter (15g) → 3.5 Tbsp butter (53g)
```

**违规条件**:
```python
sat_fat_kcal = saturated_fat_g * 9
sat_fat_pct = (sat_fat_kcal / energy_kcal) * 100

if sat_fat_pct > 12:  # >12% (健康标准<10%)
    violation = {
        'field': 'saturated_fat_g',
        'actual_pct': 14.5,
        'limit_pct': 10.0
    }
```

**修正策略**:
```python
# 减少butter/cheese 50-70%
reduction_factor = random.uniform(0.3, 0.5)
new_qty = 53g * 0.4 = 21g → "1.5 Tbsp butter"
```

---

### 8. sugars_g - 糖分超标 ⚠️

**健康标准**: 添加糖应<10% of total energy

**违规注入**:
```python
# 增加honey/syrup 100-200%
sugar_source_qty = original_qty * random.uniform(2.0, 3.0)

# 示例: 1 Tbsp honey (20g) → 2.5 Tbsp honey (50g)
```

**违规条件**:
```python
sugars_kcal = sugars_g * 4
sugars_pct = (sugars_kcal / energy_kcal) * 100
limit_pct = targets['sugars']['pct_max']  # 例如10%

if sugars_pct > limit_pct * 1.5:  # >15%
    violation = {
        'field': 'sugars_g',
        'actual_pct': 16.8,
        'limit_pct': 10.0
    }
```

**修正策略**:
```python
# 减少honey/syrup 50-70%
reduction_factor = random.uniform(0.3, 0.5)
new_qty = 50g * 0.4 = 20g → "1 Tbsp honey"
```

---

## 📈 违规分布设计

### 单一营养违规（60%）

```python
violation_types = [
    'sodium',           # 15%
    'protein_low',      # 8%
    'fat_high',         # 8%
    'carb_low',         # 8%
    'energy_high',      # 8%
    'fiber_low',        # 8%
    'saturated_fat_high',  # 3%
    'sugars_high'       # 2%
]

# 随机选择一种
violation_type = random.choice(violation_types)
```

### 偏好违规（25%）

```python
# 60%概率添加disliked食材
if random.random() < 0.6:
    add_disliked_ingredient()

# 40%概率删除liked食材
else:
    remove_liked_ingredient()
```

### 双重违规（15%）

```python
# 常见组合
combinations = [
    ('sodium', 'preference'),
    ('energy_high', 'preference'),
    ('fat_high', 'preference')
]

# 先注入营养违规，再注入偏好违规
```

---

## 🔍 质量验证

### 验证指标

```python
# 1. 违规明显性
assert (actual - limit) / limit > 0.2  # 至少偏离20%

# 2. 修正有效性
corrected_value = apply_correction(violated_value)
assert corrected_value <= limit  # 修正后满足约束

# 3. 最小修正原则
assert num_corrections <= 2  # 最多2个修正动作
```

### 成功率预期

- **单一营养违规**: 成功率95%
- **偏好违规**: 成功率90%（受营养覆盖率影响）
- **双重违规**: 成功率85%

---

## 🚀 使用方法

### 1. 测试生成（3样本）

```bash
python test_c_class_v2.py
```

### 2. 完整数据集生成

```bash
python generate_c_class_full_v2_ENHANCED.py
```

**预计时间**: 10k样本 ≈ 1.5-2小时
**预期成功率**: 90-95%

---

## 📦 文件清单

| 文件 | 说明 |
|------|------|
| `ingredient_parser.py` | Ingredient解析/重组模块 |
| `generate_c_class_full_v2_ENHANCED.py` | **增强版生成器（8种营养类型）** |
| `test_c_class_v2.py` | 测试脚本 |
| `C_CLASS_ENHANCED.md` | 本设计文档 |

---

## 💡 营养类型扩展说明

### 为什么是8种？

1. **sodium_mg** - 最常见的营养约束（高血压）
2. **protein_amdr** - AMDR核心指标之一
3. **fat_amdr** - AMDR核心指标之一
4. **carb_amdr** - AMDR核心指标之一
5. **energy_kcal** - 体重管理的关键
6. **fiber_g** - 消化健康和血糖控制
7. **saturated_fat_g** - 心血管健康（WHO推荐<10%）
8. **sugars_g** - 代谢健康（WHO推荐<10%）

### 覆盖的健康维度

✅ **宏量营养素平衡**: protein, fat, carb (AMDR)
✅ **微量营养素**: fiber
✅ **有害成分限制**: sodium, saturated_fat, sugars
✅ **能量管理**: energy_kcal

---

## 📊 示例数据

### 违规示例1: 钠超标

```json
{
  "violations": [{
    "field": "sodium_mg",
    "actual": 5200,
    "limit": 2000,
    "severity": "critical"
  }],
  "corrections": [{
    "action": "reduce_quantity",
    "ingredient": "salt",
    "from": "2.5 tsp",
    "to": "3/4 tsp"
  }]
}
```

### 违规示例2: 脂肪AMDR过高

```json
{
  "violations": [{
    "field": "fat_amdr",
    "actual_pct": 48.5,
    "target_pct": 30.0,
    "severity": "major"
  }],
  "corrections": [{
    "action": "reduce_quantity",
    "ingredient": "olive oil",
    "from": "5 Tbsp",
    "to": "2 Tbsp"
  }]
}
```

### 违规示例3: 双重违规

```json
{
  "violations": [
    {
      "field": "energy_kcal",
      "actual": 680,
      "target": 500
    },
    {
      "type": "preference_violation",
      "subtype": "disliked_ingredient_added",
      "ingredient": "bacon"
    }
  ],
  "corrections": [
    {
      "action": "reduce_quantity",
      "ingredient": "rice",
      "from": "3 cups",
      "to": "2 cups"
    },
    {
      "action": "remove_ingredient",
      "ingredient": "bacon"
    }
  ]
}
```
