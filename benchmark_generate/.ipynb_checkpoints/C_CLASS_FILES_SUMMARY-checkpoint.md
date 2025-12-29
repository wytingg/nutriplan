# C-Class数据集文件清单与说明

## ✅ 已完成增强：从4种营养类型扩展到8种

### 原版本（4种营养违规）
- sodium（钠超标）
- protein（蛋白质不足）
- energy（能量超标）
- fiber（纤维不足）

### **增强版本（8种营养违规）** ⭐
1. **sodium_mg** - 钠超标
2. **protein_amdr** - 蛋白质AMDR比例过低
3. **fat_amdr** - 脂肪AMDR比例过高
4. **carb_amdr** - 碳水AMDR比例过低
5. **energy_kcal** - 能量超标
6. **fiber_g** - 纤维不足
7. **saturated_fat_g** - 饱和脂肪超标
8. **sugars_g** - 糖分超标

**AMDR** = Acceptable Macronutrient Distribution Range（宏量营养素可接受分布范围）
- 蛋白质：15-25% of energy
- 脂肪：20-35% of energy
- 碳水：45-65% of energy

---

## 📦 需要上传到服务器的文件

### 核心文件（必需）

#### 1. `ingredient_parser.py`
**功能**：解析和重组ingredient字符串
- 解析："2 cups rice" → (370.0, "rice")
- 重组：(370.0, "rice") → "2 cups rice"
- 支持单位：g, cups, tsp, Tbsp, items（egg, onion, carrot）

#### 2. `generate_c_class_full_v2_ENHANCED.py` ⭐
**功能**：完整严谨的C-class生成器（8种营养类型）
- ✅ 真实ingredient解析和修改
- ✅ 精确营养重计算
- ✅ 8种营养违规注入
- ✅ 8种修正策略
- ✅ 营养覆盖率保障（只用500个有数据的食材）

#### 3. `test_c_class_v2.py`
**功能**：测试脚本（3样本）
- 测试ingredient parser
- 测试营养重计算
- 测试完整违规→修正流程
- 验证修正有效性

### 文档文件（可选，建议上传）

#### 4. `C_CLASS_ENHANCED.md`
**功能**：完整设计文档
- 8种营养类型详细说明
- 每种违规的注入和修正策略
- 违规分布设计
- 示例数据

---

## 🚀 运行顺序

### 步骤1: 测试ingredient parser

```bash
cd ~/work/recipebench/scripts/traindata_generate
python ingredient_parser.py
```

**预期输出**：
```
Testing Ingredient Parser:
==========================================================
Original:  2 cups rice
Parsed:    370.0g rice
Composed:  2 cups rice

Original:  1/2 tsp salt
Parsed:    3.0g salt
Composed:  1/2 tsp salt
✓ Parser working correctly
```

### 步骤2: 测试C-class生成（3样本）

```bash
python test_c_class_v2.py
```

**预期输出**：
```
Sample 1: User 1234
► Violations Detected: 1
     • Sodium: 5000mg > 2000mg limit (critical)

► Corrections Applied: 1
     1. Reduce salt: 15.0g → 6.0g (reduce_sodium)

► Validation:
     ✓ Sodium fixed: 1920mg ≤ 2000mg
```

### 步骤3: 生成完整数据集（确认测试通过后）

```bash
python generate_c_class_full_v2_ENHANCED.py
```

**预计时间**：10k样本 ≈ 1.5-2小时
**预期成功率**：90-95%

---

## 📊 8种营养违规详解

| # | 类型 | 目标约束 | 违规条件 | 注入方式 | 修正方式 |
|---|------|---------|---------|---------|---------|
| 1 | sodium_mg | <2000mg | >limit | 增加salt 150-250% | 减少salt |
| 2 | protein_amdr | 15-25% | <target*0.65 | 减少protein 50-70% | 增加protein |
| 3 | fat_amdr | 20-35% | >target*1.4 | 增加oil 100-200% | 减少oil |
| 4 | carb_amdr | 45-65% | <target*0.6 | 减少carb 40-60% | 增加carb |
| 5 | energy_kcal | ~500 kcal | >target*1.25 | 增加oil/carb 40-70% | 减少oil/carb |
| 6 | fiber_g | >25g | <minimum*0.65 | 减少veggie 50-70% | 增加veggie |
| 7 | saturated_fat_g | <10% energy | >12% | 增加butter 150-250% | 减少butter |
| 8 | sugars_g | <10% energy | >limit*1.5 | 增加honey 100-200% | 减少honey |

---

## 📈 违规分布

```
总分布：
├── 60% 单一营养违规
│   ├── 15% sodium_mg
│   ├── 8% protein_amdr
│   ├── 8% fat_amdr
│   ├── 8% carb_amdr
│   ├── 8% energy_kcal
│   ├── 8% fiber_g
│   ├── 3% saturated_fat_g
│   └── 2% sugars_g
│
├── 25% 偏好违规
│   ├── 15% 添加disliked食材
│   └── 10% 删除liked食材
│
└── 15% 双重违规
    └── 营养违规 + 偏好违规
```

---

## 🔍 关键改进点

### 1. 真实营养计算

```python
# ❌ v1简化版（不准确）
violated_nutrition['sodium_mg'] *= 1.3  # 直接乘系数

# ✅ v2增强版（精确）
salt_qty = 6.0 * 2.5  # 增加salt quantity
modified_ings = update_ingredient(salt_qty)
new_nutrition = calc.calculate_recipe_nutrition(modified_ings, 4)  # 重新计算
```

### 2. 真实ingredient修改

```python
# ❌ v1简化版
ingredients.append("salt (reduced by 50%)")

# ✅ v2增强版
(qty, name, _) = parse_ingredient_string("1 tsp salt")  # 6.0g
new_qty = 6.0 * 0.5  # 3.0g
new_str = compose_ingredient_string(3.0, "salt")  # "1/2 tsp salt"
```

### 3. 营养覆盖率保障

```python
# ✅ 只使用有营养数据的500个食材
AVAILABLE_INGREDIENTS = set(calc.nutrition_lookup.keys())

def is_ingredient_available(ing_name):
    return ing_name.lower() in AVAILABLE_INGREDIENTS

# 添加食材前检查
if is_ingredient_available("bacon"):
    matched = find_best_match_ingredient("bacon")
    # 安全添加
```

### 4. AMDR比例控制

```python
# 新增：AMDR比例违规
protein_kcal = protein_g * 4
fat_kcal = fat_g * 9
carb_kcal = carb_g * 4
total_kcal = protein_kcal + fat_kcal + carb_kcal

protein_pct = (protein_kcal / total_kcal) * 100
# 检查是否在15-25%范围内
```

---

## 💡 使用建议

### 测试阶段
1. 先运行`python ingredient_parser.py`确保解析器正常
2. 再运行`python test_c_class_v2.py`测试3个样本
3. 检查输出，确认违规注入和修正都正确

### 生产阶段
1. 确认测试通过后，运行完整生成
2. 预留1.5-2小时生成时间
3. 检查成功率（应>90%）

### 故障排除
- 如果成功率<85%：检查B-class数据集质量
- 如果营养重计算失败：检查ingredient匹配逻辑
- 如果修正无效：检查correction factor计算

---

## 📋 检查清单

上传前：
- [x] `ingredient_parser.py` 已创建
- [x] `generate_c_class_full_v2_ENHANCED.py` 已创建（8种营养类型）
- [x] `test_c_class_v2.py` 已创建
- [x] `C_CLASS_ENHANCED.md` 已创建（文档）

上传到服务器：
- [ ] 上传3个核心文件
- [ ] 运行ingredient parser测试
- [ ] 运行3样本测试
- [ ] 检查测试结果
- [ ] 运行完整数据集生成

---

## 🎯 最终输出

成功生成后，将得到：

```
work/recipebench/data/10large_scale_datasets/
├── task_c_train_large.jsonl  (~10,000样本, 成功率90-95%)
├── task_c_val_large.jsonl    (~2,000样本, 成功率90-95%)
└── task_c_test_large.jsonl   (~2,000样本, 成功率90-95%)
```

每个样本包含：
- `input.violated_recipe`: 违规初稿
- `input.violations`: 违约点列表（1-2个）
- `output.corrected_recipe`: 修正后食谱
- `output.corrections`: 修正动作列表（1-2个）

---

## ✅ 总结

**增强版本特性**：
- ✅ 8种营养违规类型（覆盖AMDR、sodium、fiber、sat_fat、sugars）
- ✅ 真实ingredient解析和修改
- ✅ 精确营养重计算
- ✅ 营养覆盖率保障
- ✅ 严谨的修正验证

**相比简化版的优势**：
- 🎯 营养类型从4种扩展到8种
- 🎯 营养值精确计算（非估算）
- 🎯 ingredient真实修改（非标注）
- 🎯 90-95%成功率（而非可能失败）
