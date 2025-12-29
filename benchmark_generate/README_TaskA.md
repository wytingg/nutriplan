# Task A: Discriminative Ranking - 数据集构建

## 📋 任务定义

**Task A (Discriminative Ranking)** - 判别式食谱排序任务

训练LLM学习：
- 从知识图谱候选集中**评分和排序**食谱
- 建立对**recipe suitability**的鲁棒理解
- 输出**结构化排序结果** + 详细评分理由

---

## 🎯 核心特性

### 1. 10个指令模板（覆盖多场景）

| # | 类型 | 示例 |
|---|------|------|
| 1 | **健康状况导向** | "I am a 45-year-old male with diabetes..." |
| 2 | **营养目标导向** | "Based on my daily nutritional requirements (Energy: 2200 kcal, Protein: 75g)..." |
| 3 | **食材偏好导向** | "I enjoy chicken, broccoli but dislike mushroom..." |
| 4 | **综合健康管理** | "As a diabetes patient aged 45..." |
| 5 | **特定营养素优化** | "I need recipes high in protein to meet my RNI of 75g..." |
| 6 | **限制性营养素控制** | "Due to hypertension, I must limit my sodium intake to 1500mg..." |
| 7 | **年龄性别特异性** | "As a 45-year-old male, please recommend age-appropriate recipes..." |
| 8 | **能量平衡** | "I need meals that provide approximately 33% of my daily energy requirement..." |
| 9 | **宏量营养素平衡** | "Please rank recipes that provide a balanced ratio of protein, carbs, and fat..." |
| 10 | **多维度综合评分** | "Considering my complete profile (demographics, health, preferences)..." |

### 2. 5维度打分系统

| 维度 | 权重 | 说明 |
|------|------|------|
| **nutrition_match** | 35% | 营养RNI匹配度（8个营养素） |
| **preference_match** | 25% | 食材偏好匹配度 |
| **cooccurrence** | 15% | 食材共现分数（从KG规则） |
| **complementarity** | 15% | 营养互补分数（从KG规则） |
| **balance** | 10% | 营养平衡分数（标签多样性） |

### 3. 可解释的推理生成

每个推荐都包含`reasoning`字段，例如：
```
"Excellent nutritional alignment with your RNI targets; contains your preferred
ingredients (chicken, broccoli); suitable for diabetes management with controlled
carbohydrate content; high ingredient synergy and nutritional complementarity"
```

---

## 📊 输出数据格式

```json
{
  "user_id": 1533,
  "instruction": "Based on my daily nutritional requirements (Energy: 2200 kcal, Protein: 75g, Fiber: 30g), please rank recipes that best meet these targets.",
  "instruction_type": "nutrition_target",
  "user_profile": {
    "gender": "male",
    "age": 45,
    "physiological_state": "diabetes",
    "nutrition_rni": {
      "energy_kcal": 2200.0,
      "protein_g": 75.0,
      "carbohydrate_g": 275.0,
      "fat_g": 61.0,
      "fiber_g": 30.0,
      "added_sugar_g": 55.0,
      "saturated_fat_g": 24.0,
      "trans_fat_g": 2.4,
      "sodium_mg": 1500.0,
      "potassium_mg": 3600.0,
      "calcium_mg": 800.0,
      "iron_mg": 12.0,
      "vitamin_c_mg": 100.0,
      "vitamin_d_ug": 15.0,
      "folate_ug": 400.0
    },
    "liked_ingredients_count": 5,
    "disliked_ingredients_count": 2
  },
  "ranked_recipes": [
    {
      "rank": 1,
      "recipe_id": "12345",
      "recipe_name": "Grilled Chicken with Quinoa and Vegetables",
      "overall_score": 0.873,
      "score_breakdown": {
        "nutrition_match": 0.920,
        "preference_match": 0.850,
        "cooccurrence": 0.780,
        "complementarity": 0.880,
        "balance": 0.910
      },
      "reasoning": "Excellent nutritional alignment with your RNI targets; contains your preferred ingredients (chicken, broccoli); suitable for diabetes management with controlled carbohydrate content; high ingredient synergy and nutritional complementarity",
      "ingredients": [
        "chicken breast",
        "quinoa",
        "broccoli",
        "olive oil",
        "garlic"
      ],
      "nutrition_per_serving": {
        "energy_kcal": 420.0,
        "protein_g": 35.0,
        "carbohydrate_g": 42.0,
        "fat_g": 12.0,
        "fiber_g": 8.0,
        "added_sugar_g": 2.0,
        "saturated_fat_g": 2.5,
        "sodium_mg": 280.0
      }
    },
    {
      "rank": 2,
      ...
    },
    {
      "rank": 3,
      ...
    }
  ]
}
```

---

## 🚀 使用方法

### 1. 修改文件路径

在脚本中修改这些路径（Line 371-374）：

```python
builder = TaskADatasetBuilder(
    kg_path="your/path/nutriplan_kg_rni_v2.graphml",
    recipe_basic_path="your/path/recipes(3column).csv",
    recipe_nutrition_path="your/path/recipe_nutrition_foodcom.csv",
    user_profile_path="your/path/updated_user_profile_15nutrients.jsonl"
)
```

### 2. 运行脚本

```bash
python build_task_a_dataset_rni.py
```

### 3. 输出文件

```
work/recipebench/data/10large_scale_datasets/
├── task_a_train_discriminative.jsonl  (10,000 样本)
├── task_a_val_discriminative.jsonl    (2,000 样本)
└── task_a_test_discriminative.jsonl   (2,000 样本)
```

---

## 📈 数据集统计

### 规模
- **训练集**: 10,000 用户 × 1样本 = 10,000 样本
- **验证集**: 2,000 用户 × 1样本 = 2,000 样本
- **测试集**: 2,000 用户 × 1样本 = 2,000 样本
- **每样本**: Top-3排序食谱（每个包含5维评分+推理）

### 指令分布
10种指令类型均匀分布（每种约10%）

### 质量指标
- 平均Top-1分数: 0.75-0.85（高质量匹配）
- 评分拆解透明度: 100%（所有样本都有5维评分）
- 推理可解释性: 100%（所有推荐都有文字理由）

---

## 🎓 训练目标

使用此数据集训练LLM学习：

### 1. **判别能力（Discriminative Ability）**
- 评估食谱与用户画像的适配性
- 区分高分食谱和低分食谱
- 理解多维度评分标准

### 2. **排序能力（Ranking Ability）**
- 在候选集中进行全局排序
- 平衡多个评分维度
- 处理trade-offs（如偏好vs营养）

### 3. **可解释性（Explainability）**
- 生成评分理由（reasoning）
- 解释推荐决策
- 提供透明的评分拆解

### 4. **泛化能力（Generalization）**
- 适应10种不同指令场景
- 处理不同人群（健康/疾病）
- 灵活应对多样化需求

---

## 🔧 关键技术点

### 1. RNI匹配算法
```python
# 正向营养素：接近单餐推荐量（RNI的1/3）
target_ratio = 0.33
diff = abs(actual_ratio - target_ratio)

# 限制性营养素：单餐不超过RNI的30%
if actual_ratio <= 0.30:
    score = 1.0
```

### 2. KG规则集成
- 从KG加载12,619条共现规则
- 从KG加载45,928条互补规则
- 从KG加载1,055个食材营养标签

### 3. 推理生成逻辑
- 基于评分拆解动态生成
- 考虑健康状况（糖尿病/高血压等）
- 整合食材偏好信息

---

## 🆚 与旧版本对比

| 特性 | 旧版本 | Task A (新版本) |
|------|-------|----------------|
| 数据格式 | label (0/1) | Top-3排序 + 评分 ✅ |
| 评分透明度 | 无拆解 | 5维拆解 ✅ |
| 可解释性 | 无 | reasoning字段 ✅ |
| 指令多样性 | 无 | 10种场景 ✅ |
| 营养素数量 | 8个 | 15个 ✅ |
| 用户属性 | 1个 | 4个（+age, gender, state）✅ |
| 任务类型 | 二分类 | 排序+评分 ✅ |

---

## 📝 样本示例

### 训练样本输入（Instruction）
```
I am a 45-year-old male with diabetes. Please recommend and rank recipes
suitable for my health condition, prioritizing nutritional safety and
disease management.
```

### 训练样本输出（Expected Response）
```
Based on your profile as a 45-year-old male with diabetes, here are my
top 3 recommended recipes ranked by suitability:

**Rank 1: Grilled Chicken with Quinoa and Vegetables** (Score: 0.873)
- Nutrition Match: 0.920 (Excellent alignment with your RNI targets)
- Preference Match: 0.850 (Contains your preferred ingredients)
- Ingredient Synergy: 0.780 (Good ingredient combinations)
- Nutritional Complementarity: 0.880 (High nutritional synergy)
- Balance: 0.910 (Well-balanced nutritional profile)

*Reasoning*: Excellent nutritional alignment with your RNI targets;
contains your preferred ingredients (chicken, broccoli); suitable for
diabetes management with controlled carbohydrate content; high ingredient
synergy and nutritional complementarity.

[Nutrition per serving: 420 kcal, 35g protein, 42g carbs, 12g fat, 8g fiber,
2g added sugar, 280mg sodium]

**Rank 2: ...**
**Rank 3: ...**
```

---

## 🎯 应用场景

1. **食谱推荐系统**：为用户推荐Top-N食谱
2. **营养咨询助手**：提供个性化饮食建议
3. **慢性病管理**：为患者筛选合适食谱
4. **健康教育**：解释推荐理由，提升营养素养

---

## ⚠️ 注意事项

1. **候选集采样**：每个用户从全量食谱中随机采样1500个候选（平衡质量和效率）
2. **Top-3选择**：只保留排名前3的食谱（高质量训练信号）
3. **评分归一化**：所有分数归一化到0-1区间
4. **推理生成**：基于规则生成，非真实LLM输出（作为训练目标）
5. **KG依赖**：需要完整的KG文件（包含共现/互补规则）

---

## 🎉 总结

Task A数据集专注于**判别式排序任务**，帮助LLM学习：
- ✅ 评估食谱适配性
- ✅ 进行全局排序
- ✅ 生成可解释的推荐
- ✅ 适应多样化指令

适合作为NutriPlan系统的**第一阶段训练任务**，为后续的生成式任务（Task B）和反思式编辑任务（Task C）打下基础。
