# 更新 RQ2 配置说明

## 在运行 RQ2 实验前必须完成的配置

### 步骤 1：查看 Table X 确定最佳模型

在完成阶段 II 的所有训练后，运行：

```bash
cat results/table_x.txt
```

查找 **SNCR** 列中数值最高的模型。例如：

```
Model                                      SNCR              UPM               K-Faith           AVC
TinyLlama/TinyLlama-1.1B-Chat-v1.0        0.650±0.012       0.580±0.015       ...
microsoft/Phi-3-mini-4k-instruct          0.720±0.008       0.690±0.010       ...
Qwen/Qwen2-7B                             0.785±0.006       0.740±0.008       ...  ⭐ 最佳
mistralai/Mistral-7B-v0.3                 0.770±0.010       0.730±0.012       ...
google/gemma-2-9b                         0.760±0.009       0.720±0.011       ...
```

在这个例子中，`Qwen/Qwen2-7B` 的 SNCR 最高 (0.785)，所以它是最佳模型。

---

### 步骤 2：更新 run_rq2_experiments.sh 配置

编辑文件：

```bash
nano scripts/run_rq2_experiments.sh
```

或者

```bash
vim scripts/run_rq2_experiments.sh
```

需要修改的行：

#### 修改点 1：数据目录（第 6 行）

**原始内容：**
```bash
DATA_DIR="D:/Downloads"
```

**修改为（根据你的实际数据路径）：**
```bash
DATA_DIR="data"
```

#### 修改点 2：最佳基础模型（第 11 行）

**原始内容：**
```bash
BEST_BASE_LLM="meta-llama/Llama-3-8B"  # Update based on Table X results
```

**修改为（根据 Table X 的结果）：**
```bash
BEST_BASE_LLM="Qwen/Qwen2-7B"  # 示例：如果 Qwen2-7B 是最佳模型
```

#### 修改点 3（可选）：知识图谱路径（第 8 行）

**原始内容：**
```bash
KG_PATH="work/recipebench/kg/nutriplan_kg4.graphml"
```

**如果 KG 文件在其他位置，修改为：**
```bash
KG_PATH="path/to/your/kg/file.graphml"
```

---

### 步骤 3：验证配置

使用以下命令快速检查配置是否正确：

```bash
# 检查数据目录
grep "^DATA_DIR=" scripts/run_rq2_experiments.sh

# 检查最佳模型
grep "^BEST_BASE_LLM=" scripts/run_rq2_experiments.sh

# 检查 KG 路径
grep "^KG_PATH=" scripts/run_rq2_experiments.sh
```

---

### 步骤 4：验证最佳模型路径存在

确保对应的实验结果存在：

```bash
# 替换 <BEST_MODEL> 为你选择的模型名（用下划线替换斜杠）
# 例如：Qwen_Qwen2-7B

ls -la experiments/rq1_<BEST_MODEL>_seed_42/best_model/

# 示例：
ls -la experiments/rq1_Qwen_Qwen2-7B_seed_42/best_model/
```

应该看到模型文件：
```
config.json
model.safetensors (或 pytorch_model.bin)
tokenizer_config.json
...
```

---

## 完整修改示例

假设 Table X 显示 `Qwen/Qwen2-7B` 是最佳模型，完整的修改后的配置：

```bash
#!/bin/bash
# Batch script for RQ2 (Stage III)
# Compares NutriPlan against all baselines

# Configuration
DATA_DIR="data"  # ✏️ 已修改
EXPERIMENTS_BASE_DIR="experiments"
KG_PATH="work/recipebench/kg/nutriplan_kg4.graphml"

# Best model from RQ1 (update after Stage II)
BEST_BASE_LLM="Qwen/Qwen2-7B"  # ✏️ 已修改，基于 Table X
BEST_BASE_LLM_CLEAN=$(echo "$BEST_BASE_LLM" | tr '/' '_')
BEST_SEED=42  # Use first seed for comparisons

# NutriPlan model path
NUTRIPLAN_MODEL_PATH="${EXPERIMENTS_BASE_DIR}/rq1_${BEST_BASE_LLM_CLEAN}_seed_${BEST_SEED}/best_model"
# ... (其余保持不变)
```

---

## 自动化修改脚本（可选）

如果你想用命令行自动修改配置：

```bash
# 备份原文件
cp scripts/run_rq2_experiments.sh scripts/run_rq2_experiments.sh.backup

# 修改数据目录
sed -i 's|^DATA_DIR="D:/Downloads"|DATA_DIR="data"|' scripts/run_rq2_experiments.sh

# 修改最佳模型（示例：改为 Qwen/Qwen2-7B）
sed -i 's|^BEST_BASE_LLM="meta-llama/Llama-3-8B"|BEST_BASE_LLM="Qwen/Qwen2-7B"|' scripts/run_rq2_experiments.sh

# 验证修改
grep -E "^(DATA_DIR|BEST_BASE_LLM)=" scripts/run_rq2_experiments.sh
```

**注意：** 在使用 `sed -i` 前务必备份文件！

---

## 修改完成后的检查清单

- [ ] `DATA_DIR` 指向正确的数据目录
- [ ] `BEST_BASE_LLM` 设置为 Table X 中 SNCR 最高的模型
- [ ] 对应的模型目录存在：`experiments/rq1_<model>_seed_42/best_model/`
- [ ] KG 文件路径正确（如果需要修改）
- [ ] 已保存修改后的文件

---

## 准备运行 RQ2

配置完成后，就可以运行 RQ2 实验了：

```bash
bash scripts/run_rq2_experiments.sh
```

脚本会自动：
1. 运行 4 个基线实验
2. 评估 NutriPlan（如果还没评估）
3. 生成 Table Y 对比结果
4. 将配置保存到 `configs/rq2_baseline_config.json`

---

## 常见问题

### Q: 我可以用除了 seed 42 以外的种子吗？

A: 可以，修改 `run_rq2_experiments.sh` 第 13 行：
```bash
BEST_SEED=123  # 或 2024
```

### Q: 如果多个模型 SNCR 相近怎么选？

A: 考虑以下优先级：
1. SNCR 最高
2. 如果 SNCR 差异 < 0.01，选择 UPM 更高的
3. 如果仍相近，选择参数量更小的（更快）

### Q: 可以同时运行多个最佳模型的对比吗？

A: 可以，但需要手动运行每个模型的 RQ2 实验，并修改输出目录名称以避免覆盖。

---

配置完成后，请参考 `SERVER_EXECUTION_GUIDE.md` 继续执行实验！
