# NutriPlan 服务器执行指南

## 📋 完整执行流程

### 上传前检查清单

在上传代码到服务器前，请确认：

- ✅ 数据文件已存在于服务器的 `data/` 目录：
  - `task_a_train_discriminative.jsonl`
  - `task_a_val_discriminative.jsonl`
  - `task_a_test_discriminative.jsonl`
  - `task_b_train_from_kg.jsonl`
  - `task_b_val_from_kg.jsonl`
  - `task_b_test_from_kg.jsonl`
  - `task_c_train_from_kg.jsonl`
  - `task_c_val_from_kg.jsonl`
  - `task_c_test_from_kg.jsonl`

- ✅ GPU 资源充足：
  - 至少 24GB VRAM（推荐 A100 或 V100）
  - 至少 200GB 存储空间

---

## 🚀 阶段 II：基础模型选择 (RQ1)

### 步骤 1：训练 15 个实验（5模型 × 3种子）

```bash
# 进入实验目录
cd /path/to/nutriplan_experiments

# 确保脚本有执行权限
chmod +x scripts/train_all_llms_PLAN_A.sh

# 运行批量训练（预计 3-7 天）
bash scripts/train_all_llms_PLAN_A.sh
```

**训练的模型：**
1. TinyLlama/TinyLlama-1.1B-Chat-v1.0
2. microsoft/Phi-3-mini-4k-instruct
3. Qwen/Qwen2-7B
4. mistralai/Mistral-7B-v0.3
5. google/gemma-2-9b

**每个模型使用种子：** 42, 123, 2024

**超参数配置（已针对 GPU 内存和 NaN 优化）：**
- Learning Rate: 3e-5
- Batch Size: 2
- Gradient Accumulation Steps: 4（有效批大小 = 8）
- Epochs: 5
- Max Grad Norm: 0.5（防止梯度爆炸）

### 步骤 2：监控训练进度

```bash
# 查看已完成的实验数量
find experiments -name "training_complete.txt" | wc -l

# 实时查看训练日志（以 TinyLlama seed 42 为例）
tail -f experiments/rq1_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed_42/logs/train.log

# 监控 GPU 使用情况
watch -n 1 nvidia-smi
```

### 步骤 3：聚合 RQ1 结果并生成 Table X

所有 15 个实验完成后：

```bash
python scripts/aggregate_rq1_results.py \
    --experiments_dir experiments \
    --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/Phi-3-mini-4k-instruct Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \
    --seeds 42 123 2024 \
    --output_file results/table_x.txt
```

**输出文件：**
- `results/table_x.txt` - 文本格式表格
- `results/table_x.csv` - CSV 格式
- `results/table_x.tex` - LaTeX 格式

**查看结果：**
```bash
cat results/table_x.txt
```

**关键任务：** 从 Table X 中识别 SNCR 分数最高的模型（这将是最佳基础模型）

---

## 🎯 阶段 III：整体性能对比 (RQ2)

### 步骤 4：更新最佳模型配置

基于 Table X 的结果，编辑 `scripts/run_rq2_experiments.sh` 第 11 行：

```bash
nano scripts/run_rq2_experiments.sh
```

将 `BEST_BASE_LLM` 更新为 Table X 中表现最好的模型，例如：
```bash
BEST_BASE_LLM="Qwen/Qwen2-7B"  # 示例，根据实际结果修改
```

**注意：** 脚本还需要更新数据路径（第 6 行）：
```bash
DATA_DIR="data"  # 如果数据在 data/ 目录
```

### 步骤 5：运行所有基线对比实验

```bash
# 确保脚本有执行权限
chmod +x scripts/run_rq2_experiments.sh

# 运行所有 RQ2 实验（预计 1-2 天）
bash scripts/run_rq2_experiments.sh
```

**运行的基线方法：**
1. **Retrieval (BM25)** - 基于检索的方法
2. **RAG** - 检索增强生成
3. **SFT (Task B only)** - 仅在 Task B 上微调
4. **Zero-shot LLM** - 零样本大语言模型

### 步骤 6：查看最终结果

RQ2 实验完成后，Table Y 会自动生成：

```bash
# 查看完整对比表格
cat results/table_y.txt

# 查看 CSV 格式（便于导入 Excel）
cat results/table_y.csv
```

---

## 📊 预期输出目录结构

```
nutriplan_experiments/
├── experiments/
│   ├── rq1_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed_42/
│   │   ├── best_model/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── eval/
│   │   │   └── aggregate_metrics.json
│   │   └── training_complete.txt
│   ├── rq1_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed_123/
│   ├── rq1_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed_2024/
│   ├── ... (共 15 个 rq1 实验)
│   ├── rq2_retrieval/
│   │   └── eval/
│   ├── rq2_rag/
│   │   └── eval/
│   ├── rq2_sft/
│   │   ├── best_model/
│   │   └── eval/
│   └── rq2_zeroshot/
│       └── eval/
├── results/
│   ├── table_x.txt
│   ├── table_x.csv
│   ├── table_x.tex
│   ├── table_y.txt
│   ├── table_y.csv
│   └── table_y.tex
└── configs/
    └── rq2_baseline_config.json (自动生成)
```

---

## ⚙️ 关键配置参数说明

### Train All LLMs (PLAN A) 配置

**文件：** `scripts/train_all_llms_PLAN_A.sh`

| 参数 | 值 | 说明 |
|------|-----|------|
| `LEARNING_RATE` | 3e-5 | 降低以防止 NaN |
| `BATCH_SIZE` | 2 | 适配 GPU 内存 |
| `GRADIENT_ACCUM_STEPS` | 4 | 有效批大小 = 8 |
| `NUM_EPOCHS` | 5 | 训练轮数 |
| `MAX_GRAD_NORM` | 0.5 | 梯度裁剪（防止梯度爆炸）|
| `TASK_A_RATIO` | 0.5 | Task A 采样比例 |
| `TASK_B_RATIO` | 0.3 | Task B 采样比例 |
| `TASK_C_RATIO` | 0.2 | Task C 采样比例 |

### RQ2 Experiments 配置

**文件：** `scripts/run_rq2_experiments.sh`

**需要手动更新的参数：**
- 第 6 行：`DATA_DIR` - 数据目录路径
- 第 11 行：`BEST_BASE_LLM` - 从 Table X 选出的最佳模型

---

## 🔍 故障排查

### 问题 1：GPU 内存不足

**症状：** `CUDA out of memory` 错误

**解决方案：**
```bash
# 编辑 train_all_llms_PLAN_A.sh
nano scripts/train_all_llms_PLAN_A.sh

# 修改第 22 行：
BATCH_SIZE=1  # 从 2 减少到 1

# 修改第 23 行（保持有效批大小）：
GRADIENT_ACCUM_STEPS=8  # 从 4 增加到 8
```

### 问题 2：训练出现 NaN loss

**症状：** 日志中显示 `loss: nan`

**解决方案：**
- 检查学习率（已设为 3e-5，如果仍有问题可降至 1e-5）
- 检查梯度裁剪（已设为 0.5，可降至 0.3）
- 检查数据是否包含异常值

### 问题 3：某个实验训练失败

**症状：** 某个模型某个种子的训练中断

**解决方案：**
```bash
# 删除失败的实验目录
rm -rf experiments/rq1_<model>_seed_<seed>

# 重新运行训练脚本（会自动跳过已完成的实验）
bash scripts/train_all_llms_PLAN_A.sh
```

### 问题 4：数据路径错误

**症状：** `FileNotFoundError: data file not found`

**解决方案：**
```bash
# 检查数据文件是否存在
ls -la data/*.jsonl

# 如果数据在其他位置，创建符号链接
ln -s /actual/data/path data

# 或者编辑脚本中的 DATA_DIR 变量
```

### 问题 5：aggregate_rq1_results.py 找不到结果

**症状：** `No results found for model X`

**解决方案：**
```bash
# 检查实验目录是否存在
ls experiments/rq1_*

# 检查评估结果是否生成
find experiments -name "aggregate_metrics.json"

# 手动运行评估（如果缺失）
python evaluation/run_evaluation.py \
    --model_path experiments/rq1_<model>_seed_<seed>/best_model \
    --data_dir data \
    --output_dir experiments/rq1_<model>_seed_<seed>/eval \
    --split test
```

---

## ⏱️ 时间估算

| 阶段 | 任务 | 预计时间 | GPU 需求 |
|------|------|----------|----------|
| **II** | 训练 15 个实验 | 3-7 天 | 24GB+ |
| **II** | 聚合 RQ1 结果 | 1-2 分钟 | 无 |
| **III** | Retrieval 基线 | 30-60 分钟 | 无 |
| **III** | RAG 基线 | 2-4 小时 | 24GB |
| **III** | SFT 基线 | 4-8 小时 | 24GB |
| **III** | Zero-shot 基线 | 3-6 小时 | 24GB |
| **III** | 生成 Table Y | 1-2 分钟 | 无 |
| **总计** | 完整流程 | **1-2 周** | **24GB+** |

---

## 📝 核心评估指标说明

### 主要指标（Primary Metrics）

1. **SNCR (Strict Nutrition Constraint Recall)** - 严格营养约束召回率
   - 范围：0.60-0.85（越高越好）
   - 衡量精确的营养约束满足度
   - **这是选择最佳模型的主要指标**

2. **UPM (User Preference Matching)** - 用户偏好匹配度
   - 范围：0.55-0.80（越高越好）
   - 衡量饮食偏好和过敏源合规性

3. **K-Faith (KG Faithfulness)** - 知识图谱忠实度
   - 范围：0.50-0.75（越高越好）
   - 衡量与知识图谱的对齐程度

4. **AVC (Average Violation Count)** - 平均违规次数
   - 范围：0.5-2.0（**越低越好**）
   - 约束违规的平均数量

### 次要指标（Secondary Metrics）

- **BLEU-1/2/3/4** - 生成质量
- **ROUGE-L** - 长序列匹配
- **Dist-2** - 生成多样性（0.40-0.70）
- **Nutrition Accuracy** - 营养准确性（0.60-0.85）

---

## ✅ 执行完成后的检查清单

- [ ] 15 个 RQ1 实验全部完成（每个都有 `training_complete.txt`）
- [ ] Table X 已生成（`results/table_x.txt`）
- [ ] 已从 Table X 识别最佳模型
- [ ] 已更新 `run_rq2_experiments.sh` 中的 `BEST_BASE_LLM`
- [ ] 4 个 RQ2 基线实验全部完成
- [ ] Table Y 已生成（`results/table_y.txt`）
- [ ] 所有评估结果的 JSON 文件完整存在

---

## 📤 结果下载

完成所有实验后，可以只下载关键结果文件：

```bash
# 打包关键结果
tar -czf nutriplan_results.tar.gz \
    results/ \
    experiments/*/eval/aggregate_metrics.json \
    experiments/*/training_complete.txt

# 下载到本地
scp user@server:/path/to/nutriplan_results.tar.gz ./
```

---

## 🎉 最终目标

成功完成后，你将获得：

1. **Table X**：5 个基础模型在 3 个随机种子上的平均性能对比
2. **Table Y**：NutriPlan vs 4 个基线方法的完整性能对比
3. **15 个训练好的模型**：可用于后续分析和部署
4. **完整的评估指标**：包括所有主要和次要指标

这些结果可以直接用于：
- 论文的实验结果部分
- 模型性能分析
- 消融研究（Ablation Studies）
- 案例研究（Case Studies）

---

## 📞 需要帮助？

如果遇到问题：
1. 检查本文档的"故障排查"部分
2. 查看 `EXECUTION_GUIDE.md` 获取更详细的说明
3. 检查训练日志：`experiments/*/logs/train.log`
4. 验证数据文件的完整性

**祝实验顺利！🚀**
