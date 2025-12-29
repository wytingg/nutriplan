# 🚀 NutriPlan 实验 - 从这里开始

## 📚 文档导航

本项目包含完整的 NutriPlan 实验框架，用于评估基于知识图谱的营养约束食谱生成。

### 📖 核心文档（按阅读顺序）

1. **SERVER_EXECUTION_GUIDE.md** ⭐ 最重要
   - 完整的服务器执行指南
   - 包含所有阶段的详细说明
   - 故障排查指南
   - 预期结果和时间估算

2. **COMMANDS_CHECKLIST.md** ⭐ 快速参考
   - 按步骤列出的所有命令
   - 包含检查点和验证步骤
   - 适合执行时对照使用

3. **QUICK_START_SERVER.sh** ⭐ 首次运行
   - 环境检查脚本
   - 上传到服务器后第一个运行的脚本
   - 自动验证数据和依赖

4. **UPDATE_RQ2_CONFIG.md**
   - 如何更新 RQ2 配置
   - 详细的修改说明和示例
   - 阶段 II 完成后必读

### 📄 其他文档

- **EXECUTION_GUIDE.md** - 详细的执行指南（包含理论背景）
- **QUICK_COMMANDS.md** - 快速命令参考
- **README.md** - 项目总览
- **COMPLETE_FILE_LIST.md** - 完整文件列表
- **QUICKSTART.md** - 快速入门

---

## 🎯 实验目标

### 阶段 II：基础模型选择（RQ1）
- 训练 5 个基础模型 × 3 个随机种子 = **15 个实验**
- 生成 **Table X**：模型性能对比表
- 识别最佳基础模型

### 阶段 III：整体性能对比（RQ2）
- 运行 4 个基线方法
- 生成 **Table Y**：NutriPlan vs 基线方法
- 完成最终性能评估

---

## ⚡ 快速开始（3 步）

### 步骤 1：上传到服务器

```bash
# 在本地执行
rsync -avz --progress C:/Users/wyt03/nutriplan_kg/nutriplan_experiments/ username@server:/path/to/destination/
```

### 步骤 2：运行环境检查

```bash
# SSH 登录服务器
ssh username@server
cd /path/to/nutriplan_experiments

# 运行检查脚本
bash QUICK_START_SERVER.sh
```

### 步骤 3：开始训练

```bash
# 在 tmux 会话中运行（防止 SSH 断开）
tmux new -s nutriplan_train

# 开始训练 15 个实验
bash scripts/train_all_llms_PLAN_A.sh
```

**就这么简单！** 🎉

---

## 📊 实验流程图

```
开始
  ↓
上传代码到服务器
  ↓
运行环境检查 (QUICK_START_SERVER.sh)
  ↓
┌─────────────────────────────────────────┐
│ 阶段 II：基础模型选择（RQ1）            │
│                                         │
│ 1. 训练 15 个实验                       │
│    bash train_all_llms_PLAN_A.sh       │
│    ⏱️  预计：3-7 天                     │
│                                         │
│ 2. 聚合结果生成 Table X                │
│    python aggregate_rq1_results.py     │
│    ⏱️  预计：1-2 分钟                   │
│                                         │
│ 3. 查看 Table X，识别最佳模型          │
│    cat results/table_x.txt             │
└─────────────────────────────────────────┘
  ↓
更新 RQ2 配置
  ↓ (参考 UPDATE_RQ2_CONFIG.md)
┌─────────────────────────────────────────┐
│ 阶段 III：整体性能对比（RQ2）           │
│                                         │
│ 1. 运行 4 个基线实验                    │
│    bash run_rq2_experiments.sh         │
│    ⏱️  预计：1-2 天                     │
│                                         │
│ 2. 自动生成 Table Y                     │
│    ⏱️  预计：1-2 分钟                   │
│                                         │
│ 3. 查看最终结果                         │
│    cat results/table_y.txt             │
└─────────────────────────────────────────┘
  ↓
下载结果到本地
  ↓
完成！🎉
```

---

## 📋 实验配置概览

### 训练的模型（Plan A）

1. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - 1.1B 参数
2. **microsoft/Phi-3-mini-4k-instruct** - 3.8B 参数
3. **Qwen/Qwen2-7B** - 7B 参数
4. **mistralai/Mistral-7B-v0.3** - 7B 参数
5. **google/gemma-2-9b** - 9B 参数

### 随机种子

- Seed 1: 42
- Seed 2: 123
- Seed 3: 2024

### 超参数（已优化）

```yaml
学习率: 3e-5
批大小: 2
梯度累积步数: 4 (有效批大小 = 8)
训练轮数: 5
最大梯度范数: 0.5
任务比例: A=0.5, B=0.3, C=0.2
```

### 基线方法（RQ2）

1. **Retrieval (BM25)** - 基于检索
2. **RAG** - 检索增强生成
3. **SFT (Task B)** - 监督微调
4. **Zero-shot LLM** - 零样本

---

## 🎯 关键评估指标

### 主要指标（用于模型选择）

| 指标 | 全称 | 范围 | 目标 |
|------|------|------|------|
| **SNCR** | Strict Nutrition Constraint Recall | 0.60-0.85 | ⬆️ 越高越好 |
| **UPM** | User Preference Matching | 0.55-0.80 | ⬆️ 越高越好 |
| **K-Faith** | KG Faithfulness | 0.50-0.75 | ⬆️ 越高越好 |
| **AVC** | Average Violation Count | 0.5-2.0 | ⬇️ 越低越好 |

**最佳模型选择标准：** SNCR 最高的模型

### 次要指标（用于质量评估）

- BLEU-1/2/3/4（生成质量）
- ROUGE-L（长序列匹配）
- Dist-2（生成多样性）
- Nutrition Accuracy（营养准确性）

---

## 📁 关键输出文件

实验完成后，你将获得：

```
results/
├── table_x.txt      # RQ1: 基础模型对比
├── table_x.csv      # CSV 格式
├── table_x.tex      # LaTeX 格式
├── table_y.txt      # RQ2: 完整方法对比
├── table_y.csv      # CSV 格式
└── table_y.tex      # LaTeX 格式

experiments/
├── rq1_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed_42/
│   ├── best_model/               # 训练好的模型
│   ├── eval/
│   │   └── aggregate_metrics.json  # 评估结果
│   └── training_complete.txt      # 完成标记
├── ... (共 15 个 rq1 实验)
├── rq2_retrieval/eval/
├── rq2_rag/eval/
├── rq2_sft/eval/
└── rq2_zeroshot/eval/

configs/
└── rq2_baseline_config.json  # RQ2 配置（自动生成）
```

---

## ⏱️ 时间线估算

| 阶段 | 时间 | 可并行 |
|------|------|--------|
| 环境检查 | 5 分钟 | ❌ |
| **阶段 II 训练** | **3-7 天** | ✅ (GPU 足够) |
| 聚合 RQ1 | 1-2 分钟 | ❌ |
| 更新配置 | 5 分钟 | ❌ |
| **阶段 III 基线** | **1-2 天** | ✅ (部分) |
| 生成 Table Y | 1-2 分钟 | ❌ |
| **总计** | **~1-2 周** | - |

---

## 💡 重要提示

### ✅ DO（推荐做的）

- ✅ 使用 `tmux` 或 `screen` 运行长时间任务
- ✅ 定期检查训练进度和 GPU 使用情况
- ✅ 在修改配置前备份原文件
- ✅ 保存所有日志以便故障排查
- ✅ 在实验完成后及时下载关键结果

### ❌ DON'T（避免做的）

- ❌ 不要在没有 tmux/screen 的情况下运行长任务
- ❌ 不要在实验进行中修改配置文件
- ❌ 不要删除实验目录（除非确认失败需要重跑）
- ❌ 不要同时运行多个训练脚本（可能导致 GPU OOM）
- ❌ 不要忽略 `training_complete.txt` 检查

---

## 🔧 常见问题速查

### Q1: GPU 内存不足怎么办？

**A:** 减少批大小，增加梯度累积步数
```bash
# 编辑 train_all_llms_PLAN_A.sh
BATCH_SIZE=1
GRADIENT_ACCUM_STEPS=8
```

### Q2: 如何知道哪个实验失败了？

**A:** 运行这个命令
```bash
find experiments -path "*/rq1_*" -type d -maxdepth 1 | while read dir; do
    if [ ! -f "$dir/training_complete.txt" ]; then
        echo "失败: $dir"
    fi
done
```

### Q3: 训练中断了怎么办？

**A:** 重新运行训练脚本，会自动跳过已完成的
```bash
bash scripts/train_all_llms_PLAN_A.sh
```

### Q4: 如何选择最佳模型？

**A:** 查看 Table X，选择 SNCR 分数最高的模型

### Q5: 可以提前终止某个实验吗？

**A:** 可以，但建议让所有实验完成以确保结果的完整性

---

## 📞 获取帮助

遇到问题时，按以下顺序查找解决方案：

1. **首先查看：** `SERVER_EXECUTION_GUIDE.md` 的故障排查部分
2. **查看日志：** `experiments/*/logs/train.log`
3. **验证数据：** 确保所有数据文件存在且格式正确
4. **检查配置：** 确保所有路径和模型名称正确

---

## 🎊 准备好了吗？

如果你已经：

- [x] 阅读了本文档
- [x] 准备好了服务器环境（GPU + 200GB 存储）
- [x] 数据文件已准备好
- [x] 了解了基本的 Linux 和 tmux 命令

那么你可以开始了！🚀

**下一步：** 打开 `COMMANDS_CHECKLIST.md`，按照步骤执行。

---

## 📊 预期成果

实验成功完成后，你将能够：

✅ 回答 **RQ1**：哪个基础 LLM 最适合 NutriPlan？
✅ 回答 **RQ2**：NutriPlan 相比基线方法的性能如何？
✅ 获得论文所需的 **Table X** 和 **Table Y**
✅ 拥有 **15 个训练好的模型**供进一步分析
✅ 完成营养约束食谱生成的**完整性能评估**

---

**祝实验顺利！如有疑问，随时参考相关文档。** 📚✨

---

*最后更新：2025-01-15*
*文档版本：v1.0*
