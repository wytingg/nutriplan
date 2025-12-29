# NutriPlan 实验执行命令检查清单

按顺序执行这些命令，完成整个实验流程。

---

## 📦 上传到服务器后的初始化

### 1. 上传代码到服务器

```bash
# 在本地机器上执行
scp -r C:/Users/wyt03/nutriplan_kg/nutriplan_experiments/ username@server:/path/to/destination/

# 或使用 rsync（推荐，支持断点续传）
rsync -avz --progress C:/Users/wyt03/nutriplan_kg/nutriplan_experiments/ username@server:/path/to/destination/
```

### 2. SSH 登录服务器

```bash
ssh username@server
cd /path/to/nutriplan_experiments
```

### 3. 运行环境检查脚本

```bash
bash QUICK_START_SERVER.sh
```

✅ **检查点：** 确保所有数据文件都显示 ✓，无 ✗

---

## 🔬 阶段 II：基础模型选择（RQ1）

### 4. 开始训练 15 个实验

```bash
# 确保在 tmux 或 screen 会话中运行（防止 SSH 断开）
tmux new -s nutriplan_train

# 或者
screen -S nutriplan_train

# 然后运行训练
bash scripts/train_all_llms_PLAN_A.sh
```

**预计时间：** 3-7 天

**如何分离 tmux 会话：** 按 `Ctrl+B` 然后按 `D`

**如何重新连接：**
```bash
tmux attach -t nutriplan_train
# 或
screen -r nutriplan_train
```

---

### 5. 监控训练进度（在新的 SSH 会话中）

#### 5.1 查看已完成的实验数量

```bash
# 应该逐渐增加到 15
find experiments -name "training_complete.txt" | wc -l
```

#### 5.2 列出所有已完成的实验

```bash
find experiments -name "training_complete.txt" -exec dirname {} \;
```

#### 5.3 查看特定模型的训练日志

```bash
# 示例：TinyLlama seed 42
tail -f experiments/rq1_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed_42/logs/train.log

# 查看最后 50 行
tail -n 50 experiments/rq1_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed_42/logs/train.log
```

#### 5.4 监控 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看当前使用情况
nvidia-smi

# 退出 watch：按 Ctrl+C
```

#### 5.5 查看磁盘使用情况

```bash
# 查看实验目录大小
du -sh experiments

# 查看每个实验的大小
du -sh experiments/rq1_*
```

---

### 6. 训练完成后，聚合 RQ1 结果

✅ **检查点：** 确保 `find experiments -name "training_complete.txt" | wc -l` 输出为 15

```bash
python scripts/aggregate_rq1_results.py \
    --experiments_dir experiments \
    --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/Phi-3-mini-4k-instruct Qwen/Qwen2-7B mistralai/Mistral-7B-v0.3 google/gemma-2-9b \
    --seeds 42 123 2024 \
    --output_file results/table_x.txt
```

**预计时间：** 1-2 分钟

---

### 7. 查看 Table X 结果

```bash
cat results/table_x.txt
```

✅ **检查点：** 记录 SNCR 列中数值最高的模型名称

**示例输出：**
```
Model                                      SNCR              UPM               ...
TinyLlama/TinyLlama-1.1B-Chat-v1.0        0.650±0.012       0.580±0.015       ...
microsoft/Phi-3-mini-4k-instruct          0.720±0.008       0.690±0.010       ...
Qwen/Qwen2-7B                             0.785±0.006       0.740±0.008       ...  ⭐
mistralai/Mistral-7B-v0.3                 0.770±0.010       0.730±0.012       ...
google/gemma-2-9b                         0.760±0.009       0.720±0.011       ...
```

**记录最佳模型：** `Qwen/Qwen2-7B`（示例）

---

## 🎯 阶段 III：整体性能对比（RQ2）

### 8. 更新 RQ2 配置文件

#### 方法 A：手动编辑（推荐）

```bash
nano scripts/run_rq2_experiments.sh
```

修改以下行：
- 第 6 行：`DATA_DIR="data"` （如果数据在 data/ 目录）
- 第 11 行：`BEST_BASE_LLM="Qwen/Qwen2-7B"` （替换为 Table X 最佳模型）

保存并退出：`Ctrl+O` → `Enter` → `Ctrl+X`

#### 方法 B：使用 sed 自动修改

```bash
# 备份原文件
cp scripts/run_rq2_experiments.sh scripts/run_rq2_experiments.sh.backup

# 修改数据目录
sed -i 's|^DATA_DIR="D:/Downloads"|DATA_DIR="data"|' scripts/run_rq2_experiments.sh

# 修改最佳模型（示例：Qwen/Qwen2-7B）
sed -i 's|^BEST_BASE_LLM="meta-llama/Llama-3-8B"|BEST_BASE_LLM="Qwen/Qwen2-7B"|' scripts/run_rq2_experiments.sh
```

#### 验证修改

```bash
grep -E "^(DATA_DIR|BEST_BASE_LLM)=" scripts/run_rq2_experiments.sh
```

✅ **检查点：** 输出应显示正确的路径和模型名称

---

### 9. 运行 RQ2 基线对比实验

```bash
# 同样在 tmux/screen 会话中运行
tmux new -s nutriplan_rq2
# 或
screen -S nutriplan_rq2

# 运行 RQ2 实验
bash scripts/run_rq2_experiments.sh
```

**预计时间：** 1-2 天

**包含的实验：**
1. Retrieval (BM25)
2. RAG
3. SFT (Task B only)
4. Zero-shot LLM

---

### 10. 监控 RQ2 进度

#### 10.1 查看已完成的基线实验

```bash
# Retrieval
ls -la experiments/rq2_retrieval/eval/aggregate_metrics.json

# RAG
ls -la experiments/rq2_rag/eval/aggregate_metrics.json

# SFT
ls -la experiments/rq2_sft/eval/aggregate_metrics.json

# Zero-shot
ls -la experiments/rq2_zeroshot/eval/aggregate_metrics.json
```

#### 10.2 查看 RQ2 日志

```bash
# 查看脚本的实时输出（在 tmux/screen 会话中）
# 或查看已保存的日志（如果有重定向）
```

---

### 11. 查看最终结果 Table Y

RQ2 实验完成后，Table Y 会自动生成：

```bash
cat results/table_y.txt
```

✅ **检查点：** Table Y 应包含 5 行数据：
- NutriPlan
- Retrieval (BM25)
- RAG
- SFT (Task B)
- Zero-shot LLM

---

## 📥 下载结果到本地

### 12. 打包关键结果

```bash
# 在服务器上执行
cd /path/to/nutriplan_experiments

tar -czf nutriplan_results.tar.gz \
    results/ \
    experiments/*/eval/aggregate_metrics.json \
    experiments/*/training_complete.txt \
    configs/rq2_baseline_config.json
```

### 13. 下载到本地

```bash
# 在本地机器上执行
scp username@server:/path/to/nutriplan_experiments/nutriplan_results.tar.gz ./

# 解压
tar -xzf nutriplan_results.tar.gz
```

---

## 📊 验证实验完整性

### 14. 最终检查清单

在服务器上运行这些命令，确保所有实验都完成：

```bash
# 检查 RQ1 实验（应该是 15）
echo "RQ1 完成的实验数："
find experiments -path "*/rq1_*/training_complete.txt" | wc -l

# 检查 RQ1 评估结果（应该是 15）
echo "RQ1 评估结果数："
find experiments -path "*/rq1_*/eval/aggregate_metrics.json" | wc -l

# 检查 RQ2 基线实验（应该是 4）
echo "RQ2 基线评估结果数："
ls experiments/rq2_*/eval/aggregate_metrics.json 2>/dev/null | wc -l

# 检查关键结果文件
echo "关键结果文件："
ls -lh results/table_x.txt results/table_y.txt

# 检查配置文件
echo "RQ2 配置文件："
ls -lh configs/rq2_baseline_config.json
```

✅ **预期输出：**
```
RQ1 完成的实验数：
15
RQ1 评估结果数：
15
RQ2 基线评估结果数：
4
关键结果文件：
-rw-r--r-- 1 user group 5.2K Jan 15 10:30 results/table_x.txt
-rw-r--r-- 1 user group 4.8K Jan 16 14:20 results/table_y.txt
RQ2 配置文件：
-rw-r--r-- 1 user group  512 Jan 16 14:20 configs/rq2_baseline_config.json
```

---

## 🐛 故障排查命令

### 某个实验失败了

```bash
# 查找失败的实验（没有 training_complete.txt 的）
for seed in 42 123 2024; do
    for model in TinyLlama_TinyLlama-1.1B-Chat-v1.0 microsoft_Phi-3-mini-4k-instruct Qwen_Qwen2-7B mistralai_Mistral-7B-v0.3 google_gemma-2-9b; do
        exp_dir="experiments/rq1_${model}_seed_${seed}"
        if [ ! -f "$exp_dir/training_complete.txt" ]; then
            echo "失败: $exp_dir"
        fi
    done
done
```

### 手动重新运行失败的实验

```bash
# 删除失败的实验目录
rm -rf experiments/rq1_<model>_seed_<seed>

# 重新运行训练脚本（会自动跳过已完成的）
bash scripts/train_all_llms_PLAN_A.sh
```

### 查看错误日志

```bash
# 查看最后 100 行日志，寻找错误信息
tail -n 100 experiments/rq1_<model>_seed_<seed>/logs/train.log | grep -i error

# 或查看完整日志
less experiments/rq1_<model>_seed_<seed>/logs/train.log
```

### 检查 GPU 内存使用

```bash
# 查看当前所有 GPU 进程
nvidia-smi

# 查看详细的内存使用
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# 如果需要终止某个进程
kill -9 <PID>
```

---

## 📈 可选：查看详细评估指标

### 查看特定模型的完整评估结果

```bash
# 以 JSON 格式查看（需要 jq 工具）
cat experiments/rq1_Qwen_Qwen2-7B_seed_42/eval/aggregate_metrics.json | jq .

# 或使用 Python 格式化
python -c "import json; print(json.dumps(json.load(open('experiments/rq1_Qwen_Qwen2-7B_seed_42/eval/aggregate_metrics.json')), indent=2))"
```

### 对比所有种子的 SNCR

```bash
echo "Model: Qwen/Qwen2-7B"
for seed in 42 123 2024; do
    sncr=$(python -c "import json; m=json.load(open('experiments/rq1_Qwen_Qwen2-7B_seed_$seed/eval/aggregate_metrics.json')); print(f\"{m['sncr']['mean']:.4f}\")")
    echo "  Seed $seed: SNCR = $sncr"
done
```

---

## ✅ 完成标志

所有实验成功完成后，你应该拥有：

- [ ] 15 个完成的 RQ1 实验目录
- [ ] `results/table_x.txt` 文件
- [ ] 4 个完成的 RQ2 基线实验目录
- [ ] `results/table_y.txt` 文件
- [ ] `configs/rq2_baseline_config.json` 文件

**恭喜！实验已全部完成！🎉**

---

## 📞 快速参考

### 常用目录路径

- 训练脚本：`scripts/train_all_llms_PLAN_A.sh`
- RQ2 脚本：`scripts/run_rq2_experiments.sh`
- 聚合脚本：`scripts/aggregate_rq1_results.py`
- 实验结果：`experiments/`
- 最终表格：`results/table_x.txt` 和 `results/table_y.txt`

### 常用检查命令

```bash
# 完成的实验数
find experiments -name "training_complete.txt" | wc -l

# GPU 状态
nvidia-smi

# 磁盘使用
df -h

# 查看 Table X
cat results/table_x.txt

# 查看 Table Y
cat results/table_y.txt
```

---

**祝实验顺利！如有问题，请参考 SERVER_EXECUTION_GUIDE.md 中的故障排查部分。**
