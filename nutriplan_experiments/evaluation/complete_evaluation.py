#!/usr/bin/env python3
"""
完整评估脚本 - 计算所有论文需要的指标
包含：
1. NutriPlan 私有指标：SNCR, UPM, K-Faith, AVC
2. 标准 NLG 指标：BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, BERTScore
3. 多样性指标：Dist-1/2/3, Self-BLEU
4. 任务特定指标：营养准确度、成分覆盖率
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
from tqdm import tqdm
import re

# NLG评估库
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# 需要先下载NLTK数据
import nltk
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')


class CompleteNutriPlanEvaluator:
    """完整的NutriPlan评估器"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()

    # ============================================================
    # NutriPlan 私有指标
    # ============================================================

    def compute_sncr(self, generated: Dict, constraints: Dict, tolerance: float = 0.1) -> float:
        """
        Strict Nutrition Constraint Recall (SNCR)
        严格营养约束召回率
        """
        nutrition_targets = constraints.get('nutrition_targets', {})
        if not nutrition_targets:
            return 1.0

        generated_nutrition = generated.get('nutrition', {})
        satisfied = 0
        total = 0

        for nutrient, target in nutrition_targets.items():
            total += 1
            gen_value = generated_nutrition.get(nutrient, 0)

            if isinstance(target, dict):
                # 范围约束: {"min": 10, "max": 50}
                min_val = target.get('min', 0)
                max_val = target.get('max', float('inf'))
                if min_val * (1 - tolerance) <= gen_value <= max_val * (1 + tolerance):
                    satisfied += 1
            else:
                # 精确约束
                if abs(gen_value - target) <= target * tolerance:
                    satisfied += 1

        return satisfied / total if total > 0 else 1.0

    def compute_upm(self, generated: Dict, reference: Dict) -> float:
        """
        User Preference Match (UPM)
        用户偏好匹配度
        """
        gen_prefs = set(generated.get('user_preferences', []))
        ref_prefs = set(reference.get('user_preferences', []))

        if not ref_prefs:
            return 1.0

        matched = len(gen_prefs & ref_prefs)
        return matched / len(ref_prefs)

    def compute_k_faith(self, generated: Dict, kg_facts: List[str]) -> float:
        """
        Knowledge Graph Faithfulness (K-Faith)
        知识图谱忠实度
        """
        if not kg_facts:
            return 1.0

        generated_text = json.dumps(generated, ensure_ascii=False).lower()
        faithful_count = 0

        for fact in kg_facts:
            # 简化版：检查关键词是否出现
            keywords = re.findall(r'\w+', fact.lower())
            if all(kw in generated_text for kw in keywords[:3]):  # 至少前3个关键词
                faithful_count += 1

        return faithful_count / len(kg_facts) if kg_facts else 1.0

    def compute_avc(self, generated: Dict, constraints: Dict) -> float:
        """
        Allergy Violation Count (AVC)
        过敏原违反计数（越低越好）
        """
        allergens = set(constraints.get('allergens', []))
        if not allergens:
            return 0.0

        ingredients = generated.get('ingredients', [])
        if isinstance(ingredients, str):
            ingredients = [ingredients]

        violations = 0
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            for allergen in allergens:
                if allergen.lower() in ingredient_lower:
                    violations += 1

        return violations

    # ============================================================
    # 标准 NLG 指标
    # ============================================================

    def compute_bleu(self, generated: str, reference: str) -> Dict[str, float]:
        """
        BLEU-1, BLEU-2, BLEU-3, BLEU-4
        """
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()

        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1/n] * n + [0] * (4-n))
            score = sentence_bleu(
                [ref_tokens],
                gen_tokens,
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            bleu_scores[f'bleu_{n}'] = score

        return bleu_scores

    def compute_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """
        ROUGE-1, ROUGE-2, ROUGE-L (F1 scores)
        """
        scores = self.rouge_scorer.score(reference, generated)

        return {
            'rouge_1': scores['rouge1'].fmeasure,
            'rouge_2': scores['rouge2'].fmeasure,
            'rouge_l': scores['rougeL'].fmeasure
        }

    def compute_meteor(self, generated: str, reference: str) -> float:
        """
        METEOR score
        """
        try:
            score = meteor_score([reference.lower().split()], generated.lower().split())
            return score
        except Exception as e:
            print(f"METEOR计算失败: {e}")
            return 0.0

    def compute_bertscore(self, generated_list: List[str], reference_list: List[str]) -> Dict[str, float]:
        """
        BERTScore (批量计算，更高效)
        返回平均 Precision, Recall, F1
        """
        try:
            P, R, F1 = bert_score(
                generated_list,
                reference_list,
                lang='en',
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"BERTScore计算失败: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }

    # ============================================================
    # 多样性指标
    # ============================================================

    def compute_distinct_n(self, texts: List[str], n: int) -> float:
        """
        Distinct-N: 衡量生成文本的多样性
        """
        all_ngrams = []

        for text in texts:
            tokens = text.lower().split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return 0.0

        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)

        return unique_ngrams / total_ngrams

    def compute_self_bleu(self, generated_list: List[str], sample_size: int = 100) -> float:
        """
        Self-BLEU: 衡量生成文本之间的相似度（越低越好，表示多样性高）
        """
        if len(generated_list) < 2:
            return 0.0

        # 采样以加速计算
        import random
        if len(generated_list) > sample_size:
            sampled = random.sample(generated_list, sample_size)
        else:
            sampled = generated_list

        bleu_scores = []
        for i, gen in enumerate(sampled):
            gen_tokens = gen.lower().split()
            # 用其他所有句子作为参考
            references = [s.lower().split() for j, s in enumerate(sampled) if j != i]

            if references:
                score = sentence_bleu(
                    references,
                    gen_tokens,
                    smoothing_function=self.smoothing.method1
                )
                bleu_scores.append(score)

        return np.mean(bleu_scores) if bleu_scores else 0.0

    # ============================================================
    # 任务特定指标
    # ============================================================

    def compute_nutrition_accuracy(self, generated: Dict, reference: Dict, tolerance: float = 0.15) -> float:
        """
        营养信息准确度（与参考食谱比较）
        """
        gen_nutrition = generated.get('nutrition', {})
        ref_nutrition = reference.get('nutrition', {})

        if not ref_nutrition:
            return 1.0

        accurate_count = 0
        total_count = 0

        for nutrient, ref_value in ref_nutrition.items():
            total_count += 1
            gen_value = gen_nutrition.get(nutrient, 0)

            if abs(gen_value - ref_value) <= ref_value * tolerance:
                accurate_count += 1

        return accurate_count / total_count if total_count > 0 else 1.0

    def compute_ingredient_coverage(self, generated: Dict, reference: Dict) -> float:
        """
        成分覆盖率
        """
        gen_ingredients = set(generated.get('ingredients', []))
        ref_ingredients = set(reference.get('ingredients', []))

        if not ref_ingredients:
            return 1.0

        covered = len(gen_ingredients & ref_ingredients)
        return covered / len(ref_ingredients)

    # ============================================================
    # 主评估函数
    # ============================================================

    def evaluate_single(
        self,
        generated: Dict,
        reference: Dict,
        constraints: Dict,
        kg_facts: List[str],
        generated_text: str,
        reference_text: str
    ) -> Dict[str, float]:
        """
        评估单个样本的所有指标
        """
        metrics = {}

        # 1. NutriPlan 私有指标
        metrics['sncr'] = self.compute_sncr(generated, constraints)
        metrics['upm'] = self.compute_upm(generated, reference)
        metrics['k_faith'] = self.compute_k_faith(generated, kg_facts)
        metrics['avc'] = self.compute_avc(generated, constraints)

        # 2. BLEU 系列
        bleu_scores = self.compute_bleu(generated_text, reference_text)
        metrics.update(bleu_scores)

        # 3. ROUGE 系列
        rouge_scores = self.compute_rouge(generated_text, reference_text)
        metrics.update(rouge_scores)

        # 4. METEOR
        metrics['meteor'] = self.compute_meteor(generated_text, reference_text)

        # 5. 任务特定指标
        metrics['nutrition_accuracy'] = self.compute_nutrition_accuracy(generated, reference)
        metrics['ingredient_coverage'] = self.compute_ingredient_coverage(generated, reference)

        return metrics

    def evaluate_all(
        self,
        predictions_file: str,
        references_file: str,
        constraints_file: str,
        kg_facts_file: str = None,
        output_dir: str = "evaluation_results"
    ) -> Dict[str, Any]:
        """
        评估所有样本并生成完整报告
        """
        print("=" * 80)
        print("NutriPlan 完整评估")
        print("=" * 80)

        # 加载数据
        predictions = self._load_jsonl(predictions_file)
        references = self._load_jsonl(references_file)
        constraints = self._load_jsonl(constraints_file)
        kg_facts = self._load_jsonl(kg_facts_file) if kg_facts_file else [[] for _ in predictions]

        print(f"加载了 {len(predictions)} 个预测样本")

        # 准备文本列表（用于批量计算）
        generated_texts = []
        reference_texts = []

        for pred, ref in zip(predictions, references):
            gen_text = pred.get('generated', json.dumps(pred, ensure_ascii=False))
            ref_text = ref.get('output', json.dumps(ref, ensure_ascii=False))
            generated_texts.append(gen_text)
            reference_texts.append(ref_text)

        # 逐样本评估
        all_metrics = []

        print("\n逐样本评估...")
        for i in tqdm(range(len(predictions))):
            metrics = self.evaluate_single(
                generated=predictions[i],
                reference=references[i],
                constraints=constraints[i],
                kg_facts=kg_facts[i] if i < len(kg_facts) else [],
                generated_text=generated_texts[i],
                reference_text=reference_texts[i]
            )
            metrics['sample_id'] = i
            all_metrics.append(metrics)

        # 批量计算 BERTScore
        print("\n计算 BERTScore...")
        bertscore_metrics = self.compute_bertscore(generated_texts, reference_texts)
        for metrics in all_metrics:
            metrics.update(bertscore_metrics)

        # 计算多样性指标（全局）
        print("\n计算多样性指标...")
        diversity_metrics = {
            'dist_1': self.compute_distinct_n(generated_texts, 1),
            'dist_2': self.compute_distinct_n(generated_texts, 2),
            'dist_3': self.compute_distinct_n(generated_texts, 3),
            'self_bleu': self.compute_self_bleu(generated_texts)
        }

        # 聚合统计
        aggregate_results = self._aggregate_metrics(all_metrics, diversity_metrics)

        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._save_results(aggregate_results, all_metrics, output_path)

        return aggregate_results

    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """加载 JSONL 文件"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _aggregate_metrics(self, per_sample: List[Dict], diversity: Dict) -> Dict:
        """聚合所有指标的统计信息"""
        # 所有要聚合的指标
        metric_names = [
            # NutriPlan 私有
            'sncr', 'upm', 'k_faith', 'avc',
            # BLEU
            'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
            # ROUGE
            'rouge_1', 'rouge_2', 'rouge_l',
            # 其他
            'meteor', 'bertscore_f1', 'bertscore_precision', 'bertscore_recall',
            # 任务特定
            'nutrition_accuracy', 'ingredient_coverage'
        ]

        aggregate = {}

        for metric in metric_names:
            values = [m[metric] for m in per_sample if metric in m]
            if values:
                aggregate[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        # 添加多样性指标
        aggregate['diversity'] = diversity

        return aggregate

    def _save_results(self, aggregate: Dict, per_sample: List[Dict], output_dir: Path):
        """保存所有结果"""

        # 1. 聚合指标 (JSON)
        with open(output_dir / 'aggregate_metrics.json', 'w') as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)

        # 2. 每个样本的指标 (CSV)
        import pandas as pd
        df = pd.DataFrame(per_sample)
        df.to_csv(output_dir / 'per_sample_metrics.csv', index=False)

        # 3. 论文表格 (LaTeX格式)
        self._generate_paper_table(aggregate, output_dir)

        print(f"\n✅ 结果已保存到: {output_dir}")

    def _generate_paper_table(self, aggregate: Dict, output_dir: Path):
        """生成论文用的表格"""

        table_file = output_dir / 'paper_table.txt'

        with open(table_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("NutriPlan 评估结果 - 论文表格\n")
            f.write("=" * 100 + "\n\n")

            # 表格头
            f.write(f"{'指标':<30} {'均值':<12} {'标准差':<12} {'中位数':<12}\n")
            f.write("-" * 100 + "\n")

            # NutriPlan 私有指标
            f.write("\n【NutriPlan 私有指标】\n")
            for metric in ['sncr', 'upm', 'k_faith', 'avc']:
                if metric in aggregate:
                    stats = aggregate[metric]
                    direction = "↓" if metric == 'avc' else "↑"
                    f.write(f"{metric.upper()} {direction:<27} {stats['mean']:.4f}      {stats['std']:.4f}      {stats['median']:.4f}\n")

            # BLEU 系列
            f.write("\n【BLEU 指标】\n")
            for i in range(1, 5):
                metric = f'bleu_{i}'
                if metric in aggregate:
                    stats = aggregate[metric]
                    f.write(f"BLEU-{i}  ↑{'':<24} {stats['mean']:.4f}      {stats['std']:.4f}      {stats['median']:.4f}\n")

            # ROUGE 系列
            f.write("\n【ROUGE 指标】\n")
            for metric in ['rouge_1', 'rouge_2', 'rouge_l']:
                if metric in aggregate:
                    stats = aggregate[metric]
                    name = metric.replace('_', '-').upper()
                    f.write(f"{name}  ↑{'':<24} {stats['mean']:.4f}      {stats['std']:.4f}      {stats['median']:.4f}\n")

            # 其他标准指标
            f.write("\n【其他标准指标】\n")
            if 'meteor' in aggregate:
                stats = aggregate['meteor']
                f.write(f"METEOR  ↑{'':<23} {stats['mean']:.4f}      {stats['std']:.4f}      {stats['median']:.4f}\n")

            if 'bertscore_f1' in aggregate:
                stats = aggregate['bertscore_f1']
                f.write(f"BERTScore-F1  ↑{'':<17} {stats['mean']:.4f}      {stats['std']:.4f}      {stats['median']:.4f}\n")

            # 多样性指标
            f.write("\n【多样性指标】\n")
            div = aggregate.get('diversity', {})
            for metric in ['dist_1', 'dist_2', 'dist_3', 'self_bleu']:
                if metric in div:
                    value = div[metric]
                    name = metric.replace('_', '-').upper()
                    direction = "↓" if metric == 'self_bleu' else "↑"
                    f.write(f"{name}  {direction:<23} {value:.4f}      -           -\n")

            # 任务特定指标
            f.write("\n【任务特定指标】\n")
            for metric in ['nutrition_accuracy', 'ingredient_coverage']:
                if metric in aggregate:
                    stats = aggregate[metric]
                    name = metric.replace('_', ' ').title()
                    f.write(f"{name}  ↑{'':<10} {stats['mean']:.4f}      {stats['std']:.4f}      {stats['median']:.4f}\n")

            f.write("\n" + "=" * 100 + "\n")

        print(f"✅ 论文表格已保存: {table_file}")

        # 同时打印到控制台
        with open(table_file, 'r') as f:
            print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(description="NutriPlan 完整评估")
    parser.add_argument('--predictions', type=str, required=True, help='预测文件 (JSONL)')
    parser.add_argument('--references', type=str, required=True, help='参考文件 (JSONL)')
    parser.add_argument('--constraints', type=str, required=True, help='约束文件 (JSONL)')
    parser.add_argument('--kg_facts', type=str, default=None, help='知识图谱事实 (JSONL, 可选)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')

    args = parser.parse_args()

    evaluator = CompleteNutriPlanEvaluator()
    results = evaluator.evaluate_all(
        predictions_file=args.predictions,
        references_file=args.references,
        constraints_file=args.constraints,
        kg_facts_file=args.kg_facts,
        output_dir=args.output_dir
    )

    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()
