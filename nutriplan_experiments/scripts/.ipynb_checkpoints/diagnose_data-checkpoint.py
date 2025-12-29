#!/usr/bin/env python3
"""
超级详细的训练数据诊断脚本
检查所有可能影响 LLM 训练质量的问题
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
import unicodedata

class DataDiagnostics:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = []
        self.issues = []
        self.warnings = []
        self.stats = defaultdict(list)

    def load_data(self):
        """加载 JSONL 数据"""
        print(f"\n{'='*80}")
        print(f"检查文件: {self.filepath}")
        print(f"{'='*80}\n")

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            self.data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            self.issues.append(f"第 {i+1} 行 JSON 解析错误: {e}")
        except Exception as e:
            self.issues.append(f"文件读取错误: {e}")
            return False

        print(f"✓ 加载了 {len(self.data)} 条数据")
        return True

    def check_basic_format(self):
        """检查基本格式"""
        print("\n[1/12] 检查基本字段...")

        required_fields = ['instruction', 'output']

        for i, item in enumerate(self.data):
            # 检查必需字段
            missing = [f for f in required_fields if f not in item]
            if missing:
                self.issues.append(f"样本 {i}: 缺少字段 {missing}")

            # 检查字段是否为空
            for field in required_fields:
                if field in item and not item[field]:
                    self.issues.append(f"样本 {i}: 字段 '{field}' 为空")

        if not self.issues:
            print("  ✓ 所有样本都包含必需字段")

    def check_encoding(self):
        """检查编码问题"""
        print("\n[2/12] 检查字符编码...")

        non_ascii_samples = []
        control_char_samples = []

        for i, item in enumerate(self.data):
            text = json.dumps(item, ensure_ascii=False)

            # 检查非 ASCII 字符（除了正常的标点）
            non_ascii = [c for c in text if ord(c) > 127]
            if non_ascii:
                # 统计字符类型
                char_types = Counter([unicodedata.category(c) for c in non_ascii])
                # 如果包含西里尔文、中文等
                if any(ord(c) > 0x0400 for c in non_ascii):
                    non_ascii_samples.append({
                        'index': i,
                        'chars': ''.join(set(non_ascii))[:50],
                        'categories': dict(char_types)
                    })

            # 检查控制字符
            control_chars = [c for c in text if unicodedata.category(c) == 'Cc' and c not in '\n\t']
            if control_chars:
                control_char_samples.append(i)

        if non_ascii_samples:
            self.warnings.append(f"发现 {len(non_ascii_samples)} 个样本包含非 ASCII 字符")
            for sample in non_ascii_samples[:5]:
                print(f"  ⚠️  样本 {sample['index']}: 包含字符 '{sample['chars']}'")
        else:
            print("  ✓ 所有样本都是纯 ASCII（英文）")

        if control_char_samples:
            self.issues.append(f"{len(control_char_samples)} 个样本包含控制字符")

    def check_length_distribution(self):
        """检查长度分布"""
        print("\n[3/12] 检查文本长度分布...")

        inst_lengths = []
        out_lengths = []

        for item in self.data:
            inst_lengths.append(len(item.get('instruction', '')))
            out_lengths.append(len(item.get('output', '')))

        self.stats['inst_length'] = inst_lengths
        self.stats['out_length'] = out_lengths

        print(f"  Instruction 长度: min={min(inst_lengths)}, max={max(inst_lengths)}, avg={sum(inst_lengths)/len(inst_lengths):.1f}")
        print(f"  Output 长度:      min={min(out_lengths)}, max={max(out_lengths)}, avg={sum(out_lengths)/len(out_lengths):.1f}")

        # 检查异常短的样本
        short_inst = [i for i, l in enumerate(inst_lengths) if l < 50]
        short_out = [i for i, l in enumerate(out_lengths) if l < 20]

        if short_inst:
            self.warnings.append(f"{len(short_inst)} 个样本的 instruction 过短 (<50 字符)")
        if short_out:
            self.warnings.append(f"{len(short_out)} 个样本的 output 过短 (<20 字符)")

        # 检查异常长的样本
        long_inst = [i for i, l in enumerate(inst_lengths) if l > 2000]
        long_out = [i for i, l in enumerate(out_lengths) if l > 4000]

        if long_inst:
            self.warnings.append(f"{len(long_inst)} 个样本的 instruction 过长 (>2000 字符)")
        if long_out:
            self.warnings.append(f"{len(long_out)} 个样本的 output 过长 (>4000 字符)")

    def check_duplicates(self):
        """检查重复样本"""
        print("\n[4/12] 检查重复样本...")

        inst_hashes = {}
        duplicates = []

        for i, item in enumerate(self.data):
            inst = item.get('instruction', '')
            if inst in inst_hashes:
                duplicates.append((i, inst_hashes[inst]))
            else:
                inst_hashes[inst] = i

        if duplicates:
            self.warnings.append(f"发现 {len(duplicates)} 对重复的 instruction")
            for dup in duplicates[:3]:
                print(f"  ⚠️  样本 {dup[0]} 和 {dup[1]} 的 instruction 相同")
        else:
            print("  ✓ 没有重复的 instruction")

    def check_format_consistency(self):
        """检查格式一致性"""
        print("\n[5/12] 检查 output 格式一致性...")

        # 检查 output 格式模式
        patterns = {
            'numbered_list': r'^\d+\.\s',  # 1. 2. 3.
            'markdown_bold': r'\*\*.*?\*\*',  # **text**
            'recipe_format': r'Ingredients:|Instructions:|Nutrition:',
            'diagnosis_format': r'Diagnosis:|Corrections:',
            'json_format': r'^\s*\{',
        }

        format_counts = Counter()

        for item in self.data:
            output = item.get('output', '')
            for fmt_name, pattern in patterns.items():
                if re.search(pattern, output, re.MULTILINE):
                    format_counts[fmt_name] += 1

        print("  Output 格式分布:")
        total = len(self.data)
        for fmt, count in format_counts.most_common():
            print(f"    - {fmt}: {count} ({count/total*100:.1f}%)")

        # 检查是否有混合格式
        if len(format_counts) > 3:
            self.warnings.append(f"检测到 {len(format_counts)} 种不同的 output 格式，可能影响训练一致性")

    def check_special_tokens(self):
        """检查特殊 token 和标记"""
        print("\n[6/12] 检查特殊 token...")

        special_patterns = [
            (r'<[^>]+>', 'HTML/XML 标签'),
            (r'\[.*?\]', '方括号标记'),
            (r'\{.*?\}', '花括号（可能是 JSON）'),
            (r'@\w+', '@mention'),
            (r'#\w+', 'hashtag'),
            (r'http[s]?://\S+', 'URL'),
        ]

        for pattern, name in special_patterns:
            count = 0
            for item in self.data:
                text = json.dumps(item, ensure_ascii=False)
                if re.search(pattern, text):
                    count += 1
            if count > 0:
                print(f"    {name}: {count} 个样本 ({count/len(self.data)*100:.1f}%)")

    def check_numeric_patterns(self):
        """检查数值模式"""
        print("\n[7/12] 检查营养数值...")

        # 提取营养数值
        nutrition_values = {
            'calories': [],
            'protein': [],
            'fiber': [],
            'sodium': []
        }

        for item in self.data:
            text = item.get('output', '') + item.get('instruction', '')

            # kcal
            cals = re.findall(r'(\d+)\s*kcal', text)
            if cals:
                nutrition_values['calories'].extend([int(c) for c in cals])

            # protein
            proteins = re.findall(r'(\d+)g?\s*protein', text)
            if proteins:
                nutrition_values['protein'].extend([int(p) for p in proteins])

            # fiber
            fibers = re.findall(r'(\d+)g?\s*fiber', text)
            if fibers:
                nutrition_values['fiber'].extend([int(f) for f in fibers])

        for nutrient, values in nutrition_values.items():
            if values:
                print(f"  {nutrient}: 范围 {min(values)}-{max(values)}, 平均 {sum(values)/len(values):.1f}")

    def check_vocabulary(self):
        """检查词汇表"""
        print("\n[8/12] 检查词汇表...")

        all_words = []
        for item in self.data:
            text = item.get('instruction', '') + ' ' + item.get('output', '')
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            all_words.extend(words)

        vocab = Counter(all_words)
        print(f"  总词汇量: {len(vocab)} 个唯一单词")
        print(f"  总 token 数: {len(all_words)}")
        print(f"  最常见单词: {vocab.most_common(10)}")

    def check_label_imbalance(self):
        """检查标签/任务不平衡"""
        print("\n[9/12] 检查任务类型分布...")

        # 尝试识别任务类型
        task_types = Counter()

        for item in self.data:
            inst = item.get('instruction', '').lower()

            if 'rank' in inst or 'score' in inst:
                task_types['ranking'] += 1
            elif 'recipe' in inst and 'generate' in inst:
                task_types['generation'] += 1
            elif 'diagnose' in inst or 'fix' in inst or 'correct' in inst:
                task_types['correction'] += 1
            else:
                task_types['other'] += 1

        print("  任务类型分布:")
        for task, count in task_types.most_common():
            print(f"    - {task}: {count} ({count/len(self.data)*100:.1f}%)")

        # 检查不平衡
        if task_types:
            max_count = max(task_types.values())
            min_count = min(task_types.values())
            ratio = max_count / min_count if min_count > 0 else float('inf')

            if ratio > 5:
                self.warnings.append(f"任务类型严重不平衡（最大/最小比例: {ratio:.1f}）")

    def check_whitespace_issues(self):
        """检查空白字符问题"""
        print("\n[10/12] 检查空白字符...")

        issues_found = []

        for i, item in enumerate(self.data):
            for field in ['instruction', 'output']:
                text = item.get(field, '')

                # 多余的空白
                if '  ' in text:  # 双空格
                    issues_found.append(f"样本 {i} {field}: 包含多余空格")
                    break

                # 前后空白
                if text != text.strip():
                    issues_found.append(f"样本 {i} {field}: 前后有多余空白")
                    break

                # Tab 字符
                if '\t' in text:
                    issues_found.append(f"样本 {i} {field}: 包含 Tab 字符")
                    break

        if issues_found:
            self.warnings.append(f"{len(issues_found)} 个样本存在空白字符问题")
            for issue in issues_found[:5]:
                print(f"  ⚠️  {issue}")
        else:
            print("  ✓ 空白字符正常")

    def check_instruction_output_mismatch(self):
        """检查 instruction 和 output 的匹配度"""
        print("\n[11/12] 检查 instruction-output 匹配...")

        mismatches = []

        for i, item in enumerate(self.data):
            inst = item.get('instruction', '').lower()
            out = item.get('output', '').lower()

            # 检查 instruction 要求排序，但 output 没有数字列表
            if ('rank' in inst or 'sort' in inst) and not re.search(r'^\d+\.', out, re.MULTILINE):
                mismatches.append(f"样本 {i}: instruction 要求排序，但 output 无编号列表")

            # 检查 instruction 要求食谱，但 output 没有食材/步骤
            if 'recipe' in inst:
                if 'ingredient' not in out and 'instruction' not in out:
                    mismatches.append(f"样本 {i}: instruction 要求食谱，但 output 缺少结构")

        if mismatches:
            self.warnings.append(f"{len(mismatches)} 个样本的 instruction-output 不匹配")
            for m in mismatches[:5]:
                print(f"  ⚠️  {m}")
        else:
            print("  ✓ instruction-output 匹配正常")

    def check_potential_data_leakage(self):
        """检查潜在的数据泄漏"""
        print("\n[12/12] 检查潜在数据泄漏...")

        leakage = []

        for i, item in enumerate(self.data):
            inst = item.get('instruction', '')
            out = item.get('output', '')

            # 检查 output 是否包含在 instruction 中（除了短语）
            if len(out) > 50 and out[:50] in inst:
                leakage.append(f"样本 {i}: output 的开头出现在 instruction 中")

        if leakage:
            self.issues.append(f"{len(leakage)} 个样本可能存在数据泄漏")
            for l in leakage[:5]:
                print(f"  ❌ {l}")
        else:
            print("  ✓ 未检测到数据泄漏")

    def generate_report(self):
        """生成最终报告"""
        print(f"\n{'='*80}")
        print("诊断报告")
        print(f"{'='*80}\n")

        if not self.issues and not self.warnings:
            print("🎉 恭喜！数据质量优秀，未发现任何问题！")
            return True

        if self.issues:
            print(f"❌ 发现 {len(self.issues)} 个严重问题:")
            for issue in self.issues:
                print(f"  - {issue}")
            print()

        if self.warnings:
            print(f"⚠️  发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()

        if self.issues:
            print("建议: 修复严重问题后再训练")
            return False
        else:
            print("建议: 警告不影响训练，可以继续")
            return True

    def run_full_diagnostics(self):
        """运行完整诊断"""
        if not self.load_data():
            return False

        self.check_basic_format()
        self.check_encoding()
        self.check_length_distribution()
        self.check_duplicates()
        self.check_format_consistency()
        self.check_special_tokens()
        self.check_numeric_patterns()
        self.check_vocabulary()
        self.check_label_imbalance()
        self.check_whitespace_issues()
        self.check_instruction_output_mismatch()
        self.check_potential_data_leakage()

        return self.generate_report()


def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_data.py <file.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]

    if not Path(filepath).exists():
        print(f"错误: 文件不存在 {filepath}")
        sys.exit(1)

    diagnostics = DataDiagnostics(filepath)
    success = diagnostics.run_full_diagnostics()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
