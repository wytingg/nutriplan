# 检查脚本
import pandas as pd

df = pd.read_csv("work/recipebench/data/4out/recipe_nutrition_foodcom.csv")

print("能量分布（kcal）:")
print(df['Calories_PerServing_kcal'].describe())
print("\n蛋白质分布（g）:")
print(df['Protein_PerServing_g'].describe())

  # 统计各能量区间的食谱数量
print("\n能量区间分布:")
print(f"< 50 kcal: {(df['Calories_PerServing_kcal'] < 50).sum()}")
print(f"50-100 kcal: {((df['Calories_PerServing_kcal'] >= 50) & (df['Calories_PerServing_kcal'] < 100)).sum()}")
print(f"100-200 kcal: {((df['Calories_PerServing_kcal'] >= 100) & (df['Calories_PerServing_kcal'] < 200)).sum()}")
print(f"200-400 kcal: {((df['Calories_PerServing_kcal'] >= 200) & (df['Calories_PerServing_kcal'] < 400)).sum()}")
print(f">= 400 kcal: {(df['Calories_PerServing_kcal'] >= 400).sum()}")