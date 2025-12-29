import pandas as pd
x = pd.read_parquet("work/recipebench/data/4out/ingredients_with_grams.parquet")
print("qty_parsed 非空占比 =", x["qty_parsed"].notna().mean())
print(x["unit_std"].isna().mean(), "  # unit_std 缺失占比")
print(x["unit_std"].value_counts().head(1))
import pandas as pd
hh = pd.read_csv("work/recipebench/data/3out/household_weights_A1.csv")
def canon(u):
    s = str(u).strip().lower().replace(".","")
    m = {"tbsp":"tablespoon","tsp":"teaspoon","cups":"cup","ounces":"ounce","oz":"ounce", "lbs":"pound","lb":"pound",
         "cans":"can","containers":"container","bananas":"banana","tomatoes":"tomato","onions":"onion","eggs":"egg",
         "cloves":"clove","leaves":"leaf","breasts":"breast","fillets":"fillet","sticks":"stick","pieces":"piece",
         "slices":"slice","bunches":"bunch","servings":"serving","packages":"package"}
    return m.get(s, s)

hh_units = set(hh["unit"].map(canon).dropna().unique())

x = pd.read_parquet("work/recipebench/data/4out/ingredients_with_grams.parquet")
x_units = x["unit_std"].dropna().astype(str).str.strip().str.lower().str.replace(".","", regex=False)
miss = x_units[~x_units.isin(hh_units)].value_counts().head(50)
print("A表没有覆盖到的 unit_std TOP 50：")
print(miss)
