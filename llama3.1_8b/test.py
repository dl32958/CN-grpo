import pandas as pd

data = pd.read_csv("results/cn-gen-20251114_2322.csv")

for i, cn in enumerate(data['GENERATED_CN']):
    print(f"[{i}] {cn}")
    print("-" * 50)