import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


df = pd.read_csv("../raw/Multitarget-CONAN.csv")
sft_data = []

for _, row in df.iterrows():
    if pd.isna(row["HATE_SPEECH"]) or pd.isna(row["COUNTER_NARRATIVE"]):
        continue
        
    conversation = {
        "messages": [
            {"role": "user", "content": str(row["HATE_SPEECH"]).strip()},
            {"role": "assistant", "content": str(row["COUNTER_NARRATIVE"]).strip()}
        ]
    }
    sft_data.append(conversation)

# split train/val ssets
train_data, val_data = train_test_split(sft_data, test_size=0.1, random_state=42)

train_file = "train.jsonl"
with open(train_file, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

val_file = "val.jsonl"
with open(val_file, "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
