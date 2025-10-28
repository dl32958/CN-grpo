from datasets import load_dataset, DatasetDict
import os

ds_all = load_dataset("csv", data_files="datasets/raw/Multitarget-CONAN.csv")["train"]

test_ratio = 0.10
val_ratio = 0.10

first = ds_all.train_test_split(test_size=test_ratio, seed=42)
second = first["train"].train_test_split(
    test_size=val_ratio/(1 - test_ratio),
    seed=42
)

dataset = {
    "train": second["train"],
    "val":   second["test"],
    "test":  first["test"],
}

dataset["train"].to_json("datasets/processed/train.jsonl", lines=True)
dataset["val"].to_json("datasets/processed/val.jsonl", lines=True)
dataset["test"].to_json("datasets/processed/test.jsonl", lines=True)