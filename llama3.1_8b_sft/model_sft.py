import os, json, yaml, torch
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
)

# ---------- load config ----------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["sft"]["model_path"]
TRAIN_PATH = config["sft"]["train_data_path"]
VAL_PATH = config["sft"]["val_data_path"]
OUTPUT_DIR = config["sft"]["output_dir"]
SYSTEM_PROMPT = config["prompts"]["system"]
USER_TEMPLATE = config["prompts"]["user"]

# ---------- tokenizer & model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
tokenizer.model_max_length = 768

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# ---------- load CSV datasets ----------
def read_csv_records(path: str):
    df = pd.read_csv(path)
    need = ["HATE_SPEECH", "COUNTER_NARRATIVE"]
    for col in need:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col} in {path}")
    df = df.dropna(subset=need)
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df.to_dict(orient="records")

train_raw = read_csv_records(TRAIN_PATH)
val_raw = read_csv_records(VAL_PATH)

# ---------- build chat-formatted text ----------
def to_chat_text(ex: Dict[str, Any]) -> str:
    user_msg = USER_TEMPLATE.format(HATE_SPEECH=ex["HATE_SPEECH"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ex["COUNTER_NARRATIVE"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

train_ds = Dataset.from_list([{"text": to_chat_text(x)} for x in train_raw])
val_ds = Dataset.from_list([{"text": to_chat_text(x)} for x in val_raw])

# ---------- completion-only data collator ----------
@dataclass
class CompletionOnlyCollatorPreferAssistant:
    tokenizer: Any
    max_length: int = 768
    keep_min_context: int = 128

    def __post_init__(self):
        # use assistant empty reply prefix to locate the start of assistant segment
        prefix_text = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}],
            tokenize=False, add_generation_prompt=False
        )
        self.prefix_ids = self.tokenizer(
            prefix_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].tolist()

    @staticmethod
    def _find_subseq_start(seq: List[int], subseq: List[int]) -> int:
        L, l = len(seq), len(subseq)
        for i in range(0, L - l + 1):
            if seq[i:i+l] == subseq:
                return i
        return -1

    def __call__(self, features: List[Dict[str, Any]]):
        texts = [f["text"] for f in features]
        batch_ids, batch_labels = [], []

        for text in texts:
            full_ids = self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            if isinstance(full_ids[0], list):
                full_ids = full_ids[0]

            pos = self._find_subseq_start(full_ids, self.prefix_ids)
            assistant_start_full = (pos + len(self.prefix_ids)) if pos != -1 else max(0, len(full_ids) - self.max_length)
            assistant_end_full = len(full_ids)
            assistant_len = assistant_end_full - assistant_start_full

            if len(full_ids) <= self.max_length:
                window_start = 0
            else:
                left_budget = self.max_length - assistant_len
                if left_budget <= 0:
                    window_start = assistant_end_full - self.max_length
                else:
                    keep_left = min(self.keep_min_context, left_budget)
                    window_start = max(0, assistant_start_full - keep_left)
                    if assistant_end_full - window_start > self.max_length:
                        window_start = assistant_end_full - self.max_length

            window_ids = full_ids[window_start:assistant_end_full]
            start_in_window = max(0, assistant_start_full - window_start)

            labels = [-100] * len(window_ids)
            for j in range(start_in_window, len(window_ids)):
                labels[j] = window_ids[j]

            batch_ids.append(window_ids)
            batch_labels.append(labels)

        B = len(batch_ids)
        maxL = max(len(x) for x in batch_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((B, maxL), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, maxL), dtype=torch.long)
        labels = torch.full((B, maxL), -100, dtype=torch.long)

        for i, ids in enumerate(batch_ids):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :L] = 1
            labels[i, :L] = torch.tensor(batch_labels[i], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

collator = CompletionOnlyCollatorPreferAssistant(tokenizer, tokenizer.model_max_length, keep_min_context=128)

# ---------- training args ----------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=4,
    load_best_model_at_end=True,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,      # collator based on "text"
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
)

# ---------- trainer ----------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
)

# check
sample_batch = collator([train_ds[0]])
print(f"Input shape: {sample_batch['input_ids'].shape}")
print(f"Labels shape: {sample_batch['labels'].shape}")
print(f"Non-masked labels: {(sample_batch['labels'][0] != -100).sum()}")

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training completed! Model saved to {OUTPUT_DIR}")
