# train_sft_hf.py  —— Supervised Fine-Tuning without TRL
import json, yaml, torch
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
USER_TEMPLATE = config["prompts"]["user_template"]

# ---------- tokenizer & model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# ---------- load dataset ----------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_raw = read_jsonl(TRAIN_PATH)
val_raw   = read_jsonl(VAL_PATH)


def to_chat_text(ex: Dict[str, Any]) -> str:
    user_msg = USER_TEMPLATE.format(hate_speech=ex["HATE_SPEECH"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ex["COUNTER_NARRATIVE"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

train_ds = Dataset.from_list([{"text": to_chat_text(x)} for x in train_raw])
val_ds   = Dataset.from_list([{"text": to_chat_text(x)} for x in val_raw])

# ---------- completion-only data collator ----------
@dataclass
class CompletionOnlyCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __post_init__(self):
        # 用"空 assistant"生成前缀，定位回复起点
        prefix_text = self.tokenizer.apply_chat_template(
            [{"role":"assistant","content":""}],
            tokenize=False, add_generation_prompt=False
        )
        self.prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        print(f"Assistant prefix pattern: {self.prefix_ids}")  # 调试信息

    def _find_subseq(self, seq, subseq):
        L, l = len(seq), len(subseq)
        for i in range(0, L - l + 1):
            if seq[i:i+l] == subseq:
                return i + l
        return None

    def __call__(self, features: List[Dict[str, Any]]):
        texts = [f["text"] for f in features]
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        # 先把 padding 置为 -100
        labels[enc["attention_mask"] == 0] = -100

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            start = self._find_subseq(ids, self.prefix_ids)
            if start is None:
                # 找不到就保守处理：除 padding 外全部 mask（防止错误监督）
                print(f"Warning: Could not find assistant prefix in sample {i}")  # 调试
                labels[i, labels[i] != -100] = -100
            else:
                # 只让 assistant 段参与 loss
                labels[i, :start] = -100

        enc["labels"] = labels
        return enc

collator = CompletionOnlyCollator(tokenizer, tokenizer.model_max_length)

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
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=4,
    load_best_model_at_end=True,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,    # 我们的 collator基于 "text"
    # 添加一些有用的参数
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_pin_memory=True,     # 加速数据加载
    dataloader_num_workers=2,       # 并行数据加载
)

# ---------- trainer ----------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,   # 关键：只训 assistant 段
)

# 训练前检查一下数据
print("Checking first training sample:")
sample_batch = collator([train_ds[0]])
print(f"Input shape: {sample_batch['input_ids'].shape}")
print(f"Labels shape: {sample_batch['labels'].shape}")
print(f"Non-masked labels: {(sample_batch['labels'][0] != -100).sum()}")

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training completed! Model saved to {OUTPUT_DIR}")