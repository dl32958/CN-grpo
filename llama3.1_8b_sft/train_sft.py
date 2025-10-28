import os, json, yaml, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments
)
from trl import SFTTrainer


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model_path"]
TRAIN_PATH = config["train_data_path"]
VAL_PATH = config["val_data_path"]
OUTPUT_DIR = config["output_dir"]

# tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,  # for h200
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# dataset
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_raw = load_data(TRAIN_PATH)
val_raw = load_data(VAL_PATH)

train_ds = Dataset.from_list(train_raw)
val_ds = Dataset.from_list(val_raw)

SYSTEM = (
"You are a helpful, factual, and empathetic assistant. "
"Given a problematic claim from an online discussion, write a short, respectful reply "
"that reduces tension, corrects inaccuracies, and promotes inclusive norms. "
"Do not repeat slurs (paraphrase if needed). No safety disclaimers or 'as an AI' statements. "
)

def formatting_func(example):
    user_msg = (
        "Reply constructively to this message from an online discussion:\n"
        f"{example['HATE_SPEECH']}"
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": example["COUNTER_NARRATIVE"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

# train args
training_args = TrainingArguments(
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
    bf16=True,   # H200
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,  # chat text won't lose columns
)

# trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    formatting_func=formatting_func,
)

trainer.train()

# save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)