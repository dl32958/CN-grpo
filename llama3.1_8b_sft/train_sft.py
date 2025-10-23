import torch
import json
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from evaluate import load


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# model
model = AutoModelForCausalLM.from_pretrained(
    config["model_path"],
    dtype=torch.bfloat16,  # for h200
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

# optional compile acceleration
if config["compute"].get("torch_compile", False):
    model = torch.compile(model)

# dataset
def format_conversation(example):
    messages = example["messages"]
    conversation = ""
    for msg in messages:
        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"Assistant: {msg['content']}\n"
    return {'text': conversation.strip()}

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=config["data"]["max_length"],
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_data = [json.loads(line) for line in open(config["train_data_path"], "r", encoding="utf-8")]
val_data = [json.loads(line) for line in open(config["val_data_path"], "r", encoding="utf-8")]

train_dataset = Dataset.from_list([format_conversation(item) for item in train_data])
val_dataset = Dataset.from_list([format_conversation(item) for item in val_data])

# shuffle train dataset
if config["data"].get("shuffle", False):
    train_dataset = train_dataset.shuffle(seed=42)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# train args
training_args = TrainingArguments(
    output_dir=config["training"]["output_dir"],
    num_train_epochs=config["training"]["num_train_epochs"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    learning_rate=float(config["training"]["learning_rate"]),
    weight_decay=float(config["training"]["weight_decay"]),
    warmup_ratio=config["training"].get("warmup_ratio", 0.03),
    logging_steps=config["training"]["logging_steps"],
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    eval_strategy=config["training"]["eval_strategy"],   # evaluation_strategy has been deprecated since version 4.46
    save_strategy=config["training"]["save_strategy"],
    save_total_limit=config["training"]["save_total_limit"],
    load_best_model_at_end=config["training"]["load_best_model_at_end"],   # load best model at the end of training
    bf16=config["compute"]["bf16"],   # for h200 GPUs
    gradient_checkpointing=config["compute"]["gradient_checkpointing"],
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    report_to="none",
)

# data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # better tensor core utilization
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

# save model
trainer.save_model(config["training"]["output_dir"])
tokenizer.save_pretrained(config["training"]["output_dir"])