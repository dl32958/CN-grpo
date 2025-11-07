import os, yaml, argparse, math, csv, random
from datetime import datetime
from typing import List, Dict, Any, Iterable
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description="Generate CN on test split (CSV)")
    p.add_argument("--limit", type=int, default=None, help="limit number of samples for quick run")
    return p.parse_args()


args = parse_args()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_path = config["generation"]["model_path"]
test_path = config["generation"]["test_path"]
output_dir = config["generation"]["output_dir"]

system_prompt = config["prompts"]["system"]
user_template = config["prompts"]["user"]
max_new_tokens = config["generation"]["max_new_tokens"]
temperature = config["generation"]["temperature"]
top_p = config["generation"]["top_p"]
batch_size = config["generation"]["batch_size"]
seed = config["generation"]["seed"]
max_input_len = 512

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def chunk(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def build_prompt_str(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---- model & tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# ---- load CSV test set ----
df = pd.read_csv(test_path)
if args.limit:
    df = df.iloc[:args.limit].copy()

if "HATE_SPEECH" not in df.columns:
    raise ValueError("CSV must contain column 'HATE_SPEECH'.")

# ---- build prompts ----
prompts = []
for hs in df["HATE_SPEECH"].fillna("").astype(str).tolist():
    user_text = user_template.format(HATE_SPEECH=hs.strip())
    prompt = build_prompt_str(tokenizer, system_prompt, user_text)
    prompts.append(prompt)

# ---- generation ----
generations = []

for batch in tqdm(list(chunk(prompts, batch_size)), desc="Generating"):
    encodings = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
    )
    encodings = {k: v.to(model.device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    input_lens = encodings["attention_mask"].sum(dim=1).tolist()
    sequences = outputs.sequences

    for i, input_len in enumerate(input_lens):
        generated_ids = sequences[i][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generations.append(text)

# ---- save results (CSV) ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(output_dir, exist_ok=True)

df_out = df.copy()
df_out["GENERATED_CN"] = generations

csv_path = os.path.join(output_dir, f"cn-gen-{timestamp}.csv")

df_out.to_csv(csv_path, index=False)
print(f"CSV saved: {csv_path}")

print(f"Total generations: {len(df_out)}")