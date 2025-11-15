import os, yaml, argparse, math, csv, random, re
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

MODEL_PATH = config["generation"]["model_path"]
TEST_DATA_PATH = config["generation"]["test_data_path"]
OUTPUT_DIR = config["generation"]["output_dir"]

SYSTEM_PROMPT = config["prompts"]["system"]
USER_TEMPLATE = config["prompts"]["user"]
MAX_NEW_TOKENS = config["generation"]["max_new_tokens"]
TEMPERATURE = config["generation"]["temperature"]
TOP_P = config["generation"]["top_p"]
BATCH_SIZE = config["generation"]["batch_size"]
MAX_INPUT_LEN = 512

SEED = config["generation"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def chunk(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def build_prompt_str(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# extract CN from response
def extract_cn(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    s = text.strip()
    if not s:
        return s

    lower = s.lower()

    pat = "assistant\n"
    idx = lower.rfind(pat)
    if idx != -1:
        s = s[idx + len(pat):]
        s = s.lstrip(" .,:;!?\n\"'""''")
        return s.strip()

    parts = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    if not parts:
        parts = [p.strip() for p in s.split("\n") if p.strip()]

    if parts:
        return parts[-1]
    else:
        return s


# ---- model & tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# ---- load CSV test set ----
df = pd.read_csv(TEST_DATA_PATH)
if args.limit:
    df = df.iloc[:args.limit].copy()

if "HATE_SPEECH" not in df.columns:
    raise ValueError("CSV must contain column 'HATE_SPEECH'.")

# ---- build prompts ----
prompts = []
for hs in df["HATE_SPEECH"].fillna("").astype(str).tolist():
    user_text = USER_TEMPLATE.format(HATE_SPEECH=hs.strip())
    prompt = build_prompt_str(tokenizer, SYSTEM_PROMPT, user_text)
    prompts.append(prompt)

# ---- generation ----
generations = []

for batch in tqdm(list(chunk(prompts, BATCH_SIZE)), desc="Generating"):
    encodings = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )
    encodings = {k: v.to(model.device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encodings,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    input_lens = encodings["attention_mask"].sum(dim=1).tolist()
    sequences = outputs.sequences

    for i, input_len in enumerate(input_lens):
        generated_ids = sequences[i][input_len:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        extracted_cn = extract_cn(raw_text)
        generations.append(extracted_cn)

# for i, cn in enumerate(generations[:50]):
#     print(f"{i} {cn}")

# ---- save results (CSV) ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_out = df.copy()
df_out["GENERATED_CN"] = generations

csv_path = os.path.join(OUTPUT_DIR, f"cn-gen-{timestamp}.csv")

df_out.to_csv(csv_path, index=False)
print(f"CSV saved: {csv_path}")

print(f"Total generations: {len(df_out)}")