import os, json, yaml, argparse, math
from datetime import datetime
from typing import List, Dict
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(description="Generate counter-narratives on test split")
    p.add_argument("--limit", type=int, default=None, help="limit number of samples for quick run")
    p.add_argument("--csv", action="store_true", help="export CSV (plus JSONL)")
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
max_input_len = config["generation"].get("max_input_len", 512)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(json.loads(s))
    return items

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chunk(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def build_prompt_str(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---- model & tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# ---- load dataset ----
records = load_jsonl(test_path)
if args.limit:
    records = records[:args.limit]

prompts = []
for rec in records:
    hate_speech = rec["HATE_SPEECH"].strip()
    user_text = user_template.format(HATE_SPEECH=hate_speech)
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

# ---- save results ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

rows = []
for rec, gen_text in zip(records, generations):
    row = {
        "INDEX": rec.get("INDEX"),
        "HATE_SPEECH": rec["HATE_SPEECH"],
        "COUNTER_NARRATIVE": rec.get("COUNTER_NARRATIVE"),
        "TARGET": rec.get("TARGET"),
        "VERSION": rec.get("VERSION"),
        "GENERATED_COUNTER_SPEECH": gen_text,
        "model_id": model_path,
        "timestamp": timestamp,
    }
    rows.append(row)

os.makedirs(output_dir, exist_ok=True)
jsonl_path = os.path.join(output_dir, f"gen-results-{timestamp}.jsonl")
write_jsonl(jsonl_path, rows)

if args.csv:
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"gen-results-{timestamp}.csv")
    df.to_csv(csv_path, index=False)

print(f"JSONL saved: {jsonl_path}")
if args.csv:
    print(f"CSV saved: {csv_path}")
print(f"Total generations: {len(rows)}")