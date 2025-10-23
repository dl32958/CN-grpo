import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import yaml
import logging

with open("config_infer.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["model_path"]
DATA_PATH = config["data_path"]
OUTPUT_PATH = config["output_path"]
MAX_NEW_TOKENS = config["max_new_tokens"]
TEMPERATURE = config["temperature"]
TOP_P = config["top_p"]
BATCH_SIZE = config["batch_size"]

os.makedirs("outputs", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,  # for h200
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = True
model.eval()

# dataset
df = pd.read_csv(DATA_PATH)

if "HATE_SPEECH" not in df.columns:
    raise ValueError("Dataset must contain column 'HATE_SPEECH'")
prompts = df["HATE_SPEECH"].tolist()
responses = []

prompts = [
    (
        "You are a helpful assistant that generates counter-speech to combat hate speech.\n"
        "Generate a counter-speech to respond to the following hate message in a respectful and persuasive way:\n"
        f"User: {text}\nAssistant:"
    )
    for text in df["HATE_SPEECH"].tolist()
]

# batch generation
def batch_generate(model, tokenizer, prompts, batch_size):
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating counter-speech"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_responses = [x.split("Assistant:")[-1].strip() for x in decoded]
        responses.extend(batch_responses)

    return responses

print("Starting generating counter-speech...")
responses = batch_generate(model, tokenizer, prompts, BATCH_SIZE)

# save results
df["GENERATED_COUNTER_SPEECH"] = responses
df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to {OUTPUT_PATH}")
print("Inference completed.")