import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import yaml
import logging

with open("config.yaml", "r") as f:
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
    dtype=torch.bfloat16,
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

# batch generation
def batch_generate(model, tokenizer, texts, batch_size):
    responses = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating counter-speech"):
        batch_texts = texts[i:i+batch_size]
        chat_prompts = []
        for text in batch_texts:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates counter-speech to combat hate speech. "
                        "Respond respectfully and persuasively."
                    )
                },
                {"role": "user", "content": text}
            ]

            # use official chat template, apply_chat_template
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            chat_prompts.append(chat_prompt)

        inputs = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():   # only feedforward, no backpropagation
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # extract assistant response only
        batch_responses = []
        for prompt_text, output_text in zip(chat_prompts, decoded_outputs):
            if output_text.startswith(prompt_text):
                output_text = output_text[len(prompt_text):].strip()
            batch_responses.append(output_text)

        responses.extend(batch_responses)

    return responses

print("Starting generating counter-speech...")
responses = batch_generate(model, tokenizer, prompts, BATCH_SIZE)

# save results
df["GENERATED_COUNTER_SPEECH"] = responses
df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to {OUTPUT_PATH}")
print("Inference completed.")