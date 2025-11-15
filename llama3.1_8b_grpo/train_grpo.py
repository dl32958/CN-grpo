import os
import sys
import math
import random
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SFT_MODEL_PATH = config["grpo"]["model_path"]
TRAIN_PATH = config["grpo"]["train_data_path"]
VAL_PATH = config["grpo"]["val_data_path"]
TEST_PATH = config["grpo"].get("test_data_path", "")
GRPO_OUTPUT_DIR = config["grpo"]["output_dir"]
EVALUATOR_PATH = config["grpo"]["evaluator_path"]
# prompts
SYSTEM_PROMPT = config["prompts"]["system"]
USER_TEMPLATE = config["prompts"]["user"]

os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)

sys.path.append(os.path.dirname(EVALUATOR_PATH))
from evaluator import (
    CounterNarrativeEvaluator,
    OVERALL_WEIGHTS,
    DIVERSITY_WEIGHTS,
    PERSUASIVENESS_WEIGHTS,
)

# Parameters
NUM_EPOCHS = 5
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
NUM_SAMPLES_PER_PROMPT = 4  # K
MAX_NEW_TOKENS = 128
LEARNING_RATE = 5e-6
KL_COEF = 0.02
WARMUP_RATIO = 0.05

GEN_TEMP = 0.7
GEN_TOP_P = 0.9

SEED = 42

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# set up cuda
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available! This code requires CUDA to run.")
device = torch.device("cuda")

# 1. Load tokenizer and model
def load_models_and_tokenizer():
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy model
    policy_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    policy_model.gradient_checkpointing_enable()
    policy_model.train()

    # reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print("models loaded")
    return tokenizer, policy_model, ref_model

# 2. build prompt
def build_chat_prompt(hate_speech, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(HATE_SPEECH=hate_speech)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text

# 3. Dataset & DataLoader
class HateSpeechDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.hates = df["HATE_SPEECH"].astype(str).tolist()

    def __len__(self):
        return len(self.hates)

    def __getitem__(self, idx):
        hs = self.hates[idx]
        return {"hate_speech": hs}

def build_dataloaders():
    train_dataset = HateSpeechDataset(TRAIN_PATH).head(100)  # for test
    val_dataset = HateSpeechDataset(VAL_PATH).head(20)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    return train_loader, val_loader

# 4. Evaluator & reward wrapper
def load_evaluator():
    evaluator = CounterNarrativeEvaluator()
    return evaluator


"""
现在 evaluator 里的 diversity 本质上是“这批句子的整体平均多样性”；在 GRPO 训练代码里，我们对每个 HS 的 K 条候选算出一个组内 diversity，
再把这个常数加到每个 reward 上。但因为 advantage 是在组内做 mean-center，diversity 这一块会被完全抵消，所以目前 diversity 只影响 
offline evaluation，不影响 RL 更新。
"""
def compute_sample_rewards(
    evaluator: CounterNarrativeEvaluator,
    hate_speech_list: List[str],
    cn_list: List[str],
) -> np.ndarray:
    """
    use evaluator's internal functions to calcuate each sample's reward (0-1)
    """
    hs = [str(x) if x is not None else "" for x in hate_speech_list]
    cn = [str(x) if x is not None else "" for x in cn_list]

    # 1. relevance
    rel_scores, _ = evaluator._compute_relevance(hs, cn, batch_size=16)

    # 2. toxicity & civility
    tox_raw, civility = evaluator._compute_toxicity(cn)
    safe_scores = 1.0 - tox_raw

    # 3. length adherence
    _, lengths = evaluator._compute_length(cn)
    len_ok = ((lengths >= 35) & (lengths <= 50)).astype(float)

    # 4. stance opposition
    stance_scores = evaluator._compute_stance_opposition(hs, cn, batch_size=16)

    # 5. answer quality (fluency + acceptability)
    answer_quality = evaluator._compute_answer_quality(cn, batch_size=8)

    # 6. diversity（全局一个分数，广播到每个样本）
    div_res = evaluator._compute_diversity(
        cn,
        sample_k=min(20, max(1, len(cn) - 1)),
        weights=DIVERSITY_WEIGHTS,
    )
    div_score = div_res["div_score"]
    div_vec = np.full(len(cn), div_score, dtype=np.float32)

    # 7. per-sample persuasiveness
    w_stance, w_civ, w_ans = PERSUASIVENESS_WEIGHTS
    pers = np.clip(
        w_stance * stance_scores +
        w_civ * civility +
        w_ans * answer_quality,
        0.0, 1.0,
    )

    # 8. overall per-sample reward
    w_rel = OVERALL_WEIGHTS["relevance"]
    w_div = OVERALL_WEIGHTS["diversity"]
    w_len = OVERALL_WEIGHTS["length"]
    w_tox = OVERALL_WEIGHTS["toxicity"]
    w_pers = OVERALL_WEIGHTS["persuasiveness"]

    rewards = (
        w_rel * rel_scores +
        w_div * div_vec +
        w_len * len_ok +
        w_tox * safe_scores +
        w_pers * pers
    )

    rewards = np.clip(rewards, 0.0, 1.0).astype(np.float32)
    return rewards

# 5. generate_cn function and logprob + KL
@torch.no_grad()
def generate_responses_for_hs(
    hate_speech: str,
    tokenizer,
    policy_model,
    num_samples: int = NUM_SAMPLES_PER_PROMPT,
) -> Dict[str, Any]:
    """
    For each HS, use policy_model to generate K CNs.
    returns:
      - responses_text: List[str]
      - sequences_ids: torch.LongTensor [num_samples, seq_len] (prompt + CN)
      - prompt_len: int
    """
    prompt_text = build_chat_prompt(hate_speech, tokenizer)
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    prompt_len = input_ids.shape[1]

    # Expand for multiple generations
    input_ids_expanded = input_ids.expand(num_samples, -1)
    attn_expanded = attention_mask.expand(num_samples, -1)

    outputs = policy_model.generate(
        input_ids=input_ids_expanded,
        attention_mask=attn_expanded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GEN_TEMP,
        top_p=GEN_TOP_P,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.eos_token_id,
    )

    sequences_ids = outputs
    responses_text = []
    for seq in sequences_ids:
        gen_ids = seq[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        responses_text.append(text)

    return {
        "responses_text": responses_text,
        "sequences_ids": sequences_ids,
        "prompt_len": prompt_len,
    }


def compute_logprobs_and_kl(
    sequences_ids: torch.LongTensor,
    prompt_len: int,
    tokenizer,
    policy_model,
    ref_model,
) -> Dict[str, torch.Tensor]:
    """
    For a batch of sequences (prompt + CN):
    - Calculate policy sequence-level logprob (generation part only)
    - Calculate KL penalty between policy vs ref (generation part only)
    """
    attention_mask = (sequences_ids != tokenizer.pad_token_id).long().to(device)
    sequences_ids = sequences_ids.to(device)

    # policy forward pass
    outputs_policy = policy_model(
        input_ids=sequences_ids,
        attention_mask=attention_mask,
    )
    logits_policy = outputs_policy.logits  # [B, L, V]

    # reference forward pass (no gradients)
    with torch.no_grad():
        outputs_ref = ref_model(
            input_ids=sequences_ids,
            attention_mask=attention_mask,
        )
        logits_ref = outputs_ref.logits  # [B, L, V]

    # shift 1: logits[:, :-1] predicts targets = input_ids[:, 1:]
    logits_pol = logits_policy[:, :-1, :]
    logits_ref = logits_ref[:, :-1, :]
    targets = sequences_ids[:, 1:]

    logprobs_pol = torch.log_softmax(logits_pol, dim=-1)
    logprobs_ref = torch.log_softmax(logits_ref, dim=-1)
    probs_pol = torch.softmax(logits_pol, dim=-1)

    # calculate only for generation part: first generated token's logits index = prompt_len - 1
    gen_start = prompt_len - 1
    gen_logprobs_pol = logprobs_pol[:, gen_start:, :]     # [B, gen_L, V]
    gen_logprobs_ref = logprobs_ref[:, gen_start:, :]     # [B, gen_L, V]
    gen_probs_pol = probs_pol[:, gen_start:, :]           # [B, gen_L, V]
    gen_targets = targets[:, gen_start:]                  # [B, gen_L]

    # sequence logprob (only for actually generated tokens)
    token_logprob_pol = gen_logprobs_pol.gather(
        2, gen_targets.unsqueeze(-1)
    ).squeeze(-1)                                         # [B, gen_L]
    seq_logprob = token_logprob_pol.sum(dim=-1)           # [B]

    # KL divergence = sum p * (log p - log q)
    kl_per_token = torch.sum(
        gen_probs_pol * (gen_logprobs_pol - gen_logprobs_ref),
        dim=-1,
    )                                                     # [B, gen_L]
    kl_seq = kl_per_token.sum(dim=-1)                     # [B]

    return {"seq_logprob": seq_logprob, "kl_seq": kl_seq}

# 6. Optimizer & Scheduler
def build_optimizer_and_scheduler(policy_model, train_loader_len: int):
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    num_update_steps_per_epoch = math.ceil(train_loader_len / GRAD_ACCUM_STEPS)
    max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(WARMUP_RATIO * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    print("Max train steps:", max_train_steps, "| Warmup steps:", num_warmup_steps)
    return optimizer, scheduler

# 7. evlaute val set
def evaluate_on_val(
    val_loader,
    tokenizer,
    policy_model,
    evaluator,
    num_batches: int = 50,
) -> float:
    """
    在 val set 上跑一小部分 batch，估计一下平均 reward。
    为了省时间 & API 调用，默认只跑 num_batches 条。
    """
    policy_model.eval()
    all_rewards: List[float] = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            hs = batch["hate_speech"][0]
            gen_res = generate_responses_for_hs(
                hs,
                tokenizer=tokenizer,
                policy_model=policy_model,
                num_samples=1,
            )
            responses = gen_res["responses_text"]

            rewards = compute_sample_rewards(
                evaluator=evaluator,
                hate_speech_list=[hs],
                cn_list=responses,
            )
            all_rewards.append(float(rewards[0]))

    policy_model.train()
    if not all_rewards:
        return 0.0
    return float(np.mean(all_rewards))

# 8. training loop
tokenizer, policy_model, ref_model = load_models_and_tokenizer()
train_loader, val_loader = build_dataloaders()
evaluator = load_evaluator()
optimizer, lr_scheduler = build_optimizer_and_scheduler(
    policy_model, len(train_loader)
)

# sanity check on val reward
init_val_reward = evaluate_on_val(
    val_loader, tokenizer, policy_model, evaluator, num_batches=5
)
print(f"Initial val reward estimate (~5 samples): {init_val_reward:.4f}")

global_step = 0

for epoch in range(NUM_EPOCHS):
    policy_model.train()
    running_loss = 0.0

    print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, batch in pbar:
        hs = batch["hate_speech"][0]

        # 1. Sample and generate K CNs
        gen_res = generate_responses_for_hs(
            hs,
            tokenizer=tokenizer,
            policy_model=policy_model,
            num_samples=NUM_SAMPLES_PER_PROMPT,
        )
        responses = gen_res["responses_text"]      # len = K
        sequences_ids = gen_res["sequences_ids"]   # [K, L]
        prompt_len = gen_res["prompt_len"]

        # 2. evaluator computes K rewards
        rewards_np = compute_sample_rewards(
            evaluator=evaluator,
            hate_speech_list=[hs] * NUM_SAMPLES_PER_PROMPT,
            cn_list=responses,
        )
        rewards = torch.tensor(
            rewards_np, device=device, dtype=torch.float32
        )  # [K]

        # 3. calculate logprob & KL
        stats = compute_logprobs_and_kl(
            sequences_ids=sequences_ids,
            prompt_len=prompt_len,
            tokenizer=tokenizer,
            policy_model=policy_model,
            ref_model=ref_model,
        )
        seq_logprob = stats["seq_logprob"]  # [K]
        kl_seq = stats["kl_seq"]            # [K]

        # 4. calculate advantage & loss
        advantage = rewards - rewards.mean()
        pg_loss = -(advantage.detach() * seq_logprob).mean()
        kl_loss = KL_COEF * kl_seq.mean()
        loss = pg_loss + kl_loss

        # 5. Gradient accumulation
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        running_loss += loss.item() * GRAD_ACCUM_STEPS
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        pbar.set_description(
            f"Epoch {epoch+1} | loss={running_loss/(step+1):.4f} | "
            f"reward_mean={rewards.mean().item():.4f} | kl={kl_seq.mean().item():.4f}"
        )

    # 6. run a small portion of val at the end of each epoch
    val_reward = evaluate_on_val(
        val_loader, tokenizer, policy_model, evaluator, num_batches=50
    )
    print(f"[Epoch {epoch+1}] approx val reward (50 samples): {val_reward:.4f}")

    # 7. save epoch checkpoint
    epoch_dir = os.path.join(GRPO_OUTPUT_DIR, f"epoch_{epoch + 1}")
    os.makedirs(epoch_dir, exist_ok=True)
    policy_model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    print(f"Saved checkpoint to: {epoch_dir}")

# save grpo model
os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)

policy_model.save_pretrained(GRPO_OUTPUT_DIR)
tokenizer.save_pretrained(GRPO_OUTPUT_DIR)
print("GRPO model saved to:", GRPO_OUTPUT_DIR)
