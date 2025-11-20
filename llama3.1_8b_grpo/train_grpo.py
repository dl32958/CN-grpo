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
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
import argparse


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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="resume checkpoint path",
    )
    return parser.parse_args()

args = parse_args()
RESUME_DIR = args.resume_dir

sys.path.append(os.path.dirname(EVALUATOR_PATH))
from evaluator import (
    CounterNarrativeEvaluator,
    EVALUATION_WEIGHTS,
)

# Parameters
NUM_EPOCHS = 5
# effective_batch_size = batch_size * K * grad_accum
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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available! This code requires CUDA to run.")
device = torch.device("cuda")

# 1. Load tokenizer and model
def load_models_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy model
    policy_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    policy_model.gradient_checkpointing_enable()
    policy_model.train()

    # reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        dtype=torch.bfloat16,
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

# 3. ds & dataLoader
class HateSpeechDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.hates = df["HATE_SPEECH"].astype(str).tolist()
        self.gts = df["COUNTER_NARRATIVE"].astype(str).tolist()

    def __len__(self):
        return len(self.hates)

    def __getitem__(self, idx):
        return {
            "hate_speech": self.hates[idx],
            "ground_truth": self.gts[idx],
        }

def build_dataloaders():
    train_dataset = HateSpeechDataset(TRAIN_PATH)
    val_dataset = HateSpeechDataset(VAL_PATH)

    # train_dataset = Subset(HateSpeechDataset(TRAIN_PATH), list(range(100)))
    # val_dataset = Subset(HateSpeechDataset(VAL_PATH), list(range(20)))

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

# 4. Evaluator (CounterNarrativeEvaluator) & reward wrapper
def load_evaluator():
    evaluator = CounterNarrativeEvaluator()
    return evaluator

def compute_sample_rewards(evaluator: CounterNarrativeEvaluator, hate_speech_list, cn_list, gt_list) -> np.ndarray:
    """use evaluator's internal functions to calcuate each sample's reward (0-1)"""
    hs = [str(x) if x is not None else "" for x in hate_speech_list]
    cn = [str(x) if x is not None else "" for x in cn_list]
    gt = [str(x) if x is not None else "" for x in gt_list]

    if not (len(hs) == len(cn) == len(gt)):
        raise ValueError(
            f"Length mismatch in compute_sample_rewards: "
            f"hs={len(hs)}, cn={len(cn)}, gt={len(gt)}"
        )

    n = len(cn)

    # safety
    scorer = evaluator._PerspectiveScorer(evaluator.perspective_api_key)
    tox_raw = scorer.score_list(cn)  # [N]
    safety = 1.0 - tox_raw   # [N]

    # refutation: MNLI contradiction prob
    enc = evaluator.refutation_tokenizer(
        hs,
        cn,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(evaluator.device)

    with torch.inference_mode():
        prob = torch.softmax(evaluator.refutation_model(**enc).logits, dim=-1)
        refutation = prob[:, evaluator._contradiction_id].detach().cpu().numpy()  # [N]

    # align GT: SBERT + BertScore
    sbert_vec, _ = evaluator._compute_sbert_cos(cn, gt, batch_size=64)   # [N]
    bert_f1 = evaluator._compute_bertscore(cn, gt)  # [N]

    align_w = EVALUATION_WEIGHTS["sub_metrics"]["align_gt"]
    align_gt_scores = (
        align_w["sbert_cosine"] * sbert_vec +
        align_w["bertscore_f1"] * bert_f1
    )  # [N]

    # language quality
    lengths = np.fromiter(
        (len(evaluator._word_re.findall(t)) for t in cn),
        dtype=int,
    )
    wc_score = CounterNarrativeEvaluator.score_wordcount(
        lengths,
        full_lo=35,
        full_hi=50,
        left_tol=20,
        right_tol=20,
    )   # [N]

    ppl_log = evaluator._compute_fluency(cn, batch_size=8)
    ppl_log_clipped = np.maximum(ppl_log, 0.0)
    fluency_arr = 1.0 / (1.0 + ppl_log_clipped)  # [N]

    gramm_arr = evaluator._compute_cola(cn, batch_size=16)  # [N]

    lang_w = EVALUATION_WEIGHTS["sub_metrics"]["language"]
    language_scores = (
        lang_w["length_score"] * wc_score +
        lang_w["fluency_score"] * fluency_arr +
        lang_w["gramm_score"] * gramm_arr
    )  # [N]

    # final rewards
    cross_weights = EVALUATION_WEIGHTS["cross_category"]
    rewards = (
        cross_weights["safety"] * safety +
        cross_weights["refutation"] * refutation +
        cross_weights["align_gt"] * align_gt_scores +
        cross_weights["language"] * language_scores
    )

    rewards = np.clip(rewards, 0.0, 1.0).astype(np.float32)
    return rewards


# 5. helper fns (generate CN & calculate log probs + kl)
@torch.no_grad()
def generate_responses_for_hs(hate_speech, tokenizer, policy_model, num_samples = NUM_SAMPLES_PER_PROMPT):
    """
    For each HS, use policy_model to generate K CNs.
    Returns:
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
    input_ids = enc["input_ids"].to(device)  # [1, L]
    attention_mask = enc["attention_mask"].to(device)

    prompt_len = input_ids.shape[1]

    outputs = policy_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=GEN_TEMP,
        top_p=GEN_TOP_P,
        num_return_sequences=num_samples,          # control K
        pad_token_id=tokenizer.eos_token_id,
    )  # [K, L_total]

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


def compute_logprobs_and_kl(sequences_ids: torch.LongTensor, prompt_len, tokenizer, policy_model, ref_model) -> Dict[str, torch.Tensor]:
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

    # shift one token: logits[:, :-1] predicts targets = input_ids[:, 1:]
    logits_pol = logits_policy[:, :-1, :]
    logits_ref = logits_ref[:, :-1, :]
    targets = sequences_ids[:, 1:]

    logprobs_pol = torch.log_softmax(logits_pol, dim=-1)
    logprobs_ref = torch.log_softmax(logits_ref, dim=-1)
    probs_pol = torch.softmax(logits_pol, dim=-1)

    # generation part only
    gen_start = prompt_len - 1  # first generated token's logits index
    gen_logprobs_pol = logprobs_pol[:, gen_start:, :]     # [B, gen_L, V]
    gen_logprobs_ref = logprobs_ref[:, gen_start:, :]     # [B, gen_L, V]
    gen_probs_pol = probs_pol[:, gen_start:, :]           # [B, gen_L, V]
    gen_targets = targets[:, gen_start:]                  # [B, gen_L]

    # sequence logprob (policy)
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

# 6. optimizer & lr-scheduler
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
def evaluate_on_val(val_loader, tokenizer, policy_model, evaluator, num_batches = 50):
    """run small batch on val set to elimiate loss"""
    policy_model.eval()
    all_rewards = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            hs = batch["hate_speech"][0]
            gt = batch["ground_truth"][0]

            gen_res = generate_responses_for_hs(
                hs,
                tokenizer=tokenizer,
                policy_model=policy_model,
                num_samples=1,
            )
            responses = gen_res["responses_text"]  # len = 1

            rewards = compute_sample_rewards(
                evaluator=evaluator,
                hate_speech_list=[hs],
                cn_list=responses,
                gt_list=[gt],
            )
            all_rewards.append(float(rewards[0]))

    policy_model.train()
    if not all_rewards:
        return 0.0
    return float(np.mean(all_rewards))


# checkpoint
def save_training_state(save_dir, epoch, global_step, policy_model, optimizer, lr_scheduler):
    """
    Save the training state at the end of current epoch to save_dir/trainer_state.pt
    epoch: the epoch to start from next time (usually pass epoch+1 here)
    """
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": policy_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    torch.save(state, os.path.join(save_dir, "trainer_state.pt"))
    print(f"[Checkpoint] trainer_state saved to {save_dir}")


def load_training_state_if_any(resume_dir, policy_model, optimizer, lr_scheduler,):
    """
    If resume_dir is provided and trainer_state.pt exists in it, load the state;
    Otherwise return (start_epoch=0, global_step=0), indicating training from scratch
    """
    if not resume_dir:
        print("[Resume] No resume_dir provided, start from scratch.")
        return 0, 0

    state_path = os.path.join(resume_dir, "trainer_state.pt")
    if not os.path.exists(state_path):
        print(f"[Resume] {state_path} not found, start from scratch.")
        return 0, 0

    print(f"[Resume] Loading training state from {state_path}")
    ckpt = torch.load(
        state_path,
        map_location=device,
        weights_only=False,  # trust saved checkpoint
    )

    policy_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # rng = ckpt.get("rng_state", None)
    # if rng is not None:
    #     random.setstate(rng["python"])
    #     np.random.set_state(rng["numpy"])
    #     torch.set_rng_state(rng["torch"])
    #     torch.cuda.set_rng_state_all(rng["cuda"])

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)

    print(f"[Resume] Next epoch index = {start_epoch}, global_step = {global_step}")
    return start_epoch, global_step


# ============= Train Loop ============
tokenizer, policy_model, ref_model = load_models_and_tokenizer()
train_loader, val_loader = build_dataloaders()
evaluator = load_evaluator()
optimizer, lr_scheduler = build_optimizer_and_scheduler(
    policy_model, len(train_loader)
)

# try to resume from checkpoint (--resume_dir)
start_epoch, global_step = load_training_state_if_any(
    RESUME_DIR, policy_model, optimizer, lr_scheduler
)

# sanity check on val reward
init_val_reward = evaluate_on_val(
    val_loader, tokenizer, policy_model, evaluator, num_batches=5
)
print(f"Initial val reward estimate (~5 samples): {init_val_reward:.4f}")

for epoch in range(start_epoch, NUM_EPOCHS):
    policy_model.train()
    running_loss = 0.0

    print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, batch in pbar:
        hs = batch["hate_speech"][0]
        gt = batch["ground_truth"][0]

        # generate K CNs
        gen_res = generate_responses_for_hs(
            hs,
            tokenizer=tokenizer,
            policy_model=policy_model,
            num_samples=NUM_SAMPLES_PER_PROMPT,
        )
        responses = gen_res["responses_text"]   # len = K
        sequences_ids = gen_res["sequences_ids"]   # [K=4, L]
        prompt_len = gen_res["prompt_len"]

        # use evaluator computes K rewards
        rewards_np = compute_sample_rewards(
            evaluator=evaluator,
            hate_speech_list=[hs] * NUM_SAMPLES_PER_PROMPT,  # [hs, hs, hs, hs]
            cn_list=responses,
            gt_list=[gt] * NUM_SAMPLES_PER_PROMPT,
        )

        rewards = torch.tensor(
            rewards_np,
            device=device,
            dtype=torch.float32
        )    # numpy->tensor, [K]

        # cal logprob & KL
        stats = compute_logprobs_and_kl(
            sequences_ids=sequences_ids,
            prompt_len=prompt_len,
            tokenizer=tokenizer,
            policy_model=policy_model,
            ref_model=ref_model,
        )
        seq_logprob = stats["seq_logprob"]  # [K]
        kl_seq = stats["kl_seq"]  # [K]

        # cal advantage & loss = pg_loss + kl_loss
        advantage = rewards - rewards.mean()
        
        # 标准化
        advantage = advantage / (advantage.std() + 1e-8)
        # 裁剪极端值
        advantage = advantage.clamp(-5, 5)

        pg_loss = -(advantage.detach() * seq_logprob).mean()
        kl_loss = KL_COEF * kl_seq.mean()
        loss = pg_loss + kl_loss

        # gradient accumulation
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

    # val eval (small batch)
    val_reward = evaluate_on_val(
        val_loader, tokenizer, policy_model, evaluator, num_batches=50
    )
    print(f"[Epoch {epoch+1}] approx val reward (50 samples): {val_reward:.4f}")

    epoch_dir = os.path.join(GRPO_OUTPUT_DIR, f"epoch_{epoch + 1}")
    os.makedirs(epoch_dir, exist_ok=True)
    policy_model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    save_training_state(
        save_dir=epoch_dir,
        epoch=epoch + 1,
        global_step=global_step,
        policy_model=policy_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    print(f"Saved checkpoint to: {epoch_dir}")


# save grpo model
os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)

policy_model.save_pretrained(GRPO_OUTPUT_DIR)
tokenizer.save_pretrained(GRPO_OUTPUT_DIR)
print("GRPO model saved to:", GRPO_OUTPUT_DIR)
