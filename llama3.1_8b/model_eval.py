import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import yaml
import os
import json

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
DATA_PATH = config["output_path"]

df = pd.read_csv(DATA_PATH)
if "GENERATED_COUNTER_SPEECH" not in df.columns:
    raise ValueError("Dataset must contain 'GENERATED_COUNTER_SPEECH' column for evaluation.")

# prepare data
refs = df['COUNTER_NARRATIVE'].tolist()
hyps = df['GENERATED_COUNTER_SPEECH'].tolist()

smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

bleu_scores = []
rougeL_scores = []

# Clean NaN values
refs = [str(ref) if pd.notna(ref) else "" for ref in refs]
hyps = [str(hyp) if pd.notna(hyp) else "" for hyp in hyps]

for ref, hyp in tqdm(zip(refs, hyps), total=len(refs), desc="Evaluating"):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    
    # BLEU
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
    bleu_scores.append(bleu)
    
    # ROUGE-L
    rouge = scorer.score(ref, hyp)
    rougeL_scores.append(rouge['rougeL'].fmeasure)

list_of_references = [[ref.split()] for ref in refs]
hypotheses = [hyp.split() for hyp in hyps]
corpus_bleu_score = corpus_bleu(list_of_references, hypotheses, smoothing_function=smooth)

results = {
    'BLEU_sentence': np.mean(bleu_scores),
    'BLEU_corpus': corpus_bleu_score,
    'ROUGE-L': np.mean(rougeL_scores),
    'avg_length': np.mean([len(hyp.split()) for hyp in hyps])
}

print("Evaluation Results:")
for k, v in results.items():
    print(f"{k:<12}: {v:.4f}")

# save results
os.makedirs("outputs", exist_ok=True)
with open("outputs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)
