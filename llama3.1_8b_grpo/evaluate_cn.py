import pandas as pd
import numpy as np
import json
import os
import yaml
from datetime import datetime
import sys

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
CN_PATH = config["evaluation"]["cn_path"]
OUTPUT_DIR = config["evaluation"]["output_dir"]
EVALUATOR_PATH = config["evaluation"]["evaluator_path"]

sys.path.append(os.path.dirname(EVALUATOR_PATH))
from evaluator import CounterNarrativeEvaluator, EVALUATION_WEIGHTS

df = pd.read_csv(CN_PATH)
if "GENERATED_CN" not in df.columns:
    raise ValueError("Dataset must contain 'GENERATED_CN' column for evaluation.")

hs = df['HATE_SPEECH'].fillna("").tolist()
gt = df['COUNTER_NARRATIVE'].fillna("").tolist()  # ground truth
cn = df['GENERATED_CN'].fillna("").tolist()

assert len(hs) == len(gt) == len(cn), "The number of hate speech, ground truths and generated cn must be the same."

evaluator = CounterNarrativeEvaluator()
results = evaluator.evaluate(
    hate_speech=hs,
    counter_narratives=cn,
    ground_truth=gt,
    batch_size=16
)

evaluator.print_summary(results)

save_data = {
    "Safety Score": results['summary']['safety_score'],
    "Refutation Score": results['summary']['refutation_score'],
    "SBERT Cosine": results['summary']['sbert_cosine'],
    "BertScore F1": results['summary']['bertscore_f1'],
    "Length Score": results['summary']['length_score'],
    "Fluency Score": results['summary']['fluency_score'],
    "Grammaticality Score": results['summary']['gramm_score'],
    "Distinct-2": results['summary']['distinct2'],
    "Self-BLEU4": results['summary']['self_bleu4'],
    "Self-SBERT": results['summary']['self_sbert'],
    
    "Safety": results['summary']['safety_score'],
    "Refutation": results['summary']['refutation_score'],
    "Align GT": results['summary']['align_gt_score'],
    "Language": results['summary']['language_score'],
    "Diversity": results['summary']['diversity_score'],
    
    "Overall Score": results['summary']['overall_score'],
    
    "Cross Category Weights": results['summary']['weights'],
    "Set Level Alpha": results['summary']['set_level_alpha']
}

# Save to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(OUTPUT_DIR, exist_ok=True)

save_path = os.path.join(OUTPUT_DIR, f"cn-eval-{timestamp}.jsonl")

with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(save_data, ensure_ascii=False, indent=None) + '\n')

print(f"Evaluation summary successfully saved to: {save_path}")