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
from evaluator import CounterNarrativeEvaluator, OVERALL_WEIGHTS

df = pd.read_csv(CN_PATH)
if "GENERATED_CN" not in df.columns:
    raise ValueError("Dataset must contain 'GENERATED_CN' column for evaluation.")

hs = df['HATE_SPEECH'].fillna("").tolist()
gt = df['COUNTER_NARRATIVE'].fillna("").tolist()  # ground truth
cn = df['GENERATED_CN'].fillna("").tolist()

assert len(gt) == len(cn), "The number of ground truths and generated cn must be the same."

evaluator = CounterNarrativeEvaluator()
results = evaluator.evaluate(
        hate_speech=hs,
        counter_narratives=cn,
        weights=OVERALL_WEIGHTS,
        batch_size=16
    )

evaluator.print_summary(results)

save_data = {
    "Relevance Score": results['summary']['relevance_mean'],
    "Distinct-1": results['summary']['distinct1'],
    "Distinct-2": results['summary']['distinct2'],
    "Self-BLEU-4": results['summary']['self_bleu4'],
    "Diversity Score": results['summary']['diversity_score'],
    "Toxicity Raw Mean": results['summary']['toxicity_raw_mean'],
    "Toxicity Safety Score": results['summary']['toxicity_safety_score'],
    "Length Score": results['summary']['length_score'],
    "Stance Opposition": results['summary']['stance_mean'],
    "Answer Quality": results['summary']['answer_quality_mean'],
    "Civility": results['summary']['civility_mean'],
    "Persuasiveness Score": results['summary']['persuasiveness_mean'],
    "Overall Weighted Score": results['summary']['overall_score'],
    "Weights": results['summary']['weights']
}

# save to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(OUTPUT_DIR, exist_ok=True)

save_path = os.path.join(OUTPUT_DIR, f"cn-eval-{timestamp}.jsonl")

with open(save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(save_data, ensure_ascii=False, indent=None) + '\n')

print("Evaluation summary successfully saved")