import pandas as pd
import numpy as np
import json
import os
import yaml
from datetime import datetime

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
CN_PATH = config["evaluation"]["cn_path"]
OUTPUT_DIR = config["evaluation"]["output_dir"]
EVALUATOR_PATH = config["evaluation"]["evaluator_path"]

sys.path.append(EVALUATOR_PATH)
from evaluator import CounterNarrativeEvaluator, DEFAULT_WEIGHTS

df = pd.read_csv(CN_PATH)
if "GENERATED_CN" not in df.columns:
    raise ValueError("Dataset must contain 'GENERATED_CN' column for evaluation.")

hs = df['HATE_SPEECH'].fillna("").tolist()
gt = df['COUNTER_NARRATIVE'].fillna("").tolist()  # ground truth
cn = df['GENERATED_CN'].fillna("").tolist()

assert len(refs) == len(hyps), "The number of references and hypotheses must be the same."

evaluator = CounterNarrativeEvaluator()
results = evaluator.evaluate(
        hate_speech=hs,
        counter_narratives=cn,
        weights=OVERALL_WEIGHTS,
        batch_size=16
    )

evaluator.print_summary(results)

save_data = {
    "Relevance Score": summary['relevance_mean'],
    "Distinct-1": summary['distinct1'],
    "Distinct-2": summary['distinct2'],
    "Self-BLEU-4": summary['self_bleu4'],
    "Diversity Score": summary['diversity_score'],
    "Toxicity Raw Mean": summary['toxicity_raw_mean'],
    "Toxicity Safety Score": summary['toxicity_safety_score'],
    "Length Score": summary['length_score'],
    "Stance Opposition": summary['stance_mean'],
    "Answer Quality": summary['answer_quality_mean'],
    "Civility": summary['civility_mean'],
    "Persuasiveness Score": summary['persuasiveness_mean'],
    "Overall Weighted Score": summary['overall_score'],
    "Weights": summary['weights']
}

# save to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

save_path = os.path.join(OUTPUT_DIR, f"cn-eval-{timestamp}.jsonl")

with open(jsonl_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(save_data, ensure_ascii=False, indent=None) + '\n')

print("Evaluation summary successfully saved")