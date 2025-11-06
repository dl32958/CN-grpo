import os
import re
import time
import json
import requests
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification
)
from sacrebleu.metrics import BLEU
from dotenv import load_dotenv


load_dotenv()

OVERALL_WEIGHTS = {
    "relevance": 0.25,
    "diversity": 0.20, 
    "length": 0.15,
    "toxicity": 0.10,
    "persuasiveness": 0.30
}
DIVERSITY_WEIGHTS = (0.4, 0.4, 0.2)  # distinct-1, distinct-2, self-BLEU
PERSUASIVENESS_WEIGHTS = (0.4, 0.3, 0.3) # stance, civility, answer quality

class CounterNarrativeEvaluator:
    def __init__(
        self, 
        relevance_model = "sentence-transformers/all-MiniLM-L6-v2",
        stance_model = "roberta-large-mnli",
        fluency_model = "gpt2-medium",
        cola_model = "textattack/roberta-base-CoLA"    # grammatical correctness
        ):

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! This evaluator requires CUDA to run.")
        
        self.device = "cuda"
        print(f"Using device: {self.device}")
        
        # load Perspective api key
        self.perspective_api_key = os.getenv("PERSPECTIVE_API_KEY", "")
        if not self.perspective_api_key:
            raise ValueError("PERSPECTIVE_API_KEY not found in environment variables.")
        print("Perspective API key loaded successfully.")
        
        # load models
        self._load_models(relevance_model, stance_model, fluency_model, cola_model)
        
        # regex for word tokenization
        self._word_re = re.compile(r"\b\w+\b", re.UNICODE)
        
    def _load_models(self, relevance_model, stance_model, fluency_model, cola_model):
        """
        Load all required models
        """
        
        print("Loading models...")
        
        # 1. Relevance model
        self.relevance_model = SentenceTransformer(relevance_model)
        
        # 2. Stance detection model (MNLI)
        self.stance_tokenizer = AutoTokenizer.from_pretrained(stance_model)
        self.stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model).to(self.device).eval()
        self._stance_label_map = {v.upper(): int(k) for k, v in self.stance_model.config.id2label.items()}
        self._contradiction_id = self._stance_label_map.get("CONTRADICTION", 0)
        self._neutral_id = self._stance_label_map.get("NEUTRAL", 1)
        
        # 3. Fluency model (GPT-2)
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_model)
        if self.fluency_tokenizer.pad_token is None:
            self.fluency_tokenizer.pad_token = self.fluency_tokenizer.eos_token
        self.fluency_model = AutoModelForCausalLM.from_pretrained(fluency_model)
        self.fluency_model.config.pad_token_id = self.fluency_tokenizer.pad_token_id
        self.fluency_model = self.fluency_model.to(self.device).eval()
        
        # 4. Grammatical acceptability (CoLA)
        self.cola_tokenizer = AutoTokenizer.from_pretrained(cola_model)
        self.cola_model = AutoModelForSequenceClassification.from_pretrained(cola_model).to(self.device).eval()
        
        print("All models loaded successfully!")

    def evaluate(self, hate_speech, counter_narratives, weights=None, batch_size=16) -> Dict[str, Any]:
        """
        Comprehensively evaluate counter-narratives

        Args:
            hate_speech: List of hate speech texts
            counter_narratives: List of counter-narrative texts  
            weights: Dimension evaluation weights
            batch_size: Batch size for processing

        Returns:
            Dictionary containing all evaluation metrics
        """
        if len(hate_speech) != len(counter_narratives):
            raise ValueError("hate_speech and counter_narratives must have the same length")
        
        if not weights:
            raise ValueError("Weights must be provided.")

        results = {}
        
        # 1. Relevance evaluation
        print("1. Computing relevance...")
        rel_scores, rel_mean = self._compute_relevance(hate_speech, counter_narratives, batch_size)
        results['relevance'] = {
            'scores': rel_scores,
            'mean': rel_mean
        }
        
        # 2. Diversity evaluation
        print("2. Computing diversity...")
        div_results = self._compute_diversity(counter_narratives, weights=DIVERSITY_WEIGHTS)
        results['diversity'] = div_results

        # 3. Toxicity evaluation
        print("3. Computing toxicity...")
        tox_raw, civility = self._compute_toxicity(counter_narratives)
        results['toxicity'] = {
            'raw_scores': tox_raw,
            'raw_mean': float(tox_raw.mean()),
            'civility_scores': civility,
            'civility_mean': float(civility.mean()),
            'safety_score': float(1.0 - tox_raw.mean())  # 1 - toxicity
        }
        
        # 4. Length Adherence evaluation
        print("4. Computing length...")
        len_score, word_lengths = self._compute_length(counter_narratives)
        results['length'] = {
            'score': len_score,
            'word_lengths': word_lengths
        }

        # 5. Persuasiveness evaluation
        print("5. Computing persuasiveness...")
        pers_results = self._compute_persuasiveness(hate_speech, counter_narratives, civility, batch_size, weights=PERSUASIVENESS_WEIGHTS)
        results['persuasiveness'] = pers_results
        
        # 6. Overall score computation
        print("6. Computing overall score...")
        overall_score = self._compute_overall_score(results, weights)
        results['overall'] = {
            'score': overall_score,
            'weights': weights
        }

        # 7. Generate summary
        summary = self._generate_summary(results)
        results['summary'] = summary
        
        print("Evaluation completed!")
        return results

    def _compute_relevance(self, hate_speech, counter_narratives, batch_size=64) -> Tuple[np.ndarray, float]:
        """compute relevance scores"""
        hs_list = [str(hs) if hs is not None else "" for hs in hate_speech]
        cn_list = [str(cn) if cn is not None else "" for cn in counter_narratives]
        
        emb_hs = self.relevance_model.encode(hs_list, convert_to_tensor=True, 
                                           normalize_embeddings=True, batch_size=batch_size)
        emb_cn = self.relevance_model.encode(cn_list, convert_to_tensor=True,
                                           normalize_embeddings=True, batch_size=batch_size)
        
        cos = util.cos_sim(emb_hs, emb_cn).diagonal().cpu().numpy()  # [-1, 1]
        rel_01 = (cos + 1.0) / 2.0  # [0, 1]
        return rel_01, float(rel_01.mean())

    def _compute_diversity(self, texts, sample_k=20, weights=DIVERSITY_WEIGHTS) -> Dict:
        """
        compute diversity metrics
        sample_k: Number of texts to randomly sample for Self-BLEU calculation
        """
        texts = [str(t) if t is not None else "" for t in texts]
        
        # Distinct-n
        d1 = self._distinct_n(texts, 1)
        d2 = self._distinct_n(texts, 2)
        
        # Self-BLEU
        sbleu4 = self._self_bleu_corpus(texts, n_gram=4, sample_k=min(sample_k, max(1, len(texts)-1)))   # 0-100
        sbleu4_norm = sbleu4 / 100.0   # 0-1
        
        # Diversity score
        s = 1.0 - sbleu4_norm  # lower BLEU -> higher diversity
        a, b, c = weights
        div_score = a * d1 + b * d2 + c * s
        
        return {
            "distinct1": d1,
            "distinct2": d2, 
            "self_bleu4": sbleu4,
            "self_bleu4_norm": sbleu4_norm,
            "div_score": div_score
        }
    
    def _distinct_n(self, texts, n=1) -> float:
        """compute distinct-n metric"""
        total, uniq = 0, set()
        for t in texts:
            tokens = self._word_re.findall(str(t).lower())
            if len(tokens) < n:
                continue
            grams = list(zip(*[tokens[i:] for i in range(n)]))
            total += len(grams)
            uniq.update(grams)
        return (len(uniq) / total) if total else 0.0

    def _self_bleu_corpus(self, texts, n_gram=4, sample_k=None, seed=42) -> float:
        """compute Self-BLEU score"""
        N = len(texts)
        if N <= 1:
            return 0.0
        
        rng = np.random.default_rng(seed)
        bleu = BLEU(max_ngram_order=n_gram, smooth_method="exp", effective_order=True)
        scores = []
        
        for i, hyp in enumerate(texts):
            if sample_k is None or sample_k >= N - 1:
                idxs = [j for j in range(N) if j != i]
            else:
                pool = [j for j in range(N) if j != i]
                idxs = rng.choice(pool, size=sample_k, replace=False).tolist()
            refs = [texts[j] for j in idxs]
            s = bleu.sentence_score(hyp, refs).score
            scores.append(s)
        return float(np.mean(scores))
    
    def _compute_toxicity(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """compute toxicity and civility scores using Perspective API"""
        if not self.perspective_api_key:
            raise ValueError("PERSPECTIVE_API_KEY not found in environment variables.")
            
        scorer = self._PerspectiveScorer(self.perspective_api_key)
        return scorer.score_list(texts)

    def _compute_length(self, texts, low=35, high=50) -> Tuple[float, np.ndarray]:
        """compute length adherence score, default 35-50 words"""
        lengths = np.fromiter((len(self._word_re.findall(str(t))) for t in texts), dtype=int)
        in_range = ((lengths >= low) & (lengths <= high)).astype(float)
        score = float(in_range.mean())
        return score, lengths

    def _compute_persuasiveness(self, hate_speech, counter_narratives, civility, batch_size=16, weights=PERSUASIVENESS_WEIGHTS) -> Dict:
        """compute persuasiveness-related metrics"""
        # stance opposition
        stance_scores = self._compute_stance_opposition(hate_speech, counter_narratives, batch_size)
        
        # answer quality
        answer_quality_scores = self._compute_answer_quality(counter_narratives, batch_size)
        
        # combined persuasiveness
        w_stance, w_civ, w_ans = weights
        persuasiveness = np.clip(
            w_stance * stance_scores + w_civ * civility + w_ans * answer_quality_scores,
            0.0, 1.0
        )
        
        return {
            "stance_scores": stance_scores,
            "stance_mean": float(stance_scores.mean()),
            "answer_quality_scores": answer_quality_scores,
            "answer_quality_mean": float(answer_quality_scores.mean()),
            "civility_mean": float(civility.mean()),
            "persuasiveness_scores": persuasiveness,
            "persuasiveness_mean": float(persuasiveness.mean()),
            "weights": {"stance": w_stance, "civility": w_civ, "answer_quality": w_ans}
        }
    
    @torch.inference_mode()
    def _compute_stance_opposition(self, hate_speech, counter_narratives, batch_size=16) -> np.ndarray:
        """use MNLI model to compute stance opposition scores"""
        hs_list = [str(hs) if hs is not None else "" for hs in hate_speech]
        cn_list = [str(cn) if cn is not None else "" for cn in counter_narratives]
        
        outs = []
        for i in range(0, len(hs_list), batch_size):
            hs_batch = hs_list[i:i+batch_size]
            cn_batch = cn_list[i:i+batch_size]
            
            enc = self.stance_tokenizer(hs_batch, cn_batch, padding=True, truncation=True,
                                      max_length=512, return_tensors="pt").to(self.device)
            
            prob = torch.softmax(self.stance_model(**enc).logits, dim=-1)
            score = prob[:, self._contradiction_id] + 0.5 * prob[:, self._neutral_id]
            outs.append(score.detach().cpu())
        
        return torch.cat(outs, dim=0).numpy()
    
    def _compute_answer_quality(self, texts, batch_size=8) -> np.ndarray:
        """compute answer quality (fluency + acceptability)"""
        texts = [str(t) if t is not None else "" for t in texts]
        
        # compute fluency
        fluency_scores = self._compute_fluency(texts, batch_size)
        
        # compute acceptability (coLA)
        acceptability_scores = self._compute_acceptability(texts, batch_size)
        
        # combined quality score
        quality_scores = 0.5 * fluency_scores + 0.5 * acceptability_scores
        return np.clip(quality_scores, 0.0, 1.0)
    
    @torch.inference_mode()
    def _compute_fluency(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """compute fluency scores using GPT-2 perplexity"""
        ppl_list = []
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.fluency_tokenizer(batch, return_tensors="pt", padding=True,
                                       truncation=True, max_length=192).to(self.device)
            
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            
            out = self.fluency_model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            mask = attn[:, 1:].contiguous().float()
            
            vocab = logits.size(-1)
            loss_tok = ce(logits.view(-1, vocab), labels.view(-1)).view(labels.size())
            loss_seq = (loss_tok * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            ppl = torch.exp(loss_seq).detach().cpu()
            ppl_list.append(ppl)
        
        ppl_array = torch.cat(ppl_list, dim=0).numpy()
        
        # convert to 0-1, lower ppl -> higher fluency
        log_ppl = np.log1p(ppl_array)
        lo, hi = np.percentile(log_ppl, [10, 90])
        if hi <= lo:
            hi = lo + 1e-6
        fluency = (hi - log_ppl) / (hi - lo)
        return np.clip(fluency, 0.0, 1.0)
    
    @torch.inference_mode()
    def _compute_acceptability(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """compute grammatical acceptability using CoLA model"""
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.cola_tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=128).to(self.device)
            prob = torch.softmax(self.cola_model(**enc).logits, dim=-1)[:, 1]  # label 1 = acceptable
            outs.append(prob.cpu())
        return torch.cat(outs, dim=0).numpy()
    
    def _compute_overall_score(self, results: Dict, weights: Dict[str, float]) -> float:
        """compute overall weighted score"""
        # normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # compute overall score
        relevance_score = results['relevance']['mean']
        diversity_score = results['diversity']['div_score']
        length_score = results['length']['score']
        toxicity_score = results['toxicity']['safety_score']  # use safe score (1 - toxicity)
        persuasiveness_score = results['persuasiveness']['persuasiveness_mean']
        
        # weighted sum
        overall = (
            normalized_weights["relevance"] * relevance_score +
            normalized_weights["diversity"] * diversity_score +
            normalized_weights["length"] * length_score +
            normalized_weights["toxicity"] * toxicity_score +
            normalized_weights["persuasiveness"] * persuasiveness_score
        )
        
        return float(overall)
    
    def _generate_summary(self, results: Dict) -> Dict:
        """generate evaluation summary"""
        return {
            "relevance_mean": results['relevance']['mean'],
            "distinct1": results['diversity']['distinct1'],
            "distinct2": results['diversity']['distinct2'],
            "self_bleu4": results['diversity']['self_bleu4'],
            "diversity_score": results['diversity']['div_score'],
            "toxicity_raw_mean": results['toxicity']['raw_mean'],
            "toxicity_safety_score": results['toxicity']['safety_score'],
            "length_score": results['length']['score'],
            "stance_mean": results['persuasiveness']['stance_mean'],
            "answer_quality_mean": results['persuasiveness']['answer_quality_mean'],
            "civility_mean": results['persuasiveness']['civility_mean'],
            "persuasiveness_mean": results['persuasiveness']['persuasiveness_mean'],
            "overall_score": results['overall']['score'],
            "weights": results['overall']['weights']
        }
    
    def print_summary(self, results: Dict):
        """print evaluation summary"""
        summary = results['summary']
        print("=" * 50)
        print("Counter-Narrative Evaluation Summary")
        print("=" * 50)
        print(f"Relevance Score         : {summary['relevance_mean']:.4f}")
        print(f"Distinct-1              : {summary['distinct1']:.4f}")
        print(f"Distinct-2              : {summary['distinct2']:.4f}")
        print(f"Self-BLEU-4             : {summary['self_bleu4']:.4f}")
        print(f"Diversity Score         : {summary['diversity_score']:.4f}")
        print(f"Toxicity Raw Mean       : {summary['toxicity_raw_mean']:.4f}")
        print(f"Toxicity Safety Score   : {summary['toxicity_safety_score']:.4f}")
        print(f"Length Score            : {summary['length_score']:.4f}")
        print(f"Stance Opposition       : {summary['stance_mean']:.4f}")
        print(f"Answer Quality          : {summary['answer_quality_mean']:.4f}")
        print(f"Civility                : {summary['civility_mean']:.4f}")
        print(f"Persuasiveness Score    : {summary['persuasiveness_mean']:.4f}")
        print("-" * 50)
        print(f"Overall Weighted Score  : {summary['overall_score']:.4f}")
        print(f"Weights: {summary['weights']}")
        print("=" * 50)
    
    class _PerspectiveScorer:
        """Perspective API Scorer"""
        def __init__(self, api_key, lang="en", qps=0.8, timeout=10.0):
            self.api_key = api_key
            self.lang = lang
            self.sleep = 1.0 / max(qps, 1e-6)
            self.timeout = timeout
            self.url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
            self.attrs = ["TOXICITY", "INSULT", "PROFANITY", "SEVERE_TOXICITY"]
        
        def score_list(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
            """batch scoring"""
            tox_raw, civ = [], []
            for t in texts:
                tr, cv = self._score_one(str(t) if t is not None else "")
                tox_raw.append(tr)
                civ.append(cv)
            return np.array(tox_raw, dtype=float), np.array(civ, dtype=float)
        
        def _score_one(self, text: str) -> Tuple[float, float]:
            """single text scoring"""
            payload = {
                "comment": {"text": text},
                "requestedAttributes": {a: {} for a in self.attrs},
                "languages": [self.lang],
                "doNotStore": True,
            }
            
            for attempt in range(2):
                try:
                    r = requests.post(self.url, json=payload, timeout=self.timeout)
                    r.raise_for_status()
                    j = r.json()["attributeScores"]
                    worst = max(j[a]["summaryScore"]["value"] for a in self.attrs)
                    time.sleep(self.sleep)
                    return float(worst), float(1.0 - worst)
                except requests.HTTPError as e:
                    code = getattr(e.response, "status_code", None)
                    if code in (429, 500, 502, 503) and attempt == 0:
                        time.sleep(2.0)
                        continue
                    return 0.0, 1.0
                except Exception:
                    return 0.0, 1.0