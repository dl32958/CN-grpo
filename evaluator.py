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
from bert_score import score as bertscore_score
from dotenv import load_dotenv

load_dotenv()


EVALUATION_WEIGHTS = {
    "cross_category": {
        "safety": 0.2,
        "refutation": 0.3,
        "align_gt": 0.3,
        "language": 0.2,
    },
    
    # submetric weights
    "sub_metrics": {
        "safety": {
            "safety_score": 1.0
        },
        "refutation": {
            "refutation_score": 1.0
        },
        "align_gt": {
            "sbert_cosine": 0.5,      # SBERT cosine
            "bertscore_f1": 0.5      # BERTScore F1
        },
        "language": {
            "length_score": 0.2,      # word-count
            "fluency_score": 0.4,           # fluency
            "gramm_score": 0.4           # grammaticality
        },
        "diversity": {
            "distinct2": 0.2,          # distinct2
            "self_bleu4": 0.4,         # self-BLEU-4
            "self_sbert": 0.4     # self-embedding cosine
        }
    },
    
    "set_level_alpha": 0.9  # 0.9-sample level, 0.1-set level
}


class CounterNarrativeEvaluator:
    def __init__(
        self, 
        refutation_model = "roberta-large-mnli",
        sbert_model = "sentence-transformers/all-MiniLM-L6-v2",
        bertscore_model = "roberta-large",
        fluency_model = "gpt2-medium",
        gramm_model = "textattack/roberta-base-CoLA"
        ):

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available! This evaluator requires CUDA to run.")
        
        self.device = "cuda"
        print(f"Using device: {self.device}")
        
        # perspective api key
        self.perspective_api_key = os.getenv("PERSPECTIVE_API_KEY", "")
        if not self.perspective_api_key:
            raise ValueError("PERSPECTIVE_API_KEY not found in environment variables.")
        print("Perspective API key loaded successfully.")
        
        # load models
        self._load_models(sbert_model, refutation_model, fluency_model, gramm_model, bertscore_model)
        
        # regex for word tokenization
        self._word_re = re.compile(r"\b\w+\b", re.UNICODE)

    def _load_models(self, sbert_model, refutation_model, fluency_model, gramm_model, bertscore_model):
        
        print("Loading models...")
        # refutation model - roberta-large-mnli
        self.refutation_tokenizer = AutoTokenizer.from_pretrained(refutation_model)
        self.refutation_model = AutoModelForSequenceClassification.from_pretrained(refutation_model).to(self.device).eval()
        self._refutation_label_map = {v.upper(): int(k) for k, v in self.refutation_model.config.id2label.items()}
        self._contradiction_id = self._refutation_label_map.get("CONTRADICTION", 0)
        self._neutral_id = self._refutation_label_map.get("NEUTRAL", 1)
        
        # SBERT model - all-MiniLM-L6-v2
        self.sbert_model = SentenceTransformer(sbert_model)

        # BertScore model - roberta-large
        self.bertscore_model = bertscore_model

        # fluency model - gpt2-medium
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_model)
        if self.fluency_tokenizer.pad_token is None:
            self.fluency_tokenizer.pad_token = self.fluency_tokenizer.eos_token
        self.fluency_model = AutoModelForCausalLM.from_pretrained(fluency_model)
        self.fluency_model.config.pad_token_id = self.fluency_tokenizer.pad_token_id
        self.fluency_model = self.fluency_model.to(self.device).eval()
        
        # grammaticality model - troberta-base-CoLA
        self.gramm_tokenizer = AutoTokenizer.from_pretrained(gramm_model)
        self.gramm_model = AutoModelForSequenceClassification.from_pretrained(gramm_model).to(self.device).eval()
        
        print("All models loaded successfully!")

    def evaluate(self, hate_speech, counter_narratives, ground_truth, batch_size=16) -> Dict[str, Any]:
        """
        Comprehensively evaluate counter-narratives

        Args:
            hate_speech: List of hate speech texts
            counter_narratives: List of counter-narrative texts  
            ground_truth: List of ground truth counter-narratives
            batch_size: Batch size for processing

        Returns:
            Dictionary containing all evaluation metrics
        """
        if len(hate_speech) != len(counter_narratives) or len(hate_speech) != len(ground_truth):
            raise ValueError("All input lists must have the same length")

        results = {}
        
        # 1. Safety evaluation
        print("1. Computing safety...")
        safety_results = self._compute_safety(counter_narratives)
        results['safety'] = safety_results

        # 2. Refutation evaluation
        print("2. Computing refutation...")
        refutation_results = self._compute_refutation(hate_speech, counter_narratives, batch_size)
        results['refutation'] = refutation_results
        
        # 3. Ground truth alignment evaluation
        print("3. Computing ground truth alignment...")
        align_results = self._compute_align_gt(counter_narratives, ground_truth, batch_size)
        results['align_gt'] = align_results

        # 4. Language quality evaluation
        print("4. Computing language quality...")
        language_results = self._compute_language(counter_narratives, batch_size)
        results['language'] = language_results

        # 5. Diversity evaluation
        print("5. Computing diversity...")
        diversity_results = self._compute_diversity(counter_narratives)
        results['diversity'] = diversity_results
        
        # 6. Overall score computation
        print("6. Computing overall score...")
        overall_score = self._compute_overall_score(results)
        results['overall'] = {
            'score': overall_score,
            'weights': EVALUATION_WEIGHTS["cross_category"]
        }

        # 7. Generate summary
        summary = self._generate_summary(results)
        results['summary'] = summary
        
        print("Evaluation completed!")
        return results

    def _compute_safety(self, texts: List[str]) -> Dict:
        """compute safety scores using Perspective API"""
        if not self.perspective_api_key:
            raise ValueError("PERSPECTIVE_API_KEY not found in environment variables.")

        scorer = self._PerspectiveScorer(self.perspective_api_key)
        tox_raw = scorer.score_list(texts)
        
        raw_mean = float(tox_raw.mean())
        safety_scores = 1.0 - tox_raw
        safety_score = float(safety_scores.mean())
        
        return {
            "safety_score": safety_score
        }
    
    @torch.inference_mode()
    def _compute_refutation(self, hate_speech, counter_narratives, batch_size=16) -> Dict:
        """compute refutation scores using MNLI model"""
        hs_list = [str(hs) if hs is not None else "" for hs in hate_speech]
        cn_list = [str(cn) if cn is not None else "" for cn in counter_narratives]
        
        outs = []
        for i in range(0, len(hs_list), batch_size):
            hs_batch = hs_list[i:i+batch_size]
            cn_batch = cn_list[i:i+batch_size]
            
            enc = self.refutation_tokenizer(hs_batch, cn_batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt").to(self.device)
            
            prob = torch.softmax(self.refutation_model(**enc).logits, dim=-1)
            score = prob[:, self._contradiction_id]
            outs.append(score.detach().cpu())
        
        stance_scores = torch.cat(outs, dim=0).numpy()
        refutation_score = float(stance_scores.mean())
        
        return {
            "refutation_score": refutation_score
        }

    def _compute_align_gt(self, counter_narratives, ground_truth, batch_size=64) -> Dict:
        """compute alignment with ground truth using embedding similarity and BertScore"""
        # SBERT cosine - cosine similarity between CN and GT in an SBERT embedding space
        emb_scores, emb_mean = self._compute_sbert_cos(counter_narratives, ground_truth, batch_size)
        
        # BertScore F1 - Token-level semantic similarity between CN and GT
        bertscore_scores = self._compute_bertscore(counter_narratives, ground_truth)
        bertscore_mean = float(bertscore_scores.mean())
        
        # Combined score
        weights = EVALUATION_WEIGHTS["sub_metrics"]["align_gt"]
        combined_score = (
            weights["sbert_cosine"] * emb_mean +
            weights["bertscore_f1"] * bertscore_mean
        )
        
        return {
            "sbert_cosine": emb_mean,
            "bertscore_f1": bertscore_mean,
            "align_gt_score": float(combined_score)
        }
    
    def _compute_language(self, texts, batch_size: int = 8) -> Dict[str, float]:
        """
        Language quality metrics (concise version):

        - length_score:   How well the text matches the desired length range (0~1; higher = closer)
        - fluency_score:  Fluency score normalized from ppl_log (0~1; higher = more fluent)
        - gramm_score:    Grammatical acceptability based on CoLA classifier (0~1; higher = better)
        - language_score: Weighted combination of the above three metrics (0~1; higher = better)
        """

        length_score, _ = self._compute_length(texts)

        ppl_log_arr = self._compute_fluency(texts, batch_size=batch_size)
        ppl_log_clipped = np.maximum(ppl_log_arr, 0.0)

        fluency_arr = 1.0 / (1.0 + ppl_log_clipped)
        fluency_score = float(fluency_arr.mean())

        gramm_arr = self._compute_cola(texts, batch_size=batch_size)
        gramm_score = float(gramm_arr.mean())

        weights = EVALUATION_WEIGHTS["sub_metrics"]["language"]
        language_score = (
            weights["length_score"] * length_score +
            weights["fluency_score"] * fluency_score +
            weights["gramm_score"] * gramm_score
        )

        return {
            "length_score": float(length_score),
            "fluency_score": float(fluency_score),
            "gramm_score": float(gramm_score),
            "language_score": float(language_score),
        }

    def _compute_diversity(self, texts, sample_k=20) -> Dict:
        """
        compute diversity metrics
        sample_k: Number of texts to randomly sample for Self-BLEU calculation
        """
        texts = [str(t) if t is not None else "" for t in texts]
        
        # distinct2
        d2 = self._distinct_n(texts, 2)
        
        # self-BLEU
        sbleu4 = self._self_bleu_corpus(texts, n_gram=4, sample_k=min(sample_k, max(1, len(texts)-1)))
        sbleu4_norm = sbleu4 / 100.0   # normalize to 0-1
        
        # self-embedding cosine
        self_emb_sim = self._compute_self_embedding_similarity(texts)
        
        # diversity score
        s_bleu = 1.0 - sbleu4_norm  # lower BLEU -> higher diversity
        s_emb = 1.0 - self_emb_sim  # lower similarity -> higher diversity
        
        weights = EVALUATION_WEIGHTS["sub_metrics"]["diversity"]
        div_score = (
            weights["distinct2"] * d2 +
            weights["self_bleu4"] * s_bleu +
            weights["self_sbert"] * s_emb
        )
        
        return {
            "distinct2": d2,
            "self_bleu4": sbleu4_norm,
            "self_sbert": self_emb_sim,
            "diversity_score": div_score
        }

    # ======= calculation details ========
    @staticmethod
    def score_wordcount(
        lengths: np.ndarray,
        full_lo: int,
        full_hi: int,
        left_tol: int,
        right_tol: int,
    ) -> np.ndarray:

        lo, hi = full_lo, full_hi
        lt, rt = left_tol, right_tol
        s = np.zeros_like(lengths, dtype=float)

        inside = (lengths >= lo) & (lengths <= hi)
        s[inside] = 1.0

        below = lengths < lo
        if lt > 0:
            s[below] = np.clip(1.0 - (lo - lengths[below]) / lt, 0.0, 1.0)

        above = lengths > hi
        if rt > 0:
            s[above] = np.clip(1.0 - (lengths[above] - hi) / rt, 0.0, 1.0)

        return s


    def _compute_sbert_cos(self, texts_a, texts_b, batch_size=64) -> Tuple[np.ndarray, float]:
        """compute SBERT cosine similarity between two text lists"""
        a_list = [str(t) if t is not None else "" for t in texts_a]
        b_list = [str(t) if t is not None else "" for t in texts_b]
        
        emb_a = self.sbert_model.encode(a_list, convert_to_tensor=True, 
                                       normalize_embeddings=True, batch_size=batch_size)
        emb_b = self.sbert_model.encode(b_list, convert_to_tensor=True,
                                       normalize_embeddings=True, batch_size=batch_size)
        
        cos = util.cos_sim(emb_a, emb_b).diagonal().cpu().numpy()  # [-1, 1]
        rel_01 = (cos + 1.0) / 2.0  # [0, 1]
        return rel_01, float(rel_01.mean())

    def _compute_bertscore(self, candidates, references) -> np.ndarray:
        """compute BertScore F1"""
        _, _, F1 = bertscore_score(
            candidates, references,
            model_type=self.bertscore_model,
            device=self.device,
            batch_size=32,
            rescale_with_baseline=True,
            lang="en",
            verbose=False
        )
        return F1.detach().cpu().numpy()

    def _compute_length(self, texts, full_lo = 35, full_hi = 50, left_tol = 20, right_tol = 20):
        """
        Compute word count + length adherence score (0~1)
        """
        lengths = np.fromiter(
            (len(self._word_re.findall(str(t))) for t in texts),
            dtype=int
        )

        wc_score = self.score_wordcount(
            lengths,
            full_lo=full_lo,
            full_hi=full_hi,
            left_tol=left_tol,
            right_tol=right_tol,
        )

        # higher = closer to desired length
        return float(wc_score.mean()), lengths

    @torch.inference_mode()
    def _compute_fluency(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """compute fluency scores using GPT-2 perplexity - return ppl_log values like notebook"""
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
        
        # return ppl_log
        ppl_log = np.log(np.clip(ppl_array, 1e-6, None))
        return ppl_log
    
    @torch.inference_mode()
    def _compute_cola(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """compute grammatical acceptability using CoLA model"""
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.gramm_tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=128).to(self.device)
            prob = torch.softmax(self.gramm_model(**enc).logits, dim=-1)[:, 1]  # label 1 = acceptable
            outs.append(prob.cpu())
        return torch.cat(outs, dim=0).numpy()

    def _compute_self_embedding_similarity(self, texts: List[str]) -> float:
        """compute mean pairwise SBERT embedding cosine similarity"""
        if len(texts) < 2:
            return 0.0
            
        embeddings = self.sbert_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        similarities = util.cos_sim(embeddings, embeddings)
        
        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(similarities, dtype=bool), diagonal=1)
        pairwise_sims = similarities[mask].cpu().numpy()
        
        return float((pairwise_sims.mean() + 1.0) / 2.0)   # [-1, 1] to [0, 1]

    def _distinct_n(self, texts, n=2) -> float:
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

    def _compute_overall_score(self, results: Dict) -> float:
        """compute overall weighted score"""
        # Get scores from each dimension
        safety_score = results['safety']['safety_score']
        refutation_score = results['refutation']['refutation_score']
        align_score = results['align_gt']['align_gt_score']
        language_score = results['language']['language_score']
        diversity_score = results['diversity']['diversity_score']
        
        cross_weights = EVALUATION_WEIGHTS["cross_category"]
        set_alpha = EVALUATION_WEIGHTS["set_level_alpha"]
        
        per_sample_score = (
            cross_weights["safety"] * safety_score +
            cross_weights["refutation"] * refutation_score +
            cross_weights["align_gt"] * align_score +
            cross_weights["language"] * language_score
        )
        
        overall = set_alpha * per_sample_score + (1 - set_alpha) * diversity_score
        
        return float(overall)
    
    def _generate_summary(self, results: Dict) -> Dict:
        """generate evaluation summary"""
        return {
            "safety_score": results['safety']['safety_score'],
            "refutation_score": results['refutation']['refutation_score'], 
            "sbert_cosine": results['align_gt']['sbert_cosine'],
            "bertscore_f1": results['align_gt']['bertscore_f1'],
            "length_score": results['language']['length_score'],
            "fluency_score": results['language']['fluency_score'],
            "gramm_score": results['language']['gramm_score'],
            "distinct2": results['diversity']['distinct2'],
            "self_bleu4": results['diversity']['self_bleu4'],
            "self_sbert": results['diversity']['self_sbert'],
            
            "safety_score": results['safety']['safety_score'],
            "refutation_score": results['refutation']['refutation_score'],
            "align_gt_score": results['align_gt']['align_gt_score'],
            "language_score": results['language']['language_score'],
            "diversity_score": results['diversity']['diversity_score'],
            
            "overall_score": results['overall']['score'],
            
            "weights": results['overall']['weights'],
            "set_level_alpha": EVALUATION_WEIGHTS["set_level_alpha"]
        }

    def print_summary(self, results: Dict):
        """print evaluation summary"""
        summary = results['summary']
        print("=" * 60)
        print("Counter-Narrative Evaluation Summary")
        print("=" * 60)
        
        # 10 Sub-metrics first
        print("Sub-metrics:")
        print(f"Safety Score: {summary['safety_score']:.4f}")
        print(f"Refutation Score: {summary['refutation_score']:.4f}")
        print(f"SBERT Cosine: {summary['sbert_cosine']:.4f}")
        print(f"BertScore F1: {summary['bertscore_f1']:.4f}")
        print(f"Length Score: {summary['length_score']:.4f}")
        print(f"Fluency Score: {summary['fluency_score']:.4f}")
        print(f"Grammaticality Score: {summary['gramm_score']:.4f}")
        print(f"Distinct-2: {summary['distinct2']:.4f}")
        print(f"Self-BLEU4: {summary['self_bleu4']:.4f}")
        print(f"Self-SBERT: {summary['self_sbert']:.4f}")
        
        print("-" * 60)
        
        # 5 Category scores second
        print("Category Scores:")
        print(f"Safety: {summary['safety_score']:.4f}")
        print(f"Refutation: {summary['refutation_score']:.4f}")
        print(f"Align GT: {summary['align_gt_score']:.4f}")
        print(f"Language: {summary['language_score']:.4f}")
        print(f"Diversity: {summary['diversity_score']:.4f}")
        
        print("-" * 60)
        
        # Overall score last
        print(f"Overall Score: {summary['overall_score']:.4f}")
        print("=" * 60)
    
    class _PerspectiveScorer:
        """Perspective API Scorer"""
        def __init__(self, api_key, lang="en", qps=0.8, timeout=10.0):
            self.api_key = api_key
            self.lang = lang
            self.sleep = 1.0 / max(qps, 1e-6)
            self.timeout = timeout
            self.url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
            self.attrs = ["TOXICITY", "INSULT", "PROFANITY", "SEVERE_TOXICITY"]
        
        def score_list(self, texts: List[str]) -> np.ndarray:
            """batch scoring"""
            tox_raw = []
            for t in texts:
                tox = self._score_one(str(t) if t is not None else "")
                tox_raw.append(tox)
            return np.array(tox_raw, dtype=float)
        
        def _score_one(self, text: str) -> float:
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
                    return float(worst)
                except requests.HTTPError as e:
                    code = getattr(e.response, "status_code", None)
                    if code in (429, 500, 502, 503) and attempt == 0:
                        time.sleep(2.0)
                        continue
                    return 0.0
                except Exception:
                    return 0.0