"""
AnswerSynthesizer: Turn evidence into structured answers with DNP-aware logic,
conformal confidence, and audit-ready formatting using only open-source tools.

Key methods (public API):
- synthesize_answer(question, evidence)
- apply_dnp_logic(answer, standards)
- calculate_confidence(evidence)
- format_response(answer)

Design notes:
- Uses open-source NLI/reranker when available (DeBERTa-v3-base-mnli via transformers);
  falls back to a lexical TF-IDF similarity if model weights are unavailable (keeps repo offline-safe).
- Confidence is computed via split conformal prediction (distribution-free) on NLI-style
  alignment scores between question and evidence.
- Supports a simple Conformal Risk Control (CRC) procedure for a monotone loss (false-negative
  on mandatory indicators) via threshold selection with a Hoeffding-style bound.
- StructuredResponse is a plain Python dict JSON-serializable with fields:
  {verdict, rationale, citations, confidence, risk_certificate} as specified.

This module is self-contained and does not require changes elsewhere. It optionally
accepts StructuredEvidence objects from evidence_processor.py but works with any evidence
objects that expose minimal fields (text, citation metadata).

Enhanced with deterministic hashing system for synthesis result consistency validation.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Optional heavy deps; we guard imports for graceful fallback
try:
    import torch
# # #     from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Module not found  # Module not found  # Module not found

    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

import numpy as np

# Import deterministic hashing system for synthesis validation
try:
    from egw_query_expansion.core.hash_policies import (
        DEFAULT_SYNTHESIS_HASHER,
        PipelineHashValidator
    )
    _HASH_VALIDATION_AVAILABLE = True
except ImportError:
    _HASH_VALIDATION_AVAILABLE = False
    DEFAULT_SYNTHESIS_HASHER = None
    PipelineHashValidator = None

# Optional sklearn components with graceful fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

    class TfidfVectorizer:  # minimal fallback
        def __init__(self, ngram_range=(1, 1), min_df=1, max_features=None, lowercase=True, stop_words=None):
            self.ngram_range = ngram_range
            self.min_df = min_df
            self.max_features = max_features
            self.lowercase = lowercase
            self.stop_words = set(stop_words) if stop_words else None
            self.vocabulary_ = {}
            self.idf_ = None

        def _tokenize(self, text: str):
            if self.lowercase:
                text = text.lower()
            tokens = text.split()
            if self.stop_words:
                tokens = [t for t in tokens if t not in self.stop_words]
            return tokens

        def fit(self, corpus):
# # #             from collections import Counter  # Module not found  # Module not found  # Module not found
            df = Counter()
            vocab = {}
            for doc in corpus:
                tokens = set(self._tokenize(doc))
                for t in tokens:
                    df[t] += 1
            # apply min_df
            items = [(t, c) for t, c in df.items() if c >= self.min_df]
            # limit features
            if self.max_features:
                items = sorted(items, key=lambda x: (-x[1], x[0]))[: self.max_features]
            vocab = {t: i for i, (t, _) in enumerate(sorted(items))}
            self.vocabulary_ = vocab
            # simple idf = 1.0 (no true idf to keep it lightweight)
            self.idf_ = np.ones(len(vocab), dtype=np.float32)
            return self

        def transform(self, corpus):
            X = np.zeros((len(corpus), len(self.vocabulary_)), dtype=np.float32)
            for i, doc in enumerate(corpus):
                tokens = self._tokenize(doc)
                for t in tokens:
                    j = self.vocabulary_.get(t)
                    if j is not None:
                        X[i, j] += 1.0
            # L2 normalize rows
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            return X / norms

        def fit_transform(self, corpus):
            return self.fit(corpus).transform(corpus)

    def cosine_similarity(A, B):  # minimal fallback
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_norm @ B_norm.T


@dataclass
class Premise:
    text: str
    evidence_id: Optional[str] = None
    citation: Optional[Dict[str, Any]] = None
    score: float = 0.0  # alignment score with question


@dataclass
class SynthesizedAnswer:
    question: str
    verdict: str  # "yes" | "no" | "unknown" or domain-specific
    rationale: str
    premises: List[Premise] = field(default_factory=list)
    unmet_conjuncts: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    confidence_interval: Optional[Tuple[float, float]] = None
    conformal_alpha: Optional[float] = None
    risk_certificate: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _synthesis_hash: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        """Compute synthesis hash after initialization if hash validation is available"""
        if _HASH_VALIDATION_AVAILABLE and DEFAULT_SYNTHESIS_HASHER:
            self._synthesis_hash = DEFAULT_SYNTHESIS_HASHER.hash_synthesized_answer(self)
    
    def get_synthesis_hash(self) -> Optional[str]:
        """Get the synthesis hash for this answer"""
        return self._synthesis_hash
    
    def verify_synthesis_integrity(self) -> bool:
        """Verify that the synthesis result hasn't been tampered with"""
        if not _HASH_VALIDATION_AVAILABLE or not DEFAULT_SYNTHESIS_HASHER:
            return True  # Can't verify, assume valid
        
        current_hash = DEFAULT_SYNTHESIS_HASHER.hash_synthesized_answer(self)
        return current_hash == self._synthesis_hash


class AnswerSynthesizer:
    """Synthesize answers from evidence with DNP-aware logic and conformal confidence.

    Conformal prediction approach:
    - We compute nonconformity scores from validation/calibration evidence (user may call
      fit_calibrator with (q, ev, label) tuples). We then compute a quantile of the scores
      at level 1 - alpha to produce an uncertainty radius and map to [0,1] confidence.
    - If no calibration is provided, we fallback to a heuristic mapping of alignment
      scores to confidence.

    CRC (Conformal Risk Control):
    - For a monotone loss like false-negative on mandatory indicators, we choose a
      threshold tau on alignment scores for a set to include an indicator as satisfied.
      Using Hoeffding's inequality, we bound risk with probability ≥ 1 - delta and
      pick the smallest tau satisfying bound ≤ alpha.
      
    Enhanced with deterministic hashing for synthesis result validation and pipeline consistency.
    """

    def __init__(
        self,
        nli_model_name: str = "microsoft/deberta-v3-base-mnli",
        device: Optional[str] = None,
        random_seed: int = 42,
        enable_hash_validation: bool = True,
    ) -> None:
        self.random = random.Random(random_seed)
        self._alpha_default = 0.1
        self._cal_scores: List[float] = []  # nonconformity scores on calibration set
        self._alpha: float = self._alpha_default
        self._rcps_config: Dict[str, Any] = {}
        
        # Hash validation components
        self.enable_hash_validation = enable_hash_validation and _HASH_VALIDATION_AVAILABLE
        if self.enable_hash_validation:
            self.synthesis_hasher = DEFAULT_SYNTHESIS_HASHER
            self.pipeline_validator = PipelineHashValidator()
        else:
            self.synthesis_hasher = None
            self.pipeline_validator = None

        # Lazy-load NLI model if available
        self._use_transformers = _TRANSFORMERS_AVAILABLE
        self._nli_model_name = nli_model_name
        self._device = device
        self._nli_model = None
        self._nli_tokenizer = None

        # Fallback TF-IDF
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._tfidf_fitted = False

    # ---------------------- Model loading helpers ----------------------
    def _ensure_nli(self) -> None:
        if not self._use_transformers or self._nli_model is not None:
            return
        try:
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self._nli_model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                self._nli_model_name
            )
            if self._device is None:
                self._device = (
                    "cuda"
                    if _TRANSFORMERS_AVAILABLE and torch.cuda.is_available()
                    else "cpu"
                )
            self._nli_model.to(self._device)
            self._nli_model.eval()
        except Exception:
            # Fallback to lexical approach if model can't be loaded
            self._use_transformers = False
            self._nli_model = None
            self._nli_tokenizer = None

    # ---------------------- Public API ----------------------
    def synthesize_answer(
        self, question: str, evidence: List[Any]
    ) -> SynthesizedAnswer:
# # #         """Compose claims from evidence with explicit premises and citations.  # Module not found  # Module not found  # Module not found

        Evidence items may be:
        - evidence_processor.StructuredEvidence
        - dict with keys {text, citation, evidence_id}
        - any object with attributes: chunk.text or text, citation, evidence_id
        """
        premises: List[Premise] = []
        citations: List[Dict[str, Any]] = []

        # Extract text and citations
        items: List[Tuple[str, Optional[str], Optional[Dict[str, Any]]]] = []
        for ev in evidence:
            text = None
            evidence_id = None
            citation = None
            # Try StructuredEvidence
            if hasattr(ev, "chunk") and hasattr(ev.chunk, "text"):
                text = getattr(ev.chunk, "text")
                evidence_id = getattr(ev, "evidence_id", None)
                # Build citation dict if present
                if hasattr(ev, "citation") and hasattr(ev.citation, "metadata"):
                    meta = ev.citation.metadata
                    citation = {
                        "document_id": getattr(meta, "document_id", None),
                        "title": getattr(meta, "title", None),
                        "author": getattr(meta, "author", None),
                        "page_number": getattr(meta, "page_number", None),
                        "inline": getattr(ev.citation, "inline_citation", None),
                    }
            # Try dict
            if text is None and isinstance(ev, dict):
                text = ev.get("text")
                evidence_id = ev.get("evidence_id")
                citation = ev.get("citation")
            # Try generic object with text
            if text is None and hasattr(ev, "text"):
                text = getattr(ev, "text")
                evidence_id = getattr(ev, "evidence_id", None)
                citation = getattr(ev, "citation", None)

            if text:
                items.append((text, evidence_id, citation))

        # Compute alignment scores of each premise with the question
        scores = self._align_scores(question, [t for t, _, _ in items])
        for (text, eid, cit), sc in zip(items, scores):
            premises.append(
                Premise(text=text, evidence_id=eid, citation=cit, score=float(sc))
            )
            if cit:
                citations.append(cit)

        # Simple verdict logic: if strong supporting premises average >= 0.6 -> yes
        avg_score = float(np.mean(scores)) if scores else 0.0
        verdict = (
            "yes" if avg_score >= 0.6 else ("no" if avg_score <= 0.3 else "unknown")
        )

        rationale = self._build_rationale(verdict, premises, question)

        ans = SynthesizedAnswer(
            question=question,
            verdict=verdict,
            rationale=rationale,
            premises=premises,
            citations=citations,
            metadata={
                "avg_alignment": avg_score,
                "premise_count": len(premises),
            },
        )
        
        # Validate synthesis result if hash validation is enabled
        if self.enable_hash_validation and self.pipeline_validator:
            self.pipeline_validator.validate_synthesis_consistency(
                "synthesize_answer", 
                ans
            )
        
        return ans

    def apply_dnp_logic(
        self, answer: SynthesizedAnswer, standards: Dict[str, Any]
    ) -> SynthesizedAnswer:
        """Apply DNP-aware rule templates; mark unmet conjuncts.

        standards example:
        {
          "mandatory_indicators": ["contains budget", "has baselines"],
          "rules": [
             {"id": "R1", "description": "Must mention year", "pattern": "202"},
             {"id": "R2", "description": "Includes % target", "any_of": ["%", "percent"]}
          ]
        }

        We mark unmet conjuncts against answer.premises and rationale text.
        """
        unmet: List[Dict[str, Any]] = []
        text_blob = (
            answer.rationale + "\n" + "\n".join(p.text for p in answer.premises)
        ).lower()

        rules = standards.get("rules", []) if isinstance(standards, dict) else []
        for rule in rules:
            rid = rule.get("id")
            desc = rule.get("description")
            pattern = rule.get("pattern")
            any_of = rule.get("any_of")
            all_of = rule.get("all_of")

            satisfied = True
            if pattern:
                satisfied = str(pattern).lower() in text_blob
            if any_of:
                any_ok = any(str(x).lower() in text_blob for x in any_of)
                satisfied = satisfied and any_ok if pattern else any_ok
            if all_of:
                all_ok = all(str(x).lower() in text_blob for x in all_of)
                satisfied = satisfied and all_ok if (pattern or any_of) else all_ok

            if not satisfied:
                unmet.append({"rule_id": rid, "description": desc})

        # Mandatory indicators monotone loss target (used by CRC later)
        mand = (
            standards.get("mandatory_indicators", [])
            if isinstance(standards, dict)
            else []
        )
        mand_satisfied = [m for m in mand if str(m).lower() in text_blob]
        mand_unmet = [m for m in mand if str(m).lower() not in text_blob]
        for m in mand_unmet:
            unmet.append(
                {
                    "rule_id": f"MAND::{m}",
                    "description": f"Mandatory indicator missing: {m}",
                }
            )

        answer.unmet_conjuncts = unmet
        # Adjust verdict if a mandatory indicator is unmet
        if mand_unmet and answer.verdict == "yes":
            answer.verdict = "unknown"  # conservative downgrade
            answer.rationale += (
                "\nNote: Verdict downgraded due to unmet mandatory indicators."
            )
        
        # Validate DNP logic result if hash validation is enabled
        if self.enable_hash_validation and self.pipeline_validator:
            # Update synthesis hash after DNP modifications
            if hasattr(answer, '_synthesis_hash') and DEFAULT_SYNTHESIS_HASHER:
                answer._synthesis_hash = DEFAULT_SYNTHESIS_HASHER.hash_synthesized_answer(answer)
            
            self.pipeline_validator.validate_synthesis_consistency(
                "apply_dnp_logic", 
                answer
            )
        
        return answer

    def calculate_confidence(
        self,
        evidence: List[Any],
        alpha: Optional[float] = None,
        apply_crc: bool = False,
        monotone_loss: str = "fnr_mandatory",
        standards: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
    ) -> Tuple[float, Tuple[float, float], Dict[str, Any]]:
# # #         """Compute conformal confidence from alignment scores.  # Module not found  # Module not found  # Module not found

        Returns (confidence_point, confidence_interval, rcps_metadata)
        - confidence_point in [0,1]
        - confidence_interval is a (lo, hi) 1-α marginal coverage interval for the mean alignment
        - rcps_metadata includes {alpha, method, quantile, crc_bound (if apply_crc)}
        """
        if alpha is None:
            alpha = self._alpha
        else:
            self._alpha = alpha

        # Build a proxy question if none provided
        q = question or "question"

        # Gather evidence texts
        texts = []
        for ev in evidence:
            if hasattr(ev, "chunk") and hasattr(ev.chunk, "text"):
                texts.append(getattr(ev.chunk, "text"))
            elif isinstance(ev, dict) and "text" in ev:
                texts.append(ev["text"])
            elif hasattr(ev, "text"):
                texts.append(getattr(ev, "text"))
        scores = self._align_scores(q, texts)
        mean_score = float(np.mean(scores)) if scores else 0.0

        # Split conformal: use calibration nonconformity scores if available
        # Nonconformity = 1 - alignment
        nonconfs = [1.0 - float(s) for s in scores]

        if self._cal_scores:
            all_scores = sorted(self._cal_scores)
            # quantile at ceil((n+1)*(1-alpha))/n in Jackknife+ spirit; for split we use simple quantile
            n = len(all_scores)
            k = max(0, min(n - 1, int(math.ceil((n + 1) * (1 - alpha)) - 1)))
            qhat = all_scores[k]
        else:
            # Heuristic quantile when no calibration present
            qhat = statistics.quantiles(nonconfs, n=10)[-1] if nonconfs else 1.0

        # Map to confidence: higher nonconformity threshold -> lower confidence
        conf_point = max(0.0, min(1.0, 1.0 - qhat))

        # Simple CI for mean alignment using conformalized interval (add qhat radius)
        lo = max(0.0, mean_score - qhat)
        hi = min(1.0, mean_score + qhat)

        rcps_meta: Dict[str, Any] = {
            "alpha": alpha,
            "method": "split-conformal",
            "quantile": qhat,
        }

        # Optional CRC for monotone loss (false negatives on mandatory indicators)
        if apply_crc and standards is not None:
            # We simulate a decision set by thresholding scores; choose tau to control FN rate
            mand = (
                standards.get("mandatory_indicators", [])
                if isinstance(standards, dict)
                else []
            )
            # Build binary indicators of found mandatory items
            text_blob = "\n".join(texts).lower()
            found = [1 if str(m).lower() in text_blob else 0 for m in mand]
            n = len(found)
            if n > 0:
                # For each candidate tau in [0,1], compute empirical FN rate proxy and Hoeffding upper bound
                taus = [i / 20.0 for i in range(0, 21)]
                # Proxy: treat all mand found if mean_score >= tau
                chosen_tau = 1.0
                best_tau = 1.0
                best_bound = 1.0
                delta = alpha  # use alpha as confidence for the bound
                for tau in taus:
                    # Empirical FN: fraction of mandatory not satisfied when decision says satisfied
                    # Here decision says satisfied if mean_score >= tau; else we abstain (not count as FN)
                    if mean_score >= tau:
                        emp_fn = float(sum(1 - x for x in found)) / n
                    else:
                        emp_fn = 0.0
                    # Hoeffding upper bound: emp + sqrt(log(1/delta)/(2n))
                    ub = emp_fn + math.sqrt(
                        max(0.0, math.log(1.0 / max(1e-12, delta))) / (2.0 * max(1, n))
                    )
                    if ub <= alpha and ub < best_bound:
                        best_bound = ub
                        best_tau = tau
                chosen_tau = best_tau
                rcps_meta.update(
                    {
                        "crc": {
                            "monotone_loss": monotone_loss,
                            "alpha": alpha,
                            "bound": best_bound,
                            "tau": chosen_tau,
                            "method": "CRC-Hoeffding",
                        }
                    }
                )

        # Log confidence calculation if hash validation is enabled
        if self.enable_hash_validation and self.pipeline_validator:
            confidence_data = {
                'confidence_point': conf_point,
                'confidence_interval': (lo, hi),
                'rcps_metadata': rcps_meta,
                'evidence_count': len(evidence),
                'mean_alignment': mean_score
            }
            self.pipeline_validator.validate_synthesis_consistency(
                "calculate_confidence",
                confidence_data
            )
        
        return conf_point, (lo, hi), rcps_meta

    def format_response(self, answer: SynthesizedAnswer) -> Dict[str, Any]:
        """Return JSON-serializable structured response with
        {verdict, rationale, citations, confidence, risk_certificate}.
        Enhanced with synthesis hash validation metadata.
        """
        formatted_response = {
            "verdict": answer.verdict,
            "rationale": answer.rationale,
            "citations": answer.citations,
            "confidence": answer.confidence,
            "risk_certificate": answer.risk_certificate
            or {
                "alpha": answer.conformal_alpha,
                "interval": answer.confidence_interval,
                "method": "split-conformal",
            },
            "premises": [
                {
                    "text": p.text,
                    "evidence_id": p.evidence_id,
                    "citation": p.citation,
                    "score": p.score,
                }
                for p in answer.premises
            ],
            "unmet_conjuncts": answer.unmet_conjuncts,
            "metadata": answer.metadata,
        }
        
        # Add hash validation metadata if enabled
        if self.enable_hash_validation and hasattr(answer, '_synthesis_hash'):
            formatted_response["_hash_validation"] = {
                "synthesis_hash": answer.get_synthesis_hash(),
                "integrity_verified": answer.verify_synthesis_integrity(),
                "validation_enabled": True
            }
            
            # Validate formatted response
            if self.pipeline_validator:
                self.pipeline_validator.validate_synthesis_consistency(
                    "format_response",
                    formatted_response
                )
        else:
            formatted_response["_hash_validation"] = {
                "synthesis_hash": None,
                "integrity_verified": None,
                "validation_enabled": False
            }
        
        return formatted_response

    # ---------------------- Calibration and utilities ----------------------
    def fit_calibrator(
        self,
        pairs: List[Tuple[str, str]],
        labels: Optional[List[int]] = None,
        alpha: float = 0.1,
    ) -> None:
        """Fit calibration nonconformity scores using split conformal.

        pairs: list of (question, evidence_text). labels are optional; if provided and binary,
        we can weight positive/negative differently. Nonconformity = 1 - alignment.
        """
        self._alpha = alpha
        qs = [q for q, _ in pairs]
        ts = [t for _, t in pairs]
        # Fit fallback vectorizer jointly to handle lexical scoring even if transformers used later
        try:
            self._tfidf.fit(qs + ts)
            self._tfidf_fitted = True
        except Exception:
            self._tfidf_fitted = False

        scores = []
        for q, t in pairs:
            s = float(self._align_scores(q, [t])[0])
            scores.append(1.0 - s)
        # Optionally adjust with labels (e.g., emphasize positives)
        if labels is not None and len(labels) == len(scores):
            wpos = 1.2
            wneg = 0.8
            self._cal_scores = []
            for sc, y in zip(scores, labels):
                reps = int(round((wpos if y == 1 else wneg) * 10))
                self._cal_scores.extend([sc] * max(1, reps))
        else:
            self._cal_scores = scores

    # ---------------------- Internal helpers ----------------------
    def _align_scores(self, question: str, texts: List[str]) -> List[float]:
        """Return alignment scores in [0,1] between question and each text.
        Prefers DeBERTa-v3-base-mnli if available; otherwise TF-IDF cosine mapping.
        """
        if not texts:
            return []

        # Try NLI-based approach; convert entailment probability to score
        if self._use_transformers:
            self._ensure_nli()
        if self._nli_model is not None and self._nli_tokenizer is not None:
            with torch.no_grad():
                # Batch encode premise-hypothesis pairs (text as premise, question as hypothesis)
                inputs = self._nli_tokenizer(
                    texts,
                    [question] * len(texts),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=256,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                logits = self._nli_model(**inputs).logits
                # DeBERTa MNLI label order is often [contradiction, neutral, entailment]
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                entail = probs[:, -1]
                # Map to [0,1]
                return [float(max(0.0, min(1.0, e))) for e in entail]

        # Fallback lexical similarity
        try:
            if not self._tfidf_fitted:
                self._tfidf.fit([question] + texts)
                self._tfidf_fitted = True
            vecs = self._tfidf.transform([question] + texts)
            qv = vecs[0]
            tv = vecs[1:]
            sims = cosine_similarity(qv, tv).flatten()
            # cosine [-] -> [0,1]
            sims = np.clip(sims, 0.0, 1.0)
            return [float(s) for s in sims]
        except Exception:
            # Final fallback: keyword overlap
            qset = set(question.lower().split())
            scores = []
            for t in texts:
                tset = set(t.lower().split())
                inter = len(qset & tset)
                union = len(qset | tset) or 1
                scores.append(inter / union)
            return scores
    
    def get_pipeline_validation_report(self) -> Optional[Dict[str, Any]]:
        """
        Get pipeline validation report from the synthesis validator
        
        Returns:
            Validation report if hash validation is enabled, None otherwise
        """
        if self.enable_hash_validation and self.pipeline_validator:
            return self.pipeline_validator.get_validation_report()
        return None
    
    def validate_synthesis_pipeline(self, synthesis_results: List[SynthesizedAnswer]) -> bool:
        """
        Validate hash consistency across a sequence of synthesis results
        
        Args:
            synthesis_results: List of synthesis results to validate
            
        Returns:
            True if all results maintain hash consistency, False otherwise
        """
        if not self.enable_hash_validation or not self.pipeline_validator:
            return True
        
        for i, result in enumerate(synthesis_results):
            stage_name = f"synthesis_pipeline_stage_{i}"
            validation_result = self.pipeline_validator.validate_synthesis_consistency(
                stage_name, 
                result
            )
            if not validation_result:
                return False
        
        return True

    def _build_rationale(
        self, verdict: str, premises: List[Premise], question: str
    ) -> str:
        parts = [f"Question: {question}", f"Verdict: {verdict.upper()}."]
        if premises:
            parts.append("Premises:")
            for i, p in enumerate(sorted(premises, key=lambda x: -x.score)[:5], 1):
                cite = (
                    f" [p.{p.citation.get('page_number')}]"
                    if p.citation and p.citation.get("page_number")
                    else ""
                )
                parts.append(f"  {i}. {p.text}{cite} (score={p.score:.2f})")
        else:
            parts.append("No supporting premises identified.")
        return "\n".join(parts)


# ---------------------- Convenience top-level functions ----------------------
def synthesize_answer(question: str, evidence: List[Any]) -> SynthesizedAnswer:
    return AnswerSynthesizer().synthesize_answer(question, evidence)


def apply_dnp_logic(
    answer: SynthesizedAnswer, standards: Dict[str, Any]
) -> SynthesizedAnswer:
    return AnswerSynthesizer().apply_dnp_logic(answer, standards)


def calculate_confidence(
    evidence: List[Any],
    alpha: float = 0.1,
    apply_crc: bool = False,
    monotone_loss: str = "fnr_mandatory",
    standards: Optional[Dict[str, Any]] = None,
    question: Optional[str] = None,
) -> float:
                """Return only the confidence point estimate as float for convenience.
    For full metadata, use the class method directly.
    """
    synth = AnswerSynthesizer()
    conf, _, _ = synth.calculate_confidence(
        evidence=evidence,
        alpha=alpha,
        apply_crc=apply_crc,
        monotone_loss=monotone_loss,
        standards=standards,
        question=question,
    )
    return float(conf)


def format_response(answer: SynthesizedAnswer) -> Dict[str, Any]:
    return AnswerSynthesizer().format_response(answer)


# ---------------------- Mini demo ----------------------
if __name__ == "__main__":
    # Minimal self-check without downloading models
    sample_ev = [
        {
            "text": "The plan includes a 20% target for vaccination coverage in 2025.",
            "evidence_id": "ev1",
            "citation": {"document_id": "docA", "page_number": 12},
        },
        {
            "text": "Baselines are established for 2023 with clear indicators.",
            "evidence_id": "ev2",
            "citation": {"document_id": "docA", "page_number": 13},
        },
    ]
    synth = AnswerSynthesizer()
    ans = synth.synthesize_answer(
        "Does the plan set measurable 2025 targets?", sample_ev
    )
    standards = {
        "mandatory_indicators": ["2025", "% target", "baselines"],
        "rules": [
            {"id": "R1", "description": "Mentions year 2025", "pattern": "2025"},
            {
                "id": "R2",
                "description": "Uses percentage target",
                "any_of": ["%", "percent"],
            },
            {"id": "R3", "description": "Has baselines", "pattern": "baseline"},
        ],
    }
    ans = synth.apply_dnp_logic(ans, standards)
    conf, ci, rcps = synth.calculate_confidence(
        sample_ev, alpha=0.1, apply_crc=True, standards=standards, question=ans.question
    )
    ans.confidence = conf
    ans.conformal_alpha = 0.1
    ans.confidence_interval = ci
    ans.risk_certificate = rcps
    out = synth.format_response(ans)
    print(json.dumps(out, ensure_ascii=False, indent=2))
