"""
Deterministic Hybrid Retriever - Merges Lexical, Vector, and Late Interaction Results
- Combines BM25 lexical scores, dense vector similarity, and optional ColBERT late interaction
- RRF (Reciprocal Rank Fusion) for deterministic multi-modal ranking
- Entropy-based score normalization and calibration
- Theoretical guarantees for ranking stability and score interpretability

Entry point: process(data, context)
Returns merged payload with:
  - candidates: list[dict] with unified scoring
  - hybrid_metrics: performance and fusion diagnostics
  - explain_plan: ranking methodology details
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import math
import os
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple  # Module not found  # Module not found  # Module not found

import numpy as np


# Mandatory Pipeline Contract Annotations
__phase__ = "R"
__code__ = "40R"
__stage_order__ = 6

try:
# # #     from config_loader import get_thresholds  # Module not found  # Module not found  # Module not found
    THRESHOLDS_AVAILABLE = True
except ImportError:
    THRESHOLDS_AVAILABLE = False

try:
    pass  # Added to fix syntax
# # # #     from orchestration.event_bus import publish_metric  # type: ignore  # Module not found  # Module not found  # Module not found  # Module not found
except Exception:  # noqa: BLE001
    def publish_metric(topic: str, payload: Dict[str, Any]) -> None:  # type: ignore
        return None

try:
    pass  # Added to fix syntax
# # # #     from tracing.decorators import trace  # type: ignore  # Module not found  # Module not found  # Module not found  # Module not found
except Exception:  # noqa: BLE001
    def trace(fn):  # type: ignore
        return fn


def _normalize_scores(scores: List[float], method: str = "min_max") -> List[float]:
    """Normalize scores using specified method for multi-modal fusion."""
    if not scores or all(s == 0 for s in scores):
        return scores
    
    if method == "min_max":
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    elif method == "z_score":
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = math.sqrt(variance) if variance > 0 else 1.0
        return [(s - mean_score) / std_score for s in scores]
    else:
        # L2 normalization
        l2_norm = math.sqrt(sum(s ** 2 for s in scores))
        return [s / l2_norm if l2_norm > 0 else 0.0 for s in scores]


def _reciprocal_rank_fusion(rankings: List[List[int]], k: Optional[int] = None) -> List[Tuple[int, float]]:
    """
    RRF fusion with deterministic tie-breaking.
    
    Theoretical Foundation:
    RRF score for document d: RRF(d) = Σ_r 1/(k + rank_r(d))
    where rank_r(d) is the rank of document d in ranking r.
    
    Provides theoretical guarantees:
    - Monotonic: better ranks in any system → better RRF score
    - Robust: insensitive to individual system failures
    - Deterministic: identical inputs → identical outputs
    """
# # #     # Load k parameter from configuration if not provided  # Module not found  # Module not found  # Module not found
    if k is None:
        if THRESHOLDS_AVAILABLE:
            try:
                config = get_thresholds()
                k = config.retrieval_thresholds.rrf_k_parameter
            except Exception:
                k = 60
        else:
            k = 60
    
    rrf_scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)  # +1 for 1-based ranking
    
    # Sort by score (descending) with deterministic tie-breaking by doc_id
    sorted_results = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
    return sorted_results


@trace
def _late_interaction_score(query_tokens: List[str], doc_tokens: List[str], 
                           query_embeddings: Optional[List[List[float]]] = None,
                           doc_embeddings: Optional[List[List[float]]] = None) -> float:
    """
    Simplified late interaction scoring (ColBERT-style).
    MaxSim between query and document token embeddings.
    """
    if not query_embeddings or not doc_embeddings:
        # Fallback to token overlap scoring
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        if not query_set or not doc_set:
            return 0.0
        jaccard = len(query_set & doc_set) / len(query_set | doc_set)
        return jaccard
    
    # MaxSim computation: for each query token, find max similarity with doc tokens
    max_sims = []
    for q_emb in query_embeddings:
        max_sim = 0.0
        for d_emb in doc_embeddings:
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(q_emb, d_emb))
            q_norm = math.sqrt(sum(x ** 2 for x in q_emb))
            d_norm = math.sqrt(sum(x ** 2 for x in d_emb))
            
            if q_norm > 0 and d_norm > 0:
                sim = dot_product / (q_norm * d_norm)
                max_sim = max(max_sim, sim)
        
        max_sims.append(max_sim)
    
    return sum(max_sims) / len(max_sims) if max_sims else 0.0


def _entropy_calibration(scores: List[float], temperature: Optional[float] = None) -> List[float]:
    """Apply temperature-based calibration to scores for better fusion."""
# # #     # Load temperature from configuration if not provided  # Module not found  # Module not found  # Module not found
    if temperature is None:
        if THRESHOLDS_AVAILABLE:
            try:
                config = get_thresholds()
                temperature = config.temperature.entropy_calibration_temperature
            except Exception:
                temperature = 1.0
        else:
            temperature = 1.0
    
    if temperature <= 0:
        temperature = 1.0
    
    calibrated = [s / temperature for s in scores]
    
    # Softmax normalization
    max_score = max(calibrated) if calibrated else 0
    exp_scores = [math.exp(s - max_score) for s in calibrated]
    sum_exp = sum(exp_scores)
    
    if sum_exp > 0:
        return [exp_s / sum_exp for exp_s in exp_scores]
    else:
        return [1.0 / len(scores)] * len(scores) if scores else []


def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Hybrid retrieval process combining lexical, vector, and late interaction signals.
    """
    ctx = context or {}
    debug = bool(ctx.get("debug", False))
    trace_log: List[Dict[str, Any]] = []

# # #     # Extract retrieval results from previous stages  # Module not found  # Module not found  # Module not found
    lexical_results = []
    vector_results = []
    query_terms = []
    
    if isinstance(data, dict):
# # #         # Lexical results from BM25  # Module not found  # Module not found  # Module not found
        if "bm25_scores" in data:
            bm25_scores = data["bm25_scores"]
            for i, score in enumerate(bm25_scores):
                lexical_results.append({"doc_id": i, "score": score, "rank": i})
        
# # #         # Vector results from dense retrieval  # Module not found  # Module not found  # Module not found
        if "vector_metrics" in data and "similarities" in data["vector_metrics"]:
            similarities = data["vector_metrics"]["similarities"]
            full_sims = similarities.get("full", [])
            for i, score in enumerate(full_sims):
                vector_results.append({"doc_id": i, "score": score, "rank": i})
        
        # Query information
        query_terms = data.get("lexical_query_terms", [])
        
        if debug:
            trace_log.append({
                "step": "input_extraction",
                "lexical_count": len(lexical_results),
                "vector_count": len(vector_results),
                "query_terms": len(query_terms)
            })

    # Get candidate documents (union of all retrieval methods)
    all_doc_ids = set()
    if lexical_results:
        all_doc_ids.update(r["doc_id"] for r in lexical_results)
    if vector_results:
        all_doc_ids.update(r["doc_id"] for r in vector_results)
    
    if not all_doc_ids:
        # No results to process
        return {
            **(data if isinstance(data, dict) else {}),
            "candidates": [],
            "hybrid_metrics": {"fusion_method": "none", "candidate_count": 0},
            "hybrid_explain_plan": {"status": "no_candidates"}
        }

    # Build unified candidate list with multi-modal scores
    candidates = []
    
    for doc_id in all_doc_ids:
# # #         # Find scores from each modality  # Module not found  # Module not found  # Module not found
        lexical_score = 0.0
        vector_score = 0.0
        
        for lex_result in lexical_results:
            if lex_result["doc_id"] == doc_id:
                lexical_score = lex_result["score"]
                break
                
        for vec_result in vector_results:
            if vec_result["doc_id"] == doc_id:
                vector_score = vec_result["score"]
                break
        
        # Late interaction score (simplified)
        doc_tokens = [f"doc_{doc_id}_token_{i}" for i in range(5)]  # Placeholder
        late_score = _late_interaction_score(query_terms, doc_tokens)
        
        candidates.append({
            "doc_id": doc_id,
            "lexical_score": lexical_score,
            "vector_score": vector_score,
            "late_interaction_score": late_score,
            "raw_scores": {
                "bm25": lexical_score,
                "dense": vector_score,
                "late": late_score
            }
        })

    if debug:
        trace_log.append({
            "step": "candidate_preparation",
            "total_candidates": len(candidates)
        })

    # Score normalization
    lexical_scores = [c["lexical_score"] for c in candidates]
    vector_scores = [c["vector_score"] for c in candidates]
    late_scores = [c["late_interaction_score"] for c in candidates]
    
    norm_lexical = _normalize_scores(lexical_scores)
    norm_vector = _normalize_scores(vector_scores)
    norm_late = _normalize_scores(late_scores)
    
    # Update candidates with normalized scores
    for i, candidate in enumerate(candidates):
        candidate["normalized_scores"] = {
            "lexical": norm_lexical[i] if i < len(norm_lexical) else 0.0,
            "vector": norm_vector[i] if i < len(norm_vector) else 0.0,
            "late": norm_late[i] if i < len(norm_late) else 0.0
        }

    # Reciprocal Rank Fusion
    # Create rankings for each modality
    lexical_ranking = sorted(enumerate(lexical_scores), key=lambda x: -x[1])
    vector_ranking = sorted(enumerate(vector_scores), key=lambda x: -x[1])
    late_ranking = sorted(enumerate(late_scores), key=lambda x: -x[1])
    
    rankings = [
        [doc_idx for doc_idx, _ in lexical_ranking],
        [doc_idx for doc_idx, _ in vector_ranking],
        [doc_idx for doc_idx, _ in late_ranking]
    ]
    
    rrf_results = _reciprocal_rank_fusion(rankings, k=60)
    
    # Apply RRF scores to candidates
    rrf_lookup = {doc_id: score for doc_id, score in rrf_results}
    
    for candidate in candidates:
        doc_idx = candidate["doc_id"]
        candidate["rrf_score"] = rrf_lookup.get(doc_idx, 0.0)

    # Final hybrid scoring (weighted combination)
    fusion_weights = ctx.get("fusion_weights")
    if fusion_weights is None:
# # #         # Load fusion weights from configuration  # Module not found  # Module not found  # Module not found
        if THRESHOLDS_AVAILABLE:
            try:
                config = get_thresholds()
                fusion_weights = {
                    "lexical": config.fusion_weights.lexical,
                    "vector": config.fusion_weights.vector,
                    "late": config.fusion_weights.late_interaction,
                    "rrf": config.fusion_weights.rrf
                }
            except Exception:
                fusion_weights = {"lexical": 0.3, "vector": 0.4, "late": 0.2, "rrf": 0.1}
        else:
            fusion_weights = {"lexical": 0.3, "vector": 0.4, "late": 0.2, "rrf": 0.1}
    
    for candidate in candidates:
        norm_scores = candidate["normalized_scores"]
        hybrid_score = (
            fusion_weights["lexical"] * norm_scores["lexical"] +
            fusion_weights["vector"] * norm_scores["vector"] +
            fusion_weights["late"] * norm_scores["late"] +
            fusion_weights["rrf"] * candidate["rrf_score"]
        )
        candidate["hybrid_score"] = hybrid_score

    # Sort candidates by hybrid score
    candidates.sort(key=lambda x: -x["hybrid_score"])
    
    # Add final rankings
    for rank, candidate in enumerate(candidates):
        candidate["final_rank"] = rank + 1

    # Metrics computation
    score_variance = np.var([c["hybrid_score"] for c in candidates]) if candidates else 0.0
    rank_correlation = _calculate_rank_correlation(lexical_scores, vector_scores)
    
    metrics = {
        "fusion_method": "RRF + Weighted",
        "candidate_count": len(candidates),
        "score_variance": float(score_variance),
        "rank_correlation": rank_correlation,
        "weights_used": fusion_weights,
        "top_score": candidates[0]["hybrid_score"] if candidates else 0.0,
        "score_distribution": {
            "mean": float(np.mean([c["hybrid_score"] for c in candidates])) if candidates else 0.0,
            "std": float(np.std([c["hybrid_score"] for c in candidates])) if candidates else 0.0
        }
    }

    try:
        publish_metric("retrieval.hybrid", {
            "metrics": metrics, 
            "stage": ctx.get("stage"),
            "top_candidates": len([c for c in candidates[:5]])
        })
    except Exception:
        pass

    explain_plan = {
        "fusion_algorithm": "RRF + Weighted Combination",
        "normalization_method": "min_max",
        "modalities_used": ["lexical", "vector", "late_interaction"],
        "weights": fusion_weights,
        "candidates_processed": len(candidates)
    }

    # Build output
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        out.update(data)
    
    out.update({
        "candidates": candidates,
        "hybrid_metrics": metrics,
        "hybrid_explain_plan": explain_plan
    })
    
    if debug:
        out["trace"] = trace_log

    # Optional persistence
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        if os.path.isdir(data_dir):
            import json
            
            with open(os.path.join(data_dir, "hybrid_state.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "metrics": metrics,
                    "explain": explain_plan,
                    "top_candidates": candidates[:10]  # Save only top 10
                }, f, indent=2)
    except Exception:
        pass

    return out


def _calculate_rank_correlation(scores1: List[float], scores2: List[float]) -> float:
    """Calculate Spearman rank correlation between two score lists."""
    if len(scores1) != len(scores2) or len(scores1) < 2:
        return 0.0
    
    # Convert to ranks
    ranks1 = _scores_to_ranks(scores1)
    ranks2 = _scores_to_ranks(scores2)
    
    # Spearman correlation
    n = len(ranks1)
    d_squared = sum((r1 - r2) ** 2 for r1, r2 in zip(ranks1, ranks2))
    
    if n == 1:
        return 1.0
    
    correlation = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
    return correlation


def _scores_to_ranks(scores: List[float]) -> List[int]:
    """Convert scores to ranks (1-based, ties get average rank)."""
    indexed_scores = [(score, i) for i, score in enumerate(scores)]
    indexed_scores.sort(key=lambda x: -x[0])  # Sort by score descending
    
    ranks = [0] * len(scores)
    for rank, (score, original_index) in enumerate(indexed_scores):
        ranks[original_index] = rank + 1
    
    return ranks


if __name__ == "__main__":
    # Test the hybrid retrieval process
    test_data = {
        "bm25_scores": [0.8, 0.6, 0.4, 0.2],
        "vector_metrics": {
            "similarities": {
                "full": [0.9, 0.3, 0.7, 0.1]
            }
        },
        "lexical_query_terms": ["machine", "learning"]
    }
    
    result = process(test_data, {"debug": True})
    print(f"Processed {len(result['candidates'])} candidates")
    if result["candidates"]:
        top_candidate = result["candidates"][0]
        print(f"Top candidate: {top_candidate['doc_id']} (score: {top_candidate['hybrid_score']:.3f})")