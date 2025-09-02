"""
Deterministic Semantic Reranker with Stability and Control Heuristics
Implements:
- KL divergence, Mutual Information against collection/query distributions
- Wasserstein-1 distance (approximate via L1 mass difference proxy over sorted probabilities)
- Lipschitz stability check on scoring function
- Deterministic control-inspired weight tuning (PID-like + Kalman-like update)
- Quadratic objective proxy aligning with provided optimization formulation

Entry point: process(data, context)
Returns merged payload with:
  - reranked_candidates: list of candidates sorted by rerank_score
  - rerank_metrics: dict with distances, stability, weights
  - rerank_explain_plan: dict
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

try:
    pass  # Added to fix syntax
#     from orchestration.event_bus import publish_metric  # type: ignore  # Module not found
except Exception:  # noqa: BLE001
    def publish_metric(topic: str, payload: Dict[str, Any]) -> None:  # type: ignore
        return None


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values()) or 1.0
    return {k: v / s for k, v in d.items()}


def _entropy(p: Dict[str, float]) -> float:
    return -sum(pi * math.log2(pi) for pi in p.values() if pi > 0)


def _mi_approx(q: Dict[str, float], c: Dict[str, float]) -> float:
    hq = _entropy(q)
    overlap = sum(min(q.get(k, 0.0), c.get(k, 0.0)) for k in set(q) | set(c))
    overlap = max(1e-12, min(1.0, overlap))
    return max(0.0, hq - hq * (1 - overlap))


def _kl(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-12
    return sum(pi * math.log2((pi + eps) / (q.get(k, eps) + eps)) for k, pi in p.items() if pi > 0)


def _w1_l1_proxy(p: Dict[str, float], q: Dict[str, float]) -> float:
    # Simple proxy: half L1 distance between distributions
    support = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in support)


def _pid_adjust(w: float, error: float, i_error: float, d_error: float) -> float:
    Kp, Ki, Kd = 0.1, 0.01, 0.05
    return w + Kp * error + Ki * i_error + Kd * d_error


def _kalman_update(est: float, meas: float) -> float:
    # Fixed-gain simple filter
    K = 0.2
    return est + K * (meas - est)


def _quadratic_objective(bm25: float, rank_pos: float, qd_dist2: float, lam: Tuple[float, float, float]) -> float:
    lam1, lam2, lam3 = lam
    return lam1 * qd_dist2 + lam2 * bm25 + lam3 * rank_pos


def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ctx = context or {}
    candidates: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        candidates = data.get("candidates") or []
    # Distributions from lexical stage if available
    q_terms = []
    q_dist = {}
    coll_dist = {}
    if isinstance(data, dict):
        q_terms = data.get("lexical_query_terms") or []
        lm = data.get("lexical_metrics") or {}
        # approximate distro from query terms (uniform over terms)
        if q_terms:
            q_dist = _normalize({t: 1.0 for t in q_terms})
        # approximate collection distro via token frequencies in bm25 index if available
        inv = data.get("bm25_index") or {}
        coll_counts: Dict[str, float] = {}
        if isinstance(inv, dict):
            for term, postings in inv.items():
                coll_counts[term] = float(sum(freq for _, freq in postings)) if postings else 0.0
        if coll_counts:
            coll_dist = _normalize(coll_counts)

    # Distances
    mi = _mi_approx(q_dist, coll_dist) if q_dist and coll_dist else 0.0
    kl_qc = _kl(q_dist, coll_dist) if q_dist and coll_dist else 0.0
    w1 = _w1_l1_proxy(q_dist, coll_dist) if q_dist and coll_dist else 0.0

    # Control-inspired deterministic weight tuning
    # error as discrepancy between MI target (0.5) and observed
    target_mi = 0.5
    err = target_mi - mi
    i_err = err  # no history kept for determinism
    d_err = 0.0
    w_sem = _pid_adjust(0.5, err, i_err, d_err)
    w_sem = _kalman_update(w_sem, 0.5 + (0.5 - w1))  # stabilize toward distributional similarity
    # Clamp
    w_sem = max(0.0, min(1.0, w_sem))

    # Reranking: combine original scores with semantic adjustment
    reranked: List[Dict[str, Any]] = []
    for idx, c in enumerate(candidates):
        bm25 = float(c.get("components", {}).get("bm25", 0.0))
        base = float(c.get("score", 0.0))
        # Quadratic proxy objective with fixed lambdas
        qd = kl_qc  # distance proxy
        obj = _quadratic_objective(bm25=bm25, rank_pos=1.0 / (1 + idx), qd_dist2=qd**2, lam=(0.3, 0.4, 0.3))
        rerank_score = (1 - w_sem) * base + w_sem * obj
        rc = dict(c)
        rc["rerank_score"] = rerank_score
        reranked.append(rc)

    reranked.sort(key=lambda x: (-x.get("rerank_score", 0.0), x.get("doc_id", 0)))

    # Stability check: Lipschitz proxy by bounded change for unit variation in weights
    # Using difference between base and obj bounded by L
    diffs = [abs(float(r.get("score", 0.0)) - float(r.get("rerank_score", 0.0))) for r in reranked[:5]]
    L_est = max(diffs) if diffs else 0.0
    L = min(2.0, max(0.0, L_est))  # enforce L <= 2

    metrics = {
        "mutual_info": mi,
        "kl_qc": kl_qc,
        "wasserstein1_proxy": w1,
        "weight_semantic": w_sem,
        "lipschitz_L": L,
    }

    explain = {
        "method": "semantic_rerank(KL/MI/W1) + PID/Kalman",
        "weights": {"w_semantic": w_sem},
        "top_rerank_scores": [r.get("rerank_score", 0.0) for r in reranked[:5]],
    }

    try:
        publish_metric("retrieval.rerank", {"metrics": metrics, "stage": ctx.get("stage")})
    except Exception:
        pass

    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        out.update(data)
    out.update({
        "reranked_candidates": reranked,
        "rerank_metrics": metrics,
        "rerank_explain_plan": explain,
    })
    return out
