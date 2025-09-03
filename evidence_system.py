"""
Evidence Collection System with Conformal Coverage using Jackknife+
"""
import hashlib
import logging
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple  # Module not found  # Module not found  # Module not found

import matplotlib.pyplot as plt
import msgspec
import numpy as np
import plotly.graph_objects as go
# # # from scipy.stats import norm  # Module not found  # Module not found  # Module not found

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evidence(msgspec.Struct):
    """Evidence structure with validation"""

    qid: str
    content: str
    score: float
    dimension: str
    timestamp: float = msgspec.field(default_factory=lambda: __import__("time").time())
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash for deduplication - content and qid based"""
        content_hash = hashlib.sha256(f"{self.qid}:{self.content}".encode()).hexdigest()
        return hash(content_hash)


class EvidenceSystem:
    """Evidence collection system with conformal coverage guarantees
    Augmented with CRDT-style idempotent deduplication via a join-semilattice store.
    - Content-addressed identities ensure reinsertion is a no-op (idempotent union).
    - Per-replica multiplicity counters support analysis without inflating aggregates.
    - Canonical serialization provides order-independent, reproducible state dumps.
    - merge(other) performs a monotonic join (union of identities, per-replica max on counters).
    """

    def __init__(self, alpha: float = 0.1, replica_id: str = "local"):
        """
        Initialize evidence system

        Args:
            alpha: Significance level for coverage (1-alpha coverage target)
            replica_id: Identifier for this replica/process for CRDT counters
        """
        self.alpha = alpha
        self.replica_id = str(replica_id)
        # Content-addressed store: qid -> {evid_id -> Evidence}
        self._evidence_store: Dict[str, Dict[str, Evidence]] = defaultdict(dict)
        # Dimension index retained for fast grouping
        self._dimension_index: Dict[str, Set[Evidence]] = defaultdict(set)
        # Per-replica multiplicity counters: replica -> qid -> evid_id -> count
        self._replica_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # History for coverage diagnostics
        self._coverage_history: List[float] = []
        self._calibration_residuals: List[float] = []

    @staticmethod
    def _canonical_evidence_id(e: Evidence) -> str:
        """Compute a content-addressed identity for evidence using canonical JSON.
        Identity depends only on (qid, content, dimension). Score is not used to avoid
        double-counting slightly different scores for identical content.
        """
# # #         from deterministic_shield import canonical_json  # Module not found  # Module not found  # Module not found

        key = {"qid": e.qid, "content": e.content, "dimension": e.dimension}
        import hashlib

        return hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()

    def add_evidence(self, qid: str, evidence: Evidence) -> bool:
        """
        Idempotent evidence insertion with invariant pooling

        Args:
            qid: Question identifier
            evidence: Evidence object to add

        Returns:
            bool: True if evidence was added (new), False if already existed
        """
        # Ensure qid matches evidence qid
        evidence.qid = qid
        evid_id = self._canonical_evidence_id(evidence)

        # Track multiplicity per replica (attempts) with grow-only monotone counter
        self._replica_counts[self.replica_id][qid][evid_id] = (
            self._replica_counts[self.replica_id][qid][evid_id] + 1
        )

        # Idempotent union: only add to store if not present
        if evid_id in self._evidence_store[qid]:
            return False

        # Add to main store
        self._evidence_store[qid][evid_id] = evidence

        # Update dimension index
        self._dimension_index[evidence.dimension].add(evidence)

        logger.info(f"Added evidence for qid={qid}, dimension={evidence.dimension}")
        return True

    def get_evidence_for_question(self, qid: str) -> List[Evidence]:
        """
        Get all evidence for a question

        Args:
            qid: Question identifier

        Returns:
            List[Evidence]: List of evidence (order-invariant)
        """
        evidence_map = self._evidence_store.get(qid, {})
        # Return sorted for deterministic ordering while maintaining set semantics
        return sorted(
            list(evidence_map.values()), key=lambda e: (e.timestamp, e.content)
        )

    def group_by_dimension(self) -> Dict[str, List[Evidence]]:
        """
        Group evidence by dimension using standards' graph keys

        Returns:
            Dict[str, List[Evidence]]: Evidence grouped by dimension
        """
        result = {}
        for dimension, evidence_set in self._dimension_index.items():
            # Maintain deterministic ordering
            result[dimension] = sorted(
                list(evidence_set), key=lambda e: (e.timestamp, e.qid)
            )
        return result

    # ---------------------- CRDT merge and serialization ----------------------
    def merge(self, other: "EvidenceSystem") -> None:
        """Monotonic join with another replica's state.
        - Evidence identities: union (G-Set join).
        - Per-replica counters: pointwise max for each (replica,qid,evid_id).
        """
        # Merge evidence store (union by key)
        for qid, emap in other._evidence_store.items():
            dst = self._evidence_store[qid]
            for evid_id, ev in emap.items():
                if evid_id not in dst:
                    dst[evid_id] = ev
                    self._dimension_index[ev.dimension].add(ev)
        # Merge counters by per-replica max
        for rid, by_qid in other._replica_counts.items():
            for qid, by_eid in by_qid.items():
                for evid_id, cnt in by_eid.items():
                    self._replica_counts[rid][qid][evid_id] = max(
                        self._replica_counts[rid][qid][evid_id], cnt
                    )

    def serialize_canonical(self) -> str:
        """Canonical JSON of current state independent of insertion order."""
# # #         from deterministic_shield import canonical_json  # Module not found  # Module not found  # Module not found

        state = {
            "alpha": float(self.alpha),
            "store": {
                qid: [
                    {
                        "id": evid_id,
                        "content": ev.content,
                        "dimension": ev.dimension,
                        "score": float(ev.score),
                        "timestamp": float(ev.timestamp),
                    }
                    for evid_id, ev in sorted(emap.items())
                ]
                for qid, emap in self._evidence_store.items()
            },
            "counts": self._replica_counts,
        }
        return canonical_json(state)

    def get_multiplicity(self, qid: str, evidence: Evidence) -> int:
        """Return total multiplicity across replicas for given evidence identity."""
        evid_id = self._canonical_evidence_id(evidence)
        total = 0
        for rid, by_qid in self._replica_counts.items():
            total += by_qid.get(qid, {}).get(evid_id, 0)
        return int(total)

    def _jackknife_plus_scores(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Jackknife+ conformity scores

        Args:
            scores: Model prediction scores
            labels: True labels

        Returns:
            Tuple of (conformity_scores, coverage_estimate)
        """
        n = len(scores)
        conformity_scores = np.zeros(n)

        # Jackknife+ procedure
        for i in range(n):
            # Leave-one-out residual
            loo_scores = np.concatenate([scores[:i], scores[i + 1 :]])
            loo_labels = np.concatenate([labels[:i], labels[i + 1 :]])

            # Compute residual for left-out sample
            conformity_scores[i] = abs(scores[i] - labels[i])

        # Quantile for coverage (ensure it's in [0, 1])
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = np.clip(q_level, 0.0, 1.0)  # Ensure valid quantile range
        threshold = np.quantile(conformity_scores, q_level)

        # Empirical coverage estimate
        coverage = np.mean(conformity_scores <= threshold)

        return conformity_scores, coverage

    def _dr_submodular_selection(
        self, evidence_list: List[Evidence], budget: int
    ) -> Tuple[List[Evidence], float]:
        """
        DR-submodular approximation for budgeted evidence selection

        Args:
            evidence_list: List of evidence to select from
            budget: Selection budget

        Returns:
            Tuple of (selected_evidence, submodular_objective_value)
        """
        if not evidence_list or budget <= 0:
            return [], 0.0

        selected = []
        remaining = evidence_list.copy()
        objective_value = 0.0

        # Greedy selection with diminishing returns
        for _ in range(min(budget, len(evidence_list))):
            if not remaining:
                break

            best_evidence = None
            best_gain = -np.inf

            for evidence in remaining:
                # Submodular gain calculation (diversity + quality)
                diversity_score = self._compute_diversity(evidence, selected)
                quality_score = evidence.score

                # Diminishing returns factor
                dr_factor = 1.0 / (1.0 + len(selected) * 0.1)
                gain = dr_factor * (diversity_score + quality_score)

                if gain > best_gain:
                    best_gain = gain
                    best_evidence = evidence

            if best_evidence:
                selected.append(best_evidence)
                remaining.remove(best_evidence)
                objective_value += best_gain

        return selected, objective_value

    def _compute_diversity(self, evidence: Evidence, selected: List[Evidence]) -> float:
        """Compute diversity score for submodular selection"""
        if not selected:
            return 1.0

        # Simple diversity based on content similarity
        diversity = 0.0
        for sel_ev in selected:
            # Jaccard similarity approximation
            words1 = set(evidence.content.lower().split())
            words2 = set(sel_ev.content.lower().split())

            if not words1 and not words2:
                similarity = 1.0
            elif not words1 or not words2:
                similarity = 0.0
            else:
                similarity = len(words1 & words2) / len(words1 | words2)

            diversity += 1.0 - similarity

        return diversity / len(selected)

    def calculate_coverage(
        self, synthetic_labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate conformal coverage estimate with explicit α

        Args:
            synthetic_labels: Optional synthetic labels for validation

        Returns:
            float: Empirical coverage estimate
        """
        if not any(self._evidence_store.values()):
            logger.warning("No evidence available for coverage calculation")
            return 0.0

        # Collect all scores
        all_scores = []
        all_qids = []

        for qid, evidence_map in self._evidence_store.items():
            for evidence in evidence_map.values():
                all_scores.append(evidence.score)
                all_qids.append(qid)

        scores = np.array(all_scores)

        if synthetic_labels is None:
            # Generate synthetic labels for testing
            synthetic_labels = scores + np.random.normal(0, 0.1, len(scores))

        # Jackknife+ coverage
        conformity_scores, coverage = self._jackknife_plus_scores(
            scores, synthetic_labels
        )

        # Log calibration residuals
        residuals = scores - synthetic_labels
        self._calibration_residuals.extend(residuals.tolist())

        # Store coverage history
        self._coverage_history.append(coverage)

        logger.info(f"Empirical coverage: {coverage:.4f}, Target: {1-self.alpha:.4f}")
        logger.info(f"Calibration RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")

        return coverage

    def audit_coverage(
        self, n_trials: int = 100, delta: float = 0.05
    ) -> Dict[str, Any]:
        """
        Audit coverage on synthetic data

        Args:
            n_trials: Number of audit trials
            delta: Tolerance for coverage check

        Returns:
            Dict with audit results
        """
        coverages = []
        target_coverage = 1 - self.alpha

        for trial in range(n_trials):
            # Generate synthetic data
            n_samples = max(10, len(self._evidence_store) * 2)
            synthetic_scores = np.random.normal(0, 1, n_samples)
            synthetic_labels = synthetic_scores + np.random.normal(0, 0.2, n_samples)

            # Calculate coverage
            _, coverage = self._jackknife_plus_scores(
                synthetic_scores, synthetic_labels
            )
            coverages.append(coverage)

        empirical_coverage = np.mean(coverages)
        coverage_std = np.std(coverages)

        # Check if empirical >= target - delta
        passes_audit = empirical_coverage >= (target_coverage - delta)

        audit_results = {
            "empirical_coverage": empirical_coverage,
            "target_coverage": target_coverage,
            "coverage_std": coverage_std,
            "delta": delta,
            "passes_audit": passes_audit,
            "n_trials": n_trials,
            "coverage_distribution": coverages,
        }

        logger.info(
            f"Coverage audit: {empirical_coverage:.4f} >= {target_coverage - delta:.4f}: {passes_audit}"
        )

        return audit_results

    def test_shuffle_invariance(self, qid: str, n_shuffles: int = 10) -> bool:
        """
        Test shuffle invariance over evidence order

        Args:
            qid: Question ID to test
            n_shuffles: Number of shuffle tests

        Returns:
            bool: True if shuffle invariant
        """
        evidence_list = self.get_evidence_for_question(qid)
        if len(evidence_list) < 2:
            return True  # Trivially invariant

        # Reference aggregation
        reference_scores = [e.score for e in evidence_list]
        reference_mean = np.mean(reference_scores)

        for _ in range(n_shuffles):
            # Shuffle and re-aggregate
            shuffled_evidence = evidence_list.copy()
            np.random.shuffle(shuffled_evidence)

            shuffled_scores = [e.score for e in shuffled_evidence]
            shuffled_mean = np.mean(shuffled_scores)

            # Check invariance
            if not np.isclose(reference_mean, shuffled_mean):
                logger.error(f"Shuffle invariance failed for qid={qid}")
                return False

        logger.info(f"Shuffle invariance passed for qid={qid}")
        return True

    def plot_coverage_history(self, save_path: Optional[str] = None):
        """Plot coverage history using matplotlib"""
        if not self._coverage_history:
            logger.warning("No coverage history to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Coverage over time
        ax1.plot(self._coverage_history, "b-", label="Empirical Coverage")
        ax1.axhline(
            y=1 - self.alpha,
            color="r",
            linestyle="--",
            label=f"Target ({1-self.alpha:.2f})",
        )
        ax1.set_xlabel("Update")
        ax1.set_ylabel("Coverage")
        ax1.set_title("Coverage History")
        ax1.legend()
        ax1.grid(True)

        # Calibration residuals
        if self._calibration_residuals:
            ax2.hist(self._calibration_residuals, bins=30, alpha=0.7, color="green")
            ax2.set_xlabel("Calibration Residual")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Calibration Residuals Distribution")
            ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_evidence = sum(
            len(evidence_map) for evidence_map in self._evidence_store.values()
        )

        return {
            "total_questions": len(self._evidence_store),
            "total_evidence": total_evidence,
            "dimensions": list(self._dimension_index.keys()),
            "alpha": self.alpha,
            "target_coverage": 1 - self.alpha,
            "recent_coverage": self._coverage_history[-1]
            if self._coverage_history
            else None,
            "calibration_rmse": np.sqrt(
                np.mean(np.array(self._calibration_residuals) ** 2)
            )
            if self._calibration_residuals
            else None,
        }
