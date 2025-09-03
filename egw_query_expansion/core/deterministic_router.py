"""
Deterministic Routing System based on Convex Optimization
Based on Abernethy et al. (2022) "Faster Projection-Free Online Learning"

Implements a deterministic routing system that guarantees unique convergence
through deterministic projections on convex polytopes, ensuring identical
inputs produce identical routes regardless of execution context.
"""

import hashlib
import json
import logging
import math
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from functools import lru_cache  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found

# # # from .import_safety import safe_import  # Module not found  # Module not found  # Module not found

# Safe imports with fallbacks
numpy_result = safe_import('numpy', attributes=['array', 'linalg'])
HAS_NUMPY = numpy_result.success

if HAS_NUMPY:
    np = numpy_result.module
else:
    # Mock numpy functionality for core deterministic logic
    class MockNumPy:
        @staticmethod
        def array(data):
            return list(data) if isinstance(data, (list, tuple)) else [data]

        @staticmethod
        def zeros_like(data):
            return [0.0] * len(data)

        @staticmethod
        def maximum(a, b):
            if isinstance(b, (int, float)):
                return [max(x, b) for x in a]
            return [max(x, y) for x, y in zip(a, b)]

        @staticmethod
        def clip(data, min_val, max_val):
            return [max(min_val, min(max_val, x)) for x in data]

        @staticmethod
        def argsort(data):
            return sorted(range(len(data)), key=lambda i: data[i], reverse=True)

        @staticmethod
        def where(condition):
            if isinstance(condition, list):
                return ([i for i, x in enumerate(condition) if x],)
            return ([],)

        @staticmethod
        def max(data):
            return max(data)

        @staticmethod
        def allclose(a, b, *args, **kwargs):
            if len(a) != len(b):
                return False
            return all(abs(x - y) < 1e-9 for x, y in zip(a, b))

        @staticmethod
        def all(data):
            return all(data)

        linalg = type(
            "linalg", (), {"norm": lambda x: math.sqrt(sum(xi**2 for xi in x))}
        )()

        random = type("random", (), {"seed": lambda x: None})()

    np = MockNumPy()


class ActionType(Enum):
    """Enumeration of possible routing actions"""

    ROUTE_TO_SPARSE = "route_to_sparse"
    ROUTE_TO_DENSE = "route_to_dense"
    ROUTE_TO_COLBERT = "route_to_colbert"
    APPLY_EGW_EXPANSION = "apply_egw_expansion"
    COMBINE_RESULTS = "combine_results"
    VALIDATE_OUTPUT = "validate_output"


@dataclass(frozen=True)
class ImmutableConfig:
    """Immutable configuration space with cryptographic hashing"""

    projection_tolerance: float = 1e-6
    max_iterations: int = 1000
    convergence_threshold: float = 1e-8
    lexicographic_tie_breaker: bool = True
    random_seed_salt: str = "egw_deterministic_v1"

    def __post_init__(self):
        """Generate and store cryptographic hash of configuration"""
        config_str = json.dumps(self.__dict__, sort_keys=True)
        object.__setattr__(
            self, "_config_hash", hashlib.sha256(config_str.encode()).hexdigest()
        )

    @property
    def config_hash(self) -> str:
        """Return cryptographic hash of configuration"""
        return getattr(self, "_config_hash")


@dataclass(frozen=True)
class RoutingContext:
    """Immutable context for routing decisions"""

    query_hash: str
    query_embedding: Tuple[float, ...]
    corpus_size: int
    retrieval_mode: str
    timestamp: Optional[str] = None

    @classmethod
    def from_query(
        cls,
        query: str,
        embedding: Union[List[float], Tuple[float, ...], Any],
        corpus_size: int,
        mode: str,
    ) -> "RoutingContext":
# # #         """Create routing context from query data"""  # Module not found  # Module not found  # Module not found
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        # Handle different embedding formats
        if HAS_NUMPY and hasattr(embedding, "flatten"):
            embedding_tuple = tuple(float(x) for x in embedding.flatten())
        elif isinstance(embedding, (list, tuple)):
            embedding_tuple = tuple(float(x) for x in embedding)
        else:
            # Fallback for mock objects or other types
            try:
                embedding_tuple = tuple(float(x) for x in embedding)
            except (TypeError, ValueError):
                # Create a simple hash-based embedding
                hash_val = int(
                    hashlib.sha256(str(embedding).encode()).hexdigest()[:8], 16
                )
                embedding_tuple = tuple(
                    float((hash_val >> (i * 4)) & 0xF) / 15.0 for i in range(8)
                )

        return cls(
            query_hash=query_hash,
            query_embedding=embedding_tuple,
            corpus_size=corpus_size,
            retrieval_mode=mode,
        )


@dataclass
class RoutingDecision:
    """Record of a single routing decision with full justification"""

    step_id: int
    context_hash: str
    action: ActionType
    projection_point: Tuple[float, ...]
    convergence_value: float
    tie_breaker_used: bool
    justification: str
    timestamp: str
    traceability_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "step_id": self.step_id,
            "context_hash": self.context_hash,
            "action": self.action.value,
            "projection_point": self.projection_point,
            "convergence_value": self.convergence_value,
            "tie_breaker_used": self.tie_breaker_used,
            "justification": self.justification,
            "timestamp": self.timestamp,
            "traceability_id": self.traceability_id,
        }


class ConvexProjector:
    """Implements deterministic projections on convex polytopes"""

    def __init__(self, config: ImmutableConfig):
        self.config = config

    def project_to_simplex(self, point: Union[List[float], Any]) -> List[float]:
        """Project point to probability simplex using deterministic algorithm"""
        # Convert to list if needed
        if not isinstance(point, list):
            if HAS_NUMPY and hasattr(point, "tolist"):
                point = point.tolist()
            else:
                point = list(point)

        n = len(point)
        sorted_indices = sorted(range(n), key=lambda i: point[i], reverse=True)

        # Find the threshold for projection
        cumsum = 0.0
        threshold = 0.0
        for i, idx in enumerate(sorted_indices):
            cumsum += point[idx]
            threshold = (cumsum - 1.0) / (i + 1)
            if i == n - 1 or point[sorted_indices[i + 1]] <= threshold:
                break

        # Apply projection
        projected = [max(point[i] - threshold, 0.0) for i in range(n)]

        # Normalize to ensure exact simplex constraint
        projected_sum = sum(projected)
        if projected_sum > 0:
            projected = [p / projected_sum for p in projected]

        return projected

    def project_to_box(
        self, point: Union[List[float], Any], bounds: Tuple[float, float]
    ) -> List[float]:
        """Project point to box constraints [a, b]^n"""
        if not isinstance(point, list):
            point = list(point)

        lower, upper = bounds
        return [max(lower, min(upper, x)) for x in point]

    def compute_convergence_metric(
        self, current: Union[List[float], Any], previous: Union[List[float], Any]
    ) -> float:
        """Compute convergence metric between iterations"""
        if not isinstance(current, list):
            current = list(current)
        if not isinstance(previous, list):
            previous = list(previous)

        # Compute L2 norm of difference
        diff_squared = [(c - p) ** 2 for c, p in zip(current, previous)]
        return math.sqrt(sum(diff_squared))


class SeedDerivation:
    """Deterministic seed derivation for routing decisions"""

    @staticmethod
    def derive_seed(
        context: RoutingContext, step_id: int, module_id: str, config_hash: str
    ) -> int:
# # #         """Derive deterministic seed from context and traceability info (backward compatible).  # Module not found  # Module not found  # Module not found
        Note: Retained for compatibility; prefers derive_step_seed when trace_id is known.
        """
        seed_string = f"{context.query_hash}:{step_id}:{module_id}:{config_hash}"
        seed_hash = hashlib.sha256(seed_string.encode()).hexdigest()
        return int(seed_hash[:8], 16) % (2**31)  # 32-bit signed integer

    @staticmethod
    def derive_step_seed(trace_id: str, step_id: int) -> int:
        """Derive seed exactly as H(trace_id||step_id) per RC contract."""
        seed_string = f"{trace_id}:{step_id}"
        seed_hash = hashlib.sha256(seed_string.encode()).hexdigest()
        return int(seed_hash[:8], 16) % (2**31)

    @staticmethod
    def generate_traceability_id(context: RoutingContext, step_id: int) -> str:
        """Generate unique traceability identifier"""
        trace_string = f"{context.query_hash}:{step_id}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(trace_string.encode()).hexdigest()[:16]


class LexicographicComparator:
    """Total order comparator with lexicographic tie breaking"""

    @staticmethod
    def compare_actions(
        action1: ActionType, action2: ActionType, context1_hash: str, context2_hash: str
    ) -> int:
        """Compare two actions with lexicographic tie breaking
        Returns: -1 if action1 < action2, 0 if equal, 1 if action1 > action2
        """
        # Primary comparison by action priority
        action_priority = {
            ActionType.ROUTE_TO_SPARSE: 0,
            ActionType.ROUTE_TO_DENSE: 1,
            ActionType.ROUTE_TO_COLBERT: 2,
            ActionType.APPLY_EGW_EXPANSION: 3,
            ActionType.COMBINE_RESULTS: 4,
            ActionType.VALIDATE_OUTPUT: 5,
        }

        priority1 = action_priority[action1]
        priority2 = action_priority[action2]

        if priority1 != priority2:
            return -1 if priority1 < priority2 else 1

        # Tie breaker using context hashes
        if context1_hash < context2_hash:
            return -1
        elif context1_hash > context2_hash:
            return 1
        else:
            return 0


class DeterministicRouter:
    """Main deterministic routing system implementing convex optimization"""

    def __init__(self, config: ImmutableConfig = None):
        self.config = config or ImmutableConfig()
        self.projector = ConvexProjector(self.config)
        self.comparator = LexicographicComparator()
        self.decisions: List[RoutingDecision] = []

        # Metadata with frozen hyperparameters hash for RC
        self.metadata: Dict[str, Any] = {"routing_hash": self.config.config_hash}

        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @lru_cache(maxsize=1024)
    def _compute_routing_weights(
        self, context_hash: str, corpus_size: int, mode: str
    ) -> Tuple[float, ...]:
        """Compute routing weights using cached deterministic computation"""
        # Use context hash as seed for deterministic computation
        seed = int(context_hash[:8], 16) % (2**31)

        # Initialize weights based on retrieval mode
        if mode == "sparse":
            base_weights = [0.7, 0.2, 0.1, 0.0]
        elif mode == "dense":
            base_weights = [0.2, 0.7, 0.1, 0.0]
        elif mode == "colbert":
            base_weights = [0.1, 0.2, 0.7, 0.0]
        else:  # hybrid
            base_weights = [0.3, 0.3, 0.3, 0.1]

        # Apply corpus size adjustment
        size_factor = min(1.0, corpus_size / 100000.0)
        base_weights[-1] = base_weights[-1] + (1 - size_factor) * 0.1

        # Project to simplex for valid probability distribution
        projected_weights = self.projector.project_to_simplex(base_weights)

        return tuple(float(w) for w in projected_weights)

    def _select_action_deterministic(
        self, weights: Union[List[float], Any], context: RoutingContext, step_id: int
    ) -> ActionType:
        """Select action deterministically based on weights and context"""
        if not isinstance(weights, list):
            weights = list(weights)

        actions = [
            ActionType.ROUTE_TO_SPARSE,
            ActionType.ROUTE_TO_DENSE,
            ActionType.ROUTE_TO_COLBERT,
            ActionType.APPLY_EGW_EXPANSION,
        ]

        # Find action with maximum weight
        max_weight = max(weights)
        max_indices = [i for i, w in enumerate(weights) if w == max_weight]

        if len(max_indices) == 1:
            return actions[max_indices[0]]

        # Tie breaker using lexicographic ordering
        tie_actions = [actions[i] for i in max_indices]
        tie_hashes = [f"{context.query_hash}:{i}" for i in max_indices]

        # Sort by lexicographic comparison
        sorted_pairs = sorted(
            zip(tie_actions, tie_hashes), key=lambda x: (x[0].value, x[1])
        )

        return sorted_pairs[0][0]

    def route_query(self, context: RoutingContext) -> List[RoutingDecision]:
        """
        Main routing function that maps context to ordered action sequence

        Args:
            context: Immutable routing context

        Returns:
            List of routing decisions with full traceability
        """
        decisions = []
        current_step = 0

        # Compute initial routing weights
        weights_tuple = self._compute_routing_weights(
            context.query_hash, context.corpus_size, context.retrieval_mode
        )
        weights = list(weights_tuple)

        # Iterative refinement using convex optimization
        previous_weights = [0.0] * len(weights)

        while (
            self.projector.compute_convergence_metric(weights, previous_weights)
            > self.config.convergence_threshold
            and current_step < self.config.max_iterations
        ):
            previous_weights = weights[:]

            # Generate traceability ID for this step
            trace_id = SeedDerivation.generate_traceability_id(context, current_step)

            # Select action deterministically
            action = self._select_action_deterministic(weights, context, current_step)

            # Create decision record
            decision = RoutingDecision(
                step_id=current_step,
                context_hash=context.query_hash,
                action=action,
                projection_point=tuple(float(w) for w in weights),
                convergence_value=self.projector.compute_convergence_metric(
                    weights, previous_weights
                ),
                tie_breaker_used=len(np.where(weights == np.max(weights))[0]) > 1,
                justification=f"Selected {action.value} with weight {max(weights):.6f}",
                timestamp=datetime.utcnow().isoformat(),
                traceability_id=trace_id,
            )

            decisions.append(decision)

            # Log decision
            self.logger.info(f"Step {current_step}: {decision.justification}")

            # Update weights using gradient step (simplified)
            gradient = self._compute_gradient(weights, context)
            weights = [
                w - 0.01 * g for w, g in zip(weights, gradient)
            ]  # Fixed step size

            # Project back to simplex
            weights = self.projector.project_to_simplex(weights)

            current_step += 1

        # Add final validation step
        final_decision = RoutingDecision(
            step_id=current_step,
            context_hash=context.query_hash,
            action=ActionType.VALIDATE_OUTPUT,
            projection_point=tuple(float(w) for w in weights),
            convergence_value=0.0,
            tie_breaker_used=False,
            justification="Final validation step",
            timestamp=datetime.utcnow().isoformat(),
            traceability_id=SeedDerivation.generate_traceability_id(
                context, current_step
            ),
        )
        decisions.append(final_decision)

        # Store decisions for later reconstruction
        self.decisions.extend(decisions)

        return decisions

    # ------------------- RC: Pure routing function ------------------- #
    def routing_fn(
        self, context: RoutingContext, steps: List[Dict[str, Any]]
    ) -> List[str]:
        """Pure function mapping (context, steps) -> ordered_step_ids.
        Deterministic: uses internal frozen config (routing_hash) and lexicographic tiebreak on
        (content_hash, step_id). Logs input digest, decision vector, and chosen route.
        Steps are a sequence of dict-like entries with at minimum:
          - 'step_id': stable identifier (stringifiable)
          - 'content' or 'content_hash': content used for hashing
        Returns list of step_id strings in chosen order.
        """
# # #         # Build decision vector deterministically from context  # Module not found  # Module not found  # Module not found
        weights = list(
            self._compute_routing_weights(
                context.query_hash, context.corpus_size, context.retrieval_mode
            )
        )
        decision_vector = {
            "weights": [float(w) for w in weights],
            "routing_hash": self.metadata["routing_hash"],
        }

        # Compute content_hash per step
        ordered_pre = []
        for idx, step in enumerate(steps):
            sid = str(step.get("step_id", str(idx)))
            if "content_hash" in step and step["content_hash"]:
                ch = str(step["content_hash"])
            else:
                content = step.get("content", "")
                ch = hashlib.sha256(
                    json.dumps(content, sort_keys=True, ensure_ascii=False).encode(
                        "utf-8"
                    )
                ).hexdigest()
            # Lexicographic tiebreak key: (-weight_rank, content_hash, step_id)
            # We map weight to a deterministic priority by index to avoid floating instability.
            # Distribute weights across steps by cycling deterministically so function stays pure.
            weight_idx = idx % max(1, len(weights))
            priority = -int(
                round(weights[weight_idx] * 1e9)
            )  # integerized for stability
            ordered_pre.append((priority, ch, sid))

        # Sort deterministically by tuple key
        ordered_keys = sorted(ordered_pre)
        ordered_step_ids = [sid for _, _, sid in ordered_keys]

        # Logging: input digest, decision vector, route
        inputs_digest = hashlib.sha256(
            json.dumps(
                {
                    "query_hash": context.query_hash,
                    "corpus_size": context.corpus_size,
                    "mode": context.retrieval_mode,
                    "steps": [(k[0], k[1], k[2]) for k in ordered_pre],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.logger.info(
            json.dumps(
                {
                    "routing_contract_log": True,
                    "routing_hash": self.metadata["routing_hash"],
                    "inputs_digest": inputs_digest,
                    "decision_vector": decision_vector,
                    "route": ordered_step_ids,
                },
                sort_keys=True,
            )
        )
        return ordered_step_ids

    def _compute_gradient(
        self, weights: List[float], context: RoutingContext
    ) -> List[float]:
        """Compute gradient for weight updates (simplified version)"""
        # Placeholder gradient computation - in practice would use actual loss
        embedding_norm = math.sqrt(sum(x**2 for x in context.query_embedding))
        gradient = [w - 0.25 for w in weights]  # Pull towards uniform distribution
        gradient[0] *= embedding_norm  # Scale by query characteristics
        return gradient

    def reconstruct_path(self, query_hash: str) -> List[Dict[str, Any]]:
# # #         """Reconstruct complete routing path from logs"""  # Module not found  # Module not found  # Module not found
        path_decisions = [d for d in self.decisions if d.context_hash == query_hash]

        return [
            decision.to_dict()
            for decision in sorted(path_decisions, key=lambda x: x.step_id)
        ]

    def verify_determinism(self, context: RoutingContext, num_trials: int = 5) -> bool:
        """Verify that identical inputs produce identical routes"""
        paths = []

        for _ in range(num_trials):
            # Clear previous decisions to avoid interference
            initial_decisions = len(self.decisions)

            decisions = self.route_query(context)
            action_sequence = [d.action.value for d in decisions]
            paths.append(tuple(action_sequence))

            # Reset decisions for next trial
            self.decisions = self.decisions[:initial_decisions]

        # Check all paths are identical
        return len(set(paths)) == 1

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about routing decisions"""
        if not self.decisions:
            return {}

        action_counts = {}
        for decision in self.decisions:
            action = decision.action.value
            action_counts[action] = action_counts.get(action, 0) + 1

        convergence_values = [d.convergence_value for d in self.decisions]
        avg_convergence = (
            sum(convergence_values) / len(convergence_values)
            if convergence_values
            else 0.0
        )

        return {
            "total_decisions": len(self.decisions),
            "action_distribution": action_counts,
            "average_convergence": avg_convergence,
            "tie_breaker_usage": sum(1 for d in self.decisions if d.tie_breaker_used),
        }


# Factory function for easy instantiation
def create_deterministic_router(config: Dict[str, Any] = None) -> DeterministicRouter:
    """Create a deterministic router with optional configuration"""
    if config is None:
        immutable_config = ImmutableConfig()
    else:
        immutable_config = ImmutableConfig(**config)

    return DeterministicRouter(immutable_config)
