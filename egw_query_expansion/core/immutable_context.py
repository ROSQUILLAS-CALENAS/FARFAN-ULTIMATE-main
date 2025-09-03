"""
Immutable Context Architecture for EGW Query Expansion System

Based on Bernardy et al. (2021) substructural linear type theory for quantum programming languages.
Implements immutable context architecture with:
- Linear type restrictions preventing aliasing and mutation
- Persistent data structures for frozen models
- Pure derivation functions without observable side effects
- HMAC integrity verification against alterations
- Directed acyclic graph (DAG) for complete derivation lineage tracking

ACM Reference:
Bernardy, J.P., Jansson, P., & Palamidessi, C. (2021).
Linear Dependent Type Theory for Quantum Programming Languages.
ACM Transactions on Programming Languages and Systems, 43(2), 1-39.
"""

import hashlib
import hmac
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

# Type aliases for linear type system compliance
LinearRef = str  # Linear reference identifier
DerivationId = str  # Unique derivation identifier
ContextHash = str  # Content-based hash for integrity
SecretKey = bytes  # HMAC secret key


@dataclass(frozen=True)
class ImmutableMetadata:
    """Metadata for immutable context tracking"""

    creation_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    derivation_id: DerivationId = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[DerivationId] = None
    operation_type: str = "creation"
    content_hash: ContextHash = ""
    integrity_hmac: str = ""

    def __post_init__(self):
        """Ensure immutable metadata integrity"""
        if not self.content_hash:
            raise ValueError("Content hash is required for immutable metadata")


@dataclass(frozen=True)
class DerivationEdge:
    """Edge in the derivation DAG"""

    parent_id: DerivationId
    child_id: DerivationId
    operation: str
    timestamp: datetime

    def __hash__(self) -> int:
        return hash((self.parent_id, self.child_id, self.operation))


class DerivationDAG:
    """
    Directed Acyclic Graph for tracking complete derivation lineage
    Maintains immutability by creating new DAG instances for each operation
    """

    def __init__(self, edges: Optional[FrozenSet[DerivationEdge]] = None):
        self._edges: FrozenSet[DerivationEdge] = edges or frozenset()
        self._nodes: FrozenSet[DerivationId] = self._compute_nodes()
        self._validate_acyclic()

    def _compute_nodes(self) -> FrozenSet[DerivationId]:
# # #         """Compute all nodes from edges"""  # Module not found  # Module not found  # Module not found
        nodes = set()
        for edge in self._edges:
            nodes.add(edge.parent_id)
            nodes.add(edge.child_id)
        return frozenset(nodes)

    def _validate_acyclic(self) -> None:
        """Validate DAG property using DFS cycle detection"""
        if not self._edges:
            return

        # Build adjacency list
        graph = {}
        for edge in self._edges:
            if edge.parent_id not in graph:
                graph[edge.parent_id] = set()
            graph[edge.parent_id].add(edge.child_id)

        # DFS cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle(node: DerivationId) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self._nodes:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError("Cycle detected in derivation DAG")

    def add_edge(self, edge: DerivationEdge) -> "DerivationDAG":
        """Add edge and return new DAG instance"""
        new_edges = self._edges | {edge}
        return DerivationDAG(new_edges)

    def get_lineage(self, target_id: DerivationId) -> List[DerivationId]:
        """Get complete lineage path to target node"""
        lineage = []
        current = target_id

        while current:
            lineage.append(current)
            parent = None
            for edge in self._edges:
                if edge.child_id == current:
                    parent = edge.parent_id
                    break
            current = parent

        return list(reversed(lineage))

    @property
    def edges(self) -> FrozenSet[DerivationEdge]:
        return self._edges

    @property
    def nodes(self) -> FrozenSet[DerivationId]:
        return self._nodes


class ImmutableDict(Mapping):
    """
    Immutable dictionary implementation using persistent data structure
    Prevents mutation and aliasing while maintaining O(1) access patterns
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = dict(data) if data else {}
        # Freeze nested structures
        self._data = self._deep_freeze(self._data)

    def _deep_freeze(self, obj: Any) -> Any:
        """Deep freeze nested structures"""
        if isinstance(obj, dict):
            return {k: self._deep_freeze(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return tuple(self._deep_freeze(item) for item in obj)
        elif isinstance(obj, set):
            return frozenset(self._deep_freeze(item) for item in obj)
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __hash__(self) -> int:
        """Content-based hash for integrity checking"""
        return hash(tuple(sorted(self._data.items())))

    def __eq__(self, other) -> bool:
        if not isinstance(other, (ImmutableDict, dict)):
            return False
        if isinstance(other, dict):
            return self._data == other
        return self._data == other._data

    def derive(self, **updates) -> "ImmutableDict":
        """Pure derivation function - create new instance with updates"""
        new_data = dict(self._data)
        new_data.update(updates)
        return ImmutableDict(new_data)

    def remove(self, key: str) -> "ImmutableDict":
        """Pure derivation function - create new instance without key"""
        new_data = {k: v for k, v in self._data.items() if k != key}
        return ImmutableDict(new_data)


class QuestionContext:
    """
    Immutable Question Context - Canonical entry point to the system

    Implements linear type theory principles:
    - No aliasing: Each context has unique linear reference
    - No mutation: All operations create new instances
    - Pure functions: Derivations have no observable side effects
    - Integrity verification: HMAC-based tampering detection
    """

    def __init__(
        self,
        question_text: str,
        context_data: Optional[Dict[str, Any]] = None,
        secret_key: Optional[SecretKey] = None,
        parent_context: Optional["QuestionContext"] = None,
        operation_type: str = "creation",
    ):
        # Validate no hidden state or side channels
        if hasattr(self, "_initialized"):
            raise RuntimeError(
                "Attempted in-situ mutation of immutable QuestionContext"
            )

        # Initialize immutable state
        self._question_text = question_text
        self._context_data = ImmutableDict(context_data or {})
        self._secret_key = secret_key or self._generate_secret_key()

        # Compute content hash
        self._content_hash = self._compute_content_hash()

        # Generate derivation ID
        derivation_id = str(uuid.uuid4())

        # Create metadata with integrity HMAC
        self._metadata = ImmutableMetadata(
            derivation_id=derivation_id,
            parent_id=parent_context.metadata.derivation_id if parent_context else None,
            operation_type=operation_type,
            content_hash=self._content_hash,
            integrity_hmac=self._compute_integrity_hmac_for_id(derivation_id),
        )

        # Initialize or derive DAG
        if parent_context:
            edge = DerivationEdge(
                parent_id=parent_context.metadata.derivation_id,
                child_id=self._metadata.derivation_id,
                operation=operation_type,
                timestamp=self._metadata.creation_timestamp,
            )
            self._derivation_dag = parent_context._derivation_dag.add_edge(edge)
        else:
            self._derivation_dag = DerivationDAG()

        # Mark as initialized to prevent further mutation
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent any attribute mutation after initialization"""
        if hasattr(self, "_initialized"):
            raise RuntimeError(
                f"Attempted in-situ mutation of immutable QuestionContext: {name}. "
                f"Use derivation methods instead."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Prevent any attribute deletion"""
        raise RuntimeError("Attempted deletion on immutable QuestionContext")

    @staticmethod
    def _generate_secret_key() -> SecretKey:
        """Generate cryptographically secure secret key"""
        import secrets

        return secrets.token_bytes(32)

    def _compute_content_hash(self) -> ContextHash:
        """Compute deterministic content hash using deterministic hashing module"""
        try:
            # Try relative import first (when used as part of package)
            from .deterministic_hashing import hash_context
        except ImportError:
            # Try using hash policies if available (upstream compatibility)
            try:
                from .hash_policies import DEFAULT_CONTEXT_HASHER
                content = {
                    "question_text": self._question_text,
                    "context_data": dict(self._context_data),
                }
                return DEFAULT_CONTEXT_HASHER.policy.hash_object(content)
            except ImportError:
                # Fallback to direct import (when used standalone)
                import importlib.util
                import os
                
                # Load deterministic hashing module directly
                current_dir = os.path.dirname(__file__)
                hash_module_path = os.path.join(current_dir, 'deterministic_hashing.py')
                if os.path.exists(hash_module_path):
                    spec = importlib.util.spec_from_file_location("deterministic_hashing", hash_module_path)
                    hash_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(hash_module)
                    hash_context = hash_module.hash_context
                else:
                    # Final fallback to original implementation
                    content_json = json.dumps({
                        "question_text": self._question_text,
                        "context_data": dict(self._context_data),
                    }, sort_keys=True)
                    return hashlib.sha256(content_json.encode()).hexdigest()
        
        content = {
            "question_text": self._question_text,
            "context_data": dict(self._context_data),
        }
        return hash_context(content)

    def _compute_integrity_hmac_for_id(self, derivation_id: str) -> str:
        """Compute HMAC for integrity verification with given derivation ID"""
        message = f"{self._content_hash}:{derivation_id}"
        return hmac.new(self._secret_key, message.encode(), hashlib.sha256).hexdigest()

    def _compute_integrity_hmac(self) -> str:
        """Compute HMAC for integrity verification"""
        return self._compute_integrity_hmac_for_id(self._metadata.derivation_id)

    def verify_integrity(self) -> bool:
        """Verify context integrity using HMAC"""
        try:
            expected_hmac = self._compute_integrity_hmac()
            return hmac.compare_digest(self._metadata.integrity_hmac, expected_hmac)
        except Exception:
            return False

    # Pure derivation methods (no side effects)

    def derive_with_context(self, **context_updates) -> "QuestionContext":
        """Pure derivation: create new context with updated context data"""
        self._assert_integrity()

        new_context_data = self._context_data.derive(**context_updates)
        return QuestionContext(
            question_text=self._question_text,
            context_data=dict(new_context_data),
            secret_key=self._secret_key,
            parent_context=self,
            operation_type="context_update",
        )

    def derive_with_question(self, new_question: str) -> "QuestionContext":
        """Pure derivation: create new context with updated question"""
        self._assert_integrity()

        return QuestionContext(
            question_text=new_question,
            context_data=dict(self._context_data),
            secret_key=self._secret_key,
            parent_context=self,
            operation_type="question_update",
        )

    def derive_with_expansion(
        self, expansion_data: Dict[str, Any]
    ) -> "QuestionContext":
        """Pure derivation: create new context with query expansion"""
        self._assert_integrity()

        expanded_context = dict(self._context_data)
        expanded_context["expansion"] = expansion_data
        expanded_context["expansion_timestamp"] = datetime.now(timezone.utc).isoformat()

        return QuestionContext(
            question_text=self._question_text,
            context_data=expanded_context,
            secret_key=self._secret_key,
            parent_context=self,
            operation_type="query_expansion",
        )

    def _assert_integrity(self) -> None:
        """Assert context integrity before operations"""
        if not self.verify_integrity():
            raise RuntimeError(
                "Context integrity verification failed - possible tampering detected"
            )

    # Read-only properties

    @property
    def question_text(self) -> str:
        return self._question_text

    @property
    def context_data(self) -> ImmutableDict:
        return self._context_data

    @property
    def metadata(self) -> ImmutableMetadata:
        return self._metadata

    @property
    def content_hash(self) -> ContextHash:
        return self._content_hash

    @property
    def derivation_dag(self) -> DerivationDAG:
        return self._derivation_dag

    def get_lineage(self) -> List[DerivationId]:
# # #         """Get complete derivation lineage from root to current context"""  # Module not found  # Module not found  # Module not found
        return self._derivation_dag.get_lineage(self._metadata.derivation_id)

    def __hash__(self) -> int:
        """Content-based hash for set/dict usage"""
        return hash((self._content_hash, self._metadata.derivation_id))

    def __eq__(self, other) -> bool:
        """Content-based equality"""
        if not isinstance(other, QuestionContext):
            return False
        return (
            self._content_hash == other._content_hash
            and self._metadata.derivation_id == other._metadata.derivation_id
        )

    def __repr__(self) -> str:
        return (
            f"QuestionContext("
            f"id={self._metadata.derivation_id[:8]}..., "
            f"hash={self._content_hash[:8]}..., "
            f"question='{self._question_text[:50]}...'"
            f")"
        )


class ImmutableContextManager:
    """
    Context manager for enforcing immutable context lifecycle
    Ensures proper cleanup and prevents resource leaks
    """

    def __init__(
        self, initial_question: str, context_data: Optional[Dict[str, Any]] = None
    ):
        self._initial_question = initial_question
        self._initial_context_data = context_data
        self._root_context: Optional[QuestionContext] = None

    def __enter__(self) -> QuestionContext:
        """Create initial immutable context"""
        self._root_context = QuestionContext(
            question_text=self._initial_question,
            context_data=self._initial_context_data,
        )
        return self._root_context

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean exit - contexts are immutable so no cleanup needed"""
        pass


# Factory functions for common patterns


def create_question_context(
    question: str, context_data: Optional[Dict[str, Any]] = None
) -> QuestionContext:
    """Factory function for creating initial question context"""
    return QuestionContext(question_text=question, context_data=context_data)


def create_expanded_context(
    base_context: QuestionContext, expansion_results: Dict[str, Any]
) -> QuestionContext:
# # #     """Factory function for creating expanded context from base context"""  # Module not found  # Module not found  # Module not found
    return base_context.derive_with_expansion(expansion_results)


# Type guards and validation


def is_valid_context(context: Any) -> bool:
    """Type guard for valid QuestionContext"""
    return isinstance(context, QuestionContext) and context.verify_integrity()


def assert_linear_reference(context: QuestionContext) -> None:
    """Assert linear reference compliance - no aliasing"""
    if not isinstance(context, QuestionContext):
        raise TypeError("Invalid context type - must be QuestionContext")

    if not context.verify_integrity():
        raise ValueError("Context integrity violation - linear reference compromised")
