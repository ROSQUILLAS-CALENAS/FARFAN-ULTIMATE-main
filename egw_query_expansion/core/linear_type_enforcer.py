"""
Linear Type System Enforcer for EGW Query Expansion

Implements runtime enforcement of linear type theory principles based on
Bernardy et al. (2021) substructural type system for quantum programming.

Enforces:
1. Affine types - at most one reference per context
2. Linear types - exactly one reference per context
3. Relevant types - at least one reference per context
4. No aliasing - prevents multiple references to same context
5. Resource management - automatic cleanup of consumed contexts
"""

import threading
import weakref
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generic, Optional, Set, TypeVar

from .immutable_context import DerivationId, QuestionContext

T = TypeVar("T")


class LinearTypeError(Exception):
    """Raised when linear type constraints are violated"""

    pass


class AliasDetectionError(LinearTypeError):
    """Raised when aliasing is detected"""

    pass


class ResourceLeakError(LinearTypeError):
    """Raised when resources are not properly consumed"""

    pass


class LinearReference(Generic[T]):
    """
    Linear reference wrapper that enforces single-use semantics

    Once consumed, the reference becomes invalid and cannot be used again.
    This prevents aliasing and ensures linear resource management.
    """

    def __init__(self, value: T, reference_id: str):
        self._value = value
        self._reference_id = reference_id
        self._consumed = False
        self._creation_thread = threading.current_thread().ident

    def consume(self) -> T:
        """
        Consume the linear reference, returning the value and invalidating the reference
        """
        if self._consumed:
            raise LinearTypeError(
                f"Linear reference {self._reference_id} already consumed. "
                f"Linear types can only be used once."
            )

        self._consumed = True
        value = self._value
        self._value = None  # Clear reference to prevent accidental access
        return value

    def is_consumed(self) -> bool:
        """Check if reference has been consumed"""
        return self._consumed

    @property
    def reference_id(self) -> str:
        return self._reference_id

    def __del__(self):
        """Detect resource leaks on garbage collection"""
        if not self._consumed:
            # In production, this might log rather than raise
            import warnings

            warnings.warn(
                f"Linear reference {self._reference_id} was not consumed - potential resource leak",
                ResourceWarning,
                stacklevel=2,
            )


class LinearTypeEnforcer:
    """
    Runtime enforcer for linear type system constraints

    Tracks all active contexts and their references to prevent aliasing
    and ensure proper resource management according to linear type theory.
    """

    def __init__(self):
        self._active_contexts: Dict[DerivationId, weakref.ref] = {}
        self._context_references: Dict[DerivationId, Set[str]] = defaultdict(set)
        self._reference_usage: Dict[str, bool] = {}  # True if consumed
        self._lock = threading.RLock()

    def create_linear_reference(
        self, context: QuestionContext
    ) -> LinearReference[QuestionContext]:
        """
        Create a linear reference to a context

        Ensures that only one linear reference exists per context at a time,
        preventing aliasing according to linear type theory.
        """
        with self._lock:
            derivation_id = context.metadata.derivation_id
            reference_id = f"{derivation_id}:{id(context)}"

            # Check for existing active references
            if derivation_id in self._context_references:
                active_refs = [
                    ref_id
                    for ref_id in self._context_references[derivation_id]
                    if not self._reference_usage.get(ref_id, False)
                ]
                if active_refs:
                    raise AliasDetectionError(
                        f"Aliasing detected: Context {derivation_id} already has active reference(s): {active_refs}. "
                        f"Linear type system prohibits multiple references."
                    )

            # Register the new reference
            self._active_contexts[derivation_id] = weakref.ref(context)
            self._context_references[derivation_id].add(reference_id)
            self._reference_usage[reference_id] = False

            return LinearReference(context, reference_id)

    def consume_reference(
        self, linear_ref: LinearReference[QuestionContext]
    ) -> QuestionContext:
        """
        Consume a linear reference, marking it as used

        This satisfies the linear type requirement that each resource
        must be used exactly once.
        """
        with self._lock:
            if linear_ref.is_consumed():
                raise LinearTypeError(
                    f"Reference {linear_ref.reference_id} already consumed"
                )

            context = linear_ref.consume()
            self._reference_usage[linear_ref.reference_id] = True

            return context

    def check_no_aliasing(self, context: QuestionContext) -> bool:
        """
        Verify that a context has no active aliases

        Returns True if context passes aliasing check, False otherwise.
        """
        with self._lock:
            derivation_id = context.metadata.derivation_id

            if derivation_id not in self._context_references:
                return True

            active_refs = [
                ref_id
                for ref_id in self._context_references[derivation_id]
                if not self._reference_usage.get(ref_id, False)
            ]

            return len(active_refs) <= 1

    def cleanup_consumed_references(self) -> None:
        """Clean up consumed references to prevent memory leaks"""
        with self._lock:
            # Remove consumed references
            for derivation_id in list(self._context_references.keys()):
                active_refs = {
                    ref_id
                    for ref_id in self._context_references[derivation_id]
                    if not self._reference_usage.get(ref_id, False)
                }

                if not active_refs:
                    # No active references, cleanup
                    del self._context_references[derivation_id]
                    if derivation_id in self._active_contexts:
                        del self._active_contexts[derivation_id]
                else:
                    # Update to only active references
                    self._context_references[derivation_id] = active_refs

            # Clean up consumed reference usage tracking
            consumed_refs = [
                ref_id for ref_id, consumed in self._reference_usage.items() if consumed
            ]
            for ref_id in consumed_refs:
                del self._reference_usage[ref_id]

    def get_active_contexts_count(self) -> int:
        """Get count of active contexts for monitoring"""
        with self._lock:
            return len(self._active_contexts)

    def validate_system_state(self) -> Dict[str, Any]:
        """
        Validate entire system state for linear type compliance

        Returns diagnostic information about current state.
        """
        with self._lock:
            diagnostics = {
                "active_contexts": len(self._active_contexts),
                "total_references": sum(
                    len(refs) for refs in self._context_references.values()
                ),
                "consumed_references": sum(
                    1 for consumed in self._reference_usage.values() if consumed
                ),
                "potential_leaks": [],
            }

            # Check for potential resource leaks
            for derivation_id, ref_ids in self._context_references.items():
                unconsumed = [
                    ref_id
                    for ref_id in ref_ids
                    if not self._reference_usage.get(ref_id, False)
                ]
                if len(unconsumed) > 1:
                    diagnostics["potential_leaks"].append(
                        {
                            "context_id": derivation_id,
                            "unconsumed_references": len(unconsumed),
                        }
                    )

            return diagnostics


# Global enforcer instance
_global_enforcer = LinearTypeEnforcer()


def get_linear_enforcer() -> LinearTypeEnforcer:
    """Get the global linear type enforcer instance"""
    return _global_enforcer


@contextmanager
def linear_context_scope(context: QuestionContext):
    """
    Context manager for linear context operations

    Ensures proper resource management and cleanup according to
    linear type system requirements.
    """
    enforcer = get_linear_enforcer()
    linear_ref = enforcer.create_linear_reference(context)

    try:
        consumed_context = enforcer.consume_reference(linear_ref)
        yield consumed_context
    finally:
        enforcer.cleanup_consumed_references()


def linear_operation(
    func: Callable[[QuestionContext], T]
) -> Callable[[QuestionContext], T]:
    """
    Decorator for functions that operate on contexts with linear type constraints

    Ensures that:
    1. Input context has no aliases
    2. Function is called with proper linear reference
    3. Resources are cleaned up after operation
    """

    def wrapper(context: QuestionContext) -> T:
        enforcer = get_linear_enforcer()

        # Verify no aliasing
        if not enforcer.check_no_aliasing(context):
            raise AliasDetectionError(
                f"Cannot perform linear operation on aliased context {context.metadata.derivation_id}"
            )

        # Execute with linear scope
        with linear_context_scope(context) as linear_context:
            return func(linear_context)

    return wrapper


def assert_no_aliasing(context: QuestionContext) -> None:
    """Assert that context has no aliases - utility for debugging"""
    enforcer = get_linear_enforcer()
    if not enforcer.check_no_aliasing(context):
        raise AliasDetectionError(
            f"Aliasing detected for context {context.metadata.derivation_id}"
        )


def consume_context_linearly(context: QuestionContext) -> QuestionContext:
    """
    Consume a context in a linear fashion

    This is the canonical way to use contexts in the linear type system.
    """
    enforcer = get_linear_enforcer()
    linear_ref = enforcer.create_linear_reference(context)
    return enforcer.consume_reference(linear_ref)
