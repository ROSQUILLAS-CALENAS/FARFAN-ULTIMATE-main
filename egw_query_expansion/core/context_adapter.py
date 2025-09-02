"""
Context Adapter for EGW Query Expansion System

Provides integration layer between immutable QuestionContext and existing
system components while maintaining linear type theory compliance.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .immutable_context import (
    QuestionContext,
    assert_linear_reference,
    is_valid_context,
)


@runtime_checkable
class ContextualComponent(Protocol):
    """Protocol for components that operate on QuestionContext"""

    def process_context(self, context: QuestionContext) -> QuestionContext:
        """Process context and return new derived context"""
        ...


class ContextValidationError(Exception):
    """Raised when context validation fails"""

    pass


class ContextAdapter:
    """
    Adapter for integrating immutable QuestionContext with existing components

    Ensures all context operations maintain linear type system compliance:
    - Validates context integrity before operations
    - Prevents mutation of input contexts
    - Returns only new derived contexts
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def validate_context(self, context: QuestionContext) -> None:
        """Validate context meets linear type requirements"""
        if not is_valid_context(context):
            raise ContextValidationError("Invalid context provided")

        assert_linear_reference(context)

    def adapt_for_query_expansion(
        self,
        context: QuestionContext,
        expansion_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Adapt QuestionContext for query expansion components
        Returns read-only view suitable for expansion algorithms
        """
        self.validate_context(context)

        return {
            "question_text": context.question_text,
            "context_data": dict(context.context_data),
            "metadata": {
                "derivation_id": context.metadata.derivation_id,
                "content_hash": context.content_hash,
                "creation_timestamp": context.metadata.creation_timestamp.isoformat(),
            },
            "expansion_params": expansion_params or {},
        }

    def create_expanded_context(
        self, base_context: QuestionContext, expansion_results: Dict[str, Any]
    ) -> QuestionContext:
        """Create new context with expansion results"""
        self.validate_context(base_context)

        return base_context.derive_with_expansion(expansion_results)

    def adapt_for_retrieval(
        self,
        context: QuestionContext,
        retrieval_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Adapt context for retrieval components"""
        self.validate_context(context)

        adapted = {
            "query_text": context.question_text,
            "context": dict(context.context_data),
            "retrieval_params": retrieval_params or {},
        }

        # Add expansion data if available
        if "expansion" in context.context_data:
            adapted["expansion_data"] = context.context_data["expansion"]

        return adapted

    def integrate_with_legacy_system(
        self, context: QuestionContext, legacy_format: str = "pdt_context"
    ) -> Dict[str, Any]:
        """
        Integrate with legacy system formats while maintaining immutability
        """
        self.validate_context(context)

        if legacy_format == "pdt_context":
            return self._adapt_to_pdt_context(context)
        else:
            raise ValueError(f"Unsupported legacy format: {legacy_format}")

    def _adapt_to_pdt_context(self, context: QuestionContext) -> Dict[str, Any]:
        """Adapt to PDT Context format"""
        base_data = {
            "question_text": context.question_text,
            "derivation_id": context.metadata.derivation_id,
            "content_hash": context.content_hash,
        }

        # Merge with context data while maintaining immutability
        for key, value in context.context_data.items():
            if key not in base_data:  # Prevent override of system fields
                base_data[key] = value

        return base_data

    def trace_derivation_lineage(
        self, context: QuestionContext
    ) -> List[Dict[str, Any]]:
        """Extract derivation lineage for debugging/auditing"""
        self.validate_context(context)

        lineage = []
        for derivation_id in context.get_lineage():
            # In a full implementation, we'd look up each context
            # For now, return the IDs and basic info
            lineage.append(
                {
                    "derivation_id": derivation_id,
                    "is_current": derivation_id == context.metadata.derivation_id,
                }
            )

        return lineage
