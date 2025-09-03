"""
Context Adapter for EGW Query Expansion System

Provides integration layer between immutable QuestionContext and existing
system components while maintaining linear type theory compliance.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .hash_policies import DEFAULT_CONTEXT_HASHER, PipelineHashValidator
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
    - Maintains hash consistency across pipeline stages
    """

    def __init__(
        self, 
        strict_mode: bool = True,
        enable_hash_validation: bool = True
    ):
        self.strict_mode = strict_mode
        self.enable_hash_validation = enable_hash_validation
        
        # Initialize hash validation components
        if enable_hash_validation:
            self.context_hasher = DEFAULT_CONTEXT_HASHER
            self.pipeline_validator = PipelineHashValidator()
        else:
            self.context_hasher = None
            self.pipeline_validator = None

    def validate_context(
        self, 
        context: QuestionContext, 
        stage_name: Optional[str] = None,
        expected_content_hash: Optional[str] = None
    ) -> None:
        """
        Validate context meets linear type requirements and hash consistency
        
        Args:
            context: QuestionContext to validate
            stage_name: Optional pipeline stage name for hash validation
            expected_content_hash: Expected content hash for validation
        """
        if not is_valid_context(context):
            raise ContextValidationError("Invalid context provided")

        assert_linear_reference(context)
        
        # Additional hash-based validation if enabled
        if self.enable_hash_validation and self.pipeline_validator and stage_name:
            validation_passed = self.pipeline_validator.validate_context_consistency(
                stage_name, 
                context, 
                expected_content_hash
            )
            if not validation_passed and self.strict_mode:
                raise ContextValidationError(
                    f"Hash validation failed for stage '{stage_name}'"
                )

    def adapt_for_query_expansion(
        self,
        context: QuestionContext,
        expansion_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Adapt QuestionContext for query expansion components
        Returns read-only view suitable for expansion algorithms
        """
        self.validate_context(context, stage_name="query_expansion_input")

        adapted_data = {
            "question_text": context.question_text,
            "context_data": dict(context.context_data),
            "metadata": {
                "derivation_id": context.metadata.derivation_id,
                "content_hash": context.content_hash,
                "creation_timestamp": context.metadata.creation_timestamp.isoformat(),
            },
            "expansion_params": expansion_params or {},
        }
        
        # Add hash validation metadata if enabled
        if self.enable_hash_validation and self.context_hasher:
            adapted_data["_hash_validation"] = {
                "content_hash": self.context_hasher.hash_context_content(context),
                "full_hash": self.context_hasher.hash_context(context),
                "stage": "query_expansion_input"
            }
        
        return adapted_data

    def create_expanded_context(
        self, base_context: QuestionContext, expansion_results: Dict[str, Any]
    ) -> QuestionContext:
        """Create new context with expansion results and validate hash consistency"""
        self.validate_context(base_context, stage_name="expansion_base_context")
        
        # Create expanded context
        expanded_context = base_context.derive_with_expansion(expansion_results)
        
        # Validate the new expanded context
        self.validate_context(expanded_context, stage_name="expansion_result_context")
        
        return expanded_context

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
    
    def get_hash_validation_report(self) -> Optional[Dict[str, Any]]:
        """
        Get hash validation report from pipeline validator
        
        Returns:
            Validation report if hash validation is enabled, None otherwise
        """
        if self.enable_hash_validation and self.pipeline_validator:
            return self.pipeline_validator.get_validation_report()
        return None
    
    def verify_pipeline_consistency(self, contexts: List[QuestionContext]) -> bool:
        """
        Verify hash consistency across a sequence of contexts in a pipeline
        
        Args:
            contexts: List of contexts representing pipeline stages
            
        Returns:
            True if all contexts maintain hash consistency, False otherwise
        """
        if not self.enable_hash_validation or not self.pipeline_validator:
            return True
        
        for i, context in enumerate(contexts):
            stage_name = f"pipeline_stage_{i}"
            validation_result = self.pipeline_validator.validate_context_consistency(
                stage_name, 
                context
            )
            if not validation_result and self.strict_mode:
                return False
        
        return True
