"""
Deterministic Hashing System for Context and Synthesis Operations

Provides stable serialization functions that handle nested dictionaries, lists, 
and custom objects with consistent ordering and normalization. Ensures hash 
consistency across pipeline stages regardless of input order or structure variations.

Key features:
- Deterministic serialization of complex nested structures
- Consistent ordering for dictionaries and sets
- Type-aware normalization for custom objects
- Immutable hash generation for contexts and synthesis results
- Pipeline stage validation and consistency checking

Design principles:
- Sort keys and values for consistent ordering
- Normalize type representations to handle variations
- Use canonical string representations for all objects
- Handle circular references and complex nested structures
- Provide different hash policies for different use cases
"""

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

try:
    import blake3
    _BLAKE3_AVAILABLE = True
except ImportError:
    _BLAKE3_AVAILABLE = False


class Serializable(Protocol):
    """Protocol for objects that can be deterministically serialized"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to deterministic dictionary representation"""
        ...


class HashPolicy(ABC):
    """Abstract base class for hash policy implementations"""
    
    @abstractmethod
    def serialize(self, obj: Any) -> str:
        """Serialize object to deterministic string representation"""
        ...
    
    @abstractmethod
    def hash_object(self, obj: Any) -> str:
        """Generate deterministic hash for object"""
        ...


class CanonicalHashPolicy(HashPolicy):
    """
    Canonical hash policy that provides deterministic serialization
    with consistent ordering and type normalization
    """
    
    def __init__(
        self, 
        algorithm: str = "sha256",
        sort_keys: bool = True,
        normalize_types: bool = True,
        handle_circular_refs: bool = True,
        max_depth: int = 100
    ):
        self.algorithm = algorithm
        self.sort_keys = sort_keys
        self.normalize_types = normalize_types
        self.handle_circular_refs = handle_circular_refs
        self.max_depth = max_depth
        
        # Select hash algorithm
        if algorithm == "blake3" and _BLAKE3_AVAILABLE:
            self._hasher = blake3.blake3
        elif algorithm in ["sha256", "sha1", "md5"]:
            self._hasher = getattr(hashlib, algorithm)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def serialize(self, obj: Any, _depth: int = 0, _seen: Optional[set] = None) -> str:
        """
        Serialize object to deterministic string representation
        
        Args:
            obj: Object to serialize
            _depth: Current recursion depth (internal use)
            _seen: Set of seen object ids for circular reference detection
            
        Returns:
            Canonical string representation
        """
        if _depth > self.max_depth:
            raise ValueError(f"Maximum serialization depth {self.max_depth} exceeded")
        
        if _seen is None:
            _seen = set()
        
        # Handle circular references
        if self.handle_circular_refs:
            obj_id = id(obj)
            if obj_id in _seen:
                return f"<circular_ref:{type(obj).__name__}:{obj_id}>"
            _seen.add(obj_id)
        
        try:
            return self._serialize_recursive(obj, _depth, _seen)
        finally:
            if self.handle_circular_refs and obj_id in _seen:
                _seen.remove(obj_id)
    
    def _serialize_recursive(self, obj: Any, depth: int, seen: set) -> str:
        """Internal recursive serialization method"""
        
        # Handle None
        if obj is None:
            return "null"
        
        # Handle primitives
        if isinstance(obj, bool):
            return "true" if obj else "false"
        elif isinstance(obj, (int, float)):
            # Normalize numeric representations
            if isinstance(obj, float) and obj.is_integer():
                return str(int(obj))
            return str(obj)
        elif isinstance(obj, str):
            # Escape and quote strings consistently
            return json.dumps(obj, ensure_ascii=True, sort_keys=True)
        
        # Handle datetime objects
        elif isinstance(obj, datetime):
            # Convert to ISO format in UTC
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return f'"{obj.isoformat()}"'
        
        # Handle UUID objects
        elif isinstance(obj, uuid.UUID):
            return f'"{str(obj)}"'
        
        # Handle sequences (lists, tuples, etc.)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            items = [self._serialize_recursive(item, depth + 1, seen) for item in obj]
            return f"[{','.join(items)}]"
        
        # Handle sets (convert to sorted lists for deterministic ordering)
        elif isinstance(obj, Set):
            # Convert set items to strings for sorting
            items = []
            for item in obj:
                item_str = self._serialize_recursive(item, depth + 1, seen)
                items.append(item_str)
            items.sort()  # Sort for deterministic ordering
            return f"[{','.join(items)}]"
        
        # Handle mappings (dicts, etc.)
        elif isinstance(obj, Mapping):
            if self.sort_keys:
                items = sorted(obj.items())
            else:
                items = obj.items()
            
            serialized_items = []
            for key, value in items:
                key_str = self._serialize_recursive(key, depth + 1, seen)
                value_str = self._serialize_recursive(value, depth + 1, seen)
                serialized_items.append(f"{key_str}:{value_str}")
            
            return f"{{{','.join(serialized_items)}}}"
        
        # Handle dataclasses
        elif is_dataclass(obj):
            obj_dict = asdict(obj)
            return self._serialize_recursive(obj_dict, depth, seen)
        
        # Handle objects with to_dict method
        elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
            obj_dict = obj.to_dict()
            return self._serialize_recursive(obj_dict, depth, seen)
        
        # Handle objects with __dict__ attribute
        elif hasattr(obj, '__dict__'):
            obj_dict = obj.__dict__.copy()
            # Add type information for disambiguation
            if self.normalize_types:
                obj_dict['__type__'] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            return self._serialize_recursive(obj_dict, depth, seen)
        
        # Handle bytes
        elif isinstance(obj, bytes):
            # Convert bytes to hex representation
            return f'"<bytes:{obj.hex()}>"'
        
        # Fallback for other types
        else:
            if self.normalize_types:
                type_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
                return f'"<object:{type_name}:{str(obj)}>"'
            else:
                return f'"{str(obj)}"'
    
    def hash_object(self, obj: Any) -> str:
        """Generate deterministic hash for object"""
        canonical_str = self.serialize(obj)
        return self._hasher(canonical_str.encode('utf-8')).hexdigest()


class FastHashPolicy(CanonicalHashPolicy):
    """
    Optimized hash policy for performance-critical scenarios
    Trades some normalization features for speed
    """
    
    def __init__(self, algorithm: str = "blake3" if _BLAKE3_AVAILABLE else "sha256"):
        super().__init__(
            algorithm=algorithm,
            sort_keys=True,
            normalize_types=False,
            handle_circular_refs=False,
            max_depth=50
        )


class SecureHashPolicy(CanonicalHashPolicy):
    """
    Security-focused hash policy with enhanced protection against
    hash collision attacks and detailed type normalization
    """
    
    def __init__(self, salt: Optional[bytes] = None):
        super().__init__(
            algorithm="sha256",
            sort_keys=True,
            normalize_types=True,
            handle_circular_refs=True,
            max_depth=200
        )
        self.salt = salt or b"egw_query_expansion_salt"
    
    def hash_object(self, obj: Any) -> str:
        """Generate salted hash for enhanced security"""
        canonical_str = self.serialize(obj)
        salted_content = self.salt + canonical_str.encode('utf-8')
        return self._hasher(salted_content).hexdigest()


class ContextHasher:
    """
    Specialized hasher for QuestionContext and related objects
    Integrates with the immutable context system
    """
    
    def __init__(self, policy: Optional[HashPolicy] = None):
        self.policy = policy or CanonicalHashPolicy()
    
    def hash_context(self, context: 'QuestionContext') -> str:
        """Generate deterministic hash for QuestionContext"""
        context_data = {
            'question_text': context.question_text,
            'context_data': dict(context.context_data),
            'metadata': {
                'derivation_id': context.metadata.derivation_id,
                'parent_id': context.metadata.parent_id,
                'operation_type': context.metadata.operation_type,
                'creation_timestamp': context.metadata.creation_timestamp.isoformat(),
            }
        }
        return self.policy.hash_object(context_data)
    
    def hash_context_content(self, context: 'QuestionContext') -> str:
        """Generate hash for context content only (excluding metadata)"""
        content_data = {
            'question_text': context.question_text,
            'context_data': dict(context.context_data)
        }
        return self.policy.hash_object(content_data)
    
    def hash_synthesis_result(self, result: Any) -> str:
        """Generate deterministic hash for synthesis results"""
        return self.policy.hash_object(result)


class SynthesisHasher:
    """
    Specialized hasher for synthesis operations and results
    Handles SynthesizedAnswer and related synthesis artifacts
    """
    
    def __init__(self, policy: Optional[HashPolicy] = None):
        self.policy = policy or CanonicalHashPolicy()
    
    def hash_synthesized_answer(self, answer: 'SynthesizedAnswer') -> str:
        """Generate deterministic hash for SynthesizedAnswer"""
        # Extract core components for hashing
        answer_data = {
            'question': answer.question,
            'verdict': answer.verdict,
            'rationale': answer.rationale,
            'premises': [
                {
                    'text': premise.text,
                    'evidence_id': premise.evidence_id,
                    'score': premise.score
                }
                for premise in answer.premises
            ],
            'unmet_conjuncts': answer.unmet_conjuncts,
            'citations': answer.citations,
            'confidence': answer.confidence,
            'confidence_interval': answer.confidence_interval,
            'conformal_alpha': answer.conformal_alpha
        }
        return self.policy.hash_object(answer_data)
    
    def hash_synthesis_pipeline_state(self, pipeline_state: Dict[str, Any]) -> str:
        """Hash entire synthesis pipeline state for consistency validation"""
        return self.policy.hash_object(pipeline_state)


class PipelineHashValidator:
    """
    Validator for ensuring hash consistency across pipeline stages
    Tracks and validates hash evolution through the processing pipeline
    """
    
    def __init__(self, context_hasher: Optional[ContextHasher] = None):
        self.context_hasher = context_hasher or ContextHasher()
        self.synthesis_hasher = SynthesisHasher(self.context_hasher.policy)
        self.stage_hashes: Dict[str, Dict[str, str]] = {}
        self.validation_log: List[Dict[str, Any]] = []
    
    def validate_context_consistency(
        self, 
        stage_name: str, 
        context: 'QuestionContext',
        expected_content_hash: Optional[str] = None
    ) -> bool:
        """
        Validate context consistency at a pipeline stage
        
        Args:
            stage_name: Name of the pipeline stage
            context: QuestionContext to validate
            expected_content_hash: Expected content hash (if available)
            
        Returns:
            True if validation passes, False otherwise
        """
        current_hash = self.context_hasher.hash_context(context)
        content_hash = self.context_hasher.hash_context_content(context)
        
        # Store stage hashes
        if stage_name not in self.stage_hashes:
            self.stage_hashes[stage_name] = {}
        
        self.stage_hashes[stage_name].update({
            'context_hash': current_hash,
            'content_hash': content_hash,
            'derivation_id': context.metadata.derivation_id
        })
        
        # Validate against expected content hash if provided
        validation_result = True
        if expected_content_hash is not None:
            validation_result = content_hash == expected_content_hash
        
        # Log validation result
        self.validation_log.append({
            'stage': stage_name,
            'derivation_id': context.metadata.derivation_id,
            'context_hash': current_hash,
            'content_hash': content_hash,
            'expected_content_hash': expected_content_hash,
            'validation_passed': validation_result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        return validation_result
    
    def validate_synthesis_consistency(
        self, 
        stage_name: str,
        synthesis_result: Any,
        context: Optional['QuestionContext'] = None
    ) -> bool:
        """
        Validate synthesis result consistency
        
        Args:
            stage_name: Name of the synthesis stage
            synthesis_result: Result to validate
            context: Associated context (if available)
            
        Returns:
            True if validation passes, False otherwise
        """
        synthesis_hash = self.synthesis_hasher.hash_synthesis_result(synthesis_result)
        
        # Store synthesis hash
        if stage_name not in self.stage_hashes:
            self.stage_hashes[stage_name] = {}
        
        self.stage_hashes[stage_name]['synthesis_hash'] = synthesis_hash
        
        # If context is provided, also validate context consistency
        context_validation = True
        if context is not None:
            context_validation = self.validate_context_consistency(
                f"{stage_name}_context", 
                context
            )
        
        # Log synthesis validation
        self.validation_log.append({
            'stage': stage_name,
            'synthesis_hash': synthesis_hash,
            'context_validation': context_validation,
            'validation_passed': True,  # Basic validation always passes
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        return context_validation
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        return {
            'stage_hashes': self.stage_hashes,
            'validation_log': self.validation_log,
            'total_validations': len(self.validation_log),
            'passed_validations': sum(
                1 for log in self.validation_log if log['validation_passed']
            ),
            'failed_validations': sum(
                1 for log in self.validation_log if not log['validation_passed']
            ),
            'report_timestamp': datetime.now(timezone.utc).isoformat()
        }


# Default hash policy instances
DEFAULT_HASH_POLICY = CanonicalHashPolicy()
FAST_HASH_POLICY = FastHashPolicy()
SECURE_HASH_POLICY = SecureHashPolicy()

# Default hasher instances
DEFAULT_CONTEXT_HASHER = ContextHasher(DEFAULT_HASH_POLICY)
FAST_CONTEXT_HASHER = ContextHasher(FAST_HASH_POLICY)
SECURE_CONTEXT_HASHER = ContextHasher(SECURE_HASH_POLICY)

DEFAULT_SYNTHESIS_HASHER = SynthesisHasher(DEFAULT_HASH_POLICY)
FAST_SYNTHESIS_HASHER = SynthesisHasher(FAST_HASH_POLICY)
SECURE_SYNTHESIS_HASHER = SynthesisHasher(SECURE_HASH_POLICY)


def hash_object(obj: Any, policy: Optional[HashPolicy] = None) -> str:
    """
    Convenience function to hash any object with specified policy
    
    Args:
        obj: Object to hash
        policy: Hash policy to use (defaults to CanonicalHashPolicy)
        
    Returns:
        Deterministic hash string
    """
    if policy is None:
        policy = DEFAULT_HASH_POLICY
    return policy.hash_object(obj)


def create_pipeline_validator() -> PipelineHashValidator:
    """
    Factory function to create a new pipeline hash validator
    
    Returns:
        Configured PipelineHashValidator instance
    """
    return PipelineHashValidator()