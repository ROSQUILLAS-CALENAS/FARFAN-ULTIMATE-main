"""
JSONCanonicalizer - Stable ID generation and audit trail utility
Provides deterministic JSON serialization, SHA-256 hash-based IDs, and comprehensive audit logging.
"""

import json
import hashlib
import time
import uuid
# # # from datetime import datetime, timezone  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union, Tuple  # Module not found  # Module not found  # Module not found
import logging
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
import traceback
import re


class CanonicalizeError(Exception):
    """Custom exception for canonicalization errors."""
    pass


class ValidationResult(Enum):
    """Validation result status."""
    SUCCESS = "success"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class TransformationStep:
    """Records a transformation step in the canonicalization process."""
    step_name: str
    input_hash: str
    output_hash: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditRecord:
    """Comprehensive audit record for canonicalization operations."""
    operation_id: str
    input_hash: str
    output_hash: str
    canonical_id: str
    transformation_steps: List[TransformationStep] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)


class JSONCanonicalizer:
    """
    Provides stable ID generation using SHA-256 hashes of normalized content,
    deterministic JSON serialization, and comprehensive audit trail logging.
    """
    
    def __init__(self, audit_enabled: bool = True, validation_enabled: bool = True):
        self.audit_enabled = audit_enabled
        self.validation_enabled = validation_enabled
        self.logger = logging.getLogger(__name__)
        self._audit_records: Dict[str, AuditRecord] = {}
        
        # Hash cache for performance optimization
        self._hash_cache: Dict[str, str] = {}
        
        # Canonicalization statistics
        self._stats = {
            "total_operations": 0,
            "cache_hits": 0,
            "validation_failures": 0,
            "edge_case_handles": 0
        }
    
    def _compute_hash(self, data: Union[str, bytes]) -> str:
        """Compute SHA-256 hash of input data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def _normalize_content(self, content: Any) -> str:
        """Normalize content for consistent hashing."""
        if content is None:
            return ""
        
        if isinstance(content, (str, bytes)):
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            # Normalize whitespace and line endings
            content = re.sub(r'\s+', ' ', content.strip())
            return content
        
        if isinstance(content, (int, float, bool)):
            return str(content)
        
        if isinstance(content, (list, tuple)):
            return json.dumps([self._normalize_content(item) for item in content], 
                            sort_keys=True, separators=(',', ':'))
        
        if isinstance(content, dict):
            return self._canonicalize_json_string(content)
        
        # Convert other types to string representation
        return str(content)
    
    def _canonicalize_json_string(self, data: Any) -> str:
        """Create deterministic JSON string with sorted keys and consistent formatting."""
        try:
            # Handle edge cases
            if data is None:
                return json.dumps(None, ensure_ascii=False, separators=(',', ':'))
            
            # Convert to JSON with deterministic formatting
            return json.dumps(
                data, 
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                separators=(',', ': '),
                default=str  # Handle non-serializable objects
            )
        except (TypeError, ValueError) as e:
            self._stats["edge_case_handles"] += 1
            self.logger.warning(f"JSON canonicalization failed: {e}")
            # Fallback to string representation
            return json.dumps(str(data), ensure_ascii=False, separators=(',', ':'))
    
    def generate_stable_id(self, content: Any, prefix: str = "canon") -> str:
# # #         """Generate stable SHA-256 based ID from normalized content."""  # Module not found  # Module not found  # Module not found
        try:
            # Normalize and canonicalize content
            if isinstance(content, str) and self._is_json_string(content):
                # Parse JSON string first to ensure consistent formatting
                try:
                    parsed_content = json.loads(content)
                    canonical_content = self._canonicalize_json_string(parsed_content)
                except json.JSONDecodeError:
                    canonical_content = self._normalize_content(content)
            else:
                canonical_content = self._normalize_content(content)
            
            # Check cache first
            cache_key = f"{prefix}:{canonical_content[:100]}"  # Limit key size
            if cache_key in self._hash_cache:
                self._stats["cache_hits"] += 1
                return self._hash_cache[cache_key]
            
            # Compute hash
            content_hash = self._compute_hash(canonical_content)
            stable_id = f"{prefix}_{content_hash[:16]}"  # Use first 16 chars for readability
            
            # Cache result
            self._hash_cache[cache_key] = stable_id
            
            return stable_id
            
        except Exception as e:
            self._stats["edge_case_handles"] += 1
            self.logger.error(f"Stable ID generation failed: {e}")
            # Fallback to UUID-based ID
            return f"{prefix}_{uuid.uuid4().hex[:16]}"
    
    def _is_json_string(self, text: str) -> bool:
        """Check if string is valid JSON."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    def canonicalize(self, data: Any, operation_context: Optional[Dict[str, Any]] = None) -> Tuple[str, str, AuditRecord]:
        """
        Main canonicalization method that returns canonical JSON, stable ID, and audit record.
        
        Args:
            data: Input data to canonicalize
            operation_context: Optional context information for audit trail
            
        Returns:
            Tuple of (canonical_json, stable_id, audit_record)
        """
        start_time = time.time()
        operation_id = f"canon_{uuid.uuid4().hex[:8]}"
        
        self._stats["total_operations"] += 1
        
        # Handle edge cases
        if data is None:
            canonical_json = json.dumps(None)
            stable_id = self.generate_stable_id(canonical_json)
            audit_record = AuditRecord(
                operation_id=operation_id,
                input_hash=self._compute_hash("null"),
                output_hash=self._compute_hash(canonical_json),
                canonical_id=stable_id,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"edge_case": "null_input"}
            )
            return canonical_json, stable_id, audit_record
        
        try:
            # Create audit record
            input_hash = self._compute_hash(self._normalize_content(data))
            audit_record = AuditRecord(
                operation_id=operation_id,
                input_hash=input_hash,
                output_hash="",  # Will be filled later
                canonical_id="",  # Will be filled later
                metadata=operation_context or {}
            )
            
            # Step 1: Input normalization
            step1_start = time.time()
            normalized_data = self._normalize_content(data)
            step1_hash = self._compute_hash(normalized_data)
            audit_record.transformation_steps.append(TransformationStep(
                step_name="input_normalization",
                input_hash=input_hash,
                output_hash=step1_hash,
                timestamp=step1_start,
                metadata={"data_type": type(data).__name__}
            ))
            
            # Step 2: JSON canonicalization
            step2_start = time.time()
            if isinstance(data, str) and self._is_json_string(data):
                # Parse and re-serialize for consistency
                parsed_data = json.loads(data)
                canonical_json = self._canonicalize_json_string(parsed_data)
            elif isinstance(data, (dict, list)):
                canonical_json = self._canonicalize_json_string(data)
            else:
                # Wrap primitive types in consistent structure
                canonical_json = self._canonicalize_json_string({"value": data, "type": type(data).__name__})
            
            step2_hash = self._compute_hash(canonical_json)
            audit_record.transformation_steps.append(TransformationStep(
                step_name="json_canonicalization",
                input_hash=step1_hash,
                output_hash=step2_hash,
                timestamp=step2_start
            ))
            
            # Step 3: ID generation
            step3_start = time.time()
            stable_id = self.generate_stable_id(canonical_json)
            audit_record.canonical_id = stable_id
            audit_record.output_hash = step2_hash
            
            audit_record.transformation_steps.append(TransformationStep(
                step_name="id_generation",
                input_hash=step2_hash,
                output_hash=self._compute_hash(stable_id),
                timestamp=step3_start,
                metadata={"stable_id": stable_id}
            ))
            
            # Step 4: Validation (if enabled)
            if self.validation_enabled:
                validation_results = self._validate_canonicalization(data, canonical_json, stable_id)
                audit_record.validation_results = validation_results
                
                if any(r.get("status") == ValidationResult.FAILED.value for r in validation_results):
                    self._stats["validation_failures"] += 1
            
            # Finalize audit record
            audit_record.execution_time_ms = (time.time() - start_time) * 1000
            
            if self.audit_enabled:
                self._audit_records[operation_id] = audit_record
            
            return canonical_json, stable_id, audit_record
            
        except Exception as e:
            self._stats["edge_case_handles"] += 1
            self.logger.error(f"Canonicalization failed: {e}\nTraceback: {traceback.format_exc()}")
            
            # Create error audit record
            error_canonical = json.dumps({"error": str(e), "input_type": type(data).__name__})
            error_id = self.generate_stable_id(error_canonical, prefix="error")
            
            audit_record = AuditRecord(
                operation_id=operation_id,
                input_hash=self._compute_hash(str(data)),
                output_hash=self._compute_hash(error_canonical),
                canonical_id=error_id,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e), "fallback": True}
            )
            
            return error_canonical, error_id, audit_record
    
    def _validate_canonicalization(self, original_data: Any, canonical_json: str, stable_id: str) -> List[Dict[str, Any]]:
        """Validate canonicalization results."""
        validation_results = []
        
        try:
            # Validation 1: JSON parsability
            try:
                parsed = json.loads(canonical_json)
                validation_results.append({
                    "validation": "json_parsability",
                    "status": ValidationResult.SUCCESS.value,
                    "message": "Canonical JSON is valid"
                })
            except json.JSONDecodeError as e:
                validation_results.append({
                    "validation": "json_parsability",
                    "status": ValidationResult.FAILED.value,
                    "message": f"Invalid JSON: {e}"
                })
            
            # Validation 2: Determinism check (repeat canonicalization)
            try:
                canonical_json2, stable_id2, _ = self.canonicalize(original_data, {"validation_repeat": True})
                if canonical_json == canonical_json2 and stable_id == stable_id2:
                    validation_results.append({
                        "validation": "determinism",
                        "status": ValidationResult.SUCCESS.value,
                        "message": "Repeated canonicalization produces identical results"
                    })
                else:
                    validation_results.append({
                        "validation": "determinism",
                        "status": ValidationResult.FAILED.value,
                        "message": "Repeated canonicalization produces different results"
                    })
            except Exception as e:
                validation_results.append({
                    "validation": "determinism",
                    "status": ValidationResult.WARNING.value,
                    "message": f"Could not validate determinism: {e}"
                })
            
            # Validation 3: ID format check
            if re.match(r'^[a-z_]+_[a-f0-9]{16}$', stable_id):
                validation_results.append({
                    "validation": "id_format",
                    "status": ValidationResult.SUCCESS.value,
                    "message": "Stable ID format is valid"
                })
            else:
                validation_results.append({
                    "validation": "id_format",
                    "status": ValidationResult.WARNING.value,
                    "message": f"Stable ID format may be non-standard: {stable_id}"
                })
            
        except Exception as e:
            validation_results.append({
                "validation": "general",
                "status": ValidationResult.FAILED.value,
                "message": f"Validation error: {e}"
            })
        
        return validation_results
    
    def save_audit_trail(self, output_file: Union[str, Path], operation_ids: Optional[List[str]] = None) -> bool:
        """Save audit trail to companion _audit.json file."""
        try:
            output_path = Path(output_file)
            
            # Create audit file path
            if output_path.suffix == '.json':
                audit_path = output_path.parent / f"{output_path.stem}_audit.json"
            else:
                audit_path = output_path.parent / f"{output_path.name}_audit.json"
            
            # Filter audit records if specific operation IDs requested
            if operation_ids:
                audit_data = {op_id: self._audit_records[op_id] 
                            for op_id in operation_ids if op_id in self._audit_records}
            else:
                audit_data = dict(self._audit_records)
            
            # Create comprehensive audit trail
            audit_trail = {
                "audit_metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "canonicalizer_version": "1.0.0",
                    "total_operations": len(audit_data),
                    "statistics": dict(self._stats)
                },
                "operations": {}
            }
            
            # Convert audit records to serializable format
            for op_id, record in audit_data.items():
                audit_trail["operations"][op_id] = {
                    "operation_id": record.operation_id,
                    "input_hash": record.input_hash,
                    "output_hash": record.output_hash,
                    "canonical_id": record.canonical_id,
                    "timestamp": record.timestamp,
                    "execution_time_ms": record.execution_time_ms,
                    "metadata": record.metadata,
                    "transformation_steps": [
                        {
                            "step_name": step.step_name,
                            "input_hash": step.input_hash,
                            "output_hash": step.output_hash,
                            "timestamp": step.timestamp,
                            "metadata": step.metadata
                        }
                        for step in record.transformation_steps
                    ],
                    "validation_results": record.validation_results
                }
            
            # Write audit trail with canonical formatting
            audit_json = self._canonicalize_json_string(audit_trail)
            
            with open(audit_path, 'w', encoding='utf-8') as f:
                f.write(audit_json)
            
            self.logger.info(f"Audit trail saved to {audit_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save audit trail: {e}")
            return False
    
    def validate_repeated_canonicalization(self, data: Any, iterations: int = 3) -> Dict[str, Any]:
        """Validate that repeated canonicalization operations produce identical outputs."""
        results = {"success": True, "iterations": [], "summary": {}}
        
        try:
            first_canonical = None
            first_id = None
            
            for i in range(iterations):
                canonical_json, stable_id, audit_record = self.canonicalize(
                    data, {"validation_iteration": i}
                )
                
                iteration_result = {
                    "iteration": i,
                    "canonical_json": canonical_json,
                    "stable_id": stable_id,
                    "execution_time_ms": audit_record.execution_time_ms,
                    "matches_first": True
                }
                
                if i == 0:
                    first_canonical = canonical_json
                    first_id = stable_id
                else:
                    if canonical_json != first_canonical or stable_id != first_id:
                        iteration_result["matches_first"] = False
                        results["success"] = False
                
                results["iterations"].append(iteration_result)
            
            results["summary"] = {
                "all_identical": results["success"],
                "total_iterations": iterations,
                "unique_canonicals": len(set(r["canonical_json"] for r in results["iterations"])),
                "unique_ids": len(set(r["stable_id"] for r in results["iterations"])),
                "avg_execution_time_ms": sum(r["execution_time_ms"] for r in results["iterations"]) / iterations
            }
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            self.logger.error(f"Validation failed: {e}")
        
        return results
    
    def handle_edge_cases(self, data: Any) -> Tuple[bool, str]:
        """Handle edge cases: empty inputs, malformed JSON, encoding issues."""
        try:
            # Empty inputs
            if data is None:
                return True, "null_input"
            
            if isinstance(data, str) and not data.strip():
                return True, "empty_string"
            
            if isinstance(data, (list, dict)) and len(data) == 0:
                return True, "empty_collection"
            
            # Malformed JSON detection
            if isinstance(data, str):
                try:
                    json.loads(data)
                except json.JSONDecodeError:
                    return True, "malformed_json"
            
            # Encoding issues
            if isinstance(data, bytes):
                try:
                    data.decode('utf-8')
                except UnicodeDecodeError:
                    return True, "encoding_issue"
            
            return False, "normal"
            
        except Exception as e:
            self.logger.warning(f"Edge case detection failed: {e}")
            return True, "detection_error"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get canonicalization statistics."""
        return {
            **self._stats,
            "cache_size": len(self._hash_cache),
            "audit_records": len(self._audit_records)
        }
    
    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._hash_cache.clear()
        self._audit_records.clear()
        self.logger.info("Canonicalizer caches cleared")


# Convenience function for quick canonicalization
def canonicalize_json(data: Any, audit_enabled: bool = True) -> Tuple[str, str]:
    """
    Quick canonicalization function that returns canonical JSON and stable ID.
    
    Args:
        data: Input data to canonicalize
        audit_enabled: Whether to enable audit trail logging
        
    Returns:
        Tuple of (canonical_json, stable_id)
    """
    canonicalizer = JSONCanonicalizer(audit_enabled=audit_enabled)
    canonical_json, stable_id, _ = canonicalizer.canonicalize(data)
    return canonical_json, stable_id


# Export main classes and functions
__all__ = [
    'JSONCanonicalizer',
    'AuditRecord', 
    'TransformationStep',
    'ValidationResult',
    'CanonicalizeError',
    'canonicalize_json'
]