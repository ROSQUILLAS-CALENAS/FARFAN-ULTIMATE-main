"""
Canonical Flow Alias: 17A
Evidence Validation Model with Total Ordering and Deterministic Processing

Source: evidence_validation_model.py
Stage: analysis_nlp
Code: 17A
"""

import hashlib
import hmac
import secrets
import time
import sys
import json
import logging
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from base64 import urlsafe_b64encode  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# # # from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin  # Module not found  # Module not found  # Module not found

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    class np:
        @staticmethod
        def random():
            return type("random", (), {
                "seed": lambda x: None,
                "shuffle": lambda x: None, 
                "normal": lambda *args, **kwargs: [0] * kwargs.get("size", 1),
            })()
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0
        
        @staticmethod
        def concatenate(arrays):
            result = []
            for arr in arrays:
                result.extend(arr)
            return result


logger = logging.getLogger(__name__)

class ValidationSeverity(str, Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationCategory(str, Enum):
    """Categories of validation checks"""
    FACTUAL_ACCURACY = "factual_accuracy"
    LOGICAL_CONSISTENCY = "logical_consistency"
    SOURCE_RELIABILITY = "source_reliability"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    BIAS_DETECTION = "bias_detection"


class DNPAlignmentCategory(str, Enum):
    """DNP alignment categories"""
    CONSTITUTIONAL = "constitutional"
    REGULATORY = "regulatory"
    PROCEDURAL = "procedural"
    ETHICAL = "ethical"
    TECHNICAL = "technical"


class EvidenceValidationRequest(TotalOrderingBase):
    """Request for evidence validation with deterministic processing"""
    
    def __init__(self, evidence_text: str, context: str = "", validation_type: str = "comprehensive"):
        super().__init__("EvidenceValidationRequest")
        
        self.evidence_text = evidence_text
        self.context = context
        self.validation_type = validation_type
        self.request_timestamp = self._get_deterministic_timestamp()
        self.request_id = self.generate_stable_id(
            {"text": evidence_text, "context": context, "type": validation_type},
            prefix="req"
        )
        
        # Update state hash
        self.update_state_hash({
            "evidence_text": evidence_text,
            "context": context,
            "validation_type": validation_type,
        })


class EvidenceValidationResponse(TotalOrderingBase):
# # #     """Response from evidence validation with deterministic processing"""  # Module not found  # Module not found  # Module not found
    
    def __init__(self, request_id: str):
        super().__init__("EvidenceValidationResponse") 
        
        self.request_id = request_id
        self.response_id = self.generate_stable_id({"request_id": request_id}, prefix="resp")
        self.validation_score = 0.0
        self.is_valid = False
        self.validation_issues: List[Dict[str, Any]] = []
        self.dnp_alignment_score = 0.0
        self.confidence_interval = (0.0, 1.0)
        self.processing_time_ms = 0
        self.response_timestamp = self._get_deterministic_timestamp()
        
        # Update state hash
        self.update_state_hash(self._get_response_state())
    
    def _get_response_state(self) -> Dict[str, Any]:
        """Get response state for hash calculation"""
        return {
            "request_id": self.request_id,
            "validation_score": self.validation_score,
            "is_valid": self.is_valid,
            "dnp_alignment_score": self.dnp_alignment_score,
            "issues_count": len(self.validation_issues),
        }


class EvidenceValidationModel(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Evidence Validation Model with deterministic processing and total ordering.
    
    Provides consistent validation results and stable ID generation across runs.
    """
    
    def __init__(self):
        super().__init__("EvidenceValidationModel")
        
        # Configuration with deterministic defaults
        self.validation_threshold = 0.6
        self.confidence_level = 0.95
        self.max_validation_time_ms = 5000
        
        # Validation rules with deterministic ordering
        self.validation_rules = self._initialize_validation_rules()
        self.dnp_alignment_criteria = self._initialize_dnp_criteria()
        
        # Statistics tracking
        self.validation_stats = {
            "requests_processed": 0,
            "valid_evidences": 0,
            "invalid_evidences": 0,
            "avg_validation_score": 0.0,
            "avg_processing_time_ms": 0.0,
        }
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "confidence_level": self.confidence_level,
            "max_validation_time_ms": self.max_validation_time_ms,
            "rules_count": len(self.validation_rules),
            "validation_threshold": self.validation_threshold,
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules with deterministic ordering"""
        rules = {
            "bias_detection": {
                "category": ValidationCategory.BIAS_DETECTION,
                "description": "Detect potential bias in evidence",
                "patterns": ["parece", "probablemente", "supuestamente", "posiblemente"],
                "severity": ValidationSeverity.MEDIUM,
                "weight": 0.15,
            },
            "completeness_check": {
                "category": ValidationCategory.COMPLETENESS,
                "description": "Check if evidence provides complete information",
                "patterns": ["completo", "detallado", "específico", "preciso"],
                "severity": ValidationSeverity.HIGH,
                "weight": 0.25,
            },
            "factual_accuracy": {
                "category": ValidationCategory.FACTUAL_ACCURACY,
                "description": "Verify factual accuracy of claims",
                "patterns": ["verificado", "documentado", "comprobado", "confirmado"],
                "severity": ValidationSeverity.CRITICAL,
                "weight": 0.30,
            },
            "logical_consistency": {
                "category": ValidationCategory.LOGICAL_CONSISTENCY,
                "description": "Check logical consistency",
                "patterns": ["por tanto", "en consecuencia", "resulta", "concluye"],
                "severity": ValidationSeverity.HIGH,
                "weight": 0.20,
            },
            "relevance_check": {
                "category": ValidationCategory.RELEVANCE,
                "description": "Assess relevance to context",
                "patterns": ["relevante", "pertinente", "aplicable", "relacionado"],
                "severity": ValidationSeverity.MEDIUM,
                "weight": 0.10,
            },
        }
        
        # Ensure deterministic ordering
        return self.sort_dict_by_keys(rules)
    
    def _initialize_dnp_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Initialize DNP alignment criteria with deterministic ordering"""
        criteria = {
            "constitutional": {
                "category": DNPAlignmentCategory.CONSTITUTIONAL,
                "description": "Constitutional compliance",
                "indicators": ["constitución", "derechos fundamentales", "principios", "garantías"],
                "weight": 0.30,
            },
            "ethical": {
                "category": DNPAlignmentCategory.ETHICAL,
                "description": "Ethical compliance",
                "indicators": ["ética", "transparencia", "integridad", "responsabilidad"],
                "weight": 0.20,
            },
            "procedural": {
                "category": DNPAlignmentCategory.PROCEDURAL,
                "description": "Procedural compliance",
                "indicators": ["procedimiento", "proceso", "metodología", "protocolo"],
                "weight": 0.20,
            },
            "regulatory": {
                "category": DNPAlignmentCategory.REGULATORY,
                "description": "Regulatory compliance",
                "indicators": ["norma", "decreto", "resolución", "reglamento"],
                "weight": 0.20,
            },
            "technical": {
                "category": DNPAlignmentCategory.TECHNICAL,
                "description": "Technical compliance",
                "indicators": ["técnico", "especificación", "estándar", "criterio"],
                "weight": 0.10,
            },
        }
        
        return self.sort_dict_by_keys(criteria)
    
    def validate_evidence(self, request: EvidenceValidationRequest) -> EvidenceValidationResponse:
        """
        Validate evidence with deterministic processing.
        
        Args:
            request: Evidence validation request
            
        Returns:
            Validation response with deterministic results
        """
        start_time = time.time()
        
        # Create response
        response = EvidenceValidationResponse(request.request_id)
        
        try:
            # Perform validation checks
            validation_results = self._perform_validation_checks_deterministic(
                request.evidence_text, request.context
            )
            
            # Calculate validation score
            validation_score = self._calculate_validation_score_deterministic(validation_results)
            
            # Check DNP alignment
            dnp_score = self._check_dnp_alignment_deterministic(request.evidence_text)
            
            # Determine if valid
            is_valid = validation_score >= self.validation_threshold
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval_deterministic(
                validation_score, len(request.evidence_text)
            )
            
            # Update response
            response.validation_score = validation_score
            response.is_valid = is_valid
            response.validation_issues = validation_results.get("issues", [])
            response.dnp_alignment_score = dnp_score
            response.confidence_interval = confidence_interval
            
            # Update statistics
            self._update_validation_stats(response, start_time)
            
        except Exception as e:
            # Handle errors deterministically
            response.validation_score = 0.0
            response.is_valid = False
            response.validation_issues = [{
                "category": "system_error",
                "description": f"Validation error: {str(e)}",
                "severity": "critical"
            }]
        
        # Calculate processing time
        response.processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Update state hash
        response.update_state_hash(response._get_response_state())
        
        return response
    
    def _perform_validation_checks_deterministic(self, evidence_text: str, context: str) -> Dict[str, Any]:
        """Perform validation checks with deterministic results"""
        
        evidence_lower = evidence_text.lower()
        context_lower = context.lower()
        
        validation_results = {
            "issues": [],
            "rule_scores": {},
            "overall_score": 0.0,
        }
        
        # Apply each validation rule
        total_weighted_score = 0.0
        total_weights = 0.0
        
        for rule_name, rule_config in sorted(self.validation_rules.items()):
            # Calculate rule score
            rule_score = self._calculate_rule_score_deterministic(
                evidence_lower, rule_config["patterns"]
            )
            
            validation_results["rule_scores"][rule_name] = rule_score
            
            # Weight the score
            weight = rule_config["weight"]
            total_weighted_score += rule_score * weight
            total_weights += weight
            
            # Generate issues if score is low
            if rule_score < 0.5:
                issue = {
                    "category": rule_config["category"].value,
                    "description": f"Low score for {rule_config['description']}: {rule_score:.2f}",
                    "rule_name": rule_name,
                    "score": rule_score,
                    "severity": rule_config["severity"].value,
                }
                validation_results["issues"].append(issue)
        
        # Calculate overall score
        if total_weights > 0:
            validation_results["overall_score"] = total_weighted_score / total_weights
        
        # Sort issues for deterministic ordering
        validation_results["issues"] = sorted(
            validation_results["issues"], 
            key=lambda x: (x["severity"], x["rule_name"])
        )
        
        return validation_results
    
    def _calculate_rule_score_deterministic(self, text: str, patterns: List[str]) -> float:
        """Calculate score for a validation rule deterministically"""
        if not patterns:
            return 0.5  # Neutral score for rules without patterns
        
        # Count pattern matches
        matches = 0
        for pattern in sorted(patterns):  # Sort for deterministic ordering
            if pattern in text:
                matches += 1
        
        # Calculate normalized score
        score = matches / len(patterns)
        
        # Apply text length adjustment
        text_length = len(text)
        if text_length > 100:
            score += 0.1  # Bonus for longer text
        elif text_length < 20:
            score -= 0.2  # Penalty for very short text
        
        # Clamp score
        return max(0.0, min(1.0, score))
    
    def _calculate_validation_score_deterministic(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score deterministically"""
        
        base_score = validation_results.get("overall_score", 0.5)
        
        # Adjust based on number of issues
        issue_count = len(validation_results.get("issues", []))
        issue_penalty = min(0.3, issue_count * 0.1)  # Max penalty of 0.3
        
        # Adjust based on issue severity
        critical_issues = sum(
            1 for issue in validation_results.get("issues", [])
            if issue.get("severity") == "critical"
        )
        critical_penalty = critical_issues * 0.2
        
        # Calculate final score
        final_score = base_score - issue_penalty - critical_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _check_dnp_alignment_deterministic(self, evidence_text: str) -> float:
        """Check DNP alignment with deterministic scoring"""
        
        evidence_lower = evidence_text.lower()
        
        total_weighted_score = 0.0
        total_weights = 0.0
        
        # Check each DNP criterion
        for criterion_name, criterion_config in sorted(self.dnp_alignment_criteria.items()):
            # Count indicator matches
            matches = 0
            indicators = criterion_config["indicators"]
            
            for indicator in sorted(indicators):
                if indicator in evidence_lower:
                    matches += 1
            
            # Calculate criterion score
            criterion_score = matches / len(indicators) if indicators else 0.0
            
            # Weight the score
            weight = criterion_config["weight"]
            total_weighted_score += criterion_score * weight
            total_weights += weight
        
        # Calculate overall DNP alignment score
        if total_weights > 0:
            dnp_score = total_weighted_score / total_weights
        else:
            dnp_score = 0.0
        
        return max(0.0, min(1.0, dnp_score))
    
    def _calculate_confidence_interval_deterministic(self, score: float, text_length: int) -> Tuple[float, float]:
        """Calculate confidence interval deterministically"""
        
        # Base confidence width based on text length
        if text_length > 200:
            base_width = 0.05
        elif text_length > 100:
            base_width = 0.10
        elif text_length > 50:
            base_width = 0.15
        else:
            base_width = 0.20
        
        # Adjust width based on score (more extreme scores have wider intervals)
        score_factor = abs(score - 0.5) * 2  # 0 to 1 range
        width = base_width * (1 + score_factor)
        
        # Calculate bounds
        lower_bound = max(0.0, score - width / 2)
        upper_bound = min(1.0, score + width / 2)
        
        return (lower_bound, upper_bound)
    
    def _update_validation_stats(self, response: EvidenceValidationResponse, start_time: float):
        """Update validation statistics"""
        self.validation_stats["requests_processed"] += 1
        
        if response.is_valid:
            self.validation_stats["valid_evidences"] += 1
        else:
            self.validation_stats["invalid_evidences"] += 1
        
        # Update average validation score
        current_avg = self.validation_stats["avg_validation_score"]
        count = self.validation_stats["requests_processed"]
        new_avg = (current_avg * (count - 1) + response.validation_score) / count
        self.validation_stats["avg_validation_score"] = new_avg
        
        # Update average processing time
        processing_time = int((time.time() - start_time) * 1000)
        current_avg_time = self.validation_stats["avg_processing_time_ms"]
        new_avg_time = (current_avg_time * (count - 1) + processing_time) / count
        self.validation_stats["avg_processing_time_ms"] = new_avg_time
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function with deterministic output.
        
        Args:
            data: Input data containing evidence to validate
            context: Processing context
            
        Returns:
            Deterministic validation results
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract evidence text
            evidence_text = ""
            if isinstance(canonical_data, str):
                evidence_text = canonical_data
            elif isinstance(canonical_data, dict):
                evidence_text = (
                    canonical_data.get("evidence_text") or 
                    canonical_data.get("text") or 
                    canonical_data.get("evidence", "")
                )
            
            # Create validation request
            request = EvidenceValidationRequest(
                evidence_text=str(evidence_text),
                context=str(canonical_context.get("context", "") if canonical_context else ""),
                validation_type=str(canonical_context.get("validation_type", "comprehensive") if canonical_context else "comprehensive")
            )
            
            # Validate evidence
            response = self.validate_evidence(request)
            
            # Generate deterministic output
            output = self._generate_deterministic_output(request, response, operation_id)
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            error_output = {
                "component": self.component_name,
                "error": str(e),
                "operation_id": operation_id,
                "status": "error",
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _generate_deterministic_output(self, request: EvidenceValidationRequest, response: EvidenceValidationResponse, operation_id: str) -> Dict[str, Any]:
        """Generate deterministic output structure"""
        
        output = {
            "component": self.component_name,
            "component_id": self.component_id,
            "metadata": self.get_deterministic_metadata(),
            "operation_id": operation_id,
            "request": {
                "context": request.context,
                "evidence_text": request.evidence_text,
                "request_id": request.request_id,
                "request_timestamp": request.request_timestamp,
                "validation_type": request.validation_type,
            },
            "response": {
                "confidence_interval": response.confidence_interval,
                "dnp_alignment_score": response.dnp_alignment_score,
                "is_valid": response.is_valid,
                "processing_time_ms": response.processing_time_ms,
                "response_id": response.response_id,
                "response_timestamp": response.response_timestamp,
                "validation_issues": response.validation_issues,
                "validation_score": response.validation_score,
            },
            "stats": self.sort_dict_by_keys(self.validation_stats),
            "status": "success",
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(output)
    
    def save_artifact(self, data: Dict[str, Any], document_stem: str, output_dir: str = "canonical_flow/analysis") -> str:
        """
        Save evidence validation output to canonical_flow/analysis/ directory with standardized naming.
        
        Args:
            data: Evidence validation data to save
            document_stem: Base document identifier (without extension)
            output_dir: Output directory (defaults to canonical_flow/analysis)
            
        Returns:
            Path to saved artifact
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate standardized filename
            filename = f"{document_stem}_validation.json"
            artifact_path = Path(output_dir) / filename
            
            # Save with consistent JSON formatting
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"EvidenceValidationModel artifact saved: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            error_msg = f"Failed to save EvidenceValidationModel artifact to {output_dir}/{document_stem}_validation.json: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


class DNPEvidenceValidator(EvidenceValidationModel):
    """Specialized DNP Evidence Validator with enhanced alignment checking"""
    
    def __init__(self):
        super().__init__()
        self._component_name = "DNPEvidenceValidator"
        
        # Enhanced DNP-specific criteria
        self.dnp_alignment_criteria.update({
            "dnp_specific": {
                "category": DNPAlignmentCategory.REGULATORY,
                "description": "DNP-specific alignment",
                "indicators": ["pdt", "plan de desarrollo", "territorial", "dnp", "planeación nacional"],
                "weight": 0.25,
            }
        })
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())


# Backward compatibility functions
def validate_evidence_text(text: str, context: str = "") -> Dict[str, Any]:
    """Validate evidence text"""
    model = EvidenceValidationModel()
    request = EvidenceValidationRequest(text, context)
    response = model.validate_evidence(request)
    
    return {
        "is_valid": response.is_valid,
        "validation_score": response.validation_score,
        "dnp_alignment_score": response.dnp_alignment_score,
        "issues": response.validation_issues,
    }


def process(data=None, context=None):
    """Backward compatible process function"""
    model = EvidenceValidationModel()
    result = model.process(data, context)
    
    # Save artifact if document_stem is provided
    if data and isinstance(data, dict) and 'document_stem' in data:
        model.save_artifact(result, data['document_stem'])
    
    return result