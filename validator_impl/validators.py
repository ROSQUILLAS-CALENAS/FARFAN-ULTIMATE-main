"""
Concrete validator implementations

These implementations depend only on validator_api interfaces and have
no direct imports from pipeline stages.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from validator_api.interfaces import IValidator, IEvidenceValidationRequest, IEvidenceValidationResponse
from validator_api.dtos import (
    ValidationRequest,
    ValidationResponse,
    ValidationResult,
    ValidationMetrics,
    ValidationSeverity,
    ValidationCategory,
    DNPAlignmentCategory
)

logger = logging.getLogger(__name__)


class ComprehensiveValidator(IValidator):
    """Comprehensive validator that performs multiple validation checks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_types = [
            "comprehensive",
            "factual_accuracy",
            "logical_consistency", 
            "source_reliability",
            "completeness",
            "relevance"
        ]
    
    def validate(self, request: IEvidenceValidationRequest) -> IEvidenceValidationResponse:
        """Perform comprehensive validation"""
        start_time = time.time()
        
        validation_results = []
        
        # Perform different types of validation based on request
        if hasattr(request, 'validation_categories'):
            categories = request.validation_categories
        else:
            categories = [ValidationCategory.FACTUAL_ACCURACY, ValidationCategory.LOGICAL_CONSISTENCY]
        
        for category in categories:
            result = self._validate_category(request.evidence_text, category, request.context)
            validation_results.append(result)
        
        # Calculate metrics
        processing_time = int((time.time() - start_time) * 1000)
        confidence_score = self._calculate_confidence(validation_results)
        
        metrics = ValidationMetrics(
            confidence_score=confidence_score,
            processing_time_ms=processing_time
        )
        
        # Create response
        if hasattr(request, 'request_id'):
            request_id = request.request_id
        else:
            request_id = hashlib.sha256(request.evidence_text.encode()).hexdigest()[:12]
        
        return ValidationResponse(
            request_id=request_id,
            validation_results=validation_results,
            confidence_score=confidence_score,
            metrics=metrics,
            processing_metadata={
                "validator_type": "comprehensive",
                "categories_checked": [cat.value for cat in categories]
            }
        )
    
    def _validate_category(self, evidence_text: str, category: ValidationCategory, context: str) -> ValidationResult:
        """Validate a specific category"""
        
        if category == ValidationCategory.FACTUAL_ACCURACY:
            return self._validate_factual_accuracy(evidence_text, context)
        elif category == ValidationCategory.LOGICAL_CONSISTENCY:
            return self._validate_logical_consistency(evidence_text, context)
        elif category == ValidationCategory.SOURCE_RELIABILITY:
            return self._validate_source_reliability(evidence_text, context)
        elif category == ValidationCategory.COMPLETENESS:
            return self._validate_completeness(evidence_text, context)
        elif category == ValidationCategory.RELEVANCE:
            return self._validate_relevance(evidence_text, context)
        else:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                category=category,
                messages=[f"Validation for {category.value} not implemented"]
            )
    
    def _validate_factual_accuracy(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate factual accuracy"""
        # Simple heuristic-based validation
        issues = []
        
        # Check for obvious factual indicators
        if any(word in evidence_text.lower() for word in ['probably', 'maybe', 'i think', 'perhaps']):
            issues.append("Evidence contains uncertainty indicators")
        
        # Check for specific claims that need verification
        if any(word in evidence_text.lower() for word in ['statistics', 'study', 'research', 'data']):
            if not any(word in evidence_text.lower() for word in ['source', 'reference', 'according to']):
                issues.append("Statistical claims lack source references")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.MEDIUM if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            category=ValidationCategory.FACTUAL_ACCURACY,
            messages=issues if issues else ["Factual accuracy check passed"]
        )
    
    def _validate_logical_consistency(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate logical consistency"""
        issues = []
        
        # Check for contradictory statements
        contradictions = [
            ('not', 'is'),
            ('never', 'always'),
            ('impossible', 'possible'),
            ('cannot', 'can')
        ]
        
        text_lower = evidence_text.lower()
        for neg, pos in contradictions:
            if neg in text_lower and pos in text_lower:
                # Simple proximity check
                neg_pos = text_lower.find(neg)
                pos_pos = text_lower.find(pos)
                if abs(neg_pos - pos_pos) < 100:  # Within 100 characters
                    issues.append(f"Potential logical contradiction: '{neg}' and '{pos}' in close proximity")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.HIGH if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            category=ValidationCategory.LOGICAL_CONSISTENCY,
            messages=issues if issues else ["Logical consistency check passed"]
        )
    
    def _validate_source_reliability(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate source reliability"""
        issues = []
        
        # Check for source indicators
        reliable_indicators = ['published', 'peer-reviewed', 'official', 'government', 'academic']
        unreliable_indicators = ['blog', 'opinion', 'rumor', 'unverified', 'alleged']
        
        text_lower = evidence_text.lower()
        
        has_reliable = any(indicator in text_lower for indicator in reliable_indicators)
        has_unreliable = any(indicator in text_lower for indicator in unreliable_indicators)
        
        if has_unreliable:
            issues.append("Evidence contains indicators of potentially unreliable sources")
        
        if not has_reliable and ('source' not in text_lower and 'reference' not in text_lower):
            issues.append("Evidence lacks clear source indicators")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.MEDIUM if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            category=ValidationCategory.SOURCE_RELIABILITY,
            messages=issues if issues else ["Source reliability check passed"]
        )
    
    def _validate_completeness(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate completeness"""
        issues = []
        
        # Simple completeness checks
        if len(evidence_text.strip()) < 50:
            issues.append("Evidence text is very short and may be incomplete")
        
        # Check for incomplete references
        if '...' in evidence_text or '[...]' in evidence_text:
            issues.append("Evidence contains truncation indicators")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.LOW if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            category=ValidationCategory.COMPLETENESS,
            messages=issues if issues else ["Completeness check passed"]
        )
    
    def _validate_relevance(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate relevance to context"""
        issues = []
        
        if context.strip():
            # Simple keyword overlap check
            evidence_words = set(evidence_text.lower().split())
            context_words = set(context.lower().split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            evidence_words -= stop_words
            context_words -= stop_words
            
            if evidence_words and context_words:
                overlap = len(evidence_words & context_words) / len(context_words)
                if overlap < 0.1:  # Less than 10% overlap
                    issues.append("Evidence has low relevance to provided context")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.MEDIUM if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            category=ValidationCategory.RELEVANCE,
            messages=issues if issues else ["Relevance check passed"]
        )
    
    def _calculate_confidence(self, validation_results: List[ValidationResult]) -> float:
        """Calculate overall confidence score"""
        if not validation_results:
            return 0.0
        
        # Weight by severity (lower severity = higher confidence)
        severity_weights = {
            ValidationSeverity.CRITICAL: 0.0,
            ValidationSeverity.HIGH: 0.3,
            ValidationSeverity.MEDIUM: 0.6,
            ValidationSeverity.LOW: 0.8,
            ValidationSeverity.INFO: 1.0
        }
        
        total_weight = 0.0
        for result in validation_results:
            if result.is_valid:
                total_weight += severity_weights.get(result.severity, 0.5)
            else:
                total_weight += severity_weights.get(result.severity, 0.5) * 0.5  # Penalize invalid results
        
        return min(1.0, total_weight / len(validation_results))
    
    def get_supported_validation_types(self) -> List[str]:
        """Get supported validation types"""
        return self.supported_types.copy()
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the validator"""
        self.config.update(config)
        logger.info(f"ComprehensiveValidator configured with: {config}")


class DNPAlignmentValidator(IValidator):
    """Validator focused on DNP (Departamento Nacional de Planeación) alignment"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_types = ["dnp_alignment", "constitutional", "regulatory", "procedural"]
    
    def validate(self, request: IEvidenceValidationRequest) -> IEvidenceValidationResponse:
        """Perform DNP alignment validation"""
        start_time = time.time()
        
        validation_results = []
        
        # Get DNP alignment categories to check
        if hasattr(request, 'dnp_alignment_categories'):
            categories = request.dnp_alignment_categories
        else:
            categories = [DNPAlignmentCategory.REGULATORY, DNPAlignmentCategory.PROCEDURAL]
        
        for category in categories:
            result = self._validate_dnp_category(request.evidence_text, category, request.context)
            validation_results.append(result)
        
        # Calculate metrics
        processing_time = int((time.time() - start_time) * 1000)
        confidence_score = self._calculate_dnp_confidence(validation_results)
        
        metrics = ValidationMetrics(
            confidence_score=confidence_score,
            processing_time_ms=processing_time
        )
        
        # Create response
        if hasattr(request, 'request_id'):
            request_id = request.request_id
        else:
            request_id = hashlib.sha256(request.evidence_text.encode()).hexdigest()[:12]
        
        return ValidationResponse(
            request_id=request_id,
            validation_results=validation_results,
            confidence_score=confidence_score,
            metrics=metrics,
            processing_metadata={
                "validator_type": "dnp_alignment",
                "categories_checked": [cat.value for cat in categories]
            }
        )
    
    def _validate_dnp_category(self, evidence_text: str, category: DNPAlignmentCategory, context: str) -> ValidationResult:
        """Validate a specific DNP category"""
        
        if category == DNPAlignmentCategory.CONSTITUTIONAL:
            return self._validate_constitutional_alignment(evidence_text, context)
        elif category == DNPAlignmentCategory.REGULATORY:
            return self._validate_regulatory_alignment(evidence_text, context)
        elif category == DNPAlignmentCategory.PROCEDURAL:
            return self._validate_procedural_alignment(evidence_text, context)
        elif category == DNPAlignmentCategory.ETHICAL:
            return self._validate_ethical_alignment(evidence_text, context)
        elif category == DNPAlignmentCategory.TECHNICAL:
            return self._validate_technical_alignment(evidence_text, context)
        else:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                dnp_alignment_category=category,
                messages=[f"DNP validation for {category.value} not implemented"]
            )
    
    def _validate_constitutional_alignment(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate constitutional alignment"""
        issues = []
        text_lower = evidence_text.lower()
        
        # Check for constitutional compliance indicators
        constitutional_keywords = [
            'constitution', 'constitutional', 'fundamental rights', 'due process',
            'equality', 'non-discrimination', 'human rights'
        ]
        
        violation_keywords = [
            'discriminat', 'violat', 'unconstitu', 'illegal', 'prohibited'
        ]
        
        has_constitutional = any(keyword in text_lower for keyword in constitutional_keywords)
        has_violations = any(keyword in text_lower for keyword in violation_keywords)
        
        if has_violations:
            issues.append("Evidence contains indicators of potential constitutional violations")
        
        if not has_constitutional and 'legal' not in text_lower:
            issues.append("Evidence lacks constitutional or legal framework references")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.HIGH if has_violations else ValidationSeverity.MEDIUM if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            dnp_alignment_category=DNPAlignmentCategory.CONSTITUTIONAL,
            messages=issues if issues else ["Constitutional alignment check passed"]
        )
    
    def _validate_regulatory_alignment(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate regulatory alignment"""
        issues = []
        text_lower = evidence_text.lower()
        
        # Check for regulatory compliance
        regulatory_keywords = [
            'regulation', 'decree', 'law', 'statute', 'ordinance',
            'compliance', 'conformity', 'standard', 'requirement'
        ]
        
        non_compliance_keywords = [
            'non-compliant', 'violation', 'breach', 'infringement', 'unauthorized'
        ]
        
        has_regulatory = any(keyword in text_lower for keyword in regulatory_keywords)
        has_violations = any(keyword in text_lower for keyword in non_compliance_keywords)
        
        if has_violations:
            issues.append("Evidence indicates potential regulatory non-compliance")
        
        if not has_regulatory and len(evidence_text) > 100:
            issues.append("Evidence lacks regulatory framework references")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.HIGH if has_violations else ValidationSeverity.LOW if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            dnp_alignment_category=DNPAlignmentCategory.REGULATORY,
            messages=issues if issues else ["Regulatory alignment check passed"]
        )
    
    def _validate_procedural_alignment(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate procedural alignment"""
        issues = []
        text_lower = evidence_text.lower()
        
        # Check for procedural compliance
        procedural_keywords = [
            'procedure', 'process', 'protocol', 'guideline', 'methodology',
            'step', 'phase', 'stage', 'approval', 'authorization'
        ]
        
        has_procedural = any(keyword in text_lower for keyword in procedural_keywords)
        
        # Check for proper procedural language
        if 'bypass' in text_lower or 'skip' in text_lower:
            issues.append("Evidence suggests procedural shortcuts or bypasses")
        
        if not has_procedural and 'implement' in text_lower:
            issues.append("Implementation evidence lacks procedural details")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.MEDIUM if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            dnp_alignment_category=DNPAlignmentCategory.PROCEDURAL,
            messages=issues if issues else ["Procedural alignment check passed"]
        )
    
    def _validate_ethical_alignment(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate ethical alignment"""
        issues = []
        text_lower = evidence_text.lower()
        
        # Check for ethical concerns
        ethical_violations = [
            'corruption', 'bribery', 'fraud', 'conflict of interest',
            'nepotism', 'favoritism', 'unethical'
        ]
        
        ethical_indicators = [
            'transparency', 'accountability', 'integrity', 'ethical',
            'fair', 'impartial', 'objective'
        ]
        
        has_violations = any(violation in text_lower for violation in ethical_violations)
        has_ethical = any(indicator in text_lower for indicator in ethical_indicators)
        
        if has_violations:
            issues.append("Evidence contains ethical violation indicators")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.CRITICAL if has_violations else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            dnp_alignment_category=DNPAlignmentCategory.ETHICAL,
            messages=issues if issues else ["Ethical alignment check passed"]
        )
    
    def _validate_technical_alignment(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate technical alignment"""
        issues = []
        text_lower = evidence_text.lower()
        
        # Check for technical standards compliance
        technical_keywords = [
            'technical', 'specification', 'standard', 'requirement',
            'criteria', 'parameter', 'metric', 'measurement'
        ]
        
        has_technical = any(keyword in text_lower for keyword in technical_keywords)
        
        # Check for vague technical language
        vague_terms = ['somehow', 'generally', 'usually', 'typically', 'approximately']
        has_vague = any(term in text_lower for term in vague_terms)
        
        if has_vague:
            issues.append("Evidence contains vague technical language")
        
        if not has_technical and any(word in text_lower for word in ['implement', 'develop', 'build']):
            issues.append("Technical implementation lacks specific standards or criteria")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.MEDIUM if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            dnp_alignment_category=DNPAlignmentCategory.TECHNICAL,
            messages=issues if issues else ["Technical alignment check passed"]
        )
    
    def _calculate_dnp_confidence(self, validation_results: List[ValidationResult]) -> float:
        """Calculate DNP alignment confidence score"""
        if not validation_results:
            return 0.0
        
        # Weight by category importance
        category_weights = {
            DNPAlignmentCategory.CONSTITUTIONAL: 1.0,
            DNPAlignmentCategory.REGULATORY: 0.9,
            DNPAlignmentCategory.ETHICAL: 0.95,
            DNPAlignmentCategory.PROCEDURAL: 0.8,
            DNPAlignmentCategory.TECHNICAL: 0.7
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in validation_results:
            category = result.dnp_alignment_category
            weight = category_weights.get(category, 0.5)
            
            if result.is_valid:
                if result.severity == ValidationSeverity.INFO:
                    score = 1.0
                elif result.severity == ValidationSeverity.LOW:
                    score = 0.8
                elif result.severity == ValidationSeverity.MEDIUM:
                    score = 0.6
                else:
                    score = 0.3
            else:
                if result.severity == ValidationSeverity.CRITICAL:
                    score = 0.0
                elif result.severity == ValidationSeverity.HIGH:
                    score = 0.1
                else:
                    score = 0.3
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_supported_validation_types(self) -> List[str]:
        """Get supported validation types"""
        return self.supported_types.copy()
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the validator"""
        self.config.update(config)
        logger.info(f"DNPAlignmentValidator configured with: {config}")


class EvidenceValidator(IValidator):
    """General evidence validator with basic validation capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_types = ["basic", "structure", "content"]
    
    def validate(self, request: IEvidenceValidationRequest) -> IEvidenceValidationResponse:
        """Perform basic evidence validation"""
        start_time = time.time()
        
        validation_results = []
        
        # Basic structure validation
        structure_result = self._validate_structure(request.evidence_text)
        validation_results.append(structure_result)
        
        # Basic content validation
        content_result = self._validate_content(request.evidence_text, request.context)
        validation_results.append(content_result)
        
        # Calculate metrics
        processing_time = int((time.time() - start_time) * 1000)
        confidence_score = sum(1.0 for r in validation_results if r.is_valid) / len(validation_results)
        
        metrics = ValidationMetrics(
            confidence_score=confidence_score,
            processing_time_ms=processing_time
        )
        
        # Create response
        if hasattr(request, 'request_id'):
            request_id = request.request_id
        else:
            request_id = hashlib.sha256(request.evidence_text.encode()).hexdigest()[:12]
        
        return ValidationResponse(
            request_id=request_id,
            validation_results=validation_results,
            confidence_score=confidence_score,
            metrics=metrics,
            processing_metadata={
                "validator_type": "evidence",
                "checks_performed": ["structure", "content"]
            }
        )
    
    def _validate_structure(self, evidence_text: str) -> ValidationResult:
        """Validate basic structure"""
        issues = []
        
        # Check basic structure requirements
        if not evidence_text or not evidence_text.strip():
            issues.append("Evidence text is empty")
        elif len(evidence_text.strip()) < 10:
            issues.append("Evidence text is too short")
        
        # Check for basic formatting
        if evidence_text == evidence_text.upper() and len(evidence_text) > 50:
            issues.append("Evidence text appears to be all uppercase")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.HIGH if not is_valid else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            category=ValidationCategory.COMPLETENESS,
            messages=issues if issues else ["Structure validation passed"]
        )
    
    def _validate_content(self, evidence_text: str, context: str) -> ValidationResult:
        """Validate basic content"""
        issues = []
        
        # Check for placeholder or dummy content
        placeholder_indicators = [
            'lorem ipsum', 'placeholder', 'dummy text', 'sample text',
            'todo', 'tbd', 'to be determined'
        ]
        
        text_lower = evidence_text.lower()
        for indicator in placeholder_indicators:
            if indicator in text_lower:
                issues.append(f"Evidence contains placeholder content: '{indicator}'")
        
        # Check for excessive repetition
        words = evidence_text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Only count longer words
                    word_lower = word.lower().strip('.,!?;:')
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count > len(words) * 0.3:  # More than 30% repetition
                issues.append("Evidence contains excessive word repetition")
        
        is_valid = len(issues) == 0
        severity = ValidationSeverity.MEDIUM if issues else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            category=ValidationCategory.COMPLETENESS,
            messages=issues if issues else ["Content validation passed"]
        )
    
    def get_supported_validation_types(self) -> List[str]:
        """Get supported validation types"""
        return self.supported_types.copy()
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the validator"""
        self.config.update(config)
        logger.info(f"EvidenceValidator configured with: {config}")