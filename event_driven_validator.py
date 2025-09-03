"""
Event-Driven Validator Components
================================

Refactored validator components that subscribe to validation events and emit
verdict events through the event bus system, eliminating direct coupling
with orchestrators and pipeline stages.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from event_bus import EventBus, event_handler, subscribe_decorated_handlers
from event_schemas import (
    EventType, BaseEvent, ValidationEventData, VerdictEventData,
    ValidationRequestedEvent, ValidationCompletedEvent, ValidationFailedEvent,
    create_validation_requested_event, create_verdict_issued_event
)


logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation operation"""
    
    def __init__(self, 
                 passed: bool,
                 confidence: float,
                 verdict: str = None,
                 evidence: List[Dict[str, Any]] = None,
                 recommendations: List[str] = None,
                 error_details: Dict[str, Any] = None):
        self.passed = passed
        self.confidence = confidence
        self.verdict = verdict or ("PASS" if passed else "FAIL")
        self.evidence = evidence or []
        self.recommendations = recommendations or []
        self.error_details = error_details or {}
        self.timestamp = datetime.utcnow()


class BaseEventValidator(ABC):
    """
    Abstract base class for event-driven validators.
    Validators subscribe to validation request events and emit verdict events.
    """
    
    def __init__(self, event_bus: EventBus, validator_name: str, validator_type: str):
        self.event_bus = event_bus
        self.validator_name = validator_name
        self.validator_type = validator_type
        
        # Validation statistics
        self.validations_performed = 0
        self.validations_passed = 0
        self.validations_failed = 0
        self.last_validation_at: Optional[datetime] = None
        
        # Subscribe to validation events
        self._subscription_ids = subscribe_decorated_handlers(
            self.event_bus, self, self.validator_name
        )
        
        logger.info(f"Validator '{self.validator_name}' initialized and subscribed to events")
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any], validation_rules: List[str]) -> ValidationResult:
        """
        Perform validation on the provided data.
        
        Args:
            data: Data to validate
            validation_rules: List of validation rules to apply
            
        Returns:
            ValidationResult with verdict and details
        """
        pass
    
    @event_handler(EventType.VALIDATION_REQUESTED, priority=0)
    def handle_validation_request(self, event: BaseEvent):
        """Handle validation request events"""
        if not isinstance(event, ValidationRequestedEvent):
            return
        
        # Check if this validator should handle this request
        if not self._should_handle_validation(event.data):
            return
        
        validation_data = event.data
        
        try:
            # Perform validation
            result = self.validate_data(
                data=validation_data.input_data,
                validation_rules=validation_data.validation_rules
            )
            
            # Update statistics
            self.validations_performed += 1
            self.last_validation_at = datetime.utcnow()
            
            if result.passed:
                self.validations_passed += 1
            else:
                self.validations_failed += 1
            
            # Create validation completed event
            completed_data = ValidationEventData(
                validator_name=self.validator_name,
                validator_type=self.validator_type,
                validation_target=validation_data.validation_target,
                validation_rules=validation_data.validation_rules,
                input_data=validation_data.input_data,
                validation_result={
                    'passed': result.passed,
                    'verdict': result.verdict,
                    'confidence': result.confidence,
                    'evidence': result.evidence,
                    'recommendations': result.recommendations,
                    'timestamp': result.timestamp.isoformat()
                },
                confidence_score=result.confidence
            )
            
            completed_event = ValidationCompletedEvent(
                data=completed_data,
                source=self.validator_name,
                correlation_id=event.correlation_id
            )
            
            self.event_bus.publish(completed_event)
            
            # Emit verdict event
            self._emit_verdict(result, event.correlation_id)
            
            logger.info(f"Validation completed by {self.validator_name}: {result.verdict}")
            
        except Exception as e:
            # Handle validation errors
            self.validations_performed += 1
            self.validations_failed += 1
            self.last_validation_at = datetime.utcnow()
            
            error_details = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'validator_name': self.validator_name
            }
            
            # Create validation failed event
            failed_data = ValidationEventData(
                validator_name=self.validator_name,
                validator_type=self.validator_type,
                validation_target=validation_data.validation_target,
                validation_rules=validation_data.validation_rules,
                input_data=validation_data.input_data,
                error_details=error_details
            )
            
            failed_event = ValidationFailedEvent(
                data=failed_data,
                source=self.validator_name,
                correlation_id=event.correlation_id
            )
            
            self.event_bus.publish(failed_event)
            
            logger.error(f"Validation failed in {self.validator_name}: {e}")
    
    def _should_handle_validation(self, validation_data: ValidationEventData) -> bool:
        """
        Determine if this validator should handle a validation request.
        Override this method to implement validation routing logic.
        
        Args:
            validation_data: Validation request data
            
        Returns:
            True if this validator should handle the request
        """
        # Default implementation: handle all requests for this validator type
        return (validation_data.validator_type == self.validator_type or
                validation_data.validator_name == self.validator_name)
    
    def _emit_verdict(self, result: ValidationResult, correlation_id: str):
        """Emit a verdict event based on validation result"""
        verdict_event = create_verdict_issued_event(
            validator_name=self.validator_name,
            verdict=result.verdict,
            confidence=result.confidence,
            source=self.validator_name,
            correlation_id=correlation_id,
            evidence=result.evidence,
            recommendations=result.recommendations,
            validation_metadata={
                'timestamp': result.timestamp.isoformat(),
                'validator_type': self.validator_type
            }
        )
        
        self.event_bus.publish(verdict_event)
    
    def request_validation(self, 
                          validation_target: str,
                          input_data: Dict[str, Any],
                          validation_rules: List[str] = None,
                          correlation_id: str = None) -> str:
        """
        Request validation by publishing a validation request event.
        This allows validators to request validation from other validators.
        
        Args:
            validation_target: Description of what is being validated
            input_data: Data to validate
            validation_rules: List of validation rules to apply
            correlation_id: Optional correlation ID
            
        Returns:
            Event ID of the published request
        """
        request_event = create_validation_requested_event(
            validator_name=self.validator_name,
            validator_type=self.validator_type,
            validation_target=validation_target,
            source=self.validator_name,
            correlation_id=correlation_id,
            input_data=input_data,
            validation_rules=validation_rules or []
        )
        
        result = self.event_bus.publish(request_event)
        return request_event.event_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics"""
        success_rate = (self.validations_passed / max(1, self.validations_performed)) * 100
        
        return {
            'validator_name': self.validator_name,
            'validator_type': self.validator_type,
            'validations_performed': self.validations_performed,
            'validations_passed': self.validations_passed,
            'validations_failed': self.validations_failed,
            'success_rate_percentage': success_rate,
            'last_validation_at': self.last_validation_at.isoformat() if self.last_validation_at else None
        }
    
    def shutdown(self):
        """Shutdown the validator and clean up resources"""
        logger.info(f"Shutting down validator '{self.validator_name}'")
        
        # Unsubscribe from events
        for sub_id in self._subscription_ids:
            self.event_bus.unsubscribe(sub_id)


class PDTDocumentValidator(BaseEventValidator):
    """Event-driven PDT document validator"""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus, "PDTDocumentValidator", "document_validator")
        
        # PDT-specific validation rules
        self.mandatory_sections = [
            "DIAGNOSTICO", "PROGRAMAS", "PRESUPUESTO", "METAS"
        ]
        self.quality_thresholds = {
            "min_completeness": 0.8,
            "min_content_length": 1000,
            "max_ocr_ratio": 0.4,
            "min_coherence": 0.6
        }
    
    def validate_data(self, data: Dict[str, Any], validation_rules: List[str]) -> ValidationResult:
        """Validate PDT document data"""
        evidence = []
        recommendations = []
        passed = True
        confidence = 1.0
        
        # Check for required document structure
        if 'sections' in data:
            sections = data['sections']
            missing_sections = [
                section for section in self.mandatory_sections
                if section not in sections
            ]
            
            if missing_sections:
                passed = False
                confidence *= 0.5
                evidence.append({
                    'type': 'missing_sections',
                    'details': {
                        'missing': missing_sections,
                        'required': self.mandatory_sections
                    }
                })
                recommendations.append(f"Add missing mandatory sections: {', '.join(missing_sections)}")
        
        # Check document quality metrics
        if 'quality_metrics' in data:
            metrics = data['quality_metrics']
            
            completeness = metrics.get('completeness_index', 0.0)
            if completeness < self.quality_thresholds['min_completeness']:
                passed = False
                confidence *= 0.7
                evidence.append({
                    'type': 'low_completeness',
                    'details': {
                        'actual': completeness,
                        'required': self.quality_thresholds['min_completeness']
                    }
                })
                recommendations.append("Improve document completeness by adding more content to existing sections")
            
            ocr_ratio = metrics.get('ocr_ratio', 0.0)
            if ocr_ratio > self.quality_thresholds['max_ocr_ratio']:
                confidence *= 0.8
                evidence.append({
                    'type': 'high_ocr_ratio',
                    'details': {
                        'actual': ocr_ratio,
                        'threshold': self.quality_thresholds['max_ocr_ratio']
                    }
                })
                recommendations.append("Consider improving document scan quality to reduce OCR dependency")
        
        # Apply custom validation rules
        for rule in validation_rules:
            rule_result = self._apply_validation_rule(rule, data)
            if not rule_result['passed']:
                passed = False
                confidence *= rule_result.get('confidence_multiplier', 0.8)
                evidence.append(rule_result['evidence'])
                recommendations.extend(rule_result.get('recommendations', []))
        
        return ValidationResult(
            passed=passed,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _apply_validation_rule(self, rule: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific validation rule"""
        if rule == "check_budget_consistency":
            return self._validate_budget_consistency(data)
        elif rule == "check_goal_alignment":
            return self._validate_goal_alignment(data)
        elif rule == "check_legal_compliance":
            return self._validate_legal_compliance(data)
        else:
            return {'passed': True}
    
    def _validate_budget_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate budget consistency within the document"""
        # Simplified validation logic
        if 'budget_data' not in data:
            return {
                'passed': False,
                'confidence_multiplier': 0.9,
                'evidence': {
                    'type': 'missing_budget_data',
                    'details': 'Budget data not found in document'
                },
                'recommendations': ['Include detailed budget information']
            }
        return {'passed': True}
    
    def _validate_goal_alignment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate alignment between goals and programs"""
        # Simplified validation logic
        if 'goals' not in data or 'programs' not in data:
            return {
                'passed': False,
                'confidence_multiplier': 0.8,
                'evidence': {
                    'type': 'incomplete_goal_program_data',
                    'details': 'Goals or programs data incomplete'
                },
                'recommendations': ['Ensure goals and programs are clearly defined and aligned']
            }
        return {'passed': True}
    
    def _validate_legal_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate legal compliance requirements"""
        # Simplified validation logic
        legal_refs = data.get('legal_references', [])
        if len(legal_refs) < 3:
            return {
                'passed': False,
                'confidence_multiplier': 0.7,
                'evidence': {
                    'type': 'insufficient_legal_references',
                    'details': f'Found {len(legal_refs)} legal references, minimum 3 required'
                },
                'recommendations': ['Add more legal references to support document validity']
            }
        return {'passed': True}


class SchemaValidator(BaseEventValidator):
    """Event-driven schema validator for data structure validation"""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus, "SchemaValidator", "schema_validator")
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_schema(self, schema_name: str, schema: Dict[str, Any]):
        """Register a validation schema"""
        self.schemas[schema_name] = schema
        logger.info(f"Registered schema '{schema_name}' in SchemaValidator")
    
    def validate_data(self, data: Dict[str, Any], validation_rules: List[str]) -> ValidationResult:
        """Validate data against registered schemas"""
        evidence = []
        recommendations = []
        passed = True
        confidence = 1.0
        
        # Determine which schema to use
        schema_name = data.get('schema_type') or (validation_rules[0] if validation_rules else 'default')
        
        if schema_name not in self.schemas:
            return ValidationResult(
                passed=False,
                confidence=0.0,
                evidence=[{
                    'type': 'unknown_schema',
                    'details': f'Schema {schema_name} not registered'
                }],
                recommendations=[f'Register schema {schema_name} before validation']
            )
        
        schema = self.schemas[schema_name]
        
        # Validate required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in data:
                passed = False
                confidence *= 0.8
                evidence.append({
                    'type': 'missing_required_field',
                    'details': {'field': field}
                })
                recommendations.append(f'Add required field: {field}')
        
        # Validate field types
        field_types = schema.get('properties', {})
        for field, type_spec in field_types.items():
            if field in data:
                if not self._validate_field_type(data[field], type_spec):
                    passed = False
                    confidence *= 0.9
                    evidence.append({
                        'type': 'invalid_field_type',
                        'details': {
                            'field': field,
                            'expected_type': type_spec.get('type'),
                            'actual_value': str(data[field])[:100]
                        }
                    })
                    recommendations.append(f'Fix data type for field: {field}')
        
        return ValidationResult(
            passed=passed,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _validate_field_type(self, value: Any, type_spec: Dict[str, Any]) -> bool:
        """Validate field type according to schema specification"""
        expected_type = type_spec.get('type')
        
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        
        return True  # Unknown type, pass validation


class QualityAssuranceValidator(BaseEventValidator):
    """Event-driven quality assurance validator"""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus, "QualityAssuranceValidator", "quality_validator")
        
        # Quality thresholds
        self.quality_thresholds = {
            'content_quality_min': 0.7,
            'structural_coherence_min': 0.6,
            'completeness_min': 0.8,
            'consistency_min': 0.75
        }
    
    def validate_data(self, data: Dict[str, Any], validation_rules: List[str]) -> ValidationResult:
        """Validate data quality metrics"""
        evidence = []
        recommendations = []
        passed = True
        confidence = 1.0
        
        # Check overall quality score
        overall_quality = data.get('overall_quality_score', 0.0)
        if overall_quality < self.quality_thresholds['content_quality_min']:
            passed = False
            confidence *= 0.6
            evidence.append({
                'type': 'low_overall_quality',
                'details': {
                    'score': overall_quality,
                    'threshold': self.quality_thresholds['content_quality_min']
                }
            })
            recommendations.append('Improve overall content quality')
        
        # Check structural metrics
        if 'structural_metrics' in data:
            structural = data['structural_metrics']
            
            coherence = structural.get('coherence', 0.0)
            if coherence < self.quality_thresholds['structural_coherence_min']:
                confidence *= 0.8
                evidence.append({
                    'type': 'low_structural_coherence',
                    'details': {
                        'score': coherence,
                        'threshold': self.quality_thresholds['structural_coherence_min']
                    }
                })
                recommendations.append('Improve document structure and organization')
        
        # Check completeness metrics
        completeness = data.get('completeness_score', 0.0)
        if completeness < self.quality_thresholds['completeness_min']:
            passed = False
            confidence *= 0.7
            evidence.append({
                'type': 'incomplete_content',
                'details': {
                    'score': completeness,
                    'threshold': self.quality_thresholds['completeness_min']
                }
            })
            recommendations.append('Add more comprehensive content to improve completeness')
        
        return ValidationResult(
            passed=passed,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _should_handle_validation(self, validation_data: ValidationEventData) -> bool:
        """Override to handle quality-related validations"""
        return (super()._should_handle_validation(validation_data) or
                'quality' in validation_data.validation_target.lower() or
                any('quality' in rule.lower() for rule in validation_data.validation_rules))


# ============================================================================
# VALIDATOR REGISTRY AND FACTORY
# ============================================================================

class ValidatorRegistry:
    """Registry for managing event-driven validators"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.validators: Dict[str, BaseEventValidator] = {}
    
    def register_validator(self, validator: BaseEventValidator):
        """Register a validator instance"""
        self.validators[validator.validator_name] = validator
        logger.info(f"Registered validator: {validator.validator_name}")
    
    def create_default_validators(self):
        """Create and register default validator instances"""
        # Create PDT document validator
        pdt_validator = PDTDocumentValidator(self.event_bus)
        self.register_validator(pdt_validator)
        
        # Create schema validator
        schema_validator = SchemaValidator(self.event_bus)
        self.register_validator(schema_validator)
        
        # Create quality assurance validator
        qa_validator = QualityAssuranceValidator(self.event_bus)
        self.register_validator(qa_validator)
        
        logger.info("Created and registered default validators")
    
    def get_validator(self, validator_name: str) -> Optional[BaseEventValidator]:
        """Get a validator by name"""
        return self.validators.get(validator_name)
    
    def list_validators(self) -> List[str]:
        """List all registered validator names"""
        return list(self.validators.keys())
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all registered validators"""
        return {
            name: validator.get_statistics()
            for name, validator in self.validators.items()
        }
    
    def shutdown_all(self):
        """Shutdown all registered validators"""
        logger.info("Shutting down all validators")
        for validator in self.validators.values():
            validator.shutdown()
        self.validators.clear()