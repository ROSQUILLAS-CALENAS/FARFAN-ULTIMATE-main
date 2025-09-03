"""
Validator Adapters for Event-Driven Architecture

Provides adapter classes that wrap existing validators to work with the event bus system,
removing direct imports of pipeline stages while maintaining backward compatibility.
"""

import logging
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from .event_bus import BaseEvent, EventHandler, SynchronousEventBus, get_event_bus
from .event_schemas import (
    ValidationRequestedEvent, ValidationCompletedEvent, ValidationOutcome,
    ValidationPayload, ValidationResult, PipelineStage, StageCompletedEvent,
    StageStartedEvent, ErrorEvent
)

logger = logging.getLogger(__name__)


class ValidatorAdapter(EventHandler, ABC):
    """
    Base adapter class for wrapping existing validators into event handlers
    """
    
    def __init__(self, validator_id: str, supported_validation_types: List[str],
                 event_bus: Optional[SynchronousEventBus] = None):
        self.validator_id = validator_id
        self.supported_validation_types = supported_validation_types
        self.event_bus = event_bus or get_event_bus()
        self.validation_history: List[ValidationResult] = []
        
        # Auto-subscribe to relevant events
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Setup event subscriptions for this validator"""
        # Subscribe to validation requests
        self.event_bus.subscribe("validation.requested", self)
        
        # Subscribe to stage events that this validator should validate
        for stage in self._get_monitored_stages():
            self.event_bus.subscribe("pipeline.stage.completed", self, 
                                   self._stage_filter_factory(stage))
    
    @abstractmethod
    def _get_monitored_stages(self) -> List[PipelineStage]:
        """Return list of pipeline stages this validator monitors"""
        pass
    
    def _stage_filter_factory(self, stage: PipelineStage):
        """Create filter function for specific stage events"""
        def stage_filter(event: BaseEvent) -> bool:
            if hasattr(event, 'stage'):
                return event.stage == stage
            return False
        return stage_filter
    
    @property
    def handler_id(self) -> str:
        return self.validator_id
    
    def can_handle(self, event: BaseEvent) -> bool:
        """Check if this adapter can handle the given event"""
        if event.event_type == "validation.requested":
            payload = event.data
            return payload.validator_id == self.validator_id or \
                   payload.validation_type in self.supported_validation_types
        
        elif event.event_type == "pipeline.stage.completed":
            # Check if this stage should trigger validation
            stage = getattr(event, 'stage', None)
            return stage in self._get_monitored_stages()
        
        return False
    
    def handle(self, event: BaseEvent) -> Optional[BaseEvent]:
        """Handle the event and return validation result"""
        try:
            if event.event_type == "validation.requested":
                return self._handle_validation_request(event)
            elif event.event_type == "pipeline.stage.completed":
                return self._handle_stage_completion(event)
        except Exception as e:
            logger.error(f"Validator {self.validator_id} failed to handle event: {e}")
            # Return error event
            return ErrorEvent(
                error_type="validation_error",
                error_message=f"Validator {self.validator_id} failed: {str(e)}",
                error_context={"event_id": event.event_id, "event_type": event.event_type}
            )
        
        return None
    
    def _handle_validation_request(self, event: ValidationRequestedEvent) -> ValidationCompletedEvent:
        """Handle explicit validation request"""
        import time
        start_time = time.time()
        
        payload = event.payload
        
        try:
            # Perform the actual validation
            outcome, confidence, errors, warnings, details = self._perform_validation(
                payload.data_to_validate, payload.validation_type, payload.validation_rules
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            result = ValidationResult(
                validator_id=self.validator_id,
                outcome=outcome,
                confidence_score=confidence,
                errors=errors,
                warnings=warnings,
                details=details,
                processing_time_ms=processing_time
            )
            
            # Add to history
            self.validation_history.append(result)
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-500:]
            
            return ValidationCompletedEvent(result, payload)
            
        except Exception as e:
            # Create failed validation result
            result = ValidationResult(
                validator_id=self.validator_id,
                outcome=ValidationOutcome.FAILED,
                confidence_score=0.0,
                errors=[f"Validation failed: {str(e)}"],
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            return ValidationCompletedEvent(result, payload)
    
    def _handle_stage_completion(self, event: StageCompletedEvent) -> Optional[ValidationCompletedEvent]:
        """Handle pipeline stage completion by triggering validation"""
        if not event.success:
            return None  # Don't validate failed stages
        
        # Create validation request for stage output
        validation_type = f"stage_{event.stage.value}_output"
        
        try:
            outcome, confidence, errors, warnings, details = self._perform_validation(
                event.results, validation_type, {}
            )
            
            result = ValidationResult(
                validator_id=self.validator_id,
                outcome=outcome,
                confidence_score=confidence,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
            return ValidationCompletedEvent(result)
            
        except Exception as e:
            logger.error(f"Stage validation failed for {event.stage}: {e}")
            return None
    
    @abstractmethod
    def _perform_validation(self, data: Any, validation_type: str, 
                           validation_rules: Dict[str, Any]) -> tuple:
        """
        Perform the actual validation logic
        
        Returns:
            Tuple of (outcome, confidence, errors, warnings, details)
        """
        pass
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {"total_validations": 0}
        
        outcomes = [result.outcome for result in self.validation_history]
        
        return {
            "total_validations": len(self.validation_history),
            "passed": sum(1 for outcome in outcomes if outcome == ValidationOutcome.PASSED),
            "failed": sum(1 for outcome in outcomes if outcome == ValidationOutcome.FAILED),
            "warnings": sum(1 for outcome in outcomes if outcome == ValidationOutcome.WARNING),
            "average_confidence": sum(result.confidence_score for result in self.validation_history) / len(self.validation_history),
            "average_processing_time": sum(result.processing_time_ms for result in self.validation_history) / len(self.validation_history)
        }


class ConstraintValidatorAdapter(ValidatorAdapter):
    """Adapter for the constraint validator"""
    
    def __init__(self, event_bus: Optional[SynchronousEventBus] = None):
        super().__init__(
            validator_id="constraint_validator",
            supported_validation_types=[
                "dimension_requirements",
                "conjunctive_conditions", 
                "mandatory_indicators",
                "satisfiability"
            ],
            event_bus=event_bus
        )
        
        # Import the actual validator class only when needed
        self._validator_instance = None
    
    def _get_monitored_stages(self) -> List[PipelineStage]:
        """Constraint validator monitors orchestration and aggregation stages"""
        return [PipelineStage.ORCHESTRATION, PipelineStage.AGGREGATION]
    
    def _get_validator_instance(self):
        """Lazy initialization of the actual validator"""
        if self._validator_instance is None:
            try:
                # Import here to avoid circular dependencies
                import sys
                from pathlib import Path
                
                # Add project root to path
                project_root = Path(__file__).resolve().parents[2]
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from constraint_validator import ConstraintValidator
                self._validator_instance = ConstraintValidator()
                
            except ImportError as e:
                logger.error(f"Failed to import ConstraintValidator: {e}")
                raise
        
        return self._validator_instance
    
    def _perform_validation(self, data: Any, validation_type: str, 
                           validation_rules: Dict[str, Any]) -> tuple:
        """Perform constraint validation using the original validator"""
        validator = self._get_validator_instance()
        
        try:
            if validation_type == "dimension_requirements":
                # Extract dimension and evidence from data
                dimension = data.get("dimension", "unknown")
                evidence = data.get("evidence", [])
                
                success, proof = validator.validate_dimension_requirements(dimension, evidence)
                
                return (
                    ValidationOutcome.PASSED if success else ValidationOutcome.FAILED,
                    1.0 - proof.crc_risk_bound if proof else 0.0,
                    [] if success else [f"Dimension validation failed for {dimension}"],
                    [] if success else proof.counterexamples if proof else [],
                    {"proof": proof.__dict__ if proof else {}}
                )
            
            elif validation_type == "conjunctive_conditions":
                point = data.get("point", "")
                scores = data.get("scores", {})
                
                success, explanation = validator.check_conjunctive_conditions(point, scores)
                
                return (
                    ValidationOutcome.PASSED if success else ValidationOutcome.FAILED,
                    explanation.get("satisfied_conditions", 0) / max(explanation.get("total_conditions", 1), 1),
                    [] if success else explanation.get("failed_literals", []),
                    [],
                    explanation
                )
            
            elif validation_type == "mandatory_indicators":
                point = data.get("point", "")
                evidence = data.get("evidence", [])
                
                success, risk_bound = validator.verify_mandatory_indicators(point, evidence)
                
                return (
                    ValidationOutcome.PASSED if success else ValidationOutcome.FAILED,
                    1.0 - risk_bound,
                    [] if success else [f"Missing mandatory indicators for {point}"],
                    [],
                    {"risk_bound": risk_bound}
                )
            
            elif validation_type == "satisfiability":
                scores = data.get("scores", {})
                
                result = validator.calculate_satisfiability(scores)
                
                satisfiability = result.get("satisfiability")
                is_success = satisfiability in ["SATISFIED", "PARTIAL"]
                
                return (
                    ValidationOutcome.PASSED if is_success else ValidationOutcome.FAILED,
                    result.get("rcps_alpha_summary", {}).get("coverage", 0.0),
                    [] if is_success else [f"Unsatisfactory scores: {satisfiability}"],
                    [],
                    result
                )
            
            else:
                return (
                    ValidationOutcome.FAILED,
                    0.0,
                    [f"Unsupported validation type: {validation_type}"],
                    [],
                    {}
                )
                
        except Exception as e:
            logger.error(f"Constraint validation failed: {e}")
            return (
                ValidationOutcome.FAILED,
                0.0,
                [f"Validation error: {str(e)}"],
                [],
                {"exception": str(e)}
            )


class RubricValidatorAdapter(ValidatorAdapter):
    """Adapter for the rubric validator"""
    
    def __init__(self, event_bus: Optional[SynchronousEventBus] = None):
        super().__init__(
            validator_id="rubric_validator",
            supported_validation_types=[
                "rubric_structure",
                "weight_validation",
                "completeness_check",
                "conjunctive_config"
            ],
            event_bus=event_bus
        )
        self._validator_instance = None
    
    def _get_monitored_stages(self) -> List[PipelineStage]:
        """Rubric validator monitors classification and evaluation stages"""
        return [PipelineStage.CLASSIFICATION]
    
    def _get_validator_instance(self):
        """Lazy initialization of the rubric validator"""
        if self._validator_instance is None:
            try:
                import sys
                from pathlib import Path
                
                # Add project root to path
                project_root = Path(__file__).resolve().parents[2]
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from rubric_validator import RubricValidator
                rubric_path = data.get("rubric_path", "config/evaluation_rubric.yaml")
                self._validator_instance = RubricValidator(rubric_path)
                
            except ImportError as e:
                logger.error(f"Failed to import RubricValidator: {e}")
                raise
        
        return self._validator_instance
    
    def _perform_validation(self, data: Any, validation_type: str, 
                           validation_rules: Dict[str, Any]) -> tuple:
        """Perform rubric validation using the original validator"""
        validator = self._get_validator_instance()
        
        try:
            # Run full validation
            is_valid, errors, warnings = validator.validate()
            
            confidence = 1.0 if is_valid and not warnings else (0.5 if warnings else 0.0)
            
            outcome = ValidationOutcome.PASSED if is_valid else (
                ValidationOutcome.WARNING if warnings and not errors else ValidationOutcome.FAILED
            )
            
            return (
                outcome,
                confidence,
                errors,
                warnings,
                {
                    "validation_type": validation_type,
                    "report": validator.generate_report()
                }
            )
            
        except Exception as e:
            logger.error(f"Rubric validation failed: {e}")
            return (
                ValidationOutcome.FAILED,
                0.0,
                [f"Rubric validation error: {str(e)}"],
                [],
                {"exception": str(e)}
            )


class NormativeValidatorAdapter(ValidatorAdapter):
    """Adapter for normative validators"""
    
    def __init__(self, event_bus: Optional[SynchronousEventBus] = None):
        super().__init__(
            validator_id="normative_validator", 
            supported_validation_types=[
                "normative_compliance",
                "policy_validation",
                "regulation_check"
            ],
            event_bus=event_bus
        )
    
    def _get_monitored_stages(self) -> List[PipelineStage]:
        """Normative validator monitors ingestion and analysis stages"""
        return [PipelineStage.INGESTION, PipelineStage.ANALYSIS]
    
    def _perform_validation(self, data: Any, validation_type: str, 
                           validation_rules: Dict[str, Any]) -> tuple:
        """Perform normative validation"""
        try:
            # Placeholder implementation - would integrate with actual normative validator
            # This demonstrates the pattern without requiring the actual validator
            
            if validation_type == "normative_compliance":
                # Check basic compliance rules
                compliance_score = self._check_compliance(data, validation_rules)
                success = compliance_score >= 0.8
                
                return (
                    ValidationOutcome.PASSED if success else ValidationOutcome.FAILED,
                    compliance_score,
                    [] if success else ["Normative compliance below threshold"],
                    [] if success else ["Consider reviewing normative requirements"],
                    {"compliance_score": compliance_score}
                )
            
            else:
                return (
                    ValidationOutcome.FAILED,
                    0.0,
                    [f"Unsupported validation type: {validation_type}"],
                    [],
                    {}
                )
                
        except Exception as e:
            logger.error(f"Normative validation failed: {e}")
            return (
                ValidationOutcome.FAILED,
                0.0,
                [f"Validation error: {str(e)}"],
                [],
                {"exception": str(e)}
            )
    
    def _check_compliance(self, data: Any, rules: Dict[str, Any]) -> float:
        """Check compliance against normative rules"""
        # Simplified compliance check
        if not isinstance(data, dict):
            return 0.5
        
        # Check for required fields
        required_fields = rules.get("required_fields", [])
        present_fields = sum(1 for field in required_fields if field in data)
        
        if not required_fields:
            return 0.9  # No specific requirements
        
        return present_fields / len(required_fields)


# Registry for validator adapters
validator_registry: Dict[str, ValidatorAdapter] = {}


def register_validator_adapter(adapter: ValidatorAdapter):
    """Register a validator adapter"""
    validator_registry[adapter.validator_id] = adapter
    logger.info(f"Registered validator adapter: {adapter.validator_id}")


def get_validator_adapter(validator_id: str) -> Optional[ValidatorAdapter]:
    """Get validator adapter by ID"""
    return validator_registry.get(validator_id)


def initialize_default_adapters(event_bus: Optional[SynchronousEventBus] = None):
    """Initialize default validator adapters"""
    adapters = [
        ConstraintValidatorAdapter(event_bus),
        RubricValidatorAdapter(event_bus),
        NormativeValidatorAdapter(event_bus)
    ]
    
    for adapter in adapters:
        register_validator_adapter(adapter)
    
    logger.info(f"Initialized {len(adapters)} validator adapters")


def cleanup_validator_adapters():
    """Cleanup all validator adapters"""
    for adapter in validator_registry.values():
        # Unsubscribe from events
        if hasattr(adapter, 'event_bus'):
            for event_type in ["validation.requested", "pipeline.stage.completed"]:
                try:
                    adapter.event_bus.unsubscribe(event_type, adapter)
                except:
                    pass
    
    validator_registry.clear()
    logger.info("Cleaned up validator adapters")