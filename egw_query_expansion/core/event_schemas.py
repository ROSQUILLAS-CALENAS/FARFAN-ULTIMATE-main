"""
Event Schema Definitions for EGW Query Expansion Pipeline

Defines typed events for pipeline stage transitions, validator verdicts,
and orchestrator communications to enable loose coupling while maintaining
type safety and data contracts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time

from .event_bus import BaseEvent, EventMetadata


class PipelineStage(Enum):
    """Pipeline stage identifiers"""
    INGESTION = "ingestion"
    ANALYSIS = "analysis"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    CLASSIFICATION = "classification"
    AGGREGATION = "aggregation"
    SEARCH_RETRIEVAL = "search_retrieval"
    SYNTHESIS = "synthesis"
    INTEGRATION = "integration"
    CONTEXT_CONSTRUCTION = "context_construction"
    ORCHESTRATION = "orchestration"


class ValidationOutcome(Enum):
    """Validation result outcomes"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class OperationType(Enum):
    """Types of operations in the pipeline"""
    TRANSFORM = "transform"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    ROUTE = "route"
    ENHANCE = "enhance"
    SCORE = "score"


@dataclass
class PipelineContext:
    """Context data passed between pipeline stages"""
    query: str
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationPayload:
    """Payload for validation events"""
    validator_id: str
    validation_type: str
    data_to_validate: Any
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    context: Optional[PipelineContext] = None
    required_confidence: float = 0.8


@dataclass
class ValidationResult:
    """Result of validation operation"""
    validator_id: str
    outcome: ValidationOutcome
    confidence_score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


# Pipeline Stage Transition Events

class StageStartedEvent(BaseEvent):
    """Event emitted when a pipeline stage starts"""
    
    def __init__(self, stage: PipelineStage, context: PipelineContext, 
                 operation_type: OperationType = OperationType.TRANSFORM,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'stage': stage,
                'context': context,
                'operation_type': operation_type
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "pipeline.stage.started"
    
    @property
    def stage(self) -> PipelineStage:
        return self.data['stage']
    
    @property
    def context(self) -> PipelineContext:
        return self.data['context']
    
    @property
    def operation_type(self) -> OperationType:
        return self.data['operation_type']


class StageCompletedEvent(BaseEvent):
    """Event emitted when a pipeline stage completes"""
    
    def __init__(self, stage: PipelineStage, context: PipelineContext,
                 results: Any, success: bool = True,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'stage': stage,
                'context': context,
                'results': results,
                'success': success
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "pipeline.stage.completed"
    
    @property
    def stage(self) -> PipelineStage:
        return self.data['stage']
    
    @property
    def context(self) -> PipelineContext:
        return self.data['context']
    
    @property
    def results(self) -> Any:
        return self.data['results']
    
    @property
    def success(self) -> bool:
        return self.data['success']


class StageFailedEvent(BaseEvent):
    """Event emitted when a pipeline stage fails"""
    
    def __init__(self, stage: PipelineStage, context: PipelineContext,
                 error: str, exception_details: Dict[str, Any] = None,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'stage': stage,
                'context': context,
                'error': error,
                'exception_details': exception_details or {}
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "pipeline.stage.failed"
    
    @property
    def stage(self) -> PipelineStage:
        return self.data['stage']
    
    @property
    def context(self) -> PipelineContext:
        return self.data['context']
    
    @property
    def error(self) -> str:
        return self.data['error']
    
    @property
    def exception_details(self) -> Dict[str, Any]:
        return self.data['exception_details']


# Data Flow Events

class DataTransformEvent(BaseEvent):
    """Event for data transformation between stages"""
    
    def __init__(self, source_stage: PipelineStage, target_stage: PipelineStage,
                 data: Any, transformation_type: str = "default",
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'source_stage': source_stage,
                'target_stage': target_stage,
                'data': data,
                'transformation_type': transformation_type
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "pipeline.data.transform"
    
    @property
    def source_stage(self) -> PipelineStage:
        return self.data['source_stage']
    
    @property
    def target_stage(self) -> PipelineStage:
        return self.data['target_stage']
    
    @property
    def transform_data(self) -> Any:
        return self.data['data']
    
    @property
    def transformation_type(self) -> str:
        return self.data['transformation_type']


# Validation Events

class ValidationRequestedEvent(BaseEvent):
    """Event requesting validation of data or state"""
    
    def __init__(self, payload: ValidationPayload,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(data=payload, metadata=metadata)
    
    @property
    def event_type(self) -> str:
        return "validation.requested"
    
    @property
    def payload(self) -> ValidationPayload:
        return self.data


class ValidationCompletedEvent(BaseEvent):
    """Event emitted when validation completes"""
    
    def __init__(self, result: ValidationResult,
                 original_request: Optional[ValidationPayload] = None,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'result': result,
                'original_request': original_request
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "validation.completed"
    
    @property
    def result(self) -> ValidationResult:
        return self.data['result']
    
    @property
    def original_request(self) -> Optional[ValidationPayload]:
        return self.data['original_request']


# Orchestrator Control Events

class OrchestratorCommandEvent(BaseEvent):
    """Command events for orchestrator control"""
    
    def __init__(self, command: str, parameters: Dict[str, Any] = None,
                 target_stage: Optional[PipelineStage] = None,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'command': command,
                'parameters': parameters or {},
                'target_stage': target_stage
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "orchestrator.command"
    
    @property
    def command(self) -> str:
        return self.data['command']
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return self.data['parameters']
    
    @property
    def target_stage(self) -> Optional[PipelineStage]:
        return self.data['target_stage']


class PipelineStateChangedEvent(BaseEvent):
    """Event for pipeline state changes"""
    
    def __init__(self, previous_state: str, new_state: str,
                 affected_stages: List[PipelineStage] = None,
                 state_data: Dict[str, Any] = None,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'previous_state': previous_state,
                'new_state': new_state,
                'affected_stages': affected_stages or [],
                'state_data': state_data or {}
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "pipeline.state.changed"
    
    @property
    def previous_state(self) -> str:
        return self.data['previous_state']
    
    @property
    def new_state(self) -> str:
        return self.data['new_state']
    
    @property
    def affected_stages(self) -> List[PipelineStage]:
        return self.data['affected_stages']
    
    @property
    def state_data(self) -> Dict[str, Any]:
        return self.data['state_data']


# Error and Recovery Events

class ErrorEvent(BaseEvent):
    """Event for error reporting across the pipeline"""
    
    def __init__(self, error_type: str, error_message: str,
                 stage: Optional[PipelineStage] = None,
                 error_context: Dict[str, Any] = None,
                 severity: str = "error",
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'error_type': error_type,
                'error_message': error_message,
                'stage': stage,
                'error_context': error_context or {},
                'severity': severity
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "pipeline.error"
    
    @property
    def error_type(self) -> str:
        return self.data['error_type']
    
    @property
    def error_message(self) -> str:
        return self.data['error_message']
    
    @property
    def stage(self) -> Optional[PipelineStage]:
        return self.data['stage']
    
    @property
    def error_context(self) -> Dict[str, Any]:
        return self.data['error_context']
    
    @property
    def severity(self) -> str:
        return self.data['severity']


class RecoveryEvent(BaseEvent):
    """Event for error recovery actions"""
    
    def __init__(self, recovery_action: str, failed_stage: PipelineStage,
                 recovery_data: Dict[str, Any] = None,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'recovery_action': recovery_action,
                'failed_stage': failed_stage,
                'recovery_data': recovery_data or {}
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "pipeline.recovery"
    
    @property
    def recovery_action(self) -> str:
        return self.data['recovery_action']
    
    @property
    def failed_stage(self) -> PipelineStage:
        return self.data['failed_stage']
    
    @property
    def recovery_data(self) -> Dict[str, Any]:
        return self.data['recovery_data']


# Performance Monitoring Events

class PerformanceMetricEvent(BaseEvent):
    """Event for performance metric reporting"""
    
    def __init__(self, stage: PipelineStage, metric_name: str,
                 metric_value: float, metric_unit: str = "",
                 additional_metrics: Dict[str, float] = None,
                 metadata: Optional[EventMetadata] = None):
        super().__init__(
            data={
                'stage': stage,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'metric_unit': metric_unit,
                'additional_metrics': additional_metrics or {}
            },
            metadata=metadata
        )
    
    @property
    def event_type(self) -> str:
        return "pipeline.performance"
    
    @property
    def stage(self) -> PipelineStage:
        return self.data['stage']
    
    @property
    def metric_name(self) -> str:
        return self.data['metric_name']
    
    @property
    def metric_value(self) -> float:
        return self.data['metric_value']
    
    @property
    def metric_unit(self) -> str:
        return self.data['metric_unit']
    
    @property
    def additional_metrics(self) -> Dict[str, float]:
        return self.data['additional_metrics']


# Utility functions for event creation

def create_stage_started_event(stage: PipelineStage, context: PipelineContext,
                              source_id: str = "") -> StageStartedEvent:
    """Create a stage started event with metadata"""
    metadata = EventMetadata(source_id=source_id)
    return StageStartedEvent(stage, context, metadata=metadata)


def create_validation_request(validator_id: str, validation_type: str,
                            data: Any, context: Optional[PipelineContext] = None,
                            source_id: str = "") -> ValidationRequestedEvent:
    """Create a validation request event"""
    payload = ValidationPayload(
        validator_id=validator_id,
        validation_type=validation_type,
        data_to_validate=data,
        context=context
    )
    metadata = EventMetadata(source_id=source_id)
    return ValidationRequestedEvent(payload, metadata=metadata)


def create_error_event(error_type: str, message: str,
                      stage: Optional[PipelineStage] = None,
                      source_id: str = "") -> ErrorEvent:
    """Create an error event"""
    metadata = EventMetadata(source_id=source_id)
    return ErrorEvent(error_type, message, stage=stage, metadata=metadata)


def create_performance_event(stage: PipelineStage, metric_name: str,
                           metric_value: float, source_id: str = "") -> PerformanceMetricEvent:
    """Create a performance metric event"""
    metadata = EventMetadata(source_id=source_id)
    return PerformanceMetricEvent(stage, metric_name, metric_value, metadata=metadata)