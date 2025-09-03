"""
Typed Event Schemas for Pipeline Event Bus System
=================================================

Defines strongly-typed event classes for various pipeline operations, including
validator interactions, stage transitions, and system events.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
import json
from uuid import uuid4


class EventType(Enum):
    """Enumeration of all supported event types in the pipeline system"""
    
    # Pipeline Stage Events
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_SKIPPED = "stage_skipped"
    
    # Validator Events  
    VALIDATION_REQUESTED = "validation_requested"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"
    VERDICT_ISSUED = "verdict_issued"
    
    # Document Processing Events
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_PROCESSED = "document_processed"
    DOCUMENT_FAILED = "document_failed"
    
    # Orchestration Events
    ORCHESTRATION_STARTED = "orchestration_started"
    ORCHESTRATION_COMPLETED = "orchestration_completed"
    ORCHESTRATION_FAILED = "orchestration_failed"
    ORCHESTRATION_PAUSED = "orchestration_paused"
    ORCHESTRATION_RESUMED = "orchestration_resumed"
    
    # System Events
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_HEALTH_CHECK = "system_health_check"
    RESOURCE_ALLOCATED = "resource_allocated"
    RESOURCE_RELEASED = "resource_released"
    
    # Enhancement Events
    ENHANCEMENT_REQUESTED = "enhancement_requested"
    ENHANCEMENT_ACTIVATED = "enhancement_activated"
    ENHANCEMENT_DEACTIVATED = "enhancement_deactivated"
    ENHANCEMENT_FAILED = "enhancement_failed"


EventData = TypeVar('EventData')


@dataclass
class BaseEvent(ABC, Generic[EventData]):
    """
    Base class for all events in the pipeline system.
    Provides common event metadata and serialization capabilities.
    """
    
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: EventType = field(init=False)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    correlation_id: Optional[str] = None
    data: EventData = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            raise TypeError(f"Event class {self.__class__.__name__} must define event_type")
    
    @abstractmethod
    def validate_data(self) -> bool:
        """Validate the event data structure"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'correlation_id': self.correlation_id,
            'data': self.data,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Serialize event to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Deserialize event from dictionary"""
        event_data = data.copy()
        if 'timestamp' in event_data:
            event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
        if 'event_type' in event_data:
            event_data['event_type'] = EventType(event_data['event_type'])
        return cls(**event_data)


# ============================================================================
# STAGE EVENTS
# ============================================================================

@dataclass
class StageEventData:
    """Data structure for stage-related events"""
    stage_name: str
    stage_type: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    error_message: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None


@dataclass
class StageStartedEvent(BaseEvent[StageEventData]):
    """Event emitted when a pipeline stage starts execution"""
    event_type: EventType = field(default=EventType.STAGE_STARTED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and 
                hasattr(self.data, 'stage_name') and 
                hasattr(self.data, 'stage_type'))


@dataclass
class StageCompletedEvent(BaseEvent[StageEventData]):
    """Event emitted when a pipeline stage completes successfully"""
    event_type: EventType = field(default=EventType.STAGE_COMPLETED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and 
                hasattr(self.data, 'stage_name') and
                hasattr(self.data, 'output_data'))


@dataclass
class StageFailedEvent(BaseEvent[StageEventData]):
    """Event emitted when a pipeline stage fails"""
    event_type: EventType = field(default=EventType.STAGE_FAILED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'stage_name') and
                hasattr(self.data, 'error_message'))


# ============================================================================
# VALIDATION EVENTS
# ============================================================================

@dataclass
class ValidationEventData:
    """Data structure for validation-related events"""
    validator_name: str
    validator_type: str
    validation_target: str
    validation_rules: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    validation_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationRequestedEvent(BaseEvent[ValidationEventData]):
    """Event emitted when validation is requested"""
    event_type: EventType = field(default=EventType.VALIDATION_REQUESTED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'validator_name') and
                hasattr(self.data, 'validation_target'))


@dataclass
class ValidationCompletedEvent(BaseEvent[ValidationEventData]):
    """Event emitted when validation completes successfully"""
    event_type: EventType = field(default=EventType.VALIDATION_COMPLETED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'validator_name') and
                hasattr(self.data, 'validation_result'))


@dataclass
class ValidationFailedEvent(BaseEvent[ValidationEventData]):
    """Event emitted when validation fails"""
    event_type: EventType = field(default=EventType.VALIDATION_FAILED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'validator_name') and
                hasattr(self.data, 'error_details'))


@dataclass
class VerdictEventData:
    """Data structure for validation verdict events"""
    validator_name: str
    verdict: str  # 'PASS', 'FAIL', 'WARNING'
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerdictIssuedEvent(BaseEvent[VerdictEventData]):
    """Event emitted when a validator issues a verdict"""
    event_type: EventType = field(default=EventType.VERDICT_ISSUED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'validator_name') and
                hasattr(self.data, 'verdict') and
                hasattr(self.data, 'confidence'))


# ============================================================================
# ORCHESTRATION EVENTS  
# ============================================================================

@dataclass
class OrchestrationEventData:
    """Data structure for orchestration events"""
    orchestrator_name: str
    orchestrator_type: str
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    stage_sequence: List[str] = field(default_factory=list)
    current_stage: Optional[str] = None
    progress_percentage: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None


@dataclass
class OrchestrationStartedEvent(BaseEvent[OrchestrationEventData]):
    """Event emitted when orchestration starts"""
    event_type: EventType = field(default=EventType.ORCHESTRATION_STARTED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'orchestrator_name') and
                hasattr(self.data, 'execution_id'))


@dataclass
class OrchestrationCompletedEvent(BaseEvent[OrchestrationEventData]):
    """Event emitted when orchestration completes"""
    event_type: EventType = field(default=EventType.ORCHESTRATION_COMPLETED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'orchestrator_name') and
                hasattr(self.data, 'execution_id'))


@dataclass
class OrchestrationFailedEvent(BaseEvent[OrchestrationEventData]):
    """Event emitted when orchestration fails"""
    event_type: EventType = field(default=EventType.ORCHESTRATION_FAILED, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'orchestrator_name') and
                hasattr(self.data, 'error_details'))


# ============================================================================
# SYSTEM EVENTS
# ============================================================================

@dataclass
class SystemEventData:
    """Data structure for system events"""
    component_name: str
    event_category: str
    severity: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None


@dataclass
class SystemErrorEvent(BaseEvent[SystemEventData]):
    """Event emitted for system errors"""
    event_type: EventType = field(default=EventType.SYSTEM_ERROR, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'component_name') and
                hasattr(self.data, 'message'))


@dataclass
class SystemWarningEvent(BaseEvent[SystemEventData]):
    """Event emitted for system warnings"""
    event_type: EventType = field(default=EventType.SYSTEM_WARNING, init=False)
    
    def validate_data(self) -> bool:
        return (self.data is not None and
                hasattr(self.data, 'component_name') and
                hasattr(self.data, 'message'))


# ============================================================================
# EVENT REGISTRY
# ============================================================================

class EventRegistry:
    """Registry for all event types and their corresponding classes"""
    
    _event_classes: Dict[EventType, Type[BaseEvent]] = {
        EventType.STAGE_STARTED: StageStartedEvent,
        EventType.STAGE_COMPLETED: StageCompletedEvent,
        EventType.STAGE_FAILED: StageFailedEvent,
        EventType.VALIDATION_REQUESTED: ValidationRequestedEvent,
        EventType.VALIDATION_COMPLETED: ValidationCompletedEvent,
        EventType.VALIDATION_FAILED: ValidationFailedEvent,
        EventType.VERDICT_ISSUED: VerdictIssuedEvent,
        EventType.ORCHESTRATION_STARTED: OrchestrationStartedEvent,
        EventType.ORCHESTRATION_COMPLETED: OrchestrationCompletedEvent,
        EventType.ORCHESTRATION_FAILED: OrchestrationFailedEvent,
        EventType.SYSTEM_ERROR: SystemErrorEvent,
        EventType.SYSTEM_WARNING: SystemWarningEvent,
    }
    
    @classmethod
    def get_event_class(cls, event_type: EventType) -> Type[BaseEvent]:
        """Get the event class for a given event type"""
        return cls._event_classes.get(event_type)
    
    @classmethod
    def register_event_class(cls, event_type: EventType, event_class: Type[BaseEvent]):
        """Register a new event class"""
        cls._event_classes[event_type] = event_class
    
    @classmethod
    def create_event(cls, event_type: EventType, **kwargs) -> BaseEvent:
        """Factory method to create typed events"""
        event_class = cls.get_event_class(event_type)
        if not event_class:
            raise ValueError(f"No event class registered for event type: {event_type}")
        return event_class(**kwargs)


# ============================================================================
# EVENT BUILDER UTILITIES
# ============================================================================

class EventBuilder:
    """Builder pattern for creating complex events"""
    
    def __init__(self, event_type: EventType):
        self.event_type = event_type
        self._kwargs = {}
    
    def with_source(self, source: str) -> 'EventBuilder':
        self._kwargs['source'] = source
        return self
    
    def with_correlation_id(self, correlation_id: str) -> 'EventBuilder':
        self._kwargs['correlation_id'] = correlation_id
        return self
    
    def with_data(self, data: Any) -> 'EventBuilder':
        self._kwargs['data'] = data
        return self
    
    def with_metadata(self, **metadata) -> 'EventBuilder':
        if 'metadata' not in self._kwargs:
            self._kwargs['metadata'] = {}
        self._kwargs['metadata'].update(metadata)
        return self
    
    def build(self) -> BaseEvent:
        """Build the event with configured parameters"""
        return EventRegistry.create_event(self.event_type, **self._kwargs)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_stage_started_event(stage_name: str, stage_type: str, source: str, 
                              correlation_id: str = None, **kwargs) -> StageStartedEvent:
    """Convenience function to create stage started events"""
    data = StageEventData(stage_name=stage_name, stage_type=stage_type, **kwargs)
    return StageStartedEvent(data=data, source=source, correlation_id=correlation_id)


def create_validation_requested_event(validator_name: str, validator_type: str,
                                    validation_target: str, source: str,
                                    correlation_id: str = None, **kwargs) -> ValidationRequestedEvent:
    """Convenience function to create validation request events"""
    data = ValidationEventData(
        validator_name=validator_name,
        validator_type=validator_type,
        validation_target=validation_target,
        **kwargs
    )
    return ValidationRequestedEvent(data=data, source=source, correlation_id=correlation_id)


def create_verdict_issued_event(validator_name: str, verdict: str, confidence: float,
                               source: str, correlation_id: str = None, **kwargs) -> VerdictIssuedEvent:
    """Convenience function to create verdict events"""
    data = VerdictEventData(
        validator_name=validator_name,
        verdict=verdict,
        confidence=confidence,
        **kwargs
    )
    return VerdictIssuedEvent(data=data, source=source, correlation_id=correlation_id)