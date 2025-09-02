"""
Event schemas for the event-driven choreography system
"""
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class EventStatus(str, Enum):
    """Event processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


class EventType(str, Enum):
    """Supported event types"""

    DOCUMENT_INGESTION_REQUESTED = "document.ingestion.requested"
    DOCUMENT_INGESTION_STARTED = "document.ingestion.started"
    DOCUMENT_INGESTION_COMPLETED = "document.ingestion.completed"
    DOCUMENT_INGESTION_FAILED = "document.ingestion.failed"

    OCR_REQUESTED = "ocr.requested"
    OCR_COMPLETED = "ocr.completed"
    OCR_FAILED = "ocr.failed"

    TABLE_EXTRACTION_REQUESTED = "table.extraction.requested"
    TABLE_EXTRACTION_COMPLETED = "table.extraction.completed"
    TABLE_EXTRACTION_FAILED = "table.extraction.failed"

    VALIDATION_REQUESTED = "validation.requested"
    VALIDATION_COMPLETED = "validation.completed"
    VALIDATION_FAILED = "validation.failed"

    PACKAGE_REQUESTED = "package.requested"
    PACKAGE_COMPLETED = "package.completed"
    PACKAGE_FAILED = "package.failed"

    COMPENSATION_REQUESTED = "compensation.requested"
    COMPENSATION_COMPLETED = "compensation.completed"
    COMPENSATION_FAILED = "compensation.failed"

    SYSTEM_HEARTBEAT = "system.heartbeat"
    SYSTEM_ALERT = "system.alert"


class RetryMetadata(BaseModel):
    """Metadata for retry and circuit breaker logic"""

    attempt_count: int = Field(default=0, description="Current attempt number")
    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    backoff_multiplier: float = Field(
        default=2.0, description="Exponential backoff multiplier"
    )
    initial_delay_seconds: float = Field(default=1.0, description="Initial retry delay")
    max_delay_seconds: float = Field(default=300.0, description="Maximum retry delay")
    circuit_breaker_threshold: int = Field(
        default=5, description="Circuit breaker failure threshold"
    )
    last_failure_reason: Optional[str] = Field(
        default=None, description="Last failure reason"
    )
    next_retry_at: Optional[datetime] = Field(
        default=None, description="Next retry timestamp"
    )


class TimeoutMetadata(BaseModel):
    """Timeout and orphan detection metadata"""

    timeout_seconds: int = Field(default=300, description="Event timeout in seconds")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(
        default=None, description="Event expiration timestamp"
    )
    heartbeat_interval_seconds: int = Field(
        default=30, description="Heartbeat interval"
    )
    last_heartbeat_at: Optional[datetime] = Field(
        default=None, description="Last heartbeat timestamp"
    )

    def __post_init__(self):
        """Set expiration timestamp if not provided"""
        if not self.expires_at and self.timeout_seconds:
            self.expires_at = self.created_at.replace(
                seconds=self.created_at.second + self.timeout_seconds
            )

    def is_expired(self) -> bool:
        """Check if event has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_orphaned(self) -> bool:
        """Check if event is orphaned based on heartbeat"""
        if not self.last_heartbeat_at:
            return self.is_expired()

        heartbeat_expiry = self.last_heartbeat_at.replace(
            seconds=self.last_heartbeat_at.second
            + (self.heartbeat_interval_seconds * 2)
        )
        return datetime.now(timezone.utc) > heartbeat_expiry


class EventRelationship(BaseModel):
    """Parent-child relationships between events"""

    parent_correlation_id: Optional[str] = Field(
        default=None, description="Parent event correlation ID"
    )
    root_correlation_id: Optional[str] = Field(
        default=None, description="Root event correlation ID"
    )
    child_correlation_ids: List[str] = Field(
        default_factory=list, description="Child event correlation IDs"
    )
    workflow_id: Optional[str] = Field(default=None, description="Workflow identifier")
    saga_id: Optional[str] = Field(
        default=None, description="Saga identifier for compensation"
    )

    def add_child(self, child_correlation_id: str):
        """Add a child correlation ID"""
        if child_correlation_id not in self.child_correlation_ids:
            self.child_correlation_ids.append(child_correlation_id)

    def is_root_event(self) -> bool:
        """Check if this is a root event"""
        return self.parent_correlation_id is None

    def is_leaf_event(self) -> bool:
        """Check if this is a leaf event"""
        return len(self.child_correlation_ids) == 0


class EventMetadata(BaseModel):
    """Complete event metadata"""

    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique correlation ID"
    )
    event_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique event ID"
    )
    event_type: EventType = Field(..., description="Type of event")
    event_version: str = Field(default="1.0", description="Event schema version")

    # Relationships
    relationships: EventRelationship = Field(default_factory=EventRelationship)

    # Timing and timeouts
    timeout_metadata: TimeoutMetadata = Field(default_factory=TimeoutMetadata)

    # Retry logic
    retry_metadata: RetryMetadata = Field(default_factory=RetryMetadata)

    # Status tracking
    status: EventStatus = Field(default=EventStatus.PENDING)

    # Context
    source_service: str = Field(..., description="Service that generated the event")
    destination_service: Optional[str] = Field(
        default=None, description="Target service"
    )
    trace_id: Optional[str] = Field(default=None, description="Distributed tracing ID")
    user_id: Optional[str] = Field(default=None, description="User ID for audit")
    tenant_id: Optional[str] = Field(
        default=None, description="Tenant ID for multi-tenancy"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Custom metadata
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata"
    )

    def update_status(self, new_status: EventStatus):
        """Update event status with timestamp"""
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)

        if new_status in [
            EventStatus.COMPLETED,
            EventStatus.FAILED,
            EventStatus.DEAD_LETTER,
        ]:
            self.completed_at = self.updated_at

    def increment_retry(self):
        """Increment retry count and calculate next retry time"""
        self.retry_metadata.attempt_count += 1

        delay = min(
            self.retry_metadata.initial_delay_seconds
            * (
                self.retry_metadata.backoff_multiplier
                ** (self.retry_metadata.attempt_count - 1)
            ),
            self.retry_metadata.max_delay_seconds,
        )

        self.retry_metadata.next_retry_at = datetime.now(timezone.utc).replace(
            seconds=datetime.now(timezone.utc).second + int(delay)
        )

        self.update_status(EventStatus.RETRYING)

    def can_retry(self) -> bool:
        """Check if event can be retried"""
        return (
            self.retry_metadata.attempt_count < self.retry_metadata.max_attempts
            and self.status in [EventStatus.FAILED, EventStatus.RETRYING]
        )


class BaseEvent(BaseModel):
    """Base event with metadata"""

    metadata: EventMetadata = Field(..., description="Event metadata")
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Event payload data"
    )

    def to_json(self) -> str:
        """Serialize event to JSON"""
        return json.dumps(self.model_dump(), default=str, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseEvent":
        """Deserialize event from JSON"""
        data = json.loads(json_str)
        return cls(**data)

    def create_child_event(
        self,
        event_type: EventType,
        source_service: str,
        payload: Dict[str, Any] = None,
        destination_service: str = None,
    ) -> "BaseEvent":
        """Create a child event with proper relationship tracking"""
        child_metadata = EventMetadata(
            event_type=event_type,
            source_service=source_service,
            destination_service=destination_service,
            relationships=EventRelationship(
                parent_correlation_id=self.metadata.correlation_id,
                root_correlation_id=self.metadata.relationships.root_correlation_id
                or self.metadata.correlation_id,
                workflow_id=self.metadata.relationships.workflow_id,
                saga_id=self.metadata.relationships.saga_id,
            ),
            trace_id=self.metadata.trace_id,
            user_id=self.metadata.user_id,
            tenant_id=self.metadata.tenant_id,
        )

        # Update parent's child list
        self.metadata.relationships.add_child(child_metadata.correlation_id)

        return BaseEvent(metadata=child_metadata, payload=payload or {})


# Specific event types for different operations
class DocumentIngestionEvent(BaseEvent):
    """Document ingestion event"""

    @field_validator("metadata")
    @classmethod
    def validate_ingestion_event(cls, v):
        if v.event_type not in [
            EventType.DOCUMENT_INGESTION_REQUESTED,
            EventType.DOCUMENT_INGESTION_STARTED,
            EventType.DOCUMENT_INGESTION_COMPLETED,
            EventType.DOCUMENT_INGESTION_FAILED,
        ]:
            raise ValueError(
                f"Invalid event type for DocumentIngestionEvent: {v.event_type}"
            )
        return v


class OCREvent(BaseEvent):
    """OCR processing event"""

    @field_validator("metadata")
    @classmethod
    def validate_ocr_event(cls, v):
        if v.event_type not in [
            EventType.OCR_REQUESTED,
            EventType.OCR_COMPLETED,
            EventType.OCR_FAILED,
        ]:
            raise ValueError(f"Invalid event type for OCREvent: {v.event_type}")
        return v


class TableExtractionEvent(BaseEvent):
    """Table extraction event"""

    @field_validator("metadata")
    @classmethod
    def validate_table_event(cls, v):
        if v.event_type not in [
            EventType.TABLE_EXTRACTION_REQUESTED,
            EventType.TABLE_EXTRACTION_COMPLETED,
            EventType.TABLE_EXTRACTION_FAILED,
        ]:
            raise ValueError(
                f"Invalid event type for TableExtractionEvent: {v.event_type}"
            )
        return v


class ValidationEvent(BaseEvent):
    """Document validation event"""

    @field_validator("metadata")
    @classmethod
    def validate_validation_event(cls, v):
        if v.event_type not in [
            EventType.VALIDATION_REQUESTED,
            EventType.VALIDATION_COMPLETED,
            EventType.VALIDATION_FAILED,
        ]:
            raise ValueError(f"Invalid event type for ValidationEvent: {v.event_type}")
        return v


class PackagingEvent(BaseEvent):
    """Document packaging event"""

    @field_validator("metadata")
    @classmethod
    def validate_packaging_event(cls, v):
        if v.event_type not in [
            EventType.PACKAGE_REQUESTED,
            EventType.PACKAGE_COMPLETED,
            EventType.PACKAGE_FAILED,
        ]:
            raise ValueError(f"Invalid event type for PackagingEvent: {v.event_type}")
        return v


class CompensationEvent(BaseEvent):
    """Compensation/rollback event"""

    @field_validator("metadata")
    @classmethod
    def validate_compensation_event(cls, v):
        if v.event_type not in [
            EventType.COMPENSATION_REQUESTED,
            EventType.COMPENSATION_COMPLETED,
            EventType.COMPENSATION_FAILED,
        ]:
            raise ValueError(
                f"Invalid event type for CompensationEvent: {v.event_type}"
            )
        return v


class SystemEvent(BaseEvent):
    """System-level event (heartbeat, alerts, etc.)"""

    @field_validator("metadata")
    @classmethod
    def validate_system_event(cls, v):
        if v.event_type not in [EventType.SYSTEM_HEARTBEAT, EventType.SYSTEM_ALERT]:
            raise ValueError(f"Invalid event type for SystemEvent: {v.event_type}")
        return v
