"""
Centralized Audit Logger for EGW Query Expansion Analysis Components

This module provides comprehensive audit logging capabilities for tracking execution traces,
performance metrics, and error handling across all analysis_nlp components (13A-20A).
"""

import json
import traceback
import time
import os
import sys
# # # from datetime import datetime, timezone  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from contextlib import contextmanager  # Module not found  # Module not found  # Module not found

# Try to import psutil, fallback to basic metrics if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AuditEventType(Enum):
    """Types of audit events that can be recorded."""
    COMPONENT_START = "component_start"
    COMPONENT_END = "component_end"
    COMPONENT_ERROR = "component_error"
    VALIDATION_STEP = "validation_step"
    PERFORMANCE_METRIC = "performance_metric"
    DATA_TRANSFORMATION = "data_transformation"


@dataclass
class MemoryMetrics:
    """Memory usage metrics captured during execution."""
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    percent: float  # Memory usage percentage
    available_mb: float  # Available memory in MB


@dataclass 
class PerformanceMetrics:
    """Performance metrics for component execution."""
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    cpu_percent: float = 0.0
    memory_metrics: Optional[MemoryMetrics] = None
    io_operations: Dict[str, int] = field(default_factory=dict)


@dataclass
class InputOutputSchema:
    """Schema information for component inputs and outputs."""
    data_type: str
    schema_keys: List[str] = field(default_factory=list)
    data_size_bytes: int = 0
    record_count: int = 0
    validation_status: str = "unknown"
    sample_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorDetails:
    """Detailed error information with full stack trace."""
    error_type: str
    error_message: str
    stack_trace: str
    component_context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass 
class AuditEvent:
    """Individual audit event record."""
    event_id: str
    component_code: str
    component_name: str
    event_type: AuditEventType
    timestamp: str
    performance_metrics: Optional[PerformanceMetrics] = None
    input_schema: Optional[InputOutputSchema] = None
    output_schema: Optional[InputOutputSchema] = None
    error_details: Optional[ErrorDetails] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    traceability_id: Optional[str] = None


class AuditLogger:
    """
    Centralized audit logger for tracking execution traces across analysis components.
    
    Provides automatic logging of component lifecycle events, performance metrics,
    input/output schemas, and error handling with stack traces.
    """
    
    def __init__(self, audit_file_path: Optional[str] = None):
        """Initialize the audit logger with configurable output path."""
        if audit_file_path is None:
            audit_file_path = "canonical_flow/analysis/_audit.json"
        
        self.audit_file_path = Path(audit_file_path)
        self.audit_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Component registry mapping codes to names
        self.component_registry = {
            "13A": "adaptive_analyzer",
            "14A": "question_analyzer", 
            "15A": "implementacion_mapeo",
            "16A": "evidence_processor",
            "17A": "extractor_evidencias_contextual",
            "18A": "evidence_validation_model",
            "19A": "evaluation_driven_processor",
            "20A": "dnp_alignment_adapter"
        }
        
        # Current execution context
        self.current_execution_id = None
        self.component_stack: List[str] = []
        self.events: List[AuditEvent] = []
        
        # Performance tracking
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
        
    def start_execution(self, execution_id: str) -> None:
        """Start a new execution trace."""
        self.current_execution_id = execution_id
        self.component_stack.clear()
        self.events.clear()
        
        # Log execution start event
        self._log_event(
            component_code="SYSTEM",
            component_name="pipeline_orchestrator",
            event_type=AuditEventType.COMPONENT_START,
            context_data={"execution_id": execution_id}
        )
    
    def _generate_timestamp(self) -> str:
        """Generate consistent ISO 8601 timestamp with timezone."""
        return datetime.now(timezone.utc).isoformat()
    
    def _capture_memory_metrics(self) -> MemoryMetrics:
        """Capture current memory usage metrics."""
        if PSUTIL_AVAILABLE and self.process:
            try:
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                virtual_memory = psutil.virtual_memory()
                
                return MemoryMetrics(
                    rss_mb=memory_info.rss / (1024 * 1024),
                    vms_mb=memory_info.vms / (1024 * 1024),
                    percent=memory_percent,
                    available_mb=virtual_memory.available / (1024 * 1024)
                )
            except Exception:
                # Fallback to basic metrics if psutil fails
                pass
        
        # Fallback implementation using basic system info
        return MemoryMetrics(
            rss_mb=0.0,
            vms_mb=0.0,
            percent=0.0,
            available_mb=1024.0  # Default 1GB available
        )
    
    def _analyze_data_schema(self, data: Any, sample_size: int = 3) -> InputOutputSchema:
        """Analyze data structure and generate schema information."""
        if data is None:
            return InputOutputSchema(data_type="null", record_count=0)
        
        data_type = type(data).__name__
        schema_keys = []
        data_size_bytes = 0
        record_count = 0
        sample_data = None
        
        try:
            if isinstance(data, dict):
                schema_keys = list(data.keys())
                record_count = 1
                data_size_bytes = len(json.dumps(data, default=str))
                # Create sample with limited data
                sample_data = {k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                              for i, (k, v) in enumerate(data.items()) if i < sample_size}
                
            elif isinstance(data, (list, tuple)):
                record_count = len(data)
                data_size_bytes = len(json.dumps(data, default=str))
                if data and isinstance(data[0], dict):
                    schema_keys = list(data[0].keys()) if data else []
                # Sample first few records
                sample_data = data[:sample_size] if len(data) <= sample_size else data[:sample_size]
                
            elif isinstance(data, str):
                record_count = 1
                data_size_bytes = len(data.encode('utf-8'))
                sample_data = data[:200] + "..." if len(data) > 200 else data
                
            else:
                data_size_bytes = len(str(data).encode('utf-8'))
                record_count = 1
                sample_data = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                
        except Exception as e:
            # Fallback for complex data types
            data_size_bytes = len(str(data).encode('utf-8'))
            record_count = 1
            sample_data = f"<{data_type}> - Unable to serialize: {str(e)}"
        
        return InputOutputSchema(
            data_type=data_type,
            schema_keys=schema_keys,
            data_size_bytes=data_size_bytes,
            record_count=record_count,
            validation_status="captured",
            sample_data=sample_data
        )
    
    def _log_event(self, 
                  component_code: str,
                  component_name: str,
                  event_type: AuditEventType,
                  performance_metrics: Optional[PerformanceMetrics] = None,
                  input_schema: Optional[InputOutputSchema] = None,
                  output_schema: Optional[InputOutputSchema] = None,
                  error_details: Optional[ErrorDetails] = None,
                  context_data: Optional[Dict[str, Any]] = None,
                  traceability_id: Optional[str] = None) -> str:
        """Log an individual audit event."""
        
        event_id = f"{component_code}_{len(self.events):04d}_{int(time.time() * 1000)}"
        
        event = AuditEvent(
            event_id=event_id,
            component_code=component_code,
            component_name=component_name,
            event_type=event_type,
            timestamp=self._generate_timestamp(),
            performance_metrics=performance_metrics,
            input_schema=input_schema,
            output_schema=output_schema,
            error_details=error_details,
            context_data=context_data or {},
            traceability_id=traceability_id or self.current_execution_id
        )
        
        self.events.append(event)
        return event_id
    
    @contextmanager
    def audit_component_execution(self, 
                                 component_code: str,
                                 input_data: Any = None,
                                 context: Optional[Dict[str, Any]] = None):
        """
        Context manager for automatically auditing component execution.
        
        Usage:
            with audit_logger.audit_component_execution("16A", input_data) as audit_ctx:
                result = component.process(input_data)
                audit_ctx.set_output(result)
        """
        
        component_name = self.component_registry.get(component_code, f"unknown_{component_code}")
        self.component_stack.append(component_code)
        
        # Start metrics capture
        start_time = time.time()
        start_memory = self._capture_memory_metrics()
        
        # Analyze input data
        input_schema = self._analyze_data_schema(input_data) if input_data is not None else None
        
        # Log component start
        start_event_id = self._log_event(
            component_code=component_code,
            component_name=component_name,
            event_type=AuditEventType.COMPONENT_START,
            input_schema=input_schema,
            context_data=context or {}
        )
        
        class AuditContext:
            """Context object for capturing output and additional metrics."""
            def __init__(self, logger_instance):
                self.logger = logger_instance
                self.output_data = None
                self.additional_metrics = {}
            
            def set_output(self, output_data: Any):
                """Set the output data for schema analysis."""
                self.output_data = output_data
            
            def add_metric(self, name: str, value: Any):
                """Add additional performance metric."""
                self.additional_metrics[name] = value
        
        audit_ctx = AuditContext(self)
        error_occurred = False
        error_details = None
        
        try:
            yield audit_ctx
            
        except Exception as e:
            error_occurred = True
            error_details = ErrorDetails(
                error_type=type(e).__name__,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                component_context={
                    "component_code": component_code,
                    "component_name": component_name,
                    "input_schema": input_schema.__dict__ if input_schema else None,
                    "context": context or {}
                }
            )
            
            # Log error event
            self._log_event(
                component_code=component_code,
                component_name=component_name,
                event_type=AuditEventType.COMPONENT_ERROR,
                error_details=error_details,
                context_data=context or {}
            )
            
            raise  # Re-raise the exception
            
        finally:
            # Calculate performance metrics
            end_time = time.time()
            end_memory = self._capture_memory_metrics()
            
            duration = end_time - start_time
            cpu_percent = 0.0
            
            if PSUTIL_AVAILABLE and self.process:
                try:
                    cpu_percent = self.process.cpu_percent()
                except Exception:
                    cpu_percent = 0.0
            
            performance_metrics = PerformanceMetrics(
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                cpu_percent=cpu_percent,
                memory_metrics=end_memory
            )
            
            # Add any additional metrics captured
            performance_metrics.io_operations = audit_ctx.additional_metrics
            
            # Analyze output data if available
            output_schema = None
            if audit_ctx.output_data is not None:
                output_schema = self._analyze_data_schema(audit_ctx.output_data)
            
            # Log component end event
            self._log_event(
                component_code=component_code,
                component_name=component_name,
                event_type=AuditEventType.COMPONENT_END,
                performance_metrics=performance_metrics,
                output_schema=output_schema,
                error_details=error_details if error_occurred else None,
                context_data={
                    **(context or {}),
                    "start_event_id": start_event_id,
                    "success": not error_occurred
                }
            )
            
# # #             # Remove from component stack  # Module not found  # Module not found  # Module not found
            if self.component_stack and self.component_stack[-1] == component_code:
                self.component_stack.pop()
    
    def serialize_audit_data(self) -> Dict[str, Any]:
        """Serialize all audit data to a structured format."""
        
        # Group events by component for hierarchical organization
        components = {}
        system_events = []
        
        for event in self.events:
            if event.component_code == "SYSTEM":
                system_events.append({
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp,
                    "context_data": event.context_data
                })
            else:
                component_code = event.component_code
                if component_code not in components:
                    components[component_code] = {
                        "component_name": event.component_name,
                        "events": [],
                        "metrics_summary": {
                            "total_events": 0,
                            "total_duration_seconds": 0.0,
                            "success_count": 0,
                            "error_count": 0,
                            "avg_memory_usage_mb": 0.0,
                            "max_memory_usage_mb": 0.0
                        }
                    }
                
                # Convert event to dictionary
                event_dict = {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp,
                    "traceability_id": event.traceability_id
                }
                
                # Add performance metrics if available
                if event.performance_metrics:
                    perf = event.performance_metrics
                    event_dict["performance_metrics"] = {
                        "duration_seconds": perf.duration_seconds,
                        "cpu_percent": perf.cpu_percent,
                        "memory_metrics": {
                            "rss_mb": perf.memory_metrics.rss_mb,
                            "vms_mb": perf.memory_metrics.vms_mb,
                            "percent": perf.memory_metrics.percent,
                            "available_mb": perf.memory_metrics.available_mb
                        } if perf.memory_metrics else None,
                        "io_operations": perf.io_operations
                    }
                
                # Add input/output schemas
                if event.input_schema:
                    event_dict["input_schema"] = {
                        "data_type": event.input_schema.data_type,
                        "schema_keys": event.input_schema.schema_keys,
                        "data_size_bytes": event.input_schema.data_size_bytes,
                        "record_count": event.input_schema.record_count,
                        "validation_status": event.input_schema.validation_status,
                        "sample_data": event.input_schema.sample_data
                    }
                
                if event.output_schema:
                    event_dict["output_schema"] = {
                        "data_type": event.output_schema.data_type,
                        "schema_keys": event.output_schema.schema_keys,
                        "data_size_bytes": event.output_schema.data_size_bytes,
                        "record_count": event.output_schema.record_count,
                        "validation_status": event.output_schema.validation_status,
                        "sample_data": event.output_schema.sample_data
                    }
                
                # Add error details if present
                if event.error_details:
                    event_dict["error_details"] = {
                        "error_type": event.error_details.error_type,
                        "error_message": event.error_details.error_message,
                        "stack_trace": event.error_details.stack_trace,
                        "component_context": event.error_details.component_context,
                        "recovery_attempted": event.error_details.recovery_attempted,
                        "recovery_successful": event.error_details.recovery_successful
                    }
                
                # Add context data
                if event.context_data:
                    event_dict["context_data"] = event.context_data
                
                components[component_code]["events"].append(event_dict)
                
                # Update metrics summary
                metrics = components[component_code]["metrics_summary"]
                metrics["total_events"] += 1
                
                if event.performance_metrics and event.performance_metrics.duration_seconds:
                    metrics["total_duration_seconds"] += event.performance_metrics.duration_seconds
                
                if event.event_type == AuditEventType.COMPONENT_END:
                    if event.error_details:
                        metrics["error_count"] += 1
                    else:
                        metrics["success_count"] += 1
                
                if event.performance_metrics and event.performance_metrics.memory_metrics:
                    mem = event.performance_metrics.memory_metrics
                    if metrics["avg_memory_usage_mb"] == 0.0:
                        metrics["avg_memory_usage_mb"] = mem.rss_mb
                        metrics["max_memory_usage_mb"] = mem.rss_mb
                    else:
                        # Simple running average
                        metrics["avg_memory_usage_mb"] = (metrics["avg_memory_usage_mb"] + mem.rss_mb) / 2
                        metrics["max_memory_usage_mb"] = max(metrics["max_memory_usage_mb"], mem.rss_mb)
        
        return {
            "audit_metadata": {
                "generated_timestamp": self._generate_timestamp(),
                "execution_id": self.current_execution_id,
                "total_components_invoked": len(components),
                "total_events_recorded": len(self.events),
                "audit_format_version": "1.0"
            },
            "system_events": system_events,
            "components": components,
            "execution_summary": {
                "total_execution_time": sum(
                    comp["metrics_summary"]["total_duration_seconds"] 
                    for comp in components.values()
                ),
                "components_with_errors": [
                    code for code, comp in components.items() 
                    if comp["metrics_summary"]["error_count"] > 0
                ],
                "successful_components": [
                    code for code, comp in components.items() 
                    if comp["metrics_summary"]["error_count"] == 0 and comp["metrics_summary"]["success_count"] > 0
                ]
            }
        }
    
    def save_audit_file(self) -> str:
        """Save the complete audit data to the standardized JSON file."""
        audit_data = self.serialize_audit_data()
        
        with open(self.audit_file_path, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(self.audit_file_path)
    
    def validate_audit_completeness(self, expected_components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate that the audit data contains complete execution traces.
        
        Args:
            expected_components: List of component codes that should be present
                               If None, uses all registered components
        
        Returns:
            Validation report with completeness status and missing components
        """
        if expected_components is None:
            expected_components = list(self.component_registry.keys())
        
        audit_data = self.serialize_audit_data()
        recorded_components = set(audit_data["components"].keys())
        expected_components_set = set(expected_components)
        
        missing_components = expected_components_set - recorded_components
        unexpected_components = recorded_components - expected_components_set
        
        # Check for complete traces (start and end events)
        incomplete_traces = []
        for component_code, component_data in audit_data["components"].items():
            events = component_data["events"]
            start_events = [e for e in events if e["event_type"] == "component_start"]
            end_events = [e for e in events if e["event_type"] == "component_end"]
            
            if len(start_events) != len(end_events):
                incomplete_traces.append(component_code)
        
        validation_report = {
            "audit_complete": len(missing_components) == 0 and len(incomplete_traces) == 0,
            "total_expected_components": len(expected_components),
            "total_recorded_components": len(recorded_components),
            "missing_components": list(missing_components),
            "unexpected_components": list(unexpected_components),
            "incomplete_traces": incomplete_traces,
            "components_with_errors": audit_data["execution_summary"]["components_with_errors"],
            "validation_timestamp": self._generate_timestamp(),
            "audit_file_path": str(self.audit_file_path)
        }
        
        return validation_report


# Singleton instance for global access
_global_audit_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger

def set_audit_logger(logger: AuditLogger) -> None:
    """Set a custom audit logger instance."""
    global _global_audit_logger
    _global_audit_logger = logger