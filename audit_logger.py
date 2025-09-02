"""
Comprehensive Audit Logging System

This module provides standardized audit logging across all stages with execution traces,
timing metrics, error details, and state transitions. Each component's process() method
creates audit entries with comprehensive metadata saved to canonical_flow/<stage_name>/<doc_stem>_audit.json.
"""

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AuditStatus(str, Enum):
    """Standardized audit status values"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class AuditLevel(str, Enum):
    """Audit logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StateTransition:
    """Represents a state transition during processing"""
    from_state: str
    to_state: str
    timestamp: str
    trigger: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure consistent metadata ordering
        if self.metadata:
            self.metadata = dict(sorted(self.metadata.items()))


@dataclass
class AuditEntry:
    """Complete audit entry for a component's process execution"""
    
    # Core identification
    component_name: str
    operation_id: str
    stage_name: str
    document_stem: str
    
    # Timing information
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    
    # File tracking
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    
    # Execution results
    status: AuditStatus = AuditStatus.SUCCESS
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    
    # State tracking
    state_transitions: List[StateTransition] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_environment: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure consistent ordering
        self.input_files = sorted(self.input_files)
        self.output_files = sorted(self.output_files)
        self.warnings = sorted(self.warnings)
        if self.metadata:
            self.metadata = dict(sorted(self.metadata.items()))
        if self.execution_environment:
            self.execution_environment = dict(sorted(self.execution_environment.items()))
        if self.performance_metrics:
            self.performance_metrics = dict(sorted(self.performance_metrics.items()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary for JSON serialization"""
        return {
            "component_name": self.component_name,
            "operation_id": self.operation_id,
            "stage_name": self.stage_name,
            "document_stem": self.document_stem,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input_files": self.input_files,
            "output_files": self.output_files,
            "status": self.status.value if isinstance(self.status, AuditStatus) else self.status,
            "error_count": self.error_count,
            "warnings": self.warnings,
            "state_transitions": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "timestamp": t.timestamp,
                    "trigger": t.trigger,
                    "metadata": t.metadata
                }
                for t in self.state_transitions
            ],
            "metadata": self.metadata,
            "execution_environment": self.execution_environment,
            "performance_metrics": self.performance_metrics
        }


class AuditLogger:
    """
    Comprehensive audit logging system for all stages.
    
    This class provides standardized audit logging that integrates with
    TotalOrderingBase components and existing process() APIs.
    """
    
    def __init__(self, component_name: str, stage_name: str):
        """
        Initialize audit logger for a specific component.
        
        Args:
            component_name: Name of the component being audited
            stage_name: Processing stage (e.g., 'analysis_nlp', 'G_aggregation_reporting')
        """
        self.component_name = component_name
        self.stage_name = stage_name
        
        # Current audit session
        self._current_entry: Optional[AuditEntry] = None
        self._start_time: Optional[float] = None
        
        # Configuration
        self.audit_base_dir = Path("canonical_flow")
        self.enable_performance_metrics = True
        self.enable_state_tracking = True
        
    def start_audit(self, document_stem: str, operation_id: str, 
                   input_files: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """
        Start audit logging for a process execution.
        
        Args:
            document_stem: Document identifier (used in filename)
            operation_id: Unique operation identifier
            input_files: List of input file paths
            metadata: Additional metadata to include
            
        Returns:
            Created audit entry
        """
        self._start_time = time.time()
        start_timestamp = datetime.now().isoformat()
        
        self._current_entry = AuditEntry(
            component_name=self.component_name,
            operation_id=operation_id,
            stage_name=self.stage_name,
            document_stem=document_stem,
            start_time=start_timestamp,
            input_files=input_files or [],
            metadata=metadata or {},
            execution_environment=self._get_execution_environment()
        )
        
        # Add initial state transition
        self.log_state_transition(
            from_state="initialized",
            to_state="processing",
            trigger="start_audit",
            metadata={"operation_id": operation_id}
        )
        
        return self._current_entry
    
    def end_audit(self, status: Union[AuditStatus, str] = AuditStatus.SUCCESS,
                 output_files: Optional[List[str]] = None,
                 error_details: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """
        End audit logging and save audit file.
        
        Args:
            status: Final execution status
            output_files: List of output file paths
            error_details: Error information if status is failed/partial
            
        Returns:
            Completed audit entry
        """
        if not self._current_entry:
            raise ValueError("No audit session started")
        
        # Calculate timing
        end_time = time.time()
        self._current_entry.end_time = datetime.now().isoformat()
        
        if self._start_time:
            duration_seconds = end_time - self._start_time
            self._current_entry.duration_ms = duration_seconds * 1000
        
        # Set final status and outputs
        if isinstance(status, str):
            self._current_entry.status = AuditStatus(status)
        else:
            self._current_entry.status = status
            
        if output_files:
            self._current_entry.output_files = sorted(output_files)
        
        # Add error details to metadata
        if error_details:
            self._current_entry.metadata["error_details"] = error_details
            self._current_entry.error_count = len(error_details.get("errors", []))
        
        # Add performance metrics
        if self.enable_performance_metrics and self._start_time:
            self._current_entry.performance_metrics = {
                "execution_time_seconds": end_time - self._start_time,
                "files_processed": len(self._current_entry.input_files),
                "files_generated": len(self._current_entry.output_files),
                "memory_peak_mb": self._get_memory_usage(),
                "cpu_time_seconds": self._get_cpu_time()
            }
        
        # Final state transition
        final_state = "completed" if status == AuditStatus.SUCCESS else "failed"
        self.log_state_transition(
            from_state="processing",
            to_state=final_state,
            trigger="end_audit",
            metadata={"final_status": status.value if isinstance(status, AuditStatus) else status}
        )
        
        # Save audit file
        self._save_audit_file()
        
        return self._current_entry
    
    def log_warning(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Log a warning during processing.
        
        Args:
            message: Warning message
            details: Additional warning details
        """
        if self._current_entry:
            warning_entry = message
            if details:
                warning_entry = f"{message}: {json.dumps(details, sort_keys=True)}"
            
            self._current_entry.warnings.append(warning_entry)
            self._current_entry.warnings = sorted(self._current_entry.warnings)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log an error during processing.
        
        Args:
            error: Exception that occurred
            context: Additional error context
        """
        if self._current_entry:
            self._current_entry.error_count += 1
            
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "context": context or {}
            }
            
            if "errors" not in self._current_entry.metadata:
                self._current_entry.metadata["errors"] = []
            
            self._current_entry.metadata["errors"].append(error_info)
    
    def log_state_transition(self, from_state: str, to_state: str, 
                           trigger: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a state transition during processing.
        
        Args:
            from_state: Previous state
            to_state: New state
            trigger: What triggered the transition
            metadata: Additional transition metadata
        """
        if not self.enable_state_tracking or not self._current_entry:
            return
        
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=datetime.now().isoformat(),
            trigger=trigger,
            metadata=metadata or {}
        )
        
        self._current_entry.state_transitions.append(transition)
    
    def add_metadata(self, key: str, value: Any):
        """
        Add metadata to current audit entry.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        if self._current_entry:
            self._current_entry.metadata[key] = value
            # Re-sort metadata
            self._current_entry.metadata = dict(sorted(self._current_entry.metadata.items()))
    
    def _save_audit_file(self):
        """Save audit entry to standardized audit file"""
        if not self._current_entry:
            return
        
        # Create directory structure: canonical_flow/<stage_name>/
        audit_dir = self.audit_base_dir / self.stage_name
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate audit filename: <doc_stem>_audit.json
        audit_filename = f"{self._current_entry.document_stem}_audit.json"
        audit_path = audit_dir / audit_filename
        
        try:
            # Save with consistent formatting
            audit_data = self._current_entry.to_dict()
            with open(audit_path, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False, sort_keys=True)
            
            logger.info(f"Audit log saved: {audit_path}")
            
        except Exception as e:
            logger.error(f"Failed to save audit log to {audit_path}: {e}")
    
    def _get_execution_environment(self) -> Dict[str, Any]:
        """Get current execution environment information"""
        import platform
        import sys
        import os
        
        return {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
            "user": os.getenv("USER", "unknown"),
            "environment": dict(sorted(os.environ.items())) if len(os.environ) < 50 else {"count": len(os.environ)}
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cpu_time(self) -> float:
        """Get current CPU time in seconds"""
        try:
            import psutil
            process = psutil.Process()
            return sum(process.cpu_times())
        except ImportError:
            return 0.0


class AuditMixin:
    """
    Mixin class to add audit logging capabilities to TotalOrderingBase components.
    
    This mixin provides standardized audit logging that integrates seamlessly
    with existing process() methods without disrupting current functionality.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize audit logger
        component_name = getattr(self, 'component_name', self.__class__.__name__)
        stage_name = getattr(self, 'stage_name', self._infer_stage_name())
        
        self._audit_logger = AuditLogger(component_name, stage_name)
    
    def _infer_stage_name(self) -> str:
        """Infer stage name from component location or type"""
        # Try to infer from module path
        module = self.__class__.__module__
        
        if 'analysis_nlp' in module or 'A_analysis_nlp' in module:
            return 'A_analysis_nlp'
        elif 'G_aggregation_reporting' in module:
            return 'G_aggregation_reporting'
        elif 'meso_aggregator' in module.lower():
            return 'G_aggregation_reporting'
        else:
            # Default fallback
            return 'unknown_stage'
    
    def process_with_audit(self, data: Any = None, context: Any = None, 
                          document_stem: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute process() method with comprehensive audit logging.
        
        This method wraps the existing process() method with audit logging
        without requiring changes to the original implementation.
        
        Args:
            data: Input data for processing
            context: Processing context
            document_stem: Document identifier for audit files
            
        Returns:
            Process results with audit metadata added
        """
        # Generate document stem if not provided
        if not document_stem:
            document_stem = self._generate_document_stem(data, context)
        
        # Generate operation ID
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        # Extract input files from data/context
        input_files = self._extract_input_files(data, context)
        
        # Start audit logging
        audit_entry = self._audit_logger.start_audit(
            document_stem=document_stem,
            operation_id=operation_id,
            input_files=input_files,
            metadata={
                "data_type": type(data).__name__ if data else "None",
                "context_keys": list(context.keys()) if isinstance(context, dict) else []
            }
        )
        
        try:
            # Log start of processing
            self._audit_logger.log_state_transition(
                from_state="audit_started",
                to_state="process_executing",
                trigger="process_call",
                metadata={"method": "process"}
            )
            
            # Execute original process method
            result = self.process(data, context)
            
            # Extract output files from result
            output_files = self._extract_output_files(result)
            
            # Determine status from result
            status = self._determine_status(result)
            
            # End audit with success
            self._audit_logger.end_audit(
                status=status,
                output_files=output_files
            )
            
            # Add audit metadata to result
            if isinstance(result, dict):
                result['audit'] = {
                    'operation_id': operation_id,
                    'audit_file': f"canonical_flow/{self._audit_logger.stage_name}/{document_stem}_audit.json",
                    'duration_ms': audit_entry.duration_ms,
                    'status': status.value if isinstance(status, AuditStatus) else status
                }
            
            return result
            
        except Exception as e:
            # Log error and end audit with failure
            self._audit_logger.log_error(e, {"method": "process"})
            self._audit_logger.end_audit(
                status=AuditStatus.FAILED,
                error_details={
                    "errors": [str(e)],
                    "exception_type": type(e).__name__
                }
            )
            
            # Re-raise the exception
            raise
    
    def _generate_document_stem(self, data: Any, context: Any) -> str:
        """Generate document stem from input data"""
        # Try various strategies to extract document identifier
        if isinstance(data, dict):
            # Check common document ID fields
            for field in ['document_id', 'doc_id', 'id', 'document_stem', 'filename', 'name']:
                if field in data and data[field]:
                    return str(data[field])
        
        if isinstance(context, dict):
            for field in ['document_id', 'doc_id', 'id', 'document_stem']:
                if field in context and context[field]:
                    return str(context[field])
        
        # Generate hash-based identifier
        if hasattr(self, 'generate_stable_id'):
            return self.generate_stable_id({"data": data, "context": context}, prefix="doc")[:16]
        
        # Fallback to timestamp-based ID
        from datetime import datetime
        return f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _extract_input_files(self, data: Any, context: Any) -> List[str]:
        """Extract input file paths from data and context"""
        files = []
        
        # Check data for file paths
        if isinstance(data, dict):
            for key in ['pdf_path', 'file_path', 'input_file', 'input_files']:
                if key in data:
                    value = data[key]
                    if isinstance(value, str):
                        files.append(value)
                    elif isinstance(value, list):
                        files.extend(str(f) for f in value)
        
        # Check context for file paths
        if isinstance(context, dict):
            for key in ['pdf_path', 'file_path', 'input_file', 'input_files']:
                if key in context:
                    value = context[key]
                    if isinstance(value, str):
                        files.append(value)
                    elif isinstance(value, list):
                        files.extend(str(f) for f in value)
        
        return sorted(list(set(files)))
    
    def _extract_output_files(self, result: Any) -> List[str]:
        """Extract output file paths from process result"""
        files = []
        
        if isinstance(result, dict):
            # Check for artifact files
            if 'artifacts' in result and isinstance(result['artifacts'], dict):
                files.extend(str(path) for path in result['artifacts'].values() if path)
            
            # Check for other output file fields
            for key in ['output_file', 'output_files', 'result_file', 'generated_files']:
                if key in result:
                    value = result[key]
                    if isinstance(value, str):
                        files.append(value)
                    elif isinstance(value, list):
                        files.extend(str(f) for f in value)
        
        return sorted(list(set(files)))
    
    def _determine_status(self, result: Any) -> AuditStatus:
        """Determine processing status from result"""
        if isinstance(result, dict):
            status = result.get('status', 'success')
            if status in ['success', 'succeeded']:
                return AuditStatus.SUCCESS
            elif status in ['partial', 'partially_successful']:
                return AuditStatus.PARTIAL
            elif status in ['failed', 'error', 'failure']:
                return AuditStatus.FAILED
        
        # Default to success if no status indicator
        return AuditStatus.SUCCESS


# Utility functions for easy integration
def create_audit_logger(component_name: str, stage_name: str) -> AuditLogger:
    """Create an audit logger instance"""
    return AuditLogger(component_name, stage_name)


def audit_process_execution(component_name: str, stage_name: str, document_stem: str):
    """
    Decorator for adding audit logging to process methods.
    
    Usage:
        @audit_process_execution("MyComponent", "analysis_nlp", "document_123")
        def process(self, data=None, context=None):
            # Processing logic here
            return result
    """
    def decorator(func):
        def wrapper(self, data=None, context=None, **kwargs):
            audit_logger = AuditLogger(component_name, stage_name)
            
            # Generate operation ID if component has the capability
            if hasattr(self, 'generate_operation_id'):
                operation_id = self.generate_operation_id("process", {"data": data, "context": context})
            else:
                operation_id = f"{component_name}_{int(time.time())}"
            
            # Start audit
            audit_logger.start_audit(
                document_stem=document_stem,
                operation_id=operation_id,
                metadata={
                    "function": func.__name__,
                    "component": component_name
                }
            )
            
            try:
                result = func(self, data, context, **kwargs)
                audit_logger.end_audit(status=AuditStatus.SUCCESS)
                return result
            except Exception as e:
                audit_logger.log_error(e)
                audit_logger.end_audit(status=AuditStatus.FAILED)
                raise
        
        return wrapper
    return decorator