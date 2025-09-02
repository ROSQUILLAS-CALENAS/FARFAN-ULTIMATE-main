"""
Centralized Audit Logger for G_aggregation_reporting stage.

This module provides comprehensive audit logging for the aggregation and reporting stage,
tracking component execution traces, timing metrics, and error details for debugging
and performance analysis.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import traceback


@dataclass
class ComponentTrace:
    """Individual component execution trace."""
    component_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "running"  # "running", "success", "failed"
    error_details: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    input_summary: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    memory_usage_mb: Optional[float] = None
    
    def complete_success(self, output_summary: Optional[Dict[str, Any]] = None):
        """Mark trace as successfully completed."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = "success"
        self.output_summary = output_summary
    
    def complete_failure(self, error: Exception, error_context: Optional[str] = None):
        """Mark trace as failed with error details."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = "failed"
        self.error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": error_context,
            "traceback": traceback.format_exc()
        }
    
    def add_warning(self, warning_message: str):
        """Add warning to component trace."""
        self.warnings.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": warning_message
        })


@dataclass
class AuditSession:
    """Complete audit session for a document processing run."""
    doc_stem: str
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    overall_status: str = "running"  # "running", "success", "failed"
    components: List[ComponentTrace] = field(default_factory=list)
    session_warnings: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_component_trace(self, component_name: str) -> ComponentTrace:
        """Add new component trace to session."""
        trace = ComponentTrace(
            component_name=component_name,
            start_time=time.time()
        )
        self.components.append(trace)
        return trace
    
    def complete_session(self, success: bool = True):
        """Complete the audit session."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.overall_status = "success" if success else "failed"
    
    def add_session_warning(self, warning_message: str):
        """Add session-level warning."""
        self.session_warnings.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": warning_message
        })


class AggregationAuditLogger:
    """
    Centralized audit logger for G_aggregation_reporting stage.
    
    Tracks execution traces, timing metrics, and error details for all components
    in the aggregation and reporting pipeline.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "canonical_flow/aggregation"):
        """
        Initialize audit logger.
        
        Args:
            output_dir: Directory to write audit files to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[AuditSession] = None
        
    def start_session(self, doc_stem: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new audit session for processing a document.
        
        Args:
            doc_stem: Document stem identifier
            context: Optional context information
            
        Returns:
            Session ID
        """
        session_id = f"{doc_stem}_{int(time.time())}"
        
        self.current_session = AuditSession(
            doc_stem=doc_stem,
            session_id=session_id,
            start_time=time.time(),
            context=context or {}
        )
        
        return session_id
    
    def end_session(self, success: bool = True) -> Optional[str]:
        """
        End current audit session and write audit file.
        
        Args:
            success: Whether the session completed successfully
            
        Returns:
            Path to written audit file, or None if no active session
        """
        if not self.current_session:
            return None
            
        self.current_session.complete_session(success)
        
        # Write audit file
        audit_file_path = self.output_dir / f"{self.current_session.doc_stem}_aggregation_audit.json"
        audit_data = self._serialize_session(self.current_session)
        
        try:
            with open(audit_file_path, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)
            
            # Clear current session
            self.current_session = None
            
            return str(audit_file_path)
            
        except Exception as e:
            # Try to log the error but don't fail the main process
            try:
                error_msg = f"Failed to write audit file {audit_file_path}: {e}"
                if self.current_session:
                    self.current_session.add_session_warning(error_msg)
            except:
                pass  # Avoid recursive errors
            
            return None
    
    @contextmanager
    def trace_component(self, 
                       component_name: str,
                       input_data: Optional[Any] = None,
                       capture_memory: bool = False):
        """
        Context manager for tracing component execution.
        
        Args:
            component_name: Name of component being traced
            input_data: Optional input data for summary
            capture_memory: Whether to capture memory usage
            
        Usage:
            with audit_logger.trace_component("meso_aggregator") as trace:
                # Component execution
                result = process_component()
                trace.complete_success({"records_processed": len(result)})
        """
        if not self.current_session:
            # Create a temporary session if none exists
            self.start_session("unknown_doc")
        
        trace = self.current_session.add_component_trace(component_name)
        
        # Capture input summary
        if input_data is not None:
            trace.input_summary = self._create_data_summary(input_data)
        
        # Capture memory if requested
        if capture_memory:
            trace.memory_usage_mb = self._get_memory_usage()
        
        try:
            yield trace
        except Exception as e:
            trace.complete_failure(e)
            raise
    
    def log_component_warning(self, component_name: str, warning_message: str):
        """Log a warning for a specific component."""
        if self.current_session:
            # Find the most recent trace for this component
            for trace in reversed(self.current_session.components):
                if trace.component_name == component_name:
                    trace.add_warning(warning_message)
                    return
            
            # If no trace found, add as session warning
            self.current_session.add_session_warning(
                f"{component_name}: {warning_message}"
            )
    
    def log_session_warning(self, warning_message: str):
        """Log a session-level warning."""
        if self.current_session:
            self.current_session.add_session_warning(warning_message)
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current session."""
        if not self.current_session:
            return None
            
        return {
            "doc_stem": self.current_session.doc_stem,
            "session_id": self.current_session.session_id,
            "status": self.current_session.overall_status,
            "components_count": len(self.current_session.components),
            "warnings_count": len(self.current_session.session_warnings),
            "duration_ms": self.current_session.duration_ms,
            "is_active": self.current_session.end_time is None
        }
    
    def _serialize_session(self, session: AuditSession) -> Dict[str, Any]:
        """Serialize audit session to JSON-compatible format."""
        return {
            "audit_metadata": {
                "version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "stage": "G_aggregation_reporting",
                "logger_type": "AggregationAuditLogger"
            },
            "session_info": {
                "doc_stem": session.doc_stem,
                "session_id": session.session_id,
                "start_time": datetime.fromtimestamp(session.start_time, timezone.utc).isoformat(),
                "end_time": (
                    datetime.fromtimestamp(session.end_time, timezone.utc).isoformat() 
                    if session.end_time else None
                ),
                "duration_ms": session.duration_ms,
                "overall_status": session.overall_status,
                "context": session.context
            },
            "component_traces": [
                self._serialize_component_trace(trace) for trace in session.components
            ],
            "session_warnings": session.session_warnings,
            "summary_statistics": self._calculate_session_statistics(session)
        }
    
    def _serialize_component_trace(self, trace: ComponentTrace) -> Dict[str, Any]:
        """Serialize component trace to JSON-compatible format."""
        return {
            "component_name": trace.component_name,
            "start_time": datetime.fromtimestamp(trace.start_time, timezone.utc).isoformat(),
            "end_time": (
                datetime.fromtimestamp(trace.end_time, timezone.utc).isoformat() 
                if trace.end_time else None
            ),
            "duration_ms": trace.duration_ms,
            "status": trace.status,
            "error_details": trace.error_details,
            "warnings": trace.warnings,
            "input_summary": trace.input_summary,
            "output_summary": trace.output_summary,
            "memory_usage_mb": trace.memory_usage_mb
        }
    
    def _calculate_session_statistics(self, session: AuditSession) -> Dict[str, Any]:
        """Calculate summary statistics for the session."""
        components = session.components
        
        return {
            "total_components": len(components),
            "successful_components": len([c for c in components if c.status == "success"]),
            "failed_components": len([c for c in components if c.status == "failed"]),
            "components_with_warnings": len([c for c in components if c.warnings]),
            "total_warnings": sum(len(c.warnings) for c in components) + len(session.session_warnings),
            "average_component_duration_ms": (
                sum(c.duration_ms for c in components if c.duration_ms) / len(components)
                if components and any(c.duration_ms for c in components) else None
            ),
            "longest_component": (
                max(components, key=lambda c: c.duration_ms or 0).component_name
                if components and any(c.duration_ms for c in components) else None
            ),
            "shortest_component": (
                min(components, key=lambda c: c.duration_ms or float('inf')).component_name
                if components and any(c.duration_ms for c in components) else None
            )
        }
    
    def _create_data_summary(self, data: Any) -> Dict[str, Any]:
        """Create summary of input/output data for audit logging."""
        if data is None:
            return {"type": "null", "value": None}
        
        data_type = type(data).__name__
        
        if isinstance(data, dict):
            return {
                "type": "dict",
                "keys_count": len(data),
                "top_keys": list(data.keys())[:5] if data else [],
                "total_size_est": len(str(data))
            }
        elif isinstance(data, (list, tuple)):
            return {
                "type": data_type,
                "length": len(data),
                "total_size_est": len(str(data))
            }
        elif isinstance(data, str):
            return {
                "type": "str",
                "length": len(data),
                "preview": data[:100] + "..." if len(data) > 100 else data
            }
        else:
            return {
                "type": data_type,
                "total_size_est": len(str(data)),
                "preview": str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
            }
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None
        except Exception:
            return None