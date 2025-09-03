"""
Centralized Audit Logging System for K_knowledge_extraction Stage

This module provides comprehensive audit logging functionality for all 6 K-stage
components (06K-11K), tracking execution metrics, performance data, and determinism
verification for reproducible outputs.
"""

import json
import time
import hashlib
import traceback
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from contextlib import contextmanager  # Module not found  # Module not found  # Module not found
import os
import threading

# Try to import psutil, fallback to basic monitoring if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
    # Mock psutil for basic functionality
    class MockProcess:
        def memory_info(self):
            class MemInfo:
                rss = 100 * 1024 * 1024  # Mock 100MB
            return MemInfo()
    
    class MockPsutil:
        def Process(self):
            return MockProcess()
    
    psutil = MockPsutil()


@dataclass
class AuditEntry:
    """Audit entry for component execution."""
    component_code: str
    component_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    success: bool = True
    error_details: Optional[str] = None
    input_artifacts: List[Dict[str, Any]] = None
    output_artifacts: List[Dict[str, Any]] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.input_artifacts is None:
            self.input_artifacts = []
        if self.output_artifacts is None:
            self.output_artifacts = []
        if self.metadata is None:
            self.metadata = {}


class KnowledgeAuditSystem:
    """Centralized audit system for K_knowledge_extraction components."""

    COMPONENT_MAPPING = {
        "06K": "knowledge_component_06",
        "07K": "knowledge_component_07", 
        "08K": "advanced_knowledge_graph_builder",
        "09K": "causal_graph",
        "10K": "causal_dnp_framework",
        "11K": "embedding_builder",
        "12K": "embedding_generator"
    }

    def __init__(self, audit_file_path: str = "canonical_flow/knowledge/knowledge_audit.json"):
        self.audit_file_path = Path(audit_file_path)
        self.audit_entries: List[AuditEntry] = []
        self._lock = threading.Lock()
        self._current_process = psutil.Process()
        
        # Ensure directory exists
        self.audit_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing audit data
        self._load_existing_audit()

    def _load_existing_audit(self):
# # #         """Load existing audit data from file."""  # Module not found  # Module not found  # Module not found
        if self.audit_file_path.exists():
            try:
                with open(self.audit_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "audit_entries" in data:
                        for entry_data in data["audit_entries"]:
                            # Convert datetime strings back to datetime objects
                            if "start_time" in entry_data:
                                entry_data["start_time"] = datetime.fromisoformat(entry_data["start_time"])
                            if "end_time" in entry_data and entry_data["end_time"]:
                                entry_data["end_time"] = datetime.fromisoformat(entry_data["end_time"])
                            
                            entry = AuditEntry(**entry_data)
                            self.audit_entries.append(entry)
            except Exception as e:
                print(f"Warning: Failed to load existing audit data: {e}")

    @contextmanager
    def audit_execution(self, component_code: str, input_data: Any = None, 
                       metadata: Dict[str, Any] = None):
        """Context manager for auditing component execution."""
        execution_id = self._generate_execution_id(component_code)
        component_name = self.COMPONENT_MAPPING.get(component_code, f"unknown_component_{component_code}")
        
        # Create audit entry
        entry = AuditEntry(
            component_code=component_code,
            component_name=component_name,
            execution_id=execution_id,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        # Record input artifacts
        if input_data is not None:
            entry.input_artifacts = self._analyze_artifacts(input_data, "input")
        
        # Record initial memory usage
        initial_memory = self._current_process.memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        
        try:
            # Monitor memory during execution
            def monitor_memory():
                nonlocal peak_memory
                while not hasattr(monitor_memory, 'stop'):
                    current_memory = self._current_process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.1)
            
            monitor_thread = threading.Thread(target=monitor_memory)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            yield entry
            
        except Exception as e:
            entry.success = False
            entry.error_details = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            raise
        finally:
            # Stop memory monitoring
            monitor_memory.stop = True
            
            # Finalize audit entry
            entry.end_time = datetime.now()
            entry.duration_ms = (entry.end_time - entry.start_time).total_seconds() * 1000
            entry.peak_memory_mb = peak_memory
            
            # Add to audit entries
            with self._lock:
                self.audit_entries.append(entry)

    def record_output_artifacts(self, execution_id: str, output_data: Any):
        """Record output artifacts for a given execution."""
        with self._lock:
            for entry in self.audit_entries:
                if entry.execution_id == execution_id:
                    entry.output_artifacts = self._analyze_artifacts(output_data, "output")
                    entry.checksum = self._compute_determinism_checksum(output_data)
                    break

    def _analyze_artifacts(self, data: Any, artifact_type: str) -> List[Dict[str, Any]]:
        """Analyze input/output artifacts and extract metadata."""
        artifacts = []
        
        if isinstance(data, dict):
            # Handle dictionary data
            for key, value in data.items():
                if isinstance(value, (str, Path)) and self._is_file_path(str(value)):
                    artifacts.append(self._get_file_info(str(value), f"{artifact_type}_{key}"))
                elif isinstance(value, bytes):
                    artifacts.append({
                        "name": f"{artifact_type}_{key}",
                        "type": "bytes",
                        "size_bytes": len(value),
                        "path": None
                    })
                elif isinstance(value, (list, tuple)) and value:
                    artifacts.append({
                        "name": f"{artifact_type}_{key}",
                        "type": "collection",
                        "size_bytes": len(str(value).encode('utf-8')),
                        "item_count": len(value),
                        "path": None
                    })
        elif isinstance(data, (str, Path)) and self._is_file_path(str(data)):
            artifacts.append(self._get_file_info(str(data), artifact_type))
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                if isinstance(item, (str, Path)) and self._is_file_path(str(item)):
                    artifacts.append(self._get_file_info(str(item), f"{artifact_type}_{i}"))
        
        return artifacts

    def _is_file_path(self, path_str: str) -> bool:
        """Check if a string represents a valid file path."""
        try:
            path = Path(path_str)
            return path.exists() and path.is_file()
        except:
            return False

    def _get_file_info(self, file_path: str, name: str) -> Dict[str, Any]:
        """Get file information for audit."""
        try:
            path = Path(file_path)
            stat = path.stat()
            return {
                "name": name,
                "type": "file",
                "path": str(path.absolute()),
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        except Exception as e:
            return {
                "name": name,
                "type": "file_error",
                "path": file_path,
                "error": str(e)
            }

    def _compute_determinism_checksum(self, data: Any) -> str:
        """Compute checksum for determinism verification."""
        try:
            # Convert data to a canonical string representation
            if isinstance(data, dict):
                # Sort keys for deterministic ordering
                canonical_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
            elif isinstance(data, (list, tuple)):
                canonical_str = json.dumps(list(data), ensure_ascii=False)
            else:
                canonical_str = str(data)
            
            return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()[:16]

    def _generate_execution_id(self, component_code: str) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.now().isoformat()
        return f"{component_code}_{int(time.time() * 1000)}"

    def validate_audit_schema(self) -> Dict[str, Any]:
        """Validate that audit entries contain all required fields."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        required_fields = [
            "component_code", "component_name", "execution_id", 
            "start_time", "success"
        ]
        
        for i, entry in enumerate(self.audit_entries):
            entry_dict = asdict(entry)
            for field in required_fields:
                if field not in entry_dict or entry_dict[field] is None:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Entry {i}: Missing required field '{field}'"
                    )
            
            # Check for completed executions
            if entry.end_time is None:
                validation_result["warnings"].append(
                    f"Entry {i}: Execution '{entry.execution_id}' appears incomplete"
                )
        
        return validation_result

    def write_audit_file(self, force_write: bool = False) -> bool:
        """Write audit data to JSON file with validation."""
        with self._lock:
            # Validate schema before writing
            validation = self.validate_audit_schema()
            
            if not validation["valid"] and not force_write:
                raise ValueError(f"Audit validation failed: {validation['errors']}")
            
            # Prepare audit data
            audit_data = {
                "audit_metadata": {
                    "system": "K_knowledge_extraction_audit",
                    "version": "1.0.0",
                    "generated_at": datetime.now().isoformat(),
                    "total_entries": len(self.audit_entries),
                    "validation_status": validation
                },
                "component_mapping": self.COMPONENT_MAPPING,
                "audit_entries": []
            }
            
            # Convert audit entries to serializable format
            for entry in self.audit_entries:
                entry_dict = asdict(entry)
                # Convert datetime objects to ISO strings
                if entry_dict["start_time"]:
                    entry_dict["start_time"] = entry.start_time.isoformat()
                if entry_dict["end_time"]:
                    entry_dict["end_time"] = entry.end_time.isoformat()
                
                audit_data["audit_entries"].append(entry_dict)
            
            # Write to file with UTF-8 encoding and proper formatting
            try:
                with open(self.audit_file_path, 'w', encoding='utf-8') as f:
                    json.dump(audit_data, f, indent=2, ensure_ascii=False)
                return True
            except Exception as e:
                raise IOError(f"Failed to write audit file: {e}")

    def get_component_stats(self, component_code: str = None) -> Dict[str, Any]:
        """Get execution statistics for a component or all components."""
        with self._lock:
            entries = self.audit_entries
            if component_code:
                entries = [e for e in entries if e.component_code == component_code]
            
            if not entries:
                return {"message": f"No audit entries found for {component_code or 'any component'}"}
            
            successful = [e for e in entries if e.success]
            failed = [e for e in entries if not e.success]
            
            durations = [e.duration_ms for e in entries if e.duration_ms is not None]
            memory_usage = [e.peak_memory_mb for e in entries if e.peak_memory_mb is not None]
            
            stats = {
                "total_executions": len(entries),
                "successful_executions": len(successful),
                "failed_executions": len(failed),
                "success_rate": len(successful) / len(entries) if entries else 0,
                "performance": {
                    "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                    "min_duration_ms": min(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0,
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                    "peak_memory_mb": max(memory_usage) if memory_usage else 0
                }
            }
            
            return stats


# Global audit system instance
_audit_system = None

def get_audit_system() -> KnowledgeAuditSystem:
    """Get or create the global audit system instance."""
    global _audit_system
    if _audit_system is None:
        _audit_system = KnowledgeAuditSystem()
    return _audit_system


def audit_component_execution(component_code: str, metadata: Dict[str, Any] = None):
    """Decorator for auditing component execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            audit_system = get_audit_system()
            
# # #             # Extract input data from arguments  # Module not found  # Module not found  # Module not found
            input_data = None
            if args:
                input_data = args[0] if args else None
            elif 'data' in kwargs:
                input_data = kwargs['data']
            
            with audit_system.audit_execution(component_code, input_data, metadata) as audit_entry:
                result = func(*args, **kwargs)
                
                # Record output artifacts
                audit_system.record_output_artifacts(audit_entry.execution_id, result)
                
                return result
        
        return wrapper
    return decorator


# Integration methods for adding missing components 06K and 07K
def create_placeholder_component(component_code: str, component_name: str):
    """Create placeholder component with audit integration."""
    def process(data=None, context=None) -> Dict[str, Any]:
        @audit_component_execution(component_code, metadata={"component": component_name})
        def _process_with_audit(data, context):
            if data is None:
                return {"error": "No input data provided"}
            
            # Placeholder processing logic
            result = {
                "component_code": component_code,
                "component_name": component_name,
                "input_received": data is not None,
                "context_received": context is not None,
                "status": "placeholder_processing_complete",
                "data_summary": {
                    "input_type": type(data).__name__,
                    "input_size": len(str(data)) if data else 0,
                }
            }
            
            # Add basic text analysis if text data provided
            if isinstance(data, dict) and "text" in data:
                text = data["text"]
                result["text_analysis"] = {
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "sentence_count": text.count('.') + text.count('!') + text.count('?')
                }
            elif isinstance(data, str):
                result["text_analysis"] = {
                    "word_count": len(data.split()),
                    "char_count": len(data),
                    "sentence_count": data.count('.') + data.count('!') + data.count('?')
                }
            
            return result
        
        return _process_with_audit(data, context)
    
    return process


# Create placeholder components for missing 06K and 07K
component_06K_process = create_placeholder_component("06K", "knowledge_component_06")
component_07K_process = create_placeholder_component("07K", "knowledge_component_07")