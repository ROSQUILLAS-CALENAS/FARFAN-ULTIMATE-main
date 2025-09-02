"""
Data Integrity Checker for Pipeline Artifacts

Provides SHA-256 hash computation and validation of JSON artifacts at stage boundaries,
automatic retry logic with exponential backoff, and comprehensive corruption reporting.
Integrates seamlessly with existing pipeline flow and audit system.
"""

import json
import hashlib
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback
import random

logger = logging.getLogger(__name__)


class CorruptionType(str, Enum):
    """Types of corruption detected"""
    HASH_MISMATCH = "hash_mismatch"
    MISSING_METADATA = "missing_metadata"
    FILE_NOT_FOUND = "file_not_found"
    PARSE_ERROR = "parse_error"
    SIZE_MISMATCH = "size_mismatch"


class RetryStatus(str, Enum):
    """Status of retry attempts"""
    SUCCESS = "success"
    RETRY = "retry"
    EXHAUSTED = "exhausted"
    FATAL = "fatal"


@dataclass
class ArtifactMetadata:
    """Required metadata for pipeline artifacts"""
    stage_name: str
    component_name: str
    document_stem: str
    created_at: str
    sha256_hash: str
    file_size: int
    schema_version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
    
    # Computed fields
    integrity_status: str = "unverified"
    last_verified: Optional[str] = None
    verification_count: int = 0
    
    def __post_init__(self):
        """Ensure deterministic ordering"""
        self.dependencies = sorted(self.dependencies)


@dataclass
class CorruptionReport:
    """Detailed corruption detection report"""
    corruption_type: CorruptionType
    stage_name: str
    component_name: str
    file_path: str
    document_stem: str
    
    # Hash comparison details
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    
    # Metadata validation details
    missing_fields: List[str] = field(default_factory=list)
    invalid_fields: Dict[str, str] = field(default_factory=dict)
    
    # File details
    expected_size: Optional[int] = None
    actual_size: Optional[int] = None
    
    # Context
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    retry_attempt: int = 0
    
    def __post_init__(self):
        """Ensure deterministic ordering"""
        self.missing_fields = sorted(self.missing_fields)
        if self.invalid_fields:
            self.invalid_fields = dict(sorted(self.invalid_fields.items()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "corruption_type": self.corruption_type.value,
            "stage_name": self.stage_name,
            "component_name": self.component_name,
            "file_path": str(self.file_path),
            "document_stem": self.document_stem,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "missing_fields": self.missing_fields,
            "invalid_fields": self.invalid_fields,
            "expected_size": self.expected_size,
            "actual_size": self.actual_size,
            "detected_at": self.detected_at,
            "error_message": self.error_message,
            "retry_attempt": self.retry_attempt
        }


@dataclass 
class RetryPolicy:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_corruption_types: Set[CorruptionType] = field(
        default_factory=lambda: {
            CorruptionType.HASH_MISMATCH,
            CorruptionType.FILE_NOT_FOUND,
            CorruptionType.PARSE_ERROR
        }
    )
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add 20% random jitter to prevent thundering herd
            jitter_range = delay * 0.2
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class DataIntegrityChecker:
    """
    Data integrity checker with corruption detection and automatic recovery.
    
    Features:
    - SHA-256 hash computation and validation
    - Required metadata field validation  
    - Automatic retry with exponential backoff
    - Detailed corruption reporting
    - Integration with existing audit system
    """
    
    def __init__(self, retry_policy: Optional[RetryPolicy] = None):
        """
        Initialize data integrity checker.
        
        Args:
            retry_policy: Retry configuration (uses default if None)
        """
        self.retry_policy = retry_policy or RetryPolicy()
        
        # File paths
        self.canonical_flow_dir = Path("canonical_flow")
        self.integrity_audit_file = Path("integrity_audit.json")
        
        # In-memory state
        self._artifact_registry: Dict[str, ArtifactMetadata] = {}
        self._corruption_history: List[CorruptionReport] = []
        
        # Required metadata fields
        self.required_metadata_fields = {
            "stage_name", "component_name", "document_stem", 
            "created_at", "sha256_hash", "file_size"
        }
        
        # Initialize audit file if needed
        self._initialize_integrity_audit()
    
    def _initialize_integrity_audit(self):
        """Initialize integrity audit file with proper structure"""
        if not self.integrity_audit_file.exists():
            initial_structure = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "schema_version": "1.0",
                    "description": "Data integrity audit trail"
                },
                "corruption_events": [],
                "recovery_attempts": [],
                "integrity_statistics": {
                    "total_artifacts_checked": 0,
                    "corruption_events_detected": 0,
                    "successful_recoveries": 0,
                    "failed_recoveries": 0
                }
            }
            self._write_audit_file(initial_structure)
    
    def compute_artifact_hash(self, artifact_data: Any) -> str:
        """
        Compute SHA-256 hash of JSON artifact data.
        
        Args:
            artifact_data: Data to hash (will be JSON serialized)
            
        Returns:
            SHA-256 hash as hexadecimal string
        """
        try:
            # Convert to deterministic JSON representation
            json_str = json.dumps(artifact_data, sort_keys=True, separators=(',', ':'))
            
            # Compute SHA-256 hash
            return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to compute artifact hash: {e}")
            raise ValueError(f"Cannot compute hash for artifact: {e}")
    
    def generate_artifact_metadata(self, artifact_data: Any, stage_name: str, 
                                 component_name: str, document_stem: str,
                                 dependencies: Optional[List[str]] = None) -> ArtifactMetadata:
        """
        Generate complete metadata for an artifact.
        
        Args:
            artifact_data: The artifact data
            stage_name: Processing stage name
            component_name: Component name
            document_stem: Document identifier
            dependencies: List of dependent artifacts
            
        Returns:
            Complete artifact metadata
        """
        # Compute hash and size based on the full artifact structure that will be saved
        sha256_hash = self.compute_artifact_hash(artifact_data)
        
        # Create the full artifact structure to get accurate file size
        temp_metadata = ArtifactMetadata(
            stage_name=stage_name,
            component_name=component_name,
            document_stem=document_stem,
            created_at=datetime.now().isoformat(),
            sha256_hash=sha256_hash,
            file_size=0,  # Will be updated
            dependencies=dependencies or []
        )
        
        artifact_with_metadata = {
            "metadata": temp_metadata.__dict__,
            "data": artifact_data
        }
        
        json_str = json.dumps(artifact_with_metadata, sort_keys=True, separators=(',', ':'))
        file_size = len(json_str.encode('utf-8'))
        
        # Update file size
        temp_metadata.file_size = file_size
        
        return temp_metadata
    
    def validate_artifact_integrity(self, file_path: Union[str, Path], 
                                   expected_metadata: Optional[ArtifactMetadata] = None) -> Tuple[bool, Optional[CorruptionReport]]:
        """
        Validate integrity of an artifact file.
        
        Args:
            file_path: Path to artifact file
            expected_metadata: Expected metadata (optional)
            
        Returns:
            Tuple of (is_valid, corruption_report)
        """
        file_path = Path(file_path)
        
        try:
            # Check file exists
            if not file_path.exists():
                report = CorruptionReport(
                    corruption_type=CorruptionType.FILE_NOT_FOUND,
                    stage_name=expected_metadata.stage_name if expected_metadata else "unknown",
                    component_name=expected_metadata.component_name if expected_metadata else "unknown",
                    file_path=str(file_path),
                    document_stem=expected_metadata.document_stem if expected_metadata else "unknown",
                    error_message=f"File not found: {file_path}"
                )
                return False, report
            
            # Load and parse artifact
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    artifact_data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                report = CorruptionReport(
                    corruption_type=CorruptionType.PARSE_ERROR,
                    stage_name=expected_metadata.stage_name if expected_metadata else "unknown",
                    component_name=expected_metadata.component_name if expected_metadata else "unknown",
                    file_path=str(file_path),
                    document_stem=expected_metadata.document_stem if expected_metadata else "unknown",
                    error_message=f"JSON parse error: {e}"
                )
                return False, report
            
            # Validate metadata if provided
            if expected_metadata:
                return self._validate_against_metadata(artifact_data, file_path, expected_metadata)
            
            # Basic validation without expected metadata
            return self._validate_basic_integrity(artifact_data, file_path)
            
        except Exception as e:
            logger.error(f"Unexpected error validating artifact {file_path}: {e}")
            report = CorruptionReport(
                corruption_type=CorruptionType.PARSE_ERROR,
                stage_name=expected_metadata.stage_name if expected_metadata else "unknown",
                component_name=expected_metadata.component_name if expected_metadata else "unknown",
                file_path=str(file_path),
                document_stem=expected_metadata.document_stem if expected_metadata else "unknown",
                error_message=f"Validation error: {e}"
            )
            return False, report
    
    def _validate_against_metadata(self, artifact_data: Any, file_path: Path, 
                                  expected_metadata: ArtifactMetadata) -> Tuple[bool, Optional[CorruptionReport]]:
        """Validate artifact against expected metadata"""
        
        # Extract the data portion if this is a structured artifact with metadata
        if isinstance(artifact_data, dict) and 'data' in artifact_data:
            data_to_validate = artifact_data['data']
        else:
            data_to_validate = artifact_data
        
        # Compute actual hash
        actual_hash = self.compute_artifact_hash(data_to_validate)
        
        # Check hash match
        if actual_hash != expected_metadata.sha256_hash:
            report = CorruptionReport(
                corruption_type=CorruptionType.HASH_MISMATCH,
                stage_name=expected_metadata.stage_name,
                component_name=expected_metadata.component_name,
                file_path=str(file_path),
                document_stem=expected_metadata.document_stem,
                expected_hash=expected_metadata.sha256_hash,
                actual_hash=actual_hash
            )
            return False, report
        
        # Check file size (allow some tolerance for timestamp differences)
        actual_size = file_path.stat().st_size
        size_tolerance = 100  # Allow 100 byte difference for timestamp/whitespace variations
        if abs(actual_size - expected_metadata.file_size) > size_tolerance:
            report = CorruptionReport(
                corruption_type=CorruptionType.SIZE_MISMATCH,
                stage_name=expected_metadata.stage_name,
                component_name=expected_metadata.component_name,
                file_path=str(file_path),
                document_stem=expected_metadata.document_stem,
                expected_size=expected_metadata.file_size,
                actual_size=actual_size
            )
            return False, report
        
        # Validate required metadata fields if present in artifact
        if isinstance(artifact_data, dict) and 'metadata' in artifact_data:
            return self._validate_metadata_fields(artifact_data['metadata'], file_path, expected_metadata)
        
        return True, None
    
    def _validate_basic_integrity(self, artifact_data: Any, file_path: Path) -> Tuple[bool, Optional[CorruptionReport]]:
        """Basic integrity validation without expected metadata"""
        
        # Check for embedded metadata
        if isinstance(artifact_data, dict) and 'metadata' in artifact_data:
            metadata = artifact_data['metadata']
            missing_fields = []
            
            for field in self.required_metadata_fields:
                if field not in metadata:
                    missing_fields.append(field)
            
            if missing_fields:
                report = CorruptionReport(
                    corruption_type=CorruptionType.MISSING_METADATA,
                    stage_name=metadata.get('stage_name', 'unknown'),
                    component_name=metadata.get('component_name', 'unknown'),
                    file_path=str(file_path),
                    document_stem=metadata.get('document_stem', 'unknown'),
                    missing_fields=missing_fields
                )
                return False, report
        
        return True, None
    
    def _validate_metadata_fields(self, metadata: Dict[str, Any], file_path: Path,
                                expected_metadata: ArtifactMetadata) -> Tuple[bool, Optional[CorruptionReport]]:
        """Validate metadata fields"""
        
        missing_fields = []
        invalid_fields = {}
        
        for field in self.required_metadata_fields:
            if field not in metadata:
                missing_fields.append(field)
            else:
                # Validate specific fields
                if field == 'stage_name' and metadata[field] != expected_metadata.stage_name:
                    invalid_fields[field] = f"Expected {expected_metadata.stage_name}, got {metadata[field]}"
                elif field == 'component_name' and metadata[field] != expected_metadata.component_name:
                    invalid_fields[field] = f"Expected {expected_metadata.component_name}, got {metadata[field]}"
                elif field == 'document_stem' and metadata[field] != expected_metadata.document_stem:
                    invalid_fields[field] = f"Expected {expected_metadata.document_stem}, got {metadata[field]}"
        
        if missing_fields or invalid_fields:
            report = CorruptionReport(
                corruption_type=CorruptionType.MISSING_METADATA,
                stage_name=expected_metadata.stage_name,
                component_name=expected_metadata.component_name,
                file_path=str(file_path),
                document_stem=expected_metadata.document_stem,
                missing_fields=missing_fields,
                invalid_fields=invalid_fields
            )
            return False, report
        
        return True, None
    
    def save_artifact_with_integrity(self, artifact_data: Any, stage_name: str,
                                   component_name: str, document_stem: str,
                                   output_path: Optional[Union[str, Path]] = None,
                                   dependencies: Optional[List[str]] = None) -> Tuple[Path, ArtifactMetadata]:
        """
        Save artifact with integrity metadata and hash computation.
        
        Args:
            artifact_data: Data to save
            stage_name: Processing stage name
            component_name: Component name  
            document_stem: Document identifier
            output_path: Output path (auto-generated if None)
            dependencies: List of dependent artifacts
            
        Returns:
            Tuple of (file_path, metadata)
        """
        # Generate metadata
        metadata = self.generate_artifact_metadata(
            artifact_data, stage_name, component_name, document_stem, dependencies
        )
        
        # Determine output path
        if output_path is None:
            stage_dir = self.canonical_flow_dir / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            output_path = stage_dir / f"{document_stem}_{component_name}_artifact.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create artifact with embedded metadata
        artifact_with_metadata = {
            "metadata": metadata.__dict__,
            "data": artifact_data
        }
        
        # Save artifact
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(artifact_with_metadata, f, indent=2, sort_keys=True)
        
        # Register in artifact registry
        registry_key = f"{stage_name}/{component_name}/{document_stem}"
        self._artifact_registry[registry_key] = metadata
        
        logger.info(f"Saved artifact with integrity: {output_path}")
        return output_path, metadata
    
    def validate_stage_boundary(self, stage_name: str, document_stem: str) -> Dict[str, Any]:
        """
        Validate all artifacts at a stage boundary before proceeding.
        
        Args:
            stage_name: Stage to validate
            document_stem: Document identifier
            
        Returns:
            Validation report
        """
        validation_report = {
            "stage_name": stage_name,
            "document_stem": document_stem,
            "timestamp": datetime.now().isoformat(),
            "artifacts_validated": 0,
            "corruption_detected": 0,
            "validation_errors": [],
            "overall_status": "success"
        }
        
        # Find all artifacts for this stage/document
        stage_dir = self.canonical_flow_dir / stage_name
        if not stage_dir.exists():
            validation_report["overall_status"] = "error"
            validation_report["validation_errors"].append(f"Stage directory not found: {stage_dir}")
            return validation_report
        
        # Validate each artifact
        artifact_files = list(stage_dir.glob(f"{document_stem}_*_artifact.json"))
        
        for artifact_file in artifact_files:
            validation_report["artifacts_validated"] += 1
            
            is_valid, corruption_report = self.validate_artifact_integrity(artifact_file)
            
            if not is_valid and corruption_report:
                validation_report["corruption_detected"] += 1
                validation_report["validation_errors"].append(corruption_report.to_dict())
                validation_report["overall_status"] = "corruption_detected"
        
        return validation_report
    
    def process_with_integrity_validation(self, process_func: callable, 
                                        stage_name: str, component_name: str,
                                        document_stem: str, *args, **kwargs) -> Any:
        """
        Wrap a process() method with integrity validation and retry logic.
        
        Args:
            process_func: The process function to wrap
            stage_name: Processing stage name
            component_name: Component name
            document_stem: Document identifier
            *args, **kwargs: Arguments for process function
            
        Returns:
            Result from successful process execution
            
        Raises:
            RuntimeError: If all retry attempts are exhausted
        """
        
        for attempt in range(self.retry_policy.max_attempts + 1):
            try:
                # Validate input artifacts if this isn't the first attempt
                if attempt > 0:
                    validation_report = self.validate_stage_boundary(stage_name, document_stem)
                    if validation_report["corruption_detected"] > 0:
                        logger.warning(f"Corruption still detected in attempt {attempt}")
                
                # Execute process function
                result = process_func(*args, **kwargs)
                
                # If we have result data, save with integrity
                if isinstance(result, dict) and result:
                    output_path, metadata = self.save_artifact_with_integrity(
                        result, stage_name, component_name, document_stem
                    )
                    
                    # Immediately validate what we just saved
                    is_valid, corruption_report = self.validate_artifact_integrity(
                        output_path, metadata
                    )
                    
                    if not is_valid:
                        error_msg = corruption_report.error_message if corruption_report else "Unknown validation error"
                        raise RuntimeError(f"Generated corrupt artifact: {error_msg}")
                
                # Success - log and return
                if attempt > 0:
                    self._log_successful_recovery(stage_name, component_name, document_stem, attempt)
                
                return result
                
            except Exception as e:
                logger.error(f"Process execution failed on attempt {attempt + 1}: {e}")
                
                # Create corruption report
                corruption_report = CorruptionReport(
                    corruption_type=CorruptionType.PARSE_ERROR,
                    stage_name=stage_name,
                    component_name=component_name,
                    file_path="process_execution",
                    document_stem=document_stem,
                    error_message=str(e),
                    retry_attempt=attempt + 1
                )
                
                # Log corruption event
                self._log_corruption_event(corruption_report)
                
                # Check if we should retry
                if attempt < self.retry_policy.max_attempts:
                    # Calculate backoff delay
                    delay = self.retry_policy.calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds (attempt {attempt + 2})")
                    time.sleep(delay)
                    continue
                else:
                    # Exhausted retries
                    self._log_failed_recovery(stage_name, component_name, document_stem, attempt + 1)
                    raise RuntimeError(f"Process failed after {self.retry_policy.max_attempts} attempts: {e}")
        
        raise RuntimeError("Unexpected end of retry loop")
    
    def _log_corruption_event(self, corruption_report: CorruptionReport):
        """Log corruption event to integrity audit file"""
        try:
            audit_data = self._read_audit_file()
            audit_data["corruption_events"].append(corruption_report.to_dict())
            audit_data["integrity_statistics"]["corruption_events_detected"] += 1
            self._write_audit_file(audit_data)
            
            # Also add to in-memory history
            self._corruption_history.append(corruption_report)
            
        except Exception as e:
            logger.error(f"Failed to log corruption event: {e}")
    
    def _log_successful_recovery(self, stage_name: str, component_name: str, 
                               document_stem: str, attempts: int):
        """Log successful recovery attempt"""
        try:
            audit_data = self._read_audit_file()
            
            recovery_event = {
                "stage_name": stage_name,
                "component_name": component_name,
                "document_stem": document_stem,
                "attempts": attempts,
                "recovered_at": datetime.now().isoformat(),
                "status": "success"
            }
            
            audit_data["recovery_attempts"].append(recovery_event)
            audit_data["integrity_statistics"]["successful_recoveries"] += 1
            self._write_audit_file(audit_data)
            
        except Exception as e:
            logger.error(f"Failed to log successful recovery: {e}")
    
    def _log_failed_recovery(self, stage_name: str, component_name: str,
                           document_stem: str, attempts: int):
        """Log failed recovery attempt"""
        try:
            audit_data = self._read_audit_file()
            
            recovery_event = {
                "stage_name": stage_name,
                "component_name": component_name, 
                "document_stem": document_stem,
                "attempts": attempts,
                "failed_at": datetime.now().isoformat(),
                "status": "failed"
            }
            
            audit_data["recovery_attempts"].append(recovery_event)
            audit_data["integrity_statistics"]["failed_recoveries"] += 1
            self._write_audit_file(audit_data)
            
        except Exception as e:
            logger.error(f"Failed to log failed recovery: {e}")
    
    def _read_audit_file(self) -> Dict[str, Any]:
        """Read integrity audit file"""
        if not self.integrity_audit_file.exists():
            self._initialize_integrity_audit()
        
        with open(self.integrity_audit_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _write_audit_file(self, audit_data: Dict[str, Any]):
        """Write integrity audit file"""
        with open(self.integrity_audit_file, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, sort_keys=True)
    
    def get_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive integrity report"""
        audit_data = self._read_audit_file()
        
        return {
            "generated_at": datetime.now().isoformat(),
            "statistics": audit_data["integrity_statistics"],
            "recent_corruption_events": audit_data["corruption_events"][-10:],  # Last 10
            "recent_recovery_attempts": audit_data["recovery_attempts"][-10:],  # Last 10
            "artifact_registry_size": len(self._artifact_registry),
            "active_artifacts": list(self._artifact_registry.keys())
        }


# Integration helpers for existing pipeline components

def integrity_validation_hook(integrity_checker: DataIntegrityChecker):
    """
    Decorator to add integrity validation to process() methods.
    
    Usage:
        @integrity_validation_hook(integrity_checker)
        def process(self, data=None, context=None):
            # Your existing process logic
            return result
    """
    def decorator(process_func):
        def wrapper(self, *args, **kwargs):
            # Extract stage and component info
            stage_name = getattr(self, 'stage_name', 'unknown')
            component_name = getattr(self, '__class__').__name__
            document_stem = "unknown"
            
            # Try to extract document_stem from context or data
            if len(args) > 1 and isinstance(args[1], dict) and 'document_stem' in args[1]:
                document_stem = args[1]['document_stem']
            elif 'context' in kwargs and isinstance(kwargs['context'], dict):
                document_stem = kwargs['context'].get('document_stem', 'unknown')
            
            return integrity_checker.process_with_integrity_validation(
                process_func, stage_name, component_name, document_stem, 
                self, *args, **kwargs
            )
        
        return wrapper
    return decorator


def add_artifact_generation_hook(component_class):
    """
    Class decorator to add artifact generation hooks to pipeline components.
    
    This modifies the process() method to automatically save artifacts with integrity metadata.
    """
    original_process = component_class.process
    
    def enhanced_process(self, data=None, context=None):
        # Create integrity checker if not exists
        if not hasattr(self, '_integrity_checker'):
            self._integrity_checker = DataIntegrityChecker()
        
        # Call original process
        result = original_process(self, data, context)
        
        # Save result as artifact if it's substantial data
        if isinstance(result, dict) and len(result) > 1:  # Non-trivial result
            stage_name = getattr(self, 'stage_name', 'unknown')
            component_name = self.__class__.__name__
            document_stem = context.get('document_stem', 'unknown') if context else 'unknown'
            
            try:
                self._integrity_checker.save_artifact_with_integrity(
                    result, stage_name, component_name, document_stem
                )
            except Exception as e:
                logger.warning(f"Failed to save artifact with integrity: {e}")
        
        return result
    
    component_class.process = enhanced_process
    return component_class


# Global integrity checker instance for easy access
_global_integrity_checker: Optional[DataIntegrityChecker] = None


def get_global_integrity_checker() -> DataIntegrityChecker:
    """Get or create global integrity checker instance"""
    global _global_integrity_checker
    if _global_integrity_checker is None:
        _global_integrity_checker = DataIntegrityChecker()
    return _global_integrity_checker


# Convenience function for easy integration
def validate_and_retry_on_corruption(stage_name: str, component_name: str, 
                                   document_stem: str):
    """
    Convenience decorator for adding integrity validation to any function.
    
    Usage:
        @validate_and_retry_on_corruption("A_analysis_nlp", "QuestionAnalyzer", "doc_123")
        def my_process_function():
            # Your processing logic
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            checker = get_global_integrity_checker()
            return checker.process_with_integrity_validation(
                func, stage_name, component_name, document_stem, *args, **kwargs
            )
        return wrapper
    return decorator