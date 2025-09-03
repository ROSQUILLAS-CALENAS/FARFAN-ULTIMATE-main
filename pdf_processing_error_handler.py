"""
Comprehensive PDF Processing Error Handling System

This module provides robust error handling, checkpointing, and resource monitoring
for PDF processing pipelines with retry mechanisms and validation.
"""

import json
import logging
import os
import pickle
import struct
import time
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Callable, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from functools import wraps  # Module not found  # Module not found  # Module not found
import threading
import hashlib


# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "106O"
__stage_order__ = 7

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Resource monitoring will be disabled.")


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors"""
    pass


class PDFValidationError(PDFProcessingError):
    """Error raised when PDF validation fails"""
    pass


class ResourceExhaustionError(PDFProcessingError):
    """Error raised when resource thresholds are exceeded"""
    pass


class CheckpointError(PDFProcessingError):
    """Error raised during checkpoint operations"""
    pass


@dataclass
class ProcessingState:
    """Represents the state of a PDF processing batch"""
    batch_id: str
    total_documents: int
    processed_documents: List[str]
    failed_documents: List[Tuple[str, str]]  # (document_path, error_message)
    current_index: int
    checkpoint_frequency: int
    start_time: datetime
    last_checkpoint_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.last_checkpoint_time:
            data['last_checkpoint_time'] = self.last_checkpoint_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingState':
# # #         """Create from dictionary"""  # Module not found  # Module not found  # Module not found
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('last_checkpoint_time'):
            data['last_checkpoint_time'] = datetime.fromisoformat(data['last_checkpoint_time'])
        return cls(**data)


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    disk_usage_gb: float
    timestamp: datetime
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


class PDFValidator:
    """Validates PDF files before processing"""
    
    def __init__(self, 
                 max_file_size_mb: int = 100,
                 min_file_size_bytes: int = 1024):
        self.max_file_size_mb = max_file_size_mb
        self.min_file_size_bytes = min_file_size_bytes
        self.logger = logging.getLogger(__name__)
    
    def validate_pdf(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Comprehensive PDF validation
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        try:
            # Check if file exists
            if not file_path.exists():
                return False, f"File does not exist: {file_path}"
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size < self.min_file_size_bytes:
                return False, f"File too small ({file_size} bytes), minimum: {self.min_file_size_bytes}"
            
            if file_size > self.max_file_size_mb * 1024 * 1024:
                return False, f"File too large ({file_size / (1024*1024):.1f}MB), maximum: {self.max_file_size_mb}MB"
            
            # Validate PDF header
            is_valid_header, header_msg = self._validate_pdf_header(file_path)
            if not is_valid_header:
                return False, header_msg
            
            # Check for corruption using basic structure parsing
            is_valid_structure, structure_msg = self._validate_pdf_structure(file_path)
            if not is_valid_structure:
                return False, structure_msg
            
            return True, "Valid PDF"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_pdf_header(self, file_path: Path) -> Tuple[bool, str]:
        """Validate PDF header signature"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
            
            # PDF files should start with "%PDF-"
            if not header.startswith(b'%PDF-'):
                return False, "Invalid PDF header signature"
            
            # Extract version
            try:
                version_part = header[5:8].decode('ascii')
                if not version_part[0].isdigit():
                    return False, "Invalid PDF version format"
            except (UnicodeDecodeError, IndexError):
                return False, "Corrupted PDF header"
            
            return True, "Valid PDF header"
            
        except Exception as e:
            return False, f"Header validation error: {str(e)}"
    
    def _validate_pdf_structure(self, file_path: Path) -> Tuple[bool, str]:
        """Basic PDF structure validation"""
        try:
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes to check basic structure
                content = f.read(1024)
                
                # Check for essential PDF elements
                if b'%PDF-' not in content:
                    return False, "Missing PDF signature"
                
                # Read last 1024 bytes to check for xref table and trailer
                f.seek(-min(1024, file_path.stat().st_size), 2)
                end_content = f.read()
                
                # Look for PDF footer elements
                if b'%%EOF' not in end_content:
                    return False, "Missing PDF end-of-file marker"
                
                # Check for cross-reference table indicator
                if b'xref' not in end_content and b'/XRefStm' not in end_content:
                    return False, "Missing cross-reference table"
                
            return True, "Valid PDF structure"
            
        except Exception as e:
            return False, f"Structure validation error: {str(e)}"


class ResourceMonitor:
    """Monitors system resources during processing"""
    
    def __init__(self,
                 memory_threshold_mb: int = 2048,
                 memory_percent_threshold: float = 85.0,
                 cpu_threshold_percent: float = 95.0,
                 disk_threshold_gb: float = 1.0):
        self.memory_threshold_mb = memory_threshold_mb
        self.memory_percent_threshold = memory_percent_threshold
        self.cpu_threshold_percent = cpu_threshold_percent
        self.disk_threshold_gb = disk_threshold_gb
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None
        self._metrics_history: List[ResourceMetrics] = []
        self._lock = threading.Lock()
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics"""
        if not PSUTIL_AVAILABLE:
            # Return dummy metrics if psutil is not available
            return ResourceMetrics(
                memory_usage_mb=0.0,
                memory_percent=0.0,
                cpu_percent=0.0,
                disk_usage_gb=100.0,  # Assume plenty of disk space
                timestamp=datetime.now()
            )
        
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/')
        
        return ResourceMetrics(
            memory_usage_mb=memory_info.rss / (1024 * 1024),
            memory_percent=system_memory.percent,
            cpu_percent=cpu_percent,
            disk_usage_gb=disk_usage.free / (1024 ** 3),
            timestamp=datetime.now()
        )
    
    def check_resource_thresholds(self) -> Tuple[bool, str]:
        """
        Check if current resource usage exceeds thresholds
        
        Returns:
            Tuple of (within_limits, warning_message)
        """
        if not PSUTIL_AVAILABLE:
            # If psutil is not available, assume resources are within limits
            return True, "Resource monitoring disabled (psutil not available)"
        
        try:
            metrics = self.get_current_metrics()
            
            # Check memory usage
            if metrics.memory_usage_mb > self.memory_threshold_mb:
                return False, f"Memory usage ({metrics.memory_usage_mb:.1f}MB) exceeds threshold ({self.memory_threshold_mb}MB)"
            
            # Check system memory percentage
            if metrics.memory_percent > self.memory_percent_threshold:
                return False, f"System memory usage ({metrics.memory_percent:.1f}%) exceeds threshold ({self.memory_percent_threshold}%)"
            
            # Check CPU usage
            if metrics.cpu_percent > self.cpu_threshold_percent:
                return False, f"CPU usage ({metrics.cpu_percent:.1f}%) exceeds threshold ({self.cpu_threshold_percent}%)"
            
            # Check disk space
            if metrics.disk_usage_gb < self.disk_threshold_gb:
                return False, f"Available disk space ({metrics.disk_usage_gb:.1f}GB) below threshold ({self.disk_threshold_gb}GB)"
            
            return True, "Resource usage within limits"
            
        except Exception as e:
            return False, f"Error checking resources: {str(e)}"
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start background resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop background resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self, interval_seconds: int):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                metrics = self.get_current_metrics()
                
                with self._lock:
                    self._metrics_history.append(metrics)
                    # Keep only last 100 metrics
                    if len(self._metrics_history) > 100:
                        self._metrics_history = self._metrics_history[-100:]
                
                # Check thresholds and log warnings
                within_limits, message = self.check_resource_thresholds()
                if not within_limits:
                    self.logger.warning(f"Resource threshold exceeded: {message}")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval_seconds)
    
    def get_metrics_history(self) -> List[ResourceMetrics]:
        """Get historical resource metrics"""
        with self._lock:
            return self._metrics_history.copy()


class ExponentialBackoffRetry:
    """Implements exponential backoff retry mechanism"""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 2.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for adding retry logic to functions"""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt == self.max_attempts:
                        self.logger.error(f"Function {func.__name__} failed after {self.max_attempts} attempts: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.exponential_base ** (attempt - 1)),
                        self.max_delay
                    )
                    
                    self.logger.warning(f"Function {func.__name__} failed (attempt {attempt}/{self.max_attempts}): {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper


class CheckpointManager:
    """Manages processing checkpoints and state recovery"""
    
    def __init__(self, checkpoint_dir: Union[str, Path] = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, state: ProcessingState) -> str:
        """
        Save processing state to checkpoint
        
        Returns:
            Checkpoint file path
        """
        try:
            state.last_checkpoint_time = datetime.now()
            checkpoint_filename = f"checkpoint_{state.batch_id}_{int(time.time())}.pkl"
            checkpoint_path = self.checkpoint_dir / checkpoint_filename
            
            # Save as pickle for complex objects
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            
            # Also save as JSON for human readability
            json_path = checkpoint_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> ProcessingState:
# # #         """Load processing state from checkpoint"""  # Module not found  # Module not found  # Module not found
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if checkpoint_path.suffix == '.pkl':
                with open(checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
            else:
                # Try JSON format
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                state = ProcessingState.from_dict(data)
            
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return state
            
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")
    
    def find_latest_checkpoint(self, batch_id: str) -> Optional[Path]:
        """Find the most recent checkpoint for a batch"""
        try:
            pattern = f"checkpoint_{batch_id}_*.pkl"
            checkpoints = list(self.checkpoint_dir.glob(pattern))
            
            if not checkpoints:
                return None
            
            # Sort by modification time, get latest
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            return latest
            
        except Exception as e:
            self.logger.error(f"Error finding latest checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, batch_id: str, keep_count: int = 3):
        """Remove old checkpoints, keeping only the most recent ones"""
        try:
            pattern = f"checkpoint_{batch_id}_*"
            checkpoints = list(self.checkpoint_dir.glob(pattern))
            
            if len(checkpoints) <= keep_count:
                return
            
            # Sort by modification time
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Remove old checkpoints
            for old_checkpoint in checkpoints[keep_count:]:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoints: {e}")


class PDFErrorHandler:
    """Main error handler for PDF processing operations"""
    
    def __init__(self,
                 checkpoint_frequency: int = 10,
                 checkpoint_dir: Union[str, Path] = "./checkpoints",
                 max_file_size_mb: int = 100,
                 memory_threshold_mb: int = 2048,
                 max_retry_attempts: int = 3,
                 base_retry_delay: float = 2.0,
                 enable_resource_monitoring: bool = True,
                 monitoring_interval: int = 30):
        
        self.checkpoint_frequency = checkpoint_frequency
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validator = PDFValidator(max_file_size_mb=max_file_size_mb)
        self.resource_monitor = ResourceMonitor(memory_threshold_mb=memory_threshold_mb)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        self.retry_decorator = ExponentialBackoffRetry(
            max_attempts=max_retry_attempts,
            base_delay=base_retry_delay
        )
        
        # Start resource monitoring if enabled
        if enable_resource_monitoring:
            self.resource_monitor.start_monitoring(monitoring_interval)
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop_monitoring()
    
    def validate_pdf_batch(self, file_paths: List[Union[str, Path]]) -> Tuple[List[Path], List[Tuple[str, str]]]:
        """
        Validate a batch of PDF files
        
        Returns:
            Tuple of (valid_files, invalid_files_with_errors)
        """
        valid_files = []
        invalid_files = []
        
        for file_path in file_paths:
            is_valid, error_msg = self.validator.validate_pdf(file_path)
            if is_valid:
                valid_files.append(Path(file_path))
            else:
                invalid_files.append((str(file_path), error_msg))
        
        self.logger.info(f"Validated batch: {len(valid_files)} valid, {len(invalid_files)} invalid")
        return valid_files, invalid_files
    
    def process_pdf_batch(self,
                         file_paths: List[Union[str, Path]],
                         processing_function: Callable[[Path], Any],
                         batch_id: Optional[str] = None,
                         resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Process a batch of PDF files with comprehensive error handling
        
        Args:
            file_paths: List of PDF file paths to process
            processing_function: Function to process each PDF file
            batch_id: Unique identifier for this batch
# # #             resume_from_checkpoint: Whether to try resuming from existing checkpoint  # Module not found  # Module not found  # Module not found
        
        Returns:
            Dictionary with processing results and statistics
        """
        if batch_id is None:
            batch_id = hashlib.md5(str(file_paths).encode()).hexdigest()[:12]
        
# # #         # Try to resume from checkpoint  # Module not found  # Module not found  # Module not found
        state = None
        if resume_from_checkpoint:
            latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint(batch_id)
            if latest_checkpoint:
                try:
                    state = self.checkpoint_manager.load_checkpoint(latest_checkpoint)
# # #                     self.logger.info(f"Resuming from checkpoint at document {state.current_index}")  # Module not found  # Module not found  # Module not found
                except CheckpointError as e:
# # #                     self.logger.warning(f"Could not resume from checkpoint: {e}")  # Module not found  # Module not found  # Module not found
        
        # Initialize state if not resumed
        if state is None:
            # Validate all files first
            valid_files, invalid_files = self.validate_pdf_batch(file_paths)
            
            if invalid_files:
                self.logger.warning(f"Found {len(invalid_files)} invalid files")
            
            state = ProcessingState(
                batch_id=batch_id,
                total_documents=len(valid_files),
                processed_documents=[],
                failed_documents=invalid_files,
                current_index=0,
                checkpoint_frequency=self.checkpoint_frequency,
                start_time=datetime.now(),
                metadata={"original_file_count": len(file_paths)}
            )
            file_paths = valid_files
        else:
            # Filter to unprocessed files when resuming
            all_processed = set(state.processed_documents + [f[0] for f in state.failed_documents])
            file_paths = [Path(p) for p in file_paths if str(p) not in all_processed]
        
        results = []
        
        try:
            # Process remaining files
            for i, file_path in enumerate(file_paths, start=state.current_index):
                try:
                    # Check resource constraints before processing
                    within_limits, resource_msg = self.resource_monitor.check_resource_thresholds()
                    if not within_limits:
                        raise ResourceExhaustionError(resource_msg)
                    
                    # Process file with retry logic
                    self.logger.info(f"Processing file {i+1}/{state.total_documents}: {file_path}")
                    result = self._process_single_pdf(processing_function, file_path)
                    
                    # Record success
                    state.processed_documents.append(str(file_path))
                    results.append({"file": str(file_path), "result": result, "status": "success"})
                    
                except Exception as e:
                    # Record failure
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    state.failed_documents.append((str(file_path), error_msg))
                    results.append({"file": str(file_path), "error": error_msg, "status": "failed"})
                    self.logger.error(f"Failed to process {file_path}: {error_msg}")
                
                # Update state
                state.current_index = i + 1
                
                # Save checkpoint if needed
                if (state.current_index % self.checkpoint_frequency == 0 or 
                    state.current_index == state.total_documents):
                    self.checkpoint_manager.save_checkpoint(state)
                    self.checkpoint_manager.cleanup_old_checkpoints(batch_id)
        
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            self.checkpoint_manager.save_checkpoint(state)
            raise
        
        except Exception as e:
            self.logger.error(f"Unexpected error in batch processing: {e}")
            self.checkpoint_manager.save_checkpoint(state)
            raise
        
        # Final summary
        end_time = datetime.now()
        processing_time = (end_time - state.start_time).total_seconds()
        
        summary = {
            "batch_id": batch_id,
            "total_files": state.total_documents,
            "successful_files": len(state.processed_documents),
            "failed_files": len(state.failed_documents),
            "processing_time_seconds": processing_time,
            "start_time": state.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "results": results,
            "failed_documents": state.failed_documents
        }
        
        self.logger.info(f"Batch processing completed: {summary['successful_files']} successful, "
                        f"{summary['failed_files']} failed, {processing_time:.1f}s total")
        
        return summary
    
    def _process_single_pdf(self, processing_function: Callable, file_path: Path) -> Any:
        """Process a single PDF with retry logic"""
        @self.retry_decorator
        def wrapped_processing():
            return processing_function(file_path)
        
        return wrapped_processing()
    
    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path],
                             processing_function: Callable[[Path], Any]) -> Dict[str, Any]:
# # #         """Resume processing from a specific checkpoint"""  # Module not found  # Module not found  # Module not found
        try:
            state = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
# # #             # Reconstruct file list from processed and failed documents  # Module not found  # Module not found  # Module not found
            all_files = state.processed_documents + [f[0] for f in state.failed_documents]
            
# # #             self.logger.info(f"Resuming batch {state.batch_id} from checkpoint")  # Module not found  # Module not found  # Module not found
            return self.process_pdf_batch(
                file_paths=all_files,
                processing_function=processing_function,
                batch_id=state.batch_id,
                resume_from_checkpoint=True
            )
            
        except Exception as e:
# # #             raise CheckpointError(f"Failed to resume from checkpoint: {e}")  # Module not found  # Module not found  # Module not found


# Convenience function for simple batch processing
def process_pdf_batch_with_error_handling(file_paths: List[Union[str, Path]],
                                         processing_function: Callable[[Path], Any],
                                         **kwargs) -> Dict[str, Any]:
    """
    Convenience function for processing PDF batches with error handling
    
    Args:
        file_paths: List of PDF files to process
        processing_function: Function that takes a Path and returns processing result
        **kwargs: Additional arguments for PDFErrorHandler
    
    Returns:
        Processing results summary
    """
    handler = PDFErrorHandler(**kwargs)
    try:
        return handler.process_pdf_batch(file_paths, processing_function)
    finally:
        # Ensure cleanup
        handler.resource_monitor.stop_monitoring()


if __name__ == "__main__":
    # Example usage
    def example_pdf_processor(file_path: Path) -> Dict[str, Any]:
        """Example processing function"""
        # Simulate processing
        time.sleep(0.1)
        return {"pages": 10, "text_length": 5000, "file": str(file_path)}
    
    # Example batch processing
    test_files = ["test1.pdf", "test2.pdf", "test3.pdf"]
    results = process_pdf_batch_with_error_handling(
        file_paths=test_files,
        processing_function=example_pdf_processor,
        checkpoint_frequency=2,
        max_retry_attempts=3
    )
    
    print(f"Processing completed: {results['successful_files']} successful, {results['failed_files']} failed")