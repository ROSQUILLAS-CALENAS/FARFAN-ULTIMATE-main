"""
Parallel PDF Processor Module

Implements a ParallelPDFProcessor class using ProcessPoolExecutor for 
memory-efficient parallel processing of large PDF documents with 
configurable worker count, progress tracking, and recovery mechanisms.
"""

import logging
import time
# # # from concurrent.futures import ProcessPoolExecutor, as_completed  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from queue import Queue  # Module not found  # Module not found  # Module not found
# # # from threading import Lock, Thread  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found

# Advanced serialization with fallback logic

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "58O"
__stage_order__ = 7

class SerializationManager:
    """Manages serialization with fallback logic across multiple backends."""
    
    def __init__(self, preferred_backend: str = "dill"):
        self.preferred_backend = preferred_backend
        self._available_backends = self._detect_available_backends()
        self._backend_order = self._get_backend_order()
        
    def _detect_available_backends(self) -> Dict[str, Any]:
        """Detect available serialization backends."""
        backends = {}
        
        try:
            import dill
            backends['dill'] = dill
        except ImportError:
            pass
            
        try:
            import cloudpickle
            backends['cloudpickle'] = cloudpickle
        except ImportError:
            pass
            
        import pickle
        backends['pickle'] = pickle
        
        return backends
    
    def _get_backend_order(self) -> List[str]:
        """Get the order of backends to try based on preference."""
        if self.preferred_backend in self._available_backends:
            order = [self.preferred_backend]
        else:
            order = []
            
        # Add remaining backends in preferred order
        preferred_order = ['dill', 'cloudpickle', 'pickle']
        for backend in preferred_order:
            if backend in self._available_backends and backend not in order:
                order.append(backend)
                
        return order
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object using fallback logic."""
        last_error = None
        
        for backend_name in self._backend_order:
            backend = self._available_backends[backend_name]
            try:
                return backend.dumps(obj)
            except Exception as e:
                last_error = e
                logging.debug(f"Serialization failed with {backend_name}: {e}")
                continue
                
        raise RuntimeError(f"All serialization backends failed. Last error: {last_error}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize object using fallback logic."""
        last_error = None
        
        for backend_name in self._backend_order:
            backend = self._available_backends[backend_name]
            try:
                return backend.loads(data)
            except Exception as e:
                last_error = e
                logging.debug(f"Deserialization failed with {backend_name}: {e}")
                continue
                
        raise RuntimeError(f"All deserialization backends failed. Last error: {last_error}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends."""
        return {
            'available_backends': list(self._available_backends.keys()),
            'backend_order': self._backend_order,
            'preferred_backend': self.preferred_backend
        }

logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """Represents a chunk of PDF pages for processing."""
    chunk_id: str
    file_path: str
    start_page: int
    end_page: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass 
class ProcessingResult:
    """Result of processing a PDF chunk."""
    chunk_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    pages_processed: int = 0


@dataclass
class ProgressTracker:
    """Tracks processing progress with thread-safe operations."""
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    retry_chunks: int = 0
    start_time: float = field(default_factory=time.time)
    _lock: Lock = field(default_factory=Lock, init=False)
    
    def update_completed(self) -> None:
        """Mark a chunk as completed."""
        with self._lock:
            self.completed_chunks += 1
    
    def update_failed(self) -> None:
        """Mark a chunk as failed."""
        with self._lock:
            self.failed_chunks += 1
    
    def update_retry(self) -> None:
        """Mark a chunk for retry."""
        with self._lock:
            self.retry_chunks += 1
            
    def get_completion_percentage(self) -> float:
        """Get completion percentage."""
        with self._lock:
            if self.total_chunks == 0:
                return 0.0
            return (self.completed_chunks / self.total_chunks) * 100.0
    
    def get_elapsed_time(self) -> float:
        """Get elapsed processing time in seconds."""
        return time.time() - self.start_time
    
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        with self._lock:
            return (self.completed_chunks + self.failed_chunks) >= self.total_chunks


class ParallelPDFProcessor:
    """
    Parallel PDF processor with configurable worker count, chunking,
    queue-based task distribution, progress tracking, and recovery mechanisms.
    """
    
    def __init__(
        self, 
        worker_count: Optional[int] = None,
        chunk_size: int = 10,
        enable_recovery: bool = True,
        recovery_dir: Optional[str] = None,
        serialization_backend: str = "dill"
    ):
        """
        Initialize the parallel PDF processor.
        
        Args:
            worker_count: Number of worker processes (4-8, auto-detected if None)
            chunk_size: Number of pages per chunk
            enable_recovery: Whether to enable recovery mechanisms
            recovery_dir: Directory to store recovery state
            serialization_backend: Preferred serialization backend ('dill', 'cloudpickle', 'pickle')
        """
        # Configure worker count (4-8 workers)
        if worker_count is None:
            import os
            worker_count = min(8, max(4, os.cpu_count() or 4))
        self.worker_count = max(4, min(8, worker_count))
        
        self.chunk_size = chunk_size
        self.enable_recovery = enable_recovery
        self.recovery_dir = Path(recovery_dir or "recovery_state")
        
        # Initialize serialization manager
        self.serialization_manager = SerializationManager(preferred_backend=serialization_backend)
        
        # Internal state
        self.task_queue: Queue = Queue()
        self.result_queue: Queue = Queue()
        self.progress_tracker = ProgressTracker()
        self.failed_chunks: List[PDFChunk] = []
        
        # Ensure recovery directory exists
        if self.enable_recovery:
            self.recovery_dir.mkdir(exist_ok=True)
            
        backend_info = self.serialization_manager.get_backend_info()
        logger.info(f"Initialized ParallelPDFProcessor with {self.worker_count} workers")
        logger.info(f"Serialization backend order: {backend_info['backend_order']}")
    
    def chunk_pdf(self, file_path: str, total_pages: Optional[int] = None) -> List[PDFChunk]:
        """
        Chunk a large PDF into smaller segments for memory-efficient processing.
        
        Args:
            file_path: Path to the PDF file
            total_pages: Total number of pages (auto-detected if None)
            
        Returns:
            List of PDF chunks
        """
        if total_pages is None:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                total_pages = len(doc)
                doc.close()
            except Exception as e:
                logger.error(f"Failed to determine page count for {file_path}: {e}")
                raise ValueError(f"Could not determine page count: {e}")
        
        chunks = []
        for start_page in range(0, total_pages, self.chunk_size):
            end_page = min(start_page + self.chunk_size - 1, total_pages - 1)
            chunk_id = f"{Path(file_path).stem}_chunk_{start_page}_{end_page}"
            
            chunk = PDFChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                start_page=start_page,
                end_page=end_page,
                metadata={
                    "total_pages": total_pages,
                    "chunk_pages": end_page - start_page + 1
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for {file_path} ({total_pages} pages)")
        return chunks
    
    def _save_recovery_state(self, chunks: List[PDFChunk], completed_chunks: List[str]) -> None:
        """Save recovery state to disk using advanced serialization."""
        if not self.enable_recovery:
            return
            
        recovery_state = {
            "chunks": chunks,
            "completed_chunks": completed_chunks,
            "failed_chunks": self.failed_chunks,
            "timestamp": time.time(),
            "serialization_backend": self.serialization_manager.preferred_backend
        }
        
        recovery_file = self.recovery_dir / "processing_state.pkl"
        try:
            serialized_data = self.serialization_manager.serialize(recovery_state)
            with open(recovery_file, 'wb') as f:
                f.write(serialized_data)
            logger.debug(f"Saved recovery state to {recovery_file}")
        except Exception as e:
            logger.warning(f"Failed to save recovery state: {e}")
    
    def _load_recovery_state(self) -> Optional[Dict[str, Any]]:
# # #         """Load recovery state from disk using advanced serialization."""  # Module not found  # Module not found  # Module not found
        if not self.enable_recovery:
            return None
            
        recovery_file = self.recovery_dir / "processing_state.pkl"
        if not recovery_file.exists():
            return None
        
        try:
            with open(recovery_file, 'rb') as f:
                serialized_data = f.read()
            recovery_state = self.serialization_manager.deserialize(serialized_data)
# # #             logger.info(f"Loaded recovery state from {recovery_file}")  # Module not found  # Module not found  # Module not found
            return recovery_state
        except Exception as e:
            logger.warning(f"Failed to load recovery state: {e}")
            return None
    
    def _process_chunk_worker(self, chunk: PDFChunk, processor_func: Callable) -> ProcessingResult:
        """
        Worker function to process a single PDF chunk.
        
        Args:
            chunk: PDF chunk to process
            processor_func: Function to process the chunk
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Initialize serialization manager in worker process
            worker_serialization = SerializationManager(
                preferred_backend=getattr(self, '_worker_serialization_backend', 'dill')
            )
            
            # Process the chunk
            result_data = processor_func(chunk)
            
            processing_time = time.time() - start_time
            pages_processed = chunk.end_page - chunk.start_page + 1
            
            return ProcessingResult(
                chunk_id=chunk.chunk_id,
                success=True,
                data=result_data,
                processing_time=processing_time,
                pages_processed=pages_processed
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Failed to process chunk {chunk.chunk_id}: {error_msg}")
            
            return ProcessingResult(
                chunk_id=chunk.chunk_id,
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def process_pdf_parallel(
        self,
        file_path: str,
        processor_func: Callable,
        total_pages: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF file in parallel using multiple workers.
        
        Args:
            file_path: Path to the PDF file
            processor_func: Function to process each chunk (should accept PDFChunk)
            total_pages: Total number of pages (auto-detected if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processing results and metadata
        """
        # Check for recovery state
        recovery_state = self._load_recovery_state()
        
        if recovery_state:
# # #             logger.info("Resuming from previous processing state")  # Module not found  # Module not found  # Module not found
            chunks = recovery_state["chunks"]
            completed_chunk_ids = set(recovery_state["completed_chunks"])
            self.failed_chunks = recovery_state.get("failed_chunks", [])
            
            # Filter out completed chunks
            pending_chunks = [c for c in chunks if c.chunk_id not in completed_chunk_ids]
        else:
            # Create new chunks
            chunks = self.chunk_pdf(file_path, total_pages)
            pending_chunks = chunks
            completed_chunk_ids = set()
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(total_chunks=len(chunks))
        self.progress_tracker.completed_chunks = len(completed_chunk_ids)
        
        results = {}
        completed_chunks = list(completed_chunk_ids)
        
        logger.info(f"Processing {len(pending_chunks)} pending chunks with {self.worker_count} workers")
        
        # Set serialization backend for worker processes
        self._worker_serialization_backend = self.serialization_manager.preferred_backend
        
        # Process chunks with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.worker_count) as executor:
            # Submit all pending chunks
            future_to_chunk = {
                executor.submit(self._process_chunk_worker, chunk, processor_func): chunk
                for chunk in pending_chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                
                try:
                    result = future.result()
                    
                    if result.success:
                        results[result.chunk_id] = result.data
                        completed_chunks.append(result.chunk_id)
                        self.progress_tracker.update_completed()
                        
                        logger.info(
                            f"Completed chunk {result.chunk_id} "
                            f"({result.pages_processed} pages in {result.processing_time:.2f}s)"
                        )
                    else:
                        # Handle retry logic
                        if chunk.retry_count < chunk.max_retries:
                            chunk.retry_count += 1
                            self.progress_tracker.update_retry()
                            
                            # Resubmit for retry
                            retry_future = executor.submit(self._process_chunk_worker, chunk, processor_func)
                            future_to_chunk[retry_future] = chunk
                            
                            logger.warning(
                                f"Retrying chunk {chunk.chunk_id} "
                                f"(attempt {chunk.retry_count}/{chunk.max_retries})"
                            )
                        else:
                            # Max retries exceeded
                            self.failed_chunks.append(chunk)
                            self.progress_tracker.update_failed()
                            
                            logger.error(
                                f"Chunk {chunk.chunk_id} failed after {chunk.max_retries} retries: "
                                f"{result.error}"
                            )
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(self.progress_tracker.get_completion_percentage())
                    
                    # Save recovery state periodically
                    if len(completed_chunks) % 5 == 0:  # Save every 5 completed chunks
                        self._save_recovery_state(chunks, completed_chunks)
                
                except Exception as e:
                    logger.error(f"Unexpected error processing chunk {chunk.chunk_id}: {e}")
                    self.failed_chunks.append(chunk)
                    self.progress_tracker.update_failed()
        
        # Final recovery state save
        self._save_recovery_state(chunks, completed_chunks)
        
        # Compile final results
        total_processing_time = self.progress_tracker.get_elapsed_time()
        total_pages_processed = sum(
            chunk.metadata.get("chunk_pages", 0) 
            for chunk in chunks 
            if chunk.chunk_id in completed_chunks
        )
        
        processing_summary = {
            "results": results,
            "total_chunks": len(chunks),
            "completed_chunks": self.progress_tracker.completed_chunks,
            "failed_chunks": len(self.failed_chunks),
            "completion_percentage": self.progress_tracker.get_completion_percentage(),
            "total_processing_time": total_processing_time,
            "total_pages_processed": total_pages_processed,
            "average_pages_per_second": (
                total_pages_processed / total_processing_time 
                if total_processing_time > 0 else 0
            ),
            "failed_chunk_details": [
                {"chunk_id": c.chunk_id, "retry_count": c.retry_count} 
                for c in self.failed_chunks
            ]
        }
        
        logger.info(
            f"Processing complete: {processing_summary['completion_percentage']:.1f}% "
            f"({processing_summary['completed_chunks']}/{processing_summary['total_chunks']} chunks) "
            f"in {total_processing_time:.2f}s"
        )
        
        return processing_summary
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        return {
            "completion_percentage": self.progress_tracker.get_completion_percentage(),
            "completed_chunks": self.progress_tracker.completed_chunks,
            "total_chunks": self.progress_tracker.total_chunks,
            "failed_chunks": self.progress_tracker.failed_chunks,
            "elapsed_time": self.progress_tracker.get_elapsed_time(),
            "is_complete": self.progress_tracker.is_complete(),
            "serialization_backend": self.serialization_manager.get_backend_info()
        }
    
    def resume_failed_chunks(self, processor_func: Callable) -> Dict[str, Any]:
        """
        Resume processing of failed chunks.
        
        Args:
            processor_func: Function to process each chunk
            
        Returns:
            Processing results for resumed chunks
        """
        if not self.failed_chunks:
            logger.info("No failed chunks to resume")
            return {"results": {}, "resumed_chunks": 0}
        
        logger.info(f"Resuming {len(self.failed_chunks)} failed chunks")
        
        # Reset retry counts
        for chunk in self.failed_chunks:
            chunk.retry_count = 0
        
        failed_chunks_copy = self.failed_chunks.copy()
        self.failed_chunks.clear()
        
        results = {}
        
        # Set serialization backend for worker processes
        self._worker_serialization_backend = self.serialization_manager.preferred_backend
        
        with ProcessPoolExecutor(max_workers=self.worker_count) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk_worker, chunk, processor_func): chunk
                for chunk in failed_chunks_copy
            }
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                
                try:
                    result = future.result()
                    
                    if result.success:
                        results[result.chunk_id] = result.data
                        logger.info(f"Successfully resumed chunk {result.chunk_id}")
                    else:
                        # Still failed after resume attempt
                        self.failed_chunks.append(chunk)
                        logger.error(f"Chunk {result.chunk_id} still failed after resume: {result.error}")
                
                except Exception as e:
                    logger.error(f"Unexpected error resuming chunk {chunk.chunk_id}: {e}")
                    self.failed_chunks.append(chunk)
        
        return {
            "results": results,
            "resumed_chunks": len(failed_chunks_copy),
            "successful_resumes": len(results),
            "still_failed": len(self.failed_chunks)
        }
    
    def cleanup_recovery_state(self) -> None:
        """Clean up recovery state files."""
        if self.enable_recovery:
            recovery_file = self.recovery_dir / "processing_state.pkl"
            if recovery_file.exists():
                recovery_file.unlink()
                logger.info("Cleaned up recovery state")


# Utility function for default PDF text extraction
def default_pdf_chunk_processor(chunk: PDFChunk) -> Dict[str, Any]:
    """
    Default processor function for PDF chunks.
# # #     Extracts text from the specified page range.  # Module not found  # Module not found  # Module not found
    
    Args:
        chunk: PDF chunk to process
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(chunk.file_path)
        text_content = []
        
        for page_num in range(chunk.start_page, chunk.end_page + 1):
            if page_num < len(doc):
                page = doc[page_num]
                text = page.get_text()
                text_content.append({
                    "page_num": page_num,
                    "text": text,
                    "char_count": len(text)
                })
        
        doc.close()
        
        return {
            "chunk_id": chunk.chunk_id,
            "pages": text_content,
            "total_chars": sum(p["char_count"] for p in text_content),
            "page_range": f"{chunk.start_page}-{chunk.end_page}"
        }
    
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF chunk {chunk.chunk_id}: {e}")


if __name__ == "__main__":
    # Example usage with configurable serialization
    processor = ParallelPDFProcessor(
        worker_count=6, 
        chunk_size=5, 
        serialization_backend="dill"  # Can be "dill", "cloudpickle", or "pickle"
    )
    
    # Example progress callback
    def progress_callback(percentage):
        print(f"Progress: {percentage:.1f}%")
    
    # Display serialization backend information
    backend_info = processor.serialization_manager.get_backend_info()
    print(f"Available serialization backends: {backend_info['available_backends']}")
    print(f"Backend order: {backend_info['backend_order']}")
    
    # Process a PDF file
    try:
        results = processor.process_pdf_parallel(
            "example.pdf",
            default_pdf_chunk_processor,
            progress_callback=progress_callback
        )
        
        print(f"Processing completed: {results['completion_percentage']:.1f}%")
        print(f"Total pages processed: {results['total_pages_processed']}")
        
    except Exception as e:
        print(f"Processing failed: {e}")