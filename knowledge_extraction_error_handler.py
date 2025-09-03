"""
Comprehensive error handling for K_knowledge_extraction stage components.

This module provides:
- Timeout decorators with configurable limits
- Memory monitoring with garbage collection triggers
- Graceful failure isolation for individual chunks
- Component-specific error logging
- Deterministic fallback behavior with stable ordering
# # # - Recovery from partial failures without corrupting downstream artifacts  # Module not found  # Module not found  # Module not found
"""

import gc
import logging
import os
import psutil
import time
import traceback
# # # from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from functools import wraps  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found
import threading


@dataclass
class ChunkProcessingResult:
    """Result of processing an individual chunk."""
    chunk_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    memory_peak: float = 0.0
    timed_out: bool = False
    
    
@dataclass
class ComponentErrorStats:
    """Error statistics for a component."""
    component_name: str
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    timeout_chunks: int = 0
    memory_errors: int = 0
    avg_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    error_messages: List[str] = field(default_factory=list)


class KnowledgeExtractionErrorHandler:
    """Comprehensive error handler for knowledge extraction components."""
    
    def __init__(self, 
                 default_timeout: float = 300.0,
                 memory_threshold: float = 0.8,
                 max_retries: int = 2,
                 log_dir: Optional[str] = None):
        """Initialize error handler.
        
        Args:
            default_timeout: Default timeout per component in seconds
            memory_threshold: Memory usage threshold to trigger garbage collection (0.0-1.0)
            max_retries: Maximum retry attempts for failed chunks
            log_dir: Directory for component-specific error logs
        """
        self.default_timeout = default_timeout
        self.memory_threshold = memory_threshold
        self.max_retries = max_retries
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path("logs/knowledge_extraction")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Component statistics
        self.component_stats: Dict[str, ComponentErrorStats] = {}
        
        # Failed chunks for recovery
        self.failed_chunks: Dict[str, ChunkProcessingResult] = OrderedDict()
        
        # Thread-safe locks
        self._stats_lock = threading.Lock()
        self._failed_chunks_lock = threading.Lock()
        
        # Process monitoring
        self.process = psutil.Process()
        self._initial_memory = self._get_memory_usage()
        
        # Setup main logger
        self.logger = self._setup_main_logger()
        
    def _setup_main_logger(self) -> logging.Logger:
        """Setup main error handler logger."""
        logger = logging.getLogger(f"knowledge_extraction_error_handler")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.log_dir / "error_handler_main.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _setup_component_logger(self, component_name: str) -> logging.Logger:
        """Setup component-specific logger."""
        logger = logging.getLogger(f"knowledge_extraction_{component_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.log_dir / f"{component_name}_errors.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [Chunk: %(chunk_id)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage as percentage of total system memory."""
        return self.process.memory_percent()
        
    def _should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        current_usage = self._get_memory_usage() / 100.0
        return current_usage >= self.memory_threshold
        
    def _trigger_garbage_collection(self):
        """Trigger garbage collection and log memory stats."""
        before_mem = self._get_memory_usage()
        collected = gc.collect()
        after_mem = self._get_memory_usage()
        
        self.logger.info(
            f"Garbage collection triggered: {collected} objects collected, "
            f"memory: {before_mem:.2f}% -> {after_mem:.2f}%"
        )
        
    def timeout_decorator(self, timeout: Optional[float] = None, component_name: str = "unknown"):
        """Decorator for adding timeout and memory monitoring to component functions."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                actual_timeout = timeout or self.default_timeout
                component_logger = self._setup_component_logger(component_name)
                
                # Extract chunk_id if available
                chunk_id = "unknown"
                if args and hasattr(args[0], 'chunk_id'):
                    chunk_id = args[0].chunk_id
                elif 'chunk_id' in kwargs:
                    chunk_id = kwargs['chunk_id']
                elif len(args) > 1 and isinstance(args[1], dict) and 'chunk_id' in args[1]:
                    chunk_id = args[1]['chunk_id']
                
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                # Check memory before processing
                if self._should_trigger_gc():
                    self._trigger_garbage_collection()
                
                try:
                    # Use ThreadPoolExecutor for timeout handling
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        result = future.result(timeout=actual_timeout)
                        
                    processing_time = time.time() - start_time
                    peak_memory = max(start_memory, self._get_memory_usage())
                    
                    # Log successful processing
                    component_logger.info(
                        f"Successfully processed in {processing_time:.2f}s, "
                        f"peak memory: {peak_memory:.2f}%",
                        extra={'chunk_id': chunk_id}
                    )
                    
                    # Update statistics
                    self._update_component_stats(
                        component_name, chunk_id, True, processing_time, peak_memory
                    )
                    
                    return result
                    
                except FutureTimeoutError:
                    processing_time = time.time() - start_time
                    error_msg = f"Component '{component_name}' timed out after {actual_timeout}s"
                    
                    component_logger.error(
                        error_msg,
                        extra={'chunk_id': chunk_id}
                    )
                    
                    self._update_component_stats(
                        component_name, chunk_id, False, processing_time, 
                        self._get_memory_usage(), timeout=True, error_msg=error_msg
                    )
                    
                    # Return deterministic fallback
                    return self._get_deterministic_fallback(component_name, chunk_id)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    error_msg = f"Component '{component_name}' failed: {str(e)}"
                    
                    component_logger.error(
                        f"{error_msg}\nTraceback:\n{traceback.format_exc()}",
                        extra={'chunk_id': chunk_id}
                    )
                    
                    self._update_component_stats(
                        component_name, chunk_id, False, processing_time,
                        self._get_memory_usage(), error_msg=error_msg
                    )
                    
                    # Return deterministic fallback
                    return self._get_deterministic_fallback(component_name, chunk_id)
                    
            return wrapper
        return decorator
        
    def _update_component_stats(self, 
                               component_name: str, 
                               chunk_id: str, 
                               success: bool, 
                               processing_time: float, 
                               memory_usage: float,
                               timeout: bool = False,
                               error_msg: Optional[str] = None):
        """Update component statistics."""
        with self._stats_lock:
            if component_name not in self.component_stats:
                self.component_stats[component_name] = ComponentErrorStats(component_name)
                
            stats = self.component_stats[component_name]
            stats.total_chunks += 1
            
            if success:
                stats.successful_chunks += 1
            else:
                stats.failed_chunks += 1
                if timeout:
                    stats.timeout_chunks += 1
                if error_msg and "memory" in error_msg.lower():
                    stats.memory_errors += 1
                if error_msg:
                    stats.error_messages.append(error_msg)
                    
            # Update averages
            stats.avg_processing_time = (
                (stats.avg_processing_time * (stats.total_chunks - 1) + processing_time) 
                / stats.total_chunks
            )
            stats.peak_memory_usage = max(stats.peak_memory_usage, memory_usage)
            
            # Track failed chunk for recovery
            if not success:
                with self._failed_chunks_lock:
                    self.failed_chunks[chunk_id] = ChunkProcessingResult(
                        chunk_id=chunk_id,
                        success=False,
                        error_message=error_msg,
                        processing_time=processing_time,
                        memory_peak=memory_usage,
                        timed_out=timeout
                    )
                    
    def _get_deterministic_fallback(self, component_name: str, chunk_id: str) -> Dict[str, Any]:
        """Generate deterministic fallback result for failed components."""
        fallback_results = {
            'key_terms_extraction': {
                'key_terms': [],
                'extraction_confidence': 0.0,
                'fallback_reason': 'component_failure'
            },
            'concept_extraction': {
                'concepts': [],
                'concept_confidence': 0.0,
                'fallback_reason': 'component_failure'
            },
            'entity_extraction': {
                'entities': [],
                'entity_confidence': 0.0,
                'fallback_reason': 'component_failure'
            }
        }
        
        return fallback_results.get(component_name, {
            'result': None,
            'confidence': 0.0,
            'fallback_reason': 'component_failure',
            'component_name': component_name,
            'chunk_id': chunk_id
        })
        
    def process_chunks_with_isolation(self, 
                                    chunks: List[Dict[str, Any]], 
                                    component_func: Callable,
                                    component_name: str,
                                    timeout: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process chunks with graceful failure isolation.
        
        Args:
            chunks: List of chunks to process
            component_func: Component function to apply
            component_name: Name of the component for logging
            timeout: Optional timeout override
            
        Returns:
            Tuple of (successful_results, failed_chunk_ids)
        """
        decorated_func = self.timeout_decorator(timeout, component_name)(component_func)
        successful_results = []
        failed_chunk_ids = []
        
        component_logger = self._setup_component_logger(component_name)
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', f"chunk_{len(successful_results) + len(failed_chunk_ids)}")
            
            try:
                result = decorated_func(chunk)
                if result is not None:
                    # Ensure stable ordering by preserving chunk_id
                    if isinstance(result, dict):
                        result['chunk_id'] = chunk_id
                        result['processing_order'] = len(successful_results)
                    successful_results.append(result)
                else:
                    failed_chunk_ids.append(chunk_id)
                    component_logger.warning(
                        f"Component returned None result",
                        extra={'chunk_id': chunk_id}
                    )
                    
            except Exception as e:
                failed_chunk_ids.append(chunk_id)
                component_logger.error(
                    f"Unexpected error in chunk isolation: {str(e)}",
                    extra={'chunk_id': chunk_id}
                )
                
        # Sort successful results by processing order to maintain deterministic output
        successful_results.sort(key=lambda x: x.get('processing_order', 0))
        
        return successful_results, failed_chunk_ids
        
    def retry_failed_chunks(self, 
                           component_func: Callable,
                           component_name: str,
                           timeout: Optional[float] = None,
                           max_retries: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retry processing of failed chunks.
        
        Args:
            component_func: Component function to retry
            component_name: Name of the component
            timeout: Optional timeout override
            max_retries: Optional max retries override
            
        Returns:
            List of successfully recovered results
        """
        actual_max_retries = max_retries or self.max_retries
        recovered_results = []
        component_logger = self._setup_component_logger(component_name)
        
        with self._failed_chunks_lock:
            failed_chunks_copy = dict(self.failed_chunks)
            
        for chunk_id, failed_result in failed_chunks_copy.items():
            if failed_result.timed_out and actual_max_retries == 0:
                # Skip retry for timed out chunks if retries disabled
                continue
                
            for attempt in range(actual_max_retries):
                try:
                    # Create mock chunk data for retry
                    retry_chunk = {'chunk_id': chunk_id}
                    decorated_func = self.timeout_decorator(timeout, component_name)(component_func)
                    result = decorated_func(retry_chunk)
                    
                    if result is not None:
                        recovered_results.append(result)
                        component_logger.info(
                            f"Successfully recovered on attempt {attempt + 1}",
                            extra={'chunk_id': chunk_id}
                        )
                        
# # #                         # Remove from failed chunks  # Module not found  # Module not found  # Module not found
                        with self._failed_chunks_lock:
                            self.failed_chunks.pop(chunk_id, None)
                        break
                        
                except Exception as e:
                    component_logger.warning(
                        f"Retry attempt {attempt + 1} failed: {str(e)}",
                        extra={'chunk_id': chunk_id}
                    )
                    
                    if attempt == actual_max_retries - 1:
                        component_logger.error(
                            f"All retry attempts exhausted",
                            extra={'chunk_id': chunk_id}
                        )
                        
        return recovered_results
        
    def get_component_stats(self, component_name: Optional[str] = None) -> Union[ComponentErrorStats, Dict[str, ComponentErrorStats]]:
        """Get error statistics for component(s)."""
        with self._stats_lock:
            if component_name:
                return self.component_stats.get(component_name)
            return dict(self.component_stats)
            
    def get_failed_chunks_summary(self) -> Dict[str, Any]:
        """Get summary of failed chunks."""
        with self._failed_chunks_lock:
            return {
                'total_failed': len(self.failed_chunks),
                'timeout_failures': sum(1 for r in self.failed_chunks.values() if r.timed_out),
                'error_failures': sum(1 for r in self.failed_chunks.values() if not r.timed_out),
                'failed_chunk_ids': list(self.failed_chunks.keys()),
                'avg_processing_time': sum(r.processing_time for r in self.failed_chunks.values()) / len(self.failed_chunks) if self.failed_chunks else 0
            }
            
    def clear_failed_chunks(self):
        """Clear failed chunks registry."""
        with self._failed_chunks_lock:
            self.failed_chunks.clear()
            
    def generate_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report."""
        report = {
            'timestamp': time.time(),
            'overall_memory_usage': self._get_memory_usage(),
            'components': {},
            'failed_chunks': self.get_failed_chunks_summary(),
            'system_info': {
                'memory_threshold': self.memory_threshold,
                'default_timeout': self.default_timeout,
                'max_retries': self.max_retries
            }
        }
        
        with self._stats_lock:
            for component_name, stats in self.component_stats.items():
                report['components'][component_name] = {
                    'total_chunks': stats.total_chunks,
                    'success_rate': stats.successful_chunks / stats.total_chunks if stats.total_chunks > 0 else 0,
                    'failed_chunks': stats.failed_chunks,
                    'timeout_chunks': stats.timeout_chunks,
                    'memory_errors': stats.memory_errors,
                    'avg_processing_time': stats.avg_processing_time,
                    'peak_memory_usage': stats.peak_memory_usage,
                    'recent_errors': stats.error_messages[-5:]  # Last 5 errors
                }
                
        return report


class KnowledgeExtractionPipeline:
    """Enhanced pipeline with comprehensive error handling."""
    
    def __init__(self, 
                 error_handler: Optional[KnowledgeExtractionErrorHandler] = None,
                 **handler_kwargs):
        """Initialize pipeline with error handling.
        
        Args:
            error_handler: Optional pre-configured error handler
            **handler_kwargs: Arguments for creating new error handler
        """
        self.error_handler = error_handler or KnowledgeExtractionErrorHandler(**handler_kwargs)
        self.logger = logging.getLogger(f"{__name__}.KnowledgeExtractionPipeline")
        
    def extract_key_terms(self, text: str, chunk_id: str = "unknown") -> Dict[str, Any]:
        """Extract key terms with error handling."""
        import re
        
        # Simulate processing time for testing
        if len(text) > 10000:
            time.sleep(0.1)
            
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        unique_words = list(set(words))[:10]
        
        return {
            'key_terms': unique_words,
            'extraction_confidence': min(1.0, len(unique_words) / 10.0),
            'text_length': len(text),
            'chunk_id': chunk_id
        }
        
    def extract_concepts(self, text: str, chunk_id: str = "unknown") -> Dict[str, Any]:
        """Extract concepts with error handling."""
        text_lower = text.lower()
        concepts = []
        
        concept_patterns = {
            'desarrollo': ['development', 'desarrollo'],
            'sostenibilidad': ['sustainable', 'sostenible', 'sustentable'],
            'comunidad': ['community', 'comunidad'],
            'medio ambiente': ['environment', 'medio ambiente', 'medioambiente']
        }
        
        for concept, patterns in concept_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                concepts.append(concept)
                
        return {
            'concepts': concepts,
            'concept_confidence': min(1.0, len(concepts) / 4.0),
            'text_length': len(text),
            'chunk_id': chunk_id
        }
        
    def extract_entities(self, text: str, chunk_id: str = "unknown") -> Dict[str, Any]:
        """Extract entities with error handling."""
        import re
        
        # Extract proper nouns and potential entities
        entities = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
        unique_entities = list(set(entities))[:5]
        
        return {
            'entities': unique_entities,
            'entity_confidence': min(1.0, len(unique_entities) / 5.0),
            'text_length': len(text),
            'chunk_id': chunk_id
        }
        
    def process_knowledge_extraction(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process knowledge extraction for all chunks with comprehensive error handling."""
        results = {
            'key_terms_results': [],
            'concept_results': [],
            'entity_results': [],
            'failed_chunks': [],
            'processing_summary': {}
        }
        
        # Decorate component functions with error handling
        safe_extract_key_terms = self.error_handler.timeout_decorator(
            timeout=60.0, component_name="key_terms_extraction"
        )(self.extract_key_terms)
        
        safe_extract_concepts = self.error_handler.timeout_decorator(
            timeout=60.0, component_name="concept_extraction"  
        )(self.extract_concepts)
        
        safe_extract_entities = self.error_handler.timeout_decorator(
            timeout=60.0, component_name="entity_extraction"
        )(self.extract_entities)
        
        # Process each component with isolation
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', f"chunk_{len(results['key_terms_results'])}")
            text = chunk.get('text', '')
            
            # Key terms extraction
            try:
                key_terms_result = safe_extract_key_terms(text, chunk_id)
                results['key_terms_results'].append(key_terms_result)
            except Exception as e:
                self.logger.warning(f"Key terms extraction failed for chunk {chunk_id}: {e}")
                results['failed_chunks'].append(chunk_id)
                
            # Concept extraction
            try:
                concepts_result = safe_extract_concepts(text, chunk_id)
                results['concept_results'].append(concepts_result)
            except Exception as e:
                self.logger.warning(f"Concept extraction failed for chunk {chunk_id}: {e}")
                
            # Entity extraction
            try:
                entities_result = safe_extract_entities(text, chunk_id)
                results['entity_results'].append(entities_result)
            except Exception as e:
                self.logger.warning(f"Entity extraction failed for chunk {chunk_id}: {e}")
                
        # Generate processing summary
        results['processing_summary'] = {
            'total_chunks': len(chunks),
            'successful_key_terms': len(results['key_terms_results']),
            'successful_concepts': len(results['concept_results']),
            'successful_entities': len(results['entity_results']),
            'failed_chunks_count': len(set(results['failed_chunks'])),
            'error_stats': self.error_handler.get_component_stats(),
            'memory_peak': self.error_handler._get_memory_usage()
        }
        
        return results