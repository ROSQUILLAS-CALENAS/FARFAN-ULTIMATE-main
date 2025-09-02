#!/usr/bin/env python3
"""
Distributed Processing System for EGW Query Expansion Pipeline

Implements distributed document processing with Redis coordination,
output aggregation, and quality validation for municipal development plans.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

try:
    import redis
except ImportError:
    redis = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

try:
    import torch
except ImportError:
    torch = None

# Advanced serialization with fallback logic
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
# Optional sklearn cosine_similarity with fallback
try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:
    def cosine_similarity(A, B):
        if np is None:
            raise ImportError("numpy is required for cosine similarity calculation")
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_norm @ B_norm.T

# Import EGW components with fallbacks
try:
    from egw_query_expansion.core.hybrid_retrieval import HybridRetrieval
    from egw_query_expansion.core.gw_alignment import GWAlignment
    from egw_query_expansion.core.query_generator import QueryGenerator
    from evidence_processor import EvidenceProcessor
    from answer_synthesizer import AnswerSynthesizer
except ImportError:
    # Create placeholder classes if EGW components aren't available
    class HybridRetrieval:
        def __init__(self, config): pass
        def search(self, query, corpus): return {}
    
    class GWAlignment:
        def __init__(self, config): pass
        def align(self, query, corpus): return []
        
    class QueryGenerator:
        def __init__(self, config): pass
        def generate_expansions(self, query): return [query]
        
    class EvidenceProcessor:
        def __init__(self, config): pass
        def extract_evidence(self, content, results): return []
        
    class AnswerSynthesizer:
        def __init__(self, config): pass
        def synthesize(self, query, evidence, context): return {"answer": "", "summary": ""}


@dataclass
class ProcessingTask:
    """Individual processing task for distributed execution"""
    task_id: str
    document_path: str
    query: str
    priority: int = 1
    created_at: float = None
    assigned_worker: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Result from distributed processing"""
    task_id: str
    worker_id: str
    status: str
    result_data: Dict[str, Any]
    processing_time: float
    quality_metrics: Dict[str, float]
    error_message: Optional[str] = None
    completed_at: float = None

    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = time.time()


@dataclass
class AggregatedResult:
    """Aggregated results from multiple processing instances"""
    request_id: str
    task_ids: List[str]
    combined_results: Dict[str, Any]
    consistency_score: float
    quality_score: float
    total_processing_time: float
    worker_results: List[ProcessingResult]
    aggregated_at: float = None

    def __post_init__(self):
        if self.aggregated_at is None:
            self.aggregated_at = time.time()


class QualityValidator:
    """Quality validation for processing results"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_relevance_score = config.get('min_relevance_score', 0.7)
        self.min_coherence_score = config.get('min_coherence_score', 0.8)
        self.max_response_time = config.get('max_response_time', 30.0)

        # Initialize embeddings for semantic validation (if transformers available)
        if AutoTokenizer and AutoModel:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            except Exception:
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None

    def validate_result(self, result: ProcessingResult, reference_query: str) -> Dict[str, float]:
        """Validate processing result quality"""
        metrics = {}

        try:
            # Relevance validation
            relevance = self._calculate_relevance(
                result.result_data.get('content', ''),
                reference_query
            )
            metrics['relevance_score'] = relevance

            # Coherence validation
            coherence = self._calculate_coherence(result.result_data)
            metrics['coherence_score'] = coherence

            # Performance validation
            performance = self._calculate_performance_score(result)
            metrics['performance_score'] = performance

            # Completeness validation
            completeness = self._calculate_completeness(result.result_data)
            metrics['completeness_score'] = completeness

            # Overall quality score
            scores = [relevance, coherence, performance, completeness]
            metrics['overall_quality'] = sum(scores) / len(scores)

            # Pass/fail determination
            metrics['quality_passed'] = (
                relevance >= self.min_relevance_score and
                coherence >= self.min_coherence_score and
                performance >= 0.6 and
                completeness >= 0.7
            )

        except Exception as e:
            logging.error(f"Quality validation failed: {e}")
            metrics = {
                'relevance_score': 0.0,
                'coherence_score': 0.0,
                'performance_score': 0.0,
                'completeness_score': 0.0,
                'overall_quality': 0.0,
                'quality_passed': False,
                'validation_error': str(e)
            }

        return metrics

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate semantic relevance between content and query"""
        try:
            # Encode content and query
            content_embedding = self._encode_text(content[:512])  # Truncate for efficiency
            query_embedding = self._encode_text(query)

            # Calculate cosine similarity
            similarity = cosine_similarity(
                content_embedding.reshape(1, -1),
                query_embedding.reshape(1, -1)
            )[0][0]

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logging.error(f"Relevance calculation failed: {e}")
            return 0.0

    def _encode_text(self, text: str):
        """Encode text to embeddings"""
        if not (self.tokenizer and self.model and torch):
            # Fallback to simple text similarity
            return list(range(len(text.split()[:50])))  # Simple word count vector
            
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embeddings

    def _calculate_coherence(self, result_data: Dict[str, Any]) -> float:
        """Calculate result coherence score"""
        try:
            # Check for required fields
            required_fields = ['content', 'evidence', 'summary']
            present_fields = sum(1 for field in required_fields if field in result_data)
            field_completeness = present_fields / len(required_fields)

            # Check content length and structure
            content = result_data.get('content', '')
            content_score = min(1.0, len(content) / 1000) if content else 0.0

            # Check evidence quality
            evidence = result_data.get('evidence', [])
            evidence_score = min(1.0, len(evidence) / 5) if evidence else 0.0

            scores = [field_completeness, content_score, evidence_score]
            return sum(scores) / len(scores)

        except Exception as e:
            logging.error(f"Coherence calculation failed: {e}")
            return 0.0

    def _calculate_performance_score(self, result: ProcessingResult) -> float:
        """Calculate performance score based on processing time"""
        if result.processing_time <= self.max_response_time:
            return 1.0
        else:
            # Exponential decay for longer processing times
            import math
            return max(0.0, math.exp(-(result.processing_time - self.max_response_time) / 10.0))

    def _calculate_completeness(self, result_data: Dict[str, Any]) -> float:
        """Calculate result completeness score"""
        expected_keys = {
            'content', 'evidence', 'summary', 'metadata',
            'query_expansion', 'relevance_scores'
        }

        present_keys = set(result_data.keys())
        completeness = len(present_keys & expected_keys) / len(expected_keys)

        return completeness


class ResultAggregator:
    """Aggregates results from multiple processing instances"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consensus_threshold = config.get('consensus_threshold', 0.7)
        self.min_results_for_consensus = config.get('min_results_for_consensus', 2)

    def aggregate_results(self, results: List[ProcessingResult],
                         request_id: str) -> AggregatedResult:
        """Aggregate multiple processing results"""
        if not results:
            raise ValueError("No results to aggregate")

        # Calculate consistency score across results
        consistency_score = self._calculate_consistency(results)

        # Calculate overall quality score
        quality_scores = [r.quality_metrics.get('overall_quality', 0.0) for r in results]
        quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Combine result data using consensus approach
        combined_results = self._combine_result_data(results)

        # Calculate total processing time
        total_time = sum(r.processing_time for r in results)

        task_ids = [r.task_id for r in results]

        return AggregatedResult(
            request_id=request_id,
            task_ids=task_ids,
            combined_results=combined_results,
            consistency_score=consistency_score,
            quality_score=quality_score,
            total_processing_time=total_time,
            worker_results=results
        )

    def _calculate_consistency(self, results: List[ProcessingResult]) -> float:
        """Calculate consistency score across multiple results"""
        if len(results) < 2:
            return 1.0

        consistency_scores = []

        # Compare content similarity across results
        contents = [r.result_data.get('content', '') for r in results]
        if all(contents):
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    # Simple Jaccard similarity for text content
                    words1 = set(contents[i].lower().split())
                    words2 = set(contents[j].lower().split())
                    if words1 or words2:
                        jaccard = len(words1 & words2) / len(words1 | words2)
                        consistency_scores.append(jaccard)

        # Compare quality metrics consistency
        quality_metrics = [r.quality_metrics for r in results]
        if all(quality_metrics):
            for metric_name in ['relevance_score', 'coherence_score']:
                values = [qm.get(metric_name, 0.0) for qm in quality_metrics]
                if values:
                    # Calculate standard deviation without numpy
                    mean_val = sum(values) / len(values)
                    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5
                    consistency_scores.append(max(0.0, 1.0 - std_dev))

        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

    def _combine_result_data(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Combine result data from multiple processing instances"""
        combined = {
            'contents': [],
            'evidence': [],
            'summaries': [],
            'metadata': {},
            'consensus_content': '',
            'consensus_evidence': [],
            'confidence_scores': {}
        }

        # Collect all content
        for result in results:
            data = result.result_data
            combined['contents'].append(data.get('content', ''))
            combined['evidence'].extend(data.get('evidence', []))
            combined['summaries'].append(data.get('summary', ''))

            # Merge metadata
            metadata = data.get('metadata', {})
            for key, value in metadata.items():
                if key not in combined['metadata']:
                    combined['metadata'][key] = []
                combined['metadata'][key].append(value)

        # Generate consensus content
        combined['consensus_content'] = self._generate_consensus_content(
            combined['contents']
        )

        # Generate consensus evidence
        combined['consensus_evidence'] = self._generate_consensus_evidence(
            combined['evidence']
        )

        # Calculate confidence scores
        combined['confidence_scores'] = self._calculate_confidence_scores(results)

        return combined

    def _generate_consensus_content(self, contents: List[str]) -> str:
        """Generate consensus content from multiple content strings"""
        if not contents or not any(contents):
            return ""

        # For simplicity, return the longest non-empty content
        # In practice, you might use more sophisticated NLP techniques
        valid_contents = [c for c in contents if c.strip()]
        if valid_contents:
            return max(valid_contents, key=len)
        return ""

    def _generate_consensus_evidence(self, evidence_list: List[Any]) -> List[Any]:
        """Generate consensus evidence from multiple evidence lists"""
        # Remove duplicates while preserving order
        seen = set()
        consensus = []

        for evidence in evidence_list:
            evidence_str = str(evidence)
            if evidence_str not in seen:
                seen.add(evidence_str)
                consensus.append(evidence)

        return consensus[:10]  # Limit to top 10 evidence items

    def _calculate_confidence_scores(self, results: List[ProcessingResult]) -> Dict[str, float]:
        """Calculate confidence scores for aggregated results"""
        scores = {}

        # Overall confidence based on result quality
        quality_scores = [r.quality_metrics.get('overall_quality', 0.0) for r in results]
        scores['overall_confidence'] = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Consensus confidence
        scores['consensus_confidence'] = self._calculate_consistency(results)

        # Performance confidence
        processing_times = [r.processing_time for r in results]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        scores['performance_confidence'] = max(0.0, 1.0 - avg_time / 60.0)  # Normalize by 1 minute

        return scores


class DistributedProcessor:
    """Main distributed processing coordinator"""

    def __init__(self, worker_id: str = None, redis_url: str = "redis://localhost:6379", 
                 serialization_backend: str = "dill"):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.redis_url = redis_url
        self.redis_client = redis.from_url(redis_url)

        # Initialize serialization manager
        self.serialization_manager = SerializationManager(preferred_backend=serialization_backend)

        # Configuration
        self.config = {
            'batch_size': int(os.getenv('BATCH_SIZE', 32)),
            'max_concurrent_tasks': int(os.getenv('MAX_CONCURRENT_TASKS', 8)),
            'chunk_size': int(os.getenv('CHUNK_SIZE', 1000)),
            'result_ttl': 3600,  # 1 hour
            'task_timeout': 300,  # 5 minutes
            'min_relevance_score': 0.7,
            'min_coherence_score': 0.8,
            'consensus_threshold': 0.7,
            'serialization_backend': serialization_backend
        }

        # Initialize components
        self.quality_validator = QualityValidator(self.config)
        self.result_aggregator = ResultAggregator(self.config)

        # Initialize EGW components
        self.hybrid_retrieval = HybridRetrieval({})
        self.gw_alignment = GWAlignment({})
        self.query_generator = QueryGenerator({})
        self.evidence_processor = EvidenceProcessor({})
        self.answer_synthesizer = AnswerSynthesizer({})

        # Processing state
        self.is_running = False
        self.active_tasks = {}
        self.completed_tasks = {}

        # Performance metrics
        self.processed_count = 0
        self.failed_count = 0
        self.total_processing_time = 0.0

        # Initialize recovery mechanism
        from recovery_system import FailedDocumentsTracker, DocumentRecoveryManager
        self.failed_docs_tracker = FailedDocumentsTracker(
            redis_client=self.redis_client,
            retention_days=config.get('failed_docs_retention_days', 7)
        )
        self.recovery_manager = DocumentRecoveryManager(
            distributed_processor=self,
            failed_docs_tracker=self.failed_docs_tracker,
            config=config
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"DistributedProcessor-{self.worker_id}")
        
        # Log serialization backend information
        backend_info = self.serialization_manager.get_backend_info()
        self.logger.info(f"Initialized with serialization backends: {backend_info['available_backends']}")
        self.logger.info(f"Backend order: {backend_info['backend_order']}")

    async def start_worker(self):
        """Start the distributed worker"""
        self.is_running = True
        self.logger.info(f"Starting distributed worker {self.worker_id}")

        # Register worker
        await self._register_worker()

        # Initialize recovery system and attempt recovery of failed documents
        await self.recovery_manager.initialize()
        await self.recovery_manager.attempt_recovery()

        # Start task processing loop
        try:
            await self._processing_loop()
        except Exception as e:
            self.logger.error(f"Worker processing loop failed: {e}")
        finally:
            await self._unregister_worker()
            self.is_running = False

    async def process_batch(self, documents: List[str], query: str,
                           request_id: str = None) -> AggregatedResult:
        """Process a batch of documents with distributed coordination"""
        if not request_id:
            request_id = f"batch-{uuid.uuid4().hex[:8]}"

        self.logger.info(f"Starting batch processing for request {request_id}")

        # Create tasks
        tasks = []
        for doc_path in documents:
            task = ProcessingTask(
                task_id=f"task-{uuid.uuid4().hex[:8]}",
                document_path=doc_path,
                query=query,
                metadata={'request_id': request_id}
            )
            tasks.append(task)

        # Submit tasks for distributed processing
        await self._submit_tasks(tasks)

        # Wait for results
        results = await self._wait_for_results(
            [task.task_id for task in tasks],
            timeout=self.config['task_timeout']
        )

        # Aggregate results
        if results:
            aggregated = self.result_aggregator.aggregate_results(results, request_id)
            await self._store_aggregated_result(aggregated)
            return aggregated
        else:
            raise RuntimeError("No results received from distributed processing")

    async def _processing_loop(self):
        """Main processing loop for worker"""
        while self.is_running:
            try:
                # Get next task from queue
                task_data = await self._get_next_task()

                if task_data:
                    task = ProcessingTask(**task_data)
                    await self._process_task(task)
                else:
                    # No tasks available, wait briefly
                    await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(5.0)

    async def _process_task(self, task: ProcessingTask):
        """Process individual task"""
        start_time = time.time()
        self.logger.info(f"Processing task {task.task_id}")

        try:
            # Mark task as processing
            self.active_tasks[task.task_id] = task
            await self._update_task_status(task.task_id, "processing", self.worker_id)

            # Perform EGW processing
            result_data = await self._perform_egw_processing(task)

            processing_time = time.time() - start_time

            # Validate result quality
            quality_metrics = self.quality_validator.validate_result(
                ProcessingResult(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    status="completed",
                    result_data=result_data,
                    processing_time=processing_time,
                    quality_metrics={}
                ),
                task.query
            )

            # Create final result
            result = ProcessingResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="completed",
                result_data=result_data,
                processing_time=processing_time,
                quality_metrics=quality_metrics
            )

            # Store result
            await self._store_result(result)

            # Update metrics
            self.processed_count += 1
            self.total_processing_time += processing_time

            # Remove from failed documents if it was previously failed
            await self.failed_docs_tracker.remove_failed_document(task.document_path)

            self.logger.info(
                f"Task {task.task_id} completed in {processing_time:.2f}s "
                f"(quality: {quality_metrics.get('overall_quality', 0.0):.2f})"
            )

        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")

            # Store failure result
            result = ProcessingResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="failed",
                result_data={},
                processing_time=time.time() - start_time,
                quality_metrics={'overall_quality': 0.0},
                error_message=str(e)
            )

            await self._store_result(result)
            self.failed_count += 1

            # Track failed document for recovery
            await self.failed_docs_tracker.track_failed_document(
                document_path=task.document_path,
                task_id=task.task_id,
                error_message=str(e),
                query=task.query,
                worker_id=self.worker_id,
                metadata=task.metadata or {}
            )

        finally:
            # Clean up
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

    async def _perform_egw_processing(self, task: ProcessingTask) -> Dict[str, Any]:
        """Perform EGW query expansion and retrieval processing"""
        # Read document
        document_content = await self._read_document(task.document_path)

        # Generate expanded query
        expanded_queries = self.query_generator.generate_variants(task.query)

        # Perform hybrid retrieval
        retrieval_results = self.hybrid_retrieval.search(
            query=task.query,
            document_content=document_content,
            top_k=10
        )

        # Apply Gromov-Wasserstein alignment
        aligned_results = self.gw_alignment.align_query_corpus(
            query_embedding=retrieval_results.get('query_embedding'),
            corpus_embeddings=retrieval_results.get('corpus_embeddings')
        )

        # Process evidence
        evidence = self.evidence_processor.extract_evidence(
            document_content, retrieval_results
        )

        # Synthesize final answer
        answer = self.answer_synthesizer.synthesize(
            query=task.query,
            evidence=evidence,
            context=retrieval_results
        )

        return {
            'content': answer.get('answer', ''),
            'evidence': evidence,
            'summary': answer.get('summary', ''),
            'query_expansion': expanded_queries,
            'relevance_scores': retrieval_results.get('scores', []),
            'metadata': {
                'document_path': task.document_path,
                'processing_method': 'egw_pipeline',
                'expanded_query_count': len(expanded_queries)
            }
        }

    async def _read_document(self, document_path: str) -> str:
        """Read document content"""
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read document {document_path}: {e}")
            return ""

    # Redis coordination methods
    async def _register_worker(self):
        """Register worker with Redis using advanced serialization"""
        worker_info = {
            'worker_id': self.worker_id,
            'status': 'active',
            'registered_at': time.time(),
            'config': self.config,
            'serialization_backend': self.serialization_manager.get_backend_info()
        }
        try:
            serialized_info = self.serialization_manager.serialize(worker_info)
            self.redis_client.hset(
                "workers",
                self.worker_id,
                serialized_info
            )
            self.redis_client.expire(f"worker:{self.worker_id}", 300)  # 5 minute TTL
            self.logger.debug(f"Worker registered with serialization backend: {self.config['serialization_backend']}")
        except Exception as e:
            # Fallback to JSON if serialization fails
            self.logger.warning(f"Advanced serialization failed for worker registration, falling back to JSON: {e}")
            self.redis_client.hset(
                "workers",
                self.worker_id,
                json.dumps(worker_info, default=str)
            )
            self.redis_client.expire(f"worker:{self.worker_id}", 300)

    async def _unregister_worker(self):
        """Unregister worker from Redis"""
        self.redis_client.hdel("workers", self.worker_id)
        self.redis_client.delete(f"worker:{self.worker_id}")

    async def _submit_tasks(self, tasks: List[ProcessingTask]):
        """Submit tasks to Redis queue using advanced serialization"""
        pipe = self.redis_client.pipeline()

        for task in tasks:
            try:
                task_data = self.serialization_manager.serialize(asdict(task))
                pipe.lpush("task_queue", task_data)
                pipe.hset("tasks", task.task_id, task_data)
            except Exception as e:
                # Fallback to JSON serialization
                self.logger.warning(f"Advanced serialization failed for task {task.task_id}, falling back to JSON: {e}")
                task_data = json.dumps(asdict(task), default=str)
                pipe.lpush("task_queue", task_data)
                pipe.hset("tasks", task.task_id, task_data)

        pipe.execute()

    async def _get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get next task from Redis queue using advanced serialization"""
        task_data = self.redis_client.brpop("task_queue", timeout=10)

        if task_data:
            try:
                # Try advanced deserialization first
                return self.serialization_manager.deserialize(task_data[1])
            except Exception as e:
                # Fallback to JSON deserialization
                self.logger.debug(f"Advanced deserialization failed, falling back to JSON: {e}")
                try:
                    return json.loads(task_data[1])
                except Exception as json_error:
                    self.logger.error(f"Both advanced and JSON deserialization failed: {json_error}")
                    return None
        return None

    async def _update_task_status(self, task_id: str, status: str, worker_id: str):
        """Update task status in Redis using advanced serialization"""
        update_data = {
            'status': status,
            'worker_id': worker_id,
            'updated_at': time.time(),
            'serialization_backend': self.config['serialization_backend']
        }
        try:
            serialized_update = self.serialization_manager.serialize(update_data)
            self.redis_client.hset(f"task:{task_id}", "status_data", serialized_update)
        except Exception as e:
            # Fallback to individual field updates
            self.logger.debug(f"Advanced serialization failed for task status update, using individual fields: {e}")
            self.redis_client.hset(f"task:{task_id}", mapping=update_data)

    async def _store_result(self, result: ProcessingResult):
        """Store processing result in Redis using advanced serialization"""
        try:
            result_data = self.serialization_manager.serialize(asdict(result))
            self.redis_client.hset("results", result.task_id, result_data)
            self.redis_client.expire(f"result:{result.task_id}", self.config['result_ttl'])
        except Exception as e:
            # Fallback to JSON serialization
            self.logger.warning(f"Advanced serialization failed for result storage, falling back to JSON: {e}")
            result_data = json.dumps(asdict(result), default=str)
            self.redis_client.hset("results", result.task_id, result_data)
            self.redis_client.expire(f"result:{result.task_id}", self.config['result_ttl'])

    async def _wait_for_results(self, task_ids: List[str],
                               timeout: float = 300) -> List[ProcessingResult]:
        """Wait for processing results using advanced serialization"""
        results = []
        start_time = time.time()

        while len(results) < len(task_ids) and (time.time() - start_time) < timeout:
            for task_id in task_ids:
                if task_id not in [r.task_id for r in results]:
                    result_data = self.redis_client.hget("results", task_id)
                    if result_data:
                        try:
                            # Try advanced deserialization
                            result_dict = self.serialization_manager.deserialize(result_data)
                            result = ProcessingResult(**result_dict)
                            results.append(result)
                        except Exception as e:
                            # Fallback to JSON deserialization
                            self.logger.debug(f"Advanced deserialization failed for result {task_id}, falling back to JSON: {e}")
                            try:
                                result_dict = json.loads(result_data)
                                result = ProcessingResult(**result_dict)
                                results.append(result)
                            except Exception as json_error:
                                self.logger.error(f"Both advanced and JSON deserialization failed for result {task_id}: {json_error}")

            if len(results) < len(task_ids):
                await asyncio.sleep(1.0)

        return results

    async def _store_aggregated_result(self, aggregated: AggregatedResult):
        """Store aggregated result in Redis using advanced serialization"""
        try:
            result_data = self.serialization_manager.serialize(asdict(aggregated))
            self.redis_client.hset(
                "aggregated_results",
                aggregated.request_id,
                result_data
            )
            self.redis_client.expire(
                f"aggregated_result:{aggregated.request_id}",
                self.config['result_ttl']
            )
        except Exception as e:
            # Fallback to JSON serialization
            self.logger.warning(f"Advanced serialization failed for aggregated result, falling back to JSON: {e}")
            result_data = json.dumps(asdict(aggregated), default=str)
            self.redis_client.hset(
                "aggregated_results",
                aggregated.request_id,
                result_data
            )
            self.redis_client.expire(
                f"aggregated_result:{aggregated.request_id}",
                self.config['result_ttl']
            )


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Distributed EGW Processor")
    parser.add_argument("--worker-mode", action="store_true",
                       help="Run in worker mode")
    parser.add_argument("--coordinator-mode", action="store_true",
                       help="Run in coordinator mode")
    parser.add_argument("--worker-id", type=str,
                       help="Worker ID (auto-generated if not provided)")
    parser.add_argument("--redis-url", type=str,
                       default=os.getenv("REDIS_URL", "redis://localhost:6379"),
                       help="Redis URL")
    parser.add_argument("--documents", type=str, nargs="+",
                       help="Documents to process")
    parser.add_argument("--query", type=str,
                       help="Query for processing")
    parser.add_argument("--serialization-backend", type=str,
                       default="dill", choices=["dill", "cloudpickle", "pickle"],
                       help="Serialization backend to use")

    args = parser.parse_args()

    processor = DistributedProcessor(
        worker_id=args.worker_id,
        redis_url=args.redis_url,
        serialization_backend=args.serialization_backend
    )

    if args.worker_mode:
        # Run as worker
        await processor.start_worker()
    elif args.coordinator_mode and args.documents and args.query:
        # Run as coordinator
        result = await processor.process_batch(args.documents, args.query)
        print(f"Processing completed:")
        print(f"Request ID: {result.request_id}")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Consistency Score: {result.consistency_score:.3f}")
        print(f"Total Processing Time: {result.total_processing_time:.2f}s")
        backend_info = processor.serialization_manager.get_backend_info()
        print(f"Serialization Backend: {backend_info['preferred_backend']}")
        print(f"Available Backends: {backend_info['available_backends']}")
    else:
        print("Please specify --worker-mode or --coordinator-mode with --documents and --query")


if __name__ == "__main__":
    asyncio.run(main())
