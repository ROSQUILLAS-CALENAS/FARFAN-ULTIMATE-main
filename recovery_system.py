#!/usr/bin/env python3
"""
Document Recovery System for Distributed Processing

Implements a comprehensive recovery mechanism that tracks previously failed
document processing attempts and automatically reprocesses them after 
import issues are resolved.
"""

import asyncio
import json
import logging
import time
import uuid
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # Mock Redis for testing without dependencies
    class MockRedis:
        def __init__(self, *args, **kwargs):
            pass
        
        def from_url(self, url):
            return self
            
        def hdel(self, *args):
            return 1
            
        def hgetall(self, key):
            return {}
            
        def hget(self, key, field):
            return None
            
        def hset(self, key, field, value):
            return 1
            
        def get(self, key):
            return None
            
        def set(self, key, value, ex=None):
            return True
    
    redis = type('redis', (), {'Redis': MockRedis, 'from_url': lambda url: MockRedis()})()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Simple numpy replacement for basic operations
    class MockNumPy:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
    np = MockNumPy()


@dataclass
class FailedDocumentRecord:
    """Record of a failed document processing attempt"""
    document_path: str
    task_id: str
    error_message: str
    failure_timestamp: float
    retry_count: int
    query: str
    worker_id: str
    metadata: Dict[str, Any]
    recovery_attempts: List[float]
    last_attempt_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailedDocumentRecord':
# # #         """Create from dictionary"""  # Module not found  # Module not found  # Module not found
        return cls(**data)
    
    def get_failure_age_hours(self) -> float:
        """Get age of failure in hours"""
        return (time.time() - self.failure_timestamp) / 3600
    
    def should_retry(self, max_retry_count: int = 3, min_retry_interval_hours: float = 1.0) -> bool:
        """Check if document should be retried"""
        if self.retry_count >= max_retry_count:
            return False
            
        if self.last_attempt_timestamp:
            hours_since_last = (time.time() - self.last_attempt_timestamp) / 3600
            return hours_since_last >= min_retry_interval_hours
            
        return True


@dataclass 
class RecoveryMetrics:
    """Metrics for recovery operations"""
    total_failed_documents: int
    documents_eligible_for_retry: int
    recovery_attempts: int
    successful_recoveries: int
    failed_recoveries: int
    recovery_success_rate: float
    average_recovery_time: float
    last_recovery_run: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class FailedDocumentsTracker:
    """Tracks and manages failed document processing attempts"""
    
    def __init__(self, redis_client: redis.Redis, retention_days: int = 7):
        self.redis_client = redis_client
        self.retention_days = retention_days
        self.failed_docs_key = "failed_documents"
        self.recovery_metrics_key = "recovery_metrics"
        self.logger = logging.getLogger(__name__)
    
    async def track_failed_document(self,
                                   document_path: str,
                                   task_id: str,
                                   error_message: str,
                                   query: str,
                                   worker_id: str,
                                   metadata: Dict[str, Any]) -> None:
        """Track a failed document processing attempt"""
        try:
            # Create failure record
            failed_record = FailedDocumentRecord(
                document_path=document_path,
                task_id=task_id,
                error_message=error_message,
                failure_timestamp=time.time(),
                retry_count=0,
                query=query,
                worker_id=worker_id,
                metadata=metadata,
                recovery_attempts=[]
            )
            
            # Check if document already failed before
            existing_record = await self._get_failed_document_record(document_path)
            if existing_record:
                # Update existing record
                failed_record.retry_count = existing_record.retry_count
                failed_record.recovery_attempts = existing_record.recovery_attempts
                failed_record.last_attempt_timestamp = existing_record.last_attempt_timestamp
            
            # Store in Redis
            await self._store_failed_document_record(failed_record)
            
            self.logger.warning(
                f"Tracked failed document: {document_path} (task: {task_id}, "
                f"retry_count: {failed_record.retry_count})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to track failed document {document_path}: {e}")
    
    async def remove_failed_document(self, document_path: str) -> bool:
# # #         """Remove a document from failed tracking (successful processing)"""  # Module not found  # Module not found  # Module not found
        try:
# # #             # Remove from Redis  # Module not found  # Module not found  # Module not found
            await asyncio.to_thread(
                self.redis_client.hdel, 
                self.failed_docs_key, 
                document_path
            )
            
# # #             self.logger.info(f"Removed successfully processed document from failed tracking: {document_path}")  # Module not found  # Module not found  # Module not found
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove failed document {document_path}: {e}")
            return False
    
    async def get_failed_documents(self,
                                  eligible_for_retry: bool = False,
                                  max_age_hours: Optional[float] = None) -> List[FailedDocumentRecord]:
        """Get failed documents, optionally filtered"""
        try:
            # Get all failed document records
            failed_docs_data = await asyncio.to_thread(
                self.redis_client.hgetall,
                self.failed_docs_key
            )
            
            failed_docs = []
            for doc_path_bytes, record_data_bytes in failed_docs_data.items():
                try:
                    doc_path = doc_path_bytes.decode('utf-8')
                    record_data = json.loads(record_data_bytes.decode('utf-8'))
                    failed_record = FailedDocumentRecord.from_dict(record_data)
                    
                    # Apply filters
                    if max_age_hours and failed_record.get_failure_age_hours() > max_age_hours:
                        continue
                        
                    if eligible_for_retry and not failed_record.should_retry():
                        continue
                    
                    failed_docs.append(failed_record)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse failed document record for {doc_path}: {e}")
                    continue
            
            return failed_docs
            
        except Exception as e:
            self.logger.error(f"Failed to get failed documents: {e}")
            return []
    
    async def update_recovery_attempt(self, document_path: str) -> None:
        """Update recovery attempt timestamp for a document"""
        try:
            failed_record = await self._get_failed_document_record(document_path)
            if failed_record:
                failed_record.recovery_attempts.append(time.time())
                failed_record.last_attempt_timestamp = time.time()
                failed_record.retry_count += 1
                
                await self._store_failed_document_record(failed_record)
                
        except Exception as e:
            self.logger.error(f"Failed to update recovery attempt for {document_path}: {e}")
    
    async def cleanup_old_records(self) -> int:
        """Clean up old failed document records"""
        try:
            cutoff_time = time.time() - (self.retention_days * 24 * 3600)
            failed_docs = await self.get_failed_documents()
            
            removed_count = 0
            for failed_doc in failed_docs:
                if failed_doc.failure_timestamp < cutoff_time:
                    await asyncio.to_thread(
                        self.redis_client.hdel,
                        self.failed_docs_key,
                        failed_doc.document_path
                    )
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old failed document records")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
            return 0
    
    async def get_recovery_metrics(self) -> RecoveryMetrics:
        """Get recovery metrics"""
        try:
            metrics_data = await asyncio.to_thread(
                self.redis_client.get,
                self.recovery_metrics_key
            )
            
            if metrics_data:
                metrics_dict = json.loads(metrics_data.decode('utf-8'))
                return RecoveryMetrics(**metrics_dict)
            else:
                # Initialize default metrics
                return RecoveryMetrics(
                    total_failed_documents=0,
                    documents_eligible_for_retry=0,
                    recovery_attempts=0,
                    successful_recoveries=0,
                    failed_recoveries=0,
                    recovery_success_rate=0.0,
                    average_recovery_time=0.0,
                    last_recovery_run=0.0
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get recovery metrics: {e}")
            return RecoveryMetrics(
                total_failed_documents=0,
                documents_eligible_for_retry=0,
                recovery_attempts=0,
                successful_recoveries=0,
                failed_recoveries=0,
                recovery_success_rate=0.0,
                average_recovery_time=0.0,
                last_recovery_run=0.0
            )
    
    async def update_recovery_metrics(self, 
                                     successful_recoveries: int = 0,
                                     failed_recoveries: int = 0,
                                     recovery_time: float = 0.0) -> None:
        """Update recovery metrics"""
        try:
            current_metrics = await self.get_recovery_metrics()
            failed_docs = await self.get_failed_documents()
            eligible_docs = await self.get_failed_documents(eligible_for_retry=True)
            
            # Update metrics
            current_metrics.total_failed_documents = len(failed_docs)
            current_metrics.documents_eligible_for_retry = len(eligible_docs)
            current_metrics.recovery_attempts += (successful_recoveries + failed_recoveries)
            current_metrics.successful_recoveries += successful_recoveries
            current_metrics.failed_recoveries += failed_recoveries
            current_metrics.last_recovery_run = time.time()
            
            # Calculate success rate
            total_attempts = current_metrics.successful_recoveries + current_metrics.failed_recoveries
            if total_attempts > 0:
                current_metrics.recovery_success_rate = current_metrics.successful_recoveries / total_attempts
            
            # Update average recovery time
            if recovery_time > 0:
                if current_metrics.average_recovery_time == 0:
                    current_metrics.average_recovery_time = recovery_time
                else:
                    # Exponential moving average
                    alpha = 0.1
                    current_metrics.average_recovery_time = (
                        alpha * recovery_time + 
                        (1 - alpha) * current_metrics.average_recovery_time
                    )
            
            # Store updated metrics
            await asyncio.to_thread(
                self.redis_client.set,
                self.recovery_metrics_key,
                json.dumps(current_metrics.to_dict()),
                ex=86400  # 24 hours TTL
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update recovery metrics: {e}")
    
    async def _get_failed_document_record(self, document_path: str) -> Optional[FailedDocumentRecord]:
# # #         """Get failed document record from Redis"""  # Module not found  # Module not found  # Module not found
        try:
            record_data = await asyncio.to_thread(
                self.redis_client.hget,
                self.failed_docs_key,
                document_path
            )
            
            if record_data:
                record_dict = json.loads(record_data.decode('utf-8'))
                return FailedDocumentRecord.from_dict(record_dict)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get failed document record for {document_path}: {e}")
            return None
    
    async def _store_failed_document_record(self, failed_record: FailedDocumentRecord) -> None:
        """Store failed document record in Redis"""
        try:
            await asyncio.to_thread(
                self.redis_client.hset,
                self.failed_docs_key,
                failed_record.document_path,
                json.dumps(failed_record.to_dict())
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store failed document record: {e}")
            raise


class DocumentRecoveryManager:
    """Manages recovery of failed documents"""
    
    def __init__(self,
                 distributed_processor,  # DistributedProcessor instance
                 failed_docs_tracker: FailedDocumentsTracker,
                 config: Dict[str, Any]):
        self.distributed_processor = distributed_processor
        self.failed_docs_tracker = failed_docs_tracker
        self.config = config
        
        # Recovery configuration
        self.max_retry_count = config.get('max_retry_count', 3)
        self.min_retry_interval_hours = config.get('min_retry_interval_hours', 1.0)
        self.recovery_batch_size = config.get('recovery_batch_size', 10)
        self.recovery_interval_minutes = config.get('recovery_interval_minutes', 30)
        
        # State
        self.is_recovery_running = False
        self.recovery_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize recovery manager"""
        try:
            # Clean up old records
            await self.failed_docs_tracker.cleanup_old_records()
            
            # Start periodic recovery if configured
            if self.config.get('enable_periodic_recovery', True):
                await self._start_periodic_recovery()
            
            self.logger.info("Document recovery manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize recovery manager: {e}")
            raise
    
    async def attempt_recovery(self, 
                              document_paths: Optional[List[str]] = None,
                              force_retry: bool = False) -> Dict[str, Any]:
        """Attempt recovery of failed documents"""
        recovery_start_time = time.time()
        self.logger.info("Starting document recovery attempt")
        
        try:
            # Get failed documents to recover
            if document_paths:
                # Specific documents requested
                failed_docs = []
                for doc_path in document_paths:
                    failed_record = await self.failed_docs_tracker._get_failed_document_record(doc_path)
                    if failed_record:
                        failed_docs.append(failed_record)
            else:
                # Get all eligible documents
                failed_docs = await self.failed_docs_tracker.get_failed_documents(
                    eligible_for_retry=not force_retry
                )
            
            if not failed_docs:
                self.logger.info("No failed documents eligible for recovery")
                return {
                    'attempted_documents': 0,
                    'successful_recoveries': 0,
                    'failed_recoveries': 0,
                    'recovery_time': 0.0
                }
            
            # Limit batch size
            failed_docs = failed_docs[:self.recovery_batch_size]
            
            self.logger.info(
                f"Attempting recovery of {len(failed_docs)} failed documents"
            )
            
            successful_recoveries = 0
            failed_recoveries = 0
            
            # Process recovery in parallel batches
            recovery_tasks = []
            for failed_doc in failed_docs:
                recovery_task = self._recover_single_document(failed_doc)
                recovery_tasks.append(recovery_task)
            
            # Wait for all recovery attempts
            recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
            
            # Count results
            for i, result in enumerate(recovery_results):
                failed_doc = failed_docs[i]
                
                # Update recovery attempt
                await self.failed_docs_tracker.update_recovery_attempt(failed_doc.document_path)
                
                if isinstance(result, Exception):
                    self.logger.error(
                        f"Recovery failed for {failed_doc.document_path}: {result}"
                    )
                    failed_recoveries += 1
                elif result.get('success', False):
                    successful_recoveries += 1
                    self.logger.info(
                        f"Successfully recovered document: {failed_doc.document_path}"
                    )
                else:
                    failed_recoveries += 1
                    self.logger.warning(
                        f"Recovery attempt failed for {failed_doc.document_path}: "
                        f"{result.get('error', 'Unknown error')}"
                    )
            
            recovery_time = time.time() - recovery_start_time
            
            # Update metrics
            await self.failed_docs_tracker.update_recovery_metrics(
                successful_recoveries=successful_recoveries,
                failed_recoveries=failed_recoveries,
                recovery_time=recovery_time
            )
            
            recovery_summary = {
                'attempted_documents': len(failed_docs),
                'successful_recoveries': successful_recoveries,
                'failed_recoveries': failed_recoveries,
                'recovery_time': recovery_time,
                'success_rate': successful_recoveries / len(failed_docs) if failed_docs else 0.0
            }
            
            self.logger.info(
                f"Recovery attempt completed: {successful_recoveries}/{len(failed_docs)} successful "
                f"in {recovery_time:.2f}s (success rate: {recovery_summary['success_rate']:.2%})"
            )
            
            return recovery_summary
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return {
                'attempted_documents': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'recovery_time': time.time() - recovery_start_time,
                'error': str(e)
            }
    
    async def _recover_single_document(self, failed_doc: FailedDocumentRecord) -> Dict[str, Any]:
        """Recover a single failed document"""
        try:
            # Create recovery task
# # #             from distributed_processor import ProcessingTask  # Import here to avoid circular import  # Module not found  # Module not found  # Module not found
            
            recovery_task = ProcessingTask(
                task_id=f"recovery-{uuid.uuid4().hex[:8]}",
                document_path=failed_doc.document_path,
                query=failed_doc.query,
                priority=2,  # Higher priority for recovery
                metadata={
                    'is_recovery': True,
                    'original_task_id': failed_doc.task_id,
                    'retry_count': failed_doc.retry_count,
                    'original_error': failed_doc.error_message,
                    **failed_doc.metadata
                }
            )
            
            # Submit task for processing
            await self.distributed_processor._submit_tasks([recovery_task])
            
            # Wait for result with timeout
            result = await self._wait_for_recovery_result(
                recovery_task.task_id, 
                timeout=self.config.get('recovery_timeout', 300)
            )
            
            if result and result.status == "completed":
                return {
                    'success': True,
                    'task_id': recovery_task.task_id,
                    'processing_time': result.processing_time,
                    'quality_score': result.quality_metrics.get('overall_quality', 0.0)
                }
            else:
                return {
                    'success': False,
                    'error': result.error_message if result else 'Timeout or no result received'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _wait_for_recovery_result(self, 
                                       task_id: str, 
                                       timeout: float = 300) -> Optional[Any]:
        """Wait for recovery task result"""
        try:
            # Wait for result with timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check if result is available
                result_key = f"result:{task_id}"
                result_data = await asyncio.to_thread(
                    self.distributed_processor.redis_client.get,
                    result_key
                )
                
                if result_data:
                    result_dict = json.loads(result_data.decode('utf-8'))
# # #                     from distributed_processor import ProcessingResult  # Import here to avoid circular import  # Module not found  # Module not found  # Module not found
                    return ProcessingResult(**result_dict)
                
                # Wait briefly before checking again
                await asyncio.sleep(1.0)
            
            return None  # Timeout
            
        except Exception as e:
            self.logger.error(f"Error waiting for recovery result {task_id}: {e}")
            return None
    
    async def _start_periodic_recovery(self) -> None:
        """Start periodic recovery task"""
        if self.is_recovery_running:
            return
        
        self.is_recovery_running = True
        self.recovery_task = asyncio.create_task(self._periodic_recovery_loop())
        
        self.logger.info(
            f"Started periodic recovery (interval: {self.recovery_interval_minutes} minutes)"
        )
    
    async def _periodic_recovery_loop(self) -> None:
        """Periodic recovery loop"""
        while self.is_recovery_running:
            try:
                # Attempt recovery
                await self.attempt_recovery()
                
                # Wait for next interval
                await asyncio.sleep(self.recovery_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in periodic recovery loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def stop_periodic_recovery(self) -> None:
        """Stop periodic recovery"""
        if self.is_recovery_running and self.recovery_task:
            self.is_recovery_running = False
            self.recovery_task.cancel()
            
            try:
                await self.recovery_task
            except asyncio.CancelledError:
                pass
            
            self.logger.info("Stopped periodic recovery")
    
    async def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery status"""
        try:
            failed_docs = await self.failed_docs_tracker.get_failed_documents()
            eligible_docs = await self.failed_docs_tracker.get_failed_documents(eligible_for_retry=True)
            metrics = await self.failed_docs_tracker.get_recovery_metrics()
            
            return {
                'recovery_running': self.is_recovery_running,
                'total_failed_documents': len(failed_docs),
                'eligible_for_recovery': len(eligible_docs),
                'recovery_metrics': metrics.to_dict(),
                'failed_documents_by_age': self._analyze_failed_documents_by_age(failed_docs),
                'most_common_errors': self._analyze_common_errors(failed_docs)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get recovery status: {e}")
            return {
                'error': str(e)
            }
    
    def _analyze_failed_documents_by_age(self, 
                                        failed_docs: List[FailedDocumentRecord]) -> Dict[str, int]:
        """Analyze failed documents by age"""
        age_buckets = {
            '< 1 hour': 0,
            '1-6 hours': 0,
            '6-24 hours': 0,
            '1-7 days': 0,
            '> 7 days': 0
        }
        
        for doc in failed_docs:
            age_hours = doc.get_failure_age_hours()
            
            if age_hours < 1:
                age_buckets['< 1 hour'] += 1
            elif age_hours < 6:
                age_buckets['1-6 hours'] += 1
            elif age_hours < 24:
                age_buckets['6-24 hours'] += 1
            elif age_hours < 168:  # 7 days
                age_buckets['1-7 days'] += 1
            else:
                age_buckets['> 7 days'] += 1
        
        return age_buckets
    
    def _analyze_common_errors(self, 
                              failed_docs: List[FailedDocumentRecord],
                              top_n: int = 5) -> List[Dict[str, Any]]:
        """Analyze most common error types"""
        error_counts = {}
        
        for doc in failed_docs:
            # Simplify error message for categorization
            error_type = self._categorize_error(doc.error_message)
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Sort by frequency and return top N
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'error_type': error_type, 'count': count}
            for error_type, count in sorted_errors[:top_n]
        ]
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message into general type"""
        error_message = error_message.lower()
        
        if 'import' in error_message or 'module' in error_message:
            return 'Import/Module Error'
        elif 'memory' in error_message or 'out of memory' in error_message:
            return 'Memory Error'
        elif 'timeout' in error_message or 'timed out' in error_message:
            return 'Timeout Error'
        elif 'file not found' in error_message or 'no such file' in error_message:
            return 'File Not Found'
        elif 'permission' in error_message or 'access denied' in error_message:
            return 'Permission Error'
        elif 'connection' in error_message or 'network' in error_message:
            return 'Network Error'
        else:
            return 'Other Error'


# Main recovery function that can be called independently
async def run_document_recovery(redis_url: str = "redis://localhost:6379",
                               config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Standalone function to run document recovery
    
    Args:
        redis_url: Redis connection URL
        config: Configuration parameters
        
    Returns:
        Recovery results summary
    """
    if config is None:
        config = {}
    
    # Initialize Redis client
    redis_client = redis.from_url(redis_url)
    
    # Initialize tracker
    tracker = FailedDocumentsTracker(redis_client, retention_days=config.get('retention_days', 7))
    
    # Create a mock distributed processor for recovery
    class MockDistributedProcessor:
        def __init__(self):
            self.redis_client = redis_client
            
        async def _submit_tasks(self, tasks):
            # In a real implementation, this would submit tasks to the processing queue
            pass
    
    mock_processor = MockDistributedProcessor()
    
    # Initialize recovery manager
    recovery_manager = DocumentRecoveryManager(
        distributed_processor=mock_processor,
        failed_docs_tracker=tracker,
        config=config
    )
    
    # Initialize and run recovery
    await recovery_manager.initialize()
    
    # Attempt recovery
    return await recovery_manager.attempt_recovery()


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'max_retry_count': 3,
            'min_retry_interval_hours': 0.5,
            'recovery_batch_size': 5,
            'enable_periodic_recovery': False
        }
        
        result = await run_document_recovery(config=config)
        print(f"Recovery completed: {result}")
    
    asyncio.run(main())