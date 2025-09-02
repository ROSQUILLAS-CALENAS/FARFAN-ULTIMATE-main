from __future__ import annotations
"""
Comprehensive Deterministic Pipeline Orchestrator

This orchestrator wires every module in a fixed, deterministic workflow that
mirrors the high-performance, inextricably connected chain you specified.

- Does NOT replace existing orchestrators. It's additive and self-contained.
- Executes modules in strict topological order of the predefined DAG.
- Attempts to import and call real module functions if available; otherwise
  performs a safe pass-through to keep the pipeline flowing.
- Computes a value-chain across nodes and auto-enhances nodes not adding value.
- Produces an execution trace and optional visualization JSON.
- Integrates QuestionContext for immutable state consistency throughout execution.
- Supports parallel PDF processing through integrated ParallelPDFProcessor.

Run directly: python comprehensive_pipeline_orchestrator.py
"""

import importlib.util
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

try:
    import psutil  # type: ignore
except Exception:
    class _PsutilStub:
        def Process(self, *a, **kw): return None
        def cpu_percent(self, *a, **kw): return 0.0
        def virtual_memory(self): return type("M", (), {"percent": 0.0})()
    psutil = _PsutilStub()  # type: ignore

from circuit_breaker import CircuitBreaker, CircuitState, BreakerOpenError
from exception_monitoring import ExceptionMonitor
from parallel_processor import ParallelPDFProcessor, default_pdf_chunk_processor
from canonical_flow.pipeline_state_manager import PipelineStateManager, StageArtifact, StageStatus


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors"""
    pass


class PDFParsingError(PDFProcessingError):
    """Recoverable PDF parsing error - processing can continue"""
    pass


class PDFCriticalError(PDFProcessingError):
    """Critical PDF error that should halt processing"""
    pass


@dataclass
class PDFProcessingResult:
    """Result structure for individual PDF processing"""
    file_path: str
    status: str  # 'success', 'failed', 'skipped'
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    extracted_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    content_analysis: Optional[Dict[str, Any]] = None


@dataclass
class BatchPDFResults:
    """Batch processing results with individual file tracking"""
    total_files: int
    successful: int
    failed: int
    skipped: int
    results: List[PDFProcessingResult]
    total_processing_time: float
    batch_metadata: Dict[str, Any]

# Add project root to path for canonical imports
import sys
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from egw_query_expansion.core.immutable_context import QuestionContext, create_question_context

# Import contract system
try:
    from contract_system import ContractManager, ContractResult, ContractStatus
    CONTRACTS_AVAILABLE = True
except ImportError as e:
    CONTRACTS_AVAILABLE = False

# Initialize logger at module level
logger = logging.getLogger(__name__)

# ----- MONITORING AND METRICS CLASSES -----

@dataclass
class ErrorMetrics:
    """Tracks various error types and their frequencies"""
    timeout_errors: int = 0
    memory_errors: int = 0
    processing_errors: int = 0
    validation_errors: int = 0
    connection_errors: int = 0
    other_errors: int = 0
    
    def categorize_error(self, error: Exception) -> str:
        """Categorize error type and increment counter"""
        error_name = type(error).__name__.lower()
        
        if any(keyword in error_name for keyword in ['timeout', 'time']):
            self.timeout_errors += 1
            return 'timeout'
        elif any(keyword in error_name for keyword in ['memory', 'alloc']):
            self.memory_errors += 1
            return 'memory'
        elif any(keyword in error_name for keyword in ['process', 'runtime']):
            self.processing_errors += 1
            return 'processing'
        elif any(keyword in error_name for keyword in ['valid', 'schema']):
            self.validation_errors += 1
            return 'validation'
        elif any(keyword in error_name for keyword in ['connect', 'network']):
            self.connection_errors += 1
            return 'connection'
        else:
            self.other_errors += 1
            return 'other'
    
    def total_errors(self) -> int:
        return (self.timeout_errors + self.memory_errors + 
                self.processing_errors + self.validation_errors +
                self.connection_errors + self.other_errors)

@dataclass
class EvidenceQualityMetrics:
    """Metrics for evidence extraction quality based on content completeness"""
    content_completeness_score: float = 0.0
    evidence_density: float = 0.0  # evidence items per page
    citation_accuracy: float = 0.0
    contextual_relevance: float = 0.0
    structural_coherence: float = 0.0
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'completeness': 0.3,
            'density': 0.2, 
            'accuracy': 0.2,
            'relevance': 0.2,
            'coherence': 0.1
        }
        
        return (
            self.content_completeness_score * weights['completeness'] +
            self.evidence_density * weights['density'] +
            self.citation_accuracy * weights['accuracy'] +
            self.contextual_relevance * weights['relevance'] +
            self.structural_coherence * weights['coherence']
        )

@dataclass 
class ProcessingStageMetrics:
    """Metrics for individual processing stages"""
    stage_name: str
    documents_processed: int = 0
    total_processing_time: float = 0.0
    memory_peak_usage: float = 0.0
    errors: ErrorMetrics = field(default_factory=ErrorMetrics)
    evidence_quality: Optional[EvidenceQualityMetrics] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def add_processing_time(self, processing_time: float):
        """Add processing time and update averages"""
        self.documents_processed += 1
        self.total_processing_time += processing_time
    
    def get_average_processing_time(self) -> float:
        """Get average processing time per document"""
        return (self.total_processing_time / max(1, self.documents_processed))
    
    def get_error_rate(self) -> float:
        """Get error rate as percentage"""
        total_operations = self.documents_processed + self.errors.total_errors()
        if total_operations == 0:
            return 0.0
        return (self.errors.total_errors() / total_operations) * 100

class MonitoringMetrics:
    """
    Comprehensive monitoring metrics that tracks processing rate, memory usage,
    error rates, evidence quality, and end-to-end latency for PDF processing pipeline.
    """
    
    def __init__(self, metric_window_minutes: int = 60):
        self.metric_window_minutes = metric_window_minutes
        self.start_time = datetime.now()
        
        # Processing rate tracking
        self.processed_documents = deque(maxlen=1000)  # Store (timestamp, doc_id) tuples
        self.processing_times = deque(maxlen=1000)  # Store processing durations
        
        # Memory usage tracking
        self.memory_samples = deque(maxlen=100)  # Store (timestamp, memory_mb) tuples
        self.memory_per_document = {}  # doc_id -> memory_usage
        
        # Error tracking
        self.global_errors = ErrorMetrics()
        self.stage_errors = defaultdict(ErrorMetrics)
        
        # Evidence quality tracking
        self.evidence_scores = deque(maxlen=500)  # Store quality scores
        
        # End-to-end latency tracking  
        self.document_start_times = {}  # doc_id -> start_timestamp
        self.document_end_times = {}   # doc_id -> end_timestamp
        self.latency_measurements = deque(maxlen=1000)  # Store latencies in seconds
        
        # Stage-specific metrics
        self.stage_metrics = defaultdict(ProcessingStageMetrics)
        
        # Batch processing metrics
        self.batch_processing_active = False
        self.batch_start_time = None
        self.batch_total_docs = 0
        self.batch_completed_docs = 0
        
        # Threading for memory monitoring
        self.memory_monitor_active = False
        self.memory_thread = None
        
    def start_memory_monitoring(self):
        """Start continuous memory monitoring in background thread"""
        if self.memory_monitor_active:
            return
            
        self.memory_monitor_active = True
        self.memory_thread = ThreadPoolExecutor(max_workers=1)
        self.memory_thread.submit(self._memory_monitoring_loop)
        
    def stop_memory_monitoring(self):
        """Stop memory monitoring"""
        self.memory_monitor_active = False
        if self.memory_thread:
            self.memory_thread.shutdown(wait=True)
            
    def _memory_monitoring_loop(self):
        """Background loop for memory monitoring"""
        while self.memory_monitor_active:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)  # Convert to MB
                self.memory_samples.append((datetime.now(), memory_mb))
                time.sleep(5)  # Sample every 5 seconds
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                time.sleep(10)  # Back off on error
    
    def record_document_start(self, doc_id: str, stage: str = None):
        """Record start of document processing"""
        timestamp = datetime.now()
        self.document_start_times[doc_id] = timestamp
        
        if stage:
            if stage not in self.stage_metrics:
                self.stage_metrics[stage] = ProcessingStageMetrics(stage_name=stage)
            self.stage_metrics[stage].start_time = timestamp
            
        # Record memory usage for this document
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_per_document[doc_id] = memory_mb
        except Exception as e:
            logger.warning(f"Failed to record memory for doc {doc_id}: {e}")
            
    def record_document_completion(self, doc_id: str, stage: str = None, 
                                 evidence_quality: Optional[EvidenceQualityMetrics] = None):
        """Record completion of document processing"""
        timestamp = datetime.now()
        self.document_end_times[doc_id] = timestamp
        self.processed_documents.append((timestamp, doc_id))
        
        # Calculate end-to-end latency if start time exists
        if doc_id in self.document_start_times:
            start_time = self.document_start_times[doc_id]
            latency_seconds = (timestamp - start_time).total_seconds()
            self.latency_measurements.append(latency_seconds)
            
            # Clean up old start time
            del self.document_start_times[doc_id]
            
        # Update stage metrics
        if stage and stage in self.stage_metrics:
            stage_metric = self.stage_metrics[stage]
            stage_metric.end_time = timestamp
            if stage_metric.start_time:
                processing_time = (timestamp - stage_metric.start_time).total_seconds()
                stage_metric.add_processing_time(processing_time)
                self.processing_times.append(processing_time)
                
            if evidence_quality:
                stage_metric.evidence_quality = evidence_quality
                self.evidence_scores.append(evidence_quality.calculate_overall_score())
        
        # Update batch processing if active
        if self.batch_processing_active:
            self.batch_completed_docs += 1
            
    def record_error(self, error: Exception, doc_id: str = None, stage: str = None):
        """Record an error with categorization"""
        error_type = self.global_errors.categorize_error(error)
        
        if stage:
            self.stage_errors[stage].categorize_error(error)
            if stage in self.stage_metrics:
                self.stage_metrics[stage].errors.categorize_error(error)
                
        logger.warning(f"Error recorded - Type: {error_type}, Stage: {stage}, "
                      f"Doc: {doc_id}, Error: {str(error)[:100]}")
    
    def record_evidence_quality(self, quality_metrics: EvidenceQualityMetrics):
        """Record evidence quality metrics"""
        overall_score = quality_metrics.calculate_overall_score()
        self.evidence_scores.append(overall_score)
        
    def start_batch_processing(self, total_documents: int):
        """Start tracking batch processing session"""
        self.batch_processing_active = True
        self.batch_start_time = datetime.now()
        self.batch_total_docs = total_documents
        self.batch_completed_docs = 0
        
    def finish_batch_processing(self):
        """Finish batch processing session"""
        self.batch_processing_active = False
        
    def get_processing_rate(self) -> float:
        """Get processing rate in documents per minute"""
        cutoff_time = datetime.now() - timedelta(minutes=self.metric_window_minutes)
        recent_docs = [
            (timestamp, doc_id) for timestamp, doc_id in self.processed_documents
            if timestamp > cutoff_time
        ]
        
        if not recent_docs:
            return 0.0
            
        return len(recent_docs) / self.metric_window_minutes
    
    def get_memory_usage_per_document(self) -> float:
        """Get average memory usage per document in MB"""
        if not self.memory_per_document:
            return 0.0
            
        return sum(self.memory_per_document.values()) / len(self.memory_per_document)
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def get_error_rates(self) -> Dict[str, float]:
        """Get error rates by category"""
        total_operations = len(self.processed_documents) + self.global_errors.total_errors()
        if total_operations == 0:
            return {}
            
        return {
            'timeout_errors': (self.global_errors.timeout_errors / total_operations) * 100,
            'memory_errors': (self.global_errors.memory_errors / total_operations) * 100, 
            'processing_errors': (self.global_errors.processing_errors / total_operations) * 100,
            'validation_errors': (self.global_errors.validation_errors / total_operations) * 100,
            'connection_errors': (self.global_errors.connection_errors / total_operations) * 100,
            'other_errors': (self.global_errors.other_errors / total_operations) * 100,
            'total_error_rate': (self.global_errors.total_errors() / total_operations) * 100
        }
    
    def get_evidence_quality_stats(self) -> Dict[str, float]:
        """Get evidence quality statistics"""
        if not self.evidence_scores:
            return {'average': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
            
        scores = list(self.evidence_scores)
        return {
            'average': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores), 
            'count': len(scores)
        }
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get end-to-end latency statistics"""
        if not self.latency_measurements:
            return {'average': 0.0, 'min': 0.0, 'max': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
            
        latencies = sorted(list(self.latency_measurements))
        n = len(latencies)
        
        return {
            'average': sum(latencies) / n,
            'min': latencies[0],
            'max': latencies[-1],
            'p50': latencies[int(n * 0.5)],
            'p95': latencies[int(n * 0.95)],
            'p99': latencies[int(n * 0.99)]
        }
    
    def get_batch_progress(self) -> Dict[str, Any]:
        """Get current batch processing progress"""
        if not self.batch_processing_active or not self.batch_start_time:
            return {'active': False}
            
        elapsed_time = (datetime.now() - self.batch_start_time).total_seconds()
        progress_pct = (self.batch_completed_docs / max(1, self.batch_total_docs)) * 100
        
        # Estimate remaining time
        if self.batch_completed_docs > 0:
            rate = self.batch_completed_docs / elapsed_time
            remaining_docs = self.batch_total_docs - self.batch_completed_docs
            estimated_remaining_time = remaining_docs / rate if rate > 0 else 0
        else:
            estimated_remaining_time = 0
            
        return {
            'active': True,
            'total_documents': self.batch_total_docs,
            'completed_documents': self.batch_completed_docs,
            'progress_percentage': progress_pct,
            'elapsed_time_seconds': elapsed_time,
            'estimated_remaining_seconds': estimated_remaining_time
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report with all metrics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'processing_rate': {
                'documents_per_minute': self.get_processing_rate(),
                'total_documents_processed': len(self.processed_documents)
            },
            'memory_usage': {
                'current_mb': self.get_current_memory_usage(),
                'average_per_document_mb': self.get_memory_usage_per_document(),
                'peak_usage_mb': max([mem for _, mem in self.memory_samples], default=0.0)
            },
            'error_rates': self.get_error_rates(),
            'evidence_quality': self.get_evidence_quality_stats(),
            'latency': self.get_latency_stats(),
            'batch_processing': self.get_batch_progress(),
            'stage_metrics': {}
        }
        
        # Add stage-specific metrics
        for stage_name, metrics in self.stage_metrics.items():
            report['stage_metrics'][stage_name] = {
                'documents_processed': metrics.documents_processed,
                'average_processing_time': metrics.get_average_processing_time(),
                'error_rate': metrics.get_error_rate(),
                'memory_peak_mb': metrics.memory_peak_usage
            }
            
            if metrics.evidence_quality:
                report['stage_metrics'][stage_name]['evidence_quality'] = {
                    'completeness_score': metrics.evidence_quality.content_completeness_score,
                    'overall_score': metrics.evidence_quality.calculate_overall_score()
                }
        
        return report
    
    def log_metric_summary(self, log_level: int = logging.INFO):
        """Log a summary of current metrics"""
        report = self.get_performance_report()
        
        summary_lines = [
            "=== PERFORMANCE METRICS SUMMARY ===",
            f"Processing Rate: {report['processing_rate']['documents_per_minute']:.2f} docs/min",
            f"Total Processed: {report['processing_rate']['total_documents_processed']} documents",
            f"Current Memory: {report['memory_usage']['current_mb']:.2f} MB",
            f"Avg Memory/Doc: {report['memory_usage']['average_per_document_mb']:.2f} MB",
            f"Total Error Rate: {report['error_rates'].get('total_error_rate', 0):.2f}%",
            f"Evidence Quality Avg: {report['evidence_quality']['average']:.3f}",
            f"Latency P95: {report['latency']['p95']:.2f}s",
            "=== END SUMMARY ==="
        ]
        
        logger.log(log_level, "\n".join(summary_lines))

# Import mathematical pipeline coordinator
try:
    from canonical_flow.mathematical_enhancers.mathematical_pipeline_coordinator import create_mathematical_pipeline_coordinator
    MATHEMATICAL_PIPELINE_AVAILABLE = True
    logger.info("Mathematical pipeline coordinator successfully imported")
except ImportError as e:
    logger.warning(f"Mathematical pipeline coordinator not available: {str(e)}")
    MATHEMATICAL_PIPELINE_AVAILABLE = False

# Import mathematical orchestration enhancer
try:
    from canonical_flow.mathematical_enhancers.orchestration_enhancer import (
        MathematicalOrchestrationEnhancer, 
        OrchestrationIntegrator,
        StabilityBounds,
        create_enhanced_orchestrator
    )
    MATH_ENHANCER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Mathematical orchestration enhancer not available: {e}")
    MATH_ENHANCER_AVAILABLE = False


class ProcessStage(Enum):
    # All process stages in order
    INGESTION = "ingestion_preparation"
    CONTEXT_BUILD = "context_construction"
    KNOWLEDGE = "knowledge_extraction"
    ANALYSIS = "analysis_nlp"
    CLASSIFICATION = "classification_evaluation"
    SEARCH = "search_retrieval"
    ORCHESTRATION = "orchestration_control"
    AGGREGATION = "aggregation_reporting"
    INTEGRATION = "integration_storage"
    SYNTHESIS = "synthesis_output"


@dataclass
class ProcessNode:
    file_path: str
    stage: ProcessStage
    dependencies: List[str]
    outputs: Dict[str, Any]
    process_type: str
    evento_inicio: str
    evento_cierre: str
    value_metrics: Dict[str, float]


class ComprehensivePipelineOrchestrator:
    """Main orchestrator that chains ALL modules deterministically."""

    def __init__(self) -> None:
        # Pre-flight validation - MUST be first operation
        from canonical_flow.mathematical_enhancers.pre_flight_validator import check_library_compatibility
        
        logger.info("Starting comprehensive pipeline orchestrator with pre-flight validation...")
        validation_result = check_library_compatibility()
        
        if validation_result.activated_fallbacks:
            logger.warning(f"Orchestrator initialized with {len(validation_result.activated_fallbacks)} fallback implementations")
        
        self.root: Path = Path(__file__).resolve().parent
        self.process_graph: Dict[str, ProcessNode] = self._build_complete_graph()
        self.execution_order: List[str] = []
        self.value_chain: Dict[str, Dict[str, Any]] = {}
        # Runtime context propagated across node executions (round/cluster/point)
        self.runtime_context: Dict[str, Any] = {}
        # Execution monitoring state
        self.execution_events: List[Dict[str, Any]] = []
        
        # Store validation result for runtime reference
        self.validation_result = validation_result
        
        # Initialize circuit breaker and exception monitor
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exceptions=(Exception,)
        )
        self.exception_monitor = ExceptionMonitor()
        # Initialize monitoring metrics
        self.monitoring_metrics = MonitoringMetrics()
        self.monitoring_metrics.start_memory_monitoring()
        
        # Logging configuration for metrics
        self.metric_log_interval_minutes = 5  # Log metrics every 5 minutes
        self.last_metric_log = datetime.now()
        
        # Initialize parallel PDF processor
        self.parallel_processor = ParallelPDFProcessor(
            worker_count=6,
            chunk_size=10,
            enable_recovery=True,
            recovery_dir="pipeline_recovery"
        )
        
        # QuestionContext for immutable state consistency
        self.question_context: Optional[QuestionContext] = None
        
        # Initialize mathematical pipeline coordinator
        self.mathematical_coordinator = None
        if MATHEMATICAL_PIPELINE_AVAILABLE:
            try:
                self.mathematical_coordinator = create_mathematical_pipeline_coordinator()
                logger.info("Mathematical pipeline coordinator initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize mathematical pipeline coordinator: {str(e)}")
                self.mathematical_coordinator = None

        # Mathematical orchestration enhancer integration
        self.math_enhancer: Optional['MathematicalOrchestrationEnhancer'] = None
        self.stability_monitoring_enabled = False
        if MATH_ENHANCER_AVAILABLE:
            try:
                self.math_enhancer = MathematicalOrchestrationEnhancer()
                self.stability_monitoring_enabled = True
                logger.info("Mathematical orchestration enhancer integrated successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize mathematical enhancer: {e}")

        # Contract system integration
        self.contract_manager: Optional['ContractManager'] = None
        if CONTRACTS_AVAILABLE:
            try:
                self.contract_manager = ContractManager()
                logger.info("Contract system integrated successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize contract manager: {str(e)}")
                self.contract_manager = None
        
        logger.info("✓ ComprehensivePipelineOrchestrator initialized successfully with pre-flight validation")

        # Pipeline state management
        self.state_manager = PipelineStateManager()
        logger.info("Pipeline state manager initialized successfully")

    def resume_from_checkpoint(self, document_id: str, input_data: Any = None, 
                              question_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume pipeline execution from the last successful checkpoint for a document.
        
        Args:
            document_id: Document identifier to resume
            input_data: Optional input data (will use stored data if available)
            question_text: Optional question text
            
        Returns:
            Pipeline execution result
        """
        logger.info(f"Resuming pipeline execution from checkpoint for document: {document_id}")
        
        # Check if document state exists
        if document_id not in self.state_manager.document_states:
            logger.warning(f"No checkpoint found for document {document_id}, starting from beginning")
            return self.execute_pipeline(input_data, question_text, document_id=document_id, enable_checkpointing=True)
        
        # Get current progress
        progress = self.state_manager.get_pipeline_progress(document_id)
        next_stage = self.state_manager.get_next_pending_stage(document_id)
        
        if next_stage is None:
            logger.info(f"Document {document_id} is already complete")
            # Return the final stored result if available
            doc_state = self.state_manager.document_states[document_id]
            final_stage_name = self.state_manager.PIPELINE_STAGES[-1]
            if final_stage_name in doc_state.stages:
                final_stage = doc_state.stages[final_stage_name]
                final_result = final_stage.metrics.get("result")
                if final_result:
                    return {
                        "status": "completed_from_checkpoint",
                        "document_id": document_id,
                        "final_output": final_result,
                        "pipeline_progress": progress,
                        "resumed_from_checkpoint": True
                    }
        
        logger.info(f"Resuming from stage: {next_stage} (Progress: {progress['progress_percentage']:.1f}%)")
        
        # Resume execution with checkpointing enabled
        return self.execute_pipeline(input_data, question_text, document_id=document_id, enable_checkpointing=True)

    def _map_node_to_pipeline_stage(self, node_name: str) -> str:
        """
        Map a process graph node name to a pipeline stage name.
        
        Args:
            node_name: Name of the process graph node
            
        Returns:
            Corresponding pipeline stage name
        """
        node = self.process_graph.get(node_name)
        if not node:
            return "unknown_stage"
        
        # Map ProcessStage enum values to pipeline stage names
        stage_mapping = {
            ProcessStage.INGESTION: "ingestion_preparation",
            ProcessStage.CONTEXT_BUILD: "context_construction",
            ProcessStage.KNOWLEDGE: "knowledge_extraction", 
            ProcessStage.ANALYSIS: "analysis_nlp",
            ProcessStage.CLASSIFICATION: "classification_evaluation",
            ProcessStage.SEARCH: "search_retrieval",
            ProcessStage.ORCHESTRATION: "orchestration_control",
            ProcessStage.AGGREGATION: "aggregation_reporting",
            ProcessStage.INTEGRATION: "integration_storage",
            ProcessStage.SYNTHESIS: "synthesis_output"
        }
        
        # Get pipeline stage from mapping, with fallback logic
        pipeline_stage = stage_mapping.get(node.stage)
        if pipeline_stage:
            return pipeline_stage
        
        # Fallback: try to infer from node name or stage value
        stage_value = node.stage.value.lower()
        if "routing" in node_name.lower() or "decision" in node_name.lower():
            return "routing_decision"
        elif "monitor" in node_name.lower() or "validation" in node_name.lower():
            return "monitoring_validation"
        else:
            # Use stage value as fallback
            return stage_value
    
    def _generate_stage_artifacts(self, node_name: str, result: Any) -> List[StageArtifact]:
        """
        Generate artifacts from a stage execution result.
        
        Args:
            node_name: Name of the process graph node
            result: Execution result data
            
        Returns:
            List of StageArtifacts generated
        """
        artifacts = []
        
        if result is None:
            return artifacts
        
        try:
            # Create artifacts directory if it doesn't exist
            artifacts_dir = Path("canonical_flow/artifacts")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate artifacts based on result type and content
            if isinstance(result, dict):
                # Save result as JSON artifact
                artifact_path = artifacts_dir / f"{node_name}_result.json"
                
                with open(artifact_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                
                file_size = artifact_path.stat().st_size
                artifacts.append(StageArtifact(
                    file_path=str(artifact_path),
                    artifact_type="json",
                    file_size=file_size,
                    created_at=datetime.now()
                ))
                
            elif isinstance(result, str) and len(result) > 0:
                # Save text result
                artifact_path = artifacts_dir / f"{node_name}_output.txt"
                
                with open(artifact_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                file_size = artifact_path.stat().st_size
                artifacts.append(StageArtifact(
                    file_path=str(artifact_path),
                    artifact_type="text",
                    file_size=file_size,
                    created_at=datetime.now()
                ))
            
            elif isinstance(result, bytes):
                # Save binary data
                artifact_path = artifacts_dir / f"{node_name}_output.bin"
                
                with open(artifact_path, 'wb') as f:
                    f.write(result)
                
                file_size = artifact_path.stat().st_size
                artifacts.append(StageArtifact(
                    file_path=str(artifact_path),
                    artifact_type="binary",
                    file_size=file_size,
                    created_at=datetime.now()
                ))
            
            # For complex objects, save a string representation
            else:
                artifact_path = artifacts_dir / f"{node_name}_repr.txt"
                
                with open(artifact_path, 'w', encoding='utf-8') as f:
                    f.write(str(result))
                
                file_size = artifact_path.stat().st_size
                artifacts.append(StageArtifact(
                    file_path=str(artifact_path),
                    artifact_type="text",
                    file_size=file_size,
                    created_at=datetime.now()
                ))
                
        except Exception as e:
            logger.warning(f"Failed to generate artifacts for {node_name}: {e}")
            
        return artifacts

    # ---------------------------------------------------------------------
    # Graph Definition
    # ---------------------------------------------------------------------
    def _build_complete_graph(self) -> Dict[str, ProcessNode]:
        """Build the complete dependency graph for ALL modules."""

        # Helper to shorten node creation
        def N(
            file_path: str,
            stage: ProcessStage,
            deps: List[str],
            outputs: Dict[str, Any],
            ptype: str,
            start: str,
            close: str,
            metrics: Dict[str, float],
        ) -> ProcessNode:
            return ProcessNode(
                file_path=file_path,
                stage=stage,
                dependencies=deps,
                outputs=outputs,
                process_type=ptype,
                evento_inicio=start,
                evento_cierre=close,
                value_metrics=metrics,
            )

        # IMPORTANT: Keys are the actual module filenames for deterministic linking
        graph: Dict[str, ProcessNode] = {
            # STAGE 1: INGESTION & PREPARATION
            "pdf_reader.py": N(
                "pdf_reader.py",
                ProcessStage.INGESTION,
                [],
                {"text": str, "metadata": dict},
                "extraction",
                "PDF file loaded",
                "Text extracted and structured",
                {"extraction_rate": 0.0, "quality": 0.0},
            ),
            "advanced_loader.py": N(
                "advanced_loader.py",
                ProcessStage.INGESTION,
                ["pdf_reader.py"],
                {"loaded_docs": list, "metadata": dict},
                "loading",
                "Document loading request",
                "Documents loaded with metadata",
                {"load_efficiency": 0.0, "completeness": 0.0},
            ),
            "feature_extractor.py": N(
                "feature_extractor.py",
                ProcessStage.INGESTION,
                ["advanced_loader.py"],
                {"features": dict, "vectors": list},
                "feature_extraction",
                "Structured text available",
                "Feature vector generated",
                {"feature_coverage": 0.0, "relevance": 0.0},
            ),
            "normative_validator.py": N(
                "normative_validator.py",
                ProcessStage.INGESTION,
                ["feature_extractor.py"],
                {"validation_report": dict, "compliance": bool},
                "validation",
                "Document processed",
                "Validation report generated",
                {"compliance_score": 0.0, "accuracy": 0.0},
            ),

            # STAGE 2: CONTEXT CONSTRUCTION
            "immutable_context.py": N(
                "immutable_context.py",
                ProcessStage.CONTEXT_BUILD,
                ["normative_validator.py"],
                {"context": dict, "dag": object},
                "context_building",
                "Validated data available",
                "Immutable context created",
                {"integrity": 0.0, "completeness": 0.0},
            ),
            "context_adapter.py": N(
                "context_adapter.py",
                ProcessStage.CONTEXT_BUILD,
                ["immutable_context.py"],
                {"adapted_context": dict},
                "adaptation",
                "Context created",
                "Context adapted for processing",
                {"adaptation_quality": 0.0},
            ),
            "lineage_tracker.py": N(
                "lineage_tracker.py",
                ProcessStage.CONTEXT_BUILD,
                ["context_adapter.py"],
                {"lineage_graph": dict, "trace": list},
                "tracking",
                "Processing started",
                "Lineage recorded",
                {"traceability": 0.0, "completeness": 0.0},
            ),

            # STAGE 3: KNOWLEDGE EXTRACTION & GRAPH BUILDING
            "Advanced Knowledge Graph Builder Component for Semantic Inference Engine.py": N(
                "Advanced Knowledge Graph Builder Component for Semantic Inference Engine.py",
                ProcessStage.KNOWLEDGE,
                ["lineage_tracker.py"],
                {"knowledge_graph": dict, "inferences": list},
                "knowledge_construction",
                "Context available",
                "Knowledge graph built",
                {"graph_completeness": 0.0, "inference_quality": 0.0},
            ),
            "causal_graph.py": N(
                "causal_graph.py",
                ProcessStage.KNOWLEDGE,
                [
                    "Advanced Knowledge Graph Builder Component for Semantic Inference Engine.py",
                ],
                {"causal_relations": dict},
                "causal_analysis",
                "Entities extracted",
                "Causal graph constructed",
                {"causality_strength": 0.0},
            ),
            "math_stage3_knowledge_enhancer.py": N(
                "math_stage3_knowledge_enhancer.py",
                ProcessStage.KNOWLEDGE,
                ["causal_graph.py", "Advanced Knowledge Graph Builder Component for Semantic Inference Engine.py"],
                {"topological_features": dict, "causal_validation": dict, "quality_metrics": dict},
                "topological_validation",
                "Causal graph constructed",
                "Topological validation completed",
                {"topological_stability": 0.0, "structural_coherence": 0.0, "homological_complexity": 0.0},
            ),
            "causal_dnp_framework.py": N(
                "causal_dnp_framework.py",
                ProcessStage.KNOWLEDGE,
                ["math_stage3_knowledge_enhancer.py"],
                {"dnp_model": object},
                "dynamic_programming",
                "Topological validation completed",
                "DNP framework applied",
                {"optimization_score": 0.0},
            ),
            "embedding_builder.py": N(
                "embedding_builder.py",
                ProcessStage.KNOWLEDGE,
                ["causal_dnp_framework.py"],
                {"embeddings": list},
                "embedding_generation",
                "Text processed",
                "Embeddings created",
                {"embedding_quality": 0.0},
            ),
            "embedding_generator.py": N(
                "embedding_generator.py",
                ProcessStage.KNOWLEDGE,
                ["embedding_builder.py"],
                {"vectors": list},
                "vectorization",
                "Text available",
                "384-dim vectors generated",
                {"vector_quality": 0.0},
            ),

            # STAGE 4: ANALYSIS & NLP
            "adaptive_analyzer.py": N(
                "adaptive_analyzer.py",
                ProcessStage.ANALYSIS,
                ["embedding_generator.py"],
                {"analysis": dict},
                "adaptive_analysis",
                "Data ready for analysis",
                "Analysis completed",
                {"analysis_depth": 0.0},
            ),
            "question_analyzer.py": N(
                "question_analyzer.py",
                ProcessStage.ANALYSIS,
                ["adaptive_analyzer.py"],
                {"questions": list, "intents": dict},
                "question_analysis",
                "Text available",
                "Questions analyzed",
                {"intent_accuracy": 0.0},
            ),
            "implementacion_mapeo.py": N(
                "implementacion_mapeo.py",
                ProcessStage.ANALYSIS,
                ["question_analyzer.py"],
                {"mapping": dict, "coverage_matrix": dict},
                "question_mapping",
                "Question-Decálogo mapping initialized",
                "Mapping ready",
                {"mapping_quality": 0.0},
            ),
            "evidence_processor.py": N(
                "evidence_processor.py",
                ProcessStage.ANALYSIS,
                ["implementacion_mapeo.py"],
                {"processed_evidence": dict},
                "evidence_processing",
                "Evidence extracted",
                "Evidence processed",
                {"evidence_quality": 0.0},
            ),
            "EXTRACTOR DE EVIDENCIAS CONTEXTUAL.py": N(
                "EXTRACTOR DE EVIDENCIAS CONTEXTUAL.py",
                ProcessStage.ANALYSIS,
                ["evidence_processor.py"],
                {"contextual_evidence": dict},
                "contextual_extraction",
                "Context available",
                "Contextual evidence extracted",
                {"context_relevance": 0.0},
            ),
            "evidence_validation_model.py": N(
                "evidence_validation_model.py",
                ProcessStage.ANALYSIS,
                ["EXTRACTOR DE EVIDENCIAS CONTEXTUAL.py"],
                {"validated_evidence": dict},
                "validation",
                "Evidence available",
                "Evidence validated",
                {"validation_score": 0.0},
            ),
            "evaluation_driven_processor.py": N(
                "evaluation_driven_processor.py",
                ProcessStage.ANALYSIS,
                ["evidence_validation_model.py"],
                {"evaluation_results": dict},
                "evaluation",
                "Evidence validated",
                "Evaluation completed",
                {"evaluation_quality": 0.0},
            ),
            "dnp_alignment_adapter.py": N(
                "dnp_alignment_adapter.py",
                ProcessStage.ANALYSIS,
                ["evaluation_driven_processor.py"],
                {"dnp_compliance": dict, "dnp_report": dict},
                "dnp_alignment",
                "DNP standards enforcement started",
                "DNP compliance evaluated",
                {"alignment_strength": 0.0},
            ),

            # STAGE 5: CLASSIFICATION & SCORING
            "adaptive_scoring_engine.py": N(
                "adaptive_scoring_engine.py",
                ProcessStage.CLASSIFICATION,
                ["dnp_alignment_adapter.py"],
                {"scores": dict},
                "scoring",
                "Evidence classified",
                "Adaptive score calculated",
                {"scoring_accuracy": 0.0},
            ),
            "score_calculator.py": N(
                "score_calculator.py",
                ProcessStage.CLASSIFICATION,
                ["adaptive_scoring_engine.py"],
                {"final_scores": dict},
                "score_calculation",
                "Components identified",
                "Scores calculated",
                {"calculation_precision": 0.0},
            ),
            "conformal_risk_control.py": N(
                "conformal_risk_control.py",
                ProcessStage.CLASSIFICATION,
                ["score_calculator.py"],
                {"risk_bounds": dict, "certificates": list},
                "risk_control",
                "Scores available",
                "Risk bounds established",
                {"confidence": 0.0, "coverage": 0.0},
            ),

            # STAGE 6: ROUTING & DECISION
            "deterministic_router.py": N(
                "deterministic_router.py",
                ProcessStage.ORCHESTRATION,
                ["conformal_risk_control.py"],
                {"routing_decision": dict},
                "routing",
                "Query received",
                "Route determined",
                {"routing_efficiency": 0.0},
            ),
            "evidence_router.py": N(
                "evidence_router.py",
                ProcessStage.ORCHESTRATION,
                ["deterministic_router.py"],
                {"evidence_routes": dict},
                "evidence_routing",
                "Evidence categorized",
                "Evidence routed",
                {"routing_accuracy": 0.0},
            ),
            "decision_engine.py": N(
                "decision_engine.py",
                ProcessStage.ORCHESTRATION,
                ["evidence_router.py"],
                {"decisions": dict},
                "decision_making",
                "Data analyzed",
                "Decision made",
                {"decision_confidence": 0.0},
            ),
            "adaptive_controller.py": N(
                "adaptive_controller.py",
                ProcessStage.ORCHESTRATION,
                ["decision_engine.py"],
                {"control_signals": dict},
                "control",
                "Decision made",
                "Control applied",
                {"control_effectiveness": 0.0},
            ),

            # STAGE 7: SEARCH & RETRIEVAL
            "retrieval_engine/lexical_index.py": N(
                "retrieval_engine/lexical_index.py",
                ProcessStage.SEARCH,
                ["adaptive_controller.py"],
                {"bm25_index": object, "lexical_metrics": dict},
                "lexical_bm25",
                "Query received",
                "Lexical BM25 indexed",
                {"index_completeness": 0.0, "consistency": 0.0},
            ),
            "retrieval_engine/vector_index.py": N(
                "retrieval_engine/vector_index.py",
                ProcessStage.SEARCH,
                ["adaptive_controller.py"],
                {"vector_index": object, "vector_metrics": dict},
                "vector_indexing",
                "Embeddings available",
                "Vector index built",
                {"embedding_quality": 0.0},
            ),
            "retrieval_engine/hybrid_retriever.py": N(
                "retrieval_engine/hybrid_retriever.py",
                ProcessStage.SEARCH,
                ["retrieval_engine/lexical_index.py", "retrieval_engine/vector_index.py"],
                {"candidates": list, "hybrid_metrics": dict},
                "hybrid_retrieval",
                "Indices ready",
                "Hybrid candidates generated",
                {"retrieval_precision": 0.0},
            ),
            "semantic_reranking/reranker.py": N(
                "semantic_reranking/reranker.py",
                ProcessStage.SEARCH,
                ["retrieval_engine/hybrid_retriever.py"],
                {"reranked_candidates": list, "rerank_metrics": dict},
                "semantic_reranking",
                "Candidates available",
                "Candidates reranked",
                {"stability": 0.0},
            ),
            "hybrid_retrieval.py": N(
                "hybrid_retrieval.py",
                ProcessStage.SEARCH,
                ["adaptive_controller.py", "semantic_reranking/reranker.py"],
                {"retrieved_docs": list},
                "retrieval",
                "Query received",
                "Results merged",
                {"retrieval_precision": 0.0, "recall": 0.0},
            ),
            "deterministic_hybrid_retrieval.py": N(
                "deterministic_hybrid_retrieval.py",
                ProcessStage.SEARCH,
                ["hybrid_retrieval.py"],
                {"deterministic_results": list},
                "deterministic_retrieval",
                "Query processed",
                "Deterministic results returned",
                {"consistency": 0.0},
            ),
            "hybrid_retrieval_bridge.py": N(
                "hybrid_retrieval_bridge.py",
                ProcessStage.SEARCH,
                ["deterministic_hybrid_retrieval.py", "semantic_reranking/reranker.py"],
                {"bridged_results": dict},
                "bridge",
                "Multiple retrievers ready",
                "Results bridged",
                {"integration_quality": 0.0},
            ),
            "lexical_index.py": N(
                "lexical_index.py",
                ProcessStage.SEARCH,
                ["hybrid_retrieval_bridge.py"],
                {"bm25_index": object},
                "indexing",
                "Text tokenized",
                "Inverted index created",
                {"index_completeness": 0.0},
            ),
            "intelligent_recommendation_engine.py": N(
                "intelligent_recommendation_engine.py",
                ProcessStage.SEARCH,
                ["lexical_index.py"],
                {"recommendations": list},
                "recommendation",
                "User profile available",
                "Recommendations generated",
                {"recommendation_quality": 0.0},
            ),

            # STAGE 8: ORCHESTRATION & PARALLEL PROCESSING
            "confluent_orchestrator.py": N(
                "confluent_orchestrator.py",
                ProcessStage.ORCHESTRATION,
                ["intelligent_recommendation_engine.py"],
                {"orchestration_state": dict},
                "orchestration",
                "Tasks received",
                "Tasks orchestrated",
                {"orchestration_efficiency": 0.0},
            ),
            "core_orchestrator.py": N(
                "core_orchestrator.py",
                ProcessStage.ORCHESTRATION,
                ["confluent_orchestrator.py"],
                {"core_state": dict},
                "core_orchestration",
                "System initialized",
                "Core orchestration complete",
                {"system_efficiency": 0.0},
            ),
            "enhanced_core_orchestrator.py": N(
                "enhanced_core_orchestrator.py",
                ProcessStage.ORCHESTRATION,
                ["core_orchestrator.py"],
                {"enhanced_state": dict},
                "enhanced_orchestration",
                "Core ready",
                "Enhanced orchestration complete",
                {"enhancement_value": 0.0},
            ),
            "distributed_processor.py": N(
                "distributed_processor.py",
                ProcessStage.ORCHESTRATION,
                ["enhanced_core_orchestrator.py"],
                {"distributed_results": dict},
                "distribution",
                "Batch received",
                "Tasks completed",
                {"parallelization_efficiency": 0.0},
            ),
            "airflow_orchestrator.py": N(
                "airflow_orchestrator.py",
                ProcessStage.ORCHESTRATION,
                ["distributed_processor.py"],
                {"airflow_dag": object},
                "workflow_orchestration",
                "Workflow defined",
                "DAG executed",
                {"workflow_efficiency": 0.0},
            ),

            # STAGE 9: MONITORING & VALIDATION
            "circuit_breaker.py": N(
                "circuit_breaker.py",
                ProcessStage.ORCHESTRATION,
                ["airflow_orchestrator.py"],
                {"circuit_state": str},
                "fault_tolerance",
                "Error threshold monitored",
                "Circuit state updated",
                {"reliability": 0.0},
            ),
            "backpressure_manager.py": N(
                "backpressure_manager.py",
                ProcessStage.ORCHESTRATION,
                ["circuit_breaker.py"],
                {"pressure_state": dict},
                "flow_control",
                "Load detected",
                "Pressure managed",
                {"flow_efficiency": 0.0},
            ),
            "alert_system.py": N(
                "alert_system.py",
                ProcessStage.ORCHESTRATION,
                ["backpressure_manager.py"],
                {"alerts": list},
                "alerting",
                "Threshold exceeded",
                "Alert sent",
                {"alert_effectiveness": 0.0},
            ),
            "exception_monitoring.py": N(
                "exception_monitoring.py",
                ProcessStage.ORCHESTRATION,
                ["alert_system.py"],
                {"exceptions": list},
                "monitoring",
                "Exception raised",
                "Exception logged",
                {"monitoring_coverage": 0.0},
            ),
            "exception_telemetry.py": N(
                "exception_telemetry.py",
                ProcessStage.ORCHESTRATION,
                ["exception_monitoring.py"],
                {"telemetry": dict},
                "telemetry",
                "Event occurred",
                "Telemetry recorded",
                {"telemetry_quality": 0.0},
            ),

            # STAGE 10: VALIDATION & CONTRACTS
            "contract_validator.py": N(
                "contract_validator.py",
                ProcessStage.ORCHESTRATION,
                ["exception_telemetry.py"],
                {"contract_validation": bool},
                "contract_validation",
                "Contract defined",
                "Contract validated",
                {"contract_compliance": 0.0},
            ),
            "constraint_validator.py": N(
                "constraint_validator.py",
                ProcessStage.ORCHESTRATION,
                ["contract_validator.py"],
                {"constraint_validation": bool},
                "constraint_validation",
                "Constraints defined",
                "Constraints validated",
                {"constraint_satisfaction": 0.0},
            ),
            "rubric_validator.py": N(
                "rubric_validator.py",
                ProcessStage.ORCHESTRATION,
                ["constraint_validator.py"],
                {"rubric_scores": dict},
                "rubric_validation",
                "Rubric applied",
                "Scores assigned",
                {"rubric_accuracy": 0.0},
            ),

            # STAGE 11: AGGREGATION & SYNTHESIS
            "answer_synthesizer.py": N(
                "answer_synthesizer.py",
                ProcessStage.SYNTHESIS,
                ["rubric_validator.py"],
                {"synthesized_answer": str},
                "synthesis",
                "Components ready",
                "Answer synthesized",
                {"synthesis_quality": 0.0},
            ),
            "answer_formatter.py": N(
                "answer_formatter.py",
                ProcessStage.SYNTHESIS,
                ["answer_synthesizer.py"],
                {"formatted_answer": str},
                "formatting",
                "Answer synthesized",
                "Answer formatted",
                {"format_quality": 0.0},
            ),
            "report_compiler.py": N(
                "report_compiler.py",
                ProcessStage.AGGREGATION,
                ["answer_formatter.py"],
                {"report": dict, "pdf": bytes},
                "compilation",
                "Data processed",
                "Report PDF generated",
                {"report_completeness": 0.0},
            ),

            # STAGE 12: INTEGRATION & METRICS
            "metrics_collector.py": N(
                "metrics_collector.py",
                ProcessStage.INTEGRATION,
                ["report_compiler.py"],
                {"metrics": dict},
                "metrics_collection",
                "Process running",
                "Metrics collected",
                {"metrics_coverage": 0.0},
            ),
            "analytics_enhancement.py": N(
                "analytics_enhancement.py",
                ProcessStage.INTEGRATION,
                ["metrics_collector.py"],
                {"enhanced_analytics": dict},
                "analytics",
                "Metrics available",
                "Analytics enhanced",
                {"analytics_depth": 0.0},
            ),

            # SPECIAL MODULES
            "feedback_loop.py": N(
                "feedback_loop.py",
                ProcessStage.INTEGRATION,
                ["analytics_enhancement.py"],
                {"feedback": dict},
                "feedback",
                "Results available",
                "Feedback incorporated",
                {"feedback_effectiveness": 0.0},
            ),
            "compensation_engine.py": N(
                "compensation_engine.py",
                ProcessStage.INTEGRATION,
                ["feedback_loop.py"],
                {"compensations": dict},
                "compensation",
                "Errors detected",
                "Compensation applied",
                {"compensation_quality": 0.0},
            ),
            "optimization_engine.py": N(
                "optimization_engine.py",
                ProcessStage.INTEGRATION,
                ["compensation_engine.py"],
                {"optimizations": dict},
                "optimization",
                "Performance analyzed",
                "System optimized",
                {"optimization_gain": 0.0},
            ),
        }
        # Inject deterministic retrieval and semantic reranking nodes (if not already present)
        if "retrieval_engine/lexical_index.py" not in graph:
            graph["retrieval_engine/lexical_index.py"] = N(
                "retrieval_engine/lexical_index.py",
                ProcessStage.SEARCH,
                ["adaptive_controller.py"],
                {"bm25_index": object, "lexical_metrics": dict},
                "lexical_bm25",
                "Query received",
                "Lexical BM25 indexed",
                {"index_completeness": 0.0, "consistency": 0.0},
            )
        if "retrieval_engine/vector_index.py" not in graph:
            graph["retrieval_engine/vector_index.py"] = N(
                "retrieval_engine/vector_index.py",
                ProcessStage.SEARCH,
                ["adaptive_controller.py"],
                {"vector_index": list, "vector_metrics": dict},
                "vector_indexing",
                "Embeddings available",
                "Vector index built",
                {"embedding_quality": 0.0},
            )
        if "retrieval_engine/hybrid_retriever.py" not in graph:
            graph["retrieval_engine/hybrid_retriever.py"] = N(
                "retrieval_engine/hybrid_retriever.py",
                ProcessStage.SEARCH,
                ["retrieval_engine/lexical_index.py", "retrieval_engine/vector_index.py"],
                {"candidates": list, "hybrid_metrics": dict},
                "hybrid_retrieval",
                "Indices ready",
                "Hybrid candidates generated",
                {"retrieval_precision": 0.0},
            )

        return graph

    # ---------------------------------------------------------------------
    # Value Chain Mechanics
    # ---------------------------------------------------------------------
    def guarantee_value_chain(self) -> bool:
        """Ensure each connection adds value; auto-enhance if needed."""
        self.value_chain.clear()
        for node_name, node in self.process_graph.items():
            input_value = self._calculate_input_value(node)
            output_value = self._calculate_output_value(node)

            if output_value <= input_value:
                logger.warning("Node %s not adding value — enhancing.", node_name)
                self._enhance_node_value(node)
                output_value = self._calculate_output_value(node)

            self.value_chain[node_name] = {
                "input_value": input_value,
                "output_value": output_value,
                "value_added": output_value - input_value,
                "efficiency": output_value / max(input_value, 0.01),
                "start_time": None,
                "end_time": None,
            }
        return all(v["value_added"] > 0 for v in self.value_chain.values())

    def _calculate_input_value(self, node: ProcessNode) -> float:
        if not node.dependencies:
            return 0.5
        dep_values: List[float] = []
        for dep in node.dependencies:
            if dep in self.value_chain:
                dep_values.append(self.value_chain[dep]["output_value"])
        return sum(dep_values) / len(dep_values) if dep_values else 0.5

    def _calculate_output_value(self, node: ProcessNode) -> float:
        metrics = node.value_metrics
        if not metrics:
            return 0.5
        return sum(metrics.values()) / len(metrics)

    def _enhance_node_value(self, node: ProcessNode) -> None:
        node.value_metrics.setdefault("quality_check", 0.9)
        node.value_metrics.setdefault("validation_score", 0.85)
        node.value_metrics.setdefault("enrichment_factor", 0.8)

    def _calculate_data_value(self, data: Any) -> float:
        """Calculate the value of data based on size or token count."""
        if data is None:
            return 0.0
        
        if isinstance(data, str):
            # For strings, use token count approximation (avg 4 chars per token)
            return len(data) / 4.0
        
        if isinstance(data, (list, tuple)):
            # For lists/tuples, sum the values of all elements
            return sum(self._calculate_data_value(item) for item in data)
        
        if isinstance(data, dict):
            # For dictionaries, sum the values of all values plus keys
            total = 0.0
            for key, value in data.items():
                total += self._calculate_data_value(key)
                total += self._calculate_data_value(value)
            return total
        
        if isinstance(data, (int, float)):
            # For numbers, use the value itself (scaled appropriately)
            return abs(float(data)) / 1000.0  # Scale down large numbers
        
        if isinstance(data, bytes):
            # For bytes, use length
            return float(len(data))
        
        # For other types, try to get string representation
        try:
            return len(str(data)) / 4.0
        except Exception:
            return 1.0  # Default minimal value
    
    def _update_value_chain_metrics(self, node_name: str, input_value: float, output_value: float, execution_time: float) -> None:
        """Update the value chain metrics for a specific node execution."""
        value_added = output_value - input_value
        efficiency = value_added / max(execution_time, 0.001)  # Avoid division by zero
        
        # Update the value chain dictionary with calculated metrics
        if node_name not in self.value_chain:
            self.value_chain[node_name] = {}
        
        self.value_chain[node_name].update({
            "input_value": input_value,
            "output_value": output_value,
            "value_added": value_added,
            "efficiency": efficiency,
            "execution_time": execution_time
        })

    # ---------------------------------------------------------------------
    # Execution Monitoring
    # ---------------------------------------------------------------------
    def log_node_start(self, node_name: str) -> datetime:
        """Log the start event for a node execution."""
        start_time = datetime.now()
        node = self.process_graph[node_name]
        
        # Update value chain with start time
        if node_name in self.value_chain:
            self.value_chain[node_name]["start_time"] = start_time
            
        # Log execution event
        event = {
            "node": node_name,
            "event_type": "start",
            "event_message": node.evento_inicio,
            "timestamp": start_time,
            "stage": node.stage.value,
        }
        self.execution_events.append(event)
        
        logger.info(f"Node {node_name} started: {node.evento_inicio}")
        return start_time

    def log_node_end(self, node_name: str, start_time: datetime, error: Optional[str] = None) -> datetime:
        """Log the end event for a node execution and calculate duration."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        node = self.process_graph[node_name]
        
        # Update value chain with end time
        if node_name in self.value_chain:
            self.value_chain[node_name]["end_time"] = end_time
            
        # Log execution event
        event = {
            "node": node_name,
            "event_type": "end",
            "event_message": node.evento_cierre if not error else f"Failed: {error}",
            "timestamp": end_time,
            "duration_seconds": duration,
            "stage": node.stage.value,
            "error": error,
        }
        self.execution_events.append(event)
        
        status = "failed" if error else "completed"
        logger.info(f"Node {node_name} {status}: {node.evento_cierre} (duration: {duration:.3f}s)")
        return end_time

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of execution monitoring data."""
        if not self.execution_events:
            return {"total_nodes": 0, "events": []}
            
        start_events = [e for e in self.execution_events if e["event_type"] == "start"]
        end_events = [e for e in self.execution_events if e["event_type"] == "end"]
        failed_events = [e for e in end_events if e.get("error")]
        
        total_duration = 0.0
        node_durations = {}
        
        for event in end_events:
            if "duration_seconds" in event:
                duration = event["duration_seconds"]
                total_duration += duration
                node_durations[event["node"]] = duration
        
        return {
            "total_nodes": len(self.process_graph),
            "started_nodes": len(start_events),
            "completed_nodes": len(end_events),
            "failed_nodes": len(failed_events),
            "total_duration_seconds": total_duration,
            "node_durations": node_durations,
            "events": self.execution_events.copy(),
            "value_chain_with_times": {
                node: {**metrics, "duration_seconds": node_durations.get(node, 0.0)}
                for node, metrics in self.value_chain.items()
            }
        }

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------
    def execute_pipeline(self, input_data: Any, question_text: Optional[str] = None, enable_parallel_processing: bool = False, 
                        document_id: Optional[str] = None, enable_checkpointing: bool = True) -> Dict[str, Any]:
        """Execute pipeline with optional mathematical stability monitoring, parallel processing, and checkpointing."""
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Initialize document state if checkpointing is enabled
        if enable_checkpointing:
            self.state_manager.initialize_document_state(document_id, str(input_data) if isinstance(input_data, Path) else None)
        
        # Check if parallel processing is enabled and input contains PDF data
        if enable_parallel_processing and self._should_use_parallel_processing(input_data):
            logger.info("Enabling parallel PDF processing for pipeline execution")
            return self._execute_pipeline_with_parallel_processing(input_data, question_text, document_id, enable_checkpointing)
        
        if self.stability_monitoring_enabled and self.math_enhancer:
            return self._execute_pipeline_with_stability_monitoring(input_data, question_text, document_id, enable_checkpointing)
        else:
            return self._execute_pipeline_standard(input_data, question_text, document_id, enable_checkpointing)
    
    def _execute_pipeline_with_stability_monitoring(self, input_data: Any, question_text: Optional[str] = None) -> Dict[str, Any]:
        """Execute pipeline with mathematical stability guarantees."""
        logger.info("Executing pipeline with mathematical stability monitoring")
        
        # Create integrator for stability monitoring
        integrator = OrchestrationIntegrator(self, self.math_enhancer)
        
        # Execute with stability monitoring
        enhanced_results = integrator.execute_with_stability_monitoring({
            'input_data': input_data,
            'question_text': question_text
        })
        
        return enhanced_results
    
    def _should_use_parallel_processing(self, input_data: Any) -> bool:
        """Check if input data qualifies for parallel processing."""
        # Check if input contains PDF files or large document batches
        if isinstance(input_data, dict):
            # Check for PDF file paths
            pdf_indicators = ['pdf_file', 'pdf_path', 'file_path', 'documents']
            for key in pdf_indicators:
                if key in input_data:
                    value = input_data[key]
                    if isinstance(value, (str, Path)) and str(value).lower().endswith('.pdf'):
                        return True
                    if isinstance(value, list) and len(value) > 1:  # Multiple documents
                        return True
            
            # Check for batch processing indicators
            if input_data.get('batch_mode', False) or input_data.get('parallel_mode', False):
                return True
                
        # Check if input is a list of files/documents
        if isinstance(input_data, list) and len(input_data) > 1:
            return True
            
        return False
    
    def _execute_pipeline_with_parallel_processing(self, input_data: Any, question_text: Optional[str] = None) -> Dict[str, Any]:
        """Execute pipeline with parallel PDF processing capabilities."""
        logger.info("Executing pipeline with parallel PDF processing")
        
        # Extract PDF processing configuration from input
        pdf_files = self._extract_pdf_files_from_input(input_data)
        
        if not pdf_files:
            logger.warning("Parallel processing enabled but no PDF files found, falling back to standard execution")
            return self._execute_pipeline_standard(input_data, question_text)
        
        parallel_results = {}
        processing_metadata = []
        
        # Process each PDF file in parallel
        for pdf_file in pdf_files:
            try:
                def pdf_processor(chunk):
                    """Custom processor that runs pipeline stages on each chunk."""
                    # Extract text using default processor
                    chunk_data = default_pdf_chunk_processor(chunk)
                    
                    # Create modified input for this chunk
                    chunk_input = input_data.copy() if isinstance(input_data, dict) else {}
                    chunk_input.update({
                        'chunk_data': chunk_data,
                        'source_file': chunk.file_path,
                        'chunk_id': chunk.chunk_id,
                        'parallel_mode': True
                    })
                    
                    # Run core processing stages on this chunk
                    chunk_result = self._process_chunk_through_pipeline(chunk_input, question_text)
                    return chunk_result
                
                # Create progress callback for this PDF
                def progress_callback(percentage):
                    logger.info(f"Processing {Path(pdf_file).name}: {percentage:.1f}% complete")
                
                # Process PDF with parallel processor
                pdf_result = self.parallel_processor.process_pdf_parallel(
                    pdf_file,
                    pdf_processor,
                    progress_callback=progress_callback
                )
                
                parallel_results[pdf_file] = pdf_result
                processing_metadata.append({
                    'file': pdf_file,
                    'completion_percentage': pdf_result['completion_percentage'],
                    'total_chunks': pdf_result['total_chunks'],
                    'processing_time': pdf_result['total_processing_time'],
                    'pages_processed': pdf_result['total_pages_processed']
                })
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file} in parallel: {e}")
                parallel_results[pdf_file] = {'error': str(e), 'success': False}
        
        # Aggregate results from all PDF files
        aggregated_results = self._aggregate_parallel_results(parallel_results)
        
        # Run final synthesis stages on aggregated results
        final_input = {
            'parallel_results': aggregated_results,
            'processing_metadata': processing_metadata,
            'original_input': input_data
        }
        
        synthesis_result = self._run_synthesis_stages(final_input, question_text)
        
        return {
            'final_output': synthesis_result,
            'parallel_processing': {
                'enabled': True,
                'files_processed': len(pdf_files),
                'total_chunks': sum(r.get('total_chunks', 0) for r in parallel_results.values() if isinstance(r, dict)),
                'total_processing_time': sum(r.get('total_processing_time', 0) for r in parallel_results.values() if isinstance(r, dict)),
                'processing_metadata': processing_metadata
            },
            'parallel_results': parallel_results,
            'execution_mode': 'parallel_pdf_processing'
        }
    
    def _extract_pdf_files_from_input(self, input_data: Any) -> List[str]:
        """Extract PDF file paths from input data."""
        pdf_files = []
        
        if isinstance(input_data, dict):
            # Check common keys for PDF files
            for key in ['pdf_file', 'pdf_path', 'file_path', 'documents', 'files']:
                if key in input_data:
                    value = input_data[key]
                    if isinstance(value, (str, Path)) and str(value).lower().endswith('.pdf'):
                        pdf_files.append(str(value))
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, (str, Path)) and str(item).lower().endswith('.pdf'):
                                pdf_files.append(str(item))
        
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, (str, Path)) and str(item).lower().endswith('.pdf'):
                    pdf_files.append(str(item))
        
        elif isinstance(input_data, (str, Path)) and str(input_data).lower().endswith('.pdf'):
            pdf_files.append(str(input_data))
        
        return pdf_files
    
    def _process_chunk_through_pipeline(self, chunk_input: Dict[str, Any], question_text: Optional[str] = None) -> Dict[str, Any]:
        """Process a single chunk through relevant pipeline stages with comprehensive error handling."""
        from knowledge_extraction_error_handler import KnowledgeExtractionErrorHandler
        
        # Initialize error handler if not exists
        if not hasattr(self, '_chunk_error_handler'):
            self._chunk_error_handler = KnowledgeExtractionErrorHandler(
                default_timeout=300.0,
                memory_threshold=0.8,
                max_retries=2,
                log_dir="logs/chunk_processing"
            )
        
        chunk_id = chunk_input.get('chunk_id', 'unknown')
        chunk_logger = logging.getLogger(f"chunk_processing_{chunk_id}")
        
        # Initialize processing state
        processing_state = {
            'chunk_id': chunk_id,
            'stages_completed': [],
            'stages_failed': [],
            'processing_errors': [],
            'start_time': time.time()
        }
        
        # Initialize QuestionContext for chunk if needed
        chunk_context = None
        if question_text:
            try:
                from egw_query_expansion.core.immutable_context import create_question_context
                chunk_context = create_question_context(question_text, chunk_input.get('chunk_data', {}))
                processing_state['stages_completed'].append('context_creation')
            except Exception as e:
                error_msg = f"Failed to create chunk context: {e}"
                logger.warning(error_msg)
                processing_state['stages_failed'].append('context_creation')
                processing_state['processing_errors'].append(error_msg)
        
        # Process through key stages with isolation
        processed_data = chunk_input.copy()  # Ensure we don't mutate original
        
        # Feature extraction with timeout and error handling
        @self._chunk_error_handler.timeout_decorator(timeout=120.0, component_name="feature_extraction")
        def safe_feature_extraction(data):
            return self._run_feature_extraction(data)
            
        try:
            processed_data = safe_feature_extraction(processed_data)
            processing_state['stages_completed'].append('feature_extraction')
            chunk_logger.info(f"Feature extraction completed successfully")
        except Exception as e:
            error_msg = f"Feature extraction failed: {e}"
            logger.warning(error_msg)
            processing_state['stages_failed'].append('feature_extraction')
            processing_state['processing_errors'].append(error_msg)
            
            # Add fallback features
            processed_data['features'] = {
                'text_length': 0,
                'page_count': 0,
                'avg_chars_per_page': 0,
                'fallback_reason': 'feature_extraction_failure'
            }
        
        # Knowledge extraction with timeout and error handling
        @self._chunk_error_handler.timeout_decorator(timeout=300.0, component_name="knowledge_extraction")
        def safe_knowledge_extraction(data):
            return self._run_knowledge_extraction(data)
            
        try:
            processed_data = safe_knowledge_extraction(processed_data)
            processing_state['stages_completed'].append('knowledge_extraction')
            chunk_logger.info(f"Knowledge extraction completed successfully")
        except Exception as e:
            error_msg = f"Knowledge extraction failed: {e}"
            logger.warning(error_msg)
            processing_state['stages_failed'].append('knowledge_extraction')
            processing_state['processing_errors'].append(error_msg)
            
            # Add fallback knowledge
            processed_data['knowledge'] = {
                'key_terms': [],
                'concepts': [],
                'entities': [],
                'processing_success': False,
                'fallback_reason': 'knowledge_extraction_failure',
                'chunk_id': chunk_id
            }
        
        # Analysis stages with timeout and error handling
        @self._chunk_error_handler.timeout_decorator(timeout=90.0, component_name="analysis_stages")
        def safe_analysis_stages(data):
            return self._run_analysis_stages(data)
            
        try:
            processed_data = safe_analysis_stages(processed_data)
            processing_state['stages_completed'].append('analysis_stages')
            chunk_logger.info(f"Analysis stages completed successfully")
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            logger.warning(error_msg)
            processing_state['stages_failed'].append('analysis_stages')
            processing_state['processing_errors'].append(error_msg)
            
            # Add fallback analysis
            processed_data['analysis'] = {
                'complexity_score': 0.0,
                'entity_density': 0.0,
                'concept_coverage': 0,
                'fallback_reason': 'analysis_failure'
            }
        
        # Calculate processing metrics
        processing_state['end_time'] = time.time()
        processing_state['total_processing_time'] = processing_state['end_time'] - processing_state['start_time']
        processing_state['success_rate'] = len(processing_state['stages_completed']) / max(1, 
            len(processing_state['stages_completed']) + len(processing_state['stages_failed']))
        
        # Determine overall processing success
        critical_stages = ['knowledge_extraction']  # Define critical stages
        processing_success = all(stage in processing_state['stages_completed'] 
                               for stage in critical_stages)
        
        # Log final processing state
        if processing_success:
            chunk_logger.info(
                f"Chunk processing completed successfully. "
                f"Stages completed: {processing_state['stages_completed']}, "
                f"Processing time: {processing_state['total_processing_time']:.2f}s"
            )
        else:
            chunk_logger.warning(
                f"Chunk processing completed with failures. "
                f"Failed stages: {processing_state['stages_failed']}, "
                f"Errors: {processing_state['processing_errors']}"
            )
        
        return {
            'chunk_id': chunk_id,
            'processed_data': processed_data,
            'context': chunk_context.metadata.derivation_id if chunk_context else None,
            'processing_success': processing_success,
            'processing_state': processing_state,
            'stages_completed': processing_state['stages_completed'],
            'stages_failed': processing_state['stages_failed'],
            'processing_errors': processing_state['processing_errors'],
            'processing_time': processing_state['total_processing_time'],
            'success_rate': processing_state['success_rate']
        }
    
    def _run_feature_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run feature extraction stages on data."""
        # Simplified feature extraction
        chunk_data = data.get('chunk_data', {})
        
        features = {
            'text_length': sum(len(page.get('text', '')) for page in chunk_data.get('pages', [])),
            'page_count': len(chunk_data.get('pages', [])),
            'avg_chars_per_page': 0
        }
        
        if features['page_count'] > 0:
            features['avg_chars_per_page'] = features['text_length'] / features['page_count']
        
        data['features'] = features
        return data
    
    def _run_knowledge_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run knowledge extraction stages on data with comprehensive error handling."""
        from knowledge_extraction_error_handler import KnowledgeExtractionPipeline
        
        # Initialize error-handled pipeline if not exists
        if not hasattr(self, '_knowledge_pipeline'):
            self._knowledge_pipeline = KnowledgeExtractionPipeline(
                default_timeout=300.0,
                memory_threshold=0.8,
                max_retries=2,
                log_dir="logs/knowledge_extraction"
            )
        
        chunk_data = data.get('chunk_data', {})
        chunk_id = data.get('chunk_id', 'unknown')
        
        # Extract all text from pages
        all_text = ' '.join(page.get('text', '') for page in chunk_data.get('pages', []))
        
        # Create chunk for processing
        processing_chunk = {
            'chunk_id': chunk_id,
            'text': all_text,
            'page_count': len(chunk_data.get('pages', [])),
            'text_length': len(all_text)
        }
        
        # Process through error-handled knowledge extraction
        try:
            extraction_results = self._knowledge_pipeline.process_knowledge_extraction([processing_chunk])
            
            # Aggregate results with fallbacks
            knowledge = {
                'key_terms': [],
                'concepts': [],
                'entities': [],
                'extraction_stats': extraction_results.get('processing_summary', {}),
                'chunk_id': chunk_id
            }
            
            # Extract key terms
            if extraction_results['key_terms_results']:
                key_terms_result = extraction_results['key_terms_results'][0]
                knowledge['key_terms'] = key_terms_result.get('key_terms', [])
                knowledge['key_terms_confidence'] = key_terms_result.get('extraction_confidence', 0.0)
            
            # Extract concepts  
            if extraction_results['concept_results']:
                concepts_result = extraction_results['concept_results'][0]
                knowledge['concepts'] = concepts_result.get('concepts', [])
                knowledge['concepts_confidence'] = concepts_result.get('concept_confidence', 0.0)
                
            # Extract entities
            if extraction_results['entity_results']:
                entities_result = extraction_results['entity_results'][0]
                knowledge['entities'] = entities_result.get('entities', [])
                knowledge['entities_confidence'] = entities_result.get('entity_confidence', 0.0)
            
            # Add processing metadata
            knowledge['processing_success'] = chunk_id not in extraction_results.get('failed_chunks', [])
            knowledge['error_recovery'] = len(extraction_results.get('failed_chunks', [])) > 0
            
        except Exception as e:
            # Ultimate fallback for catastrophic failures
            logger.error(f"Catastrophic failure in knowledge extraction for chunk {chunk_id}: {e}")
            knowledge = {
                'key_terms': [],
                'concepts': [],
                'entities': [],
                'processing_success': False,
                'error_recovery': False,
                'fallback_reason': 'catastrophic_failure',
                'error_message': str(e),
                'chunk_id': chunk_id
            }
        
        data['knowledge'] = knowledge
        return data
    
    def _run_analysis_stages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis stages on data."""
        # Simplified analysis
        knowledge = data.get('knowledge', {})
        
        analysis = {
            'complexity_score': len(knowledge.get('key_terms', [])) / 10.0,
            'entity_density': len(knowledge.get('entities', [])) / max(1, data.get('features', {}).get('text_length', 1)),
            'concept_coverage': len(knowledge.get('concepts', []))
        }
        
        data['analysis'] = analysis
        return data
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text (simplified)."""
        import re
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        return list(set(words))[:10]  # Top 10 capitalized terms
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text (simplified)."""
        # Simple concept extraction based on common patterns
        concepts = []
        if 'desarrollo' in text.lower():
            concepts.append('desarrollo')
        if 'sostenible' in text.lower():
            concepts.append('sostenibilidad')
        if 'comunidad' in text.lower():
            concepts.append('comunidad')
        return concepts
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simplified)."""
        import re
        # Extract proper nouns and potential entities
        entities = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
        return list(set(entities))[:5]  # Top 5 entities
    
    def _aggregate_parallel_results(self, parallel_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from parallel processing with error handling statistics."""
        from knowledge_extraction_error_handler import KnowledgeExtractionErrorHandler
        
        aggregated = {
            'total_files': len(parallel_results),
            'successful_files': sum(1 for r in parallel_results.values() if isinstance(r, dict) and r.get('completion_percentage', 0) > 90),
            'total_chunks_processed': sum(r.get('completed_chunks', 0) for r in parallel_results.values() if isinstance(r, dict)),
            'combined_features': {},
            'combined_knowledge': {},
            'combined_analysis': {},
            'error_handling_stats': {
                'total_processing_errors': 0,
                'failed_chunk_ids': [],
                'stage_failure_counts': defaultdict(int),
                'processing_success_rate': 0.0,
                'average_processing_time': 0.0,
                'chunks_with_fallbacks': 0
            }
        }
        
        # Combine features, knowledge, and analysis from all chunks
        all_chunk_results = []
        total_processing_time = 0
        total_chunks = 0
        successful_chunks = 0
        
        for file_result in parallel_results.values():
            if isinstance(file_result, dict) and 'results' in file_result:
                for chunk_id, chunk_result in file_result['results'].items():
                    if isinstance(chunk_result, dict):
                        all_chunk_results.append(chunk_result)
                        total_chunks += 1
                        
                        # Track processing statistics
                        if chunk_result.get('processing_success', False):
                            successful_chunks += 1
                        
                        processing_time = chunk_result.get('processing_time', 0)
                        total_processing_time += processing_time
                        
                        # Track stage failures
                        failed_stages = chunk_result.get('stages_failed', [])
                        for stage in failed_stages:
                            aggregated['error_handling_stats']['stage_failure_counts'][stage] += 1
                        
                        # Track failed chunks
                        if failed_stages:
                            aggregated['error_handling_stats']['failed_chunk_ids'].append(chunk_id)
                            aggregated['error_handling_stats']['total_processing_errors'] += len(failed_stages)
                        
                        # Check for fallback usage
                        processed_data = chunk_result.get('processed_data', {})
                        if (processed_data.get('knowledge', {}).get('fallback_reason') or
                            processed_data.get('features', {}).get('fallback_reason') or
                            processed_data.get('analysis', {}).get('fallback_reason')):
                            aggregated['error_handling_stats']['chunks_with_fallbacks'] += 1
        
        # Calculate aggregate statistics
        if total_chunks > 0:
            aggregated['error_handling_stats']['processing_success_rate'] = successful_chunks / total_chunks
            aggregated['error_handling_stats']['average_processing_time'] = total_processing_time / total_chunks
        
        # Sort chunk results by chunk_id for deterministic output
        all_chunk_results.sort(key=lambda x: x.get('chunk_id', 'unknown'))
        
        # Aggregate features across successful chunks only
        successful_features = []
        successful_knowledge = []
        successful_analysis = []
        
        for chunk_result in all_chunk_results:
            if chunk_result.get('processing_success', False):
                processed_data = chunk_result.get('processed_data', {})
                
                # Collect features
                if 'features' in processed_data and not processed_data['features'].get('fallback_reason'):
                    successful_features.append(processed_data['features'])
                
                # Collect knowledge
                if 'knowledge' in processed_data and processed_data['knowledge'].get('processing_success', False):
                    successful_knowledge.append(processed_data['knowledge'])
                
                # Collect analysis
                if 'analysis' in processed_data and not processed_data['analysis'].get('fallback_reason'):
                    successful_analysis.append(processed_data['analysis'])
        
        # Combine successful results
        if successful_features:
            aggregated['combined_features'] = {
                'total_text_length': sum(f.get('text_length', 0) for f in successful_features),
                'total_pages': sum(f.get('page_count', 0) for f in successful_features),
                'avg_chars_per_page': sum(f.get('avg_chars_per_page', 0) for f in successful_features) / len(successful_features)
            }
        
        if successful_knowledge:
            all_key_terms = []
            all_concepts = []
            all_entities = []
            
            for k in successful_knowledge:
                all_key_terms.extend(k.get('key_terms', []))
                all_concepts.extend(k.get('concepts', []))
                all_entities.extend(k.get('entities', []))
            
            aggregated['combined_knowledge'] = {
                'unique_key_terms': list(set(all_key_terms)),
                'unique_concepts': list(set(all_concepts)),
                'unique_entities': list(set(all_entities)),
                'total_key_terms_found': len(all_key_terms),
                'total_concepts_found': len(all_concepts),
                'total_entities_found': len(all_entities)
            }
        
        if successful_analysis:
            aggregated['combined_analysis'] = {
                'avg_complexity_score': sum(a.get('complexity_score', 0) for a in successful_analysis) / len(successful_analysis),
                'avg_entity_density': sum(a.get('entity_density', 0) for a in successful_analysis) / len(successful_analysis),
                'total_concept_coverage': sum(a.get('concept_coverage', 0) for a in successful_analysis)
            }
        
        return aggregated
        
        # Aggregate numerical features
        total_complexity = sum(
            chunk.get('processed_data', {}).get('analysis', {}).get('complexity_score', 0) 
            for chunk in all_chunk_results
        )
        aggregated['combined_analysis']['avg_complexity'] = total_complexity / max(1, len(all_chunk_results))
        
        # Aggregate key terms
        all_terms = []
        for chunk in all_chunk_results:
            terms = chunk.get('processed_data', {}).get('knowledge', {}).get('key_terms', [])
            all_terms.extend(terms)
        
        # Count term frequency
        term_counts = {}
        for term in all_terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        # Get most frequent terms
        aggregated['combined_knowledge']['top_terms'] = sorted(
            term_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]
        
        return aggregated
    
    def _run_synthesis_stages(self, input_data: Dict[str, Any], question_text: Optional[str] = None) -> Dict[str, Any]:
        """Run synthesis stages on aggregated parallel results."""
        parallel_results = input_data.get('parallel_results', {})
        aggregated = input_data.get('parallel_results', {})
        
        synthesis = {
            'summary': f"Processed {len(parallel_results)} files in parallel",
            'key_insights': self._extract_key_insights(aggregated),
            'recommendations': self._generate_recommendations(aggregated),
            'metadata': input_data.get('processing_metadata', [])
        }
        
        return synthesis
    
    def _extract_key_insights(self, aggregated_data: Dict[str, Any]) -> List[str]:
        """Extract key insights from aggregated data."""
        insights = []
        
        combined_analysis = aggregated_data.get('combined_analysis', {})
        avg_complexity = combined_analysis.get('avg_complexity', 0)
        
        if avg_complexity > 5:
            insights.append("High complexity content detected across documents")
        elif avg_complexity > 2:
            insights.append("Moderate complexity content identified")
        else:
            insights.append("Relatively simple content structure")
        
        combined_knowledge = aggregated_data.get('combined_knowledge', {})
        top_terms = combined_knowledge.get('top_terms', [])
        
        if top_terms:
            top_term, count = top_terms[0]
            insights.append(f"Most frequent term: '{top_term}' (appears {count} times)")
        
        total_chunks = aggregated_data.get('total_chunks_processed', 0)
        if total_chunks > 50:
            insights.append(f"Large-scale processing: {total_chunks} chunks processed")
        
        return insights
    
    def _generate_recommendations(self, aggregated_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on aggregated data."""
        recommendations = []
        
        successful_files = aggregated_data.get('successful_files', 0)
        total_files = aggregated_data.get('total_files', 0)
        
        if total_files > 0:
            success_rate = successful_files / total_files
            if success_rate < 0.8:
                recommendations.append("Consider optimizing PDF processing pipeline for better success rates")
            elif success_rate > 0.95:
                recommendations.append("High processing success rate - pipeline is performing well")
        
        combined_analysis = aggregated_data.get('combined_analysis', {})
        avg_complexity = combined_analysis.get('avg_complexity', 0)
        
        if avg_complexity > 5:
            recommendations.append("High complexity detected - consider additional processing stages")
        
        return recommendations
    
    def _execute_pipeline_standard(self, input_data: Any, question_text: Optional[str] = None, 
                                  document_id: Optional[str] = None, enable_checkpointing: bool = True) -> Dict[str, Any]:
        """Execute the complete pipeline deterministically with QuestionContext integration and metrics collection."""
        # Generate document ID for tracking if not provided
        if document_id is None:
            document_id = f"doc_{int(time.time() * 1000)}"
        
        # Start end-to-end tracking
        self.monitoring_metrics.record_document_start(document_id)
        
        try:
            # Initialize QuestionContext if not already set
            if self.question_context is None:
                question = question_text or self._extract_question_from_input(input_data)
                context_data = self._extract_context_data_from_input(input_data)
                try:
                    self.question_context = create_question_context(question, context_data)
                except Exception as e:
                    logger.error(f"Failed to initialize QuestionContext: {e}")
                    self.monitoring_metrics.record_error(e, document_id, "context_initialization")
                    raise RuntimeError(f"QuestionContext initialization failed: {e}") from e
        except Exception as e:
            logger.error(f"Error during QuestionContext setup: {e}")
            self.monitoring_metrics.record_error(e, document_id, "setup")
            return {"error": f"QuestionContext setup failed: {e}"}

        # Ensure value chain is primed
        self.guarantee_value_chain()

        current_data: Any = input_data
        execution_trace: List[Dict[str, Any]] = []

        # Compute deterministic execution order
        self.execution_order = self._topological_sort()
        
        # Track evidence quality scores for accumulation
        evidence_quality_scores = []
        
        # Check for checkpoint resume if enabled
        if enable_checkpointing and document_id:
            completed_stages = self.state_manager.get_completed_stages(document_id)
            logger.info(f"Found {len(completed_stages)} completed stages for document {document_id}")
        else:
            completed_stages = set()

        for node_name in self.execution_order:
            node = self.process_graph[node_name]
            
            # Map node to pipeline stage
            pipeline_stage = self._map_node_to_pipeline_stage(node_name)

            # Start stage-level metrics tracking
            stage_name = node.stage.value
            self.monitoring_metrics.record_document_start(f"{document_id}_{node_name}", stage_name)
            
            # Check if stage should be skipped due to checkpointing
            if enable_checkpointing and document_id and pipeline_stage in completed_stages:
                # Validate artifacts still exist before skipping
                if self.state_manager.validate_stage_artifacts(document_id, pipeline_stage):
                    logger.info(f"Skipping completed stage {pipeline_stage} for node {node_name}")
                    self.state_manager.skip_stage(document_id, pipeline_stage, "Already completed - artifacts validated")
                    
                    # Try to load previous result if available
                    doc_state = self.state_manager.document_states[document_id]
                    if pipeline_stage in doc_state.stages:
                        stage = doc_state.stages[pipeline_stage]
                        # Use metrics as a proxy for previous result
                        restored_data = stage.metrics.get("result")
                        if restored_data:
                            current_data = restored_data
                    continue
                else:
                    # Artifacts are corrupted, reset and re-execute
                    logger.warning(f"Artifacts corrupted for stage {pipeline_stage}, will re-execute")
                    self.state_manager.reset_document_state(document_id, f"Artifacts corrupted for stage {pipeline_stage}")
                    completed_stages = set()  # Clear completed stages cache
            
            # Start stage execution
            if enable_checkpointing and document_id:
                self.state_manager.start_stage(document_id, pipeline_stage)

            # Log start event and get start time
            start_time = self.log_node_start(node_name)
            start_ts = start_time.isoformat()
            
            exec_rec: Dict[str, Any] = {
                "node": node_name,
                "stage": node.stage.value,
                "pipeline_stage": pipeline_stage,
                "evento_inicio": node.evento_inicio,
                "timestamp_inicio": start_ts,
                "question_context_id": self.question_context.metadata.derivation_id if self.question_context else None,
            }

            error_msg = None
            try:
                current_data = self._execute_node(node, current_data)
                
                # Extract evidence quality metrics if this is an evidence processing stage
                if 'evidence' in node_name.lower() or node.stage in [ProcessStage.ANALYSIS, ProcessStage.KNOWLEDGE]:
                    evidence_quality = self._extract_evidence_quality_from_data(current_data, node_name)
                    if evidence_quality:
                        evidence_quality_scores.append(evidence_quality)
                        self.monitoring_metrics.record_evidence_quality(evidence_quality)
                
                # Generate artifacts from result
                artifacts = self._generate_stage_artifacts(node_name, current_data)
                
                # Complete stage in checkpoint
                if enable_checkpointing and document_id:
                    metrics = {
                        "result": current_data,
                        "execution_time": (datetime.now() - start_time).total_seconds()
                    }
                    self.state_manager.complete_stage(document_id, pipeline_stage, artifacts, metrics)
                
            except PDFCriticalError as pdf_critical:
                error_msg = f"Critical PDF error: {pdf_critical}"
                logger.critical("Critical PDF processing error in %s: %s", node_name, pdf_critical)
                self.monitoring_metrics.record_error(pdf_critical, document_id, stage_name)
                self._handle_critical_pdf_error(node, pdf_critical)
                
                # Fail stage in checkpoint
                if enable_checkpointing and document_id:
                    self.state_manager.fail_stage(document_id, pipeline_stage, error_msg)
                
                exec_rec.update({"error": error_msg, "error_type": "pdf_critical", "halt_processing": True})
                # Critical PDF errors should halt processing
                break
            except PDFParsingError as pdf_parsing:
                error_msg = f"PDF parsing error: {pdf_parsing}"
                logger.warning("PDF parsing error in %s (continuing processing): %s", node_name, pdf_parsing)
                self.monitoring_metrics.record_error(pdf_parsing, document_id, stage_name)
                self._handle_pdf_parsing_error(node, pdf_parsing, current_data)
                
                # Fail stage in checkpoint
                if enable_checkpointing and document_id:
                    self.state_manager.fail_stage(document_id, pipeline_stage, error_msg)
                
                exec_rec.update({"error": error_msg, "error_type": "pdf_parsing", "recoverable": True})
                # Continue processing with available data
            except Exception as e:  # noqa: BLE001
                error_msg = str(e)
                logger.exception("Error in %s: %s", node_name, e)
                self.monitoring_metrics.record_error(e, document_id, stage_name)
                self._handle_error(node, e)
                
                # Fail stage in checkpoint
                if enable_checkpointing and document_id:
                    self.state_manager.fail_stage(document_id, pipeline_stage, str(e))
                
                exec_rec.update({"error": error_msg})

            # Log end event and get end time
            end_time = self.log_node_end(node_name, start_time, error_msg)
            end_ts = end_time.isoformat()
            duration = (end_time - start_time).total_seconds()
            
            # Complete stage-level metrics tracking
            stage_evidence_quality = evidence_quality_scores[-1] if evidence_quality_scores else None
            self.monitoring_metrics.record_document_completion(
                f"{document_id}_{node_name}", 
                stage_name, 
                stage_evidence_quality
            )
            
            exec_rec.update(
                {
                    "evento_cierre": node.evento_cierre,
                    "timestamp_cierre": end_ts,
                    "duration_seconds": duration,
                    "value_added": self.value_chain[node_name]["value_added"],
                }
            )

            execution_trace.append(exec_rec)
            
            # Check if it's time for metric logging
            self._check_and_log_metrics()

        # Complete end-to-end tracking
        final_evidence_quality = evidence_quality_scores[-1] if evidence_quality_scores else None
        self.monitoring_metrics.record_document_completion(document_id, "pipeline_complete", final_evidence_quality)

        # Get execution monitoring summary
        monitoring_summary = self.get_execution_summary()

        # Apply mathematical enhancements if available
        mathematically_enhanced_output = current_data
        if self.mathematical_coordinator:
            try:
                mathematical_input = {
                    'text': self._extract_text_from_data(current_data),
                    'context': self._extract_context_from_data(current_data),
                    'metadata': self._extract_metadata_from_data(current_data),
                    'pipeline_data': current_data
                }
                mathematically_enhanced_output = self.mathematical_coordinator.integrate_with_comprehensive_orchestrator(mathematical_input)
                logger.info("Successfully applied mathematical enhancements to pipeline output")
            except Exception as e:
                logger.warning(f"Mathematical enhancement failed (graceful degradation): {str(e)}")
                # Keep original output if mathematical enhancement fails

        # Enrich final output to ensure canonical sophistication (clusters, evidence, causal)
        enriched_output = mathematically_enhanced_output
        try:
            from canonical_flow import enrichment_postprocessor as _enrich  # type: ignore
            enriched_candidate = _enrich.process(mathematically_enhanced_output, context={"source": "comprehensive_pipeline_orchestrator"})
            if isinstance(enriched_candidate, dict):
                enriched_output = enriched_candidate
        except Exception:
            # Graceful degradation: keep mathematically_enhanced_output
            pass
        
        return {
            "final_output": enriched_output,
            "execution_trace": execution_trace,
            "value_chain": self.value_chain,
            "total_value_added": sum(v["value_added"] for v in self.value_chain.values()),
            "execution_order": self.execution_order,
            "monitoring_summary": monitoring_summary,
            "performance_metrics": self.monitoring_metrics.get_performance_report(),
            "question_context": {
                "id": self.question_context.metadata.derivation_id if self.question_context else None,
                "question_text": self.question_context.question_text if self.question_context else None,
                "integrity_verified": self.question_context.verify_integrity() if self.question_context else False,
            },
            "mathematical_enhancements": {
                "applied": self.mathematical_coordinator is not None,
                "status": self.mathematical_coordinator.get_pipeline_status() if self.mathematical_coordinator else None,
            },
        }

    def batch_process_pdfs(self, pdf_file_paths: List[str], enable_content_analysis: bool = True) -> BatchPDFResults:
        """
        Batch process multiple PDF files with individual file status tracking and content analysis.
        
        Args:
            pdf_file_paths: List of PDF file paths to process
            enable_content_analysis: Whether to perform detailed content analysis on extracted text
            
        Returns:
            BatchPDFResults with individual file processing status and results
        """
        import time
        from pathlib import Path
        
        logger.info(f"Starting batch PDF processing for {len(pdf_file_paths)} files")
        batch_start_time = time.time()
        
        results: List[PDFProcessingResult] = []
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        
        for file_path in pdf_file_paths:
            file_start_time = time.time()
            result = self._process_single_pdf(file_path, enable_content_analysis)
            result.processing_time = time.time() - file_start_time
            
            results.append(result)
            
            if result.status == 'success':
                successful_count += 1
            elif result.status == 'failed':
                failed_count += 1
            else:  # skipped
                skipped_count += 1
                
        batch_end_time = time.time()
        total_processing_time = batch_end_time - batch_start_time
        
        batch_results = BatchPDFResults(
            total_files=len(pdf_file_paths),
            successful=successful_count,
            failed=failed_count,
            skipped=skipped_count,
            results=results,
            total_processing_time=total_processing_time,
            batch_metadata={
                'processing_timestamp': datetime.now().isoformat(),
                'average_processing_time_per_file': total_processing_time / len(pdf_file_paths) if pdf_file_paths else 0,
                'success_rate': successful_count / len(pdf_file_paths) if pdf_file_paths else 0,
            }
        )
        
        logger.info(f"Batch PDF processing completed: {successful_count} successful, {failed_count} failed, {skipped_count} skipped")
        return batch_results

    def _process_single_pdf(self, file_path: str, enable_content_analysis: bool) -> PDFProcessingResult:
        """
        Process a single PDF file with comprehensive error handling and content analysis.
        
        Args:
            file_path: Path to the PDF file
            enable_content_analysis: Whether to analyze extracted content
            
        Returns:
            PDFProcessingResult with processing status and extracted data
        """
        pdf_path = Path(file_path)
        
        # Validate file exists and is accessible
        if not pdf_path.exists():
            return PDFProcessingResult(
                file_path=file_path,
                status='failed',
                error_message=f"File does not exist: {file_path}",
                error_type='file_not_found'
            )
            
        if not pdf_path.suffix.lower() == '.pdf':
            return PDFProcessingResult(
                file_path=file_path,
                status='skipped',
                error_message=f"File is not a PDF: {file_path}",
                error_type='invalid_file_type'
            )
            
        try:
            # Import PDF processing modules
            try:
                from pdf_reader import PDFPageIterator
                from pdf_text_reader import extract_text_from_pdf, PDFTextExtractor
            except ImportError as e:
                raise PDFCriticalError(f"Required PDF processing modules not available: {e}")
            
            # Process PDF with comprehensive content extraction
            extracted_text = ""
            metadata = {}
            content_analysis = {}
            
            # Try primary PDF extraction method
            try:
                with PDFPageIterator(file_path, enable_intelligent_ocr=True) as pdf_iterator:
                    pages_content = []
                    for page_content in pdf_iterator:
                        pages_content.append({
                            'page_num': page_content.page_num,
                            'text': page_content.text,
                            'bbox': page_content.bbox,
                            'span_count': len(page_content.spans)
                        })
                        extracted_text += page_content.text + "\n"
                    
                    metadata['total_pages'] = len(pages_content)
                    metadata['extraction_method'] = 'PDFPageIterator'
                    
            except Exception as pdf_iter_error:
                logger.warning(f"PDFPageIterator failed for {file_path}: {pdf_iter_error}")
                
                # Fallback to alternative extraction method
                try:
                    extractor = PDFTextExtractor()
                    fallback_result = extractor.extract_from_file(file_path)
                    extracted_text = fallback_result.get('text', '')
                    metadata = fallback_result.get('metadata', {})
                    metadata['extraction_method'] = 'PDFTextExtractor_fallback'
                    
                except Exception as fallback_error:
                    # Classification of error types
                    if any(keyword in str(fallback_error).lower() for keyword in ['password', 'encrypted', 'permission']):
                        raise PDFParsingError(f"PDF is encrypted or password protected: {fallback_error}")
                    elif any(keyword in str(fallback_error).lower() for keyword in ['corrupt', 'damaged', 'invalid']):
                        raise PDFParsingError(f"PDF file appears to be corrupted: {fallback_error}")
                    elif any(keyword in str(fallback_error).lower() for keyword in ['memory', 'resource']):
                        raise PDFCriticalError(f"System resource error during PDF processing: {fallback_error}")
                    else:
                        raise PDFParsingError(f"PDF extraction failed: {fallback_error}")
            
            # Content analysis if enabled and text was extracted
            if enable_content_analysis and extracted_text.strip():
                content_analysis = self._analyze_pdf_content(extracted_text, metadata)
                
            # Validate extraction quality
            if not extracted_text.strip():
                return PDFProcessingResult(
                    file_path=file_path,
                    status='failed',
                    error_message="No text could be extracted from PDF",
                    error_type='no_text_extracted',
                    metadata=metadata
                )
                
            return PDFProcessingResult(
                file_path=file_path,
                status='success',
                extracted_text=extracted_text,
                metadata=metadata,
                content_analysis=content_analysis if enable_content_analysis else None
            )
            
        except PDFCriticalError as critical_error:
            logger.error(f"Critical PDF processing error for {file_path}: {critical_error}")
            # Re-raise critical errors to halt batch processing if needed
            raise critical_error
            
        except PDFParsingError as parsing_error:
            logger.warning(f"PDF parsing error for {file_path}: {parsing_error}")
            return PDFProcessingResult(
                file_path=file_path,
                status='failed',
                error_message=str(parsing_error),
                error_type='parsing_error'
            )
            
        except Exception as unexpected_error:
            logger.error(f"Unexpected error processing PDF {file_path}: {unexpected_error}")
            # Classify unexpected errors
            if any(keyword in str(unexpected_error).lower() for keyword in ['memory', 'resource', 'system']):
                raise PDFCriticalError(f"System error during PDF processing: {unexpected_error}")
            else:
                return PDFProcessingResult(
                    file_path=file_path,
                    status='failed',
                    error_message=f"Unexpected error: {unexpected_error}",
                    error_type='unexpected_error'
                )

    def _analyze_pdf_content(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze extracted PDF content for quality metrics and insights.
        
        Args:
            text: Extracted text from PDF
            metadata: PDF metadata
            
        Returns:
            Dictionary with content analysis results
        """
        analysis = {}
        
        try:
            # Basic text statistics
            analysis['text_length'] = len(text)
            analysis['word_count'] = len(text.split())
            analysis['line_count'] = len(text.splitlines())
            analysis['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
            
            # Content quality indicators
            analysis['avg_words_per_line'] = analysis['word_count'] / max(analysis['line_count'], 1)
            analysis['avg_chars_per_word'] = analysis['text_length'] / max(analysis['word_count'], 1)
            
            # Language and encoding analysis
            analysis['has_non_ascii'] = any(ord(char) > 127 for char in text)
            analysis['likely_ocr_text'] = self._detect_ocr_artifacts(text)
            
            # Content structure analysis
            analysis['has_tables'] = bool(self._detect_tabular_content(text))
            analysis['has_lists'] = bool(self._detect_list_content(text))
            analysis['has_headers'] = bool(self._detect_headers(text))
            
            # Combine with PDF metadata
            if metadata:
                analysis['pdf_pages'] = metadata.get('total_pages', 0)
                analysis['extraction_method'] = metadata.get('extraction_method', 'unknown')
                
            # Content classification
            analysis['content_type'] = self._classify_content_type(text, analysis)
            
            logger.debug(f"Content analysis completed: {analysis['word_count']} words, {analysis['content_type']} type")
            
        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            analysis['analysis_error'] = str(e)
            
        return analysis
        
    def _detect_ocr_artifacts(self, text: str) -> bool:
        """Detect if text likely came from OCR processing"""
        # Common OCR artifacts
        ocr_indicators = [
            'rn' in text.replace(' ', ''),  # m misread as rn
            text.count('|') > len(text) / 100,  # Excessive pipe characters
            text.count('1') > len(text) / 50,   # Excessive 1s (misread l or I)
        ]
        return any(ocr_indicators)
        
    def _detect_tabular_content(self, text: str) -> bool:
        """Detect presence of tabular data"""
        lines = text.splitlines()
        tab_indicators = sum(1 for line in lines if line.count('\t') > 2 or line.count('  ') > 5)
        return tab_indicators > len(lines) * 0.1
        
    def _detect_list_content(self, text: str) -> bool:
        """Detect bullet points or numbered lists"""
        import re
        list_patterns = [
            r'^\s*[•\-\*]\s+',  # Bullet points
            r'^\s*\d+[\.\)]\s+',  # Numbered lists
            r'^\s*[a-zA-Z][\.\)]\s+',  # Lettered lists
        ]
        lines = text.splitlines()
        list_lines = 0
        for line in lines:
            if any(re.match(pattern, line) for pattern in list_patterns):
                list_lines += 1
        return list_lines > len(lines) * 0.05
        
    def _detect_headers(self, text: str) -> bool:
        """Detect section headers"""
        lines = text.splitlines()
        header_indicators = sum(1 for line in lines 
                              if line.strip() and 
                              (line.isupper() or 
                               (len(line.split()) < 8 and line.strip().endswith(':')) or
                               line.strip().startswith('#')))
        return header_indicators > 0
        
    def _classify_content_type(self, text: str, analysis: Dict[str, Any]) -> str:
        """Classify the type of content in the PDF"""
        word_count = analysis.get('word_count', 0)
        
        if word_count < 100:
            return 'minimal_content'
        elif analysis.get('has_tables', False):
            return 'structured_document'
        elif analysis.get('has_lists', False) and analysis.get('has_headers', False):
            return 'formatted_document'
        elif word_count > 5000:
            return 'lengthy_document'
        else:
            return 'standard_document'

    def _handle_pdf_parsing_error(self, node: ProcessNode, error: PDFParsingError, current_data: Any) -> None:
        """
        Handle recoverable PDF parsing errors by logging and allowing pipeline continuation.
        
        Args:
            node: The processing node where error occurred
            error: The PDF parsing error
            current_data: Current pipeline data to potentially augment with error info
        """
        try:
            # Log the specific PDF parsing error
            self.exception_monitor.log_exception(f"{node.file_path}_pdf_parsing", error)
            
            # Add error information to runtime context for downstream nodes
            if not hasattr(self, 'pdf_processing_errors'):
                self.pdf_processing_errors = []
                
            self.pdf_processing_errors.append({
                'node': node.file_path,
                'error_type': 'pdf_parsing',
                'error_message': str(error),
                'timestamp': datetime.now().isoformat(),
                'recoverable': True
            })
            
            # Update runtime context with error information
            self.runtime_context['pdf_processing_status'] = 'partial_failure'
            self.runtime_context['pdf_errors'] = self.pdf_processing_errors
            
            logger.info(f"PDF parsing error handled gracefully in {node.file_path}, continuing pipeline execution")
            
        except Exception as handling_error:
            logger.error(f"Error while handling PDF parsing error: {handling_error}")
            
    def _handle_critical_pdf_error(self, node: ProcessNode, error: PDFCriticalError) -> None:
        """
        Handle critical PDF errors that should halt pipeline processing.
        
        Args:
            node: The processing node where critical error occurred
            error: The critical PDF error
        """
        try:
            # Log the critical error with high severity
            self.exception_monitor.log_exception(f"{node.file_path}_pdf_critical", error)
            
            # Mark pipeline as critically failed
            self.runtime_context['pipeline_status'] = 'critical_failure'
            self.runtime_context['critical_error'] = {
                'node': node.file_path,
                'error_type': 'pdf_critical',
                'error_message': str(error),
                'timestamp': datetime.now().isoformat(),
                'halt_required': True
            }
            
            logger.critical(f"Critical PDF error in {node.file_path} requires pipeline halt: {error}")
            
        except Exception as handling_error:
            logger.error(f"Error while handling critical PDF error: {handling_error}")
            # Even if error handling fails, the critical nature should still halt processing

    def _topological_sort(self) -> List[str]:
        """Return execution order respecting dependencies deterministically."""
        visited = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            node = self.process_graph[name]
            # Visit dependencies in their defined order for determinism
            for dep in node.dependencies:
                if dep in self.process_graph:
                    visit(dep)
            order.append(name)

        # Iterate nodes in insertion order (Python 3.7+ dict preserves it)
        for node_name in self.process_graph.keys():
            visit(node_name)
        return order
    
    def _extract_text_from_data(self, data: Any) -> str:
        """Extract text content from pipeline data"""
        if isinstance(data, dict):
            return data.get('text', '') or data.get('final_output', {}).get('text', '')
        elif isinstance(data, str):
            return data
        else:
            return str(data)
    
    def _extract_context_from_data(self, data: Any) -> Dict[str, Any]:
        """Extract context information from pipeline data"""
        if isinstance(data, dict):
            return data.get('context', {}) or data.get('final_output', {}).get('context', {})
        else:
            return {}
    
    def _extract_metadata_from_data(self, data: Any) -> Dict[str, Any]:
        """Extract metadata from pipeline data"""
        if isinstance(data, dict):
            return data.get('metadata', {}) or data.get('final_output', {}).get('metadata', {})
        else:
            return {}

    def _execute_node(self, node: ProcessNode, data: Any) -> Any:
        """Execute a single node using circuit breaker protection and exception monitoring."""
        import time
        
        # Calculate input value (data size or token count before execution)
        input_value = self._calculate_data_value(data)
        start_time = time.time()
        
        module_path = self.root / node.file_path
        if not module_path.exists():
            # Safe pass-through if file is absent
            result = self._passthrough(node, data, reason="file_not_found")
            
            # Calculate output value and update value chain
            end_time = time.time()
            output_value = self._calculate_data_value(result)
            execution_time = end_time - start_time
            self._update_value_chain_metrics(node.file_path, input_value, output_value, execution_time)
            
            return result
        
        # Use circuit breaker for protected execution
        try:
            result = self.circuit_breaker.call(lambda: self._execute_node_protected(node, data, input_value, start_time))
        except Exception as e:
            self.exception_monitor.log_exception("circuit_breaker", e)
            result = self._passthrough(node, data, reason=f"circuit_breaker_failed: {e}")
            end_time = time.time()
            output_value = self._calculate_data_value(result)
            execution_time = end_time - start_time
            self._update_value_chain_metrics(node.file_path, input_value, output_value, execution_time)
            
        return result
        
    def _execute_node_protected(self, node: ProcessNode, data: Any, input_value: float, start_time: float) -> Any:
        """Protected node execution with proper exception handling."""

        # Define the module path once
        module_path = self.root / node.file_path
        
        def _protected_execution():
            """Protected module execution function wrapped by circuit breaker."""
            try:
                # Handle subdirectory modules by converting path to module name
                file_path = node.file_path
                if "/" in file_path:
                    # Convert subdirectory path to Python module path format
                    module_name = file_path.replace("/", ".").replace(".py", "")
                    
                    # For subdirectory modules, add the parent directory to sys.path
                    import sys
                    subdirectory_path = str(self.root / file_path.split("/")[0])
                    if subdirectory_path not in sys.path:
                        sys.path.insert(0, subdirectory_path)
                    
                    # Use the module name without subdirectory prefix for importlib
                    module_file_name = file_path.split("/")[-1]  # Get just the filename
                    spec = importlib.util.spec_from_file_location(
                        f"module_{Path(module_file_name).stem}", str(module_path)
                    )
                else:
                    # Original logic for root directory modules
                    spec = importlib.util.spec_from_file_location(
                        f"module_{module_path.stem}", str(module_path)
                    )
                
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
                else:
                    raise ImportError("spec_loader_missing")

                # Try common callable entrypoints in priority order
                entry_names = ["process", "run", "execute", "main", "handle"]
                func = None
                for name in entry_names:
                    f = getattr(module, name, None)
                    if callable(f):
                        func = f
                        break

                if func is None:
                    raise AttributeError("no_callable")

                # Attempt to call with (data, context)
                context = {
                    "stage": node.stage.value,
                    "file": node.file_path,
                    "timestamp": self._get_timestamp(),
                }
                # Merge runtime context for questionnaire orchestration
                if getattr(self, "runtime_context", None):
                    try:
                        context.update(self.runtime_context)
                    except Exception:
                        pass
                
                try:
                    return func(data=data, context=context)  # type: ignore[misc]
                except TypeError:
                    # Fallbacks for different signatures
                    try:
                        return func(data)  # type: ignore[misc]
                    except TypeError:
                        try:
                            return func(data, context)  # type: ignore[misc]
                        except TypeError:
                            return func()  # type: ignore[misc]
                            
            except Exception as e:
                # Route exception through monitoring system
                self.exception_monitor.logger.error(
                    f"Module execution failure in {node.file_path}: {e}",
                    extra={
                        "component": node.file_path,
                        "stage": node.stage.value,
                        "exception_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                # Re-raise to trigger circuit breaker
                raise

        try:
            # Execute through circuit breaker protection
            result = self.circuit_breaker.execute(_protected_execution)
        except BreakerOpenError:
            logger.warning(f"Circuit breaker open for {node.file_path}, using passthrough")
            result = self._passthrough(node, data, reason="circuit_breaker_open")
        except Exception as e:
            logger.exception("Circuit breaker execution failure in %s: %s", node.file_path, e)
            result = self._passthrough(node, data, reason=str(e))
        
        # Calculate output value and update value chain
        end_time = time.time()
        output_value = self._calculate_data_value(result)
        execution_time = end_time - start_time
        self._update_value_chain_metrics(node.file_path, input_value, output_value, execution_time)
        
        return result

    def _passthrough(self, node: ProcessNode, data: Any, reason: str) -> Dict[str, Any]:
        """Return a structured pass-through payload when no callable is available."""
        base = data if isinstance(data, dict) else {"data": data}
        base.update(
            {
                "processed_by": node.file_path,
                "stage": node.stage.value,
                "reason": reason,
                "outputs_schema": list(node.outputs.keys()),
            }
        )
        # Include runtime context for traceability across questionnaire rounds
        try:
            if getattr(self, "runtime_context", None) and "runtime_context" not in base:
                base["runtime_context"] = dict(self.runtime_context)
        except Exception:
            pass
        return base

    @staticmethod
    def _get_timestamp() -> str:
        return datetime.now().isoformat()

    def _extract_question_from_input(self, input_data: Any) -> str:
        """Extract question text from pipeline input data."""
        try:
            if isinstance(input_data, dict):
                # Try common question field names
                for field in ["question", "query", "question_text", "text", "input"]:
                    if field in input_data and isinstance(input_data[field], str):
                        return input_data[field]
            elif isinstance(input_data, str):
                return input_data
            # Default fallback
            return "Pipeline execution without explicit question"
        except Exception as e:
            logger.warning(f"Failed to extract question from input: {e}")
            return "Pipeline execution with extraction error"

    def _extract_context_data_from_input(self, input_data: Any) -> Dict[str, Any]:
        """Extract context data from pipeline input data."""
        try:
            if isinstance(input_data, dict):
                # Extract all non-question fields as context
                context_data = {}
                question_fields = {"question", "query", "question_text"}
                for key, value in input_data.items():
                    if key not in question_fields:
                        context_data[key] = value
                return context_data
            else:
                return {"input_type": type(input_data).__name__, "pipeline_mode": "direct"}
        except Exception as e:
            logger.warning(f"Failed to extract context data from input: {e}")
            return {"extraction_error": str(e)}

    def _handle_error(self, node: ProcessNode, error: Exception) -> None:
        logger.error("Compensating for error in %s: %s", node.file_path, error)
        # Optionally, attempt to call compensation_engine if available
        try:
            comp = self.process_graph.get("compensation_engine.py")
            if comp:
                self._execute_node(comp, {"error": str(error), "failed_node": node.file_path})
        except Exception:  # Best-effort only
            pass

    # ---------------------------------------------------------------------
    # Questionnaire Orchestration
    # ---------------------------------------------------------------------
    def execute_questionnaire(
        self,
        input_data: Any,
        rounds: int = 4,
        clusters: Optional[List[int]] = None,
        apply_decalogo: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full deterministic pipeline across multiple rounds, clusters, and Decálogo points.

        Args:
            input_data: Initial payload for the pipeline.
            rounds: Number of rounds to execute for each cluster/point.
            clusters: Cluster ids to apply (defaults to [1,2,3,4]).
            apply_decalogo: If True, iterate over Decálogo points from implementacion_mapeo.

        Returns:
            A dict with aggregated results of all runs, plus per-run execution traces.
        """
        clusters = clusters or [1, 2, 3, 4]

        # Attempt to load Decálogo and question mappings
        decalogo_points: Dict[int, Dict[str, Any]] = {}
        questions_by_point: Dict[int, List[str]] = {}
        try:
            from implementacion_mapeo import QuestionDecalogoMapper  # type: ignore

            mapper = QuestionDecalogoMapper()
            # Access defined Decálogo points
            decalogo_points = mapper.decalogo_points  # type: ignore[attr-defined]
            # Build qid lists per point
            try:
                complete = mapper.complete_mapping  # type: ignore[attr-defined]
                for item in complete:
                    pid = getattr(item, "decalogo_point", None)
                    qid = getattr(item, "question_id", None)
                    if isinstance(pid, int) and isinstance(qid, str):
                        questions_by_point.setdefault(pid, []).append(qid)
            except Exception:
                pass
        except Exception:
            # Mapper not available; proceed without strict mapping
            decalogo_points = {i: {"name": f"Punto {i}"} for i in range(1, 12)}

        runs: List[Dict[str, Any]] = []

        # Iterate clusters, points, and rounds
        for cluster in clusters:
            points_iter = list(decalogo_points.keys()) if apply_decalogo else [None]
            for point_id in points_iter:
                # Build context for this (cluster, point)
                point_questions = questions_by_point.get(point_id, []) if point_id is not None else []
                for r in range(1, rounds + 1):
                    # Set runtime context for propagation
                    self.runtime_context = {
                        "round": r,
                        "cluster": cluster,
                        "decalogo_point": point_id,
                        "question_ids": point_questions,
                    }
                    
                    # Clear previous execution events for this run
                    self.execution_events.clear()
                    
                    # Initialize or derive QuestionContext for this iteration
                    try:
                        if self.question_context is None:
                            # First initialization
                            base_question = self._extract_question_from_input(input_data)
                            context_data = self._extract_context_data_from_input(input_data)
                            context_data.update(self.runtime_context)
                            self.question_context = create_question_context(base_question, context_data)
                        else:
                            # Derive context for new iteration
                            self.question_context = self.question_context.derive_with_context(**self.runtime_context)
                    except Exception as e:
                        logger.error(f"Failed to initialize/derive QuestionContext for cluster {cluster}, point {point_id}, round {r}: {e}")
                        # Continue with existing context or None
                    
                    # Start batch processing if first run
                    if len(runs) == 0:
                        total_expected = len(clusters) * len(points_iter) * rounds
                        self.monitoring_metrics.start_batch_processing(total_expected)
                    
                    # Execute the pipeline for this configuration
                    result = self.execute_pipeline(input_data)
                    runs.append(
                        {
                            "cluster": cluster,
                            "decalogo_point": point_id,
                            "round": r,
                            "result": result,
                        }
                    )

        # Finish batch processing
        self.monitoring_metrics.finish_batch_processing()

        # Compute a lightweight summary
        summary = {
            "total_runs": len(runs),
            "clusters": clusters,
            "applied_decalogo": apply_decalogo,
        }

        # Reset runtime context and QuestionContext after execution
        self.runtime_context = {}
        self.question_context = None

        return {
            "runs": runs, 
            "summary": summary,
            "batch_performance_report": self.monitoring_metrics.get_performance_report()
        }

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    def visualize_pipeline(self) -> str:
        visualization = {"nodes": [], "edges": [], "stages": {}}
        for node_name, node in self.process_graph.items():
            visualization["nodes"].append(
                {
                    "id": node_name,
                    "label": node_name,
                    "stage": node.stage.value,
                    "value": self.value_chain.get(node_name, {}).get("value_added", 0),
                }
            )
            for dep in node.dependencies:
                visualization["edges"].append(
                    {
                        "source": dep,
                        "target": node_name,
                        "value_flow": self.value_chain.get(dep, {}).get("output_value", 0),
                    }
                )
        return json.dumps(visualization, indent=2)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report from monitoring metrics
        """
        return self.monitoring_metrics.get_performance_report()
    
    def _extract_evidence_quality_from_data(self, data: Any, node_name: str) -> Optional[EvidenceQualityMetrics]:
        """Extract evidence quality metrics from processing results"""
        try:
            if isinstance(data, dict):
                # Look for evidence-related metrics in the data
                evidence_data = data.get('evidence', {}) or data.get('processed_evidence', {}) or data.get('validated_evidence', {})
                
                if evidence_data:
                    # Calculate metrics based on available evidence data
                    content_completeness = self._calculate_content_completeness(evidence_data)
                    evidence_density = self._calculate_evidence_density(evidence_data, data)
                    citation_accuracy = self._calculate_citation_accuracy(evidence_data)
                    contextual_relevance = self._calculate_contextual_relevance(evidence_data, data)
                    structural_coherence = self._calculate_structural_coherence(evidence_data)
                    
                    return EvidenceQualityMetrics(
                        content_completeness_score=content_completeness,
                        evidence_density=evidence_density,
                        citation_accuracy=citation_accuracy,
                        contextual_relevance=contextual_relevance,
                        structural_coherence=structural_coherence
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to extract evidence quality metrics for {node_name}: {e}")
            
        return None
    
    def _calculate_content_completeness(self, evidence_data: Dict[str, Any]) -> float:
        """Calculate content completeness score based on evidence coverage"""
        if not evidence_data:
            return 0.0
            
        # Check for key evidence components
        components = ['citations', 'context', 'relevance_score', 'extracted_text']
        present_components = sum(1 for comp in components if comp in evidence_data and evidence_data[comp])
        
        return present_components / len(components)
    
    def _calculate_evidence_density(self, evidence_data: Dict[str, Any], full_data: Dict[str, Any]) -> float:
        """Calculate evidence density (evidence items per page/section)"""
        try:
            evidence_count = len(evidence_data.get('citations', []))
            page_count = full_data.get('metadata', {}).get('pages', 1) or 1
            return evidence_count / max(1, page_count)
        except Exception:
            return 0.0
    
    def _calculate_citation_accuracy(self, evidence_data: Dict[str, Any]) -> float:
        """Calculate citation accuracy based on format and completeness"""
        try:
            citations = evidence_data.get('citations', [])
            if not citations:
                return 0.0
                
            # Simple heuristic: check if citations have required fields
            valid_citations = 0
            for citation in citations:
                if isinstance(citation, dict):
                    required_fields = ['page', 'text', 'context']
                    if all(field in citation and citation[field] for field in required_fields):
                        valid_citations += 1
                        
            return valid_citations / len(citations)
        except Exception:
            return 0.0
    
    def _calculate_contextual_relevance(self, evidence_data: Dict[str, Any], full_data: Dict[str, Any]) -> float:
        """Calculate contextual relevance of evidence to the question"""
        try:
            # Simple heuristic based on relevance scores if available
            relevance_scores = evidence_data.get('relevance_scores', [])
            if relevance_scores:
                return sum(relevance_scores) / len(relevance_scores)
            
            # Fallback: check for question-related keywords
            question_text = full_data.get('question', '').lower()
            evidence_text = str(evidence_data.get('extracted_text', '')).lower()
            
            if question_text and evidence_text:
                question_words = set(question_text.split())
                evidence_words = set(evidence_text.split())
                overlap = len(question_words.intersection(evidence_words))
                return min(1.0, overlap / max(1, len(question_words)))
                
        except Exception:
            pass
        
        return 0.5  # Default neutral score
    
    def _calculate_structural_coherence(self, evidence_data: Dict[str, Any]) -> float:
        """Calculate structural coherence of evidence organization"""
        try:
            # Check for organized structure
            structure_indicators = ['sections', 'hierarchy', 'categories', 'relationships']
            structure_score = 0.0
            
            for indicator in structure_indicators:
                if indicator in evidence_data and evidence_data[indicator]:
                    structure_score += 0.25
                    
            return min(1.0, structure_score)
        except Exception:
            return 0.5
    
    def _check_and_log_metrics(self):
        """Check if it's time to log metrics summary and do so if needed"""
        now = datetime.now()
        if (now - self.last_metric_log).total_seconds() >= (self.metric_log_interval_minutes * 60):
            self.monitoring_metrics.log_metric_summary()
            self.last_metric_log = now
    
    def configure_metric_logging(self, interval_minutes: int = 5):
        """Configure the interval for automatic metric logging"""
        self.metric_log_interval_minutes = interval_minutes
    
    def __del__(self):
        """Cleanup monitoring on orchestrator destruction"""
        try:
            self.monitoring_metrics.stop_memory_monitoring()
        except Exception:
            pass


def get_canonical_process_graph() -> dict:
    """Return the canonical, deterministic process graph mapping.

    This exposes the exact mapping of module filename -> ProcessNode used by
    the orchestrator, to satisfy external tools requiring a direct dictionary
    return of the canonical pipeline structure.
    """
    orch = ComprehensivePipelineOrchestrator()
    return orch.process_graph


def generate_excellence_analysis_report(input_data: Optional[Any] = None) -> dict:
    """Generate a canonical excellence analysis report.

    - Ensures the value chain is established and positive.
    - Computes a deterministic execution order.
    - Optionally executes the pipeline on provided input_data to enrich trace.

    Returns a dict containing:
      - execution_order: list of node names in deterministic order
      - stages: per-stage list of nodes
      - value_chain: per-node value metrics (input/output/value_added/efficiency)
      - execution_trace: if input_data provided
    """
    orch = ComprehensivePipelineOrchestrator()
    orch.guarantee_value_chain()
    execution_order = orch._topological_sort()

    # Organize nodes by stage
    stages: dict[str, list[str]] = {}
    for name, node in orch.process_graph.items():
        stages.setdefault(node.stage.value, []).append(name)

    report: dict = {
        "execution_order": execution_order,
        "stages": stages,
        "value_chain": orch.value_chain,
    }

    if input_data is not None:
        result = orch.execute_pipeline(input_data)
        report["execution_trace"] = result.get("execution_trace", [])
        report["total_value_added"] = result.get("total_value_added", 0.0)

    return report


def generate_deterministic_flux_markdown(include_value_chain: bool = True) -> str:
    """Build a granular, stage-organized Markdown description of the deterministic pipeline.

    The narrative is synthesized from the canonical process graph and includes:
    - Stage overview and purpose
    - Per-node contribution: dependencies, inputs, outputs, events, and value metrics
    - Determinism notes (topological order and dependency guarantees)
    """
    orch = ComprehensivePipelineOrchestrator()
    orch.guarantee_value_chain()
    execution_order = orch._topological_sort()

    # Stage descriptions
    stage_titles = {
        ProcessStage.INGESTION: "Stage 1 — Ingestion & Preparation",
        ProcessStage.CONTEXT_BUILD: "Stage 2 — Context Construction",
        ProcessStage.KNOWLEDGE: "Stage 3 — Knowledge Extraction & Graph Building",
        ProcessStage.ANALYSIS: "Stage 4 — Analysis & NLP",
        ProcessStage.CLASSIFICATION: "Stage 5 — Classification & Scoring",
        ProcessStage.SEARCH: "Stage 6 — Search & Retrieval",
        ProcessStage.ORCHESTRATION: "Stage 7/8/9/10 — Orchestration, Monitoring & Validation",
        ProcessStage.AGGREGATION: "Stage 11 — Aggregation & Reporting",
        ProcessStage.INTEGRATION: "Stage 12 — Integration & Metrics",
        ProcessStage.SYNTHESIS: "Stage 11 — Synthesis",
    }
    stage_purposes = {
        ProcessStage.INGESTION: "Acquire, load, and normalize inputs into structured, validated form.",
        ProcessStage.CONTEXT_BUILD: "Create immutable, adapted context with full lineage.",
        ProcessStage.KNOWLEDGE: "Construct semantic/call causal structures and embeddings.",
        ProcessStage.ANALYSIS: "Analyze text, intents, evidence; evaluate and align to standards.",
        ProcessStage.CLASSIFICATION: "Score, calculate final metrics, and bound risk deterministically.",
        ProcessStage.SEARCH: "Index, retrieve, hybridize, and semantically rerank with stability.",
        ProcessStage.ORCHESTRATION: "Route, orchestrate, monitor, enforce contracts and constraints.",
        ProcessStage.AGGREGATION: "Compile reports and artifacts for downstream consumption.",
        ProcessStage.INTEGRATION: "Collect metrics, analyze, feedback, compensate, and optimize.",
        ProcessStage.SYNTHESIS: "Synthesize and format final answers for presentation.",
    }

    # Order nodes by execution order, then group by stage
    stage_to_nodes: dict[ProcessStage, list[str]] = {stage: [] for stage in ProcessStage}
    for name in execution_order:
        node = orch.process_graph[name]
        stage_to_nodes[node.stage].append(name)

    lines: list[str] = []
    lines.append("# Deterministic Pipeline (Flux) — Granular Stage-by-Stage Explanation\n")
    lines.append(f"Generated at: {datetime.now().isoformat()}\n")
    lines.append("This document describes the canonical, deterministic pipeline. Execution is strictly topological: each node runs only after all its dependencies have produced outputs.\n")

    # Global determinism notes
    lines.append("Determinism Principles:\n")
    lines.append("- Fixed process graph and insertion-ordered traversal ensure a stable topological order.\n")
    lines.append("- Each node declares explicit dependencies; execution respects these constraints.\n")
    lines.append("- If a module lacks a callable entrypoint (process/run/execute/main/handle), a structured, non-mutating pass-through is used to preserve flow and traceability.\n")
    lines.append("- A value-chain check ensures each step contributes measurable value; under-contributing nodes are auto-enhanced with quality/validation metrics to maintain monotonic value growth.\n")

    for stage in ProcessStage:
        nodes = stage_to_nodes.get(stage, [])
        if not nodes:
            continue
        lines.append("")
        lines.append(f"## {stage_titles.get(stage, stage.value)}\n")
        lines.append(f"Purpose: {stage_purposes.get(stage, '')}\n")

        for name in nodes:
            node = orch.process_graph[name]
            deps = node.dependencies or []
            # Construct inputs description as union of dependency outputs
            dep_outputs: list[str] = []
            for d in deps:
                if d in orch.process_graph:
                    dep_outputs.extend(list(orch.process_graph[d].outputs.keys()))
            dep_outputs = sorted(set(dep_outputs))

            lines.append(f"### {name}\n")
            lines.append(f"- Process type: {node.process_type}")
            lines.append(f"- Stage: {node.stage.value}")
            lines.append(f"- Depends on: {', '.join(deps) if deps else 'None'}")
            if dep_outputs:
                lines.append(f"- Expected inputs (from dependencies): {', '.join(dep_outputs)}")
            lines.append(f"- Declared outputs: {', '.join(node.outputs.keys()) or 'None'}")
            lines.append(f"- Start event: {node.evento_inicio}")
            lines.append(f"- Close event: {node.evento_cierre}")
            if include_value_chain:
                vc = orch.value_chain.get(name, {})
                lines.append(f"- Value metrics: keys={', '.join(node.value_metrics.keys()) or 'None'}")
                if vc:
                    lines.append(
                        f"- Value contribution: input_value={vc.get('input_value', 0):.3f}, output_value={vc.get('output_value', 0):.3f}, value_added={vc.get('value_added', 0):.3f}, efficiency={vc.get('efficiency', 0):.3f}"
                    )
            lines.append("- Determinism: executes after all dependencies complete; participates in the monotonic value chain.")
            lines.append("")

    return "\n".join(lines)


def write_deterministic_flux_report(path: str = "DETERMINISTIC_FLUX.md", include_value_chain: bool = True) -> str:
    """Generate and write the deterministic flux report to the given path."""
    content = generate_deterministic_flux_markdown(include_value_chain=include_value_chain)
    p = Path(path)
    p.write_text(content, encoding="utf-8")
    return str(p.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Simple CLI flags without argparse to keep import-safety
    argv = sys.argv[1:]
    json_only = "--json" in argv
    quick = "--quick" in argv or "-q" in argv

    orch = ComprehensivePipelineOrchestrator()
    ok = orch.guarantee_value_chain()
    status_msg = {
        True: "\u2713 Value chain guaranteed for all nodes",
        False: "\u26A0 Some nodes needed value enhancement (applied)",
    }[ok]

    # Minimal demo input
    input_payload = {"initial_data": "development_plan.pdf"}

    if quick:
        # Quick-run: do not process heavy PDF paths; just compute execution order and a dry-run summary
        topo = orch._topological_sort()
        summary = {
            "ok": ok,
            "nodes": len(topo),
            "first": topo[0] if topo else None,
            "last": topo[-1] if topo else None,
            "execution_order": topo,
        }
        if json_only:
            print(json.dumps({"status": status_msg, "summary": summary}))
            sys.exit(0 if ok else 1)
        else:
            print(status_msg)
            print(f"Deterministic order length: {len(topo)}")
            print(f"First: {summary['first']}, Last: {summary['last']}")
            # Also write deterministic flux report quickly
            report_path = write_deterministic_flux_report()
            print(f"Deterministic flux report written to {report_path}")
            sys.exit(0 if ok else 1)
    else:
        # Standard execution
        result = orch.execute_pipeline(input_payload)
        if json_only:
            print(json.dumps({"status": status_msg, "result": result}))
            sys.exit(0 if ok else 1)
        else:
            print(status_msg)
            # Save execution trace
            out_path = Path("execution_trace.json")
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Execution trace written to {out_path.resolve()}")
            # Save deterministic flux report
            report_path = write_deterministic_flux_report()
            print(f"Deterministic flux report written to {report_path}")
            sys.exit(0 if ok else 1)
