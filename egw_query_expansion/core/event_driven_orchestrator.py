"""
Event-Driven Orchestrator for EGW Query Expansion Pipeline

This orchestrator publishes events for pipeline stage transitions instead of
directly calling pipeline components. It maintains synchronous execution flow
while enabling loose coupling through the event bus system.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict, deque

from .event_bus import (
    SynchronousEventBus, get_event_bus, EventHandler, BaseEvent,
    EventProcessingResult, EventPriority
)
from .event_schemas import (
    PipelineStage, PipelineContext, ValidationPayload, ValidationResult,
    StageStartedEvent, StageCompletedEvent, StageFailedEvent,
    ValidationRequestedEvent, ValidationCompletedEvent, ValidationOutcome,
    DataTransformEvent, OrchestratorCommandEvent, PipelineStateChangedEvent,
    ErrorEvent, PerformanceMetricEvent, create_stage_started_event,
    create_validation_request, create_error_event, create_performance_event
)

logger = logging.getLogger(__name__)


class OrchestrationState(Enum):
    """States of the orchestration process"""
    IDLE = "idle"
    INITIALIZING = "initializing" 
    RUNNING = "running"
    WAITING_FOR_VALIDATION = "waiting_for_validation"
    COMPLETING = "completing"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class StageConfiguration:
    """Configuration for a pipeline stage"""
    stage: PipelineStage
    required_validations: List[str] = field(default_factory=list)
    optional_validations: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3
    dependencies: List[PipelineStage] = field(default_factory=list)
    stage_processor: Optional[Callable] = None


@dataclass  
class ExecutionPlan:
    """Plan for pipeline execution"""
    stages: List[StageConfiguration] = field(default_factory=list)
    execution_order: List[PipelineStage] = field(default_factory=list)
    validation_requirements: Dict[PipelineStage, List[str]] = field(default_factory=dict)
    parallel_stages: Dict[PipelineStage, List[PipelineStage]] = field(default_factory=dict)


class EventDrivenOrchestrator(EventHandler):
    """
    Event-driven orchestrator that coordinates pipeline execution through events
    """
    
    def __init__(self, event_bus: Optional[SynchronousEventBus] = None):
        self.event_bus = event_bus or get_event_bus()
        self.orchestrator_id = "event_driven_orchestrator"
        
        # Orchestration state
        self.state = OrchestrationState.IDLE
        self.current_context: Optional[PipelineContext] = None
        self.execution_plan: Optional[ExecutionPlan] = None
        self.completed_stages: List[PipelineStage] = []
        self.failed_stages: List[PipelineStage] = []
        self.stage_results: Dict[PipelineStage, Any] = {}
        
        # Validation tracking
        self.pending_validations: Dict[str, ValidationPayload] = {}
        self.validation_results: Dict[str, ValidationResult] = {}
        self.validation_timeouts: Dict[str, float] = {}
        
        # Performance tracking
        self.stage_start_times: Dict[PipelineStage, float] = {}
        self.performance_metrics: Dict[PipelineStage, Dict[str, float]] = defaultdict(dict)
        
        # Error handling
        self.error_count = 0
        self.max_errors = 10
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Setup event subscriptions
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Setup event subscriptions for orchestrator"""
        self.event_bus.subscribe("validation.completed", self)
        self.event_bus.subscribe("pipeline.error", self)
        self.event_bus.subscribe("orchestrator.command", self)
        
        logger.debug(f"Orchestrator {self.orchestrator_id} subscribed to events")
    
    @property
    def handler_id(self) -> str:
        return self.orchestrator_id
    
    def can_handle(self, event: BaseEvent) -> bool:
        """Check if orchestrator can handle the event"""
        return event.event_type in [
            "validation.completed",
            "pipeline.error", 
            "orchestrator.command"
        ]
    
    def handle(self, event: BaseEvent) -> Optional[BaseEvent]:
        """Handle orchestrator events"""
        try:
            if event.event_type == "validation.completed":
                return self._handle_validation_completed(event)
            elif event.event_type == "pipeline.error":
                return self._handle_pipeline_error(event)
            elif event.event_type == "orchestrator.command":
                return self._handle_orchestrator_command(event)
        except Exception as e:
            logger.error(f"Orchestrator failed to handle event: {e}")
            self._record_error(f"Event handling failed: {str(e)}")
        
        return None
    
    def execute_pipeline(self, query: str, document_id: Optional[str] = None,
                        execution_plan: Optional[ExecutionPlan] = None) -> Dict[str, Any]:
        """
        Execute the pipeline using event-driven coordination
        
        Args:
            query: Query to process
            document_id: Optional document identifier
            execution_plan: Optional custom execution plan
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        
        try:
            # Initialize execution
            self.state = OrchestrationState.INITIALIZING
            self.current_context = PipelineContext(
                query=query,
                document_id=document_id,
                metadata={"execution_start": start_time}
            )
            self.execution_plan = execution_plan or self._create_default_execution_plan()
            
            # Reset state
            self.completed_stages.clear()
            self.failed_stages.clear() 
            self.stage_results.clear()
            self.pending_validations.clear()
            self.validation_results.clear()
            self.stage_start_times.clear()
            
            # Publish state change event
            self._publish_state_change(OrchestrationState.IDLE, OrchestrationState.RUNNING)
            
            # Execute stages in order
            self.state = OrchestrationState.RUNNING
            
            for stage in self.execution_plan.execution_order:
                if not self._execute_stage(stage):
                    self.state = OrchestrationState.FAILED
                    break
            
            # Finalize execution
            if self.state != OrchestrationState.FAILED:
                self.state = OrchestrationState.COMPLETED
            
            execution_time = time.time() - start_time
            
            # Publish completion event
            self._publish_state_change(OrchestrationState.RUNNING, self.state)
            
            return {
                "success": self.state == OrchestrationState.COMPLETED,
                "completed_stages": [stage.value for stage in self.completed_stages],
                "failed_stages": [stage.value for stage in self.failed_stages],
                "stage_results": {stage.value: result for stage, result in self.stage_results.items()},
                "validation_results": {k: v.__dict__ for k, v in self.validation_results.items()},
                "execution_time_seconds": execution_time,
                "performance_metrics": dict(self.performance_metrics),
                "error_count": self.error_count
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.state = OrchestrationState.FAILED
            self._record_error(f"Pipeline execution failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "completed_stages": [stage.value for stage in self.completed_stages],
                "failed_stages": [stage.value for stage in self.failed_stages]
            }
    
    def _create_default_execution_plan(self) -> ExecutionPlan:
        """Create default execution plan for the pipeline"""
        stages = [
            StageConfiguration(
                stage=PipelineStage.INGESTION,
                required_validations=["normative_compliance"]
            ),
            StageConfiguration(
                stage=PipelineStage.ANALYSIS,
                required_validations=["analysis_quality"],
                dependencies=[PipelineStage.INGESTION]
            ),
            StageConfiguration(
                stage=PipelineStage.KNOWLEDGE_EXTRACTION,
                required_validations=["extraction_completeness"],
                dependencies=[PipelineStage.ANALYSIS]
            ),
            StageConfiguration(
                stage=PipelineStage.CLASSIFICATION,
                required_validations=["rubric_structure", "classification_accuracy"],
                dependencies=[PipelineStage.KNOWLEDGE_EXTRACTION]
            ),
            StageConfiguration(
                stage=PipelineStage.AGGREGATION,
                required_validations=["constraint_validation", "satisfiability"],
                dependencies=[PipelineStage.CLASSIFICATION]
            ),
            StageConfiguration(
                stage=PipelineStage.SEARCH_RETRIEVAL,
                optional_validations=["retrieval_relevance"],
                dependencies=[PipelineStage.AGGREGATION]
            ),
            StageConfiguration(
                stage=PipelineStage.SYNTHESIS,
                required_validations=["synthesis_coherence"],
                dependencies=[PipelineStage.SEARCH_RETRIEVAL]
            ),
            StageConfiguration(
                stage=PipelineStage.INTEGRATION,
                optional_validations=["integration_completeness"],
                dependencies=[PipelineStage.SYNTHESIS]
            )
        ]
        
        execution_order = [config.stage for config in stages]
        
        validation_requirements = {
            config.stage: config.required_validations + config.optional_validations
            for config in stages
        }
        
        return ExecutionPlan(
            stages=stages,
            execution_order=execution_order,
            validation_requirements=validation_requirements
        )
    
    def _execute_stage(self, stage: PipelineStage) -> bool:
        """
        Execute a pipeline stage through event publishing
        
        Args:
            stage: Stage to execute
            
        Returns:
            True if stage executed successfully
        """
        try:
            # Check dependencies
            stage_config = self._get_stage_config(stage)
            if stage_config and not self._dependencies_satisfied(stage_config.dependencies):
                logger.error(f"Dependencies not satisfied for stage {stage}")
                return False
            
            # Record start time
            self.stage_start_times[stage] = time.time()
            
            # Update context
            if self.current_context:
                self.current_context.processing_history.append(stage.value)
            
            # Publish stage started event
            started_event = StageStartedEvent(
                stage=stage,
                context=self.current_context,
                metadata=self._create_event_metadata()
            )
            
            result = self.event_bus.publish(started_event)
            
            if not result.success:
                logger.error(f"Failed to publish stage started event for {stage}")
                return False
            
            # Execute stage logic (placeholder - would integrate with actual stage processors)
            stage_result = self._process_stage(stage)
            
            if stage_result is None:
                self._publish_stage_failed(stage, "Stage processing returned no result")
                return False
            
            # Store result
            self.stage_results[stage] = stage_result
            
            # Publish stage completed event  
            completed_event = StageCompletedEvent(
                stage=stage,
                context=self.current_context,
                results=stage_result,
                success=True,
                metadata=self._create_event_metadata()
            )
            
            result = self.event_bus.publish(completed_event)
            
            if not result.success:
                logger.error(f"Failed to publish stage completed event for {stage}")
                return False
            
            # Request validations
            if not self._request_stage_validations(stage):
                logger.warning(f"Validation requests failed for stage {stage}")
                # Continue execution even if validations fail to start
            
            # Wait for required validations
            if not self._wait_for_validations(stage):
                logger.error(f"Required validations failed for stage {stage}")
                return False
            
            # Record completion
            self.completed_stages.append(stage)
            self._record_stage_performance(stage)
            
            logger.info(f"Stage {stage} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Stage {stage} execution failed: {e}")
            self._publish_stage_failed(stage, str(e))
            self.failed_stages.append(stage)
            return False
    
    def _get_stage_config(self, stage: PipelineStage) -> Optional[StageConfiguration]:
        """Get configuration for a stage"""
        if not self.execution_plan:
            return None
            
        for config in self.execution_plan.stages:
            if config.stage == stage:
                return config
        return None
    
    def _dependencies_satisfied(self, dependencies: List[PipelineStage]) -> bool:
        """Check if stage dependencies are satisfied"""
        return all(dep in self.completed_stages for dep in dependencies)
    
    def _process_stage(self, stage: PipelineStage) -> Any:
        """
        Process stage logic - placeholder for actual stage processors
        In a real implementation, this would delegate to the appropriate stage processor
        """
        import time
        time.sleep(0.1)  # Simulate processing time
        
        # Return mock result based on stage
        if stage == PipelineStage.INGESTION:
            return {"documents": ["doc1", "doc2"], "metadata": {"count": 2}}
        elif stage == PipelineStage.ANALYSIS:
            return {"analysis_result": "positive", "confidence": 0.85}
        elif stage == PipelineStage.KNOWLEDGE_EXTRACTION:
            return {"entities": ["entity1", "entity2"], "relations": ["rel1"]}
        elif stage == PipelineStage.CLASSIFICATION:
            return {"classification": "category_a", "score": 0.92}
        elif stage == PipelineStage.AGGREGATION:
            return {"aggregated_score": 0.88, "components": ["comp1", "comp2"]}
        elif stage == PipelineStage.SEARCH_RETRIEVAL:
            return {"results": ["result1", "result2"], "relevance_scores": [0.9, 0.8]}
        elif stage == PipelineStage.SYNTHESIS:
            return {"synthesized_output": "Generated response", "quality_score": 0.85}
        elif stage == PipelineStage.INTEGRATION:
            return {"final_output": "Integrated result", "completeness": 0.95}
        else:
            return {"processed": True, "stage": stage.value}
    
    def _request_stage_validations(self, stage: PipelineStage) -> bool:
        """Request validations for a completed stage"""
        if not self.execution_plan:
            return True
            
        validation_types = self.execution_plan.validation_requirements.get(stage, [])
        
        if not validation_types:
            return True  # No validations required
        
        success = True
        
        for validation_type in validation_types:
            try:
                validation_request = create_validation_request(
                    validator_id=f"{validation_type}_validator",
                    validation_type=validation_type,
                    data={
                        "stage": stage,
                        "results": self.stage_results.get(stage),
                        "context": self.current_context
                    },
                    context=self.current_context,
                    source_id=self.orchestrator_id
                )
                
                # Track pending validation
                self.pending_validations[validation_request.event_id] = validation_request.payload
                self.validation_timeouts[validation_request.event_id] = time.time() + 30  # 30 second timeout
                
                result = self.event_bus.publish(validation_request)
                
                if not result.success:
                    logger.error(f"Failed to request validation {validation_type} for stage {stage}")
                    success = False
                
            except Exception as e:
                logger.error(f"Failed to create validation request {validation_type}: {e}")
                success = False
        
        return success
    
    def _wait_for_validations(self, stage: PipelineStage) -> bool:
        """Wait for required validations to complete"""
        if not self.execution_plan:
            return True
            
        stage_config = self._get_stage_config(stage)
        if not stage_config or not stage_config.required_validations:
            return True  # No required validations
        
        # Set state to waiting
        self.state = OrchestrationState.WAITING_FOR_VALIDATION
        
        # Wait for validations with timeout
        timeout = time.time() + 60  # 60 second timeout
        
        while time.time() < timeout:
            # Check if all required validations are complete
            required_complete = True
            
            for validation_id, payload in self.pending_validations.items():
                if payload.validation_type in stage_config.required_validations:
                    if validation_id not in self.validation_results:
                        required_complete = False
                        break
            
            if required_complete:
                # Check if all required validations passed
                for validation_id, result in self.validation_results.items():
                    payload = self.pending_validations.get(validation_id)
                    if payload and payload.validation_type in stage_config.required_validations:
                        if result.outcome == ValidationOutcome.FAILED:
                            logger.error(f"Required validation {payload.validation_type} failed for stage {stage}")
                            self.state = OrchestrationState.RUNNING
                            return False
                
                self.state = OrchestrationState.RUNNING
                return True
            
            time.sleep(0.1)  # Brief pause
        
        # Timeout reached
        logger.error(f"Validation timeout for stage {stage}")
        self.state = OrchestrationState.RUNNING
        return False
    
    def _handle_validation_completed(self, event: ValidationCompletedEvent) -> Optional[BaseEvent]:
        """Handle validation completion events"""
        result = event.result
        
        # Find corresponding validation request
        request_id = None
        for vid, payload in self.pending_validations.items():
            if payload.validator_id == result.validator_id:
                request_id = vid
                break
        
        if request_id:
            self.validation_results[request_id] = result
            logger.debug(f"Validation {result.validator_id} completed with outcome {result.outcome}")
        
        return None
    
    def _handle_pipeline_error(self, event: ErrorEvent) -> Optional[BaseEvent]:
        """Handle pipeline error events"""
        self.error_count += 1
        logger.error(f"Pipeline error: {event.error_message}")
        
        # Check if error limit exceeded
        if self.error_count >= self.max_errors:
            self.state = OrchestrationState.FAILED
            logger.error("Maximum error count exceeded, failing pipeline")
        
        return None
    
    def _handle_orchestrator_command(self, event: OrchestratorCommandEvent) -> Optional[BaseEvent]:
        """Handle orchestrator command events"""
        command = event.command
        parameters = event.parameters
        
        if command == "pause":
            logger.info("Pipeline execution paused")
            # Implementation would pause execution
        elif command == "resume":
            logger.info("Pipeline execution resumed")
            # Implementation would resume execution
        elif command == "abort":
            logger.info("Pipeline execution aborted")
            self.state = OrchestrationState.FAILED
        else:
            logger.warning(f"Unknown orchestrator command: {command}")
        
        return None
    
    def _publish_stage_failed(self, stage: PipelineStage, error_message: str):
        """Publish stage failed event"""
        failed_event = StageFailedEvent(
            stage=stage,
            context=self.current_context,
            error=error_message,
            metadata=self._create_event_metadata()
        )
        
        self.event_bus.publish(failed_event)
    
    def _publish_state_change(self, old_state: OrchestrationState, new_state: OrchestrationState):
        """Publish pipeline state change event"""
        state_event = PipelineStateChangedEvent(
            previous_state=old_state.value,
            new_state=new_state.value,
            affected_stages=[],
            metadata=self._create_event_metadata()
        )
        
        self.event_bus.publish(state_event)
    
    def _record_stage_performance(self, stage: PipelineStage):
        """Record performance metrics for a stage"""
        if stage in self.stage_start_times:
            processing_time = time.time() - self.stage_start_times[stage]
            
            self.performance_metrics[stage]["processing_time_seconds"] = processing_time
            
            # Publish performance event
            perf_event = create_performance_event(
                stage=stage,
                metric_name="processing_time",
                metric_value=processing_time,
                source_id=self.orchestrator_id
            )
            
            self.event_bus.publish(perf_event)
    
    def _record_error(self, error_message: str):
        """Record and publish error"""
        error_event = create_error_event(
            error_type="orchestration_error",
            message=error_message,
            source_id=self.orchestrator_id
        )
        
        self.event_bus.publish(error_event)
    
    def _create_event_metadata(self):
        """Create event metadata with orchestrator info"""
        from .event_bus import EventMetadata
        return EventMetadata(source_id=self.orchestrator_id)
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        return {
            "current_state": self.state.value,
            "completed_stages": len(self.completed_stages),
            "failed_stages": len(self.failed_stages),
            "pending_validations": len(self.pending_validations),
            "completed_validations": len(self.validation_results),
            "error_count": self.error_count,
            "performance_metrics": dict(self.performance_metrics)
        }