"""
Event-Driven Pipeline Orchestrator
=================================

Refactored orchestrator that uses the event bus system instead of direct imports
between orchestrators and concrete pipeline stages. All stage interactions happen
through typed events, eliminating direct coupling.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from uuid import uuid4

from event_bus import EventBus, event_handler, subscribe_decorated_handlers
from event_schemas import (
    EventType, BaseEvent, StageEventData, OrchestrationEventData,
    create_stage_started_event, StageCompletedEvent, StageFailedEvent,
    OrchestrationStartedEvent, OrchestrationCompletedEvent, OrchestrationFailedEvent
)


logger = logging.getLogger(__name__)


class PipelineExecutionContext:
    """Context for tracking pipeline execution state"""
    
    def __init__(self, execution_id: str, pipeline_config: Dict[str, Any]):
        self.execution_id = execution_id
        self.pipeline_config = pipeline_config
        self.start_time = datetime.utcnow()
        self.current_stage: Optional[str] = None
        self.completed_stages: List[str] = []
        self.failed_stages: List[str] = []
        self.stage_results: Dict[str, Any] = {}
        self.pipeline_data: Dict[str, Any] = {}
        self.error_details: Optional[Dict[str, Any]] = None
        
    def is_complete(self, required_stages: List[str]) -> bool:
        """Check if all required stages have completed"""
        return all(stage in self.completed_stages for stage in required_stages)
    
    def has_failures(self) -> bool:
        """Check if any stages have failed"""
        return len(self.failed_stages) > 0
    
    def get_progress_percentage(self, total_stages: int) -> float:
        """Calculate execution progress as percentage"""
        if total_stages == 0:
            return 100.0
        return (len(self.completed_stages) / total_stages) * 100.0


class EventDrivenOrchestrator:
    """
    Pipeline orchestrator that coordinates execution through events.
    Eliminates direct imports between orchestrators and concrete pipeline stages.
    """
    
    def __init__(self, event_bus: EventBus, orchestrator_name: str = None):
        self.event_bus = event_bus
        self.orchestrator_name = orchestrator_name or self.__class__.__name__
        
        # Execution tracking
        self._active_executions: Dict[str, PipelineExecutionContext] = {}
        self._stage_handlers: Dict[str, Callable] = {}
        
        # Subscribe to events
        self._subscription_ids = subscribe_decorated_handlers(
            self.event_bus, self, self.orchestrator_name
        )
        
        logger.info(f"EventDrivenOrchestrator '{self.orchestrator_name}' initialized")
    
    def register_stage_handler(self, stage_name: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Register a handler function for a specific stage.
        This replaces direct imports of stage implementations.
        
        Args:
            stage_name: Name of the pipeline stage
            handler: Function that processes stage input and returns output
        """
        self._stage_handlers[stage_name] = handler
        logger.info(f"Registered handler for stage '{stage_name}'")
    
    def start_pipeline_execution(self, 
                                pipeline_config: Dict[str, Any],
                                input_data: Dict[str, Any],
                                execution_id: str = None) -> str:
        """
        Start a new pipeline execution.
        
        Args:
            pipeline_config: Pipeline configuration
            input_data: Initial input data
            execution_id: Optional execution ID (generated if not provided)
            
        Returns:
            Execution ID for tracking
        """
        if execution_id is None:
            execution_id = str(uuid4())
        
        # Create execution context
        context = PipelineExecutionContext(execution_id, pipeline_config)
        context.pipeline_data = input_data.copy()
        self._active_executions[execution_id] = context
        
        # Extract stage sequence from config
        stage_sequence = pipeline_config.get('stages', [])
        if not stage_sequence:
            logger.error(f"No stages defined in pipeline config for execution {execution_id}")
            return execution_id
        
        # Publish orchestration started event
        orchestration_data = OrchestrationEventData(
            orchestrator_name=self.orchestrator_name,
            orchestrator_type="EventDriven",
            pipeline_config=pipeline_config,
            execution_id=execution_id,
            stage_sequence=stage_sequence,
            current_stage=None,
            progress_percentage=0.0
        )
        
        orchestration_event = OrchestrationStartedEvent(
            data=orchestration_data,
            source=self.orchestrator_name,
            correlation_id=execution_id
        )
        
        self.event_bus.publish(orchestration_event)
        
        # Start first stage
        if stage_sequence:
            self._start_next_stage(execution_id)
        
        logger.info(f"Started pipeline execution {execution_id} with {len(stage_sequence)} stages")
        return execution_id
    
    def _start_next_stage(self, execution_id: str):
        """Start the next stage in the pipeline"""
        context = self._active_executions.get(execution_id)
        if not context:
            logger.error(f"No execution context found for {execution_id}")
            return
        
        stage_sequence = context.pipeline_config.get('stages', [])
        
        # Find next stage to execute
        next_stage = None
        for stage_name in stage_sequence:
            if stage_name not in context.completed_stages and stage_name not in context.failed_stages:
                next_stage = stage_name
                break
        
        if not next_stage:
            # No more stages - pipeline complete
            self._complete_pipeline_execution(execution_id)
            return
        
        context.current_stage = next_stage
        
        # Publish stage started event
        stage_event = create_stage_started_event(
            stage_name=next_stage,
            stage_type="pipeline_stage",
            source=self.orchestrator_name,
            correlation_id=execution_id,
            input_data=context.pipeline_data.copy()
        )
        
        self.event_bus.publish(stage_event)
        
        # Execute stage if handler is registered
        if next_stage in self._stage_handlers:
            self._execute_stage_handler(execution_id, next_stage)
        else:
            logger.warning(f"No handler registered for stage '{next_stage}' in execution {execution_id}")
    
    def _execute_stage_handler(self, execution_id: str, stage_name: str):
        """Execute a stage handler function"""
        context = self._active_executions.get(execution_id)
        if not context:
            return
        
        handler = self._stage_handlers.get(stage_name)
        if not handler:
            logger.warning(f"No handler found for stage '{stage_name}', skipping")
            # Treat as successful completion with no output
            stage_data = StageEventData(
                stage_name=stage_name,
                stage_type="pipeline_stage",
                input_data=context.pipeline_data,
                output_data={}
            )
            
            completed_event = StageCompletedEvent(
                data=stage_data,
                source=self.orchestrator_name,
                correlation_id=execution_id
            )
            
            self.event_bus.publish(completed_event)
            return
        
        try:
            # Execute stage handler with current pipeline data
            stage_output = handler(context.pipeline_data)
            
            # Update pipeline data with stage output
            if isinstance(stage_output, dict):
                context.pipeline_data.update(stage_output)
                context.stage_results[stage_name] = stage_output
            
            # Publish stage completed event
            stage_data = StageEventData(
                stage_name=stage_name,
                stage_type="pipeline_stage",
                input_data=context.pipeline_data,
                output_data=stage_output if isinstance(stage_output, dict) else {},
                execution_time_ms=None  # Could add timing here
            )
            
            completed_event = StageCompletedEvent(
                data=stage_data,
                source=self.orchestrator_name,
                correlation_id=execution_id
            )
            
            self.event_bus.publish(completed_event)
            
        except Exception as e:
            logger.error(f"Stage '{stage_name}' failed in execution {execution_id}: {e}")
            
            # Publish stage failed event
            stage_data = StageEventData(
                stage_name=stage_name,
                stage_type="pipeline_stage",
                input_data=context.pipeline_data,
                error_message=str(e)
            )
            
            failed_event = StageFailedEvent(
                data=stage_data,
                source=self.orchestrator_name,
                correlation_id=execution_id
            )
            
            self.event_bus.publish(failed_event)
    
    def _complete_pipeline_execution(self, execution_id: str):
        """Complete a pipeline execution"""
        context = self._active_executions.get(execution_id)
        if not context:
            return
        
        stage_sequence = context.pipeline_config.get('stages', [])
        
        # Create completion event
        orchestration_data = OrchestrationEventData(
            orchestrator_name=self.orchestrator_name,
            orchestrator_type="EventDriven",
            pipeline_config=context.pipeline_config,
            execution_id=execution_id,
            stage_sequence=stage_sequence,
            current_stage=None,
            progress_percentage=100.0
        )
        
        if context.has_failures():
            # Pipeline failed
            orchestration_data.error_details = {
                'failed_stages': context.failed_stages,
                'error_message': f"Pipeline failed with {len(context.failed_stages)} failed stages"
            }
            
            failed_event = OrchestrationFailedEvent(
                data=orchestration_data,
                source=self.orchestrator_name,
                correlation_id=execution_id
            )
            
            self.event_bus.publish(failed_event)
            logger.error(f"Pipeline execution {execution_id} failed")
            
        else:
            # Pipeline completed successfully
            completed_event = OrchestrationCompletedEvent(
                data=orchestration_data,
                source=self.orchestrator_name,
                correlation_id=execution_id
            )
            
            self.event_bus.publish(completed_event)
            logger.info(f"Pipeline execution {execution_id} completed successfully")
        
        # Keep execution context for final status retrieval
        # In production, you might want to move to a completed_executions dict
        # For now, just mark as completed but keep in active for testing
        pass
    
    # Event handlers
    
    @event_handler(EventType.STAGE_COMPLETED, priority=10)
    def handle_stage_completed(self, event: BaseEvent):
        """Handle stage completion events"""
        if not hasattr(event, 'data') or not event.data:
            return
        
        execution_id = event.correlation_id
        stage_name = event.data.stage_name
        
        context = self._active_executions.get(execution_id)
        if not context:
            return
        
        # Update execution context
        context.completed_stages.append(stage_name)
        context.current_stage = None
        
        logger.info(f"Stage '{stage_name}' completed in execution {execution_id}")
        
        # Start next stage
        self._start_next_stage(execution_id)
    
    @event_handler(EventType.STAGE_FAILED, priority=10)
    def handle_stage_failed(self, event: BaseEvent):
        """Handle stage failure events"""
        if not hasattr(event, 'data') or not event.data:
            return
        
        execution_id = event.correlation_id
        stage_name = event.data.stage_name
        error_message = getattr(event.data, 'error_message', 'Unknown error')
        
        context = self._active_executions.get(execution_id)
        if not context:
            return
        
        # Update execution context
        context.failed_stages.append(stage_name)
        context.current_stage = None
        context.error_details = {
            'failed_stage': stage_name,
            'error_message': error_message
        }
        
        logger.error(f"Stage '{stage_name}' failed in execution {execution_id}: {error_message}")
        
        # Determine failure handling strategy
        failure_strategy = context.pipeline_config.get('failure_strategy', 'stop_on_failure')
        
        if failure_strategy == 'continue_on_failure':
            # Continue with next stage
            self._start_next_stage(execution_id)
        else:
            # Stop execution and complete pipeline as failed
            self._complete_pipeline_execution(execution_id)
    
    # Management methods
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a pipeline execution"""
        context = self._active_executions.get(execution_id)
        if not context:
            return None
        
        stage_sequence = context.pipeline_config.get('stages', [])
        
        return {
            'execution_id': execution_id,
            'orchestrator_name': self.orchestrator_name,
            'start_time': context.start_time.isoformat(),
            'current_stage': context.current_stage,
            'completed_stages': context.completed_stages.copy(),
            'failed_stages': context.failed_stages.copy(),
            'total_stages': len(stage_sequence),
            'progress_percentage': context.get_progress_percentage(len(stage_sequence)),
            'is_complete': context.is_complete(stage_sequence),
            'has_failures': context.has_failures(),
            'pipeline_config': context.pipeline_config
        }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active pipeline executions"""
        return [
            self.get_execution_status(execution_id) 
            for execution_id in self._active_executions.keys()
        ]
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active pipeline execution"""
        if execution_id not in self._active_executions:
            return False
        
        context = self._active_executions[execution_id]
        
        # Publish orchestration failed event for cancellation
        orchestration_data = OrchestrationEventData(
            orchestrator_name=self.orchestrator_name,
            orchestrator_type="EventDriven",
            pipeline_config=context.pipeline_config,
            execution_id=execution_id,
            stage_sequence=context.pipeline_config.get('stages', []),
            current_stage=context.current_stage,
            error_details={'reason': 'cancelled_by_user'}
        )
        
        failed_event = OrchestrationFailedEvent(
            data=orchestration_data,
            source=self.orchestrator_name,
            correlation_id=execution_id
        )
        
        self.event_bus.publish(failed_event)
        
        # Clean up
        del self._active_executions[execution_id]
        
        logger.info(f"Cancelled pipeline execution {execution_id}")
        return True
    
    def shutdown(self):
        """Shutdown the orchestrator and clean up resources"""
        logger.info(f"Shutting down orchestrator '{self.orchestrator_name}'")
        
        # Unsubscribe from events
        for sub_id in self._subscription_ids:
            self.event_bus.unsubscribe(sub_id)
        
        # Cancel all active executions
        active_ids = list(self._active_executions.keys())
        for execution_id in active_ids:
            self.cancel_execution(execution_id)
        
        # Clear handlers
        self._stage_handlers.clear()