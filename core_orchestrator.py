"""
Core Event-Driven Orchestrator for PDT Analysis System
"""

import asyncio
import logging
# # # from typing import Dict, List, Optional, Any  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timezone  # Module not found  # Module not found  # Module not found

# # # from models import (  # Module not found  # Module not found  # Module not found

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "65O"
__stage_order__ = 7

    Event, EventType, WorkflowDefinition, ProcessingContext,
    SystemHealthMetrics, WorkflowStatus
)
# # # from event_bus import EventBus  # Module not found  # Module not found  # Module not found
# # # from workflow_engine import WorkflowEngine  # Module not found  # Module not found  # Module not found
# # # from compensation_engine import CompensationEngine, register_default_compensation_handlers  # Module not found  # Module not found  # Module not found
# # # from workflow_definitions import WORKFLOW_REGISTRY, get_workflow_definition  # Module not found  # Module not found  # Module not found
# # # from step_handlers import register_default_step_handlers  # Module not found  # Module not found  # Module not found
# Optional enhanced core orchestrator integration
# # # from advanced_loader import get_hyper_advanced_core  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


class EventDrivenOrchestrator:
    """
    Main orchestration engine that coordinates event-driven workflows
    for the PDT analysis system with automatic failure recovery
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.event_bus = EventBus(
            max_workers=self.config.get('event_workers', 10),
            event_retention_hours=self.config.get('event_retention_hours', 24)
        )
        
        self.compensation_engine = CompensationEngine(self.event_bus)
        self.workflow_engine = WorkflowEngine(self.event_bus, self.compensation_engine)
        
        # System state
        self.running = False
        self.health_metrics = SystemHealthMetrics()
        
        # Workflow triggers mapping
        self.event_workflow_mapping: Dict[EventType, List[str]] = {}
        
        # System integration points
        self.ray_integrations: Dict[str, Any] = {}
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize orchestrator components"""
        # Verify paths and log which orchestrator modules are active/newest
        try:
# # #             from path_verification import verify_orchestration_paths  # Module not found  # Module not found  # Module not found
            verify_orchestration_paths(strict=False)
        except Exception as e:
            logger.debug(f"Orchestration path verification skipped or failed: {e}")
        # Register default handlers
        register_default_compensation_handlers(self.compensation_engine)
        register_default_step_handlers(self.workflow_engine, self)
        
        # Register workflows
        for workflow_id, workflow_factory in WORKFLOW_REGISTRY.items():
            workflow_def = workflow_factory()
            self.workflow_engine.register_workflow(workflow_def)
            
            # Map triggers to workflows
            for trigger in workflow_def.triggers:
                if trigger not in self.event_workflow_mapping:
                    self.event_workflow_mapping[trigger] = []
                self.event_workflow_mapping[trigger].append(workflow_id)
        
        # Subscribe to trigger events
        for event_type in self.event_workflow_mapping.keys():
            self.event_bus.subscribe(event_type, self._handle_workflow_trigger)
        
        # Subscribe to system events
        self.event_bus.subscribe(EventType.SYSTEM_ERROR, self._handle_system_error)
        self.event_bus.subscribe(EventType.WORKFLOW_FAILED, self._handle_workflow_failure)
        
        logger.info("Event-driven orchestrator initialized")
    
    async def start(self):
        """Start the orchestration system"""
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        self.running = True
        
        # Start core components
        await self.event_bus.start()
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        # Start system monitoring workflow
        await self.trigger_workflow("system_monitoring", {})
        
        logger.info("Event-driven orchestrator started")
    
    async def stop(self):
        """Stop the orchestration system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        await self.event_bus.stop()
        
        logger.info("Event-driven orchestrator stopped")
    
    async def trigger_workflow(self, workflow_id: str, context: Dict[str, Any],
                             correlation_id: str = None) -> str:
        """
        Manually trigger a workflow execution
        
        Args:
            workflow_id: ID of workflow to execute
            context: Workflow context data
            correlation_id: Optional correlation ID
            
        Returns:
            Execution ID
        """
        execution = await self.workflow_engine.start_workflow(
            workflow_id, context, correlation_id
        )
        
        logger.info(f"Triggered workflow {workflow_id} with execution ID {execution.id}")
        return execution.id
    
    async def process_document(self, document_uri: str, 
                             processing_context: ProcessingContext) -> str:
        """
        Process a document through the PDT analysis pipeline
        
        Args:
            document_uri: URI of document to process
            processing_context: Processing context and options
            
        Returns:
            Correlation ID for tracking
        """
        # Publish document upload event
        event = Event(
            type=EventType.DOCUMENT_UPLOADED,
            source="orchestrator",
            data={
                'document_uri': document_uri,
                'document_id': processing_context.document_id,
                'document_type': processing_context.document_type,
                'processing_options': processing_context.processing_options,
                'municipal_context': processing_context.municipal_context,
                'priority': processing_context.priority
            },
            correlation_id=f"doc_{processing_context.document_id}_{int(datetime.now().timestamp())}"
        )
        
        await self.event_bus.publish(event)
        
        logger.info(f"Started document processing for {document_uri}")
        return event.correlation_id
    
    async def _handle_workflow_trigger(self, event: Event):
        """Handle events that trigger workflows"""
        workflow_ids = self.event_workflow_mapping.get(event.type, [])
        
        for workflow_id in workflow_ids:
            try:
# # #                 # Build workflow context from event  # Module not found  # Module not found  # Module not found
                context = {
                    'trigger_event': {
                        'id': event.id,
                        'type': event.type.value,
                        'data': event.data,
                        'timestamp': event.timestamp
                    },
                    **event.data  # Include event data in context
                }
                
                await self.workflow_engine.start_workflow(
                    workflow_id, context, event.correlation_id
                )
                
# # #                 logger.info(f"Triggered workflow {workflow_id} from event {event.type.value}")  # Module not found  # Module not found  # Module not found
                
            except Exception as e:
                logger.error(f"Failed to trigger workflow {workflow_id}: {e}")
                
                # Publish system error
                await self.event_bus.publish(Event(
                    type=EventType.SYSTEM_ERROR,
                    source="orchestrator",
                    data={
                        'error': str(e),
                        'workflow_id': workflow_id,
                        'trigger_event_id': event.id
                    },
                    correlation_id=event.correlation_id
                ))
    
    async def _handle_system_error(self, event: Event):
        """Handle system errors with automatic recovery"""
        logger.warning(f"System error detected: {event.data}")
        
        # Update health metrics
        self.health_metrics.error_rates['system_errors'] = \
            self.health_metrics.error_rates.get('system_errors', 0) + 1
        
        # Check if we should trigger recovery workflow
        error_type = event.data.get('error_type', 'unknown')
        if error_type in ['workflow_failure', 'service_unavailable', 'timeout']:
            # Trigger failure recovery workflow
            try:
                context = {
                    'error_event': event.data,
                    'error_type': error_type,
                    'timestamp': event.timestamp
                }
                
                await self.trigger_workflow("failure_recovery", context, event.correlation_id)
                
            except Exception as e:
                logger.error(f"Failed to trigger recovery workflow: {e}")
    
    async def _handle_workflow_failure(self, event: Event):
        """Handle workflow failures"""
        workflow_id = event.data.get('workflow_id')
        execution_id = event.data.get('execution_id')
        
        logger.error(f"Workflow {workflow_id} failed (execution: {execution_id})")
        
        # Update health metrics
        self.health_metrics.failed_workflows_last_hour += 1
        
        # Check if this is a critical workflow that needs immediate attention
        critical_workflows = ['document_upload', 'pdt_analysis']
        if workflow_id in critical_workflows:
            # Publish high-priority system error
            await self.event_bus.publish(Event(
                type=EventType.SYSTEM_ERROR,
                source="orchestrator",
                data={
                    'error_type': 'critical_workflow_failure',
                    'workflow_id': workflow_id,
                    'execution_id': execution_id,
                    'priority': 'high'
                },
                correlation_id=event.correlation_id
            ))
    
    async def _health_monitoring_loop(self):
        """Background health monitoring"""
        while self.running:
            try:
                await self._update_health_metrics()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _update_health_metrics(self):
        """Update system health metrics"""
        self.health_metrics.timestamp = datetime.now(timezone.utc)
        
        # Update workflow metrics
        active_executions = self.workflow_engine.list_active_executions()
        self.health_metrics.active_workflows = len(active_executions)
        
        # Update event bus metrics
        event_stats = self.event_bus.get_stats()
        self.health_metrics.pending_events = event_stats.get('queue_size', 0)
        
        # Calculate average workflow duration
        if active_executions:
            total_duration = sum(
                exec_info.get('duration_seconds', 0) 
                for exec_info in active_executions
                if exec_info.get('duration_seconds')
            )
            self.health_metrics.average_workflow_duration_seconds = \
                total_duration / len(active_executions)
        
        # Update queue depths
        self.health_metrics.queue_depths = {
            'event_queue': event_stats.get('queue_size', 0),
            'failed_events': event_stats.get('failed_events', 0)
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        return {
            'status': 'healthy' if self.running else 'stopped',
            'timestamp': self.health_metrics.timestamp,
            'active_workflows': self.health_metrics.active_workflows,
            'pending_events': self.health_metrics.pending_events,
            'failed_workflows_last_hour': self.health_metrics.failed_workflows_last_hour,
            'average_workflow_duration': self.health_metrics.average_workflow_duration_seconds,
            'system_load': self.health_metrics.system_load_percentage,
            'memory_usage': self.health_metrics.memory_usage_percentage,
            'queue_depths': self.health_metrics.queue_depths,
            'error_rates': self.health_metrics.error_rates,
            'event_bus_stats': self.event_bus.get_stats()
        }
    
    def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow execution"""
        return self.workflow_engine.get_execution_status(execution_id)
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflow executions"""
        return self.workflow_engine.list_active_executions()
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a workflow execution"""
        return await self.workflow_engine.pause_workflow(execution_id)
    
    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume a paused workflow execution"""
        return await self.workflow_engine.resume_workflow(execution_id)
    
    def register_ray_integration(self, service_name: str, ray_actor_handle):
        """Register Ray-based microservice integration"""
        self.ray_integrations[service_name] = ray_actor_handle
        logger.info(f"Registered Ray integration for {service_name}")
    
    async def call_ray_service(self, service_name: str, method_name: str, *args, **kwargs):
        """Call a Ray-based microservice method"""
        if service_name not in self.ray_integrations:
            raise ValueError(f"Ray service not registered: {service_name}")
        
        ray_actor = self.ray_integrations[service_name]
        method = getattr(ray_actor, method_name)
        
        # Call Ray method (will return a future)
        result = await method.remote(*args, **kwargs)
        return result
    
    async def batch_process_documents(self, document_contexts: List[ProcessingContext]) -> str:
        """Process multiple documents in batch"""
        batch_context = {
            'documents': [ctx.__dict__ for ctx in document_contexts],
            'batch_size': len(document_contexts),
            'batch_id': f"batch_{int(datetime.now().timestamp())}"
        }
        
        return await self.trigger_workflow("batch_processing", batch_context)


# Convenience class aliases for backward compatibility
PDTAnalysisWorkflow = lambda: get_workflow_definition("pdt_analysis")
DocumentUploadWorkflow = lambda: get_workflow_definition("document_upload")


def get_preferred_core_orchestrator(config: Dict[str, Any] = None):
    """
    Factory to obtain the preferred core orchestrator.
    If FARFAN_USE_ENHANCED_CORE=1 and enhanced module is available, returns
    HyperAdvancedOrchestrator; otherwise returns EventDrivenOrchestrator.
    """
    advanced = get_hyper_advanced_core(config=config)
    if advanced is not None:
# # #         logger.info("Using HyperAdvancedOrchestrator from advanced modules")  # Module not found  # Module not found  # Module not found
        return advanced
    return EventDrivenOrchestrator(config=config)