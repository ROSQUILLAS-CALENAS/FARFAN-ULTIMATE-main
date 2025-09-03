"""
Workflow Engine for managing workflow definitions and execution

Note: With the introduction of the Confluent Orchestrator (egw_query_expansion.core.confluent_orchestrator),
several capabilities of this engine overlap. This module remains supported as a
compatibility layer for existing EventDrivenOrchestrator/core_orchestrator integrations.
Use orchestration_redundancy.assess_redundancy() for a detailed overlap report.
"""

import asyncio
import logging
import warnings
# # # from typing import Dict, List, Optional, Any, Set  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timezone  # Module not found  # Module not found  # Module not found

# # # from models import (  # Module not found  # Module not found  # Module not found

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "64O"
__stage_order__ = 7

    WorkflowDefinition, WorkflowExecution, WorkflowStep, WorkflowStatus, 
    StepStatus, EventType, Event
)
# # # from compensation_engine import CompensationEngine  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)

# Emit a gentle deprecation notice (does not break runtime)
warnings.warn(
    "workflow_engine: Consider migrating DAG-like workflows to the Confluent Orchestrator. "
    "This engine remains for backward compatibility.",
    category=DeprecationWarning,
    stacklevel=1,
)


class WorkflowEngine:
    """
    Core workflow execution engine with support for:
    - Parallel and sequential step execution
    - Dependency resolution
    - Timeout handling
    - Automatic retry logic
    - Failure recovery
    """
    
    def __init__(self, event_bus=None, compensation_engine: CompensationEngine = None):
        self.event_bus = event_bus
        self.compensation_engine = compensation_engine
        
        # Workflow registry
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        
        # Step handlers registry
        self.step_handlers: Dict[str, callable] = {}
        
        # Execution control
        self._running = False
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition"""
        self.workflow_definitions[workflow.id] = workflow
        logger.info(f"Registered workflow: {workflow.name} (v{workflow.version})")
    
    def register_step_handler(self, handler_name: str, handler: callable):
        """Register a step handler function"""
        self.step_handlers[handler_name] = handler
        logger.info(f"Registered step handler: {handler_name}")
    
    async def start_workflow(self, workflow_id: str, context: Dict[str, Any],
                           correlation_id: str = None) -> WorkflowExecution:
        """
        Start a new workflow execution
        
        Args:
            workflow_id: ID of workflow definition to execute
            context: Initial context data
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            WorkflowExecution instance
        """
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow_def = self.workflow_definitions[workflow_id]
        
        # Create execution instance
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            correlation_id=correlation_id or f"wf_{workflow_id}_{int(datetime.now().timestamp())}",
            context=context.copy(),
            started_at=datetime.now(timezone.utc),
            status=WorkflowStatus.RUNNING
        )
        
        # Initialize step executions
        for step in workflow_def.steps:
            execution.step_executions[step.id] = WorkflowStep(
                id=step.id,
                name=step.name,
                handler=step.handler,
                dependencies=step.dependencies.copy(),
                compensation_handler=step.compensation_handler,
                timeout_seconds=step.timeout_seconds,
                retry_attempts=step.retry_attempts,
                retry_delay_seconds=step.retry_delay_seconds,
                parameters=step.parameters.copy()
            )
        
        # Store active execution
        self.active_executions[execution.id] = execution
        
        # Publish workflow started event
        if self.event_bus:
            await self.event_bus.publish(Event(
                type=EventType.WORKFLOW_STARTED,
                source="workflow_engine",
                data={
                    'workflow_id': workflow_id,
                    'execution_id': execution.id,
                    'total_steps': len(workflow_def.steps)
                },
                correlation_id=execution.correlation_id
            ))
        
        # Start execution task
        execution_task = asyncio.create_task(
            self._execute_workflow(execution, workflow_def)
        )
        self._execution_tasks[execution.id] = execution_task
        
        logger.info(f"Started workflow {workflow_id} with execution ID {execution.id}")
        
        return execution
    
    async def _execute_workflow(self, execution: WorkflowExecution, 
                               workflow_def: WorkflowDefinition):
        """Execute a workflow with proper error handling"""
        try:
            await self._run_workflow_steps(execution, workflow_def)
            
            # Mark as completed if all steps succeeded
            if not execution.failed_steps:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.now(timezone.utc)
                
                if self.event_bus:
                    await self.event_bus.publish(Event(
                        type=EventType.WORKFLOW_COMPLETED,
                        source="workflow_engine",
                        data={
                            'workflow_id': execution.workflow_id,
                            'execution_id': execution.id,
                            'duration_seconds': (execution.completed_at - execution.started_at).total_seconds(),
                            'completed_steps': len(execution.completed_steps)
                        },
                        correlation_id=execution.correlation_id
                    ))
                
                logger.info(f"Workflow {execution.id} completed successfully")
            else:
                await self._handle_workflow_failure(execution, workflow_def)
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            await self._handle_workflow_failure(execution, workflow_def, str(e))
        finally:
            # Clean up execution task
            if execution.id in self._execution_tasks:
                del self._execution_tasks[execution.id]
    
    async def _run_workflow_steps(self, execution: WorkflowExecution,
                                 workflow_def: WorkflowDefinition):
        """Run workflow steps with dependency resolution"""
        
        while execution.current_steps or self._has_ready_steps(execution):
            # Find steps ready to execute
            ready_steps = self._get_ready_steps(execution)
            
            if not ready_steps and execution.current_steps:
                # Wait for running steps to complete
                await asyncio.sleep(0.1)
                continue
            
            if not ready_steps and not execution.current_steps:
                # No more steps to run
                break
            
            # Limit parallel execution
            max_parallel = workflow_def.max_parallel_steps
            available_slots = max_parallel - len(execution.current_steps)
            
            steps_to_start = ready_steps[:available_slots]
            
            # Start ready steps
            for step_id in steps_to_start:
                await self._start_step(execution, step_id)
            
            # Check for completed steps
            await self._check_running_steps(execution)
    
    def _has_ready_steps(self, execution: WorkflowExecution) -> bool:
        """Check if there are steps ready to execute"""
        return bool(self._get_ready_steps(execution))
    
    def _get_ready_steps(self, execution: WorkflowExecution) -> List[str]:
        """Get list of steps ready to execute"""
        ready_steps = []
        
        for step_id, step in execution.step_executions.items():
            if step.status != StepStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_completed = all(
                dep_id in execution.completed_steps 
                for dep_id in step.dependencies
            )
            
            if deps_completed and step_id not in execution.current_steps:
                ready_steps.append(step_id)
        
        return ready_steps
    
    async def _start_step(self, execution: WorkflowExecution, step_id: str):
        """Start execution of a single step"""
        step = execution.step_executions[step_id]
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        
        execution.current_steps.append(step_id)
        
        logger.debug(f"Starting step {step_id} in workflow {execution.id}")
        
        # Create step execution task
        step_task = asyncio.create_task(
            self._execute_step(execution, step)
        )
        
        # Store task for monitoring
        if not hasattr(execution, '_step_tasks'):
            execution._step_tasks = {}
        execution._step_tasks[step_id] = step_task
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep):
        """Execute a single step with retry logic"""
        
        for attempt in range(step.retry_attempts + 1):
            try:
                # Get step handler
                handler = self.step_handlers.get(step.handler)
                if not handler:
                    raise ValueError(f"Step handler not found: {step.handler}")
                
                # Prepare step context
                step_context = {
                    'workflow_context': execution.context.copy(),
                    'step_parameters': step.parameters.copy(),
                    'execution_id': execution.id,
                    'correlation_id': execution.correlation_id,
                    'step_id': step.id
                }
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._call_step_handler(handler, step_context),
                    timeout=step.timeout_seconds
                )
                
                # Step completed successfully
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now(timezone.utc)
                step.result = result
                
                # Update workflow context if step returns context updates
                if isinstance(result, dict) and 'context_updates' in result:
                    execution.context.update(result['context_updates'])
                
                logger.debug(f"Step {step.id} completed successfully")
                return
                
            except asyncio.TimeoutError:
                error_msg = f"Step {step.id} timed out after {step.timeout_seconds}s"
                logger.error(error_msg)
                
                if attempt == step.retry_attempts:
                    step.status = StepStatus.FAILED
                    step.error = error_msg
                    return
                
            except Exception as e:
                error_msg = f"Step {step.id} failed: {str(e)}"
                logger.error(error_msg)
                
                if attempt == step.retry_attempts:
                    step.status = StepStatus.FAILED
                    step.error = error_msg
                    return
                
                # Wait before retry
                if attempt < step.retry_attempts:
                    await asyncio.sleep(step.retry_delay_seconds * (2 ** attempt))
    
    async def _call_step_handler(self, handler: callable, context: Dict[str, Any]):
        """Call step handler (async or sync)"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(context)
        else:
            # Run sync handler in thread pool
            return await asyncio.get_event_loop().run_in_executor(
                None, handler, context
            )
    
    async def _check_running_steps(self, execution: WorkflowExecution):
        """Check status of currently running steps"""
        completed_steps = []
        
        for step_id in execution.current_steps.copy():
            step = execution.step_executions[step_id]
            
            if step.status in [StepStatus.COMPLETED, StepStatus.FAILED]:
                completed_steps.append(step_id)
                execution.current_steps.remove(step_id)
                
                if step.status == StepStatus.COMPLETED:
                    execution.completed_steps.append(step_id)
                else:
                    execution.failed_steps.append(step_id)
                
                # Clean up step task
                if hasattr(execution, '_step_tasks') and step_id in execution._step_tasks:
                    del execution._step_tasks[step_id]
    
    async def _handle_workflow_failure(self, execution: WorkflowExecution,
                                     workflow_def: WorkflowDefinition, error: str = None):
        """Handle workflow failure and trigger compensation if needed"""
        execution.status = WorkflowStatus.FAILED
        execution.completed_at = datetime.now(timezone.utc)
        execution.error = error or "One or more steps failed"
        
        logger.error(f"Workflow {execution.id} failed: {execution.error}")
        
        # Publish failure event
        if self.event_bus:
            await self.event_bus.publish(Event(
                type=EventType.WORKFLOW_FAILED,
                source="workflow_engine",
                data={
                    'workflow_id': execution.workflow_id,
                    'execution_id': execution.id,
                    'error': execution.error,
                    'failed_steps': execution.failed_steps.copy(),
                    'completed_steps': execution.completed_steps.copy()
                },
                correlation_id=execution.correlation_id
            ))
        
        # Trigger compensation if needed
        if self.compensation_engine and execution.completed_steps:
            execution.status = WorkflowStatus.COMPENSATING
            
            if self.event_bus:
                await self.event_bus.publish(Event(
                    type=EventType.COMPENSATION_TRIGGERED,
                    source="workflow_engine",
                    data={
                        'workflow_id': execution.workflow_id,
                        'execution_id': execution.id
                    },
                    correlation_id=execution.correlation_id
                ))
            
            compensation_success = await self.compensation_engine.compensate_workflow(execution)
            
            if compensation_success:
                execution.status = WorkflowStatus.COMPENSATED
                logger.info(f"Compensation completed for workflow {execution.id}")
            else:
                logger.error(f"Compensation failed for workflow {execution.id}")
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = WorkflowStatus.PAUSED
        
        # Cancel execution task
        if execution_id in self._execution_tasks:
            self._execution_tasks[execution_id].cancel()
        
        logger.info(f"Paused workflow execution {execution_id}")
        return True
    
    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume a paused workflow"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        if execution.status != WorkflowStatus.PAUSED:
            return False
        
        execution.status = WorkflowStatus.RUNNING
        
        # Restart execution
        workflow_def = self.workflow_definitions[execution.workflow_id]
        execution_task = asyncio.create_task(
            self._execute_workflow(execution, workflow_def)
        )
        self._execution_tasks[execution.id] = execution_task
        
        logger.info(f"Resumed workflow execution {execution_id}")
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed execution status"""
        if execution_id not in self.active_executions:
            return None
        
        execution = self.active_executions[execution_id]
        
        step_status_counts = {}
        for status in StepStatus:
            step_status_counts[status.value] = sum(
                1 for step in execution.step_executions.values()
                if step.status == status
            )
        
        duration = None
        if execution.started_at:
            end_time = execution.completed_at or datetime.now(timezone.utc)
            duration = (end_time - execution.started_at).total_seconds()
        
        return {
            'execution_id': execution_id,
            'workflow_id': execution.workflow_id,
            'status': execution.status.value,
            'started_at': execution.started_at,
            'completed_at': execution.completed_at,
            'duration_seconds': duration,
            'step_status_counts': step_status_counts,
            'completed_steps': len(execution.completed_steps),
            'failed_steps': len(execution.failed_steps),
            'current_steps': execution.current_steps.copy(),
            'error': execution.error
        }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active workflow executions"""
        return [
            self.get_execution_status(execution_id)
            for execution_id in self.active_executions.keys()
        ]