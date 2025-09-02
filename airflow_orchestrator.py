"""
Apache Airflow-based orchestration engine with dynamic DAG generation
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path

import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.kafka.operators.produce import ProduceToTopicOperator
from airflow.providers.celery.operators.celery import CeleryOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from process_inventory import ProcessInventoryManager
from service_discovery import ServiceDiscoveryManager
from circuit_breaker import CircuitBreaker
from models import WorkflowDefinition, ProcessDefinition
# Optional advanced orchestrator integration
from advanced_loader import get_hyper_airflow_orchestrator, get_advanced_dag_generator

logger = logging.getLogger(__name__)


class DynamicDAGGenerator:
    """
    Generates Apache Airflow DAGs dynamically based on process definitions
    stored in a centralized registry
    """
    
    def __init__(self, 
                 process_inventory: ProcessInventoryManager,
                 service_discovery: ServiceDiscoveryManager,
                 dag_storage_path: str = "/opt/airflow/dags"):
        self.process_inventory = process_inventory
        self.service_discovery = service_discovery
        self.dag_storage_path = Path(dag_storage_path)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        # DAG generation templates
        self.operator_templates = {
            'kubernetes': self._create_kubernetes_operator,
            'python': self._create_python_operator,
            'bash': self._create_bash_operator,
            'kafka': self._create_kafka_operator,
            'celery': self._create_celery_operator
        }
        
    def generate_dag_from_process(self, process_def: ProcessDefinition) -> DAG:
        """
        Generate a DAG from a process definition
        
        Args:
            process_def: Process definition containing workflow steps
            
        Returns:
            Generated Airflow DAG
        """
        default_args = {
            'owner': process_def.metadata.get('owner', 'system'),
            'depends_on_past': False,
            'start_date': days_ago(1),
            'email_on_failure': True,
            'email_on_retry': False,
            'retries': process_def.metadata.get('retries', 2),
            'retry_delay': timedelta(minutes=process_def.metadata.get('retry_delay_minutes', 5)),
            'max_active_runs': process_def.metadata.get('max_active_runs', 1),
            'catchup': False
        }
        
        dag_id = f"dynamic_{process_def.id}_{process_def.version}"
        
        dag = DAG(
            dag_id=dag_id,
            default_args=default_args,
            description=process_def.description,
            schedule_interval=process_def.schedule or None,
            catchup=False,
            tags=['dynamic', 'orchestrated'] + process_def.tags
        )
        
        # Track task objects for dependency setup
        task_map = {}
        
        # Create task groups for parallel execution
        with dag:
            for step_group in process_def.execution_plan:
                if len(step_group.steps) > 1:
                    # Create task group for parallel steps
                    with TaskGroup(f"group_{step_group.id}") as tg:
                        group_tasks = []
                        for step in step_group.steps:
                            task = self._create_task_from_step(step, dag)
                            group_tasks.append(task)
                            task_map[step.id] = task
                        
                        # Set intra-group dependencies if specified
                        for step in step_group.steps:
                            if step.depends_on:
                                for dependency in step.depends_on:
                                    if dependency in task_map:
                                        task_map[dependency] >> task_map[step.id]
                else:
                    # Single task
                    step = step_group.steps[0]
                    task = self._create_task_from_step(step, dag)
                    task_map[step.id] = task
        
        # Set inter-group dependencies
        self._setup_task_dependencies(process_def, task_map)
        
        return dag
    
    def _create_task_from_step(self, step, dag) -> airflow.models.BaseOperator:
        """Create an Airflow operator from a workflow step"""
        # Validate that step has required fields
        if not hasattr(step, 'id'):
            raise ValueError(f"Step missing required 'id' field: {step}")
        if not hasattr(step, 'name'):
            raise ValueError(f"Step missing required 'name' field: {step}")
        if not hasattr(step, 'config'):
            raise ValueError(f"Step missing required 'config' field: {step}")
        if not hasattr(step, 'operator_type'):
            raise ValueError(f"Step missing required 'operator_type' field: {step}")
            
        operator_type = step.operator_type
        
        if operator_type not in self.operator_templates:
            logger.warning(f"Unknown operator type {operator_type}, using python operator")
            operator_type = 'python'
            
        return self.operator_templates[operator_type](step, dag)
    
    def _create_kubernetes_operator(self, step, dag) -> KubernetesPodOperator:
        """Create Kubernetes pod operator"""
        return KubernetesPodOperator(
            task_id=step.id,
            name=step.name,
            namespace=step.config.get('namespace', 'default'),
            image=step.config['image'],
            cmds=step.config.get('commands', []),
            arguments=step.config.get('args', []),
            labels=step.config.get('labels', {}),
            env_vars=step.config.get('env_vars', {}),
            get_logs=True,
            delete_option_kwargs={"propagationPolicy": "Background"},
            dag=dag,
            retries=step.config.get('retries', 2),
            is_delete_operator_pod=step.config.get('cleanup_pod', True)
        )
    
    def _create_python_operator(self, step, dag) -> PythonOperator:
        """Create Python callable operator"""
        
        def execute_step(**context):
            """Execute the step with circuit breaker protection"""
            return self.circuit_breaker.call(
                self._execute_python_step,
                step,
                context
            )
        
        return PythonOperator(
            task_id=step.id,
            python_callable=execute_step,
            dag=dag,
            provide_context=True,
            op_kwargs=step.config.get('kwargs', {})
        )
    
    def _execute_python_step(self, step, context):
        """Execute a Python step with service discovery"""
        # Get service endpoints from discovery
        services = self.service_discovery.get_available_services()
        
        # Import and execute the specified function
        module_path = step.config['module']
        function_name = step.config['function']
        
        # Dynamic import
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, function_name)
        
        # Execute with context and services
        return func(context=context, services=services, **step.config.get('kwargs', {}))
    
    def _create_bash_operator(self, step, dag) -> BashOperator:
        """Create Bash operator"""
        return BashOperator(
            task_id=step.id,
            bash_command=step.config['command'],
            dag=dag,
            env=step.config.get('env', {}),
            cwd=step.config.get('cwd')
        )
    
    def _create_kafka_operator(self, step, dag) -> ProduceToTopicOperator:
        """Create Kafka producer operator"""
        return ProduceToTopicOperator(
            task_id=step.id,
            topic=step.config['topic'],
            producer_config=step.config.get('producer_config', {}),
            kafka_config_id=step.config.get('kafka_conn_id', 'kafka_default'),
            dag=dag
        )
    
    def _create_celery_operator(self, step, dag) -> CeleryOperator:
        """Create Celery task operator"""
        return CeleryOperator(
            task_id=step.id,
            task_name=step.config['task_name'],
            args=step.config.get('args', []),
            kwargs=step.config.get('kwargs', {}),
            queue=step.config.get('queue', 'default'),
            dag=dag
        )
    
    def _setup_task_dependencies(self, process_def: ProcessDefinition, task_map: Dict[str, Any]):
        """Setup task dependencies based on process definition"""
        for step_group in process_def.execution_plan:
            for step in step_group.steps:
                if step.depends_on:
                    for dependency in step.depends_on:
                        if dependency in task_map:
                            task_map[dependency] >> task_map[step.id]
    
    def generate_all_dags(self) -> List[DAG]:
        """Generate DAGs for all process definitions in the registry"""
        dags = []
        
        try:
            process_definitions = self.process_inventory.get_all_processes()
            
            for process_def in process_definitions:
                try:
                    dag = self.generate_dag_from_process(process_def)
                    dags.append(dag)
                    
                    # Save DAG to file system
                    self._save_dag_to_file(dag, process_def)
                    
                    logger.info(f"Generated DAG for process {process_def.id}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate DAG for process {process_def.id}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to retrieve process definitions: {e}")
        
        return dags
    
    def _save_dag_to_file(self, dag: DAG, process_def: ProcessDefinition):
        """Save generated DAG to file system for Airflow to discover"""
        dag_file_path = self.dag_storage_path / f"{dag.dag_id}.py"
        
        dag_code = self._generate_dag_file_content(dag, process_def)
        
        with open(dag_file_path, 'w') as f:
            f.write(dag_code)
        
        logger.info(f"Saved DAG file: {dag_file_path}")
    
    def _generate_dag_file_content(self, dag: DAG, process_def: ProcessDefinition) -> str:
        """Generate the Python code for a DAG file"""
        return f'''
"""
Auto-generated DAG for process: {process_def.id}
Version: {process_def.version}
Generated at: {datetime.now()}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.dates import days_ago

from orchestration.airflow_orchestrator import DynamicDAGGenerator
from orchestration.process_inventory import ProcessInventoryManager
from orchestration.service_discovery import ServiceDiscoveryManager

# Initialize components
process_inventory = ProcessInventoryManager()
service_discovery = ServiceDiscoveryManager()
dag_generator = DynamicDAGGenerator(process_inventory, service_discovery)

# Get process definition
process_def = process_inventory.get_process("{process_def.id}", "{process_def.version}")

# Generate DAG
dag = dag_generator.generate_dag_from_process(process_def)

# Export for Airflow
globals()[dag.dag_id] = dag
'''


class AirflowOrchestrator:
    """
    Main Airflow orchestration controller that manages dynamic DAG generation,
    workflow execution, and integration with Kubernetes and Kafka
    """
    
    def __init__(self, 
                 process_inventory: ProcessInventoryManager,
                 service_discovery: ServiceDiscoveryManager,
                 airflow_config: Dict[str, Any] = None):
        self.process_inventory = process_inventory
        self.service_discovery = service_discovery
        self.config = airflow_config or {}
        
        # Initialize DAG generator (prefer advanced if enabled and available)
        advanced = get_advanced_dag_generator(
            process_inventory=process_inventory,
            service_discovery=service_discovery,
            monitoring_stack=None,
            dag_storage_path=self.config.get('dag_storage_path', '/opt/airflow/dags'),
            enable_optimization=True,
        )
        if advanced is not None:
            self.dag_generator = advanced
            logger.info("Using AdvancedDAGGenerator from advanced modules")
        else:
            self.dag_generator = DynamicDAGGenerator(
                process_inventory=process_inventory,
                service_discovery=service_discovery,
                dag_storage_path=self.config.get('dag_storage_path', '/opt/airflow/dags')
            )
        
        # Workflow triggers registry
        self.event_workflow_triggers = {}
        
    def register_process_definition(self, process_def: ProcessDefinition):
        """Register a new process definition and generate corresponding DAG"""
        # Store in inventory
        self.process_inventory.register_process(process_def)
        
        # Generate DAG
        dag = self.dag_generator.generate_dag_from_process(process_def)
        
        logger.info(f"Registered process {process_def.id} with generated DAG {dag.dag_id}")
        return dag.dag_id
    
    def trigger_workflow(self, 
                        process_id: str, 
                        version: str = None,
                        execution_context: Dict[str, Any] = None) -> str:
        """
        Trigger workflow execution via Airflow
        
        Args:
            process_id: ID of process to execute
            version: Process version (latest if None)
            execution_context: Execution context data
            
        Returns:
            DAG run ID
        """
        from airflow.models import DagBag
        from airflow.api.client.local_client import Client
        
        # Get process definition
        process_def = self.process_inventory.get_process(process_id, version)
        if not process_def:
            raise ValueError(f"Process not found: {process_id}")
        
        dag_id = f"dynamic_{process_def.id}_{process_def.version}"
        
        # Trigger DAG run
        client = Client(None, None)
        dag_run = client.trigger_dag(
            dag_id=dag_id,
            conf=execution_context or {},
            run_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info(f"Triggered workflow {process_id} - DAG run: {dag_run}")
        return dag_run
    
    def setup_event_triggers(self):
        """Setup event-driven workflow triggers using Kafka"""
        # This would typically be done through Kafka consumers
        # that trigger Airflow DAGs when certain events occur
        pass
    
    def get_workflow_status(self, dag_id: str, run_id: str = None) -> Dict[str, Any]:
        """Get status of a workflow execution"""
        from airflow.models import DagRun, TaskInstance
        
        if run_id:
            dag_run = DagRun.find(dag_id=dag_id, run_id=run_id)
        else:
            dag_run = DagRun.find(dag_id=dag_id)[-1]  # Latest run
        
        if not dag_run:
            return {"status": "not_found"}
        
        # Get task statuses
        task_instances = TaskInstance.find(dag_id=dag_id, run_id=dag_run.run_id)
        
        return {
            "dag_id": dag_id,
            "run_id": dag_run.run_id,
            "state": dag_run.state,
            "start_date": dag_run.start_date,
            "end_date": dag_run.end_date,
            "tasks": [
                {
                    "task_id": ti.task_id,
                    "state": ti.state,
                    "start_date": ti.start_date,
                    "end_date": ti.end_date,
                    "duration": ti.duration
                }
                for ti in task_instances
            ]
        }
    
    def regenerate_all_dags(self):
        """Regenerate all DAGs from current process definitions"""
        return self.dag_generator.generate_all_dags()
    
    def setup_dead_letter_queue(self, queue_name: str = "workflow_dlq"):
        """Setup dead letter queue for failed workflow tasks"""
        # This would integrate with the messaging system to handle failed tasks
        pass


def get_preferred_airflow_orchestrator(process_inventory: ProcessInventoryManager,
                                       service_discovery: ServiceDiscoveryManager,
                                       airflow_config: Dict[str, Any] = None):
    """
    Factory to obtain the preferred Airflow orchestrator.
    If FARFAN_USE_ADVANCED_AIRFLOW=1 and advanced module is available, returns
    HyperAirflowOrchestrator; otherwise returns AirflowOrchestrator.
    """
    # Try advanced high-level orchestrator first
    hyper = get_hyper_airflow_orchestrator(
        process_inventory=process_inventory,
        service_discovery=service_discovery,
        monitoring_stack=None,
        config=airflow_config or {},
    )
    if hyper is not None:
        logger.info("Using HyperAirflowOrchestrator from advanced modules")
        return hyper
    return AirflowOrchestrator(process_inventory, service_discovery, airflow_config or {})