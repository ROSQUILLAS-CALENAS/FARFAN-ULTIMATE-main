"""
Tests for the event-driven orchestrator
"""

import pytest
import time
from unittest.mock import Mock, patch

from egw_query_expansion.core.event_bus import SynchronousEventBus
from egw_query_expansion.core.event_driven_orchestrator import (
    EventDrivenOrchestrator, OrchestrationState, StageConfiguration, ExecutionPlan
)
from egw_query_expansion.core.event_schemas import (
    PipelineStage, ValidationOutcome, ValidationResult, ValidationCompletedEvent,
    ValidationPayload, ErrorEvent, OrchestratorCommandEvent
)


class TestEventDrivenOrchestrator:
    """Test cases for EventDrivenOrchestrator"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = SynchronousEventBus()
        self.orchestrator = EventDrivenOrchestrator(self.event_bus)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        assert self.orchestrator.state == OrchestrationState.IDLE
        assert self.orchestrator.orchestrator_id == "event_driven_orchestrator"
        assert self.orchestrator.event_bus == self.event_bus
        assert len(self.orchestrator.completed_stages) == 0
        assert len(self.orchestrator.failed_stages) == 0
    
    def test_handler_registration(self):
        """Test that orchestrator registers as event handler"""
        handlers = self.event_bus.get_subscribers("validation.completed")
        assert self.orchestrator.handler_id in handlers
        
        handlers = self.event_bus.get_subscribers("pipeline.error")
        assert self.orchestrator.handler_id in handlers
    
    def test_can_handle_events(self):
        """Test event handling capability checking"""
        validation_event = ValidationCompletedEvent(
            result=ValidationResult(
                validator_id="test",
                outcome=ValidationOutcome.PASSED,
                confidence_score=0.9
            )
        )
        
        error_event = ErrorEvent(
            error_type="test_error",
            error_message="Test error"
        )
        
        command_event = OrchestratorCommandEvent(
            command="pause"
        )
        
        assert self.orchestrator.can_handle(validation_event)
        assert self.orchestrator.can_handle(error_event)
        assert self.orchestrator.can_handle(command_event)
    
    def test_default_execution_plan_creation(self):
        """Test creation of default execution plan"""
        plan = self.orchestrator._create_default_execution_plan()
        
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.stages) > 0
        assert len(plan.execution_order) > 0
        assert PipelineStage.INGESTION in plan.execution_order
        assert PipelineStage.ANALYSIS in plan.execution_order
        
        # Check dependencies are properly set
        analysis_config = None
        for config in plan.stages:
            if config.stage == PipelineStage.ANALYSIS:
                analysis_config = config
                break
        
        assert analysis_config is not None
        assert PipelineStage.INGESTION in analysis_config.dependencies
    
    def test_stage_dependency_checking(self):
        """Test stage dependency satisfaction checking"""
        self.orchestrator.completed_stages = [PipelineStage.INGESTION]
        
        # Should be satisfied
        deps = [PipelineStage.INGESTION]
        assert self.orchestrator._dependencies_satisfied(deps)
        
        # Should not be satisfied
        deps = [PipelineStage.INGESTION, PipelineStage.ANALYSIS]
        assert not self.orchestrator._dependencies_satisfied(deps)
    
    def test_stage_configuration_retrieval(self):
        """Test retrieval of stage configuration"""
        plan = self.orchestrator._create_default_execution_plan()
        self.orchestrator.execution_plan = plan
        
        config = self.orchestrator._get_stage_config(PipelineStage.INGESTION)
        assert config is not None
        assert config.stage == PipelineStage.INGESTION
        
        # Non-existent stage
        custom_stage = PipelineStage.SYNTHESIS  # Assuming this exists in plan
        config = self.orchestrator._get_stage_config(custom_stage)
        assert config is not None or custom_stage not in [c.stage for c in plan.stages]
    
    @patch('time.sleep')  # Speed up tests by mocking sleep
    def test_simple_pipeline_execution(self, mock_sleep):
        """Test simple pipeline execution"""
        # Create minimal execution plan
        stages = [
            StageConfiguration(
                stage=PipelineStage.INGESTION,
                required_validations=[]
            ),
            StageConfiguration(
                stage=PipelineStage.ANALYSIS,
                required_validations=[],
                dependencies=[PipelineStage.INGESTION]
            )
        ]
        
        plan = ExecutionPlan(
            stages=stages,
            execution_order=[PipelineStage.INGESTION, PipelineStage.ANALYSIS],
            validation_requirements={}
        )
        
        result = self.orchestrator.execute_pipeline(
            query="test query",
            execution_plan=plan
        )
        
        assert result["success"]
        assert len(result["completed_stages"]) == 2
        assert "ingestion" in result["completed_stages"]
        assert "analysis" in result["completed_stages"]
        assert len(result["failed_stages"]) == 0
    
    def test_validation_request_creation(self):
        """Test creation of validation requests"""
        # Setup stage results
        self.orchestrator.stage_results[PipelineStage.INGESTION] = {"test": "data"}
        self.orchestrator.current_context = Mock()
        
        # Create execution plan with validations
        plan = ExecutionPlan(
            validation_requirements={
                PipelineStage.INGESTION: ["normative_compliance"]
            }
        )
        self.orchestrator.execution_plan = plan
        
        # Mock successful event publishing
        with patch.object(self.event_bus, 'publish') as mock_publish:
            mock_publish.return_value = Mock(success=True)
            
            result = self.orchestrator._request_stage_validations(PipelineStage.INGESTION)
            
            assert result
            assert mock_publish.called
            assert len(self.orchestrator.pending_validations) > 0
    
    def test_validation_completion_handling(self):
        """Test handling of validation completion events"""
        # Setup pending validation
        payload = ValidationPayload(
            validator_id="test_validator",
            validation_type="test",
            data_to_validate={}
        )
        self.orchestrator.pending_validations["test_id"] = payload
        
        # Create validation result
        result = ValidationResult(
            validator_id="test_validator",
            outcome=ValidationOutcome.PASSED,
            confidence_score=0.9
        )
        
        event = ValidationCompletedEvent(result)
        
        # Handle the event
        self.orchestrator._handle_validation_completed(event)
        
        # Check that result was recorded
        assert len(self.orchestrator.validation_results) > 0
    
    def test_error_event_handling(self):
        """Test handling of error events"""
        initial_error_count = self.orchestrator.error_count
        
        error_event = ErrorEvent(
            error_type="test_error",
            error_message="Test error message"
        )
        
        self.orchestrator._handle_pipeline_error(error_event)
        
        assert self.orchestrator.error_count == initial_error_count + 1
    
    def test_orchestrator_command_handling(self):
        """Test handling of orchestrator commands"""
        # Test pause command
        command_event = OrchestratorCommandEvent(command="pause")
        self.orchestrator._handle_orchestrator_command(command_event)
        
        # Test abort command
        command_event = OrchestratorCommandEvent(command="abort")
        self.orchestrator._handle_orchestrator_command(command_event)
        
        assert self.orchestrator.state == OrchestrationState.FAILED
    
    def test_stage_performance_recording(self):
        """Test stage performance metric recording"""
        stage = PipelineStage.INGESTION
        start_time = time.time()
        self.orchestrator.stage_start_times[stage] = start_time
        
        # Mock event publishing
        with patch.object(self.event_bus, 'publish') as mock_publish:
            mock_publish.return_value = Mock(success=True)
            
            self.orchestrator._record_stage_performance(stage)
            
            assert stage in self.orchestrator.performance_metrics
            assert "processing_time_seconds" in self.orchestrator.performance_metrics[stage]
            assert mock_publish.called
    
    def test_error_recording_and_publishing(self):
        """Test error recording and event publishing"""
        with patch.object(self.event_bus, 'publish') as mock_publish:
            mock_publish.return_value = Mock(success=True)
            
            self.orchestrator._record_error("Test error message")
            
            assert mock_publish.called
            call_args = mock_publish.call_args[0][0]
            assert call_args.event_type == "pipeline.error"
            assert call_args.error_message == "Test error message"
    
    def test_orchestration_statistics(self):
        """Test orchestration statistics retrieval"""
        # Set some state
        self.orchestrator.state = OrchestrationState.RUNNING
        self.orchestrator.completed_stages = [PipelineStage.INGESTION]
        self.orchestrator.error_count = 2
        
        stats = self.orchestrator.get_orchestration_stats()
        
        assert stats["current_state"] == "running"
        assert stats["completed_stages"] == 1
        assert stats["failed_stages"] == 0
        assert stats["error_count"] == 2
    
    def test_stage_failure_handling(self):
        """Test handling of stage failures"""
        stage = PipelineStage.ANALYSIS
        
        with patch.object(self.orchestrator, '_process_stage') as mock_process:
            mock_process.return_value = None  # Simulate failure
            
            with patch.object(self.event_bus, 'publish') as mock_publish:
                mock_publish.return_value = Mock(success=True)
                
                result = self.orchestrator._execute_stage(stage)
                
                assert not result
                assert stage in self.orchestrator.failed_stages
                assert mock_publish.called
    
    @patch('time.sleep')
    def test_validation_timeout(self, mock_sleep):
        """Test validation timeout handling"""
        # Create stage config with required validation
        stage_config = StageConfiguration(
            stage=PipelineStage.INGESTION,
            required_validations=["test_validation"]
        )
        
        # Mock the config retrieval
        with patch.object(self.orchestrator, '_get_stage_config', return_value=stage_config):
            # Create pending validation that won't complete
            payload = ValidationPayload(
                validator_id="test_validator",
                validation_type="test_validation",
                data_to_validate={}
            )
            self.orchestrator.pending_validations["test_id"] = payload
            
            # This should timeout
            result = self.orchestrator._wait_for_validations(PipelineStage.INGESTION)
            
            assert not result
            assert self.orchestrator.state == OrchestrationState.RUNNING


class TestStageConfiguration:
    """Test cases for StageConfiguration"""
    
    def test_configuration_creation(self):
        """Test stage configuration creation"""
        config = StageConfiguration(
            stage=PipelineStage.ANALYSIS,
            required_validations=["validation1", "validation2"],
            timeout_seconds=600,
            dependencies=[PipelineStage.INGESTION]
        )
        
        assert config.stage == PipelineStage.ANALYSIS
        assert len(config.required_validations) == 2
        assert config.timeout_seconds == 600
        assert PipelineStage.INGESTION in config.dependencies
    
    def test_default_configuration_values(self):
        """Test default configuration values"""
        config = StageConfiguration(stage=PipelineStage.SYNTHESIS)
        
        assert len(config.required_validations) == 0
        assert len(config.optional_validations) == 0
        assert config.timeout_seconds == 300
        assert config.retry_count == 3
        assert len(config.dependencies) == 0


class TestExecutionPlan:
    """Test cases for ExecutionPlan"""
    
    def test_execution_plan_creation(self):
        """Test execution plan creation"""
        stages = [
            StageConfiguration(stage=PipelineStage.INGESTION),
            StageConfiguration(stage=PipelineStage.ANALYSIS)
        ]
        
        execution_order = [PipelineStage.INGESTION, PipelineStage.ANALYSIS]
        validation_requirements = {
            PipelineStage.INGESTION: ["validation1"]
        }
        
        plan = ExecutionPlan(
            stages=stages,
            execution_order=execution_order,
            validation_requirements=validation_requirements
        )
        
        assert len(plan.stages) == 2
        assert len(plan.execution_order) == 2
        assert PipelineStage.INGESTION in plan.validation_requirements
    
    def test_empty_execution_plan(self):
        """Test empty execution plan"""
        plan = ExecutionPlan()
        
        assert len(plan.stages) == 0
        assert len(plan.execution_order) == 0
        assert len(plan.validation_requirements) == 0
        assert len(plan.parallel_stages) == 0


if __name__ == "__main__":
    pytest.main([__file__])