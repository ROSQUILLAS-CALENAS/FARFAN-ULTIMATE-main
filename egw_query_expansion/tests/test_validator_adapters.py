"""
Tests for validator adapters
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from egw_query_expansion.core.event_bus import SynchronousEventBus
from egw_query_expansion.core.validator_adapters import (
    ValidatorAdapter, ConstraintValidatorAdapter, RubricValidatorAdapter,
    NormativeValidatorAdapter, register_validator_adapter, get_validator_adapter,
    initialize_default_adapters, cleanup_validator_adapters
)
from egw_query_expansion.core.event_schemas import (
    ValidationRequestedEvent, ValidationCompletedEvent, ValidationPayload,
    ValidationResult, ValidationOutcome, PipelineStage, PipelineContext,
    StageCompletedEvent
)


class TestValidatorAdapter:
    """Test cases for base ValidatorAdapter class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = SynchronousEventBus()
        
        class TestAdapter(ValidatorAdapter):
            def _get_monitored_stages(self):
                return [PipelineStage.INGESTION]
            
            def _perform_validation(self, data, validation_type, validation_rules):
                return (
                    ValidationOutcome.PASSED,
                    0.9,
                    [],
                    [],
                    {"test": "details"}
                )
        
        self.adapter = TestAdapter(
            validator_id="test_validator",
            supported_validation_types=["test_validation"],
            event_bus=self.event_bus
        )
    
    def test_adapter_initialization(self):
        """Test adapter initialization"""
        assert self.adapter.validator_id == "test_validator"
        assert "test_validation" in self.adapter.supported_validation_types
        assert self.adapter.event_bus == self.event_bus
        assert len(self.adapter.validation_history) == 0
    
    def test_handler_properties(self):
        """Test event handler properties"""
        assert self.adapter.handler_id == "test_validator"
    
    def test_can_handle_validation_request(self):
        """Test validation request handling capability"""
        payload = ValidationPayload(
            validator_id="test_validator",
            validation_type="test_validation",
            data_to_validate={}
        )
        
        event = ValidationRequestedEvent(payload)
        
        assert self.adapter.can_handle(event)
    
    def test_can_handle_stage_completion(self):
        """Test stage completion handling capability"""
        context = PipelineContext(query="test")
        event = StageCompletedEvent(
            stage=PipelineStage.INGESTION,
            context=context,
            results={"test": "data"},
            success=True
        )
        
        assert self.adapter.can_handle(event)
        
        # Should not handle non-monitored stages
        event.data['stage'] = PipelineStage.ANALYSIS
        assert not self.adapter.can_handle(event)
    
    def test_handle_validation_request(self):
        """Test handling of validation request events"""
        payload = ValidationPayload(
            validator_id="test_validator",
            validation_type="test_validation",
            data_to_validate={"test": "data"}
        )
        
        event = ValidationRequestedEvent(payload)
        response = self.adapter.handle(event)
        
        assert isinstance(response, ValidationCompletedEvent)
        assert response.result.validator_id == "test_validator"
        assert response.result.outcome == ValidationOutcome.PASSED
        assert response.result.confidence_score == 0.9
        assert len(self.adapter.validation_history) == 1
    
    def test_handle_stage_completion(self):
        """Test handling of stage completion events"""
        context = PipelineContext(query="test")
        event = StageCompletedEvent(
            stage=PipelineStage.INGESTION,
            context=context,
            results={"processed": True},
            success=True
        )
        
        response = self.adapter.handle(event)
        
        assert isinstance(response, ValidationCompletedEvent)
        assert response.result.validator_id == "test_validator"
    
    def test_validation_history_tracking(self):
        """Test validation history tracking"""
        payload = ValidationPayload(
            validator_id="test_validator",
            validation_type="test_validation",
            data_to_validate={}
        )
        
        event = ValidationRequestedEvent(payload)
        
        # Process multiple validations
        for i in range(3):
            self.adapter.handle(event)
        
        assert len(self.adapter.validation_history) == 3
        
        # Test history size limit
        for i in range(1000):
            self.adapter.handle(event)
        
        assert len(self.adapter.validation_history) <= 1000
    
    def test_validation_statistics(self):
        """Test validation statistics generation"""
        # Initially empty stats
        stats = self.adapter.get_validation_stats()
        assert stats["total_validations"] == 0
        
        # Add some validation history
        results = [
            ValidationResult("test", ValidationOutcome.PASSED, 0.9),
            ValidationResult("test", ValidationOutcome.FAILED, 0.3),
            ValidationResult("test", ValidationOutcome.PASSED, 0.8)
        ]
        
        self.adapter.validation_history = results
        stats = self.adapter.get_validation_stats()
        
        assert stats["total_validations"] == 3
        assert stats["passed"] == 2
        assert stats["failed"] == 1
        assert stats["average_confidence"] == pytest.approx(0.67, abs=0.01)


class TestConstraintValidatorAdapter:
    """Test cases for ConstraintValidatorAdapter"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = SynchronousEventBus()
        self.adapter = ConstraintValidatorAdapter(self.event_bus)
    
    def test_constraint_adapter_initialization(self):
        """Test constraint validator adapter initialization"""
        assert self.adapter.validator_id == "constraint_validator"
        assert "dimension_requirements" in self.adapter.supported_validation_types
        assert "conjunctive_conditions" in self.adapter.supported_validation_types
        assert PipelineStage.ORCHESTRATION in self.adapter._get_monitored_stages()
    
    @patch('sys.path')
    def test_lazy_validator_loading(self, mock_path):
        """Test lazy loading of actual validator"""
        # Mock the import to avoid actual file dependencies
        mock_validator = Mock()
        mock_validator.validate_dimension_requirements.return_value = (True, Mock())
        
        with patch('importlib.import_module') as mock_import:
            with patch.dict('sys.modules', {'constraint_validator': Mock(ConstraintValidator=lambda: mock_validator)}):
                instance = self.adapter._get_validator_instance()
                assert instance == mock_validator
    
    def test_dimension_requirements_validation(self):
        """Test dimension requirements validation"""
        # Mock the validator instance
        mock_validator = Mock()
        mock_proof = Mock()
        mock_proof.crc_risk_bound = 0.1
        mock_proof.__dict__ = {"test": "proof"}
        mock_validator.validate_dimension_requirements.return_value = (True, mock_proof)
        
        with patch.object(self.adapter, '_get_validator_instance', return_value=mock_validator):
            result = self.adapter._perform_validation(
                data={"dimension": "test_dim", "evidence": [{"score": 0.8}]},
                validation_type="dimension_requirements",
                validation_rules={}
            )
            
            outcome, confidence, errors, warnings, details = result
            assert outcome == ValidationOutcome.PASSED
            assert confidence == 0.9  # 1.0 - 0.1 risk bound
            assert len(errors) == 0
    
    def test_conjunctive_conditions_validation(self):
        """Test conjunctive conditions validation"""
        mock_validator = Mock()
        mock_explanation = {
            "satisfied_conditions": 3,
            "total_conditions": 4,
            "failed_literals": []
        }
        mock_validator.check_conjunctive_conditions.return_value = (True, mock_explanation)
        
        with patch.object(self.adapter, '_get_validator_instance', return_value=mock_validator):
            result = self.adapter._perform_validation(
                data={"point": "P1", "scores": {"dim1": 0.8, "dim2": 0.9}},
                validation_type="conjunctive_conditions",
                validation_rules={}
            )
            
            outcome, confidence, errors, warnings, details = result
            assert outcome == ValidationOutcome.PASSED
            assert confidence == 0.75  # 3/4 conditions satisfied
    
    def test_validation_error_handling(self):
        """Test error handling during validation"""
        mock_validator = Mock()
        mock_validator.validate_dimension_requirements.side_effect = Exception("Validation error")
        
        with patch.object(self.adapter, '_get_validator_instance', return_value=mock_validator):
            result = self.adapter._perform_validation(
                data={"dimension": "test", "evidence": []},
                validation_type="dimension_requirements",
                validation_rules={}
            )
            
            outcome, confidence, errors, warnings, details = result
            assert outcome == ValidationOutcome.FAILED
            assert confidence == 0.0
            assert len(errors) > 0
            assert "Validation error" in errors[0]


class TestRubricValidatorAdapter:
    """Test cases for RubricValidatorAdapter"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = SynchronousEventBus()
        self.adapter = RubricValidatorAdapter(self.event_bus)
    
    def test_rubric_adapter_initialization(self):
        """Test rubric validator adapter initialization"""
        assert self.adapter.validator_id == "rubric_validator"
        assert "rubric_structure" in self.adapter.supported_validation_types
        assert PipelineStage.CLASSIFICATION in self.adapter._get_monitored_stages()
    
    def test_rubric_validation_success(self):
        """Test successful rubric validation"""
        mock_validator = Mock()
        mock_validator.validate.return_value = (True, [], [])
        mock_validator.generate_report.return_value = "Validation successful"
        
        with patch.object(self.adapter, '_get_validator_instance', return_value=mock_validator):
            result = self.adapter._perform_validation(
                data={"rubric_path": "test.yaml"},
                validation_type="rubric_structure",
                validation_rules={}
            )
            
            outcome, confidence, errors, warnings, details = result
            assert outcome == ValidationOutcome.PASSED
            assert confidence == 1.0
            assert len(errors) == 0
    
    def test_rubric_validation_with_warnings(self):
        """Test rubric validation with warnings"""
        mock_validator = Mock()
        mock_validator.validate.return_value = (True, [], ["Warning 1"])
        mock_validator.generate_report.return_value = "Validation with warnings"
        
        with patch.object(self.adapter, '_get_validator_instance', return_value=mock_validator):
            result = self.adapter._perform_validation(
                data={},
                validation_type="rubric_structure",
                validation_rules={}
            )
            
            outcome, confidence, errors, warnings, details = result
            assert outcome == ValidationOutcome.WARNING
            assert confidence == 0.5


class TestNormativeValidatorAdapter:
    """Test cases for NormativeValidatorAdapter"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = SynchronousEventBus()
        self.adapter = NormativeValidatorAdapter(self.event_bus)
    
    def test_normative_adapter_initialization(self):
        """Test normative validator adapter initialization"""
        assert self.adapter.validator_id == "normative_validator"
        assert "normative_compliance" in self.adapter.supported_validation_types
        assert PipelineStage.INGESTION in self.adapter._get_monitored_stages()
    
    def test_normative_compliance_validation(self):
        """Test normative compliance validation"""
        data = {"field1": "value1", "field2": "value2"}
        rules = {"required_fields": ["field1", "field2"]}
        
        result = self.adapter._perform_validation(
            data=data,
            validation_type="normative_compliance",
            validation_rules=rules
        )
        
        outcome, confidence, errors, warnings, details = result
        assert outcome == ValidationOutcome.PASSED
        assert confidence == 1.0  # All required fields present
        assert "compliance_score" in details
    
    def test_partial_compliance(self):
        """Test partial normative compliance"""
        data = {"field1": "value1"}  # Missing field2
        rules = {"required_fields": ["field1", "field2"]}
        
        result = self.adapter._perform_validation(
            data=data,
            validation_type="normative_compliance",
            validation_rules=rules
        )
        
        outcome, confidence, errors, warnings, details = result
        assert outcome == ValidationOutcome.FAILED
        assert confidence == 0.5  # 1 out of 2 fields present


class TestValidatorRegistry:
    """Test cases for validator registry functions"""
    
    def setup_method(self):
        """Setup test environment"""
        cleanup_validator_adapters()  # Start clean
        self.event_bus = SynchronousEventBus()
    
    def teardown_method(self):
        """Cleanup after tests"""
        cleanup_validator_adapters()
    
    def test_adapter_registration(self):
        """Test validator adapter registration"""
        adapter = NormativeValidatorAdapter(self.event_bus)
        register_validator_adapter(adapter)
        
        retrieved = get_validator_adapter("normative_validator")
        assert retrieved == adapter
    
    def test_adapter_retrieval_not_found(self):
        """Test retrieval of non-existent adapter"""
        adapter = get_validator_adapter("nonexistent_validator")
        assert adapter is None
    
    def test_default_adapters_initialization(self):
        """Test initialization of default adapters"""
        initialize_default_adapters(self.event_bus)
        
        constraint_adapter = get_validator_adapter("constraint_validator")
        rubric_adapter = get_validator_adapter("rubric_validator")
        normative_adapter = get_validator_adapter("normative_validator")
        
        assert constraint_adapter is not None
        assert rubric_adapter is not None
        assert normative_adapter is not None
    
    def test_adapter_cleanup(self):
        """Test cleanup of validator adapters"""
        initialize_default_adapters(self.event_bus)
        
        # Verify adapters exist
        assert get_validator_adapter("constraint_validator") is not None
        
        # Cleanup
        cleanup_validator_adapters()
        
        # Verify adapters are removed
        assert get_validator_adapter("constraint_validator") is None


class TestAdapterEventIntegration:
    """Test integration between adapters and event system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = SynchronousEventBus()
        initialize_default_adapters(self.event_bus)
    
    def teardown_method(self):
        """Cleanup after tests"""
        cleanup_validator_adapters()
    
    def test_validation_request_processing(self):
        """Test processing of validation requests through event system"""
        payload = ValidationPayload(
            validator_id="normative_validator",
            validation_type="normative_compliance",
            data_to_validate={"field1": "value1"},
            validation_rules={"required_fields": ["field1"]}
        )
        
        event = ValidationRequestedEvent(payload)
        result = self.event_bus.publish(event)
        
        assert result.success
        assert len(result.response_events) > 0
        
        response_event = result.response_events[0]
        assert isinstance(response_event, ValidationCompletedEvent)
        assert response_event.result.validator_id == "normative_validator"
    
    def test_stage_completion_triggers_validation(self):
        """Test that stage completion events trigger validations"""
        context = PipelineContext(query="test")
        event = StageCompletedEvent(
            stage=PipelineStage.INGESTION,
            context=context,
            results={"processed": True},
            success=True
        )
        
        result = self.event_bus.publish(event)
        
        assert result.success
        # Should have validation responses
        validation_responses = [
            e for e in result.response_events 
            if isinstance(e, ValidationCompletedEvent)
        ]
        assert len(validation_responses) > 0
    
    def test_adapter_error_handling_in_event_system(self):
        """Test error handling when adapters fail during event processing"""
        # Create adapter that will fail
        class FailingAdapter(NormativeValidatorAdapter):
            def _perform_validation(self, data, validation_type, validation_rules):
                raise Exception("Validation failure")
        
        failing_adapter = FailingAdapter(self.event_bus)
        register_validator_adapter(failing_adapter)
        
        payload = ValidationPayload(
            validator_id="normative_validator",
            validation_type="normative_compliance",
            data_to_validate={}
        )
        
        event = ValidationRequestedEvent(payload)
        result = self.event_bus.publish(event)
        
        # Event processing should handle the error gracefully
        assert not result.success
        assert result.error_message is not None


if __name__ == "__main__":
    pytest.main([__file__])