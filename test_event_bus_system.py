"""
Test Suite for Event Bus System
==============================

Comprehensive tests for the event bus system, typed events, validators,
and the refactored orchestrator architecture.
"""

import unittest
import time
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any, List

from event_bus import EventBus, EventSubscription, event_handler
from event_schemas import (
    EventType, StageEventData, ValidationEventData, VerdictEventData,
    StageStartedEvent, StageCompletedEvent, ValidationRequestedEvent,
    VerdictIssuedEvent, EventRegistry, create_stage_started_event
)
from event_driven_validator import (
    BaseEventValidator, PDTDocumentValidator, SchemaValidator,
    QualityAssuranceValidator, ValidatorRegistry, ValidationResult
)
from event_driven_orchestrator import EventDrivenOrchestrator, PipelineExecutionContext
from refactored_pipeline_orchestrator import RefactoredPipelineOrchestrator


class TestEventSchemas(unittest.TestCase):
    """Test event schema classes and utilities"""
    
    def test_event_data_classes(self):
        """Test event data structures"""
        # Test StageEventData
        stage_data = StageEventData(
            stage_name="test_stage",
            stage_type="processing",
            input_data={'key': 'value'},
            execution_time_ms=150.5
        )
        
        self.assertEqual(stage_data.stage_name, "test_stage")
        self.assertEqual(stage_data.stage_type, "processing")
        self.assertEqual(stage_data.execution_time_ms, 150.5)
    
    def test_stage_events(self):
        """Test stage event classes"""
        # Test StageStartedEvent
        stage_data = StageEventData("test_stage", "processing")
        event = StageStartedEvent(
            data=stage_data,
            source="test_orchestrator"
        )
        
        self.assertEqual(event.event_type, EventType.STAGE_STARTED)
        self.assertTrue(event.validate_data())
        self.assertIsNotNone(event.event_id)
        
        # Test event serialization
        event_dict = event.to_dict()
        self.assertIn('event_id', event_dict)
        self.assertIn('event_type', event_dict)
        self.assertEqual(event_dict['source'], "test_orchestrator")
    
    def test_validation_events(self):
        """Test validation event classes"""
        validation_data = ValidationEventData(
            validator_name="test_validator",
            validator_type="document",
            validation_target="sample_document",
            validation_rules=["rule1", "rule2"]
        )
        
        event = ValidationRequestedEvent(
            data=validation_data,
            source="test_source",
            correlation_id="test-123"
        )
        
        self.assertEqual(event.event_type, EventType.VALIDATION_REQUESTED)
        self.assertTrue(event.validate_data())
        self.assertEqual(event.correlation_id, "test-123")
    
    def test_verdict_events(self):
        """Test verdict event creation"""
        verdict_data = VerdictEventData(
            validator_name="test_validator",
            verdict="PASS",
            confidence=0.85,
            evidence=[{'type': 'test', 'details': 'test evidence'}],
            recommendations=["improve quality"]
        )
        
        event = VerdictIssuedEvent(
            data=verdict_data,
            source="test_validator"
        )
        
        self.assertEqual(event.event_type, EventType.VERDICT_ISSUED)
        self.assertTrue(event.validate_data())
        self.assertEqual(event.data.verdict, "PASS")
        self.assertEqual(event.data.confidence, 0.85)
    
    def test_event_registry(self):
        """Test event registry functionality"""
        # Test getting event class
        event_class = EventRegistry.get_event_class(EventType.STAGE_STARTED)
        self.assertEqual(event_class, StageStartedEvent)
        
        # Test creating event
        stage_data = StageEventData("test", "test")
        event = EventRegistry.create_event(
            EventType.STAGE_STARTED,
            data=stage_data,
            source="test"
        )
        
        self.assertIsInstance(event, StageStartedEvent)
    
    def test_convenience_functions(self):
        """Test convenience functions for event creation"""
        event = create_stage_started_event(
            stage_name="test_stage",
            stage_type="test_type",
            source="test_source",
            correlation_id="test-123",
            execution_time_ms=100.0
        )
        
        self.assertIsInstance(event, StageStartedEvent)
        self.assertEqual(event.data.stage_name, "test_stage")
        self.assertEqual(event.correlation_id, "test-123")
        self.assertEqual(event.data.execution_time_ms, 100.0)


class TestEventBus(unittest.TestCase):
    """Test event bus functionality"""
    
    def setUp(self):
        """Set up test event bus"""
        self.event_bus = EventBus(max_event_history=100)
        self.received_events = []
    
    def tearDown(self):
        """Clean up event bus"""
        self.event_bus.shutdown()
    
    def test_subscribe_and_publish(self):
        """Test basic subscription and publishing"""
        def handler(event):
            self.received_events.append(event)
        
        # Subscribe to events
        sub_id = self.event_bus.subscribe(
            subscriber_name="test_subscriber",
            event_types=[EventType.STAGE_STARTED],
            handler=handler
        )
        
        self.assertIsNotNone(sub_id)
        
        # Publish event
        event = create_stage_started_event("test", "test", "source")
        result = self.event_bus.publish(event)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['handlers_called'], 1)
        self.assertEqual(result['handlers_succeeded'], 1)
        self.assertEqual(len(self.received_events), 1)
    
    def test_multiple_subscribers(self):
        """Test multiple subscribers to same event"""
        received_1 = []
        received_2 = []
        
        def handler_1(event):
            received_1.append(event)
        
        def handler_2(event):
            received_2.append(event)
        
        # Subscribe multiple handlers
        self.event_bus.subscribe("sub1", EventType.STAGE_STARTED, handler_1, priority=10)
        self.event_bus.subscribe("sub2", EventType.STAGE_STARTED, handler_2, priority=5)
        
        # Publish event
        event = create_stage_started_event("test", "test", "source")
        result = self.event_bus.publish(event)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['handlers_called'], 2)
        self.assertEqual(len(received_1), 1)
        self.assertEqual(len(received_2), 1)
    
    def test_event_filtering(self):
        """Test event type filtering"""
        stage_events = []
        validation_events = []
        
        def stage_handler(event):
            stage_events.append(event)
        
        def validation_handler(event):
            validation_events.append(event)
        
        # Subscribe to different event types
        self.event_bus.subscribe("stage_sub", EventType.STAGE_STARTED, stage_handler)
        self.event_bus.subscribe("validation_sub", EventType.VALIDATION_REQUESTED, validation_handler)
        
        # Publish stage event
        stage_event = create_stage_started_event("test", "test", "source")
        self.event_bus.publish(stage_event)
        
        # Publish validation event
        validation_data = ValidationEventData("validator", "type", "target")
        validation_event = ValidationRequestedEvent(data=validation_data, source="source")
        self.event_bus.publish(validation_event)
        
        # Check filtering worked
        self.assertEqual(len(stage_events), 1)
        self.assertEqual(len(validation_events), 1)
    
    def test_handler_errors(self):
        """Test error handling in event handlers"""
        def failing_handler(event):
            raise ValueError("Test error")
        
        def working_handler(event):
            self.received_events.append(event)
        
        # Subscribe both handlers
        self.event_bus.subscribe("failing", EventType.STAGE_STARTED, failing_handler)
        self.event_bus.subscribe("working", EventType.STAGE_STARTED, working_handler)
        
        # Publish event
        event = create_stage_started_event("test", "test", "source")
        result = self.event_bus.publish(event)
        
        # Should have partial success
        self.assertFalse(result['success'])  # Overall failed due to error
        self.assertEqual(result['handlers_called'], 2)
        self.assertEqual(result['handlers_succeeded'], 1)
        self.assertEqual(result['handlers_failed'], 1)
        self.assertEqual(len(self.received_events), 1)  # Working handler still worked
    
    def test_subscription_management(self):
        """Test subscription lifecycle management"""
        def handler(event):
            pass
        
        # Subscribe
        sub_id = self.event_bus.subscribe("test", EventType.STAGE_STARTED, handler)
        
        # Check subscription exists
        sub_info = self.event_bus.get_subscription_info(sub_id)
        self.assertIsNotNone(sub_info)
        self.assertEqual(sub_info['subscriber_name'], "test")
        
        # Unsubscribe
        success = self.event_bus.unsubscribe(sub_id)
        self.assertTrue(success)
        
        # Check subscription removed
        sub_info = self.event_bus.get_subscription_info(sub_id)
        self.assertIsNone(sub_info)
    
    def test_event_history(self):
        """Test event history tracking"""
        # Publish several events
        for i in range(5):
            event = create_stage_started_event(f"stage_{i}", "test", "source")
            self.event_bus.publish(event)
        
        # Check history
        history = self.event_bus.get_event_history()
        self.assertEqual(len(history), 5)
        
        # Check history content
        for i, event_record in enumerate(history):
            self.assertIn('event_id', event_record)
            self.assertIn('event_type', event_record)
            self.assertEqual(event_record['source'], "source")
    
    def test_statistics(self):
        """Test event bus statistics"""
        def handler(event):
            pass
        
        # Subscribe and publish
        self.event_bus.subscribe("test", EventType.STAGE_STARTED, handler)
        
        event = create_stage_started_event("test", "test", "source")
        self.event_bus.publish(event)
        
        # Check statistics
        stats = self.event_bus.get_stats()
        self.assertEqual(stats['events_published'], 1)
        self.assertEqual(stats['events_delivered'], 1)
        self.assertEqual(stats['active_subscriptions'], 1)
        self.assertGreater(stats['success_rate'], 0)


class TestEventDrivenValidators(unittest.TestCase):
    """Test event-driven validator components"""
    
    def setUp(self):
        """Set up test environment"""
        self.event_bus = EventBus()
        self.received_verdicts = []
        
        # Subscribe to verdict events
        def verdict_handler(event):
            self.received_verdicts.append(event)
        
        self.event_bus.subscribe(
            "test_verdict_collector",
            EventType.VERDICT_ISSUED,
            verdict_handler
        )
    
    def tearDown(self):
        """Clean up"""
        self.event_bus.shutdown()
    
    def test_pdt_document_validator(self):
        """Test PDT document validator"""
        validator = PDTDocumentValidator(self.event_bus)
        
        # Test validation with good data
        good_data = {
            'sections': ['DIAGNOSTICO', 'PROGRAMAS', 'PRESUPUESTO', 'METAS'],
            'quality_metrics': {
                'completeness_index': 0.9,
                'ocr_ratio': 0.2
            },
            'budget_data': {'total': 1000000},
            'goals': ['goal1', 'goal2'],
            'programs': ['program1', 'program2'],
            'legal_references': ['ley1', 'ley2', 'decreto1']
        }
        
        result = validator.validate_data(good_data, ['check_budget_consistency'])
        
        self.assertTrue(result.passed)
        self.assertGreater(result.confidence, 0.5)
        self.assertEqual(result.verdict, "PASS")
    
    def test_pdt_validator_missing_sections(self):
        """Test PDT validator with missing sections"""
        validator = PDTDocumentValidator(self.event_bus)
        
        # Test with missing mandatory sections
        bad_data = {
            'sections': ['DIAGNOSTICO'],  # Missing required sections
            'quality_metrics': {'completeness_index': 0.5}
        }
        
        result = validator.validate_data(bad_data, [])
        
        self.assertFalse(result.passed)
        self.assertEqual(result.verdict, "FAIL")
        self.assertGreater(len(result.evidence), 0)
        self.assertGreater(len(result.recommendations), 0)
    
    def test_schema_validator(self):
        """Test schema validator"""
        validator = SchemaValidator(self.event_bus)
        
        # Register a test schema
        test_schema = {
            'required': ['name', 'type'],
            'properties': {
                'name': {'type': 'string'},
                'type': {'type': 'string'},
                'count': {'type': 'number'}
            }
        }
        validator.register_schema('test_schema', test_schema)
        
        # Test valid data
        valid_data = {
            'schema_type': 'test_schema',
            'name': 'test_item',
            'type': 'test_type',
            'count': 42
        }
        
        result = validator.validate_data(valid_data, [])
        self.assertTrue(result.passed)
        
        # Test invalid data
        invalid_data = {
            'schema_type': 'test_schema',
            'name': 'test_item',
            # Missing 'type' field
            'count': 'not_a_number'  # Wrong type
        }
        
        result = validator.validate_data(invalid_data, [])
        self.assertFalse(result.passed)
        self.assertGreater(len(result.evidence), 0)
    
    def test_quality_assurance_validator(self):
        """Test quality assurance validator"""
        validator = QualityAssuranceValidator(self.event_bus)
        
        # Test high quality data
        high_quality_data = {
            'overall_quality_score': 0.85,
            'structural_metrics': {'coherence': 0.8},
            'completeness_score': 0.9
        }
        
        result = validator.validate_data(high_quality_data, [])
        self.assertTrue(result.passed)
        
        # Test low quality data
        low_quality_data = {
            'overall_quality_score': 0.5,  # Below threshold
            'structural_metrics': {'coherence': 0.4},  # Below threshold
            'completeness_score': 0.6  # Below threshold
        }
        
        result = validator.validate_data(low_quality_data, [])
        self.assertFalse(result.passed)
        self.assertGreater(len(result.evidence), 0)
    
    def test_validation_event_flow(self):
        """Test complete validation event flow"""
        validator = PDTDocumentValidator(self.event_bus)
        
        # Create validation request event
        validation_data = ValidationEventData(
            validator_name="PDTDocumentValidator",
            validator_type="document_validator",
            validation_target="test_document",
            input_data={
                'sections': ['DIAGNOSTICO', 'PROGRAMAS', 'PRESUPUESTO', 'METAS'],
                'quality_metrics': {'completeness_index': 0.8}
            },
            validation_rules=[]
        )
        
        request_event = ValidationRequestedEvent(
            data=validation_data,
            source="test_requester",
            correlation_id="test-123"
        )
        
        # Publish request
        self.event_bus.publish(request_event)
        
        # Give a moment for async processing
        time.sleep(0.1)
        
        # Check that verdict was issued
        self.assertEqual(len(self.received_verdicts), 1)
        verdict_event = self.received_verdicts[0]
        self.assertEqual(verdict_event.event_type, EventType.VERDICT_ISSUED)
        self.assertEqual(verdict_event.correlation_id, "test-123")
    
    def test_validator_registry(self):
        """Test validator registry functionality"""
        registry = ValidatorRegistry(self.event_bus)
        
        # Create default validators
        registry.create_default_validators()
        
        # Check validators were registered
        validators = registry.list_validators()
        self.assertIn("PDTDocumentValidator", validators)
        self.assertIn("SchemaValidator", validators)
        self.assertIn("QualityAssuranceValidator", validators)
        
        # Get specific validator
        pdt_validator = registry.get_validator("PDTDocumentValidator")
        self.assertIsInstance(pdt_validator, PDTDocumentValidator)
        
        # Get statistics
        stats = registry.get_all_statistics()
        self.assertIn("PDTDocumentValidator", stats)


class TestEventDrivenOrchestrator(unittest.TestCase):
    """Test event-driven orchestrator"""
    
    def setUp(self):
        """Set up test orchestrator"""
        self.event_bus = EventBus()
        self.orchestrator = EventDrivenOrchestrator(self.event_bus, "TestOrchestrator")
        
        # Track events for verification
        self.orchestration_events = []
        self.stage_events = []
        
        def orchestration_handler(event):
            self.orchestration_events.append(event)
        
        def stage_handler(event):
            self.stage_events.append(event)
        
        self.event_bus.subscribe("test_orchestration", 
                                [EventType.ORCHESTRATION_STARTED, EventType.ORCHESTRATION_COMPLETED],
                                orchestration_handler)
        self.event_bus.subscribe("test_stage", 
                                [EventType.STAGE_STARTED, EventType.STAGE_COMPLETED],
                                stage_handler)
    
    def tearDown(self):
        """Clean up"""
        self.orchestrator.shutdown()
        self.event_bus.shutdown()
    
    def test_stage_handler_registration(self):
        """Test registering stage handlers"""
        def test_handler(data):
            return {'result': 'success'}
        
        self.orchestrator.register_stage_handler("test_stage", test_handler)
        
        # Verify handler is registered
        self.assertIn("test_stage", self.orchestrator._stage_handlers)
    
    def test_pipeline_execution_lifecycle(self):
        """Test complete pipeline execution lifecycle"""
        # Register test stage handlers
        def stage1_handler(data):
            return {'stage1_result': 'completed', 'data_from_stage1': True}
        
        def stage2_handler(data):
            # Should receive output from stage1
            self.assertIn('data_from_stage1', data)
            return {'stage2_result': 'completed'}
        
        self.orchestrator.register_stage_handler("stage1", stage1_handler)
        self.orchestrator.register_stage_handler("stage2", stage2_handler)
        
        # Start pipeline execution
        pipeline_config = {
            'stages': ['stage1', 'stage2'],
            'failure_strategy': 'stop_on_failure'
        }
        
        input_data = {'initial_data': 'test'}
        
        execution_id = self.orchestrator.start_pipeline_execution(
            pipeline_config=pipeline_config,
            input_data=input_data
        )
        
        self.assertIsNotNone(execution_id)
        
        # Wait for execution to complete
        for i in range(20):  # Max 2 second wait
            status = self.orchestrator.get_execution_status(execution_id)
            if status and status['is_complete']:
                break
            time.sleep(0.1)
        
        # Check final status
        final_status = self.orchestrator.get_execution_status(execution_id)
        self.assertIsNotNone(final_status, f"Execution status should not be None after {i+1} attempts")
        if final_status:
            self.assertTrue(final_status['is_complete'])
            self.assertFalse(final_status['has_failures'])
            self.assertEqual(len(final_status['completed_stages']), 2)
    
    def test_pipeline_failure_handling(self):
        """Test pipeline failure handling"""
        def failing_stage(data):
            raise ValueError("Intentional test failure")
        
        def normal_stage(data):
            return {'result': 'success'}
        
        self.orchestrator.register_stage_handler("normal_stage", normal_stage)
        self.orchestrator.register_stage_handler("failing_stage", failing_stage)
        
        # Start pipeline with failing stage
        pipeline_config = {
            'stages': ['normal_stage', 'failing_stage'],
            'failure_strategy': 'stop_on_failure'
        }
        
        execution_id = self.orchestrator.start_pipeline_execution(
            pipeline_config=pipeline_config,
            input_data={'test': 'data'}
        )
        
        # Wait for completion
        for i in range(20):
            status = self.orchestrator.get_execution_status(execution_id)
            if status and status['is_complete']:
                break
            time.sleep(0.1)
        
        # Check failure was handled correctly
        final_status = self.orchestrator.get_execution_status(execution_id)
        self.assertIsNotNone(final_status, f"Execution status should not be None after {i+1} attempts")
        if final_status:
            self.assertTrue(final_status['has_failures'])
            self.assertIn('failing_stage', final_status['failed_stages'])
    
    def test_execution_context(self):
        """Test pipeline execution context"""
        context = PipelineExecutionContext("test-123", {'stages': ['stage1', 'stage2']})
        
        self.assertEqual(context.execution_id, "test-123")
        self.assertFalse(context.is_complete(['stage1', 'stage2']))
        self.assertFalse(context.has_failures())
        
        # Complete stages
        context.completed_stages.append('stage1')
        context.completed_stages.append('stage2')
        
        self.assertTrue(context.is_complete(['stage1', 'stage2']))
        self.assertEqual(context.get_progress_percentage(2), 100.0)


class TestRefactoredPipelineOrchestrator(unittest.TestCase):
    """Test refactored pipeline orchestrator"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = RefactoredPipelineOrchestrator()
        
        # Check components are initialized
        self.assertIsNotNone(orchestrator.event_bus)
        self.assertIsNotNone(orchestrator.validator_registry)
        
        # Check validators are registered
        validators = orchestrator.validator_registry.list_validators()
        self.assertGreater(len(validators), 0)
        
        # Check stage handlers are registered
        self.assertGreater(len(orchestrator._stage_handlers), 0)
        
        orchestrator.shutdown()
    
    def test_pipeline_execution(self):
        """Test complete pipeline execution"""
        orchestrator = RefactoredPipelineOrchestrator()
        
        try:
            # Execute pipeline
            input_data = {
                'document_path': '/test/sample.pdf',
                'document_type': 'PDT',
                'pages': 10
            }
            
            execution_id = orchestrator.execute_pipeline(input_data)
            self.assertIsNotNone(execution_id)
            
            # Wait for completion
            for _ in range(20):  # Max 2 second wait
                status = orchestrator.get_pipeline_status(execution_id)
                if status and status['is_complete']:
                    break
                time.sleep(0.1)
            
            # Check execution completed
            final_status = orchestrator.get_pipeline_status(execution_id)
            self.assertIsNotNone(final_status)
            
            # Check that we have validator and event bus statistics
            self.assertIn('validator_statistics', final_status)
            self.assertIn('event_bus_statistics', final_status)
            
        finally:
            orchestrator.shutdown()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete event bus system"""
    
    def test_end_to_end_validation_flow(self):
        """Test end-to-end validation flow through events"""
        # Create components
        event_bus = EventBus()
        registry = ValidatorRegistry(event_bus)
        registry.create_default_validators()
        
        # Track validation results
        validation_results = []
        
        def result_handler(event):
            validation_results.append(event)
        
        event_bus.subscribe("test_results", 
                          [EventType.VALIDATION_COMPLETED, EventType.VERDICT_ISSUED],
                          result_handler)
        
        try:
            # Request validation through events
            pdt_validator = registry.get_validator("PDTDocumentValidator")
            
            validation_id = pdt_validator.request_validation(
                validation_target="test_document",
                input_data={
                    'sections': ['DIAGNOSTICO', 'PROGRAMAS', 'PRESUPUESTO'],
                    'quality_metrics': {'completeness_index': 0.9}
                },
                validation_rules=['check_budget_consistency'],
                correlation_id="integration-test"
            )
            
            # Wait for results
            time.sleep(0.2)
            
            # Check we got validation results
            self.assertGreater(len(validation_results), 0)
            
            # Check verdict was issued
            verdict_events = [e for e in validation_results if e.event_type == EventType.VERDICT_ISSUED]
            self.assertEqual(len(verdict_events), 1)
            
            verdict = verdict_events[0]
            self.assertEqual(verdict.correlation_id, "integration-test")
            self.assertIn(verdict.data.verdict, ["PASS", "FAIL", "WARNING"])
            
        finally:
            registry.shutdown_all()
            event_bus.shutdown()


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)