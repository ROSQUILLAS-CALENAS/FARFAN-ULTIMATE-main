#!/usr/bin/env python3
"""
Demo of the Event-Driven Architecture for EGW Query Expansion Pipeline

This demo shows how the synchronous event bus system eliminates circular import
dependencies while maintaining deterministic execution flow.
"""

import sys
import time
from pathlib import Path

# Add the package to path for demo
sys.path.insert(0, str(Path(__file__).parent.parent))

from egw_query_expansion.core.event_bus import SynchronousEventBus, get_event_bus
from egw_query_expansion.core.event_schemas import (
    PipelineStage, PipelineContext, ValidationOutcome,
    create_stage_started_event, create_validation_request, create_error_event,
    create_performance_event
)
from egw_query_expansion.core.event_driven_orchestrator import (
    EventDrivenOrchestrator, StageConfiguration, ExecutionPlan
)
from egw_query_expansion.core.validator_adapters import (
    initialize_default_adapters, get_validator_adapter
)


def demo_basic_event_bus():
    """Demonstrate basic event bus functionality"""
    print("=== Basic Event Bus Demo ===")
    
    # Create event bus
    event_bus = SynchronousEventBus()
    
    # Create some events
    context = PipelineContext(query="What is machine learning?", document_id="doc_123")
    
    stage_event = create_stage_started_event(
        stage=PipelineStage.INGESTION,
        context=context,
        source_id="demo_orchestrator"
    )
    
    validation_event = create_validation_request(
        validator_id="demo_validator",
        validation_type="structure_check",
        data={"field1": "value1", "field2": "value2"},
        source_id="demo_processor"
    )
    
    error_event = create_error_event(
        error_type="demo_error",
        message="This is a demo error",
        stage=PipelineStage.ANALYSIS,
        source_id="demo_component"
    )
    
    # Create a simple event handler
    from egw_query_expansion.core.event_bus import EventHandler, BaseEvent
    
    class DemoHandler(EventHandler):
        def __init__(self):
            self.handled_events = []
        
        @property
        def handler_id(self) -> str:
            return "demo_handler"
        
        def can_handle(self, event: BaseEvent) -> bool:
            return event.event_type.startswith("pipeline.")
        
        def handle(self, event: BaseEvent):
            self.handled_events.append(event)
            print(f"  Handler processed: {event.event_type}")
            return None
    
    # Subscribe handler
    handler = DemoHandler()
    event_bus.subscribe("pipeline.stage.started", handler)
    event_bus.subscribe("pipeline.error", handler)
    
    # Publish events
    print("Publishing events...")
    
    result1 = event_bus.publish(stage_event)
    print(f"  Stage event published: success={result1.success}")
    
    result2 = event_bus.publish(error_event)  
    print(f"  Error event published: success={result2.success}")
    
    result3 = event_bus.publish(validation_event)
    print(f"  Validation event published: success={result3.success}")
    
    # Show statistics
    stats = event_bus.get_stats()
    print(f"\nEvent Bus Stats:")
    print(f"  Total events processed: {stats['total_events_processed']}")
    print(f"  Event types: {stats['event_types']}")
    print(f"  Active subscriptions: {stats['active_subscriptions']}")
    print(f"  Handler processed {len(handler.handled_events)} events")


def demo_validator_adapters():
    """Demonstrate validator adapters working with event bus"""
    print("\n=== Validator Adapters Demo ===")
    
    # Initialize event bus and adapters
    event_bus = SynchronousEventBus()
    initialize_default_adapters(event_bus)
    
    # Show registered adapters
    print("Registered validator adapters:")
    from egw_query_expansion.core.validator_adapters import validator_registry
    for validator_id in validator_registry.keys():
        print(f"  - {validator_id}")
    
    # Create validation requests
    normative_request = create_validation_request(
        validator_id="normative_validator",
        validation_type="normative_compliance",
        data={
            "field1": "value1",
            "field2": "value2",
            "metadata": {"source": "demo"}
        },
        source_id="demo_orchestrator"
    )
    
    print(f"\nPublishing normative validation request...")
    result = event_bus.publish(normative_request)
    
    print(f"  Request processed: success={result.success}")
    print(f"  Response events: {len(result.response_events)}")
    
    if result.response_events:
        response = result.response_events[0]
        if hasattr(response, 'result'):
            validation_result = response.result
            print(f"  Validation outcome: {validation_result.outcome.value}")
            print(f"  Confidence score: {validation_result.confidence_score}")
            print(f"  Processing time: {validation_result.processing_time_ms:.2f}ms")
    
    # Test constraint validation (will fail gracefully without actual validator)
    constraint_request = create_validation_request(
        validator_id="constraint_validator", 
        validation_type="dimension_requirements",
        data={
            "dimension": "quality",
            "evidence": [
                {"score": 0.8, "indicators": ["completeness"]},
                {"score": 0.9, "indicators": ["accuracy"]}
            ]
        },
        source_id="demo_orchestrator"
    )
    
    print(f"\nPublishing constraint validation request...")
    result = event_bus.publish(constraint_request)
    print(f"  Request processed: success={result.success}")
    
    # Show adapter statistics
    normative_adapter = get_validator_adapter("normative_validator")
    if normative_adapter:
        stats = normative_adapter.get_validation_stats()
        print(f"\nNormative Validator Stats:")
        print(f"  Total validations: {stats['total_validations']}")
        if stats['total_validations'] > 0:
            print(f"  Average confidence: {stats.get('average_confidence', 0):.2f}")


def demo_event_driven_orchestrator():
    """Demonstrate event-driven orchestrator"""
    print("\n=== Event-Driven Orchestrator Demo ===")
    
    # Create orchestrator with event bus
    event_bus = SynchronousEventBus()
    initialize_default_adapters(event_bus)
    orchestrator = EventDrivenOrchestrator(event_bus)
    
    print(f"Orchestrator state: {orchestrator.state.value}")
    
    # Create a simple execution plan
    stages = [
        StageConfiguration(
            stage=PipelineStage.INGESTION,
            required_validations=[],  # Simplified for demo
            timeout_seconds=10
        ),
        StageConfiguration(
            stage=PipelineStage.ANALYSIS,
            required_validations=[],
            dependencies=[PipelineStage.INGESTION],
            timeout_seconds=10
        )
    ]
    
    plan = ExecutionPlan(
        stages=stages,
        execution_order=[PipelineStage.INGESTION, PipelineStage.ANALYSIS],
        validation_requirements={}  # Simplified for demo
    )
    
    print("Executing simplified pipeline...")
    start_time = time.time()
    
    result = orchestrator.execute_pipeline(
        query="How does machine learning work?",
        document_id="demo_doc",
        execution_plan=plan
    )
    
    execution_time = time.time() - start_time
    
    print(f"\nExecution Results:")
    print(f"  Success: {result['success']}")
    print(f"  Execution time: {execution_time:.2f}s")
    print(f"  Completed stages: {result['completed_stages']}")
    print(f"  Failed stages: {result['failed_stages']}")
    print(f"  Error count: {result['error_count']}")
    
    # Show orchestrator statistics
    stats = orchestrator.get_orchestration_stats()
    print(f"\nOrchestrator Stats:")
    print(f"  Current state: {stats['current_state']}")
    print(f"  Completed stages: {stats['completed_stages']}")
    print(f"  Performance metrics: {len(stats['performance_metrics'])} stages")


def demo_circular_dependency_elimination():
    """Demonstrate how events eliminate circular dependencies"""
    print("\n=== Circular Dependency Elimination Demo ===")
    
    print("Traditional Architecture (with circular dependencies):")
    print("  Orchestrator -> imports -> Validator")
    print("  Validator -> imports -> PipelineStage")
    print("  PipelineStage -> imports -> Orchestrator")
    print("  ❌ Circular import error!")
    
    print("\nEvent-Driven Architecture (no circular dependencies):")
    print("  Orchestrator -> publishes events -> EventBus")
    print("  ValidatorAdapter -> subscribes to events -> EventBus")
    print("  ValidatorAdapter -> wraps -> OriginalValidator")
    print("  ✅ No direct imports between components!")
    
    # Demonstrate loose coupling
    event_bus = SynchronousEventBus()
    
    # Components can be added/removed dynamically
    print("\nDynamic component registration:")
    
    # Add validator
    from egw_query_expansion.core.validator_adapters import NormativeValidatorAdapter
    validator = NormativeValidatorAdapter(event_bus)
    print(f"  ✓ Added validator: {validator.handler_id}")
    
    # Remove validator
    event_bus.unsubscribe("validation.requested", validator)
    print(f"  ✓ Removed validator: {validator.handler_id}")
    
    # Show event bus still works
    test_event = create_error_event("test", "Still working", source_id="demo")
    result = event_bus.publish(test_event)
    print(f"  ✓ Event bus operational: {result.success}")


def main():
    """Run all demos"""
    print("Event-Driven Architecture Demo for EGW Query Expansion Pipeline")
    print("================================================================")
    
    try:
        demo_basic_event_bus()
        demo_validator_adapters()
        demo_event_driven_orchestrator()
        demo_circular_dependency_elimination()
        
        print("\n✅ All demos completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("  • Eliminated circular import dependencies")
        print("  • Maintained synchronous execution flow") 
        print("  • Enabled loose coupling between components")
        print("  • Preserved backward compatibility")
        print("  • Added dynamic component registration")
        print("  • Provided comprehensive event tracking")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)