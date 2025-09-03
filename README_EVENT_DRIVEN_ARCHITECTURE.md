# Event-Driven Architecture for EGW Query Expansion Pipeline

This document describes the synchronous event bus system implemented to eliminate circular import dependencies while maintaining deterministic execution flow in the EGW Query Expansion Pipeline.

## Architecture Overview

The event-driven architecture decouples orchestrators from validators using a synchronous event bus, allowing components to communicate through typed events instead of direct imports.

### Key Components

1. **SynchronousEventBus** - Central event coordinator that maintains execution order
2. **Event Schemas** - Typed event definitions for pipeline transitions and validation
3. **Validator Adapters** - Wrapper classes that convert existing validators to event handlers
4. **Event-Driven Orchestrator** - Orchestrator that publishes events instead of direct calls

## Benefits

- **Eliminates Circular Dependencies**: Components no longer import each other directly
- **Maintains Synchronous Execution**: Events are processed immediately in order
- **Preserves Backward Compatibility**: Existing validators work without modification
- **Enables Dynamic Registration**: Components can be added/removed at runtime
- **Provides Comprehensive Tracking**: All events and validations are logged

## Core Components

### Event Bus (`egw_query_expansion/core/event_bus.py`)

The synchronous event bus provides:
- Type-safe event publishing and subscription
- Synchronous processing to maintain execution order
- Event filtering and handler management
- Comprehensive event history and statistics
- Error handling and recovery

```python
from egw_query_expansion.core.event_bus import get_event_bus

# Get global event bus instance
bus = get_event_bus()

# Subscribe handler to event type
bus.subscribe("validation.requested", my_handler)

# Publish event for synchronous processing
result = bus.publish(my_event)
```

### Event Schemas (`egw_query_expansion/core/event_schemas.py`)

Defines typed events for all pipeline communications:

**Pipeline Events:**
- `StageStartedEvent` - Pipeline stage initialization
- `StageCompletedEvent` - Pipeline stage completion
- `StageFailedEvent` - Pipeline stage failure

**Validation Events:**
- `ValidationRequestedEvent` - Request for validation
- `ValidationCompletedEvent` - Validation results
- `ValidationPayload` - Validation input data
- `ValidationResult` - Validation output data

**Control Events:**
- `OrchestratorCommandEvent` - Orchestrator commands (pause, resume, abort)
- `PipelineStateChangedEvent` - Pipeline state transitions
- `ErrorEvent` - Error reporting
- `PerformanceMetricEvent` - Performance monitoring

### Validator Adapters (`egw_query_expansion/core/validator_adapters.py`)

Adapter classes wrap existing validators to work with the event system:

- **ValidatorAdapter** - Base class for all validator adapters
- **ConstraintValidatorAdapter** - Wraps constraint validation logic
- **RubricValidatorAdapter** - Wraps rubric validation logic  
- **NormativeValidatorAdapter** - Wraps normative compliance validation

```python
from egw_query_expansion.core.validator_adapters import initialize_default_adapters

# Initialize all default validator adapters
initialize_default_adapters(event_bus)

# Validators automatically subscribe to relevant events
# No direct imports between orchestrator and validators needed
```

### Event-Driven Orchestrator (`egw_query_expansion/core/event_driven_orchestrator.py`)

Orchestrator that coordinates pipeline execution through event publishing:

```python
from egw_query_expansion.core.event_driven_orchestrator import EventDrivenOrchestrator

orchestrator = EventDrivenOrchestrator()

# Execute pipeline through events
result = orchestrator.execute_pipeline(
    query="How does machine learning work?",
    document_id="doc123"
)
```

## Event Flow Example

1. **Orchestrator starts stage:**
   ```python
   # Publishes StageStartedEvent
   stage_event = StageStartedEvent(stage=PipelineStage.ANALYSIS, context=context)
   bus.publish(stage_event)
   ```

2. **Stage completes processing:**
   ```python
   # Publishes StageCompletedEvent
   completed_event = StageCompletedEvent(stage=stage, results=results, success=True)
   bus.publish(completed_event)
   ```

3. **Validator adapters auto-trigger:**
   ```python
   # Adapters subscribed to stage completion events automatically receive them
   # No direct import of validator by orchestrator needed
   ```

4. **Validation request created:**
   ```python
   # Adapter publishes ValidationRequestedEvent
   validation_request = ValidationRequestedEvent(payload=validation_payload)
   bus.publish(validation_request)
   ```

5. **Validation completes:**
   ```python
   # Validator publishes ValidationCompletedEvent
   validation_result = ValidationCompletedEvent(result=result)
   bus.publish(validation_result)
   ```

## Usage

### Basic Setup

```python
from egw_query_expansion.core.event_bus import get_event_bus
from egw_query_expansion.core.validator_adapters import initialize_default_adapters
from egw_query_expansion.core.event_driven_orchestrator import EventDrivenOrchestrator

# Initialize event system
event_bus = get_event_bus()
initialize_default_adapters(event_bus)
orchestrator = EventDrivenOrchestrator(event_bus)

# Execute pipeline
result = orchestrator.execute_pipeline("Your query here")
```

### Custom Validator Adapter

```python
from egw_query_expansion.core.validator_adapters import ValidatorAdapter
from egw_query_expansion.core.event_schemas import PipelineStage, ValidationOutcome

class CustomValidatorAdapter(ValidatorAdapter):
    def __init__(self, event_bus):
        super().__init__(
            validator_id="custom_validator",
            supported_validation_types=["custom_check"],
            event_bus=event_bus
        )
    
    def _get_monitored_stages(self):
        return [PipelineStage.SYNTHESIS]
    
    def _perform_validation(self, data, validation_type, validation_rules):
        # Your validation logic here
        return (
            ValidationOutcome.PASSED,
            0.95,  # confidence
            [],    # errors
            [],    # warnings  
            {}     # details
        )
```

### Custom Event Handler

```python
from egw_query_expansion.core.event_bus import EventHandler

class CustomEventHandler(EventHandler):
    @property
    def handler_id(self):
        return "custom_handler"
    
    def can_handle(self, event):
        return event.event_type == "my.custom.event"
    
    def handle(self, event):
        # Process event
        print(f"Handling: {event.data}")
        return None

# Register handler
handler = CustomEventHandler()
event_bus.subscribe("my.custom.event", handler)
```

## Testing

Comprehensive test suites are provided:

- `test_event_bus.py` - Event bus functionality
- `test_event_schemas.py` - Event schema definitions
- `test_validator_adapters.py` - Validator adapter functionality
- `test_event_driven_orchestrator.py` - Orchestrator functionality

Run tests:
```bash
python -m pytest egw_query_expansion/tests/test_event_*.py -v
```

## Demo

Run the comprehensive demo:
```bash
python examples/event_bus_demo.py
```

The demo shows:
- Basic event bus operations
- Validator adapter functionality
- Event-driven orchestration
- Circular dependency elimination

## Migration from Direct Imports

### Before (Circular Dependencies)
```python
# orchestrator.py
from validators import ConstraintValidator  # Direct import

class Orchestrator:
    def __init__(self):
        self.validator = ConstraintValidator()  # Direct dependency
    
    def run_stage(self):
        # Direct call creates tight coupling
        result = self.validator.validate(data)
```

### After (Event-Driven)
```python
# orchestrator.py - No validator imports needed
from egw_query_expansion.core.event_schemas import ValidationRequestedEvent

class EventDrivenOrchestrator:
    def run_stage(self):
        # Publish event instead of direct call
        validation_event = ValidationRequestedEvent(payload)
        result = self.event_bus.publish(validation_event)
```

## Performance

The synchronous event bus maintains performance while adding benefits:

- **Minimal Overhead**: Events are processed immediately without queuing delays
- **Direct Handler Invocation**: No async/await overhead
- **Efficient Filtering**: Only relevant handlers process each event
- **Batching Support**: Multiple events can be processed together

## Configuration

Event bus behavior can be configured:

```python
# Custom event bus with larger queue
bus = SynchronousEventBus(max_queue_size=50000)

# Event metadata for tracking
metadata = EventMetadata(
    source_id="my_component",
    priority=EventPriority.HIGH,
    timeout_ms=10000
)

event = MyEvent(data, metadata=metadata)
```

## Error Handling

Robust error handling throughout the event system:

- Failed handlers don't stop event processing
- Detailed error reporting and logging
- Graceful degradation when components fail
- Recovery mechanisms for transient failures

## Monitoring and Debugging

Built-in monitoring capabilities:

```python
# Get event bus statistics
stats = event_bus.get_stats()
print(f"Total events: {stats['total_events_processed']}")

# Get event history
history = event_bus.get_event_history("validation.completed", limit=10)

# Get validator statistics  
adapter = get_validator_adapter("constraint_validator")
stats = adapter.get_validation_stats()
```

## Best Practices

1. **Use Typed Events**: Always use the defined event schemas for type safety
2. **Handle Errors Gracefully**: Implement proper error handling in event handlers
3. **Monitor Performance**: Track event processing times and success rates
4. **Validate Event Data**: Ensure event payloads contain expected data
5. **Document Custom Events**: Clearly document any custom event types
6. **Test Event Flows**: Write comprehensive tests for event publishing and handling
7. **Use Filters Judiciously**: Apply event filters to reduce unnecessary processing

## Troubleshooting

### Common Issues

**Events not being handled:**
- Verify handler is subscribed to correct event type
- Check that `can_handle()` returns True for the event
- Ensure handler is registered before event is published

**Validation failures:**
- Check validator adapter implementation
- Verify event payload contains expected data structure
- Review validation logic for correctness

**Performance issues:**
- Monitor event processing times with built-in metrics
- Check for inefficient event filters
- Consider batching related events

### Debug Mode

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Event bus operations will be logged in detail
```

This event-driven architecture provides a clean, scalable solution that eliminates circular dependencies while maintaining the synchronous execution flow required for deterministic pipeline processing.