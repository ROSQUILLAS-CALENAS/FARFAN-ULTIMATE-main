# EGW Query Expansion Audit Logging System

This document describes the comprehensive audit logging system implemented for the EGW Query Expansion analysis components (13A-20A).

## Overview

The audit logging system provides complete execution tracing, performance monitoring, and error handling for all 9 analysis_nlp components. It captures detailed metrics including start/end times, input/output schemas, memory usage, CPU utilization, and complete error details with stack traces.

## Architecture

### Core Components

1. **AuditLogger** (`canonical_flow/analysis/audit_logger.py`)
   - Centralized logging of all component executions
   - Automatic performance metrics collection
   - Input/output schema analysis
   - Error handling with stack trace capture
   - Memory and CPU usage tracking

2. **AuditValidator** (`canonical_flow/analysis/audit_validation.py`) 
   - Validation of audit data completeness
   - Component execution trace verification
   - Integrity checking of audit records
   - Comprehensive reporting

3. **Demo System** (`canonical_flow/analysis/demo_audit_system.py`)
   - Complete demonstration of audit functionality
   - Simulation of all 9 components
   - Example error handling scenarios
   - Validation testing

## Component Integration

### Supported Components

The audit system tracks all 9 analysis_nlp components:

| Code | Component Name | Module |
|------|----------------|--------|
| 13A | adaptive_analyzer | adaptive_analyzer.py |
| 14A | question_analyzer | question_analyzer.py |
| 15A | implementacion_mapeo | implementacion_mapeo.py |
| 16A | evidence_processor | evidence_processor.py |
| 17A | extractor_evidencias_contextual | EXTRACTOR DE EVIDENCIAS CONTEXTUAL.py |
| 18A | evidence_validation_model | evidence_validation_model.py |
| 19A | evaluation_driven_processor | evaluation_driven_processor.py |
| 20A | dnp_alignment_adapter | dnp_alignment_adapter.py |

### Integration Pattern

Each component's `process()` method has been enhanced with audit logging:

```python
# Import audit logger
from canonical_flow.analysis.audit_logger import get_audit_logger

def process(self, input_data):
    """Component process method with audit logging."""
    audit_logger = get_audit_logger() if get_audit_logger else None
    
    if audit_logger:
        with audit_logger.audit_component_execution("16A", input_data) as audit_ctx:
            result = self._process_internal(input_data)
            audit_ctx.set_output(result)
            return result
    else:
        return self._process_internal(input_data)
```

## Audit Data Format

### JSON Structure

The audit system generates a standardized `_audit.json` file with the following structure:

```json
{
  "audit_metadata": {
    "generated_timestamp": "2025-01-20T23:45:12.123456+00:00",
    "execution_id": "EGW_DEMO_20250120_234512",
    "total_components_invoked": 9,
    "total_events_recorded": 18,
    "audit_format_version": "1.0"
  },
  "system_events": [...],
  "components": {
    "16A": {
      "component_name": "evidence_processor",
      "events": [
        {
          "event_id": "16A_0001_1737413112150",
          "event_type": "component_start",
          "timestamp": "2025-01-20T23:45:12.150000+00:00",
          "input_schema": {...},
          "performance_metrics": {...}
        }
      ],
      "metrics_summary": {
        "total_events": 2,
        "total_duration_seconds": 0.2,
        "success_count": 1,
        "error_count": 0,
        "avg_memory_usage_mb": 187.4
      }
    }
  },
  "execution_summary": {
    "total_execution_time": 0.45,
    "components_with_errors": [],
    "successful_components": ["13A", "14A", "16A"]
  }
}
```

### Event Types

- **component_start**: Component execution begins
- **component_end**: Component execution completes successfully  
- **component_error**: Component execution fails with error
- **validation_step**: Validation checkpoints
- **performance_metric**: Performance milestone events

### Performance Metrics

Each component execution captures:

```python
{
  "duration_seconds": 0.2,
  "cpu_percent": 31.7,
  "memory_metrics": {
    "rss_mb": 187.4,        # Resident Set Size
    "vms_mb": 1089.2,       # Virtual Memory Size 
    "percent": 4.1,         # Memory usage percentage
    "available_mb": 8141.8  # Available system memory
  },
  "io_operations": {...}    # Additional metrics
}
```

### Input/Output Schemas

The system automatically analyzes and records data structures:

```python
{
  "data_type": "dict",
  "schema_keys": ["chunks_count", "question_id", "dimension"],
  "data_size_bytes": 198,
  "record_count": 1,
  "validation_status": "captured",
  "sample_data": {
    "chunks_count": 25,
    "question_id": "Q001"
  }
}
```

## Error Handling

### Error Details

Complete error information is captured:

```python
{
  "error_type": "ValueError",
  "error_message": "Invalid input format",
  "stack_trace": "Traceback (most recent call last):\n...",
  "component_context": {
    "component_code": "16A",
    "input_schema": {...}
  },
  "recovery_attempted": false,
  "recovery_successful": false
}
```

### Graceful Degradation

The audit system is designed to never break the pipeline:
- Falls back gracefully when psutil is unavailable
- Continues execution if audit logging fails
- Provides meaningful defaults for missing metrics

## Usage Examples

### Basic Usage

```python
from canonical_flow.analysis.audit_logger import get_audit_logger

# Get the global audit logger instance
audit_logger = get_audit_logger()

# Start an execution trace
audit_logger.start_execution("PIPELINE_RUN_001")

# Audit a component execution
with audit_logger.audit_component_execution("16A", input_data) as audit_ctx:
    result = process_evidence(input_data)
    audit_ctx.set_output(result)
    audit_ctx.add_metric("evidence_found", len(result))

# Save audit data
audit_file_path = audit_logger.save_audit_file()
```

### Validation

```python
from canonical_flow.analysis.audit_validation import print_audit_validation_report

# Validate audit completeness
expected_components = {"13A", "14A", "16A", "18A"}
validation_passed = print_audit_validation_report(
    audit_file_path="canonical_flow/analysis/_audit.json",
    invoked_components=expected_components
)
```

### Custom Audit File Location

```python
from canonical_flow.analysis.audit_logger import AuditLogger, set_audit_logger

# Create custom audit logger
custom_logger = AuditLogger("custom/path/audit.json")
set_audit_logger(custom_logger)
```

## Validation Features

### Completeness Checks

The validator ensures:
- All expected components have execution traces
- Each component has balanced start/end events  
- Performance metrics are recorded
- Input/output schemas are captured
- Error handling is properly logged

### Validation Report

```
EGW QUERY EXPANSION AUDIT VALIDATION REPORT
============================================================
Generated: 2025-01-20T23:45:12.123456+00:00
Audit File: canonical_flow/analysis/_audit.json

Overall Validation Status: ✓ PASSED

✓ File Exists
✓ Structure Valid
✓ Components Complete
✓ Execution Summary Valid

COMPONENT STATUS:
  ✓ 13A (adaptive_analyzer)
  ✓ 14A (question_analyzer)
  ✓ 16A (evidence_processor)
  ✗ 18A (evidence_validation_model)
      • Component 18A has unbalanced events: 1 starts, 0 ends
```

## Dependencies

### Required
- Python 3.7+
- Standard library modules: json, time, datetime, pathlib, dataclasses, enum, contextlib

### Optional  
- `psutil`: For advanced memory and CPU metrics (graceful fallback if not available)

### Installation

The audit system is self-contained within the canonical_flow/analysis/ directory. No additional installation is required beyond the existing project dependencies.

## Performance Impact

The audit system is designed for minimal performance overhead:
- Lightweight context managers
- Efficient JSON serialization
- Optional metrics collection
- Asynchronous-friendly design
- Fallback implementations for resource-constrained environments

Typical overhead per component execution: < 5ms

## Files Structure

```
canonical_flow/analysis/
├── __init__.py                    # Module initialization
├── audit_logger.py               # Core audit logging functionality
├── audit_validation.py           # Audit data validation
├── demo_audit_system.py         # Complete demo and testing
├── sample_audit_output.json     # Example audit output
└── _audit.json                   # Generated audit data (runtime)
```

## Testing

### Demo Execution

Run the complete demo to verify functionality:

```bash
python3 canonical_flow/analysis/demo_audit_system.py
```

### Manual Testing  

```python
# Simple test
from canonical_flow.analysis.audit_logger import AuditLogger

logger = AuditLogger()
logger.start_execution("TEST_001")

with logger.audit_component_execution("16A", {"test": "data"}) as ctx:
    ctx.set_output({"result": "success"})

logger.save_audit_file()
```

## Best Practices

1. **Always use context managers** for component execution auditing
2. **Set meaningful output data** using `audit_ctx.set_output()`
3. **Add relevant metrics** with `audit_ctx.add_metric()`
4. **Validate audit data** after pipeline execution
5. **Use consistent component codes** (13A-20A)
6. **Handle audit failures gracefully** in production code

## Troubleshooting

### Common Issues

1. **Missing psutil**: Install with `pip install psutil` or use fallback mode
2. **Permission errors**: Ensure write access to audit directory
3. **Large audit files**: Consider rotating or compressing old audit files
4. **Memory issues**: Reduce audit detail level for large-scale processing

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check audit file generation:

```python
from pathlib import Path
audit_path = Path("canonical_flow/analysis/_audit.json")
print(f"Audit file exists: {audit_path.exists()}")
print(f"Audit file size: {audit_path.stat().st_size if audit_path.exists() else 0} bytes")
```

## Future Enhancements

- Integration with external monitoring systems (Prometheus, Grafana)
- Real-time audit streaming
- Distributed audit collection
- Advanced anomaly detection in audit patterns
- Audit data compression and archiving
- Performance regression detection

## License

This audit system is part of the EGW Query Expansion project and follows the same licensing terms.