# Audit Logging System for G_aggregation_reporting Stage

This document describes the centralized audit logging system implemented for the G_aggregation_reporting stage, providing comprehensive tracking of component execution traces, timing metrics, and error details.

## Overview

The audit logging system generates `_aggregation_audit.json` files that contain detailed information about the processing pipeline, including:

- Component execution traces with start/end times and duration
- Success/failure status for each component  
- Detailed error information including tracebacks
- Warning messages at both component and session levels
- Input/output data summaries for debugging
- Performance statistics and timing analysis

## Architecture

### Core Components

1. **AggregationAuditLogger** - Central logging coordinator
2. **ComponentTrace** - Individual component execution tracking
3. **AuditSession** - Complete session management with statistics
4. **JSON Output** - Structured audit files with consistent formatting

### File Structure

Audit files are written to: `canonical_flow/aggregation/<doc_stem>_aggregation_audit.json`

## Integration

### Meso Aggregator Integration

The `meso_aggregator.py` component automatically integrates audit logging:

```python
def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    from canonical_flow.G_aggregation_reporting.audit_logger import AggregationAuditLogger
    
    audit_logger = AggregationAuditLogger()
    doc_stem = context.get("doc_stem", "unknown") if context else "unknown"
    session_id = audit_logger.start_session(doc_stem, context)
    
    try:
        with audit_logger.trace_component("meso_aggregator", input_data=data) as trace:
            # Component processing logic...
            trace.complete_success({
                "meso_summary_items": len(meso_summary.get("items", {})),
                "coverage_matrix_components": len(coverage_matrix),
                # ... other metrics
            })
            return out
    except Exception as e:
        audit_logger.end_session(success=False)
        raise
    finally:
        if audit_logger.current_session:
            audit_logger.end_session(success=True)
```

### Report Compiler Integration  

The `report_compiler.py` component includes audit logging:

```python  
def process(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
    from canonical_flow.G_aggregation_reporting.audit_logger import AggregationAuditLogger
    
    audit_logger = AggregationAuditLogger()
    doc_stem = context.get("doc_stem", "unknown") if context else "unknown"
    
    if not audit_logger.current_session:
        session_id = audit_logger.start_session(doc_stem, context)
    
    try:
        with audit_logger.trace_component("report_compiler", input_data=data) as trace:
            # Report compilation logic...
            trace.complete_success({
                "reports_generated": reports_generated,
                "plan_name": plan_name,
                # ... other metrics  
            })
            return out
    except Exception as e:
        audit_logger.end_session(success=False)
        raise
    finally:
        if not context.get("preserve_audit_session"):
            audit_logger.end_session(success=True)
```

## Audit File Format

### Structure

```json
{
  "audit_metadata": {
    "version": "1.0",
    "generated_at": "2025-08-27T01:04:57.577490+00:00",
    "stage": "G_aggregation_reporting", 
    "logger_type": "AggregationAuditLogger"
  },
  "session_info": {
    "doc_stem": "document_name",
    "session_id": "document_name_1756256697",
    "start_time": "2025-08-27T01:04:57.565020+00:00",
    "end_time": "2025-08-27T01:04:57.577445+00:00",
    "duration_ms": 12.425,
    "overall_status": "success",
    "context": {
      "plan_name": "Development Plan Name"
    }
  },
  "component_traces": [
    {
      "component_name": "meso_aggregator",
      "start_time": "2025-08-27T01:04:57.565027+00:00", 
      "end_time": "2025-08-27T01:04:57.577422+00:00",
      "duration_ms": 12.395,
      "status": "success",
      "error_details": null,
      "warnings": [],
      "input_summary": {
        "type": "dict",
        "keys_count": 1,
        "top_keys": ["cluster_audit"],
        "total_size_est": 1024
      },
      "output_summary": {
        "meso_summary_items": 25,
        "coverage_matrix_components": 10,
        "total_questions": 25
      },
      "memory_usage_mb": 45.2
    }
  ],
  "session_warnings": [],
  "summary_statistics": {
    "total_components": 2,
    "successful_components": 2,
    "failed_components": 0,
    "components_with_warnings": 0,
    "total_warnings": 0,
    "average_component_duration_ms": 12.395,
    "longest_component": "meso_aggregator",
    "shortest_component": "report_compiler"
  }
}
```

### Key Fields

- **audit_metadata**: Version and generation information
- **session_info**: Overall session timing and context
- **component_traces**: Detailed per-component execution data
- **session_warnings**: System-level warnings and issues
- **summary_statistics**: Aggregated performance metrics

## Usage Examples

### Basic Component Tracing

```python
from canonical_flow.G_aggregation_reporting.audit_logger import AggregationAuditLogger

logger = AggregationAuditLogger()
session_id = logger.start_session("my_document", {"plan_name": "Test Plan"})

with logger.trace_component("my_component", input_data) as trace:
    # Component processing
    result = process_data(input_data)
    trace.complete_success({"records_processed": len(result)})

audit_file_path = logger.end_session(success=True)
```

### Error Handling

```python
try:
    with logger.trace_component("risky_component") as trace:
        result = risky_operation()
        trace.complete_success({"success": True})
except Exception as e:
    # Error automatically captured by trace
    logger.end_session(success=False)
    raise
```

### Warning Logging

```python
# Component-level warning
logger.log_component_warning("meso_aggregator", "Missing data for cluster C3")

# Session-level warning  
logger.log_session_warning("Processing completed with data quality issues")
```

## Performance Monitoring

The audit system captures comprehensive timing information:

- **Component Duration**: Individual component execution time
- **Session Duration**: Total processing time
- **Performance Statistics**: Min/max/average component times
- **Memory Usage**: Optional memory tracking per component

## Debugging Support

Audit files provide extensive debugging information:

- **Input/Output Summaries**: Data structure overviews
- **Error Details**: Full exception information and tracebacks
- **Warning Context**: Detailed warning messages with timestamps
- **Execution Flow**: Component execution order and timing

## Configuration

### Output Directory

By default, audit files are written to `canonical_flow/aggregation/`. This can be configured:

```python
logger = AggregationAuditLogger(output_dir="custom/audit/path")
```

### Memory Tracking

Memory usage tracking can be enabled per component:

```python
with logger.trace_component("memory_intensive", capture_memory=True) as trace:
    # Processing that may use significant memory
    pass
```

## File Management

- **Encoding**: All files use UTF-8 encoding
- **Format**: Pretty-printed JSON with 2-space indentation  
- **Naming**: `<doc_stem>_aggregation_audit.json`
- **Overwrite**: Files are overwritten if they exist

## Integration Guidelines

1. **Initialize Early**: Create logger at start of processing pipeline
2. **Use Context Managers**: Always use `trace_component()` context manager
3. **Handle Errors**: Ensure `end_session()` is called even on errors
4. **Log Warnings**: Use appropriate warning levels for issues
5. **Provide Context**: Include meaningful doc_stem and context data

## Testing

The audit logging system includes comprehensive tests:

- `test_audit_logger.py` - Basic functionality testing
- `test_simple_audit.py` - Integration testing
- `test_report_compiler_integration.py` - Component integration
- `demo_integration.py` - Full workflow demonstration

## Benefits

1. **Debugging**: Detailed error traces and component timing
2. **Performance Analysis**: Component-level performance metrics  
3. **Quality Assurance**: Warning tracking and data validation
4. **Compliance**: Complete audit trail for processing
5. **Monitoring**: Session-level health and status tracking

The audit logging system provides comprehensive observability into the G_aggregation_reporting stage without impacting normal execution flow, enabling effective debugging, performance optimization, and quality assurance.