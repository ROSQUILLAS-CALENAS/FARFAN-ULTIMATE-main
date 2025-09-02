# Auto-Enhancement Orchestration System

A comprehensive auto-enhancement orchestration system that performs preflight validation, auto-deactivation monitoring, and provenance tracking for calibration safety governance.

## Overview

The system consists of four main components:

1. **Preflight Validator** - Comprehensive validation before enhancement activation
2. **Auto-Deactivation Monitor** - Stability drift and performance regression monitoring
3. **Provenance Tracker** - Enhancement metadata and audit trail generation
4. **Enhancement Orchestrator** - Main coordination system

## Components

### Preflight Validator (`preflight_validator.py`)

Performs comprehensive validation including:

- **Input Schema Verification**: Validates enhancement requests against predefined JSON schemas
- **Library Version Compatibility**: Checks library versions against `requirements.txt`
- **Threshold Satisfaction**: Validates current metrics against unified thresholds configuration

Key features:
- Configurable validation thresholds via `thresholds.json`
- Detailed validation scoring and error reporting
- Support for multiple validation check types
- Integration with requirements.txt for dependency validation

### Auto-Deactivation Monitor (`auto_deactivation_monitor.py`)

Tracks system stability and automatically deactivates enhancements when thresholds are exceeded:

- **Stability Drift Analysis**: Score variance and trend analysis
- **Evidence Quality Tracking**: Quality degradation detection
- **Performance Regression Detection**: Multi-metric performance monitoring
- **Auto-Deactivation Logic**: Configurable triggers and cooldown periods

Key features:
- Consecutive violation tracking
- Multiple severity levels (minor, major, critical)
- Configurable cooldown periods
- Comprehensive monitoring summary generation

### Provenance Tracker (`provenance_tracker.py`)

Generates comprehensive audit trails and enhancement metadata:

- **Enhancement Lifecycle Tracking**: Complete state transitions
- **Activation Criteria Management**: Weighted criteria evaluation
- **Performance Impact Analysis**: Baseline vs current metrics
- **Audit Trail Generation**: Timestamped lifecycle events

Key features:
- JSON metadata artifact generation (`enhancement_metadata.json`)
- Complete audit trail export
- Performance impact trend analysis
- Configurable activation criteria weighting

### Enhancement Orchestrator (`orchestrator.py`)

Main coordination system that integrates all components:

- **Request Processing**: Enhancement request validation and queuing
- **Continuous Monitoring**: Background monitoring of active enhancements
- **Lifecycle Management**: Complete enhancement lifecycle orchestration
- **Reporting**: Comprehensive system status and recommendations

Key features:
- Multiple operation modes (automatic, semi-automatic, manual)
- Concurrent enhancement limit management
- Integrated monitoring and alerting
- Comprehensive report generation

## Configuration

### Thresholds Configuration (`thresholds.json`)

Unified configuration for all validation and monitoring thresholds:

```json
{
  "validation": {
    "schema_compliance": {
      "minimum_fields_present": 0.95,
      "data_type_accuracy": 0.98
    },
    "threshold_satisfaction": {
      "mandatory_clause_compliance": 1.0,
      "proxy_score_minimum": 0.7,
      "confidence_alpha": 0.95
    }
  },
  "stability_monitoring": {
    "score_variance": {
      "maximum_stddev": 0.15,
      "stability_coefficient": 0.8
    },
    "performance_regression": {
      "response_time_increase": 1.5,
      "accuracy_degradation": 0.05
    }
  },
  "auto_deactivation": {
    "triggers": {
      "stability_breach": {
        "consecutive_violations": 3
      }
    },
    "cooldown_periods": {
      "critical_deactivation": "PT24H"
    }
  }
}
```

## Usage

### Basic Usage

```python
from calibration_safety_governance import EnhancementOrchestrator, OrchestrationConfig, OrchestrationMode

# Initialize orchestrator
config = OrchestrationConfig(
    mode=OrchestrationMode.AUTOMATIC,
    monitoring_interval_seconds=30,
    max_concurrent_enhancements=5
)

orchestrator = EnhancementOrchestrator(config)

# Submit enhancement request
result = orchestrator.submit_enhancement_request(
    enhancement_id="adaptive_scoring_v1",
    enhancement_type="adaptive_scoring",
    description="Adaptive scoring with neural networks",
    configuration={
        "model_type": "neural_network",
        "learning_rate": 0.001
    },
    activation_criteria=[
        {
            "type": "performance_threshold",
            "description": "Performance requirement",
            "threshold": 0.85
        }
    ],
    baseline_metrics={
        "accuracy": 0.82,
        "response_time": 0.3
    }
)

# Start monitoring
orchestrator.start_monitoring()

# Get status
status = orchestrator.get_orchestration_status()
print(f"Active enhancements: {status['current_state']['active_enhancements']}")

# Generate report
report = orchestrator.generate_orchestration_report()

# Export metadata
orchestrator.export_enhancement_metadata()

# Shutdown
orchestrator.shutdown()
```

### Individual Component Usage

```python
# Preflight validation
from calibration_safety_governance import PreflightValidator

validator = PreflightValidator()
validation_result = validator.run_comprehensive_validation(
    input_data=enhancement_request,
    schema_type="enhancement_request",
    current_metrics=system_metrics
)

# Auto-deactivation monitoring
from calibration_safety_governance import AutoDeactivationMonitor

monitor = AutoDeactivationMonitor()
monitoring_result = monitor.monitor_enhancement(
    enhancement_id="test_enhancement",
    performance_metrics=current_performance,
    evidence_quality=quality_metrics,
    score=current_score
)

# Provenance tracking
from calibration_safety_governance import ProvenanceTracker

tracker = ProvenanceTracker()
metadata = tracker.create_enhancement_metadata(
    enhancement_id="test_enhancement",
    enhancement_type="adaptive_scoring",
    description="Test enhancement",
    configuration={},
    activation_criteria=[],
    baseline_metrics={}
)
```

## Generated Artifacts

### Enhancement Metadata JSON (`enhancement_metadata.json`)

Complete enhancement metadata including:

- Enhancement lifecycle events with timestamps
- Activation criteria satisfaction tracking
- Performance impact metrics and trends
- Complete audit trail with triggering conditions
- Configuration and dependency tracking

### Directory Structure

```
calibration_safety_governance/
├── __init__.py
├── preflight_validator.py
├── auto_deactivation_monitor.py
├── provenance_tracker.py
├── orchestrator.py
├── thresholds.json
├── metadata/
│   ├── enhancement_adaptive_scoring_v1.json
│   ├── enhancement_dynamic_thresholding_v1.json
│   └── ...
├── enhancement_metadata.json
└── README.md
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest test_orchestration_system.py -v
```

## Demo

Run the demonstration script to see all components in action:

```bash
python demo_orchestration.py
```

The demo will:

1. Demonstrate preflight validation with valid/invalid requests
2. Show auto-deactivation monitoring with simulated degradation
3. Demonstrate provenance tracking lifecycle management
4. Run full orchestration system with multiple enhancements
5. Generate comprehensive reports and export metadata

## Integration

The system integrates with:

- **Requirements validation**: Uses `requirements.txt` for dependency checks
- **Unified thresholds**: Single `thresholds.json` for all configuration
- **Audit logging**: Comprehensive logging and audit trail generation
- **Performance monitoring**: Integration with system performance metrics
- **Safety governance**: Compliance with safety and governance requirements

## Key Features

### Comprehensive Validation
- Multi-layer preflight validation
- Schema validation with detailed error reporting
- Library compatibility verification
- Threshold satisfaction validation

### Intelligent Monitoring
- Real-time stability drift detection
- Evidence quality degradation tracking
- Performance regression analysis
- Automatic deactivation with cooldown periods

### Complete Provenance
- Full lifecycle event tracking
- Performance impact analysis
- Audit trail generation
- Metadata export capabilities

### Orchestrated Management
- Multiple operation modes
- Concurrent enhancement management
- Integrated monitoring and reporting
- Graceful shutdown and recovery

This system ensures safe, monitored, and fully auditable enhancement operations with comprehensive validation, monitoring, and provenance tracking capabilities.