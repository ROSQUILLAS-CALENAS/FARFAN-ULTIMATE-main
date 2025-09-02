# Calibration Dashboard Artifact Generation System

## Overview

The Calibration Dashboard system provides standardized JSON report generation for pipeline calibration monitoring across three major stages: retrieval, confidence estimation, and aggregation. Each stage generates deterministic artifacts with quality gates, coverage statistics, and drift detection capabilities.

## Architecture

### Core Components

1. **CalibrationArtifact (Base Class)** - Common fields for all calibration reports
2. **RetrievalCalibrationArtifact** - Retrieval stage with temperature stability metrics  
3. **ConfidenceCalibrationArtifact** - Confidence stage with interval coverage metrics
4. **AggregationCalibrationArtifact** - Aggregation stage with convergence metrics
5. **CalibrationDashboard** - Central management system for artifact generation

### Canonical File Structure

```
canonical_flow/calibration/
├── retrieval_calibration.json     # Deterministic retrieval stage report
├── confidence_calibration.json    # Deterministic confidence stage report  
├── aggregation_calibration.json   # Deterministic aggregation stage report
└── README.md                      # This documentation
```

## Standard Fields

Every calibration artifact contains these core fields:

- **calibration_quality_score**: Overall calibration quality (0.0-1.0)
- **coverage_percentage**: Coverage of validation data (0-100%)
- **decisions_made**: Number of decisions processed
- **quality_gate_status**: "PASS", "WARNING", or "FAIL"
- **timestamp**: ISO format generation timestamp
- **stage_version**: Semantic version of stage implementation
- **calibration_parameters**: Configuration used for calibration

## Stage-Specific Metrics

### Retrieval Stage
- **temperature_stability**: Thermal calibration stability (0.0-1.0)
- **entropy_calibration_score**: Entropy-based calibration quality
- **retrieval_precision/recall**: Standard IR metrics
- **fusion_coherence**: Multi-retriever fusion quality
- **temperature_parameter**: Calibration temperature setting

### Confidence Stage  
- **interval_coverage**: Prediction interval coverage (0.0-1.0)
- **calibration_error**: Difference between predicted and actual confidence
- **sharpness_score**: Precision of confidence intervals
- **reliability_score**: Consistency of confidence estimates
- **prediction_efficiency**: Computational efficiency metric

### Aggregation Stage
- **convergence_rate**: Mathematical convergence quality (0.0-1.0)
- **aggregation_stability**: Stability across iterations
- **consensus_score**: Agreement between aggregation methods
- **uncertainty_quantification_quality**: Quality of uncertainty estimates
- **spectral_gap**: Eigenvalue separation for stability analysis

## Quality Gate System

### Thresholds

Quality gates use configurable thresholds per stage:

**Retrieval**: calibration_quality_score ≥ 0.7, coverage_percentage ≥ 80%
**Confidence**: calibration_quality_score ≥ 0.75, interval_coverage ≥ 0.9  
**Aggregation**: calibration_quality_score ≥ 0.8, convergence_rate ≥ 0.9

### Status Logic

- **PASS**: All metrics meet thresholds
- **WARNING**: Some metrics below threshold but above 70% of threshold
- **FAIL**: Any metric below 70% of threshold

## Usage Examples

### Basic Artifact Generation

```python
from canonical_flow.calibration import CalibrationDashboard

dashboard = CalibrationDashboard()

# Generate retrieval stage artifact
retrieval_artifact = dashboard.generate_retrieval_artifact(
    calibration_quality_score=0.85,
    coverage_percentage=82.5,
    decisions_made=150,
    temperature_stability=0.92,
    entropy_calibration_score=0.78,
    retrieval_precision=0.72,
    retrieval_recall=0.68,
    fusion_coherence=0.85
)

print(f"Quality Gate: {retrieval_artifact.quality_gate_status}")
```

### Drift Detection

```python
# Detect calibration drift over time
drift_analysis = dashboard.detect_calibration_drift(
    current_artifact,
    historical_artifacts,
    drift_threshold=0.1
)

if drift_analysis['drift_detected']:
    print("⚠️ Calibration drift detected!")
```

### Summary Dashboard

```python
# Get overall calibration status
summary = dashboard.get_calibration_summary()
print(f"Overall Status: {summary['overall_status']}")
```

## Integration Points

### Pipeline Integration

The calibration system integrates with existing pipeline stages:

1. **Retrieval Stage**: Hook into `canonical_flow/R_search_retrieval/`
2. **Confidence Stage**: Hook into conformal prediction systems
3. **Aggregation Stage**: Hook into `canonical_flow/G_aggregation_reporting/`

### Quality Gate Enforcement

Quality gates can block pipeline progression:

```python
if not artifact.check_quality_gates():
    raise PipelineQualityGateFailure(
        f"Quality gate failed: {artifact.quality_gate_status}"
    )
```

## Drift Detection Algorithm

The system uses statistical analysis to detect calibration drift:

1. **Z-score Analysis**: Compare current metrics to historical distribution
2. **Threshold**: Drift detected if Z-score > 2.0 (2 standard deviations)
3. **Minimum History**: Requires ≥2 historical artifacts for comparison

## File Format Specification

Each JSON artifact follows this structure:

```json
{
  "calibration_quality_score": 0.85,
  "coverage_percentage": 82.5,
  "decisions_made": 150,
  "quality_gate_status": "PASS",
  "timestamp": "2025-08-27T03:32:01.611244",
  "stage_version": "1.0.0",
  "calibration_parameters": {
    "temperature_parameter": 1.0,
    "fusion_weights": [0.33, 0.33, 0.34]
  },
  "quality_thresholds": {
    "calibration_quality_score": 0.7,
    "coverage_percentage": 80.0
  },
  "temperature_stability": 0.92,
  // ... stage-specific fields
}
```

## Dependencies

- Python 3.7+
- json (built-in)
- dataclasses (built-in)
- pathlib (built-in)
- datetime (built-in)
- Optional: numpy (for advanced drift detection)

## Testing

Run the test suite:

```bash
python3 test_calibration_dashboard.py
```

Expected outputs:
- All artifact files created in `canonical_flow/calibration/`
- Quality gates evaluated correctly
- JSON validation passes
- File deterministic naming verified