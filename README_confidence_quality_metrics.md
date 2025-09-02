# Confidence and Quality Metrics Propagation System

This document describes the comprehensive confidence and quality metrics system implemented for the EGW Query Expansion project. The system provides hierarchical metrics that flow from individual question evaluations through dimension, point, meso, and macro aggregation levels.

## Overview

The confidence and quality metrics system consists of:

1. **Evidence density calculation** - Measures evidence count per question
2. **Coverage validation** - Tracks answered vs total questions ratios
3. **Model agreement scoring** - Assesses consistency across multiple analysis models
4. **Quality metric aggregation** - Combines evidence credibility, temporal recency, and authority rankings
5. **Hierarchical propagation** - Flows metrics through all aggregation levels
6. **Artifact integration** - Adds confidence_score and quality_score fields to all JSON outputs

## Key Components

### 1. Evidence Density Metrics

**Formula**: `min(evidence_count / (question_count * optimal_evidence_count), 1.0)`

- Penalizes insufficient evidence per question
- Rewards comprehensive evidence coverage
- Optimal evidence count defaults to 5 per question
- Minimum threshold of 3 evidence items per question

```python
from confidence_quality_metrics import EvidenceDensityMetrics

metrics = EvidenceDensityMetrics(
    total_evidence_count=15,
    question_count=3,
    optimal_evidence_count=5
)
# Result: density_ratio = 1.0 (perfect coverage)
```

### 2. Coverage Validation Metrics

**Formula**: `(answered_questions / total_questions) * (1 - critical_gaps * gap_penalty_factor)`

- Base score from answer completion ratio
- Penalty for critical coverage gaps
- Gap penalty factor defaults to 0.1 per critical gap

```python
from confidence_quality_metrics import CoverageValidationMetrics

metrics = CoverageValidationMetrics(
    answered_questions=8,
    total_questions=10,
    critical_gaps=1,
    gap_penalty_factor=0.1
)
# Result: coverage_ratio = 0.8 * (1 - 1 * 0.1) = 0.72
```

### 3. Model Agreement Scoring

**Formula**: `exp(-variance / agreement_threshold) * (1 - outlier_penalty)`

- Higher agreement for consistent model outputs
- Exponential penalty for high variance
- Additional penalty for significant outliers
- Agreement threshold defaults to 0.1

```python
from confidence_quality_metrics import ModelAgreementMetrics

metrics = ModelAgreementMetrics(
    model_scores=[0.8, 0.82, 0.78, 0.81],
    agreement_threshold=0.1
)
# Result: High consensus_score due to low variance
```

### 4. Quality Components

**Formula**: `credibility_score * 0.4 + recency_weight * 0.3 + authority_ranking * 0.3`

- **Evidence Credibility** (40% weight): Based on source quality and validation
- **Temporal Recency** (30% weight): Exponential decay over time
- **Authority Ranking** (30% weight): Source authority scores

```python
from confidence_quality_metrics import QualityComponents

quality = QualityComponents(
    credibility_score=0.8,
    recency_weight=0.6, 
    authority_ranking=0.9
)
# Result: composite_score = 0.77
```

## Hierarchical Aggregation Levels

### Question Level
- Individual question confidence and quality calculation
- Considers evidence density, coverage, model agreement, and quality components
- Base level for all propagation

### Dimension Level  
- Aggregates multiple question scores within a dimension
- Uses confidence-weighted averaging
- Combines evidence gaps and uncertainty factors

### Point Level (Decálogo)
- Aggregates dimension scores within a Decálogo point
- Uses confidence-weighted averaging
- Maintains traceability of gaps and uncertainty

### Meso Level
- Aggregates point scores using harmonic mean
- Penalizes low scores more heavily
- Provides mid-level operational analysis

### Macro Level
- Aggregates meso scores using geometric mean
- Provides balanced high-level strategic overview
- Final aggregation level

## Propagation Rules

### Confidence Score Propagation

1. **Question → Dimension**: Confidence-weighted average
2. **Dimension → Point**: Confidence-weighted average  
3. **Point → Meso**: Harmonic mean (penalizes low scores)
4. **Meso → Macro**: Geometric mean (balanced aggregation)

### Quality Score Propagation

Similar hierarchical rules with appropriate weighting for each level.

### Evidence Gaps and Uncertainty Factors

- Aggregated and deduplicated at each level
- Maintained for transparency and debugging
- Include factors like:
  - `insufficient_evidence_count`
  - `missing_answer`
  - `coverage_gap`
  - `model_disagreement`

## Artifact Integration

The system integrates seamlessly with existing artifact generation:

### Question-Level Artifacts
```json
{
  "question": "What is the impact?",
  "confidence_score": 0.75,
  "quality_score": 0.82,
  "metrics_metadata": {
    "evidence_density": {"total_evidence": 4, "density_ratio": 0.8},
    "coverage_validation": {"coverage_ratio": 1.0},
    "model_agreement": {"consensus_score": 0.85},
    "evidence_gaps": [],
    "uncertainty_factors": [],
    "calculation_timestamp": "2024-01-01T00:00:00Z"
  }
}
```

### Dimension-Level Artifacts
```json
{
  "dimension": "DE-1",
  "confidence_score": 0.71,
  "quality_score": 0.78,
  "metrics_metadata": {
    "question_count": 5,
    "evidence_density_ratio": 0.76,
    "coverage_ratio": 0.8,
    "evidence_gaps": ["insufficient_evidence_count"],
    "uncertainty_factors": ["low_evidence_density"]
  }
}
```

## Usage Examples

### Basic Usage

```python
from confidence_quality_metrics import (
    ConfidenceQualityCalculator,
    ArtifactMetricsIntegrator
)

# Calculate question-level metrics
calculator = ConfidenceQualityCalculator()
question_data = {
    'question': 'What is the effectiveness?',
    'evidence': [
        {'text': 'Evidence 1', 'source_credibility': 0.8},
        {'text': 'Evidence 2', 'source_credibility': 0.7}
    ],
    'answer': 'Positive results observed',
    'nlp_score': 0.8
}

metrics = calculator.calculate_question_level_metrics(question_data)
print(f"Confidence: {metrics.confidence_score:.3f}")
print(f"Quality: {metrics.quality_score:.3f}")

# Integrate with artifacts
integrator = ArtifactMetricsIntegrator()
enhanced_artifact = integrator.add_metrics_to_question_artifact(question_data)
```

### Convenience Functions

```python
from confidence_quality_metrics import (
    calculate_question_confidence,
    calculate_question_quality,
    validate_metrics_bounded
)

# Quick confidence calculation
confidence = calculate_question_confidence(question_data)

# Quick quality calculation  
quality = calculate_question_quality(question_data)

# Ensure bounds
safe_score = validate_metrics_bounded(score, 'metric_name')
```

## Score Bounds and Validation

All confidence and quality scores are strictly bounded between 0.0 and 1.0:

- **0.0**: No confidence/quality (missing evidence, no answers)
- **0.1-0.3**: Low confidence/quality (insufficient evidence, gaps)
- **0.4-0.6**: Medium confidence/quality (adequate coverage)
- **0.7-0.8**: High confidence/quality (good evidence, consistency)
- **0.9-1.0**: Very high confidence/quality (comprehensive, validated)

The `validate_metrics_bounded()` function ensures all scores remain within bounds and logs warnings for out-of-bounds values.

## Integration Points

The system integrates with existing components:

1. **Question Analyzer** - Adds confidence/quality to question analysis results
2. **Evidence Processor** - Enhances evidence processing with quality metrics  
3. **Meso Aggregator** - Includes meso-level confidence/quality calculation
4. **Report Compiler** - Adds macro-level metrics to compiled reports

## Testing

Run the comprehensive test suite:

```bash
python3 test_confidence_quality_metrics.py
```

Tests cover:
- Individual component calculations
- Hierarchical propagation
- Artifact integration
- Bounds validation
- Edge cases and error handling

## Configuration

Key parameters can be configured:

```python
calculator = ConfidenceQualityCalculator()
calculator.temporal_decay_days = 365  # Days for temporal decay
calculator.authority_sources = {'source_id': 0.9}  # Authority mappings
calculator.credibility_thresholds = {  # Credibility mappings
    'high': 0.8,
    'medium': 0.6, 
    'low': 0.3
}
```

## Mathematical Foundations

The system uses well-established mathematical principles:

- **Exponential decay** for temporal recency weighting
- **Harmonic mean** for penalizing low scores at meso level
- **Geometric mean** for balanced aggregation at macro level
- **Confidence-weighted averaging** for maintaining score quality
- **Variance-based agreement** for model consensus assessment

All formulas are documented in the code and this documentation for transparency and reproducibility.