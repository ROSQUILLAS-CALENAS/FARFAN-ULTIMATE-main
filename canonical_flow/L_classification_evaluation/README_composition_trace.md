# Point Composition Trace System

The Point Composition Trace System provides complete mathematical transparency and audit trails for Decálogo point score calculations in the L_classification_evaluation stage.

## Overview

This system captures detailed composition artifacts for each of the 10 Decálogo points, including:
- Dimension scores (DE-1, DE-2, DE-3, DE-4) 
- Applied weights (Decálogo weights: DE-1: 0.30, DE-2: 0.25, DE-3: 0.25, DE-4: 0.20)
- Weighted sum calculations 
- Final point scores
- Capping policy enforcement
- Complete mathematical provenance

## File Structure

Trace files are saved in: `canonical_flow/classification/<doc_id>/P{n}_composition.json`

Where:
- `<doc_id>` is the document identifier
- `{n}` is the Decálogo point number (1-10)

## JSON Structure

Each composition trace file contains:

```json
{
  "metadata": {
    "point_id": 1,
    "document_id": "DOC_001",
    "timestamp": "2025-08-27T03:16:39.367251+00:00",
    "system_version": "1.0.0",
    "deterministic": true
  },
  "scoring_configuration": {
    "base_scores": {"Sí": 1.0, "Parcial": 0.5, "No": 0.0, "NI": 0.0},
    "decalogo_weights": {"DE-1": 0.3, "DE-2": 0.25, "DE-3": 0.25, "DE-4": 0.2},
    "evidence_multiplier_range": [0.5, 1.2],
    "precision": 4
  },
  "dimension_details": [
    {
      "dimension_id": "DE-1",
      "total_questions": 12,
      "aggregated_score": 0.8456,
      "question_breakdown": [
        {
          "question_id": "DE1_Q1",
          "response": "Sí",
          "base_score": 1.0,
          "evidence_completeness": 0.9,
          "page_reference_quality": 1.0,
          "evidence_multiplier": 1.151,
          "final_score": 1.151
        }
      ]
    }
  ],
  "composition_calculation": {
    "point_id": 1,
    "dimension_scores": {"DE-1": 0.8456, "DE-2": 0.7234, "DE-3": 0.6789, "DE-4": 0.5432},
    "weights": {"DE-1": 0.3, "DE-2": 0.25, "DE-3": 0.25, "DE-4": 0.2},
    "weighted_contributions": {"DE-1": 0.2537, "DE-2": 0.1809, "DE-3": 0.1697, "DE-4": 0.1086},
    "weighted_sum": 0.7129,
    "total_weight": 1.0,
    "final_score": 0.7129,
    "capping_applied": false,
    "capping_details": null
  },
  "mathematical_audit": {
    "weighted_sum_validation": true,
    "weight_sum_validation": true,
    "precision_consistency": true
  }
}
```

## Usage

### Basic Usage

```python
from decalogo_scoring_system import ScoringSystem

scoring_system = ScoringSystem(precision=4)

# Evaluation data structure
evaluation_data = {
    "DE-1": [
        {"question_id": "DE1_Q1", "response": "Sí", "evidence_completeness": 0.9, "page_reference_quality": 1.0},
        # ... more questions
    ],
    "DE-2": [...],
    "DE-3": [...],
    "DE-4": [...]
}

# Process with trace saving enabled
point_score = scoring_system.process_point_evaluation(
    point_id=1,
    evaluation_data=evaluation_data,
    doc_id="DOCUMENT_123",
    save_trace=True  # Enables composition trace saving
)
```

### Trace File Features

1. **Deterministic JSON Serialization**: All JSON files use `sort_keys=True` for consistent ordering
2. **Mathematical Transparency**: Every calculation step is documented with full precision
3. **Evidence Quality Tracking**: Complete evidence multiplier calculations
4. **Audit Trail**: Built-in validation of weighted sums and mathematical operations
5. **Debugging Support**: Question-level breakdown for troubleshooting

### Mathematical Validation

Each trace file includes automatic validation:
- Weighted sum validation: Ensures sum of contributions equals weighted_sum
- Weight sum validation: Ensures weights sum to 1.0
- Precision consistency: All calculations use consistent decimal precision

## Integration Points

The trace system integrates with:
- `ScoringSystem.process_point_evaluation()` - Main entry point
- `ScoringSystem.compose_point_score()` - Mathematical composition with tracing
- Evidence quality multiplier calculations
- Dimension-level aggregation

## Testing

Run the test suite:
```bash
python3 canonical_flow/L_classification_evaluation/test_composition_trace.py
```

Or use the simple test runner:
```bash
python3 test_trace_system.py
```

## Files

- `decalogo_scoring_system.py` - Main implementation with `CompositionTrace` class
- `test_composition_trace.py` - Comprehensive test suite
- `demo_decalogo_scoring.py` - Demonstration with trace generation
- `README_composition_trace.md` - This documentation

## Features

✓ Complete mathematical transparency  
✓ Deterministic JSON serialization  
✓ Question-level evidence tracking  
✓ Automatic validation and audit  
✓ File organization by document and point  
✓ Debugging and troubleshooting support  
✓ Integration with existing ScoringSystem  