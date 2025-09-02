# L Classification Evaluation Technical Specification

## Table of Contents

1. [Overview](#overview)
2. [API Schemas](#api-schemas)
3. [Mathematical Formulas](#mathematical-formulas)
4. [Evidence Quality System](#evidence-quality-system)
5. [DNP Baseline Integration](#dnp-baseline-integration)
6. [Acceptance Criteria](#acceptance-criteria)
7. [Downstream Integration](#downstream-integration)

## Overview

The L Classification Evaluation system provides comprehensive question evaluation capabilities with multi-dimensional scoring, evidence quality assessment, and DNP (Did Not Pass) baseline integration. The system processes 470 questions across four evaluation dimensions (DE-1 through DE-4) with deterministic output guarantees.

## API Schemas

### QuestionEvalInput

JSON Schema for question evaluation input payload:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "QuestionEvalInput",
  "description": "Input schema for question evaluation processing",
  "required": ["question_id", "question_text", "context", "dimensions"],
  "properties": {
    "question_id": {
      "type": "string",
      "pattern": "^[A-Z0-9]{3,10}$",
      "description": "Unique identifier for the question"
    },
    "question_text": {
      "type": "string",
      "minLength": 10,
      "maxLength": 2000,
      "description": "The question content to be evaluated"
    },
    "context": {
      "type": "object",
      "required": ["document_id", "section", "page_num"],
      "properties": {
        "document_id": {
          "type": "string",
          "description": "Source document identifier"
        },
        "section": {
          "type": "string",
          "description": "Document section reference"
        },
        "page_num": {
          "type": "integer",
          "minimum": 1,
          "description": "Source page number"
        },
        "exact_text": {
          "type": "string",
          "description": "Exact text excerpt from source"
        }
      }
    },
    "dimensions": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["DE-1", "DE-2", "DE-3", "DE-4"]
      },
      "minItems": 1,
      "maxItems": 4,
      "uniqueItems": true,
      "description": "Evaluation dimensions to process"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "priority": {
          "type": "string",
          "enum": ["high", "medium", "low"],
          "default": "medium"
        },
        "tags": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    }
  },
  "additionalProperties": false
}
```

#### Example QuestionEvalInput:

```json
{
  "question_id": "Q001",
  "question_text": "What are the compliance requirements for data retention?",
  "context": {
    "document_id": "DOC_2024_001",
    "section": "Data Management",
    "page_num": 45,
    "exact_text": "Data retention policies must comply with regulatory requirements..."
  },
  "dimensions": ["DE-1", "DE-2", "DE-3"],
  "metadata": {
    "priority": "high",
    "tags": ["compliance", "data-retention"]
  }
}
```

### DimensionEvalOutput

JSON Schema for dimension-level evaluation results:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "DimensionEvalOutput",
  "description": "Output schema for individual dimension evaluation",
  "required": ["dimension_id", "base_score", "evidence_quality", "final_score", "classification"],
  "properties": {
    "dimension_id": {
      "type": "string",
      "enum": ["DE-1", "DE-2", "DE-3", "DE-4"],
      "description": "Dimension identifier"
    },
    "base_score": {
      "type": "number",
      "enum": [0.0, 0.5, 1.0],
      "description": "Base scoring: Sí=1.0, Parcial=0.5, No/NI=0.0"
    },
    "evidence_quality": {
      "type": "object",
      "required": ["completeness_score", "multiplier"],
      "properties": {
        "completeness_score": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Evidence completeness assessment"
        },
        "multiplier": {
          "type": "number",
          "minimum": 0.7,
          "maximum": 1.0,
          "description": "Quality multiplier applied to base score"
        },
        "has_page_reference": {
          "type": "boolean",
          "description": "Whether page number is provided"
        },
        "has_exact_text": {
          "type": "boolean",
          "description": "Whether exact text excerpt is provided"
        }
      }
    },
    "final_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Base score * evidence quality multiplier"
    },
    "classification": {
      "type": "string",
      "enum": ["compliant", "partial", "non-compliant", "insufficient"],
      "description": "Classification based on final score"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Confidence level in evaluation"
    }
  },
  "additionalProperties": false
}
```

#### Example DimensionEvalOutput:

```json
{
  "dimension_id": "DE-1",
  "base_score": 1.0,
  "evidence_quality": {
    "completeness_score": 0.9,
    "multiplier": 0.95,
    "has_page_reference": true,
    "has_exact_text": true
  },
  "final_score": 0.95,
  "classification": "compliant",
  "confidence": 0.92
}
```

### PointEvalOutput

JSON Schema for point-level aggregated results:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "PointEvalOutput",
  "description": "Output schema for point-level evaluation aggregation",
  "required": ["point_id", "dimensions", "weighted_score", "bounds_applied", "final_classification"],
  "properties": {
    "point_id": {
      "type": "string",
      "pattern": "^P[0-9]{3}$",
      "description": "Point identifier (P001-P470)"
    },
    "dimensions": {
      "type": "array",
      "items": {"$ref": "#/definitions/DimensionEvalOutput"},
      "minItems": 1,
      "maxItems": 4,
      "description": "Individual dimension evaluations"
    },
    "weights": {
      "type": "object",
      "required": ["DE-1", "DE-2", "DE-3", "DE-4"],
      "properties": {
        "DE-1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "DE-2": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "DE-3": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "DE-4": {"type": "number", "minimum": 0.0, "maximum": 1.0}
      },
      "description": "Configurable dimension weights (must sum to 1.0)"
    },
    "weighted_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Weighted aggregate score across dimensions"
    },
    "bounds_applied": {
      "type": "object",
      "properties": {
        "lower_bound": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "default": 0.0
        },
        "upper_bound": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "default": 1.0
        },
        "capped": {
          "type": "boolean",
          "description": "Whether score was capped due to bounds"
        }
      }
    },
    "final_classification": {
      "type": "string",
      "enum": ["compliant", "partial", "non-compliant", "insufficient"],
      "description": "Final point classification"
    },
    "dnp_baseline": {
      "type": "object",
      "properties": {
        "threshold": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "meets_threshold": {
          "type": "boolean"
        },
        "synchronized": {
          "type": "boolean"
        }
      }
    }
  },
  "definitions": {
    "DimensionEvalOutput": {
      "type": "object"
    }
  },
  "additionalProperties": false
}
```

#### Example PointEvalOutput:

```json
{
  "point_id": "P001",
  "dimensions": [
    {
      "dimension_id": "DE-1",
      "base_score": 1.0,
      "evidence_quality": {
        "completeness_score": 0.9,
        "multiplier": 0.95,
        "has_page_reference": true,
        "has_exact_text": true
      },
      "final_score": 0.95,
      "classification": "compliant",
      "confidence": 0.92
    }
  ],
  "weights": {
    "DE-1": 0.4,
    "DE-2": 0.3,
    "DE-3": 0.2,
    "DE-4": 0.1
  },
  "weighted_score": 0.89,
  "bounds_applied": {
    "lower_bound": 0.0,
    "upper_bound": 1.0,
    "capped": false
  },
  "final_classification": "compliant",
  "dnp_baseline": {
    "threshold": 0.7,
    "meets_threshold": true,
    "synchronized": true
  }
}
```

### StageMeta

JSON Schema for pipeline stage metadata:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "StageMeta",
  "description": "Metadata schema for pipeline stage tracking",
  "required": ["stage_id", "timestamp", "version", "status"],
  "properties": {
    "stage_id": {
      "type": "string",
      "pattern": "^STAGE_[A-Z0-9_]+$",
      "description": "Unique stage identifier"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp"
    },
    "version": {
      "type": "string",
      "pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+$",
      "description": "Semantic version of processing stage"
    },
    "status": {
      "type": "string",
      "enum": ["pending", "processing", "completed", "failed"],
      "description": "Current stage status"
    },
    "processing_stats": {
      "type": "object",
      "properties": {
        "questions_processed": {
          "type": "integer",
          "minimum": 0,
          "maximum": 470
        },
        "success_rate": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "avg_processing_time_ms": {
          "type": "number",
          "minimum": 0
        }
      }
    },
    "deterministic_hash": {
      "type": "string",
      "pattern": "^[a-f0-9]{64}$",
      "description": "SHA-256 hash ensuring deterministic output"
    }
  },
  "additionalProperties": false
}
```

#### Example StageMeta:

```json
{
  "stage_id": "STAGE_L_CLASSIFICATION",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "v1.2.3",
  "status": "completed",
  "processing_stats": {
    "questions_processed": 470,
    "success_rate": 0.998,
    "avg_processing_time_ms": 125.5
  },
  "deterministic_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
}
```

## Mathematical Formulas

### Base Scoring System

The base scoring system uses discrete values based on evaluation outcomes:

```
Base Score (Bs) = {
  1.0  if evaluation = "Sí" (Yes/Compliant)
  0.5  if evaluation = "Parcial" (Partial)
  0.0  if evaluation = "No" ∨ "NI" (No/Not Identified)
}
```

### Evidence Quality Multiplier

Evidence quality is calculated based on completeness factors:

```
Completeness Score (Cs) = 0.5 × page_factor + 0.5 × text_factor

Where:
page_factor = 1.0 if page_num exists, 0.0 otherwise
text_factor = 1.0 if exact_text exists, 0.0 otherwise

Evidence Quality Multiplier (Em) = 0.7 + 0.3 × Cs

Bounds: Em ∈ [0.7, 1.0]
```

### Dimension Weighted Aggregation

Point-level scores aggregate across dimensions DE-1 through DE-4:

```
Weighted Score (Ws) = Σ(i=1 to 4) wi × (Bsi × Emi)

Where:
- wi = weight for dimension DE-i
- Bsi = base score for dimension DE-i
- Emi = evidence multiplier for dimension DE-i
- Σwi = 1.0 (normalized weights)

Constraint: wi ∈ [0.0, 1.0] ∀i
```

### Bounds Enforcement and Capping

Score bounds are enforced with configurable limits:

```
Final Score (Fs) = min(max(Ws, lower_bound), upper_bound)

Default bounds:
- lower_bound = 0.0
- upper_bound = 1.0

Capping indicator:
capped = (Ws < lower_bound) ∨ (Ws > upper_bound)
```

### Classification Thresholds

Final classification based on score ranges:

```
Classification = {
  "compliant"      if Fs ≥ 0.8
  "partial"        if 0.5 ≤ Fs < 0.8
  "non-compliant"  if 0.2 ≤ Fs < 0.5
  "insufficient"   if Fs < 0.2
}
```

## Evidence Quality System

### Completeness Score Derivation

Evidence quality assessment follows a systematic approach:

#### Page Number Assessment

```
page_score = {
  1.0  if page_num is provided AND page_num > 0
  0.0  otherwise
}
```

#### Exact Text Assessment

```
text_score = {
  1.0  if exact_text is provided AND length(exact_text) > 10
  0.8  if exact_text is provided AND 5 < length(exact_text) ≤ 10
  0.5  if exact_text is provided AND 0 < length(exact_text) ≤ 5
  0.0  if exact_text is null OR empty
}
```

#### Combined Completeness

```
completeness_score = (page_score + text_score) / 2
```

#### Quality Multiplier Mapping

```
evidence_multiplier = 0.7 + (0.3 × completeness_score)
```

This ensures the multiplier range [0.7, 1.0], where:
- Perfect evidence (page + full text) = 1.0 multiplier
- No evidence = 0.7 multiplier (minimum penalty)
- Partial evidence = linear interpolation

## DNP Baseline Integration

### Threshold Synchronization

DNP (Did Not Pass) baseline integration maintains consistency with evaluation thresholds:

```
dnp_threshold = configurable_threshold (default: 0.7)

meets_dnp_threshold = final_score ≥ dnp_threshold
```

### Compliance Classification Logic

```
compliance_classification = {
  "compliant"      if meets_dnp_threshold AND final_score ≥ 0.8
  "partial"        if meets_dnp_threshold AND 0.5 ≤ final_score < 0.8
  "non-compliant"  if ¬meets_dnp_threshold
  "insufficient"   if final_score < 0.2
}
```

### Synchronization Verification

```
synchronized = (classification_consistent AND threshold_aligned)

Where:
classification_consistent = (dnp_result == l_classification_result)
threshold_aligned = (dnp_threshold == l_threshold ± tolerance)
tolerance = 0.01 (1% threshold alignment tolerance)
```

## Acceptance Criteria

### Coverage Requirements

1. **Question Coverage**: System MUST process all 470 questions
   ```
   processed_questions = 470
   coverage_rate = processed_questions / 470 = 1.0 (100%)
   ```

2. **Dimension Coverage**: Each question MUST evaluate relevant dimensions
   ```
   min_dimensions_per_question = 1
   max_dimensions_per_question = 4
   ```

### Deterministic Output Guarantees

1. **Reproducibility**: Identical input MUST produce identical output
   ```
   hash(output_t1) == hash(output_t2) for identical input
   ```

2. **Hash Verification**: Each processing stage generates deterministic hash
   ```
   deterministic_hash = SHA256(canonical_json(output))
   ```

3. **Precision Consistency**: Numerical outputs MUST maintain precision
   ```
   score_precision = 6 decimal places
   multiplier_precision = 3 decimal places
   ```

### Performance Criteria

1. **Processing Time**: Average question processing ≤ 200ms
2. **Success Rate**: ≥ 99.5% successful evaluations
3. **Memory Usage**: ≤ 2GB peak memory consumption

### Validation Requirements

1. **Schema Compliance**: All outputs MUST validate against JSON schemas
2. **Bounds Verification**: All scores MUST respect defined bounds
3. **Weight Normalization**: Dimension weights MUST sum to 1.0 (±0.001)

## Downstream Integration

### Pipeline Stage Interface

The L Classification Evaluation integrates with downstream stages through standardized interfaces:

#### Output Format

```json
{
  "batch_id": "string",
  "processing_metadata": "StageMeta",
  "results": [
    {
      "question_id": "string",
      "point_evaluation": "PointEvalOutput"
    }
  ],
  "summary_statistics": {
    "total_questions": 470,
    "avg_weighted_score": "number",
    "classification_distribution": {
      "compliant": "integer",
      "partial": "integer", 
      "non-compliant": "integer",
      "insufficient": "integer"
    }
  }
}
```

#### Consumption Interface

Downstream stages MUST implement:

1. **Input Validation**: Verify schema compliance and completeness
2. **Hash Verification**: Validate deterministic hash integrity
3. **Error Handling**: Process failed evaluations appropriately
4. **Metadata Preservation**: Maintain pipeline traceability

#### Integration Checkpoints

1. **Pre-processing**: Validate input availability and format
2. **Processing**: Monitor progress and handle failures
3. **Post-processing**: Verify output completeness and quality
4. **Handoff**: Confirm successful downstream consumption

### API Endpoints

#### Evaluation Endpoint
```
POST /api/v1/evaluate/batch
Content-Type: application/json

Request Body: Array of QuestionEvalInput
Response: Batch processing result with PointEvalOutput array
```

#### Status Endpoint
```
GET /api/v1/status/{batch_id}
Response: StageMeta with current processing status
```

#### Results Endpoint
```
GET /api/v1/results/{batch_id}
Response: Complete evaluation results for batch
```

### Error Handling

#### Error Categories

1. **Input Validation Errors**: Malformed input data
2. **Processing Errors**: Evaluation computation failures  
3. **Integration Errors**: Downstream communication failures
4. **System Errors**: Infrastructure or resource issues

#### Error Response Format

```json
{
  "error": {
    "code": "string",
    "message": "string", 
    "details": "object",
    "timestamp": "string (ISO 8601)",
    "trace_id": "string"
  },
  "partial_results": "array (if applicable)"
}
```

This technical specification provides comprehensive coverage of the L Classification Evaluation system's API schemas, mathematical formulations, and integration requirements, ensuring deterministic processing of the 470-question dataset with robust evidence quality assessment and DNP baseline synchronization.