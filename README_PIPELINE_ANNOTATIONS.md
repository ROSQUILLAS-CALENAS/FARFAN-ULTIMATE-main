# Pipeline Component Contract Annotations

## Overview

All pipeline components in this system now include mandatory static contract annotations that define their phase, code, and position in the canonical I→X→K→A→L→R→O→G→T→S sequence. These annotations ensure proper component classification, enable automated validation, and maintain pipeline integrity.

## Required Annotations

Every pipeline component must include these three module-level constants:

```python
# Mandatory Pipeline Contract Annotations
__phase__ = "A"        # Pipeline phase (I, X, K, A, L, R, O, G, T, S)
__code__ = "25A"       # Unique component code (format: NNX)
__stage_order__ = 4    # Stage order in canonical sequence (1-10)
```

## Pipeline Phases

The canonical pipeline follows this sequence:

| Phase | Name | Stage Order | Purpose |
|-------|------|-------------|---------|
| **I** | Ingestion Preparation | 1 | PDF reading, data loading, feature extraction |
| **X** | Context Construction | 2 | Context building, lineage tracking, immutable context |
| **K** | Knowledge Extraction | 3 | Graph building, embedding generation, causal analysis |
| **A** | Analysis NLP | 4 | Question analysis, evidence processing, NLP operations |
| **L** | Classification Evaluation | 5 | Scoring, classification, conformal prediction |
| **R** | Search Retrieval | 6 | Hybrid retrieval, indexing, recommendation systems |
| **O** | Orchestration Control | 7 | Workflow management, routing, validation |
| **G** | Aggregation Reporting | 8 | Report compilation, meso aggregation, audit logging |
| **T** | Integration Storage | 9 | Metrics collection, analytics, feedback loops |
| **S** | Synthesis Output | 10 | Answer synthesis, formatting, final output |

## Component Code Format

Component codes follow the format `NNX` where:
- `NN` = Sequential number within phase (01-99)
- `X` = Phase character (I, X, K, A, L, R, O, G, T, S)

Examples:
- `01I` = First ingestion component
- `15A` = Fifteenth analysis component  
- `33R` = Thirty-third retrieval component

## Automated Validation

### CI/CD Pipeline Validation

The system includes automated validation that runs on every commit:

```yaml
# .github/workflows/validate-pipeline-annotations.yml
name: Validate Pipeline Component Annotations
on: [push, pull_request]
jobs:
  validate-annotations:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Validate Annotations
      run: python3 scripts/validate_pipeline_annotations.sh
```

### Pre-commit Hooks

Pre-commit hooks prevent commits with invalid annotations:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pipeline-component-annotations
      name: Validate Pipeline Component Annotations
      entry: python3 scripts/pre_commit_hook.py
      language: system
      files: \.py$
```

## Tools and Scripts

### Bulk Annotation Tool

Add annotations to all components:

```bash
python3 scripts/bulk_annotate_components.py
```

### Validation Script

Validate existing annotations:

```bash
python3 scripts/validate_pipeline_annotations.sh
```

### Component Scanner

Scan and analyze component annotations:

```bash
python3 canonical_flow/component_scanner_update.py
```

## Current Status

✅ **COMPLETE**: All 216 pipeline components have been successfully annotated with mandatory contract annotations.

### Phase Distribution

- **I (Ingestion)**: 17 components
- **X (Context)**: 13 components  
- **K (Knowledge)**: 31 components
- **A (Analysis)**: 44 components
- **L (Classification)**: 34 components
- **R (Retrieval)**: 51 components
- **O (Orchestration)**: 157 components
- **G (Aggregation)**: 62 components
- **T (Integration)**: 65 components
- **S (Synthesis)**: 55 components

## Implementation Details

### Adding Annotations to New Components

When creating new pipeline components, add annotations after the docstring but before imports:

```python
"""
My Pipeline Component
Does important pipeline work.
"""

# Mandatory Pipeline Contract Annotations
__phase__ = "A"
__code__ = "45A"
__stage_order__ = 4

import sys
import logging
# ... rest of imports

class MyProcessor:
    def process(self, data, context):
        # Component implementation
        pass
```

### Annotation Rules

1. **Required**: All three annotations must be present
2. **Format**: Phase must be valid character, code must match `\d{2}[IXKALROGTS]`, stage_order must be integer
3. **Consistency**: Code suffix must match phase character
4. **Uniqueness**: No duplicate codes allowed
5. **Order**: Stage order must match canonical sequence

### Error Handling

The validation system provides clear error messages:

```
❌ Found 5 components missing required annotations
Components missing annotations:
  - ./new_processor.py
  
Required annotations for all pipeline components:
  __phase__ = "X"  # Pipeline phase (I, X, K, A, L, R, O, G, T, S)
  __code__ = "XXX"  # Component code (e.g., "01I", "15A")  
  __stage_order__ = N  # Stage order in pipeline sequence
```

## Benefits

1. **Automated Classification**: Components are automatically classified by phase
2. **Pipeline Integrity**: Ensures proper component sequencing
3. **CI/CD Integration**: Prevents invalid components from entering the codebase
4. **Documentation**: Self-documenting component metadata
5. **Debugging**: Easy identification of component roles and positions

## Future Enhancements

- [ ] Component dependency validation based on phase ordering
- [ ] Automated phase coverage analysis
- [ ] Integration with pipeline orchestration systems
- [ ] Performance metrics by phase
- [ ] Automated component documentation generation

## Troubleshooting

### Common Issues

**Missing Annotations**
```bash
# Fix: Run bulk annotator
python3 scripts/bulk_annotate_components.py
```

**Invalid Code Format**
```python
# Wrong: 
__code__ = "A15"

# Correct:
__code__ = "15A"
```

**Phase Mismatch**
```python
# Wrong:
__phase__ = "A"
__code__ = "15K"  # K doesn't match A

# Correct:
__phase__ = "A" 
__code__ = "15A"  # A matches A
```

### Support

For issues with pipeline annotations:
1. Run validation: `python3 scripts/validate_pipeline_annotations.sh`
2. Check error messages for specific issues
3. Use bulk annotator for missing annotations
4. Review this documentation for format requirements