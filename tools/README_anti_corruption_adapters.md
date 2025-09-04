# Anti-Corruption Adapters

This module implements adapter classes to break circular dependencies between retrieval and analysis phases by translating data transfer objects and blocking backward imports.

## Key Features

1. **Dependency Management**: Prevents circular dependencies between retrieval and analysis phases
2. **DTO Translation**: Translates between phase-specific data transfer objects
3. **Schema Validation**: Validates data contracts and logs mismatches
4. **Import Guards**: Prevents backward dependencies at runtime
5. **Lineage Tracking**: Integrates with lineage tracking system for monitoring

## Architecture

### Data Transfer Objects (DTOs)

- `RetrievalResultDTO`: Standardized format for retrieval phase outputs
- `AnalysisInputDTO`: Standardized format for analysis phase inputs  
- `AnalysisResultDTO`: Standardized format for analysis phase outputs

### Adapters

- `RetrievalToAnalysisAdapter`: Translates retrieval results to analysis inputs
- `AnalysisToRetrievalAdapter`: Limited feedback from analysis to retrieval (discouraged)

### Validation & Monitoring

- `SchemaViolationLogger`: Logs schema mismatches and forwards to lineage tracker
- `ImportGuard`: Runtime protection against backward dependencies
- Integration with lineage tracking system

## Usage Examples

### Basic Translation

```python
from tools.anti_corruption_adapters import (
    RetrievalResultDTO,
    RetrievalToAnalysisAdapter
)

# Create retrieval result
retrieval_result = RetrievalResultDTO(
    query="What is machine learning?",
    documents=[{"id": "doc1", "content": "ML is..."}],
    scores=[0.9],
    retrieval_method="hybrid"
)

# Translate to analysis inputs
adapter = RetrievalToAnalysisAdapter()
analysis_inputs = adapter.translate(retrieval_result)
```

### Import Protection

```python
from tools.anti_corruption_adapters import install_import_guards

# Install runtime import guards
install_import_guards()

# This will now raise BackwardDependencyError
# from analysis_nlp.question_analyzer import something
# from retrieval_engine.hybrid_retriever import something_else
```

### Schema Monitoring

```python
from tools.anti_corruption_adapters import get_schema_violations

# Get recent schema violations
violations = get_schema_violations(limit=10)
for violation in violations:
    print(f"Schema mismatch: {violation['source_schema']} -> {violation['target_schema']}")
```

## Design Principles

1. **Unidirectional Flow**: Enforces retrieval -> analysis data flow
2. **Loose Coupling**: DTOs prevent tight coupling between phases
3. **Contract Monitoring**: Logs and tracks data contract violations
4. **Runtime Safety**: Import guards prevent backward dependencies
5. **Observability**: Integrates with lineage tracking for transparency

## Testing

Run validation tests:

```bash
python3 tools/validate_adapters.py
python3 tools/demo_adapters.py
```

## Integration

The module integrates with:

- **Lineage Tracker**: For schema violation monitoring
- **Retrieval Engine**: Consumes retrieval results
- **Analysis NLP**: Provides analysis inputs
- **Import System**: Guards against backward dependencies

This ensures clean separation between architectural phases while maintaining data flow integrity.