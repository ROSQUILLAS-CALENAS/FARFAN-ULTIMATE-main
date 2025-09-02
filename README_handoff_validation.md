# Comprehensive Handoff Validation System

A robust validation system that ensures data integrity and schema compliance between critical pipeline stage pairs, providing early failure detection and detailed logging for validation issues.

## Overview

The Handoff Validation System enforces strict data quality standards at critical pipeline transition points:

- **I_ingestion_preparation** → **C_context_establishment**
- **K_knowledge_extraction** → **A_analysis_nlp** 
- **L_classification_evaluation** → **S_search_retrieval**

## Key Features

### ✅ Mandatory Field Validation
- Verifies all required fields exist in stage outputs
- Checks for null/empty values with appropriate severity levels
- Provides specific error messages and suggested fixes

### ✅ Schema Compliance Checking
- Validates data structures against expected schemas
- Supports multiple integrity levels (strict, moderate, relaxed)
- Detects type mismatches and structural violations

### ✅ Cross-Stage Dependency Validation
- Ensures doc_id consistency across pipeline stages
- Validates artifact presence and integrity
- Checks metadata consistency and content hashes

### ✅ Early Failure Detection
- Fail-fast mode stops pipeline on critical errors
- Comprehensive error logging with detailed diagnostics
- Validation checkpoints at critical handoff points

### ✅ Detailed Logging & Reporting
- Structured validation results with error categorization
- Suggested fixes for common validation failures
- Historical tracking and success rate statistics

## File Structure

```
schemas/
├── __init__.py                     # Schema package initialization
├── api_models.py                   # API model definitions  
└── pipeline_schemas.py             # Core pipeline stage schemas

handoff_validation_system.py       # Main validation system
test_handoff_validation.py         # Comprehensive test suite
demo_handoff_validation.py         # Demo and examples
validate_handoff_system.py         # Basic validation script
README_handoff_validation.md       # This documentation
```

## Quick Start

### Basic Usage

```python
from handoff_validation_system import validate_ingestion_to_context
from datetime import datetime

# Sample stage output data
output_data = {
    "metadata": {
        "doc_id": "doc_001",
        "source_path": "/data/document.pdf",
        "content_hash": "abc123def456", 
        "created_timestamp": datetime.now().isoformat(),
        "stage_processed": "I_ingestion_preparation"
    },
    "extracted_text": "Document text content...",
    "document_structure": {"pages": 5, "sections": 3},
    "page_count": 5,
    "extraction_confidence": 0.95,
    "content_blocks": [{"type": "paragraph", "content": "..."}],
    "semantic_markers": ["introduction", "conclusion"]
}

# Validate handoff
result = validate_ingestion_to_context(output_data)

if result.is_valid:
    print("✅ Validation passed - ready for next stage")
else:
    print("❌ Validation failed:")
    for error in result.validation_errors:
        print(f"  {error.severity}: {error.field_name} - {error.error_message}")
```

### Checkpoint Validation

```python
from handoff_validation_system import create_checkpoint_validator
from schemas.pipeline_schemas import StageType

# Create validator with checkpoint configuration
validator = create_checkpoint_validator()

# Create validation checkpoint
checkpoint = validator.create_checkpoint(
    StageType.INGESTION_PREPARATION,
    StageType.CONTEXT_ESTABLISHMENT, 
    output_data
)

print(f"Checkpoint: {checkpoint.checkpoint_id}")
print(f"Status: {'✅ VALID' if checkpoint.validation_results[0].is_valid else '❌ INVALID'}")
```

### Advanced Configuration

```python
from handoff_validation_system import HandoffValidationSystem, CheckpointValidationConfig
from schemas.pipeline_schemas import StageType, DataIntegrityLevel

# Custom validation configuration  
config = CheckpointValidationConfig(
    stage_pair=(StageType.KNOWLEDGE_EXTRACTION, StageType.ANALYSIS_NLP),
    integrity_level=DataIntegrityLevel.STRICT,
    enforce_schema=True,
    validate_cross_dependencies=True,
    fail_fast=True,
    log_warnings_as_errors=False
)

# Create validator with custom config
validator = HandoffValidationSystem()
result = validator.validate_handoff(
    StageType.KNOWLEDGE_EXTRACTION,
    StageType.ANALYSIS_NLP,
    knowledge_output_data,
    config
)
```

## Schema Definitions

### Core Schemas

The system defines strict schemas for each stage output:

- **IngestionOutput**: Text extraction, document structure, content blocks
- **ContextOutput**: Context graphs, entity references, embeddings  
- **KnowledgeOutput**: Knowledge entities, concept relations, factual claims
- **AnalysisOutput**: Sentiment analysis, topic classification, NLP features
- **ClassificationOutput**: Classification labels, confidence scores, predictions
- **RetrievalOutput**: Search results, similarity scores, retrieved documents

### Metadata Requirements

All stages must include consistent `DocumentMetadata`:

```python
@dataclass
class DocumentMetadata:
    doc_id: str                    # Unique document identifier
    source_path: str              # Original document path
    content_hash: str             # Content integrity hash
    created_timestamp: datetime   # Creation timestamp
    stage_processed: str          # Current processing stage
    processing_version: str = "1.0"
    content_length: int = 0
    language: Optional[str] = None
    document_type: Optional[str] = None
```

## Validation Levels

### Strict Mode (Default)
- All required fields must be present and non-empty
- Schema compliance strictly enforced
- Cross-stage dependencies validated
- Fails fast on critical errors

### Moderate Mode  
- Required fields must exist (can be empty with warnings)
- Flexible type checking
- Cross-stage dependencies validated
- Continues on non-critical issues

### Relaxed Mode
- Basic validation only
- Warnings for missing fields
- Minimal schema enforcement
- Continues unless critical system errors

## Error Types and Fixes

### Common Validation Errors

| Error Type | Description | Suggested Fix |
|------------|-------------|---------------|
| `missing_field` | Required field not present | Ensure previous stage outputs field |
| `null_value` | Required field is null | Provide valid value in previous stage |
| `empty_value` | Required field is empty | Verify processing logic |
| `doc_id_mismatch` | Document ID inconsistent | Preserve doc_id across stages |
| `hash_mismatch` | Content hash changed | Verify content integrity |
| `schema_violation` | Data doesn't match schema | Review data structure |
| `missing_dependency` | Cross-stage dependency missing | Ensure previous stage outputs dependency |

### Error Severity Levels

- **error**: Critical issues that block pipeline execution
- **warning**: Non-critical issues that may affect quality
- **info**: Informational notices for optimization

## Testing

### Run Basic Tests

```bash
python3 validate_handoff_system.py
```

### Run Comprehensive Test Suite

```bash
python3 -m pytest test_handoff_validation.py -v
```

### Run Demo

```bash
python3 demo_handoff_validation.py
```

## Integration Examples

### With Existing Pipeline

```python
from handoff_validation_system import HandoffValidationError, validate_ingestion_to_context

def process_ingestion_to_context(ingestion_output):
    """Process with validation"""
    try:
        # Validate before proceeding
        result = validate_ingestion_to_context(ingestion_output)
        
        if not result.is_valid:
            raise HandoffValidationError(
                "Critical validation failure", 
                result
            )
        
        # Proceed with context establishment
        return context_establishment_stage(ingestion_output)
        
    except HandoffValidationError as e:
        logger.error(f"Handoff validation failed: {e}")
        # Handle validation failure appropriately
        raise
```

### With Error Recovery

```python
def robust_stage_transition(source_stage, target_stage, output_data):
    """Stage transition with error recovery"""
    validator = create_checkpoint_validator()
    
    try:
        result = validator.validate_handoff(source_stage, target_stage, output_data)
        
        if result.is_valid:
            return proceed_to_next_stage(output_data)
        else:
            # Attempt to fix common issues
            fixed_data = attempt_auto_fix(output_data, result.validation_errors)
            
            # Re-validate after fixes
            retry_result = validator.validate_handoff(source_stage, target_stage, fixed_data)
            
            if retry_result.is_valid:
                logger.info("Auto-fix successful, proceeding")
                return proceed_to_next_stage(fixed_data)
            else:
                # Manual intervention required
                raise HandoffValidationError("Manual fix required", retry_result)
                
    except Exception as e:
        logger.error(f"Stage transition failed: {e}")
        raise
```

## Performance Considerations

- **Validation Overhead**: ~1-5ms per handoff for typical documents
- **Memory Usage**: Minimal additional memory footprint
- **Scalability**: Supports concurrent validation across multiple pipelines
- **Caching**: Validation results cached for repeated validations

## Customization

### Custom Validators

```python
from handoff_validation_system import ArtifactValidator, ValidationResult

class CustomJSONValidator(ArtifactValidator):
    def __init__(self, custom_rules):
        self.custom_rules = custom_rules
    
    def validate(self, artifact_path):
        # Custom validation logic
        return ValidationResult(
            artifact_path=artifact_path,
            is_valid=True,
            errors=[],
            warnings=[]
        )
```

### Custom Schema Extensions

```python
from schemas.pipeline_schemas import IngestionOutput
from dataclasses import dataclass

@dataclass
class ExtendedIngestionOutput(IngestionOutput):
    custom_field: str = ""
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
```

## Monitoring and Metrics

### Validation Statistics

```python
validator = create_checkpoint_validator()

# After running validations...
summary = validator.get_validation_summary()

print(f"Success Rate: {summary['success_rate']:.2%}")
print(f"Total Validations: {summary['total_validations']}")

# Stage-specific statistics
for stage_pair, stats in summary['stage_pair_statistics'].items():
    print(f"{stage_pair}: {stats['successful']}/{stats['total']} successful")
```

### Logging Integration

The system integrates with Python's logging framework:

```python
import logging

# Configure logging level
logging.getLogger("HandoffValidationSystem").setLevel(logging.INFO)
logging.getLogger("SchemaValidator").setLevel(logging.DEBUG)
logging.getLogger("CrossStageValidator").setLevel(logging.WARNING)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all schema files are in the Python path
2. **Schema Mismatches**: Check that stage outputs match expected schemas
3. **DateTime Serialization**: Use ISO format for datetime fields
4. **Memory Issues**: Use validation caching for large datasets
5. **Performance**: Consider relaxed mode for non-critical validations

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enables detailed validation logging
validator = HandoffValidationSystem(log_level=logging.DEBUG)
```

## Contributing

When extending the validation system:

1. Add new schemas to `schemas/pipeline_schemas.py`
2. Update `HANDOFF_SCHEMAS` mapping for new stage pairs
3. Add corresponding tests in `test_handoff_validation.py`
4. Update documentation with new validation rules

## License

Part of the EGW Query Expansion system. See main project LICENSE for details.