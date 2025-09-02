# Data Integrity Checker

A comprehensive data integrity system for pipeline artifacts that provides SHA-256 hash computation, corruption detection, automatic retry logic with exponential backoff, and detailed audit trails.

## Features

- **SHA-256 Hash Computation**: Deterministic hashing of JSON artifacts at stage boundaries
- **Corruption Detection**: Validates file integrity, metadata completeness, and content consistency  
- **Automatic Retry Logic**: Exponential backoff retry on corruption with configurable policies
- **Comprehensive Reporting**: Detailed corruption reports with file paths, hash mismatches, and missing metadata
- **Pipeline Integration**: Seamless integration with existing pipeline components via decorators and hooks
- **Audit Trail**: Central `integrity_audit.json` logging of all corruption events and recovery attempts

## Quick Start

### Basic Usage

```python
from data_integrity_checker import DataIntegrityChecker

# Initialize checker
checker = DataIntegrityChecker()

# Save artifact with integrity metadata
artifact_data = {"analysis": "complete", "confidence": 0.89}
output_path, metadata = checker.save_artifact_with_integrity(
    artifact_data, "A_analysis_nlp", "QuestionAnalyzer", "doc_123"
)

# Validate artifact integrity  
is_valid, corruption_report = checker.validate_artifact_integrity(
    output_path, metadata
)

if not is_valid:
    print(f"Corruption detected: {corruption_report.corruption_type}")
```

### Integration with Existing Components

#### Method 1: Decorator Approach (Recommended)

```python
from data_integrity_checker import integrity_validation_hook, DataIntegrityChecker

class QuestionAnalyzer:
    def __init__(self):
        self.stage_name = "A_analysis_nlp"
    
    @integrity_validation_hook(DataIntegrityChecker())
    def process(self, data=None, context=None):
        # Your existing process logic
        return {"questions_analyzed": 5, "categories": ["DE-1", "DE-2"]}
```

#### Method 2: Class Decorator for Auto-Saving

```python
from data_integrity_checker import add_artifact_generation_hook

@add_artifact_generation_hook
class MesoAggregator:
    def __init__(self):
        self.stage_name = "G_aggregation_reporting"
    
    def process(self, data=None, context=None):
        # Artifacts automatically saved with integrity metadata
        return {"aggregation": "complete"}
```

#### Method 3: Function Decorator

```python
from data_integrity_checker import validate_and_retry_on_corruption

@validate_and_retry_on_corruption("R_search_retrieval", "HybridRetriever", "query_001")
def hybrid_search(query):
    return {"results": [...], "query": query}
```

## Core Classes

### DataIntegrityChecker

Main class providing integrity validation and corruption recovery.

**Key Methods:**
- `compute_artifact_hash(artifact_data)` - SHA-256 hash computation
- `save_artifact_with_integrity()` - Save with embedded metadata
- `validate_artifact_integrity()` - Validate against expected metadata
- `validate_stage_boundary()` - Validate all artifacts at stage boundary
- `process_with_integrity_validation()` - Wrap process with retry logic

### ArtifactMetadata

Metadata structure for artifacts:
```python
@dataclass
class ArtifactMetadata:
    stage_name: str
    component_name: str  
    document_stem: str
    created_at: str
    sha256_hash: str
    file_size: int
    schema_version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
```

### CorruptionReport

Detailed corruption detection report:
```python
@dataclass  
class CorruptionReport:
    corruption_type: CorruptionType  # HASH_MISMATCH, MISSING_METADATA, etc.
    stage_name: str
    component_name: str
    file_path: str
    document_stem: str
    expected_hash: Optional[str]
    actual_hash: Optional[str]
    missing_fields: List[str]
    error_message: Optional[str]
    retry_attempt: int
```

### RetryPolicy

Configurable retry behavior:
```python
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
```

## Integration Patterns

### Pipeline Component Integration

Add integrity validation to existing pipeline components:

```python
class ExistingComponent:
    def __init__(self):
        self.stage_name = "your_stage"
        
    def process(self, data=None, context=None):
        # Add this at the start for input validation
        document_stem = context.get('document_stem', 'unknown') if context else 'unknown'
        
        # Your existing logic
        result = your_processing_logic(data)
        
        # Optional: Manual artifact saving with integrity
        if hasattr(self, '_integrity_checker'):
            self._integrity_checker.save_artifact_with_integrity(
                result, self.stage_name, self.__class__.__name__, document_stem
            )
        
        return result
```

### Stage Boundary Validation

Add validation hooks at the beginning of each stage's process() method:

```python
def process(self, data=None, context=None):
    # Validate inputs from previous stage
    document_stem = context.get('document_stem', 'unknown') if context else 'unknown'
    checker = get_global_integrity_checker()
    
    validation_report = checker.validate_stage_boundary(self.stage_name, document_stem)
    if validation_report['corruption_detected'] > 0:
        logger.warning(f"Input corruption detected: {validation_report}")
        # Handle corruption or trigger recovery
    
    # Continue with normal processing
    return your_process_logic(data, context)
```

## File Structure

### Artifact Files
```
canonical_flow/<stage_name>/<document_stem>_<component_name>_artifact.json
```

Example artifact structure:
```json
{
  "metadata": {
    "stage_name": "A_analysis_nlp",
    "component_name": "QuestionAnalyzer", 
    "document_stem": "doc_123",
    "created_at": "2024-01-15T10:30:00Z",
    "sha256_hash": "abc123...",
    "file_size": 1024,
    "schema_version": "1.0",
    "dependencies": []
  },
  "data": {
    "questions_analyzed": 5,
    "categories": ["DE-1", "DE-2"],
    "confidence_scores": [0.85, 0.92]
  }
}
```

### Integrity Audit File
```
integrity_audit.json
```

Structure:
```json
{
  "metadata": {
    "created_at": "2024-01-15T09:00:00Z",
    "schema_version": "1.0",
    "description": "Data integrity audit trail"
  },
  "corruption_events": [
    {
      "corruption_type": "hash_mismatch",
      "stage_name": "A_analysis_nlp",
      "component_name": "QuestionAnalyzer",
      "file_path": "canonical_flow/A_analysis_nlp/doc_123_QuestionAnalyzer_artifact.json",
      "expected_hash": "abc123...",
      "actual_hash": "def456...",
      "detected_at": "2024-01-15T10:15:00Z",
      "retry_attempt": 1
    }
  ],
  "recovery_attempts": [
    {
      "stage_name": "A_analysis_nlp",
      "component_name": "QuestionAnalyzer", 
      "document_stem": "doc_123",
      "attempts": 2,
      "recovered_at": "2024-01-15T10:16:00Z",
      "status": "success"
    }
  ],
  "integrity_statistics": {
    "total_artifacts_checked": 150,
    "corruption_events_detected": 3,
    "successful_recoveries": 2,
    "failed_recoveries": 1
  }
}
```

## Corruption Types

- **HASH_MISMATCH**: SHA-256 hash doesn't match expected value
- **MISSING_METADATA**: Required metadata fields are missing
- **FILE_NOT_FOUND**: Artifact file doesn't exist  
- **PARSE_ERROR**: JSON parsing failed
- **SIZE_MISMATCH**: File size differs significantly from expected

## Retry Logic

### Exponential Backoff
- Base delay: 1 second
- Exponential factor: 2.0
- Maximum delay: 60 seconds
- Jitter: ±20% random variation

### Retry Sequence Example
```
Attempt 1: Immediate
Attempt 2: ~1.0 second delay
Attempt 3: ~2.0 second delay  
Attempt 4: ~4.0 second delay
```

### Recovery Strategies

1. **Transient Failures**: Automatic retry with backoff
2. **Corruption Detection**: Regenerate artifact from source
3. **Missing Dependencies**: Validate and retry dependent stages
4. **Fatal Errors**: Mark stage as failed after max attempts

## Configuration

### Custom Retry Policy
```python
from data_integrity_checker import RetryPolicy, CorruptionType

custom_policy = RetryPolicy(
    max_attempts=5,
    base_delay=2.0,
    max_delay=120.0,
    exponential_base=1.5,
    jitter=False,
    retry_on_corruption_types={
        CorruptionType.HASH_MISMATCH,
        CorruptionType.FILE_NOT_FOUND
    }
)

checker = DataIntegrityChecker(retry_policy=custom_policy)
```

### Global Configuration
```python
from data_integrity_checker import get_global_integrity_checker

# Get shared instance
checker = get_global_integrity_checker()

# Configure paths
checker.canonical_flow_dir = Path("custom/artifacts")
checker.integrity_audit_file = Path("custom/integrity.json")
```

## Monitoring and Reporting

### Generate Integrity Report
```python
checker = DataIntegrityChecker()
report = checker.get_integrity_report()

print(f"Artifacts checked: {report['statistics']['total_artifacts_checked']}")
print(f"Corruption events: {report['statistics']['corruption_events_detected']}")
print(f"Recovery success rate: {report['statistics']['successful_recoveries']}")
```

### Integration with Existing Audit System
The integrity checker integrates with the existing audit logging system:

```python
from audit_logger import AuditLogger

# Standard audit logging continues to work
audit_logger = AuditLogger("ComponentName", "stage_name")

# Integrity checker adds corruption detection and recovery
integrity_checker = DataIntegrityChecker()

# Both systems work together
def process(self, data=None, context=None):
    audit_entry = audit_logger.start_audit(document_stem, operation_id)
    
    result = integrity_checker.process_with_integrity_validation(
        your_process_function, stage_name, component_name, document_stem
    )
    
    audit_logger.end_audit(status="success")
    return result
```

## Testing

Run the test suite:
```bash
python3 run_integrity_tests.py
```

Run integration examples:
```bash
python3 integration_example.py
```

## Best Practices

1. **Use Decorators**: Prefer `@integrity_validation_hook` for minimal code changes
2. **Validate at Boundaries**: Check integrity when transitioning between stages  
3. **Handle Context**: Pass `document_stem` in context for proper artifact tracking
4. **Monitor Reports**: Regularly review `integrity_audit.json` for corruption patterns
5. **Configure Retries**: Adjust retry policy based on your pipeline's reliability needs
6. **Test Corruption**: Use corruption injection for testing recovery logic

## Pipeline Determinism

The integrity checker ensures pipeline determinism by:
- Using deterministic JSON serialization (sorted keys)
- Consistent hash computation across runs
- Artifact versioning and dependency tracking
- Reproducible retry behavior with controlled randomness (jitter)

## Performance Considerations

- Hash computation: ~1ms for typical artifacts (<100KB)
- File I/O: Artifacts stored with minimal overhead
- Memory usage: In-memory registry for active artifacts only
- Concurrent access: Thread-safe operations on audit files

## Error Handling

The integrity checker provides graceful degradation:
- Non-blocking: Pipeline continues even if integrity checks fail
- Configurable: Can be disabled for performance-critical scenarios  
- Isolated: Integrity failures don't crash the main pipeline
- Observable: All events logged for debugging

## Integration with CI/CD

For continuous integration, add integrity validation:

```yaml
# In your CI pipeline
- name: Validate Pipeline Integrity
  run: |
    python3 run_integrity_tests.py
    python3 -c "
    from data_integrity_checker import get_global_integrity_checker
    checker = get_global_integrity_checker()
    report = checker.get_integrity_report()
    assert report['statistics']['failed_recoveries'] == 0
    "
```

This ensures that integrity issues are caught early in the development cycle.