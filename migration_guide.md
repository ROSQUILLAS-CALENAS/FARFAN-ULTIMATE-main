# Migration Guide: Validator Port/Adapter Architecture

This guide explains how to migrate from the old direct import approach to the new port/adapter architecture with dependency injection.

## Overview

The new architecture introduces:
- **`validator_api`** package: Abstract interfaces and DTOs
- **`validator_impl`** package: Concrete implementations
- **Dependency injection**: Components depend on interfaces, not concrete implementations

## Architecture Benefits

1. **Decoupling**: No direct imports between validation and pipeline components
2. **Testability**: Easy to mock interfaces for testing
3. **Extensibility**: Add new validators/processors without changing existing code
4. **Maintainability**: Clear separation of concerns

## Migration Steps

### 1. Old Code Pattern (Before)

```python
# ❌ Direct imports from pipeline components
from models import SectionBlock, SectionType, QualityIndicators
from evidence_validation_model import EvidenceValidationModel
from canonical_flow.analysis.audit_logger import get_audit_logger

class PDTValidator:
    def validate_document(self, blocks):
        # Direct coupling to pipeline components
        pass
```

### 2. New Code Pattern (After)

```python
# ✅ Import only from validator_api interfaces
from validator_api import ValidatorPort, ValidationRequest, ValidationResult
from validator_impl import PDTValidator, ValidationServiceImpl

class ValidatorOrchestrator:
    def __init__(self):
        # Dependency injection
        self.validation_service = ValidationServiceImpl()
        self._setup_validators()
    
    def _setup_validators(self):
        pdt_validator = PDTValidator()
        self.validation_service.register_validator("pdt", pdt_validator)
```

## Key Changes

### A. Validator Components

#### Before:
```python
# Direct imports and tight coupling
from models import SectionBlock, SectionType
import pipeline_component

class OldValidator:
    def validate(self, data):
        # Direct method calls to pipeline components
        return pipeline_component.process(data)
```

#### After:
```python
# Clean interfaces only
from validator_api import ValidatorPort, ValidationResult

class NewValidator(ValidatorPort):
    def validate(self, request: ValidationRequest) -> ValidationResult:
        # Self-contained validation logic
        return ValidationResult(...)
```

### B. Evidence Processing

#### Before:
```python
# Direct imports from pipeline stages
from canonical_flow.analysis.audit_logger import get_audit_logger
from evidence_validation_model import EvidenceValidationModel

class EvidenceProcessor:
    def __init__(self):
        self.audit_logger = get_audit_logger()  # Direct dependency
```

#### After:
```python
# Interface-based design
from validator_api import EvidenceProcessorPort

class EvidenceProcessorImpl(EvidenceProcessorPort):
    def process_evidence(self, request: EvidenceProcessingRequest):
        # No direct pipeline dependencies
```

### C. Service Orchestration

#### Before:
```python
# Direct instantiation and coupling
validator = PDTValidator()
evidence_processor = EvidenceProcessor()
result = validator.check_mandatory_sections(blocks)
```

#### After:
```python
# Dependency injection through service
service = ValidationServiceImpl()
service.register_validator("pdt", PDTValidator())
service.register_evidence_processor("default", EvidenceProcessorImpl())

result = service.validate_single(data, "pdt")
```

## Usage Examples

### Example 1: Document Validation

```python
from validator_refactored import ValidatorOrchestrator

# Create orchestrator (handles DI internally)
orchestrator = ValidatorOrchestrator()

# Validate document
document_data = {
    "blocks": [
        {
            "section_type": "diagnostico",
            "text": "Detailed diagnostic text...",
            "confidence": 0.9
        }
    ]
}

result = orchestrator.validate_document(
    document_data=document_data,
    document_type="pdt"
)

print(f"Validation success: {result['success']}")
print(f"Status: {result['status']}")
```

### Example 2: Evidence Processing

```python
from evidence_processor_refactored import EvidenceProcessingOrchestrator

# Create orchestrator
orchestrator = EvidenceProcessingOrchestrator()

# Process evidence
raw_evidence = [
    {
        "text": "Evidence text...",
        "metadata": {"document_id": "doc_001", "title": "Report"}
    }
]

result = orchestrator.process_evidence(
    raw_evidence=raw_evidence,
    processor_type="default"
)

print(f"Processing success: {result['success']}")
print(f"Evidence count: {result['processing_result']['evidence_count']}")
```

### Example 3: Custom Validator

```python
from validator_api import ValidatorPort, ValidationResult, ValidationStatus
from validator_refactored import ValidatorOrchestrator

class CustomValidator(ValidatorPort):
    def validate(self, request):
        # Custom validation logic
        return ValidationResult(
            status=ValidationStatus.PASSED,
            message="Custom validation passed",
            details={},
            errors=[],
            warnings=[]
        )
    
    def get_validation_rules(self):
        return {"custom_rules": "example"}
    
    def supports_data_type(self, data_type):
        return data_type == "custom"

# Register custom validator
orchestrator = ValidatorOrchestrator()
orchestrator.add_custom_validator("custom", CustomValidator())
```

## Testing Strategy

### Unit Tests

```python
# Test validator in isolation
def test_pdt_validator():
    validator = PDTValidator()
    request = ValidationRequest(data=test_data, validation_type="pdt")
    result = validator.validate(request)
    assert result.is_success()

# Test with mocked interfaces
def test_with_mock_service():
    mock_service = Mock(spec=ValidationService)
    orchestrator = ValidatorOrchestrator()
    orchestrator.validation_service = mock_service
    # Test behavior
```

### Integration Tests

```python
def test_full_pipeline():
    service = ValidationServiceImpl()
    service.register_validator("pdt", PDTValidator())
    
    result = service.validate_single(test_data, "pdt")
    assert result.is_success()
```

## Benefits Achieved

1. **No Direct Pipeline Dependencies**: Validators and processors are self-contained
2. **Easy Testing**: Mock interfaces instead of complex pipeline components
3. **Flexible Configuration**: Register different implementations at runtime
4. **Clear Boundaries**: Well-defined interfaces between components
5. **Maintainable Code**: Changes to one component don't affect others

## Migration Checklist

- [ ] Create `validator_api` package with interfaces and DTOs
- [ ] Create `validator_impl` package with concrete implementations
- [ ] Refactor existing validators to implement `ValidatorPort`
- [ ] Refactor evidence processors to implement `EvidenceProcessorPort`
- [ ] Create service orchestrators using dependency injection
- [ ] Update all imports to use interface packages only
- [ ] Add tests for new architecture
- [ ] Update documentation and examples

## Common Pitfalls

1. **Don't import pipeline components** in validator_impl - keep implementations self-contained
2. **Use interfaces consistently** - always program to interfaces, not implementations
3. **Handle registration errors** - validate that components implement required interfaces
4. **Test boundary contracts** - ensure implementations satisfy interface contracts
5. **Document dependencies** - clearly specify what each component needs

## Next Steps

1. Run migration tests: `python3 test_validator_api.py`
2. Update existing code to use new orchestrators
3. Add custom validators/processors as needed
4. Integrate with CI/CD pipelines
5. Monitor performance and adjust as needed