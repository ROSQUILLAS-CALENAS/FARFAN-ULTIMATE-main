# Validator System Refactoring

This document describes the refactoring of the validator system to use dependency injection and clean architecture principles.

## Overview

The refactoring creates a clear separation between:
- **validator_api**: Abstract interfaces and data transfer objects
- **validator_impl**: Concrete implementations that depend only on validator_api
- **Pipeline components**: Refactored to use dependency injection

## Architecture

### validator_api Package

Contains abstract interfaces and DTOs that define the contract for validator operations:

```
validator_api/
├── __init__.py          # Package exports
├── interfaces.py        # Abstract interfaces (IValidator, IEvidenceProcessor, etc.)
└── dtos.py             # Data transfer objects (ValidationRequest, ValidationResponse, etc.)
```

**Key Interfaces:**
- `IValidator`: Abstract base for all validators
- `IEvidenceProcessor`: Abstract base for evidence processors
- `IValidatorFactory`: Factory interface for creating validators
- `IEvidenceProcessorFactory`: Factory interface for creating processors

**Key DTOs:**
- `ValidationRequest`: Request for validation operations
- `ValidationResponse`: Response from validation operations
- `EvidenceItem`: Structured evidence item
- `ValidationMetrics`: Metrics for validation operations

### validator_impl Package

Contains concrete implementations that depend only on validator_api interfaces:

```
validator_impl/
├── __init__.py              # Package exports
├── validators.py            # Concrete validator implementations
├── evidence_processors.py   # Concrete evidence processor implementations
└── factories.py            # Factory implementations and DI container
```

**Implemented Validators:**
- `ComprehensiveValidator`: Multi-category validation
- `DNPAlignmentValidator`: DNP-specific compliance validation
- `EvidenceValidator`: Basic evidence structure validation

**Implemented Processors:**
- `DefaultEvidenceProcessor`: General evidence processing
- `DNPEvidenceProcessor`: DNP-enhanced evidence processing

## Dependency Injection

The system uses a dependency injection container for managing components:

```python
from validator_impl.factories import get_global_container, configure_global_defaults

# Setup container
container = get_global_container()
configure_global_defaults()

# Get components
validator = container.get_validator('comprehensive', singleton=True)
processor = container.get_processor('dnp', singleton=False)
```

## Refactored Pipeline Integration

The refactored orchestrator uses dependency injection:

```python
from canonical_flow.A_analysis_nlp.refactored_stage_orchestrator import create_analysis_orchestrator
from validator_impl.factories import get_global_container

container = get_global_container()
orchestrator = create_analysis_orchestrator(
    validator_factory=container.get_validator_factory(),
    processor_factory=container.get_processor_factory()
)
```

## Usage Examples

### Basic Usage

```python
from validator_impl.factories import ValidatorFactory, EvidenceProcessorFactory
from validator_api.dtos import ValidationRequest, ValidationCategory

# Create factories
validator_factory = ValidatorFactory()
processor_factory = EvidenceProcessorFactory()

# Create validator
validator = validator_factory.create_validator('comprehensive')

# Create validation request
request = ValidationRequest(
    evidence_text="Sample evidence text",
    validation_categories=[ValidationCategory.FACTUAL_ACCURACY]
)

# Perform validation
response = validator.validate(request)
print(f"Valid: {response.is_valid}, Confidence: {response.confidence_score}")
```

### Evidence Processing

```python
# Create processor
processor = processor_factory.create_processor('dnp')

# Process evidence
evidence_data = {
    'content': 'Legal evidence content',
    'source': 'Legal document',
    'type': 'legal_reference'
}

processed = processor.process_evidence(evidence_data)
print(f"Processed {len(processed)} items")
```

### Full Integration

```python
from validator_api.dtos import ValidationCategory, DNPAlignmentCategory

results = orchestrator.process_analysis_request(
    text_content="Analysis content",
    context="Context information",
    validation_categories=[ValidationCategory.FACTUAL_ACCURACY],
    dnp_categories=[DNPAlignmentCategory.REGULATORY],
    evidence_data={'items': [evidence_item]}
)

print(f"Analysis status: {results['status']}")
print(f"Overall validation: {results['validation']['overall']['is_valid']}")
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_validator_refactor.py
```

The test suite validates:
- ✅ Package imports
- ✅ Validator creation and functionality
- ✅ Evidence processor creation and functionality
- ✅ Complete validation workflows
- ✅ Dependency injection container
- ✅ Orchestrator integration
- ✅ Interface compliance

## Benefits of Refactoring

1. **Clean Separation of Concerns**: Clear boundaries between interfaces and implementations
2. **Dependency Injection**: Easy testing and component swapping
3. **Extensibility**: Easy to add new validators and processors
4. **Testability**: Components can be easily mocked and tested in isolation
5. **Maintainability**: Clear dependencies and reduced coupling
6. **Compliance**: All implementations properly implement defined interfaces

## Migration Guide

### For Existing Pipeline Components

1. Replace direct imports of concrete validators with validator_api interfaces:
   ```python
   # Before
   from evidence_processor import EvidenceProcessor
   
   # After
   from validator_api.interfaces import IEvidenceProcessor
   ```

2. Use dependency injection to obtain implementations:
   ```python
   # Before
   processor = EvidenceProcessor()
   
   # After
   processor = processor_factory.create_processor('default')
   ```

3. Update method signatures to use DTOs:
   ```python
   # Before
   def validate(self, text: str, context: str) -> dict:
   
   # After
   def validate(self, request: IEvidenceValidationRequest) -> IEvidenceValidationResponse:
   ```

### For New Components

1. Define interfaces in validator_api if needed
2. Implement concrete classes in validator_impl
3. Register with factories
4. Use dependency injection for integration

This refactoring provides a solid foundation for scalable, testable, and maintainable validator operations in the pipeline system.