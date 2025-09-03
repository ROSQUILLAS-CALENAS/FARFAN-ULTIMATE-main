# Validator Port/Adapter Architecture

## Overview

This document describes the new port/adapter (hexagonal) architecture implemented for validation operations. The architecture decouples validation logic from concrete pipeline components using dependency injection and clean interfaces.

## Architecture Components

### 1. `validator_api` Package (Ports)

Contains abstract interfaces and data transfer objects that define contracts for validation operations.

#### Key Interfaces:

- **`ValidatorPort`**: Abstract interface for all validators
- **`EvidenceProcessorPort`**: Abstract interface for evidence processors  
- **`ValidationService`**: Service interface for orchestrating operations

#### Data Transfer Objects:

- **`ValidationRequest/Response`**: Request/response objects for validation
- **`EvidenceProcessingRequest/Response`**: Request/response for evidence processing
- **`ValidationResult`**: Standardized result object
- **`EvidenceItem/Metadata`**: Structured evidence objects

### 2. `validator_impl` Package (Adapters)

Contains concrete implementations that depend only on the `validator_api` interfaces.

#### Key Implementations:

- **`PDTValidator`**: Validates PDT documents according to DNP standards
- **`EvidenceProcessorImpl`**: Processes raw evidence into structured format
- **`ValidationServiceImpl`**: Orchestrates validation operations

### 3. Refactored Orchestrators

High-level orchestrators that demonstrate proper usage of the new architecture:

- **`ValidatorOrchestrator`**: Handles document validation with dependency injection
- **`EvidenceProcessingOrchestrator`**: Manages evidence processing operations

## Benefits

### 1. Decoupling
- No direct imports between validation and pipeline components
- Validators are self-contained and portable
- Clear separation of concerns

### 2. Testability
- Easy to mock interfaces for unit testing
- Isolated component testing
- Reduced test complexity

### 3. Extensibility
- Add new validators without changing existing code
- Plugin architecture support
- Runtime component registration

### 4. Maintainability
- Clear boundaries between components
- Interface contracts prevent breaking changes
- Easier refactoring

## Usage Patterns

### Basic Document Validation

```python
from validator_refactored import ValidatorOrchestrator

# Create orchestrator (dependency injection handled internally)
orchestrator = ValidatorOrchestrator()

# Validate document
result = orchestrator.validate_document(
    document_data=document,
    document_type="pdt"
)
```

### Evidence Processing

```python
from evidence_processor_refactored import EvidenceProcessingOrchestrator

# Create orchestrator
orchestrator = EvidenceProcessingOrchestrator()

# Process evidence
result = orchestrator.process_evidence(
    raw_evidence=raw_data,
    processor_type="default"
)
```

### Custom Validator Registration

```python
from validator_api import ValidatorPort
from validator_impl import ValidationServiceImpl

class CustomValidator(ValidatorPort):
    def validate(self, request):
        # Custom validation logic
        pass

# Register with service
service = ValidationServiceImpl()
service.register_validator("custom", CustomValidator())
```

## File Structure

```
validator_api/
├── __init__.py                 # API exports
├── validation_interfaces.py    # Abstract interfaces
└── dtos.py                    # Data transfer objects

validator_impl/
├── __init__.py                # Implementation exports
├── pdt_validator.py           # PDT validator implementation
├── evidence_processor_impl.py # Evidence processor implementation
└── validation_service_impl.py # Service orchestrator implementation

# Refactored modules
validator_refactored.py        # Document validation orchestrator
evidence_processor_refactored.py # Evidence processing orchestrator

# Migration support
migration_guide.md            # Detailed migration guide
test_validator_api.py         # Architecture validation tests
```

## Interface Contracts

### ValidatorPort

```python
class ValidatorPort(ABC):
    @abstractmethod
    def validate(self, request: ValidationRequest) -> ValidationResult:
        """Validate input data according to defined rules."""
        
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules used by this validator."""
        
    @abstractmethod
    def supports_data_type(self, data_type: str) -> bool:
        """Check if validator supports given data type."""
```

### EvidenceProcessorPort

```python
class EvidenceProcessorPort(ABC):
    @abstractmethod
    def process_evidence(self, request: EvidenceProcessingRequest) -> EvidenceProcessingResponse:
        """Process raw evidence into structured format."""
        
    @abstractmethod
    def validate_evidence_structure(self, evidence: Dict[str, Any]) -> ValidationResult:
        """Validate evidence data structure."""
        
    @abstractmethod
    def extract_metadata(self, raw_data: Any) -> Dict[str, Any]:
        """Extract metadata from raw evidence."""
```

### ValidationService

```python
class ValidationService(ABC):
    @abstractmethod
    def register_validator(self, name: str, validator: ValidatorPort) -> None:
        """Register a validator with the service."""
        
    @abstractmethod
    def validate_with_pipeline(self, data: Any, validator_names: List[str]) -> Dict[str, ValidationResult]:
        """Run validation through pipeline of validators."""
        
    @abstractmethod
    def process_and_validate_evidence(self, raw_evidence: Any, processor_name: str, validator_names: Optional[List[str]]) -> Dict[str, Any]:
        """Process evidence and optionally validate it."""
```

## Testing Strategy

### Unit Tests

Test individual components in isolation:

```python
def test_pdt_validator():
    validator = PDTValidator()
    request = ValidationRequest(data=test_data, validation_type="pdt")
    result = validator.validate(request)
    
    assert result.is_success()
    assert result.confidence_score > 0.8
```

### Integration Tests

Test component interactions:

```python
def test_validation_service_integration():
    service = ValidationServiceImpl()
    service.register_validator("pdt", PDTValidator())
    
    result = service.validate_single(test_data, "pdt")
    assert result.is_success()
```

### Architecture Tests

Verify architectural constraints:

```python
def test_no_pipeline_imports():
    """Ensure validator_impl has no direct pipeline imports."""
    # Check import statements in implementation files
```

## Performance Considerations

### Initialization
- Validators are lightweight and fast to initialize
- Service registration is a one-time setup cost
- Consider lazy loading for heavy validators

### Validation Performance
- Interface calls have minimal overhead
- Validators can maintain internal caches
- Batch processing supported for multiple documents

### Memory Usage
- DTOs use dataclasses for efficient memory usage
- Optional fields with defaults reduce memory footprint
- Streaming support for large evidence batches

## Migration Path

1. **Phase 1**: Create interface packages (`validator_api`)
2. **Phase 2**: Implement concrete classes (`validator_impl`) 
3. **Phase 3**: Create orchestrators with dependency injection
4. **Phase 4**: Migrate existing code to use orchestrators
5. **Phase 5**: Remove old direct-import code

## Best Practices

### Interface Design
- Keep interfaces focused and cohesive
- Use DTOs for complex parameter passing
- Document interface contracts clearly
- Version interfaces for backward compatibility

### Implementation Guidelines
- Implementations should be self-contained
- No direct imports from pipeline components
- Use dependency injection for external dependencies
- Implement comprehensive error handling

### Service Configuration
- Register validators/processors at startup
- Use descriptive names for registration
- Implement health checks for registered components
- Support runtime component replacement

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure clean separation between api and impl packages
2. **Interface Mismatches**: Verify implementations satisfy interface contracts
3. **Registration Failures**: Check component types match expected interfaces
4. **Performance Issues**: Consider batch processing and caching strategies

### Debug Tools

- Health check endpoints for service status
- Component registration introspection
- Validation pipeline tracing
- Performance metrics collection

## Future Enhancements

1. **Plugin System**: Dynamic validator/processor loading
2. **Configuration Management**: External configuration for component behavior
3. **Metrics Collection**: Built-in performance and usage metrics
4. **Async Support**: Non-blocking validation operations
5. **Distributed Processing**: Multi-node validation pipeline support