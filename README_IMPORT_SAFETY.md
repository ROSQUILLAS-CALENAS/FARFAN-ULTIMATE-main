# Import Safety Module

The `import_safety` module provides robust dependency management for the EGW Query Expansion system, implementing graceful degradation when optional dependencies are unavailable.

## Features

### Dynamic Import Handling
- **Safe Import Functions**: Try-catch wrappers for all dependency imports
- **Fallback Implementations**: Minimal mock implementations when packages are missing  
- **Alternative Names**: Support for multiple package names (e.g., `faiss`, `faiss-cpu`, `faiss-gpu`)
- **Version Checking**: Minimum version validation for critical dependencies
- **Attribute Validation**: Verify required attributes exist in imported modules

### Comprehensive Logging
- **Import Status Tracking**: Global registry of successful and failed imports
- **Fallback Usage Logging**: Track when and which fallbacks are used
- **Degraded Mode Warnings**: Informative messages about reduced functionality
- **Import Reports**: Detailed summaries of dependency availability

### Specialized Handlers
- **NumPy**: Core numerical computing with pure Python fallbacks
- **SciPy**: Scientific computing with basic implementations for distances, optimization, and stats
- **PyTorch**: Deep learning framework with tensor operation mocks
- **scikit-learn**: Machine learning library with basic preprocessing fallbacks  
- **FAISS**: Vector search with linear search fallback implementation
- **Transformers**: Hugging Face library with mock tokenization and models
- **Sentence Transformers**: Sentence embeddings with random vector fallbacks
- **POT (Python Optimal Transport)**: Optimal transport with basic coupling fallbacks

## Usage

### Basic Import Safety

```python
from egw_query_expansion.core.import_safety import safe_import

# Safe import with fallback
result = safe_import('numpy', fallback_factory=lambda: MockNumPy())
if result.success:
    np = result.module
    # Use numpy or fallback
else:
    print(f"Import failed: {result.error}")
```

### Specialized Imports

```python
from egw_query_expansion.core.import_safety import (
    safe_import_numpy, safe_import_scipy, safe_import_torch,
    safe_import_sklearn, safe_import_faiss
)

# Import with built-in fallbacks
numpy_result = safe_import_numpy()
np = numpy_result.module  # Either numpy or fallback

scipy_result = safe_import_scipy()  
scipy = scipy_result.module  # Either scipy or mock with basic functions

torch_result = safe_import_torch()
torch = torch_result.module  # Either torch or mock tensors
```

### Batch Dependency Checking

```python
from egw_query_expansion.core.import_safety import check_dependencies

# Check multiple dependencies at once
dependencies = ['numpy', 'scipy', 'torch', 'sklearn', 'faiss']
results = check_dependencies(dependencies, verbose=True)
# Output:
# numpy: ✓ Available
# scipy: ✗ Missing (using fallback)
# torch: ✓ Available  
# sklearn: ✗ Missing (using registry)
# faiss: ✓ Available
```

### Import Reporting

```python
from egw_query_expansion.core.import_safety import get_import_report, log_import_summary

# Get detailed report
report = get_import_report()
print(f"Total attempts: {report['summary']['total_attempts']}")
print(f"Successful: {report['summary']['successful_imports']}")  
print(f"Failed: {report['summary']['failed_imports']}")
print(f"Using fallbacks: {report['summary']['fallbacks_used']}")

# Log summary to console
log_import_summary()
```

### Custom Fallbacks

```python
from egw_query_expansion.core.import_safety import register_fallback, register_mock

# Register custom fallback factory
def my_numpy_fallback():
    class MockNumPy:
        def array(self, data):
            return data  # Simplistic fallback
    return MockNumPy()

register_fallback('numpy', my_numpy_fallback)

# Register pre-built mock
class CustomMock:
    def special_function(self):
        return "fallback result"

register_mock('custom_package', CustomMock())
```

### Decorator Pattern

```python
from egw_query_expansion.core.import_safety import import_with_fallback

@import_with_fallback(fallback_value=[])
def function_needing_numpy():
    import numpy as np
    return np.array([1, 2, 3]).tolist()

# Returns fallback value if numpy import fails
result = function_needing_numpy()  # Either [1, 2, 3] or []
```

## Fallback Implementations

### SciPy Fallbacks
- `spatial.distance.cosine()`: Pure Python cosine distance
- `spatial.distance.euclidean()`: Pure Python Euclidean distance  
- `optimize.minimize()`: Basic optimization stub
- `stats.entropy()`: Shannon and cross-entropy calculations
- `stats.wasserstein_distance()`: Simplified Earth Mover's Distance

### FAISS Fallbacks
- `IndexFlatL2`: Linear search with L2 distance
- `IndexFlatIP`: Linear search with inner product similarity
- Support for basic `add()` and `search()` operations

### Transformers Fallbacks  
- `AutoTokenizer`: Hash-based tokenization with tensor support
- `AutoModel`: Mock transformer with random embeddings
- `pipeline`: Basic feature extraction and text generation stubs

### Sentence Transformers Fallbacks
- `SentenceTransformer`: Random 384-dimensional embeddings
- `encode()`: Batch encoding with normalization support
- `similarity()`: Cosine similarity computation

## Configuration

The module uses a singleton pattern to maintain global state:

```python
from egw_query_expansion.core.import_safety import ImportSafety

# Access global instance
safety = ImportSafety()

# Configure logging level
safety.logger.setLevel(logging.INFO)

# Clear cache for testing
safety.clear_cache()

# Check internal state
print(f"Failed imports: {len(safety.failed_imports)}")
print(f"Successful imports: {len(safety.successful_imports)}")
```

## Integration with EGW System

The import safety module integrates seamlessly with the EGW Query Expansion system:

1. **Automatic Fallbacks**: Core modules automatically use safe imports
2. **Graceful Degradation**: System continues operating with reduced functionality
3. **Performance Monitoring**: Track impact of missing dependencies
4. **Development Support**: Easy testing with selective dependency availability

## Testing

```bash
# Run the demo to see functionality
python3 demo_import_safety.py

# Check basic functionality  
python3 validate_import_safety.py
```

## Error Handling

The module provides comprehensive error handling:

- **ImportError**: Caught and logged with fallback activation
- **Version Errors**: Minimum version requirements enforced
- **Attribute Errors**: Required module attributes validated
- **Fallback Errors**: Graceful handling of fallback failures

## Performance Considerations

- **Caching**: Import results cached to avoid repeated attempts
- **Lazy Loading**: Fallbacks only created when needed
- **Minimal Overhead**: Fast-path for successful imports
- **Memory Efficient**: Fallbacks designed to be lightweight

This module ensures the EGW Query Expansion system remains robust and functional even in environments with limited dependency availability.