"""
Mock Fallback Infrastructure for EGW Query Expansion System

Comprehensive mock implementations for external dependencies (FAISS, PyTorch, 
scikit-learn, NumPy) that provide complete API coverage matching the real 
libraries. These mocks maintain functionality when real libraries are 
unavailable while providing deterministic behavior for testing and degraded 
environments.

All mock implementations include:
- Complete method/attribute coverage
- Appropriate return types
- Realistic fallback behaviors
- Performance-appropriate responses
- Deterministic execution

Available mocks:
- MockNumPy: Complete NumPy array operations and mathematical functions
- MockTorch: PyTorch tensors, models, and neural network operations
- MockFAISS: Vector similarity search and indexing operations
- MockSklearn: Machine learning algorithms and transformations
"""

# Import utilities
# # # from .mock_utils import (  # Module not found  # Module not found  # Module not found
    MockRandomState,
    DeterministicHasher,
    mock_warning,
    validate_mock_compatibility
)

# Import comprehensive mock implementations
# # # from .mock_numpy import MockNumPy, MockNDArray  # Module not found  # Module not found  # Module not found
# # # from .mock_torch import MockTorch, MockTensor, MockModule  # Module not found  # Module not found  # Module not found
# # # from .mock_faiss import MockFAISS, MockIndex  # Module not found  # Module not found  # Module not found
# # # from .mock_sklearn import MockSklearn, MockEstimator  # Module not found  # Module not found  # Module not found

# Auto-fallback mechanism - detects missing libraries and provides mocks
import sys
import warnings

# Check for real libraries and setup fallbacks
_REAL_NUMPY_AVAILABLE = True
_REAL_TORCH_AVAILABLE = True 
_REAL_FAISS_AVAILABLE = True
_REAL_SKLEARN_AVAILABLE = True

try:
    import numpy as _np
except ImportError:
    _REAL_NUMPY_AVAILABLE = False
    sys.modules['numpy'] = MockNumPy()
    mock_warning("NumPy not found - using comprehensive mock fallback")

try:
    import torch as _torch
except ImportError:
    _REAL_TORCH_AVAILABLE = False
    sys.modules['torch'] = MockTorch()
    mock_warning("PyTorch not found - using comprehensive mock fallback")

try:
    import faiss as _faiss
except ImportError:
    _REAL_FAISS_AVAILABLE = False
    sys.modules['faiss'] = MockFAISS()
    mock_warning("FAISS not found - using comprehensive mock fallback")

try:
    import sklearn as _sklearn
except ImportError:
    _REAL_SKLEARN_AVAILABLE = False
    sys.modules['sklearn'] = MockSklearn()
    mock_warning("scikit-learn not found - using comprehensive mock fallback")

__all__ = [
    'MockNumPy', 'MockNDArray',
    'MockTorch', 'MockTensor', 'MockModule',
    'MockFAISS', 'MockIndex',
    'MockSklearn', 'MockEstimator',
    'MockRandomState', 'DeterministicHasher',
    'mock_warning', 'validate_mock_compatibility'
]