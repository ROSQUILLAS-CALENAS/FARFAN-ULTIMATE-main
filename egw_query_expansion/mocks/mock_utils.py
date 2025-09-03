"""
Mock Utilities - Common functionality for mock implementations

Provides shared utilities, deterministic helpers, and validation functions
used across all mock implementations in the EGW fallback system.
"""

import hashlib
import struct
import warnings
# # # from typing import Any, List, Union, Tuple, Optional  # Module not found  # Module not found  # Module not found


class MockRandomState:
    """Deterministic random state for consistent mock behavior"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._state = seed
        
    def random(self, size: Optional[Tuple[int, ...]] = None):
        """Generate deterministic random values"""
        if size is None:
            return self._next_float()
        
        # Generate array of random values
        result = []
        flat_size = 1
        for dim in size:
            flat_size *= dim
            
        for _ in range(flat_size):
            result.append(self._next_float())
            
        return self._reshape_list(result, size)
    
    def randint(self, low: int, high: int, size: Optional[Tuple[int, ...]] = None):
        """Generate deterministic random integers"""
        if size is None:
            return int(self._next_float() * (high - low)) + low
            
        result = []
        flat_size = 1
        for dim in size:
            flat_size *= dim
            
        for _ in range(flat_size):
            result.append(int(self._next_float() * (high - low)) + low)
            
        return self._reshape_list(result, size)
    
    def randn(self, *size):
        """Generate deterministic normally distributed values"""
        if not size:
            return self._next_normal()
            
        result = []
        flat_size = 1
        for dim in size:
            flat_size *= dim
            
        for _ in range(flat_size):
            result.append(self._next_normal())
            
        return self._reshape_list(result, size)
    
    def choice(self, a, size=None, replace=True, p=None):
        """Make deterministic random choices"""
        if isinstance(a, int):
            choices = list(range(a))
        else:
            choices = list(a)
            
        if size is None:
            idx = int(self._next_float() * len(choices))
            return choices[idx]
            
        if isinstance(size, int):
            size = (size,)
            
        result = []
        flat_size = 1
        for dim in size:
            flat_size *= dim
            
        for _ in range(flat_size):
            idx = int(self._next_float() * len(choices))
            result.append(choices[idx])
            
        return self._reshape_list(result, size)
    
    def _next_float(self) -> float:
        """Generate next deterministic float"""
        self._state = (self._state * 1103515245 + 12345) & 0x7fffffff
        return self._state / 0x7fffffff
    
    def _next_normal(self) -> float:
        """Generate normally distributed value using Box-Muller"""
        if not hasattr(self, '_spare_normal'):
            # Box-Muller transformation
            u1 = self._next_float()
            u2 = self._next_float()
            
            import math
            mag = 0.1 * math.sqrt(-2.0 * math.log(u1))
            self._spare_normal = mag * math.cos(2.0 * math.pi * u2)
            return mag * math.sin(2.0 * math.pi * u2)
        else:
            result = self._spare_normal
            delattr(self, '_spare_normal')
            return result
    
    def _reshape_list(self, flat_list: List[Any], shape: Tuple[int, ...]) -> List:
        """Reshape flat list into nested structure"""
        if len(shape) == 1:
            return flat_list[:shape[0]]
        
        result = []
        items_per_row = len(flat_list) // shape[0]
        
        for i in range(shape[0]):
            start_idx = i * items_per_row
            end_idx = start_idx + items_per_row
            row_data = flat_list[start_idx:end_idx]
            
            if len(shape) > 2:
                # Recursively reshape for higher dimensions
                new_shape = shape[1:]
                result.append(self._reshape_list(row_data, new_shape))
            else:
                result.append(row_data)
                
        return result


class DeterministicHasher:
    """Creates deterministic hashes for consistent mock behavior"""
    
    @staticmethod
    def hash_array(data: Any, salt: str = "mock") -> int:
# # #         """Create deterministic hash from array-like data"""  # Module not found  # Module not found  # Module not found
        # Convert data to string representation
        if hasattr(data, 'tolist'):
            str_data = str(data.tolist())
        else:
            str_data = str(data)
            
        # Add salt and create hash
        salted_data = f"{salt}:{str_data}"
        hash_bytes = hashlib.sha256(salted_data.encode()).digest()
        
        # Convert to integer
        return struct.unpack('>Q', hash_bytes[:8])[0]
    
    @staticmethod
    def hash_shape(shape: Tuple[int, ...], dtype: str = "float32") -> int:
# # #         """Create hash from array shape and dtype"""  # Module not found  # Module not found  # Module not found
        shape_str = f"{shape}:{dtype}"
        hash_bytes = hashlib.sha256(shape_str.encode()).digest()
        return struct.unpack('>Q', hash_bytes[:8])[0]


def mock_warning(message: str) -> None:
    """Issue a warning about mock fallback usage"""
    warnings.warn(
        f"EGW Mock Fallback: {message}",
        UserWarning,
        stacklevel=3
    )


def validate_mock_compatibility(mock_obj: Any, expected_methods: List[str]) -> bool:
    """Validate that mock object implements expected methods"""
    for method_name in expected_methods:
        if not hasattr(mock_obj, method_name):
            return False
        if not callable(getattr(mock_obj, method_name)):
            return False
    return True


def create_deterministic_data(shape: Tuple[int, ...], dtype: str = "float32", seed: int = 42) -> List:
    """Create deterministic data arrays for mock implementations"""
    rng = MockRandomState(seed)
    
    # Generate data based on dtype
    if dtype in ["float32", "float64", "float"]:
        return rng.random(shape)
    elif dtype in ["int32", "int64", "int"]:
        return rng.randint(0, 100, shape)
    elif dtype == "bool":
        return [[bool(x > 0.5) for x in row] if isinstance(row, list) else bool(row > 0.5) 
                for row in rng.random(shape)]
    else:
        # Default to float
        return rng.random(shape)


def ensure_tuple(value: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Ensure value is a tuple"""
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def flatten_shape(shape: Tuple[int, ...]) -> int:
    """Calculate total number of elements in shape"""
    result = 1
    for dim in shape:
        result *= dim
    return result