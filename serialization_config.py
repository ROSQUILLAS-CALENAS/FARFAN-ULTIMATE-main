"""
Serialization Configuration Module

Provides centralized configuration and utilities for function serialization
in distributed processing pipelines.
"""

import os
# # # from typing import Dict, Any, Optional  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found


@dataclass
class SerializationConfig:
    """Configuration for serialization backends."""
    
    backend: str = "dill"  # Default to dill
    fallback_enabled: bool = True
    timeout_seconds: float = 30.0
    compression_enabled: bool = False
    compression_level: int = 6
    max_retry_attempts: int = 3
    
    @classmethod
    def from_environment(cls) -> 'SerializationConfig':
# # #         """Create configuration from environment variables."""  # Module not found  # Module not found  # Module not found
        return cls(
            backend=os.getenv('SERIALIZATION_BACKEND', 'dill'),
            fallback_enabled=os.getenv('SERIALIZATION_FALLBACK', 'true').lower() == 'true',
            timeout_seconds=float(os.getenv('SERIALIZATION_TIMEOUT', '30.0')),
            compression_enabled=os.getenv('SERIALIZATION_COMPRESSION', 'false').lower() == 'true',
            compression_level=int(os.getenv('SERIALIZATION_COMPRESSION_LEVEL', '6')),
            max_retry_attempts=int(os.getenv('SERIALIZATION_MAX_RETRIES', '3'))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'backend': self.backend,
            'fallback_enabled': self.fallback_enabled,
            'timeout_seconds': self.timeout_seconds,
            'compression_enabled': self.compression_enabled,
            'compression_level': self.compression_level,
            'max_retry_attempts': self.max_retry_attempts
        }


def get_default_serialization_config() -> SerializationConfig:
    """Get default serialization configuration."""
    return SerializationConfig.from_environment()


def validate_serialization_backend(backend: str) -> bool:
    """Validate if a serialization backend is available."""
    try:
        if backend == "dill":
            import dill
            return True
        elif backend == "cloudpickle":
            import cloudpickle
            return True
        elif backend == "pickle":
            import pickle
            return True
        else:
            return False
    except ImportError:
        return False


def get_available_backends() -> list[str]:
    """Get list of available serialization backends."""
    backends = []
    
    for backend in ["dill", "cloudpickle", "pickle"]:
        if validate_serialization_backend(backend):
            backends.append(backend)
    
    return backends


def get_recommended_backend() -> str:
    """Get recommended serialization backend based on availability."""
    for backend in ["dill", "cloudpickle", "pickle"]:
        if validate_serialization_backend(backend):
            return backend
    
    raise RuntimeError("No serialization backend available")