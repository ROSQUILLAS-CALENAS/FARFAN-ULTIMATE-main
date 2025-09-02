"""
Import Safety Utility for EGW Query Expansion System

Provides standardized try/except patterns for critical dependencies with
graceful degradation and comprehensive logging. Maintains a global registry
of import failures and coordinates with existing mock systems.
"""

import logging
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from functools import wraps
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ImportResult:
    """Container for import attempt results"""
    module_name: str
    success: bool
    module: Optional[Any] = None
    error: Optional[Exception] = None
    fallback_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImportSafety:
    """
    Centralized import safety manager with consistent error handling,
    logging, and fallback coordination.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern for global import registry"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self.failed_imports: Dict[str, ImportResult] = {}
        self.successful_imports: Dict[str, ImportResult] = {}
        self.fallback_registry: Dict[str, Callable] = {}
        self.mock_registry: Dict[str, Any] = {}
        self._lock = Lock()
        self._initialized = True
        
        # Setup logging handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(levelname)s] %(name)s: %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
    
    def safe_import(
        self,
        module_name: str,
        package: Optional[str] = None,
        fallback_factory: Optional[Callable] = None,
        mock_factory: Optional[Callable] = None,
        required: bool = True,
        min_version: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        alternative_names: Optional[List[str]] = None
    ) -> ImportResult:
        """
        Safely import a module with comprehensive error handling and fallback support.
        
        Args:
            module_name: Primary module name to import
            package: Package context for relative imports
            fallback_factory: Factory function to create fallback implementation
            mock_factory: Factory function to create mock implementation
            required: Whether import failure should trigger warnings
            min_version: Minimum required version (if applicable)
            attributes: List of required attributes to verify
            alternative_names: Alternative module names to try
            
        Returns:
            ImportResult containing import status and module/fallback
        """
        cache_key = f"{module_name}:{package or ''}"
        
        # Check cache first
        with self._lock:
            if cache_key in self.successful_imports:
                return self.successful_imports[cache_key]
            if cache_key in self.failed_imports:
                cached = self.failed_imports[cache_key]
                if cached.fallback_used or not required:
                    return cached
        
        # Attempt import with comprehensive error handling
        result = self._attempt_import(
            module_name=module_name,
            package=package,
            min_version=min_version,
            attributes=attributes,
            alternative_names=alternative_names
        )
        
        # Handle import failure
        if not result.success:
            result = self._handle_import_failure(
                result=result,
                fallback_factory=fallback_factory,
                mock_factory=mock_factory,
                required=required
            )
        
        # Cache and return result
        with self._lock:
            if result.success:
                self.successful_imports[cache_key] = result
            else:
                self.failed_imports[cache_key] = result
        
        return result
    
    def _attempt_import(
        self,
        module_name: str,
        package: Optional[str] = None,
        min_version: Optional[str] = None,
        attributes: Optional[List[str]] = None,
        alternative_names: Optional[List[str]] = None
    ) -> ImportResult:
        """Attempt to import module with version and attribute validation"""
        
        # List of names to try (primary + alternatives)
        names_to_try = [module_name]
        if alternative_names:
            names_to_try.extend(alternative_names)
        
        last_error = None
        
        for name in names_to_try:
            try:
                # Attempt import
                if package:
                    module = __import__(f"{package}.{name}", fromlist=[name])
                    module = getattr(module, name, module)
                else:
                    module = __import__(name)
                    # Handle dotted imports (e.g., 'torch.nn.functional')
                    components = name.split('.')
                    for component in components[1:]:
                        module = getattr(module, component)
                
                # Version validation
                if min_version and hasattr(module, '__version__'):
                    if not self._check_version(module.__version__, min_version):
                        raise ImportError(
                            f"Module {name} version {module.__version__} "
                            f"is below minimum required version {min_version}"
                        )
                
                # Attribute validation
                if attributes:
                    missing_attrs = [
                        attr for attr in attributes 
                        if not hasattr(module, attr)
                    ]
                    if missing_attrs:
                        raise ImportError(
                            f"Module {name} missing required attributes: {missing_attrs}"
                        )
                
                # Success
                return ImportResult(
                    module_name=module_name,
                    success=True,
                    module=module,
                    metadata={'actual_name': name, 'has_version': hasattr(module, '__version__')}
                )
                
            except Exception as e:
                last_error = e
                continue
        
        # All attempts failed
        return ImportResult(
            module_name=module_name,
            success=False,
            error=last_error or ImportError(f"Could not import {module_name}")
        )
    
    def _handle_import_failure(
        self,
        result: ImportResult,
        fallback_factory: Optional[Callable] = None,
        mock_factory: Optional[Callable] = None,
        required: bool = True
    ) -> ImportResult:
        """Handle import failure with fallbacks and logging"""
        
        module_name = result.module_name
        error = result.error
        
        # Try fallback factory
        if fallback_factory:
            try:
                fallback_module = fallback_factory()
                result.module = fallback_module
                result.fallback_used = "factory"
                result.success = True
                
                if required:
                    self.logger.warning(
                        f"Import failed for {module_name}: {error}. "
                        f"Using fallback implementation."
                    )
                
                return result
            except Exception as fallback_error:
                self.logger.warning(
                    f"Fallback factory failed for {module_name}: {fallback_error}"
                )
        
        # Try mock factory
        if mock_factory:
            try:
                mock_module = mock_factory()
                result.module = mock_module
                result.fallback_used = "mock"
                result.success = True
                
                if required:
                    self.logger.warning(
                        f"Import failed for {module_name}: {error}. "
                        f"Using mock implementation."
                    )
                
                return result
            except Exception as mock_error:
                self.logger.warning(
                    f"Mock factory failed for {module_name}: {mock_error}"
                )
        
        # Try registered fallbacks
        if module_name in self.fallback_registry:
            try:
                fallback_module = self.fallback_registry[module_name]()
                result.module = fallback_module
                result.fallback_used = "registry"
                result.success = True
                
                if required:
                    self.logger.warning(
                        f"Import failed for {module_name}: {error}. "
                        f"Using registered fallback."
                    )
                
                return result
            except Exception as registry_error:
                self.logger.warning(
                    f"Registry fallback failed for {module_name}: {registry_error}"
                )
        
        # Try registered mocks
        if module_name in self.mock_registry:
            result.module = self.mock_registry[module_name]
            result.fallback_used = "mock_registry"
            result.success = True
            
            if required:
                self.logger.warning(
                    f"Import failed for {module_name}: {error}. "
                    f"Using registered mock."
                )
            
            return result
        
        # Complete failure
        if required:
            self.logger.error(
                f"Critical import failed for {module_name}: {error}. "
                f"No fallback available. System may have degraded functionality."
            )
        else:
            self.logger.debug(
                f"Optional import failed for {module_name}: {error}. "
                f"Continuing without this dependency."
            )
        
        return result
    
    def register_fallback(self, module_name: str, factory: Callable):
        """Register a fallback factory for a module"""
        with self._lock:
            self.fallback_registry[module_name] = factory
    
    def register_mock(self, module_name: str, mock_module: Any):
        """Register a mock implementation for a module"""
        with self._lock:
            self.mock_registry[module_name] = mock_module
    
    def get_import_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of import status"""
        with self._lock:
            successful_count = len(self.successful_imports)
            failed_count = len(self.failed_imports)
            fallback_count = len([
                r for r in self.failed_imports.values() 
                if r.fallback_used
            ])
            
            critical_failures = [
                name for name, result in self.failed_imports.items()
                if not result.fallback_used
            ]
            
            return {
                'summary': {
                    'total_attempts': successful_count + failed_count,
                    'successful_imports': successful_count,
                    'failed_imports': failed_count,
                    'fallbacks_used': fallback_count,
                    'critical_failures': len(critical_failures)
                },
                'successful_modules': list(self.successful_imports.keys()),
                'failed_modules': list(self.failed_imports.keys()),
                'critical_failures': critical_failures,
                'fallback_types': {
                    name: result.fallback_used 
                    for name, result in self.failed_imports.items()
                    if result.fallback_used
                },
                'registered_fallbacks': list(self.fallback_registry.keys()),
                'registered_mocks': list(self.mock_registry.keys())
            }
    
    def clear_cache(self):
        """Clear import cache (useful for testing)"""
        with self._lock:
            self.failed_imports.clear()
            self.successful_imports.clear()
    
    def _check_version(self, current: str, minimum: str) -> bool:
        """Simple version comparison (major.minor.patch)"""
        try:
            def parse_version(v):
                return tuple(map(int, v.split('.')))
            return parse_version(current) >= parse_version(minimum)
        except Exception:
            return True  # If can't parse, assume OK
    
    # Convenience methods for common import patterns
    def safe_import_torch(self) -> ImportResult:
        """Safely import PyTorch with appropriate fallbacks"""
        return self.safe_import(
            'torch',
            attributes=['tensor', 'nn', 'optim'],
            min_version='1.7.0'
        )
    
    def safe_import_numpy(self) -> ImportResult:
        """Safely import NumPy with appropriate fallbacks"""
        return self.safe_import(
            'numpy',
            attributes=['array', 'ndarray', 'linalg'],
            alternative_names=['np']
        )
    
    def safe_import_sklearn(self) -> ImportResult:
        """Safely import scikit-learn with appropriate fallbacks"""
        return self.safe_import(
            'sklearn',
            attributes=['metrics', 'preprocessing'],
            alternative_names=['sklearn']
        )
    
    def safe_import_faiss(self) -> ImportResult:
        """Safely import FAISS with CPU fallback"""
        def faiss_fallback():
            """Minimal FAISS-like interface for basic operations"""
            class MockFAISS:
                class IndexFlatL2:
                    def __init__(self, d):
                        self.d = d
                        self.ntotal = 0
                        self._vectors = []
                    
                    def add(self, vectors):
                        import numpy as np
                        if hasattr(vectors, 'shape'):
                            self._vectors.extend(vectors.tolist())
                            self.ntotal += len(vectors)
                    
                    def search(self, queries, k):
                        import numpy as np
                        if not self._vectors:
                            return np.array([]), np.array([])
                        
                        # Minimal linear search
                        vectors = np.array(self._vectors)
                        distances = []
                        indices = []
                        
                        for query in queries:
                            dists = np.linalg.norm(vectors - query, axis=1)
                            idx = np.argsort(dists)[:k]
                            distances.append(dists[idx])
                            indices.append(idx)
                        
                        return np.array(distances), np.array(indices)
            
            return MockFAISS()
        
        return self.safe_import(
            'faiss',
            fallback_factory=faiss_fallback,
            alternative_names=['faiss-cpu', 'faiss-gpu']
        )
    
    def safe_import_transformers(self) -> ImportResult:
        """Safely import transformers library"""
        return self.safe_import(
            'transformers',
            attributes=['AutoModel', 'AutoTokenizer'],
            min_version='4.0.0'
        )


# Global instance
_import_safety = ImportSafety()

# Convenience functions for common usage patterns
def safe_import(module_name: str, **kwargs) -> ImportResult:
    """Convenience wrapper for ImportSafety.safe_import()"""
    return _import_safety.safe_import(module_name, **kwargs)

def get_import_report() -> Dict[str, Any]:
    """Get global import status report"""
    return _import_safety.get_import_report()

def register_fallback(module_name: str, factory: Callable):
    """Register a global fallback factory"""
    _import_safety.register_fallback(module_name, factory)

def register_mock(module_name: str, mock_module: Any):
    """Register a global mock implementation"""
    _import_safety.register_mock(module_name, mock_module)

def import_with_fallback(fallback_value: Any = None):
    """Decorator for functions that need import-safe behavior"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                _import_safety.logger.warning(
                    f"Import error in {func.__name__}: {e}. Using fallback."
                )
                return fallback_value
        return wrapper
    return decorator