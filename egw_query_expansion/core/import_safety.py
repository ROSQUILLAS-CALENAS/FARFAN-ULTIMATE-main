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
    
    def safe_import_scipy(self) -> ImportResult:
        """Safely import SciPy with appropriate fallbacks"""
        def scipy_fallback():
            """Minimal SciPy-like interface for basic operations"""
            import math
            
            class MockSciPy:
                class spatial:
                    class distance:
                        @staticmethod
                        def cosine(u, v):
                            """Basic cosine distance implementation"""
                            try:
                                np = _import_safety.safe_import_numpy().module
                                if np:
                                    dot = np.dot(u, v)
                                    norms = np.linalg.norm(u) * np.linalg.norm(v)
                                    return 1 - (dot / norms) if norms != 0 else 0
                                else:
                                    # Pure Python fallback
                                    dot = sum(a * b for a, b in zip(u, v))
                                    norm_u = math.sqrt(sum(a * a for a in u))
                                    norm_v = math.sqrt(sum(b * b for b in v))
                                    return 1 - (dot / (norm_u * norm_v)) if norm_u * norm_v != 0 else 0
                            except Exception:
                                return 0.5  # Neutral distance
                        
                        @staticmethod
                        def euclidean(u, v):
                            """Basic euclidean distance implementation"""
                            try:
                                return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))
                            except Exception:
                                return float('inf')
                
                class optimize:
                    class OptimizeResult:
                        def __init__(self, x, fun, success=True, message="Fallback optimization"):
                            self.x = x
                            self.fun = fun
                            self.success = success
                            self.message = message
                    
                    @staticmethod
                    def minimize(func, x0, **kwargs):
                        """Basic optimization fallback"""
                        return MockSciPy.optimize.OptimizeResult(x0, func(x0), False, "No SciPy available")
                
                class stats:
                    @staticmethod
                    def entropy(pk, qk=None, base=None):
                        """Basic entropy calculation"""
                        try:
                            if qk is None:
                                # Shannon entropy
                                entropy_val = -sum(p * math.log(p) for p in pk if p > 0)
                            else:
                                # Cross entropy
                                entropy_val = -sum(p * math.log(q) for p, q in zip(pk, qk) if q > 0)
                            
                            if base is not None:
                                entropy_val /= math.log(base)
                            return entropy_val
                        except Exception:
                            return 0.0
                    
                    @staticmethod
                    def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
                        """Basic Wasserstein distance approximation"""
                        # This is a very simplified approximation
                        try:
                            if u_weights is None:
                                u_weights = [1/len(u_values)] * len(u_values)
                            if v_weights is None:
                                v_weights = [1/len(v_values)] * len(v_values)
                            
                            # Sort and compute basic Earth Mover's Distance approximation
                            u_sorted = sorted(zip(u_values, u_weights))
                            v_sorted = sorted(zip(v_values, v_weights))
                            
                            # Simple approximation based on sorted values
                            return sum(abs(u - v) * min(wu, wv) 
                                     for (u, wu), (v, wv) in zip(u_sorted, v_sorted))
                        except Exception:
                            return float('inf')
            
            return MockSciPy()
        
        return self.safe_import(
            'scipy',
            fallback_factory=scipy_fallback,
            attributes=['spatial', 'optimize', 'stats'],
            min_version='1.7.0'
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
                        np = _import_safety.safe_import_numpy().module
                        if np and hasattr(vectors, 'shape'):
                            self._vectors.extend(vectors.tolist())
                            self.ntotal += len(vectors)
                    
                    def search(self, queries, k):
                        np = _import_safety.safe_import_numpy().module
                        if not np or not self._vectors:
                            return [], []
                        
                        # Minimal linear search using numpy if available
                        vectors = np.array(self._vectors)
                        distances = []
                        indices = []
                        
                        for query in queries:
                            dists = np.linalg.norm(vectors - query, axis=1)
                            idx = np.argsort(dists)[:k]
                            distances.append(dists[idx])
                            indices.append(idx)
                        
                        return np.array(distances), np.array(indices)
                
                class IndexFlatIP:
                    def __init__(self, d):
                        self.d = d
                        self.ntotal = 0
                        self._vectors = []
                    
                    def add(self, vectors):
                        np = _import_safety.safe_import_numpy().module
                        if np and hasattr(vectors, 'shape'):
                            self._vectors.extend(vectors.tolist())
                            self.ntotal += len(vectors)
                    
                    def search(self, queries, k):
                        np = _import_safety.safe_import_numpy().module
                        if not np or not self._vectors:
                            return [], []
                        
                        vectors = np.array(self._vectors)
                        scores = []
                        indices = []
                        
                        for query in queries:
                            # Inner product scores (higher is better)
                            inner_products = np.dot(vectors, query)
                            idx = np.argsort(inner_products)[::-1][:k]  # Descending order
                            scores.append(inner_products[idx])
                            indices.append(idx)
                        
                        return np.array(scores), np.array(indices)
            
            return MockFAISS()
        
        return self.safe_import(
            'faiss',
            fallback_factory=faiss_fallback,
            alternative_names=['faiss-cpu', 'faiss-gpu']
        )
    
    def safe_import_transformers(self) -> ImportResult:
        """Safely import transformers library"""
        def transformers_fallback():
            """Minimal transformers-like interface for basic operations"""
            class MockTransformers:
                class AutoTokenizer:
                    def __init__(self, model_name):
                        self.model_name = model_name
                        self.vocab_size = 50000  # Dummy vocab size
                    
                    @classmethod
                    def from_pretrained(cls, model_name_or_path, **kwargs):
                        return cls(model_name_or_path)
                    
                    def encode(self, text, **kwargs):
                        """Basic hash-based encoding fallback"""
                        if isinstance(text, str):
                            # Simple hash-based token generation
                            tokens = [hash(word) % self.vocab_size for word in text.split()]
                            if kwargs.get('return_tensors') == 'pt':
                                torch = _import_safety.safe_import_torch().module
                                if torch:
                                    return torch.tensor(tokens).unsqueeze(0)
                            return tokens
                        return []
                    
                    def decode(self, token_ids, **kwargs):
                        """Fallback decode - returns placeholder"""
                        return f"[DECODED_TOKENS_{len(token_ids) if hasattr(token_ids, '__len__') else 'UNKNOWN'}]"
                    
                    def __call__(self, text, **kwargs):
                        """Tokenize with basic fallback"""
                        if isinstance(text, str):
                            tokens = text.split()
                            encoding = {
                                'input_ids': self.encode(text, **kwargs),
                                'attention_mask': [1] * len(tokens)
                            }
                            if kwargs.get('return_tensors') == 'pt':
                                torch = _import_safety.safe_import_torch().module
                                if torch:
                                    encoding = {k: torch.tensor(v).unsqueeze(0) if isinstance(v, list) else v 
                                              for k, v in encoding.items()}
                            return encoding
                        return {'input_ids': [], 'attention_mask': []}
                
                class AutoModel:
                    def __init__(self, config=None):
                        self.config = config or {}
                    
                    @classmethod
                    def from_pretrained(cls, model_name_or_path, **kwargs):
                        return cls()
                    
                    def __call__(self, **kwargs):
                        """Mock forward pass"""
                        torch = _import_safety.safe_import_torch().module
                        if torch:
                            batch_size = 1
                            seq_len = 512
                            hidden_size = 768
                            
                            if 'input_ids' in kwargs:
                                input_ids = kwargs['input_ids']
                                if hasattr(input_ids, 'shape'):
                                    batch_size, seq_len = input_ids.shape
                            
                            # Return mock outputs
                            last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
                            pooler_output = torch.randn(batch_size, hidden_size)
                            
                            class MockOutput:
                                def __init__(self):
                                    self.last_hidden_state = last_hidden_state
                                    self.pooler_output = pooler_output
                                    self.hidden_states = (last_hidden_state,)
                            
                            return MockOutput()
                        
                        # Non-torch fallback
                        class MockOutput:
                            def __init__(self):
                                self.last_hidden_state = [[0.0] * 768] * 512
                                self.pooler_output = [0.0] * 768
                        
                        return MockOutput()
                    
                    def eval(self):
                        return self
                    
                    def to(self, device):
                        return self
                
                class pipeline:
                    def __init__(self, task, model=None, tokenizer=None, **kwargs):
                        self.task = task
                        self.model = model
                        self.tokenizer = tokenizer
                    
                    def __call__(self, inputs, **kwargs):
                        if self.task == "feature-extraction":
                            if isinstance(inputs, str):
                                # Return mock embeddings
                                return [[0.1] * 768]  # Mock 768-dim embeddings
                            elif isinstance(inputs, list):
                                return [[0.1] * 768] * len(inputs)
                        elif self.task == "text-classification":
                            return [{"label": "POSITIVE", "score": 0.5}]
                        elif self.task == "text-generation":
                            return [{"generated_text": inputs + " [GENERATED]"}]
                        return []
            
            return MockTransformers()
        
        return self.safe_import(
            'transformers',
            fallback_factory=transformers_fallback,
            attributes=['AutoModel', 'AutoTokenizer'],
            min_version='4.0.0'
        )
    
    def safe_import_sentence_transformers(self) -> ImportResult:
        """Safely import sentence-transformers library"""
        def sentence_transformers_fallback():
            """Minimal sentence-transformers-like interface"""
            class MockSentenceTransformer:
                def __init__(self, model_name_or_path='mock-model'):
                    self.model_name = model_name_or_path
                    self.max_seq_length = 512
                
                def encode(self, sentences, **kwargs):
                    """Mock encoding that returns random vectors"""
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    
                    # Mock 384-dimensional embeddings (common size)
                    np = _import_safety.safe_import_numpy().module
                    if np:
                        embeddings = np.random.randn(len(sentences), 384).astype(np.float32)
                        if kwargs.get('normalize_embeddings', False):
                            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                        return embeddings
                    else:
                        # Pure Python fallback
                        import random
                        return [[random.gauss(0, 1) for _ in range(384)] for _ in sentences]
                
                def similarity(self, embeddings1, embeddings2):
                    """Mock similarity computation"""
                    np = _import_safety.safe_import_numpy().module
                    if np:
                        return np.dot(embeddings1, embeddings2.T)
                    else:
                        # Basic dot product fallback
                        if len(embeddings1.shape) == 1:
                            embeddings1 = [embeddings1]
                        if len(embeddings2.shape) == 1:
                            embeddings2 = [embeddings2]
                        
                        similarities = []
                        for emb1 in embeddings1:
                            row = []
                            for emb2 in embeddings2:
                                sim = sum(a * b for a, b in zip(emb1, emb2))
                                row.append(sim)
                            similarities.append(row)
                        return similarities
                
                def get_sentence_embedding_dimension(self):
                    return 384
            
            class MockSentenceTransformers:
                SentenceTransformer = MockSentenceTransformer
            
            return MockSentenceTransformers()
        
        return self.safe_import(
            'sentence_transformers',
            fallback_factory=sentence_transformers_fallback,
            attributes=['SentenceTransformer'],
            min_version='2.0.0'
        )
    
    def safe_import_pot(self) -> ImportResult:
        """Safely import Python Optimal Transport (POT) library"""
        def pot_fallback():
            """Minimal POT-like interface for optimal transport"""
            class MockPOT:
                @staticmethod
                def emd(a, b, M, **kwargs):
                    """Mock Earth Mover's Distance"""
                    np = _import_safety.safe_import_numpy().module
                    if np:
                        # Return identity transport plan
                        n, m = len(a), len(b)
                        G = np.zeros((n, m))
                        # Simple greedy assignment
                        for i in range(min(n, m)):
                            G[i, i] = min(a[i], b[i])
                        return G
                    else:
                        # Basic fallback
                        return [[0.0] * len(b) for _ in range(len(a))]
                
                @staticmethod
                def gromov_wasserstein(C1, C2, p, q, loss_fun='square_loss', **kwargs):
                    """Mock Gromov-Wasserstein distance"""
                    np = _import_safety.safe_import_numpy().module
                    if np:
                        n, m = len(p), len(q)
                        # Return uniform coupling
                        G = np.outer(p, q)
                        return G
                    else:
                        return [[p[i] * q[j] for j in range(len(q))] for i in range(len(p))]
                
                @staticmethod
                def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun='square_loss', epsilon=0.1, **kwargs):
                    """Mock Entropic Gromov-Wasserstein distance"""
                    return MockPOT.gromov_wasserstein(C1, C2, p, q, loss_fun, **kwargs)
                
                class ot:
                    emd = MockPOT.emd
                    gromov_wasserstein = MockPOT.gromov_wasserstein
                    entropic_gromov_wasserstein = MockPOT.entropic_gromov_wasserstein
            
            return MockPOT()
        
        return self.safe_import(
            'ot',  # POT imports as 'ot'
            fallback_factory=pot_fallback,
            attributes=['emd', 'gromov_wasserstein'],
            min_version='0.8.0'
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

# Convenience functions for common libraries
def safe_import_numpy(**kwargs) -> ImportResult:
    """Safely import NumPy"""
    return _import_safety.safe_import_numpy()

def safe_import_scipy(**kwargs) -> ImportResult:
    """Safely import SciPy"""
    return _import_safety.safe_import_scipy()

def safe_import_torch(**kwargs) -> ImportResult:
    """Safely import PyTorch"""
    return _import_safety.safe_import_torch()

def safe_import_sklearn(**kwargs) -> ImportResult:
    """Safely import scikit-learn"""
    return _import_safety.safe_import_sklearn()

def safe_import_faiss(**kwargs) -> ImportResult:
    """Safely import FAISS"""
    return _import_safety.safe_import_faiss()

def safe_import_transformers(**kwargs) -> ImportResult:
    """Safely import transformers"""
    return _import_safety.safe_import_transformers()

def safe_import_sentence_transformers(**kwargs) -> ImportResult:
    """Safely import sentence-transformers"""
    return _import_safety.safe_import_sentence_transformers()

def safe_import_pot(**kwargs) -> ImportResult:
    """Safely import Python Optimal Transport (POT)"""
    return _import_safety.safe_import_pot()

def check_dependencies(dependencies: List[str], verbose: bool = True) -> Dict[str, bool]:
    """
    Check availability of multiple dependencies at once
    
    Args:
        dependencies: List of module names to check
        verbose: Whether to print status messages
        
    Returns:
        Dictionary mapping module names to availability status
    """
    results = {}
    
    for dep in dependencies:
        result = safe_import(dep, required=False)
        results[dep] = result.success
        
        if verbose:
            status = "✓ Available" if result.success else "✗ Missing"
            fallback_info = f" (using {result.fallback_used})" if result.fallback_used else ""
            print(f"{dep}: {status}{fallback_info}")
    
    return results

def log_import_summary():
    """Log a summary of all import attempts"""
    report = get_import_report()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Import Summary: {report['summary']['successful_imports']} successful, "
               f"{report['summary']['failed_imports']} failed, "
               f"{report['summary']['fallbacks_used']} using fallbacks")
    
    if report['summary']['critical_failures'] > 0:
        logger.warning(f"Critical failures: {report['critical_failures']}")
    
    return report