"""
Comprehensive FAISS Mock Implementation

Provides complete API coverage for FAISS vector similarity search, indexing,
clustering, and other core FAISS functionality. Maintains deterministic
behavior and realistic return types for testing and fallback scenarios.
"""

import math
from typing import Any, List, Tuple, Union, Optional, Dict
from .mock_utils import MockRandomState, DeterministicHasher, create_deterministic_data


class MockIndex:
    """Mock implementation of FAISS Index with complete API coverage"""
    
    def __init__(self, dimension: int, metric_type: str = "L2"):
        self.d = dimension  # FAISS convention for dimension
        self.ntotal = 0     # Number of vectors in index
        self.metric_type = metric_type
        self.is_trained = True
        self.verbose = False
        
        # Storage for vectors and metadata
        self._vectors = []
        self._ids = []
        self._rng = MockRandomState(42)
    
    def add(self, vectors):
        """Add vectors to index"""
        if hasattr(vectors, 'shape'):
            # Numpy-like array
            n_vectors = vectors.shape[0] if len(vectors.shape) > 1 else 1
            if len(vectors.shape) > 1 and vectors.shape[1] != self.d:
                raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.d}")
        elif isinstance(vectors, list):
            # List of vectors
            n_vectors = len(vectors)
            if n_vectors > 0 and len(vectors[0]) != self.d:
                raise ValueError(f"Vector dimension {len(vectors[0])} doesn't match index dimension {self.d}")
        else:
            n_vectors = 1
            
        # Store vectors (simplified - just store count and dimension info)
        for i in range(n_vectors):
            self._vectors.append(self._create_mock_vector())
            self._ids.append(self.ntotal + i)
            
        self.ntotal += n_vectors
    
    def add_with_ids(self, vectors, ids):
        """Add vectors with specific IDs"""
        if hasattr(vectors, 'shape'):
            n_vectors = vectors.shape[0] if len(vectors.shape) > 1 else 1
        elif isinstance(vectors, list):
            n_vectors = len(vectors)
        else:
            n_vectors = 1
            
        if hasattr(ids, '__len__') and len(ids) != n_vectors:
            raise ValueError("Number of vectors and IDs must match")
            
        # Store vectors with provided IDs
        for i in range(n_vectors):
            self._vectors.append(self._create_mock_vector())
            vector_id = ids[i] if hasattr(ids, '__getitem__') else ids
            self._ids.append(vector_id)
            
        self.ntotal += n_vectors
    
    def search(self, query_vectors, k: int):
        """Search for k nearest neighbors"""
        if hasattr(query_vectors, 'shape'):
            n_queries = query_vectors.shape[0] if len(query_vectors.shape) > 1 else 1
        elif isinstance(query_vectors, list):
            n_queries = len(query_vectors)
        else:
            n_queries = 1
            
        # Generate mock results
        distances = []
        indices = []
        
        for q in range(n_queries):
            # Generate deterministic but realistic distances and indices
            query_distances = []
            query_indices = []
            
            # Ensure we don't return more results than available
            actual_k = min(k, self.ntotal) if self.ntotal > 0 else 0
            
            for i in range(actual_k):
                # Generate deterministic distance based on query and index
                distance = self._generate_distance(q, i)
                index = i % self.ntotal if self.ntotal > 0 else -1
                
                query_distances.append(distance)
                query_indices.append(index)
            
            # Pad with -1s if not enough results
            while len(query_distances) < k:
                query_distances.append(float('inf'))
                query_indices.append(-1)
                
            distances.append(query_distances)
            indices.append(query_indices)
        
        return distances, indices
    
    def train(self, vectors):
        """Train index (no-op for most index types)"""
        self.is_trained = True
    
    def reset(self):
        """Reset index to empty state"""
        self.ntotal = 0
        self._vectors.clear()
        self._ids.clear()
    
    def remove_ids(self, ids):
        """Remove vectors by IDs"""
        # Simplified implementation
        if hasattr(ids, '__iter__'):
            n_removed = len(ids)
        else:
            n_removed = 1
            
        self.ntotal = max(0, self.ntotal - n_removed)
        return n_removed
    
    def reconstruct(self, key: int):
        """Reconstruct vector by key"""
        if 0 <= key < len(self._vectors):
            return self._vectors[key]
        else:
            return self._create_mock_vector()
    
    def reconstruct_n(self, n0: int, n: int):
        """Reconstruct n vectors starting from n0"""
        vectors = []
        for i in range(n):
            if n0 + i < len(self._vectors):
                vectors.append(self._vectors[n0 + i])
            else:
                vectors.append(self._create_mock_vector())
        return vectors
    
    def _create_mock_vector(self):
        """Create a mock vector for this index"""
        return [self._rng.random() for _ in range(self.d)]
    
    def _generate_distance(self, query_idx: int, result_idx: int) -> float:
        """Generate deterministic distance for search results"""
        # Use hash-based deterministic distance generation
        hash_val = DeterministicHasher.hash_array(f"{query_idx}:{result_idx}")
        
        # Generate distance based on metric type
        if self.metric_type == "L2":
            # L2 distances are non-negative, typically small for good matches
            base_distance = (hash_val % 1000) / 1000.0
            return base_distance * (result_idx + 1)  # Closer results have smaller distances
        elif self.metric_type == "IP":
            # Inner product - higher is better, can be negative
            base_score = (hash_val % 2000) / 1000.0 - 1.0  # Range [-1, 1]
            return base_score / (result_idx + 1)  # Better results have higher scores
        else:
            return (hash_val % 1000) / 1000.0


class MockIndexFlat(MockIndex):
    """Mock implementation of IndexFlat"""
    
    def __init__(self, dimension: int, metric=None):
        metric_type = "L2" if metric is None else str(metric)
        super().__init__(dimension, metric_type)


class MockIndexFlatIP(MockIndex):
    """Mock implementation of IndexFlatIP (Inner Product)"""
    
    def __init__(self, dimension: int):
        super().__init__(dimension, "IP")


class MockIndexFlatL2(MockIndex):
    """Mock implementation of IndexFlatL2"""
    
    def __init__(self, dimension: int):
        super().__init__(dimension, "L2")


class MockIndexIVFFlat(MockIndex):
    """Mock implementation of IndexIVFFlat"""
    
    def __init__(self, quantizer, dimension: int, nlist: int, metric=None):
        metric_type = "L2" if metric is None else str(metric)
        super().__init__(dimension, metric_type)
        self.quantizer = quantizer
        self.nlist = nlist
        self.nprobe = 1  # Default number of probes
        self.is_trained = False  # IVF indices need training
    
    def train(self, vectors):
        """Train IVF index"""
        super().train(vectors)
        # Mock training process
        if hasattr(vectors, 'shape'):
            n_vectors = vectors.shape[0] if len(vectors.shape) > 1 else 1
        else:
            n_vectors = len(vectors) if isinstance(vectors, list) else 1
            
        if n_vectors >= self.nlist:
            self.is_trained = True
        else:
            self.is_trained = False


class MockIndexHNSW(MockIndex):
    """Mock implementation of IndexHNSW"""
    
    def __init__(self, dimension: int, M: int = 16, metric=None):
        metric_type = "L2" if metric is None else str(metric)
        super().__init__(dimension, metric_type)
        self.M = M  # Number of connections
        self.efConstruction = 200  # Construction parameter
        self.efSearch = 16  # Search parameter
    
    def set_ef(self, ef: int):
        """Set search parameter"""
        self.efSearch = ef


class MockIndexLSH(MockIndex):
    """Mock implementation of IndexLSH"""
    
    def __init__(self, dimension: int, nbits: int, metric=None):
        metric_type = "L2" if metric is None else str(metric)
        super().__init__(dimension, metric_type)
        self.nbits = nbits


class MockFAISS:
    """Comprehensive FAISS mock with complete API coverage"""
    
    def __init__(self):
        # Index types
        self.IndexFlat = MockIndexFlat
        self.IndexFlatIP = MockIndexFlatIP  
        self.IndexFlatL2 = MockIndexFlatL2
        self.IndexIVFFlat = MockIndexIVFFlat
        self.IndexHNSWFlat = MockIndexHNSW
        self.IndexLSH = MockIndexLSH
        
        # Metrics
        self.METRIC_L2 = "L2"
        self.METRIC_INNER_PRODUCT = "IP"
        self.METRIC_L1 = "L1"
        self.METRIC_Linf = "Linf"
        
        # Global random state
        self._global_rng = MockRandomState(42)
    
    # Index factory functions
    def index_factory(self, dimension: int, description: str, metric=None):
        """Create index from description string"""
        desc_lower = description.lower()
        metric_type = self.METRIC_L2 if metric is None else metric
        
        if "flat" in desc_lower:
            if "ip" in desc_lower or metric == self.METRIC_INNER_PRODUCT:
                return MockIndexFlatIP(dimension)
            else:
                return MockIndexFlatL2(dimension)
        elif "ivf" in desc_lower:
            # Extract nlist from description (simplified)
            nlist = 100  # Default
            if "ivf" in desc_lower:
                parts = desc_lower.split("ivf")
                if len(parts) > 1:
                    try:
                        nlist = int(''.join(filter(str.isdigit, parts[1][:10])))
                    except:
                        nlist = 100
            
            quantizer = MockIndexFlat(dimension, metric_type)
            return MockIndexIVFFlat(quantizer, dimension, nlist, metric_type)
        elif "hnsw" in desc_lower:
            return MockIndexHNSW(dimension, metric=metric_type)
        elif "lsh" in desc_lower:
            return MockIndexLSH(dimension, 64, metric_type)  # Default 64 bits
        else:
            # Default to flat index
            return MockIndexFlat(dimension, metric_type)
    
    # I/O operations
    def write_index(self, index, filename: str):
        """Write index to file (no-op in mock)"""
        # In a real implementation, this would serialize the index
        pass
    
    def read_index(self, filename: str):
        """Read index from file (returns empty index)"""
        # In a real implementation, this would deserialize the index
        # Return a basic flat index as placeholder
        return MockIndexFlat(128)  # Default dimension
    
    def write_VectorTransform(self, transform, filename: str):
        """Write vector transform to file (no-op)"""
        pass
    
    def read_VectorTransform(self, filename: str):
        """Read vector transform from file (returns None)"""
        return None
    
    # Clustering
    def Kmeans(self, dimension: int, k: int, **kwargs):
        """K-means clustering"""
        class MockKmeans:
            def __init__(self, d: int, k: int):
                self.d = d
                self.k = k
                self.centroids = None
                self.obj = []  # Objective values
                
            def train(self, vectors):
                """Train k-means"""
                # Generate mock centroids
                rng = MockRandomState(42)
                self.centroids = []
                for _ in range(self.k):
                    centroid = [rng.random() for _ in range(self.d)]
                    self.centroids.append(centroid)
                
                # Mock objective values
                self.obj = [1.0 / (i + 1) for i in range(5)]  # Decreasing objectives
        
        return MockKmeans(dimension, k)
    
    # Utility functions
    def normalize_L2(self, vectors):
        """L2 normalize vectors (no-op in mock)"""
        return vectors
    
    def pca_matrix(self, vectors, dimension_out: int):
        """Compute PCA matrix"""
        if hasattr(vectors, 'shape'):
            dimension_in = vectors.shape[1] if len(vectors.shape) > 1 else vectors.shape[0]
        else:
            dimension_in = len(vectors[0]) if isinstance(vectors, list) and vectors else 128
        
        # Return identity-like matrix (simplified)
        rng = MockRandomState(42)
        pca_matrix = []
        for i in range(dimension_out):
            row = []
            for j in range(dimension_in):
                if i == j:
                    row.append(1.0)
                else:
                    row.append(rng.random() * 0.1)  # Small random values
            pca_matrix.append(row)
        
        return pca_matrix
    
    def vector_to_array(self, vectors):
        """Convert vector to array format"""
        return vectors
    
    def array_to_vector(self, array):
        """Convert array to vector format"""
        return array
    
    # Random number generation
    def seed_global_rng(self, seed: int):
        """Set global random seed"""
        self._global_rng = MockRandomState(seed)
    
    # Distance computations
    def compute_distance_subset(self, vectors1, vectors2, ids1=None, ids2=None):
        """Compute distances between vector subsets"""
        # Simplified distance computation
        if ids1 is None:
            n1 = len(vectors1) if isinstance(vectors1, list) else 1
        else:
            n1 = len(ids1)
        
        if ids2 is None:
            n2 = len(vectors2) if isinstance(vectors2, list) else 1
        else:
            n2 = len(ids2)
        
        # Generate mock distance matrix
        distances = []
        for i in range(n1):
            row = []
            for j in range(n2):
                # Deterministic distance based on indices
                hash_val = DeterministicHasher.hash_array(f"{i}:{j}")
                distance = (hash_val % 1000) / 1000.0
                row.append(distance)
            distances.append(row)
        
        return distances
    
    # Index operations
    def clone_index(self, index):
        """Clone an index"""
        new_index = MockIndex(index.d, index.metric_type)
        new_index.ntotal = index.ntotal
        new_index.is_trained = index.is_trained
        return new_index
    
    def downcast_index(self, index):
        """Downcast index to specific type"""
        return index
    
    def extract_index_ivf(self, index):
        """Extract IVF index"""
        if isinstance(index, MockIndexIVFFlat):
            return index
        return None
    
    # Search parameters
    def ParameterSpace(self):
        """Create parameter space for tuning"""
        class MockParameterSpace:
            def __init__(self):
                self.parameters = {}
            
            def set_index_parameter(self, index, parameter: str, value):
                """Set index parameter"""
                self.parameters[parameter] = value
            
            def explore(self, index, vectors, queries):
                """Explore parameter space"""
                return {"best_params": {}, "best_score": 1.0}
        
        return MockParameterSpace()
    
    # GPU operations (all no-ops in mock)
    def StandardGpuResources(self):
        """Create GPU resources"""
        class MockGpuResources:
            def __init__(self):
                self.device = 0
            
            def setDefaultNullStreamAllDevices(self):
                pass
        
        return MockGpuResources()
    
    def index_cpu_to_gpu(self, resources, device: int, index):
        """Move index to GPU"""
        return index  # Return same index in mock
    
    def index_gpu_to_cpu(self, gpu_index):
        """Move index to CPU"""
        return gpu_index  # Return same index in mock
    
    # Vector transforms
    def PCAMatrix(self, dimension_in: int, dimension_out: int):
        """Create PCA transform matrix"""
        class MockPCAMatrix:
            def __init__(self, d_in: int, d_out: int):
                self.d_in = d_in
                self.d_out = d_out
                self.is_trained = False
            
            def train(self, vectors):
                """Train PCA"""
                self.is_trained = True
            
            def apply_py(self, vectors):
                """Apply PCA transform"""
                # Simplified transform - just truncate or pad
                if hasattr(vectors, 'shape'):
                    n_vectors = vectors.shape[0] if len(vectors.shape) > 1 else 1
                else:
                    n_vectors = len(vectors)
                
                result = []
                for i in range(n_vectors):
                    if isinstance(vectors, list):
                        vec = vectors[i] if i < len(vectors) else []
                    else:
                        vec = [0] * self.d_in
                    
                    # Truncate or pad to output dimension
                    transformed = vec[:self.d_out]
                    while len(transformed) < self.d_out:
                        transformed.append(0.0)
                    
                    result.append(transformed)
                
                return result
        
        return MockPCAMatrix(dimension_in, dimension_out)
    
    def OPQMatrix(self, dimension: int, M: int):
        """Create OPQ (Optimized Product Quantization) matrix"""
        class MockOPQMatrix:
            def __init__(self, d: int, M: int):
                self.d = d
                self.M = M
                self.is_trained = False
            
            def train(self, vectors):
                self.is_trained = True
            
            def apply_py(self, vectors):
                return vectors  # No transformation in mock
        
        return MockOPQMatrix(dimension, M)
    
    # Product quantization
    def ProductQuantizer(self, dimension: int, M: int, nbits: int = 8):
        """Create product quantizer"""
        class MockProductQuantizer:
            def __init__(self, d: int, M: int, nbits: int):
                self.d = d
                self.M = M
                self.nbits = nbits
                self.is_trained = False
            
            def train(self, vectors):
                self.is_trained = True
            
            def compute_codes(self, vectors):
                # Return mock codes
                if hasattr(vectors, 'shape'):
                    n_vectors = vectors.shape[0] if len(vectors.shape) > 1 else 1
                else:
                    n_vectors = len(vectors)
                
                codes = []
                for i in range(n_vectors):
                    code = [i % 256 for _ in range(self.M)]  # Mock codes
                    codes.append(code)
                
                return codes
        
        return MockProductQuantizer(dimension, M, nbits)


# Global module functions
def seed_global_rng(seed: int):
    """Set global random seed"""
    global _MOCK_FAISS_INSTANCE
    if '_MOCK_FAISS_INSTANCE' not in globals():
        _MOCK_FAISS_INSTANCE = MockFAISS()
    _MOCK_FAISS_INSTANCE.seed_global_rng(seed)


def write_index(index, filename: str):
    """Write index to file"""
    global _MOCK_FAISS_INSTANCE
    if '_MOCK_FAISS_INSTANCE' not in globals():
        _MOCK_FAISS_INSTANCE = MockFAISS()
    _MOCK_FAISS_INSTANCE.write_index(index, filename)


def read_index(filename: str):
    """Read index from file"""
    global _MOCK_FAISS_INSTANCE
    if '_MOCK_FAISS_INSTANCE' not in globals():
        _MOCK_FAISS_INSTANCE = MockFAISS()
    return _MOCK_FAISS_INSTANCE.read_index(filename)


# Initialize global instance for module-level functions
_MOCK_FAISS_INSTANCE = None