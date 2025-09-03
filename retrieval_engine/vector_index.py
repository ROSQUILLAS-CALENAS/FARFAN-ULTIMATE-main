"""
FAISS-based Vector Retrieval Module

Implements semantic search using dense vector embeddings and FAISS indexing.
Provides standardized retrieval interface for vector similarity search.
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import json
import os
import pickle
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "R"
__code__ = "38R"
__stage_order__ = 6

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    # Mock FAISS for basic functionality
    class MockFAISSIndex:
        def __init__(self, dim):
            self.dim = dim
            self.vectors = []
        def add(self, vectors):
            if hasattr(vectors, '__iter__'):
                self.vectors.extend(vectors)
        def search(self, query, k):
            scores = [0.9, 0.8, 0.7][:k]
            indices = list(range(min(k, len(self.vectors))))
            return [scores], [indices]
    
    class MockFAISS:
        IndexFlatIP = MockFAISSIndex
        IndexFlatL2 = MockFAISSIndex
        IndexHNSWFlat = MockFAISSIndex
        IndexIVFFlat = MockFAISSIndex
        @staticmethod
        def normalize_L2(x):
            return x
        @staticmethod
        def write_index(idx, path):
            pass
        @staticmethod
        def read_index(path):
            return MockFAISSIndex(384)
    
    faiss = MockFAISS()

try:
# # #     from sentence_transformers import SentenceTransformer  # Module not found  # Module not found  # Module not found
except ImportError:
    SentenceTransformer = None


class VectorIndex:
    """FAISS-based vector similarity index."""
    
    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = "Flat",
        metric: str = "cosine",
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None
    ):
        """
        Initialize vector index.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: FAISS index type ("Flat", "HNSW", "IVF")
            metric: Distance metric ("cosine", "euclidean")
            model_name: Sentence transformer model for embeddings
            index_path: Path to load existing index from
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.model_name = model_name
        
        # Initialize embedding model
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.model = None
        else:
            self.model = None
        
        # Index components
        self.index: faiss.Index = self._create_faiss_index()
        self.document_ids: List[str] = []
        self.document_texts: List[str] = []
        self.document_embeddings: Optional[np.ndarray] = None
        
        if index_path and os.path.exists(f"{index_path}.faiss"):
            self.load_index(index_path)
    
    def _create_faiss_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.metric == "cosine":
            # Use inner product for cosine similarity (vectors will be normalized)
            if self.index_type == "Flat":
                index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == "HNSW":
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # M=32 connections
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 100
            elif self.index_type == "IVF":
                nlist = min(100, max(1, int(np.sqrt(1000))))  # Adaptive nlist
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            else:
                index = faiss.IndexFlatIP(self.embedding_dim)
        else:  # euclidean
            if self.index_type == "Flat":
                index = faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == "HNSW":
                index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            elif self.index_type == "IVF":
                nlist = min(100, max(1, int(np.sqrt(1000))))
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            else:
                index = faiss.IndexFlatL2(self.embedding_dim)
        
        return index
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings."""
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        else:
            # Fallback: simple hash-based embeddings for testing
            embeddings = []
            for text in texts:
                # Simple deterministic embedding based on text hash
                hash_val = hash(text)
                # Simple deterministic embedding without external dependencies
                embedding = [(hash_val + i) % 1000 / 1000.0 for i in range(self.embedding_dim)]
                norm = sum(x*x for x in embedding) ** 0.5
                embedding = [x / (norm + 1e-8) for x in embedding]
                embeddings.append(embedding)
            
            # Create mock array structure
            class MockArray:
                def __init__(self, data):
                    self.data = data
                    self.shape = (len(data), len(data[0]) if data else 0)
                def astype(self, dtype):
                    return self
                def __iter__(self):
                    return iter(self.data)
            
            if HAS_NUMPY:
                embeddings = np.array(embeddings, dtype=np.float32)
            else:
                embeddings = MockArray(embeddings)
        
        # Normalize for cosine similarity if using inner product
        if self.metric == "cosine" and HAS_FAISS:
            faiss.normalize_L2(embeddings)
        
        if HAS_NUMPY:
            return embeddings.astype(np.float32)
        else:
            return embeddings
    
    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        embedding = self._embed_texts([text])
        
        self.document_ids.append(doc_id)
        self.document_texts.append(text)
        
        if self.document_embeddings is None:
            self.document_embeddings = embedding
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embedding])
        
        self.index.add(embedding)
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained and len(self.document_ids) >= 100:
            self.index.train(self.document_embeddings)
    
    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """Add multiple documents to the index."""
        if not documents:
            return
        
        doc_ids, texts = zip(*documents)
        embeddings = self._embed_texts(list(texts))
        
        self.document_ids.extend(doc_ids)
        self.document_texts.extend(texts)
        
        if self.document_embeddings is None:
            self.document_embeddings = embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embeddings])
        
        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(self.document_ids) >= max(100, self.index.nlist if hasattr(self.index, 'nlist') else 100):
                self.index.train(self.document_embeddings)
        
        self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of ranked documents with scores
        """
        if len(self.document_ids) == 0:
            return []
        
        # Convert query to embedding
        query_embedding = self._embed_texts([query])
        
        # Search in FAISS index
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            # For untrained IVF indices, use brute force search
            if self.document_embeddings is not None:
                if self.metric == "cosine":
                    # Compute cosine similarities
                    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                    doc_norms = self.document_embeddings / (np.linalg.norm(self.document_embeddings, axis=1, keepdims=True) + 1e-8)
                    scores = np.dot(doc_norms, query_norm.T).flatten()
                else:
                    # Compute euclidean distances (convert to similarities)
                    distances = np.linalg.norm(self.document_embeddings - query_embedding, axis=1)
                    scores = 1.0 / (1.0 + distances)  # Convert distance to similarity
                
                # Get top-k
                top_indices = np.argsort(scores)[::-1][:top_k]
                similarities = scores[top_indices]
            else:
                return []
        else:
            # Use FAISS search
            similarities, top_indices = self.index.search(query_embedding, top_k)
            similarities = similarities[0]  # Get first (and only) query result
            top_indices = top_indices[0]
        
        # Build results
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(self.document_ids) and (not hasattr(similarities, '__len__') or similarities[i] > 0):
                results.append({
                    'doc_id': self.document_ids[idx],
                    'score': float(similarities[i]) if hasattr(similarities, '__len__') else float(similarities),
                    'text': self.document_texts[idx],
                    'method': 'vector_similarity'
                })
        
        return results
    
    def save_index(self, path: str) -> None:
        """Save the FAISS index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'model_name': self.model_name,
            'document_ids': self.document_ids,
            'document_texts': self.document_texts,
        }
        
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save embeddings if they exist
        if self.document_embeddings is not None:
            np.save(f"{path}_embeddings.npy", self.document_embeddings)
    
    def load_index(self, path: str) -> None:
# # #         """Load the FAISS index and metadata from disk."""  # Module not found  # Module not found  # Module not found
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load metadata
        with open(f"{path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.embedding_dim = metadata['embedding_dim']
        self.index_type = metadata['index_type']
        self.metric = metadata['metric']
        self.model_name = metadata['model_name']
        self.document_ids = metadata['document_ids']
        self.document_texts = metadata['document_texts']
        
        # Load embeddings if they exist
        embeddings_path = f"{path}_embeddings.npy"
        if os.path.exists(embeddings_path):
            self.document_embeddings = np.load(embeddings_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'num_documents': len(self.document_ids),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'model_name': self.model_name,
            'is_trained': getattr(self.index, 'is_trained', True)
        }


class VectorRetriever:
    """Standard interface for FAISS-based vector retrieval."""
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_dim: int = 384,
        index_type: str = "Flat",
        metric: str = "cosine",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector retriever.
        
        Args:
            index_path: Path to FAISS index file (embeddings.faiss)
            embedding_dim: Dimension of embedding vectors
            index_type: FAISS index type
            metric: Distance metric
            model_name: Sentence transformer model
        """
        self.index_path = index_path or "embeddings"
        self.index = VectorIndex(
            embedding_dim=embedding_dim,
            index_type=index_type,
            metric=metric,
            model_name=model_name,
            index_path=self.index_path if os.path.exists(f"{self.index_path}.faiss") else None
        )
    
    def build_index(self, documents: List[Tuple[str, str]]) -> None:
        """
# # #         Build FAISS index from documents.  # Module not found  # Module not found  # Module not found
        
        Args:
            documents: List of (doc_id, text) tuples
        """
        self.index.add_documents(documents)
        self.index.save_index(self.index_path)
    
    def add_document(self, doc_id: str, text: str) -> None:
        """Add a single document to the index."""
        self.index.add_document(doc_id, text)
        self.index.save_index(self.index_path)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query using vector similarity.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of ranked documents with standardized format:
            [{'doc_id': str, 'score': float, 'text': str, 'method': str}]
        """
        return self.index.search(query, top_k)
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get index statistics and metrics."""
        return self.index.get_statistics()


def create_vector_retriever(
    documents: Optional[List[Tuple[str, str]]] = None,
    index_path: str = "embeddings",
    embedding_dim: int = 384,
    index_type: str = "Flat",
    metric: str = "cosine",
    model_name: str = "all-MiniLM-L6-v2"
) -> VectorRetriever:
    """
    Factory function to create and optionally build a vector retriever.
    
    Args:
        documents: Optional documents to build index from
        index_path: Path for FAISS index file
        embedding_dim: Embedding dimension
        index_type: FAISS index type
        metric: Distance metric
        model_name: Sentence transformer model
        
    Returns:
        Configured VectorRetriever instance
    """
    retriever = VectorRetriever(
        index_path=index_path,
        embedding_dim=embedding_dim,
        index_type=index_type,
        metric=metric,
        model_name=model_name
    )
    
    if documents:
        retriever.build_index(documents)
    
    return retriever


# Compatibility function for comprehensive_pipeline_orchestrator.py
def process(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process function for pipeline integration.
    
    Maintains compatibility with existing orchestrator while providing
    enhanced FAISS vector functionality.
    """
    ctx = context or {}
    
# # #     # Extract query and documents from input data  # Module not found  # Module not found  # Module not found
    query = ""
    documents = []
    
    if isinstance(data, dict):
        query = data.get("query", "") or ctx.get("query", "")
        
# # #         # Try to extract documents from various possible keys  # Module not found  # Module not found  # Module not found
        docs = data.get("documents") or data.get("loaded_docs") or data.get("text", "")
        if isinstance(docs, str) and docs:
            documents = [("doc_0", docs)]
        elif isinstance(docs, list):
            documents = [(f"doc_{i}", doc) if isinstance(doc, str) else (str(doc.get("id", i)), str(doc.get("text", ""))) 
                        for i, doc in enumerate(docs)]
    
    # Create retriever
    retriever = create_vector_retriever(documents if documents else None)
    
    # Perform search if query is provided
    results = []
    if query and len(retriever.index.document_ids) > 0:
        results = retriever.retrieve(query, top_k=ctx.get("top_k", 10))
    
    # Build response
    output = {}
    if isinstance(data, dict):
        output.update(data)
    
    output.update({
        "vector_index": retriever.index,
        "vector_results": results,
        "vector_metrics": retriever.get_index_statistics(),
        "embeddings": retriever.index.document_embeddings.tolist() if retriever.index.document_embeddings is not None else [],
        "method": "vector_similarity"
    })
    
    return output


if __name__ == "__main__":
    # Example usage
    sample_docs = [
        ("doc1", "The quick brown fox jumps over the lazy dog"),
        ("doc2", "Python is a programming language"),
        ("doc3", "Information retrieval using vector similarity"),
        ("doc4", "Machine learning and natural language processing")
    ]
    
    retriever = create_vector_retriever(documents=sample_docs)
    results = retriever.retrieve("programming language", top_k=3)
    
    print("FAISS Vector Retrieval Results:")
    for result in results:
        print(f"Doc ID: {result['doc_id']}, Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:100]}...")
        print("-" * 50)