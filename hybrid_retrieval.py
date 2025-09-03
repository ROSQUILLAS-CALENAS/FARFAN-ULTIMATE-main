"""
Hybrid Retrieval Module

Implements hybrid dense/sparse retrieval with FAISS, SPLADE grounding,
ColBERTv2 late interaction, and multilingual E5 embeddings.
"""

import logging
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found

import faiss
import numpy as np
import torch
# # # from sentence_transformers import SentenceTransformer  # Module not found  # Module not found  # Module not found
# Optional sklearn with fallback
try:
# # #     from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore  # Module not found  # Module not found  # Module not found
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

    class TfidfVectorizer:  # minimal fallback
        def __init__(self, max_features=None, lowercase=True, stop_words=None, ngram_range=(1, 1)):
            self.max_features = max_features
            self.lowercase = lowercase
            self.stop_words = set(stop_words) if stop_words else None
            self.ngram_range = ngram_range
            self.vocabulary_ = {}

        def _tokenize(self, text: str):
            if self.lowercase:
                text = text.lower()
            tokens = text.split()
            if self.stop_words:
                tokens = [t for t in tokens if t not in self.stop_words]
            return tokens

        def fit(self, corpus):
# # #             from collections import Counter  # Module not found  # Module not found  # Module not found
            df = Counter()
            for doc in corpus:
                for t in set(self._tokenize(doc)):
                    df[t] += 1
            items = list(df.items())
            if self.max_features:
                items = sorted(items, key=lambda x: (-x[1], x[0]))[: self.max_features]
            self.vocabulary_ = {t: i for i, (t, _) in enumerate(sorted(items))}
            return self

        def transform(self, corpus):
            import numpy as np
            X = np.zeros((len(corpus), len(self.vocabulary_)), dtype=np.float32)
            for i, doc in enumerate(corpus):
                for t in self._tokenize(doc):
                    j = self.vocabulary_.get(t)
                    if j is not None:
                        X[i, j] += 1.0
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            return X / norms

        def fit_transform(self, corpus):
            return self.fit(corpus).transform(corpus)


@dataclass
class RetrievalResult:
    """Container for retrieval results"""

    doc_ids: List[str]
    scores: List[float]
    method: str
    query: str


@dataclass
class HybridScore:
    """Container for hybrid scoring results"""

    doc_id: str
    sparse_score: float
    dense_score: float
    late_interaction_score: float
    final_score: float


class HybridRetriever:
    """
    Hybrid retrieval system combining sparse (SPLADE-style), dense (E5),
    and late interaction (ColBERTv2-style) methods with FAISS indexing.
    """

    def __init__(
        self,
        embedding_model_name: str = "intfloat/e5-base-v2",
        device: str = "cpu",
        index_type: str = "IVF",
        sparse_alpha: float = 0.3,
        dense_alpha: float = 0.4,
        late_interaction_alpha: float = 0.3,
        max_docs: int = 1000,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedding_model_name: Dense embedding model
            device: Computation device
            index_type: FAISS index type (IVF, Flat, HNSW)
            sparse_alpha: Weight for sparse component
            dense_alpha: Weight for dense component
            late_interaction_alpha: Weight for late interaction component
            max_docs: Maximum documents in collection
        """
        self.device = device
        self.embedding_model_name = embedding_model_name
        self.index_type = index_type

        # Hybrid weighting
        self.sparse_alpha = sparse_alpha
        self.dense_alpha = dense_alpha
        self.late_interaction_alpha = late_interaction_alpha

        # Normalize weights
        total_alpha = sparse_alpha + dense_alpha + late_interaction_alpha
        self.sparse_alpha /= total_alpha
        self.dense_alpha /= total_alpha
        self.late_interaction_alpha /= total_alpha

        self.max_docs = max_docs

        # Initialize components
        self.logger = logging.getLogger(__name__)
        self._initialize_models()
        self._initialize_indices()

        # Document storage
        self.documents = []
        self.doc_ids = []
        self.doc_embeddings = None

    def _initialize_models(self):
        """Initialize embedding models."""
        self.logger.info("Loading embedding models...")

        # E5 model for dense embeddings
        self.dense_model = SentenceTransformer(
            self.embedding_model_name, device=self.device
        )
        self.embedding_dim = self.dense_model.get_sentence_embedding_dimension()

        # SPLADE-style sparse vectorizer (using TF-IDF as proxy)
        self.sparse_vectorizer = TfidfVectorizer(
            max_features=10000, lowercase=True, stop_words="english", ngram_range=(1, 2)
        )

        # For ColBERTv2-style late interaction, we'll use the same E5 model
        self.late_interaction_model = self.dense_model

        self.logger.info("Models initialized successfully")

    def _initialize_indices(self):
        """Initialize FAISS indices."""
        self.logger.info(f"Initializing FAISS indices with {self.index_type}")

        # Dense FAISS index
        if self.index_type == "Flat":
            self.dense_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.dense_index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, 100
            )  # 100 clusters
        elif self.index_type == "HNSW":
            self.dense_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.dense_index.hnsw.efConstruction = 40
        else:
            self.dense_index = faiss.IndexFlatIP(self.embedding_dim)

        # Sparse index (using approximate with FAISS for consistency)
        self.sparse_dim = 10000  # Will be updated after sparse vectorizer fit
        self.sparse_index = faiss.IndexFlatIP(self.sparse_dim)

        # Late interaction index (using same as dense for now)
        self.late_interaction_index = faiss.IndexFlatIP(self.embedding_dim)

        self.logger.info("FAISS indices initialized")

    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """
        Add documents to the hybrid retrieval indices.

        Args:
            documents: List of document texts
            doc_ids: Optional document IDs (will generate if not provided)
        """
        if doc_ids is None:
            doc_ids = [
                f"doc_{i}"
                for i in range(
                    len(self.documents), len(self.documents) + len(documents)
                )
            ]

        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)

        self.logger.info(f"Adding {len(documents)} documents to indices...")

        # Generate dense embeddings
        dense_embeddings = self._generate_dense_embeddings(documents)

        # Generate sparse embeddings
        sparse_embeddings = self._generate_sparse_embeddings(documents)

        # Generate late interaction embeddings (same as dense for simplicity)
        late_interaction_embeddings = dense_embeddings

        # Add to indices
        self._add_to_dense_index(dense_embeddings)
        self._add_to_sparse_index(sparse_embeddings)
        self._add_to_late_interaction_index(late_interaction_embeddings)

        # Store embeddings
        if self.doc_embeddings is None:
            self.doc_embeddings = dense_embeddings
        else:
            self.doc_embeddings = np.vstack([self.doc_embeddings, dense_embeddings])

        self.logger.info(f"Successfully added documents. Total: {len(self.documents)}")

    def _generate_dense_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate dense embeddings using E5."""
        # Add E5 instruction prefix
        prefixed_docs = [f"passage: {doc}" for doc in documents]
        embeddings = self.dense_model.encode(
            prefixed_docs, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings

    def _generate_sparse_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate sparse embeddings using TF-IDF (SPLADE proxy)."""
        if not hasattr(self.sparse_vectorizer, "vocabulary_"):
            # First time - fit the vectorizer
            all_docs = self.documents + documents if self.documents else documents
            self.sparse_vectorizer.fit(all_docs)
            self.sparse_dim = len(self.sparse_vectorizer.vocabulary_)

            # Reinitialize sparse index with correct dimensions
            self.sparse_index = faiss.IndexFlatIP(self.sparse_dim)

            # Re-index existing documents if any
            if self.documents:
                existing_sparse = (
                    self.sparse_vectorizer.transform(self.documents)
                    .toarray()
                    .astype(np.float32)
                )
                self.sparse_index.add(existing_sparse)

        sparse_matrix = self.sparse_vectorizer.transform(documents)
        return sparse_matrix.toarray().astype(np.float32)

    def _add_to_dense_index(self, embeddings: np.ndarray):
        """Add embeddings to dense FAISS index."""
        if self.index_type == "IVF" and not self.dense_index.is_trained:
            if len(self.documents) >= 100:  # Need enough data to train
                self.dense_index.train(embeddings)

        self.dense_index.add(embeddings)

    def _add_to_sparse_index(self, embeddings: np.ndarray):
        """Add embeddings to sparse FAISS index."""
        self.sparse_index.add(embeddings)

    def _add_to_late_interaction_index(self, embeddings: np.ndarray):
        """Add embeddings to late interaction index."""
        self.late_interaction_index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
        return_scores: bool = True,
    ) -> Union[RetrievalResult, List[HybridScore]]:
        """
        Search documents using specified method.

        Args:
            query: Search query
            top_k: Number of results to return
            method: Retrieval method (sparse, dense, late_interaction, hybrid)
            return_scores: Whether to return detailed scores

        Returns:
            Retrieval results or detailed hybrid scores
        """
        if len(self.documents) == 0:
            self.logger.warning("No documents in index")
            return RetrievalResult([], [], method, query)

        if method == "sparse":
            return self._sparse_search(query, top_k)
        elif method == "dense":
            return self._dense_search(query, top_k)
        elif method == "late_interaction":
            return self._late_interaction_search(query, top_k)
        elif method == "hybrid":
            return self._hybrid_search(query, top_k, return_scores)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _sparse_search(self, query: str, top_k: int) -> RetrievalResult:
        """Perform sparse retrieval."""
        query_vector = (
            self.sparse_vectorizer.transform([query]).toarray().astype(np.float32)
        )

        scores, indices = self.sparse_index.search(
            query_vector, min(top_k, len(self.documents))
        )

        doc_ids = [self.doc_ids[idx] for idx in indices[0] if idx != -1]
        scores = scores[0][scores[0] != -1].tolist()

        return RetrievalResult(doc_ids, scores, "sparse", query)

    def _dense_search(self, query: str, top_k: int) -> RetrievalResult:
        """Perform dense retrieval."""
        query_embedding = self.dense_model.encode(
            [f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True
        )

        scores, indices = self.dense_index.search(
            query_embedding, min(top_k, len(self.documents))
        )

        doc_ids = [self.doc_ids[idx] for idx in indices[0] if idx != -1]
        scores = scores[0][scores[0] != -1].tolist()

        return RetrievalResult(doc_ids, scores, "dense", query)

    def _late_interaction_search(self, query: str, top_k: int) -> RetrievalResult:
        """Perform late interaction retrieval (ColBERTv2-style)."""
        # For simplicity, using same as dense search
        # In practice, this would involve token-level interactions
        query_tokens = query.split()

        # Simulate late interaction with token-level embeddings
        if len(query_tokens) > 1:
            # Create interaction query
            interaction_query = f"[CLS] {query}"
        else:
            interaction_query = f"query: {query}"

        query_embedding = self.late_interaction_model.encode(
            [interaction_query], convert_to_numpy=True, normalize_embeddings=True
        )

        scores, indices = self.late_interaction_index.search(
            query_embedding, min(top_k, len(self.documents))
        )

        doc_ids = [self.doc_ids[idx] for idx in indices[0] if idx != -1]
        scores = scores[0][scores[0] != -1].tolist()

        return RetrievalResult(doc_ids, scores, "late_interaction", query)

    def _hybrid_search(
        self, query: str, top_k: int, return_scores: bool = True
    ) -> Union[RetrievalResult, List[HybridScore]]:
        """Perform hybrid retrieval combining all methods."""
# # #         # Get results from all methods  # Module not found  # Module not found  # Module not found
        sparse_result = self._sparse_search(query, top_k * 2)  # Get more for fusion
        dense_result = self._dense_search(query, top_k * 2)
        late_interaction_result = self._late_interaction_search(query, top_k * 2)

        # Combine scores
        doc_scores = self._fuse_scores(
            sparse_result, dense_result, late_interaction_result
        )

        # Sort by final score
        sorted_results = sorted(
            doc_scores.items(), key=lambda x: x[1].final_score, reverse=True
        )

        if return_scores:
            return [score_obj for _, score_obj in sorted_results[:top_k]]
        else:
            doc_ids = [doc_id for doc_id, _ in sorted_results[:top_k]]
            final_scores = [
                score_obj.final_score for _, score_obj in sorted_results[:top_k]
            ]
            return RetrievalResult(doc_ids, final_scores, "hybrid", query)

    def _fuse_scores(
        self,
        sparse_result: RetrievalResult,
        dense_result: RetrievalResult,
        late_interaction_result: RetrievalResult,
    ) -> Dict[str, HybridScore]:
# # #         """Fuse scores from different retrieval methods."""  # Module not found  # Module not found  # Module not found
        doc_scores = {}

        # Normalize scores to [0, 1] range
        def normalize_scores(scores):
            if not scores:
                return scores
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                return [1.0] * len(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]

        # Process sparse results
        sparse_scores_norm = normalize_scores(sparse_result.scores)
        for doc_id, score in zip(sparse_result.doc_ids, sparse_scores_norm):
            doc_scores[doc_id] = HybridScore(doc_id, score, 0.0, 0.0, 0.0)

        # Process dense results
        dense_scores_norm = normalize_scores(dense_result.scores)
        for doc_id, score in zip(dense_result.doc_ids, dense_scores_norm):
            if doc_id in doc_scores:
                doc_scores[doc_id].dense_score = score
            else:
                doc_scores[doc_id] = HybridScore(doc_id, 0.0, score, 0.0, 0.0)

        # Process late interaction results
        late_scores_norm = normalize_scores(late_interaction_result.scores)
        for doc_id, score in zip(late_interaction_result.doc_ids, late_scores_norm):
            if doc_id in doc_scores:
                doc_scores[doc_id].late_interaction_score = score
            else:
                doc_scores[doc_id] = HybridScore(doc_id, 0.0, 0.0, score, 0.0)

        # Compute final hybrid scores
        for doc_id, score_obj in doc_scores.items():
            score_obj.final_score = (
                self.sparse_alpha * score_obj.sparse_score
                + self.dense_alpha * score_obj.dense_score
                + self.late_interaction_alpha * score_obj.late_interaction_score
            )

        return doc_scores

    def batch_search(
        self, queries: List[str], top_k: int = 10, method: str = "hybrid"
    ) -> List[RetrievalResult]:
        """
        Perform batch search for multiple queries.

        Args:
            queries: List of queries
            top_k: Number of results per query
            method: Retrieval method

        Returns:
            List of retrieval results
        """
        results = []
        for query in queries:
            result = self.search(query, top_k, method, return_scores=False)
            results.append(result)
        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the indices."""
        stats = {
            "num_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "sparse_dim": self.sparse_dim,
            "dense_index_size": self.dense_index.ntotal,
            "sparse_index_size": self.sparse_index.ntotal,
            "late_interaction_index_size": self.late_interaction_index.ntotal,
            "index_type": self.index_type,
            "hybrid_weights": {
                "sparse": self.sparse_alpha,
                "dense": self.dense_alpha,
                "late_interaction": self.late_interaction_alpha,
            },
        }
        return stats

    def save_index(self, save_path: str):
        """Save FAISS indices to disk."""
        faiss.write_index(self.dense_index, f"{save_path}_dense.faiss")
        faiss.write_index(self.sparse_index, f"{save_path}_sparse.faiss")
        faiss.write_index(
            self.late_interaction_index, f"{save_path}_late_interaction.faiss"
        )

        # Save metadata
        import pickle

        metadata = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "doc_embeddings": self.doc_embeddings,
            "sparse_vectorizer": self.sparse_vectorizer,
            "embedding_dim": self.embedding_dim,
            "sparse_dim": self.sparse_dim,
        }

        with open(f"{save_path}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        self.logger.info(f"Saved indices and metadata to {save_path}")

    def load_index(self, save_path: str):
# # #         """Load FAISS indices from disk."""  # Module not found  # Module not found  # Module not found
        self.dense_index = faiss.read_index(f"{save_path}_dense.faiss")
        self.sparse_index = faiss.read_index(f"{save_path}_sparse.faiss")
        self.late_interaction_index = faiss.read_index(
            f"{save_path}_late_interaction.faiss"
        )

        # Load metadata
        import pickle

        with open(f"{save_path}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.documents = metadata["documents"]
        self.doc_ids = metadata["doc_ids"]
        self.doc_embeddings = metadata["doc_embeddings"]
        self.sparse_vectorizer = metadata["sparse_vectorizer"]
        self.embedding_dim = metadata["embedding_dim"]
        self.sparse_dim = metadata["sparse_dim"]

# # #         self.logger.info(f"Loaded indices and metadata from {save_path}")  # Module not found  # Module not found  # Module not found
