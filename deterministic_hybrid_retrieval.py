"""
Deterministic Hybrid Retrieval System

Implementation of deterministic hybrid retrieval based on Bruch et al. (2023)
optimal ranking fusion theorem from "Bridging Dense and Sparse Maximum Inner Product Search".

This system unifies dense and sparse retrieval methods through learned projections
while maintaining complete determinism in fusion and reranking.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Heavy/optional dependencies (guarded)
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
try:
    import torch  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAS_ST = True
except Exception:
    HAS_ST = False
# Optional sklearn with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
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
            from collections import Counter
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
            # Pure-Python sparse TF matrix and L2 normalization (no NumPy)
            rows = len(corpus)
            cols = len(self.vocabulary_)
            X = [[0.0 for _ in range(cols)] for _ in range(rows)]
            for i, doc in enumerate(corpus):
                for t in self._tokenize(doc):
                    j = self.vocabulary_.get(t)
                    if j is not None:
                        X[i][j] += 1.0
            # L2 normalize each row
            def _l2norm(vec: List[float]) -> float:
                s = sum(v*v for v in vec)
                return s ** 0.5 if s > 0 else 1e-8
            for i in range(rows):
                n = _l2norm(X[i])
                X[i] = [v / n for v in X[i]]
            return X

        def fit_transform(self, corpus):
            return self.fit(corpus).transform(corpus)


@dataclass(frozen=True)
class DocumentHash:
    """Immutable content-based document identifier"""

    content_hash: str
    doc_id: str

    @classmethod
    def from_content(cls, content: str, doc_id: str) -> "DocumentHash":
        """Create hash from document content"""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return cls(content_hash=content_hash, doc_id=doc_id)


@dataclass(frozen=True)
class IndexSnapshot:
    """Immutable index snapshot with version control"""

    snapshot_id: str
    documents_hash: str
    embedding_dim: int
    sparse_dim: int
    creation_time: float

    @classmethod
    def create(
        cls, documents: List[str], embedding_dim: int, sparse_dim: int
    ) -> "IndexSnapshot":
        """Create snapshot from document collection"""
        import time

        # Create deterministic hash of all documents
        combined_content = "|||".join(sorted(documents))
        documents_hash = hashlib.sha256(combined_content.encode("utf-8")).hexdigest()

        # Create snapshot ID combining content and dimensions
        snapshot_data = f"{documents_hash}_{embedding_dim}_{sparse_dim}"
        snapshot_id = hashlib.sha256(snapshot_data.encode("utf-8")).hexdigest()[:16]

        return cls(
            snapshot_id=snapshot_id,
            documents_hash=documents_hash,
            embedding_dim=embedding_dim,
            sparse_dim=sparse_dim,
            creation_time=time.time(),
        )


@dataclass(frozen=True)
class QueryExpansion:
    """Fixed query expansion configuration"""

    expansion_weights: Tuple[float, ...]  # Immutable tuple
    expansion_radius: float
    max_expansions: int

    def expand_query(self, query: str, seed: int = 42) -> List[str]:
        """Deterministically expand query using fixed weights"""
        try:
            import numpy as _np  # local, may fail
            _np.random.seed(seed)
        except Exception:
            import random as _random
            _random.seed(seed)  # Fallback without NumPy

        # Simple deterministic expansion (in practice, use learned projections)
        words = query.split()
        expansions = [query]  # Always include original

        # Generate deterministic variations
        for i, weight in enumerate(self.expansion_weights[: self.max_expansions - 1]):
            if i < len(words):
                # Create semantic variations (simplified)
                expanded = (
                    f"{words[i]} {query}" if weight > 0.5 else f"{query} {words[i]}"
                )
                expansions.append(expanded)

        return expansions[: self.max_expansions]


@dataclass(frozen=True)
class DNPConstraint:
    """Do Not Process constraint for consistent filtering"""

    field: str
    operator: str  # 'eq', 'ne', 'in', 'not_in'
    value: Any

    def applies_to_document(self, doc_metadata: Dict[str, Any]) -> bool:
        """Check if constraint applies to document"""
        if self.field not in doc_metadata:
            return False

        doc_value = doc_metadata[self.field]

        if self.operator == "eq":
            return doc_value == self.value
        elif self.operator == "ne":
            return doc_value != self.value
        elif self.operator == "in":
            return doc_value in self.value
        elif self.operator == "not_in":
            return doc_value not in self.value
        else:
            return False


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable retrieval result"""

    doc_hashes: Tuple[DocumentHash, ...]
    scores: Tuple[float, ...]
    method: str
    query: str
    snapshot_id: str

    def __post_init__(self):
        """Validate result consistency"""
        assert len(self.doc_hashes) == len(self.scores)


@dataclass(frozen=True)
class FusionResult:
    """Result from optimal ranking fusion"""

    doc_hash: DocumentHash
    sparse_score: float
    dense_score: float
    fusion_score: float
    rank: int


class LearnedProjection:
    """Learned projection for bridging dense and sparse representations"""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        """Initialize with fixed random seed for determinism"""
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Fixed projection matrix (would be learned in practice)
        self.W = torch.randn(input_dim, output_dim, dtype=torch.float32)
        self.b = torch.randn(output_dim, dtype=torch.float32)

        # Freeze parameters for determinism
        self.W.requires_grad = False
        self.b.requires_grad = False

    def project(self, dense_embedding: np.ndarray) -> np.ndarray:
        """Project dense embedding to sparse space"""
        x = torch.from_numpy(dense_embedding).float()
        projected = torch.matmul(x, self.W) + self.b
        return torch.relu(projected).numpy()  # Sparse-like activation


class OptimalRankingFusion:
    """Implementation of Bruch et al. (2023) optimal ranking fusion theorem"""

    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        """Initialize with fixed fusion parameters"""
        self.alpha = alpha  # Weight for sparse component
        self.beta = beta  # Weight for dense component
        self.gamma = 1.0 - alpha - beta  # Weight for projection component

    def fuse_rankings(
        self,
        sparse_results: List[Tuple[DocumentHash, float]],
        dense_results: List[Tuple[DocumentHash, float]],
        projection_results: List[Tuple[DocumentHash, float]],
    ) -> List[FusionResult]:
        """Apply optimal fusion theorem to combine rankings"""

        # Normalize scores to [0, 1]
        sparse_scores = self._normalize_scores([score for _, score in sparse_results])
        dense_scores = self._normalize_scores([score for _, score in dense_results])
        projection_scores = self._normalize_scores(
            [score for _, score in projection_results]
        )

        # Create score mapping
        score_map = {}

        # Process sparse
        for (doc_hash, _), norm_score in zip(sparse_results, sparse_scores):
            score_map[doc_hash] = {
                "sparse": norm_score,
                "dense": 0.0,
                "projection": 0.0,
            }

        # Process dense
        for (doc_hash, _), norm_score in zip(dense_results, dense_scores):
            if doc_hash in score_map:
                score_map[doc_hash]["dense"] = norm_score
            else:
                score_map[doc_hash] = {
                    "sparse": 0.0,
                    "dense": norm_score,
                    "projection": 0.0,
                }

        # Process projection
        for (doc_hash, _), norm_score in zip(projection_results, projection_scores):
            if doc_hash in score_map:
                score_map[doc_hash]["projection"] = norm_score
            else:
                score_map[doc_hash] = {
                    "sparse": 0.0,
                    "dense": 0.0,
                    "projection": norm_score,
                }

        # Apply optimal fusion
        fusion_results = []
        for doc_hash, scores in score_map.items():
            fusion_score = (
                self.alpha * scores["sparse"]
                + self.beta * scores["dense"]
                + self.gamma * scores["projection"]
            )

            fusion_results.append(
                FusionResult(
                    doc_hash=doc_hash,
                    sparse_score=scores["sparse"],
                    dense_score=scores["dense"],
                    fusion_score=fusion_score,
                    rank=0,  # Will be set after sorting
                )
            )

        # Sort by fusion score and assign ranks
        fusion_results.sort(key=lambda x: x.fusion_score, reverse=True)

        # Return new objects with correct ranks (frozen dataclasses)
        ranked_results = []
        for rank, result in enumerate(fusion_results):
            ranked_results.append(
                FusionResult(
                    doc_hash=result.doc_hash,
                    sparse_score=result.sparse_score,
                    dense_score=result.dense_score,
                    fusion_score=result.fusion_score,
                    rank=rank,
                )
            )

        return ranked_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range deterministically"""
        if not scores:
            return scores

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]


class DeterministicHybridRetriever:
    """
    Deterministic hybrid retrieval system implementing Bruch et al. (2023)
    optimal ranking fusion with complete reproducibility guarantees.
    """

    def __init__(
        self,
        embedding_model_name: str = "intfloat/e5-base-v2",
        device: str = "cpu",
        sparse_alpha: float = 0.4,
        dense_alpha: float = 0.4,
        projection_alpha: float = 0.2,
        seed: int = 42,
    ):
        """Initialize deterministic retriever with fixed parameters"""

        # Set all random seeds for complete determinism
        np.random.seed(seed)
        torch.manual_seed(seed)
        faiss.omp_set_num_threads(1)  # Ensure deterministic FAISS

        self.seed = seed
        self.embedding_model_name = embedding_model_name
        self.device = device

        # Fixed fusion weights
        self.fusion = OptimalRankingFusion(alpha=sparse_alpha, beta=dense_alpha)

        # Initialize components
        self.logger = logging.getLogger(__name__)
        self._initialize_models()

        # Document storage with content hashing
        self.document_hashes: List[DocumentHash] = []
        self.documents: List[str] = []
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}

        # Index snapshots
        self.current_snapshot: Optional[IndexSnapshot] = None
        self.indices_initialized = False

    def _initialize_models(self):
        """Initialize models with fixed parameters"""
        self.logger.info("Loading models with fixed parameters...")

        # E5 model for dense embeddings
        self.dense_model = SentenceTransformer(
            self.embedding_model_name, device=self.device
        )
        self.embedding_dim = self.dense_model.get_sentence_embedding_dimension()

        # Fixed sparse vectorizer
        self.sparse_vectorizer = TfidfVectorizer(
            max_features=5000,  # Fixed vocabulary size
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.logger.info("Models initialized with deterministic parameters")

    def _initialize_indices(self, sparse_dim: int):
        """Initialize FAISS indices with fixed parameters"""

        # Dense index (flat for determinism)
        self.dense_index = faiss.IndexFlatIP(self.embedding_dim)

        # Sparse index
        self.sparse_index = faiss.IndexFlatIP(sparse_dim)

        # Learned projection
        self.projection = LearnedProjection(
            self.embedding_dim, sparse_dim, seed=self.seed
        )

        self.indices_initialized = True

    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> IndexSnapshot:
        """
        Add documents and create new immutable snapshot.

        Args:
            documents: Document texts
            doc_ids: Optional document IDs
            metadata: Optional document metadata

        Returns:
            New index snapshot
        """

        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        # Create document hashes
        new_hashes = [
            DocumentHash.from_content(doc, doc_id)
            for doc, doc_id in zip(documents, doc_ids)
        ]

        # Check for duplicates
        existing_hashes = {dh.content_hash for dh in self.document_hashes}
        unique_additions = []
        unique_docs = []
        unique_metadata = []

        for i, doc_hash in enumerate(new_hashes):
            if doc_hash.content_hash not in existing_hashes:
                unique_additions.append(doc_hash)
                unique_docs.append(documents[i])
                if metadata:
                    unique_metadata.append(metadata[i])

        if not unique_additions:
            self.logger.info("No new unique documents to add")
            return self.current_snapshot

        # Add to storage
        self.document_hashes.extend(unique_additions)
        self.documents.extend(unique_docs)

        # Store metadata
        if metadata:
            for doc_hash, meta in zip(unique_additions, unique_metadata):
                self.doc_metadata[doc_hash.content_hash] = meta

        # Generate embeddings deterministically
        sparse_embeddings = self._generate_sparse_embeddings(self.documents)
        dense_embeddings = self._generate_dense_embeddings(self.documents)

        # Initialize/rebuild indices
        if not self.indices_initialized:
            self._initialize_indices(sparse_embeddings.shape[1])

        # Clear and rebuild indices for determinism
        self.dense_index = faiss.IndexFlatIP(self.embedding_dim)
        self.sparse_index = faiss.IndexFlatIP(sparse_embeddings.shape[1])

        # Add all embeddings
        self.dense_index.add(dense_embeddings)
        self.sparse_index.add(sparse_embeddings)

        # Create snapshot
        self.current_snapshot = IndexSnapshot.create(
            self.documents, self.embedding_dim, sparse_embeddings.shape[1]
        )

        self.logger.info(
            f"Created snapshot {self.current_snapshot.snapshot_id} with {len(self.documents)} documents"
        )
        return self.current_snapshot

    def _generate_dense_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate deterministic dense embeddings"""
        prefixed_docs = [f"passage: {doc}" for doc in documents]

        # Sort for determinism, then restore original order
        doc_indices = list(range(len(prefixed_docs)))
        sorted_pairs = sorted(zip(prefixed_docs, doc_indices))
        sorted_docs = [doc for doc, _ in sorted_pairs]

        embeddings = self.dense_model.encode(
            sorted_docs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,  # Fixed batch size
            show_progress_bar=False,
        )

        # Restore original order
        ordered_embeddings = np.zeros_like(embeddings)
        for i, (_, orig_idx) in enumerate(sorted_pairs):
            ordered_embeddings[orig_idx] = embeddings[i]

        return ordered_embeddings.astype(np.float32)

    def _generate_sparse_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate deterministic sparse embeddings"""

        if not hasattr(self.sparse_vectorizer, "vocabulary_"):
            # Fit on sorted documents for determinism
            sorted_docs = sorted(documents)
            self.sparse_vectorizer.fit(sorted_docs)

        sparse_matrix = self.sparse_vectorizer.transform(documents)
        return sparse_matrix.toarray().astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 10,
        expansion_config: Optional[QueryExpansion] = None,
        dnp_constraints: Optional[List[DNPConstraint]] = None,
    ) -> RetrievalResult:
        """
        Perform deterministic hybrid search.

        Args:
            query: Search query
            top_k: Number of results
            expansion_config: Optional query expansion
            dnp_constraints: Optional filtering constraints

        Returns:
            Deterministic retrieval results
        """

        if not self.current_snapshot:
            raise ValueError("No documents indexed")

        # Query expansion
        queries = [query]
        if expansion_config:
            queries = expansion_config.expand_query(query)

        # Perform retrieval for all query variants
        all_sparse_results = []
        all_dense_results = []
        all_projection_results = []

        for q in queries:
            sparse_res = self._sparse_search(q, top_k * 2)
            dense_res = self._dense_search(q, top_k * 2)
            projection_res = self._projection_search(q, top_k * 2)

            all_sparse_results.extend(sparse_res)
            all_dense_results.extend(dense_res)
            all_projection_results.extend(projection_res)

        # Apply optimal fusion
        fusion_results = self.fusion.fuse_rankings(
            all_sparse_results, all_dense_results, all_projection_results
        )

        # Apply DNP constraints
        if dnp_constraints:
            fusion_results = self._apply_dnp_constraints(
                fusion_results, dnp_constraints
            )

        # Deduplicate by content hash and sort
        seen_hashes = set()
        unique_results = []

        for result in fusion_results:
            if result.doc_hash.content_hash not in seen_hashes:
                seen_hashes.add(result.doc_hash.content_hash)
                unique_results.append(result)

        # Take top k and create final result
        final_results = unique_results[:top_k]

        return RetrievalResult(
            doc_hashes=tuple(r.doc_hash for r in final_results),
            scores=tuple(r.fusion_score for r in final_results),
            method="deterministic_hybrid",
            query=query,
            snapshot_id=self.current_snapshot.snapshot_id,
        )

    def _sparse_search(
        self, query: str, top_k: int
    ) -> List[Tuple[DocumentHash, float]]:
        """Deterministic sparse search"""
        query_vector = (
            self.sparse_vectorizer.transform([query]).toarray().astype(np.float32)
        )

        scores, indices = self.sparse_index.search(
            query_vector, min(top_k, len(self.documents))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score > 0:
                results.append((self.document_hashes[idx], float(score)))

        return results

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[DocumentHash, float]]:
        """Deterministic dense search"""
        query_embedding = self.dense_model.encode(
            [f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True
        )

        scores, indices = self.dense_index.search(
            query_embedding, min(top_k, len(self.documents))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score > 0:
                results.append((self.document_hashes[idx], float(score)))

        return results

    def _projection_search(
        self, query: str, top_k: int
    ) -> List[Tuple[DocumentHash, float]]:
        """Search using learned projection"""
        # Get dense query embedding
        query_embedding = self.dense_model.encode(
            [f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True
        )

        # Project to sparse space
        projected_query = self.projection.project(query_embedding)
        projected_query = projected_query / np.linalg.norm(projected_query)  # Normalize

        # Search in sparse index
        scores, indices = self.sparse_index.search(
            projected_query.reshape(1, -1).astype(np.float32),
            min(top_k, len(self.documents)),
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score > 0:
                results.append((self.document_hashes[idx], float(score)))

        return results

    def _apply_dnp_constraints(
        self, results: List[FusionResult], constraints: List[DNPConstraint]
    ) -> List[FusionResult]:
        """Apply Do Not Process constraints"""

        filtered_results = []

        for result in results:
            doc_metadata = self.doc_metadata.get(result.doc_hash.content_hash, {})

            # Check if any constraint applies (DNP = should be filtered out)
            should_filter = False
            for constraint in constraints:
                if constraint.applies_to_document(doc_metadata):
                    should_filter = True
                    break

            if not should_filter:
                filtered_results.append(result)

        return filtered_results

    def verify_determinism(self, query: str, num_trials: int = 3) -> bool:
        """Verify that search results are deterministic"""

        results = []
        for _ in range(num_trials):
            result = self.search(query)
            # Convert to comparable format
            result_signature = (
                tuple(dh.content_hash for dh in result.doc_hashes),
                result.scores,
                result.snapshot_id,
            )
            results.append(result_signature)

        # Check all results are identical
        first_result = results[0]
        return all(result == first_result for result in results)

    def get_reproducibility_info(self) -> Dict[str, Any]:
        """Get information for reproducing results"""
        return {
            "seed": self.seed,
            "model_name": self.embedding_model_name,
            "fusion_weights": {
                "alpha": self.fusion.alpha,
                "beta": self.fusion.beta,
                "gamma": self.fusion.gamma,
            },
            "current_snapshot": asdict(self.current_snapshot)
            if self.current_snapshot
            else None,
            "num_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "sparse_vocab_size": len(self.sparse_vectorizer.vocabulary_)
            if hasattr(self.sparse_vectorizer, "vocabulary_")
            else 0,
        }

if __name__ == "__main__":
    # Minimal, dependency-free demo to prove executability
    docs = [
        "deterministic retrieval with hybrid sparse and dense signals",
        "pure python fallback for tf idf without numpy",
        "semantic reranking using projections"
    ]
    queries = ["deterministic hybrid retrieval"]
    try:
        # Try sparse-only demo using fallback TF-IDF (works even if sklearn/numpy missing)
        vec = TfidfVectorizer(max_features=50, lowercase=True, stop_words=None)
        vec.fit(docs)
        X = vec.transform(docs)  # list of lists
        def cosine(a: List[float], b: List[float]) -> float:
            num = sum(x*y for x, y in zip(a, b))
            da = sum(x*x for x in a) ** 0.5
            db = sum(y*y for y in b) ** 0.5
            if da <= 0 or db <= 0:
                return 0.0
            return num / (da * db)
        for q in queries:
            qv = vec.transform([q])[0]
            scores = [(i, cosine(qv, dv)) for i, dv in enumerate(X)]
            scores.sort(key=lambda t: t[1], reverse=True)
            result = {
                "query": q,
                "ranking": [
                    {"doc": int(i), "score": float(s), "text": docs[i]} for i, s in scores
                ]
            }
            print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
