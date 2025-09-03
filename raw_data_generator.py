"""
Raw Data Artifacts Generator

This module generates four deterministic artifacts using hash-stable algorithms
to ensure identical outputs across runs given the same input data:
1. features.parquet - Feature matrix with deterministic encoding
2. embeddings.faiss - Vector embeddings index with stable initialization
3. bm25.idx - BM25 lexical index with deterministic document ordering
4. vec.idx - Vector index with hash-stable construction

Implements hash verification mechanisms to validate artifact integrity and
ensure reproducible pipeline execution for the comprehensive orchestrator's
raw data dependencies.
"""

import hashlib
import json
import os
import random
import struct
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found

import faiss
import numpy as np
import pandas as pd
import torch
# Optional sklearn with fallback for TF-IDF
try:
# # #     from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore  # Module not found  # Module not found  # Module not found
except Exception:
    class TfidfVectorizer:  # minimal fallback
        def __init__(self, ngram_range=(1, 1), min_df=1, max_features=None, lowercase=True, stop_words=None):
            self.ngram_range = ngram_range
            self.min_df = min_df
            self.max_features = max_features
            self.lowercase = lowercase
            self.stop_words = set(stop_words) if stop_words else None
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
# # # from transformers import AutoModel, AutoTokenizer  # Module not found  # Module not found  # Module not found

# # # from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found

# Set deterministic seeds for all random number generators
DETERMINISTIC_SEED = 42


class HashStableRandom(TotalOrderingBase):
    """Hash-stable random number generator for deterministic artifacts."""

    def __init__(self, seed_data: Union[str, bytes, int]):
# # #         """Initialize with deterministic seed from input data."""  # Module not found  # Module not found  # Module not found
        super().__init__(component_name="HashStableRandom")

        if isinstance(seed_data, (str, bytes)):
# # #             # Create deterministic seed from hash of input data  # Module not found  # Module not found  # Module not found
            hash_digest = hashlib.sha256(
                seed_data.encode() if isinstance(seed_data, str) else seed_data
            ).digest()
            # Convert first 8 bytes to integer seed
            self.seed = struct.unpack('Q', hash_digest[:8])[0]
        else:
            self.seed = int(seed_data)

        self.rng = np.random.RandomState(self.seed)

        # State tracking
        self._random_calls = 0
        self._seed_data_hash = self.generate_stable_id(str(seed_data), prefix="seed")

    def random(self):
        """Generate deterministic random float."""
        self._random_calls += 1
        return self.rng.random()

    def randint(self, low, high):
        """Generate deterministic random integer."""
        self._random_calls += 1
        return self.rng.randint(low, high)

    def choice(self, array, size=None, replace=True):
        """Make deterministic random choice."""
        self._random_calls += 1
        return self.rng.choice(array, size=size, replace=replace)

    def shuffle(self, array):
        """Shuffle array deterministically in place."""
        self._random_calls += 1
        self.rng.shuffle(array)
        return array

    def __lt__(self, other):
        """Comparison based on seed for stable sorting"""
        if not isinstance(other, HashStableRandom):
            return NotImplemented
        return self.seed < other.seed

    def to_json_dict(self) -> Dict[str, Any]:
        """Generate canonical JSON representation"""
        return self.serialize_output({
            "component_id": self.component_id,
            "component_name": self.component_name,
            "seed": self.seed,
            "seed_data_hash": self._seed_data_hash,
            "random_calls": self._random_calls,
            "metadata": self.get_deterministic_metadata()
        })


class DeterministicFeatureGenerator(TotalOrderingBase):
# # #     """Generates deterministic feature matrices from text data."""  # Module not found  # Module not found  # Module not found

    def __init__(self, seed_data: Union[str, bytes]):
        super().__init__(component_name="DeterministicFeatureGenerator")

        self.hasher = HashStableRandom(seed_data)

        # State tracking
        self._features_generated = 0
        self._documents_processed = 0

        # Generate configuration ID
        config_data = {
            "seed_hash": self.hasher._seed_data_hash,
            "generator_type": "features"
        }
        self._config_id = self.generate_stable_id(config_data, prefix="featgen")

    def generate_features_parquet(
        self,
        documents: List[str],
        output_path: str,
        feature_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate deterministic features.parquet file.

        Args:
            documents: List of input documents
            output_path: Path for output parquet file
            feature_config: Configuration for feature extraction

        Returns:
            SHA256 hash of generated file
        """
        self._features_generated += 1
        self._documents_processed += len(documents)

        if feature_config is None:
            feature_config = {
                'max_features': 1000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95
            }

        # Generate operation ID for this generation
        operation_id = self.generate_operation_id(
            "generate_features_parquet",
            {
                "document_count": len(documents),
                "output_path": output_path,
                "config": self.sort_dict_by_keys(feature_config)
            }
        )

        # Sort documents deterministically by content hash
        doc_hashes = [(hashlib.sha256(doc.encode()).hexdigest(), doc)
                     for doc in documents]
        doc_hashes.sort()  # Sort by hash for deterministic ordering
        sorted_documents = [doc for _, doc in doc_hashes]

        # Create deterministic TF-IDF features
        vectorizer = TfidfVectorizer(
            random_state=self.hasher.seed,
            **feature_config
        )

        tfidf_matrix = vectorizer.fit_transform(sorted_documents)
        feature_names = vectorizer.get_feature_names_out()

        # Convert to DataFrame with deterministic column ordering
        df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=sorted(feature_names)  # Sort feature names for consistency
        )

        # Add document metadata
        df['doc_id'] = range(len(sorted_documents))
        df['doc_hash'] = [hash_val for hash_val, _ in doc_hashes]
        df['doc_length'] = [len(doc) for doc in sorted_documents]

        # Sort columns deterministically
        df = df[sorted(df.columns)]

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save with deterministic options
        df.to_parquet(
            output_path,
            index=False,
            compression='snappy',
            engine='pyarrow'
        )

        # Update state tracking
        state_data = {
            "features_generated": self._features_generated,
            "documents_processed": self._documents_processed,
            "config_id": self._config_id
        }
        self.update_state_hash(state_data)

        return self._compute_file_hash(output_path)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def __lt__(self, other):
        """Comparison based on features generated for stable sorting"""
        if not isinstance(other, DeterministicFeatureGenerator):
            return NotImplemented
        return self._features_generated < other._features_generated

    def to_json_dict(self) -> Dict[str, Any]:
        """Generate canonical JSON representation"""
        return self.serialize_output({
            "component_id": self.component_id,
            "component_name": self.component_name,
            "config_id": self._config_id,
            "features_generated": self._features_generated,
            "documents_processed": self._documents_processed,
            "hasher_state": self.hasher.to_json_dict(),
            "metadata": self.get_deterministic_metadata()
        })


class DeterministicEmbeddingGenerator(TotalOrderingBase):
    """Generates deterministic FAISS embedding indexes."""

    def __init__(self, seed_data: Union[str, bytes]):
        super().__init__(component_name="DeterministicEmbeddingGenerator")

        self.hasher = HashStableRandom(seed_data)
        self._set_deterministic_torch_state()

        # State tracking
        self._embeddings_generated = 0
        self._documents_processed = 0

        # Generate configuration ID
        config_data = {
            "seed_hash": self.hasher._seed_data_hash,
            "generator_type": "embeddings"
        }
        self._config_id = self.generate_stable_id(config_data, prefix="embgen")

    def _set_deterministic_torch_state(self):
        """Set deterministic state for PyTorch operations."""
        torch.manual_seed(self.hasher.seed)
        torch.cuda.manual_seed_all(self.hasher.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate_embeddings_faiss(
        self,
        documents: List[str],
        output_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate deterministic embeddings.faiss index.

        Args:
            documents: List of input documents
            output_path: Path for output FAISS index
            model_name: Hugging Face model for embeddings
            embedding_config: Configuration for embedding generation

        Returns:
            SHA256 hash of generated index file
        """
        if embedding_config is None:
            embedding_config = {
                'batch_size': 32,
                'max_length': 512,
                'index_type': 'IndexFlatIP'  # Inner product for cosine similarity
            }

        # Load model with deterministic settings
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode for deterministic behavior

        # Sort documents by hash for deterministic ordering
        doc_hashes = [(hashlib.sha256(doc.encode()).hexdigest(), doc)
                     for doc in documents]
        doc_hashes.sort()
        sorted_documents = [doc for _, doc in doc_hashes]

        # Generate embeddings deterministically
        embeddings = []
        batch_size = embedding_config['batch_size']

        with torch.no_grad():
            for i in range(0, len(sorted_documents), batch_size):
                batch_docs = sorted_documents[i:i + batch_size]

                # Tokenize with deterministic settings
                tokens = tokenizer(
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=embedding_config['max_length'],
                    return_tensors='pt'
                )

                # Generate embeddings
                outputs = model(**tokens)
                # Use mean pooling for sentence embeddings
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                # Normalize embeddings for cosine similarity
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, p=2, dim=1
                )

                embeddings.append(batch_embeddings.cpu().numpy())

        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings).astype(np.float32)

        # Create FAISS index with deterministic initialization
        dimension = all_embeddings.shape[1]

        if embedding_config['index_type'] == 'IndexFlatIP':
            index = faiss.IndexFlatIP(dimension)
        elif embedding_config['index_type'] == 'IndexFlatL2':
            index = faiss.IndexFlatL2(dimension)
        else:
            # Default to inner product
            index = faiss.IndexFlatIP(dimension)

        # Add embeddings to index
        index.add(all_embeddings)

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(index, output_path)

        return self._compute_file_hash(output_path)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class DeterministicBM25Generator(TotalOrderingBase):
    """Generates deterministic BM25 lexical indexes."""

    def __init__(self, seed_data: Union[str, bytes]):
        super().__init__(component_name="DeterministicBM25Generator")

        self.hasher = HashStableRandom(seed_data)

        # State tracking
        self._indexes_generated = 0
        self._documents_processed = 0

        # Generate configuration ID
        config_data = {
            "seed_hash": self.hasher._seed_data_hash,
            "generator_type": "bm25"
        }
        self._config_id = self.generate_stable_id(config_data, prefix="bm25gen")

    def generate_bm25_index(
        self,
        documents: List[str],
        output_path: str,
        bm25_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate deterministic BM25 index file.

        Args:
            documents: List of input documents
            output_path: Path for output BM25 index
            bm25_config: Configuration for BM25 parameters

        Returns:
            SHA256 hash of generated index file
        """
        if bm25_config is None:
            bm25_config = {
                'k1': 1.2,
                'b': 0.75,
                'min_df': 2,
                'max_df': 0.95
            }

        # Sort documents by hash for deterministic ordering
        doc_hashes = [(hashlib.sha256(doc.encode()).hexdigest(), doc)
                     for doc in documents]
        doc_hashes.sort()
        sorted_documents = [doc for _, doc in doc_hashes]

        # Create BM25 vectorizer with deterministic settings
        vectorizer = TfidfVectorizer(
            use_idf=True,
            norm=None,  # Don't normalize for BM25
            min_df=bm25_config['min_df'],
            max_df=bm25_config['max_df'],
            lowercase=True,
            stop_words='english'
        )

        # Fit and transform documents
        tfidf_matrix = vectorizer.fit_transform(sorted_documents)

        # Create BM25 index structure
        bm25_index = {
            'documents': sorted_documents,
            'doc_hashes': [hash_val for hash_val, _ in doc_hashes],
            'vocabulary': dict(vectorizer.vocabulary_),
            'idf_scores': vectorizer.idf_.tolist(),
            'feature_names': vectorizer.get_feature_names_out().tolist(),
            'doc_frequencies': (tfidf_matrix > 0).sum(axis=0).A1.tolist(),
            'doc_lengths': [len(doc.split()) for doc in sorted_documents],
            'avg_doc_length': np.mean([len(doc.split()) for doc in sorted_documents]),
            'k1': bm25_config['k1'],
            'b': bm25_config['b'],
            'total_docs': len(sorted_documents)
        }

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON with deterministic formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                bm25_index,
                f,
                indent=2,
                sort_keys=True,  # Ensure deterministic key ordering
                ensure_ascii=False
            )

        return self._compute_file_hash(output_path)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class DeterministicVectorGenerator(TotalOrderingBase):
    """Generates deterministic vector indexes."""

    def __init__(self, seed_data: Union[str, bytes]):
        super().__init__(component_name="DeterministicVectorGenerator")

        self.hasher = HashStableRandom(seed_data)

        # State tracking
        self._vectors_generated = 0
        self._documents_processed = 0

        # Generate configuration ID
        config_data = {
            "seed_hash": self.hasher._seed_data_hash,
            "generator_type": "vector"
        }
        self._config_id = self.generate_stable_id(config_data, prefix="vecgen")

    def generate_vector_index(
        self,
        documents: List[str],
        output_path: str,
        vector_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate deterministic vector index file.

        Args:
            documents: List of input documents
            output_path: Path for output vector index
            vector_config: Configuration for vector generation

        Returns:
            SHA256 hash of generated index file
        """
        if vector_config is None:
            vector_config = {
                'vector_size': 300,
                'min_count': 2,
                'window': 5,
                'sg': 1,  # Skip-gram
                'workers': 1  # Single worker for determinism
            }

        # Sort documents by hash for deterministic ordering
        doc_hashes = [(hashlib.sha256(doc.encode()).hexdigest(), doc)
                     for doc in documents]
        doc_hashes.sort()
        sorted_documents = [doc for _, doc in doc_hashes]

        # Tokenize documents deterministically
        tokenized_docs = []
        for doc in sorted_documents:
            # Simple whitespace tokenization with deterministic lowercase
            tokens = doc.lower().split()
            # Sort tokens within each document for additional determinism
            tokens.sort()
            tokenized_docs.append(tokens)

        # Create vocabulary with deterministic ordering
        vocabulary = set()
        for tokens in tokenized_docs:
            vocabulary.update(tokens)

        # Filter by minimum count
        token_counts = {}
        for tokens in tokenized_docs:
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        vocabulary = {token for token in vocabulary
                     if token_counts[token] >= vector_config['min_count']}
        vocabulary = sorted(vocabulary)  # Deterministic ordering

        # Create deterministic random vectors
        vector_size = vector_config['vector_size']
        word_vectors = {}

        for word in vocabulary:
            # Use word as seed for its vector
            word_hasher = HashStableRandom(word + str(self.hasher.seed))
            # Generate normalized random vector
            vector = np.array([word_hasher.random() - 0.5
                              for _ in range(vector_size)])
            vector = vector / np.linalg.norm(vector)  # Normalize
            word_vectors[word] = vector.tolist()

        # Create vector index structure
        vector_index = {
            'vocabulary': vocabulary,
            'vectors': word_vectors,
            'vector_size': vector_size,
            'total_words': len(vocabulary),
            'documents_processed': len(sorted_documents),
            'config': vector_config,
            'doc_hashes': [hash_val for hash_val, _ in doc_hashes]
        }

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON with deterministic formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                vector_index,
                f,
                indent=2,
                sort_keys=True,  # Ensure deterministic key ordering
                ensure_ascii=False
            )

        return self._compute_file_hash(output_path)

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class RawDataArtifactGenerator(TotalOrderingBase):
    """Main orchestrator for generating all deterministic artifacts."""

    def __init__(self, base_output_dir: str = "data/raw_artifacts"):
        super().__init__(component_name="RawDataArtifactGenerator")

        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self._generations_performed = 0
        self._total_documents_processed = 0
        self._artifacts_created = 0

        # Generate configuration ID
        config_data = {
            "base_output_dir": str(self.base_output_dir.resolve()),
            "generator_type": "artifact_orchestrator"
        }
        self._config_id = self.generate_stable_id(config_data, prefix="artgen")

    def generate_all_artifacts(
        self,
        documents: List[str],
        artifact_configs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate all four deterministic artifacts.

        Args:
            documents: List of input documents
            artifact_configs: Configurations for each artifact type

        Returns:
            Dictionary mapping artifact names to their SHA256 hashes
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self._generations_performed += 1
        self._total_documents_processed += len(documents)

        if artifact_configs is None:
            artifact_configs = {}

        # Generate operation ID for this generation
        operation_id = self.generate_operation_id(
            "generate_all_artifacts",
            {
                "document_count": len(documents),
                "config_keys": sorted(artifact_configs.keys()),
                "base_output_dir": str(self.base_output_dir)
            }
        )

# # #         # Create deterministic seed from all input documents (sorted)  # Module not found  # Module not found  # Module not found
        sorted_documents = sorted(documents)
        combined_content = "\n".join(sorted_documents)
        seed_data = hashlib.sha256(combined_content.encode()).digest()

        results = {}

        # Define generators in deterministic order
        generators = [
            ("features.parquet", DeterministicFeatureGenerator, "generate_features_parquet", "features"),
            ("embeddings.faiss", DeterministicEmbeddingGenerator, "generate_embeddings_faiss", "embeddings"),
            ("bm25.idx", DeterministicBM25Generator, "generate_bm25_index", "bm25"),
            ("vec.idx", DeterministicVectorGenerator, "generate_vector_index", "vector")
        ]

        # Generate artifacts in deterministic order
        for artifact_name, generator_class, method_name, config_key in generators:
            generator = generator_class(seed_data)
            artifact_path = self.base_output_dir / artifact_name
            config = artifact_configs.get(config_key, None)

            # Call the appropriate method
            if hasattr(generator, method_name):
                method = getattr(generator, method_name)
                if config_key == "embeddings":
                    # Handle special case for embeddings
                    results[artifact_name] = method(
                        sorted_documents, str(artifact_path), embedding_config=config
                    )
                else:
                    results[artifact_name] = method(
                        sorted_documents, str(artifact_path), config
                    )

                self._artifacts_created += 1

        # Sort results by key for deterministic output
        sorted_results = self.sort_dict_by_keys(results)

        # Save hash verification manifest
        self._save_verification_manifest(sorted_results, sorted_documents)

        # Update state tracking
        state_data = {
            "generations_performed": self._generations_performed,
            "total_documents_processed": self._total_documents_processed,
            "artifacts_created": self._artifacts_created,
            "config_id": self._config_id
        }
        self.update_state_hash(state_data)

        return sorted_results

    def verify_artifacts(
        self,
        expected_hashes: Dict[str, str],
        recompute_on_mismatch: bool = False
    ) -> Dict[str, bool]:
        """
        Verify integrity of generated artifacts against expected hashes.

        Args:
            expected_hashes: Dictionary mapping artifact names to expected SHA256 hashes
            recompute_on_mismatch: Whether to recompute hash if file doesn't match

        Returns:
            Dictionary mapping artifact names to verification results
        """
        verification_results = {}

        for artifact_name, expected_hash in expected_hashes.items():
            artifact_path = self.base_output_dir / artifact_name

            if not artifact_path.exists():
                verification_results[artifact_name] = False
                continue

            # Compute current hash
            current_hash = self._compute_file_hash(str(artifact_path))

            if current_hash == expected_hash:
                verification_results[artifact_name] = True
            else:
                verification_results[artifact_name] = False
                if recompute_on_mismatch:
                    print(f"Warning: Hash mismatch for {artifact_name}")
                    print(f"Expected: {expected_hash}")
                    print(f"Current:  {current_hash}")

        return verification_results

    def _save_verification_manifest(
        self,
        hashes: Dict[str, str],
        documents: List[str]
    ):
        """Save verification manifest with metadata."""
        combined_content = "\n".join(sorted(documents))
        input_hash = hashlib.sha256(combined_content.encode()).hexdigest()

        manifest = {
            'input_documents_count': len(documents),
            'input_content_hash': input_hash,
            'artifacts': hashes,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'generator_version': '1.0.0'
        }

        manifest_path = self.base_output_dir / "verification_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    def __lt__(self, other):
        """Comparison based on generations performed for stable sorting"""
        if not isinstance(other, RawDataArtifactGenerator):
            return NotImplemented
        return self._generations_performed < other._generations_performed

    def to_json_dict(self) -> Dict[str, Any]:
        """Generate canonical JSON representation"""
        return self.serialize_output({
            "component_id": self.component_id,
            "component_name": self.component_name,
            "config_id": self._config_id,
            "base_output_dir": str(self.base_output_dir),
            "generations_performed": self._generations_performed,
            "total_documents_processed": self._total_documents_processed,
            "artifacts_created": self._artifacts_created,
            "metadata": self.get_deterministic_metadata()
        })

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


def create_raw_data_generator(
    base_output_dir: str = "data/raw_artifacts"
) -> RawDataArtifactGenerator:
    """
    Factory function to create a RawDataArtifactGenerator instance.

    Args:
        base_output_dir: Base directory for output artifacts

    Returns:
        Configured RawDataArtifactGenerator instance
    """
    return RawDataArtifactGenerator(base_output_dir)


# Demo usage
if __name__ == "__main__":
    # Sample documents for testing
    sample_documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Information retrieval systems help users find relevant documents.",
        "Deterministic algorithms produce consistent results across runs."
    ]

    # Create generator
    generator = create_raw_data_generator()

    # Generate all artifacts
    print("Generating deterministic artifacts...")
    hashes = generator.generate_all_artifacts(sample_documents)

    print("\nGenerated artifacts and their hashes:")
    for artifact, hash_value in hashes.items():
        print(f"{artifact}: {hash_value}")

    # Verify artifacts
    print("\nVerifying artifacts...")
    verification = generator.verify_artifacts(hashes)

    for artifact, is_valid in verification.items():
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"{artifact}: {status}")

    print(f"\nAll artifacts saved to: {generator.base_output_dir}")
