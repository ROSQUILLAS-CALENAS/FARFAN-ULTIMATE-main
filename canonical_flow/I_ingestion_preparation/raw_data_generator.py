"""
05I_raw_data_generator.py - Deterministic Raw Data Artifacts Generator

Canonical Flow Module: I_ingestion_preparation/05I_raw_data_generator.py
Stage: I - Ingestion & Preparation 
Order: 05

This module generates four deterministic artifacts using hash-stable algorithms
to ensure identical outputs across runs given the same input data:
1. features.parquet - Feature matrix with deterministic encoding
2. embeddings.faiss - Vector embeddings index with stable initialization  
3. bm25.idx - BM25 lexical index with deterministic document ordering
4. vec.idx - Vector index with hash-stable construction

Implements hash verification mechanisms to validate artifact integrity and
ensure reproducible pipeline execution for the comprehensive orchestrator's
raw data dependencies.

Alias metadata:
- alias_source: raw_data_generator.py
- alias_stage: ingestion_preparation
- alias_code: 05I
"""

import hashlib
import json
import logging
import os
import random
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import with fallbacks for missing dependencies
try:
    import faiss
except ImportError:
    faiss = None

try:
    import numpy as np
except ImportError:
    np = None
    
try:
    import pandas as pd
except ImportError:
    pd = None
    
try:
    import torch
except ImportError:
    torch = None
    
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None
    
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

# Configure logging
logger = logging.getLogger(__name__)

# Set deterministic seeds for all random number generators
DETERMINISTIC_SEED = 42


class HashStableRandom:
    """Hash-stable random number generator for deterministic artifacts."""
    
    def __init__(self, seed_data: Union[str, bytes, int]):
        """Initialize with deterministic seed from input data."""
        if isinstance(seed_data, (str, bytes)):
            # Create deterministic seed from hash of input data
            hash_digest = hashlib.sha256(
                seed_data.encode() if isinstance(seed_data, str) else seed_data
            ).digest()
            # Convert first 8 bytes to integer seed
            self.seed = struct.unpack('Q', hash_digest[:8])[0]
        else:
            self.seed = int(seed_data)
        
        self.rng = np.random.RandomState(self.seed)
        
    def random(self):
        """Generate deterministic random float."""
        return self.rng.random()
    
    def randint(self, low, high):
        """Generate deterministic random integer."""
        return self.rng.randint(low, high)
    
    def choice(self, array, size=None, replace=True):
        """Make deterministic random choice."""
        return self.rng.choice(array, size=size, replace=replace)
    
    def shuffle(self, array):
        """Shuffle array deterministically in place."""
        self.rng.shuffle(array)
        return array


class DeterministicFeatureGenerator:
    """Generates deterministic feature matrices from text data."""
    
    def __init__(self, seed_data: Union[str, bytes]):
        self.hasher = HashStableRandom(seed_data)
        
        # Check for required dependencies
        if not pd or not TfidfVectorizer:
            raise ImportError("pandas and sklearn are required for feature generation")
        
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
        if feature_config is None:
            feature_config = {
                'max_features': 1000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95
            }
        
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
        
        return self._compute_file_hash(output_path)
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class DeterministicEmbeddingGenerator:
    """Generates deterministic FAISS embedding indexes."""
    
    def __init__(self, seed_data: Union[str, bytes]):
        self.hasher = HashStableRandom(seed_data)
        self._set_deterministic_torch_state()
        
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


class DeterministicBM25Generator:
    """Generates deterministic BM25 lexical indexes."""
    
    def __init__(self, seed_data: Union[str, bytes]):
        self.hasher = HashStableRandom(seed_data)
        
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


class DeterministicVectorGenerator:
    """Generates deterministic vector indexes."""
    
    def __init__(self, seed_data: Union[str, bytes]):
        self.hasher = HashStableRandom(seed_data)
        
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


class RawDataArtifactGenerator:
    """Main orchestrator for generating all deterministic artifacts."""
    
    def __init__(self, base_output_dir: str = "data/raw_artifacts"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for minimal dependencies
        if not np:
            raise ImportError("numpy is required for raw data artifact generation")
        
    def generate_all_artifacts(
        self,
        documents: List[str],
        artifact_configs: Optional[Dict[str, Any]] = None,
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Generate all four deterministic artifacts with resilience for failed documents.
        
        Args:
            documents: List of input documents
            artifact_configs: Configurations for each artifact type
            document_metadata: Optional metadata for each document (includes status info)
            
        Returns:
            Dictionary mapping artifact names to their SHA256 hashes
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        if artifact_configs is None:
            artifact_configs = {}
        
        # Filter out failed/no_content documents for processing
        filtered_documents, corpus_stats = self._filter_documents_for_processing(
            documents, document_metadata
        )
        
        if not filtered_documents:
            logger.warning("No valid documents available for processing after filtering")
            # Generate empty artifacts with proper structure
            return self._generate_empty_artifacts(artifact_configs, corpus_stats)
        
        logger.info(f"Processing {len(filtered_documents)} valid documents out of {len(documents)} total")
        
        # Create deterministic seed from filtered documents
        combined_content = "\n".join(sorted(filtered_documents))
        seed_data = hashlib.sha256(combined_content.encode()).digest()
        
        results = {}
        
        try:
            # Generate features.parquet
            feature_gen = DeterministicFeatureGenerator(seed_data)
            features_path = self.base_output_dir / "features.parquet"
            feature_config = artifact_configs.get('features', None)
            results['features.parquet'] = feature_gen.generate_features_parquet(
                filtered_documents, str(features_path), feature_config
            )
            
            # Generate embeddings.faiss
            embedding_gen = DeterministicEmbeddingGenerator(seed_data)
            embeddings_path = self.base_output_dir / "embeddings.faiss"
            embedding_config = artifact_configs.get('embeddings', None)
            results['embeddings.faiss'] = embedding_gen.generate_embeddings_faiss(
                filtered_documents, str(embeddings_path), 
                embedding_config=embedding_config
            )
            
            # Generate bm25.idx
            bm25_gen = DeterministicBM25Generator(seed_data)
            bm25_path = self.base_output_dir / "bm25.idx"
            bm25_config = artifact_configs.get('bm25', None)
            results['bm25.idx'] = bm25_gen.generate_bm25_index(
                filtered_documents, str(bm25_path), bm25_config
            )
            
            # Generate vec.idx
            vector_gen = DeterministicVectorGenerator(seed_data)
            vector_path = self.base_output_dir / "vec.idx"
            vector_config = artifact_configs.get('vector', None)
            results['vec.idx'] = vector_gen.generate_vector_index(
                filtered_documents, str(vector_path), vector_config
            )
            
            # Save hash verification manifest with corpus statistics
            self._save_verification_manifest(results, filtered_documents, corpus_stats)
            
            # Log partial corpus statistics
            self._log_corpus_statistics(corpus_stats, len(filtered_documents))
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating artifacts: {e}")
            # Log partial statistics even on failure
            self._log_corpus_statistics(corpus_stats, 0)
            raise
    
    def _filter_documents_for_processing(
        self,
        documents: List[str],
        document_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Filter out failed/no_content documents and collect statistics.
        
        Args:
            documents: List of input documents
            document_metadata: Optional metadata with document status
            
        Returns:
            Tuple of (filtered_documents, corpus_statistics)
        """
        if not document_metadata:
            # No metadata available, process all documents
            corpus_stats = {
                "total_documents": len(documents),
                "valid_documents": len(documents),
                "failed_documents": 0,
                "no_content_documents": 0,
                "unknown_status_documents": 0,
                "filter_applied": False
            }
            return documents, corpus_stats
        
        filtered_documents = []
        failed_count = 0
        no_content_count = 0
        unknown_status_count = 0
        
        # Process documents with metadata
        for i, doc in enumerate(documents):
            if i < len(document_metadata):
                metadata = document_metadata[i]
                status = metadata.get("status", "unknown")
                
                if status == "success":
                    filtered_documents.append(doc)
                elif status == "failed":
                    failed_count += 1
                    logger.debug(f"Excluding failed document: {metadata.get('doc_id', f'doc_{i}')}")
                elif status == "no_content":
                    no_content_count += 1
                    logger.debug(f"Excluding no-content document: {metadata.get('doc_id', f'doc_{i}')}")
                else:
                    unknown_status_count += 1
                    logger.warning(f"Unknown status '{status}' for document {i}, excluding")
            else:
                # No metadata for this document, include it
                filtered_documents.append(doc)
                unknown_status_count += 1
        
        corpus_stats = {
            "total_documents": len(documents),
            "valid_documents": len(filtered_documents),
            "failed_documents": failed_count,
            "no_content_documents": no_content_count,
            "unknown_status_documents": unknown_status_count,
            "filter_applied": True,
            "success_rate": len(filtered_documents) / len(documents) if documents else 0.0
        }
        
        return filtered_documents, corpus_stats
    
    def _generate_empty_artifacts(
        self,
        artifact_configs: Dict[str, Any],
        corpus_stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate empty but valid artifacts when no documents are available.
        
        Args:
            artifact_configs: Artifact configuration
            corpus_stats: Corpus statistics
            
        Returns:
            Dictionary of artifact hashes
        """
        logger.warning("Generating empty artifacts due to no valid documents")
        
        results = {}
        
        # Generate empty features.parquet
        import pandas as pd
        empty_df = pd.DataFrame({"doc_id": [], "doc_hash": [], "doc_length": []})
        features_path = self.base_output_dir / "features.parquet"
        empty_df.to_parquet(str(features_path), index=False)
        results['features.parquet'] = self._compute_file_hash(str(features_path))
        
        # Generate empty embeddings.faiss
        import faiss
        embeddings_path = self.base_output_dir / "embeddings.faiss"
        # Create empty index with default dimension
        empty_index = faiss.IndexFlatIP(384)  # Default dimension
        faiss.write_index(empty_index, str(embeddings_path))
        results['embeddings.faiss'] = self._compute_file_hash(str(embeddings_path))
        
        # Generate empty bm25.idx
        empty_bm25 = {
            "documents": [],
            "doc_hashes": [],
            "vocabulary": {},
            "total_docs": 0,
            "k1": artifact_configs.get('bm25', {}).get('k1', 1.2),
            "b": artifact_configs.get('bm25', {}).get('b', 0.75)
        }
        bm25_path = self.base_output_dir / "bm25.idx"
        with open(bm25_path, 'w') as f:
            json.dump(empty_bm25, f, indent=2, sort_keys=True)
        results['bm25.idx'] = self._compute_file_hash(str(bm25_path))
        
        # Generate empty vec.idx
        empty_vector = {
            "vocabulary": [],
            "vectors": {},
            "vector_size": artifact_configs.get('vector', {}).get('vector_size', 300),
            "total_words": 0,
            "documents_processed": 0
        }
        vector_path = self.base_output_dir / "vec.idx"
        with open(vector_path, 'w') as f:
            json.dump(empty_vector, f, indent=2, sort_keys=True)
        results['vec.idx'] = self._compute_file_hash(str(vector_path))
        
        # Save manifest with empty corpus stats
        self._save_verification_manifest(results, [], corpus_stats)
        
        return results
    
    def _log_corpus_statistics(
        self,
        corpus_stats: Dict[str, Any],
        processed_documents: int
    ):
        """
        Log comprehensive corpus statistics.
        
        Args:
            corpus_stats: Statistics about the corpus
            processed_documents: Number of documents actually processed
        """
        logger.info("=== Corpus Processing Statistics ===")
        logger.info(f"Total Documents: {corpus_stats['total_documents']}")
        logger.info(f"Valid Documents: {corpus_stats['valid_documents']}")
        logger.info(f"Failed Documents: {corpus_stats['failed_documents']}")
        logger.info(f"No-Content Documents: {corpus_stats['no_content_documents']}")
        logger.info(f"Unknown Status Documents: {corpus_stats['unknown_status_documents']}")
        logger.info(f"Documents Processed for Artifacts: {processed_documents}")
        
        if corpus_stats['total_documents'] > 0:
            success_rate = corpus_stats['valid_documents'] / corpus_stats['total_documents']
            logger.info(f"Success Rate: {success_rate:.2%}")
            
            if success_rate < 0.5:
                logger.warning("Low success rate detected - less than 50% of documents were processable")
            elif success_rate < 0.8:
                logger.warning("Moderate success rate - some documents failed processing")
        
        logger.info("=====================================")
        
        # Save statistics to file
        stats_path = self.base_output_dir / "corpus_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump({
                **corpus_stats,
                "documents_processed_for_artifacts": processed_documents,
                "timestamp": str(pd.Timestamp.now())
            }, f, indent=2)
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
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
        documents: List[str],
        corpus_stats: Optional[Dict[str, Any]] = None
    ):
        """Save verification manifest with metadata and corpus statistics."""
        combined_content = "\n".join(sorted(documents))
        input_hash = hashlib.sha256(combined_content.encode()).hexdigest() if documents else "empty_corpus"
        
        manifest = {
            'input_documents_count': len(documents),
            'input_content_hash': input_hash,
            'artifacts': hashes,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'generator_version': '1.0.0'
        }
        
        # Add corpus statistics if provided
        if corpus_stats:
            manifest['corpus_statistics'] = corpus_stats
        
        manifest_path = self.base_output_dir / "verification_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    
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


def process(data=None, context=None) -> Dict[str, Any]:
    """
    Process API for raw data generator component (05I).
    
    Generates raw data artifacts from validation results and writes 
    standardized artifacts using ArtifactManager.
    
    Args:
        data: Input data (validation results or document content)
        context: Processing context with metadata
        
    Returns:
        Dictionary with processing results and output paths
    """
    # Import ArtifactManager locally to avoid circular imports
    try:
        from canonical_flow.ingestion import ArtifactManager
    except ImportError:
        return {"error": "ArtifactManager not available"}
    
    artifact_manager = ArtifactManager()
    
    # Process input data
    if not data:
        return {"error": "No input data provided"}
    
    results = []
    
    # Handle different input formats
    if isinstance(data, dict) and 'results' in data:
        # Input from 04I component
        validation_results = data['results']
    elif isinstance(data, list):
        validation_results = data
    else:
        validation_results = [data]
    
    # Collect document texts for raw data generation
    documents = []
    document_metadata = []
    stems = []
    
    for validation_result in validation_results:
        try:
            # Extract stem and validation data
            if isinstance(validation_result, dict):
                stem = validation_result.get('stem', validation_result.get('document_stem', 'unknown'))
                stems.append(stem)
                
                # For actual processing, we need to read the validation file
                validation_path = validation_result.get('output_path')
                if validation_path and Path(validation_path).exists():
                    with open(validation_path, 'r', encoding='utf-8') as f:
                        validation_data = json.load(f)
                else:
                    # Use validation_result directly if no file path
                    validation_data = validation_result
                
                # Extract document text (if available) or create placeholder
                document_text = validation_data.get('document_text', f"Sample document content for {stem}")
                documents.append(document_text)
                
                # Create metadata
                metadata = {
                    "document_stem": stem,
                    "status": "processed" if validation_result.get('success', False) else "failed",
                    "compliance_level": validation_data.get('compliance_result', {}).get('overall_level', 'UNKNOWN'),
                    "validation_score": validation_data.get('compliance_result', {}).get('compliance_percentage', 0.0)
                }
                document_metadata.append(metadata)
                
        except Exception as e:
            # Add failed document with minimal metadata
            stem = validation_result.get('stem', 'unknown') if isinstance(validation_result, dict) else 'unknown'
            stems.append(stem)
            documents.append(f"Failed processing for {stem}")
            document_metadata.append({
                "document_stem": stem,
                "status": "failed",
                "error": str(e)
            })
    
    if not documents:
        return {"error": "No documents to process"}
    
    try:
        # Create raw data generator
        generator = RawDataArtifactGenerator(base_output_dir=str(artifact_manager.base_path / "raw_artifacts"))
        
        # Generate all artifacts
        artifact_hashes = generator.generate_all_artifacts(
            documents=documents,
            document_metadata=document_metadata
        )
        
        # Create consolidated raw data artifact
        raw_data_artifact = {
            "generation_metadata": {
                "component": "05I",
                "processor": "RawDataArtifactGenerator",
                "timestamp": str(__import__('datetime').datetime.now()),
                "total_documents": len(documents),
                "successful_documents": len([m for m in document_metadata if m["status"] == "processed"])
            },
            "artifact_hashes": artifact_hashes,
            "document_metadata": document_metadata,
            "generation_statistics": {
                "features_generated": "features.parquet" in artifact_hashes,
                "embeddings_generated": "embeddings.faiss" in artifact_hashes,
                "bm25_generated": "bm25.idx" in artifact_hashes,
                "vector_generated": "vec.idx" in artifact_hashes
            }
        }
        
        # Write consolidated artifact for each input document stem
        for stem in set(stems):  # Use set to avoid duplicates
            stem_artifact_path = artifact_manager.write_artifact(stem, "raw_data", raw_data_artifact)
            
            results.append({
                "stem": stem,
                "success": True,
                "output_path": str(stem_artifact_path),
                "artifacts_generated": list(artifact_hashes.keys()),
                "artifact_type": "raw_data"
            })
        
    except Exception as e:
        # Write error artifacts for all stems
        for stem in stems:
            error_data = {
                "document_stem": stem,
                "error": str(e),
                "processing_metadata": {
                    "component": "05I",
                    "status": "failed",
                    "timestamp": str(__import__('datetime').datetime.now())
                }
            }
            
            try:
                output_path = artifact_manager.write_artifact(stem, "raw_data", error_data)
                results.append({
                    "stem": stem,
                    "success": False,
                    "error": str(e),
                    "output_path": str(output_path),
                    "artifact_type": "raw_data"
                })
            except Exception as artifact_error:
                results.append({
                    "stem": stem,
                    "success": False,
                    "error": f"Processing failed: {str(e)}, Artifact writing failed: {str(artifact_error)}"
                })
    
    return {
        "component": "05I",
        "results": results,
        "total_inputs": len(validation_results),
        "successful_generations": len([r for r in results if r.get('success', False)])
    }


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