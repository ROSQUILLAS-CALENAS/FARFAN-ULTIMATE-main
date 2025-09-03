"""
Constructor de embeddings usando sentence-transformers (10K).
"""

import gc
import hashlib
import json
import logging
import threading
import time
import warnings
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple  # Module not found  # Module not found  # Module not found

import numpy as np
# # # from langdetect import detect  # Module not found  # Module not found  # Module not found
# # # from sentence_transformers import SentenceTransformer  # Module not found  # Module not found  # Module not found
# # # from tqdm import tqdm  # Module not found  # Module not found  # Module not found
# # # from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found

# Optional sklearn cosine_similarity for analysis (used in a small section); provide fallback

# Mandatory Pipeline Contract Annotations
__phase__ = "K"
__code__ = "18K"
__stage_order__ = 3

try:
# # #     from sklearn.metrics.pairwise import cosine_similarity  # type: ignore  # Module not found  # Module not found  # Module not found
except Exception:
    def cosine_similarity(A, B=None):
        import numpy as np
        A = np.asarray(A, dtype=np.float32)
        if B is None:
            B = A
        else:
            B = np.asarray(B, dtype=np.float32)
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_norm @ B_norm.T

# JSON Schema validation
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    warnings.warn("jsonschema library not available. Install with: pip install jsonschema")

# Alias metadata
alias_code = "10K"
alias_stage = "knowledge_extraction"
component_name = "Embedding Builder"

# JSON Schemas for validation
EMBEDDING_SCHEMA = {
    "type": "object",
    "properties": {
        "provenance": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "document_stem": {"type": "string"},
                "processing_status": {"type": "string", "enum": ["success", "failed", "partial"]}
            },
            "required": ["component_id", "timestamp", "document_stem", "processing_status"]
        },
        "embeddings": {
            "type": "object",
            "properties": {
                "node_embeddings": {"type": "object"},
                "edge_embeddings": {"type": "object"},
                "graph_embedding": {"type": "array"},
                "embedding_metadata": {"type": "object"}
            }
        },
        "embedding_analysis": {
            "type": "object",
            "properties": {
                "quality_metrics": {"type": "object"},
                "dimensionality_info": {"type": "object"},
                "similarity_analysis": {"type": "object"}
            }
        }
    },
    "required": ["provenance", "embeddings"]
}

def _write_knowledge_artifact(data: Dict[str, Any], document_stem: str, suffix: str, processing_status: str = "success") -> bool:
    """
    Write knowledge artifact to canonical_flow/knowledge/ directory with standardized naming and validation.

    Args:
        data: The data to write
        document_stem: Document identifier stem
        suffix: Component-specific suffix
        processing_status: Processing status (success/failed/partial)

    Returns:
        bool: True if write was successful
    """
    try:
        # Create output directory
        output_dir = Path("canonical_flow/knowledge")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create standardized filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{document_stem}_{alias_code}_{suffix}_{timestamp}.json"
        output_path = output_dir / filename

        # Add provenance metadata
        artifact = {
            "provenance": {
                "component_id": alias_code,
                "timestamp": datetime.now().isoformat(),
                "document_stem": document_stem,
                "processing_status": processing_status
            },
            **data
        }

        # Validate JSON schema if available
        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(artifact, EMBEDDING_SCHEMA)
                logging.info(f"JSON schema validation passed for {filename}")
            except jsonschema.exceptions.ValidationError as e:
                logging.warning(f"JSON schema validation failed for {filename}: {e}")
                # Continue with writing despite validation failure for debugging

        # Write JSON with UTF-8 encoding and standardized formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)

        logging.info(f"Successfully wrote knowledge artifact: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Failed to write knowledge artifact for {document_stem}_{suffix}: {e}")
        # Try to write a minimal artifact for debugging purposes
        try:
            error_artifact = {
                "provenance": {
                    "component_id": alias_code,
                    "timestamp": datetime.now().isoformat(),
                    "document_stem": document_stem,
                    "processing_status": "failed"
                },
                "error": str(e),
                "attempted_data": str(data)[:1000]  # Truncate for safety
            }

            error_filename = f"{document_stem}_{alias_code}_{suffix}_error_{timestamp}.json"
            error_path = output_dir / error_filename

            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_artifact, f, indent=2, ensure_ascii=False)

            logging.info(f"Wrote error artifact for debugging: {error_path}")
        except Exception as inner_e:
            logging.error(f"Failed to write error artifact: {inner_e}")

        return False

logger = logging.getLogger(__name__)


class EmbeddingCacheManager:
    """LRU cache manager for embeddings with size limits and automatic cleanup."""

    def __init__(self, max_cache_size: int = 10000, max_memory_mb: int = 500):
        self.max_cache_size = max_cache_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.cache_memory_usage = 0
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0

    def _calculate_memory_size(self, embedding: np.ndarray) -> int:
        """Calculate memory size of embedding in bytes."""
        return embedding.nbytes

    def _generate_cache_key(self, text: str, model_name: str = "") -> str:
# # #         """Generate cache key from text and model."""  # Module not found  # Module not found  # Module not found
        combined = f"{model_name}:{text}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def get(self, text: str, model_name: str = "") -> Optional[np.ndarray]:
# # #         """Get embedding from cache, moving to end (most recently used)."""  # Module not found  # Module not found  # Module not found
        cache_key = self._generate_cache_key(text, model_name)

        with self.lock:
            if cache_key in self.cache:
                # Move to end (most recently used)
                embedding = self.cache.pop(cache_key)
                self.cache[cache_key] = embedding
                self.hit_count += 1
                return embedding.copy()
            else:
                self.miss_count += 1
                return None

    def put(self, text: str, embedding: np.ndarray, model_name: str = ""):
        """Put embedding in cache with LRU eviction."""
        cache_key = self._generate_cache_key(text, model_name)
        embedding_size = self._calculate_memory_size(embedding)

        with self.lock:
            # Remove existing entry if it exists
            if cache_key in self.cache:
                old_embedding = self.cache.pop(cache_key)
                self.cache_memory_usage -= self._calculate_memory_size(old_embedding)

            # Evict LRU entries if necessary
            while (len(self.cache) >= self.max_cache_size or
                   self.cache_memory_usage + embedding_size > self.max_memory_bytes):
                if not self.cache:
                    break

                lru_key, lru_embedding = self.cache.popitem(last=False)
                self.cache_memory_usage -= self._calculate_memory_size(lru_embedding)
                logger.debug(f"Evicted LRU embedding cache entry: {lru_key[:16]}...")

            # Add new entry
            self.cache[cache_key] = embedding.copy()
            self.cache_memory_usage += embedding_size

            logger.debug(f"Cached embedding for key: {cache_key[:16]}... "
                        f"(cache size: {len(self.cache)}, memory: {self.cache_memory_usage/1024/1024:.1f}MB)")

    def clear_cache(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.cache_memory_usage = 0
            logger.info("Embedding cache cleared")

    def clear_batch(self):
        """Clear cache between batches to free memory."""
        self.clear_cache()
        gc.collect()
        logger.info("Embedding cache cleared between batches")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

            return {
                "cache_size": len(self.cache),
                "max_cache_size": self.max_cache_size,
                "memory_usage_mb": self.cache_memory_usage / 1024 / 1024,
                "max_memory_mb": self.max_memory_bytes / 1024 / 1024,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate
            }


class EmbeddingBuilder(TotalOrderingBase):
    """Generador de embeddings semánticos para chunks de texto con cache y gestión de memoria."""

    def __init__(
        self,
        model_name: str = "multi-qa-mpnet-base-dot-v1",
        device: str = "cpu",
        enable_cache: bool = True,
        max_cache_size: int = 10000,
        max_cache_memory_mb: int = 500
    ):
        """
        Inicializa el constructor de embeddings.

        Args:
            model_name: Nombre del modelo de sentence-transformers
            device: Dispositivo para ejecutar el modelo ("cpu" | "cuda")
            enable_cache: Habilitar cache de embeddings
            max_cache_size: Número máximo de embeddings en cache
            max_cache_memory_mb: Memoria máxima para cache en MB
        """
        super().__init__("EmbeddingBuilder")
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize cache if enabled
        self.cache_manager = None
        if enable_cache:
            self.cache_manager = EmbeddingCacheManager(
                max_cache_size=max_cache_size,
                max_memory_mb=max_cache_memory_mb
            )

        # Update state hash for deterministic processing
        self.update_state_hash(self._get_initial_state())

        logger.info(
            f"EmbeddingBuilder inicializado con modelo {model_name}, dim={self.embedding_dim}, "
            f"cache={'habilitado' if enable_cache else 'deshabilitado'}"
        )

    def _get_comparison_key(self) -> Tuple[str, ...]:
        """Return comparison key for deterministic ordering"""
        return (
            self.component_id,
            str(self.model_name),
            str(self.device),
            str(self.embedding_dim),
            str(self.cache_manager is not None),
            str(self._state_hash or "")
        )

    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "has_cache": self.cache_manager is not None,
        }

    def generate_embeddings(
        self, chunk_text: str, normalize: bool = True
    ) -> np.ndarray:
        """
        Genera embedding para un chunk de texto con cache.

        Args:
            chunk_text: Texto del chunk
            normalize: Si normalizar el embedding

        Returns:
            Embedding como array numpy
        """
        try:
            # Preprocesar texto para multilingüe
            processed_text = self._preprocess_text(chunk_text)

            # Check cache first if enabled
            if self.cache_manager:
                cached_embedding = self.cache_manager.get(processed_text, self.model_name)
                if cached_embedding is not None:
# # #                     logger.debug("Retrieved embedding from cache")  # Module not found  # Module not found  # Module not found
                    return cached_embedding

            # Generar embedding
            embedding = self.model.encode(
                processed_text, normalize_embeddings=normalize
            )

            embedding_array = embedding.astype(np.float32)

            # Cache the result if cache is enabled
            if self.cache_manager:
                self.cache_manager.put(processed_text, embedding_array, self.model_name)

            # Explicit garbage collection after embedding generation
            gc.collect()

            return embedding_array

        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            raise

    def batch_embeddings(
        self, chunks_list: List[str], batch_size: int = 32, show_progress: bool = True,
        clear_cache_between_batches: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Genera embeddings en lote para mejor eficiencia con gestión de memoria.

        Args:
            chunks_list: Lista de textos de chunks
            batch_size: Tamaño del lote
            show_progress: Mostrar barra de progreso
            clear_cache_between_batches: Limpiar cache entre lotes grandes

        Returns:
            Tupla con matriz de embeddings y lista de chunk_ids
        """
        try:
            # Preprocesar todos los textos
            processed_texts = [self._preprocess_text(text) for text in chunks_list]
            chunk_ids = [f"chunk_{i:06d}" for i in range(len(chunks_list))]

            # Check cache for existing embeddings
            embeddings_list = []
            texts_to_process = []
            indices_to_process = []

            if self.cache_manager:
                logger.info("Checking cache for existing embeddings...")
                for i, text in enumerate(processed_texts):
                    cached = self.cache_manager.get(text, self.model_name)
                    if cached is not None:
                        embeddings_list.append(cached)
                    else:
                        embeddings_list.append(None)  # Placeholder
                        texts_to_process.append(text)
                        indices_to_process.append(i)

                cache_hits = len(processed_texts) - len(texts_to_process)
                logger.info(f"Cache hits: {cache_hits}/{len(processed_texts)}")
            else:
                texts_to_process = processed_texts
                indices_to_process = list(range(len(processed_texts)))

            # Process remaining texts in batches
            if texts_to_process:
                logger.info(f"Processing {len(texts_to_process)} texts in batches of {batch_size}")

                # Process in smaller batches to manage memory
                new_embeddings = []
                for i in range(0, len(texts_to_process), batch_size):
                    batch_texts = texts_to_process[i:i + batch_size]

                    # Generate embeddings for batch
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=min(batch_size, len(batch_texts)),
                        show_progress_bar=show_progress and i == 0,  # Show progress only for first batch
                        normalize_embeddings=True,
                    )

                    new_embeddings.append(batch_embeddings)

                    # Cache new embeddings if cache is enabled
                    if self.cache_manager:
                        for j, text in enumerate(batch_texts):
                            embedding = batch_embeddings[j].astype(np.float32)
                            self.cache_manager.put(text, embedding, self.model_name)

                    # Force garbage collection between batches
                    gc.collect()

                    if show_progress:
                        logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts_to_process) + batch_size - 1)//batch_size}")

                # Concatenate all batch results
                if new_embeddings:
                    all_new_embeddings = np.vstack(new_embeddings)
                else:
                    all_new_embeddings = np.array([]).reshape(0, self.embedding_dim)

                # Combine with cached embeddings
                if self.cache_manager:
                    # Fill in the placeholders with new embeddings
                    new_idx = 0
                    for i in indices_to_process:
                        embeddings_list[i] = all_new_embeddings[new_idx].astype(np.float32)
                        new_idx += 1

                    final_embeddings = np.array(embeddings_list)
                else:
                    final_embeddings = all_new_embeddings.astype(np.float32)
            else:
                # All embeddings were cached
                final_embeddings = np.array(embeddings_list)

            # Clear cache between large batches if requested
            if clear_cache_between_batches and len(chunks_list) > 1000 and self.cache_manager:
                self.cache_manager.clear_batch()

            # Final garbage collection
            gc.collect()

            logger.info(f"Generados {len(final_embeddings)} embeddings en lotes "
                       f"(shape: {final_embeddings.shape})")
            return final_embeddings, chunk_ids

        except Exception as e:
            logger.error(f"Error en batch_embeddings: {e}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """
        Codifica una query de búsqueda con cache.

        Args:
            query: Texto de la query

        Returns:
            Embedding de la query
        """
        # Para queries, usar prefijo específico si el modelo lo soporta
        if "multi-qa" in self.model_name.lower():
            processed_query = f"Query: {self._preprocess_text(query)}"
        else:
            processed_query = self._preprocess_text(query)

        # Check cache first if enabled
        if self.cache_manager:
            cached_embedding = self.cache_manager.get(processed_query, f"query_{self.model_name}")
            if cached_embedding is not None:
# # #                 logger.debug("Retrieved query embedding from cache")  # Module not found  # Module not found  # Module not found
                return cached_embedding

        embedding = self.model.encode(processed_query, normalize_embeddings=True).astype(
            np.float32
        )

        # Cache the query embedding
        if self.cache_manager:
            self.cache_manager.put(processed_query, embedding, f"query_{self.model_name}")

        # Explicit garbage collection
        gc.collect()

        return embedding

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesa texto para embeddings multilingües.

        Args:
            text: Texto original

        Returns:
            Texto preprocesado
        """
        if not text or not text.strip():
            return ""

        # Detectar idioma
        try:
            lang = detect(text)
        except Exception as e:
            try:
# # #                 from exception_telemetry import log_structured_exception  # Module not found  # Module not found  # Module not found
                log_structured_exception(
                    type(e),
                    e,
                    e.__traceback__,
                    context={
                        "text_length": len(text),
                        "text_preview": text[:100] if text else None,
                        "has_text": bool(text and text.strip()),
                    },
                    component="embedding_builder",
                    operation="detect_language",
                )
            except ImportError:
                logger.warning(f"Language detection failed: {e}")
            lang = "es"  # Default a español

        # Limpiar texto
        cleaned_text = text.strip()

        # Truncar si es muy largo (limite típico 512 tokens)
        if len(cleaned_text.split()) > 400:
            cleaned_text = " ".join(cleaned_text.split()[:400])

        return cleaned_text

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache_manager:
            self.cache_manager.clear_cache()

    def clear_cache_between_batches(self):
        """Clear cache and force garbage collection between batches."""
        if self.cache_manager:
            self.cache_manager.clear_batch()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return {"cache_enabled": False}

    def get_embedding_dimension(self) -> int:
        """Retorna la dimensión del embedding."""
        return self.embedding_dim

    def save_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del modelo para metadatos.

        Returns:
            Diccionario con información del modelo
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_seq_length": getattr(self.model, "max_seq_length", 512),
        }

def process_embedding_generation(causal_graph: Any,
                               dnp_results: Optional[Dict[str, Any]] = None,
                               document_stem: str = "unknown") -> Dict[str, Any]:
    """
    Main processing function for embedding generation stage (10K).

    Args:
# # #         causal_graph: Input causal graph from previous stage  # Module not found  # Module not found  # Module not found
        dnp_results: Optional DNP optimization results
        document_stem: Document identifier for output naming

    Returns:
        Dict containing generated embeddings and analysis
    """
    try:
        logging.info(f"Starting embedding generation for document: {document_stem}")

        # Create embedding builder
        builder = EmbeddingBuilder()

# # #         # Extract text content from causal graph  # Module not found  # Module not found  # Module not found
        texts = []
        node_texts = {}
        edge_texts = {}

        if hasattr(causal_graph, 'nodes'):
            for node_id, node_data in causal_graph.nodes(data=True):
                node_text = node_data.get('text', node_id)
                texts.append(node_text)
                node_texts[node_id] = node_text

        if hasattr(causal_graph, 'edges'):
            for source, target, edge_data in causal_graph.edges(data=True):
                edge_text = f"{source} -> {target}"
                if 'relation_type' in edge_data:
                    edge_text += f" ({edge_data['relation_type']})"
                texts.append(edge_text)
                edge_texts[f"{source}->{target}"] = edge_text

        # Generate embeddings
        if texts:
            embeddings = builder.build_embeddings(texts)

            # Separate node and edge embeddings
            node_embeddings = {}
            edge_embeddings = {}

            idx = 0
            for node_id in node_texts:
                if idx < len(embeddings):
                    node_embeddings[node_id] = embeddings[idx].tolist()
                    idx += 1

            for edge_key in edge_texts:
                if idx < len(embeddings):
                    edge_embeddings[edge_key] = embeddings[idx].tolist()
                    idx += 1

            # Generate graph-level embedding (mean of all embeddings)
            if embeddings:
                graph_embedding = np.mean(embeddings, axis=0).tolist()
            else:
                graph_embedding = [0.0] * builder.get_embedding_dimension()
        else:
            node_embeddings = {}
            edge_embeddings = {}
            graph_embedding = [0.0] * builder.get_embedding_dimension()

        # Compute quality metrics
        quality_metrics = {
            "num_nodes_embedded": len(node_embeddings),
            "num_edges_embedded": len(edge_embeddings),
            "embedding_coverage": len(texts) / max(1, len(node_texts) + len(edge_texts)),
            "avg_embedding_norm": float(np.mean([np.linalg.norm(emb) for emb in embeddings])) if embeddings else 0.0
        }

        # Dimensionality info
        dimensionality_info = {
            "embedding_dimension": builder.get_embedding_dimension(),
            "total_embeddings": len(embeddings) if embeddings else 0,
            "model_info": builder.save_model_info()
        }

        # Similarity analysis (compute pairwise similarities for first few embeddings)
        similarity_analysis = {}
        if embeddings and len(embeddings) > 1:
            similarities = cosine_similarity(embeddings[:min(10, len(embeddings))])
            similarity_analysis = {
                "mean_similarity": float(np.mean(similarities[np.triu_indices_from(similarities, k=1)])),
                "max_similarity": float(np.max(similarities[np.triu_indices_from(similarities, k=1)])),
                "min_similarity": float(np.min(similarities[np.triu_indices_from(similarities, k=1)]))
            }

        # Prepare output data
        embedding_data = {
            "embeddings": {
                "node_embeddings": node_embeddings,
                "edge_embeddings": edge_embeddings,
                "graph_embedding": graph_embedding,
                "embedding_metadata": {
                    "model_name": builder.model_name,
                    "embedding_dimension": builder.get_embedding_dimension(),
                    "generation_timestamp": datetime.now().isoformat()
                }
            }
        }

        analysis_data = {
            "embedding_analysis": {
                "quality_metrics": quality_metrics,
                "dimensionality_info": dimensionality_info,
                "similarity_analysis": similarity_analysis
            }
        }

        # Write artifacts to canonical_flow/knowledge/
        _write_knowledge_artifact(embedding_data, document_stem, "embeddings")
        _write_knowledge_artifact(analysis_data, document_stem, "embedding_analysis")

        # Clean up
        builder.clear_cache()

        logging.info(f"Completed embedding generation for document: {document_stem}")
        return {**embedding_data, **analysis_data}

    except Exception as e:
        logging.error(f"Error in embedding generation for {document_stem}: {e}")

        # Write error artifacts for debugging
        error_data = {"error": str(e), "input_type": type(causal_graph).__name__}
        _write_knowledge_artifact(error_data, document_stem, "embeddings", "failed")
        _write_knowledge_artifact(error_data, document_stem, "embedding_analysis", "failed")

        return {
            "embeddings": {
                "node_embeddings": {},
                "edge_embeddings": {},
                "graph_embedding": [],
                "embedding_metadata": {}
            },
            "embedding_analysis": {
                "quality_metrics": {},
                "dimensionality_info": {},
                "similarity_analysis": {}
            },
            "error": str(e)
        }

# Main execution for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the embedding generation process
    import networkx as nx

    mock_graph = nx.DiGraph()
    mock_graph.add_node("economia", text="La economía nacional")
    mock_graph.add_node("empleo", text="Tasas de empleo")
    mock_graph.add_edge("economia", "empleo", relation_type="affects", confidence=0.8)

    result = process_embedding_generation(mock_graph, document_stem="test_document")

    print(f"Embedding generation results:")
    print(f"Node embeddings: {len(result['embeddings']['node_embeddings'])}")
    print(f"Edge embeddings: {len(result['embeddings']['edge_embeddings'])}")
    print(f"Graph embedding dimension: {len(result['embeddings']['graph_embedding'])}")
    print(f"Embedding coverage: {result['embedding_analysis']['quality_metrics'].get('embedding_coverage', 0):.3f}")

    print("="*80)
    print("EMBEDDING BUILDER COMPLETED")
    print("="*80)
    print(f"Component: {component_name} ({alias_code})")
    print(f"Stage: {alias_stage}")
    print(f"Artifacts written to: canonical_flow/knowledge/")
    print("="*80)


    def process(self, data=None, context=None) -> Dict[str, Any]:
        """
        Process method with audit integration for 11K component.

        Args:
            data: Input data containing chunks or text to process
            context: Additional context information

        Returns:
            Dict containing processed embeddings and metadata
        """
# # #         from canonical_flow.knowledge.knowledge_audit_system import audit_component_execution  # Module not found  # Module not found  # Module not found

        @audit_component_execution("11K", metadata={"model_name": self.model_name})
        def _process_with_audit(data, context):
            if data is None:
                return {"error": "No input data provided"}

            try:
# # #                 # Extract text chunks from data  # Module not found  # Module not found  # Module not found
                chunks_list = []
                if isinstance(data, dict):
                    if "chunks" in data:
                        chunks_list = data["chunks"]
                    elif "text" in data:
                        chunks_list = [data["text"]]
                    elif "texts" in data:
                        chunks_list = data["texts"]
                elif isinstance(data, list):
                    chunks_list = data
                elif isinstance(data, str):
                    chunks_list = [data]

                if not chunks_list:
                    return {"error": "No valid text chunks found in input data"}

                # Generate embeddings
                embeddings_matrix, chunk_ids = self.batch_embeddings(
                    chunks_list=chunks_list,
                    batch_size=getattr(context, 'batch_size', 32) if context else 32,
                    show_progress=getattr(context, 'show_progress', True) if context else True
                )

                # Prepare result
                result = {
                    "embeddings": embeddings_matrix.tolist(),
                    "chunk_ids": chunk_ids,
                    "embedding_dimension": self.get_embedding_dimension(),
                    "model_info": self.save_model_info(),
                    "cache_stats": self.get_cache_stats(),
                    "num_processed": len(chunk_ids)
                }

                return result

            except Exception as e:
                logger.error(f"Error in embedding_builder process: {e}")
                raise

        return _process_with_audit(data, context)
