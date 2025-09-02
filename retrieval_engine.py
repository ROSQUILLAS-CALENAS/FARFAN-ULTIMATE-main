"""
Retrieval engine with integrated traceability.
Performs document retrieval operations with complete audit trail.
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from traceability import TraceabilityManager, NodeType, ProcessingMetadata
from document_processor import DocumentChunk
from embedding_generator import ChunkEmbedding, EmbeddingGenerator


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query_id: str
    retrieved_chunks: List[DocumentChunk]
    similarity_scores: List[float]
    embedding_similarities: List[float]
    retrieval_metadata: Dict[str, Any]


@dataclass
class Query:
    """Query representation with metadata."""
    query_id: str
    query_text: str
    query_embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]


class RetrievalEngine:
    """Retrieve relevant chunks with traceability integration."""
    
    def __init__(
        self, 
        traceability_manager: TraceabilityManager,
        embedding_generator: EmbeddingGenerator,
        similarity_threshold: float = 0.1,
        max_results: int = 10
    ):
        self.traceability_manager = traceability_manager
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Storage for embeddings and chunks
        self.chunk_embeddings: Dict[str, ChunkEmbedding] = {}
        self.chunks: Dict[str, DocumentChunk] = {}
    
    def index_chunks(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: List[ChunkEmbedding]
    ) -> None:
        """Index chunks and their embeddings for retrieval."""
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks[chunk.chunk_id] = chunk
            self.chunk_embeddings[chunk.chunk_id] = embedding
    
    def retrieve(
        self, 
        query: Query, 
        chain_id: str,
        max_results: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query with traceability.
        
        Args:
            query: Query object with text and metadata
            chain_id: Traceability chain ID
            max_results: Maximum number of results (overrides default)
            
        Returns:
            RetrievalResult with retrieved chunks and scores
        """
        start_time = time.time()
        max_results = max_results or self.max_results
        
        # Generate query embedding if not provided
        if query.query_embedding is None:
            query.query_embedding = self.embedding_generator._simulate_embedding(query.query_text)
        
        # Find embedding generation nodes as parents for retrieval
        chain = self.traceability_manager.get_chain(chain_id)
        embedding_nodes = [node for node in chain.nodes.values() 
                          if node.node_type == NodeType.EMBEDDING_GENERATION]
        
        parent_node_ids = [node.node_id for node in embedding_nodes]
        
        # Perform similarity search
        chunk_embeddings_list = list(self.chunk_embeddings.values())
        similarity_results = self.embedding_generator.similarity_search(
            query.query_embedding, 
            chunk_embeddings_list, 
            top_k=max_results
        )
        
        # Filter by threshold
        filtered_results = [
            (chunk_emb, score) for chunk_emb, score in similarity_results 
            if score >= self.similarity_threshold
        ]
        
        # Prepare retrieval result
        retrieved_chunks = []
        similarity_scores = []
        embedding_similarities = []
        
        for chunk_emb, score in filtered_results:
            chunk = self.chunks[chunk_emb.chunk_id]
            retrieved_chunks.append(chunk)
            similarity_scores.append(score)
            embedding_similarities.append(score)  # Same as similarity for this implementation
        
        execution_time = (time.time() - start_time) * 1000
        
        retrieval_result = RetrievalResult(
            query_id=query.query_id,
            retrieved_chunks=retrieved_chunks,
            similarity_scores=similarity_scores,
            embedding_similarities=embedding_similarities,
            retrieval_metadata={
                "execution_time_ms": execution_time,
                "total_candidates": len(chunk_embeddings_list),
                "threshold_used": self.similarity_threshold,
                "results_returned": len(retrieved_chunks)
            }
        )
        
        # Create traceability node for retrieval operation
        processing_metadata = ProcessingMetadata(
            step_name="retrieval_operation",
            parameters={
                "query_text": query.query_text,
                "similarity_threshold": self.similarity_threshold,
                "max_results": max_results,
                "total_candidates": len(chunk_embeddings_list)
            },
            execution_time_ms=execution_time,
            algorithm_config={
                "similarity_metric": "cosine",
                "embedding_model": self.embedding_generator.model_name
            }
        )
        
        retrieval_node_content = {
            "query_id": query.query_id,
            "query_text": query.query_text,
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved_chunks],
            "similarity_scores": similarity_scores,
            "retrieval_stats": {
                "total_candidates": len(chunk_embeddings_list),
                "results_above_threshold": len(retrieved_chunks),
                "average_score": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
                "max_score": max(similarity_scores) if similarity_scores else 0,
                "min_score": min(similarity_scores) if similarity_scores else 0
            }
        }
        
        retrieval_node_id = self.traceability_manager.create_node(
            chain_id=chain_id,
            node_type=NodeType.RETRIEVAL_OPERATION,
            content=retrieval_node_content,
            parent_ids=parent_node_ids,
            metadata=processing_metadata
        )
        
        return retrieval_result
    
    def batch_retrieve(
        self, 
        queries: List[Query], 
        chain_id: str
    ) -> List[RetrievalResult]:
        """
        Retrieve results for multiple queries with batch traceability.
        
        Args:
            queries: List of Query objects
            chain_id: Traceability chain ID
            
        Returns:
            List of RetrievalResult objects
        """
        results = []
        
        for query in queries:
            result = self.retrieve(query, chain_id)
            results.append(result)
        
        return results
    
    def get_retrieval_statistics(self, chain_id: str) -> Dict[str, Any]:
        """
        Get statistics about retrieval operations in a chain.
        
        Args:
            chain_id: Traceability chain ID
            
        Returns:
            Dictionary with retrieval statistics
        """
        chain = self.traceability_manager.get_chain(chain_id)
        if not chain:
            return {}
        
        retrieval_nodes = [
            node for node in chain.nodes.values() 
            if node.node_type == NodeType.RETRIEVAL_OPERATION
        ]
        
        if not retrieval_nodes:
            return {"total_retrievals": 0}
        
        total_results = 0
        total_execution_time = 0
        all_scores = []
        
        for node in retrieval_nodes:
            content = node.content
            total_results += len(content.get("retrieved_chunk_ids", []))
            total_execution_time += node.metadata.execution_time_ms
            all_scores.extend(content.get("similarity_scores", []))
        
        return {
            "total_retrievals": len(retrieval_nodes),
            "total_results_returned": total_results,
            "average_results_per_query": total_results / len(retrieval_nodes) if retrieval_nodes else 0,
            "total_execution_time_ms": total_execution_time,
            "average_execution_time_ms": total_execution_time / len(retrieval_nodes) if retrieval_nodes else 0,
            "score_statistics": {
                "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0,
                "max_score": max(all_scores) if all_scores else 0,
                "min_score": min(all_scores) if all_scores else 0,
                "total_scores": len(all_scores)
            }
        }