"""
BM25-based Text Retrieval Module

Implements traditional lexical search using BM25 scoring algorithm.
Provides standardized retrieval interface for query processing and document ranking.
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import json
import math
import os
import pickle
# # # from collections import Counter, defaultdict  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Tuple, Optional  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "R"
__code__ = "39R"
__stage_order__ = 6

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Mock numpy for basic functionality
    class MockNumpy:
        @staticmethod
        def zeros(shape):
            if isinstance(shape, int):
                return [0.0] * shape
            return [0.0] * shape[0]
        @staticmethod
        def argsort(arr):
            return sorted(range(len(arr)), key=lambda i: arr[i])
    np = MockNumpy()


class BM25Index:
    """BM25-based lexical retrieval index."""
    
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        index_path: Optional[str] = None
    ):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
            index_path: Path to load existing index from
        """
        self.k1 = k1
        self.b = b
        
        # Index components
        self.document_frequencies: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> [(doc_id, tf)]
        self.document_lengths: List[int] = []
        self.document_ids: List[str] = []
        self.document_texts: List[str] = []
        self.average_document_length: float = 0.0
        self.num_documents: int = 0
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization - can be enhanced with more sophisticated methods."""
        import re
        # Remove punctuation, convert to lowercase, split on whitespace
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index."""
        tokens = self.tokenize(text)
        
        # Store document
        internal_doc_id = len(self.document_ids)
        self.document_ids.append(doc_id)
        self.document_texts.append(text)
        self.document_lengths.append(len(tokens))
        
        # Build inverted index
        term_frequencies = Counter(tokens)
        for term, tf in term_frequencies.items():
            self.inverted_index[term].append((internal_doc_id, tf))
            if term not in self.document_frequencies:
                self.document_frequencies[term] = 0
            self.document_frequencies[term] += 1
        
        self.num_documents = len(self.document_ids)
        self.average_document_length = sum(self.document_lengths) / self.num_documents if self.num_documents > 0 else 0.0
    
    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """Add multiple documents to the index."""
        for doc_id, text in documents:
            self.add_document(doc_id, text)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25 scoring.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of ranked documents with scores
        """
        if self.num_documents == 0:
            return []
        
        query_terms = self.tokenize(query)
        if not query_terms:
            return []
        
        # Calculate BM25 scores for all documents
        scores = np.zeros(self.num_documents)
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
                
            # IDF calculation
            df = self.document_frequencies[term]
            idf = math.log((self.num_documents - df + 0.5) / (df + 0.5) + 1.0)
            
            # Calculate term contribution to each document
            for doc_id, tf in self.inverted_index[term]:
                doc_length = self.document_lengths[doc_id]
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
                
                scores[doc_id] += idf * (numerator / denominator)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return documents with non-zero scores
                results.append({
                    'doc_id': self.document_ids[idx],
                    'score': float(scores[idx]),
                    'text': self.document_texts[idx],
                    'method': 'bm25'
                })
        
        return results
    
    def save_index(self, path: str) -> None:
        """Save the index to disk."""
        index_data = {
            'k1': self.k1,
            'b': self.b,
            'document_frequencies': self.document_frequencies,
            'inverted_index': dict(self.inverted_index),  # Convert defaultdict to dict
            'document_lengths': self.document_lengths,
            'document_ids': self.document_ids,
            'document_texts': self.document_texts,
            'average_document_length': self.average_document_length,
            'num_documents': self.num_documents
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, path: str) -> None:
# # #         """Load the index from disk."""  # Module not found  # Module not found  # Module not found
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.k1 = index_data['k1']
        self.b = index_data['b']
        self.document_frequencies = index_data['document_frequencies']
        self.inverted_index = defaultdict(list, index_data['inverted_index'])
        self.document_lengths = index_data['document_lengths']
        self.document_ids = index_data['document_ids']
        self.document_texts = index_data['document_texts']
        self.average_document_length = index_data['average_document_length']
        self.num_documents = index_data['num_documents']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'num_documents': self.num_documents,
            'num_terms': len(self.document_frequencies),
            'average_document_length': self.average_document_length,
            'total_tokens': sum(self.document_lengths),
            'parameters': {'k1': self.k1, 'b': self.b}
        }


class LexicalRetriever:
    """Standard interface for BM25-based lexical retrieval."""
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        k1: float = 1.2,
        b: float = 0.75
    ):
        """
        Initialize lexical retriever.
        
        Args:
            index_path: Path to BM25 index file (bm25.idx)
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
        """
        self.index_path = index_path or "bm25.idx"
        self.index = BM25Index(k1=k1, b=b)
        
        # Try to load existing index
        if os.path.exists(self.index_path):
            try:
                self.index.load_index(self.index_path)
            except Exception as e:
# # #                 print(f"Warning: Could not load index from {self.index_path}: {e}")  # Module not found  # Module not found  # Module not found
    
    def build_index(self, documents: List[Tuple[str, str]]) -> None:
        """
# # #         Build BM25 index from documents.  # Module not found  # Module not found  # Module not found
        
        Args:
            documents: List of (doc_id, text) tuples
        """
        self.index.add_documents(documents)
        
        # Save index to disk
        self.index.save_index(self.index_path)
    
    def add_document(self, doc_id: str, text: str) -> None:
        """Add a single document to the index."""
        self.index.add_document(doc_id, text)
        self.index.save_index(self.index_path)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query using BM25 scoring.
        
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


def create_lexical_retriever(
    documents: Optional[List[Tuple[str, str]]] = None,
    index_path: str = "bm25.idx",
    k1: float = 1.2,
    b: float = 0.75
) -> LexicalRetriever:
    """
    Factory function to create and optionally build a lexical retriever.
    
    Args:
        documents: Optional documents to build index from
        index_path: Path for BM25 index file
        k1: BM25 parameter
        b: BM25 parameter
        
    Returns:
        Configured LexicalRetriever instance
    """
    retriever = LexicalRetriever(index_path=index_path, k1=k1, b=b)
    
    if documents:
        retriever.build_index(documents)
    
    return retriever


# Compatibility function for comprehensive_pipeline_orchestrator.py
def process(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process function for pipeline integration.
    
    Maintains compatibility with existing orchestrator while providing
    enhanced BM25 functionality.
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
    retriever = create_lexical_retriever(documents if documents else None)
    
    # Perform search if query is provided
    results = []
    if query and retriever.index.num_documents > 0:
        results = retriever.retrieve(query, top_k=ctx.get("top_k", 10))
    
    # Build response
    output = {}
    if isinstance(data, dict):
        output.update(data)
    
    output.update({
        "bm25_index": retriever.index,
        "lexical_results": results,
        "lexical_metrics": retriever.get_index_statistics(),
        "lexical_query_terms": retriever.index.tokenize(query) if query else [],
        "method": "BM25"
    })
    
    return output


if __name__ == "__main__":
    # Example usage
    sample_docs = [
        ("doc1", "The quick brown fox jumps over the lazy dog"),
        ("doc2", "Python is a programming language"),
        ("doc3", "Information retrieval using BM25 algorithm"),
        ("doc4", "Machine learning and natural language processing")
    ]
    
    retriever = create_lexical_retriever(documents=sample_docs)
    results = retriever.retrieve("programming language", top_k=3)
    
    print("BM25 Lexical Retrieval Results:")
    for result in results:
        print(f"Doc ID: {result['doc_id']}, Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:100]}...")
        print("-" * 50)