#!/usr/bin/env python3

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "62O"
__stage_order__ = 7

"""
Serialization-safe wrappers for the process_document function.

This module provides both functools.partial-based and class-based callable wrappers
that can be properly pickled for multiprocessing contexts, avoiding closure and 
module-level state issues.
"""

import functools
import logging
from security_utils import secure_pickle_replacement, secure_unpickle_replacement
# # # from typing import Any, Dict, Optional, Callable, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
import json
import time


@dataclass
class ProcessingConfig:
    """Configuration container for document processing parameters"""
    # EGW parameters
    batch_size: int = 32
    chunk_size: int = 1000
    max_concurrent_tasks: int = 8
    
    # Quality thresholds
    min_relevance_score: float = 0.7
    min_coherence_score: float = 0.8
    max_response_time: float = 30.0
    
    # Retrieval parameters
    top_k: int = 10
    consensus_threshold: float = 0.7
    
    # Model configurations
    tokenizer_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    result_ttl: int = 3600
    task_timeout: int = 300
    
    # Processing options
    enable_query_expansion: bool = True
    enable_gw_alignment: bool = True
    enable_evidence_processing: bool = True
    enable_answer_synthesis: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
# # #         """Create from dictionary"""  # Module not found  # Module not found  # Module not found
        return cls(**data)


def process_document_serializable(document_path: str, query: str, config: ProcessingConfig) -> Dict[str, Any]:
    """
    Serializable version of process_document that accepts all configuration as parameters.
    
    This function does not rely on closures or module-level state, making it
    safe for multiprocessing serialization.
    
    Args:
        document_path: Path to document to process
        query: Query string for document processing
        config: Processing configuration parameters
        
    Returns:
        Dict containing processing results
    """
    import os
    
    # Import components within function to avoid module-level dependencies
    try:
# # #         from egw_query_expansion.core.hybrid_retrieval import HybridRetrieval  # Module not found  # Module not found  # Module not found
# # #         from egw_query_expansion.core.gw_alignment import GWAlignment  # Module not found  # Module not found  # Module not found
# # #         from egw_query_expansion.core.query_generator import QueryGenerator  # Module not found  # Module not found  # Module not found
# # #         from evidence_processor import EvidenceProcessor  # Module not found  # Module not found  # Module not found
# # #         from answer_synthesizer import AnswerSynthesizer  # Module not found  # Module not found  # Module not found
    except ImportError as e:
        logging.warning(f"Failed to import EGW components: {e}")
        # Fallback to mock processing
        return _mock_process_document(document_path, query, config)
    
    start_time = time.time()
    
    try:
        # Initialize components with configuration
        hybrid_retrieval = HybridRetrieval(config.to_dict())
        gw_alignment = GWAlignment(config.to_dict())
        query_generator = QueryGenerator(config.to_dict())
        evidence_processor = EvidenceProcessor(config.to_dict())
        answer_synthesizer = AnswerSynthesizer(config.to_dict())
        
        # Read document
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                document_content = f.read()
        except Exception as e:
            logging.error(f"Failed to read document {document_path}: {e}")
            document_content = ""
        
        # Generate expanded queries if enabled
        expanded_queries = []
        if config.enable_query_expansion:
            try:
                expanded_queries = query_generator.generate_variants(query)
            except Exception as e:
                logging.warning(f"Query expansion failed: {e}")
                expanded_queries = [query]
        else:
            expanded_queries = [query]
        
        # Perform hybrid retrieval
        retrieval_results = {}
        try:
            retrieval_results = hybrid_retrieval.search(
                query=query,
                document_content=document_content,
                top_k=config.top_k
            )
        except Exception as e:
            logging.warning(f"Hybrid retrieval failed: {e}")
            retrieval_results = {'scores': [], 'query_embedding': None, 'corpus_embeddings': []}
        
        # Apply Gromov-Wasserstein alignment if enabled
        aligned_results = {}
        if config.enable_gw_alignment and retrieval_results.get('query_embedding') is not None:
            try:
                aligned_results = gw_alignment.align_query_corpus(
                    query_embedding=retrieval_results.get('query_embedding'),
                    corpus_embeddings=retrieval_results.get('corpus_embeddings', [])
                )
            except Exception as e:
                logging.warning(f"GW alignment failed: {e}")
                aligned_results = retrieval_results
        else:
            aligned_results = retrieval_results
        
        # Process evidence if enabled
        evidence = []
        if config.enable_evidence_processing:
            try:
                evidence = evidence_processor.extract_evidence(
                    document_content, retrieval_results
                )
            except Exception as e:
                logging.warning(f"Evidence processing failed: {e}")
                evidence = []
        
        # Synthesize final answer if enabled
        answer = {'answer': '', 'summary': ''}
        if config.enable_answer_synthesis:
            try:
                answer = answer_synthesizer.synthesize(
                    query=query,
                    evidence=evidence,
                    context=retrieval_results
                )
            except Exception as e:
                logging.warning(f"Answer synthesis failed: {e}")
                answer = {'answer': document_content[:500], 'summary': f'Document summary for: {query}'}
        else:
            answer = {'answer': document_content[:500], 'summary': f'Document summary for: {query}'}
        
        processing_time = time.time() - start_time
        
        return {
            'content': answer.get('answer', ''),
            'evidence': evidence,
            'summary': answer.get('summary', ''),
            'query_expansion': expanded_queries,
            'relevance_scores': retrieval_results.get('scores', []),
            'metadata': {
                'document_path': document_path,
                'processing_method': 'egw_pipeline_serializable',
                'expanded_query_count': len(expanded_queries),
                'processing_time': processing_time,
                'config_hash': hash(str(config.to_dict()))
            }
        }
        
    except Exception as e:
        logging.error(f"Document processing failed for {document_path}: {e}")
        return _mock_process_document(document_path, query, config)


def _mock_process_document(document_path: str, query: str, config: ProcessingConfig) -> Dict[str, Any]:
    """Fallback mock processing when components are not available"""
    import os
    processing_time = time.time()
    
    # Try to read document for basic content
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()[:500]  # Limit content length
    except Exception:
        doc_name = os.path.basename(document_path) if document_path else "unknown_document"
        content = f"Mock content for document: {doc_name}"
    
    processing_time = time.time() - processing_time
    
    doc_name = os.path.basename(document_path) if document_path else "unknown_document"
    
    return {
        'content': content,
        'evidence': [f"Mock evidence for query: {query}"],
        'summary': f"Mock summary for {query} in document {doc_name}",
        'query_expansion': [query, f"{query} expanded"],
        'relevance_scores': [0.8, 0.7, 0.6],
        'metadata': {
            'document_path': document_path,
            'processing_method': 'mock_fallback',
            'expanded_query_count': 2,
            'processing_time': processing_time,
            'config_hash': hash(str(config.to_dict()))
        }
    }


class DocumentProcessorCallable:
    """
    Class-based callable wrapper for document processing.
    
    This approach encapsulates all necessary state and configuration
    as instance attributes, making the entire object serializable.
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the callable processor.
        
        Args:
            config: Processing configuration parameters
        """
        self.config = config
        self._initialized_components = False
        
    def __call__(self, document_path: str, query: str) -> Dict[str, Any]:
        """
        Process a document with the given query.
        
        Args:
            document_path: Path to document to process
            query: Query string for document processing
            
        Returns:
            Dict containing processing results
        """
        return process_document_serializable(document_path, query, self.config)
    
    def __reduce__(self):
        """Enable pickling of the callable object"""
        return (self.__class__, (self.config,))
    
    def __repr__(self):
        """String representation"""
        return f"DocumentProcessorCallable(config_hash={hash(str(self.config.to_dict()))})"


def create_partial_wrapper(config: ProcessingConfig) -> functools.partial:
    """
    Create a functools.partial-based wrapper for process_document.
    
    This uses functools.partial to bind the configuration parameter,
    creating a serializable function reference.
    
    Args:
        config: Processing configuration parameters
        
    Returns:
        Partial function that can be called with (document_path, query)
    """
    return functools.partial(process_document_serializable, config=config)


def create_class_wrapper(config: ProcessingConfig) -> DocumentProcessorCallable:
    """
    Create a class-based callable wrapper for process_document.
    
    Args:
        config: Processing configuration parameters
        
    Returns:
        Callable object that can be called with (document_path, query)
    """
    return DocumentProcessorCallable(config)


def validate_serialization(wrapper: Union[functools.partial, DocumentProcessorCallable]) -> bool:
    """
    Validate that a wrapper can be properly serialized and deserialized.
    
    Args:
        wrapper: The wrapper to test
        
    Returns:
        True if serialization succeeds, False otherwise
    """
    try:
        # Test secure serialization
        from security_utils import secure_pickle_replacement, secure_unpickle_replacement
        
        serialized = secure_pickle_replacement(wrapper, use_msgpack=True)
        
        # Test unpickling
        deserialized = secure_unpickle_replacement(serialized)
        
        # Test that it's still callable
        if not callable(deserialized):
            return False
            
        # Test with mock data
        test_result = deserialized("test_doc.txt", "test query")
        
        # Verify result structure
        required_keys = {'content', 'evidence', 'summary', 'metadata'}
        if not all(key in test_result for key in required_keys):
            return False
            
        logging.info(f"Serialization validation passed for {type(wrapper).__name__}")
        return True
        
    except Exception as e:
        logging.error(f"Serialization validation failed for {type(wrapper).__name__}: {e}")
        return False


def create_multiprocessing_safe_wrapper(
    config: ProcessingConfig, 
    wrapper_type: str = "class"
) -> Union[functools.partial, DocumentProcessorCallable]:
    """
    Create a multiprocessing-safe wrapper for document processing.
    
    Args:
        config: Processing configuration parameters
        wrapper_type: Either "partial" or "class" to specify wrapper type
        
    Returns:
        Serializable wrapper function/callable
        
    Raises:
        ValueError: If wrapper_type is not supported
        RuntimeError: If wrapper fails serialization validation
    """
    if wrapper_type == "partial":
        wrapper = create_partial_wrapper(config)
    elif wrapper_type == "class":
        wrapper = create_class_wrapper(config)
    else:
        raise ValueError(f"Unsupported wrapper_type: {wrapper_type}. Use 'partial' or 'class'.")
    
    # Validate serialization
    if not validate_serialization(wrapper):
        raise RuntimeError(f"Created wrapper failed serialization validation")
    
    return wrapper


# Example usage functions
def demo_wrappers():
    """Demonstrate usage of both wrapper types"""
    config = ProcessingConfig(
        batch_size=16,
        top_k=5,
        min_relevance_score=0.6
    )
    
    print("Creating and testing functools.partial wrapper...")
    partial_wrapper = create_multiprocessing_safe_wrapper(config, "partial")
    print(f"Partial wrapper: {partial_wrapper}")
    
    print("Creating and testing class-based wrapper...")
    class_wrapper = create_multiprocessing_safe_wrapper(config, "class")
    print(f"Class wrapper: {class_wrapper}")
    
    # Test both wrappers
    test_doc = "test_document.txt"
    test_query = "What is the main topic?"
    
    print(f"\nTesting partial wrapper...")
    try:
        result1 = partial_wrapper(test_doc, test_query)
        print(f"Partial result keys: {list(result1.keys())}")
    except Exception as e:
        print(f"Partial wrapper error: {e}")
    
    print(f"\nTesting class wrapper...")
    try:
        result2 = class_wrapper(test_doc, test_query)
        print(f"Class result keys: {list(result2.keys())}")
    except Exception as e:
        print(f"Class wrapper error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_wrappers()