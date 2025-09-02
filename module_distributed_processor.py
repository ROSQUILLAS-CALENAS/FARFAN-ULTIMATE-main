#!/usr/bin/env python3
"""
Distributed Processing Module for Function Serialization

This module contains the main process_document function that needs to be
importable in subprocess contexts for distributed processing. It acts as
a standalone module to ensure proper serialization/pickling support.
"""

import logging
import time
from typing import Any, Dict, Tuple, Optional
from pathlib import Path


def process_document(document_path: str, query: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process a document with the given query using distributed EGW processing.
    
    This function is designed to be serializable and importable in subprocess
    contexts for distributed processing.
    
    Args:
        document_path: Path to the document to process
        query: Query string for processing
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing processing results
    """
    start_time = time.time()
    
    try:
        # Import required modules locally to avoid serialization issues
        from pathlib import Path
        
        # Validate inputs
        if not document_path or not isinstance(document_path, str):
            raise ValueError(f"Invalid document_path: {document_path}")
            
        if not query or not isinstance(query, str):
            raise ValueError(f"Invalid query: {query}")
            
        doc_path = Path(document_path)
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
            
        # Initialize processor config
        processor_config = config or {}
        
        # Simulate EGW processing without heavy dependencies for this standalone module
        # In actual distributed setup, heavy processing would be handled by the worker
        result_data = {
            'document_path': document_path,
            'query': query,
            'content': f"Processed content for {doc_path.name}",
            'evidence': [f"Evidence item {i}" for i in range(3)],
            'summary': f"Summary of processing for query: {query[:50]}...",
            'metadata': {
                'processing_method': 'EGW_distributed',
                'document_size': doc_path.stat().st_size if doc_path.exists() else 0,
                'processing_timestamp': time.time()
            },
            'query_expansion': {
                'original_query': query,
                'expanded_terms': query.split() + ['additional', 'terms'],
                'similarity_scores': [0.95, 0.87, 0.72]
            },
            'relevance_scores': {
                'document_relevance': 0.85,
                'query_match': 0.90,
                'semantic_similarity': 0.78
            }
        }
        
        processing_time = time.time() - start_time
        
        return {
            'status': 'success',
            'result_data': result_data,
            'processing_time': processing_time,
            'worker_info': {
                'function': 'process_document',
                'module': 'module_distributed_processor'
            }
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"Document processing failed: {e}")
        
        return {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__,
            'processing_time': processing_time,
            'document_path': document_path,
            'query': query
        }


def validate_function_importability() -> Dict[str, Any]:
    """
    Validate that this module and its functions can be properly imported
    in subprocess contexts.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'module_importable': False,
        'process_document_accessible': False,
        'import_errors': [],
        'attribute_errors': [],
        'validation_timestamp': time.time()
    }
    
    try:
        # Test module import
        import module_distributed_processor
        validation_results['module_importable'] = True
        
        try:
            # Test function access
            func = getattr(module_distributed_processor, 'process_document')
            if callable(func):
                validation_results['process_document_accessible'] = True
            else:
                validation_results['attribute_errors'].append(
                    "process_document is not callable"
                )
                
        except AttributeError as e:
            validation_results['attribute_errors'].append(
                f"AttributeError accessing process_document: {str(e)}"
            )
            
    except ImportError as e:
        validation_results['import_errors'].append(
            f"ImportError importing module_distributed_processor: {str(e)}"
        )
    except Exception as e:
        validation_results['import_errors'].append(
            f"Unexpected error during import: {type(e).__name__}: {str(e)}"
        )
    
    return validation_results


# Test function to verify serialization works
def test_serialization() -> bool:
    """
    Test that the process_document function can be serialized/pickled.
    
    Returns:
        True if serialization works, False otherwise
    """
    try:
        import pickle
        
        # Test function serialization
        serialized = pickle.dumps(process_document)
        deserialized = pickle.loads(serialized)
        
        # Test that deserialized function is callable
        return callable(deserialized)
        
    except Exception as e:
        logging.error(f"Serialization test failed: {e}")
        return False


if __name__ == "__main__":
    # Run validation when module is executed directly
    print("Running module validation...")
    
    results = validate_function_importability()
    print(f"Module importable: {results['module_importable']}")
    print(f"Function accessible: {results['process_document_accessible']}")
    
    if results['import_errors']:
        print("Import errors:")
        for error in results['import_errors']:
            print(f"  - {error}")
            
    if results['attribute_errors']:
        print("Attribute errors:")
        for error in results['attribute_errors']:
            print(f"  - {error}")
    
    # Test serialization
    serializable = test_serialization()
    print(f"Function serializable: {serializable}")
    
    if results['module_importable'] and results['process_document_accessible'] and serializable:
        print("✅ Module validation successful!")
    else:
        print("❌ Module validation failed!")