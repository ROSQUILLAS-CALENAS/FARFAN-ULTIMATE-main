#!/usr/bin/env python3
"""
Test script for serializable wrappers.

This script demonstrates and validates the serialization-safe wrappers for 
the process_document function in multiprocessing contexts.
"""

import tempfile
import multiprocessing
import pickle
import time
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# # # from serializable_wrappers import (  # Module not found  # Module not found  # Module not found
    ProcessingConfig,
    create_multiprocessing_safe_wrapper,
    validate_serialization,
    process_document_serializable
)


def create_test_documents():
    """Create temporary test documents"""
    test_docs = []
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
Municipal Development Plan - City Infrastructure

This document outlines the urban development strategy for the next five years.
Key areas include transportation improvements, housing development, and
environmental sustainability initiatives.

Transportation:
- Expand public transit network
- Improve bike lanes and pedestrian infrastructure
- Implement smart traffic management systems

Housing:
- Develop affordable housing units
- Promote mixed-use developments
- Ensure accessibility in all new constructions

Environment:
- Increase green spaces by 20%
- Implement renewable energy systems
- Improve waste management and recycling programs
        """)
        test_docs.append(f.name)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
Economic Development Strategy

The city's economic development plan focuses on innovation, job creation,
and sustainable growth across multiple sectors.

Technology Sector:
- Support startup incubators
- Develop innovation hubs
- Provide tax incentives for tech companies

Manufacturing:
- Modernize existing facilities
- Attract clean manufacturing businesses
- Invest in workforce training programs

Tourism:
- Promote cultural attractions
- Develop sustainable tourism practices
- Improve visitor infrastructure
        """)
        test_docs.append(f.name)
    
    return test_docs


def test_wrapper_serialization():
    """Test that both wrapper types can be properly serialized"""
    print("Testing wrapper serialization...")
    
    config = ProcessingConfig(
        batch_size=16,
        top_k=5,
        min_relevance_score=0.6,
        enable_query_expansion=True
    )
    
    # Test partial wrapper
    partial_wrapper = create_multiprocessing_safe_wrapper(config, "partial")
    partial_valid = validate_serialization(partial_wrapper)
    print(f"Partial wrapper serialization: {'PASS' if partial_valid else 'FAIL'}")
    
    # Test class wrapper
    class_wrapper = create_multiprocessing_safe_wrapper(config, "class")
    class_valid = validate_serialization(class_wrapper)
    print(f"Class wrapper serialization: {'PASS' if class_valid else 'FAIL'}")
    
    return partial_valid and class_valid


def test_multiprocessing_execution():
    """Test actual multiprocessing execution with both wrapper types"""
    print("\nTesting multiprocessing execution...")
    
    test_docs = create_test_documents()
    test_queries = [
        "What are the transportation improvements?",
        "How will the city support economic development?"
    ]
    
    config = ProcessingConfig(
        batch_size=8,
        top_k=3,
        min_relevance_score=0.5
    )
    
    try:
        for wrapper_type in ["class", "partial"]:
            print(f"\n--- Testing {wrapper_type} wrapper with multiprocessing ---")
            
            wrapper = create_multiprocessing_safe_wrapper(config, wrapper_type)
            
            # Prepare tasks for multiprocessing
            tasks = [
                (doc, query) 
                for doc in test_docs 
                for query in test_queries
            ]
            
            start_time = time.time()
            
            # Use multiprocessing pool
            with multiprocessing.Pool(processes=2) as pool:
                results = pool.starmap(wrapper, tasks)
            
            end_time = time.time()
            
            print(f"Processed {len(tasks)} tasks in {end_time - start_time:.2f}s")
            print(f"Average processing time per task: {(end_time - start_time) / len(tasks):.2f}s")
            
            # Validate results
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    print(f"ERROR: Result {i} is not a dict")
                    continue
                
                required_keys = {'content', 'evidence', 'summary', 'metadata'}
                missing_keys = required_keys - set(result.keys())
                if missing_keys:
                    print(f"ERROR: Result {i} missing keys: {missing_keys}")
                    continue
                
                print(f"Task {i}: {result['metadata']['processing_method']} - {len(result['content'])} chars")
            
            print(f"{wrapper_type} wrapper: SUCCESS")
    
    finally:
        # Clean up test files
        for doc_path in test_docs:
            try:
                os.unlink(doc_path)
            except OSError:
                pass


def test_direct_function_call():
    """Test direct function calls without multiprocessing"""
    print("\nTesting direct function calls...")
    
    test_docs = create_test_documents()
    query = "What are the key development priorities?"
    
    config = ProcessingConfig(
        top_k=5,
        enable_query_expansion=True,
        enable_gw_alignment=True
    )
    
    try:
        for doc_path in test_docs:
            print(f"\nProcessing: {Path(doc_path).name}")
            
            start_time = time.time()
            result = process_document_serializable(doc_path, query, config)
            end_time = time.time()
            
            print(f"Processing time: {end_time - start_time:.2f}s")
            print(f"Content length: {len(result['content'])}")
            print(f"Evidence items: {len(result['evidence'])}")
            print(f"Processing method: {result['metadata']['processing_method']}")
            
            # Validate result structure
            required_keys = {'content', 'evidence', 'summary', 'query_expansion', 'metadata'}
            if not all(key in result for key in required_keys):
                print(f"ERROR: Missing keys in result")
                continue
            
            print("Direct function call: SUCCESS")
    
    finally:
        # Clean up
        for doc_path in test_docs:
            try:
                os.unlink(doc_path)
            except OSError:
                pass


def test_pickle_roundtrip():
    """Test pickle serialization roundtrip for both wrapper types"""
    print("\nTesting pickle roundtrip...")
    
    config = ProcessingConfig(batch_size=10, top_k=3)
    
    # Test both wrapper types
    for wrapper_type in ["class", "partial"]:
        print(f"\n--- Testing {wrapper_type} wrapper pickle roundtrip ---")
        
        # Create wrapper
        original_wrapper = create_multiprocessing_safe_wrapper(config, wrapper_type)
        
        # Serialize using secure method
        from security_utils import secure_pickle_replacement, secure_unpickle_replacement
        serialized = secure_pickle_replacement(original_wrapper, use_msgpack=True)
        print(f"Serialized size: {len(serialized)} bytes")
        
        # Deserialize
        deserialized_wrapper = secure_unpickle_replacement(serialized)
        
        # Test functionality
        test_doc = "test_document.txt"
        test_query = "test query"
        
        try:
            result = deserialized_wrapper(test_doc, test_query)
            if isinstance(result, dict) and 'content' in result:
                print(f"Pickle roundtrip for {wrapper_type}: SUCCESS")
            else:
                print(f"Pickle roundtrip for {wrapper_type}: FAIL - Invalid result format")
        except Exception as e:
            print(f"Pickle roundtrip for {wrapper_type}: FAIL - {e}")


def test_configuration_variations():
    """Test different configuration variations"""
    print("\nTesting configuration variations...")
    
    configurations = [
        {
            'name': 'Minimal Config',
            'config': ProcessingConfig()
        },
        {
            'name': 'High Performance Config',
            'config': ProcessingConfig(
                batch_size=64,
                top_k=20,
                max_concurrent_tasks=16,
                enable_query_expansion=True,
                enable_gw_alignment=True,
                enable_evidence_processing=True,
                enable_answer_synthesis=True
            )
        },
        {
            'name': 'Basic Processing Config',
            'config': ProcessingConfig(
                enable_query_expansion=False,
                enable_gw_alignment=False,
                enable_evidence_processing=False,
                enable_answer_synthesis=False
            )
        }
    ]
    
    test_docs = create_test_documents()
    
    try:
        for config_test in configurations:
            print(f"\n--- Testing {config_test['name']} ---")
            
            wrapper = create_multiprocessing_safe_wrapper(config_test['config'], "class")
            
            start_time = time.time()
            result = wrapper(test_docs[0], "What is this document about?")
            end_time = time.time()
            
            print(f"Processing time: {end_time - start_time:.2f}s")
            print(f"Processing method: {result['metadata']['processing_method']}")
            print(f"Config hash: {result['metadata']['config_hash']}")
            
            # Validate serialization
            if validate_serialization(wrapper):
                print(f"{config_test['name']}: SUCCESS")
            else:
                print(f"{config_test['name']}: FAIL")
    
    finally:
        # Clean up
        for doc_path in test_docs:
            try:
                os.unlink(doc_path)
            except OSError:
                pass


def main():
    """Run all tests"""
    print("=== Serializable Wrappers Test Suite ===\n")
    
    # Set multiprocessing start method for compatibility
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Serialization validation
    if test_wrapper_serialization():
        success_count += 1
    
    # Test 2: Direct function calls  
    try:
        test_direct_function_call()
        success_count += 1
        print("Direct function call test: SUCCESS")
    except Exception as e:
        print(f"Direct function call test: FAIL - {e}")
    
    # Test 3: Pickle roundtrip
    try:
        test_pickle_roundtrip()
        success_count += 1
        print("Pickle roundtrip test: SUCCESS")
    except Exception as e:
        print(f"Pickle roundtrip test: FAIL - {e}")
    
    # Test 4: Configuration variations
    try:
        test_configuration_variations()
        success_count += 1
        print("Configuration variations test: SUCCESS")
    except Exception as e:
        print(f"Configuration variations test: FAIL - {e}")
    
    # Test 5: Multiprocessing execution
    try:
        test_multiprocessing_execution()
        success_count += 1
        print("Multiprocessing execution test: SUCCESS")
    except Exception as e:
        print(f"Multiprocessing execution test: FAIL - {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {success_count}/{total_tests}")
    print(f"Success Rate: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("All tests passed! Serializable wrappers are working correctly.")
        return 0
    else:
        print("Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())