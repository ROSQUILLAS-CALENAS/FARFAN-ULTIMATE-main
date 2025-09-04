#!/usr/bin/env python3
"""
Test script for anti_corruption_adapters module
"""

def test_module_import():
    """Test that the module imports correctly"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from tools.anti_corruption_adapters import (
            RetrievalResultDTO,
            AnalysisInputDTO,
            AnalysisResultDTO,
            RetrievalToAnalysisAdapter,
            AdapterFactory,
            install_import_guards
        )
        print("✓ Module imports successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dto_creation():
    """Test DTO creation and validation"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from tools.anti_corruption_adapters import RetrievalResultDTO, AnalysisInputDTO
        
        # Test RetrievalResultDTO
        retrieval_dto = RetrievalResultDTO(
            query="test query",
            documents=[{"id": "doc1", "content": "test content"}],
            scores=[0.8],
            retrieval_method="hybrid"
        )
        
        errors = retrieval_dto.validate()
        if errors:
            print(f"✗ RetrievalResultDTO validation failed: {errors}")
            return False
        
        # Test AnalysisInputDTO
        analysis_dto = AnalysisInputDTO(
            content="test content",
            document_metadata={"id": "doc1"},
            processing_context={"query": "test"},
            analysis_type="standard"
        )
        
        errors = analysis_dto.validate()
        if errors:
            print(f"✗ AnalysisInputDTO validation failed: {errors}")
            return False
            
        print("✓ DTO creation and validation works")
        return True
        
    except Exception as e:
        print(f"✗ DTO test failed: {e}")
        return False


def test_adapter_translation():
    """Test adapter translation functionality"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from tools.anti_corruption_adapters import (
            RetrievalResultDTO,
            RetrievalToAnalysisAdapter
        )
        
        # Create test retrieval result
        retrieval_result = RetrievalResultDTO(
            query="What is machine learning?",
            documents=[
                {"id": "doc1", "content": "Machine learning is a subset of AI", "title": "ML Basics"},
                {"id": "doc2", "content": "Deep learning uses neural networks", "title": "DL Guide"}
            ],
            scores=[0.9, 0.7],
            retrieval_method="hybrid"
        )
        
        # Create adapter and translate
        adapter = RetrievalToAnalysisAdapter()
        analysis_inputs = adapter.translate(retrieval_result)
        
        if len(analysis_inputs) != 2:
            print(f"✗ Expected 2 analysis inputs, got {len(analysis_inputs)}")
            return False
        
        # Validate first input
        first_input = analysis_inputs[0]
        if first_input.content != "Machine learning is a subset of AI":
            print(f"✗ Content mismatch: {first_input.content}")
            return False
            
        print("✓ Adapter translation works")
        return True
        
    except Exception as e:
        print(f"✗ Adapter test failed: {e}")
        return False


def test_import_guards():
    """Test import guard functionality"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from tools.anti_corruption_adapters import ImportGuard, BackwardDependencyError
        
        # Test backward dependency detection
        try:
            ImportGuard.check_import('analysis_nlp.test', 'retrieval_engine.hybrid_retriever')
            print("✗ Import guard should have blocked this import")
            return False
        except BackwardDependencyError:
            print("✓ Import guard correctly blocked backward dependency")
            
        # Test allowed import
        try:
            ImportGuard.check_import('retrieval_engine.test', 'some_other_module')
            print("✓ Import guard allows valid imports")
        except BackwardDependencyError:
            print("✗ Import guard blocked valid import")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Import guard test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing anti_corruption_adapters module...")
    print("=" * 50)
    
    tests = [
        test_module_import,
        test_dto_creation,
        test_adapter_translation,
        test_import_guards
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)