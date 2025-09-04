#!/usr/bin/env python3
"""
Demo script showing anti_corruption_adapters functionality
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_basic_translation():
    """Demonstrate basic DTO translation"""
    print("=== Basic Translation Demo ===")
    
    from tools.anti_corruption_adapters import (
        RetrievalResultDTO,
        RetrievalToAnalysisAdapter
    )
    
    # Create sample retrieval result
    retrieval_result = RetrievalResultDTO(
        query="What are the benefits of renewable energy?",
        documents=[
            {
                "id": "doc1",
                "title": "Solar Energy Benefits", 
                "content": "Solar energy provides clean, renewable power that reduces carbon emissions and energy costs.",
                "source": "energy_database"
            },
            {
                "id": "doc2", 
                "title": "Wind Power Advantages",
                "content": "Wind power is sustainable and can generate electricity without producing greenhouse gases.",
                "source": "research_papers"
            }
        ],
        scores=[0.92, 0.85],
        retrieval_method="hybrid_search",
        metadata={"search_time": 0.15, "total_candidates": 1000}
    )
    
    print(f"Created retrieval result with {len(retrieval_result.documents)} documents")
    
    # Translate using adapter
    adapter = RetrievalToAnalysisAdapter()
    analysis_inputs = adapter.translate(retrieval_result)
    
    print(f"Translated to {len(analysis_inputs)} analysis inputs")
    
    # Show first translation
    first_input = analysis_inputs[0]
    print(f"\nFirst Analysis Input:")
    print(f"  Content: {first_input.content[:60]}...")
    print(f"  Analysis Type: {first_input.analysis_type}")
    print(f"  Priority: {first_input.priority}")
    print(f"  Document ID: {first_input.document_metadata['document_id']}")
    

def demo_import_guards():
    """Demonstrate import guard functionality"""
    print("\n=== Import Guard Demo ===")
    
    from tools.anti_corruption_adapters import ImportGuard, BackwardDependencyError
    
    # Test blocked import
    try:
        ImportGuard.check_import('analysis_nlp.question_analyzer', 'retrieval_engine.hybrid_retriever')
        print("✗ Should have blocked this import!")
    except BackwardDependencyError as e:
        print(f"✓ Correctly blocked: {e.importing_module} -> {e.blocked_module}")
        print(f"  Reason: {e.reason}")
    
    # Test allowed import  
    try:
        ImportGuard.check_import('retrieval_engine.vector_index', 'numpy')
        print("✓ Allowed valid import: retrieval_engine.vector_index -> numpy")
    except BackwardDependencyError as e:
        print(f"✗ Should have allowed this import: {e}")


def demo_schema_validation():
    """Demonstrate schema validation and error handling"""
    print("\n=== Schema Validation Demo ===")
    
    from tools.anti_corruption_adapters import (
        RetrievalResultDTO,
        AnalysisInputDTO,
        get_schema_violations,
        clear_schema_violations
    )
    
    # Clear previous violations
    clear_schema_violations()
    
    # Test valid DTO
    valid_dto = RetrievalResultDTO(
        query="test query",
        documents=[{"id": "1", "content": "test"}],
        scores=[0.8],
        retrieval_method="test"
    )
    
    errors = valid_dto.validate()
    print(f"Valid DTO errors: {errors}")
    
    # Test invalid DTO
    try:
        invalid_dto = AnalysisInputDTO(
            content="",  # Invalid: empty content
            document_metadata="not_a_dict",  # Invalid: should be dict
            processing_context={},
            analysis_type="",  # Invalid: empty analysis type
            priority=-1  # Invalid: negative priority
        )
        
        errors = invalid_dto.validate()
        print(f"Invalid DTO errors: {errors}")
        
    except Exception as e:
        print(f"Error creating invalid DTO: {e}")
    
    # Show any schema violations
    violations = get_schema_violations()
    print(f"Total schema violations logged: {len(violations)}")


def demo_adapter_factory():
    """Demonstrate adapter factory usage"""
    print("\n=== Adapter Factory Demo ===")
    
    from tools.anti_corruption_adapters import AdapterFactory
    
    # Create retrieval->analysis adapter
    adapter = AdapterFactory.create_adapter('retrieval', 'analysis')
    print(f"Created adapter: {type(adapter).__name__}")
    
    # Try creating analysis->retrieval adapter (should warn)
    try:
        reverse_adapter = AdapterFactory.create_adapter('analysis', 'retrieval') 
        print(f"Created reverse adapter: {type(reverse_adapter).__name__}")
    except Exception as e:
        print(f"Error creating reverse adapter: {e}")


def main():
    """Run all demos"""
    print("Anti-Corruption Adapters Demo")
    print("=" * 50)
    
    try:
        demo_basic_translation()
        demo_import_guards() 
        demo_schema_validation()
        demo_adapter_factory()
        
        print("\n✓ All demos completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)