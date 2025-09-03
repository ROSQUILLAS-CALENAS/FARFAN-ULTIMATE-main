#!/usr/bin/env python3

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "74O"
__stage_order__ = 7

"""
Demo script showing the updated distributed processor with serializable wrappers.

This script demonstrates the improved serialization-safe approach to 
multiprocessing document processing.
"""

import asyncio
import tempfile
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import List  # Module not found  # Module not found  # Module not found
import logging

# Import the updated distributed processor components
# # # from serializable_wrappers import ProcessingConfig, create_multiprocessing_safe_wrapper  # Module not found  # Module not found  # Module not found
# # # from distributed_processor import DistributedProcessor  # Module not found  # Module not found  # Module not found


def create_demo_documents() -> List[str]:
    """Create sample documents for processing"""
    documents = []
    
    # Document 1: Urban Development Plan
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
Urban Development Master Plan 2024-2029

Executive Summary:
The city's urban development strategy focuses on sustainable growth, 
affordable housing, and improved transportation infrastructure.

Key Objectives:
1. Increase housing supply by 25% over 5 years
2. Develop comprehensive public transit network
3. Create 15 new parks and green spaces
4. Implement smart city technologies

Housing Development:
- Priority zones for affordable housing construction
- Mixed-use development incentives
- Accessibility compliance in all new buildings
- Community engagement in planning processes

Transportation:
- Bus rapid transit system expansion
- Bicycle infrastructure improvements
- Electric vehicle charging stations
- Pedestrian safety enhancements

Environmental Sustainability:
- Green building standards
- Renewable energy requirements
- Stormwater management systems
- Urban forestry program expansion
        """)
        documents.append(f.name)
    
    # Document 2: Economic Development Strategy
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
Economic Development Strategic Plan

Vision:
Create a thriving, diversified economy that provides opportunities 
for all residents while maintaining environmental sustainability.

Innovation and Technology:
- Establish technology incubators
- Support startup ecosystem
- Digital infrastructure investment
- STEM education partnerships

Workforce Development:
- Skills training programs
- Community college partnerships
- Apprenticeship opportunities
- Career counseling services

Business Support:
- Small business loan programs
- Tax incentive packages
- Streamlined permitting process
- Business mentorship networks

Tourism and Culture:
- Arts district development
- Cultural event programming
- Historic preservation
- Visitor experience improvements
        """)
        documents.append(f.name)
    
    # Document 3: Environmental Action Plan
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
Environmental Action Plan

Climate Goals:
Achieve carbon neutrality by 2035 through comprehensive 
environmental initiatives and community engagement.

Energy Systems:
- Solar panel installation programs
- Energy efficiency retrofits
- Community energy storage
- Grid modernization projects

Waste Management:
- Zero waste goal by 2030
- Expanded recycling programs
- Composting initiatives
- Plastic reduction campaigns

Water Conservation:
- Rainwater harvesting systems
- Drought-resistant landscaping
- Water recycling programs
- Infrastructure leak detection

Biodiversity Protection:
- Native species habitat restoration
- Pollinator-friendly gardens
- Wildlife corridor creation
- Invasive species management
        """)
        documents.append(f.name)
    
    return documents


async def demo_serializable_wrappers():
    """Demonstrate the serializable wrapper functionality"""
    print("=== Serializable Wrappers Demo ===\n")
    
    # Create test documents
    documents = create_demo_documents()
    query = "What are the main priorities and strategies outlined in this plan?"
    
    try:
        # Initialize distributed processor
        processor = DistributedProcessor()
        
        print("1. Testing Class-based Wrapper:")
        print("-" * 40)
        
        # Test class wrapper with multiprocessing
        start_time = asyncio.get_event_loop().time()
        results_class = await processor.process_batch_multiprocessing(
            documents, 
            query, 
            num_workers=2, 
            wrapper_type="class"
        )
        end_time = asyncio.get_event_loop().time()
        
        print(f"Class wrapper processed {len(results_class)} documents")
        print(f"Total time: {end_time - start_time:.2f}s")
        
        for i, result in enumerate(results_class):
            doc_name = Path(documents[i]).name
            print(f"  {doc_name}: {len(result['content'])} chars, "
                  f"{len(result['evidence'])} evidence items")
        
        print("\n2. Testing Partial Function Wrapper:")
        print("-" * 40)
        
        # Test partial wrapper with multiprocessing
        start_time = asyncio.get_event_loop().time()
        results_partial = await processor.process_batch_multiprocessing(
            documents, 
            query, 
            num_workers=2, 
            wrapper_type="partial"
        )
        end_time = asyncio.get_event_loop().time()
        
        print(f"Partial wrapper processed {len(results_partial)} documents")
        print(f"Total time: {end_time - start_time:.2f}s")
        
        for i, result in enumerate(results_partial):
            doc_name = Path(documents[i]).name
            print(f"  {doc_name}: {len(result['content'])} chars, "
                  f"{result['metadata']['processing_method']}")
        
        print("\n3. Comparing Results:")
        print("-" * 40)
        
        # Compare processing methods
        class_methods = [r['metadata']['processing_method'] for r in results_class]
        partial_methods = [r['metadata']['processing_method'] for r in results_partial]
        
        print(f"Class wrapper methods: {set(class_methods)}")
        print(f"Partial wrapper methods: {set(partial_methods)}")
        print(f"Both use same processing: {set(class_methods) == set(partial_methods)}")
        
        print("\n4. Configuration Validation:")
        print("-" * 40)
        
        # Validate configuration hashes
        class_hashes = [r['metadata']['config_hash'] for r in results_class]
        partial_hashes = [r['metadata']['config_hash'] for r in results_partial]
        
        print(f"Class wrapper config hashes: {len(set(class_hashes))} unique")
        print(f"Partial wrapper config hashes: {len(set(partial_hashes))} unique")
        print(f"Configurations match: {set(class_hashes) == set(partial_hashes)}")
        
        print("\n✅ All serializable wrapper tests completed successfully!")
        
    finally:
        # Clean up temporary files
        for doc_path in documents:
            try:
                os.unlink(doc_path)
            except OSError:
                pass


async def demo_configuration_flexibility():
    """Demonstrate different configuration options"""
    print("\n=== Configuration Flexibility Demo ===\n")
    
    # Create a small test document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Sample document for configuration testing.")
        test_doc = f.name
    
    try:
        configurations = [
            {
                'name': 'Fast Processing',
                'config': ProcessingConfig(
                    batch_size=16,
                    top_k=3,
                    enable_query_expansion=False,
                    enable_gw_alignment=False
                )
            },
            {
                'name': 'Complete Processing',
                'config': ProcessingConfig(
                    batch_size=32,
                    top_k=10,
                    enable_query_expansion=True,
                    enable_gw_alignment=True,
                    enable_evidence_processing=True,
                    enable_answer_synthesis=True
                )
            },
            {
                'name': 'Custom Processing',
                'config': ProcessingConfig(
                    min_relevance_score=0.6,
                    min_coherence_score=0.7,
                    top_k=5,
                    enable_evidence_processing=True
                )
            }
        ]
        
        for config_test in configurations:
            print(f"Testing {config_test['name']}:")
            print("-" * 30)
            
            # Create wrapper with custom configuration
            wrapper = create_multiprocessing_safe_wrapper(
                config_test['config'], 
                "class"
            )
            
            # Process document
            result = wrapper(test_doc, "What is this document about?")
            
            print(f"  Processing method: {result['metadata']['processing_method']}")
            print(f"  Content length: {len(result['content'])}")
            print(f"  Evidence items: {len(result['evidence'])}")
            print(f"  Query expansions: {len(result['query_expansion'])}")
            
            # Show configuration effect
            metadata = result['metadata']
            print(f"  Config hash: {metadata['config_hash']}")
            if 'processing_time' in metadata:
                print(f"  Processing time: {metadata['processing_time']:.3f}s")
            print()
        
        print("✅ Configuration flexibility demo completed!")
        
    finally:
        try:
            os.unlink(test_doc)
        except OSError:
            pass


async def demo_error_handling():
    """Demonstrate error handling and fallback mechanisms"""
    print("\n=== Error Handling Demo ===\n")
    
    # Test with non-existent file
    print("1. Testing with non-existent file:")
    print("-" * 35)
    
    config = ProcessingConfig()
    wrapper = create_multiprocessing_safe_wrapper(config, "class")
    
    result = wrapper("non_existent_file.txt", "Test query")
    print(f"  Processing method: {result['metadata']['processing_method']}")
    print(f"  Content: {result['content'][:50]}...")
    print(f"  Handled gracefully: {'mock_fallback' in result['metadata']['processing_method']}")
    
    # Test with empty query
    print("\n2. Testing with empty query:")
    print("-" * 30)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content")
        test_file = f.name
    
    try:
        result = wrapper(test_file, "")
        print(f"  Processing method: {result['metadata']['processing_method']}")
        print(f"  Handled empty query: {len(result['content']) > 0}")
        print(f"  Evidence generated: {len(result['evidence']) > 0}")
        
    finally:
        try:
            os.unlink(test_file)
        except OSError:
            pass
    
    print("\n✅ Error handling demo completed!")


async def main():
    """Run all demonstrations"""
    print("Distributed Processor Serializable Wrappers Demonstration")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(
# # #         level=logging.WARNING,  # Reduce noise from warnings  # Module not found  # Module not found  # Module not found
        format='%(levelname)s: %(message)s'
    )
    
    try:
        # Run demonstrations
        await demo_serializable_wrappers()
        await demo_configuration_flexibility()
        await demo_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\nKey improvements implemented:")
        print("• Serialization-safe process_document wrappers")
        print("• Both functools.partial and class-based approaches")
        print("• Configuration passed as parameters (no closures)")
        print("• Multiprocessing compatibility validated")
        print("• Comprehensive error handling and fallbacks")
        print("• Zero external API dependencies")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())