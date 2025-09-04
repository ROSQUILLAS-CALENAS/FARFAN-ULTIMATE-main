"""
Demo: Anti-Corruption Adapter Usage

Demonstrates how to use the adapter modules to break circular dependencies
and enforce proper separation between retrieval and analysis phases.
"""

import logging
import json
from typing import Any, Dict

# Import adapter components
from .retrieval_analysis_adapter import RetrievalAnalysisAdapter
from .import_blocker import ImportBlocker
from .lineage_tracker import LineageTracker


def demo_adapter_translation():
    """Demonstrate adapter translation between retrieval and analysis"""
    
    print("=== Anti-Corruption Adapter Demo ===\n")
    
    # Initialize adapter
    adapter = RetrievalAnalysisAdapter("demo_adapter")
    
    # Simulate retrieval output
    sample_retrieval_output = {
        'query_id': 'demo_query_001',
        'retrieved_chunks': [
            {
                'chunk_id': 'chunk_1',
                'content': 'This is sample document content about DNP compliance.',
                'source': 'document_1.pdf',
                'metadata': {'page': 1, 'confidence': 0.95}
            },
            {
                'chunk_id': 'chunk_2', 
                'content': 'Additional content about territorial development.',
                'source': 'document_2.pdf',
                'metadata': {'page': 2, 'confidence': 0.87}
            }
        ],
        'similarity_scores': [0.95, 0.87],
        'retrieval_metadata': {
            'algorithm': 'hybrid_retrieval',
            'total_candidates': 1000,
            'execution_time_ms': 150
        }
    }
    
    # Translate retrieval output to analysis input
    print("1. Translating retrieval output to analysis input format...")
    
    analysis_input = adapter.translate_retrieval_to_analysis(
        retrieval_output=sample_retrieval_output,
        context={'user_query': 'Check DNP compliance', 'priority': 'high'}
    )
    
    print("✓ Translation completed successfully")
    print(f"   Chunks translated: {len(analysis_input['document_chunks'])}")
    print(f"   Context keys: {list(analysis_input['context'].keys())}")
    print(f"   Processing metadata: {list(analysis_input['processing_metadata'].keys())}\n")
    
    # Validate translation
    print("2. Validating translation...")
    
    validation = adapter.validate_translation(sample_retrieval_output, analysis_input)
    print(f"✓ Validation passed: {validation['validation_passed']}")
    print(f"   Input schema valid: {validation['input_schema_valid']}")
    print(f"   Output schema valid: {validation['output_schema_valid']}")
    print(f"   Data preserved: {validation['data_preserved']}\n")
    
    # Show adapter statistics
    print("3. Adapter statistics:")
    stats = adapter.get_adapter_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    return adapter


def demo_import_blocking():
    """Demonstrate import blocking functionality"""
    
    print("=== Import Blocking Demo ===\n")
    
    # Initialize lineage tracker and import blocker
    lineage_tracker = LineageTracker()
    import_blocker = ImportBlocker(lineage_tracker)
    
    print("1. Import blocker installed and active")
    
    # Test import validation
    print("\n2. Testing import restrictions:")
    
    test_cases = [
        ('analysis_module', 'retrieval_engine'),
        ('analysis_module', 'hybrid_retrieval'),
        ('retrieval_module', 'analysis_nlp_orchestrator'),
        ('other_module', 'some_library'),  # Should be allowed
    ]
    
    for calling_module, imported_module in test_cases:
        allowed = import_blocker.is_import_allowed(calling_module, imported_module)
        status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
        print(f"   {calling_module} -> {imported_module}: {status}")
    
    # Show violation summary
    print("\n3. Import violation summary:")
    summary = import_blocker.get_violation_summary()
    for key, value in summary.items():
        if key != 'restricted_patterns':
            print(f"   {key}: {value}")
    
    print()
    return import_blocker


def demo_lineage_tracking():
    """Demonstrate lineage tracking functionality"""
    
    print("=== Lineage Tracking Demo ===\n")
    
    lineage_tracker = LineageTracker()
    
    # Track some component operations
    print("1. Tracking component operations...")
    
    operations = [
        {
            'component_id': 'retrieval_engine',
            'operation_type': 'document_retrieval',
            'input_schema': 'query_input',
            'output_schema': 'retrieval_output',
            'dependencies': ['vector_index', 'lexical_index']
        },
        {
            'component_id': 'analysis_processor',
            'operation_type': 'dnp_analysis',
            'input_schema': 'analysis_input',
            'output_schema': 'analysis_output', 
            'dependencies': ['retrieval_analysis_adapter']
        },
        {
            'component_id': 'report_generator',
            'operation_type': 'report_generation',
            'input_schema': 'analysis_output',
            'output_schema': 'final_report',
            'dependencies': ['analysis_processor']
        }
    ]
    
    for op in operations:
        lineage_tracker.track_component_operation(**op)
        print(f"   ✓ Tracked {op['component_id']}: {op['operation_type']}")
    
    # Check for circular dependencies
    print("\n2. Checking for circular dependencies...")
    cycles = lineage_tracker.detect_circular_dependencies()
    if cycles:
        print(f"   ⚠️ Found {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles):
            print(f"     Cycle {i+1}: {' -> '.join(cycle)}")
    else:
        print("   ✓ No circular dependencies detected")
    
    # Get lineage info for a component
    print("\n3. Component lineage example:")
    lineage_info = lineage_tracker.get_component_lineage('analysis_processor')
    print(f"   Component: {lineage_info['component_id']}")
    print(f"   Direct dependencies: {lineage_info['direct_dependencies']}")
    print(f"   Dependents: {lineage_info['dependents']}")
    print(f"   Health score: {lineage_info['lineage_health']['health_score']}")
    
    # System summary
    print("\n4. System lineage summary:")
    summary = lineage_tracker.get_system_lineage_summary()
    for key, value in summary.items():
        if key not in ['cycles_detected', 'timestamp']:
            print(f"   {key}: {value}")
    
    print()
    return lineage_tracker


def demo_full_pipeline():
    """Demonstrate full pipeline with all adapter components"""
    
    print("=== Full Pipeline Demo ===\n")
    
    # Initialize all components
    lineage_tracker = LineageTracker()
    import_blocker = ImportBlocker(lineage_tracker)
    adapter = RetrievalAnalysisAdapter("pipeline_adapter")
    
    # Simulate a full retrieval -> analysis pipeline
    print("1. Simulating full pipeline...")
    
    # Mock retrieval phase
    retrieval_output = {
        'query_id': 'pipeline_query_001',
        'retrieved_chunks': [
            {
                'chunk_id': 'doc1_chunk1',
                'content': 'Budget allocation follows DNP guidelines for territorial development.',
                'source': 'pdt_document.pdf',
                'metadata': {'section': 'budget', 'page': 5}
            }
        ],
        'similarity_scores': [0.92],
        'retrieval_metadata': {'method': 'hybrid', 'time_ms': 200}
    }
    
    # Track retrieval operation
    lineage_tracker.track_component_operation(
        component_id='retrieval_engine',
        operation_type='document_search',
        input_schema='search_query',
        output_schema='retrieval_result',
        dependencies=['vector_store', 'text_index']
    )
    
    # Translate through adapter
    analysis_input = adapter.translate_retrieval_to_analysis(retrieval_output)
    
    # Track analysis operation
    lineage_tracker.track_component_operation(
        component_id='dnp_analyzer',
        operation_type='compliance_check',
        input_schema='analysis_input',
        output_schema='compliance_report',
        dependencies=['pipeline_adapter']
    )
    
    print("   ✓ Retrieval phase completed")
    print("   ✓ Adapter translation completed")
    print("   ✓ Analysis phase started")
    
    # Show pipeline health
    print("\n2. Pipeline health check:")
    system_summary = lineage_tracker.get_system_lineage_summary()
    adapter_stats = adapter.get_adapter_statistics()
    
    print(f"   System health: {system_summary['system_health']}")
    print(f"   Components tracked: {system_summary['total_components']}")
    print(f"   Adapter success rate: {adapter_stats['success_rate']:.2%}")
    print(f"   Schema mismatches: {adapter_stats['schema_mismatches']}")
    
    print("\n✓ Pipeline demo completed successfully!")
    
    return {
        'lineage_tracker': lineage_tracker,
        'import_blocker': import_blocker,
        'adapter': adapter
    }


def run_demo():
    """Run the adapter demo - separate function for testing"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Run all demos
        print("Starting Anti-Corruption Adapter Demos...\n")
        
        adapter = demo_adapter_translation()
        import_blocker = demo_import_blocking()
        lineage_tracker = demo_lineage_tracking()
        components = demo_full_pipeline()
        
        print("\n🎉 All demos completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("- ✓ Clean separation between retrieval and analysis")
        print("- ✓ Schema mismatch detection and logging")
        print("- ✓ Import blocking prevents circular dependencies")
        print("- ✓ Lineage tracking for dependency monitoring")
        print("- ✓ Graceful handling of data translation failures")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = run_demo()
    if not success:
        exit(1)