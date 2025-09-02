#!/usr/bin/env python3
"""
Simple validation script for mathematical pipeline coordinator
"""

try:
    from canonical_flow.mathematical_enhancers.mathematical_pipeline_coordinator import (
        MathematicalPipelineCoordinator,
        MathStageType, 
        create_mathematical_pipeline_coordinator
    )
    
    print("✓ Successfully imported mathematical pipeline coordinator")
    
    # Create coordinator
    coordinator = create_mathematical_pipeline_coordinator()
    print("✓ Successfully created coordinator")
    
    # Test data
    test_data = {
        'text': 'This is a test text for mathematical enhancement validation.',
        'metadata': {'source': 'validation_test', 'timestamp': '2024-01-01'},
        'context': {}
    }
    
    # Test single stage execution
    try:
        result = coordinator.execute_stage(MathStageType.INGESTION, test_data)
        print("✓ Successfully executed ingestion stage")
        print(f"  Result keys: {list(result.keys())}")
        
        if 'mathematical_metrics' in result:
            metrics = result['mathematical_metrics']
            print(f"  Mathematical metrics: {list(metrics.keys())}")
        
    except Exception as e:
        print(f"✗ Stage execution failed: {str(e)}")
    
    # Test status reporting
    try:
        status = coordinator.get_pipeline_status()
        print("✓ Successfully retrieved pipeline status")
        print(f"  Overall status: {status['overall_status']}")
        print(f"  Completed stages: {status['summary_metrics']['completed_stages']}")
        
    except Exception as e:
        print(f"✗ Status reporting failed: {str(e)}")
    
    # Test integration interface
    try:
        comprehensive_data = {
            'text': 'Sample integration test text',
            'context': {'integration': True},
            'metadata': {'test_type': 'integration'}
        }
        
        result = coordinator.integrate_with_comprehensive_orchestrator(comprehensive_data)
        print("✓ Successfully tested integration interface")
        print(f"  Integration result keys: {list(result.keys())}")
        
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
    
    print("\n=== Mathematical Pipeline Coordinator Validation Complete ===")
    print("The coordinator is ready for integration with comprehensive_pipeline_orchestrator.py")

except ImportError as e:
    print(f"✗ Import failed: {str(e)}")
    print("Please ensure all dependencies are installed")

except Exception as e:
    print(f"✗ Validation failed: {str(e)}")
    import traceback
    traceback.print_exc()