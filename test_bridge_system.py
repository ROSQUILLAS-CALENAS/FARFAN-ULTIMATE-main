"""
Test Suite for Dependency-Inverted Bridge System

Tests the bridge modules, registration system, and configuration injection
mechanisms to ensure proper dependency inversion and loose coupling.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add canonical_flow to path for imports
sys.path.insert(0, str(Path(__file__).parent / "canonical_flow"))

from canonical_flow.bridge_registry import get_bridge_registry, initialize_bridge_system
from canonical_flow.configuration_injection import get_configuration_injector
from canonical_flow.mathematical_enhancers.api_interfaces import ProcessingPhase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_interfaces():
    """Test mathematical enhancer API interfaces"""
    print("\n=== Testing API Interfaces ===")
    
    try:
        from canonical_flow.mathematical_enhancers.api_interfaces import (
            MathematicalEnhancerAPI,
            ProcessingContext,
            ProcessingResult,
            ProcessingPhase
        )
        
        # Test ProcessingContext creation
        context = ProcessingContext(
            stage_id="test_stage",
            phase=ProcessingPhase.INGESTION_PREPARATION,
            pipeline_state={},
            metadata={},
            upstream_results={}
        )
        
        print(f"✓ ProcessingContext created: {context.stage_id}")
        print(f"✓ Processing phases available: {len(ProcessingPhase)}")
        
        return True
        
    except Exception as e:
        print(f"✗ API interface test failed: {e}")
        return False


def test_bridge_modules():
    """Test bridge module imports and instantiation"""
    print("\n=== Testing Bridge Modules ===")
    
    bridge_modules = [
        ("canonical_flow.I_ingestion_preparation.bridge_ingestion_enhancer", "IngestionEnhancementBridge"),
        ("canonical_flow.X_context_construction.bridge_context_enhancer", "ContextEnhancementBridge"),
        ("canonical_flow.K_knowledge_extraction.bridge_knowledge_enhancer", "KnowledgeEnhancementBridge"),
        ("canonical_flow.A_analysis_nlp.bridge_analysis_enhancer", "AnalysisEnhancementBridge"),
        ("canonical_flow.L_classification_evaluation.bridge_scoring_enhancer", "ScoringEnhancementBridge"),
        ("canonical_flow.O_orchestration_control.bridge_orchestration_enhancer", "OrchestrationEnhancementBridge"),
        ("canonical_flow.R_search_retrieval.bridge_retrieval_enhancer", "RetrievalEnhancementBridge"),
        ("canonical_flow.S_synthesis_output.bridge_synthesis_enhancer", "SynthesisEnhancementBridge"),
        ("canonical_flow.G_aggregation_reporting.bridge_aggregation_enhancer", "AggregationEnhancementBridge"),
        ("canonical_flow.T_integration_storage.bridge_integration_enhancer", "IntegrationEnhancementBridge"),
    ]
    
    success_count = 0
    
    for module_path, class_name in bridge_modules:
        try:
            # Import module
            import importlib
            module = importlib.import_module(module_path)
            
            # Get bridge class
            bridge_class = getattr(module, class_name)
            
            # Create instance without enhancer
            bridge = bridge_class()
            
            # Test standard interface
            if hasattr(bridge, 'process'):
                # Test process function with dummy data
                dummy_data = {"test": "data"}
                dummy_context = {
                    "stage_id": "test",
                    "pipeline_state": {},
                    "metadata": {},
                    "upstream_results": {}
                }
                
                result = bridge.process(dummy_data, dummy_context)
                
                # Verify result structure
                required_keys = ["success", "data", "metadata", "performance_metrics"]
                if all(key in result for key in required_keys):
                    print(f"✓ {class_name}: Import, instantiation, and process test successful")
                    success_count += 1
                else:
                    print(f"✗ {class_name}: Invalid result structure: {list(result.keys())}")
            else:
                print(f"✗ {class_name}: Missing process method")
                
        except Exception as e:
            print(f"✗ {class_name}: Failed - {e}")
    
    print(f"Bridge modules test: {success_count}/{len(bridge_modules)} successful")
    return success_count == len(bridge_modules)


def test_bridge_registry():
    """Test bridge registry system"""
    print("\n=== Testing Bridge Registry ===")
    
    try:
        # Initialize registry
        registry = get_bridge_registry()
        
        # Test discovery
        print(f"✓ Registry initialized with {len(registry.registered_bridges)} bridges")
        
        # Test phase mapping
        for phase in ProcessingPhase:
            bridges = registry.get_bridges_for_phase(phase)
            print(f"✓ Phase {phase.value}: {len(bridges)} bridges")
        
        # Test bridge instantiation
        sample_bridge_id = "ingestion_enhancement_bridge"
        if sample_bridge_id in registry.registered_bridges:
            bridge_instance = registry.create_bridge_instance(sample_bridge_id)
            if bridge_instance:
                print(f"✓ Successfully created bridge instance: {sample_bridge_id}")
            else:
                print(f"✗ Failed to create bridge instance: {sample_bridge_id}")
                return False
        
        # Test registry info
        info = registry.get_registry_info()
        print(f"✓ Registry info: {info['total_bridges']} total, {info['enabled_bridges']} enabled")
        
        return True
        
    except Exception as e:
        print(f"✗ Bridge registry test failed: {e}")
        return False


def test_configuration_injection():
    """Test configuration injection system"""
    print("\n=== Testing Configuration Injection ===")
    
    try:
        # Get configuration injector
        injector = get_configuration_injector()
        
        # Test injector initialization
        summary = injector.get_injection_summary()
        print(f"✓ Configuration injector initialized")
        print(f"✓ Available enhancers: {summary['total_enhancers']}")
        print(f"✓ Injection mappings: {summary['total_injections']}")
        
        # Test bridge creation with registry
        registry = get_bridge_registry()
        bridge_id = "ingestion_enhancement_bridge"
        
        if bridge_id in registry.registered_bridges:
            # Create bridge instance
            bridge = registry.create_bridge_instance(bridge_id)
            
            if bridge:
                # Test enhancer injection
                injection_success = injector.inject_enhancer_into_bridge(
                    bridge, bridge_id, "math_stage1_ingestion"
                )
                
                if injection_success:
                    print(f"✓ Successfully injected enhancer into {bridge_id}")
                    
                    # Test bridge with injected enhancer
                    dummy_data = {"test": "data"}
                    dummy_context = {
                        "stage_id": "test",
                        "pipeline_state": {},
                        "metadata": {},
                        "upstream_results": {}
                    }
                    
                    result = bridge.process(dummy_data, dummy_context)
                    print(f"✓ Bridge processing with enhancer: {result['success']}")
                    
                else:
                    print(f"✗ Failed to inject enhancer into {bridge_id}")
                    return False
            else:
                print(f"✗ Failed to create bridge instance: {bridge_id}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration injection test failed: {e}")
        return False


def test_index_json_integration():
    """Test index.json integration"""
    print("\n=== Testing Index.json Integration ===")
    
    try:
        # Read index.json
        index_path = Path("canonical_flow/index.json")
        if not index_path.exists():
            print("✗ index.json not found")
            return False
        
        import json
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Count bridge entries
        bridge_entries = [
            entry for entry in index_data 
            if entry.get('bridge_metadata', {}).get('enhancement_bridge', False)
        ]
        
        print(f"✓ Found {len(bridge_entries)} bridge entries in index.json")
        
        # Verify bridge metadata
        for entry in bridge_entries:
            bridge_meta = entry.get('bridge_metadata', {})
            required_fields = ['bridge_id', 'bridge_name', 'api_interface', 'priority']
            
            if all(field in bridge_meta for field in required_fields):
                print(f"✓ Bridge entry valid: {bridge_meta['bridge_id']}")
            else:
                print(f"✗ Invalid bridge entry: {entry.get('code', 'unknown')}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Index.json integration test failed: {e}")
        return False


def test_end_to_end_pipeline():
    """Test end-to-end pipeline with bridges"""
    print("\n=== Testing End-to-End Pipeline ===")
    
    try:
        # Initialize systems
        registry = initialize_bridge_system()
        injector = get_configuration_injector()
        
        # Test pipeline flow through multiple phases
        phases_to_test = [
            ProcessingPhase.INGESTION_PREPARATION,
            ProcessingPhase.CONTEXT_CONSTRUCTION,
            ProcessingPhase.KNOWLEDGE_EXTRACTION,
            ProcessingPhase.ANALYSIS_NLP
        ]
        
        pipeline_data = {"document": "Test document content", "metadata": {}}
        pipeline_context = {
            "stage_id": "pipeline_test",
            "pipeline_state": {"phase_results": {}},
            "metadata": {"test": True},
            "upstream_results": {}
        }
        
        for phase in phases_to_test:
            # Get bridges for phase
            bridges = registry.get_bridges_for_phase(phase)
            
            if bridges:
                bridge_registration = bridges[0]  # Use first bridge
                bridge_id = bridge_registration.metadata.bridge_id
                
                # Create and configure bridge
                bridge = registry.create_bridge_instance(bridge_id)
                if bridge:
                    # Inject enhancer
                    injector.inject_enhancer_into_bridge(bridge, bridge_id)
                    
                    # Process data
                    result = bridge.process(pipeline_data, pipeline_context)
                    
                    if result['success']:
                        print(f"✓ Phase {phase.value}: Processing successful")
                        pipeline_data = result['data']  # Pass data to next phase
                        pipeline_context['upstream_results'][phase.value] = result
                    else:
                        print(f"✗ Phase {phase.value}: Processing failed")
                        return False
                else:
                    print(f"✗ Phase {phase.value}: Bridge creation failed")
                    return False
            else:
                print(f"! Phase {phase.value}: No bridges available (expected for some phases)")
        
        print("✓ End-to-end pipeline test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ End-to-end pipeline test failed: {e}")
        return False


def run_all_tests():
    """Run all test suites"""
    print("🧪 Starting Dependency-Inverted Bridge System Tests")
    print("=" * 60)
    
    tests = [
        ("API Interfaces", test_api_interfaces),
        ("Bridge Modules", test_bridge_modules),
        ("Bridge Registry", test_bridge_registry),
        ("Configuration Injection", test_configuration_injection),
        ("Index.json Integration", test_index_json_integration),
        ("End-to-End Pipeline", test_end_to_end_pipeline)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🏆 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Bridge system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)