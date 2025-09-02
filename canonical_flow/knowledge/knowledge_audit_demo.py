"""
Demo script for Knowledge Audit System

This script demonstrates the audit functionality for all 6 K-stage components,
showing how execution metrics are tracked and written to the audit file.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from canonical_flow.knowledge.knowledge_audit_system import (
    get_audit_system, 
    component_06K_process, 
    component_07K_process
)


def run_audit_demo():
    """Demonstrate audit system with all K-stage components."""
    print("=" * 60)
    print("KNOWLEDGE AUDIT SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Get audit system
    audit_system = get_audit_system()
    
    # Sample data for processing
    sample_data = {
        "text": "Investment in technology leads to economic growth. Innovation drives market expansion.",
        "chunks": [
            {"text": "Technology investment increases productivity", "chunk_id": "chunk_001"},
            {"text": "Economic growth follows technological advancement", "chunk_id": "chunk_002"}
        ],
        "entities": [
            {"id": "tech_investment", "text": "Technology Investment", "type": "economic_factor"},
            {"id": "productivity", "text": "Productivity", "type": "outcome"}
        ],
        "relationships": [
            {"source": "tech_investment", "target": "productivity", "type": "direct_cause", "confidence": 0.85}
        ],
        "patterns": [
            "Investment leads to growth",
            "Technology drives innovation"
        ]
    }
    
    print("Testing all K-stage components with audit logging...\n")
    
    # Test 06K component (placeholder)
    print("1. Testing 06K Component (knowledge_component_06)")
    try:
        result_06K = component_06K_process(sample_data, {"batch_size": 16})
        print(f"   ✓ Component 06K completed successfully")
        print(f"   ✓ Status: {result_06K.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Component 06K failed: {e}")
    
    time.sleep(0.5)  # Small delay to show time differences
    
    # Test 07K component (placeholder) 
    print("\n2. Testing 07K Component (knowledge_component_07)")
    try:
        result_07K = component_07K_process(sample_data, {"confidence_threshold": 0.7})
        print(f"   ✓ Component 07K completed successfully")
        print(f"   ✓ Status: {result_07K.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Component 07K failed: {e}")
    
    time.sleep(0.5)
    
    # Test 08K component (advanced_knowledge_graph_builder) - placeholder for now
    print("\n3. Testing 08K Component (advanced_knowledge_graph_builder)")
    try:
        from canonical_flow.knowledge.knowledge_audit_system import create_placeholder_component
        process_08K = create_placeholder_component("08K", "advanced_knowledge_graph_builder")
        result_08K = process_08K(sample_data, {"enable_parallel": False})
        print(f"   ✓ Component 08K completed successfully")
        print(f"   ✓ Status: {result_08K.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Component 08K failed: {e}")
    
    time.sleep(0.5)
    
    # Test 09K component (causal_graph) - placeholder for now
    print("\n4. Testing 09K Component (causal_graph)")
    try:
        from canonical_flow.knowledge.knowledge_audit_system import create_placeholder_component
        process_09K = create_placeholder_component("09K", "causal_graph")
        result_09K = process_09K(sample_data)
        print(f"   ✓ Component 09K completed successfully") 
        print(f"   ✓ Status: {result_09K.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Component 09K failed: {e}")
    
    time.sleep(0.5)
    
    # Test 10K component (causal_dnp_framework) - placeholder for now
    print("\n5. Testing 10K Component (causal_dnp_framework)")
    try:
        from canonical_flow.knowledge.knowledge_audit_system import create_placeholder_component
        process_10K = create_placeholder_component("10K", "causal_dnp_framework")
        result_10K = process_10K(sample_data)
        print(f"   ✓ Component 10K completed successfully")
        print(f"   ✓ Status: {result_10K.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Component 10K failed: {e}")
    
    time.sleep(0.5)
    
    # Test 11K component (embedding_builder) - placeholder for now
    print("\n6. Testing 11K Component (embedding_builder)")
    try:
        from canonical_flow.knowledge.knowledge_audit_system import create_placeholder_component
        process_11K = create_placeholder_component("11K", "embedding_builder")
        result_11K = process_11K(sample_data)
        print(f"   ✓ Component 11K completed successfully")
        print(f"   ✓ Status: {result_11K.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Component 11K failed: {e}")
    
    time.sleep(0.5)
    
    # Test 12K component (embedding_generator) - placeholder for now
    print("\n7. Testing 12K Component (embedding_generator)")
    try:
        from canonical_flow.knowledge.knowledge_audit_system import create_placeholder_component
        process_12K = create_placeholder_component("12K", "embedding_generator")
        result_12K = process_12K(sample_data)
        print(f"   ✓ Component 12K completed successfully")
        print(f"   ✓ Status: {result_12K.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Component 12K failed: {e}")
    
    print("\n" + "=" * 60)
    print("AUDIT SYSTEM ANALYSIS")
    print("=" * 60)
    
    # Display audit statistics
    print("\nComponent Execution Statistics:")
    for component_code in ["06K", "07K", "08K", "09K", "10K", "11K", "12K"]:
        stats = audit_system.get_component_stats(component_code)
        if "message" not in stats:  # If we have actual stats
            print(f"\n{component_code} ({audit_system.COMPONENT_MAPPING[component_code]}):")
            print(f"   Executions: {stats['total_executions']}")
            print(f"   Success Rate: {stats['success_rate']:.1%}")
            print(f"   Avg Duration: {stats['performance']['avg_duration_ms']:.2f}ms")
            print(f"   Peak Memory: {stats['performance']['peak_memory_mb']:.2f}MB")
    
    # Validate and write audit file
    print("\nWriting audit file...")
    try:
        validation = audit_system.validate_audit_schema()
        if validation["valid"]:
            audit_system.write_audit_file()
            print(f"✓ Audit file written successfully to: {audit_system.audit_file_path}")
            print(f"✓ Total audit entries: {len(audit_system.audit_entries)}")
        else:
            print(f"✗ Audit validation failed: {validation['errors']}")
            
        if validation["warnings"]:
            print(f"⚠ Warnings: {len(validation['warnings'])}")
            
    except Exception as e:
        print(f"✗ Failed to write audit file: {e}")
    
    print("\n" + "=" * 60)
    print("AUDIT DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_audit_demo()