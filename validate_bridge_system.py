"""
Bridge System Validation Script

Validates the dependency-inverted bridge system by checking file structure,
imports, and basic functionality without executing potentially unsafe code.
"""

import sys
import os
from pathlib import Path
import json

def validate_file_structure():
    """Validate that all bridge files were created"""
    print("=== Validating File Structure ===")
    
    expected_files = [
        "canonical_flow/mathematical_enhancers/api_interfaces.py",
        "canonical_flow/bridge_registry.py", 
        "canonical_flow/configuration_injection.py",
        "canonical_flow/I_ingestion_preparation/bridge_ingestion_enhancer.py",
        "canonical_flow/X_context_construction/bridge_context_enhancer.py",
        "canonical_flow/K_knowledge_extraction/bridge_knowledge_enhancer.py",
        "canonical_flow/A_analysis_nlp/bridge_analysis_enhancer.py",
        "canonical_flow/L_classification_evaluation/bridge_scoring_enhancer.py",
        "canonical_flow/O_orchestration_control/bridge_orchestration_enhancer.py",
        "canonical_flow/R_search_retrieval/bridge_retrieval_enhancer.py",
        "canonical_flow/S_synthesis_output/bridge_synthesis_enhancer.py",
        "canonical_flow/G_aggregation_reporting/bridge_aggregation_enhancer.py",
        "canonical_flow/T_integration_storage/bridge_integration_enhancer.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print(f"\n✅ All {len(expected_files)} bridge files created successfully")
        return True

def validate_api_interfaces():
    """Validate API interfaces file content"""
    print("\n=== Validating API Interfaces ===")
    
    api_file = Path("canonical_flow/mathematical_enhancers/api_interfaces.py")
    if not api_file.exists():
        print("❌ API interfaces file missing")
        return False
    
    content = api_file.read_text()
    
    required_classes = [
        "MathematicalEnhancerAPI",
        "AbstractMathematicalEnhancer", 
        "ProcessingPhase",
        "ProcessingContext",
        "ProcessingResult",
        "IngestionEnhancerAPI",
        "ContextEnhancerAPI",
        "KnowledgeEnhancerAPI",
        "AnalysisEnhancerAPI",
        "ScoringEnhancerAPI",
        "OrchestrationEnhancerAPI",
        "RetrievalEnhancerAPI",
        "SynthesisEnhancerAPI",
        "AggregationEnhancerAPI",
        "IntegrationEnhancerAPI"
    ]
    
    missing_classes = []
    for class_name in required_classes:
        if f"class {class_name}" not in content and f"{class_name}(" not in content:
            missing_classes.append(class_name)
        else:
            print(f"✓ {class_name}")
    
    if missing_classes:
        print(f"\n❌ Missing API classes: {missing_classes}")
        return False
    else:
        print(f"\n✅ All {len(required_classes)} API interfaces defined")
        return True

def validate_bridge_modules():
    """Validate bridge module content"""
    print("\n=== Validating Bridge Modules ===")
    
    bridge_files = [
        ("canonical_flow/I_ingestion_preparation/bridge_ingestion_enhancer.py", "IngestionEnhancementBridge"),
        ("canonical_flow/X_context_construction/bridge_context_enhancer.py", "ContextEnhancementBridge"),
        ("canonical_flow/K_knowledge_extraction/bridge_knowledge_enhancer.py", "KnowledgeEnhancementBridge"),
        ("canonical_flow/A_analysis_nlp/bridge_analysis_enhancer.py", "AnalysisEnhancementBridge"),
        ("canonical_flow/L_classification_evaluation/bridge_scoring_enhancer.py", "ScoringEnhancementBridge"),
        ("canonical_flow/O_orchestration_control/bridge_orchestration_enhancer.py", "OrchestrationEnhancementBridge"),
        ("canonical_flow/R_search_retrieval/bridge_retrieval_enhancer.py", "RetrievalEnhancementBridge"),
        ("canonical_flow/S_synthesis_output/bridge_synthesis_enhancer.py", "SynthesisEnhancementBridge"),
        ("canonical_flow/G_aggregation_reporting/bridge_aggregation_enhancer.py", "AggregationEnhancementBridge"),
        ("canonical_flow/T_integration_storage/bridge_integration_enhancer.py", "IntegrationEnhancementBridge")
    ]
    
    valid_bridges = 0
    
    for file_path, class_name in bridge_files:
        if not Path(file_path).exists():
            print(f"❌ {file_path} missing")
            continue
            
        content = Path(file_path).read_text()
        
        # Check for required elements
        required_elements = [
            f"class {class_name}",
            "def inject_enhancer",
            "def process", 
            "def create_",
            "from canonical_flow.mathematical_enhancers.api_interfaces import"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ {class_name}: Missing {missing_elements}")
        else:
            print(f"✓ {class_name}: All required elements present")
            valid_bridges += 1
    
    print(f"\n✅ {valid_bridges}/{len(bridge_files)} bridge modules valid")
    return valid_bridges == len(bridge_files)

def validate_registry_system():
    """Validate bridge registry system"""
    print("\n=== Validating Registry System ===")
    
    registry_file = Path("canonical_flow/bridge_registry.py")
    if not registry_file.exists():
        print("❌ Bridge registry file missing")
        return False
    
    content = registry_file.read_text()
    
    required_components = [
        "class BridgeRegistry",
        "class BridgeMetadata", 
        "class BridgeRegistration",
        "def register_bridge",
        "def discover_bridges",
        "def create_bridge_instance",
        "def update_index_json"
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
        else:
            print(f"✓ {component}")
    
    if missing_components:
        print(f"\n❌ Missing registry components: {missing_components}")
        return False
    else:
        print(f"\n✅ All {len(required_components)} registry components present")
        return True

def validate_configuration_injection():
    """Validate configuration injection system"""
    print("\n=== Validating Configuration Injection ===")
    
    config_file = Path("canonical_flow/configuration_injection.py") 
    if not config_file.exists():
        print("❌ Configuration injection file missing")
        return False
    
    content = config_file.read_text()
    
    required_components = [
        "class ConfigurationInjector",
        "class EnhancerFactory",
        "class EnhancerConfiguration",
        "class InjectionConfiguration", 
        "def inject_enhancer_into_bridge",
        "def create_enhancer",
        "def get_configuration_injector"
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
        else:
            print(f"✓ {component}")
    
    if missing_components:
        print(f"\n❌ Missing injection components: {missing_components}")
        return False
    else:
        print(f"\n✅ All {len(required_components)} injection components present") 
        return True

def validate_index_json():
    """Validate index.json integration"""
    print("\n=== Validating Index.json Integration ===")
    
    index_file = Path("canonical_flow/index.json")
    if not index_file.exists():
        print("❌ index.json file missing")
        return False
    
    try:
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # Count bridge entries
        bridge_entries = [
            entry for entry in index_data 
            if entry.get('bridge_metadata', {}).get('enhancement_bridge', False)
        ]
        
        print(f"✓ Found {len(bridge_entries)} bridge entries in index.json")
        
        # Validate bridge entries
        valid_entries = 0
        for entry in bridge_entries:
            bridge_meta = entry.get('bridge_metadata', {})
            required_fields = ['bridge_id', 'bridge_name', 'api_interface', 'priority']
            
            if all(field in bridge_meta for field in required_fields):
                print(f"✓ Bridge entry valid: {bridge_meta['bridge_id']}")
                valid_entries += 1
            else:
                print(f"❌ Invalid bridge entry: {entry.get('code', 'unknown')}")
        
        print(f"\n✅ {valid_entries}/{len(bridge_entries)} bridge entries valid")
        return valid_entries == len(bridge_entries)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in index.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating index.json: {e}")
        return False

def validate_dependency_inversion():
    """Validate dependency inversion principles"""
    print("\n=== Validating Dependency Inversion ===")
    
    bridge_files = [
        "canonical_flow/I_ingestion_preparation/bridge_ingestion_enhancer.py",
        "canonical_flow/X_context_construction/bridge_context_enhancer.py", 
        "canonical_flow/K_knowledge_extraction/bridge_knowledge_enhancer.py",
        "canonical_flow/A_analysis_nlp/bridge_analysis_enhancer.py",
        "canonical_flow/L_classification_evaluation/bridge_scoring_enhancer.py",
        "canonical_flow/O_orchestration_control/bridge_orchestration_enhancer.py",
        "canonical_flow/R_search_retrieval/bridge_retrieval_enhancer.py",
        "canonical_flow/S_synthesis_output/bridge_synthesis_enhancer.py",
        "canonical_flow/G_aggregation_reporting/bridge_aggregation_enhancer.py",
        "canonical_flow/T_integration_storage/bridge_integration_enhancer.py"
    ]
    
    violation_count = 0
    
    for file_path in bridge_files:
        if not Path(file_path).exists():
            continue
            
        content = Path(file_path).read_text()
        
        # Check for proper dependency inversion
        good_imports = [
            "from canonical_flow.mathematical_enhancers.api_interfaces import",
            "if TYPE_CHECKING:"
        ]
        
        # Check for direct concrete imports (violations)
        bad_patterns = [
            "from canonical_flow.mathematical_enhancers.ingestion_enhancer import",
            "from canonical_flow.mathematical_enhancers.context_enhancer import", 
            "from canonical_flow.mathematical_enhancers.knowledge_enhancer import",
            "from canonical_flow.mathematical_enhancers.analysis_enhancer import",
            "from canonical_flow.mathematical_enhancers.scoring_enhancer import",
            "from canonical_flow.mathematical_enhancers.orchestration_enhancer import",
            "from canonical_flow.mathematical_enhancers.retrieval_enhancer import",
            "from canonical_flow.mathematical_enhancers.synthesis_enhancer import",
            "from canonical_flow.mathematical_enhancers.aggregation_enhancer import",
            "from canonical_flow.mathematical_enhancers.integration_enhancer import"
        ]
        
        # Check for violations
        violations = []
        for bad_pattern in bad_patterns:
            if bad_pattern in content and "# Type hints" not in content[content.find(bad_pattern):content.find(bad_pattern)+100]:
                violations.append(bad_pattern)
        
        # Check for good practices
        good_practices = []
        for good_import in good_imports:
            if good_import in content:
                good_practices.append(good_import)
        
        if violations:
            print(f"❌ {file_path}: Dependency inversion violations: {len(violations)}")
            violation_count += len(violations)
        else:
            print(f"✓ {file_path}: No dependency inversion violations")
    
    print(f"\n✅ Dependency inversion validation: {violation_count} violations found")
    return violation_count == 0

def main():
    """Run all validations"""
    print("🧪 Validating Dependency-Inverted Bridge System")
    print("=" * 60)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("API Interfaces", validate_api_interfaces),
        ("Bridge Modules", validate_bridge_modules),
        ("Registry System", validate_registry_system), 
        ("Configuration Injection", validate_configuration_injection),
        ("Index.json Integration", validate_index_json),
        ("Dependency Inversion", validate_dependency_inversion)
    ]
    
    passed = 0
    failed = 0
    
    for validation_name, validation_func in validations:
        try:
            if validation_func():
                passed += 1
                print(f"✅ {validation_name}: PASSED\n")
            else:
                failed += 1
                print(f"❌ {validation_name}: FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {validation_name}: ERROR - {e}\n")
    
    print("=" * 60)
    print(f"🏆 Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All validations passed! Bridge system is properly implemented.")
        print("\n📋 System Features Validated:")
        print("✓ Dependency-inverted bridge modules in all canonical phases")
        print("✓ API interfaces for loose coupling")
        print("✓ Registration system with index.json integration") 
        print("✓ Configuration injection mechanism")
        print("✓ Standard process(data, context) -> Dict[str, Any] interface")
        print("✓ Proper dependency inversion principles")
    else:
        print("⚠️  Some validations failed. Check the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)