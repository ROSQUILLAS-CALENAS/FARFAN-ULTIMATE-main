"""
Simple test to validate the phases directory structure and APIs.
"""

import os
import sys
import importlib.util
from pathlib import Path


def test_phases_structure():
    """Test that all phase directories exist with correct structure."""
    print("Testing phases directory structure...")
    
    phases_dir = Path("phases")
    expected_phases = ['I', 'X', 'K', 'A', 'L', 'R', 'O', 'G', 'T', 'S']
    
    # Check main phases directory exists
    assert phases_dir.exists(), "phases/ directory not found"
    assert (phases_dir / "__init__.py").exists(), "phases/__init__.py not found"
    
    # Check each phase subdirectory
    for phase in expected_phases:
        phase_dir = phases_dir / phase
        phase_init = phase_dir / "__init__.py"
        
        assert phase_dir.exists(), f"Phase {phase} directory not found"
        assert phase_init.exists(), f"Phase {phase}/__init__.py not found"
        
        print(f"  ✓ Phase {phase}: structure OK")
    
    print("All phase directories have correct structure")


def test_phase_apis():
    """Test that phase APIs can be imported and have required interfaces."""
    print("\nTesting phase API imports...")
    
    sys.path.insert(0, str(Path.cwd()))
    
    phase_mappings = {
        'I': 'Ingestion',
        'X': 'ContextConstruction',
        'K': 'KnowledgeExtraction',
        'A': 'Analysis', 
        'L': 'Classification',
        'R': 'Retrieval',
        'O': 'Orchestration',
        'G': 'Aggregation',
        'T': 'Integration',
        'S': 'Synthesis'
    }
    
    for phase, name in phase_mappings.items():
        try:
            # Import phase module
            module_spec = importlib.util.spec_from_file_location(
                f"phases_{phase}", f"phases/{phase}/__init__.py"
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            
            # Check required exports exist
            required_exports = [
                f'{name}Data',
                f'{name}Context',
                f'{name}Processor', 
                f'{name}Phase'
            ]
            
            for export in required_exports:
                assert hasattr(module, export), f"Phase {phase} missing {export}"
            
            # Check __all__ declaration
            assert hasattr(module, '__all__'), f"Phase {phase} missing __all__"
            assert len(module.__all__) == 4, f"Phase {phase} __all__ should have 4 items"
            
            print(f"  ✓ Phase {phase}: API OK")
            
        except Exception as e:
            print(f"  ✗ Phase {phase}: {e}")
            raise


def test_validation_system():
    """Test that the validation system can be imported."""
    print("\nTesting validation system...")
    
    try:
        spec = importlib.util.spec_from_file_location(
            "phases_validation", "phases/_validation.py"
        )
        validation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validation_module)
        
        # Test validator can be created
        validator = validation_module.PhaseAccessValidator()
        assert validator is not None
        
        # Test basic validation
        result = validator.validate_import("phases.I", "IngestionPhase", "test_module")
        assert isinstance(result, bool)
        
        print("  ✓ Validation system: OK")
        
    except Exception as e:
        print(f"  ✗ Validation system: {e}")
        raise


def test_compatibility_layer():
    """Test that the compatibility layer can be imported."""
    print("\nTesting compatibility layer...")
    
    try:
        spec = importlib.util.spec_from_file_location(
            "phases_compatibility", "phases/backward_compatibility.py"
        )
        compatibility_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compatibility_module)
        
        # Test adapter can be created
        adapter = compatibility_module.CanonicalToPhaseAdapter()
        assert adapter is not None
        
        # Test basic mapping
        mapping = adapter.get_phase_equivalent("canonical_flow.I_ingestion_preparation")
        assert mapping == "phases.I"
        
        print("  ✓ Compatibility layer: OK")
        
    except Exception as e:
        print(f"  ✗ Compatibility layer: {e}")
        raise


def test_import_linter_config():
    """Test that import linter configuration exists."""
    print("\nTesting import linter configuration...")
    
    config_files = [".importlinter", "pyproject.toml"]
    config_found = False
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"  ✓ Found {config_file}")
            config_found = True
    
    assert config_found, "No import linter configuration found"


if __name__ == "__main__":
    try:
        test_phases_structure()
        test_phase_apis()
        test_validation_system()
        test_compatibility_layer()
        test_import_linter_config()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("Phase architecture is correctly implemented.")
        print("="*50)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)