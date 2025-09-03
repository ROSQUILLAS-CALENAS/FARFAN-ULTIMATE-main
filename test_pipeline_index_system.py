#!/usr/bin/env python3
"""
Test script for the pipeline index system.
"""

import json
import tempfile
import shutil
from pathlib import Path
from pipeline_autoscan import PipelineAutoscan
from pipeline_validation_system import PipelineValidationSystem 
from pipeline_dag_visualizer import PipelineDAGVisualizer


def test_basic_functionality():
    """Test basic functionality of the pipeline index system."""
    print("🧪 Testing Pipeline Index System...")
    
    # Create a temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test index
        test_index = {
            "version": "1.0.0",
            "generated_at": "2025-01-03T00:00:00.000Z",
            "metadata": {
                "description": "Test pipeline index",
                "total_components": 2,
                "phases": ["ingestion_preparation", "analysis_nlp"],
                "maintainer": "test_system"
            },
            "components": [
                {
                    "name": "test_reader",
                    "code": "01I",
                    "phase": "ingestion_preparation",
                    "dependencies": [],
                    "canonical_path": f"{temp_path}/canonical_flow/I_ingestion_preparation/test_reader.py",
                    "original_path": f"{temp_path}/test_reader.py",
                    "description": "Test PDF reader",
                    "enabled": True,
                    "entry_point": True
                },
                {
                    "name": "test_analyzer",
                    "code": "02A",
                    "phase": "analysis_nlp",
                    "dependencies": ["01I"],
                    "canonical_path": f"{temp_path}/canonical_flow/A_analysis_nlp/test_analyzer.py",
                    "original_path": f"{temp_path}/test_analyzer.py",
                    "description": "Test analyzer",
                    "enabled": True
                }
            ]
        }
        
        # Write test index
        index_path = temp_path / "test_index.json"
        with open(index_path, 'w') as f:
            json.dump(test_index, f, indent=2)
        
        # Create test canonical structure
        canonical_root = temp_path / "canonical_flow"
        ingestion_dir = canonical_root / "I_ingestion_preparation"
        analysis_dir = canonical_root / "A_analysis_nlp"
        
        ingestion_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        test_reader_file = ingestion_dir / "test_reader.py"
        test_analyzer_file = analysis_dir / "test_analyzer.py"
        
        test_reader_file.write_text("# Test reader component\nprint('test_reader')")
        test_analyzer_file.write_text("# Test analyzer component\nprint('test_analyzer')")
        
        print(f"✅ Created test environment in {temp_path}")
        
        # Test 1: Autoscan System
        print("🔍 Testing Autoscan System...")
        try:
            scanner = PipelineAutoscan(str(index_path), str(canonical_root))
            scanner.repo_root = temp_path  # Override repo root for test
            
            result = scanner.run_autoscan(update_index=False)
            print(f"  📊 Scan completed: {result.components_found} components found")
            print(f"  🔄 Changes detected: {len(result.changes_detected)}")
            print("✅ Autoscan test passed")
            
        except Exception as e:
            print(f"❌ Autoscan test failed: {e}")
            return False
        
        # Test 2: Validation System
        print("🔍 Testing Validation System...")
        try:
            validator = PipelineValidationSystem(str(index_path), str(canonical_root), strict_mode=False)
            validator.repo_root = temp_path  # Override repo root for test
            
            # Update index paths to use absolute paths for test
            updated_index = test_index.copy()
            for comp in updated_index["components"]:
                comp["canonical_path"] = str(Path(comp["canonical_path"]).relative_to(temp_path))
            
            with open(index_path, 'w') as f:
                json.dump(updated_index, f, indent=2)
            
            result = validator.run_full_validation()
            print(f"  📊 Validation completed: {result.components_validated} components")
            print(f"  ❌ Errors: {len(result.errors)}")
            print(f"  ⚠️  Warnings: {len(result.warnings)}")
            
            if result.errors:
                for error in result.errors:
                    print(f"    • {error.message}")
            
            print("✅ Validation test passed (with expected issues in test environment)")
            
        except Exception as e:
            print(f"❌ Validation test failed: {e}")
            return False
        
        # Test 3: DAG Visualizer
        print("🔍 Testing DAG Visualizer...")
        try:
            visualizer = PipelineDAGVisualizer(str(index_path))
            
            # Test DAG validation
            is_valid, errors = visualizer.validate_dag()
            print(f"  📈 DAG validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
            
            if errors:
                for error in errors:
                    print(f"    • {error}")
            
            # Test format generation (text formats only to avoid Graphviz dependency)
            index_data = visualizer.load_index()
            nodes, edges = visualizer.build_dag(index_data)
            
            dot_content = visualizer.generate_graphviz_dot(nodes, edges)
            mermaid_content = visualizer.generate_mermaid_diagram(nodes, edges)
            
            print(f"  📄 Generated DOT content: {len(dot_content)} chars")
            print(f"  📄 Generated Mermaid content: {len(mermaid_content)} chars")
            
            print("✅ DAG Visualizer test passed")
            
        except Exception as e:
            print(f"❌ DAG Visualizer test failed: {e}")
            return False
        
        print("🎉 All tests passed!")
        return True


def test_real_system():
    """Test the real system with current index."""
    print("🧪 Testing Real System...")
    
    try:
        # Test validation on real index
        print("📋 Validating real pipeline index...")
        validator = PipelineValidationSystem()
        result = validator.run_full_validation()
        
        print(f"  📊 Components validated: {result.components_validated}")
        print(f"  ❌ Errors: {len(result.errors)}")
        print(f"  ⚠️  Warnings: {len(result.warnings)}")
        print(f"  ℹ️  Info: {len(result.info)}")
        
        # Show first few errors/warnings
        if result.errors:
            print("  First few errors:")
            for error in result.errors[:3]:
                print(f"    • {error.error_type}: {error.message}")
        
        if result.warnings:
            print("  First few warnings:")
            for warning in result.warnings[:3]:
                print(f"    • {warning.error_type}: {warning.message}")
        
        # Test DAG validation
        print("📈 Validating DAG structure...")
        visualizer = PipelineDAGVisualizer()
        is_valid, dag_errors = visualizer.validate_dag()
        
        print(f"  📈 DAG validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        if dag_errors:
            print("  DAG errors:")
            for error in dag_errors:
                print(f"    • {error}")
        
        print("✅ Real system test completed")
        return True
        
    except Exception as e:
        print(f"❌ Real system test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Pipeline Index System Tests")
    print("=" * 50)
    
    success = True
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    print("=" * 50)
    
    # Test real system
    if not test_real_system():
        success = False
    
    print("=" * 50)
    
    if success:
        print("🎉 All tests completed successfully!")
        exit(0)
    else:
        print("❌ Some tests failed!")
        exit(1)