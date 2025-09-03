#!/usr/bin/env python3
"""
Validation script for L-stage test suite
"""

import sys
import json
import hashlib
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "canonical_flow"))

def test_imports():
    """Test that all required imports work."""
    try:
# # #         from canonical_flow.A_analysis_nlp.question_analyzer import (  # Module not found  # Module not found  # Module not found
            get_decalogo_question_registry,
            DecalogoQuestionRegistry
        )
# # #         from canonical_flow.L_classification_evaluation.decalogo_scoring_system import (  # Module not found  # Module not found  # Module not found
            ScoringSystem,
            QuestionResponse,
            DimensionScore,
            PointScore
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_registry_basic_functionality():
    """Test basic registry functionality."""
    try:
# # #         from canonical_flow.A_analysis_nlp.question_analyzer import get_decalogo_question_registry  # Module not found  # Module not found  # Module not found
        
        # Get registry
        registry = get_decalogo_question_registry()
        questions = registry.get_all_questions()
        
        # Validate count
        if len(questions) != 470:
            print(f"✗ Expected 470 questions, got {len(questions)}")
            return False
        
        # Validate points
        points = set(q.point_number for q in questions)
        if points != set(range(1, 11)):
            print(f"✗ Expected points 1-10, got {points}")
            return False
        
        # Validate dimensions
        dimensions = set(q.dimension_code for q in questions)
        expected_dims = {"DE1", "DE2", "DE3", "DE4", "GEN"}
        if not expected_dims.issubset(dimensions):
            print(f"✗ Missing dimensions: {expected_dims - dimensions}")
            return False
        
        print("✓ Registry basic functionality validated")
        return True
    except Exception as e:
        print(f"✗ Registry test error: {e}")
        return False

def test_scoring_system_basic():
    """Test basic scoring system functionality."""
    try:
# # #         from canonical_flow.L_classification_evaluation.decalogo_scoring_system import ScoringSystem  # Module not found  # Module not found  # Module not found
        
        # Create system
        scoring_system = ScoringSystem(precision=4)
        
        # Test system info
        system_info = scoring_system.get_system_info()
        
        if not system_info.get("deterministic"):
            print("✗ Scoring system not configured as deterministic")
            return False
        
        if system_info.get("precision") != 4:
            print(f"✗ Expected precision 4, got {system_info.get('precision')}")
            return False
        
        # Test score calculation
        base_score, multiplier, final_score = scoring_system.calculate_final_score("Sí", 0.8, 0.7)
        
        if not (0.0 <= float(final_score) <= 1.2):
            print(f"✗ Final score {final_score} outside bounds [0, 1.2]")
            return False
        
        print("✓ Scoring system basic functionality validated")
        return True
    except Exception as e:
        print(f"✗ Scoring system test error: {e}")
        return False

def test_checksum_determinism():
    """Test checksum determinism."""
    try:
# # #         from canonical_flow.A_analysis_nlp.question_analyzer import get_decalogo_question_registry  # Module not found  # Module not found  # Module not found
        
        # Get multiple registry instances
        registry1 = get_decalogo_question_registry()
        registry2 = get_decalogo_question_registry()
        
        # Calculate checksums
        questions1 = registry1.get_all_questions()
        questions2 = registry2.get_all_questions()
        
        # Create deterministic representations
        data1 = [(q.question_id, q.point_number, q.dimension_code) for q in questions1]
        data2 = [(q.question_id, q.point_number, q.dimension_code) for q in questions2]
        
        data1.sort()
        data2.sort()
        
        if data1 != data2:
            print("✗ Registry data not deterministic between instances")
            return False
        
        # Calculate checksums
        json1 = json.dumps(data1, sort_keys=True)
        json2 = json.dumps(data2, sort_keys=True)
        
        checksum1 = hashlib.sha256(json1.encode('utf-8')).hexdigest()
        checksum2 = hashlib.sha256(json2.encode('utf-8')).hexdigest()
        
        if checksum1 != checksum2:
            print(f"✗ Checksums differ: {checksum1[:16]}... vs {checksum2[:16]}...")
            return False
        
        print("✓ Registry checksum determinism validated")
        return True
    except Exception as e:
        print(f"✗ Checksum test error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Validating L-stage test suite components...")
    print("=" * 50)
    
    tests = [
        ("Import validation", test_imports),
        ("Registry functionality", test_registry_basic_functionality),
        ("Scoring system", test_scoring_system_basic),
        ("Checksum determinism", test_checksum_determinism),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All validation tests PASSED")
        print("L-stage test suite is ready for use")
    else:
        print("✗ Some validation tests FAILED")
        print("Please fix issues before using L-stage test suite")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())