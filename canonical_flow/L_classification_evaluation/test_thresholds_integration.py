#!/usr/bin/env python3
"""
Integration test for unified thresholds.json configuration
Tests that both ScoringSystem and AdaptiveScoringEngine load identical thresholds.
"""

import sys
import json
from pathlib import Path

# Add paths to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from decalogo_scoring_system import ScoringSystem
from adaptive_scoring_engine import AdaptiveScoringEngine


def test_threshold_consistency():
    """Test that both scoring systems load identical threshold values"""
    
    print("🧪 Testing unified thresholds configuration integration...")
    
    # Initialize both scoring systems
    try:
        scoring_system = ScoringSystem()
        adaptive_engine = AdaptiveScoringEngine(models_path="test_models")
        
        print("✓ Both systems initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize systems: {e}")
        return False
    
    # Test 1: Base scores consistency
    try:
        scoring_info = scoring_system.get_system_info()
        base_scores_scoring = scoring_info["base_scores"]
        
        # AdaptiveScoringEngine doesn't have direct access to base scores,
        # but we can check if thresholds are loaded properly
        print("✓ Base scores loaded in ScoringSystem")
        print(f"  Base scores: {base_scores_scoring}")
        
    except Exception as e:
        print(f"✗ Base scores test failed: {e}")
        return False
    
    # Test 2: Compliance thresholds consistency
    try:
        # Check dimensional thresholds
        scoring_compliance = scoring_info.get("compliance_thresholds", {})
        adaptive_compliance = adaptive_engine.compliance_thresholds
        
        print("✓ Compliance thresholds comparison:")
        
        # Compare DE-1 vs DE1 format
        for dim in ["DE-1", "DE-2", "DE-3", "DE-4"]:
            dim_compat = dim.replace("-", "")
            
            if dim in scoring_compliance and dim_compat in adaptive_compliance:
                scoring_cumple = scoring_compliance[dim]["CUMPLE"]
                adaptive_cumple = adaptive_compliance[dim_compat]["CUMPLE"]
                
                if abs(scoring_cumple - adaptive_cumple) < 0.001:
                    print(f"  ✓ {dim}: {scoring_cumple} = {adaptive_cumple}")
                else:
                    print(f"  ✗ {dim}: {scoring_cumple} ≠ {adaptive_cumple}")
                    return False
        
    except Exception as e:
        print(f"✗ Compliance thresholds test failed: {e}")
        return False
    
    # Test 3: Weights consistency
    try:
        scoring_weights = scoring_info["decalogo_weights"]
        adaptive_weights = adaptive_engine.dimension_weights
        
        print("✓ Dimension weights comparison:")
        
        # Compare weights with format conversion
        for dim_scoring, weight_scoring in scoring_weights.items():
            dim_adaptive = dim_scoring.replace("-", "")
            if dim_adaptive in adaptive_weights:
                weight_adaptive = adaptive_weights[dim_adaptive]
                
                if abs(weight_scoring - weight_adaptive) < 0.001:
                    print(f"  ✓ {dim_scoring}: {weight_scoring} = {weight_adaptive}")
                else:
                    print(f"  ✗ {dim_scoring}: {weight_scoring} ≠ {weight_adaptive}")
                    return False
        
    except Exception as e:
        print(f"✗ Weights consistency test failed: {e}")
        return False
    
    # Test 4: Schema validation
    try:
        thresholds_file = Path(__file__).parent.parent / "thresholds.json"
        
        if thresholds_file.exists():
            with open(thresholds_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Check required schema sections
            required_sections = [
                "dnp_baseline_standards",
                "compliance_classification_boundaries",
                "correction_factors",
                "validation_schema",
                "artifact_serialization"
            ]
            
            for section in required_sections:
                if section not in config:
                    print(f"✗ Missing required schema section: {section}")
                    return False
            
            print("✓ Schema validation passed")
            print(f"  Schema version: {config.get('version', 'unknown')}")
            
        else:
            print(f"✗ Thresholds file not found: {thresholds_file}")
            return False
        
    except Exception as e:
        print(f"✗ Schema validation failed: {e}")
        return False
    
    # Test 5: Compliance classification consistency
    try:
        test_cases = [
            (0.8, "DE-1", "CUMPLE"),
            (0.6, "DE-1", "CUMPLE_PARCIAL"),
            (0.3, "DE-1", "NO_CUMPLE"),
            (0.76, "P1", "CUMPLE"),
            (0.55, "P1", "CUMPLE_PARCIAL"),
            (0.2, "P1", "NO_CUMPLE")
        ]
        
        print("✓ Compliance classification tests:")
        
        for score, entity, expected in test_cases:
            try:
                classification = scoring_system.classify_compliance(score, entity)
                if classification == expected:
                    print(f"  ✓ {entity} score {score} = {classification}")
                else:
                    print(f"  ✗ {entity} score {score}: expected {expected}, got {classification}")
                    return False
                    
            except Exception as e:
                print(f"  ✗ Classification failed for {entity}: {e}")
                return False
        
    except Exception as e:
        print(f"✗ Compliance classification test failed: {e}")
        return False
    
    print("🎉 All threshold integration tests passed!")
    return True


def test_correction_factor_serialization():
    """Test correction factor serialization functionality"""
    
    print("\n🧪 Testing correction factor serialization...")
    
    try:
        adaptive_engine = AdaptiveScoringEngine(models_path="test_models")
        
        # Mock correction data
        correction_data = {
            "correction_factor": 1.25,
            "causal_distance": 0.35,
            "dnp_deviation_score": 0.45,
            "correction_status": "MODERATE_DEVIATION",
            "causal_validity_score": 0.75,
            "robustness_score": 0.62,
            "evidence_count": 5,
            "recommendations": ["Enhance evidence quality", "Strengthen causal logic"]
        }
        
        # Mock evidence references
        evidence_references = [
            {
                "reference_id": "ref_1",
                "page_number": 15,
                "context_excerpt": "Desarrollo económico inclusivo para grupos étnicos",
                "quality_score": 0.85,
                "reference_type": "documentary",
                "source_credibility": 0.9
            },
            {
                "reference_id": "ref_2", 
                "page_number": 22,
                "context_excerpt": "Protección ambiental y sostenibilidad",
                "quality_score": 0.78,
                "reference_type": "statistical",
                "source_credibility": 0.8
            }
        ]
        
        # Test serialization
        serialized = adaptive_engine.serialize_correction_factors(
            correction_data, 
            evidence_references,
            output_file="test_correction_factors.json",
            validate_schema=True
        )
        
        # Verify serialized data structure
        required_keys = [
            "metadata", "correction_factors", "dnp_deviation_scores", 
            "evidence_references", "causal_analysis", "validation_results"
        ]
        
        for key in required_keys:
            if key not in serialized:
                print(f"✗ Missing required key in serialized data: {key}")
                return False
        
        print("✓ Correction factor serialization structure validated")
        
        # Test validation results
        validation = serialized.get("validation_results", {})
        if validation.get("is_valid", False):
            print("✓ Schema validation passed")
        else:
            print(f"✗ Schema validation failed: {validation.get('errors', [])}")
            return False
        
        # Test artifact file creation
        if "artifact_location" in serialized:
            artifact_path = Path(serialized["artifact_location"])
            if artifact_path.exists():
                print(f"✓ Artifact file created: {artifact_path}")
                
                # Clean up test file
                artifact_path.unlink()
                print("✓ Test artifact cleaned up")
                
            else:
                print(f"✗ Artifact file not found: {artifact_path}")
                return False
        
        print("🎉 Correction factor serialization tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Correction factor serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    
    print("=" * 60)
    print("UNIFIED THRESHOLDS INTEGRATION TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Threshold consistency
    if not test_threshold_consistency():
        all_passed = False
    
    # Test 2: Correction factor serialization  
    if not test_correction_factor_serialization():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✓ Unified thresholds configuration working correctly")
        print("✓ Both ScoringSystem and AdaptiveScoringEngine use identical values")
        print("✓ Correction factor serialization functional")
    else:
        print("❌ SOME INTEGRATION TESTS FAILED!")
        print("⚠️  Check threshold configuration and system integration")
    
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)