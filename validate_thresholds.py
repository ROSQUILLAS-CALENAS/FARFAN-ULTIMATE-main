#!/usr/bin/env python3
"""
Simple validation script for thresholds configuration.
"""

import json
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

def main():
    """Validate thresholds.json structure."""
    print("Validating thresholds configuration...")
    
    # Check file exists
    thresholds_path = Path("thresholds.json")
    if not thresholds_path.exists():
        print(f"✗ thresholds.json not found")
        return 1
    
    try:
        # Load JSON
        with open(thresholds_path, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = [
            "version", "temperature", "scoring_bounds", "fusion_weights", 
            "evidence_multipliers", "quality_thresholds", "conformal_prediction",
            "retrieval_thresholds", "decalogo_scoring", "adaptive_scoring",
            "dnp_alignment", "statistical_bounds", "aggregation_thresholds", "validation"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"✗ Missing sections: {missing_sections}")
            return 1
        
        # Check temperature bounds
        temp = config["temperature"]
        if temp["min_temperature"] >= temp["max_temperature"]:
            print("✗ Invalid temperature bounds")
            return 1
        
        # Check fusion weights sum to ~1.0
        weights = config["fusion_weights"]
        weight_sum = weights["lexical"] + weights["vector"] + weights["late_interaction"] + weights["rrf"]
        if abs(weight_sum - 1.0) > 0.01:
            print(f"✗ Fusion weights sum to {weight_sum}, expected ~1.0")
            return 1
        
        # Check evidence multiplier bounds
        mult = config["evidence_multipliers"]
        if mult["MIN_MULTIPLIER"] >= mult["MAX_MULTIPLIER"]:
            print("✗ Invalid multiplier bounds")
            return 1
        
        # Check conformal alpha in valid range
        alpha = config["conformal_prediction"]["alpha"]
        if not (0 < alpha < 1):
            print(f"✗ Invalid alpha: {alpha}")
            return 1
        
        print("✓ thresholds.json validation passed")
        print(f"✓ Configuration version: {config['version']}")
        print(f"✓ Found {len(required_sections)} required sections")
        print(f"✓ Temperature range: [{temp['min_temperature']}, {temp['max_temperature']}]")
        print(f"✓ Fusion weights sum: {weight_sum:.3f}")
        print(f"✓ Alpha: {alpha}")
        
        return 0
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())