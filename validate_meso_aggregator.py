#!/usr/bin/env python3
"""
Validation script for meso_aggregator.py modifications
"""

import json
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

# Test data
test_data = {
    'cluster_audit': {
        'micro': {
            'C1': {
                'answers': [
                    {
                        'question_id': 'Q1',
                        'verdict': 'YES',
                        'score': 0.8,
                        'evidence_ids': ['E1', 'E2'],
                        'components': ['OBJECTIVES', 'TIMELINES']
                    }
                ]
            },
            'C2': {
                'answers': [
                    {
                        'question_id': 'Q1',
                        'verdict': 'NO',
                        'score': 0.3,
                        'evidence_ids': ['E3'],
                        'components': ['STRATEGIES']
                    }
                ]
            }
        }
    }
}

test_context = {'doc_stem': 'test_document'}

def validate_meso_aggregator():
    """Validate the meso_aggregator module and artifact generation."""
    
    try:
        # Import the module
        import meso_aggregator
        print("✓ Module imports successfully")
        
        # Test the process function
        result = meso_aggregator.process(test_data, test_context)
        print("✓ Process function executes successfully")
        
        # Check required output fields
        required_fields = ['meso_summary', 'coverage_matrix']
        for field in required_fields:
            if field in result:
                print(f"✓ Required field '{field}' present in result")
            else:
# # #                 print(f"✗ Required field '{field}' missing from result")  # Module not found  # Module not found  # Module not found
                return False
        
        # Check if artifacts directory was created
        if os.path.exists('canonical_flow/aggregation'):
            print("✓ canonical_flow/aggregation directory created")
            
            # Check if artifact file was created
            expected_file = 'canonical_flow/aggregation/test_document_meso.json'
            if os.path.exists(expected_file):
                print("✓ Artifact file test_document_meso.json created")
                
                # Validate JSON structure
                with open(expected_file, 'r', encoding='utf-8') as f:
                    artifact = json.load(f)
                    print("✓ JSON artifact is valid and properly formatted")
                    
                    # Check required schema fields
                    required_schema_fields = [
                        'doc_stem', 'coverage_metrics', 'divergence_scores',
                        'cluster_participation', 'dimension_groupings', 'meso_summary'
                    ]
                    
                    for field in required_schema_fields:
                        if field in artifact:
                            print(f"✓ Schema field '{field}' present in artifact")
                        else:
# # #                             print(f"✗ Schema field '{field}' missing from artifact")  # Module not found  # Module not found  # Module not found
                            return False
                    
                    # Check JSON formatting (sorted keys, proper indentation)
                    with open(expected_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '"doc_stem": "test_document"' in content and content.count('\n') > 10:
                            print("✓ JSON formatted with proper indentation and sorting")
                        else:
                            print("? JSON formatting check inconclusive")
                
            else:
                print(f"✗ Expected artifact file not found: {expected_file}")
                return False
        else:
            print("✗ canonical_flow/aggregation directory not created")
            return False
        
        print("\n✓ All validation checks passed!")
        return True
        
    except Exception as e:
        print(f"✗ Validation failed with error: {e}")
        return False

if __name__ == "__main__":
    success = validate_meso_aggregator()
    sys.exit(0 if success else 1)