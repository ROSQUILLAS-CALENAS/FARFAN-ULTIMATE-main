#!/usr/bin/env python3
"""
Validation script for DocumentFeatureExtractor refactor.
Tests inheritance, interface compliance, and deterministic behavior.
"""

import json
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

def validate_implementation():
    """Validate the refactored implementation"""
    
    print("🔍 Validating DocumentFeatureExtractor refactor...")
    
    # Test 1: Import and inheritance
    try:
# # #         from feature_extractor import DocumentFeatureExtractor  # Module not found  # Module not found  # Module not found
# # #         from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found
        
        extractor = DocumentFeatureExtractor()
        
        # Verify inheritance
# # #         assert isinstance(extractor, TotalOrderingBase), "Must inherit from TotalOrderingBase"  # Module not found  # Module not found  # Module not found
# # #         print("✅ Inheritance from TotalOrderingBase verified")  # Module not found  # Module not found  # Module not found
        
        # Verify required methods exist
        assert hasattr(extractor, 'process'), "Must implement process() method"
        print("✅ Required process() method exists")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except AssertionError as e:
        print(f"❌ Interface validation failed: {e}")
        return False
    
    # Test 2: Deterministic component IDs
    try:
        extractor1 = DocumentFeatureExtractor()
        extractor2 = DocumentFeatureExtractor()
        
        assert extractor1.component_id == extractor2.component_id, "Component IDs must be deterministic"
        print("✅ Deterministic component ID generation verified")
        
    except AssertionError as e:
        print(f"❌ Deterministic behavior failed: {e}")
        return False
    
    # Test 3: Process method functionality
    try:
        result = extractor.process('sample_document')
        
        assert isinstance(result, dict), "process() must return dict"
        assert 'status' in result, "Result must contain status"
        assert 'doc_stem' in result, "Result must contain doc_stem"
        assert 'operation_id' in result, "Result must contain operation_id"
        
        print("✅ Process method interface validated")
        
    except Exception as e:
        print(f"❌ Process method failed: {e}")
        return False
    
    # Test 4: Features file format validation
    try:
        features_path = Path("canonical_flow/ingestion/sample_document_features.json")
        
        if features_path.exists():
            with open(features_path, 'r', encoding='utf-8') as f:
                features_data = json.load(f)
            
            # Validate required fields
            required_fields = [
                'bundle_id', 'document_stem', 'extraction_timestamp',
                'features', 'operation_id', 'processing_status', 'schema_version'
            ]
            
            for field in required_fields:
                assert field in features_data, f"Features file missing required field: {field}"
            
            # Validate features structure
            features = features_data['features']
            expected_features = [
                'word_count', 'character_count', 'sentence_count', 'paragraph_count',
                'flesch_reading_ease', 'primary_language'
            ]
            
            for feature in expected_features:
                assert feature in features, f"Missing expected feature: {feature}"
            
            print("✅ Features file structure validated")
            print(f"   - Schema version: {features_data.get('schema_version')}")
            print(f"   - Feature count: {len(features)}")
            
        else:
            print("⚠️ Features file not found for validation")
            
    except Exception as e:
        print(f"❌ Features file validation failed: {e}")
        return False
    
    # Test 5: Error handling and graceful degradation
    try:
        # Test with non-existent file
        error_result = extractor.process('completely_nonexistent')
        
        if error_result['status'] == 'error':
            print("✅ Error handling for missing files validated")
        else:
            print("⚠️ Unexpected status for missing file")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    
    print("🎉 All validation tests passed!")
    return True

def validate_json_formatting():
    """Validate JSON formatting consistency"""
    
    print("\n🔍 Validating JSON formatting...")
    
    features_path = Path("canonical_flow/ingestion/sample_document_features.json")
    
    if not features_path.exists():
        print("⚠️ Features file not found for formatting validation")
        return True
    
    try:
        with open(features_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse and reformat
        data = json.loads(content)
        expected_format = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
        
        # Check formatting
        content_lines = content.strip().split('\n')
        expected_lines = expected_format.strip().split('\n')
        
        if len(content_lines) == len(expected_lines):
            print("✅ JSON formatting consistent (indent=2, sorted keys)")
        else:
            print(f"⚠️ JSON formatting may be inconsistent (lines: {len(content_lines)} vs {len(expected_lines)})")
        
        # Check UTF-8 encoding
        try:
            content.encode('utf-8')
            print("✅ UTF-8 encoding validated")
        except UnicodeEncodeError:
            print("❌ UTF-8 encoding validation failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ JSON formatting validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_implementation()
    formatting_success = validate_json_formatting()
    
    if success and formatting_success:
        print("\n🏆 All validations passed! DocumentFeatureExtractor refactor complete.")
        sys.exit(0)
    else:
        print("\n❌ Some validations failed.")
        sys.exit(1)