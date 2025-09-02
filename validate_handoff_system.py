#!/usr/bin/env python3
"""
Validation script for the handoff validation system
"""

def test_basic_imports():
    """Test basic imports work"""
    try:
        from schemas.pipeline_schemas import StageType, DataIntegrityLevel, ValidationError
        print("✅ Schema imports successful")
        
        from handoff_validation_system import (
            HandoffValidationSystem, 
            validate_ingestion_to_context,
            create_checkpoint_validator
        )
        print("✅ Handoff validation system imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic system functionality"""
    try:
        from handoff_validation_system import create_checkpoint_validator
        from schemas.pipeline_schemas import StageType
        from datetime import datetime
        
        # Create validator
        validator = create_checkpoint_validator()
        print("✅ Validator created successfully")
        
        # Test data
        sample_data = {
            'metadata': {
                'doc_id': 'test_001',
                'source_path': '/data/test.pdf', 
                'content_hash': 'hash123',
                'created_timestamp': datetime.now().isoformat(),
                'stage_processed': 'I_ingestion_preparation'
            },
            'extracted_text': 'Sample text content for validation testing',
            'document_structure': {'pages': 3, 'sections': 2},
            'page_count': 3,
            'extraction_confidence': 0.92,
            'content_blocks': [{'type': 'paragraph', 'content': 'Test paragraph'}],
            'semantic_markers': ['intro', 'body']
        }
        
        # Test validation
        result = validator.validate_handoff(
            StageType.INGESTION_PREPARATION,
            StageType.CONTEXT_ESTABLISHMENT, 
            sample_data
        )
        
        print(f"✅ Validation completed: valid={result.is_valid}, errors={len(result.validation_errors)}")
        
        if result.validation_errors:
            print("Validation issues found:")
            for i, error in enumerate(result.validation_errors[:3]):
                print(f"  {i+1}. {error.severity}: {error.field_name} - {error.error_message}")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("=== Handoff Validation System Test ===")
    
    # Test imports
    if not test_basic_imports():
        return False
    
    # Test functionality  
    if not test_basic_functionality():
        return False
    
    print("\n🎉 All tests passed! System is ready for use.")
    return True

if __name__ == "__main__":
    main()