"""
Validation script for StageValidationMiddleware

Tests the middleware functionality without pytest dependency.
"""

import json
import logging
from stage_validation_middleware import (
    StageValidationMiddleware,
    StageSchema, 
    FieldSpec,
    ValidationRule,
    ValidationStatus,
    IntegrityChecker,
    create_data_completeness_rule,
    create_non_empty_string_rule,
    create_basic_pipeline_schema,
    demo_validation_middleware
)


def test_integrity_checker():
    """Test the integrity checker functionality."""
    print("Testing IntegrityChecker...")
    
    # Test checksum calculation
    data = {"key1": "value1", "key2": "value2"}
    checksum = IntegrityChecker.calculate_checksum(data)
    print(f"✓ Checksum calculated: {checksum[:16]}...")
    
    # Test checksum verification
    assert IntegrityChecker.verify_checksum(data, checksum)
    print("✓ Checksum verification passed")
    
    # Test different algorithms
    sha256 = IntegrityChecker.calculate_checksum(data, "sha256")
    md5 = IntegrityChecker.calculate_checksum(data, "md5")
    sha1 = IntegrityChecker.calculate_checksum(data, "sha1")
    
    assert len(sha256) == 64
    assert len(md5) == 32
    assert len(sha1) == 40
    print("✓ Different hash algorithms work correctly")
    
    # Test field integrity validation
    test_data = {
        "content": {"key": "value"},
        "content_checksum": IntegrityChecker.calculate_checksum({"key": "value"}),
    }
    
    failures = IntegrityChecker.validate_field_integrity(test_data, ["content_checksum"])
    assert len(failures) == 0
    print("✓ Field integrity validation passed")
    
    # Test integrity failure detection
    test_data["content_checksum"] = "invalid_checksum"
    failures = IntegrityChecker.validate_field_integrity(test_data, ["content_checksum"])
    assert len(failures) == 1
    print("✓ Integrity failure detection works")
    
    print("IntegrityChecker tests completed successfully\n")


def test_stage_validation_middleware():
    """Test the main middleware functionality."""
    print("Testing StageValidationMiddleware...")
    
    # Create middleware instance
    middleware = StageValidationMiddleware(halt_on_failure=False)
    
    # Create and register test schema
    schema = StageSchema(
        stage_name="test_stage",
        version="1.0",
        required_fields=[
            FieldSpec(name="data", field_type=dict),
            FieldSpec(name="stage_id", field_type=str, min_length=3),
            FieldSpec(name="timestamp", field_type=str)
        ],
        optional_fields=[
            FieldSpec(name="metadata", field_type=dict)
        ],
        validation_rules=[
            create_data_completeness_rule(min_fields=3),
            create_non_empty_string_rule("stage_id")
        ],
        checksum_fields=["data_checksum"]
    )
    
    middleware.register_stage_schema(schema)
    print("✓ Schema registered successfully")
    
    # Test with valid data
    valid_data = {
        "data": {"content": "test data", "value": 123},
        "stage_id": "test_001",
        "timestamp": "2024-12-20T10:00:00Z",
        "metadata": {"version": "1.0"},
        "data_checksum": IntegrityChecker.calculate_checksum({"content": "test data", "value": 123})
    }
    
    report = middleware.validate_stage_input("test_stage", valid_data)
    assert report.overall_status == ValidationStatus.PASSED
    assert not report.has_failures()
    print("✓ Valid data validation passed")
    
    # Test with invalid data
    invalid_data = {
        "data": {},  # Empty data
        "stage_id": "ab",  # Too short
        # Missing timestamp
        "data_checksum": "invalid_checksum"
    }
    
    report = middleware.validate_stage_input("test_stage", invalid_data)
    assert report.overall_status == ValidationStatus.FAILED
    assert report.has_failures()
    print("✓ Invalid data validation failed as expected")
    
    # Test validation summary
    summary = middleware.get_validation_summary()
    assert summary["total_validations"] == 2
    assert "test_stage" in summary["registered_schemas"]
    print("✓ Validation summary generated correctly")
    
    print("StageValidationMiddleware tests completed successfully\n")


def test_stage_interception():
    """Test stage execution interception."""
    print("Testing stage execution interception...")
    
    middleware = StageValidationMiddleware(halt_on_failure=False)
    
    # Register basic schema
    schema = create_basic_pipeline_schema("intercept_stage")
    middleware.register_stage_schema(schema)
    
    # Mock stage executor
    def mock_executor(data):
        return {"result": "executed", "input_data": data}
    
    # Valid data
    valid_data = {
        "data": {"test": "data"},
        "stage_id": "test_001",
        "timestamp": "2024-12-20T10:00:00Z",
        "data_checksum": IntegrityChecker.calculate_checksum({"test": "data"})
    }
    
    # Test interception with valid data
    result = middleware.intercept_stage_execution(
        "intercept_stage", mock_executor, valid_data
    )
    
    assert result["result"] == "executed"
    print("✓ Stage execution interception works with valid data")
    
    print("Stage interception tests completed successfully\n")


def test_validation_rules():
    """Test custom validation rules."""
    print("Testing validation rules...")
    
    # Test data completeness rule
    rule = create_data_completeness_rule(min_fields=3)
    assert rule.validator_func({"a": 1, "b": 2, "c": 3})
    assert not rule.validator_func({"a": 1})
    print("✓ Data completeness rule works")
    
    # Test non-empty string rule
    rule = create_non_empty_string_rule("test_field")
    assert rule.validator_func({"test_field": "valid_string"})
    assert not rule.validator_func({"test_field": ""})
    assert not rule.validator_func({"test_field": "   "})
    print("✓ Non-empty string rule works")
    
    print("Validation rules tests completed successfully\n")


def run_comprehensive_demo():
    """Run the comprehensive demonstration."""
    print("=" * 50)
    print("STAGE VALIDATION MIDDLEWARE DEMONSTRATION")
    print("=" * 50)
    print()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run built-in demo
    demo_validation_middleware()


if __name__ == "__main__":
    try:
        test_integrity_checker()
        test_stage_validation_middleware()
        test_stage_interception()
        test_validation_rules()
        
        print("=" * 50)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 50)
        print()
        
        # Run comprehensive demo
        run_comprehensive_demo()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()