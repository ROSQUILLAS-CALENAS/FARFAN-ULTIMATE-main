#!/usr/bin/env python3
"""
Demo script for EarlyErrorDetector system

This script demonstrates how to use the EarlyErrorDetector for validating
pipeline stages with schema compliance, data integrity checks, and real-time
monitoring capabilities.
"""

import json
import logging
import time
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# # # from egw_query_expansion.core.early_error_detector import (  # Module not found  # Module not found  # Module not found
    EarlyErrorDetector,
    StageSchema,
    LoggingHook,
    WebhookHook,
    ValidationError,
    ValidationErrorType,
    ValidationSeverity,
    validate_stage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_schemas():
    """Create sample schemas for demonstration"""
    
    # Data ingestion stage schema
    ingestion_schema = StageSchema(
        stage_name="data_ingestion",
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "format": {"type": "string", "enum": ["json", "csv", "xml"]},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 10000}
            },
            "required": ["source", "format"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "records": {"type": "array"},
                "metadata": {"type": "object"}
            },
            "required": ["records"]
        },
        required_fields={"source", "format"},
        optional_fields={"batch_size", "options"},
        value_ranges={
            "batch_size": {"min": 1, "max": 10000},
            "format": {"allowed": ["json", "csv", "xml"]},
            "priority": {"min": 1, "max": 10}
        },
        artifact_patterns=[r".*\.log$", r"ingestion_.*\.json$"],
        metadata_requirements={
            "ingestion_time": {"required": True, "type": float},
            "source_count": {"required": False, "type": int}
        }
    )
    
    # Data processing stage schema
    processing_schema = StageSchema(
        stage_name="data_processing",
        input_schema={
            "type": "object",
            "properties": {
                "data": {"type": "array"},
                "algorithm": {"type": "string"}
            },
            "required": ["data"]
        },
        output_schema={},
        required_fields={"processed_data"},
        optional_fields={"statistics", "quality_metrics"},
        value_ranges={
            "algorithm": {"allowed": ["linear", "polynomial", "neural"]},
            "quality_score": {"min": 0.0, "max": 1.0}
        },
        artifact_patterns=[r"processed_.*\.json$", r"model_.*\.pkl$"],
        metadata_requirements={
            "processing_time": {"required": True, "type": float},
            "algorithm_version": {"required": True, "type": str}
        }
    )
    
    # Output generation stage schema  
    output_schema = StageSchema(
        stage_name="output_generation",
        input_schema={},
        output_schema={},
        required_fields={"final_output", "summary"},
        optional_fields={"diagnostics"},
        artifact_patterns=[r"output_.*\.json$", r"report_.*\.pdf$"],
        metadata_requirements={
            "generation_time": {"required": True, "type": float},
            "output_format": {"required": True, "type": str}
        }
    )
    
    return [ingestion_schema, processing_schema, output_schema]


def demo_basic_validation():
    """Demonstrate basic validation functionality"""
    print("\n" + "="*60)
    print("DEMO: Basic Validation")
    print("="*60)
    
    detector = EarlyErrorDetector()
    schemas = create_sample_schemas()
    
    # Register schemas
    for schema in schemas:
        detector.register_schema(schema)
    
    # Test valid input
    print("\n1. Testing valid input:")
    valid_input = {
        "source": "database",
        "format": "json", 
        "batch_size": 1000,
        "metadata": {
            "ingestion_time": time.time()
        }
    }
    
    result = detector.validate_stage_input("data_ingestion", valid_input)
    print(f"Valid input result: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    # Test invalid input
    print("\n2. Testing invalid input (missing required field):")
    invalid_input = {
        "format": "json",  # Missing required "source" field
        "batch_size": 1000
    }
    
    result = detector.validate_stage_input("data_ingestion", invalid_input)
    print(f"Invalid input result: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    
    for error in result.errors:
        print(f"  - {error.error_type.value}: {error.message}")
    
    # Test range violation
    print("\n3. Testing range violation:")
    range_violation_input = {
        "source": "file",
        "format": "json",
        "batch_size": 50000,  # Exceeds maximum
        "priority": 15  # Exceeds maximum
    }
    
    result = detector.validate_stage_input("data_ingestion", range_violation_input)
    print(f"Range violation result: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    
    for error in result.errors:
        print(f"  - {error.error_type.value}: {error.message} (Field: {error.field_path})")


def demo_hooks_and_monitoring():
    """Demonstrate hooks and real-time monitoring"""
    print("\n" + "="*60)
    print("DEMO: Hooks and Monitoring")
    print("="*60)
    
    detector = EarlyErrorDetector()
    schemas = create_sample_schemas()
    
    # Register schemas
    for schema in schemas:
        detector.register_schema(schema)
    
    # Add logging hook
    logging_hook = LoggingHook()
    detector.add_hook(logging_hook)
    
    # Add mock webhook hook (in real scenario, use actual webhook URL)
    class MockWebhookHook:
        def __init__(self):
            self.notifications = []
        
        def on_validation_start(self, stage_name, data):
            pass
        
        def on_validation_error(self, error):
            if error.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                self.notifications.append({
                    "type": "error",
                    "stage": error.stage_name,
                    "message": error.message,
                    "severity": error.severity.value
                })
                print(f"📢 WEBHOOK ALERT: {error.severity.value.upper()} in {error.stage_name}: {error.message}")
        
        def on_validation_complete(self, result):
            if not result.is_valid and result.critical_errors:
                self.notifications.append({
                    "type": "critical_failure",
                    "critical_errors": len(result.critical_errors)
                })
                print(f"🚨 CRITICAL FAILURE ALERT: {len(result.critical_errors)} critical errors detected!")
    
    webhook_hook = MockWebhookHook()
    detector.add_hook(webhook_hook)
    
    print("\n1. Testing validation with monitoring hooks:")
    
    # Test with errors to trigger notifications
    invalid_data = {
        "source": "database",
        "format": "invalid_format",  # Not in allowed values
        "batch_size": -5,  # Below minimum
        "metadata": {}  # Missing required ingestion_time
    }
    
    result = detector.validate_stage_input("data_ingestion", invalid_data)
    
    print(f"\nValidation completed. Webhook notifications sent: {len(webhook_hook.notifications)}")
    for notification in webhook_hook.notifications:
        print(f"  - {notification}")


def demo_referential_integrity():
    """Demonstrate referential integrity validation"""
    print("\n" + "="*60)
    print("DEMO: Referential Integrity")
    print("="*60)
    
    detector = EarlyErrorDetector()
    schemas = create_sample_schemas()
    
    # Register schemas
    for schema in schemas:
        detector.register_schema(schema)
    
    print("\n1. Setting up pipeline with stage dependencies:")
    
    # Stage 1: Data ingestion (produces output)
    stage1_output = {
        "records": [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}],
        "metadata": {
            "ingestion_time": time.time(),
            "record_count": 2
        }
    }
    
    result1 = detector.validate_stage_output("data_ingestion", stage1_output)
    print(f"Stage 1 output validation: {result1.is_valid}")
    
    # Stage 2: Data processing (references stage 1 output)
    stage2_input = {
        "processed_data": [{"id": 1, "processed_value": "A_processed"}],
        "references": {
            "data_ingestion": ["records", "metadata"]  # Valid references
        },
        "artifacts": ["processed_data.json"],
        "metadata": {
            "processing_time": time.time(),
            "algorithm_version": "v1.0"
        }
    }
    
    result2 = detector.validate_stage_input("data_processing", stage2_input)
    print(f"Stage 2 input validation (valid references): {result2.is_valid}")
    
    # Test invalid reference
    print("\n2. Testing invalid referential integrity:")
    
    stage2_invalid = {
        "processed_data": [{"id": 1, "processed_value": "A_processed"}],
        "references": {
            "data_ingestion": ["records", "nonexistent_field"],  # Invalid reference
            "missing_stage": ["some_field"]  # Stage doesn't exist
        },
        "artifacts": ["processed_data.json"],
        "metadata": {
            "processing_time": time.time(),
            "algorithm_version": "v1.0"
        }
    }
    
    result2_invalid = detector.validate_stage_input("data_processing", stage2_invalid)
    print(f"Stage 2 input validation (invalid references): {result2_invalid.is_valid}")
    
    integrity_errors = [
        e for e in result2_invalid.errors 
        if e.error_type == ValidationErrorType.REFERENTIAL_INTEGRITY
    ]
    print(f"Referential integrity errors: {len(integrity_errors)}")
    
    for error in integrity_errors:
        print(f"  - {error.message}")


def demo_artifact_validation():
    """Demonstrate artifact and naming convention validation"""
    print("\n" + "="*60)
    print("DEMO: Artifact Validation")
    print("="*60)
    
    detector = EarlyErrorDetector()
    schemas = create_sample_schemas()
    
    # Register schemas
    for schema in schemas:
        detector.register_schema(schema)
    
    print("\n1. Testing artifact validation:")
    
    # Valid artifacts
    valid_data = {
        "processed_data": [1, 2, 3],
        "artifacts": [
            "processed_results.json",  # Matches processed_.*\.json$
            "model_v1.pkl"  # Matches model_.*\.pkl$
        ],
        "metadata": {
            "processing_time": time.time(),
            "algorithm_version": "v2.0"
        }
    }
    
    result_valid = detector.validate_stage_input("data_processing", valid_data)
    print(f"Valid artifacts result: {result_valid.is_valid}")
    
    # Invalid artifacts (missing artifacts)
    missing_artifacts = {
        "processed_data": [1, 2, 3],
        # No artifacts provided
        "metadata": {
            "processing_time": time.time(),
            "algorithm_version": "v2.0"
        }
    }
    
    result_missing = detector.validate_stage_input("data_processing", missing_artifacts)
    print(f"Missing artifacts result: {result_missing.is_valid}")
    
    artifact_errors = [
        e for e in result_missing.errors 
        if e.error_type == ValidationErrorType.ARTIFACT_MISSING
    ]
    print(f"Artifact errors: {len(artifact_errors)}")
    
    # Wrong naming convention
    wrong_naming = {
        "processed_data": [1, 2, 3],
        "artifacts": [
            "wrong_name.json",  # Doesn't match patterns
            "another_file.txt"
        ],
        "metadata": {
            "processing_time": time.time(),
            "algorithm_version": "v2.0"
        }
    }
    
    result_wrong = detector.validate_stage_input("data_processing", wrong_naming)
    print(f"Wrong naming result: {result_wrong.is_valid}")
    
    naming_warnings = [
        e for e in (result_wrong.errors + result_wrong.warnings)
        if e.error_type == ValidationErrorType.NAMING_CONVENTION
    ]
    print(f"Naming convention warnings: {len(naming_warnings)}")


def demo_custom_validators():
    """Demonstrate custom validation logic"""
    print("\n" + "="*60)
    print("DEMO: Custom Validators")
    print("="*60)
    
    def business_logic_validator(data, stage_name):
        """Custom validator implementing business logic"""
        errors = []
        
        # Example: Check that batch_size is reasonable for the source type
        if "source" in data and "batch_size" in data:
            source = data["source"]
            batch_size = data["batch_size"]
            
            if source == "database" and batch_size > 5000:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.RANGE_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message="Large batch sizes may impact database performance",
                    field_path="batch_size",
                    stage_name=stage_name,
                    context={"recommendation": "Consider batch_size <= 5000 for database sources"}
                ))
            
            elif source == "file" and batch_size < 100:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.RANGE_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message="Small batch sizes may be inefficient for file processing",
                    field_path="batch_size", 
                    stage_name=stage_name,
                    context={"recommendation": "Consider batch_size >= 100 for file sources"}
                ))
        
        return errors
    
    def data_quality_validator(data, stage_name):
        """Custom validator for data quality checks"""
        errors = []
        
        if "processed_data" in data:
            processed_data = data["processed_data"]
            
            # Check for empty results
            if not processed_data:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.SCHEMA_VIOLATION,
                    severity=ValidationSeverity.ERROR,
                    message="Processed data cannot be empty",
                    field_path="processed_data",
                    stage_name=stage_name
                ))
            
            # Check for duplicate records
            elif isinstance(processed_data, list):
                seen = set()
                duplicates = []
                for item in processed_data:
                    if isinstance(item, dict) and "id" in item:
                        if item["id"] in seen:
                            duplicates.append(item["id"])
                        seen.add(item["id"])
                
                if duplicates:
                    errors.append(ValidationError(
                        error_type=ValidationErrorType.SCHEMA_VIOLATION,
                        severity=ValidationSeverity.WARNING,
                        message=f"Duplicate IDs found: {duplicates}",
                        field_path="processed_data",
                        stage_name=stage_name,
                        context={"duplicate_ids": duplicates}
                    ))
        
        return errors
    
    # Create detector with custom validators
    detector = EarlyErrorDetector()
    
    # Create schema with custom validators
    custom_schema = StageSchema(
        stage_name="custom_validation_stage",
        input_schema={},
        output_schema={},
        required_fields={"source"},
        custom_validators=[business_logic_validator, data_quality_validator]
    )
    
    detector.register_schema(custom_schema)
    
    print("\n1. Testing custom business logic validator:")
    
    # Test database with large batch size
    db_large_batch = {
        "source": "database",
        "batch_size": 8000  # Should trigger warning
    }
    
    result1 = detector.validate_stage_input("custom_validation_stage", db_large_batch)
    print(f"Database large batch result: {result1.is_valid}")
    print(f"Warnings: {len(result1.warnings)}")
    
    if result1.warnings:
        for warning in result1.warnings:
            print(f"  - {warning.message}")
            if warning.context:
                print(f"    Recommendation: {warning.context.get('recommendation', 'N/A')}")
    
    # Register processing schema with data quality validator
    processing_schema_custom = StageSchema(
        stage_name="processing_with_quality",
        input_schema={},
        output_schema={},
        required_fields={"processed_data"},
        custom_validators=[data_quality_validator]
    )
    
    detector.register_schema(processing_schema_custom)
    
    print("\n2. Testing data quality validator:")
    
    # Test with duplicate IDs
    duplicate_data = {
        "processed_data": [
            {"id": 1, "value": "A"},
            {"id": 2, "value": "B"}, 
            {"id": 1, "value": "A_duplicate"}  # Duplicate ID
        ]
    }
    
    result2 = detector.validate_stage_input("processing_with_quality", duplicate_data)
    print(f"Duplicate data result: {result2.is_valid}")
    print(f"Warnings: {len(result2.warnings)}")
    
    if result2.warnings:
        for warning in result2.warnings:
            print(f"  - {warning.message}")


def demo_decorator_usage():
    """Demonstrate the validate_stage decorator"""
    print("\n" + "="*60)
    print("DEMO: Decorator Usage")
    print("="*60)
    
    detector = EarlyErrorDetector()
    
    # Simple schema for decorator demo
    simple_schema = StageSchema(
        stage_name="math_operation",
        input_schema={},
        output_schema={},
        required_fields=set(),
        value_ranges={
            "args.0": {"min": 0},  # First argument should be non-negative
        }
    )
    
    detector.register_schema(simple_schema)
    
    @validate_stage("math_operation", detector)
    def square_root(x):
        """Function with automatic validation"""
        return x ** 0.5
    
    print("\n1. Testing decorated function with valid input:")
    try:
        result = square_root(25)
        print(f"square_root(25) = {result}")
    except ValueError as e:
        print(f"Validation error: {e}")
    
    print("\n2. Testing decorated function with invalid input:")
    try:
        result = square_root(-5)  # Should fail validation
        print(f"square_root(-5) = {result}")
    except ValueError as e:
        print(f"Validation error: {e}")


def demo_complete_pipeline():
    """Demonstrate a complete pipeline with all features"""
    print("\n" + "="*60)
    print("DEMO: Complete Pipeline")
    print("="*60)
    
    detector = EarlyErrorDetector()
    
    # Add monitoring
    logging_hook = LoggingHook()
    detector.add_hook(logging_hook)
    
    # Register all schemas
    schemas = create_sample_schemas()
    for schema in schemas:
        detector.register_schema(schema)
    
    print("\n🚀 Starting complete pipeline execution:")
    
    try:
        # Stage 1: Data Ingestion
        print("\n📥 Stage 1: Data Ingestion")
        stage1_input = {
            "source": "database",
            "format": "json",
            "batch_size": 1000,
            "metadata": {
                "ingestion_time": time.time(),
                "source_count": 5000
            },
            "artifacts": ["ingestion.log", "ingestion_batch_001.json"]
        }
        
        result1 = detector.validate_stage_input("data_ingestion", stage1_input)
        if not result1.is_valid:
            print(f"❌ Stage 1 validation failed: {len(result1.errors)} errors")
            return
        print("✅ Stage 1 input validation passed")
        
        # Simulate stage 1 processing
        stage1_output = {
            "records": [
                {"id": i, "value": f"record_{i}"}
                for i in range(1, 6)
            ],
            "metadata": {
                "ingestion_time": stage1_input["metadata"]["ingestion_time"],
                "record_count": 5
            }
        }
        
        result1_out = detector.validate_stage_output("data_ingestion", stage1_output)
        if not result1_out.is_valid:
            print(f"❌ Stage 1 output validation failed")
            return
        print("✅ Stage 1 output validation passed")
        
        # Stage 2: Data Processing
        print("\n⚙️  Stage 2: Data Processing")
        stage2_input = {
            "processed_data": [
                {"id": i, "processed_value": f"processed_{i}"}
                for i in range(1, 6)
            ],
            "algorithm": "linear",
            "quality_score": 0.95,
            "references": {
                "data_ingestion": ["records", "metadata"]
            },
            "artifacts": ["processed_batch_001.json", "model_linear_v1.pkl"],
            "metadata": {
                "processing_time": time.time(),
                "algorithm_version": "v1.2"
            }
        }
        
        result2 = detector.validate_stage_input("data_processing", stage2_input)
        if not result2.is_valid:
            print(f"❌ Stage 2 validation failed: {len(result2.errors)} errors")
            for error in result2.errors:
                print(f"   - {error.message}")
            return
        print("✅ Stage 2 input validation passed")
        
        stage2_output = {
            "processed_data": stage2_input["processed_data"],
            "statistics": {"mean": 3.0, "std": 1.58},
            "quality_metrics": {"accuracy": 0.95, "completeness": 1.0}
        }
        
        result2_out = detector.validate_stage_output("data_processing", stage2_output)
        if not result2_out.is_valid:
            print(f"❌ Stage 2 output validation failed")
            return
        print("✅ Stage 2 output validation passed")
        
        # Stage 3: Output Generation
        print("\n📤 Stage 3: Output Generation")
        stage3_input = {
            "final_output": {
                "processed_records": 5,
                "quality_score": 0.95,
                "processing_algorithm": "linear"
            },
            "summary": {
                "total_records": 5,
                "processing_time": 1.23,
                "quality_metrics": stage2_output["quality_metrics"]
            },
            "references": {
                "data_processing": ["processed_data", "statistics"]
            },
            "artifacts": ["output_report.json", "report_final.pdf"],
            "metadata": {
                "generation_time": time.time(),
                "output_format": "json+pdf"
            }
        }
        
        result3 = detector.validate_stage_input("output_generation", stage3_input)
        if not result3.is_valid:
            print(f"❌ Stage 3 validation failed: {len(result3.errors)} errors")
            return
        print("✅ Stage 3 input validation passed")
        
        print("\n🎉 Pipeline completed successfully!")
        
        # Show summary
        summary = detector.get_validation_summary()
        print(f"\n📊 Pipeline Summary:")
        print(f"   - Stages validated: {len(summary['registered_schemas'])}")
        print(f"   - Artifacts stored: {len(summary['stored_artifacts'])}")
        print(f"   - Monitoring hooks: {summary['active_hooks']}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with exception: {e}")


def main():
    """Run all demos"""
    print("🔍 EarlyErrorDetector System Demo")
    print("This demo showcases comprehensive validation capabilities for data pipelines")
    
    try:
        demo_basic_validation()
        demo_hooks_and_monitoring() 
        demo_referential_integrity()
        demo_artifact_validation()
        demo_custom_validators()
        demo_decorator_usage()
        demo_complete_pipeline()
        
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()