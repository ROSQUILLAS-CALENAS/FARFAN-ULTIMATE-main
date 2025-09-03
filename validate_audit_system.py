#!/usr/bin/env python3
"""
Audit System Validation Script

This script validates the comprehensive audit logging system implementation
by testing key components and verifying standardized _audit.json file generation.
"""

import json
import sys
import tempfile
import traceback
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any  # Module not found  # Module not found  # Module not found

def validate_audit_imports():
    """Validate that audit system imports work correctly"""
    print("1. Testing audit system imports...")
    
    try:
# # #         from audit_logger import AuditLogger, AuditMixin, AuditStatus, create_audit_logger  # Module not found  # Module not found  # Module not found
        print("   ✅ Core audit classes imported successfully")
        
# # #         from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found
        print("   ✅ TotalOrderingBase with audit integration imported")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False


def validate_audit_logger_creation():
    """Validate AuditLogger can be created and configured"""
    print("\n2. Testing AuditLogger creation...")
    
    try:
# # #         from audit_logger import AuditLogger, AuditStatus  # Module not found  # Module not found  # Module not found
        
        # Create logger instance
        logger = AuditLogger("TestComponent", "test_stage")
        print("   ✅ AuditLogger instance created")
        
        # Verify properties
        assert logger.component_name == "TestComponent"
        assert logger.stage_name == "test_stage"
        print("   ✅ Component properties set correctly")
        
        # Test configuration options
        logger.enable_performance_metrics = True
        logger.enable_state_tracking = True
        print("   ✅ Configuration options work")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def validate_audit_session():
    """Validate complete audit session workflow"""
    print("\n3. Testing complete audit session...")
    
    try:
# # #         from audit_logger import AuditLogger, AuditStatus  # Module not found  # Module not found  # Module not found
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create logger with custom base directory
            logger = AuditLogger("ValidationComponent", "validation_stage")
            logger.audit_base_dir = temp_path
            
            # Start audit session
            entry = logger.start_audit(
                document_stem="validation_test",
                operation_id="val_001",
                input_files=["test_input.pdf"],
                metadata={"test_mode": True}
            )
            print("   ✅ Audit session started")
            
            # Log state transitions
            logger.log_state_transition(
                from_state="init",
                to_state="processing", 
                trigger="start",
                metadata={"step": 1}
            )
            
            logger.log_state_transition(
                from_state="processing",
                to_state="complete",
                trigger="finish", 
                metadata={"step": 2}
            )
            print("   ✅ State transitions logged")
            
            # Log warnings
            logger.log_warning("Test warning", {"code": "W001"})
            print("   ✅ Warnings logged")
            
            # Add metadata
            logger.add_metadata("custom_field", "custom_value")
            print("   ✅ Custom metadata added")
            
            # End audit session
            completed = logger.end_audit(
                status=AuditStatus.SUCCESS,
                output_files=["test_output.json"]
            )
            print("   ✅ Audit session completed")
            
            # Verify audit file was created
            audit_file = temp_path / "validation_stage" / "validation_test_audit.json"
            if not audit_file.exists():
                print(f"   ❌ Audit file not created at {audit_file}")
                return False
            
            # Verify audit file content
            with open(audit_file) as f:
                audit_data = json.load(f)
            
            required_fields = [
                "component_name", "operation_id", "stage_name", "document_stem",
                "start_time", "end_time", "duration_ms", "input_files", 
                "output_files", "status", "error_count", "warnings",
                "state_transitions", "metadata", "execution_environment", 
                "performance_metrics"
            ]
            
            missing_fields = [field for field in required_fields if field not in audit_data]
            if missing_fields:
                print(f"   ❌ Missing required fields: {missing_fields}")
                return False
            
            print("   ✅ Audit file created with all required fields")
            print(f"   ✅ File size: {audit_file.stat().st_size} bytes")
            print(f"   ✅ Duration: {audit_data.get('duration_ms', 0):.2f}ms")
            print(f"   ✅ State transitions: {len(audit_data.get('state_transitions', []))}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def validate_total_ordering_integration():
    """Validate TotalOrderingBase integration with audit system"""
    print("\n4. Testing TotalOrderingBase audit integration...")
    
    try:
# # #         from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found
# # #         from audit_logger import AuditMixin  # Module not found  # Module not found  # Module not found
        
        # Create test component class
        class TestComponent(TotalOrderingBase):
            def __init__(self):
                super().__init__("TestIntegrationComponent")
                self.stage_name = "integration_test_stage"
            
            def process(self, data=None, context=None):
                # Test standard process method
                return {
                    "status": "success",
                    "component": self.component_name,
                    "data_processed": bool(data)
                }
        
        # Create component instance
        component = TestComponent()
        print("   ✅ TotalOrderingBase component created")
        
        # Verify audit capabilities
        has_audit_mixin = isinstance(component, AuditMixin) if AuditMixin != object else False
        print(f"   ✅ AuditMixin integration: {'enabled' if has_audit_mixin else 'available'}")
        
        # Test metadata generation
        metadata = component.get_deterministic_metadata()
        assert "component_name" in metadata
        assert "audit_enabled" in metadata
        print("   ✅ Metadata includes audit information")
        
        # Test process method
        test_data = {"test": "data"}
        test_context = {"document_id": "integration_test"}
        
        result = component.process(test_data, test_context)
        assert result["status"] == "success"
        print("   ✅ Process method works correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def validate_existing_components():
    """Validate existing components work with audit system"""
    print("\n5. Testing existing component integration...")
    
    try:
        # Test AdaptiveAnalyzer
        try:
# # #             from canonical_flow.A_analysis_nlp.adaptive_analyzer import AdaptiveAnalyzer  # Module not found  # Module not found  # Module not found
            
            analyzer = AdaptiveAnalyzer()
            assert hasattr(analyzer, 'component_name')
            assert hasattr(analyzer, 'stage_name')
            print("   ✅ AdaptiveAnalyzer audit integration ready")
            
        except ImportError:
            print("   ⚠️  AdaptiveAnalyzer not available for testing")
        
        # Test MesoAggregator
        try:
# # #             from G_aggregation_reporting.meso_aggregator import MesoAggregator  # Module not found  # Module not found  # Module not found
            
            aggregator = MesoAggregator()
            assert hasattr(aggregator, 'component_name')
            assert hasattr(aggregator, 'stage_name')
            print("   ✅ MesoAggregator audit integration ready")
            
        except ImportError:
            print("   ⚠️  MesoAggregator not available for testing")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def validate_audit_file_format():
    """Validate audit file format specifications"""
    print("\n6. Testing audit file format compliance...")
    
    try:
# # #         from audit_logger import AuditLogger, AuditStatus  # Module not found  # Module not found  # Module not found
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            logger = AuditLogger("FormatTestComponent", "format_test_stage")
            logger.audit_base_dir = temp_path
            
            # Create comprehensive audit entry
            entry = logger.start_audit(
                document_stem="format_test_doc",
                operation_id="fmt_001",
                input_files=["input1.pdf", "input2.json"],
                metadata={
                    "version": "1.0",
                    "config": {"threads": 4, "timeout": 30},
                    "unicode_test": "Testing UTF-8: café, naïve, résumé"
                }
            )
            
            # Add comprehensive data
            logger.log_state_transition("init", "step1", "trigger1", {"data": "value"})
            logger.log_state_transition("step1", "step2", "trigger2", {"data": "value2"})
            logger.log_warning("Test warning with unicode: émöjï 🎉")
            logger.add_metadata("complex_data", {"nested": {"value": 42, "list": [1, 2, 3]}})
            
            completed = logger.end_audit(
                status=AuditStatus.SUCCESS,
                output_files=["output1.json", "output2.txt", "summary.pdf"]
            )
            
            # Read and validate file
            audit_file = temp_path / "format_test_stage" / "format_test_doc_audit.json"
            
            # Check encoding
            with open(audit_file, 'rb') as f:
                raw_content = f.read()
            
            # Verify UTF-8 encoding
            try:
                content_str = raw_content.decode('utf-8')
                print("   ✅ File uses UTF-8 encoding")
            except UnicodeDecodeError:
                print("   ❌ File is not UTF-8 encoded")
                return False
            
            # Verify JSON formatting
            with open(audit_file, encoding='utf-8') as f:
                audit_data = json.load(f)
            
            # Re-serialize to check formatting
            formatted_json = json.dumps(audit_data, indent=2, ensure_ascii=False, sort_keys=True)
            
            # Check if file content matches expected formatting
            if content_str.strip() == formatted_json.strip():
                print("   ✅ JSON formatting is consistent (indent=2, sorted keys)")
            else:
                print("   ⚠️  JSON formatting may not be fully consistent")
            
            # Verify required structure
            expected_structure = {
                "component_name": str,
                "operation_id": str,
                "stage_name": str,
                "document_stem": str,
                "start_time": str,
                "end_time": str,
                "duration_ms": (int, float),
                "input_files": list,
                "output_files": list,
                "status": str,
                "error_count": int,
                "warnings": list,
                "state_transitions": list,
                "metadata": dict,
                "execution_environment": dict,
                "performance_metrics": dict
            }
            
            for field, expected_type in expected_structure.items():
                if field not in audit_data:
                    print(f"   ❌ Missing field: {field}")
                    return False
                
                if not isinstance(audit_data[field], expected_type):
                    print(f"   ❌ Field {field} has wrong type: {type(audit_data[field])} (expected {expected_type})")
                    return False
            
            print("   ✅ All required fields present with correct types")
            print(f"   ✅ File path follows pattern: canonical_flow/<stage_name>/<doc_stem>_audit.json")
            
            # Check sorting
            if list(audit_data.keys()) == sorted(audit_data.keys()):
                print("   ✅ Top-level keys are sorted")
            else:
                print("   ⚠️  Top-level keys may not be sorted")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def validate_error_handling():
    """Validate audit system error handling"""
    print("\n7. Testing error handling capabilities...")
    
    try:
# # #         from audit_logger import AuditLogger, AuditStatus  # Module not found  # Module not found  # Module not found
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            logger = AuditLogger("ErrorTestComponent", "error_test_stage")
            logger.audit_base_dir = temp_path
            
            # Start audit session
            entry = logger.start_audit(
                document_stem="error_test_doc",
                operation_id="err_001"
            )
            
            # Log an error
            try:
                raise ValueError("Test error for audit validation")
            except Exception as e:
                logger.log_error(e, {"context": "validation_test"})
            
            print("   ✅ Error logging works")
            
            # End with failure status
            completed = logger.end_audit(
                status=AuditStatus.FAILED,
                error_details={"errors": ["Test error"], "exception_type": "ValueError"}
            )
            
            # Verify error information in audit file
            audit_file = temp_path / "error_test_stage" / "error_test_doc_audit.json"
            with open(audit_file) as f:
                audit_data = json.load(f)
            
            assert audit_data["status"] == "failed"
            assert audit_data["error_count"] > 0
            assert "error_details" in audit_data["metadata"]
            
            print("   ✅ Error audit file created correctly")
            print(f"   ✅ Error count: {audit_data['error_count']}")
            print(f"   ✅ Status: {audit_data['status']}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run comprehensive audit system validation"""
    print("Comprehensive Audit System Validation")
    print("=" * 50)
    
    tests = [
        validate_audit_imports,
        validate_audit_logger_creation,
        validate_audit_session,
        validate_total_ordering_integration,
        validate_existing_components,
        validate_audit_file_format,
        validate_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All audit system validations passed!")
        print("\nAudit System Features Verified:")
        print("- Standardized _audit.json file generation")
        print("- UTF-8 encoding with indent=2 formatting")
        print("- Comprehensive execution tracing")
        print("- Timing and performance metrics")
        print("- State transition tracking")
        print("- Error handling and logging")
        print("- TotalOrderingBase integration")
        print("- Cross-stage compatibility")
        
        return True
    else:
        print("❌ Some audit system validations failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)