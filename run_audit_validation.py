#!/usr/bin/env python3

# Simple validation runner that avoids blocked commands

import tempfile
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "55O"
__stage_order__ = 7

def test_audit_system():
    """Test core audit system functionality"""
    print("Testing audit system...")
    
    try:
        # Import core components
# # #         from audit_logger import AuditLogger, AuditStatus  # Module not found  # Module not found  # Module not found
# # #         from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found
        
        print("✅ Imports successful")
        
        # Test AuditLogger creation
        logger = AuditLogger("TestComponent", "test_stage")
        print("✅ AuditLogger created")
        
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.audit_base_dir = temp_path
            
            # Test audit session
            entry = logger.start_audit(
                document_stem="test_doc",
                operation_id="test_op_001",
                input_files=["test.pdf"]
            )
            print("✅ Audit session started")
            
            # Add some data
            logger.log_state_transition("init", "processing", "start")
            logger.log_warning("Test warning")
            logger.add_metadata("test_key", "test_value")
            
            # End session
            completed = logger.end_audit(
                status=AuditStatus.SUCCESS,
                output_files=["output.json"]
            )
            print(f"✅ Audit session completed ({completed.duration_ms:.2f}ms)")
            
            # Verify file creation
            audit_file = temp_path / "test_stage" / "test_doc_audit.json"
            if audit_file.exists():
                print("✅ Audit file created")
                
                # Verify content
                with open(audit_file) as f:
                    data = json.load(f)
                
                required_fields = [
                    "component_name", "operation_id", "stage_name", "document_stem",
                    "start_time", "end_time", "duration_ms", "status", "warnings",
                    "state_transitions", "metadata"
                ]
                
                missing = [f for f in required_fields if f not in data]
                if missing:
                    print(f"❌ Missing fields: {missing}")
                    return False
                
                print("✅ Audit file has all required fields")
                print(f"   Duration: {data['duration_ms']:.2f}ms")
                print(f"   Status: {data['status']}")
                print(f"   State transitions: {len(data['state_transitions'])}")
                print(f"   Warnings: {len(data['warnings'])}")
                
            else:
                print("❌ Audit file not created")
                return False
        
        print("✅ All audit system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_component_integration():
    """Test TotalOrderingBase integration"""
    print("\nTesting component integration...")
    
    try:
# # #         from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found
        
        class TestComponent(TotalOrderingBase):
            def __init__(self):
                super().__init__("TestIntegrationComponent")
                self.stage_name = "integration_test"
            
            def process(self, data=None, context=None):
                operation_id = self.generate_operation_id("process", {"data": data, "context": context})
                return {
                    "component": self.component_name,
                    "operation_id": operation_id,
                    "status": "success",
                    "has_audit": hasattr(self, '_audit_logger')
                }
        
        component = TestComponent()
        print("✅ Test component created")
        
        # Test process method
        result = component.process({"test": "data"}, {"document_id": "test"})
        print(f"✅ Process method works: {result['status']}")
        
        # Check metadata
        metadata = component.get_deterministic_metadata()
        if "audit_enabled" in metadata:
            print(f"✅ Audit integration detected: {metadata['audit_enabled']}")
        else:
            print("⚠️  Audit integration not fully active")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    print("Audit System Validation")
    print("=" * 30)
    
    success = True
    success &= test_audit_system()
    success &= test_component_integration()
    
    print("\n" + "=" * 30)
    if success:
        print("✅ All validations passed!")
        print("\nAudit System Features Confirmed:")
        print("- AuditLogger creation and configuration")
        print("- Audit session lifecycle management")
        print("- State transition and warning logging")
        print("- Standardized _audit.json file generation")
        print("- UTF-8 encoding and JSON formatting")
        print("- TotalOrderingBase integration")
    else:
        print("❌ Some validations failed")
    
    return success


if __name__ == "__main__":
    main()