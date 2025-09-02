"""
Comprehensive Unit Tests for Code Quality Fixes
===============================================

This test file validates all the code quality fixes implemented across the 
analysis_nlp and calibration_safety_governance modules. It ensures that:

1. Type hint imports work correctly
2. ErrorCodes and ArtifactPathBuilder classes initialize properly with __init__ methods
3. Static methods in auto_deactivation_monitor.py function correctly
4. Previously fixed issues don't regress (missing imports, unbound variables, etc.)
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest for basic functionality
    class pytest:
        @staticmethod
        def skip(reason):
            print(f"SKIPPED: {reason}")
            return
        
        @staticmethod
        def fail(message):
            raise AssertionError(message)
        
        @staticmethod
        def main(args):
            print("pytest not available - running basic tests")
            return 0

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
try:
    from unittest.mock import Mock, patch, MagicMock
except ImportError:
    # Basic mock for older Python versions
    class Mock:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# Test imports that were previously causing issues
try:
    from typing import Dict, Any, Optional, Union
    typing_imports_success = True
    typing_import_error = None
except ImportError as e:
    typing_imports_success = False
    typing_import_error = str(e)

# Test analysis_nlp imports
try:
    from analysis_nlp import (
        ErrorCodes,
        ArtifactPathBuilder, 
        BaseAnalysisContract,
        ProcessingResult,
        AnalysisError,
        ValidationError,
        ProcessingError,
        TimeoutError,
        COMPONENT_SUFFIXES
    )
    analysis_nlp_imports_success = True
    analysis_nlp_import_error = None
except ImportError as e:
    analysis_nlp_imports_success = False
    analysis_nlp_import_error = str(e)

# Test calibration_safety_governance imports
try:
    from calibration_safety_governance.auto_deactivation_monitor import (
        AutoDeactivationMonitor,
        StabilityDriftAnalyzer,
        EvidenceQualityTracker,
        PerformanceRegressionDetector,
        DeactivationTriggerType,
        DeactivationSeverity,
        MonitoringPoint,
        DeactivationEvent
    )
    calibration_imports_success = True
    calibration_import_error = None
except ImportError as e:
    calibration_imports_success = False
    calibration_import_error = str(e)


class TestTypeHintImports:
    """Test that type hint imports work correctly"""
    
    def test_typing_imports_successful(self):
        """Verify typing module imports don't fail"""
        assert typing_imports_success, f"Typing imports failed: {typing_import_error}"
    
    def test_dict_type_hint_usable(self):
        """Test that Dict type hint can be used"""
        def test_function(data: Dict[str, Any]) -> Dict[str, str]:
            return {str(k): str(v) for k, v in data.items()}
        
        result = test_function({"key": "value", "number": 42})
        assert result == {"key": "value", "number": "42"}
    
    def test_optional_type_hint_usable(self):
        """Test that Optional type hint can be used"""
        def test_function(value: Optional[str] = None) -> str:
            return value if value is not None else "default"
        
        assert test_function("test") == "test"
        assert test_function() == "default"
    
    def test_union_type_hint_usable(self):
        """Test that Union type hint can be used"""
        def test_function(value: Union[str, int]) -> str:
            return str(value)
        
        assert test_function("string") == "string"
        assert test_function(42) == "42"


class TestAnalysisNLPImports:
    """Test that analysis_nlp imports work correctly"""
    
    def test_analysis_nlp_imports_successful(self):
        """Verify analysis_nlp module imports don't fail"""
        assert analysis_nlp_imports_success, f"analysis_nlp imports failed: {analysis_nlp_import_error}"
    
    def test_all_required_classes_imported(self):
        """Verify all required classes are available"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        # Test that all classes exist and are classes
        assert isinstance(ErrorCodes, type), "ErrorCodes should be a class"
        assert isinstance(ArtifactPathBuilder, type), "ArtifactPathBuilder should be a class"
        assert isinstance(BaseAnalysisContract, type), "BaseAnalysisContract should be a class"
        assert isinstance(ProcessingResult, type), "ProcessingResult should be a class"
        
        # Test exception classes
        assert issubclass(AnalysisError, Exception), "AnalysisError should be an Exception subclass"
        assert issubclass(ValidationError, AnalysisError), "ValidationError should inherit from AnalysisError"
        assert issubclass(ProcessingError, AnalysisError), "ProcessingError should inherit from AnalysisError"
        assert issubclass(TimeoutError, AnalysisError), "TimeoutError should inherit from AnalysisError"
        
        # Test constants
        assert isinstance(COMPONENT_SUFFIXES, dict), "COMPONENT_SUFFIXES should be a dictionary"


class TestErrorCodesClass:
    """Test ErrorCodes class initialization and functionality"""
    
    def test_error_codes_class_instantiation(self):
        """Test that ErrorCodes can be instantiated (has proper __init__)"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        # Test class can be instantiated
        error_codes = ErrorCodes()
        assert error_codes is not None
    
    def test_error_codes_constants_exist(self):
        """Test that all expected error code constants exist"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        expected_codes = [
            "VALIDATION_FAILED",
            "PROCESSING_TIMEOUT", 
            "PROCESSING_ERROR",
            "INVALID_PDF_PATH",
            "INVALID_DOCUMENT_ID",
            "INVALID_OUTPUT_DIR",
            "ARTIFACT_GENERATION_FAILED",
            "INSUFFICIENT_RESOURCES",
            "DEPENDENCY_ERROR"
        ]
        
        for code in expected_codes:
            assert hasattr(ErrorCodes, code), f"ErrorCodes missing constant: {code}"
            assert isinstance(getattr(ErrorCodes, code), str), f"ErrorCodes.{code} should be a string"
    
    def test_error_codes_values_are_strings(self):
        """Test that error code values are proper strings"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        error_codes = ErrorCodes()
        
        # Test that constants can be accessed from instance
        assert error_codes.VALIDATION_FAILED == "VALIDATION_FAILED"
        assert error_codes.PROCESSING_ERROR == "PROCESSING_ERROR"
        assert error_codes.INVALID_PDF_PATH == "INVALID_PDF_PATH"


class TestArtifactPathBuilderClass:
    """Test ArtifactPathBuilder class initialization and functionality"""
    
    def test_artifact_path_builder_instantiation(self):
        """Test that ArtifactPathBuilder can be instantiated"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        # Test class can be instantiated
        builder = ArtifactPathBuilder()
        assert builder is not None
    
    def test_build_path_static_method(self):
        """Test that build_path works as a static method"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        # Test static method can be called without instantiation
        path = ArtifactPathBuilder.build_path("/tmp", "doc123", "_test")
        expected = str(Path("/tmp") / "doc123_test.json")
        assert path == expected
    
    def test_build_path_with_missing_underscore(self):
        """Test that build_path adds underscore prefix if missing"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        path = ArtifactPathBuilder.build_path("/tmp", "doc123", "test")
        expected = str(Path("/tmp") / "doc123_test.json")
        assert path == expected
    
    def test_get_suffix_for_component_static_method(self):
        """Test that get_suffix_for_component works as a static method"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        # Test known component
        suffix = ArtifactPathBuilder.get_suffix_for_component("adaptive_analyzer")
        assert suffix == "_adaptive"
        
        # Test unknown component gets default
        suffix = ArtifactPathBuilder.get_suffix_for_component("unknown_component")
        assert suffix == "_metadata"
    
    def test_artifact_path_builder_handles_pathlib_paths(self):
        """Test that ArtifactPathBuilder properly handles Path objects"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            path = ArtifactPathBuilder.build_path(temp_dir, "doc456", "_analysis")
            
            # Verify path is a string and properly formatted
            assert isinstance(path, str)
            assert path.endswith("doc456_analysis.json")
            assert temp_dir in path


class TestCalibrationSafetyGovernanceImports:
    """Test that calibration_safety_governance imports work correctly"""
    
    def test_calibration_imports_successful(self):
        """Verify calibration_safety_governance imports don't fail"""
        assert calibration_imports_success, f"calibration_safety_governance imports failed: {calibration_import_error}"
    
    def test_required_classes_available(self):
        """Test that all required classes are available"""
        if not calibration_imports_success:
            pytest.skip(f"calibration_safety_governance imports failed: {calibration_import_error}")
        
        # Test main classes exist
        assert isinstance(AutoDeactivationMonitor, type)
        assert isinstance(StabilityDriftAnalyzer, type)
        assert isinstance(EvidenceQualityTracker, type)
        assert isinstance(PerformanceRegressionDetector, type)
        
        # Test enum classes
        assert hasattr(DeactivationTriggerType, 'STABILITY_DRIFT')
        assert hasattr(DeactivationSeverity, 'CRITICAL')
    
    def test_dataclass_imports_work(self):
        """Test that dataclass imports work properly"""
        if not calibration_imports_success:
            pytest.skip(f"calibration_safety_governance imports failed: {calibration_import_error}")
            
        # Test that dataclasses can be instantiated
        monitoring_point = MonitoringPoint(
            timestamp=datetime.now(),
            metric_name="test_metric",
            value=0.5
        )
        assert monitoring_point.metric_name == "test_metric"
        assert monitoring_point.value == 0.5
        
        deactivation_event = DeactivationEvent(
            trigger_type=DeactivationTriggerType.STABILITY_DRIFT,
            severity=DeactivationSeverity.MAJOR,
            enhancement_id="test_enhancement",
            trigger_condition="test condition",
            metrics={"score": 0.8},
            timestamp=datetime.now(),
            cooldown_until=datetime.now() + timedelta(hours=1)
        )
        assert deactivation_event.enhancement_id == "test_enhancement"


class TestStaticMethodsInAutoDeactivationMonitor:
    """Test static methods in auto_deactivation_monitor.py"""
    
    def test_parse_duration_to_minutes_method_exists(self):
        """Test that _parse_duration_to_minutes method exists and works"""
        if not calibration_imports_success:
            pytest.skip(f"calibration_safety_governance imports failed: {calibration_import_error}")
            
        # Create instance to test the method
        monitor = AutoDeactivationMonitor()
        
        # Test various duration formats
        assert monitor._parse_duration_to_minutes("PT15M") == 15
        assert monitor._parse_duration_to_minutes("PT1H") == 60
        assert monitor._parse_duration_to_minutes("PT24H") == 1440
        assert monitor._parse_duration_to_minutes("PT2H") == 120
    
    def test_parse_duration_handles_invalid_input(self):
        """Test that _parse_duration_to_minutes handles invalid input"""
        if not calibration_imports_success:
            pytest.skip(f"calibration_safety_governance imports failed: {calibration_import_error}")
            
        monitor = AutoDeactivationMonitor()
        
        # Test invalid formats default to 60 minutes
        assert monitor._parse_duration_to_minutes("INVALID") == 60
        assert monitor._parse_duration_to_minutes("") == 60
        assert monitor._parse_duration_to_minutes("PT") == 60
    
    def test_analyze_quality_trend_method_exists(self):
        """Test that _analyze_quality_trend method exists in EvidenceQualityTracker"""
        if not calibration_imports_success:
            pytest.skip(f"calibration_safety_governance imports failed: {calibration_import_error}")
            
        tracker = EvidenceQualityTracker()
        
        # Test with sufficient data
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = tracker._analyze_quality_trend(scores)
        
        assert "trend" in result
        assert "recent_average" in result
        assert "early_average" in result
        assert "change_magnitude" in result
        assert result["trend"] == "declining"
    
    def test_analyze_quality_trend_handles_insufficient_data(self):
        """Test that _analyze_quality_trend handles insufficient data"""
        if not calibration_imports_success:
            pytest.skip(f"calibration_safety_governance imports failed: {calibration_import_error}")
            
        tracker = EvidenceQualityTracker()
        
        # Test with insufficient data
        result = tracker._analyze_quality_trend([0.5, 0.6])
        assert result["trend"] == "insufficient_data"
        
        result = tracker._analyze_quality_trend([])
        assert result["trend"] == "insufficient_data"


class TestRegressionPrevention:
    """Test cases that would catch the specific issues that were previously fixed"""
    
    def test_no_unbound_variable_errors_in_error_codes(self):
        """Test that ErrorCodes doesn't have unbound variable issues"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
        
        # Test that all error codes can be accessed without NameError
        try:
            codes = [
                ErrorCodes.VALIDATION_FAILED,
                ErrorCodes.PROCESSING_TIMEOUT,
                ErrorCodes.PROCESSING_ERROR,
                ErrorCodes.INVALID_PDF_PATH,
                ErrorCodes.INVALID_DOCUMENT_ID,
                ErrorCodes.INVALID_OUTPUT_DIR,
                ErrorCodes.ARTIFACT_GENERATION_FAILED,
                ErrorCodes.INSUFFICIENT_RESOURCES,
                ErrorCodes.DEPENDENCY_ERROR
            ]
            assert all(isinstance(code, str) for code in codes)
        except NameError as e:
            pytest.fail(f"NameError accessing ErrorCodes constants: {e}")
    
    def test_no_missing_import_errors(self):
        """Test that there are no missing import errors"""
        # This test passes if the imports at the top of the file succeeded
        assert typing_imports_success, f"Missing typing imports: {typing_import_error}"
        assert analysis_nlp_imports_success, f"Missing analysis_nlp imports: {analysis_nlp_import_error}" 
        assert calibration_imports_success, f"Missing calibration imports: {calibration_import_error}"
    
    def test_dataclass_initialization_works(self):
        """Test that dataclasses can be properly initialized (no missing __init__ methods)"""
        if not calibration_imports_success:
            pytest.skip(f"calibration_safety_governance imports failed: {calibration_import_error}")
            
        # Test MonitoringPoint dataclass initialization
        try:
            point = MonitoringPoint(
                timestamp=datetime.now(),
                metric_name="test",
                value=1.0,
                metadata={"key": "value"}
            )
            assert point.metric_name == "test"
            assert point.value == 1.0
            assert point.metadata == {"key": "value"}
        except TypeError as e:
            pytest.fail(f"MonitoringPoint initialization failed: {e}")
            
        # Test DeactivationEvent dataclass initialization
        try:
            event = DeactivationEvent(
                trigger_type=DeactivationTriggerType.STABILITY_DRIFT,
                severity=DeactivationSeverity.MINOR,
                enhancement_id="test",
                trigger_condition="test condition",
                metrics={"test": 1.0},
                timestamp=datetime.now(),
                cooldown_until=datetime.now() + timedelta(minutes=30),
                metadata={"test": "data"}
            )
            assert event.enhancement_id == "test"
            assert event.metadata == {"test": "data"}
        except TypeError as e:
            pytest.fail(f"DeactivationEvent initialization failed: {e}")
    
    def test_static_methods_dont_require_instance(self):
        """Test that static methods work without class instantiation"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
        
        # Test ArtifactPathBuilder static methods work without instantiation
        try:
            path = ArtifactPathBuilder.build_path("/test", "doc", "_suffix")
            assert isinstance(path, str)
            
            suffix = ArtifactPathBuilder.get_suffix_for_component("test_component")
            assert isinstance(suffix, str)
        except TypeError as e:
            pytest.fail(f"Static methods require instance when they shouldn't: {e}")
    
    def test_exception_classes_have_proper_inheritance(self):
        """Test that exception classes have proper inheritance and __init__ methods"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
        
        # Test AnalysisError
        try:
            error = AnalysisError("test message", "TEST_CODE", "test_component")
            assert error.message == "test message"
            assert error.error_code == "TEST_CODE"
            assert error.component == "test_component"
            assert isinstance(error.to_dict(), dict)
        except Exception as e:
            pytest.fail(f"AnalysisError initialization failed: {e}")
            
        # Test ValidationError
        try:
            error = ValidationError("validation failed", "test_comp", {"field": "error"})
            assert error.error_code == ErrorCodes.VALIDATION_FAILED
            assert error.validation_details == {"field": "error"}
            error_dict = error.to_dict()
            assert "validation_details" in error_dict
        except Exception as e:
            pytest.fail(f"ValidationError initialization failed: {e}")
    
    def test_processing_result_initialization(self):
        """Test that ProcessingResult class initializes properly"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
        
        try:
            result = ProcessingResult(
                status="success",
                artifacts={"output": "/path/to/artifact.json"},
                error_details={},
                processing_metadata={"component": "test"}
            )
            assert result.status == "success"
            assert result.artifacts == {"output": "/path/to/artifact.json"}
            
            # Test to_dict method
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "status" in result_dict
            assert "artifacts" in result_dict
            
            # Test to_json method
            json_str = result.to_json()
            assert isinstance(json_str, str)
            # Verify it's valid JSON
            parsed = json.loads(json_str)
            assert parsed["status"] == "success"
            
        except Exception as e:
            pytest.fail(f"ProcessingResult initialization failed: {e}")


class TestIntegrationScenarios:
    """Test integration scenarios that combine multiple fixed components"""
    
    def test_end_to_end_artifact_path_building(self):
        """Test end-to-end artifact path building with all components"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test building paths for all known components
            for component_name, expected_suffix in COMPONENT_SUFFIXES.items():
                path = ArtifactPathBuilder.build_path(temp_dir, "test_doc", expected_suffix)
                
                assert isinstance(path, str)
                assert path.endswith(f"test_doc{expected_suffix}.json")
                assert temp_dir in path
                
                # Verify path can be used with pathlib
                path_obj = Path(path)
                assert path_obj.parent == Path(temp_dir)
                assert path_obj.suffix == ".json"
    
    def test_monitoring_system_integration(self):
        """Test integration of monitoring system components"""
        if not calibration_imports_success:
            pytest.skip(f"calibration_safety_governance imports failed: {calibration_import_error}")
            
        # Create temporary thresholds file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            thresholds = {
                "stability_monitoring": {
                    "score_variance": {"stability_coefficient": 0.8},
                    "performance_regression": {
                        "response_time_increase": 1.5,
                        "accuracy_degradation": 0.05
                    }
                },
                "auto_deactivation": {
                    "triggers": {
                        "safety_violation": {"single_critical_failure": True},
                        "stability_breach": {"consecutive_violations": 3}
                    },
                    "cooldown_periods": {
                        "critical_deactivation": "PT24H",
                        "major_deactivation": "PT1H"
                    }
                }
            }
            json.dump(thresholds, f)
            temp_thresholds_path = f.name
        
        try:
            # Test AutoDeactivationMonitor initialization with thresholds
            monitor = AutoDeactivationMonitor(thresholds_path=temp_thresholds_path)
            assert monitor is not None
            
            # Test monitoring an enhancement
            performance_metrics = {
                "response_time": 0.5,
                "accuracy": 0.95,
                "throughput": 100.0,
                "error_rate": 0.01
            }
            
            evidence_quality = {
                "overall_quality": 0.9,
                "consistency": 0.85,
                "coverage": 0.95,
                "coherence": 0.88
            }
            
            result = monitor.monitor_enhancement(
                enhancement_id="test_enhancement",
                performance_metrics=performance_metrics,
                evidence_quality=evidence_quality,
                score=0.92
            )
            
            assert "enhancement_id" in result
            assert "monitoring_results" in result
            assert "deactivation_decision" in result
            assert result["enhancement_id"] == "test_enhancement"
            
            # Test status retrieval
            status = monitor.get_enhancement_status("test_enhancement")
            assert "enhancement_id" in status
            assert "is_active" in status
            
        finally:
            # Clean up temporary file
            Path(temp_thresholds_path).unlink()
    
    def test_error_handling_integration(self):
        """Test that error handling works across all components"""
        if not analysis_nlp_imports_success:
            pytest.skip(f"analysis_nlp imports failed: {analysis_nlp_import_error}")
        
        # Test that exceptions can be properly created and converted to dict
        validation_error = ValidationError(
            "Invalid input parameters",
            "test_component",
            {"pdf_path": "File not found", "document_id": "Empty ID"}
        )
        
        error_dict = validation_error.to_dict()
        assert error_dict["error_code"] == ErrorCodes.VALIDATION_FAILED
        assert error_dict["component"] == "test_component"
        assert "validation_details" in error_dict
        assert error_dict["validation_details"]["pdf_path"] == "File not found"
        
        # Test ProcessingResult with error
        result = ProcessingResult(
            status="failed",
            error_details=error_dict,
            processing_metadata={"component": "test_component"}
        )
        
        result_dict = result.to_dict()
        assert result_dict["status"] == "failed"
        assert "error_details" in result_dict
        assert result_dict["error_details"]["error_code"] == ErrorCodes.VALIDATION_FAILED


def run_basic_tests():
    """Run basic tests without pytest"""
    print("=" * 60)
    print("Running Code Quality Fixes Validation Tests")
    print("=" * 60)
    
    test_classes = [
        TestTypeHintImports,
        TestAnalysisNLPImports, 
        TestErrorCodesClass,
        TestArtifactPathBuilderClass,
        TestCalibrationSafetyGovernanceImports,
        TestStaticMethodsInAutoDeactivationMonitor,
        TestRegressionPrevention,
        TestIntegrationScenarios
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            method = getattr(instance, test_method)
            
            try:
                print(f"  {test_method}...", end=" ")
                method()
                print("PASS")
                passed_tests += 1
            except AssertionError as e:
                print(f"FAIL: {e}")
                failed_tests += 1
            except Exception as e:
                if "skip" in str(e).lower():
                    print("SKIP")
                    skipped_tests += 1
                else:
                    print(f"ERROR: {e}")
                    failed_tests += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Skipped: {skipped_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
    
    return failed_tests == 0


if __name__ == "__main__":
    if HAS_PYTEST:
        # Run tests with verbose output
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        # Run basic tests without pytest
        success = run_basic_tests()
        exit(0 if success else 1)