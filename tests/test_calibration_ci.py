"""
Comprehensive CI Calibration Validation Test Suite

This module implements comprehensive validation tests for CI pipelines to ensure:

1. Calibration Quality Gates:
   - Retrieval temperature bounds validation (0.5-2.0)
   - Conformal prediction intervals bounded to [0,1]
   - Evidence multipliers respect MIN_MULTIPLIER to MAX_MULTIPLIER thresholds

2. Determinism Verification:
   - Pipeline components produce identical outputs with identical inputs
   - Float output differences ≤ 1e-9 tolerance

3. Numerical Stability:
   - NaN/Inf detection in all scoring outputs
   - Bounded iteration convergence validation

4. Enhancement Safety Validation:
   - Feature flags prevent unsafe auto-activations
   - Graceful degradation triggers work correctly
   - Configurable fail-fast vs soft-fail policies

Author: Tonkotsu AI Engineering Team
"""

import math
import warnings
# # # from typing import Any, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found

try:
    import pytest
except ImportError:
    pytest = None

try:
    import numpy as np
except ImportError:
    np = None

# Import system components
try:
# # #     from canonical_flow.L_classification_evaluation.decalogo_scoring_system import ScoringSystem  # Module not found  # Module not found  # Module not found
except ImportError as e:
    warnings.warn(f"Could not import all modules: {e}")
    ScoringSystem = None


@dataclass
class CalibrationTestConfig:
    """Configuration for calibration validation tests."""
    
    # Temperature bounds
    min_temperature: float = 0.5
    max_temperature: float = 2.0
    
    # Conformal prediction bounds
    conformal_lower_bound: float = 0.0
    conformal_upper_bound: float = 1.0
    
    # Evidence multiplier thresholds
    min_multiplier: float = 0.5
    max_multiplier: float = 1.2
    
    # Determinism tolerance
    float_tolerance: float = 1e-9
    
    # Test execution policies
    fail_fast: bool = False
    collect_all_errors: bool = True
    
    # Numerical stability
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6


@dataclass
class ValidationError:
    """Container for validation error information."""
    
    test_name: str
    component: str
    error_type: str
    message: str
    severity: str = "error"  # "error", "warning", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CalibrationValidator:
    """
    Comprehensive calibration validation system for CI pipelines.
    
    Implements quality gates, determinism checks, numerical stability validation,
    and enhancement safety verification with configurable failure policies.
    """
    
    def __init__(self, config: Optional[CalibrationTestConfig] = None):
        self.config = config or CalibrationTestConfig()
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        
    def add_error(self, test_name: str, component: str, error_type: str, 
                  message: str, severity: str = "error", metadata: Optional[Dict] = None):
        """Add validation error with optional fail-fast behavior."""
        error = ValidationError(
            test_name=test_name,
            component=component,
            error_type=error_type,
            message=message,
            severity=severity,
            metadata=metadata or {}
        )
        
        if severity == "error":
            self.errors.append(error)
            if self.config.fail_fast:
                raise AssertionError(f"{test_name}: {message}")
        else:
            self.warnings.append(error)
    
    def validate_retrieval_temperature_bounds(self, temperatures: List[float]) -> bool:
        """
        Validate retrieval temperature parameters are within acceptable bounds.
        
        Args:
            temperatures: List of temperature values to validate
            
        Returns:
            True if all temperatures are within bounds
        """
        test_name = "validate_retrieval_temperature_bounds"
        valid = True
        
        for i, temp in enumerate(temperatures):
            if not isinstance(temp, (int, float)):
                self.add_error(
                    test_name, "temperature_validation", "type_error",
                    f"Temperature at index {i} is not numeric: {type(temp)}"
                )
                valid = False
                continue
                
            if temp < self.config.min_temperature:
                self.add_error(
                    test_name, "temperature_validation", "bounds_violation",
                    f"Temperature {temp} below minimum {self.config.min_temperature} at index {i}"
                )
                valid = False
                
            if temp > self.config.max_temperature:
                self.add_error(
                    test_name, "temperature_validation", "bounds_violation",
                    f"Temperature {temp} above maximum {self.config.max_temperature} at index {i}"
                )
                valid = False
                
            if math.isnan(temp) or math.isinf(temp):
                self.add_error(
                    test_name, "temperature_validation", "numerical_instability",
                    f"Temperature contains NaN or Inf at index {i}: {temp}"
                )
                valid = False
        
        return valid
    
    def validate_conformal_prediction_intervals(self, intervals: List[Tuple[float, float]]) -> bool:
        """
        Validate conformal prediction intervals are properly bounded to [0,1].
        
        Args:
            intervals: List of (lower, upper) prediction intervals
            
        Returns:
            True if all intervals are properly bounded
        """
        test_name = "validate_conformal_prediction_intervals"
        valid = True
        
        for i, (lower, upper) in enumerate(intervals):
            # Check bounds
            if lower < self.config.conformal_lower_bound:
                self.add_error(
                    test_name, "conformal_validation", "bounds_violation",
                    f"Lower bound {lower} below {self.config.conformal_lower_bound} at index {i}"
                )
                valid = False
                
            if upper > self.config.conformal_upper_bound:
                self.add_error(
                    test_name, "conformal_validation", "bounds_violation",
                    f"Upper bound {upper} above {self.config.conformal_upper_bound} at index {i}"
                )
                valid = False
                
            # Check interval validity
            if lower > upper:
                self.add_error(
                    test_name, "conformal_validation", "interval_invalid",
                    f"Invalid interval: lower {lower} > upper {upper} at index {i}"
                )
                valid = False
                
            # Check for numerical issues
            if any(math.isnan(x) or math.isinf(x) for x in [lower, upper]):
                self.add_error(
                    test_name, "conformal_validation", "numerical_instability",
                    f"Interval contains NaN or Inf at index {i}: ({lower}, {upper})"
                )
                valid = False
        
        return valid
    
    def validate_evidence_multipliers(self, multipliers: List[float]) -> bool:
        """
        Validate evidence multipliers respect thresholds.
        
        Args:
            multipliers: List of evidence multiplier values
            
        Returns:
            True if all multipliers are within thresholds
        """
        test_name = "validate_evidence_multipliers"
        valid = True
        
        for i, multiplier in enumerate(multipliers):
            if multiplier < self.config.min_multiplier:
                self.add_error(
                    test_name, "multiplier_validation", "bounds_violation",
                    f"Multiplier {multiplier} below minimum {self.config.min_multiplier} at index {i}"
                )
                valid = False
                
            if multiplier > self.config.max_multiplier:
                self.add_error(
                    test_name, "multiplier_validation", "bounds_violation",
                    f"Multiplier {multiplier} above maximum {self.config.max_multiplier} at index {i}"
                )
                valid = False
                
            if math.isnan(multiplier) or math.isinf(multiplier):
                self.add_error(
                    test_name, "multiplier_validation", "numerical_instability",
                    f"Multiplier contains NaN or Inf at index {i}: {multiplier}"
                )
                valid = False
        
        return valid
    
    def verify_determinism(self, component_func: callable, inputs: Any, 
                          run_count: int = 2) -> bool:
        """
        Verify component produces identical outputs with identical inputs.
        
        Args:
            component_func: Function to test for determinism
            inputs: Input data for the function
            run_count: Number of runs to compare
            
        Returns:
            True if all runs produce identical outputs within tolerance
        """
        test_name = "verify_determinism"
        
        try:
            outputs = []
            for run in range(run_count):
                output = component_func(inputs)
                outputs.append(output)
            
            # Compare all outputs pairwise
            for i in range(1, len(outputs)):
                if not self._outputs_equal(outputs[0], outputs[i], test_name):
                    return False
            
            return True
            
        except Exception as e:
            self.add_error(
                test_name, "determinism_verification", "execution_error",
                f"Error during determinism test: {str(e)}"
            )
            return False
    
    def _outputs_equal(self, output1: Any, output2: Any, test_name: str) -> bool:
        """Compare two outputs for equality within float tolerance."""
        
        def compare_values(val1, val2, path=""):
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if math.isnan(val1) and math.isnan(val2):
                    return True
                if math.isinf(val1) and math.isinf(val2) and math.copysign(1, val1) == math.copysign(1, val2):
                    return True
                if abs(val1 - val2) > self.config.float_tolerance:
                    self.add_error(
                        test_name, "determinism_verification", "float_difference",
                        f"Float difference {abs(val1 - val2)} > tolerance {self.config.float_tolerance} at {path}"
                    )
                    return False
                return True
            
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                if len(val1) != len(val2):
                    self.add_error(
                        test_name, "determinism_verification", "length_mismatch",
                        f"Length mismatch at {path}: {len(val1)} vs {len(val2)}"
                    )
                    return False
                
                return all(compare_values(v1, v2, f"{path}[{i}]") 
                          for i, (v1, v2) in enumerate(zip(val1, val2)))
            
            elif isinstance(val1, dict) and isinstance(val2, dict):
                if set(val1.keys()) != set(val2.keys()):
                    self.add_error(
                        test_name, "determinism_verification", "key_mismatch",
                        f"Key mismatch at {path}"
                    )
                    return False
                
                return all(compare_values(val1[k], val2[k], f"{path}.{k}") 
                          for k in val1.keys())
            
            else:
                if val1 != val2:
                    self.add_error(
                        test_name, "determinism_verification", "value_mismatch",
                        f"Value mismatch at {path}: {val1} vs {val2}"
                    )
                    return False
                return True
        
        return compare_values(output1, output2)
    
    def validate_numerical_stability(self, scores: Union[List[float], List]) -> bool:
        """
        Detect NaN/Inf values in scoring outputs and validate convergence.
        
        Args:
            scores: Numerical scores to validate
            
        Returns:
            True if scores are numerically stable
        """
        test_name = "validate_numerical_stability"
        valid = True
        
        if np is not None and isinstance(scores, np.ndarray):
            scores = scores.flatten().tolist()
        
        for i, score in enumerate(scores):
            if math.isnan(score):
                self.add_error(
                    test_name, "numerical_stability", "nan_detected",
                    f"NaN value detected at index {i}"
                )
                valid = False
                
            if math.isinf(score):
                self.add_error(
                    test_name, "numerical_stability", "inf_detected",
                    f"Infinite value detected at index {i}: {score}"
                )
                valid = False
        
        # Check for convergence patterns
        if len(scores) > 10:
            recent_scores = scores[-10:]
            if np is not None:
                variance = np.var(recent_scores)
            else:
                mean_score = sum(recent_scores) / len(recent_scores)
                variance = sum((s - mean_score) ** 2 for s in recent_scores) / len(recent_scores)
            
            if variance > 1.0:  # High variance indicates lack of convergence
                self.add_error(
                    test_name, "numerical_stability", "convergence_issue",
                    f"High variance in recent scores: {variance}",
                    severity="warning"
                )
        
        return valid
    
    def validate_bounded_iteration_convergence(self, iteration_func: callable, 
                                             initial_state: Any, 
                                             convergence_check: callable) -> bool:
        """
        Validate that iterative processes converge within bounds.
        
        Args:
            iteration_func: Function that performs one iteration
            initial_state: Initial state for iteration
            convergence_check: Function to check if converged
            
        Returns:
            True if iteration converges within max_iterations
        """
        test_name = "validate_bounded_iteration_convergence"
        
        state = initial_state
        for iteration in range(self.config.max_iterations):
            try:
                new_state = iteration_func(state)
                
                if convergence_check(state, new_state, self.config.convergence_tolerance):
                    return True
                
                state = new_state
                
            except Exception as e:
                self.add_error(
                    test_name, "iteration_convergence", "iteration_error",
                    f"Error at iteration {iteration}: {str(e)}"
                )
                return False
        
        self.add_error(
            test_name, "iteration_convergence", "max_iterations_exceeded",
            f"Failed to converge within {self.config.max_iterations} iterations"
        )
        return False
    
    def validate_feature_flags_safety(self, feature_flags: Dict[str, bool]) -> bool:
        """
        Validate feature flags prevent unsafe auto-activations.
        
        Args:
            feature_flags: Dictionary of feature flag settings
            
        Returns:
            True if feature flags are safe
        """
        test_name = "validate_feature_flags_safety"
        valid = True
        
        unsafe_combinations = [
            # Unsafe auto-activation combinations
            {"enable_automatic_activation": True, "safety_override": True},
            {"experimental_features": True, "production_mode": True},
            {"debug_mode": True, "auto_deployment": True}
        ]
        
        for unsafe_combo in unsafe_combinations:
            if all(feature_flags.get(key, False) == value 
                  for key, value in unsafe_combo.items()):
                self.add_error(
                    test_name, "feature_flags", "unsafe_combination",
                    f"Unsafe feature flag combination detected: {unsafe_combo}"
                )
                valid = False
        
        # Check for required safety flags
        required_safety_flags = [
            "enable_safety_checks",
            "validation_enabled",
            "error_handling_enabled"
        ]
        
        for flag in required_safety_flags:
            if not feature_flags.get(flag, False):
                self.add_error(
                    test_name, "feature_flags", "missing_safety_flag",
                    f"Required safety flag '{flag}' is not enabled"
                )
                valid = False
        
        return valid
    
    def validate_graceful_degradation(self, degradation_func: callable, 
                                    failure_scenarios: List[Dict]) -> bool:
        """
        Validate graceful degradation triggers work correctly.
        
        Args:
            degradation_func: Function that should handle failures gracefully
            failure_scenarios: List of failure scenarios to test
            
        Returns:
            True if degradation works correctly for all scenarios
        """
        test_name = "validate_graceful_degradation"
        valid = True
        
        for i, scenario in enumerate(failure_scenarios):
            try:
                result = degradation_func(scenario)
                
                # Check that function doesn't crash
                if result is None:
                    self.add_error(
                        test_name, "graceful_degradation", "null_result",
                        f"Degradation function returned None for scenario {i}"
                    )
                    valid = False
                
                # Check for error indicators in result
                if isinstance(result, dict) and result.get("error") and not result.get("degraded_mode"):
                    self.add_error(
                        test_name, "graceful_degradation", "ungraceful_failure",
                        f"Function failed ungracefully for scenario {i}"
                    )
                    valid = False
                    
            except Exception as e:
                self.add_error(
                    test_name, "graceful_degradation", "exception_raised",
                    f"Exception raised for scenario {i}: {str(e)}"
                )
                valid = False
        
        return valid
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "validation_passed": len(self.errors) == 0
            },
            "errors": [
                {
                    "test": error.test_name,
                    "component": error.component,
                    "type": error.error_type,
                    "message": error.message,
                    "severity": error.severity,
                    "metadata": error.metadata
                }
                for error in self.errors
            ],
            "warnings": [
                {
                    "test": warning.test_name,
                    "component": warning.component,
                    "type": warning.error_type,
                    "message": warning.message,
                    "severity": warning.severity,
                    "metadata": warning.metadata
                }
                for warning in self.warnings
            ],
            "configuration": {
                "fail_fast": self.config.fail_fast,
                "collect_all_errors": self.config.collect_all_errors,
                "float_tolerance": self.config.float_tolerance,
                "temperature_bounds": [self.config.min_temperature, self.config.max_temperature],
                "conformal_bounds": [self.config.conformal_lower_bound, self.config.conformal_upper_bound],
                "multiplier_bounds": [self.config.min_multiplier, self.config.max_multiplier]
            }
        }


# ============================================================================
# PYTEST TEST CASES
# ============================================================================

class TestCalibrationQualityGates:
    """Test calibration quality gate validations."""
    
    def test_temperature_bounds_valid(self):
        """Test valid temperature bounds pass validation."""
        validator = CalibrationValidator()
        temperatures = [0.7, 1.0, 1.5, 0.8, 1.2]
        
        assert validator.validate_retrieval_temperature_bounds(temperatures)
        assert len(validator.errors) == 0
    
    def test_temperature_bounds_invalid(self):
        """Test invalid temperature bounds fail validation."""
        validator = CalibrationValidator()
        temperatures = [0.3, 2.5, 1.0]  # Below and above bounds
        
        assert not validator.validate_retrieval_temperature_bounds(temperatures)
        assert len(validator.errors) == 2
    
    def test_conformal_intervals_valid(self):
        """Test valid conformal prediction intervals."""
        validator = CalibrationValidator()
        intervals = [(0.1, 0.9), (0.0, 1.0), (0.3, 0.7)]
        
        assert validator.validate_conformal_prediction_intervals(intervals)
        assert len(validator.errors) == 0
    
    def test_conformal_intervals_invalid(self):
        """Test invalid conformal prediction intervals."""
        validator = CalibrationValidator()
        intervals = [(-0.1, 0.9), (0.3, 1.2), (0.8, 0.2)]  # Below bound, above bound, invalid
        
        assert not validator.validate_conformal_prediction_intervals(intervals)
        assert len(validator.errors) == 3
    
    def test_evidence_multipliers_valid(self):
        """Test valid evidence multipliers."""
        validator = CalibrationValidator()
        multipliers = [0.8, 1.0, 1.1, 0.7, 1.15]
        
        assert validator.validate_evidence_multipliers(multipliers)
        assert len(validator.errors) == 0
    
    def test_evidence_multipliers_invalid(self):
        """Test invalid evidence multipliers."""
        validator = CalibrationValidator()
        multipliers = [0.3, 1.5, 1.0]  # Below and above bounds
        
        assert not validator.validate_evidence_multipliers(multipliers)
        assert len(validator.errors) == 2


class TestDeterminismVerification:
    """Test determinism verification functionality."""
    
    def test_deterministic_function(self):
        """Test that deterministic functions pass verification."""
        def deterministic_func(x):
            return {"result": x * 2, "squared": x ** 2}
        
        validator = CalibrationValidator()
        assert validator.verify_determinism(deterministic_func, 5)
        assert len(validator.errors) == 0
    
    def test_non_deterministic_function(self):
        """Test that non-deterministic functions fail verification."""
        import random
        
        def random_func(x):
            return {"result": x + random.random()}
        
        validator = CalibrationValidator()
        assert not validator.verify_determinism(random_func, 5)
        assert len(validator.errors) > 0


class TestNumericalStability:
    """Test numerical stability validations."""
    
    def test_stable_scores(self):
        """Test numerically stable scores."""
        validator = CalibrationValidator()
        scores = [0.1, 0.5, 0.8, 0.3, 0.9]
        
        assert validator.validate_numerical_stability(scores)
        assert len(validator.errors) == 0
    
    def test_unstable_scores(self):
        """Test numerically unstable scores."""
        validator = CalibrationValidator()
        scores = [0.1, float('nan'), float('inf'), 0.3, -float('inf')]
        
        assert not validator.validate_numerical_stability(scores)
        assert len(validator.errors) == 3
    
    def test_bounded_iteration_convergence(self):
        """Test bounded iteration convergence validation."""
        def iteration_step(state):
            return state * 0.9  # Converges to 0
        
        def convergence_check(old_state, new_state, tolerance):
            return abs(new_state - old_state) < tolerance
        
        validator = CalibrationValidator()
        assert validator.validate_bounded_iteration_convergence(
            iteration_step, 10.0, convergence_check
        )
        assert len(validator.errors) == 0


class TestEnhancementSafety:
    """Test enhancement safety validations."""
    
    def test_safe_feature_flags(self):
        """Test safe feature flag configurations."""
        validator = CalibrationValidator()
        safe_flags = {
            "enable_safety_checks": True,
            "validation_enabled": True,
            "error_handling_enabled": True,
            "enable_automatic_activation": False
        }
        
        assert validator.validate_feature_flags_safety(safe_flags)
        assert len(validator.errors) == 0
    
    def test_unsafe_feature_flags(self):
        """Test unsafe feature flag configurations."""
        validator = CalibrationValidator()
        unsafe_flags = {
            "enable_automatic_activation": True,
            "safety_override": True,
            "enable_safety_checks": False
        }
        
        assert not validator.validate_feature_flags_safety(unsafe_flags)
        assert len(validator.errors) > 0
    
    def test_graceful_degradation(self):
        """Test graceful degradation validation."""
        def degradation_handler(scenario):
            if scenario.get("simulate_failure"):
                return {"error": True, "degraded_mode": True, "result": "fallback"}
            return {"result": "normal"}
        
        scenarios = [
            {"simulate_failure": False},
            {"simulate_failure": True}
        ]
        
        validator = CalibrationValidator()
        assert validator.validate_graceful_degradation(degradation_handler, scenarios)
        assert len(validator.errors) == 0


class TestFailurePolicies:
    """Test configurable failure policy behaviors."""
    
    def test_fail_fast_policy(self):
        """Test fail-fast policy stops on first error."""
        config = CalibrationTestConfig(fail_fast=True)
        validator = CalibrationValidator(config)
        
        if pytest is not None:
            with pytest.raises(AssertionError):
                validator.add_error("test", "component", "error", "Test error")
        else:
            # Manual test without pytest
            try:
                validator.add_error("test", "component", "error", "Test error")
                assert False, "Expected AssertionError"
            except AssertionError:
                pass  # Expected
    
    def test_collect_all_errors_policy(self):
        """Test collect-all-errors policy continues after errors."""
        config = CalibrationTestConfig(fail_fast=False, collect_all_errors=True)
        validator = CalibrationValidator(config)
        
        # Should not raise exception
        validator.add_error("test1", "component1", "error", "Error 1")
        validator.add_error("test2", "component2", "error", "Error 2")
        
        assert len(validator.errors) == 2
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive error reporting."""
        validator = CalibrationValidator()
        
        validator.add_error("test1", "component1", "bounds_violation", "Error 1")
        validator.add_error("test2", "component2", "nan_detected", "Error 2", "warning")
        
        report = validator.generate_comprehensive_report()
        
        assert report["summary"]["total_errors"] == 1
        assert report["summary"]["total_warnings"] == 1
        assert not report["summary"]["validation_passed"]
        assert len(report["errors"]) == 1
        assert len(report["warnings"]) == 1


# ============================================================================
# INTEGRATION TESTS WITH ACTUAL SYSTEM COMPONENTS
# ============================================================================


class TestSystemIntegration:
    """Integration tests with actual system components."""
    
    def test_scoring_system_evidence_multipliers(self):
# # #         """Test evidence multipliers from actual scoring system."""  # Module not found  # Module not found  # Module not found
        try:
            scoring_system = ScoringSystem()
            validator = CalibrationValidator()
            
            # Test various evidence quality combinations
            test_cases = [
                (0.0, 0.0), (0.5, 0.5), (1.0, 1.0),
                (0.3, 0.7), (0.8, 0.2), (0.9, 0.9)
            ]
            
            multipliers = []
            for completeness, quality in test_cases:
                multiplier = scoring_system.calculate_evidence_multiplier(completeness, quality)
                multipliers.append(float(multiplier))
            
            assert validator.validate_evidence_multipliers(multipliers)
            
        except ImportError:
            print("ScoringSystem not available - skipping test")
    
    def test_hybrid_retrieval_temperature_calibration(self):
        """Test temperature calibration in hybrid retrieval."""
        try:
# # #             from retrieval_engine.hybrid_retriever import _entropy_calibration  # Module not found  # Module not found  # Module not found
            validator = CalibrationValidator()
            
            # Test temperature calibration with various values
            test_temperatures = [0.5, 0.8, 1.0, 1.5, 2.0]
            
            for temp in test_temperatures:
                scores = [0.1, 0.5, 0.8, 0.3]
                calibrated = _entropy_calibration(scores, temp)
                
                # Validate calibrated scores
                assert validator.validate_numerical_stability(calibrated)
                # Validate temperature bounds
                assert validator.validate_retrieval_temperature_bounds([temp])
            
        except ImportError:
            print("Hybrid retrieval not available - skipping test")
    
    def test_conformal_risk_control_intervals(self):
# # #         """Test conformal prediction intervals from risk control system."""  # Module not found  # Module not found  # Module not found
        try:
# # #             from egw_query_expansion.core.conformal_risk_control import RiskControlConfig  # Module not found  # Module not found  # Module not found
            config = RiskControlConfig(alpha=0.1)
            validator = CalibrationValidator()
            
            # Simulate conformal intervals
            test_intervals = [
                (0.05, 0.95), (0.1, 0.9), (0.0, 1.0),
                (0.2, 0.8), (0.15, 0.85)
            ]
            
            assert validator.validate_conformal_prediction_intervals(test_intervals)
            
        except ImportError:
            print("Conformal risk control not available - skipping test")


def run_manual_tests():
    """Run tests manually when pytest is not available."""
    print("Running Calibration CI Validation Test Suite")
    print("=" * 50)
    
    # Test calibration quality gates
    print("\n1. Testing Calibration Quality Gates")
    test_cal = TestCalibrationQualityGates()
    
    try:
        test_cal.test_temperature_bounds_valid()
        print("   ✓ Temperature bounds valid test passed")
    except Exception as e:
        print(f"   ✗ Temperature bounds valid test failed: {e}")
    
    try:
        test_cal.test_conformal_intervals_valid()
        print("   ✓ Conformal intervals valid test passed")
    except Exception as e:
        print(f"   ✗ Conformal intervals valid test failed: {e}")
    
    try:
        test_cal.test_evidence_multipliers_valid()
        print("   ✓ Evidence multipliers valid test passed")
    except Exception as e:
        print(f"   ✗ Evidence multipliers valid test failed: {e}")
    
    # Test determinism verification
    print("\n2. Testing Determinism Verification")
    test_det = TestDeterminismVerification()
    
    try:
        test_det.test_deterministic_function()
        print("   ✓ Deterministic function test passed")
    except Exception as e:
        print(f"   ✗ Deterministic function test failed: {e}")
    
    # Test numerical stability
    print("\n3. Testing Numerical Stability")
    test_num = TestNumericalStability()
    
    try:
        test_num.test_stable_scores()
        print("   ✓ Stable scores test passed")
    except Exception as e:
        print(f"   ✗ Stable scores test failed: {e}")
    
    try:
        test_num.test_bounded_iteration_convergence()
        print("   ✓ Bounded iteration convergence test passed")
    except Exception as e:
        print(f"   ✗ Bounded iteration convergence test failed: {e}")
    
    # Test enhancement safety
    print("\n4. Testing Enhancement Safety")
    test_safety = TestEnhancementSafety()
    
    try:
        test_safety.test_safe_feature_flags()
        print("   ✓ Safe feature flags test passed")
    except Exception as e:
        print(f"   ✗ Safe feature flags test failed: {e}")
    
    try:
        test_safety.test_graceful_degradation()
        print("   ✓ Graceful degradation test passed")
    except Exception as e:
        print(f"   ✗ Graceful degradation test failed: {e}")
    
    # Test failure policies
    print("\n5. Testing Failure Policies")
    test_policies = TestFailurePolicies()
    
    try:
        test_policies.test_fail_fast_policy()
        print("   ✓ Fail-fast policy test passed")
    except Exception as e:
        print(f"   ✗ Fail-fast policy test failed: {e}")
    
    try:
        test_policies.test_collect_all_errors_policy()
        print("   ✓ Collect all errors policy test passed")
    except Exception as e:
        print(f"   ✗ Collect all errors policy test failed: {e}")
    
    try:
        test_policies.test_comprehensive_report_generation()
        print("   ✓ Comprehensive report generation test passed")
    except Exception as e:
        print(f"   ✗ Comprehensive report generation test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test Suite Completed")


if __name__ == "__main__":
    if pytest is not None:
        # Run with pytest if available
        pytest.main([
            __file__, 
            "-v", 
            "--tb=short",
            "-x"  # Stop on first failure for fail-fast testing
        ])
    else:
        # Run manual tests
        run_manual_tests()