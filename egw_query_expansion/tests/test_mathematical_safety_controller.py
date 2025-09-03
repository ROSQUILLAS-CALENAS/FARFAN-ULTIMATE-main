"""
Tests for Mathematical Enhancement Safety Controller

This module tests the comprehensive safety system for mathematical enhancements
including feature flags, numerical stability, and graceful degradation.
"""

import pytest
import numpy as np
import torch
import time
import tempfile
import json
# # # from unittest.mock import Mock, patch  # Module not found  # Module not found  # Module not found

# # # from egw_query_expansion.core.mathematical_safety_controller import (  # Module not found  # Module not found  # Module not found
    MathematicalEnhancementSafetyController,
    SafetyThresholds,
    FeatureFlag,
    EnhancementStatus,
    DegradationReason,
    NumericalStabilityGuard,
    safe_mathematical_enhancement,
    create_default_safety_controller,
    create_strict_safety_controller
)


class TestNumericalStabilityGuard:
    """Test numerical stability checking"""
    
    def setup_method(self):
        self.thresholds = SafetyThresholds()
        self.guard = NumericalStabilityGuard(self.thresholds)
    
    def test_tensor_stability_valid(self):
        """Test stable tensor passes checks"""
        tensor = np.array([1.0, 2.0, 3.0])
        stable, reason = self.guard.check_tensor_stability(tensor)
        assert stable is True
        assert reason is None
    
    def test_tensor_stability_nan(self):
        """Test NaN detection"""
        tensor = np.array([1.0, np.nan, 3.0])
        stable, reason = self.guard.check_tensor_stability(tensor)
        assert stable is False
        assert reason == DegradationReason.NAN_DETECTED
    
    def test_tensor_stability_inf(self):
        """Test Inf detection"""
        tensor = np.array([1.0, np.inf, 3.0])
        stable, reason = self.guard.check_tensor_stability(tensor)
        assert stable is False
        assert reason == DegradationReason.INF_DETECTED
    
    def test_matrix_condition_good(self):
        """Test well-conditioned matrix"""
        matrix = np.eye(3)
        stable, reason = self.guard.check_matrix_condition(matrix)
        assert stable is True
        assert reason is None
    
    def test_matrix_condition_bad(self):
        """Test ill-conditioned matrix"""
        # Create ill-conditioned matrix
        matrix = np.array([[1e-15, 1.0], [0.0, 1.0]])
        stable, reason = self.guard.check_matrix_condition(matrix)
        assert stable is False
        assert reason == DegradationReason.NUMERICAL_INSTABILITY
    
    def test_safe_division(self):
        """Test epsilon-protected division"""
        # Normal division
        result = self.guard.safe_division(6.0, 2.0)
        assert result == 3.0
        
        # Division by small number
        result = self.guard.safe_division(1.0, 1e-15)
        assert not np.isnan(result) and not np.isinf(result)
        
        # Division by zero
        result = self.guard.safe_division(1.0, 0.0)
        assert not np.isnan(result) and not np.isinf(result)
    
    def test_safe_log(self):
        """Test epsilon-protected logarithm"""
        # Normal log
        result = self.guard.safe_log(np.e)
        assert np.isclose(result, 1.0)
        
        # Log of small number
        result = self.guard.safe_log(1e-20)
        assert not np.isnan(result) and not np.isinf(result)
        
        # Log of zero
        result = self.guard.safe_log(0.0)
        assert not np.isnan(result) and not np.isinf(result)
    
    def test_safe_sqrt(self):
        """Test epsilon-protected square root"""
        # Normal sqrt
        result = self.guard.safe_sqrt(4.0)
        assert result == 2.0
        
        # Sqrt of small number
        result = self.guard.safe_sqrt(1e-20)
        assert not np.isnan(result) and not np.isinf(result)
        
        # Sqrt of negative (should be protected)
        result = self.guard.safe_sqrt(-1.0)
        assert not np.isnan(result) and not np.isinf(result)


class TestMathematicalEnhancementSafetyController:
    """Test main safety controller"""
    
    def setup_method(self):
        self.controller = MathematicalEnhancementSafetyController()
        
        # Mock functions for testing
        def mock_enhancement(x):
            return x * 2
        
        def mock_baseline(x):
            return x
        
        def mock_failing_enhancement(x):
            raise RuntimeError("Enhancement failed")
        
        def mock_unstable_enhancement(x):
            return np.array([np.nan, np.inf, 1.0])
        
        self.mock_enhancement = mock_enhancement
        self.mock_baseline = mock_baseline
        self.mock_failing_enhancement = mock_failing_enhancement
        self.mock_unstable_enhancement = mock_unstable_enhancement
    
    def test_register_enhancement(self):
        """Test enhancement registration"""
        self.controller.register_enhancement(
            "test_enhancement",
            self.mock_enhancement,
            self.mock_baseline
        )
        
        assert "test_enhancement" in self.controller.feature_flags
        assert "test_enhancement" in self.controller.enhancement_metrics
        assert "test_enhancement" in self.controller.baseline_implementations
    
    def test_enable_disable_enhancement(self):
        """Test enabling/disabling enhancements"""
        # Register first
        self.controller.register_enhancement(
            "test_enhancement",
            self.mock_enhancement,
            self.mock_baseline
        )
        
        # Test enabling
        success = self.controller.enable_enhancement("test_enhancement")
        assert success is True
        assert self.controller.enhancement_status["test_enhancement"] == EnhancementStatus.ENABLED
        
        # Test disabling
        success = self.controller.disable_enhancement("test_enhancement")
        assert success is True
        assert self.controller.enhancement_status["test_enhancement"] == EnhancementStatus.DISABLED
    
    def test_safe_compute_success(self):
        """Test successful safe computation"""
        self.controller.register_enhancement(
            "test_enhancement",
            self.mock_enhancement,
            self.mock_baseline
        )
        self.controller.enable_enhancement("test_enhancement")
        
        result = self.controller.safe_compute_with_fallback(
            "test_enhancement", 5, enhanced_fn=self.mock_enhancement
        )
        
        assert result == 10  # 5 * 2
        metrics = self.controller.enhancement_metrics["test_enhancement"]
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
    
    def test_safe_compute_fallback(self):
        """Test fallback to baseline on failure"""
        self.controller.register_enhancement(
            "test_enhancement",
            self.mock_failing_enhancement,
            self.mock_baseline
        )
        self.controller.enable_enhancement("test_enhancement")
        
        result = self.controller.safe_compute_with_fallback(
            "test_enhancement", 5, enhanced_fn=self.mock_failing_enhancement
        )
        
        assert result == 5  # baseline result
        metrics = self.controller.enhancement_metrics["test_enhancement"]
        assert metrics.failure_count == 1
        assert self.controller.enhancement_status["test_enhancement"] == EnhancementStatus.DEGRADED
    
    def test_feature_flag_dependencies(self):
        """Test feature flag dependencies"""
        # Create dependent enhancement
        flag = FeatureFlag(
            name="dependent_enhancement",
            dependencies=["base_enhancement"]
        )
        
        self.controller.register_enhancement(
            "dependent_enhancement",
            self.mock_enhancement,
            self.mock_baseline,
            flag
        )
        
        self.controller.register_enhancement(
            "base_enhancement",
            self.mock_enhancement,
            self.mock_baseline
        )
        
        # Should fail to enable without dependency
        success = self.controller.enable_enhancement("dependent_enhancement")
        assert success is False
        
        # Enable dependency first
        self.controller.enable_enhancement("base_enhancement")
        success = self.controller.enable_enhancement("dependent_enhancement")
        assert success is True
    
    def test_auto_disable_on_high_failure_rate(self):
        """Test auto-disable on high failure rate"""
        # Set low degradation threshold
        thresholds = SafetyThresholds(max_degradation_rate=0.1)
        controller = MathematicalEnhancementSafetyController(thresholds)
        
        controller.register_enhancement(
            "failing_enhancement",
            self.mock_failing_enhancement,
            self.mock_baseline
        )
        controller.enable_enhancement("failing_enhancement")
        
        # Force multiple failures
        for _ in range(5):
            try:
                controller.safe_compute_with_fallback(
                    "failing_enhancement", 5, enhanced_fn=self.mock_failing_enhancement
                )
            except:
                pass
        
        # Should be auto-disabled
        assert controller.enhancement_status["failing_enhancement"] == EnhancementStatus.AUTO_DISABLED
    
    def test_enhancement_status_tracking(self):
        """Test enhancement status and metrics tracking"""
        self.controller.register_enhancement(
            "test_enhancement",
            self.mock_enhancement,
            self.mock_baseline
        )
        self.controller.enable_enhancement("test_enhancement")
        
        # Successful execution
        self.controller.safe_compute_with_fallback(
            "test_enhancement", 5, enhanced_fn=self.mock_enhancement
        )
        
        status = self.controller.get_enhancement_status("test_enhancement")
        assert status['status'] == EnhancementStatus.ENABLED.value
        assert status['metrics']['success_count'] == 1
        assert status['metrics']['success_rate'] == 1.0
    
    def test_system_health_metrics(self):
        """Test system health reporting"""
        self.controller.register_enhancement(
            "test_enhancement",
            self.mock_enhancement,
            self.mock_baseline
        )
        self.controller.enable_enhancement("test_enhancement")
        
        # Execute some operations
        self.controller.safe_compute_with_fallback(
            "test_enhancement", 5, enhanced_fn=self.mock_enhancement
        )
        
        health = self.controller.get_system_health()
        assert health['total_activations'] == 1
        assert health['total_successes'] == 1
        assert health['global_success_rate'] == 1.0
        assert health['active_enhancements'] == 1
    
    def test_metrics_export(self):
        """Test metrics export functionality"""
        self.controller.register_enhancement(
            "test_enhancement",
            self.mock_enhancement,
            self.mock_baseline
        )
        
        # Export to dictionary
        metrics = self.controller.export_metrics()
        assert 'timestamp' in metrics
        assert 'system_health' in metrics
        assert 'enhancement_status' in metrics
        
        # Export to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        self.controller.export_metrics(filename)
        
        # Verify file contents
        with open(filename, 'r') as f:
            file_metrics = json.load(f)
            assert 'timestamp' in file_metrics
    
    def test_safe_enhancement_context(self):
        """Test safe enhancement context manager"""
        self.controller.register_enhancement(
            "test_enhancement",
            self.mock_enhancement,
            self.mock_baseline
        )
        self.controller.enable_enhancement("test_enhancement")
        
        # Successful context
        with self.controller.safe_enhancement_context("test_enhancement"):
            pass  # Simulation of successful computation
        
        metrics = self.controller.enhancement_metrics["test_enhancement"]
        assert metrics.success_count == 1
        
        # Failing context
        with pytest.raises(RuntimeError):
            with self.controller.safe_enhancement_context("test_enhancement"):
                raise RuntimeError("Test failure")
        
        assert metrics.failure_count == 1


class TestSafeMathematicalEnhancementDecorator:
    """Test the safety decorator"""
    
    def setup_method(self):
        self.controller = create_default_safety_controller()
    
    def test_decorator_basic_usage(self):
        """Test basic decorator usage"""
        def baseline_fn(x):
            return x
        
        @safe_mathematical_enhancement("decorated_enhancement", self.controller, baseline_fn)
        def enhanced_fn(x):
            return x * 2
        
        # Enable the enhancement
        self.controller.enable_enhancement("decorated_enhancement")
        
        result = enhanced_fn(5)
        assert result == 10
        
        # Check metrics
        status = self.controller.get_enhancement_status("decorated_enhancement")
        assert status['metrics']['success_count'] == 1
    
    def test_decorator_fallback(self):
        """Test decorator fallback behavior"""
        def baseline_fn(x):
            return x
        
        @safe_mathematical_enhancement("failing_enhancement", self.controller, baseline_fn)
        def failing_fn(x):
            raise RuntimeError("Enhancement failed")
        
        # Enable the enhancement
        self.controller.enable_enhancement("failing_enhancement")
        
        result = failing_fn(5)
        assert result == 5  # Should fall back to baseline
        
        # Check metrics
        status = self.controller.get_enhancement_status("failing_enhancement")
        assert status['metrics']['failure_count'] == 1


class TestSafetyControllerFactories:
    """Test safety controller factory functions"""
    
    def test_default_controller(self):
        """Test default controller creation"""
        controller = create_default_safety_controller()
        assert isinstance(controller, MathematicalEnhancementSafetyController)
        assert controller.thresholds.max_iterations == 1000
    
    def test_strict_controller(self):
        """Test strict controller creation"""
        controller = create_strict_safety_controller()
        assert isinstance(controller, MathematicalEnhancementSafetyController)
        assert controller.thresholds.max_iterations == 500
        assert controller.thresholds.computation_timeout == 15.0


class TestIntegrationScenarios:
    """Test complex integration scenarios"""
    
    def setup_method(self):
        self.controller = create_default_safety_controller()
    
    def test_numerical_instability_detection_and_recovery(self):
# # #         """Test detection and recovery from numerical instability"""  # Module not found  # Module not found  # Module not found
        def unstable_enhancement(x):
            # Simulate numerical instability
            if hasattr(unstable_enhancement, 'call_count'):
                unstable_enhancement.call_count += 1
            else:
                unstable_enhancement.call_count = 1
            
            if unstable_enhancement.call_count <= 2:
                return np.array([np.nan, x])  # Unstable first two calls
            else:
                return np.array([x, x * 2])  # Stable afterward
        
        def stable_baseline(x):
            return np.array([x, x])
        
        self.controller.register_enhancement(
            "unstable_enhancement",
            unstable_enhancement,
            stable_baseline
        )
        self.controller.enable_enhancement("unstable_enhancement")
        
        # First calls should fall back
        result1 = self.controller.safe_compute_with_fallback(
            "unstable_enhancement", 5, enhanced_fn=unstable_enhancement
        )
        result2 = self.controller.safe_compute_with_fallback(
            "unstable_enhancement", 5, enhanced_fn=unstable_enhancement
        )
        
        # Should fall back to baseline
        np.testing.assert_array_equal(result1, [5, 5])
        np.testing.assert_array_equal(result2, [5, 5])
        
        # Check degradation was recorded
        status = self.controller.get_enhancement_status("unstable_enhancement")
        assert status['status'] == EnhancementStatus.DEGRADED.value
    
    def test_multi_enhancement_coordination(self):
        """Test coordination between multiple enhancements"""
        def enhancement_a(x):
            return x + 1
        
        def enhancement_b(x):
            return x * 2
        
        def baseline_a(x):
            return x
        
        def baseline_b(x):
            return x
        
        # Register enhancements
        self.controller.register_enhancement("enhancement_a", enhancement_a, baseline_a)
        self.controller.register_enhancement("enhancement_b", enhancement_b, baseline_b)
        
        # Enable both
        self.controller.enable_enhancement("enhancement_a")
        self.controller.enable_enhancement("enhancement_b")
        
        # Use both enhancements
        result_a = self.controller.safe_compute_with_fallback(
            "enhancement_a", 5, enhanced_fn=enhancement_a
        )
        result_b = self.controller.safe_compute_with_fallback(
            "enhancement_b", result_a, enhanced_fn=enhancement_b
        )
        
        assert result_a == 6  # 5 + 1
        assert result_b == 12  # 6 * 2
        
        # Check system health
        health = self.controller.get_system_health()
        assert health['total_activations'] == 2
        assert health['active_enhancements'] == 2
    
    def test_performance_monitoring_and_optimization(self):
        """Test performance monitoring capabilities"""
        def slow_enhancement(x):
            time.sleep(0.1)  # Simulate slow computation
            return x * 2
        
        def fast_baseline(x):
            return x
        
        self.controller.register_enhancement(
            "slow_enhancement",
            slow_enhancement,
            fast_baseline
        )
        self.controller.enable_enhancement("slow_enhancement")
        
        # Execute enhancement
        start_time = time.time()
        result = self.controller.safe_compute_with_fallback(
            "slow_enhancement", 5, enhanced_fn=slow_enhancement
        )
        end_time = time.time()
        
        assert result == 10
        
        # Check timing metrics
        status = self.controller.get_enhancement_status("slow_enhancement")
        assert status['metrics']['average_computation_time'] > 0.05  # Should record the delay
        
        # Verify performance history
        metrics = self.controller.enhancement_metrics["slow_enhancement"]
        assert len(metrics.performance_history) == 1
        assert metrics.performance_history[0] > 0.05


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestMathematicalEnhancementSafetyController::test_safe_compute_success", "-v"])