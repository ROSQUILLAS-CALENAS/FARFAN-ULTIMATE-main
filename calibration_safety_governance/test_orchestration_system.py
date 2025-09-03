"""
Test Suite for Auto-Enhancement Orchestration System
====================================================

Comprehensive tests for preflight validation, auto-deactivation monitoring,
and provenance tracking components.
"""

import pytest
import json
import tempfile
import shutil
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from unittest.mock import Mock, patch  # Module not found  # Module not found  # Module not found

# # # from preflight_validator import PreflightValidator, ValidationResult  # Module not found  # Module not found  # Module not found
# # # from auto_deactivation_monitor import (  # Module not found  # Module not found  # Module not found
    AutoDeactivationMonitor, 
    StabilityDriftAnalyzer, 
    EvidenceQualityTracker,
    PerformanceRegressionDetector,
    DeactivationTriggerType,
    DeactivationSeverity
)
# # # from provenance_tracker import (  # Module not found  # Module not found  # Module not found
    ProvenanceTracker, 
    EnhancementLifecycleState,
    ActivationCriteriaType
)
# # # from orchestrator import EnhancementOrchestrator, OrchestrationConfig, OrchestrationMode  # Module not found  # Module not found  # Module not found


class TestPreflightValidator:
    """Test preflight validation system"""
    
    @pytest.fixture
    def validator(self):
        """Create validator with temporary thresholds"""
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds_path = Path(temp_dir) / "thresholds.json"
            
            # Create minimal thresholds file
            thresholds = {
                "validation": {
                    "schema_compliance": {
                        "minimum_fields_present": 0.95,
                        "data_type_accuracy": 0.98
                    },
                    "library_compatibility": {
                        "version_mismatch_tolerance": 0.1,
                        "critical_dependency_alignment": 1.0
                    },
                    "threshold_satisfaction": {
                        "mandatory_clause_compliance": 1.0,
                        "proxy_score_minimum": 0.7
                    }
                }
            }
            
            with open(thresholds_path, 'w') as f:
                json.dump(thresholds, f)
                
            yield PreflightValidator(str(thresholds_path))
    
    def test_valid_schema_validation(self, validator):
        """Test validation of valid enhancement request"""
        valid_request = {
            "enhancement_id": "test_enhancement",
            "enhancement_type": "adaptive_scoring",
            "configuration": {"param1": "value1"},
            "priority": "high",
            "activation_criteria": {"threshold": 0.8},
            "metadata": {}
        }
        
        result = validator.validate_input_schema(valid_request, "enhancement_request")
        
        assert isinstance(result, ValidationResult)
        assert result.passed
        assert result.score > 0.8
        assert len(result.errors) == 0
    
    def test_invalid_schema_validation(self, validator):
        """Test validation of invalid enhancement request"""
        invalid_request = {
            "enhancement_id": "test_enhancement",
            "enhancement_type": "invalid_type",  # Invalid enum value
            # Missing required fields
        }
        
        result = validator.validate_input_schema(invalid_request, "enhancement_request")
        
        assert isinstance(result, ValidationResult)
        assert not result.passed
        assert result.score < 0.5
        assert len(result.errors) > 0
    
    def test_threshold_satisfaction_validation(self, validator):
        """Test threshold satisfaction validation"""
        metrics = {
            "mandatory_compliance": 0.95,
            "proxy_score": 0.8,
            "confidence_alpha": 0.9,
            "sigma_presence": 0.1,
            "governance_completeness": 0.85
        }
        
        result = validator.validate_threshold_satisfaction(metrics)
        
        assert isinstance(result, ValidationResult)
        assert result.passed
        assert result.score > 0.7
    
    def test_insufficient_threshold_satisfaction(self, validator):
        """Test insufficient threshold satisfaction"""
        metrics = {
            "mandatory_compliance": 0.6,  # Below threshold
            "proxy_score": 0.5,           # Below threshold
            "confidence_alpha": 0.8,
            "sigma_presence": 0.05,
            "governance_completeness": 0.7
        }
        
        result = validator.validate_threshold_satisfaction(metrics)
        
        assert isinstance(result, ValidationResult)
        assert not result.passed
        assert len(result.errors) > 0


class TestStabilityDriftAnalyzer:
    """Test stability drift analysis"""
    
    @pytest.fixture
    def analyzer(self):
        return StabilityDriftAnalyzer(window_size=5, variance_threshold=0.15)
    
    def test_stable_scores(self, analyzer):
        """Test analysis of stable scores"""
        enhancement_id = "test_enhancement"
        
        # Add stable scores
        for score in [0.85, 0.84, 0.86, 0.85, 0.87]:
            analyzer.add_score(enhancement_id, score)
        
        analysis = analyzer.analyze_stability_drift(enhancement_id)
        
        assert analysis["sufficient_data"]
        assert analysis["stability_coefficient"] > 0.8
        assert not analysis["exceeds_variance_threshold"]
        assert not analysis["is_degrading"]
    
    def test_unstable_scores(self, analyzer):
        """Test analysis of unstable scores"""
        enhancement_id = "test_enhancement"
        
        # Add unstable scores
        for score in [0.9, 0.7, 0.95, 0.6, 0.8]:
            analyzer.add_score(enhancement_id, score)
        
        analysis = analyzer.analyze_stability_drift(enhancement_id)
        
        assert analysis["sufficient_data"]
        assert analysis["exceeds_variance_threshold"]
        assert analysis["stability_coefficient"] < 0.8
    
    def test_degrading_scores(self, analyzer):
        """Test analysis of degrading scores"""
        enhancement_id = "test_enhancement"
        
        # Add degrading scores
        for score in [0.9, 0.85, 0.8, 0.75, 0.7]:
            analyzer.add_score(enhancement_id, score)
        
        analysis = analyzer.analyze_stability_drift(enhancement_id)
        
        assert analysis["sufficient_data"]
        assert analysis["is_degrading"]


class TestEvidenceQualityTracker:
    """Test evidence quality tracking"""
    
    @pytest.fixture
    def tracker(self):
        return EvidenceQualityTracker(degradation_threshold=0.1, quality_minimum=0.75)
    
    def test_quality_degradation_detection(self, tracker):
        """Test detection of quality degradation"""
        enhancement_id = "test_enhancement"
        
        # Record degrading quality
        qualities = [
            {"overall_quality": 0.9, "consistency": 0.85, "coverage": 0.9, "coherence": 0.88},
            {"overall_quality": 0.8, "consistency": 0.8, "coverage": 0.85, "coherence": 0.83},
            {"overall_quality": 0.7, "consistency": 0.75, "coverage": 0.8, "coherence": 0.78}
        ]
        
        for quality in qualities:
            tracker.record_evidence_quality(enhancement_id, quality)
        
        analysis = tracker.detect_quality_degradation(enhancement_id)
        
        assert analysis["sufficient_data"]
        assert analysis["below_minimum_quality"]
        assert analysis["degradation_rate_exceeded"]
    
    def test_stable_quality(self, tracker):
        """Test stable quality detection"""
        enhancement_id = "test_enhancement"
        
        # Record stable quality
        for _ in range(3):
            tracker.record_evidence_quality(enhancement_id, {
                "overall_quality": 0.85,
                "consistency": 0.9,
                "coverage": 0.88,
                "coherence": 0.87
            })
        
        analysis = tracker.detect_quality_degradation(enhancement_id)
        
        assert analysis["sufficient_data"]
        assert not analysis["below_minimum_quality"]
        assert not analysis["degradation_rate_exceeded"]


class TestPerformanceRegressionDetector:
    """Test performance regression detection"""
    
    @pytest.fixture
    def detector(self):
        return PerformanceRegressionDetector()
    
    def test_regression_detection(self, detector):
        """Test detection of performance regressions"""
        enhancement_id = "test_enhancement"
        
        # Establish baseline
        baseline_metrics = {
            "response_time": 0.5,
            "accuracy": 0.85,
            "throughput": 100.0,
            "error_rate": 0.01
        }
        detector.establish_baseline(enhancement_id, baseline_metrics)
        
        # Record degraded performance
        degraded_metrics = {
            "response_time": 1.0,    # 2x increase
            "accuracy": 0.8,         # 0.05 decrease
            "throughput": 70.0,      # 30% decrease
            "error_rate": 0.04       # 0.03 increase
        }
        detector.record_performance(enhancement_id, degraded_metrics)
        
        # Check for regressions
        thresholds = {
            "response_time_increase": 1.5,
            "accuracy_degradation": 0.03,
            "throughput_decrease": 0.25,
            "error_rate_increase": 0.02
        }
        
        analysis = detector.detect_regressions(enhancement_id, thresholds)
        
        assert analysis["sufficient_data"]
        assert analysis["critical_regression_detected"]
        assert analysis["regression_score"] > 0.5


class TestAutoDeactivationMonitor:
    """Test auto-deactivation monitoring system"""
    
    @pytest.fixture
    def monitor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds_path = Path(temp_dir) / "thresholds.json"
            
            # Create minimal thresholds
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
                        "stability_breach": {"consecutive_violations": 2},
                        "performance_degradation": {"consecutive_regressions": 2},
                        "safety_violation": {"single_critical_failure": True}
                    },
                    "cooldown_periods": {
                        "minor_deactivation": "PT15M",
                        "major_deactivation": "PT1H",
                        "critical_deactivation": "PT24H"
                    }
                }
            }
            
            with open(thresholds_path, 'w') as f:
                json.dump(thresholds, f)
                
            yield AutoDeactivationMonitor(str(thresholds_path))
    
    def test_stability_deactivation_trigger(self, monitor):
        """Test deactivation trigger due to stability issues"""
        enhancement_id = "test_enhancement"
        
        # Record unstable performance multiple times
        for i in range(3):
            result = monitor.monitor_enhancement(
                enhancement_id=enhancement_id,
                performance_metrics={
                    "response_time": 0.5 + (i * 0.5),  # Increasing
                    "accuracy": 0.85 - (i * 0.1),      # Decreasing
                    "throughput": 100.0,
                    "error_rate": 0.01
                },
                evidence_quality={
                    "overall_quality": 0.8,
                    "consistency": 0.85,
                    "coverage": 0.9,
                    "coherence": 0.82
                },
                score=0.8 - (i * 0.2)  # Degrading score
            )
        
        # Should trigger deactivation after consecutive violations
        assert result["deactivation_decision"]["should_deactivate"] or \
               result["deactivation_triggers"] > 0


class TestProvenanceTracker:
    """Test provenance tracking system"""
    
    @pytest.fixture
    def tracker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ProvenanceTracker(temp_dir)
    
    def test_enhancement_metadata_creation(self, tracker):
        """Test creation of enhancement metadata"""
        enhancement_id = "test_enhancement"
        
        metadata = tracker.create_enhancement_metadata(
            enhancement_id=enhancement_id,
            enhancement_type="adaptive_scoring",
            description="Test enhancement",
            configuration={"param1": "value1"},
            activation_criteria=[
                {"type": "performance_threshold", "description": "Performance", "threshold": 0.8}
            ],
            baseline_metrics={"accuracy": 0.82},
            dependencies=["base_system"],
            tags=["test"]
        )
        
        assert metadata.enhancement_id == enhancement_id
        assert metadata.current_state == EnhancementLifecycleState.PENDING_ACTIVATION
        assert len(metadata.activation_criteria) == 1
        assert len(metadata.lifecycle_events) == 1  # Creation event
    
    def test_activation_criteria_evaluation(self, tracker):
        """Test evaluation of activation criteria"""
        enhancement_id = "test_enhancement"
        
        # Create metadata
        tracker.create_enhancement_metadata(
            enhancement_id=enhancement_id,
            enhancement_type="adaptive_scoring",
            description="Test enhancement",
            configuration={},
            activation_criteria=[
                {"type": "performance_threshold", "description": "Performance", "threshold": 0.8},
                {"type": "stability_requirement", "description": "Stability", "threshold": 0.85}
            ],
            baseline_metrics={}
        )
        
        # Evaluate with sufficient metrics
        current_metrics = {
            "performance_score": 0.85,
            "stability_score": 0.9
        }
        
        decision = tracker.evaluate_activation_criteria(enhancement_id, current_metrics)
        
        assert decision["should_activate"]
        assert decision["satisfied_criteria"] == 2
        assert decision["satisfaction_score"] > 0.8
    
    def test_activation_recording(self, tracker):
        """Test recording of activation"""
        enhancement_id = "test_enhancement"
        
        # Create and evaluate
        tracker.create_enhancement_metadata(
            enhancement_id=enhancement_id,
            enhancement_type="adaptive_scoring",
            description="Test enhancement",
            configuration={},
            activation_criteria=[],
            baseline_metrics={}
        )
        
        # Record activation
        success = tracker.record_activation(enhancement_id, {"context": "test"})
        
        assert success
        
        metadata = tracker.enhancement_metadata[enhancement_id]
        assert metadata.current_state == EnhancementLifecycleState.ACTIVE
        assert metadata.activated_at is not None
        assert len(metadata.lifecycle_events) >= 2  # Creation + activation
    
    def test_performance_impact_recording(self, tracker):
        """Test recording of performance impacts"""
        enhancement_id = "test_enhancement"
        
        # Create enhancement
        tracker.create_enhancement_metadata(
            enhancement_id=enhancement_id,
            enhancement_type="adaptive_scoring",
            description="Test enhancement",
            configuration={},
            activation_criteria=[],
            baseline_metrics={"accuracy": 0.8, "response_time": 0.5}
        )
        
        # Record performance impact
        tracker.record_performance_impact(enhancement_id, {
            "accuracy": 0.85,
            "response_time": 0.4
        })
        
        metadata = tracker.enhancement_metadata[enhancement_id]
        assert len(metadata.performance_impacts) == 2  # One for each metric
        
        # Check impact calculation
        accuracy_impact = next(
            impact for impact in metadata.performance_impacts 
            if impact.metric_name == "accuracy"
        )
        assert accuracy_impact.impact_percentage > 0  # Improvement


class TestEnhancementOrchestrator:
    """Test full orchestration system"""
    
    @pytest.fixture
    def orchestrator(self):
        config = OrchestrationConfig(
            mode=OrchestrationMode.AUTOMATIC,
            monitoring_interval_seconds=1,
            max_concurrent_enhancements=2
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds_path = Path(temp_dir) / "thresholds.json"
            
            # Create comprehensive thresholds
            thresholds = {
                "validation": {
                    "schema_compliance": {"minimum_fields_present": 0.95, "data_type_accuracy": 0.98},
                    "library_compatibility": {"version_mismatch_tolerance": 0.1},
                    "threshold_satisfaction": {
                        "mandatory_clause_compliance": 1.0,
                        "proxy_score_minimum": 0.7,
                        "confidence_alpha": 0.95,
                        "sigma_presence": 0.05,
                        "governance_completeness": 0.85
                    }
                },
                "stability_monitoring": {
                    "score_variance": {"stability_coefficient": 0.8},
                    "performance_regression": {
                        "response_time_increase": 1.5,
                        "accuracy_degradation": 0.05
                    }
                },
                "enhancement_activation": {
                    "auto_activation": {"stability_requirement": 0.8}
                },
                "auto_deactivation": {
                    "triggers": {
                        "stability_breach": {"consecutive_violations": 2},
                        "performance_degradation": {"consecutive_regressions": 2}
                    },
                    "cooldown_periods": {"minor_deactivation": "PT15M"}
                }
            }
            
            with open(thresholds_path, 'w') as f:
                json.dump(thresholds, f)
            
            # Set metadata directory to temp dir
            orchestrator = EnhancementOrchestrator(config, str(thresholds_path))
            orchestrator.provenance.metadata_dir = Path(temp_dir) / "metadata"
            orchestrator.provenance.metadata_dir.mkdir(exist_ok=True)
            
            yield orchestrator
    
    def test_enhancement_request_submission(self, orchestrator):
        """Test submission of enhancement request"""
        result = orchestrator.submit_enhancement_request(
            enhancement_id="test_enhancement",
            enhancement_type="adaptive_scoring",
            description="Test enhancement",
            configuration={"param1": "value1"},
            activation_criteria=[
                {"type": "performance_threshold", "description": "Performance", "threshold": 0.8}
            ],
            baseline_metrics={"accuracy": 0.8},
            priority="medium"
        )
        
        assert "status" in result
        assert result["status"] in ["submitted", "validation_failed"]
    
    def test_orchestration_status(self, orchestrator):
        """Test orchestration status retrieval"""
        status = orchestrator.get_orchestration_status()
        
        assert "orchestration_config" in status
        assert "current_state" in status
        assert "status_timestamp" in status
        assert status["orchestration_config"]["mode"] == "automatic"
    
    def test_orchestration_report_generation(self, orchestrator):
        """Test comprehensive report generation"""
        # Submit an enhancement first
        orchestrator.submit_enhancement_request(
            enhancement_id="test_enhancement",
            enhancement_type="adaptive_scoring",
            description="Test enhancement",
            configuration={},
            activation_criteria=[],
            baseline_metrics={}
        )
        
        report = orchestrator.generate_orchestration_report()
        
        assert "report_metadata" in report
        assert "orchestration_status" in report
        assert "system_metrics" in report
        assert "recommendations" in report


def run_tests():
    """Run all tests"""
    import sys
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)