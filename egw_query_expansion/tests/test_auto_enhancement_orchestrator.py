"""
Tests for AutoEnhancementOrchestrator
"""

import json
import time
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from egw_query_expansion.core.auto_enhancement_orchestrator import (
    AutoEnhancementOrchestrator,
    PreflightValidationError,
    DriftDetectionError,
    PerformanceMetric,
    DriftAnalysis,
    ThresholdConfig
)


class TestAutoEnhancementOrchestrator:
    """Test suite for AutoEnhancementOrchestrator"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_orchestrator(self, temp_dir):
        """Create orchestrator with temporary files"""
        thresholds_path = temp_dir / "test_thresholds.json"
        metadata_path = temp_dir / "test_metadata.json"
        
        # Create test thresholds file
        test_thresholds = {
            "score_variance": {
                "min_value": 0.0,
                "max_value": 1.0,
                "tolerance": 0.15,
                "calibration_requirement": 0.8
            },
            "processing_time": {
                "min_value": 0.0,
                "max_value": 30.0,
                "tolerance": 0.25,
                "calibration_requirement": 0.9
            }
        }
        
        with open(thresholds_path, 'w') as f:
            json.dump(test_thresholds, f)
        
        return AutoEnhancementOrchestrator(
            thresholds_path=str(thresholds_path),
            metadata_output_path=str(metadata_path)
        )
    
    def test_initialization(self, mock_orchestrator):
        """Test orchestrator initialization"""
        assert mock_orchestrator.enable_drift_detection
        assert mock_orchestrator.enable_provenance_tracking
        assert len(mock_orchestrator.thresholds) >= 2
        assert "score_variance" in mock_orchestrator.thresholds
        assert "processing_time" in mock_orchestrator.thresholds
    
    def test_threshold_config_validation(self):
        """Test ThresholdConfig validation"""
        # Valid configuration
        config = ThresholdConfig(
            metric_name="test",
            min_value=0.0,
            max_value=1.0,
            tolerance=0.1,
            calibration_requirement=0.8
        )
        assert config.metric_name == "test"
        
        # Invalid configuration (max <= min)
        with pytest.raises(ValueError):
            ThresholdConfig(
                metric_name="test",
                min_value=1.0,
                max_value=0.5,
                tolerance=0.1,
                calibration_requirement=0.8
            )
    
    def test_preflight_validation_success(self, mock_orchestrator):
        """Test successful preflight validation"""
        valid_input = {
            "query": "test query",
            "enhancement_params": {"param1": "value1"},
            "metadata": {"source": "test"}
        }
        
        result = mock_orchestrator.preflight_validation(valid_input)
        assert result is True
    
    def test_preflight_validation_invalid_input(self, mock_orchestrator):
        """Test preflight validation with invalid input"""
        invalid_input = {
            "query": "",  # Empty query
            "enhancement_params": {},
            "metadata": {}
        }
        
        with pytest.raises(PreflightValidationError):
            mock_orchestrator.preflight_validation(invalid_input)
    
    def test_enhancement_activation(self, mock_orchestrator):
        """Test enhancement activation"""
        enhancement_id = "test_enhancement_1"
        
        activation_id = mock_orchestrator.activate_enhancement(
            enhancement_id,
            query="test query",
            enhancement_params={"param1": "value1"}
        )
        
        assert activation_id is not None
        assert len(activation_id) == 16  # SHA256 hash prefix
        assert enhancement_id in mock_orchestrator.active_enhancements
        
        # Check metadata
        metadata = mock_orchestrator.active_enhancements[enhancement_id]
        assert metadata.enhancement_id == enhancement_id
        assert metadata.activation_timestamp > 0
        assert len(metadata.audit_trail) == 1
        assert metadata.audit_trail[0]["event"] == "enhancement_activated"
    
    def test_performance_monitoring(self, mock_orchestrator):
        """Test performance monitoring context manager"""
        enhancement_id = "test_enhancement_2"
        
        # Activate enhancement
        mock_orchestrator.activate_enhancement(
            enhancement_id,
            query="test query"
        )
        
        # Monitor performance
        with mock_orchestrator.monitor_performance(enhancement_id):
            time.sleep(0.1)  # Simulate work
        
        # Check recorded metrics
        assert enhancement_id in mock_orchestrator.performance_history
        metrics = mock_orchestrator.performance_history[enhancement_id]
        assert len(metrics) > 0
        
        # Check metric types
        metric_names = {m.name for m in metrics}
        expected_names = {"processing_time", "memory_delta", "memory_usage"}
        assert expected_names.issubset(metric_names)
    
    def test_drift_detection(self, mock_orchestrator):
        """Test stability drift detection"""
        enhancement_id = "test_enhancement_3"
        
        # Activate enhancement
        mock_orchestrator.activate_enhancement(
            enhancement_id,
            query="test query"
        )
        
        # Create metrics with significant drift
        drift_metrics = [
            PerformanceMetric("processing_time", 25.0, time.time(), "seconds"),  # High processing time
            PerformanceMetric("memory_usage", 100.0, time.time(), "MB")
        ]
        
        # Build up history with normal values
        for i in range(15):
            mock_orchestrator._record_performance_metric(
                enhancement_id,
                PerformanceMetric("processing_time", 1.0 + i * 0.1, time.time(), "seconds")
            )
        
        # Detect drift with extreme values
        mock_orchestrator._detect_stability_drift(enhancement_id, drift_metrics)
        
        # Check if enhancement was auto-deactivated
        metadata = mock_orchestrator.active_enhancements.get(enhancement_id)
        if metadata:
            # Should have drift detection results
            assert len(metadata.drift_detection_results) > 0
        else:
            # Enhancement was auto-deactivated - check this doesn't cause error
            assert enhancement_id not in mock_orchestrator.active_enhancements
    
    @patch('psutil.Process')
    def test_baseline_establishment(self, mock_process, mock_orchestrator):
        """Test performance baseline establishment"""
        mock_process_instance = Mock()
        mock_process_instance.cpu_percent.return_value = 5.0
        mock_process_instance.memory_info.return_value = Mock(rss=1024 * 1024 * 100)  # 100 MB
        mock_process.return_value = mock_process_instance
        
        baselines = mock_orchestrator._establish_baselines()
        
        assert "cpu_percent" in baselines
        assert "memory_mb" in baselines
        assert "processing_time" in baselines
        assert baselines["cpu_percent"] == 5.0
        assert baselines["memory_mb"] == 100.0
    
    def test_manual_deactivation(self, mock_orchestrator):
        """Test manual enhancement deactivation"""
        enhancement_id = "test_enhancement_4"
        
        # Activate enhancement
        mock_orchestrator.activate_enhancement(
            enhancement_id,
            query="test query"
        )
        
        assert enhancement_id in mock_orchestrator.active_enhancements
        
        # Deactivate manually
        result = mock_orchestrator.deactivate_enhancement(
            enhancement_id,
            reason="test_completion"
        )
        
        assert result is True
        assert enhancement_id not in mock_orchestrator.active_enhancements
    
    def test_enhancement_status(self, mock_orchestrator):
        """Test getting enhancement status"""
        enhancement_id = "test_enhancement_5"
        
        # Test non-existent enhancement
        status = mock_orchestrator.get_enhancement_status(enhancement_id)
        assert status is None
        
        # Activate enhancement
        mock_orchestrator.activate_enhancement(
            enhancement_id,
            query="test query"
        )
        
        # Add some performance metrics
        for i in range(5):
            mock_orchestrator._record_performance_metric(
                enhancement_id,
                PerformanceMetric("processing_time", 1.0 + i * 0.1, time.time(), "seconds")
            )
        
        # Get status
        status = mock_orchestrator.get_enhancement_status(enhancement_id)
        
        assert status is not None
        assert status["enhancement_id"] == enhancement_id
        assert status["status"] == "active"
        assert "uptime_seconds" in status
        assert "performance_summary" in status
        assert status["total_audit_events"] >= 1
    
    def test_system_status(self, mock_orchestrator):
        """Test getting system status"""
        # Activate multiple enhancements
        for i in range(3):
            mock_orchestrator.activate_enhancement(
                f"test_enhancement_{i}",
                query=f"test query {i}"
            )
        
        status = mock_orchestrator.get_system_status()
        
        assert status["active_enhancements"] == 3
        assert status["drift_detection_enabled"] is True
        assert status["provenance_tracking_enabled"] is True
        assert "timestamp" in status
        assert "version_info" in status
        assert len(status["enhancement_ids"]) == 3
    
    def test_calibration_score_calculation(self, mock_orchestrator):
        """Test threshold calibration score calculation"""
        config = ThresholdConfig(
            metric_name="test",
            min_value=0.0,
            max_value=10.0,
            tolerance=1.0,  # 10% of range
            calibration_requirement=0.8
        )
        
        score = mock_orchestrator._calculate_calibration_score("test", config)
        assert 0.0 <= score <= 1.0
        assert score >= 0.85  # Should be high score for reasonable tolerance
    
    def test_trend_calculation(self, mock_orchestrator):
        """Test trend calculation from values"""
        # Stable values
        stable_values = [1.0, 1.1, 0.9, 1.0, 1.05]
        trend = mock_orchestrator._calculate_trend(stable_values)
        assert trend == "stable"
        
        # Increasing values
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = mock_orchestrator._calculate_trend(increasing_values)
        assert trend == "increasing"
        
        # Decreasing values
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = mock_orchestrator._calculate_trend(decreasing_values)
        assert trend == "decreasing"
        
        # Insufficient data
        few_values = [1.0, 2.0]
        trend = mock_orchestrator._calculate_trend(few_values)
        assert trend == "insufficient_data"
    
    def test_metadata_persistence(self, mock_orchestrator, temp_dir):
        """Test enhancement metadata persistence"""
        enhancement_id = "test_enhancement_persist"
        
        # Activate enhancement
        mock_orchestrator.activate_enhancement(
            enhancement_id,
            query="test query"
        )
        
        # Force metadata save
        mock_orchestrator._save_enhancement_metadata()
        
        # Check metadata file exists
        metadata_file = Path(mock_orchestrator.metadata_output_path)
        assert metadata_file.exists()
        
        # Load and verify metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert enhancement_id in metadata
        assert "_system" in metadata
        assert metadata[enhancement_id]["status"] == "active"
        assert metadata["_system"]["total_active_enhancements"] == 1
    
    def test_error_handling_invalid_thresholds(self, temp_dir):
        """Test error handling with invalid thresholds file"""
        thresholds_path = temp_dir / "invalid_thresholds.json"
        metadata_path = temp_dir / "test_metadata.json"
        
        # Create invalid thresholds file
        with open(thresholds_path, 'w') as f:
            f.write("invalid json content")
        
        # Should fall back to default thresholds
        orchestrator = AutoEnhancementOrchestrator(
            thresholds_path=str(thresholds_path),
            metadata_output_path=str(metadata_path)
        )
        
        # Should have default thresholds
        assert len(orchestrator.thresholds) >= 3
        assert "score_variance" in orchestrator.thresholds
    
    def test_drift_analysis_dataclass(self):
        """Test DriftAnalysis dataclass"""
        analysis = DriftAnalysis(
            metric_name="test_metric",
            current_value=1.5,
            baseline_value=1.0,
            variance=0.1,
            drift_magnitude=0.5,
            exceeded_tolerance=True,
            tolerance_bound=0.2,
            timestamp=time.time(),
            additional_context={"test": "value"}
        )
        
        assert analysis.metric_name == "test_metric"
        assert analysis.exceeded_tolerance is True
        assert analysis.additional_context["test"] == "value"
    
    def test_performance_metric_dataclass(self):
        """Test PerformanceMetric dataclass"""
        timestamp = time.time()
        metric = PerformanceMetric(
            name="test_metric",
            value=42.0,
            timestamp=timestamp,
            unit="seconds",
            metadata={"source": "test"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.timestamp == timestamp
        assert metric.unit == "seconds"
        assert metric.metadata["source"] == "test"