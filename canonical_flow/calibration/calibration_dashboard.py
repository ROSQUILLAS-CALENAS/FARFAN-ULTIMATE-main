"""
Calibration Dashboard System

Manages calibration artifact generation, quality gate enforcement, and drift detection.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    class MockNumpy:
        def mean(self, values):
            return sum(values) / len(values) if values else 0.0
        def std(self, values):
            if not values or len(values) < 2:
                return 0.0
            mean_val = self.mean(values)
            variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
            return variance ** 0.5
    np = MockNumpy()
from dataclasses import asdict

from .calibration_artifacts import (
    CalibrationArtifact,
    RetrievalCalibrationArtifact,
    ConfidenceCalibrationArtifact,
    AggregationCalibrationArtifact
)


class CalibrationDashboard:
    """
    Central system for managing calibration artifacts across pipeline stages.
    Handles artifact generation, quality gate enforcement, and drift detection.
    """
    
    def __init__(self, calibration_dir: Union[str, Path] = "canonical_flow/calibration"):
        """
        Initialize calibration dashboard.
        
        Args:
            calibration_dir: Directory for storing calibration artifacts
        """
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Default quality thresholds
        self.default_thresholds = {
            "retrieval": {
                "calibration_quality_score": 0.7,
                "coverage_percentage": 80.0,
                "temperature_stability": 0.8,
                "retrieval_precision": 0.6,
                "retrieval_recall": 0.5
            },
            "confidence": {
                "calibration_quality_score": 0.75,
                "coverage_percentage": 85.0,
                "interval_coverage": 0.9,
                "calibration_error": 0.1,
                "reliability_score": 0.8
            },
            "aggregation": {
                "calibration_quality_score": 0.8,
                "coverage_percentage": 90.0,
                "convergence_rate": 0.9,
                "consensus_score": 0.85,
                "aggregation_stability": 0.75
            }
        }
    
    def generate_retrieval_artifact(
        self,
        calibration_quality_score: float,
        coverage_percentage: float,
        decisions_made: int,
        temperature_stability: float,
        entropy_calibration_score: float,
        retrieval_precision: float,
        retrieval_recall: float,
        fusion_coherence: float,
        temperature_parameter: float = 1.0,
        fusion_weights: Optional[List[float]] = None,
        retrieval_k_values: Optional[List[int]] = None,
        retrieval_latency_ms: float = 0.0,
        index_coherence_score: float = 0.0,
        cross_encoder_agreement: float = 0.0,
        stage_version: str = "1.0.0",
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> RetrievalCalibrationArtifact:
        """Generate retrieval stage calibration artifact."""
        
        # Determine quality gate status
        quality_gate_status = self._evaluate_quality_gates(
            {
                "calibration_quality_score": calibration_quality_score,
                "coverage_percentage": coverage_percentage,
                "temperature_stability": temperature_stability,
                "retrieval_precision": retrieval_precision,
                "retrieval_recall": retrieval_recall
            },
            self.default_thresholds["retrieval"]
        )
        
        # Build calibration parameters
        calibration_parameters = {
            "temperature_parameter": temperature_parameter,
            "fusion_weights": fusion_weights or [0.33, 0.33, 0.34],
            "retrieval_k_values": retrieval_k_values or [10, 50, 100],
            "entropy_calibration_enabled": True,
            **(custom_parameters or {})
        }
        
        artifact = RetrievalCalibrationArtifact(
            calibration_quality_score=calibration_quality_score,
            coverage_percentage=coverage_percentage,
            decisions_made=decisions_made,
            quality_gate_status=quality_gate_status,
            timestamp=datetime.utcnow().isoformat(),
            stage_version=stage_version,
            calibration_parameters=calibration_parameters,
            quality_thresholds=self.default_thresholds["retrieval"],
            temperature_stability=temperature_stability,
            entropy_calibration_score=entropy_calibration_score,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            fusion_coherence=fusion_coherence,
            temperature_parameter=temperature_parameter,
            fusion_weights=fusion_weights or [0.33, 0.33, 0.34],
            retrieval_k_values=retrieval_k_values or [10, 50, 100],
            retrieval_latency_ms=retrieval_latency_ms,
            index_coherence_score=index_coherence_score,
            cross_encoder_agreement=cross_encoder_agreement
        )
        
        # Save to canonical location
        artifact_path = self.calibration_dir / "retrieval_calibration.json"
        artifact.save(artifact_path)
        
        self.logger.info(f"Generated retrieval calibration artifact: {quality_gate_status}")
        return artifact
    
    def generate_confidence_artifact(
        self,
        calibration_quality_score: float,
        coverage_percentage: float,
        decisions_made: int,
        interval_coverage: float,
        calibration_error: float,
        sharpness_score: float,
        reliability_score: float,
        prediction_efficiency: float,
        adaptive_coverage_gap: float = 0.0,
        prediction_set_size_avg: float = 0.0,
        nonconformity_score_stats: Optional[Dict[str, float]] = None,
        confidence_level: float = 0.9,
        alpha_significance: float = 0.1,
        adaptive_quantile_level: float = 0.9,
        distribution_shift_detected: bool = False,
        ks_test_p_value: float = 1.0,
        bootstrap_intervals: Optional[Dict[str, List[float]]] = None,
        stage_version: str = "1.0.0",
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> ConfidenceCalibrationArtifact:
        """Generate confidence stage calibration artifact."""
        
        # Determine quality gate status
        quality_gate_status = self._evaluate_quality_gates(
            {
                "calibration_quality_score": calibration_quality_score,
                "coverage_percentage": coverage_percentage,
                "interval_coverage": interval_coverage,
                "calibration_error": calibration_error,
                "reliability_score": reliability_score
            },
            self.default_thresholds["confidence"]
        )
        
        # Build calibration parameters
        calibration_parameters = {
            "confidence_level": confidence_level,
            "alpha_significance": alpha_significance,
            "adaptive_quantile_level": adaptive_quantile_level,
            "conformal_prediction_enabled": True,
            "distribution_monitoring": True,
            **(custom_parameters or {})
        }
        
        artifact = ConfidenceCalibrationArtifact(
            calibration_quality_score=calibration_quality_score,
            coverage_percentage=coverage_percentage,
            decisions_made=decisions_made,
            quality_gate_status=quality_gate_status,
            timestamp=datetime.utcnow().isoformat(),
            stage_version=stage_version,
            calibration_parameters=calibration_parameters,
            quality_thresholds=self.default_thresholds["confidence"],
            interval_coverage=interval_coverage,
            calibration_error=calibration_error,
            sharpness_score=sharpness_score,
            reliability_score=reliability_score,
            prediction_efficiency=prediction_efficiency,
            adaptive_coverage_gap=adaptive_coverage_gap,
            prediction_set_size_avg=prediction_set_size_avg,
            nonconformity_score_stats=nonconformity_score_stats or {},
            confidence_level=confidence_level,
            alpha_significance=alpha_significance,
            adaptive_quantile_level=adaptive_quantile_level,
            distribution_shift_detected=distribution_shift_detected,
            ks_test_p_value=ks_test_p_value,
            bootstrap_intervals=bootstrap_intervals or {}
        )
        
        # Save to canonical location
        artifact_path = self.calibration_dir / "confidence_calibration.json"
        artifact.save(artifact_path)
        
        self.logger.info(f"Generated confidence calibration artifact: {quality_gate_status}")
        return artifact
    
    def generate_aggregation_artifact(
        self,
        calibration_quality_score: float,
        coverage_percentage: float,
        decisions_made: int,
        convergence_rate: float,
        aggregation_stability: float,
        consensus_score: float,
        uncertainty_quantification_quality: float,
        spectral_gap: float = 0.0,
        eigenvalue_stability: float = 0.0,
        manifold_coherence: float = 0.0,
        aggregation_method: str = "consensus",
        convergence_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization_strength: float = 0.01,
        epistemic_uncertainty: float = 0.0,
        aleatoric_uncertainty: float = 0.0,
        total_uncertainty: float = 0.0,
        calibration_curve_auc: float = 0.0,
        stage_version: str = "1.0.0",
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> AggregationCalibrationArtifact:
        """Generate aggregation stage calibration artifact."""
        
        # Determine quality gate status
        quality_gate_status = self._evaluate_quality_gates(
            {
                "calibration_quality_score": calibration_quality_score,
                "coverage_percentage": coverage_percentage,
                "convergence_rate": convergence_rate,
                "consensus_score": consensus_score,
                "aggregation_stability": aggregation_stability
            },
            self.default_thresholds["aggregation"]
        )
        
        # Build calibration parameters
        calibration_parameters = {
            "aggregation_method": aggregation_method,
            "convergence_tolerance": convergence_tolerance,
            "max_iterations": max_iterations,
            "regularization_strength": regularization_strength,
            "uncertainty_decomposition": True,
            "spectral_analysis_enabled": True,
            **(custom_parameters or {})
        }
        
        artifact = AggregationCalibrationArtifact(
            calibration_quality_score=calibration_quality_score,
            coverage_percentage=coverage_percentage,
            decisions_made=decisions_made,
            quality_gate_status=quality_gate_status,
            timestamp=datetime.utcnow().isoformat(),
            stage_version=stage_version,
            calibration_parameters=calibration_parameters,
            quality_thresholds=self.default_thresholds["aggregation"],
            convergence_rate=convergence_rate,
            aggregation_stability=aggregation_stability,
            consensus_score=consensus_score,
            uncertainty_quantification_quality=uncertainty_quantification_quality,
            spectral_gap=spectral_gap,
            eigenvalue_stability=eigenvalue_stability,
            manifold_coherence=manifold_coherence,
            aggregation_method=aggregation_method,
            convergence_tolerance=convergence_tolerance,
            max_iterations=max_iterations,
            regularization_strength=regularization_strength,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            calibration_curve_auc=calibration_curve_auc
        )
        
        # Save to canonical location
        artifact_path = self.calibration_dir / "aggregation_calibration.json"
        artifact.save(artifact_path)
        
        self.logger.info(f"Generated aggregation calibration artifact: {quality_gate_status}")
        return artifact
    
    def _evaluate_quality_gates(
        self, 
        metrics: Dict[str, float], 
        thresholds: Dict[str, float]
    ) -> str:
        """
        Evaluate quality gate status based on metrics and thresholds.
        
        Args:
            metrics: Current metric values
            thresholds: Threshold values for quality gates
            
        Returns:
            Quality gate status: "PASS", "WARNING", or "FAIL"
        """
        fails = 0
        warnings = 0
        total = 0
        
        for metric, value in metrics.items():
            if metric in thresholds:
                threshold = thresholds[metric]
                total += 1
                
                # Handle inverse metrics (lower is better)
                if metric in ["calibration_error", "adaptive_coverage_gap"]:
                    if value > threshold * 1.5:
                        fails += 1
                    elif value > threshold:
                        warnings += 1
                else:
                    # Standard metrics (higher is better)
                    if value < threshold * 0.7:
                        fails += 1
                    elif value < threshold:
                        warnings += 1
        
        if fails > 0:
            return "FAIL"
        elif warnings > 0:
            return "WARNING"
        else:
            return "PASS"
    
    def detect_calibration_drift(
        self, 
        current_artifact: CalibrationArtifact,
        historical_artifacts: Optional[List[CalibrationArtifact]] = None,
        drift_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect calibration drift by comparing current artifact with historical data.
        
        Args:
            current_artifact: Current calibration artifact
            historical_artifacts: List of historical artifacts for comparison
            drift_threshold: Threshold for detecting significant drift
            
        Returns:
            Drift analysis results
        """
        if not historical_artifacts:
            # Try to load historical artifacts from disk
            historical_artifacts = self._load_historical_artifacts(type(current_artifact).__name__)
        
        if len(historical_artifacts) < 2:
            return {
                "drift_detected": False,
                "reason": "Insufficient historical data",
                "artifact_count": len(historical_artifacts)
            }
        
        # Analyze key metrics for drift
        key_metrics = [
            "calibration_quality_score",
            "coverage_percentage"
        ]
        
        drift_results = {}
        significant_drift = False
        
        for metric in key_metrics:
            if hasattr(current_artifact, metric):
                current_value = getattr(current_artifact, metric)
                historical_values = [
                    getattr(artifact, metric) 
                    for artifact in historical_artifacts 
                    if hasattr(artifact, metric)
                ]
                
                if historical_values:
                    historical_mean = np.mean(historical_values)
                    historical_std = np.std(historical_values)
                    
                    # Z-score based drift detection
                    if historical_std > 0:
                        z_score = abs(current_value - historical_mean) / historical_std
                        is_drift = z_score > 2.0  # 2 standard deviations
                        
                        drift_results[metric] = {
                            "current_value": current_value,
                            "historical_mean": historical_mean,
                            "historical_std": historical_std,
                            "z_score": z_score,
                            "drift_detected": is_drift
                        }
                        
                        if is_drift:
                            significant_drift = True
        
        return {
            "drift_detected": significant_drift,
            "metrics_analysis": drift_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _load_historical_artifacts(self, artifact_type: str) -> List[CalibrationArtifact]:
        """Load historical artifacts from disk for drift analysis."""
        # Map artifact types to filenames
        filename_map = {
            "RetrievalCalibrationArtifact": "retrieval_calibration.json",
            "ConfidenceCalibrationArtifact": "confidence_calibration.json", 
            "AggregationCalibrationArtifact": "aggregation_calibration.json"
        }
        
        artifacts = []
        if artifact_type in filename_map:
            artifact_path = self.calibration_dir / filename_map[artifact_type]
            if artifact_path.exists():
                # For simplicity, just load the current artifact
                # In production, this would load from a time-series database
                try:
                    with open(artifact_path, 'r') as f:
                        data = json.load(f)
                    
                    # Create appropriate artifact type
                    if artifact_type == "RetrievalCalibrationArtifact":
                        artifacts.append(RetrievalCalibrationArtifact.from_dict(data))
                    elif artifact_type == "ConfidenceCalibrationArtifact":
                        artifacts.append(ConfidenceCalibrationArtifact.from_dict(data))
                    elif artifact_type == "AggregationCalibrationArtifact":
                        artifacts.append(AggregationCalibrationArtifact.from_dict(data))
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load historical artifact {artifact_path}: {e}")
        
        return artifacts
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of all calibration artifacts."""
        summary = {
            "retrieval": None,
            "confidence": None,
            "aggregation": None,
            "overall_status": "UNKNOWN",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Load each artifact if it exists
        artifact_files = [
            ("retrieval", "retrieval_calibration.json", RetrievalCalibrationArtifact),
            ("confidence", "confidence_calibration.json", ConfidenceCalibrationArtifact),
            ("aggregation", "aggregation_calibration.json", AggregationCalibrationArtifact)
        ]
        
        all_status = []
        
        for stage, filename, artifact_class in artifact_files:
            artifact_path = self.calibration_dir / filename
            if artifact_path.exists():
                try:
                    artifact = artifact_class.load(artifact_path)
                    summary[stage] = {
                        "quality_gate_status": artifact.quality_gate_status,
                        "calibration_quality_score": artifact.calibration_quality_score,
                        "coverage_percentage": artifact.coverage_percentage,
                        "decisions_made": artifact.decisions_made,
                        "timestamp": artifact.timestamp
                    }
                    all_status.append(artifact.quality_gate_status)
                except Exception as e:
                    self.logger.error(f"Failed to load {stage} artifact: {e}")
        
        # Determine overall status
        if "FAIL" in all_status:
            summary["overall_status"] = "FAIL"
        elif "WARNING" in all_status:
            summary["overall_status"] = "WARNING"
        elif all_status and all(status == "PASS" for status in all_status):
            summary["overall_status"] = "PASS"
        
        return summary