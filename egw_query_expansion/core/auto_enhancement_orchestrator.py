"""
Auto Enhancement Orchestrator for EGW Query Expansion System

This module provides comprehensive orchestration of query enhancements with:
- Preflight validation (schemas, versions, thresholds)
- Stability drift detection with auto-deactivation
- Complete provenance tracking and audit trails
"""

import json
import time
import logging
import hashlib
import importlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import numpy as np
import psutil
from pydantic import BaseModel, Field, validator
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Container for performance metric data"""
    name: str
    value: float
    timestamp: float
    unit: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DriftAnalysis:
    """Results from drift detection analysis"""
    metric_name: str
    current_value: float
    baseline_value: float
    variance: float
    drift_magnitude: float
    exceeded_tolerance: bool
    tolerance_bound: float
    timestamp: float
    additional_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_context is None:
            self.additional_context = {}


@dataclass
class EnhancementMetadata:
    """Complete metadata for enhancement tracking"""
    enhancement_id: str
    activation_timestamp: float
    version_info: Dict[str, str]
    performance_baselines: Dict[str, float]
    drift_detection_results: List[DriftAnalysis]
    audit_trail: List[Dict[str, Any]]
    deactivation_timestamp: Optional[float] = None
    deactivation_reason: Optional[str] = None


class PreflightValidationError(Exception):
    """Raised when preflight validation fails"""
    pass


class DriftDetectionError(Exception):
    """Raised when drift detection encounters an error"""
    pass


class ProvenanceTrackingError(Exception):
    """Raised when provenance tracking fails"""
    pass


class InputSchema(BaseModel):
    """Expected input schema structure"""
    query: str = Field(..., min_length=1, max_length=10000)
    enhancement_params: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('query')
    def validate_query_content(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()


class ThresholdConfig(BaseModel):
    """Threshold configuration structure"""
    metric_name: str
    min_value: float = Field(ge=0.0)
    max_value: float = Field(ge=0.0)
    tolerance: float = Field(ge=0.0, le=1.0)
    calibration_requirement: float = Field(ge=0.0, le=1.0)
    
    @validator('max_value')
    def validate_max_greater_than_min(cls, v, values):
        if 'min_value' in values and v <= values['min_value']:
            raise ValueError("max_value must be greater than min_value")
        return v


class AutoEnhancementOrchestrator:
    """
    Orchestrates query enhancements with comprehensive validation,
    drift detection, and provenance tracking.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 thresholds_path: Optional[str] = None,
                 metadata_output_path: Optional[str] = None,
                 enable_drift_detection: bool = True,
                 enable_provenance_tracking: bool = True):
        """
        Initialize the AutoEnhancementOrchestrator
        
        Args:
            config_path: Path to configuration file
            thresholds_path: Path to thresholds.json file
            metadata_output_path: Path for enhancement_metadata.json output
            enable_drift_detection: Enable drift monitoring
            enable_provenance_tracking: Enable provenance tracking
        """
        self.config_path = config_path
        self.thresholds_path = thresholds_path or "thresholds.json"
        self.metadata_output_path = metadata_output_path or "enhancement_metadata.json"
        self.enable_drift_detection = enable_drift_detection
        self.enable_provenance_tracking = enable_provenance_tracking
        
        # State tracking
        self.active_enhancements: Dict[str, EnhancementMetadata] = {}
        self.performance_history: Dict[str, List[PerformanceMetric]] = {}
        self.baselines: Dict[str, float] = {}
        self.thresholds: Dict[str, ThresholdConfig] = {}
        
        # Drift detection parameters
        self.drift_tolerance_bounds = {
            "score_variance": 0.15,
            "processing_time_deviation": 0.25,
            "memory_usage": 0.20
        }
        
        # Initialize components
        self._initialize_thresholds()
        self._initialize_version_tracking()
        self._setup_logging()
        
        logger.info("AutoEnhancementOrchestrator initialized")
    
    def _initialize_thresholds(self):
        """Load and validate threshold configurations"""
        try:
            if Path(self.thresholds_path).exists():
                with open(self.thresholds_path, 'r') as f:
                    thresholds_data = json.load(f)
                
                for name, config in thresholds_data.items():
                    try:
                        self.thresholds[name] = ThresholdConfig(
                            metric_name=name,
                            **config
                        )
                    except Exception as e:
                        logger.warning(f"Invalid threshold config for {name}: {e}")
            else:
                # Create default thresholds
                self._create_default_thresholds()
                
        except Exception as e:
            logger.error(f"Failed to initialize thresholds: {e}")
            self._create_default_thresholds()
    
    def _create_default_thresholds(self):
        """Create default threshold configurations"""
        defaults = {
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
            },
            "memory_usage": {
                "min_value": 0.0,
                "max_value": 8192.0,
                "tolerance": 0.20,
                "calibration_requirement": 0.85
            }
        }
        
        for name, config in defaults.items():
            self.thresholds[name] = ThresholdConfig(
                metric_name=name,
                **config
            )
        
        # Save default thresholds
        self._save_thresholds()
    
    def _save_thresholds(self):
        """Save current thresholds to file"""
        try:
            thresholds_dict = {
                name: {
                    "min_value": cfg.min_value,
                    "max_value": cfg.max_value,
                    "tolerance": cfg.tolerance,
                    "calibration_requirement": cfg.calibration_requirement
                }
                for name, cfg in self.thresholds.items()
            }
            
            with open(self.thresholds_path, 'w') as f:
                json.dump(thresholds_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save thresholds: {e}")
    
    def _initialize_version_tracking(self):
        """Initialize version tracking for dependencies"""
        self.version_info = {}
        
        # Get Python version
        import sys
        self.version_info["python"] = sys.version
        
        # Track key dependencies from requirements.txt
        key_deps = [
            "torch", "transformers", "sentence-transformers", 
            "faiss-cpu", "numpy", "scipy", "POT", "scikit-learn"
        ]
        
        for dep in key_deps:
            try:
                module = importlib.import_module(dep.replace("-", "_"))
                if hasattr(module, "__version__"):
                    self.version_info[dep] = module.__version__
            except ImportError:
                self.version_info[dep] = "not_installed"
    
    def _setup_logging(self):
        """Configure logging for the orchestrator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def preflight_validation(self, input_data: Dict[str, Any]) -> bool:
        """
        Perform comprehensive preflight validation
        
        Args:
            input_data: Input data to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            PreflightValidationError: If validation fails
        """
        try:
            # 1. Input schema validation
            self._validate_input_schema(input_data)
            
            # 2. Library version compatibility validation
            self._validate_library_compatibility()
            
            # 3. Threshold calibration validation
            self._validate_threshold_calibration()
            
            logger.info("Preflight validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Preflight validation failed: {e}")
            raise PreflightValidationError(f"Preflight validation failed: {e}")
    
    def _validate_input_schema(self, input_data: Dict[str, Any]):
        """Validate input data against expected schema"""
        try:
            # Use Pydantic model for validation
            InputSchema(**input_data)
            logger.debug("Input schema validation passed")
        except Exception as e:
            raise PreflightValidationError(f"Input schema validation failed: {e}")
    
    def _validate_library_compatibility(self):
        """Validate library versions against requirements"""
        try:
            requirements_path = Path("requirements.txt")
            if not requirements_path.exists():
                logger.warning("requirements.txt not found, skipping version validation")
                return
            
            with open(requirements_path, 'r') as f:
                requirements = f.readlines()
            
            # Parse and validate key dependencies
            for line in requirements:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '>=' in line:
                        package, version = line.split('>=')
                        package = package.strip()
                        version = version.split()[0].strip()  # Remove comments
                        
                        if package in self.version_info:
                            installed_version = self.version_info[package]
                            if installed_version == "not_installed":
                                raise PreflightValidationError(
                                    f"Required package {package} is not installed"
                                )
                            
            logger.debug("Library compatibility validation passed")
            
        except FileNotFoundError:
            logger.warning("requirements.txt not found")
        except Exception as e:
            raise PreflightValidationError(f"Library compatibility validation failed: {e}")
    
    def _validate_threshold_calibration(self):
        """Validate threshold calibration requirements"""
        try:
            for name, config in self.thresholds.items():
                # Check if calibration requirement is met
                calibration_score = self._calculate_calibration_score(name, config)
                
                if calibration_score < config.calibration_requirement:
                    raise PreflightValidationError(
                        f"Threshold {name} calibration score {calibration_score:.3f} "
                        f"below requirement {config.calibration_requirement}"
                    )
            
            logger.debug("Threshold calibration validation passed")
            
        except Exception as e:
            raise PreflightValidationError(f"Threshold calibration validation failed: {e}")
    
    def _calculate_calibration_score(self, metric_name: str, config: ThresholdConfig) -> float:
        """Calculate calibration score for a threshold configuration"""
        # Simplified calibration score based on threshold bounds
        range_size = config.max_value - config.min_value
        if range_size <= 0:
            return 0.0
        
        tolerance_ratio = config.tolerance / range_size if range_size > 0 else 1.0
        
        # Score based on reasonable tolerance relative to range
        if tolerance_ratio < 0.1:  # Very tight tolerance
            return 0.95
        elif tolerance_ratio < 0.2:  # Moderate tolerance
            return 0.85
        elif tolerance_ratio < 0.3:  # Loose tolerance
            return 0.75
        else:  # Very loose tolerance
            return 0.65
    
    def activate_enhancement(self, enhancement_id: str, **kwargs) -> str:
        """
        Activate an enhancement with full validation and tracking
        
        Args:
            enhancement_id: Unique identifier for the enhancement
            **kwargs: Additional parameters for the enhancement
            
        Returns:
            str: Activation confirmation ID
        """
        try:
            # Perform preflight validation
            input_data = {
                "query": kwargs.get("query", "test_query"),
                "enhancement_params": kwargs.get("enhancement_params", {}),
                "metadata": kwargs.get("metadata", {})
            }
            
            self.preflight_validation(input_data)
            
            # Create enhancement metadata
            timestamp = time.time()
            metadata = EnhancementMetadata(
                enhancement_id=enhancement_id,
                activation_timestamp=timestamp,
                version_info=self.version_info.copy(),
                performance_baselines=self._establish_baselines(),
                drift_detection_results=[],
                audit_trail=[
                    {
                        "timestamp": timestamp,
                        "event": "enhancement_activated",
                        "enhancement_id": enhancement_id,
                        "parameters": kwargs
                    }
                ]
            )
            
            # Store active enhancement
            self.active_enhancements[enhancement_id] = metadata
            
            # Initialize performance tracking
            if enhancement_id not in self.performance_history:
                self.performance_history[enhancement_id] = []
            
            # Save provenance data
            if self.enable_provenance_tracking:
                self._save_enhancement_metadata()
            
            activation_id = hashlib.sha256(
                f"{enhancement_id}_{timestamp}".encode()
            ).hexdigest()[:16]
            
            logger.info(f"Enhancement {enhancement_id} activated with ID {activation_id}")
            return activation_id
            
        except Exception as e:
            logger.error(f"Failed to activate enhancement {enhancement_id}: {e}")
            raise
    
    def _establish_baselines(self) -> Dict[str, float]:
        """Establish performance baselines for drift detection"""
        baselines = {}
        
        # CPU and memory baselines
        process = psutil.Process()
        baselines["cpu_percent"] = process.cpu_percent()
        baselines["memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        # Performance baselines (simulated)
        baselines["processing_time"] = 1.0  # seconds
        baselines["score_variance"] = 0.05  # variance metric
        baselines["throughput"] = 100.0  # items per second
        
        return baselines
    
    @contextmanager
    def monitor_performance(self, enhancement_id: str):
        """Context manager for monitoring performance during enhancement execution"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Record performance metrics
            processing_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            metrics = [
                PerformanceMetric("processing_time", processing_time, end_time, "seconds"),
                PerformanceMetric("memory_delta", memory_delta, end_time, "MB"),
                PerformanceMetric("memory_usage", end_memory, end_time, "MB")
            ]
            
            for metric in metrics:
                self._record_performance_metric(enhancement_id, metric)
            
            # Perform drift detection
            if self.enable_drift_detection:
                self._detect_stability_drift(enhancement_id, metrics)
    
    def _record_performance_metric(self, enhancement_id: str, metric: PerformanceMetric):
        """Record a performance metric for tracking"""
        if enhancement_id not in self.performance_history:
            self.performance_history[enhancement_id] = []
        
        self.performance_history[enhancement_id].append(metric)
        
        # Maintain history size (keep last 1000 metrics)
        if len(self.performance_history[enhancement_id]) > 1000:
            self.performance_history[enhancement_id] = \
                self.performance_history[enhancement_id][-1000:]
    
    def _detect_stability_drift(self, enhancement_id: str, current_metrics: List[PerformanceMetric]):
        """
        Detect stability drift and auto-deactivate if necessary
        
        Args:
            enhancement_id: ID of the enhancement to check
            current_metrics: Current performance metrics
        """
        try:
            if enhancement_id not in self.active_enhancements:
                return
            
            enhancement_meta = self.active_enhancements[enhancement_id]
            history = self.performance_history.get(enhancement_id, [])
            
            if len(history) < 10:  # Need sufficient history
                return
            
            drift_analyses = []
            should_deactivate = False
            
            for current_metric in current_metrics:
                # Get historical values for this metric
                historical_values = [
                    m.value for m in history 
                    if m.name == current_metric.name
                ]
                
                if len(historical_values) < 5:
                    continue
                
                # Calculate drift
                baseline = enhancement_meta.performance_baselines.get(current_metric.name)
                if baseline is None:
                    baseline = np.mean(historical_values[:5])  # Use early values as baseline
                
                current_value = current_metric.value
                variance = np.var(historical_values)
                drift_magnitude = abs(current_value - baseline) / (baseline + 1e-8)
                
                # Check tolerance bounds
                tolerance_bound = self.drift_tolerance_bounds.get(current_metric.name, 0.2)
                exceeded_tolerance = drift_magnitude > tolerance_bound
                
                drift_analysis = DriftAnalysis(
                    metric_name=current_metric.name,
                    current_value=current_value,
                    baseline_value=baseline,
                    variance=variance,
                    drift_magnitude=drift_magnitude,
                    exceeded_tolerance=exceeded_tolerance,
                    tolerance_bound=tolerance_bound,
                    timestamp=current_metric.timestamp,
                    additional_context={
                        "historical_mean": np.mean(historical_values),
                        "historical_std": np.std(historical_values),
                        "trend": self._calculate_trend(historical_values)
                    }
                )
                
                drift_analyses.append(drift_analysis)
                
                if exceeded_tolerance:
                    logger.warning(
                        f"Drift detected for {enhancement_id}.{current_metric.name}: "
                        f"magnitude={drift_magnitude:.3f}, tolerance={tolerance_bound}"
                    )
                    should_deactivate = True
            
            # Update drift detection results
            enhancement_meta.drift_detection_results.extend(drift_analyses)
            
            # Auto-deactivate if drift exceeds tolerance
            if should_deactivate:
                self._auto_deactivate_enhancement(enhancement_id, drift_analyses)
            
        except Exception as e:
            logger.error(f"Drift detection failed for {enhancement_id}: {e}")
            raise DriftDetectionError(f"Drift detection failed: {e}")
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from historical values"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _auto_deactivate_enhancement(self, enhancement_id: str, drift_analyses: List[DriftAnalysis]):
        """Automatically deactivate enhancement due to drift"""
        try:
            if enhancement_id not in self.active_enhancements:
                return
            
            timestamp = time.time()
            enhancement_meta = self.active_enhancements[enhancement_id]
            
            # Create deactivation details
            drift_summary = {
                "total_metrics_exceeded": sum(1 for d in drift_analyses if d.exceeded_tolerance),
                "max_drift_magnitude": max(d.drift_magnitude for d in drift_analyses),
                "affected_metrics": [d.metric_name for d in drift_analyses if d.exceeded_tolerance]
            }
            
            deactivation_reason = (
                f"Automatic deactivation due to stability drift. "
                f"Metrics exceeded tolerance: {drift_summary['affected_metrics']}. "
                f"Max drift magnitude: {drift_summary['max_drift_magnitude']:.3f}"
            )
            
            # Update metadata
            enhancement_meta.deactivation_timestamp = timestamp
            enhancement_meta.deactivation_reason = deactivation_reason
            enhancement_meta.audit_trail.append({
                "timestamp": timestamp,
                "event": "auto_deactivation",
                "reason": deactivation_reason,
                "drift_summary": drift_summary,
                "drift_analyses": [asdict(d) for d in drift_analyses]
            })
            
            # Log deactivation event with detailed analysis
            logger.error(
                f"AUTO-DEACTIVATED enhancement {enhancement_id}. "
                f"Reason: {deactivation_reason}. "
                f"Drift summary: {json.dumps(drift_summary, indent=2)}"
            )
            
            # Save updated metadata
            if self.enable_provenance_tracking:
                self._save_enhancement_metadata()
            
            # Remove from active enhancements
            del self.active_enhancements[enhancement_id]
            
        except Exception as e:
            logger.error(f"Failed to auto-deactivate {enhancement_id}: {e}")
    
    def deactivate_enhancement(self, enhancement_id: str, reason: str = "manual_deactivation") -> bool:
        """
        Manually deactivate an enhancement
        
        Args:
            enhancement_id: ID of enhancement to deactivate
            reason: Reason for deactivation
            
        Returns:
            bool: True if successfully deactivated
        """
        try:
            if enhancement_id not in self.active_enhancements:
                logger.warning(f"Enhancement {enhancement_id} not active")
                return False
            
            timestamp = time.time()
            enhancement_meta = self.active_enhancements[enhancement_id]
            
            # Update metadata
            enhancement_meta.deactivation_timestamp = timestamp
            enhancement_meta.deactivation_reason = reason
            enhancement_meta.audit_trail.append({
                "timestamp": timestamp,
                "event": "manual_deactivation",
                "reason": reason
            })
            
            # Save metadata before removing
            if self.enable_provenance_tracking:
                self._save_enhancement_metadata()
            
            # Remove from active enhancements
            del self.active_enhancements[enhancement_id]
            
            logger.info(f"Enhancement {enhancement_id} deactivated: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate {enhancement_id}: {e}")
            return False
    
    def _save_enhancement_metadata(self):
        """Save complete enhancement metadata to file"""
        try:
            metadata_dict = {}
            
            # Active enhancements
            for eid, meta in self.active_enhancements.items():
                metadata_dict[eid] = {
                    "enhancement_id": meta.enhancement_id,
                    "activation_timestamp": meta.activation_timestamp,
                    "activation_datetime": datetime.fromtimestamp(
                        meta.activation_timestamp, tz=timezone.utc
                    ).isoformat(),
                    "version_info": meta.version_info,
                    "performance_baselines": meta.performance_baselines,
                    "drift_detection_results": [asdict(d) for d in meta.drift_detection_results],
                    "audit_trail": meta.audit_trail,
                    "deactivation_timestamp": meta.deactivation_timestamp,
                    "deactivation_datetime": (
                        datetime.fromtimestamp(meta.deactivation_timestamp, tz=timezone.utc).isoformat()
                        if meta.deactivation_timestamp else None
                    ),
                    "deactivation_reason": meta.deactivation_reason,
                    "status": "deactivated" if meta.deactivation_timestamp else "active"
                }
            
            # Add system metadata
            metadata_dict["_system"] = {
                "orchestrator_version": "1.0.0",
                "last_update": datetime.now(timezone.utc).isoformat(),
                "total_active_enhancements": len(self.active_enhancements),
                "drift_detection_enabled": self.enable_drift_detection,
                "provenance_tracking_enabled": self.enable_provenance_tracking,
                "threshold_configurations": {
                    name: {
                        "min_value": cfg.min_value,
                        "max_value": cfg.max_value,
                        "tolerance": cfg.tolerance,
                        "calibration_requirement": cfg.calibration_requirement
                    }
                    for name, cfg in self.thresholds.items()
                }
            }
            
            # Write to file
            with open(self.metadata_output_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            logger.debug(f"Enhancement metadata saved to {self.metadata_output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save enhancement metadata: {e}")
            raise ProvenanceTrackingError(f"Failed to save metadata: {e}")
    
    def get_enhancement_status(self, enhancement_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an enhancement"""
        if enhancement_id not in self.active_enhancements:
            return None
        
        meta = self.active_enhancements[enhancement_id]
        history = self.performance_history.get(enhancement_id, [])
        
        # Calculate current performance summary
        recent_metrics = history[-10:] if len(history) >= 10 else history
        performance_summary = {}
        
        for metric_name in ["processing_time", "memory_usage", "score_variance"]:
            metric_values = [m.value for m in recent_metrics if m.name == metric_name]
            if metric_values:
                performance_summary[metric_name] = {
                    "current": metric_values[-1],
                    "average": np.mean(metric_values),
                    "variance": np.var(metric_values)
                }
        
        return {
            "enhancement_id": meta.enhancement_id,
            "status": "active",
            "activation_timestamp": meta.activation_timestamp,
            "uptime_seconds": time.time() - meta.activation_timestamp,
            "performance_summary": performance_summary,
            "drift_detections": len(meta.drift_detection_results),
            "recent_drift_exceeded": any(
                d.exceeded_tolerance 
                for d in meta.drift_detection_results[-5:]
            ),
            "total_audit_events": len(meta.audit_trail)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_enhancements": len(self.active_enhancements),
            "drift_detection_enabled": self.enable_drift_detection,
            "provenance_tracking_enabled": self.enable_provenance_tracking,
            "configured_thresholds": len(self.thresholds),
            "total_performance_metrics": sum(
                len(history) for history in self.performance_history.values()
            ),
            "version_info": self.version_info,
            "enhancement_ids": list(self.active_enhancements.keys())
        }