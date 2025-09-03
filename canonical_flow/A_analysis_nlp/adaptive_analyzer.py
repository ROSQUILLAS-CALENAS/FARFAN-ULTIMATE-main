"""
Canonical Flow Alias: 13A
Adaptive Analyzer with Total Ordering and Deterministic Processing

Source: adaptive_analyzer.py
Stage: analysis_nlp
Code: 13A
"""

import asyncio
import json
import logging
import statistics
import os
# # # from collections import deque, OrderedDict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional, Tuple  # Module not found  # Module not found  # Module not found
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# # # from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


class AnalysisMode(str, Enum):
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    PROACTIVE = "proactive"
    LEARNING = "learning"


class SystemState(str, Enum):
    OPTIMAL = "optimal"
    STABLE = "stable"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class AdaptationAction(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    ADJUST_TIMEOUT = "adjust_timeout"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    ADJUST_BACKPRESSURE = "adjust_backpressure"
    REROUTE_TRAFFIC = "reroute_traffic"
    ENABLE_CACHING = "enable_caching"
    ADJUST_PARALLELISM = "adjust_parallelism"
    TRIGGER_MAINTENANCE = "trigger_maintenance"
    NO_ACTION = "no_action"


@dataclass
class AdaptationRecommendation:
    action: AdaptationAction
    component: str
    reason: str
    confidence: float
    expected_impact: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    estimated_execution_time: float = 0.0
    rollback_procedure: Optional[str] = None
    
    def __post_init__(self):
        # Ensure deterministic ordering of parameters
        if self.parameters:
            self.parameters = OrderedDict(sorted(self.parameters.items()))


class AdaptiveAnalyzer(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Adaptive Analyzer with deterministic processing, total ordering, and comprehensive audit logging.
    
    Provides consistent analysis results, stable ID generation across runs, and complete
    execution traceability through standardized audit logs.
    """
    
    def __init__(self, mode: AnalysisMode = AnalysisMode.PREDICTIVE):
        super().__init__("AdaptiveAnalyzer")
        self.mode = mode
        self.stage_name = "A_analysis_nlp"  # Set stage name for audit logging
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.adaptation_handlers: Dict[str, Callable] = {}
        self.history: List[Dict[str, Any]] = []
        self.last_analysis: Optional[Dict[str, Any]] = None
        
        # Configuration with deterministic defaults
        self.analysis_interval = 30.0
        self.metric_retention_hours = 24
        self.anomaly_threshold = 2.5
        self.min_confidence_for_action = 0.7
        
        # Internal state
        self._running = False
        self._analysis_task: Optional[asyncio.Task] = None
        
        self._setup_default_thresholds()
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "mode": self.mode.value,
            "thresholds": self.sort_dict_by_keys(self.thresholds),
            "configuration": {
                "analysis_interval": self.analysis_interval,
                "metric_retention_hours": self.metric_retention_hours,
                "anomaly_threshold": self.anomaly_threshold,
                "min_confidence_for_action": self.min_confidence_for_action,
            }
        }
    
    def _setup_default_thresholds(self):
        """Setup deterministic default thresholds"""
        default_thresholds = {
            "cpu_usage": {"critical": 85.0, "optimal": 50.0, "warning": 70.0},
            "error_rate": {"critical": 0.10, "optimal": 0.01, "warning": 0.05},
            "memory_usage": {"critical": 90.0, "optimal": 60.0, "warning": 75.0},
            "queue_size": {"critical": 5000, "optimal": 100, "warning": 1000},
            "response_time": {"critical": 10.0, "optimal": 1.0, "warning": 5.0},
            "throughput": {"critical": 10, "optimal": 100, "warning": 50},
        }
        self.thresholds = self.sort_dict_by_keys(default_thresholds)
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function with deterministic output and comprehensive audit logging.
        Schema validation applied via decorator when available.
        
        Args:
            data: Input data for analysis
            context: Processing context
            
        Returns:
            Deterministic analysis results with audit metadata
        """
        # Use audit-enabled processing if available
        if hasattr(self, 'process_with_audit'):
            return self.process_with_audit(data, context)
        
        # Fallback to standard processing
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Perform analysis
            analysis_results = self._perform_deterministic_analysis(
                canonical_data, canonical_context
            )
            
            # Generate deterministic output
            output = self._generate_deterministic_output(analysis_results, operation_id)
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            error_output = {
                "component": self.component_name,
                "operation_id": operation_id,
                "error": str(e),
                "status": "error",
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _perform_deterministic_analysis(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deterministic analysis with stable results"""
        
        # Extract metrics with stable ordering
        metrics = self._extract_metrics_deterministic(data)
        
        # Analyze system state
        system_state = self._analyze_system_state_deterministic(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations_deterministic(system_state, metrics)
        
        return {
            "metrics": self.sort_dict_by_keys(metrics),
            "system_state": system_state,
            "recommendations": self.sort_collection(recommendations, key_func=lambda r: r.action.value),
            "analysis_timestamp": self._get_deterministic_timestamp(),
        }
    
    def _extract_metrics_deterministic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics with deterministic processing"""
        metrics = {}
        
        # Standard metric extraction with sorted keys
        if "performance" in data:
            perf_data = data["performance"]
            if isinstance(perf_data, dict):
                for key, value in sorted(perf_data.items()):
                    if isinstance(value, (int, float)):
                        metrics[f"performance_{key}"] = float(value)
        
        # Resource metrics
        if "resources" in data:
            res_data = data["resources"]
            if isinstance(res_data, dict):
                for key, value in sorted(res_data.items()):
                    if isinstance(value, (int, float)):
                        metrics[f"resource_{key}"] = float(value)
        
        # Error metrics
        if "errors" in data:
            error_data = data["errors"]
            if isinstance(error_data, (list, int)):
                metrics["error_count"] = len(error_data) if isinstance(error_data, list) else int(error_data)
        
        return self.sort_dict_by_keys(metrics)
    
    def _analyze_system_state_deterministic(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system state with deterministic logic"""
        state_scores = []
        component_states = {}
        
        # Analyze each metric against thresholds
        for metric_name, value in sorted(metrics.items()):
            base_name = metric_name.split('_', 1)[-1] if '_' in metric_name else metric_name
            
            if base_name in self.thresholds:
                thresholds = self.thresholds[base_name]
                state = self._calculate_metric_state(value, thresholds)
                component_states[metric_name] = state
                state_scores.append(self._state_to_numeric(state))
        
        # Calculate overall state
        overall_state = "unknown"
        if state_scores:
            avg_score = sum(state_scores) / len(state_scores)
            overall_state = self._numeric_to_state(avg_score)
        
        return {
            "overall_state": overall_state,
            "component_states": self.sort_dict_by_keys(component_states),
            "confidence": min(1.0, len(state_scores) / 5.0),
            "health_score": max(0.0, 100.0 - (avg_score * 25.0)) if state_scores else 50.0,
        }
    
    def _calculate_metric_state(self, value: float, thresholds: Dict[str, float]) -> str:
        """Calculate state for a metric value"""
        if "critical" in thresholds and value >= thresholds["critical"]:
            return "critical"
        elif "warning" in thresholds and value >= thresholds["warning"]:
            return "degraded"
        elif "optimal" in thresholds and value <= thresholds["optimal"]:
            return "optimal"
        else:
            return "stable"
    
    def _state_to_numeric(self, state: str) -> float:
        """Convert state to numeric value for calculations"""
        state_map = {
            "optimal": 0.0,
            "stable": 1.0,
            "degraded": 2.0,
            "critical": 3.0,
            "unknown": 1.5,
        }
        return state_map.get(state, 1.5)
    
    def _numeric_to_state(self, value: float) -> str:
        """Convert numeric value back to state"""
        if value <= 0.5:
            return "optimal"
        elif value <= 1.5:
            return "stable"
        elif value <= 2.5:
            return "degraded"
        else:
            return "critical"
    
    def _generate_recommendations_deterministic(self, system_state: Dict[str, Any], metrics: Dict[str, Any]) -> List[AdaptationRecommendation]:
        """Generate recommendations with deterministic ordering"""
        recommendations = []
        
        overall_state = system_state.get("overall_state", "unknown")
        component_states = system_state.get("component_states", {})
        
        # Generate recommendations based on component states
        for component, state in sorted(component_states.items()):
            if state == "critical":
                rec = AdaptationRecommendation(
                    action=AdaptationAction.SCALE_UP,
                    component=component,
                    reason=f"Critical state detected in {component}",
                    confidence=0.9,
                    expected_impact="Reduce system load",
                    priority=3,
                    parameters={"component": component, "state": state}
                )
                recommendations.append(rec)
            elif state == "degraded":
                rec = AdaptationRecommendation(
                    action=AdaptationAction.ADJUST_TIMEOUT,
                    component=component,
                    reason=f"Performance degradation in {component}",
                    confidence=0.7,
                    expected_impact="Improve response times",
                    priority=2,
                    parameters={"component": component, "state": state}
                )
                recommendations.append(rec)
        
        return recommendations
    
    def _generate_deterministic_output(self, analysis_results: Dict[str, Any], operation_id: str) -> Dict[str, Any]:
        """Generate deterministic output structure"""
        output = {
            "component": self.component_name,
            "component_id": self.component_id,
            "operation_id": operation_id,
            "analysis_mode": self.mode.value,
            "results": analysis_results,
            "metadata": self.get_deterministic_metadata(),
            "status": "success",
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(output)
    
    def save_artifact(self, data: Dict[str, Any], document_stem: str, output_dir: str = "canonical_flow/analysis") -> str:
        """
        Save analysis output to canonical_flow/analysis/ directory with standardized naming.
        
        Args:
            data: Analysis data to save
            document_stem: Base document identifier (without extension)
            output_dir: Output directory (defaults to canonical_flow/analysis)
            
        Returns:
            Path to saved artifact
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate standardized filename
            filename = f"{document_stem}_adaptive.json"
            artifact_path = Path(output_dir) / filename
            
            # Save with consistent JSON formatting
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"AdaptiveAnalyzer artifact saved: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            error_msg = f"Failed to save AdaptiveAnalyzer artifact to {output_dir}/{document_stem}_adaptive.json: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Maintain backward compatibility with original interface
def process(data=None, context=None):
    """Backward compatible process function"""
    analyzer = AdaptiveAnalyzer()
    result = analyzer.process(data, context)
    
    # Save artifact if document_stem is provided
    if data and isinstance(data, dict) and 'document_stem' in data:
        analyzer.save_artifact(result, data['document_stem'])
    
    return result


if __name__ == "__main__":
    # Minimal demo: run with a tiny deterministic input and print JSON
    demo_input = {
        "document_stem": "demo",
        "performance": {"latency_ms": 120.0, "throughput": 85.0},
        "resources": {"cpu_usage": 42.0, "memory_usage": 61.5},
        "errors": []
    }
    output = process(demo_input, context={"mode": "predictive"})
    print(json.dumps(output, indent=2, ensure_ascii=False))
