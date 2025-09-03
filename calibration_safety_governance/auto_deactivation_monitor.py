"""
Auto-Deactivation Monitoring System
====================================

Tracks stability drift, evidence quality degradation, and performance regressions
to automatically disable enhancements when stability thresholds are exceeded.
"""

import json
import logging
import statistics
# # # from collections import deque, defaultdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any, Optional, Tuple, Set  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
import numpy as np
# # # from pydantic import BaseModel  # Module not found  # Module not found  # Module not found



# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "118O"
__stage_order__ = 7

class DeactivationTriggerType(Enum):
    """Types of deactivation triggers"""
    STABILITY_DRIFT = "stability_drift"
    EVIDENCE_QUALITY = "evidence_quality"
    PERFORMANCE_REGRESSION = "performance_regression"
    SAFETY_VIOLATION = "safety_violation"


class DeactivationSeverity(Enum):
    """Severity levels for deactivation events"""
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class MonitoringPoint:
    """Single monitoring data point"""
    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict[str, Any] = None


@dataclass
class DeactivationEvent:
    """Deactivation event record"""
    trigger_type: DeactivationTriggerType
    severity: DeactivationSeverity
    enhancement_id: str
    trigger_condition: str
    metrics: Dict[str, float]
    timestamp: datetime
    cooldown_until: datetime
    metadata: Dict[str, Any] = None


class StabilityDriftAnalyzer:
    """Analyzes score variance and stability drift"""
    
    def __init__(self, window_size: int = 10, variance_threshold: float = 0.15):
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.score_windows = defaultdict(lambda: deque(maxlen=window_size))
        
    def add_score(self, enhancement_id: str, score: float, timestamp: datetime = None):
        """Add new score measurement"""
        if timestamp is None:
            timestamp = datetime.now()
            
        point = MonitoringPoint(timestamp, "score", score)
        self.score_windows[enhancement_id].append(point)
        
    def analyze_stability_drift(self, enhancement_id: str) -> Dict[str, Any]:
        """Analyze stability drift for specific enhancement"""
        window = self.score_windows.get(enhancement_id)
        if not window or len(window) < 3:
            return {"sufficient_data": False}
            
        scores = [point.value for point in window]
        
        # Calculate variance and standard deviation
        variance = statistics.variance(scores)
        stddev = statistics.stdev(scores)
        mean_score = statistics.mean(scores)
        
        # Calculate drift coefficient (trend analysis)
        if len(scores) >= 5:
            x = list(range(len(scores)))
            drift_coefficient = np.corrcoef(x, scores)[0, 1] if len(set(scores)) > 1 else 0.0
        else:
            drift_coefficient = 0.0
            
        # Calculate stability metrics
        stability_coefficient = 1.0 - min(stddev / mean_score if mean_score > 0 else 1.0, 1.0)
        
        # Check for concerning patterns
        recent_scores = scores[-3:] if len(scores) >= 3 else scores
        is_degrading = len(recent_scores) >= 2 and all(
            recent_scores[i] >= recent_scores[i+1] for i in range(len(recent_scores)-1)
        )
        
        return {
            "sufficient_data": True,
            "variance": variance,
            "stddev": stddev,
            "mean_score": mean_score,
            "stability_coefficient": stability_coefficient,
            "drift_coefficient": drift_coefficient,
            "is_degrading": is_degrading,
            "exceeds_variance_threshold": stddev > self.variance_threshold,
            "sample_count": len(scores)
        }


class EvidenceQualityTracker:
    """Tracks evidence quality degradation"""
    
    def __init__(self, degradation_threshold: float = 0.1, quality_minimum: float = 0.75):
        self.degradation_threshold = degradation_threshold
        self.quality_minimum = quality_minimum
        self.quality_history = defaultdict(list)
        
    def record_evidence_quality(self, enhancement_id: str, quality_metrics: Dict[str, float]):
        """Record evidence quality metrics"""
        timestamp = datetime.now()
        
        # Store quality metrics with timestamp
        quality_record = {
            "timestamp": timestamp,
            "overall_quality": quality_metrics.get("overall_quality", 0.0),
            "consistency": quality_metrics.get("consistency", 0.0),
            "coverage": quality_metrics.get("coverage", 0.0),
            "coherence": quality_metrics.get("coherence", 0.0)
        }
        
        self.quality_history[enhancement_id].append(quality_record)
        
        # Maintain reasonable history size
        if len(self.quality_history[enhancement_id]) > 50:
            self.quality_history[enhancement_id] = self.quality_history[enhancement_id][-40:]
            
    def detect_quality_degradation(self, enhancement_id: str) -> Dict[str, Any]:
        """Detect evidence quality degradation patterns"""
        history = self.quality_history.get(enhancement_id, [])
        if len(history) < 3:
            return {"sufficient_data": False}
            
        # Calculate degradation rate over recent period
        recent_period = history[-5:] if len(history) >= 5 else history
        
        quality_scores = [record["overall_quality"] for record in recent_period]
        current_quality = quality_scores[-1] if quality_scores else 0.0
        
        # Calculate degradation rate
        if len(quality_scores) >= 2:
            initial_quality = quality_scores[0]
            degradation_rate = (initial_quality - current_quality) / initial_quality if initial_quality > 0 else 0.0
        else:
            degradation_rate = 0.0
            
        # Check consistency trends
        consistency_scores = [record["consistency"] for record in recent_period]
        consistency_degrading = len(consistency_scores) >= 3 and all(
            consistency_scores[i] >= consistency_scores[i+1] for i in range(len(consistency_scores)-1)
        )
        
        # Check coverage adequacy
        coverage_scores = [record["coverage"] for record in recent_period]
        current_coverage = coverage_scores[-1] if coverage_scores else 0.0
        
        return {
            "sufficient_data": True,
            "current_quality": current_quality,
            "degradation_rate": degradation_rate,
            "below_minimum_quality": current_quality < self.quality_minimum,
            "degradation_rate_exceeded": degradation_rate > self.degradation_threshold,
            "consistency_degrading": consistency_degrading,
            "coverage_adequate": current_coverage >= 0.9,
            "trend_analysis": EvidenceQualityTracker._analyze_quality_trend(quality_scores)
        }
        
    @staticmethod
    def _analyze_quality_trend(scores: List[float]) -> Dict[str, Any]:
        """Analyze quality trend patterns"""
        if len(scores) < 3:
            return {"trend": "insufficient_data"}
            
        # Simple trend analysis
        recent_half = scores[len(scores)//2:]
        early_half = scores[:len(scores)//2]
        
        recent_avg = sum(recent_half) / len(recent_half)
        early_avg = sum(early_half) / len(early_half)
        
        if recent_avg < early_avg - 0.05:
            trend = "declining"
        elif recent_avg > early_avg + 0.05:
            trend = "improving"
        else:
            trend = "stable"
            
        return {
            "trend": trend,
            "recent_average": recent_avg,
            "early_average": early_avg,
            "change_magnitude": abs(recent_avg - early_avg)
        }


class PerformanceRegressionDetector:
    """Detects performance regressions across multiple metrics"""
    
    def __init__(self):
        self.performance_history = defaultdict(lambda: defaultdict(list))
        self.baseline_metrics = {}
        
    def establish_baseline(self, enhancement_id: str, metrics: Dict[str, float]):
        """Establish performance baseline for enhancement"""
        self.baseline_metrics[enhancement_id] = {
            "response_time": metrics.get("response_time", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "throughput": metrics.get("throughput", 0.0),
            "error_rate": metrics.get("error_rate", 0.0),
            "established_at": datetime.now()
        }
        
    def record_performance(self, enhancement_id: str, metrics: Dict[str, float]):
        """Record performance metrics"""
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            self.performance_history[enhancement_id][metric_name].append({
                "value": value,
                "timestamp": timestamp
            })
            
        # Maintain reasonable history size
        for metric_name in self.performance_history[enhancement_id]:
            if len(self.performance_history[enhancement_id][metric_name]) > 50:
                self.performance_history[enhancement_id][metric_name] = \
                    self.performance_history[enhancement_id][metric_name][-40:]
                    
    def detect_regressions(self, enhancement_id: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Detect performance regressions"""
        baseline = self.baseline_metrics.get(enhancement_id)
        history = self.performance_history.get(enhancement_id, {})
        
        if not baseline or not history:
            return {"sufficient_data": False}
            
        regressions = {}
        current_performance = {}
        
        # Check response time regression
        if "response_time" in history and history["response_time"]:
            current_response_time = history["response_time"][-1]["value"]
            baseline_response_time = baseline["response_time"]
            response_time_increase = (current_response_time - baseline_response_time) / baseline_response_time if baseline_response_time > 0 else 0.0
            
            current_performance["response_time"] = current_response_time
            regressions["response_time"] = {
                "current": current_response_time,
                "baseline": baseline_response_time,
                "increase_ratio": response_time_increase,
                "regression_detected": response_time_increase > thresholds.get("response_time_increase", 1.5)
            }
            
        # Check accuracy regression
        if "accuracy" in history and history["accuracy"]:
            current_accuracy = history["accuracy"][-1]["value"]
            baseline_accuracy = baseline["accuracy"]
            accuracy_degradation = baseline_accuracy - current_accuracy
            
            current_performance["accuracy"] = current_accuracy
            regressions["accuracy"] = {
                "current": current_accuracy,
                "baseline": baseline_accuracy,
                "degradation": accuracy_degradation,
                "regression_detected": accuracy_degradation > thresholds.get("accuracy_degradation", 0.05)
            }
            
        # Check throughput regression
        if "throughput" in history and history["throughput"]:
            current_throughput = history["throughput"][-1]["value"]
            baseline_throughput = baseline["throughput"]
            throughput_decrease = (baseline_throughput - current_throughput) / baseline_throughput if baseline_throughput > 0 else 0.0
            
            current_performance["throughput"] = current_throughput
            regressions["throughput"] = {
                "current": current_throughput,
                "baseline": baseline_throughput,
                "decrease_ratio": throughput_decrease,
                "regression_detected": throughput_decrease > thresholds.get("throughput_decrease", 0.2)
            }
            
        # Check error rate regression
        if "error_rate" in history and history["error_rate"]:
            current_error_rate = history["error_rate"][-1]["value"]
            baseline_error_rate = baseline["error_rate"]
            error_rate_increase = current_error_rate - baseline_error_rate
            
            current_performance["error_rate"] = current_error_rate
            regressions["error_rate"] = {
                "current": current_error_rate,
                "baseline": baseline_error_rate,
                "increase": error_rate_increase,
                "regression_detected": error_rate_increase > thresholds.get("error_rate_increase", 0.02)
            }
            
        # Calculate overall regression score
        detected_regressions = sum(1 for r in regressions.values() if r.get("regression_detected", False))
        total_metrics = len(regressions)
        regression_score = detected_regressions / total_metrics if total_metrics > 0 else 0.0
        
        return {
            "sufficient_data": True,
            "regressions": regressions,
            "current_performance": current_performance,
            "regression_score": regression_score,
            "critical_regression_detected": detected_regressions >= 2,
            "detected_regressions": detected_regressions,
            "total_metrics": total_metrics
        }


class AutoDeactivationMonitor:
    """Main auto-deactivation monitoring system"""
    
    def __init__(self, thresholds_path: str = "calibration_safety_governance/thresholds.json"):
        self.logger = logging.getLogger(__name__)
        self.thresholds_path = thresholds_path
        self.thresholds = self._load_thresholds()
        
        # Initialize analyzers
        self.stability_analyzer = StabilityDriftAnalyzer()
        self.quality_tracker = EvidenceQualityTracker()
        self.performance_detector = PerformanceRegressionDetector()
        
        # Deactivation tracking
        self.deactivation_events = []
        self.consecutive_violations = defaultdict(int)
        self.active_cooldowns = {}
        
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load monitoring thresholds"""
        try:
            with open(self.thresholds_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load thresholds: {e}")
            return {}
            
    def monitor_enhancement(
        self, 
        enhancement_id: str,
        performance_metrics: Dict[str, float],
        evidence_quality: Dict[str, float],
        score: float
    ) -> Dict[str, Any]:
        """Monitor enhancement for deactivation triggers"""
        
        monitoring_results = {}
        deactivation_triggers = []
        
        # Record metrics
        self.stability_analyzer.add_score(enhancement_id, score)
        self.quality_tracker.record_evidence_quality(enhancement_id, evidence_quality)
        self.performance_detector.record_performance(enhancement_id, performance_metrics)
        
        # Analyze stability drift
        stability_config = self.thresholds.get("stability_monitoring", {}).get("score_variance", {})
        stability_analysis = self.stability_analyzer.analyze_stability_drift(enhancement_id)
        monitoring_results["stability_analysis"] = stability_analysis
        
        if stability_analysis.get("sufficient_data", False):
            if (stability_analysis.get("exceeds_variance_threshold", False) or
                stability_analysis.get("stability_coefficient", 1.0) < stability_config.get("stability_coefficient", 0.8)):
                
                deactivation_triggers.append({
                    "type": DeactivationTriggerType.STABILITY_DRIFT,
                    "severity": DeactivationSeverity.MAJOR if stability_analysis.get("is_degrading", False) else DeactivationSeverity.MINOR,
                    "details": stability_analysis
                })
                
        # Analyze evidence quality degradation
        quality_analysis = self.quality_tracker.detect_quality_degradation(enhancement_id)
        monitoring_results["quality_analysis"] = quality_analysis
        
        if quality_analysis.get("sufficient_data", False):
            if (quality_analysis.get("below_minimum_quality", False) or
                quality_analysis.get("degradation_rate_exceeded", False)):
                
                severity = DeactivationSeverity.CRITICAL if quality_analysis.get("below_minimum_quality", False) else DeactivationSeverity.MAJOR
                deactivation_triggers.append({
                    "type": DeactivationTriggerType.EVIDENCE_QUALITY,
                    "severity": severity,
                    "details": quality_analysis
                })
                
        # Analyze performance regressions
        performance_config = self.thresholds.get("stability_monitoring", {}).get("performance_regression", {})
        regression_analysis = self.performance_detector.detect_regressions(enhancement_id, performance_config)
        monitoring_results["regression_analysis"] = regression_analysis
        
        if regression_analysis.get("sufficient_data", False):
            if regression_analysis.get("critical_regression_detected", False):
                deactivation_triggers.append({
                    "type": DeactivationTriggerType.PERFORMANCE_REGRESSION,
                    "severity": DeactivationSeverity.CRITICAL,
                    "details": regression_analysis
                })
                
        # Process deactivation triggers
        deactivation_decision = self._process_deactivation_triggers(enhancement_id, deactivation_triggers)
        
        return {
            "enhancement_id": enhancement_id,
            "monitoring_results": monitoring_results,
            "deactivation_triggers": len(deactivation_triggers),
            "deactivation_decision": deactivation_decision,
            "trigger_details": deactivation_triggers,
            "timestamp": datetime.now().isoformat()
        }
        
    def _process_deactivation_triggers(
        self, 
        enhancement_id: str, 
        triggers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process deactivation triggers and make deactivation decision"""
        
        if not triggers:
            # Reset consecutive violations if no triggers
            self.consecutive_violations[enhancement_id] = 0
            return {"should_deactivate": False, "reason": "no_triggers"}
            
        # Check if enhancement is in cooldown
        if enhancement_id in self.active_cooldowns:
            if datetime.now() < self.active_cooldowns[enhancement_id]:
                return {
                    "should_deactivate": False, 
                    "reason": "in_cooldown",
                    "cooldown_until": self.active_cooldowns[enhancement_id].isoformat()
                }
            else:
                # Cooldown expired
                del self.active_cooldowns[enhancement_id]
                
        # Analyze trigger severity
        critical_triggers = [t for t in triggers if t["severity"] == DeactivationSeverity.CRITICAL]
        major_triggers = [t for t in triggers if t["severity"] == DeactivationSeverity.MAJOR]
        
        deactivation_config = self.thresholds.get("auto_deactivation", {}).get("triggers", {})
        
        # Check for immediate critical deactivation
        if critical_triggers and deactivation_config.get("safety_violation", {}).get("single_critical_failure", True):
            return self._execute_deactivation(
                enhancement_id, 
                DeactivationTriggerType.SAFETY_VIOLATION,
                DeactivationSeverity.CRITICAL,
                "Critical safety violation detected",
                {"critical_triggers": len(critical_triggers)}
            )
            
        # Track consecutive violations for major triggers
        if major_triggers or critical_triggers:
            self.consecutive_violations[enhancement_id] += 1
        else:
            self.consecutive_violations[enhancement_id] = 0
            
        consecutive_count = self.consecutive_violations[enhancement_id]
        
        # Check consecutive violation thresholds
        stability_threshold = deactivation_config.get("stability_breach", {}).get("consecutive_violations", 3)
        performance_threshold = deactivation_config.get("performance_degradation", {}).get("consecutive_regressions", 2)
        
        performance_triggers = [t for t in triggers if t["type"] == DeactivationTriggerType.PERFORMANCE_REGRESSION]
        stability_triggers = [t for t in triggers if t["type"] == DeactivationTriggerType.STABILITY_DRIFT]
        
        if performance_triggers and consecutive_count >= performance_threshold:
            return self._execute_deactivation(
                enhancement_id,
                DeactivationTriggerType.PERFORMANCE_REGRESSION,
                DeactivationSeverity.MAJOR,
                f"Consecutive performance regressions: {consecutive_count}",
                {"consecutive_violations": consecutive_count, "performance_triggers": len(performance_triggers)}
            )
            
        if stability_triggers and consecutive_count >= stability_threshold:
            return self._execute_deactivation(
                enhancement_id,
                DeactivationTriggerType.STABILITY_DRIFT,
                DeactivationSeverity.MAJOR,
                f"Consecutive stability violations: {consecutive_count}",
                {"consecutive_violations": consecutive_count, "stability_triggers": len(stability_triggers)}
            )
            
        return {
            "should_deactivate": False,
            "reason": "threshold_not_met",
            "consecutive_violations": consecutive_count,
            "trigger_summary": {
                "critical": len(critical_triggers),
                "major": len(major_triggers),
                "total": len(triggers)
            }
        }
        
    def _execute_deactivation(
        self,
        enhancement_id: str,
        trigger_type: DeactivationTriggerType,
        severity: DeactivationSeverity,
        reason: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute deactivation and establish cooldown period"""
        
        # Determine cooldown period
        cooldown_config = self.thresholds.get("auto_deactivation", {}).get("cooldown_periods", {})
        
        if severity == DeactivationSeverity.CRITICAL:
            cooldown_duration = cooldown_config.get("critical_deactivation", "PT24H")
        elif severity == DeactivationSeverity.MAJOR:
            cooldown_duration = cooldown_config.get("major_deactivation", "PT1H")
        else:
            cooldown_duration = cooldown_config.get("minor_deactivation", "PT15M")
            
        # Parse ISO 8601 duration (simplified)
        cooldown_minutes = AutoDeactivationMonitor._parse_duration_to_minutes(cooldown_duration)
        cooldown_until = datetime.now() + timedelta(minutes=cooldown_minutes)
        
        # Record deactivation event
        deactivation_event = DeactivationEvent(
            trigger_type=trigger_type,
            severity=severity,
            enhancement_id=enhancement_id,
            trigger_condition=reason,
            metrics=metrics,
            timestamp=datetime.now(),
            cooldown_until=cooldown_until,
            metadata={"consecutive_violations": self.consecutive_violations[enhancement_id]}
        )
        
        self.deactivation_events.append(deactivation_event)
        self.active_cooldowns[enhancement_id] = cooldown_until
        
        # Reset consecutive violations
        self.consecutive_violations[enhancement_id] = 0
        
        self.logger.warning(f"Enhancement {enhancement_id} deactivated: {reason}")
        
        return {
            "should_deactivate": True,
            "reason": reason,
            "trigger_type": trigger_type.value,
            "severity": severity.value,
            "cooldown_until": cooldown_until.isoformat(),
            "cooldown_duration_minutes": cooldown_minutes,
            "deactivation_event_id": len(self.deactivation_events) - 1
        }
        
    @staticmethod
    def _parse_duration_to_minutes(duration: str) -> int:
        """Parse ISO 8601 duration to minutes (simplified parser)"""
        # Simple parser for PT format (PT15M, PT1H, PT24H)
        duration = duration.upper()
        if not duration.startswith("PT"):
            return 60  # Default to 1 hour
            
        time_part = duration[2:]  # Remove "PT"
        
        if "H" in time_part:
            hours = int(time_part.split("H")[0])
            return hours * 60
        elif "M" in time_part:
            return int(time_part.split("M")[0])
        else:
            return 60  # Default to 1 hour
            
    def get_enhancement_status(self, enhancement_id: str) -> Dict[str, Any]:
        """Get current status of enhancement monitoring"""
        is_in_cooldown = enhancement_id in self.active_cooldowns
        cooldown_until = self.active_cooldowns.get(enhancement_id)
        
        recent_events = [
            event for event in self.deactivation_events[-10:]  # Last 10 events
            if event.enhancement_id == enhancement_id
        ]
        
        return {
            "enhancement_id": enhancement_id,
            "is_active": not is_in_cooldown,
            "in_cooldown": is_in_cooldown,
            "cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
            "consecutive_violations": self.consecutive_violations.get(enhancement_id, 0),
            "recent_deactivation_events": len(recent_events),
            "last_deactivation": recent_events[-1].timestamp.isoformat() if recent_events else None
        }
        
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        active_enhancements = set(self.stability_analyzer.score_windows.keys())
        
        summary = {
            "total_monitored_enhancements": len(active_enhancements),
            "active_cooldowns": len(self.active_cooldowns),
            "total_deactivation_events": len(self.deactivation_events),
            "recent_deactivations": len([e for e in self.deactivation_events if e.timestamp > datetime.now() - timedelta(hours=24)]),
            "enhancement_statuses": {
                enhancement_id: self.get_enhancement_status(enhancement_id)
                for enhancement_id in active_enhancements
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return summary