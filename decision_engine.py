"""
Metrics-driven decision engine with exponential moving averages and threshold-based triggers
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Optional heavy dependencies (guarded)
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    # Minimal fallback for required functions
    class _NP:
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0.0
        @staticmethod
        def std(x):
            m = _NP.mean(x)
            return (sum((v - m) * (v - m) for v in x) / max(1, len(x))) ** 0.5
    np = _NP()  # type: ignore
try:
    from scipy import stats  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    class _Stats:
        @staticmethod
        def zscore(data):
            m = np.mean(data)
            s = np.std(data)
            s = s if s > 0 else 1.0
            return [(v - m) / s for v in data]
    stats = _Stats()  # type: ignore


class ScalingAction(Enum):
    """Types of scaling actions"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE_DOWN = "emergency_scale_down"


class MetricTrend(Enum):
    """Trend directions for metrics"""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class ScalingDecision:
    """Represents a scaling decision with rationale"""

    action: ScalingAction
    target_workers: int
    target_window_size: int
    target_frequency: float  # Processing frequency adjustment
    confidence: float  # 0-1 confidence in decision
    reasons: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "target_workers": self.target_workers,
            "target_window_size": self.target_window_size,
            "target_frequency": self.target_frequency,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "metrics_snapshot": self.metrics_snapshot,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ThresholdConfiguration:
    """Configuration for metric thresholds and bounds"""

    # Resource utilization thresholds
    cpu_high_threshold: float = 0.8
    cpu_low_threshold: float = 0.3
    memory_high_threshold: float = 0.85
    memory_low_threshold: float = 0.4

    # Performance thresholds
    latency_p95_threshold: float = 5000  # milliseconds
    latency_p99_threshold: float = 10000
    throughput_min_threshold: float = 0.5  # docs/second
    error_rate_threshold: float = 5.0  # percentage

    # Scaling bounds
    min_workers: int = 1
    max_workers: int = 50
    min_window_size: int = 1
    max_window_size: int = 100
    min_frequency: float = 0.1  # Hz
    max_frequency: float = 10.0

    # Decision thresholds
    confidence_threshold: float = 0.7
    trend_detection_window: int = 10  # number of samples

    def validate(self) -> bool:
        """Validate threshold configuration"""
        return (
            0 < self.cpu_low_threshold < self.cpu_high_threshold < 1.0
            and 0 < self.memory_low_threshold < self.memory_high_threshold < 1.0
            and self.min_workers <= self.max_workers
            and self.min_window_size <= self.max_window_size
            and self.min_frequency <= self.max_frequency
            and 0 <= self.confidence_threshold <= 1.0
        )


class ExponentialMovingAverage:
    """Exponential moving average calculator with trend detection"""

    def __init__(self, alpha: float = 0.3, min_samples: int = 3):
        """
        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
            min_samples: Minimum samples before calculations are reliable
        """
        self.alpha = alpha
        self.min_samples = min_samples
        self.values: List[float] = []
        self.timestamps: List[datetime] = []
        self.current_ema: Optional[float] = None

    def update(self, value: float, timestamp: Optional[datetime] = None) -> float:
        """Update EMA with new value"""
        if timestamp is None:
            timestamp = datetime.now()

        self.values.append(value)
        self.timestamps.append(timestamp)

        # Keep only recent values (last 100 samples)
        if len(self.values) > 100:
            self.values = self.values[-100:]
            self.timestamps = self.timestamps[-100:]

        if self.current_ema is None:
            self.current_ema = value
        else:
            self.current_ema = self.alpha * value + (1 - self.alpha) * self.current_ema

        return self.current_ema

    def get_value(self) -> Optional[float]:
        """Get current EMA value"""
        return self.current_ema

    def get_trend(self, window_size: int = 5) -> MetricTrend:
        """Detect trend over recent window"""
        if len(self.values) < max(window_size, self.min_samples):
            return MetricTrend.STABLE

        recent_values = self.values[-window_size:]

        # Calculate linear regression slope
        x = np.arange(len(recent_values))
        slope, _, r_value, _, _ = stats.linregress(x, recent_values)

        # Determine trend based on slope and correlation
        if abs(r_value) < 0.5:  # Low correlation suggests volatility
            return MetricTrend.VOLATILE
        elif slope > 0.1:
            return MetricTrend.INCREASING
        elif slope < -0.1:
            return MetricTrend.DECREASING
        else:
            return MetricTrend.STABLE

    def get_variance(self, window_size: int = 10) -> float:
        """Get variance over recent window"""
        if len(self.values) < window_size:
            return 0.0
        recent_values = self.values[-window_size:]
        return np.var(recent_values)

    def is_stable(self) -> bool:
        """Check if metric has sufficient samples and is stable"""
        return len(self.values) >= self.min_samples and self.get_trend() in [
            MetricTrend.STABLE,
            MetricTrend.DECREASING,
        ]


class DecisionEngine:
    """
    Metrics-driven decision engine that analyzes performance patterns
    and generates scaling decisions with confidence ratings.
    """

    def __init__(
        self,
        thresholds: Optional[ThresholdConfiguration] = None,
        ema_alpha: float = 0.3,
    ):
        self.logger = logging.getLogger(__name__)
        self.thresholds = thresholds or ThresholdConfiguration()

        if not self.thresholds.validate():
            raise ValueError("Invalid threshold configuration")

        # Exponential moving averages for key metrics
        self.emas = {
            "cpu_usage": ExponentialMovingAverage(ema_alpha),
            "memory_usage": ExponentialMovingAverage(ema_alpha),
            "throughput": ExponentialMovingAverage(ema_alpha),
            "latency_p95": ExponentialMovingAverage(ema_alpha),
            "latency_p99": ExponentialMovingAverage(ema_alpha),
            "error_rate": ExponentialMovingAverage(ema_alpha),
            "queue_depth": ExponentialMovingAverage(ema_alpha),
            "workload_complexity": ExponentialMovingAverage(ema_alpha),
        }

        # Current system state
        self.current_workers = 1
        self.current_window_size = 10
        self.current_frequency = 1.0

        # Decision history for learning
        self.decision_history: List[ScalingDecision] = []
        self.max_history = 100

        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update EMAs with new metrics data"""
        timestamp = datetime.now()

        # Update system metrics
        if "system" in metrics:
            system = metrics["system"]
            self.emas["cpu_usage"].update(system.get("cpu_usage", 0), timestamp)
            self.emas["memory_usage"].update(system.get("memory_usage", 0), timestamp)

        # Update processing metrics
        if "processing" in metrics:
            processing = metrics["processing"]
            self.emas["throughput"].update(processing.get("throughput", 0), timestamp)
            self.emas["latency_p95"].update(processing.get("latency_p95", 0), timestamp)
            self.emas["latency_p99"].update(processing.get("latency_p99", 0), timestamp)
            self.emas["error_rate"].update(processing.get("error_rate", 0), timestamp)
            self.emas["queue_depth"].update(processing.get("queue_depth", 0), timestamp)

        # Update workload metrics
        if "workload" in metrics:
            workload = metrics["workload"]
            complexity = (
                workload.get("document_complexity", 0.5)
                + workload.get("ocr_ratio", 0.3)
                + workload.get("table_density", 0.15)
            ) / 3
            self.emas["workload_complexity"].update(complexity, timestamp)

    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns and trends"""
        analysis = {
            "trends": {},
            "stability": {},
            "alerts": [],
            "workload_drift": False,
        }

        # Analyze trends for each metric
        for metric_name, ema in self.emas.items():
            trend = ema.get_trend()
            stability = ema.is_stable()
            current_value = ema.get_value()
            variance = ema.get_variance()

            analysis["trends"][metric_name] = {
                "trend": trend.value,
                "current_value": current_value,
                "variance": variance,
                "stable": stability,
            }

            # Generate alerts for concerning patterns
            if (
                metric_name == "cpu_usage"
                and current_value
                and current_value > self.thresholds.cpu_high_threshold
            ):
                analysis["alerts"].append(f"High CPU usage: {current_value:.1%}")
            elif (
                metric_name == "memory_usage"
                and current_value
                and current_value > self.thresholds.memory_high_threshold
            ):
                analysis["alerts"].append(f"High memory usage: {current_value:.1%}")
            elif (
                metric_name == "error_rate"
                and current_value
                and current_value > self.thresholds.error_rate_threshold
            ):
                analysis["alerts"].append(f"High error rate: {current_value:.1%}")
            elif (
                metric_name == "latency_p95"
                and current_value
                and current_value > self.thresholds.latency_p95_threshold
            ):
                analysis["alerts"].append(f"High P95 latency: {current_value:.0f}ms")

        # Detect workload drift
        workload_complexity = self.emas["workload_complexity"]
        if workload_complexity.get_trend() == MetricTrend.VOLATILE:
            analysis["workload_drift"] = True
            analysis["alerts"].append("Workload complexity drift detected")

        return analysis

    def generate_scaling_decision(self) -> ScalingDecision:
        """Generate scaling decision based on current metrics and patterns"""
        analysis = self.analyze_performance_patterns()
        reasons = []
        confidence = 0.5  # Base confidence

        # Get current metric values
        cpu = self.emas["cpu_usage"].get_value() or 0
        memory = self.emas["memory_usage"].get_value() or 0
        throughput = self.emas["throughput"].get_value() or 0
        latency_p95 = self.emas["latency_p95"].get_value() or 0
        error_rate = self.emas["error_rate"].get_value() or 0
        queue_depth = self.emas["queue_depth"].get_value() or 0

        # Decision logic with multiple factors
        scale_up_score = 0
        scale_down_score = 0

        # Resource pressure indicators
        if cpu > self.thresholds.cpu_high_threshold:
            scale_up_score += 2
            reasons.append(f"High CPU usage ({cpu:.1%})")
            confidence += 0.1
        elif cpu < self.thresholds.cpu_low_threshold:
            scale_down_score += 1
            reasons.append(f"Low CPU usage ({cpu:.1%})")

        if memory > self.thresholds.memory_high_threshold:
            scale_up_score += 2
            reasons.append(f"High memory usage ({memory:.1%})")
            confidence += 0.1
        elif memory < self.thresholds.memory_low_threshold:
            scale_down_score += 1
            reasons.append(f"Low memory usage ({memory:.1%})")

        # Performance indicators
        if latency_p95 > self.thresholds.latency_p95_threshold:
            scale_up_score += 3
            reasons.append(f"High P95 latency ({latency_p95:.0f}ms)")
            confidence += 0.2

        if throughput < self.thresholds.throughput_min_threshold:
            scale_up_score += 2
            reasons.append(f"Low throughput ({throughput:.2f} docs/sec)")
            confidence += 0.1

        if error_rate > self.thresholds.error_rate_threshold:
            scale_up_score += (
                1  # Might need more resources, but could also be other issues
            )
            reasons.append(f"High error rate ({error_rate:.1%})")

        if queue_depth > 10:  # Arbitrary threshold for queue backlog
            scale_up_score += 2
            reasons.append(f"Queue backlog ({queue_depth:.0f})")
            confidence += 0.1

        # Trend-based adjustments
        cpu_trend = analysis["trends"]["cpu_usage"]["trend"]
        throughput_trend = analysis["trends"]["throughput"]["trend"]

        if cpu_trend == "increasing":
            scale_up_score += 1
            reasons.append("CPU usage trending up")
        elif throughput_trend == "decreasing":
            scale_up_score += 1
            reasons.append("Throughput trending down")

        # Workload drift considerations
        if analysis["workload_drift"]:
            scale_up_score += 1
            reasons.append("Workload complexity drift detected")
            confidence -= 0.1  # Less confident during drift

        # Make decision
        if scale_up_score >= 4 or (scale_up_score >= 2 and scale_down_score == 0):
            action = ScalingAction.SCALE_UP
            target_workers = min(
                self.current_workers + max(1, scale_up_score // 2),
                self.thresholds.max_workers,
            )
            target_window_size = min(
                self.current_window_size + 5, self.thresholds.max_window_size
            )
            target_frequency = min(
                self.current_frequency * 1.2, self.thresholds.max_frequency
            )
        elif (
            scale_down_score >= 2
            and scale_up_score == 0
            and len(analysis["alerts"]) == 0
        ):
            action = ScalingAction.SCALE_DOWN
            target_workers = max(self.current_workers - 1, self.thresholds.min_workers)
            target_window_size = max(
                self.current_window_size - 2, self.thresholds.min_window_size
            )
            target_frequency = max(
                self.current_frequency * 0.9, self.thresholds.min_frequency
            )
        elif cpu > 0.95 or memory > 0.95:  # Emergency conditions
            action = ScalingAction.EMERGENCY_SCALE_DOWN
            target_workers = max(self.current_workers // 2, 1)
            target_window_size = max(self.current_window_size // 2, 1)
            target_frequency = self.current_frequency * 0.5
            reasons.append("Emergency resource exhaustion")
            confidence = 0.9
        else:
            action = ScalingAction.MAINTAIN
            target_workers = self.current_workers
            target_window_size = self.current_window_size
            target_frequency = self.current_frequency

        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))

        # Create decision
        decision = ScalingDecision(
            action=action,
            target_workers=target_workers,
            target_window_size=target_window_size,
            target_frequency=target_frequency,
            confidence=confidence,
            reasons=reasons,
            metrics_snapshot={
                "cpu_usage": cpu,
                "memory_usage": memory,
                "throughput": throughput,
                "latency_p95": latency_p95,
                "error_rate": error_rate,
                "queue_depth": queue_depth,
                "analysis": analysis,
            },
        )

        # Store decision in history
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history :]

        self.logger.info(
            f"Generated scaling decision: {decision.action.value} "
            f"(workers: {target_workers}, confidence: {confidence:.2f})"
        )

        return decision

    def update_system_state(self, workers: int, window_size: int, frequency: float):
        """Update current system state after scaling action"""
        self.current_workers = workers
        self.current_window_size = window_size
        self.current_frequency = frequency

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        recent_decisions = self.decision_history[-limit:]
        return [decision.to_dict() for decision in recent_decisions]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with key metrics"""
        summary = {
            "current_state": {
                "workers": self.current_workers,
                "window_size": self.current_window_size,
                "frequency": self.current_frequency,
            },
            "current_metrics": {},
            "stability_status": {},
            "recent_decisions": len(
                [
                    d
                    for d in self.decision_history[-10:]
                    if d.action != ScalingAction.MAINTAIN
                ]
            ),
        }

        # Current metric values
        for metric_name, ema in self.emas.items():
            current_value = ema.get_value()
            if current_value is not None:
                summary["current_metrics"][metric_name] = {
                    "value": current_value,
                    "trend": ema.get_trend().value,
                    "stable": ema.is_stable(),
                }

        return summary

    def reset_state(self):
        """Reset decision engine state (useful for testing)"""
        for ema in self.emas.values():
            ema.values.clear()
            ema.timestamps.clear()
            ema.current_ema = None

        self.decision_history.clear()
        self.current_workers = 1
        self.current_window_size = 10
        self.current_frequency = 1.0

if __name__ == "__main__":
    # Minimal demo to guarantee execution without numpy/scipy
    logging.basicConfig(level=logging.INFO)
    engine = DecisionEngine()
    sample_metrics = {
        "cpu_usage": 0.42,
        "memory_usage": 0.55,
        "latency_ms": 1200,
        "throughput": 2.5,
        "error_rate": 0.5,
    }
    engine.update_metrics(sample_metrics)
    decision = engine.generate_scaling_decision()
    print(json.dumps({
        "decision": decision.to_dict(),
        "summary": engine.get_performance_summary()
    }))
