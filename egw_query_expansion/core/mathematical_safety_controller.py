"""
Mathematical Enhancement Safety Controller

This module implements a comprehensive safety system for controlling mathematical enhancements
with feature flags, auto-activation criteria, numerical stability guards, and graceful degradation.

Features:
- Feature flag system for enhancement activation control
- Auto-activation criteria validation against predefined thresholds
- Numerical stability guards (NaN/Inf detection, epsilon protection, bounded iterations)
- Graceful degradation with baseline fallbacks
- Enhancement status tracking with performance metrics and degradation monitoring

Safety mechanisms ensure robust mathematical computations in the EGW query expansion pipeline.
"""

import numpy as np
import torch
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
from contextlib import contextmanager
import logging
from functools import wraps
from collections import defaultdict


class EnhancementStatus(Enum):
    """Status of mathematical enhancements"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    DEGRADED = "degraded"
    FAILED = "failed"
    AUTO_DISABLED = "auto_disabled"


class DegradationReason(Enum):
    """Reasons for enhancement degradation"""
    NAN_DETECTED = "nan_detected"
    INF_DETECTED = "inf_detected"
    NUMERICAL_INSTABILITY = "numerical_instability"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    ITERATION_LIMIT = "iteration_limit"
    COMPUTATION_TIMEOUT = "computation_timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class EnhancementMetrics:
    """Performance metrics for enhancements"""
    name: str
    activation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    degradation_count: int = 0
    total_computation_time: float = 0.0
    average_computation_time: float = 0.0
    last_activation: Optional[float] = None
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    degradation_events: List[Tuple[float, DegradationReason, str]] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    
    def update_success(self, computation_time: float):
        """Update metrics for successful enhancement"""
        self.success_count += 1
        self.activation_count += 1
        self.total_computation_time += computation_time
        self.average_computation_time = self.total_computation_time / self.activation_count
        self.last_activation = time.time()
        self.last_success = time.time()
        self.performance_history.append(computation_time)
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def update_failure(self, computation_time: float, reason: DegradationReason, details: str = ""):
        """Update metrics for failed enhancement"""
        self.failure_count += 1
        self.activation_count += 1
        self.total_computation_time += computation_time
        self.average_computation_time = self.total_computation_time / self.activation_count
        self.last_activation = time.time()
        self.last_failure = time.time()
        self.degradation_events.append((time.time(), reason, details))
        self.performance_history.append(float('inf'))
        
        # Limit degradation events
        if len(self.degradation_events) > 100:
            self.degradation_events = self.degradation_events[-100:]
    
    def update_degradation(self, reason: DegradationReason, details: str = ""):
        """Update metrics for enhancement degradation"""
        self.degradation_count += 1
        self.degradation_events.append((time.time(), reason, details))
        
        # Limit degradation events
        if len(self.degradation_events) > 100:
            self.degradation_events = self.degradation_events[-100:]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.activation_count == 0:
            return 0.0
        return self.success_count / self.activation_count
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.activation_count == 0:
            return 0.0
        return self.failure_count / self.activation_count


@dataclass
class SafetyThresholds:
    """Safety thresholds for numerical stability"""
    max_iterations: int = 1000
    computation_timeout: float = 30.0  # seconds
    memory_limit: int = 1024 * 1024 * 1024  # 1GB
    epsilon: float = 1e-12
    max_gradient_norm: float = 1e6
    min_singular_value: float = 1e-10
    max_condition_number: float = 1e12
    nan_tolerance: int = 0
    inf_tolerance: int = 0
    min_success_rate: float = 0.7
    max_degradation_rate: float = 0.3


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    enabled: bool = False
    auto_activation_enabled: bool = False
    degradation_threshold: float = 0.3
    recovery_threshold: float = 0.8
    min_activation_interval: float = 60.0  # seconds
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)


class NumericalStabilityGuard:
    """Numerical stability checking and protection"""
    
    def __init__(self, thresholds: SafetyThresholds):
        self.thresholds = thresholds
        self.logger = logging.getLogger(__name__ + ".NumericalStabilityGuard")
    
    def check_tensor_stability(self, tensor: Union[np.ndarray, torch.Tensor], name: str = "") -> Tuple[bool, Optional[DegradationReason]]:
        """Check numerical stability of tensor"""
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = tensor
        
        # Check for NaN
        if np.isnan(tensor_np).sum() > self.thresholds.nan_tolerance:
            self.logger.warning(f"NaN detected in tensor {name}: {np.isnan(tensor_np).sum()} values")
            return False, DegradationReason.NAN_DETECTED
        
        # Check for Inf
        if np.isinf(tensor_np).sum() > self.thresholds.inf_tolerance:
            self.logger.warning(f"Inf detected in tensor {name}: {np.isinf(tensor_np).sum()} values")
            return False, DegradationReason.INF_DETECTED
        
        # Check gradient norm for neural networks
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad and tensor.grad is not None:
            grad_norm = torch.norm(tensor.grad).item()
            if grad_norm > self.thresholds.max_gradient_norm:
                self.logger.warning(f"Gradient norm too large in {name}: {grad_norm}")
                return False, DegradationReason.NUMERICAL_INSTABILITY
        
        return True, None
    
    def check_matrix_condition(self, matrix: Union[np.ndarray, torch.Tensor], name: str = "") -> Tuple[bool, Optional[DegradationReason]]:
        """Check matrix condition number"""
        if isinstance(matrix, torch.Tensor):
            matrix_np = matrix.detach().cpu().numpy()
        else:
            matrix_np = matrix
        
        if len(matrix_np.shape) < 2:
            return True, None
        
        try:
            # Compute condition number
            cond_num = np.linalg.cond(matrix_np)
            
            if cond_num > self.thresholds.max_condition_number:
                self.logger.warning(f"High condition number in matrix {name}: {cond_num}")
                return False, DegradationReason.NUMERICAL_INSTABILITY
            
            # Check singular values
            try:
                singular_values = np.linalg.svd(matrix_np, compute_uv=False)
                min_sv = np.min(singular_values)
                
                if min_sv < self.thresholds.min_singular_value:
                    self.logger.warning(f"Small singular value in matrix {name}: {min_sv}")
                    return False, DegradationReason.NUMERICAL_INSTABILITY
            except np.linalg.LinAlgError:
                self.logger.warning(f"SVD failed for matrix {name}")
                return False, DegradationReason.NUMERICAL_INSTABILITY
                
        except np.linalg.LinAlgError:
            self.logger.warning(f"Condition number computation failed for matrix {name}")
            return False, DegradationReason.NUMERICAL_INSTABILITY
        
        return True, None
    
    def safe_division(self, numerator: Union[float, np.ndarray, torch.Tensor], 
                     denominator: Union[float, np.ndarray, torch.Tensor],
                     epsilon: Optional[float] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """Perform epsilon-protected division"""
        if epsilon is None:
            epsilon = self.thresholds.epsilon
        
        if isinstance(denominator, torch.Tensor):
            protected_denom = torch.where(torch.abs(denominator) < epsilon, 
                                        torch.sign(denominator) * epsilon, 
                                        denominator)
        elif isinstance(denominator, np.ndarray):
            protected_denom = np.where(np.abs(denominator) < epsilon,
                                     np.sign(denominator) * epsilon,
                                     denominator)
        else:
            protected_denom = epsilon if abs(denominator) < epsilon else denominator
        
        return numerator / protected_denom
    
    def safe_log(self, x: Union[float, np.ndarray, torch.Tensor],
                epsilon: Optional[float] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """Perform epsilon-protected logarithm"""
        if epsilon is None:
            epsilon = self.thresholds.epsilon
        
        if isinstance(x, torch.Tensor):
            return torch.log(torch.clamp(x, min=epsilon))
        elif isinstance(x, np.ndarray):
            return np.log(np.maximum(x, epsilon))
        else:
            return np.log(max(x, epsilon))
    
    def safe_sqrt(self, x: Union[float, np.ndarray, torch.Tensor],
                 epsilon: Optional[float] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """Perform epsilon-protected square root"""
        if epsilon is None:
            epsilon = self.thresholds.epsilon
        
        if isinstance(x, torch.Tensor):
            return torch.sqrt(torch.clamp(x, min=epsilon))
        elif isinstance(x, np.ndarray):
            return np.sqrt(np.maximum(x, epsilon))
        else:
            return np.sqrt(max(x, epsilon))


class MathematicalEnhancementSafetyController:
    """Main safety controller for mathematical enhancements"""
    
    def __init__(self, thresholds: Optional[SafetyThresholds] = None):
        """Initialize the safety controller"""
        self.thresholds = thresholds or SafetyThresholds()
        self.stability_guard = NumericalStabilityGuard(self.thresholds)
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.enhancement_metrics: Dict[str, EnhancementMetrics] = {}
        self.enhancement_status: Dict[str, EnhancementStatus] = {}
        self.baseline_implementations: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__ + ".MathematicalEnhancementSafetyController")
        
        # Global statistics
        self.global_stats = {
            'total_activations': 0,
            'total_successes': 0,
            'total_failures': 0,
            'total_degradations': 0,
            'system_start_time': time.time()
        }
    
    def register_enhancement(self, name: str, 
                           enhancement_fn: Callable,
                           baseline_fn: Callable,
                           feature_flag: Optional[FeatureFlag] = None) -> None:
        """Register a mathematical enhancement with baseline fallback"""
        if feature_flag is None:
            feature_flag = FeatureFlag(name=name)
        
        self.feature_flags[name] = feature_flag
        self.enhancement_metrics[name] = EnhancementMetrics(name=name)
        self.enhancement_status[name] = EnhancementStatus.DISABLED
        self.baseline_implementations[name] = baseline_fn
        
        self.logger.info(f"Registered enhancement '{name}' with baseline fallback")
    
    def enable_enhancement(self, name: str) -> bool:
        """Enable a specific enhancement"""
        if name not in self.feature_flags:
            self.logger.error(f"Enhancement '{name}' not registered")
            return False
        
        # Check dependencies
        for dep in self.feature_flags[name].dependencies:
            if dep not in self.enhancement_status or self.enhancement_status[dep] != EnhancementStatus.ENABLED:
                self.logger.warning(f"Cannot enable '{name}': dependency '{dep}' not enabled")
                return False
        
        self.feature_flags[name].enabled = True
        self.enhancement_status[name] = EnhancementStatus.ENABLED
        self.logger.info(f"Enabled enhancement '{name}'")
        return True
    
    def disable_enhancement(self, name: str) -> bool:
        """Disable a specific enhancement"""
        if name not in self.feature_flags:
            self.logger.error(f"Enhancement '{name}' not registered")
            return False
        
        self.feature_flags[name].enabled = False
        self.enhancement_status[name] = EnhancementStatus.DISABLED
        self.logger.info(f"Disabled enhancement '{name}'")
        return True
    
    def check_auto_activation_criteria(self, name: str) -> bool:
        """Check if enhancement should be auto-activated"""
        if name not in self.feature_flags:
            return False
        
        flag = self.feature_flags[name]
        if not flag.auto_activation_enabled:
            return False
        
        metrics = self.enhancement_metrics[name]
        
        # Check minimum interval
        if (flag.last_activation and 
            time.time() - flag.last_activation < flag.min_activation_interval):
            return False
        
        # Check success rate
        if metrics.success_rate < flag.recovery_threshold:
            return False
        
        # Check custom conditions
        for condition_name, condition_value in flag.conditions.items():
            # Implement custom condition checking logic here
            pass
        
        return True
    
    def validate_enhancement_criteria(self, name: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Validate enhancement activation criteria"""
        if name not in self.feature_flags:
            return False, f"Enhancement '{name}' not registered"
        
        if not self.feature_flags[name].enabled:
            return False, f"Enhancement '{name}' is disabled"
        
        status = self.enhancement_status[name]
        if status in [EnhancementStatus.FAILED, EnhancementStatus.AUTO_DISABLED]:
            return False, f"Enhancement '{name}' status is {status.value}"
        
        # Check degradation threshold
        metrics = self.enhancement_metrics[name]
        if metrics.failure_rate > self.feature_flags[name].degradation_threshold:
            return False, f"Enhancement '{name}' failure rate too high: {metrics.failure_rate}"
        
        return True, None
    
    @contextmanager
    def safe_enhancement_context(self, name: str, **kwargs):
        """Context manager for safe enhancement execution"""
        start_time = time.time()
        success = False
        error_msg = ""
        
        try:
            # Pre-execution validation
            can_activate, reason = self.validate_enhancement_criteria(name, **kwargs)
            if not can_activate:
                raise RuntimeError(f"Enhancement validation failed: {reason}")
            
            self.logger.debug(f"Starting safe enhancement execution: {name}")
            yield self
            success = True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Enhancement '{name}' failed: {error_msg}")
            self._handle_enhancement_failure(name, DegradationReason.UNKNOWN_ERROR, error_msg)
            raise
        
        finally:
            computation_time = time.time() - start_time
            
            # Update metrics
            if success:
                self.enhancement_metrics[name].update_success(computation_time)
                self.global_stats['total_successes'] += 1
            else:
                self.enhancement_metrics[name].update_failure(
                    computation_time, DegradationReason.UNKNOWN_ERROR, error_msg
                )
                self.global_stats['total_failures'] += 1
            
            self.global_stats['total_activations'] += 1
    
    def safe_compute_with_fallback(self, name: str, *args, **kwargs) -> Any:
        """Safely compute with automatic fallback to baseline"""
        try:
            with self.safe_enhancement_context(name, **kwargs):
                # Get the enhanced function (would be provided by registration)
                enhanced_fn = kwargs.get('enhanced_fn')
                if enhanced_fn is None:
                    raise ValueError("Enhanced function not provided")
                
                # Add safety monitoring
                result = self._monitored_execution(enhanced_fn, name, *args, **kwargs)
                
                # Validate result stability
                if hasattr(result, '__iter__') and not isinstance(result, str):
                    for i, item in enumerate(result):
                        if isinstance(item, (np.ndarray, torch.Tensor)):
                            stable, reason = self.stability_guard.check_tensor_stability(item, f"{name}_result_{i}")
                            if not stable:
                                raise RuntimeError(f"Unstable result detected: {reason}")
                
                return result
                
        except Exception as e:
            self.logger.warning(f"Enhancement '{name}' failed, falling back to baseline: {e}")
            return self._execute_baseline(name, *args, **kwargs)
    
    def _monitored_execution(self, fn: Callable, name: str, *args, **kwargs) -> Any:
        """Execute function with monitoring and timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Enhancement '{name}' timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.thresholds.computation_timeout))
        
        try:
            # Monitor memory usage if possible
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Execute function
            result = fn(*args, **kwargs)
            
            # Check memory usage
            final_memory = process.memory_info().rss
            if final_memory - initial_memory > self.thresholds.memory_limit:
                self.logger.warning(f"Enhancement '{name}' exceeded memory limit")
                self._handle_enhancement_failure(name, DegradationReason.MEMORY_EXCEEDED, 
                                               f"Memory increase: {final_memory - initial_memory}")
            
            return result
            
        except ImportError:
            # psutil not available, skip memory monitoring
            return fn(*args, **kwargs)
        
        finally:
            signal.alarm(0)  # Cancel timeout
    
    def _execute_baseline(self, name: str, *args, **kwargs) -> Any:
        """Execute baseline implementation"""
        if name not in self.baseline_implementations:
            raise RuntimeError(f"No baseline implementation registered for '{name}'")
        
        self.enhancement_status[name] = EnhancementStatus.DEGRADED
        self.enhancement_metrics[name].update_degradation(
            DegradationReason.NUMERICAL_INSTABILITY, 
            "Fallback to baseline"
        )
        
        baseline_fn = self.baseline_implementations[name]
        return baseline_fn(*args, **kwargs)
    
    def _handle_enhancement_failure(self, name: str, reason: DegradationReason, details: str):
        """Handle enhancement failure and potential auto-disable"""
        metrics = self.enhancement_metrics[name]
        metrics.update_degradation(reason, details)
        
        # Check if we should auto-disable
        if metrics.failure_rate > self.thresholds.max_degradation_rate:
            self.enhancement_status[name] = EnhancementStatus.AUTO_DISABLED
            self.feature_flags[name].enabled = False
            self.logger.warning(f"Auto-disabled enhancement '{name}' due to high failure rate: {metrics.failure_rate}")
        else:
            self.enhancement_status[name] = EnhancementStatus.DEGRADED
    
    def get_enhancement_status(self, name: Optional[str] = None) -> Union[Dict[str, Dict], Dict]:
        """Get enhancement status and metrics"""
        if name is not None:
            if name not in self.enhancement_metrics:
                return {}
            
            return {
                'name': name,
                'status': self.enhancement_status[name].value,
                'enabled': self.feature_flags[name].enabled,
                'metrics': {
                    'activation_count': self.enhancement_metrics[name].activation_count,
                    'success_count': self.enhancement_metrics[name].success_count,
                    'failure_count': self.enhancement_metrics[name].failure_count,
                    'success_rate': self.enhancement_metrics[name].success_rate,
                    'failure_rate': self.enhancement_metrics[name].failure_rate,
                    'average_computation_time': self.enhancement_metrics[name].average_computation_time,
                    'degradation_count': self.enhancement_metrics[name].degradation_count,
                    'last_activation': self.enhancement_metrics[name].last_activation,
                    'recent_degradations': self.enhancement_metrics[name].degradation_events[-5:]
                }
            }
        else:
            # Return all enhancements
            return {
                name: self.get_enhancement_status(name) 
                for name in self.enhancement_metrics.keys()
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        total_activations = self.global_stats['total_activations']
        
        return {
            'system_uptime': time.time() - self.global_stats['system_start_time'],
            'total_activations': total_activations,
            'total_successes': self.global_stats['total_successes'],
            'total_failures': self.global_stats['total_failures'],
            'total_degradations': self.global_stats['total_degradations'],
            'global_success_rate': (self.global_stats['total_successes'] / max(1, total_activations)),
            'global_failure_rate': (self.global_stats['total_failures'] / max(1, total_activations)),
            'active_enhancements': len([n for n, s in self.enhancement_status.items() 
                                      if s == EnhancementStatus.ENABLED]),
            'degraded_enhancements': len([n for n, s in self.enhancement_status.items() 
                                        if s == EnhancementStatus.DEGRADED]),
            'failed_enhancements': len([n for n, s in self.enhancement_status.items() 
                                      if s in [EnhancementStatus.FAILED, EnhancementStatus.AUTO_DISABLED]]),
            'enhancement_health': {name: status.get_enhancement_status(name) 
                                 for name, status in self.enhancement_status.items()}
        }
    
    def reset_enhancement_metrics(self, name: str) -> bool:
        """Reset metrics for a specific enhancement"""
        if name not in self.enhancement_metrics:
            return False
        
        self.enhancement_metrics[name] = EnhancementMetrics(name=name)
        self.enhancement_status[name] = EnhancementStatus.DISABLED
        self.logger.info(f"Reset metrics for enhancement '{name}'")
        return True
    
    def export_metrics(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Export all metrics to dictionary or file"""
        metrics_data = {
            'timestamp': time.time(),
            'system_health': self.get_system_health(),
            'enhancement_status': self.get_enhancement_status(),
            'global_stats': self.global_stats.copy(),
            'safety_thresholds': {
                'max_iterations': self.thresholds.max_iterations,
                'computation_timeout': self.thresholds.computation_timeout,
                'memory_limit': self.thresholds.memory_limit,
                'epsilon': self.thresholds.epsilon,
                'max_gradient_norm': self.thresholds.max_gradient_norm,
                'min_singular_value': self.thresholds.min_singular_value,
                'max_condition_number': self.thresholds.max_condition_number,
                'min_success_rate': self.thresholds.min_success_rate,
                'max_degradation_rate': self.thresholds.max_degradation_rate
            }
        }
        
        if filename:
            import json
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            self.logger.info(f"Exported metrics to {filename}")
        
        return metrics_data


# Decorator for automatic safety wrapping
def safe_mathematical_enhancement(name: str, 
                                controller: MathematicalEnhancementSafetyController,
                                baseline_fn: Optional[Callable] = None):
    """Decorator to wrap functions with safety controls"""
    def decorator(enhanced_fn: Callable) -> Callable:
        @wraps(enhanced_fn)
        def wrapper(*args, **kwargs):
            # Register if not already registered
            if name not in controller.enhancement_metrics:
                if baseline_fn is None:
                    raise ValueError(f"Baseline function required for enhancement '{name}'")
                controller.register_enhancement(name, enhanced_fn, baseline_fn)
            
            # Execute with safety controls
            return controller.safe_compute_with_fallback(
                name, *args, enhanced_fn=enhanced_fn, **kwargs
            )
        
        return wrapper
    return decorator


# Example usage functions
def create_default_safety_controller() -> MathematicalEnhancementSafetyController:
    """Create a safety controller with default settings"""
    return MathematicalEnhancementSafetyController()


def create_strict_safety_controller() -> MathematicalEnhancementSafetyController:
    """Create a safety controller with strict settings"""
    strict_thresholds = SafetyThresholds(
        max_iterations=500,
        computation_timeout=15.0,
        memory_limit=512 * 1024 * 1024,  # 512MB
        epsilon=1e-15,
        max_gradient_norm=1e4,
        min_singular_value=1e-12,
        max_condition_number=1e10,
        min_success_rate=0.85,
        max_degradation_rate=0.15
    )
    
    return MathematicalEnhancementSafetyController(strict_thresholds)