"""
Mathematical Enhancement Safety Controller

Provides comprehensive safety management for mathematical enhancements across
the EGW query expansion pipeline with feature flag management, numerical
stability protection, and graceful degradation mechanisms.

Key Features:
- Feature flag management for safe activation/deactivation of enhancements
- Auto-activation based on predefined numerical criteria 
- Numerical stability guards (NaN/Inf detection and handling)
- Epsilon-based floating point comparison protection
- Bounded iteration limits to prevent runaway calculations
- Graceful degradation with automatic enhancement disabling
- Enhancement status tracking in all pipeline artifacts
"""

import numpy as np
import torch
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from contextlib import contextmanager
from enum import Enum
import warnings
from functools import wraps


class EnhancementStatus(Enum):
    """Status of mathematical enhancements"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO_DISABLED = "auto_disabled"
    DEGRADED = "degraded"
    FAILED = "failed"


class StabilityLevel(Enum):
    """Numerical stability levels"""
    STRICT = "strict"      # Highest safety, lowest tolerance
    MODERATE = "moderate"  # Balanced safety and performance  
    RELAXED = "relaxed"    # Performance-focused, higher tolerance


@dataclass
class EnhancementConfig:
    """Configuration for a mathematical enhancement"""
    name: str
    enabled: bool = True
    auto_activation_threshold: float = 0.8
    max_iterations: int = 1000
    numerical_tolerance: float = 1e-8
    stability_level: StabilityLevel = StabilityLevel.MODERATE
    dependencies: List[str] = field(default_factory=list)
    auto_disable_on_failure: bool = True
    failure_count_limit: int = 3


@dataclass
class StabilityMetrics:
    """Metrics for numerical stability monitoring"""
    nan_count: int = 0
    inf_count: int = 0
    underflow_count: int = 0
    overflow_count: int = 0
    convergence_failures: int = 0
    iteration_exceeded: int = 0
    total_operations: int = 0
    last_reset: float = field(default_factory=time.time)


@dataclass
class EnhancementState:
    """State tracking for individual enhancement"""
    status: EnhancementStatus = EnhancementStatus.ENABLED
    config: EnhancementConfig = None
    metrics: StabilityMetrics = field(default_factory=StabilityMetrics)
    failure_count: int = 0
    last_failure: Optional[str] = None
    last_failure_time: Optional[float] = None
    activation_criteria_met: bool = False
    dependent_enhancements: List[str] = field(default_factory=list)


class MathEnhancementSafetyController:
    """
    Central controller for mathematical enhancement safety management
    
    Provides:
    - Feature flag management with dependency tracking
    - Numerical stability monitoring and protection  
    - Auto-activation based on criteria evaluation
    - Graceful degradation and recovery mechanisms
    - Comprehensive logging and status tracking
    """
    
    def __init__(self, 
                 default_stability_level: StabilityLevel = StabilityLevel.MODERATE,
                 enable_auto_degradation: bool = True,
                 metric_collection_interval: float = 60.0):
        """
        Initialize the safety controller
        """
        self.default_stability_level = default_stability_level
        self.enable_auto_degradation = enable_auto_degradation
        self.metric_collection_interval = metric_collection_interval
        
        # Enhancement registry
        self.enhancements: Dict[str, EnhancementState] = {}
        
        # Global configuration
        self.global_config = {
            'max_global_failures': 10,
            'failure_window': 300.0,  # 5 minutes
            'emergency_shutdown_threshold': 0.5,  # Disable if >50% fail
            'recovery_cooldown': 60.0,  # 1 minute recovery wait
        }
        
        # Numerical stability thresholds by level
        self.stability_thresholds = {
            StabilityLevel.STRICT: {
                'nan_tolerance': 0,
                'inf_tolerance': 0, 
                'numerical_epsilon': 1e-12,
                'convergence_tolerance': 1e-10,
                'max_condition_number': 1e12
            },
            StabilityLevel.MODERATE: {
                'nan_tolerance': 0,
                'inf_tolerance': 0,
                'numerical_epsilon': 1e-8,
                'convergence_tolerance': 1e-6,
                'max_condition_number': 1e14
            },
            StabilityLevel.RELAXED: {
                'nan_tolerance': 1,  # Allow occasional NaN
                'inf_tolerance': 1,  # Allow occasional Inf
                'numerical_epsilon': 1e-6,
                'convergence_tolerance': 1e-4,
                'max_condition_number': 1e16
            }
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def register_enhancement(self, 
                           name: str,
                           config: Optional[EnhancementConfig] = None,
                           activation_function: Optional[Callable] = None) -> None:
        """Register a mathematical enhancement for safety management"""
        with self._lock:
            if config is None:
                config = EnhancementConfig(name=name)
                
            state = EnhancementState(
                status=EnhancementStatus.ENABLED if config.enabled else EnhancementStatus.DISABLED,
                config=config,
                metrics=StabilityMetrics()
            )
            
            # Check dependencies
            for dep in config.dependencies:
                if dep not in self.enhancements:
                    self.logger.warning(f"Enhancement {name} depends on unregistered {dep}")
                else:
                    self.enhancements[dep].dependent_enhancements.append(name)
            
            self.enhancements[name] = state
            self.logger.info(f"Registered enhancement: {name} with status: {state.status.value}")
            
            # Store activation function
            if activation_function:
                setattr(self, f"_activation_fn_{name}", activation_function)
    
    def is_enhancement_active(self, name: str) -> bool:
        """Check if an enhancement is currently active"""
        with self._lock:
            if name not in self.enhancements:
                self.logger.warning(f"Unknown enhancement: {name}")
                return False
                
            state = self.enhancements[name]
            return state.status == EnhancementStatus.ENABLED
    
    def enable_enhancement(self, name: str, force: bool = False) -> bool:
        """Enable a mathematical enhancement"""
        with self._lock:
            if name not in self.enhancements:
                self.logger.error(f"Cannot enable unknown enhancement: {name}")
                return False
                
            state = self.enhancements[name]
            
            # Check dependencies unless forced
            if not force:
                for dep in state.config.dependencies:
                    if not self.is_enhancement_active(dep):
                        self.logger.warning(f"Cannot enable {name}: dependency {dep} not active")
                        return False
            
            # Reset failure count on manual enable
            state.failure_count = 0
            state.last_failure = None
            state.status = EnhancementStatus.ENABLED
            
            self.logger.info(f"Enabled enhancement: {name}")
            return True
    
    def disable_enhancement(self, name: str, reason: str = "manual") -> bool:
        """Disable a mathematical enhancement"""
        with self._lock:
            if name not in self.enhancements:
                self.logger.error(f"Cannot disable unknown enhancement: {name}")
                return False
                
            state = self.enhancements[name]
            old_status = state.status
            
            if reason == "auto":
                state.status = EnhancementStatus.AUTO_DISABLED
            else:
                state.status = EnhancementStatus.DISABLED
                
            # Disable dependent enhancements
            for dep_name in state.dependent_enhancements:
                if self.is_enhancement_active(dep_name):
                    self.disable_enhancement(dep_name, f"dependency_{name}_disabled")
            
            self.logger.info(f"Disabled enhancement {name}: {old_status.value} -> {state.status.value} (reason: {reason})")
            return True
    
    @contextmanager
    def safe_computation(self, 
                        enhancement_name: str,
                        operation_name: str = "unknown",
                        max_iterations: Optional[int] = None):
        """Context manager for numerically safe computation"""
        if not self.is_enhancement_active(enhancement_name):
            raise ValueError(f"Enhancement {enhancement_name} is not active")
            
        state = self.enhancements[enhancement_name]
        iteration_limit = max_iterations or state.config.max_iterations
        
        # Context tracking
        context = SafeComputationContext(
            enhancement_name=enhancement_name,
            operation_name=operation_name,
            iteration_limit=iteration_limit,
            tolerance=state.config.numerical_tolerance,
            stability_level=state.config.stability_level,
            controller=self
        )
        
        try:
            yield context
            
            # Update success metrics
            state.metrics.total_operations += 1
            
        except NumericalInstabilityError as e:
            # Handle numerical instability
            self._handle_numerical_failure(enhancement_name, str(e), operation_name)
            raise
            
        except Exception as e:
            # Handle general errors
            self._handle_general_failure(enhancement_name, str(e), operation_name)
            raise
    
    def get_enhancement_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get status information for enhancements"""
        with self._lock:
            if name:
                if name not in self.enhancements:
                    return {}
                return self._serialize_enhancement_state(name, self.enhancements[name])
            else:
                return {
                    name: self._serialize_enhancement_state(name, state)
                    for name, state in self.enhancements.items()
                }
    
    def get_pipeline_artifact_metadata(self) -> Dict[str, Any]:
        """Get metadata about enhancement status for pipeline artifacts"""
        with self._lock:
            active_enhancements = [
                name for name, state in self.enhancements.items()
                if state.status == EnhancementStatus.ENABLED
            ]
            
            disabled_enhancements = [
                name for name, state in self.enhancements.items()
                if state.status in [EnhancementStatus.AUTO_DISABLED, EnhancementStatus.DISABLED]
            ]
            
            stability_summary = {}
            for name, state in self.enhancements.items():
                stability_summary[name] = {
                    'total_operations': state.metrics.total_operations,
                    'failure_rate': state.failure_count / max(1, state.metrics.total_operations),
                    'stability_level': state.config.stability_level.value
                }
            
            return {
                'enhancement_status': {
                    'active': active_enhancements,
                    'disabled': disabled_enhancements,
                    'timestamp': time.time()
                },
                'stability_summary': stability_summary,
                'safety_controller_version': '1.0.0'
            }
    
    def create_stability_guard(self, 
                              enhancement_name: str,
                              operation_type: str = "general") -> 'StabilityGuard':
        """Create a stability guard for numerical operations"""
        if enhancement_name not in self.enhancements:
            raise ValueError(f"Unknown enhancement: {enhancement_name}")
            
        state = self.enhancements[enhancement_name]
        thresholds = self.stability_thresholds[state.config.stability_level]
        
        return StabilityGuard(
            enhancement_name=enhancement_name,
            operation_type=operation_type,
            thresholds=thresholds,
            controller=self
        )
    
    def _handle_numerical_failure(self, 
                                 enhancement_name: str, 
                                 error_msg: str,
                                 operation_name: str) -> None:
        """Handle numerical instability failures"""
        with self._lock:
            state = self.enhancements[enhancement_name]
            state.failure_count += 1
            state.last_failure = f"Numerical: {error_msg} in {operation_name}"
            state.last_failure_time = time.time()
            
            # Auto-disable if failure limit exceeded
            if (state.config.auto_disable_on_failure and 
                state.failure_count >= state.config.failure_count_limit):
                self.disable_enhancement(enhancement_name, "auto")
                
            self.logger.error(f"Numerical failure in {enhancement_name}: {error_msg}")
    
    def _handle_general_failure(self, 
                               enhancement_name: str, 
                               error_msg: str,
                               operation_name: str) -> None:
        """Handle general computation failures"""
        with self._lock:
            state = self.enhancements[enhancement_name]
            state.failure_count += 1
            state.last_failure = f"General: {error_msg} in {operation_name}"
            state.last_failure_time = time.time()
            
            self.logger.error(f"General failure in {enhancement_name}: {error_msg}")
    
    def _serialize_enhancement_state(self, name: str, state: EnhancementState) -> Dict[str, Any]:
        """Serialize enhancement state for status reporting"""
        return {
            'name': name,
            'status': state.status.value,
            'enabled': state.status == EnhancementStatus.ENABLED,
            'failure_count': state.failure_count,
            'last_failure': state.last_failure,
            'metrics': {
                'total_operations': state.metrics.total_operations,
                'nan_count': state.metrics.nan_count,
                'inf_count': state.metrics.inf_count,
            }
        }


class SafeComputationContext:
    """Context for safe mathematical computations with stability checking"""
    
    def __init__(self, 
                 enhancement_name: str,
                 operation_name: str,
                 iteration_limit: int,
                 tolerance: float,
                 stability_level: StabilityLevel,
                 controller: MathEnhancementSafetyController):
        self.enhancement_name = enhancement_name
        self.operation_name = operation_name
        self.iteration_limit = iteration_limit
        self.tolerance = tolerance
        self.stability_level = stability_level
        self.controller = controller
        self.iteration_count = 0
        
        # Get stability thresholds
        self.thresholds = controller.stability_thresholds[stability_level]
    
    def check_iteration_limit(self) -> None:
        """Check if iteration limit is exceeded"""
        self.iteration_count += 1
        if self.iteration_count > self.iteration_limit:
            raise IterationLimitExceededError(
                f"Iteration limit {self.iteration_limit} exceeded in {self.operation_name}"
            )
    
    def check_stability(self, 
                       values: Union[np.ndarray, torch.Tensor, float],
                       operation_step: str = "unknown") -> None:
        """Check numerical stability of values"""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, (int, float)):
            values = np.array([values])
        elif not isinstance(values, np.ndarray):
            values = np.array(values)
        
        # Check for NaN
        nan_count = np.sum(np.isnan(values))
        if nan_count > self.thresholds['nan_tolerance']:
            raise NumericalInstabilityError(
                f"NaN detected ({nan_count} values) in {operation_step}"
            )
        
        # Check for Inf
        inf_count = np.sum(np.isinf(values))
        if inf_count > self.thresholds['inf_tolerance']:
            raise NumericalInstabilityError(
                f"Inf detected ({inf_count} values) in {operation_step}"
            )
        
        # Update metrics
        state = self.controller.enhancements[self.enhancement_name]
        state.metrics.nan_count += nan_count
        state.metrics.inf_count += inf_count
    
    def safe_comparison(self, a: float, b: float) -> Tuple[bool, bool, bool]:
        """Epsilon-based floating point comparison"""
        epsilon = self.thresholds['numerical_epsilon']
        diff = abs(a - b)
        
        is_equal = diff < epsilon
        is_less = (a - b) < -epsilon
        is_greater = (a - b) > epsilon
        
        return is_equal, is_less, is_greater


class StabilityGuard:
    """Numerical stability guard for mathematical operations"""
    
    def __init__(self, 
                 enhancement_name: str,
                 operation_type: str,
                 thresholds: Dict[str, float],
                 controller: MathEnhancementSafetyController):
        self.enhancement_name = enhancement_name
        self.operation_type = operation_type
        self.thresholds = thresholds
        self.controller = controller
    
    def guard_matrix_operation(self, matrix: np.ndarray) -> np.ndarray:
        """Guard matrix operations with numerical stability checks"""
        # Check for NaN/Inf
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            raise NumericalInstabilityError("Matrix contains NaN or Inf values")
        
        # Check condition number for invertibility
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            try:
                cond_num = np.linalg.cond(matrix)
                if cond_num > self.thresholds['max_condition_number']:
                    # Add regularization
                    epsilon = self.thresholds['numerical_epsilon']
                    matrix = matrix + epsilon * np.eye(matrix.shape[0])
            except np.linalg.LinAlgError:
                pass  # Skip condition check if computation fails
        
        return matrix
    
    def guard_eigenvalue_computation(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Guard eigenvalue computation with stability checks"""
        matrix = self.guard_matrix_operation(matrix)
        
        try:
            eigenvals, eigenvecs = np.linalg.eig(matrix)
            
            # Check for complex eigenvalues in real matrices
            if np.any(np.iscomplex(eigenvals)) and not np.iscomplexobj(matrix):
                warnings.warn("Complex eigenvalues detected in real matrix", RuntimeWarning)
            
            # Sort eigenvalues for consistency
            idx = np.argsort(np.real(eigenvals))[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            return eigenvals, eigenvecs
            
        except np.linalg.LinAlgError as e:
            raise NumericalInstabilityError(f"Eigenvalue computation failed: {e}")


# Custom exceptions for numerical stability
class NumericalInstabilityError(Exception):
    """Raised when numerical instability is detected"""
    pass


class IterationLimitExceededError(Exception):
    """Raised when iteration limits are exceeded"""
    pass


class ConvergenceError(Exception):
    """Raised when convergence fails or divergence is detected"""
    pass


# Decorator for automatic enhancement safety
def safe_enhancement(enhancement_name: str, 
                    operation_name: Optional[str] = None):
    """Decorator to automatically wrap functions with safety controller"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get controller from kwargs or create default
            controller = kwargs.pop('safety_controller', None)
            if controller is None:
                controller = MathEnhancementSafetyController()
            
            op_name = operation_name or func.__name__
            
            with controller.safe_computation(enhancement_name, op_name) as ctx:
                result = func(*args, **kwargs)
                
                # Auto-check result if it's numerical
                if isinstance(result, (np.ndarray, torch.Tensor, float, int)):
                    ctx.check_stability(result, "function_output")
                
                return result
        
        return wrapper
    return decorator