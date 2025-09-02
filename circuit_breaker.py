"""
Circuit breaker pattern implementation for fault tolerance.
"""

import time
from enum import Enum
from typing import Any, Callable, Optional, Type, Union, Tuple
import threading
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing fast, not calling function
    HALF_OPEN = "half_open"  # Testing if service has recovered


class BreakerOpenError(Exception):
    """Exception raised when circuit breaker is in open state."""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Failing fast, calls immediately raise BreakerOpenError
    - HALF_OPEN: Testing recovery, limited calls allowed through
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        name: Optional[str] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exceptions: Exception types that trigger the circuit breaker
            name: Optional name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.name = name or f"CircuitBreaker-{id(self)}"
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
    def execute(self, func: Callable[[], Any]) -> Any:
        """
        Execute function through circuit breaker protection.
        
        Args:
            func: Callable to execute
            
        Returns:
            Result of function execution
            
        Raises:
            BreakerOpenError: When circuit is open
            Any exception raised by the function
        """
        with self._lock:
            current_time = time.time()
            
            # Check if we should transition from OPEN to HALF_OPEN
            if (self.state == CircuitState.OPEN and 
                current_time >= self.last_failure_time + self.recovery_timeout):
                self._transition_to_half_open()
            
            # Handle OPEN state
            if self.state == CircuitState.OPEN:
                logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise BreakerOpenError(f"Circuit breaker {self.name} is open")
            
            # Execute function
            try:
                result = func()
                self._on_success()
                return result
                
            except self.expected_exceptions as e:
                self._on_failure(e, current_time)
                raise
            except Exception as e:
                # Unexpected exceptions don't affect circuit state
                logger.warning(
                    f"Circuit breaker {self.name} encountered unexpected exception: {e}"
                )
                raise
                
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker {self.name} recovered, closing circuit")
            self._reset()
        # Success in CLOSED state doesn't change anything
        
    def _on_failure(self, exception: Exception, current_time: float):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = current_time
        
        logger.warning(
            f"Circuit breaker {self.name} failure {self.failure_count}/{self.failure_threshold}: {exception}"
        )
        
        if self.failure_count >= self.failure_threshold:
            self._transition_to_open()
            
    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        self.state = CircuitState.OPEN
        logger.error(
            f"Circuit breaker {self.name} opened after {self.failure_count} failures"
        )
        
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN for testing")
        
    def _reset(self):
        """Reset circuit breaker to healthy state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self._reset()
            logger.info(f"Circuit breaker {self.name} manually reset")
            
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self.last_failure_time,
                "recovery_timeout": self.recovery_timeout,
                "time_until_retry": max(
                    0, 
                    self.last_failure_time + self.recovery_timeout - time.time()
                ) if self.state == CircuitState.OPEN else 0
            }


__all__ = ["CircuitBreaker", "CircuitState", "BreakerOpenError"]