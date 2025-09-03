"""Exception telemetry and structured logging module for monitoring and debugging."""

import json
import logging
import threading
import traceback
import uuid
# # # from collections import Counter, defaultdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, Optional, Type  # Module not found  # Module not found  # Module not found


class ExceptionTelemetry:
    """Centralized exception telemetry and logging system."""

    def __init__(self, logger_name: str = "exception_telemetry"):
        self.logger = logging.getLogger(logger_name)
        self._exception_stats = defaultdict(Counter)
        self._lock = threading.Lock()

        # Configure structured logging format
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.ERROR)

    def log_exception(
        self,
        exc_type: Type[Exception],
        exc_value: Exception,
        exc_traceback: Any,
        context: Optional[Dict[str, Any]] = None,
        component: str = "unknown",
        operation: str = "unknown",
    ) -> str:
        """
        Log exception with structured format and telemetry data collection.

        Args:
            exc_type: Exception type
            exc_value: Exception instance
            exc_traceback: Exception traceback
            context: Additional context variables
            component: Component where exception occurred
            operation: Operation being performed when exception occurred

        Returns:
            Exception ID for correlation
        """
        exc_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat()

        # Collect telemetry data
        with self._lock:
            self._exception_stats[component][exc_type.__name__] += 1

        # Build structured log entry
        log_data = {
            "exception_id": exc_id,
            "timestamp": timestamp,
            "component": component,
            "operation": operation,
            "exception_type": exc_type.__name__,
            "exception_message": str(exc_value),
            "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback),
            "context": context or {},
        }

        # Log structured exception data
        self.logger.error(
            f"Exception in {component}.{operation} [ID: {exc_id}]: "
            f"{exc_type.__name__}: {exc_value}",
            extra={"structured_data": json.dumps(log_data, default=str)},
        )

        return exc_id

    def get_exception_stats(self) -> Dict[str, Dict[str, int]]:
        """Get current exception statistics for monitoring."""
        with self._lock:
            return dict(self._exception_stats)

    def clear_stats(self):
        """Clear exception statistics."""
        with self._lock:
            self._exception_stats.clear()


# Global telemetry instance
_telemetry = ExceptionTelemetry()


def log_structured_exception(
    exc_type: Type[Exception],
    exc_value: Exception,
    exc_traceback: Any,
    context: Optional[Dict[str, Any]] = None,
    component: str = "unknown",
    operation: str = "unknown",
) -> str:
    """
    Convenience function for logging exceptions with structured format.

    Usage:
        try:
            # some operation
            pass
        except Exception as e:
            log_structured_exception(
                type(e), e, e.__traceback__,
                context={"variable": value},
                component="feature_extractor",
                operation="extract_features"
            )
    """
    return _telemetry.log_exception(
        exc_type, exc_value, exc_traceback, context, component, operation
    )


def get_telemetry_stats() -> Dict[str, Dict[str, int]]:
    """Get current exception telemetry statistics."""
    return _telemetry.get_exception_stats()


def clear_telemetry_stats():
    """Clear exception telemetry statistics."""
    _telemetry.clear_stats()
