# Lightweight deterministic shim for Hypothesis to ensure tests run without external dependency
# This shim provides the subset of API used by our tests: given, settings, and strategies as st.
# It is intentionally deterministic to satisfy the project's determinism requirements.

from typing import Any, Callable

# Re-export strategies implemented in local shim
from . import strategies as st  # type: ignore


def _eval_strategy(v: Any) -> Any:
    """Evaluate a strategy-like value produced by our local strategies shim.
    If it's a callable (zero-arg), call it to produce a deterministic example.
    Otherwise, return as-is.
    """
    try:
        if callable(v):
            return v()
    except Exception:
        pass
    return v


def given(**arg_strategies: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Minimal @given decorator: calls the test once with one deterministic example.
    The real Hypothesis generates many cases; our shim provides one deterministic case
    to validate invariants without introducing non-determinism.
    """
    def decorator(test_func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build concrete arguments by evaluating provided strategies
            concrete_kwargs = {k: _eval_strategy(v) for k, v in arg_strategies.items()}
            concrete_kwargs.update(kwargs)
            return test_func(*args, **concrete_kwargs)
        return wrapper
    return decorator


def settings(**_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """No-op settings decorator for compatibility (accepts any kwargs)."""
    def decorator(test_func: Callable[..., Any]) -> Callable[..., Any]:
        return test_func
    return decorator
