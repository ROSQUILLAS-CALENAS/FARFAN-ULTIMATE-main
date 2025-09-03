# Minimal deterministic strategy implementations for Hypothesis shim
# This module provides a subset of APIs used in tests:
# - integers, floats, booleans, lists, sampled_from, composite

# # # from typing import Any, Callable, Iterable, List as TList  # Module not found  # Module not found  # Module not found


def integers(min_value: int = -10, max_value: int = 10) -> Callable[[], int]:
    if min_value is None and max_value is None:
        return lambda: 0
    if min_value is None:
        min_value = -10
    if max_value is None:
        max_value = 10
    # Choose midpoint deterministically
    return lambda: int((min_value + max_value) // 2)


def floats(
    min_value: float = -1.0,
    max_value: float = 1.0,
    allow_nan: bool = False,
    allow_infinity: bool = False,
) -> Callable[[], float]:
    if min_value is None and max_value is None:
        return lambda: 0.0
    if min_value is None:
        min_value = -1.0
    if max_value is None:
        max_value = 1.0
    # Deterministic midpoint
    return lambda: (float(min_value) + float(max_value)) / 2.0


def booleans() -> Callable[[], bool]:
    return lambda: False


def lists(
    element_strategy: Callable[[], Any],
    min_size: int = 0,
    max_size: int = 0,
) -> Callable[[], TList[Any]]:
    size = min_size if min_size is not None else 0
    def produce() -> TList[Any]:
        return [element_strategy() for _ in range(size)]
    return produce


def sampled_from(values: Iterable[Any]) -> Callable[[], Any]:
    values = list(values)
    if not values:
        return lambda: None
    return lambda: values[0]


def composite(func: Callable[..., Any]) -> Callable[..., Callable[[], Any]]:
    """
    Decorator to mimic @st.composite behavior. The resulting callable returns
    a zero-argument function producing a deterministic example by calling the
    original function with a simple 'draw' that evaluates strategies.
    """
    def wrapper(*args, **kwargs) -> Callable[[], Any]:
        def draw(strategy_callable: Callable[[], Any]) -> Any:
            return strategy_callable()
        def produce() -> Any:
            return func(draw, *args, **kwargs)
        return produce
    return wrapper
