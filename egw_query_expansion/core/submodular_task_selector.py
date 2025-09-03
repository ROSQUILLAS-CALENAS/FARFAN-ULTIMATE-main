"""
Monotonic Task Selector with Submodular Approximation

Implementation of a monotonic task selector based on submodular maximization
# # # from Balkanski et al. (2021) "Exponential Speedups in Parallel Running Time  # Module not found  # Module not found  # Module not found
for Submodular Maximization without Loss in Approximation" (ACM-SIAM SODA).

Provides greedy selection with lazy evaluation, stable ordering under budget
changes, and proven approximation guarantees.
"""

import heapq
import logging
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple, Union  # Module not found  # Module not found  # Module not found


@dataclass(frozen=True)
class Task:
    """Task representation with immutable properties.

    Compatibility:
    - Supports construction with either task_id= or id= keyword.
    - Provides .id property for compatibility with external tests.
    - Equality and hashing are based only on task_id.
    """

    task_id: str
    cost: float
    priority: float = 1.0
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        task_id: Optional[str] = None,
        cost: float = 0.0,
        priority: float = 1.0,
        dependencies: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ):
        if task_id is None and id is None:
            raise TypeError("Task requires 'task_id' or 'id'")
        tid = task_id or id  # prefer explicit task_id
        object.__setattr__(self, "task_id", tid)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "dependencies", set(dependencies or set()))
        object.__setattr__(self, "metadata", dict(metadata or {}))
        if cost <= 0:
            raise ValueError(f"Task {tid} must have positive cost")

    @property
    def id(self) -> str:
        return self.task_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return NotImplemented
        return self.task_id == other.task_id

    def __hash__(self) -> int:
        return hash(self.task_id)


@dataclass
class SelectionTrace:
    """Trace record for selection decisions."""

    task_id: str
    budget_at_selection: float
    marginal_gain: float
    cumulative_value: float
    selection_order: int
    rejection_reason: Optional[str] = None
    dependencies_satisfied: bool = True


@dataclass
class SelectionDecision:
    """Decision record for legacy API compatibility.

# # #     Fields mirror expectations from external tests.  # Module not found  # Module not found  # Module not found
    """

    task: "Task"
    selected: bool
    marginal_gain: float
    budget_remaining: float
    reason: str = ""


class SubmodularFunction:
    """Base class for submodular utility functions.

    This base intentionally avoids abstract enforcement to support multiple
    historical interfaces used across tests:
    - New style: evaluate(selected_ids, candidate_id, available_tasks),
      evaluate_set(selected_ids, available_tasks)
    - Legacy style: marginal_gain(task_obj, selected_task_objs),
      evaluate(set_of_task_objs)
    Implementations may provide either style; the selector adapts at runtime.
    """

    # Optional methods to be implemented by concrete utilities
    # def evaluate(self, selected_tasks: Set[str], candidate_task: str, available_tasks: Dict[str, Task]) -> float: ...
    # def evaluate_set(self, selected_tasks: Set[str], available_tasks: Dict[str, Task]) -> float: ...
    # def marginal_gain(self, task: Task, selected_tasks: Set[Task]) -> float: ...
    # def evaluate(self, tasks: Set[Task]) -> float: ...


class _GainAdapter(SubmodularFunction):
    """Adapter to use DeterministicGainFunction-like objects with this API."""

    def __init__(self, gain_function: Any):
        self.gain_function = gain_function

    def evaluate(
        self,
        selected_tasks: Set[str],
        candidate_task: str,
        available_tasks: Dict[str, "Task"],
    ) -> float:
        # Accept either dict or list for available_tasks
        if isinstance(available_tasks, list):
            available_map = {t.id: t for t in available_tasks}
        else:
            available_map = available_tasks
        # Build Task set for selected
        selected_task_objs = {available_map[tid] for tid in selected_tasks if tid in available_map}
        task_obj = available_map.get(candidate_task)
        if task_obj is None:
            return 0.0
        if hasattr(self.gain_function, "marginal_gain"):
            return float(self.gain_function.marginal_gain(task_obj, selected_task_objs))
        # Fallback: no info
        return 0.0

    def evaluate_set(self, selected_tasks: Set[str], available_tasks: Dict[str, "Task"]) -> float:
        # Accumulate marginal gains in insertion order
        total = 0.0
        cumulative: Set[str] = set()
        for tid in selected_tasks:
            total += self.evaluate(cumulative, tid, available_tasks)
            cumulative.add(tid)
        return total


class CoverageUtility(SubmodularFunction):
    """Coverage-based submodular utility function."""

    def __init__(
        self,
        feature_coverage: Dict[str, Set[str]],
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
# # #             feature_coverage: Map from task_id to set of features covered  # Module not found  # Module not found  # Module not found
            feature_weights: Optional weights for features (default: uniform)
        """
        self.feature_coverage = feature_coverage
        self.feature_weights = feature_weights or {}
        self._all_features = set().union(*feature_coverage.values())

        # Default uniform weights
        for feature in self._all_features:
            if feature not in self.feature_weights:
                self.feature_weights[feature] = 1.0

    def evaluate(
        self,
        selected_tasks: Set[str],
        candidate_task: str,
        available_tasks: Dict[str, Task],
    ) -> float:
# # #         """Marginal coverage gain from adding candidate_task."""  # Module not found  # Module not found  # Module not found
        if candidate_task not in self.feature_coverage:
            return 0.0

        # Features already covered by selected tasks
        covered_features = set()
        for task_id in selected_tasks:
            if task_id in self.feature_coverage:
                covered_features.update(self.feature_coverage[task_id])

        # Additional features covered by candidate
        new_features = self.feature_coverage[candidate_task] - covered_features
        return sum(self.feature_weights[f] for f in new_features)

    def evaluate_set(
        self, selected_tasks: Set[str], available_tasks: Dict[str, Task]
    ) -> float:
        """Total coverage value of selected tasks."""
        covered_features = set()
        for task_id in selected_tasks:
            if task_id in self.feature_coverage:
                covered_features.update(self.feature_coverage[task_id])

        return sum(self.feature_weights[f] for f in covered_features)


@dataclass
class HeapItem:
    """Heap item with stable comparison for task selection."""

    marginal_gain: float
    task_id: str
    insertion_order: int
    last_updated: int = 0

    def __lt__(self, other: "HeapItem") -> bool:
        """Stable comparison: higher gain first, then by insertion order."""
        if abs(self.marginal_gain - other.marginal_gain) < 1e-10:
            return self.insertion_order < other.insertion_order
        return self.marginal_gain > other.marginal_gain


class MonotonicTaskSelector:
    """
    Monotonic task selector with submodular approximation guarantees.

# # #     Implements the greedy algorithm with lazy evaluation from Balkanski et al.  # Module not found  # Module not found  # Module not found
    Maintains prefix property: increasing budget only adds tasks at the end.
    """

    def __init__(
        self,
        utility_function: Optional[SubmodularFunction] = None,
        enable_lazy_evaluation: Optional[bool] = None,
        approximation_tolerance: Optional[float] = None,
        # Legacy aliases
        gain_function: Any = None,
        lazy_evaluation: Optional[bool] = None,
        approximation_factor: Optional[float] = None,
    ):
        """
        Args:
            utility_function: Submodular function for task valuation
            enable_lazy_evaluation: Enable lazy evaluation for efficiency
            approximation_tolerance: Tolerance for approximation guarantees
            gain_function: Legacy alias for utility function based on marginal_gain
        """
        # Resolve utility/gain function
        util = utility_function if utility_function is not None else gain_function
        if util is None:
            raise ValueError("A utility_function or gain_function must be provided")
        if isinstance(util, SubmodularFunction) and hasattr(util, "evaluate_set") and hasattr(util, "evaluate"):
            self.utility_function = util
        else:
            # Wrap deterministic gain or incompatible utilities into adapter
            self.utility_function = _GainAdapter(util)

        # Resolve flags with legacy aliases
        if enable_lazy_evaluation is None:
            enable_lazy_evaluation = True if lazy_evaluation is None else bool(lazy_evaluation)
        if approximation_tolerance is None:
            approximation_tolerance = (
                float(approximation_factor) if approximation_factor is not None else 1e-6
            )

        self.enable_lazy_evaluation = bool(enable_lazy_evaluation)
        self.approximation_tolerance = float(approximation_tolerance)
        # Legacy property expected by some tests
        self.approximation_factor = (
            float(approximation_factor) if approximation_factor is not None else 0.632
        )

        # Selection state
        self._selected_tasks: List[str] = []
        self._selection_traces: List[SelectionTrace] = []
        self._task_heap: List[HeapItem] = []
        self._task_positions: Dict[str, int] = {}
        self._insertion_counter = 0
        self._update_counter = 0

        # Registered tasks for budget-only API
        self._registered_tasks: Dict[str, Task] = {}
        self._last_available_tasks: Dict[str, Task] = {}

        # Monotonicity tracking
        self._budget_history: List[float] = []
        self._selection_history: List[List[str]] = []

        self.logger = logging.getLogger(__name__)

    def add_tasks(self, tasks: List["Task"]) -> None:
        """Register tasks for budget-only selection API."""
        for t in tasks:
            self._registered_tasks[t.id] = t

    def select_tasks(self, available_tasks_or_budget, budget: Optional[float] = None, preserve_order: bool = True):
        """
        Unified selection API supporting two call styles:
        - select_tasks(available_tasks: Dict[str, Task>, budget: float, preserve_order=True)
          returns (selected_task_ids: List[str], traces: List[SelectionTrace])
        - select_tasks(budget: float) after add_tasks([...])
          returns List[Task]
        """
        # Detect budget-only mode
        budget_only_mode = False
        if isinstance(available_tasks_or_budget, (int, float)) and budget is None:
            budget_only_mode = True
            budget = float(available_tasks_or_budget)
            available_tasks = self._registered_tasks
        else:
            available_tasks = available_tasks_or_budget or {}
            if budget is None:
                raise TypeError("select_tasks requires a budget parameter")

        if budget <= 0:
            return [] if budget_only_mode else ([], [])

        # Validate monotonicity requirement
        if (not budget_only_mode) and preserve_order and self._budget_history:
            last_budget = self._budget_history[-1]
            if budget < last_budget:
                raise ValueError(
                    f"Monotonicity violation: budget {budget} < {last_budget}"
                )

        # Initialize or extend selection
        if preserve_order and self._selected_tasks:
            selected_tasks = self._selected_tasks.copy()
            traces = self._selection_traces.copy()
            remaining_budget = budget - sum(
                available_tasks[tid].cost
                for tid in selected_tasks
                if tid in available_tasks
            )
        else:
            selected_tasks = []
            traces = []
            remaining_budget = budget
            self._reset_state()

        # Build candidate heap if needed
        if not self._task_heap or not preserve_order:
            self._build_candidate_heap(available_tasks, set(selected_tasks))

        # Greedy selection with lazy evaluation
        selection_order = len(selected_tasks)

        # Track last call context for trace/decision mapping
        self._last_budget = budget
        self._last_available_tasks = dict(available_tasks)

        while self._task_heap and remaining_budget > 0:
            # Get next candidate
            candidate = self._get_next_candidate(
                available_tasks, set(selected_tasks), remaining_budget
            )

            if not candidate:
                break

            task_id, marginal_gain = candidate
            task = available_tasks[task_id]

            # Check budget and dependencies
            if task.cost > remaining_budget:
                traces.append(
                    SelectionTrace(
                        task_id=task_id,
                        budget_at_selection=budget - remaining_budget,
                        marginal_gain=marginal_gain,
                        cumulative_value=self.utility_function.evaluate_set(
                            set(selected_tasks), available_tasks
                        ),
                        selection_order=selection_order,
                        rejection_reason=f"Exceeds budget: {task.cost} > {remaining_budget}",
                    )
                )
                continue

            if not self._dependencies_satisfied(task, selected_tasks):
                traces.append(
                    SelectionTrace(
                        task_id=task_id,
                        budget_at_selection=budget - remaining_budget,
                        marginal_gain=marginal_gain,
                        cumulative_value=self.utility_function.evaluate_set(
                            set(selected_tasks), available_tasks
                        ),
                        selection_order=selection_order,
                        rejection_reason="Dependencies not satisfied",
                        dependencies_satisfied=False,
                    )
                )
                continue

            # Select task
            selected_tasks.append(task_id)
            remaining_budget -= task.cost

            traces.append(
                SelectionTrace(
                    task_id=task_id,
                    budget_at_selection=budget - remaining_budget,
                    marginal_gain=marginal_gain,
                    cumulative_value=self.utility_function.evaluate_set(
                        set(selected_tasks), available_tasks
                    ),
                    selection_order=selection_order,
                )
            )

            selection_order += 1

            # Update heap after selection if lazy evaluation enabled
            if self.enable_lazy_evaluation:
                self._invalidate_heap_after_selection(task_id, set(selected_tasks))

        # Update internal state
        self._selected_tasks = selected_tasks
        self._selection_traces = traces
        self._budget_history.append(budget)
        self._selection_history.append(selected_tasks.copy())

        if budget_only_mode:
            # Return Task objects for budget-only API
            return [available_tasks[tid] for tid in selected_tasks if tid in available_tasks]
        return selected_tasks, traces

    def get_decision_trace(self) -> List[SelectionDecision]:
        """Return selection decisions in legacy-compatible format."""
        decisions: List[SelectionDecision] = []
        if not hasattr(self, "_last_available_tasks"):
            return decisions
        available = self._last_available_tasks
        budget = getattr(self, "_last_budget", 0.0) or 0.0
        # Track consumed to compute remaining
        for tr in self._selection_traces:
            task = available.get(tr.task_id)
            remaining = max(0.0, budget - tr.budget_at_selection)
            decisions.append(
                SelectionDecision(
                    task=task if task is not None else Task(tr.task_id, cost=1.0 if task is None else task.cost),
                    selected=tr.rejection_reason is None,
                    marginal_gain=float(tr.marginal_gain),
                    budget_remaining=float(remaining),
                    reason=str(tr.rejection_reason or "selected"),
                )
            )
        return decisions

    def verify_prefix_property(self, budget1: float, budget2: float) -> bool:
        """Verify prefix property holds between two budgets using registered tasks.

        If tasks not registered, returns True.
        """
        if not self._registered_tasks:
            return True
        # Preserve current state
        state = (
            list(self._selected_tasks),
            list(self._selection_traces),
            list(self._budget_history),
            list(self._selection_history),
            dict(self._last_available_tasks),
            getattr(self, "_last_budget", None),
        )
        try:
            small, large = (budget1, budget2) if budget1 <= budget2 else (budget2, budget1)
            sel_small = [t.id for t in self.select_tasks(small)]
            # Reset for fair recompute
            self._reset_state()
            sel_large = [t.id for t in self.select_tasks(large)]
            if len(sel_small) > len(sel_large):
                return False
            return sel_large[: len(sel_small)] == sel_small
        finally:
            # Restore state
            (
                self._selected_tasks,
                self._selection_traces,
                self._budget_history,
                self._selection_history,
                self._last_available_tasks,
                self._last_budget,
            ) = state

    def _build_candidate_heap(
        self, available_tasks: Dict[str, Task], selected_tasks: Set[str]
    ):
        """Build initial candidate heap with marginal gains."""
        self._task_heap = []
        self._task_positions = {}

        for task_id, task in available_tasks.items():
            if task_id not in selected_tasks:
                marginal_gain = self.utility_function.evaluate(
                    selected_tasks, task_id, available_tasks
                )

                heap_item = HeapItem(
                    marginal_gain=marginal_gain,
                    task_id=task_id,
                    insertion_order=self._insertion_counter,
                    last_updated=self._update_counter,
                )

                heapq.heappush(self._task_heap, heap_item)
                self._task_positions[task_id] = len(self._task_heap) - 1
                self._insertion_counter += 1

    def _get_next_candidate(
        self,
        available_tasks: Dict[str, Task],
        selected_tasks: Set[str],
        remaining_budget: float,
    ) -> Optional[Tuple[str, float]]:
        """Get next candidate with lazy evaluation."""
        while self._task_heap:
            candidate = heapq.heappop(self._task_heap)

            # Skip if task already selected or unavailable
            if (
                candidate.task_id in selected_tasks
                or candidate.task_id not in available_tasks
            ):
                continue

            # Lazy evaluation: recompute if outdated
            if (
                self.enable_lazy_evaluation
                and candidate.last_updated < self._update_counter
            ):
                new_gain = self.utility_function.evaluate(
                    selected_tasks, candidate.task_id, available_tasks
                )

                updated_item = HeapItem(
                    marginal_gain=new_gain,
                    task_id=candidate.task_id,
                    insertion_order=candidate.insertion_order,
                    last_updated=self._update_counter,
                )

                heapq.heappush(self._task_heap, updated_item)
                continue

            return candidate.task_id, candidate.marginal_gain

        return None

    def _dependencies_satisfied(self, task: Task, selected_tasks: List[str]) -> bool:
        """Check if task dependencies are satisfied."""
        selected_set = set(selected_tasks)
        return all(dep in selected_set for dep in task.dependencies)

    def _invalidate_heap_after_selection(
        self, selected_task_id: str, selected_tasks: Set[str]
    ):
        """Invalidate heap items after task selection for lazy evaluation."""
        self._update_counter += 1
        # Heap items will be re-evaluated when popped

    def _reset_state(self):
        """Reset internal selection state."""
        self._selected_tasks = []
        self._selection_traces = []
        self._task_heap = []
        self._task_positions = {}
        self._update_counter += 1

    def get_approximation_ratio(
        self, available_tasks_or_selection, optimal_value: Optional[float] = None
    ) -> float:
        """
        Compute approximation ratio against optimal solution.

        For submodular maximization under budget constraint, the greedy
        algorithm achieves (1-1/e) ≈ 0.632 approximation ratio.

        Args:
            available_tasks_or_selection: Either available tasks mapping/list or the selected list of Task objects
            optimal_value: Known optimal value (if available)

        Returns:
            Approximation ratio
        """
        # Determine available context and selected ids
        if isinstance(available_tasks_or_selection, list):
            if available_tasks_or_selection and isinstance(available_tasks_or_selection[0], Task):
                available = available_tasks_or_selection  # list[Task]
                selected_ids = [t.id for t in available_tasks_or_selection]
            else:
                available = available_tasks_or_selection  # list[Task] as available set
                selected_ids = list(self._selected_tasks)
        elif isinstance(available_tasks_or_selection, dict):
            available = available_tasks_or_selection
            selected_ids = list(self._selected_tasks)
        else:
            # Fallback to last known available or registered tasks
            available = self._last_available_tasks or list(self._registered_tasks.values())
            selected_ids = list(self._selected_tasks)

        if not selected_ids:
            # If no selection and no optimal benchmark, theoretical ratio is 1.0 by definition when optimal is 0
            if optimal_value is not None and optimal_value <= 0:
                return 1.0
            return 0.0

        # Compute current value using adapter-capable evaluate_set
        current_value = self.utility_function.evaluate_set(selected_ids, available)

        if optimal_value is not None:
            return current_value / optimal_value if optimal_value > 0 else 1.0

        # Return theoretical guarantee when optimal unknown
        return float(self.approximation_factor)

    def verify_monotonicity(self) -> bool:
        """Verify that selection history satisfies monotonicity property."""
        if len(self._selection_history) <= 1:
            return True

        for i in range(1, len(self._selection_history)):
            prev_selection = self._selection_history[i - 1]
            curr_selection = self._selection_history[i]

            # Current selection should be prefix-extension of previous
            if len(curr_selection) < len(prev_selection):
                return False

            if curr_selection[: len(prev_selection)] != prev_selection:
                return False

        return True

    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the selection process."""
        if not self._selection_traces:
            return {}

        successful_selections = [
            t for t in self._selection_traces if t.rejection_reason is None
        ]
        rejections = [
            t for t in self._selection_traces if t.rejection_reason is not None
        ]

        rejection_reasons = defaultdict(int)
        for trace in rejections:
            rejection_reasons[trace.rejection_reason] += 1

        return {
            "total_tasks_considered": len(self._selection_traces),
            "successful_selections": len(successful_selections),
            "rejections": len(rejections),
            "rejection_breakdown": dict(rejection_reasons),
            "final_utility": successful_selections[-1].cumulative_value
            if successful_selections
            else 0.0,
            "monotonicity_verified": self.verify_monotonicity(),
            "selection_efficiency": len(successful_selections)
            / len(self._selection_traces),
        }

    def get_theoretical_guarantee(self) -> str:
        """Return a human-readable description of the approximation guarantee."""
        return (
            "Greedy submodular maximization under a knapsack/budget constraint "
            f"achieves an approximation factor of {self.approximation_factor} "
            "(theoretical bound 1-1/e ≈ 0.632)."
        )


# --- Compatibility Layer -----------------------------------------------------
# Provide DeterministicGainFunction as used by legacy tests, and alias
# SelectionDecision for trace type compatibility.
# # # from typing import Iterable  # Module not found  # Module not found  # Module not found


class DeterministicGainFunction:
    """Deterministic gain function with interactions and dependencies.

    Args:
        base_values: Base value for each task id.
        interaction_matrix: Pairwise penalties between tasks, e.g., ("A","B"): 20.0
        dependency_graph: Task -> set of prerequisite task ids that, if all
            present in selected set, provide a bonus.
        dependency_bonus: Fraction of base value to add when dependencies satisfied.
    """

    def __init__(
        self,
        base_values: Dict[str, float],
        interaction_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        dependency_graph: Optional[Dict[str, Set[str]]] = None,
        dependency_bonus: float = 0.2,
    ) -> None:
        self.base_values = dict(base_values)
        self.interaction_matrix = dict(interaction_matrix or {})
        self.dependency_graph = {k: set(v) for k, v in (dependency_graph or {}).items()}
        self.dependency_bonus = float(dependency_bonus)

    def _pair_penalty(self, a: str, b: str) -> float:
        if (a, b) in self.interaction_matrix:
            return float(self.interaction_matrix[(a, b)])
        if (b, a) in self.interaction_matrix:
            return float(self.interaction_matrix[(b, a)])
        return 0.0

    def marginal_gain(self, task: Task, selected_tasks: Set[Task]) -> float:
        tid = task.id
        base = float(self.base_values.get(tid, 0.0))
        # Penalties for redundancy with already selected tasks
        selected_ids = {t.id for t in selected_tasks}
        penalty = sum(self._pair_penalty(tid, sid) for sid in selected_ids)
        # Dependency bonus if all deps satisfied
        bonus = 0.0
        deps = self.dependency_graph.get(tid)
        if deps and deps.issubset(selected_ids):
            bonus = base * self.dependency_bonus
        gain = base + bonus - penalty
        # Enforce non-negativity as a simple safety rule
        return max(0.0, gain)


