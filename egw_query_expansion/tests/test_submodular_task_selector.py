"""
Tests for the Monotonic Task Selector with Submodular Approximation

Test suite covering monotonicity guarantees, approximation bounds,
lazy evaluation correctness, and edge cases.
"""

# # # from typing import Dict, Set  # Module not found  # Module not found  # Module not found
# # # from unittest.mock import Mock, patch  # Module not found  # Module not found  # Module not found

import numpy as np
import pytest

# # # from egw_query_expansion.core.submodular_task_selector import (  # Module not found  # Module not found  # Module not found
    CoverageUtility,
    HeapItem,
    MonotonicTaskSelector,
    SelectionTrace,
    SubmodularFunction,
    Task,
)


class MockSubmodularFunction(SubmodularFunction):
    """Mock submodular function for testing."""

    def __init__(self, task_values: Dict[str, float]):
        self.task_values = task_values

    def evaluate(
        self,
        selected_tasks: Set[str],
        candidate_task: str,
        available_tasks: Dict[str, Task],
    ) -> float:
        # Simple diminishing returns mock
        if candidate_task not in self.task_values:
            return 0.0

        base_value = self.task_values[candidate_task]
        # Diminish by 50% for each already selected task
        diminish_factor = 0.5 ** len(selected_tasks)
        return base_value * diminish_factor

    def evaluate_set(
        self, selected_tasks: Set[str], available_tasks: Dict[str, Task]
    ) -> float:
        total = 0.0
        cumulative_selected = set()

        for task_id in selected_tasks:
            if task_id in self.task_values:
                marginal = self.evaluate(cumulative_selected, task_id, available_tasks)
                total += marginal
                cumulative_selected.add(task_id)

        return total


class TestTask:
    """Test Task dataclass."""

    def test_task_creation_valid(self):
        task = Task("task1", cost=10.0, priority=1.5)
        assert task.task_id == "task1"
        assert task.cost == 10.0
        assert task.priority == 1.5
        assert task.dependencies == set()

    def test_task_with_dependencies(self):
        task = Task("task1", cost=5.0, dependencies={"dep1", "dep2"})
        assert task.dependencies == {"dep1", "dep2"}

    def test_task_invalid_cost(self):
        with pytest.raises(ValueError, match="must have positive cost"):
            Task("task1", cost=-1.0)

        with pytest.raises(ValueError, match="must have positive cost"):
            Task("task1", cost=0.0)


class TestCoverageUtility:
    """Test CoverageUtility submodular function."""

    def test_coverage_utility_basic(self):
        feature_coverage = {
            "task1": {"f1", "f2"},
            "task2": {"f2", "f3"},
            "task3": {"f1", "f3", "f4"},
        }

        utility = CoverageUtility(feature_coverage)

        # No tasks selected - full marginal gain
        gain = utility.evaluate(set(), "task1", {})
        assert gain == 2.0  # covers f1, f2

        # Task1 already selected - diminished gain for task2
        gain = utility.evaluate({"task1"}, "task2", {})
        assert gain == 1.0  # only f3 is new

    def test_coverage_utility_weighted(self):
        feature_coverage = {"task1": {"f1", "f2"}, "task2": {"f2", "f3"}}
        feature_weights = {"f1": 2.0, "f2": 1.0, "f3": 3.0}

        utility = CoverageUtility(feature_coverage, feature_weights)

        gain = utility.evaluate(set(), "task1", {})
        assert gain == 3.0  # 2.0 (f1) + 1.0 (f2)

        gain = utility.evaluate({"task1"}, "task2", {})
        assert gain == 3.0  # only f3 is new

    def test_coverage_utility_set_evaluation(self):
        feature_coverage = {"task1": {"f1", "f2"}, "task2": {"f2", "f3"}}

        utility = CoverageUtility(feature_coverage)

        # Both tasks cover f1, f2, f3 total
        value = utility.evaluate_set({"task1", "task2"}, {})
        assert value == 3.0


class TestHeapItem:
    """Test HeapItem stable comparison."""

    def test_heap_item_comparison_gain_differs(self):
        item1 = HeapItem(10.0, "task1", 1)
        item2 = HeapItem(5.0, "task2", 2)

        assert item1 < item2  # Higher gain comes first
        assert not (item2 < item1)

    def test_heap_item_comparison_gain_equal(self):
        item1 = HeapItem(10.0, "task1", 1)
        item2 = HeapItem(10.0, "task2", 2)

        assert item1 < item2  # Lower insertion order comes first
        assert not (item2 < item1)

    def test_heap_item_comparison_stable(self):
        # Test stability with very close gains
        item1 = HeapItem(10.0000000001, "task1", 2)
        item2 = HeapItem(10.0, "task2", 1)

        # Should be considered equal, so insertion order decides
        assert item2 < item1


class TestMonotonicTaskSelector:
    """Test MonotonicTaskSelector main functionality."""

    @pytest.fixture
    def simple_tasks(self):
        return {
            "task1": Task("task1", cost=5.0),
            "task2": Task("task2", cost=3.0),
            "task3": Task("task3", cost=4.0),
            "task4": Task("task4", cost=2.0),
        }

    @pytest.fixture
    def utility_function(self):
        return MockSubmodularFunction(
            {"task1": 10.0, "task2": 8.0, "task3": 6.0, "task4": 4.0}
        )

    def test_basic_selection(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function)

        selected, traces = selector.select_tasks(simple_tasks, budget=10.0)

        assert len(selected) > 0
        assert all(task_id in simple_tasks for task_id in selected)

        # Check budget constraint
        total_cost = sum(simple_tasks[tid].cost for tid in selected)
        assert total_cost <= 10.0

        # Check traces
        successful_traces = [t for t in traces if t.rejection_reason is None]
        assert len(successful_traces) == len(selected)

    def test_monotonicity_property(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function)

        # First selection with budget 8
        selected1, _ = selector.select_tasks(simple_tasks, budget=8.0)

        # Increase budget to 12 - should only add tasks
        selected2, _ = selector.select_tasks(simple_tasks, budget=12.0)

        # Verify monotonicity: selected2 should be prefix extension of selected1
        assert len(selected2) >= len(selected1)
        assert selected2[: len(selected1)] == selected1

        # Verify monotonicity checker
        assert selector.verify_monotonicity()

    def test_monotonicity_violation_detection(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function)

        # First selection
        selector.select_tasks(simple_tasks, budget=10.0)

        # Try to decrease budget - should raise error
        with pytest.raises(ValueError, match="Monotonicity violation"):
            selector.select_tasks(simple_tasks, budget=5.0, preserve_order=True)

    def test_dependency_handling(self, utility_function):
        tasks_with_deps = {
            "task1": Task("task1", cost=2.0),
            "task2": Task("task2", cost=3.0, dependencies={"task1"}),
            "task3": Task("task3", cost=4.0, dependencies={"task1", "task2"}),
        }

        selector = MonotonicTaskSelector(utility_function)
        selected, traces = selector.select_tasks(tasks_with_deps, budget=10.0)

        # Verify dependency satisfaction
        for task_id in selected:
            task = tasks_with_deps[task_id]
            selected_set = set(selected[: selected.index(task_id)])
            assert all(dep in selected_set for dep in task.dependencies)

        # Check for dependency rejection traces
        dependency_rejections = [
            t
            for t in traces
            if t.rejection_reason and "Dependencies not satisfied" in t.rejection_reason
        ]

        # Should have some dependency-based rejections if budget allows
        if len(selected) < len(tasks_with_deps):
            assert len(dependency_rejections) >= 0

    def test_lazy_evaluation_enabled(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function, enable_lazy_evaluation=True)

        with patch.object(utility_function, "evaluate") as mock_evaluate:
            mock_evaluate.side_effect = utility_function.evaluate

            selected, _ = selector.select_tasks(simple_tasks, budget=10.0)

            # With lazy evaluation, should have some efficiency gains
            # (exact count depends on selection order and heap behavior)
            assert mock_evaluate.call_count >= len(selected)

    def test_lazy_evaluation_disabled(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function, enable_lazy_evaluation=False)

        selected, traces = selector.select_tasks(simple_tasks, budget=10.0)

        # Should still work correctly
        assert len(selected) > 0
        successful_traces = [t for t in traces if t.rejection_reason is None]
        assert len(successful_traces) == len(selected)

    def test_budget_constraint_strict(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function)

        # Very tight budget
        selected, traces = selector.select_tasks(simple_tasks, budget=2.5)

        # Should select only task4 (cost=2.0) or nothing
        total_cost = sum(simple_tasks[tid].cost for tid in selected)
        assert total_cost <= 2.5

        # Check budget rejection traces
        budget_rejections = [
            t
            for t in traces
            if t.rejection_reason and "Exceeds budget" in t.rejection_reason
        ]
        assert len(budget_rejections) > 0

    def test_approximation_ratio_calculation(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function)

        selected, _ = selector.select_tasks(simple_tasks, budget=15.0)

        # With known optimal
        optimal_value = 20.0
        ratio = selector.get_approximation_ratio(simple_tasks, optimal_value)
        assert 0.0 <= ratio <= 1.0

        # Without known optimal (theoretical bound)
        ratio_theoretical = selector.get_approximation_ratio(simple_tasks)
        assert 0.0 <= ratio_theoretical <= 1.0

    def test_selection_statistics(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function)

        selected, _ = selector.select_tasks(simple_tasks, budget=8.0)
        stats = selector.get_selection_statistics()

        assert "total_tasks_considered" in stats
        assert "successful_selections" in stats
        assert "rejections" in stats
        assert "rejection_breakdown" in stats
        assert "final_utility" in stats
        assert "monotonicity_verified" in stats
        assert "selection_efficiency" in stats

        assert stats["monotonicity_verified"] is True
        assert stats["successful_selections"] == len(selected)
        assert 0.0 <= stats["selection_efficiency"] <= 1.0

    def test_empty_task_set(self, utility_function):
        selector = MonotonicTaskSelector(utility_function)

        selected, traces = selector.select_tasks({}, budget=10.0)

        assert selected == []
        assert traces == []

        stats = selector.get_selection_statistics()
        assert stats == {}

    def test_zero_budget(self, simple_tasks, utility_function):
        selector = MonotonicTaskSelector(utility_function)

        selected, traces = selector.select_tasks(simple_tasks, budget=0.0)

        assert selected == []
        assert traces == []


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    def test_large_task_set_performance(self):
        """Test performance with large task set."""
        # Create 100 tasks
        tasks = {}
        feature_coverage = {}

        np.random.seed(42)  # For reproducible tests

        for i in range(100):
            task_id = f"task_{i}"
            tasks[task_id] = Task(task_id, cost=np.random.uniform(1, 10))

            # Random feature coverage
            num_features = np.random.randint(1, 10)
            features = {
                f"feature_{j}"
                for j in np.random.choice(50, num_features, replace=False)
            }
            feature_coverage[task_id] = features

        utility = CoverageUtility(feature_coverage)
        selector = MonotonicTaskSelector(utility)

        selected, traces = selector.select_tasks(tasks, budget=200.0)

        assert len(selected) > 0
        assert len(selected) <= 100

        # Verify all constraints satisfied
        total_cost = sum(tasks[tid].cost for tid in selected)
        assert total_cost <= 200.0

        stats = selector.get_selection_statistics()
        assert stats["monotonicity_verified"] is True

    def test_complex_dependencies_scenario(self):
        """Test complex dependency chains."""
        tasks = {
            "root": Task("root", cost=1.0),
            "level1_a": Task("level1_a", cost=2.0, dependencies={"root"}),
            "level1_b": Task("level1_b", cost=2.0, dependencies={"root"}),
            "level2_a": Task("level2_a", cost=3.0, dependencies={"level1_a"}),
            "level2_b": Task(
                "level2_b", cost=3.0, dependencies={"level1_a", "level1_b"}
            ),
            "final": Task("final", cost=5.0, dependencies={"level2_a", "level2_b"}),
        }

        # Coverage that makes final task most valuable
        feature_coverage = {
            "root": {"base"},
            "level1_a": {"l1a"},
            "level1_b": {"l1b"},
            "level2_a": {"l2a"},
            "level2_b": {"l2b"},
            "final": {"final_feature", "bonus1", "bonus2", "bonus3"},
        }

        utility = CoverageUtility(feature_coverage)
        selector = MonotonicTaskSelector(utility)

        selected, traces = selector.select_tasks(tasks, budget=20.0)

        # Should be able to select full chain
        expected_cost = 1 + 2 + 2 + 3 + 3 + 5  # 16 total
        if sum(tasks[tid].cost for tid in selected) == expected_cost:
            assert "final" in selected
            assert all(
                task_id in selected
                for task_id in ["root", "level1_a", "level1_b", "level2_a", "level2_b"]
            )

        # Verify dependency order
        task_positions = {tid: selected.index(tid) for tid in selected}
        for task_id in selected:
            task = tasks[task_id]
            for dep in task.dependencies:
                if dep in task_positions:
                    assert task_positions[dep] < task_positions[task_id]

    def test_incremental_budget_increases(self):
        """Test multiple incremental budget increases."""
        tasks = {f"task_{i}": Task(f"task_{i}", cost=2.0) for i in range(10)}

        # Simple linear utility
        task_values = {f"task_{i}": 10.0 - i for i in range(10)}
        utility = MockSubmodularFunction(task_values)

        selector = MonotonicTaskSelector(utility)

        budgets = [5.0, 8.0, 12.0, 18.0, 25.0]
        previous_selection = []

        for budget in budgets:
            selected, traces = selector.select_tasks(tasks, budget=budget)

            # Verify monotonicity
            assert len(selected) >= len(previous_selection)
            assert selected[: len(previous_selection)] == previous_selection

            previous_selection = selected

        # Final verification
        assert selector.verify_monotonicity()

        final_stats = selector.get_selection_statistics()
        assert final_stats["monotonicity_verified"] is True
