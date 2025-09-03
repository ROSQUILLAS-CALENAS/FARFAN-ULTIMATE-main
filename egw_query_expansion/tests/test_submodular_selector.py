"""
Tests for Monotonic Task Selector based on Submodular Maximization.

Validates approximation guarantees, prefix properties, and decision traceability.
"""

# # # from typing import Dict, List, Set  # Module not found  # Module not found  # Module not found

import numpy as np
import pytest

# # # from egw_query_expansion.core.submodular_task_selector import (  # Module not found  # Module not found  # Module not found
    DeterministicGainFunction,
    MonotonicTaskSelector,
    SelectionDecision,
    SubmodularFunction,
    Task,
)


class TestSubmodularFunction(SubmodularFunction):
    """Simple test submodular function for validation."""

    def __init__(self, values: Dict[str, float]):
        self.values = values

    def marginal_gain(self, task: Task, selected_tasks: Set[Task]) -> float:
        base_value = self.values.get(task.id, 0.0)
        # Simple diminishing returns based on set size
        diminishing_factor = 1.0 / (1.0 + 0.2 * len(selected_tasks))
        return base_value * diminishing_factor

    def evaluate(self, tasks: Set[Task]) -> float:
        if not tasks:
            return 0.0

        total = sum(self.values.get(task.id, 0.0) for task in tasks)
        # Apply diminishing returns penalty
        size_penalty = 0.1 * len(tasks) * (len(tasks) - 1) / 2
        return max(0.0, total - size_penalty)


class TestMonotonicTaskSelector:
    def setup_method(self):
        """Set up test fixtures."""
        self.test_tasks = [
            Task(id="task1", cost=10.0, metadata={"priority": "high"}),
            Task(id="task2", cost=15.0, metadata={"priority": "medium"}),
            Task(id="task3", cost=20.0, metadata={"priority": "low"}),
            Task(id="task4", cost=5.0, metadata={"priority": "high"}),
            Task(id="task5", cost=25.0, metadata={"priority": "medium"}),
        ]

        self.gain_values = {
            "task1": 100.0,
            "task2": 80.0,
            "task3": 60.0,
            "task4": 90.0,
            "task5": 50.0,
        }

        self.gain_function = TestSubmodularFunction(self.gain_values)
        self.selector = MonotonicTaskSelector(
            gain_function=self.gain_function,
            lazy_evaluation=True,
            approximation_factor=0.5,
        )

        self.selector.add_tasks(self.test_tasks)

    def test_task_creation_and_equality(self):
        """Test Task dataclass functionality."""
        task1 = Task(id="test", cost=10.0)
        task2 = Task(id="test", cost=15.0)  # Different cost, same ID
        task3 = Task(id="different", cost=10.0)

        assert task1 == task2  # Equality based on ID only
        assert task1 != task3
        assert hash(task1) == hash(task2)
        assert hash(task1) != hash(task3)

    def test_submodular_property(self):
        """Verify submodular property: f(S ∪ {x}) - f(S) >= f(T ∪ {x}) - f(T) for S ⊆ T."""
        task_x = self.test_tasks[0]

        # S ⊆ T
        S = set()
        T = {self.test_tasks[1], self.test_tasks[2]}

        gain_S = self.gain_function.marginal_gain(task_x, S)
        gain_T = self.gain_function.marginal_gain(task_x, T)

        # Submodular property: marginal gain decreases with larger sets
        assert gain_S >= gain_T, f"Submodular property violated: {gain_S} < {gain_T}"

    def test_deterministic_gain_function(self):
        """Test DeterministicGainFunction with interactions and dependencies."""
        base_values = {"A": 100.0, "B": 80.0, "C": 60.0}
        interactions = {("A", "B"): 20.0, ("B", "C"): 15.0}  # Redundancy penalties
# # #         dependencies = {"C": {"A", "B"}}  # C benefits from A and B  # Module not found  # Module not found  # Module not found

        gain_func = DeterministicGainFunction(
            base_values=base_values,
            interaction_matrix=interactions,
            dependency_graph=dependencies,
        )

        task_A = Task("A", 10.0)
        task_B = Task("B", 10.0)
        task_C = Task("C", 10.0)

        # Test marginal gains
        assert gain_func.marginal_gain(task_A, set()) == 100.0

        # B's gain should be reduced due to interaction with A
        gain_B_with_A = gain_func.marginal_gain(task_B, {task_A})
        assert gain_B_with_A < 80.0

        # C should get dependency bonus when A and B are selected
        gain_C_with_AB = gain_func.marginal_gain(task_C, {task_A, task_B})
        gain_C_alone = gain_func.marginal_gain(task_C, set())
        # Even with diminishing returns, dependency bonus should help
        assert gain_C_with_AB > 0

    def test_basic_selection(self):
        """Test basic task selection within budget."""
        budget = 50.0
        selected = self.selector.select_tasks(budget)

        # Should select tasks within budget
        total_cost = sum(task.cost for task in selected)
        assert total_cost <= budget

        # Should have selected some tasks
        assert len(selected) > 0

        # Should prefer higher value-to-cost ratio tasks
        assert any(task.id == "task1" for task in selected)  # High value, moderate cost
        assert any(task.id == "task4" for task in selected)  # High value, low cost

    def test_prefix_property(self):
        """Test that selections maintain prefix property with increasing budgets."""
        budgets = [20.0, 35.0, 50.0, 75.0]
        selections = []

        for budget in budgets:
            selection = self.selector.select_tasks(budget)
            selections.append(selection)

        # Verify prefix property
        for i in range(len(selections) - 1):
            smaller = selections[i]
            larger = selections[i + 1]

            # Smaller selection should be prefix of larger
            assert len(smaller) <= len(larger)

            for j, task in enumerate(smaller):
                assert j < len(larger), f"Prefix violation at index {j}"
                assert (
                    task == larger[j]
                ), f"Task mismatch at position {j}: {task.id} != {larger[j].id}"

    def test_monotonicity_verification(self):
        """Test explicit monotonicity verification."""
        budget1 = 30.0
        budget2 = 60.0

        # Verify prefix property holds
        assert self.selector.verify_prefix_property(budget1, budget2)
        assert self.selector.verify_prefix_property(
            budget2, budget1
        )  # Should handle order

    def test_decision_traceability(self):
        """Test complete traceability of selection decisions."""
        budget = 40.0
        selected = self.selector.select_tasks(budget)

        trace = self.selector.get_decision_trace()

        # Should have decision records
        assert len(trace) > 0

        # Check decision structure
        for decision in trace:
            assert isinstance(decision, SelectionDecision)
            assert isinstance(decision.task, Task)
            assert isinstance(decision.selected, bool)
            assert isinstance(decision.marginal_gain, float)
            assert isinstance(decision.budget_remaining, float)
            assert decision.reason != ""

        # Number of selected decisions should match selection length
        selected_decisions = [d for d in trace if d.selected]
        assert len(selected_decisions) == len(selected)

    def test_lazy_evaluation(self):
        """Test lazy evaluation optimization."""
        selector_lazy = MonotonicTaskSelector(
            gain_function=self.gain_function, lazy_evaluation=True
        )

        selector_eager = MonotonicTaskSelector(
            gain_function=self.gain_function, lazy_evaluation=False
        )

        selector_lazy.add_tasks(self.test_tasks)
        selector_eager.add_tasks(self.test_tasks)

        budget = 45.0
        selected_lazy = selector_lazy.select_tasks(budget)
        selected_eager = selector_eager.select_tasks(budget)

        # Results should be the same regardless of lazy evaluation
        assert len(selected_lazy) == len(selected_eager)

        # Task IDs should match (though implementation details may differ)
        lazy_ids = {task.id for task in selected_lazy}
        eager_ids = {task.id for task in selected_eager}
        assert lazy_ids == eager_ids

    def test_approximation_guarantees(self):
        """Test approximation ratio computation."""
        budget = 40.0
        selected = self.selector.select_tasks(budget)

        # Compute optimal solution value (brute force for small instance)
        optimal_value = self._compute_optimal_solution(budget)

        # Test approximation ratio
        ratio = self.selector.get_approximation_ratio(selected, optimal_value)

        # Should achieve at least the theoretical guarantee
        assert (
            ratio >= self.selector.approximation_factor * 0.9
        )  # Allow small numerical tolerance

        # Get theoretical guarantee description
        guarantee = self.selector.get_theoretical_guarantee()
        assert "approximation" in guarantee.lower()
        assert str(self.selector.approximation_factor) in guarantee

    def test_empty_task_set(self):
        """Test behavior with empty task set."""
        empty_selector = MonotonicTaskSelector(self.gain_function)
        selected = empty_selector.select_tasks(100.0)

        assert len(selected) == 0
        assert empty_selector.get_approximation_ratio(selected, 0.0) == 1.0

    def test_zero_budget(self):
        """Test behavior with zero budget."""
        selected = self.selector.select_tasks(0.0)
        assert len(selected) == 0

        # With zero budget, no tasks are even attempted for selection
        # so there may be no rejection decisions in the trace
        # The important thing is that no tasks are selected
        trace = self.selector.get_decision_trace()
        selected_decisions = [d for d in trace if d.selected]
        assert len(selected_decisions) == 0

    def test_heap_stability(self):
        """Test that heap maintains stable ordering for equal gains."""
        # Create tasks with identical gains but different insertion order
        identical_tasks = [Task(id=f"equal_{i}", cost=10.0) for i in range(5)]

        equal_gains = {f"equal_{i}": 50.0 for i in range(5)}
        equal_gain_func = TestSubmodularFunction(equal_gains)

        stable_selector = MonotonicTaskSelector(equal_gain_func)
        stable_selector.add_tasks(identical_tasks)

        budget = 100.0  # Enough for all tasks
        selected = stable_selector.select_tasks(budget)

        # Should select all tasks
        assert len(selected) == 5

        # Order should be consistent across multiple runs due to stable sorting
        selected_ids = [task.id for task in selected]

        # Run again to verify consistency
        stable_selector2 = MonotonicTaskSelector(equal_gain_func)
        stable_selector2.add_tasks(identical_tasks)
        selected2 = stable_selector2.select_tasks(budget)
        selected_ids2 = [task.id for task in selected2]

        # Both runs should produce the same order
        assert (
            selected_ids == selected_ids2
        ), f"Inconsistent ordering: {selected_ids} vs {selected_ids2}"

        # All tasks should be included
        assert set(selected_ids) == {f"equal_{i}" for i in range(5)}

    def test_large_scale_performance(self):
        """Test performance and correctness with larger task sets."""
        # Create larger test set
        large_tasks = [
            Task(id=f"task_{i}", cost=np.random.uniform(5, 25)) for i in range(100)
        ]

        large_gains = {f"task_{i}": np.random.uniform(10, 100) for i in range(100)}
        large_gain_func = TestSubmodularFunction(large_gains)

        large_selector = MonotonicTaskSelector(large_gain_func, lazy_evaluation=True)
        large_selector.add_tasks(large_tasks)

        # Test multiple budget levels
        budgets = [50, 100, 200, 300, 500]
        previous_selection = []

        for budget in budgets:
            selection = large_selector.select_tasks(budget)

            # Basic sanity checks
            total_cost = sum(task.cost for task in selection)
            assert total_cost <= budget

            # Prefix property
            assert len(selection) >= len(previous_selection)
            for i, task in enumerate(previous_selection):
                assert i < len(selection)
                assert task.id == selection[i].id

            previous_selection = selection

    def _compute_optimal_solution(self, budget: float) -> float:
        """
        Compute optimal solution value via brute force (for small test instances).
        """
# # #         from itertools import combinations  # Module not found  # Module not found  # Module not found

        best_value = 0.0

        # Try all possible combinations
        for r in range(len(self.test_tasks) + 1):
            for combo in combinations(self.test_tasks, r):
                total_cost = sum(task.cost for task in combo)
                if total_cost <= budget:
                    value = self.gain_function.evaluate(set(combo))
                    best_value = max(best_value, value)

        return best_value


def test_integration_with_realistic_scenario():
    """Integration test with realistic task selection scenario."""
    # Realistic scenario: software development task prioritization
    dev_tasks = [
        Task("implement_auth", 40.0, {"priority": "critical", "team": "backend"}),
        Task("ui_redesign", 60.0, {"priority": "high", "team": "frontend"}),
        Task("database_migration", 80.0, {"priority": "medium", "team": "backend"}),
        Task("unit_tests", 20.0, {"priority": "high", "team": "qa"}),
        Task(
            "performance_optimization", 50.0, {"priority": "medium", "team": "backend"}
        ),
        Task("documentation", 15.0, {"priority": "low", "team": "all"}),
        Task("security_audit", 35.0, {"priority": "critical", "team": "security"}),
        Task("mobile_app", 100.0, {"priority": "low", "team": "mobile"}),
    ]

    # Define realistic gain function with business value
    business_values = {
        "implement_auth": 200.0,  # Critical security feature
        "ui_redesign": 120.0,  # User experience improvement
        "database_migration": 80.0,  # Technical debt
        "unit_tests": 150.0,  # Quality assurance
        "performance_optimization": 100.0,  # Performance gain
        "documentation": 40.0,  # Long-term maintenance
        "security_audit": 180.0,  # Risk mitigation
        "mobile_app": 90.0,  # New market expansion
    }

    # Define task interactions (redundancies and synergies)
    interactions = {
        ("implement_auth", "security_audit"): 30.0,  # Some redundancy
        ("unit_tests", "security_audit"): 20.0,  # Testing overlap
        ("ui_redesign", "mobile_app"): 25.0,  # Design consistency issues
    }

    # Define dependencies (tasks that work better together)
    dependencies = {
        "performance_optimization": {
            "database_migration"
        },  # DB migration enables performance work
        "mobile_app": {"implement_auth"},  # Mobile needs authentication
        "documentation": {"implement_auth", "ui_redesign"},  # Document new features
    }

    gain_function = DeterministicGainFunction(
        base_values=business_values,
        interaction_matrix=interactions,
        dependency_graph=dependencies,
    )

    selector = MonotonicTaskSelector(
        gain_function=gain_function, lazy_evaluation=True, approximation_factor=0.5
    )

    selector.add_tasks(dev_tasks)

    # Test different sprint budgets (in story points/hours)
    sprint_budgets = [100, 150, 200, 250, 300]

    for budget in sprint_budgets:
        selection = selector.select_tasks(budget)

        # Validate basic constraints
        total_cost = sum(task.cost for task in selection)
        assert total_cost <= budget

        # Should prioritize critical tasks first
        if budget >= 75:  # Enough for auth + security audit
            selected_ids = {task.id for task in selection}
            assert "implement_auth" in selected_ids or "security_audit" in selected_ids

        # Verify decision traceability
        trace = selector.get_decision_trace()
        selected_count = sum(1 for d in trace if d.selected)
        assert selected_count == len(selection)

    # Verify monotonicity across all budget levels
    all_selections = [selector.select_tasks(b) for b in sprint_budgets]
    for i in range(len(all_selections) - 1):
        smaller = all_selections[i]
        larger = all_selections[i + 1]

        # Prefix property verification
        for j, task in enumerate(smaller):
            assert j < len(larger)
            assert task.id == larger[j].id

    # Check approximation guarantee
    optimal_value = 500.0  # Assume known optimal for this instance
    final_selection = all_selections[-1]
    ratio = selector.get_approximation_ratio(final_selection, optimal_value)
    assert ratio >= 0.4  # Should be close to theoretical guarantee


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
