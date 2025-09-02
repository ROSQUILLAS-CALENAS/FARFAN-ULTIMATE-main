#!/usr/bin/env python3
"""
Basic functionality test for the Monotonic Task Selector.
This test validates core functionality without external dependencies.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from egw_query_expansion.core.submodular_task_selector import (
    CoverageUtility,
    HeapItem,
    MonotonicTaskSelector,
    Task,
)


def test_task_creation():
    """Test basic Task creation."""
    print("Testing Task creation...")

    # Valid task
    task = Task("task1", cost=10.0, priority=1.5)
    assert task.task_id == "task1"
    assert task.cost == 10.0
    assert task.priority == 1.5
    print("✅ Task creation successful")

    # Task with dependencies
    task_with_deps = Task("task2", cost=5.0, dependencies={"task1"})
    assert task_with_deps.dependencies == {"task1"}
    print("✅ Task with dependencies successful")


def test_coverage_utility():
    """Test CoverageUtility function."""
    print("\nTesting CoverageUtility...")

    feature_coverage = {
        "task1": {"f1", "f2"},
        "task2": {"f2", "f3"},
        "task3": {"f1", "f3", "f4"},
    }

    utility = CoverageUtility(feature_coverage)

    # Test marginal gain
    gain = utility.evaluate(set(), "task1", {})
    assert gain == 2.0  # covers f1, f2
    print(f"✅ Marginal gain calculation: {gain}")

    # Test diminished gain
    gain = utility.evaluate({"task1"}, "task2", {})
    assert gain == 1.0  # only f3 is new
    print(f"✅ Diminished gain calculation: {gain}")


def test_heap_item():
    """Test HeapItem comparison."""
    print("\nTesting HeapItem comparison...")

    item1 = HeapItem(10.0, "task1", 1)
    item2 = HeapItem(5.0, "task2", 2)

    # Higher gain should come first
    assert item1 < item2
    print("✅ HeapItem comparison successful")


def test_basic_selection():
    """Test basic task selection."""
    print("\nTesting basic task selection...")

    # Create simple tasks
    tasks = {
        "task1": Task("task1", cost=5.0),
        "task2": Task("task2", cost=3.0),
        "task3": Task("task3", cost=4.0),
        "task4": Task("task4", cost=2.0),
    }

    # Simple coverage utility
    feature_coverage = {
        "task1": {"f1", "f2"},
        "task2": {"f2", "f3"},
        "task3": {"f3", "f4"},
        "task4": {"f4", "f5"},
    }

    utility = CoverageUtility(feature_coverage)
    selector = MonotonicTaskSelector(utility)

    # Select with budget constraint
    selected, traces = selector.select_tasks(tasks, budget=10.0)

    assert len(selected) > 0
    print(f"✅ Selected {len(selected)} tasks: {selected}")

    # Check budget constraint
    total_cost = sum(tasks[tid].cost for tid in selected)
    assert total_cost <= 10.0
    print(f"✅ Budget constraint satisfied: ${total_cost} <= $10.0")


def test_monotonicity():
    """Test monotonicity property."""
    print("\nTesting monotonicity property...")

    tasks = {f"task{i}": Task(f"task{i}", cost=2.0) for i in range(5)}

    feature_coverage = {f"task{i}": {f"f{i}", f"f{i+1}"} for i in range(5)}

    utility = CoverageUtility(feature_coverage)
    selector = MonotonicTaskSelector(utility)

    # First selection
    selected1, _ = selector.select_tasks(tasks, budget=6.0)

    # Increase budget
    selected2, _ = selector.select_tasks(tasks, budget=10.0)

    # Check monotonicity
    assert len(selected2) >= len(selected1)
    assert selected2[: len(selected1)] == selected1
    print(f"✅ Monotonicity verified: {selected1} -> {selected2}")

    # Internal monotonicity check
    assert selector.verify_monotonicity()
    print("✅ Internal monotonicity verification successful")


def test_dependency_handling():
    """Test dependency constraint handling."""
    print("\nTesting dependency handling...")

    tasks = {
        "task1": Task("task1", cost=2.0),
        "task2": Task("task2", cost=3.0, dependencies={"task1"}),
        "task3": Task("task3", cost=4.0, dependencies={"task1", "task2"}),
    }

    feature_coverage = {
        "task1": {"f1"},
        "task2": {"f2"},
        "task3": {"f3", "f4", "f5"},  # High value
    }

    utility = CoverageUtility(feature_coverage)
    selector = MonotonicTaskSelector(utility)

    selected, traces = selector.select_tasks(tasks, budget=10.0)

    # Verify dependency order
    for task_id in selected:
        task = tasks[task_id]
        selected_index = selected.index(task_id)
        selected_so_far = set(selected[:selected_index])

        missing_deps = task.dependencies - selected_so_far
        assert len(missing_deps) == 0, f"Task {task_id} missing deps: {missing_deps}"

    print(f"✅ Dependency order verified for selection: {selected}")


def main():
    """Run all basic functionality tests."""
    print("🧪 Running Basic Functionality Tests")
    print("=" * 50)

    try:
        test_task_creation()
        test_coverage_utility()
        test_heap_item()
        test_basic_selection()
        test_monotonicity()
        test_dependency_handling()

        print("\n🎉 All tests passed successfully!")
        print("\n✅ Key functionality verified:")
        print("  - Task creation and validation")
        print("  - Submodular utility functions")
        print("  - Heap-based selection algorithm")
        print("  - Budget constraint satisfaction")
        print("  - Monotonicity property")
        print("  - Dependency constraint handling")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
