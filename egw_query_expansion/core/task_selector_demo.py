"""
Demonstration of Monotonic Task Selector with Submodular Approximation

This demo showcases the key features of the task selector including:
- Monotonic selection under increasing budgets
- Submodular approximation guarantees
- Lazy evaluation for efficiency
- Dependency constraint handling
- Comprehensive tracing and statistics
"""

# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Set  # Module not found  # Module not found  # Module not found

import matplotlib.pyplot as plt
import numpy as np

# # # from egw_query_expansion.core.submodular_task_selector import (  # Module not found  # Module not found  # Module not found
    CoverageUtility,
    MonotonicTaskSelector,
    SubmodularFunction,
    Task,
)


class ProjectUtility(SubmodularFunction):
    """
    Project-based utility function modeling software development tasks.

    Models diminishing returns as more tasks of similar type are selected,
    and synergy effects between complementary tasks.
    """

    def __init__(
        self,
        task_categories: Dict[str, str],
        category_base_values: Dict[str, float],
        synergy_matrix: Dict[tuple, float] = None,
    ):
        """
        Args:
# # #             task_categories: Map from task_id to category  # Module not found  # Module not found  # Module not found
            category_base_values: Base utility values per category
            synergy_matrix: Bonus for category pairs (cat1, cat2) -> bonus
        """
        self.task_categories = task_categories
        self.category_base_values = category_base_values
        self.synergy_matrix = synergy_matrix or {}

    def evaluate(
        self,
        selected_tasks: Set[str],
        candidate_task: str,
        available_tasks: Dict[str, Task],
    ) -> float:
        """Marginal gain with diminishing returns and synergy bonuses."""
        if candidate_task not in self.task_categories:
            return 0.0

        candidate_category = self.task_categories[candidate_task]
        base_value = self.category_base_values.get(candidate_category, 0.0)

        # Count tasks in same category (diminishing returns)
        same_category_count = sum(
            1
            for tid in selected_tasks
            if self.task_categories.get(tid) == candidate_category
        )

        # Diminishing returns: 0.8^n factor
        diminish_factor = 0.8**same_category_count
        diminished_value = base_value * diminish_factor

        # Add synergy bonuses
        synergy_bonus = 0.0
        for selected_task in selected_tasks:
            selected_category = self.task_categories.get(selected_task)
            if selected_category:
                # Check both directions for synergy
                synergy_key1 = (candidate_category, selected_category)
                synergy_key2 = (selected_category, candidate_category)

                synergy_bonus += self.synergy_matrix.get(synergy_key1, 0.0)
                synergy_bonus += self.synergy_matrix.get(synergy_key2, 0.0)

        return diminished_value + synergy_bonus

    def evaluate_set(
        self, selected_tasks: Set[str], available_tasks: Dict[str, Task]
    ) -> float:
        """Total utility of selected task set."""
        total = 0.0
        cumulative_selected = set()

        for task_id in selected_tasks:
            marginal = self.evaluate(cumulative_selected, task_id, available_tasks)
            total += marginal
            cumulative_selected.add(task_id)

        return total


def create_software_project_scenario():
    """Create a realistic software project task scenario."""

    # Define tasks with dependencies and categories
    tasks = {
        # Infrastructure
        "setup_repo": Task("setup_repo", cost=2.0),
        "ci_pipeline": Task("ci_pipeline", cost=4.0, dependencies={"setup_repo"}),
        "docker_config": Task("docker_config", cost=3.0, dependencies={"setup_repo"}),
        # Backend Development
        "api_design": Task("api_design", cost=5.0),
        "database_schema": Task("database_schema", cost=4.0),
        "auth_service": Task(
            "auth_service", cost=8.0, dependencies={"api_design", "database_schema"}
        ),
        "user_service": Task("user_service", cost=6.0, dependencies={"auth_service"}),
        "notification_service": Task(
            "notification_service", cost=5.0, dependencies={"api_design"}
        ),
        # Frontend Development
        "ui_components": Task("ui_components", cost=7.0),
        "login_page": Task("login_page", cost=3.0, dependencies={"ui_components"}),
        "dashboard": Task(
            "dashboard", cost=6.0, dependencies={"ui_components", "user_service"}
        ),
        "notifications_ui": Task(
            "notifications_ui",
            cost=4.0,
            dependencies={"ui_components", "notification_service"},
        ),
        # Testing
        "unit_tests": Task("unit_tests", cost=5.0),
        "integration_tests": Task(
            "integration_tests", cost=6.0, dependencies={"unit_tests"}
        ),
        "e2e_tests": Task(
            "e2e_tests", cost=8.0, dependencies={"integration_tests", "dashboard"}
        ),
        # Deployment
        "staging_deploy": Task(
            "staging_deploy", cost=3.0, dependencies={"ci_pipeline", "docker_config"}
        ),
        "prod_deploy": Task(
            "prod_deploy", cost=4.0, dependencies={"staging_deploy", "e2e_tests"}
        ),
    }

    # Task categories
    task_categories = {
        "setup_repo": "infrastructure",
        "ci_pipeline": "infrastructure",
        "docker_config": "infrastructure",
        "api_design": "backend",
        "database_schema": "backend",
        "auth_service": "backend",
        "user_service": "backend",
        "notification_service": "backend",
        "ui_components": "frontend",
        "login_page": "frontend",
        "dashboard": "frontend",
        "notifications_ui": "frontend",
        "unit_tests": "testing",
        "integration_tests": "testing",
        "e2e_tests": "testing",
        "staging_deploy": "deployment",
        "prod_deploy": "deployment",
    }

    # Base values per category
    category_base_values = {
        "infrastructure": 15.0,
        "backend": 20.0,
        "frontend": 18.0,
        "testing": 16.0,
        "deployment": 25.0,
    }

    # Synergy matrix (bonus for complementary categories)
    synergy_matrix = {
        ("backend", "frontend"): 3.0,
        ("frontend", "backend"): 3.0,
        ("testing", "backend"): 2.0,
        ("testing", "frontend"): 2.0,
        ("deployment", "testing"): 4.0,
        ("infrastructure", "deployment"): 2.0,
    }

    utility = ProjectUtility(task_categories, category_base_values, synergy_matrix)

    return tasks, utility


def demonstrate_monotonic_selection():
    """Demonstrate monotonic task selection with increasing budgets."""

    print("🚀 Monotonic Task Selector Demo - Software Project")
    print("=" * 60)

    tasks, utility = create_software_project_scenario()
    selector = MonotonicTaskSelector(utility, enable_lazy_evaluation=True)

    # Test with increasing budgets
    budgets = [10, 20, 35, 50, 70, 90]
    results = []

    print("\n📊 Testing Monotonic Selection with Increasing Budgets:")
    print("-" * 60)

    for budget in budgets:
        selected, traces = selector.select_tasks(tasks, budget=budget)

        total_cost = sum(tasks[tid].cost for tid in selected)
        total_utility = utility.evaluate_set(set(selected), tasks)

        results.append(
            {
                "budget": budget,
                "selected": selected.copy(),
                "cost": total_cost,
                "utility": total_utility,
                "count": len(selected),
            }
        )

        print(
            f"Budget ${budget:2d}: Selected {len(selected):2d} tasks, "
            f"Cost ${total_cost:5.1f}, Utility {total_utility:6.1f}"
        )
        print(
            f"           Tasks: {', '.join(selected[:3])}"
            f"{'...' if len(selected) > 3 else ''}"
        )

    # Verify monotonicity
    monotonic_verified = selector.verify_monotonicity()
    print(f"\n✅ Monotonicity Verified: {monotonic_verified}")

    # Show prefix property
    print("\n🔍 Demonstrating Prefix Property:")
    print("-" * 40)
    for i in range(1, len(results)):
        prev_selection = results[i - 1]["selected"]
        curr_selection = results[i]["selected"]

        is_prefix = curr_selection[: len(prev_selection)] == prev_selection
        print(
            f"Budget ${results[i-1]['budget']:2d} -> ${results[i]['budget']:2d}: "
            f"Prefix preserved = {is_prefix}"
        )

    return results, selector


def demonstrate_approximation_guarantees():
    """Demonstrate approximation ratio analysis."""

    print("\n📈 Approximation Ratio Analysis:")
    print("-" * 40)

    tasks, utility = create_software_project_scenario()
    selector = MonotonicTaskSelector(utility)

    # Select with reasonable budget
    selected, traces = selector.select_tasks(tasks, budget=60.0)

    # Get statistics
    stats = selector.get_selection_statistics()

    print(f"Tasks Considered: {stats['total_tasks_considered']}")
    print(f"Successful Selections: {stats['successful_selections']}")
    print(f"Selection Efficiency: {stats['selection_efficiency']:.3f}")
    print(f"Final Utility: {stats['final_utility']:.1f}")

    # Approximation ratio (theoretical)
    approx_ratio = selector.get_approximation_ratio(tasks)
    print(f"Approximation Ratio: {approx_ratio:.3f}")

    # Breakdown rejections
    print(f"\nRejection Breakdown:")
    for reason, count in stats["rejection_breakdown"].items():
        print(f"  {reason}: {count}")

    return stats


def demonstrate_dependency_handling():
    """Demonstrate complex dependency constraint handling."""

    print("\n🔗 Dependency Constraint Handling:")
    print("-" * 40)

    tasks, utility = create_software_project_scenario()
    selector = MonotonicTaskSelector(utility)

    # Select with moderate budget to show dependency ordering
    selected, traces = selector.select_tasks(tasks, budget=45.0)

    print(f"Selected {len(selected)} tasks in dependency-respecting order:")

    for i, task_id in enumerate(selected):
        task = tasks[task_id]
        deps_str = (
            f"[deps: {', '.join(task.dependencies)}]"
            if task.dependencies
            else "[no deps]"
        )
        print(f"  {i+1:2d}. {task_id:<20} (cost: ${task.cost:4.1f}) {deps_str}")

    # Verify dependency satisfaction
    print(f"\n✅ Dependency Verification:")
    valid_dependencies = True

    for i, task_id in enumerate(selected):
        task = tasks[task_id]
        selected_so_far = set(selected[:i])

        missing_deps = task.dependencies - selected_so_far
        if missing_deps:
            print(f"  ❌ {task_id}: Missing dependencies {missing_deps}")
            valid_dependencies = False
        else:
            print(f"  ✅ {task_id}: All dependencies satisfied")

    print(f"\nAll dependencies valid: {valid_dependencies}")

    return selected


def plot_selection_analysis(results):
    """Create visualization of selection analysis."""

    try:
        budgets = [r["budget"] for r in results]
        utilities = [r["utility"] for r in results]
        costs = [r["cost"] for r in results]
        counts = [r["count"] for r in results]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Utility vs Budget
        ax1.plot(budgets, utilities, "bo-", linewidth=2, markersize=6)
        ax1.set_xlabel("Budget ($)")
        ax1.set_ylabel("Total Utility")
        ax1.set_title("Utility vs Budget")
        ax1.grid(True, alpha=0.3)

        # Cost Efficiency
        efficiency = [u / c if c > 0 else 0 for u, c in zip(utilities, costs)]
        ax2.plot(budgets, efficiency, "ro-", linewidth=2, markersize=6)
        ax2.set_xlabel("Budget ($)")
        ax2.set_ylabel("Utility per Dollar")
        ax2.set_title("Cost Efficiency")
        ax2.grid(True, alpha=0.3)

        # Task Count
        ax3.plot(budgets, counts, "go-", linewidth=2, markersize=6)
        ax3.set_xlabel("Budget ($)")
        ax3.set_ylabel("Number of Tasks")
        ax3.set_title("Task Count vs Budget")
        ax3.grid(True, alpha=0.3)

        # Budget Utilization
        utilization = [c / b * 100 for c, b in zip(costs, budgets)]
        ax4.plot(budgets, utilization, "mo-", linewidth=2, markersize=6)
        ax4.set_xlabel("Budget ($)")
        ax4.set_ylabel("Budget Utilization (%)")
        ax4.set_title("Budget Utilization")
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig("task_selection_analysis.png", dpi=150, bbox_inches="tight")
        print(f"\n📊 Analysis plots saved to 'task_selection_analysis.png'")

    except ImportError:
        print(f"\n📊 Matplotlib not available - skipping visualization")


def main():
    """Run complete demonstration of the monotonic task selector."""

    print("🎯 Monotonic Task Selector with Submodular Approximation")
    print("Based on Balkanski et al. (2021) SODA paper")
    print("=" * 80)

    # Core demonstrations
    results, selector = demonstrate_monotonic_selection()
    stats = demonstrate_approximation_guarantees()
    selected = demonstrate_dependency_handling()

    # Performance analysis
    print(f"\n⚡ Performance Summary:")
    print("-" * 30)
    print(f"Lazy evaluation enabled: {selector.enable_lazy_evaluation}")
    print(f"Approximation tolerance: {selector.approximation_tolerance}")
    print(f"Total budget increases tested: {len(results)}")
    print(f"Monotonicity maintained: {selector.verify_monotonicity()}")

    # Generate visualization
    plot_selection_analysis(results)

    print(f"\n🎉 Demo completed successfully!")
    print(f"Key features demonstrated:")
    print(f"  ✅ Monotonic selection (prefix property)")
    print(f"  ✅ Submodular approximation guarantees")
    print(f"  ✅ Lazy evaluation for efficiency")
    print(f"  ✅ Complex dependency handling")
    print(f"  ✅ Comprehensive tracing and statistics")


if __name__ == "__main__":
    main()
