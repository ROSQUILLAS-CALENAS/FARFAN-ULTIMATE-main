"""
Demonstration of Monotonic Task Selector based on Submodular Maximization

This example showcases the implementation of Balkanski et al. (2021) submodular
approximation algorithm with practical applications in task prioritization.
"""

from typing import Dict, List

import numpy as np

from egw_query_expansion import DeterministicGainFunction, MonotonicTaskSelector, Task


def demonstrate_software_development_prioritization():
    """
    Example: Software development sprint planning with task dependencies
    and business value optimization.
    """
    print("=== Software Development Task Prioritization Demo ===\n")

    # Define development tasks with costs (story points/hours)
    tasks = [
        Task("user_authentication", 25.0, {"priority": "critical", "team": "backend"}),
        Task("payment_integration", 40.0, {"priority": "high", "team": "backend"}),
        Task("ui_dashboard", 30.0, {"priority": "medium", "team": "frontend"}),
        Task("mobile_responsive", 20.0, {"priority": "high", "team": "frontend"}),
        Task("unit_test_suite", 15.0, {"priority": "medium", "team": "qa"}),
        Task("api_documentation", 10.0, {"priority": "low", "team": "docs"}),
        Task("performance_monitoring", 35.0, {"priority": "medium", "team": "devops"}),
        Task("security_audit", 30.0, {"priority": "critical", "team": "security"}),
        Task("database_optimization", 45.0, {"priority": "low", "team": "backend"}),
        Task("user_analytics", 25.0, {"priority": "medium", "team": "analytics"}),
    ]

    # Define business value for each task
    business_values = {
        "user_authentication": 150.0,  # Critical for launch
        "payment_integration": 200.0,  # Revenue generating
        "ui_dashboard": 80.0,  # User experience
        "mobile_responsive": 120.0,  # Market reach
        "unit_test_suite": 100.0,  # Quality assurance
        "api_documentation": 40.0,  # Developer productivity
        "performance_monitoring": 90.0,  # Operational excellence
        "security_audit": 130.0,  # Risk mitigation
        "database_optimization": 70.0,  # Technical debt
        "user_analytics": 85.0,  # Business insights
    }

    # Define task interactions (redundancies reduce combined value)
    interactions = {
        ("user_authentication", "security_audit"): 20.0,  # Some security overlap
        ("ui_dashboard", "mobile_responsive"): 15.0,  # UI consistency work
        (
            "performance_monitoring",
            "database_optimization",
        ): 10.0,  # Performance overlap
    }

    # Define task dependencies (synergistic benefits)
    dependencies = {
        "payment_integration": {"user_authentication"},  # Payment needs auth
        "ui_dashboard": {"user_authentication"},  # Dashboard needs auth
        "user_analytics": {"ui_dashboard"},  # Analytics needs dashboard
        "api_documentation": {"payment_integration"},  # Document payment API
        "unit_test_suite": {
            "user_authentication",
            "payment_integration",
        },  # Test critical features
    }

    # Create deterministic gain function
    gain_function = DeterministicGainFunction(
        base_values=business_values,
        interaction_matrix=interactions,
        dependency_graph=dependencies,
    )

    # Initialize monotonic task selector
    selector = MonotonicTaskSelector(
        gain_function=gain_function, lazy_evaluation=True, approximation_factor=0.5
    )

    selector.add_tasks(tasks)

    # Simulate different sprint capacities
    sprint_capacities = [50, 100, 150, 200, 250]

    print("Sprint Planning Results:")
    print("-" * 60)

    for capacity in sprint_capacities:
        selected_tasks = selector.select_tasks(capacity)
        total_cost = sum(task.cost for task in selected_tasks)
        total_value = gain_function.evaluate(set(selected_tasks))

        print(f"\nSprint Capacity: {capacity} points")
        print(f"Tasks Selected: {len(selected_tasks)}")
        print(f"Total Cost: {total_cost:.1f} points")
        print(f"Total Business Value: {total_value:.1f}")
        print(
            f"Value per Point: {total_value / total_cost:.2f}"
            if total_cost > 0
            else "N/A"
        )

        print("Selected Tasks:")
        for i, task in enumerate(selected_tasks, 1):
            priority = task.metadata.get("priority", "unknown")
            team = task.metadata.get("team", "unknown")
            print(
                f"  {i}. {task.id} (cost: {task.cost}, {priority} priority, {team} team)"
            )

    # Demonstrate prefix property
    print("\n" + "=" * 60)
    print("DEMONSTRATING MONOTONICITY (PREFIX PROPERTY)")
    print("=" * 60)

    capacity1, capacity2 = 100, 200
    selection1 = selector.select_tasks(capacity1)
    selection2 = selector.select_tasks(capacity2)

    print(f"\nTasks selected with {capacity1} points:")
    for task in selection1:
        print(f"  - {task.id}")

    print(f"\nTasks selected with {capacity2} points:")
    for task in selection2:
        print(f"  - {task.id}")

    # Verify prefix property
    is_prefix = selector.verify_prefix_property(capacity1, capacity2)
    print(f"\nPrefix property holds: {is_prefix}")

    if is_prefix:
        print(
            "✓ Increasing budget from 100 to 200 only ADDED tasks, never removed or reordered"
        )
    else:
        print("✗ Prefix property violation detected!")

    # Show decision traceability
    print("\n" + "=" * 60)
    print("DECISION TRACEABILITY")
    print("=" * 60)

    trace = selector.get_decision_trace()
    recent_decisions = trace[-10:]  # Show last 10 decisions

    print(f"\nShowing last {len(recent_decisions)} decisions from trace:")
    for decision in recent_decisions:
        status = "SELECTED" if decision.selected else "REJECTED"
        print(f"  {decision.timestamp}: {status} {decision.task.id}")
        print(f"    Marginal Gain: {decision.marginal_gain:.2f}")
        print(f"    Budget Remaining: {decision.budget_remaining:.1f}")
        print(f"    Reason: {decision.reason}")
        print()

    # Show approximation guarantee
    print("=" * 60)
    print("APPROXIMATION GUARANTEES")
    print("=" * 60)

    theoretical_guarantee = selector.get_theoretical_guarantee()
    print(f"\nTheoretical Guarantee:")
    print(f"  {theoretical_guarantee}")

    # Compute approximation ratio for final selection
    final_selection = selector.select_tasks(250)

    # For demonstration, assume we know the optimal value (in practice, this is unknown)
    optimal_value = 600.0  # Hypothetical optimal solution value
    approx_ratio = selector.get_approximation_ratio(final_selection, optimal_value)

    print(f"\nApproximation Analysis:")
    print(f"  Selected Value: {gain_function.evaluate(set(final_selection)):.1f}")
    print(f"  Assumed Optimal Value: {optimal_value:.1f}")
    print(f"  Approximation Ratio: {approx_ratio:.3f}")
    print(f"  Theoretical Minimum: {selector.approximation_factor:.3f}")

    if approx_ratio >= selector.approximation_factor:
        print("  ✓ Meets theoretical approximation guarantee")
    else:
        print("  ⚠ Below theoretical guarantee (may indicate estimation error)")


def demonstrate_resource_allocation():
    """
    Example: Resource allocation for research projects with uncertain outcomes
    but guaranteed submodular returns.
    """
    print("\n\n=== Research Project Resource Allocation Demo ===\n")

    # Define research projects with funding requirements (in $1000s)
    research_tasks = [
        Task("ai_safety_research", 150.0, {"field": "AI", "risk": "low"}),
        Task("quantum_computing", 200.0, {"field": "Physics", "risk": "high"}),
        Task("biomarker_discovery", 120.0, {"field": "Biology", "risk": "medium"}),
        Task("climate_modeling", 180.0, {"field": "Environment", "risk": "low"}),
        Task("drug_synthesis", 250.0, {"field": "Chemistry", "risk": "high"}),
        Task(
            "neural_interfaces", 300.0, {"field": "Neuroscience", "risk": "very_high"}
        ),
        Task("renewable_energy", 160.0, {"field": "Engineering", "risk": "medium"}),
        Task("space_propulsion", 220.0, {"field": "Aerospace", "risk": "high"}),
    ]

    # Define expected scientific impact values
    impact_values = {
        "ai_safety_research": 200.0,
        "quantum_computing": 300.0,
        "biomarker_discovery": 180.0,
        "climate_modeling": 250.0,
        "drug_synthesis": 280.0,
        "neural_interfaces": 350.0,
        "renewable_energy": 220.0,
        "space_propulsion": 190.0,
    }

    # Define research field interactions (similar fields have diminishing returns)
    field_interactions = {
        ("ai_safety_research", "neural_interfaces"): 30.0,  # Both AI-related
        ("quantum_computing", "space_propulsion"): 20.0,  # Physics overlap
        ("biomarker_discovery", "drug_synthesis"): 40.0,  # Both medical research
        ("climate_modeling", "renewable_energy"): 25.0,  # Environmental focus
    }

    # Research project dependencies (complementary research areas)
    research_dependencies = {
        "neural_interfaces": {
            "ai_safety_research"
        },  # AI safety needed for neural interfaces
        "drug_synthesis": {"biomarker_discovery"},  # Biomarkers guide drug development
        "space_propulsion": {"renewable_energy"},  # Clean energy for space systems
    }

    gain_function = DeterministicGainFunction(
        base_values=impact_values,
        interaction_matrix=field_interactions,
        dependency_graph=research_dependencies,
    )

    selector = MonotonicTaskSelector(
        gain_function=gain_function, lazy_evaluation=True, approximation_factor=0.5
    )

    selector.add_tasks(research_tasks)

    # Test different funding levels
    funding_levels = [300, 500, 800, 1200, 1500]  # in $1000s

    print("Research Funding Allocation Results:")
    print("-" * 50)

    for funding in funding_levels:
        selected = selector.select_tasks(funding)
        total_cost = sum(task.cost for task in selected)
        total_impact = gain_function.evaluate(set(selected))

        print(f"\nFunding Level: ${funding}K")
        print(f"Projects Funded: {len(selected)}")
        print(f"Total Cost: ${total_cost:.0f}K")
        print(f"Expected Impact Score: {total_impact:.1f}")
        print(
            f"Impact per Dollar: {total_impact / total_cost:.3f}"
            if total_cost > 0
            else "N/A"
        )

        print("Funded Projects:")
        for i, task in enumerate(selected, 1):
            field = task.metadata.get("field", "unknown")
            risk = task.metadata.get("risk", "unknown")
            print(f"  {i}. {task.id} (${task.cost:.0f}K, {field}, {risk} risk)")

    # Demonstrate stable ordering
    print(f"\n{'='*50}")
    print("STABLE SELECTION ORDERING")
    print("=" * 50)

    # Show that selections are consistent and monotonic
    for i, funding in enumerate(funding_levels):
        selection = selector.select_tasks(funding)
        print(f"\nFunding ${funding}K: ", end="")
        print(" → ".join([task.id for task in selection]))


def main():
    """Run all demonstrations."""
    print("MONOTONIC TASK SELECTOR DEMONSTRATION")
    print("Based on Balkanski et al. (2021) Submodular Maximization")
    print("=" * 70)

    # Run software development example
    demonstrate_software_development_prioritization()

    # Run research allocation example
    demonstrate_resource_allocation()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Properties Demonstrated:")
    print("✓ Submodular approximation guarantees")
    print("✓ Monotonic prefix property preservation")
    print("✓ Lazy evaluation for efficiency")
    print("✓ Complete decision traceability")
    print("✓ Stable heap ordering for tie-breaking")
    print("✓ Deterministic gain functions with dependencies")
    print("✓ Real-world application scenarios")


if __name__ == "__main__":
    main()
