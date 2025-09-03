#!/usr/bin/env python3
"""
Plan Diff Tool

Compares and visualizes inclusion relationships between task selection plans
# # # from different budget scenarios. Shows whether B↑ maintains monotonic inclusion.  # Module not found  # Module not found  # Module not found
"""

import argparse
import json
import sys
# # # from typing import Dict, List, Set, Tuple, Any  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # # from egw_query_expansion.core.submodular_task_selector import (  # Module not found  # Module not found  # Module not found
    MonotonicTaskSelector,
    CoverageUtility,
    Task,
)


@dataclass
class PlanComparison:
    """Comparison result between two plans."""
    budget1: float
    budget2: float
    plan1: List[str]
    plan2: List[str]
    is_subset: bool
    is_prefix: bool
    added_tasks: List[str]
    removed_tasks: List[str]
    common_tasks: List[str]
    inclusion_ratio: float  # |intersection| / |smaller_plan|


def create_test_scenario(num_tasks: int = 15) -> Tuple[Dict[str, Task], CoverageUtility]:
    """Create a test scenario with tasks and utility function."""
    tasks = {}
    feature_coverage = {}
    
    # Create tasks with varying costs
    for i in range(num_tasks):
        task_id = f"task_{i:02d}"
        cost = 2.0 + i * 0.8  # Increasing costs
        tasks[task_id] = Task(task_id, cost=cost, priority=1.0)
        
        # Each task covers overlapping features (submodular structure)
        features = {f"feature_{j}" for j in range(i, min(i + 4, num_tasks + 5))}
        feature_coverage[task_id] = features
    
    # Weight features with diminishing returns
    feature_weights = {
        f"feature_{i}": max(1.0, 8.0 - i * 0.2) 
        for i in range(num_tasks + 5)
    }
    
    utility = CoverageUtility(feature_coverage, feature_weights)
    return tasks, utility


def generate_plans(tasks: Dict[str, Task], utility: CoverageUtility, budgets: List[float]) -> Dict[float, List[str]]:
    """Generate task selection plans for different budgets."""
    selector = MonotonicTaskSelector(utility)
    plans = {}
    
    for budget in budgets:
        selector._reset_state()
        # Force fresh selection without order preservation to test monotonicity
        selected, _ = selector.select_tasks(tasks, budget=budget, preserve_order=False)
        plans[budget] = selected
    
    return plans


def compare_plans(budget1: float, plan1: List[str], budget2: float, plan2: List[str]) -> PlanComparison:
    """Compare two task selection plans."""
    set1 = set(plan1)
    set2 = set(plan2)
    
    common = list(set1 & set2)
    added = [t for t in plan2 if t not in set1]
    removed = [t for t in plan1 if t not in set2]
    
    is_subset = set1.issubset(set2) if budget1 <= budget2 else set2.issubset(set1)
    
    # Check prefix property for monotonic budgets
    is_prefix = False
    if budget1 <= budget2 and len(plan1) <= len(plan2):
        is_prefix = plan2[:len(plan1)] == plan1
    elif budget2 <= budget1 and len(plan2) <= len(plan1):
        is_prefix = plan1[:len(plan2)] == plan2
    
    smaller_size = min(len(plan1), len(plan2))
    inclusion_ratio = len(common) / smaller_size if smaller_size > 0 else 1.0
    
    return PlanComparison(
        budget1=budget1,
        budget2=budget2,
        plan1=plan1,
        plan2=plan2,
        is_subset=is_subset,
        is_prefix=is_prefix,
        added_tasks=added,
        removed_tasks=removed,
        common_tasks=common,
        inclusion_ratio=inclusion_ratio
    )


def print_plan_summary(budget: float, plan: List[str], tasks: Dict[str, Task]):
    """Print summary of a single plan."""
    total_cost = sum(tasks[tid].cost for tid in plan if tid in tasks)
    print(f"Budget {budget:6.1f}: {len(plan):2d} tasks, cost {total_cost:6.2f}")
    print(f"  Tasks: {', '.join(plan[:10])}")
    if len(plan) > 10:
        print(f"         ... and {len(plan) - 10} more")


def print_comparison(comparison: PlanComparison, tasks: Dict[str, Task], verbose: bool = False):
    """Print detailed comparison between two plans."""
    b1, b2 = comparison.budget1, comparison.budget2
    
    print(f"\n{'='*60}")
    print(f"COMPARISON: Budget {b1:.1f} vs Budget {b2:.1f}")
    print(f"{'='*60}")
    
    print_plan_summary(b1, comparison.plan1, tasks)
    print_plan_summary(b2, comparison.plan2, tasks)
    
    print(f"\nRelationship Analysis:")
    print(f"  Subset Property:  {'✓' if comparison.is_subset else '✗'} ")
    print(f"  Prefix Property:  {'✓' if comparison.is_prefix else '✗'}")
    print(f"  Inclusion Ratio:  {comparison.inclusion_ratio:.3f}")
    
    if comparison.added_tasks:
        print(f"  Added Tasks:      {', '.join(comparison.added_tasks)}")
    if comparison.removed_tasks:
        print(f"  Removed Tasks:    {', '.join(comparison.removed_tasks)}")
    
    if verbose:
        print(f"  Common Tasks:     {', '.join(comparison.common_tasks)}")


def analyze_monotonicity_chain(plans: Dict[float, List[str]], tasks: Dict[str, Task]) -> bool:
    """Analyze whether a chain of plans satisfies monotonicity."""
    budgets = sorted(plans.keys())
    
    print(f"\nMONOTONICITY CHAIN ANALYSIS")
    print(f"{'='*60}")
    
    chain_valid = True
    
    for i in range(len(budgets) - 1):
        b1, b2 = budgets[i], budgets[i + 1]
        comparison = compare_plans(b1, plans[b1], b2, plans[b2])
        
        if not comparison.is_subset or not comparison.is_prefix:
            chain_valid = False
            print(f"VIOLATION: Budget {b1:.1f} → {b2:.1f}")
            print(f"  Subset: {'✓' if comparison.is_subset else '✗'}")
            print(f"  Prefix: {'✓' if comparison.is_prefix else '✗'}")
        else:
            print(f"OK: Budget {b1:.1f} → {b2:.1f} (added {len(comparison.added_tasks)} tasks)")
    
    print(f"\nChain Status: {'✓ VALID' if chain_valid else '✗ INVALID'}")
    return chain_valid


def generate_inclusion_matrix(plans: Dict[float, List[str]]) -> Dict[Tuple[float, float], PlanComparison]:
    """Generate pairwise inclusion analysis matrix."""
    budgets = sorted(plans.keys())
    matrix = {}
    
    for i, b1 in enumerate(budgets):
        for j, b2 in enumerate(budgets):
            if i != j:  # Skip self-comparison
                comparison = compare_plans(b1, plans[b1], b2, plans[b2])
                matrix[(b1, b2)] = comparison
    
    return matrix


def print_inclusion_matrix(matrix: Dict[Tuple[float, float], PlanComparison]):
    """Print inclusion relationship matrix."""
    budgets = sorted(set(b1 for b1, b2 in matrix.keys()))
    
    print(f"\nINCLUSION MATRIX (Subset ✓/✗, Prefix ✓/✗)")
    print(f"{'='*60}")
    print(f"{'Budget':<8} ", end="")
    for b in budgets:
        print(f"{b:>8.1f} ", end="")
    print()
    
    for b1 in budgets:
        print(f"{b1:<8.1f} ", end="")
        for b2 in budgets:
            if b1 == b2:
                print("    --   ", end="")
            else:
                comp = matrix[(b1, b2)]
                subset_sym = "✓" if comp.is_subset else "✗"
                prefix_sym = "✓" if comp.is_prefix else "✗"
                print(f"  {subset_sym}{prefix_sym}    ", end="")
        print()


def main():
    """Main entry point for plan diff tool."""
    parser = argparse.ArgumentParser(description="Analyze task selection plan inclusion relationships")
    parser.add_argument("--budgets", nargs="+", type=float, 
                       default=[10.0, 20.0, 35.0, 50.0, 75.0],
                       help="Budget values to test")
    parser.add_argument("--tasks", type=int, default=15,
                       help="Number of tasks in test scenario")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output with detailed comparisons")
    parser.add_argument("--output", "-o", type=str,
                       help="Save results to JSON file")
    parser.add_argument("--matrix", action="store_true",
                       help="Show inclusion matrix")
    
    args = parser.parse_args()
    
    print("Plan Diff Tool - BMC Property Analysis")
    print("="*60)
    
    # Create test scenario
    print(f"Creating scenario with {args.tasks} tasks...")
    tasks, utility = create_test_scenario(args.tasks)
    
    # Generate plans
    print(f"Generating plans for budgets: {args.budgets}")
    plans = generate_plans(tasks, utility, args.budgets)
    
    # Show individual plans
    if args.verbose:
        print(f"\nINDIVIDUAL PLANS")
        print("="*60)
        for budget in sorted(args.budgets):
            print_plan_summary(budget, plans[budget], tasks)
    
    # Analyze monotonicity chain
    chain_valid = analyze_monotonicity_chain(plans, tasks)
    
    # Detailed pairwise comparisons
    if args.verbose:
        budgets = sorted(args.budgets)
        print(f"\nDETAILED PAIRWISE COMPARISONS")
        for i in range(len(budgets) - 1):
            b1, b2 = budgets[i], budgets[i + 1]
            comparison = compare_plans(b1, plans[b1], b2, plans[b2])
            print_comparison(comparison, tasks, verbose=True)
    
    # Show inclusion matrix
    if args.matrix:
        matrix = generate_inclusion_matrix(plans)
        print_inclusion_matrix(matrix)
    
    # Summary
    print(f"\nSUMMARY")
    print("="*60)
    total_comparisons = len(args.budgets) - 1
    print(f"Budget Chain:     {len(args.budgets)} budgets")
    print(f"Chain Monotonic:  {'✓ YES' if chain_valid else '✗ NO'}")
    print(f"Tasks Available:  {len(tasks)}")
    
    # Calculate selection efficiency
    max_budget = max(args.budgets)
    max_selection_size = len(plans[max_budget])
    efficiency = max_selection_size / len(tasks)
    print(f"Selection Rate:   {max_selection_size}/{len(tasks)} ({efficiency:.1%})")
    
    # Save results if requested
    if args.output:
        results = {
            "budgets": args.budgets,
            "plans": {str(b): plan for b, plan in plans.items()},
            "chain_valid": chain_valid,
            "task_count": len(tasks),
            "summary": {
                "max_selection_size": max_selection_size,
                "selection_efficiency": efficiency,
                "monotonic_chain": chain_valid
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()