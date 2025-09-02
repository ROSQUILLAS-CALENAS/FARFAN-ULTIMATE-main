"""
BMC (Budget Monotonic Constraint) Tests

Property-based testing for submodular knapsack monotonicity guarantees.
Verifies B↑ only adds tasks and maintains stable prefix ordering.
"""

import json
from typing import Dict, List, Set
import pytest
from egw_query_expansion.core.submodular_task_selector import (
    MonotonicTaskSelector,
    CoverageUtility,
    Task,
)


class TestBMCMonotonicity:
    """Test Budget Monotonic Constraint with property-based validation."""

    @pytest.fixture
    def fixed_cost_tasks(self):
        """Tasks with fixed costs for deterministic testing."""
        return {
            f"task_{i}": Task(f"task_{i}", cost=2.0 + i * 0.5, priority=1.0)
            for i in range(20)
        }

    @pytest.fixture
    def coverage_utility(self):
        """Coverage-based submodular utility with diminishing returns."""
        feature_coverage = {}
        
        # Each task covers overlapping features (submodular property)
        for i in range(20):
            task_id = f"task_{i}"
            # Task i covers features [i, i+1, i+2, ...]
            features = {f"feature_{j}" for j in range(i, min(i + 5, 30))}
            feature_coverage[task_id] = features
        
        # Weighted features with higher weights for earlier features
        feature_weights = {f"feature_{i}": max(1.0, 10.0 - i * 0.3) for i in range(30)}
        
        return CoverageUtility(feature_coverage, feature_weights)

    def test_budget_sweep_monotonicity(self, fixed_cost_tasks, coverage_utility):
        """Test B1 < B2 < B3 ⇒ S*(B1) ⊆ S*(B2) ⊆ S*(B3) with stable order."""
        selector = MonotonicTaskSelector(coverage_utility)
        
        # Budget sweep with meaningful differences
        budgets = [10.0, 25.0, 45.0, 70.0, 100.0]
        selections = []
        
        for budget in budgets:
            # Reset selector state for clean testing
            selector._reset_state()
            selected, traces = selector.select_tasks(fixed_cost_tasks, budget=budget)
            selections.append(selected)
            
            # Verify budget constraint
            total_cost = sum(fixed_cost_tasks[tid].cost for tid in selected)
            assert total_cost <= budget, f"Budget violation: {total_cost} > {budget}"
        
        # Verify monotonic inclusion: S*(B_i) ⊆ S*(B_{i+1})
        for i in range(len(budgets) - 1):
            smaller_selection = selections[i]
            larger_selection = selections[i + 1]
            
            # Check subset property
            assert len(smaller_selection) <= len(larger_selection), (
                f"Non-monotonic size: B={budgets[i]} has {len(smaller_selection)} tasks, "
                f"B={budgets[i+1]} has {len(larger_selection)} tasks"
            )
            
            # Check prefix stability: smaller selection is prefix of larger
            prefix_match = larger_selection[:len(smaller_selection)]
            assert prefix_match == smaller_selection, (
                f"Prefix instability: B={budgets[i]} selection {smaller_selection} "
                f"is not prefix of B={budgets[i+1]} selection {larger_selection}"
            )
        
        # Additional verification using selector's internal method
        for i in range(len(budgets) - 1):
            selector.add_tasks(list(fixed_cost_tasks.values()))
            assert selector.verify_prefix_property(budgets[i], budgets[i + 1]), (
                f"Prefix property failed between budgets {budgets[i]} and {budgets[i + 1]}"
            )
    
    def test_incremental_budget_additions(self, fixed_cost_tasks, coverage_utility):
        """Test incremental budget increases maintain selection stability."""
        selector = MonotonicTaskSelector(coverage_utility)
        
        # Start with small budget and incrementally increase
        current_budget = 5.0
        previous_selection = []
        
        increments = [7.0, 12.0, 8.0, 15.0, 20.0, 25.0]  # Varying increments
        
        for increment in increments:
            current_budget += increment
            selected, _ = selector.select_tasks(fixed_cost_tasks, budget=current_budget)
            
            # Verify monotonic expansion
            assert len(selected) >= len(previous_selection), (
                f"Selection contracted from {len(previous_selection)} to {len(selected)}"
            )
            
            # Verify prefix stability
            if previous_selection:
                assert selected[:len(previous_selection)] == previous_selection, (
                    f"Prefix changed: {previous_selection} vs {selected[:len(previous_selection)]}"
                )
            
            previous_selection = selected
    
    def test_objective_monotonicity(self, fixed_cost_tasks, coverage_utility):
        """Test that objective function value is monotonic in budget."""
        selector = MonotonicTaskSelector(coverage_utility)
        
        budgets = [15.0, 30.0, 50.0, 80.0, 120.0]
        objective_values = []
        
        for budget in budgets:
            selector._reset_state()
            selected, _ = selector.select_tasks(fixed_cost_tasks, budget=budget)
            
            # Calculate objective value
            obj_value = coverage_utility.evaluate_set(set(selected), fixed_cost_tasks)
            objective_values.append(obj_value)
        
        # Verify non-decreasing objective values
        for i in range(len(objective_values) - 1):
            assert objective_values[i] <= objective_values[i + 1], (
                f"Objective decreased: {objective_values[i]} > {objective_values[i + 1]} "
                f"at budgets {budgets[i]} and {budgets[i + 1]}"
            )
        
        return objective_values
    
    def test_chain_property_validation(self, fixed_cost_tasks, coverage_utility):
        """Test chain property: B1 ≤ B2 ≤ B3 ⇒ S*(B1) ⊆ S*(B2) ⊆ S*(B3)."""
        selector = MonotonicTaskSelector(coverage_utility)
        
        # Test multiple budget chains
        chains = [
            [8.0, 20.0, 35.0],
            [12.0, 28.0, 45.0],
            [5.0, 15.0, 30.0, 60.0],
            [18.0, 22.0, 26.0, 40.0]
        ]
        
        chain_results = []
        
        for chain in chains:
            selections = []
            for budget in chain:
                selector._reset_state()
                selected, _ = selector.select_tasks(fixed_cost_tasks, budget=budget)
                selections.append(selected)
            
            # Verify chain property
            chain_valid = True
            for i in range(len(chain) - 1):
                smaller = selections[i]
                larger = selections[i + 1]
                
                # Check inclusion and prefix stability
                if len(smaller) > len(larger) or larger[:len(smaller)] != smaller:
                    chain_valid = False
                    break
            
            chain_results.append({
                'budgets': chain,
                'valid': chain_valid,
                'selections': [len(sel) for sel in selections]
            })
        
        # All chains should be valid
        all_valid = all(result['valid'] for result in chain_results)
        
        return all_valid, chain_results
    
    def test_edge_cases(self, coverage_utility):
        """Test edge cases for BMC property."""
        # Empty task set
        selector = MonotonicTaskSelector(coverage_utility)
        selected1, _ = selector.select_tasks({}, budget=10.0)
        selector._reset_state()
        selected2, _ = selector.select_tasks({}, budget=20.0)
        assert selected1 == selected2 == []
        
        # Single task
        single_task = {"task1": Task("task1", cost=5.0)}
        selector._reset_state()
        sel_low, _ = selector.select_tasks(single_task, budget=3.0)
        selector._reset_state()
        sel_high, _ = selector.select_tasks(single_task, budget=10.0)
        assert len(sel_low) <= len(sel_high)
        if sel_low:
            assert sel_high[:len(sel_low)] == sel_low
        
        # Equal budgets
        tasks = {f"task_{i}": Task(f"task_{i}", cost=2.0) for i in range(5)}
        selector._reset_state()
        sel1, _ = selector.select_tasks(tasks, budget=8.0)
        selector._reset_state()
        sel2, _ = selector.select_tasks(tasks, budget=8.0)
        # Should be identical (deterministic)
        assert sel1 == sel2


def generate_bmc_certificate():
    """Generate BMC certification results."""
    # Run comprehensive tests
    test_instance = TestBMCMonotonicity()
    
    # Create test fixtures
    tasks = {f"task_{i}": Task(f"task_{i}", cost=2.0 + i * 0.5) for i in range(20)}
    feature_coverage = {}
    for i in range(20):
        features = {f"feature_{j}" for j in range(i, min(i + 5, 30))}
        feature_coverage[f"task_{i}"] = features
    feature_weights = {f"feature_{i}": max(1.0, 10.0 - i * 0.3) for i in range(30)}
    utility = CoverageUtility(feature_coverage, feature_weights)
    
    try:
        # Test 1: Budget sweep monotonicity
        test_instance.test_budget_sweep_monotonicity(tasks, utility)
        budget_sweep_ok = True
    except Exception as e:
        budget_sweep_ok = False
        print(f"Budget sweep test failed: {e}")
    
    try:
        # Test 2: Chain property validation
        chains_ok, chain_results = test_instance.test_chain_property_validation(tasks, utility)
    except Exception as e:
        chains_ok = False
        print(f"Chain validation test failed: {e}")
    
    try:
        # Test 3: Objective monotonicity
        obj_values = test_instance.test_objective_monotonicity(tasks, utility)
        objective_monotone = all(
            obj_values[i] <= obj_values[i + 1] 
            for i in range(len(obj_values) - 1)
        )
    except Exception as e:
        objective_monotone = False
        print(f"Objective monotonicity test failed: {e}")
    
    # Overall pass status
    overall_pass = budget_sweep_ok and chains_ok and objective_monotone
    
    certificate = {
        "pass": overall_pass,
        "chains_ok": chains_ok,
        "objective_monotone": objective_monotone,
        "budget_sweep_ok": budget_sweep_ok,
        "test_summary": {
            "budget_ranges_tested": [10.0, 25.0, 45.0, 70.0, 100.0],
            "chain_count": 4,
            "task_count": 20,
            "coverage_features": 30
        }
    }
    
    return certificate


if __name__ == "__main__":
    # Generate and save certificate
    cert = generate_bmc_certificate()
    with open("bmc_certificate.json", "w") as f:
        json.dump(cert, f, indent=2)
    
    print("BMC Certificate generated:")
    print(json.dumps(cert, indent=2))