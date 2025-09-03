#!/usr/bin/env python3
"""
Optional Z3 integration demo for Horn Logic verification.
Falls back to custom evaluator if Z3 not available.
"""

import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Import MCC components
sys.path.append(str(Path(__file__).parent))
# # # from tests.test_mcc import SatLabel, Clause, Evidence, HornEvaluator  # Module not found  # Module not found  # Module not found


def test_z3_verification():
    """Demo Z3 integration for formal verification"""
    print('Testing Z3 integration (optional):')
    
    try:
        import z3
        print('✓ Z3 available - formal verification enabled')
        
        # Example Z3 usage for Horn clause verification
        solver = z3.Solver()
        
        # Define propositional variables
        a = z3.Bool('a')
        b = z3.Bool('b') 
        goal = z3.Bool('goal')
        
        # Add Horn clauses as implications
        solver.add(z3.Implies(a, b))  # a -> b
        solver.add(z3.Implies(b, goal))  # b -> goal
        
        # Add evidence
        solver.add(a)  # fact: a
        
        # Check if goal is derivable
        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            goal_value = model.eval(goal)
            print(f'Z3 verification: goal = {goal_value}')
            
            # Compare with custom evaluator
            evidence = Evidence(
                facts={'a'},
                clauses=[
                    Clause({'a'}, 'b'),
                    Clause({'b'}, 'goal')
                ]
            )
            evaluator = HornEvaluator({'goal'})
            custom_label = evaluator.evaluate(evidence)
            
            z3_satisfies = goal_value == True
            custom_satisfies = custom_label == SatLabel.SAT
            
            print(f'Z3 result: {z3_satisfies}')
            print(f'Custom evaluator: {custom_satisfies}')
            print(f'Agreement: {z3_satisfies == custom_satisfies}')
        
    except ImportError:
        print('ℹ Z3 not available - using custom evaluator only')
        
        # Demonstrate custom evaluator
        evidence = Evidence(
            facts={'a'},
            clauses=[
                Clause({'a'}, 'b'),
                Clause({'b'}, 'goal')
            ]
        )
        evaluator = HornEvaluator({'goal'})
        label = evaluator.evaluate(evidence)
        print(f'Custom evaluator result: {label.name}')


if __name__ == '__main__':
    test_z3_verification()