"""
Tests for Monotone Consistency Check (MCC) in Horn Logic systems.
Verifies that adding non-contradictory evidence never decreases satisfaction labels.
UNSAT < PARTIAL < SAT ordering must be preserved.
"""

import json
# # # from typing import Set, Dict, List, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from enum import IntEnum  # Module not found  # Module not found  # Module not found

# Handle pytest import gracefully
try:
    import pytest
except ImportError:
    # Mock pytest functions for standalone execution
    class MockPytest:
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)
        
        @staticmethod
        def main(args):
            return 0
    
    pytest = MockPytest()


class SatLabel(IntEnum):
    """Satisfaction labels with monotone ordering: UNSAT < PARTIAL < SAT"""
    UNSAT = 0
    PARTIAL = 1
    SAT = 2


@dataclass
class Clause:
    """Horn clause representation: body → head"""
    body: Set[str]  # conjunction of literals
    head: str       # single literal


@dataclass 
class Evidence:
    """Evidence set with facts and clauses"""
    facts: Set[str]
    clauses: List[Clause]
    
    def __hash__(self):
        return hash((frozenset(self.facts), tuple(sorted(str(c.body) + "->" + c.head for c in self.clauses))))
    
    def is_subset_of(self, other: 'Evidence') -> bool:
        """Check if this evidence is subset of other"""
        return (self.facts.issubset(other.facts) and 
                all(any(c1.body == c2.body and c1.head == c2.head for c2 in other.clauses) 
                    for c1 in self.clauses))


class HornEvaluator:
    """Horn clause evaluator with forward chaining"""
    
    def __init__(self, target_facts: Set[str]):
        self.target_facts = target_facts
        self.triggered_clauses: List[Clause] = []
    
    def evaluate(self, evidence: Evidence) -> SatLabel:
        """Evaluate evidence and return satisfaction label"""
        self.triggered_clauses = []
        derived_facts = evidence.facts.copy()
        
        # Forward chaining until fixpoint
        changed = True
        while changed:
            changed = False
            for clause in evidence.clauses:
                if clause.body.issubset(derived_facts) and clause.head not in derived_facts:
                    derived_facts.add(clause.head)
                    self.triggered_clauses.append(clause)
                    changed = True
        
        # Check satisfaction against targets
        satisfied_targets = len(self.target_facts.intersection(derived_facts))
        total_targets = len(self.target_facts)
        
        if satisfied_targets == 0:
            return SatLabel.UNSAT
        elif satisfied_targets == total_targets:
            return SatLabel.SAT
        else:
            return SatLabel.PARTIAL


def test_monotone_consistency_basic():
    """Test basic monotonicity: E ⊆ E' ⇒ label(E') ≥ label(E)"""
    evaluator = HornEvaluator(target_facts={'goal'})
    
    # Base evidence
    e1 = Evidence(
        facts={'a'},
        clauses=[Clause(body={'a'}, head='b')]
    )
    
    # Extended evidence (superset)
    e2 = Evidence(
        facts={'a', 'c'},
        clauses=[
            Clause(body={'a'}, head='b'),
            Clause(body={'b', 'c'}, head='goal')
        ]
    )
    
    assert e1.is_subset_of(e2)
    
    label1 = evaluator.evaluate(e1)
    label2 = evaluator.evaluate(e2)
    
    assert label2 >= label1, f"Monotonicity violated: {label1} -> {label2}"


def test_monotone_consistency_comprehensive():
    """Comprehensive test of monotonicity with multiple scenarios"""
    results = {'pass': True, 'upgrades': 0, 'illegal_downgrades': 0}
    
    target_facts = {'goal1', 'goal2'}
    evaluator = HornEvaluator(target_facts)
    
    test_cases = [
        # (base_evidence, extended_evidence, description)
        (
            Evidence(facts=set(), clauses=[]),
            Evidence(facts={'a'}, clauses=[]),
            "Adding facts to empty set"
        ),
        (
            Evidence(facts={'a'}, clauses=[]),
            Evidence(facts={'a', 'b'}, clauses=[]),
            "Adding more facts"
        ),
        (
            Evidence(facts={'a'}, clauses=[Clause({'a'}, 'b')]),
            Evidence(facts={'a'}, clauses=[
                Clause({'a'}, 'b'),
                Clause({'b'}, 'goal1')
            ]),
            "Adding enabling clauses"
        ),
        (
            Evidence(facts={'a'}, clauses=[Clause({'a'}, 'goal1')]),
            Evidence(facts={'a', 'c'}, clauses=[
                Clause({'a'}, 'goal1'),
                Clause({'c'}, 'goal2')
            ]),
            "Independent path expansion"
        ),
    ]
    
    for base_ev, ext_ev, desc in test_cases:
        assert base_ev.is_subset_of(ext_ev), f"Subset relation failed: {desc}"
        
        label_base = evaluator.evaluate(base_ev)
        label_ext = evaluator.evaluate(ext_ev)
        
        if label_ext > label_base:
            results['upgrades'] += 1
        elif label_ext < label_base:
            results['illegal_downgrades'] += 1
            results['pass'] = False
            pytest.fail(f"Monotonicity violation in {desc}: {label_base} -> {label_ext}")
    
    # Write certificate
    with open('mcc_certificate.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    assert results['illegal_downgrades'] == 0, "Found illegal downgrades in monotonicity"


def test_mandatory_failure_deterministic_downgrade():
    """Test that mandatory clause failures lead to deterministic downgrades"""
    # Test case 1: Incomplete satisfaction (some targets unreachable)
    evaluator = HornEvaluator(target_facts={'goal1', 'goal2', 'goal3'})
    
    evidence = Evidence(
        facts={'a'},
        clauses=[
            Clause({'a'}, 'goal1'),  # Can reach goal1
            # goal2 and goal3 are unreachable -> PARTIAL
        ]
    )
    
    label = evaluator.evaluate(evidence)
    assert label == SatLabel.PARTIAL, f"Expected PARTIAL but got {label}"
    
    # Test case 2: Complete failure (no targets reachable)  
    evaluator_fail = HornEvaluator(target_facts={'unreachable_goal'})
    
    evidence_fail = Evidence(
        facts={'a'},
        clauses=[
            Clause({'a'}, 'b'),
            # unreachable_goal cannot be derived -> UNSAT
        ]
    )
    
    label_fail = evaluator_fail.evaluate(evidence_fail)
    assert label_fail == SatLabel.UNSAT, f"Expected UNSAT but got {label_fail}"


def test_clause_triggering_trace():
    """Test that we can track which clauses were triggered"""
    evaluator = HornEvaluator(target_facts={'goal'})
    
    evidence = Evidence(
        facts={'a', 'c'},
        clauses=[
            Clause({'a'}, 'b'),
            Clause({'b', 'c'}, 'goal'),
            Clause({'d'}, 'unused')  # Should not trigger
        ]
    )
    
    label = evaluator.evaluate(evidence)
    
    assert label == SatLabel.SAT
    assert len(evaluator.triggered_clauses) == 2
    
    triggered_heads = {c.head for c in evaluator.triggered_clauses}
    assert 'b' in triggered_heads
    assert 'goal' in triggered_heads
    assert 'unused' not in triggered_heads


if __name__ == '__main__':
    pytest.main([__file__, '-v'])