#!/usr/bin/env python3
"""
Horn Logic Label Explainer
Lists triggered clauses and derivation chains for satisfaction labels.
Provides transparency into monotone reasoning process.
"""

import json
import sys
from typing import Set, List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Import from test module for consistency
sys.path.append(str(Path(__file__).parent.parent))
from tests.test_mcc import SatLabel, Clause, Evidence, HornEvaluator


@dataclass
class ExplanationTrace:
    """Detailed explanation of label derivation"""
    evidence_hash: str
    label: SatLabel
    triggered_clauses: List[Dict]
    derivation_steps: List[Dict]
    target_coverage: Dict[str, bool]
    
    
class LabelExplainer:
    """Explains how satisfaction labels are derived in Horn logic"""
    
    def __init__(self, target_facts: Set[str]):
        self.target_facts = target_facts
        self.evaluator = HornEvaluator(target_facts)
    
    def explain_label(self, evidence: Evidence, verbose: bool = True) -> ExplanationTrace:
        """Generate detailed explanation of label derivation"""
        # Evaluate evidence
        label = self.evaluator.evaluate(evidence)
        
        # Track derivation steps
        derivation_steps = []
        derived_facts = evidence.facts.copy()
        
        # Re-run evaluation to capture steps
        step_num = 0
        changed = True
        while changed:
            changed = False
            step_num += 1
            new_facts_this_step = set()
            
            for clause in evidence.clauses:
                if clause.body.issubset(derived_facts) and clause.head not in derived_facts:
                    derived_facts.add(clause.head)
                    new_facts_this_step.add(clause.head)
                    changed = True
                    
                    if verbose:
                        derivation_steps.append({
                            'step': step_num,
                            'clause': {
                                'body': list(clause.body),
                                'head': clause.head
                            },
                            'new_fact': clause.head,
                            'justification': f"From {clause.body} derive {clause.head}"
                        })
        
        # Analyze target coverage
        target_coverage = {
            fact: fact in derived_facts 
            for fact in self.target_facts
        }
        
        # Convert triggered clauses to serializable format
        triggered_clause_dicts = [
            {
                'body': list(clause.body),
                'head': clause.head,
                'triggered_by': list(clause.body.intersection(evidence.facts))
            }
            for clause in self.evaluator.triggered_clauses
        ]
        
        return ExplanationTrace(
            evidence_hash=str(hash(evidence)),
            label=label,
            triggered_clauses=triggered_clause_dicts,
            derivation_steps=derivation_steps,
            target_coverage=target_coverage
        )
    
    def compare_explanations(self, evidence1: Evidence, evidence2: Evidence) -> Dict:
        """Compare explanations between two evidence sets"""
        trace1 = self.explain_label(evidence1, verbose=False)
        trace2 = self.explain_label(evidence2, verbose=False)
        
        return {
            'evidence1': {
                'label': trace1.label.name,
                'triggered_count': len(trace1.triggered_clauses),
                'satisfied_targets': sum(trace1.target_coverage.values())
            },
            'evidence2': {
                'label': trace2.label.name,
                'triggered_count': len(trace2.triggered_clauses),
                'satisfied_targets': sum(trace2.target_coverage.values())
            },
            'monotonicity_preserved': trace2.label >= trace1.label,
            'new_clauses_triggered': len(trace2.triggered_clauses) - len(trace1.triggered_clauses),
            'additional_targets_satisfied': (
                sum(trace2.target_coverage.values()) - sum(trace1.target_coverage.values())
            )
        }


def main():
    """CLI interface for label explanation"""
    if len(sys.argv) < 2:
        print("Usage: python label_explain.py <evidence_file.json>")
        print("       python label_explain.py compare <evidence1.json> <evidence2.json>")
        sys.exit(1)
    
    if sys.argv[1] == 'compare' and len(sys.argv) >= 4:
        # Compare mode
        explainer = LabelExplainer(target_facts={'goal'})  # Default target
        
        # Load evidence files (simplified format for demo)
        with open(sys.argv[2], 'r') as f:
            ev1_data = json.load(f)
        with open(sys.argv[3], 'r') as f:
            ev2_data = json.load(f)
        
        # Convert to Evidence objects (simplified)
        ev1 = Evidence(
            facts=set(ev1_data.get('facts', [])),
            clauses=[
                Clause(body=set(c['body']), head=c['head'])
                for c in ev1_data.get('clauses', [])
            ]
        )
        ev2 = Evidence(
            facts=set(ev2_data.get('facts', [])),
            clauses=[
                Clause(body=set(c['body']), head=c['head'])
                for c in ev2_data.get('clauses', [])
            ]
        )
        
        comparison = explainer.compare_explanations(ev1, ev2)
        print(json.dumps(comparison, indent=2))
        
    else:
        # Single explanation mode
        print("Demo explanation for test case:")
        
        # Demo case
        explainer = LabelExplainer(target_facts={'goal1', 'goal2'})
        
        evidence = Evidence(
            facts={'a', 'c'},
            clauses=[
                Clause({'a'}, 'b'),
                Clause({'b'}, 'goal1'),
                Clause({'c'}, 'goal2')
            ]
        )
        
        trace = explainer.explain_label(evidence)
        
        print(f"\nEvidence Hash: {trace.evidence_hash}")
        print(f"Satisfaction Label: {trace.label.name}")
        print(f"Target Coverage: {trace.target_coverage}")
        
        print(f"\nTriggered Clauses ({len(trace.triggered_clauses)}):")
        for i, clause in enumerate(trace.triggered_clauses, 1):
            print(f"  {i}. {clause['body']} → {clause['head']}")
            print(f"     Triggered by: {clause['triggered_by']}")
        
        print(f"\nDerivation Steps ({len(trace.derivation_steps)}):")
        for step in trace.derivation_steps:
            print(f"  Step {step['step']}: {step['justification']}")


if __name__ == '__main__':
    main()