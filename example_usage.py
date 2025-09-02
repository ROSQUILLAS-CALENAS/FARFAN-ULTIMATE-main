"""
Example usage of the Evidence System with Conformal Coverage
"""
import time

import numpy as np

from evidence_system import Evidence, EvidenceSystem


def main():
    """Demonstrate the Evidence System functionality"""
    print("🔬 Evidence System with Conformal Coverage Demo")
    print("=" * 50)

    # Initialize system with 90% coverage target (α = 0.1)
    system = EvidenceSystem(alpha=0.1)
    print(f"Initialized system with target coverage: {1 - system.alpha:.1%}")

    print("\n1. Adding Evidence (Idempotent)")
    print("-" * 30)

    # Add evidence for different questions
    questions_evidence = [
        ("q1", "This is high quality evidence", 0.92, "quality"),
        ("q1", "Additional supporting evidence", 0.85, "quality"),
        ("q1", "Relevant contextual information", 0.78, "relevance"),
        ("q2", "Strong methodological evidence", 0.88, "methodology"),
        ("q2", "Peer reviewed citation", 0.91, "credibility"),
        ("q3", "Statistical significance confirmed", 0.84, "statistics"),
        ("q3", "Replicated results", 0.87, "reproducibility"),
    ]

    for qid, content, score, dimension in questions_evidence:
        evidence = Evidence(qid=qid, content=content, score=score, dimension=dimension)
        added = system.add_evidence(qid, evidence)
        print(f"  {qid}: {'Added' if added else 'Already exists'} - {content[:30]}...")

    # Test idempotency
    duplicate_evidence = Evidence(
        qid="q1",
        content="This is high quality evidence",
        score=0.92,
        dimension="quality",
    )
    added = system.add_evidence("q1", duplicate_evidence)
    print(f"  q1: {'Added' if added else 'Already exists'} (duplicate test)")

    print(f"\n2. Evidence Retrieval")
    print("-" * 20)

    for qid in ["q1", "q2", "q3"]:
        evidence_list = system.get_evidence_for_question(qid)
        print(f"  {qid}: {len(evidence_list)} evidence items")
        for i, ev in enumerate(evidence_list):
            print(f"    {i+1}. [{ev.dimension}] Score: {ev.score:.2f}")

    print(f"\n3. Grouping by Dimension")
    print("-" * 25)

    grouped = system.group_by_dimension()
    for dimension, evidence_list in grouped.items():
        print(f"  {dimension}: {len(evidence_list)} items")
        avg_score = np.mean([e.score for e in evidence_list])
        print(f"    Average score: {avg_score:.3f}")

    print(f"\n4. Shuffle Invariance Test")
    print("-" * 26)

    for qid in ["q1", "q2"]:
        invariant = system.test_shuffle_invariance(qid, n_shuffles=5)
        print(f"  {qid}: {'✓ PASS' if invariant else '✗ FAIL'}")

    print(f"\n5. Coverage Calculation")
    print("-" * 22)

    # Calculate coverage with synthetic labels
    coverage = system.calculate_coverage()
    target = 1 - system.alpha
    print(f"  Empirical Coverage: {coverage:.3f}")
    print(f"  Target Coverage:    {target:.3f}")
    print(f"  Difference:         {coverage - target:+.3f}")

    print(f"\n6. Coverage Audit")
    print("-" * 16)

    audit_results = system.audit_coverage(n_trials=50, delta=0.05)
    print(f"  Audit Trials:       {audit_results['n_trials']}")
    print(f"  Empirical Coverage: {audit_results['empirical_coverage']:.3f}")
    print(f"  Coverage Std:       {audit_results['coverage_std']:.3f}")
    print(
        f"  Audit Result:       {'✓ PASS' if audit_results['passes_audit'] else '✗ FAIL'}"
    )

    print(f"\n7. DR-Submodular Selection")
    print("-" * 28)

    # Test submodular selection
    all_evidence = []
    for evidence_set in system._evidence_store.values():
        all_evidence.extend(list(evidence_set))

    selected, objective_value = system._dr_submodular_selection(all_evidence, budget=3)
    print(f"  Selected {len(selected)}/3 evidence items")
    print(f"  Submodular objective: {objective_value:.3f}")
    print("  Selected items:")
    for i, ev in enumerate(selected):
        print(
            f"    {i+1}. [{ev.dimension}] {ev.content[:40]}... (score: {ev.score:.2f})"
        )

    print(f"\n8. System Statistics")
    print("-" * 18)

    stats = system.get_stats()
    for key, value in stats.items():
        if isinstance(value, float) and value is not None:
            print(f"  {key:20}: {value:.3f}")
        elif isinstance(value, list):
            print(f"  {key:20}: {len(value)} items")
        else:
            print(f"  {key:20}: {value}")

    print(f"\n9. Coverage Visualization")
    print("-" * 23)

    # Add more coverage calculations to build history
    for _ in range(5):
        system.calculate_coverage()
        time.sleep(0.01)  # Small delay for timestamp variation

    try:
        system.plot_coverage_history(save_path="coverage_history.png")
        print("  Coverage plot saved to 'coverage_history.png'")
    except Exception as e:
        print(f"  Plot generation skipped: {str(e)}")

    print(f"\n✅ Demo completed successfully!")
    print(f"   System contains {stats['total_evidence']} evidence items")
    print(f"   across {stats['total_questions']} questions")
    print(f"   with {len(stats['dimensions'])} dimensions")


if __name__ == "__main__":
    main()
