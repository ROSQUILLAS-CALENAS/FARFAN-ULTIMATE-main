"""
Example usage of the tamper-evident lineage tracking system
demonstrating integration with the evidence collection pipeline.
"""
import os
import tempfile
from typing import Any, Dict, List

from evidence_system import Evidence, EvidenceSystem
from lineage_tracker import EventType, LineageTracker, create_evidence_lineage_adapter


def simulate_question_processing_pipeline(
    question_id: str, tracker: LineageTracker, evidence_system: EvidenceSystem
) -> str:
    """
    Simulate a complete question processing pipeline with lineage tracking.

    Args:
        question_id: The question to process
        tracker: Lineage tracker instance
        evidence_system: Evidence collection system

    Returns:
        str: Trace ID for the processing pipeline
    """
    print(f"\n=== Processing Question: {question_id} ===")

    # Create lineage trace
    trace_id = tracker.create_trace(
        question_id,
        metadata={
            "domain": "municipal_planning",
            "user": "analyst_001",
            "priority": "high",
        },
    )
    print(f"Created trace: {trace_id}")

    # Step 1: Evidence Collection
    print("\n1. Evidence Collection Phase")
    evidence_items = [
        Evidence(
            qid=question_id,
            content="Budget allocation document",
            score=0.85,
            dimension="financial",
        ),
        Evidence(
            qid=question_id,
            content="Population demographics data",
            score=0.92,
            dimension="demographic",
        ),
        Evidence(
            qid=question_id,
            content="Infrastructure assessment report",
            score=0.78,
            dimension="infrastructure",
        ),
        Evidence(
            qid=question_id,
            content="Community feedback survey",
            score=0.87,
            dimension="social",
        ),
    ]

    for evidence in evidence_items:
        # Add to evidence system
        added = evidence_system.add_evidence(question_id, evidence)

        # Track in lineage
        tracker.log_evidence_source(
            trace_id, evidence.content, f"evidence_system_{evidence.dimension}"
        )
        print(f"  Added evidence: {evidence.content[:30]}... (score: {evidence.score})")

    # Step 2: Evidence Aggregation & Scoring
    print("\n2. Evidence Aggregation Phase")
    all_evidence = evidence_system.get_evidence_for_question(question_id)
    aggregate_score = sum(e.score for e in all_evidence) / len(all_evidence)

    aggregation_result = {
        "total_evidence": len(all_evidence),
        "aggregate_score": aggregate_score,
        "coverage_dimensions": len(set(e.dimension for e in all_evidence)),
    }

    tracker.log_processing_step(trace_id, "evidence_aggregation", aggregation_result)
    print(
        f"  Aggregated {len(all_evidence)} pieces of evidence, score: {aggregate_score:.3f}"
    )

    # Step 3: Conformal Prediction Coverage
    print("\n3. Conformal Prediction Phase")
    coverage = evidence_system.calculate_coverage()

    conformal_result = {
        "empirical_coverage": coverage,
        "target_coverage": 1 - evidence_system.alpha,
        "coverage_gap": coverage - (1 - evidence_system.alpha),
    }

    tracker.log_processing_step(trace_id, "conformal_prediction", conformal_result)
    print(f"  Coverage: {coverage:.3f} (target: {1-evidence_system.alpha:.3f})")

    # Step 4: Final Answer Generation
    print("\n4. Answer Generation Phase")
    recommendation = {
        "recommendation": "Increase infrastructure budget by 15%",
        "confidence": aggregate_score,
        "supporting_dimensions": list(set(e.dimension for e in all_evidence)),
        "evidence_count": len(all_evidence),
    }

    tracker.log_processing_step(trace_id, "answer_generation", recommendation)
    print(f"  Generated recommendation: {recommendation['recommendation']}")

    return trace_id


def demonstrate_audit_and_verification(tracker: LineageTracker, trace_id: str):
    """Demonstrate audit trail generation and integrity verification."""
    print(f"\n=== Audit & Verification for {trace_id} ===")

    # Generate complete audit trail
    print("\n1. Generating Audit Trail")
    audit_trail = tracker.generate_audit_trail(trace_id)

    print(f"  📊 Total Events: {audit_trail.total_events}")
    print(f"  🔐 Merkle Root: {audit_trail.merkle_root[:16]}...")
    print(
        f"  ⏰ Duration: {audit_trail.last_update - audit_trail.creation_time:.2f} seconds"
    )

    # Show event timeline
    print("\n2. Event Timeline")
    for i, event in enumerate(audit_trail.events):
        event_time = event.timestamp - audit_trail.creation_time
        print(
            f"  {i+1}. [{event_time:6.3f}s] {event.event_type.value:<18} | {event.event_id[:16]}..."
        )

    # Verify integrity
    print("\n3. Integrity Verification")
    integrity_results = tracker.verify_trace_integrity(trace_id)

    status = "✅ PASS" if not integrity_results["tamper_detected"] else "❌ FAIL"
    print(f"  Tamper Detection: {status}")
    print(
        f"  Ratchet Chain: {'✅ INTACT' if integrity_results['ratchet_integrity'] else '❌ BROKEN'}"
    )
    print(
        f"  Consistency Check: {'✅ PASS' if integrity_results['consistency_check'] else '❌ FAIL'}"
    )

    # Verify inclusion proofs
    print("\n4. Inclusion Proof Verification")
    valid_proofs = 0
    for event_id, proof_valid in integrity_results["inclusion_proof_results"].items():
        if proof_valid:
            valid_proofs += 1
        else:
            print(f"  ❌ Failed: {event_id[:16]}...")

    print(
        f"  Valid Proofs: {valid_proofs}/{len(integrity_results['inclusion_proof_results'])}"
    )

    # Show failed checks if any
    if integrity_results["failed_checks"]:
        print("\n5. ⚠️  Failed Integrity Checks:")
        for check in integrity_results["failed_checks"]:
            print(f"  - {check}")
    else:
        print("\n5. ✅ All integrity checks passed!")

    return audit_trail, integrity_results


def demonstrate_cross_question_analysis(tracker: LineageTracker, trace_ids: List[str]):
    """Demonstrate analysis across multiple question traces."""
    global summary
    print(f"\n=== Cross-Question Analysis ===")

    print(f"\nAnalyzing {len(trace_ids)} question traces:")

    total_events = 0
    processing_times = []
    event_types = set()

    for trace_id in trace_ids:
        summary = tracker.get_trace_summary(trace_id)
        if summary["exists"]:
            total_events += summary["event_count"]
            processing_times.append(summary["duration_seconds"])
            event_types.update(summary["event_distribution"].keys())

            print(
                f"  📝 {trace_id[:24]}... | {summary['event_count']} events | {summary['duration_seconds']:.2f}s"
            )

    print(f"\n📊 Aggregate Statistics:")
    print(f"  Total Events: {total_events}")
    print(f"  Avg Processing Time: {sum(processing_times)/len(processing_times):.3f}s")
    print(f"  Event Types: {', '.join(sorted(event_types))}")
    print(f"  Hash Algorithm: {summary.get('hash_algorithm', 'sha256')}")


def demonstrate_tamper_detection(tracker: LineageTracker, trace_id: str):
    """Demonstrate tamper detection by corrupting data."""
    print(f"\n=== Tamper Detection Demo ===")

    # First verify baseline integrity
    print("\n1. Baseline Integrity Check")
    baseline = tracker.verify_trace_integrity(trace_id)
    print(
        f"  Baseline Status: {'✅ CLEAN' if not baseline['tamper_detected'] else '❌ COMPROMISED'}"
    )

    # Simulate tampering by corrupting database
    print("\n2. Simulating Tampering")
    try:
        import sqlite3

        with sqlite3.connect(tracker.db_path) as conn:
            # Corrupt a random event hash
            cursor = conn.execute(
                "SELECT event_id FROM events WHERE trace_id = ? LIMIT 1", (trace_id,)
            )
            event_id = cursor.fetchone()
            if event_id:
                conn.execute(
                    "UPDATE events SET event_hash = 'corrupted_hash_for_demo' WHERE event_id = ?",
                    (event_id[0],),
                )
                print(f"  🔴 Corrupted event: {event_id[0][:16]}...")

        # Clear cache to force reload
        tracker._merkle_cache.clear()

        # Re-verify integrity
        print("\n3. Post-Tampering Integrity Check")
        tampered = tracker.verify_trace_integrity(trace_id)

        if tampered["tamper_detected"]:
            print("  ✅ Tampering successfully detected!")
            print(f"  🔍 Failed checks: {len(tampered['failed_checks'])}")
            for check in tampered["failed_checks"]:
                print(f"    - {check}")
        else:
            print("  ❌ Tampering not detected (unexpected)")

    except Exception as e:
        print(f"  ⚠️  Could not demonstrate tampering: {e}")


def main():
    """Main demonstration of lineage tracking system."""
    print("🔐 Tamper-Evident Lineage Tracking System Demo")
    print("=" * 60)

    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    try:
        # Initialize systems
        tracker = LineageTracker(db_path)
        evidence_system = EvidenceSystem(alpha=0.1)

        # Process multiple questions to show system capabilities
        question_ids = [
            "Q001_budget_analysis",
            "Q002_infrastructure_planning",
            "Q003_community_engagement",
        ]
        trace_ids = []

        for question_id in question_ids:
            trace_id = simulate_question_processing_pipeline(
                question_id, tracker, evidence_system
            )
            trace_ids.append(trace_id)

        # Demonstrate audit capabilities
        if trace_ids:
            demonstrate_audit_and_verification(tracker, trace_ids[0])
            demonstrate_cross_question_analysis(tracker, trace_ids)
            demonstrate_tamper_detection(tracker, trace_ids[0])

        print(f"\n🎉 Demo completed successfully!")
        print(f"📄 Database saved at: {db_path}")

    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        raise
    finally:
        # Cleanup (comment out to inspect database)
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"🗑️  Cleaned up temporary database")


if __name__ == "__main__":
    main()
