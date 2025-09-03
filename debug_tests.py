#!/usr/bin/env python3
"""Debug failing tests"""

import numpy as np

# # # from evidence_system import Evidence, EvidenceSystem  # Module not found  # Module not found  # Module not found


def debug_retrieval():
    print("=== Debug Evidence Retrieval ===")
    system = EvidenceSystem()
    qid = "q1"
    evidence = Evidence(
        qid=qid, content="Test evidence", score=0.8, dimension="quality"
    )
    result = system.add_evidence(qid, evidence)
    print(f"Added evidence: {result}")

    retrieved = system.get_evidence_for_question(qid)
    print(f"Retrieved evidence count: {len(retrieved)}")
    for i, e in enumerate(retrieved):
        print(f"  {i}: {e.qid} - {e.content}")


def debug_grouping():
    print("=== Debug Grouping ===")
    system = EvidenceSystem()
    evidence1 = Evidence(qid="q1", content="Content 1", score=0.8, dimension="quality")
    evidence2 = Evidence(qid="q2", content="Content 2", score=0.7, dimension="quality")
    evidence3 = Evidence(
        qid="q3", content="Content 3", score=0.9, dimension="relevance"
    )

    system.add_evidence("q1", evidence1)
    system.add_evidence("q2", evidence2)
    system.add_evidence("q3", evidence3)

    grouped = system.group_by_dimension()
    print(f"Grouped keys: {list(grouped.keys())}")
    for key, items in grouped.items():
        print(f"  {key}: {len(items)} items")


def debug_stats():
    print("=== Debug System Stats ===")
    system = EvidenceSystem()

    for i in range(3):
        evidence = Evidence(
            qid=f"q{i}",
            content=f"Evidence {i}",
            score=0.8,
            dimension="quality" if i < 2 else "relevance",
        )
        system.add_evidence(f"q{i}", evidence)

    try:
        stats = system.get_stats()
        print("Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error getting stats: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_retrieval()
    print()
    debug_grouping()
    print()
    debug_stats()
