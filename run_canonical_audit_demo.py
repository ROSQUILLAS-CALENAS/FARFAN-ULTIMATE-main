#!/usr/bin/env python3
"""
Demo: Canonical Audit End-to-End
- Builds a minimal 4-cluster payload with evidence linked to questions.
- Runs the canonical_output_auditor and public_transformer_adapter.
- Prints audit summary and public report hash.
"""
from __future__ import annotations

from typing import Dict, Any, List

from evidence_system import Evidence, EvidenceSystem
import canonical_output_auditor as coa
import public_transformer_adapter as pta


def build_demo_payload() -> Dict[str, Any]:
    # Prepare evidence system with two questions
    evsys = EvidenceSystem(alpha=0.1)

    def add(qid: str, txt: str, dim: str, score: float) -> Evidence:
        e = Evidence(qid=qid, content=txt, dimension=dim, score=score)
        evsys.add_evidence(qid, e)
        # The canonical id will be stable from serialize_canonical
        # but for simplicity the answers will reference the computed id via serialize
        return e

    # Add evidence objects
    e1 = add("Q1", "Texto de evidencia Q1-A", "diagnostico", 0.8)
    e2 = add("Q1", "Texto de evidencia Q1-B", "programa", 0.7)
    e3 = add("Q2", "Texto de evidencia Q2-A", "presupuesto", 0.9)

    # Build evidence id index from the system
    import json
    store = json.loads(evsys.serialize_canonical())["store"]
    def ids_for(qid: str) -> List[str]:
        return [item["id"] for item in store.get(qid, [])]

    # Minimal 4-cluster answers
    clusters = ["C1", "C2", "C3", "C4"]
    cluster_answers = {
        "C1": [
            {"question_id": "Q1", "verdict": "Sí", "score": 0.9, "evidence_ids": ids_for("Q1")[:1]},
            {"question_id": "Q2", "verdict": "Parcial", "score": 0.6, "evidence_ids": ids_for("Q2")[:1]},
        ],
        "C2": [
            {"question_id": "Q1", "verdict": "Sí", "score": 0.85, "evidence_ids": ids_for("Q1")[:1]},
        ],
        "C3": [
            {"question_id": "Q1", "verdict": "Parcial", "score": 0.5, "evidence_ids": ids_for("Q1")[:1]},
        ],
        "C4": [
            {"question_id": "Q2", "verdict": "Sí", "score": 0.95, "evidence_ids": ids_for("Q2")[:1]},
        ],
    }

    data = {
        "clusters": clusters,
        "cluster_answers": cluster_answers,
        "evidence_system": evsys,
        # Signal DNP usage and causal information minimally
        "dnp_alignment": {"rule": "DNP-2024-Std", "status": "used"},
        "causal_correction": {"method": "proximal"},
    }
    return data


def main() -> None:
    data = build_demo_payload()
    audited = coa.process(data, context={"source": "demo"})
    public = pta.process(audited, context={"source": "demo"})

    audit = audited.get("canonical_audit", {})
    print("Canonical Audit Summary:")
    print({
        "four_clusters_confirmed": audit.get("four_clusters_confirmed"),
        "clusters_complete": audit.get("clusters_complete"),
        "non_redundant": audit.get("non_redundant"),
        "uses_dnp_standards": audit.get("uses_dnp_standards"),
        "causal_correction_signals": audit.get("causal_correction_signals"),
        "reporting_levels": audit.get("reporting_levels"),
        "gaps": audit.get("gaps"),
    })

    print("Public Report Hash:", public.get("public_report", {}).get("hash"))


if __name__ == "__main__":
    main()
