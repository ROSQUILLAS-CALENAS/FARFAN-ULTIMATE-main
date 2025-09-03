"""
Canonical Output Auditor
- Verifies canonical process compliance after pipeline execution.
- Ensures four-cluster application, evidence linkage, DNP standards usage,
  causal correction calibration triggers, and micro/meso/macro reporting.
- Produces an audit dictionary with explicit gap flags and replicability markers.

Usage: process(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

# # # from typing import Any, Dict, List, Optional, Set  # Module not found  # Module not found  # Module not found
import hashlib
import time

# Best-effort imports of optional modules; auditor degrades gracefully
try:
    import cluster_execution_controller  # type: ignore
except Exception:
    cluster_execution_controller = None  # type: ignore

try:
    import meso_aggregator  # type: ignore
except Exception:
    meso_aggregator = None  # type: ignore

# Constants
REQUIRED_CLUSTERS_LABEL = ("C1", "C2", "C3", "C4")
REQUIRED_CLUSTERS_NUM = (1, 2, 3, 4)


def _stable_hash_dict(d: Dict[str, Any]) -> str:
    try:
        import json
        content = json.dumps(d, sort_keys=True, ensure_ascii=False)
    except Exception:
        content = str(d)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _collect_evidence_ids_from_answers(answers: List[Dict[str, Any]]) -> Set[str]:
    eids: Set[str] = set()
    for a in answers or []:
        if not isinstance(a, dict):
            continue
        for key in ("evidence_ids", "citations"):
            vals = a.get(key) or []
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str):
                        eids.add(v)
    return eids


def _ensure_micro_and_meso(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    cluster_audit = out.get("cluster_audit")
    # Attempt to build micro via cluster_execution_controller if missing
    if not isinstance(cluster_audit, dict) and cluster_execution_controller:
        try:
            out = cluster_execution_controller.process(out, context={})
            cluster_audit = out.get("cluster_audit")
        except Exception:
            pass

    # Build meso if missing and micro present
    if (not out.get("meso_summary")) and meso_aggregator and isinstance(cluster_audit, dict):
        try:
            out = meso_aggregator.process(out, context={})
        except Exception:
            pass
    return out


def _compute_macro_synthesis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight macro synthesis: alignment extent and risk based on meso divergence and presence of DNP validations."""
    meso = data.get("meso_summary") or {}
    div = meso.get("divergence_stats") or {}
    max_div = float(div.get("max", 0.0))
    avg_div = float(div.get("avg", 0.0))

    # Evidence coverage proxy
    total_items = int(div.get("count", 0))
    evidence_total = 0
    items = (meso.get("items") or {}).values()
    for slot in items:
        evidence_total += int(slot.get("evidence_coverage", 0))
    coverage_proxy = float(evidence_total) / max(1, total_items)

    # DNP validation presence
    uses_dnp = False
    dnp_keys = ("dnp_validation_results", "dnp_compliance", "dnp_alignment")
    def _has_dnp(x: Any) -> bool:
        if isinstance(x, dict):
            return any(k in x for k in dnp_keys)
        return False

    uses_dnp = _has_dnp(data) or _has_dnp(data.get("report", {}))

    # Simple alignment score combining low divergence and coverage
    alignment_score = max(0.0, 1.0 - min(1.0, (max_div + avg_div) / 2.0)) * (0.5 + 0.5 * min(1.0, coverage_proxy))

    macro = {
        "alignment_score": round(alignment_score, 4),
        "uses_dnp_standards": bool(uses_dnp),
        "divergence": {"max": max_div, "avg": avg_div},
        "coverage_proxy": round(coverage_proxy, 4),
        "timestamp": time.time(),
    }
    return macro


def process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = context or {}
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        out.update(data)

    # Ensure micro/meso exist where possible
    out = _ensure_micro_and_meso(out)

    # Four-cluster verification and micro evidence linkage
    cluster_ok = False
    gaps: List[str] = []
    cluster_audit = out.get("cluster_audit") or {}
    present_clusters: List[str] = []
    micro_ev_detected = False
    per_cluster_resolution: Dict[str, Dict[str, Any]] = {}

    if isinstance(cluster_audit, dict):
        present_clusters = cluster_audit.get("present") or []
        micro = cluster_audit.get("micro") or {}
        # Accept either labeled C1..C4 or numeric 1..4
        set_labels = set([str(c) for c in present_clusters])
        cluster_ok = set("C" + str(n) for n in REQUIRED_CLUSTERS_NUM).issubset(set_labels) or set(
            str(n) for n in REQUIRED_CLUSTERS_NUM
        ).issubset(set_labels)
# # #         # Non-redundancy comes from cluster_audit if present  # Module not found  # Module not found  # Module not found
        non_redundant = bool(cluster_audit.get("non_redundant", False))
        if not non_redundant:
            gaps.append("redundant_items")
        # Evidence linkage check at micro level
        micro_ev_detected = False
        if isinstance(micro, dict):
            micro_ev_detected = True
            for c, payload in micro.items():
                answers = payload.get("answers") or []
                if answers and not payload.get("evidence_linked"):
                    # Fallback: try to detect evidence_ids directly
                    eids = _collect_evidence_ids_from_answers(answers)
                    if not eids:
                        micro_ev_detected = False
                        break
        else:
            micro_ev_detected = False

        # Completeness check per cluster
        if not bool(cluster_audit.get("complete")):
            gaps.append("incomplete_cluster_execution")
    else:
        # If we lack cluster_audit, we cannot confirm four clusters
        gaps.append("missing_cluster_audit")

    if not cluster_ok:
        gaps.append("missing_required_clusters")

    # DNP standards & scoring calibration checks
    uses_dnp = False
    dnp_fields = ("dnp_validation_results", "dnp_compliance", "dnp_alignment")
    for k in dnp_fields:
        if k in out or (isinstance(out.get("report"), dict) and k in out.get("report", {})):
            uses_dnp = True
            break

    # Look for causal-correction signals
    causal_signals = False
    for k in ("causal_correction", "causal_graph", "causal_adjusted_scores", "causal_dnp"):
        if k in out or (isinstance(out.get("report"), dict) and k in out.get("report", {})):
            causal_signals = True
            break

    # Calibration/causal correction proxy: if divergence high and coverage low => flag recalibration
    meso = out.get("meso_summary") or {}
    div = meso.get("divergence_stats") or {}
    max_div = float(div.get("max", 0.0))
    avg_div = float(div.get("avg", 0.0))
    recalibration_required = (max_div > 0.5 and avg_div > 0.25)

    # Compute macro synthesis
    macro = out.get("macro_synthesis") or _compute_macro_synthesis(out)
    out["macro_synthesis"] = macro

    # Raw data sufficiency checks (best-effort)
    raw_presence = {
        "features": isinstance(out.get("features"), dict),
        "vectors": isinstance(out.get("vectors"), list) and len(out.get("vectors") or []) > 0,
        "bm25_index": bool(out.get("bm25_index")),
        "vector_index": bool(out.get("vector_index")),
        "evidence_store": bool(out.get("evidence")) or bool(out.get("evidence_system")),
    }

    # Evidence resolution against evidence store
    unresolved_evidence: List[str] = []
    try:
        import json as _json
    except Exception:
        _json = None  # type: ignore

    evidence_index_by_qid: Dict[str, set] = {}
    # Try evidence_system first
    evsys = out.get("evidence_system")
    if evsys is not None:
        try:
            if hasattr(evsys, "serialize_canonical") and callable(evsys.serialize_canonical):
                raw = evsys.serialize_canonical()
                if isinstance(raw, str) and _json:
                    doc = _json.loads(raw)
                    store = doc.get("store", {})
                    for qid, items in store.items():
                        ids = set()
                        for it in items or []:
                            eid = it.get("id")
                            if isinstance(eid, str):
                                ids.add(eid)
                        evidence_index_by_qid[qid] = ids
        except Exception:
            pass
    # Fallback: evidence dict directly
    if not evidence_index_by_qid and isinstance(out.get("evidence"), dict):
        try:
            for qid, items in (out.get("evidence") or {}).items():
                ids = set()
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            eid = it.get("id") or it.get("evidence_id")
                            if isinstance(eid, str):
                                ids.add(eid)
                evidence_index_by_qid[qid] = ids
        except Exception:
            pass

    # Walk micro answers and resolve evidence ids
    if isinstance(cluster_audit, dict):
        micro = cluster_audit.get("micro") or {}
        for c, payload in (micro or {}).items():
            answers = payload.get("answers") or []
            unresolved_count = 0
            checked = 0
            for a in answers:
                if not isinstance(a, dict):
                    continue
                qid = str(a.get("question_id") or a.get("question") or "")
                eids = a.get("evidence_ids") or []
                if isinstance(eids, list):
                    for eid in eids:
                        if isinstance(eid, str):
                            checked += 1
                            universe = evidence_index_by_qid.get(qid) or set()
                            if universe and eid not in universe:
                                unresolved_evidence.append(f"{qid}:{eid}")
                                unresolved_count += 1
            per_cluster_resolution[c] = {
                "checked": checked,
                "unresolved": unresolved_count,
            }

    if unresolved_evidence:
        gaps.append("unresolved_evidence_ids")

    # Compute additional gaps based on presence of reporting levels and raw sufficiency
    if not out.get("cluster_audit"):
        gaps.append("missing_micro_level")
    if not out.get("meso_summary"):
        gaps.append("missing_meso_summary")
    if not out.get("macro_synthesis"):
        gaps.append("missing_macro_synthesis")
    if not uses_dnp:
        gaps.append("missing_dnp_alignment")
    if not causal_signals:
        gaps.append("missing_causal_correction_signal")
    if not raw_presence.get("evidence_store"):
        gaps.append("insufficient_raw_evidence_store")

    # Replicability hash on key outputs
    replicability = {
        "cluster_audit_hash": _stable_hash_dict(out.get("cluster_audit", {})) if out.get("cluster_audit") else None,
        "meso_summary_hash": _stable_hash_dict(out.get("meso_summary", {})) if out.get("meso_summary") else None,
        "macro_synthesis_hash": _stable_hash_dict(macro) if macro else None,
    }

    # External transformation path availability
    try:
        import public_transformer_adapter as _pta  # type: ignore
        public_adapter_available = hasattr(_pta, "process")
    except Exception:
        public_adapter_available = False
        gaps.append("missing_public_adapter")

    # Calibration advisory
    calibration_trigger: Optional[Dict[str, Any]] = None
    if recalibration_required or not causal_signals:
        calibration_trigger = {
            "reason": "high_divergence_or_missing_causal_correction" if recalibration_required else "missing_causal_correction",
            "action": "invoke_adaptive_recalibration_and_update_scores",
            "metrics": {"max_div": max_div, "avg_div": avg_div},
        }

    # Mathematical enhancers default-programming check
    math_info = out.get("mathematical_enhancements") or {}
    math_programmed_by_default = bool(math_info.get("applied", False))
    if not math_programmed_by_default:
        gaps.append("math_enhancers_not_programmed_by_default")

    canonical_audit = {
        "four_clusters_confirmed": bool(cluster_ok),
        "clusters_complete": bool(cluster_audit.get("complete")) if isinstance(cluster_audit, dict) else False,
        "non_redundant": bool(cluster_audit.get("non_redundant")) if isinstance(cluster_audit, dict) else False,
        "evidence_linked_micro": bool(micro_ev_detected),
        "uses_dnp_standards": bool(uses_dnp),
        "causal_correction_signals": bool(causal_signals),
        "recalibration_required": bool(recalibration_required),
        "reporting_levels": {
            "micro": bool(out.get("cluster_audit")),
            "meso": bool(out.get("meso_summary")),
            "macro": bool(out.get("macro_synthesis")),
        },
        "raw_data_presence": raw_presence,
        "replicability": replicability,
        "public_adapter_available": public_adapter_available,
        "mathematical_enhancers": {
            "programmed_by_default": math_programmed_by_default,
            "applied": math_programmed_by_default,
        },
        "evidence_traceability": {
            "per_cluster": per_cluster_resolution,
            "unresolved_sample": unresolved_evidence[:10],
        },
        "calibration_trigger": calibration_trigger,
        "gaps": sorted(set(gaps)),
        "timestamp": time.time(),
    }

    out["canonical_audit"] = canonical_audit
    return out
