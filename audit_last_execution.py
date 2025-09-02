#!/usr/bin/env python3
"""
Audit Last Execution — Step-by-Step
- Loads the most recent execution artifacts (execution_trace.json preferred).
- Extracts per-step statuses, timestamps, and errors.
- Runs canonical_output_auditor on available results to validate four clusters,
  evidence linkage, DNP usage, causal correction signals, and reporting levels.
- Produces enhancement suggestions based on gaps and detected runtime errors.
- Writes audit_last_execution.json and prints a concise human-readable summary.

Usage:
  python3 audit_last_execution.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional imports; the auditor degrades gracefully
try:
    import canonical_output_auditor as coa  # type: ignore
except Exception:  # pragma: no cover
    coa = None  # type: ignore

try:
    import pipeline_orchestrator_audit as poa  # type: ignore
except Exception:  # pragma: no cover
    poa = None  # type: ignore

ROOT = Path(os.getcwd())
CANDIDATE_FILES = [
    ROOT / "execution_trace.json",
    ROOT / "orchestrator_audit.json",
    ROOT / "processing_results.json",
    ROOT / "governance_report.json",
    ROOT / "development_plan_analysis_report.json",
]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _pick_latest_existing(files: List[Path]) -> Optional[Path]:
    existing = [p for p in files if p.exists() and p.is_file()]
    if not existing:
        return None
    # Prefer execution_trace.json if present; otherwise pick most recently modified
    for p in existing:
        if p.name == "execution_trace.json":
            return p
    return max(existing, key=lambda p: p.stat().st_mtime)


def _extract_trace(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize different execution trace shapes into a common structure.
    Returns dict with keys: steps (list), result (dict|None)
    """
    steps: List[Dict[str, Any]] = []
    result_payload: Optional[Dict[str, Any]] = None

    # pipeline_orchestrator format: { 'result': {...}, 'execution_trace': [ ... ], 'canonical_audit': {...} }
    if isinstance(doc.get("execution_trace"), list) or isinstance(doc.get("result"), dict):
        steps = list(doc.get("execution_trace") or [])
        res = doc.get("result")
        if isinstance(res, dict):
            result_payload = res

    # comprehensive_pipeline_orchestrator format: the whole document is the result; step logs may be embedded
    if not steps:
        # Try to parse steps heuristically from known keys
        # Some orchestrators store 'logs' or 'trace' or 'timeline'
        for k in ("steps", "trace", "timeline", "events"):
            v = doc.get(k)
            if isinstance(v, list):
                steps = v
                break

    # Alternative layouts: 'final_output' indicates result-like structure
    if result_payload is None:
        if isinstance(doc.get("final_output"), dict):
            # Prefer 'final_output' payload for auditing
            result_payload = doc["final_output"]
        else:
            # As a last resort, use doc itself
            if isinstance(doc, dict):
                result_payload = doc

    return {"steps": steps, "result": result_payload}


def _suggest_enhancements(audit: Dict[str, Any], steps: List[Dict[str, Any]], doc: Dict[str, Any]) -> List[str]:
    suggestions: List[str] = []
    gaps = list(audit.get("gaps") or [])

    # Map gaps to actionable suggestions
    gap_actions = {
        "missing_cluster_audit": "Integrar cluster_execution_controller.process para construir 'cluster_audit' antes del agregado.",
        "missing_required_clusters": "Asegurar ejecución de 4 clústeres (C1..C4) y proporcionar 'clusters' + 'cluster_answers'.",
        "incomplete_cluster_execution": "Rellenar respuestas por clúster y reintentar; todas deben tener al menos un ítem.",
        "redundant_items": "Eliminar duplicados por 'question_id' dentro de cada clúster.",
        "unresolved_evidence_ids": "Sincronizar 'evidence_ids' con EvidenceSystem.serialize_canonical(); usar IDs canónicos.",
        "missing_micro_level": "Proveer 'cluster_audit.micro' (por clúster) con respuestas y evidencias.",
        "missing_meso_summary": "Ejecutar meso_aggregator.process tras el micro para generar divergencias.",
        "missing_macro_synthesis": "Calcular 'macro_synthesis' (p.ej. via canonical_output_auditor).",
        "missing_dnp_alignment": "Incluir 'dnp_alignment'/'dnp_compliance'/'dnp_validation_results' en el payload.",
        "missing_causal_correction_signal": "Activar módulo de corrección causal (causal_graph/causal_adjusted_scores).",
        "insufficient_raw_evidence_store": "Adjuntar 'evidence_system' o 'evidence' con IDs canónicos por pregunta.",
        "missing_public_adapter": "Usar public_transformer_adapter.process para el reporte pedagógico.",
    }
    for g in gaps:
        if g in gap_actions:
            suggestions.append(gap_actions[g])

    # Parse runtime errors from steps or the document
    runtime_errors: List[str] = []
    for s in steps:
        if isinstance(s, dict):
            err = s.get("error") or s.get("exception")
            if isinstance(err, str) and err.strip():
                runtime_errors.append(err)
    # Heuristic: common error fields
    for k in ("error", "reason", "exception"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            runtime_errors.append(v)

    # Add targeted suggestions for known errors
    for err in runtime_errors:
        if "pm4py" in err.lower():
            suggestions.append("Instalar dependencia 'pm4py' o proveer stub de importación para evitar fallos en integración de procesos.")
        if ">=" in err and "int" in err and "str" in err:
            suggestions.append("Corregir comparación tipada en evaluador (evitar comparar int con str); normalizar tipos antes de comparar.")

    # Calibration advice passthrough
    calib = audit.get("calibration_trigger")
    if isinstance(calib, dict):
        reason = calib.get("reason")
        if reason:
            suggestions.append(f"Recalibración sugerida: {reason} (ejecutar motor adaptativo y actualizar puntajes).")

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for s in suggestions:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def audit_last_execution() -> Dict[str, Any]:
    picked = _pick_latest_existing(CANDIDATE_FILES)
    source_path = str(picked) if picked else None

    base_doc: Dict[str, Any] = {}
    if picked:
        loaded = _read_json(picked)
        if isinstance(loaded, dict):
            base_doc = loaded

    trace = _extract_trace(base_doc)
    steps = trace["steps"]
    result_payload = trace["result"]

    # Optionally enrich the payload before auditing to execute suggested improvements
    enriched_payload: Optional[Dict[str, Any]] = None
    try:
        if isinstance(result_payload, dict):
            from canonical_flow import enrichment_postprocessor as _enrich  # type: ignore
            enriched_candidate = _enrich.process(result_payload, context={"source": "audit_last_execution"})
            if isinstance(enriched_candidate, dict):
                enriched_payload = enriched_candidate
    except Exception:
        enriched_payload = None

    payload_for_audit = enriched_payload or (result_payload if isinstance(result_payload, dict) else {})

    # Run canonical audit if possible
    audit_out: Dict[str, Any] = {}
    audited_payload: Dict[str, Any] = {}
    if coa and isinstance(payload_for_audit, dict):
        try:
            audited_payload = coa.process(payload_for_audit, context={"source": "audit_last_execution"})
            audit_out = audited_payload.get("canonical_audit", {})
        except Exception:
            audit_out = {}

    # Also run static orchestrator graph audit if importable
    orchestrator_graph_validation: Optional[Dict[str, Any]] = None
    if poa:
        try:
            orchestrator_graph_validation = poa.validate()
        except Exception:
            orchestrator_graph_validation = None

    # Build per-step summary
    step_summaries: List[Dict[str, Any]] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        step_summaries.append(
            {
                "module": s.get("module") or s.get("name") or s.get("stage") or "?",
                "status": s.get("status") or s.get("state") or "unknown",
                "timestamp": s.get("timestamp") or s.get("time"),
                "error": s.get("error") or s.get("exception"),
            }
        )

    flow_ok = all((x.get("status") == "success" and not x.get("error")) for x in step_summaries) if step_summaries else True

    # Optionally run calibration controller to produce segmented plan and recalibrated scores
    calibration_payload: Dict[str, Any] = {}
    try:
        # Always produce a segmented calibration plan to increase sophistication and transparency
        from canonical_flow import calibration_controller as _cal  # type: ignore
        calibrated = _cal.process(payload_for_audit, context={"source": "audit_last_execution"}, steps=steps)
        if isinstance(calibrated, dict):
            calibration_payload = {
                "calibration_report": calibrated.get("calibration_report"),
                "calibrated_scores": calibrated.get("calibrated_scores"),
            }
            # Propagate back to audited_payload if available
            audited_payload.update(calibrated)
            audit_out = (audited_payload.get("canonical_audit") or audit_out)
    except Exception:
        calibration_payload = {}

    # Compose final report
    report: Dict[str, Any] = {
        "source_file": source_path,
        "generated_at": time.time(),
        "flow": {
            "steps_count": len(step_summaries),
            "flow_ok": bool(flow_ok),
        },
        "steps": step_summaries,
        "canonical_audit": audit_out,
        "orchestrator_graph_validation": orchestrator_graph_validation,
        "enhancements": _suggest_enhancements(audit_out, steps, base_doc),
        "applied_improvements": {
            "enrichment_applied": bool(enriched_payload),
            "enrichment_context": "audit_last_execution" if enriched_payload else None,
        },
        "calibration": calibration_payload,
    }

    # Persist
    out_path = ROOT / "audit_last_execution.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return report


def main() -> None:
    report = audit_last_execution()
    # Console summary
    print("Step-by-step Audit Summary:")
    print({
        "source": report.get("source_file"),
        "steps": report.get("flow", {}).get("steps_count"),
        "flow_ok": report.get("flow", {}).get("flow_ok"),
        "gaps": (report.get("canonical_audit", {}) or {}).get("gaps"),
        "enhancements": report.get("enhancements", [])[:5],
    })


if __name__ == "__main__":
    main()
