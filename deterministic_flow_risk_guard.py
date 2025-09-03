"""
Deterministic Flow Risk Guard

Purpose: Enhance interconnections of deterministic flow and proactively mitigate
error corruption risks as the project grows. This module is additive and
optional. If imported and used by orchestrators, it performs:

- Preflight checks on orchestrator wiring (modules/executors/dependencies).
- Lightweight runtime IO fingerprinting at step boundaries.
- Schema drift detection (shape/type signature changes across steps).
- Audit report generation (JSON-serializable dict) for CI and local audits.

Design constraints:
- No external dependencies beyond standard library.
- Safe to import in any environment.
- Zero side-effects unless explicitly invoked by host orchestrator.
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import hashlib
import json
import os
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple  # Module not found  # Module not found  # Module not found


def _stable_json_dumps(obj: Any) -> str:
    """Best-effort stable JSON serialization for hashing/fingerprinting.
    Falls back to repr if not JSON-serializable.
    """
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(obj)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _schema_signature(obj: Any) -> Dict[str, str]:
    """Compute a lightweight schema signature mapping keys to type names.
    For nested dicts/lists, it captures top-level structure and basic nesting types.
    """
    sig: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            if isinstance(v, dict):
                sig[str(k)] = "dict"
            elif isinstance(v, list):
                inner = set(type(x).__name__ for x in v[:3]) if v else set()
                sig[str(k)] = f"list<{','.join(sorted(inner))}>"
            else:
                sig[str(k)] = type(v).__name__
    else:
        sig["__root__"] = type(obj).__name__
    return sig


@dataclass
class RiskEvent:
    module: str
    stage: Optional[str]
    event: str
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RiskReport:
    preflight_issues: List[Dict[str, Any]] = field(default_factory=list)
    runtime_anomalies: List[RiskEvent] = field(default_factory=list)
    step_fingerprints: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preflight_issues": self.preflight_issues,
            "runtime_anomalies": [e.__dict__ for e in self.runtime_anomalies],
            "step_fingerprints": self.step_fingerprints,
            "stats": self.stats,
            "metadata": self.metadata,
        }


class DeterministicFlowRiskGuard:
    def __init__(self, strict: bool = False, write_artifact: bool = True, artifact_path: Optional[str] = None):
        self.strict = strict
        self.write_artifact = write_artifact
        self.artifact_path = artifact_path or os.path.join(os.getcwd(), "deterministic_flow_audit.json")
        self.report = RiskReport(metadata={
            "created_at": datetime.now().isoformat(),
            "strict": strict,
            "version": "1.0.0"
        })
        # Keep last schema signature to detect drift step-to-step
        self._last_schema: Optional[Dict[str, str]] = None
        self._last_output_hash: Optional[str] = None

    # ---------------------- Preflight ----------------------
    def preflight_checks(self, orchestrator: Any) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        try:
            modules = getattr(orchestrator, "modules", {}) or {}
            executors = getattr(orchestrator, "executors", {}) or {}
            # Missing executors
            for name in modules.keys():
                if name not in executors:
                    issues.append({
                        "module": name,
                        "issue": "Missing executor",
                        "severity": "ERROR"
                    })
            # Missing dependencies
            for name, module in modules.items():
                for dep in getattr(module, "dependencies", []) or []:
                    if dep not in modules:
                        issues.append({
                            "module": name,
                            "issue": f"Missing dependency: {dep}",
                            "severity": "WARNING"
                        })
            # Non-deterministic ordering risk: ensure stable iteration order notice
            # (PipelineOrchestrator._get_execution_order returns list(self.modules.keys()))
            # We record the order fingerprint for reference.
            order = list(modules.keys())
            self.report.stats["execution_order_fingerprint"] = _hash_text("||".join(order))
        except Exception as e:
            issues.append({"issue": f"Preflight error: {e}", "severity": "ERROR"})
        self.report.preflight_issues.extend(issues)
        return issues

    # ---------------------- Runtime Probes ----------------------
    def before_step(self, module: str, data_in: Any, stage: Optional[str] = None) -> None:
        try:
            schema = _schema_signature(data_in)
            payload = _stable_json_dumps(data_in)[:100000]  # cap for safety
            in_hash = _hash_text(payload)
            self.report.step_fingerprints.append({
                "module": module,
                "stage": stage,
                "event": "before",
                "schema": schema,
                "hash": in_hash,
                "timestamp": datetime.now().isoformat(),
            })
            # Detect sudden oscillation loop (repeated identical IO)
            if self._last_output_hash and self._last_output_hash == in_hash:
                self.report.runtime_anomalies.append(RiskEvent(
                    module=module,
                    stage=stage,
                    event="input_same_as_last_output",
                    details={"hash": in_hash}
                ))
        except Exception as e:
            self.report.runtime_anomalies.append(RiskEvent(
                module=module,
                stage=stage,
                event="before_step_probe_error",
                details={"error": str(e)}
            ))

    def after_step(self, module: str, data_out: Any, stage: Optional[str] = None) -> None:
        try:
            schema = _schema_signature(data_out)
            payload = _stable_json_dumps(data_out)[:100000]
            out_hash = _hash_text(payload)
            self.report.step_fingerprints.append({
                "module": module,
                "stage": stage,
                "event": "after",
                "schema": schema,
                "hash": out_hash,
                "timestamp": datetime.now().isoformat(),
            })
            # Schema drift detection (coarse): if top-level type signatures reduce unexpectedly
            if self._last_schema is not None:
                missing_keys = [k for k in self._last_schema.keys() if k not in schema.keys()]
                if missing_keys:
                    self.report.runtime_anomalies.append(RiskEvent(
                        module=module,
                        stage=stage,
                        event="schema_keys_missing",
                        details={"missing_keys": missing_keys}
                    ))
            self._last_schema = schema
            self._last_output_hash = out_hash
        except Exception as e:
            self.report.runtime_anomalies.append(RiskEvent(
                module=module,
                stage=stage,
                event="after_step_probe_error",
                details={"error": str(e)}
            ))

    # ---------------------- Finalize ----------------------
    def finalize_report(self, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if extra_metadata:
            self.report.metadata.update(extra_metadata)
        # Simple anomaly counts
        self.report.stats.update({
            "anomalies_count": len(self.report.runtime_anomalies),
            "preflight_issues_count": len(self.report.preflight_issues),
            "steps_fingerprinted": len(self.report.step_fingerprints),
        })
        result = self.report.to_dict()
        if self.write_artifact:
            try:
                with open(self.artifact_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return result


# Convenience function for CLI/script usage

def run_standalone_preflight(orchestrator: Any, strict: bool = False) -> Tuple[bool, Dict[str, Any]]:
    guard = DeterministicFlowRiskGuard(strict=strict, write_artifact=True)
    issues = guard.preflight_checks(orchestrator)
    report = guard.finalize_report(extra_metadata={"mode": "preflight_only"})
    ok = not issues if strict else True
    return ok, report
