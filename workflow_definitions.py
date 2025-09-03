"""
Unified Workflow Definitions (Schema 2.0) and Normalizer

This module defines a stable, minimal schema and a normalize(definition) function
that upcasts legacy definitions to schema "2.0", canonicalizes metadata, validates
contracts, checks acyclicity (topological order), and freezes the effective per-step
metadata via hashing for traceability.

Schema (per step):
- step_id: str
- handler: str|Callable (string recommended)
- depends_on: list[str]
- metadata: dict with canonical keys and semantics (see METADATA_KEYS)

The output of normalize(definition) is a dict:
{
  "schema": "2.0",
  "steps": [
     {
       "step_id": ..., "handler": ..., "depends_on": [...],
       "metadata": { ... canonical ... },
       "effective_metadata": { ... defaults applied ... },
       "effective_metadata_hash": "sha256(...)",
     }, ...
  ],
  "topological_order": [step_id...],
  "selected_step_ids": [step_id...],   # deterministic selection for tests (budget>=cost)
  "digest": "sha256(canonical_json(output))"
}

Note: This file intentionally avoids any execution engine/queues/schedulers and only
captures the intent of workflows as stable data with validation.
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import hashlib
import json
# # # from typing import Any, Dict, List, Set, Tuple  # Module not found  # Module not found  # Module not found

# ---------------- Canonicalization helpers ---------------- #


def _to_canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_canonical(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple, set)):
        canon = [_to_canonical(x) for x in obj]
        return sorted(
            canon, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"))
        )
    return obj


def canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(
        _to_canonical(dict(data)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


# ---------------- Schema, defaults, and legacy mapping ---------------- #

DEFAULTS: Dict[str, Any] = {
    "commutative": True,
    "associative": True,
    "ordered": False,
    "seed": None,
    "idempotent": True,
    "join_semantics": "gset",  # default conservative set semantics
    "backoff": "expo",
    "revision": 1,
    "deprecated": False,
    "timeout_s": 60,
    "resources": {"cpu": 0.0, "mem_mb": 0},
}

# Keys that belong under metadata if encountered at step level (legacy upcast)
METADATA_KEYS: Set[str] = set(
    [
        # Determinism
        "commutative",
        "associative",
        "ordered",
        "seed",
        # Routing / snapshot / ordering
        "routing_key",
        "routing_hash",
        "snapshot_id",
        "score_vector",
        # Budget & monotonicity
        "cost",
        "gain_fn_id",
        "budget",
        # IO/state / CRDT
        "input",
        "output",
        "idempotent",
        "join_semantics",
        # Risk
        "risk",  # dict
        # OT alignment
        "ot",  # dict with reg, max_iter, tol, seed
        # Logic compliance
        "rules_uri",
        "labels_poset",
        # Fallbacks and failures
        "fallback",
        "error_map",
        # Traceability
        "trace_id_strategy",
        "evidence_keys",
        # Versioning
        "revision",
        "deprecated",
        # Resources/time
        "timeout_s",
        "resources",
        # Guards/flags
        "preconditions",
        "requires_snapshot",
    ]
)

LEGACY_MAP: Dict[str, str] = {
    # Common legacy fields to move under metadata with new names if needed
    "dependencies": "depends_on",
    "id": "step_id",
    "name": None,  # ignored at schema level
    "timeout_seconds": "timeout_s",
}


# ---------------- Utility: Topological order and validation ---------------- #


def _toposort(steps: List[Dict[str, Any]]) -> List[str]:
    graph: Dict[str, Set[str]] = {}
    indeg: Dict[str, int] = {}
    for st in steps:
        sid = st["step_id"]
        deps = list(st.get("depends_on") or [])
        graph.setdefault(sid, set())
        indeg.setdefault(sid, 0)
        for d in deps:
            graph.setdefault(d, set()).add(sid)
            indeg[sid] = indeg.get(sid, 0) + 1
            indeg.setdefault(d, indeg.get(d, 0))
    # Kahn's algorithm
    order: List[str] = []
    queue = sorted([n for n, deg in indeg.items() if deg == 0])
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in sorted(graph.get(n, set())):
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
                queue.sort()
    if len(order) != len(indeg):
        raise ValueError("Cycle detected in workflow steps (acyclicity violated)")
    return order


def _freeze_hash(obj: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _normalize_step_legacy(step: Dict[str, Any]) -> Dict[str, Any]:
    # Rename legacy keys
    st = dict(step)
    if "dependencies" in st and "depends_on" not in st:
        st["depends_on"] = st.pop("dependencies")
    if "id" in st and "step_id" not in st:
        st["step_id"] = st.pop("id")
    if "timeout_seconds" in st and "timeout_s" not in st:
        st["timeout_s"] = st.pop("timeout_seconds")
    # Ensure required fields
    st.setdefault("step_id", str(st.get("step_id") or st.get("name") or ""))
    st.setdefault("handler", st.get("handler") or "")
    st.setdefault("depends_on", list(st.get("depends_on") or []))
    # Move known metadata-like fields under metadata
    md = dict(st.get("metadata") or {})
    for k in list(st.keys()):
        if k in METADATA_KEYS:
            md.setdefault(k, st.pop(k))
    st["metadata"] = md
    return st


def _validate_contracts(eff: Dict[str, Any]) -> None:
    # commutative=false ⇒ ordered=true
    if eff.get("commutative") is False and eff.get("ordered") is False:
        raise ValueError("If commutative=false then ordered must be true")
    # requires_snapshot ⇒ snapshot_id present
    if eff.get("requires_snapshot"):
        if not eff.get("snapshot_id"):
            raise ValueError(
                "requires_snapshot=true but snapshot_id is missing in metadata"
            )
    # If join_semantics!="multiset" ⇒ ignore multiplicity (no action; just ensure no 'multiplicity' field leaks)
    if eff.get("join_semantics") != "multiset":
        if "multiplicity" in eff:
            eff.pop("multiplicity", None)
    # If budget present ⇒ cost and gain_fn_id required
    if eff.get("budget") is not None:
        if eff.get("cost") is None or eff.get("gain_fn_id") in (None, ""):
            raise ValueError("budget specified but cost and gain_fn_id are required")
    # If rules_uri ⇒ labels_poset obligatory
    if eff.get("rules_uri") and not eff.get("labels_poset"):
        raise ValueError("rules_uri provided but labels_poset is missing")


def _effective_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    # Apply defaults without overwriting present keys (deep for resources)
    eff = dict(DEFAULTS)
    eff.update({k: v for k, v in md.items() if v is not None})
    # Ensure resources has both keys
    res = dict(DEFAULTS["resources"])
    res.update(eff.get("resources") or {})
    eff["resources"] = res
    return eff


def _canonicalize_depends_on(
    depends_on: List[str], eff_md: Dict[str, Any]
) -> List[str]:
    deps = list(dict.fromkeys(depends_on or []))  # unique preserving order
    # If commutative/associative, order does not matter => sort to canonicalize
    if eff_md.get("commutative", True) and eff_md.get("associative", True):
        return sorted(deps)
    return deps


# ---------------- Public API ---------------- #


def normalize(definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a workflow definition to schema "2.0" with canonical metadata and
    validations. Returns a normalized dict as described in the module docstring.
    """
    if not isinstance(definition, dict):
        raise TypeError("definition must be a dict")

    schema = str(definition.get("schema") or "2.0")
    steps_in = definition.get("steps") or []
    if schema != "2.0":
        # Upcast legacy by forcing schema and processing steps
        schema = "2.0"

    # Normalize steps
    norm_steps: List[Dict[str, Any]] = []
    for step in steps_in:
        st = _normalize_step_legacy(dict(step))
        eff = _effective_metadata(st.get("metadata") or {})
        _validate_contracts(eff)
        # Canonicalize depends_on based on commutativity/associativity
        st["depends_on"] = _canonicalize_depends_on(st.get("depends_on", []), eff)
        # Freeze effective metadata and attach
        st["effective_metadata"] = eff
        st["effective_metadata_hash"] = _freeze_hash(eff)
        norm_steps.append(
            {
                "step_id": str(st["step_id"]),
                "handler": st.get("handler"),
                "depends_on": st.get("depends_on", []),
                "metadata": st.get("metadata", {}),
                "effective_metadata": eff,
                "effective_metadata_hash": st["effective_metadata_hash"],
            }
        )

    # Topological order and acyclicity
    topo = _toposort(norm_steps)

    # Deterministic selected_step_ids for tests: select steps whose budget >= cost
    selected: List[str] = []
    for sid in topo:
        st = next(s for s in norm_steps if s["step_id"] == sid)
        md = st["effective_metadata"]
        budget = md.get("budget")
        cost = md.get("cost")
        if budget is not None and cost is not None and float(budget) >= float(cost):
            selected.append(sid)

    out = {
        "schema": "2.0",
        "steps": norm_steps,
        "topological_order": topo,
        "selected_step_ids": selected,
    }
    out["digest"] = _freeze_hash(out)
    return out
