"""
Snapshot Manager (SC — Snapshot Contract)

Provides:
- Global snapshot_id per run (Merkle-like root over corpus+indices+standards digests).
- Read-only resolver by snapshot_id; rejects if missing or checksum diverges.
- Decorator to enforce requires_snapshot=True for handlers.
- Deterministic, canonical serialization for replay.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

_lock = threading.Lock()
_current_snapshot_id: Optional[str] = None
_registry: Dict[str, Dict[str, Any]] = {}


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple, set)):
        return sorted(
            [_canonical(x) for x in obj],
            key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")),
        )
    return obj


def _canon_json(data: Dict[str, Any]) -> str:
    return json.dumps(
        _canonical(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def compute_snapshot_id(
    *, corpus_digest: str, index_digest: str, standards_digest: str
) -> str:
    payload = {
        "corpus": corpus_digest,
        "indices": index_digest,
        "standards": standards_digest,
    }
    return hashlib.sha256(_canon_json(payload).encode("utf-8")).hexdigest()


def mount_snapshot(state: Dict[str, Any]) -> str:
    """Mount a snapshot for this run. Returns snapshot_id. Thread-safe, idempotent for identical state."""
    global _current_snapshot_id
    with _lock:
        # Expect state has digests; if not, compute minimal ones deterministically
        corpus_digest = str(
            state.get("corpus_digest")
            or hashlib.sha256(
                _canon_json(state.get("corpus", {})).encode("utf-8")
            ).hexdigest()
        )
        index_digest = str(
            state.get("index_digest")
            or hashlib.sha256(
                _canon_json(state.get("indices", {})).encode("utf-8")
            ).hexdigest()
        )
        standards_digest = str(
            state.get("standards_digest")
            or hashlib.sha256(
                _canon_json(state.get("standards", {})).encode("utf-8")
            ).hexdigest()
        )
        sid = compute_snapshot_id(
            corpus_digest=corpus_digest,
            index_digest=index_digest,
            standards_digest=standards_digest,
        )
        # Store read-only copy
        payload = {
            "snapshot_id": sid,
            "corpus_digest": corpus_digest,
            "index_digest": index_digest,
            "standards_digest": standards_digest,
            "frozen": _canonical(
                {
                    "corpus": state.get("corpus", {}),
                    "indices": state.get("indices", {}),
                    "standards": state.get("standards", {}),
                }
            ),
        }
        _registry[sid] = payload
        _current_snapshot_id = sid
        return sid


def get_current_snapshot_id() -> Optional[str]:
    return _current_snapshot_id


def resolve_snapshot(snapshot_id: str) -> Dict[str, Any]:
    """Return a read-only view for the snapshot. Raises if absent."""
    if snapshot_id not in _registry:
        raise KeyError(f"Snapshot not found: {snapshot_id}")
    # Return a deep-ish read-only dict (prevent mutation by returning canonical JSON parsed back)
    data = _registry[snapshot_id]
    # Verify checksum consistency
    expected = compute_snapshot_id(
        corpus_digest=data["corpus_digest"],
        index_digest=data["index_digest"],
        standards_digest=data["standards_digest"],
    )
    if expected != snapshot_id:
        raise ValueError("Snapshot checksum divergence detected")
    # Provide frozen projection
    frozen_json = _canon_json(data["frozen"])  # canonical form for replay
    return {"snapshot_id": snapshot_id, "frozen_json": frozen_json}


def requires_snapshot(handler: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator enforcing that a snapshot is mounted before handler execution and marks requires_snapshot=True."""
    setattr(handler, "requires_snapshot", True)

    def wrapper(*args, **kwargs):
        sid = get_current_snapshot_id()
        if not sid:
            raise RuntimeError("No snapshot mounted")
        # Pass snapshot_id to handler if it accepts it
        return handler(*args, snapshot_id=sid, **kwargs)

    return wrapper


def replay_output(
    snapshot_id: str, func: Callable[..., Any], *, inputs: Dict[str, Any]
) -> str:
    """Run func under fixed snapshot and return canonical JSON of outputs for byte-for-byte replay comparison."""
    resolved = resolve_snapshot(snapshot_id)
    # Freeze inputs canonically
    payload = {
        "snapshot_id": snapshot_id,
        "inputs": inputs,
        "frozen": resolved["frozen_json"],
    }
    output = func(**inputs)
    return _canon_json({"payload": payload, "output": output})
