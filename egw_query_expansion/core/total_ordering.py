"""
Total Ordering System (Deterministic and Reproducible)

This module implements a total ordering comparator and sorting utilities inspired by
Haslbeck & Nipkow (2022): "Formally Verified Comparison Algorithms for Partial Orders"
(Journal of Automated Reasoning). While not a formal proof inside this codebase, the
design follows the same discipline: define a total comparator that refines equivalence
classes (by score vectors) into a strict total order using a deterministic, lexicographic
chain of tie-breakers over unique identifier tuples.

Goals satisfied:
- Total comparator over score vectors with deterministic lexicographic tie-break over
  tuples of unique identifiers (UIDs) to refine partial equivalences to a total order.
- Stability: sorting is stable and preserves relative order of items that are exactly
  equal under the full comparator (scores and UIDs).
- Avoids nondeterministic reductions on parallel hardware by ensuring a pure comparator
  and an explicit, serializable total key.
- Canonical serializers that preserve order regardless of storage format/OS/CPU.
- Exact reproducibility across OS/architectures by using canonical JSON encoding and
  explicit total keys that include the full tie-break chain.
- Tie-break chain documentation via explain_comparison / explain_tie_breaks.

API overview:
- make_total_key(scores, uids, *, descending=True) -> tuple: build total order key.
- total_compare(a, b) -> int: comparator over (scores, uids) pairs.
- sort_total(items, get_scores, get_uids, *, descending=True, stable=True) -> list: stable sorting.
- serialize_ordered(items, get_scores, get_uids, *, descending=True) -> str: canonical JSON.
- explain_comparison(scores_a, uids_a, scores_b, uids_b, *, descending=True) -> dict
- explain_tie_breaks(items, get_scores, get_uids, *, descending=True) -> dict

Notes on numeric handling for reproducibility:
- Floating values are normalized:
  - NaN values are mapped to a dedicated sentinel that sorts worse than all numbers.
  - +inf and -inf are preserved with deterministic order.
  - -0.0 is normalized to 0.0 for ordering equality, but the UID tie-break ensures a
    deterministic outcome when necessary.
- The comparator is pure-Python and avoids any platform-dependent math.

This module is self-contained and does not change other pipelines. Import and use
where deterministic total ordering is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite, isnan
from typing import Any, Callable, Iterable, List, Mapping, Sequence, Tuple, TypeVar

# Local canonicalization helpers (duplicated minimal logic to avoid hard dependency)
# We keep these minimal to avoid circular imports; deterministic_shield also provides
# similar utilities but this keeps the module standalone.


def _to_canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_canonical(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (set, tuple, list)):
        # Only canonicalize content, don't reorder lists/tuples to preserve ordered data
        canon_list = [_to_canonical(x) for x in obj]
        # For sets, we still need to sort for determinism
        if isinstance(obj, set):
            return sorted(
                canon_list,
                key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")),
            )
        return canon_list
    return obj


def canonical_json(data: Mapping[str, Any]) -> str:
    canon = _to_canonical(dict(data))
    return json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------- Key construction ---------------------------- #

# The key is constructed to be a total order key:
# - First component: normalized score vector with direction (descending/ascending).
# - Second component: UID tuple (as strings) for lexicographic tie-break.
# The result is a tuple that Python can compare lexicographically in a deterministic way.

Score = float
UID = str


def _normalize_score(x: Any) -> Tuple[float, bool, bool, bool]:
    """
    Normalize numeric score for deterministic ordering.
    Returns a tuple: (value, is_nan, is_pos_inf, is_neg_inf)
    - value is finite float (with -0.0 normalized to 0.0) or 0.0 if non-finite
    - booleans indicate special categories
    """
    try:
        xf = float(x)
    except Exception:
        return (0.0, True, False, False)

    if isnan(xf):
        return (0.0, True, False, False)
    if xf == float("inf"):
        return (0.0, False, True, False)
    if xf == float("-inf"):
        return (0.0, False, False, True)
    if not isfinite(xf):
        return (0.0, True, False, False)
    if xf == 0.0:
        xf = 0.0
    return (xf, False, False, False)


def _directional_value(
    norm: Tuple[float, bool, bool, bool], descending: bool
) -> Tuple[int, float]:
    val, is_nan, is_pos_inf, is_neg_inf = norm
    # Establish category rank so that smaller rank sorts first (Python tuple ordering)
    if descending:
        # Best to worst: +inf (0), finite (1), -inf (2), NaN (3)
        if is_pos_inf:
            cat = 0
            adj = 0.0
        elif not (is_nan or is_neg_inf):
            cat = 1
            adj = -val  # flip for descending
        elif is_neg_inf:
            cat = 2
            adj = 0.0
        else:  # NaN
            cat = 3
            adj = 0.0
    else:
        # Best to worst ascending: -inf (0), finite (1), +inf (2), NaN (3)
        if is_neg_inf:
            cat = 0
            adj = 0.0
        elif not (is_nan or is_pos_inf):
            cat = 1
            adj = val
        elif is_pos_inf:
            cat = 2
            adj = 0.0
        else:  # NaN
            cat = 3
            adj = 0.0
    return (cat, adj)


def make_total_key(
    scores: Sequence[Score], uids: Sequence[UID], *, descending: bool = True
) -> Tuple[Tuple[int, float], ...]:
    """
    Build a total key for lexicographic comparison.
    - scores: vector of numeric scores.
    - uids: tuple/list of unique identifiers used for deterministic tie-breaking.
    - descending: True -> higher scores come first.

    The key is a tuple composed of:
    - normalized and direction-adjusted score entries (per coordinate), each as (category, value)
    - a final separator marker
    - normalized UID entries (as strings) to break ties lexicographically

    Using categories ensures that NaN sorts consistently (worst), and +/-inf are ordered.
    """
    norm_scores = tuple(
        _directional_value(_normalize_score(s), descending) for s in scores
    )
    # Use a distinct separator type: a single-element tuple marker never collides with score pairs
    sep = ((99, 0.0),)  # arbitrary marker greater than any category we use for scores
    norm_uids = tuple(
        (3, float("inf")) for _ in ()
    )  # placeholder to keep types consistent
    # For UID tie-breaks, we append their string forms as separate comparable items after the separator.
    # Python compares tuples element-wise; mixing types is allowed as long as positions are consistent.
    # We will embed UIDs as a parallel tuple after scores using an inner tuple marker to ensure order.
    uid_section = tuple((4, 0.0, str(u)) for u in uids)
    return (*norm_scores, *sep, *uid_section)


# ------------------------------- Comparator ------------------------------- #


def total_compare(
    a: Tuple[Sequence[Score], Sequence[UID]],
    b: Tuple[Sequence[Score], Sequence[UID]],
    *,
    descending: bool = True,
) -> int:
    """Return -1 if a<b, 0 if equal, 1 if a>b under the total order defined by scores then UIDs."""
    ka = make_total_key(a[0], a[1], descending=descending)
    kb = make_total_key(b[0], b[1], descending=descending)
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


# ------------------------------- Sorting ---------------------------------- #

T = TypeVar("T")


def sort_total(
    items: Sequence[T],
    get_scores: Callable[[T], Sequence[Score]],
    get_uids: Callable[[T], Sequence[UID]],
    *,
    descending: bool = True,
    stable: bool = True,
) -> List[T]:
    """
    Deterministically sort items using the total order. Python's sort is already stable.
    - items: list/sequence of items
    - get_scores(item): returns score vector
    - get_uids(item): returns UID tuple/list
    - descending: higher scores first by default
    - stable: kept for API clarity; Python's Timsort is stable regardless
    """
    keys = [
        (make_total_key(get_scores(it), get_uids(it), descending=descending), idx, it)
        for idx, it in enumerate(items)
    ]
    # Include original index as a last resort to preserve stability if keys fully equal (identical)
    keys.sort(key=lambda x: (x[0], x[1]))
    return [it for _, _, it in keys]


# -------------------------- Canonical serialization ------------------------ #


def serialize_ordered(
    items: Sequence[T],
    get_scores: Callable[[T], Sequence[Score]],
    get_uids: Callable[[T], Sequence[UID]],
    *,
    descending: bool = True,
) -> str:
    """
    Serialize the ordered items into canonical JSON that preserves the order and is
    independent of platform.
    Output format:
    {
      "order": [
         {"rank": 0, "uids": [..], "scores": [..]},
         ...
      ]
    }
    """
    ordered = sort_total(items, get_scores, get_uids, descending=descending)
    order_list = []
    for rank, it in enumerate(ordered):
        order_list.append(
            {
                "rank": int(rank),
                "uids": [str(u) for u in get_uids(it)],
                "scores": [float(s) if s is not None else None for s in get_scores(it)],
            }
        )
    return canonical_json({"order": order_list})


# ------------------------- Tie-break chain explanation --------------------- #


def explain_comparison(
    scores_a: Sequence[Score],
    uids_a: Sequence[UID],
    scores_b: Sequence[Score],
    uids_b: Sequence[UID],
    *,
    descending: bool = True,
) -> Mapping[str, Any]:
    """
    Provide a human-readable, deterministic explanation of how two items are ordered.
    Returns a dict with the decision and the applied tie-break steps.
    """
    steps: List[Mapping[str, Any]] = []
    # Compare coordinate-wise
    for i, (sa, sb) in enumerate(zip(scores_a, scores_b)):
        na = _normalize_score(sa)
        nb = _normalize_score(sb)
        va = _directional_value(na, descending)
        vb = _directional_value(nb, descending)
        if va != vb:
            winner = (
                "A" if va < vb else "B"
            )  # lower key wins (because descending flips sign)
            steps.append(
                {
                    "type": "score",
                    "dimension": i,
                    "descending": bool(descending),
                    "A": {"raw": sa, "norm": na, "dir": va},
                    "B": {"raw": sb, "norm": nb, "dir": vb},
                    "decision": f"{winner} wins on score[{i}]",
                }
            )
            return {"result": -1 if winner == "A" else 1, "steps": steps}
        else:
            steps.append(
                {
                    "type": "score",
                    "dimension": i,
                    "descending": bool(descending),
                    "A": {"raw": sa, "norm": na, "dir": va},
                    "B": {"raw": sb, "norm": nb, "dir": vb},
                    "decision": "tie",
                }
            )
    # If all score dims equal under normalization, use UID lexicographic tie-break
    for j, (ua, ub) in enumerate(zip(uids_a, uids_b)):
        sa = str(ua)
        sb = str(ub)
        if sa != sb:
            winner = "A" if sa < sb else "B"
            steps.append(
                {
                    "type": "uid",
                    "position": j,
                    "A": sa,
                    "B": sb,
                    "decision": f"{winner} wins on uid[{j}] lexicographic",
                }
            )
            return {"result": -1 if winner == "A" else 1, "steps": steps}
        else:
            steps.append(
                {
                    "type": "uid",
                    "position": j,
                    "A": sa,
                    "B": sb,
                    "decision": "tie",
                }
            )
    # All equal
    steps.append({"type": "final", "decision": "identical under comparator"})
    return {"result": 0, "steps": steps}


def explain_tie_breaks(
    items: Sequence[T],
    get_scores: Callable[[T], Sequence[Score]],
    get_uids: Callable[[T], Sequence[UID]],
    *,
    descending: bool = True,
) -> Mapping[str, Any]:
    """
    For items with identical normalized score vectors, show which UID positions are used
    to break ties. Returns a mapping with groups and applied tie-break positions.
    """
    # Group by normalized scores
    from collections import defaultdict

    def norm_scores_key(scores: Sequence[Score]) -> Tuple[Tuple[int, float], ...]:
        return tuple(
            _directional_value(_normalize_score(s), descending) for s in scores
        )

    groups: Mapping[
        Tuple[Tuple[int, float], ...], List[Tuple[List[str], T]]
    ] = defaultdict(list)
    for it in items:
        ns = norm_scores_key(get_scores(it))
        groups[ns].append(([str(u) for u in get_uids(it)], it))

    report: dict = {"groups": []}
    for ns_key, members in groups.items():
        # Determine the first UID position that breaks ties among members (if any)
        used_positions: List[int] = []
        if len(members) > 1:
            max_len = max(len(uids) for uids, _ in members)
            for pos in range(max_len):
                vals = [uids[pos] if pos < len(uids) else "" for uids, _ in members]
                if len(set(vals)) > 1:
                    used_positions.append(pos)
                    # We continue to list all positions that could break at any comparison
        report["groups"].append(
            {
                "normalized_scores": ns_key,
                "size": len(members),
                "uid_positions_used_for_tiebreak": used_positions,
            }
        )
    return report


# ------------------------------- Data model ------------------------------- #


@dataclass(frozen=True)
class OrderedRecord:
    """Simple data model for convenience; optional to use.
    - scores: score vector
    - uids: tuple/list of unique identifiers (must be deterministic strings when cast)
    - payload: any additional data to carry along
    """

    scores: Tuple[Score, ...]
    uids: Tuple[UID, ...]
    payload: Any = None
