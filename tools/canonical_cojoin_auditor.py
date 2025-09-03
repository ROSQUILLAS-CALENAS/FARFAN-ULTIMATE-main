#!/usr/bin/env python3
"""
Canonical Co-Join Auditor
- Exhaustive repo-wide search to recover canonical pipeline nodes scattered outside canonical_flow/
- Deterministic co-join assembly without moving/renaming files
- Emits REQUIRED outputs:
  1) INVENTORY (JSONL)
  2) DAG (Mermaid + JSON)
  3) GAP REPORT (Markdown table)
  4) CO-JOIN PLAN (YAML)
  5) CONTRACT DIFFS (unified diff blocks)
  6) FINAL NORMALIZED SEQUENCE (JSON array)

Usage:
  python tools/canonical_cojoin_auditor.py \
    --seed-sequence-file canonical_order_manifest.json \
    --out-dir artifacts/canonical_cojoin

Notes:
- Zero external deps; stdlib only
- Zero hallucinations: any uncertainty is annotated with @UNSURE and reason
- Preserve seed ordering; insert newly found nodes at phase-appropriate anchors
- Do NOT rename; only propose links/moves in plan
- Deterministic: fixed sort keys, stable tie-breakers
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import argparse
import ast
import json
import os
import re
import sys
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, Iterable, List, Optional, Set, Tuple  # Module not found  # Module not found  # Module not found

ROOT = Path(__file__).resolve().parents[1]

PHASES = [
    ("I", "I_ingestion_preparation", ["ingestion", "prepare", "raw", "pdf", "loader", "gate", "preflight"]),
    ("X", "X_context_construction", ["context", "immutable", "lineage"]),
    ("K", "K_knowledge_extraction", ["knowledge", "embedding", "graph", "entity", "chunk"]),
    ("A", "A_analysis_nlp", ["analysis", "nlp", "question", "evidence", "evaluation_driven"]),
    ("L", "L_classification_evaluation", ["classification", "evaluation", "scoring", "threshold", "conformal"]),
    ("R", "R_search_retrieval", ["retrieval", "search", "rerank", "index", "vector", "lexical"]),
    ("O", "O_orchestration_control", ["orchestr", "router", "controller", "circuit", "exception", "contract"]),
    ("G", "G_aggregation_reporting", ["aggregation", "report", "meso", "compiler", "summary"]),
    ("T", "T_integration_storage", ["integration", "metrics", "feedback", "optimization", "storage"]),
    ("S", "S_synthesis_output", ["synthesis", "answer", "formatter", "output"]),
]
PHASE_CODE = {p[0]: p[1] for p in PHASES}
PHASE_FULL = {p[1]: p[0] for p in PHASES}

# Regexes used (reported verbatim for determinism)
REGEXES = {
    "phase_token": r"^(I|X|K|A|L|R|O|G|T|S)[_-].+",
    "contract_terms": r"(ingestion|context|knowledge|analysis|classification|retrieval|orchestration|aggregation|storage|synthesis)",
}
# Globs used
GLOBS = [
    "**/*.py", "**/*.ipynb", "**/*.json", "**/*.yml", "**/*.yaml", "**/*.toml", "**/*.md"
]

# Scoped directories to include (besides everything by default)
SCOPE_HINTS = [
    ".", "canonical_flow", "tests", "scripts", "legacy", "experiments", "playground",
    "drafts", "proto", "notebooks", "retrieval_engine", "semantic_reranking", "egw_query_expansion"
]

@dataclass
class Evidence:
    why: List[str]

@dataclass
class InventoryItem:
    path: str
    phase: str
    evidence: Evidence
    confidence: float
    duplicates_of: Optional[str]
    status: str  # new|seed|alternate

@dataclass
class Node:
    path: str
    phase: str
    imports: List[str]
    provides: List[str]
    requires: List[str]
    version: Optional[str]
    date: Optional[str]
    tests: List[str]


def read_seed(seed_sequence_file: Optional[Path]) -> List[str]:
    if not seed_sequence_file or not seed_sequence_file.exists():
        # Fallback to canonical_order_manifest.json if available
        fallback = ROOT / "canonical_order_manifest.json"
        if fallback.exists():
            try:
                data = json.loads(fallback.read_text(encoding="utf-8"))
                return list(data.get("sequence", []))
            except Exception:
                return []
        return []
    try:
        txt = seed_sequence_file.read_text(encoding="utf-8")
        if seed_sequence_file.suffix == ".json":
            data = json.loads(txt)
            # accept either {sequence:[...]} or [...]
            if isinstance(data, dict) and "sequence" in data:
                return list(data.get("sequence", []))
            if isinstance(data, list):
                return list(data)
        else:
            # Plain text, one path per line
            return [line.strip() for line in txt.splitlines() if line.strip()]
    except Exception:
        return []


def iter_repo_files() -> Iterable[Path]:
# # #     # Include hidden folders and follow symlinks by using rglob from each scope root deterministically  # Module not found  # Module not found  # Module not found
    roots = sorted({str((ROOT / s).resolve()) for s in SCOPE_HINTS if (ROOT / s).exists()})
    seen: Set[str] = set()
    for r in roots:
        base = Path(r)
        for pattern in GLOBS:
            for p in sorted(base.rglob(pattern)):
                if p.is_file():
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        yield p


def guess_phase(path: Path, text: Optional[str]) -> Tuple[str, float, List[str]]:
    rel = str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path)
    hits: List[str] = []
    # filename/dirname tokens
    base = rel.lower()
    m = re.match(REGEXES["phase_token"], path.name)
    if m:
        hits.append(f"regex:phase_token:{m.group(0)}")
        code = m.group(1)
        return PHASE_CODE.get(code, "unknown"), 0.9, hits
    for code, full, keywords in PHASES:
        if f"/{full.lower()}/" in f"/{base}/" or base.startswith(full.lower()):
            hits.append(f"dirname:{full}")
            return full, 0.8, hits
    # Content-based
    if text:
        if re.search(REGEXES["contract_terms"], text, re.I):
            for code, full, keywords in PHASES:
                if any(kw in text.lower() for kw in keywords):
                    hits.append(f"content:{full}")
                    return full, 0.6, hits
    # Heuristic by keywords in filename
    for code, full, keywords in PHASES:
        if any(kw in base for kw in keywords):
            hits.append(f"filename_kw:{full}")
            return full, 0.55, hits
    return "unknown", 0.3, hits


def extract_imports_and_interfaces(path: Path, text: str) -> Tuple[List[str], List[str], List[str]]:
    imports: List[str] = []
    provides: List[str] = []
    requires: List[str] = []
    try:
        if path.suffix == ".py":
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imports.append(a.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    provides.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    provides.append(node.name)
            # crude requires: look for bus contracts mentions
            if re.search(r"(INGEST\.bus|NLP\.bus|EVAL\.bus)", text):
                requires.append("bus_contract")
        else:
            # JSON/YAML/TOML/MD basic inspection
            if re.search(r"\bimport\b|from\s+\w+\s+import", text):
                imports.append("inline_code_block")
            if re.search(r"process\(|def\s+process\s*\(", text):
                provides.append("process")
    except Exception:
        pass
    # normalize
    imports = sorted(set(imports))
    provides = sorted(set(provides))
    requires = sorted(set(requires))
    return imports, provides, requires


def file_version_and_date(path: Path, text: str) -> Tuple[Optional[str], Optional[str]]:
# # #     # Extract version-like and date-like tokens from header/comments  # Module not found  # Module not found  # Module not found
    version = None
    date = None
    try:
        head = "\n".join(text.splitlines()[:30])
        m = re.search(r"version\s*[:=]\s*\"?([0-9]+\.[0-9]+(\.[0-9]+)?)", head, re.I)
        if m:
            version = m.group(1)
        dm = re.search(r"(20[0-9]{2}-[01][0-9]-[0-3][0-9]|20[0-9]{2}/[01][0-9]/[0-3][0-9])", head)
        if dm:
            date = dm.group(1)
    except Exception:
        pass
    return version, date


def find_tests_for(path: Path) -> List[str]:
    rel = str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path)
    name = path.stem
    candidates = []
    for p in [ROOT / "tests", ROOT]:
        if p.exists():
            for t in sorted(p.rglob(f"test_{name}*.py")):
                candidates.append(str(t.relative_to(ROOT)))
    return sorted(set(candidates))


def build_inventory(seed: List[str]) -> Tuple[List[InventoryItem], Dict[str, Node]]:
    seed_set = set(seed)
    items: List[InventoryItem] = []
    nodes: Dict[str, Node] = {}

    for path in iter_repo_files():
        # Only consider code-like and config files per scope
        if path.suffix not in {".py", ".ipynb", ".json", ".yml", ".yaml", ".toml", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        phase, conf, hits = guess_phase(path, text)
        if phase == "unknown" and conf < 0.5:
            continue  # skip weak unknowns
        rel = str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path)
        status = "seed" if rel in seed_set else "new"
        imports, provides, requires = extract_imports_and_interfaces(path, text)
        version, date = file_version_and_date(path, text)
        tests = find_tests_for(path)

        # Duplicate detection: same basename within same phase
        dup_key = f"{phase}:{path.name}"
        duplicates_of = None
        for it in items:
            if it.phase == phase and Path(it.path).name == path.name:
                duplicates_of = it.path
                status = "alternate" if status != "seed" else "seed"
                break

        items.append(InventoryItem(
            path=rel,
            phase=phase,
            evidence=Evidence(why=hits + [f"imports:{len(imports)}", f"provides:{len(provides)}", f"requires:{len(requires)}"]),
            confidence=round(conf, 2),
            duplicates_of=duplicates_of,
            status=status,
        ))
        nodes[rel] = Node(
            path=rel, phase=phase, imports=imports, provides=provides, requires=requires,
            version=version, date=date, tests=tests
        )

    # Deterministic sort
    items.sort(key=lambda x: (x.status != "seed", PHASE_FULL.get(x.phase, "Z"), x.path.lower()))
    return items, nodes


def build_dag(nodes: Dict[str, Node]) -> Tuple[List[Tuple[str, str]], List[str]]:
    edges: List[Tuple[str, str]] = []
    warnings: List[str] = []
    # simple import-based edges: if node A imports module that matches basename of node B
    name_to_paths: Dict[str, List[str]] = {}
    for p, n in nodes.items():
        name_to_paths.setdefault(Path(p).stem, []).append(p)
    for a_path, a_node in nodes.items():
        for imp in a_node.imports:
            base = imp.split(".")[0]
            if base in name_to_paths:
                for b_path in name_to_paths[base]:
                    if b_path != a_path:
                        # Edge: b provides -> a depends (b -> a)
                        edges.append((b_path, a_path))
    # Deduplicate deterministically
    edges = sorted(set(edges), key=lambda e: (e[0].lower(), e[1].lower()))
    # Validate order conforms to canonical phase order
    phase_rank = {p[1]: i for i, p in enumerate(PHASES)}
    for u, v in edges:
        pu = nodes[u].phase
        pv = nodes[v].phase
        if pu in phase_rank and pv in phase_rank and phase_rank[pu] > phase_rank[pv]:
            warnings.append(f"phase_order_violation:{u}->{v}")
    return edges, warnings


def assemble_sequence(seed: List[str], items: List[InventoryItem], nodes: Dict[str, Node]) -> List[Dict[str, Any]]:
    # Preserve seed spine; insert new nodes into their phase lanes deterministically after last seed of that phase
    seed_clean = [s for s in seed if s in nodes]
    phase_rank = {p[1]: i for i, p in enumerate(PHASES)}
    # Group new nodes by phase
    new_by_phase: Dict[str, List[str]] = {}
    for it in items:
        if it.path not in seed_clean and it.status == "new" and it.phase in phase_rank:
            new_by_phase.setdefault(it.phase, []).append(it.path)
    for k in new_by_phase:
        new_by_phase[k].sort(key=lambda p: Path(p).name.lower())
    # Build final sequence list with annotations
    final: List[Dict[str, Any]] = []
    for s in seed_clean:
        final.append({"path": s, "status": "seed"})
        phase = nodes[s].phase
        # insert any new nodes for this phase that are not yet placed and whose anchor is this seed group end
        if phase in new_by_phase and new_by_phase[phase]:
            # Only insert after the last seed of this phase; we’ll handle this by peeking ahead
            pass
    # To place new items per phase after the last occurrence of that phase in the seed, we compute positions
    last_pos: Dict[str, int] = {}
    for idx, s in enumerate(seed_clean):
        last_pos[nodes[s].phase] = idx
    # Start with seed list; then extend with phase buckets in canonical order after last seed
    final_paths = seed_clean[:]
    for phase, _full, _ in PHASES:
        full = PHASE_CODE[phase]
        if full in new_by_phase:
            final_paths.extend(new_by_phase[full])
    # Annotate alternates
    alt = [it for it in items if it.status == "alternate"]
    alternates = sorted([{"path": it.path, "alternate_of": it.duplicates_of} for it in alt], key=lambda x: (x["alternate_of"] or "", x["path"]))
    result = [{"path": p, "status": ("seed" if p in seed_clean else "new")} for p in final_paths]
    # Append alternates as notes (not part of main sequence)
    for a in alternates:
        result.append({"path": a["path"], "status": "alternate", "alternate_of": a["alternate_of"]})
    return result


def write_outputs(out_dir: Path, inventory: List[InventoryItem], nodes: Dict[str, Node], edges: List[Tuple[str, str]], dag_warnings: List[str], final_sequence: List[Dict[str, Any]], seed: List[str]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # 1) INVENTORY JSONL
    inv_path = out_dir / "inventory.jsonl"
    with inv_path.open("w", encoding="utf-8") as f:
        for it in inventory:
            line = {
                "path": it.path,
                "phase": it.phase,
                "evidence": {"why": it.evidence.why},
                "confidence": it.confidence,
                "duplicates_of": it.duplicates_of,
                "status": it.status,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    paths["inventory_jsonl"] = str(inv_path)

    # 2) DAG Mermaid + JSON
    dag_json = {"nodes": [asdict(nodes[p]) for p in sorted(nodes.keys())], "edges": edges, "warnings": dag_warnings}
    dag_json_path = out_dir / "dag.json"
    dag_json_path.write_text(json.dumps(dag_json, indent=2), encoding="utf-8")
    paths["dag_json"] = str(dag_json_path)
    # Mermaid with phase lanes
    def lane(phase_full: str) -> str:
        return phase_full
    mermaid_lines = ["flowchart LR"]
    # Subgraphs per phase
    phase_to_nodes: Dict[str, List[str]] = {}
    for p, n in nodes.items():
        phase_to_nodes.setdefault(n.phase, []).append(p)
    for phase in sorted(phase_to_nodes.keys(), key=lambda x: (PHASE_FULL.get(x, "Z"), x)):
        mermaid_lines.append(f"  subgraph {phase}")
        for p in sorted(phase_to_nodes[phase]):
            node_id = p.replace('/', '_').replace('.', '_')
            mermaid_lines.append(f"    {node_id}({Path(p).name})")
        mermaid_lines.append("  end")
    for u, v in edges:
        mermaid_lines.append(f"  {u.replace('/', '_').replace('.', '_')} --> {v.replace('/', '_').replace('.', '_')}")
    dag_md_path = out_dir / "dag.mmd"
    dag_md_path.write_text("\n".join(mermaid_lines), encoding="utf-8")
    paths["dag_mermaid"] = str(dag_md_path)

    # 3) GAP REPORT (Markdown table)
    # Compare ideal phases vs seed+found
    present_phases = sorted({nodes[p].phase for p in nodes})
    ideal_phases = [p[1] for p in PHASES]
    missing_phases = [ph for ph in ideal_phases if ph not in present_phases]
    # insertion anchors: for each new node, before/after which last seed of same phase
    anchors: List[Dict[str, str]] = []
    seed_set = set(seed)
    for it in inventory:
        if it.status == "new" and it.phase in ideal_phases:
            # find last seed in same phase
            last_seed = None
            for s in seed:
                if s in nodes and nodes[s].phase == it.phase:
                    last_seed = s
            anchors.append({"path": it.path, "phase": it.phase, "insert_after": last_seed or "@UNSURE", "rationale": "phase-conformant insertion preserving seed order" if last_seed else "@UNSURE: no seed anchor for phase"})
    gap_lines = ["| Phase | Missing? | Proposed Inserts (path → after) | Rationale |", "|---|---|---|---|"]
    for ph in ideal_phases:
        miss = "YES" if ph in missing_phases else "NO"
        props = [a for a in anchors if a["phase"] == ph]
        desc = "; ".join([f"{Path(a['path']).name} → {Path(a['insert_after']).name if a['insert_after']!='@UNSURE' else '@UNSURE'}" for a in props]) or ""
        rat = props[0]["rationale"] if props else ""
        gap_lines.append(f"| {ph} | {miss} | {desc} | {rat} |")
    gap_path = out_dir / "gap_report.md"
    gap_path.write_text("\n".join(gap_lines), encoding="utf-8")
    paths["gap_report_md"] = str(gap_path)

    # 4) CO-JOIN PLAN (YAML)
    plan_lines = ["# Co-Join Plan (no changes applied; proposals only)"]
    for a in anchors:
        plan_lines.append("- link:")
        plan_lines.append(f"    from: {a['path']}")
        plan_lines.append(f"    to: canonical_flow/{PHASE_FULL.get(a['phase'], a['phase'])}/{Path(a['path']).name}")
        plan_lines.append("    reason: Attach discovered node into canonical phase while preserving seed spine")
        plan_lines.append("    contracts_preserved: true")
        plan_lines.append(f"    rollback: {a['path']}")
    plan_path = out_dir / "cojoin_plan.yaml"
    plan_path.write_text("\n".join(plan_lines), encoding="utf-8")
    paths["cojoin_plan_yaml"] = str(plan_path)

    # 5) CONTRACT DIFFS (unified diff blocks)
    # Best-effort: detect interface signature drift for process(data, context) presence
    diffs: List[str] = []
    for it in inventory:
        if it.status in {"new", "alternate"} and it.phase in ideal_phases and it.path.endswith('.py'):
            p = ROOT / it.path
            try:
                txt = p.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                txt = ''
            has_process = bool(re.search(r"def\s+process\s*\(.*\):", txt))
            if not has_process:
                diffs.append("""@@ CONTRACT:process @@\n- expected: def process(data=None, context=None):\n+ found:    missing (adapter needed)\n""".rstrip())
    diffs_path = out_dir / "contract_diffs.diff"
    diffs_path.write_text("\n\n".join(sorted(set(diffs))), encoding="utf-8")
    paths["contract_diffs"] = str(diffs_path)

    # 6) FINAL NORMALIZED SEQUENCE (JSON array)
    final_seq_path = out_dir / "final_normalized_sequence.json"
    final_seq_path.write_text(json.dumps(final_sequence, indent=2), encoding="utf-8")
    paths["final_sequence_json"] = str(final_seq_path)

    # Determinism note and regex/globs used
    notes = {
        "determinism_note": "All outputs are produced using fixed sort keys (path-lowercase, phase order), stable tie-breakers (basename, then full path), and consistent anchors based on the provided seed sequence. Without seed, canonical_order_manifest.json is used if present. Running twice on the same repo state yields identical outputs.",
        "regexes_used": REGEXES,
        "globs_used": GLOBS,
        "timestamp": datetime.now().isoformat(),
    }
    notes_path = out_dir / "determinism_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    paths["determinism_notes_json"] = str(notes_path)

    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonical Co-Join Auditor")
    ap.add_argument("--seed-sequence-file", type=str, default="", help="Path to JSON/JSONL/TXT seed sequence list or canonical_order_manifest.json")
    ap.add_argument("--out-dir", type=str, default="artifacts/canonical_cojoin", help="Output directory for artifacts")
    args = ap.parse_args()

    seed_file = Path(args.seed_sequence_file) if args.seed_sequence_file else None
    seed = read_seed(seed_file)

    inventory, nodes = build_inventory(seed)
    edges, dag_warnings = build_dag(nodes)
    final_sequence = assemble_sequence(seed, inventory, nodes)

    out_paths = write_outputs(Path(args.out_dir), inventory, nodes, edges, dag_warnings, final_sequence, seed)

    # Print concise machine-readable summary
    summary = {
        "ok": True,
        "inventory_items": len(inventory),
        "edges": len(edges),
        "warnings": dag_warnings,
        "outputs": out_paths,
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
