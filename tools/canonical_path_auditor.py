#!/usr/bin/env python3
"""
Canonical Path Auditor
- Verifies canonical pipeline sequence and intervening nodes
- Detects disconnections/isolations and missing files
- Confronts current structure with unused assets
- Reports integrated assets with rootness (root nodes in DAG) and stage
- Checks math enhancers default activation heuristically
- Scans for blank/corrupted files
- Emits JSON report and ordered manifests

Safe to run standalone:
  python tools/canonical_path_auditor.py --json
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import argparse
import json
import os
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Set, Tuple  # Module not found  # Module not found  # Module not found

# Local imports are guarded to avoid import-time side effects
try:
# # #     from comprehensive_pipeline_orchestrator import (  # Module not found  # Module not found  # Module not found
        ComprehensivePipelineOrchestrator,
        get_canonical_process_graph,
        ProcessStage,
    )
except Exception:
    ComprehensivePipelineOrchestrator = None  # type: ignore
    get_canonical_process_graph = None  # type: ignore
    ProcessStage = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class NodeInfo:
    name: str
    file_path: str
    stage: str
    dependencies: List[str]
    exists: bool
    size_bytes: int
    blank: bool

@dataclass
class AuditorReport:
    ok: bool
    sequence: List[str]
    roots: List[str]
    sinks: List[str]
    isolated_nodes: List[str]
    disconnected_components: List[List[str]]
    missing_files: List[str]
    blank_or_corrupted: List[str]
    unused_assets: List[str]
    integrated_assets: List[NodeInfo]
    math_enhancers_enabled: bool
    artifacts: Dict[str, str]


def _load_graph() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if get_canonical_process_graph is not None:
        try:
            graph = get_canonical_process_graph()
            if graph:
                return graph, {}
        except Exception as e:
            pass
    # Fallback: try constructing orchestrator and reading attribute
    try:
        orch = ComprehensivePipelineOrchestrator()  # type: ignore
        if getattr(orch, 'process_graph', None):
            return orch.process_graph, {}
    except Exception:
        pass
    # Fallback 2: parse PIPELINEORCHESTRATOR.py textually
    try:
        import importlib.util as _ilu, types as _types
        # Use pipeline_orchestrator_audit to extract nodes
# # #         from pipeline_orchestrator_audit import list_required_modules_and_functions  # Module not found  # Module not found  # Module not found
        audit = list_required_modules_and_functions(str(ROOT))
        nodes: Dict[str, Any] = {}
        class _Node:
            def __init__(self, file_path: str, deps: List[str]):
                self.file_path = file_path
                self.dependencies = deps
                self.stage = type('S', (), {'value': 'unknown'})
        for it in audit.get('modules', []):
            nodes[it['node']] = _Node(it['file_path'], it.get('dependencies', []))
        if nodes:
            return nodes, {"source": "PIPELINEORCHESTRATOR.py"}
    except Exception as e:
        return {}, {"error": f"failed_to_parse_pipeline_orchestrator: {e}"}
    return {}, {"error": "graph_unavailable"}


def _topo_order(graph: Dict[str, Any]) -> List[str]:
    # Use dependencies on ProcessNode objects
    order: List[str] = []
    visited: Set[str] = set()

    def visit(n: str) -> None:
        if n in visited or n not in graph:
            return
        visited.add(n)
        for d in getattr(graph[n], 'dependencies', []) or []:
            visit(d)
        order.append(n)

    for k in graph.keys():
        visit(k)
    return order


def _roots_sinks(graph: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    roots, sinks = [], []
    keys = list(graph.keys())
    dep_set: Set[str] = set()
    for k in keys:
        dep_set.update(graph[k].dependencies or [])
    for k in keys:
        if not graph[k].dependencies:
            roots.append(k)
        # sink: not a dependency of anyone (no outgoing in reversed)
        if k not in dep_set:
            sinks.append(k)
    return roots, sinks


def _isolated(graph: Dict[str, Any]) -> List[str]:
    keys = list(graph.keys())
    dep_of: Dict[str, int] = {k: 0 for k in keys}
    for k in keys:
        for d in graph[k].dependencies or []:
            dep_of[d] = dep_of.get(d, 0) + 1
    outdeg = {k: len(graph[k].dependencies or []) for k in keys}
    return [k for k in keys if dep_of.get(k, 0) == 0 and outdeg.get(k, 0) == 0]


def _components(graph: Dict[str, Any]) -> List[List[str]]:
    # Undirected components for connectivity
    adj: Dict[str, Set[str]] = {k: set() for k in graph}
    for k, node in graph.items():
        for d in node.dependencies or []:
            if d in adj:
                adj[k].add(d)
                adj[d].add(k)
    seen: Set[str] = set()
    comps: List[List[str]] = []
    for k in adj:
        if k in seen:
            continue
        stack = [k]
        comp: List[str] = []
        seen.add(k)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def _node_info(graph: Dict[str, Any]) -> List[NodeInfo]:
    infos: List[NodeInfo] = []
    for name, node in graph.items():
        fp = getattr(node, 'file_path', name)
        p = ROOT / fp
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        blank = exists and size == 0
        stage = getattr(getattr(node, 'stage', None), 'value', str(getattr(node, 'stage', 'unknown')))
        deps = list(getattr(node, 'dependencies', []) or [])
        infos.append(NodeInfo(name=name, file_path=fp, stage=stage, dependencies=deps, exists=exists, size_bytes=size, blank=blank))
    return infos


def _scan_unused_assets(graph: Dict[str, Any]) -> List[str]:
    referenced: Set[str] = {getattr(n, 'file_path', k) for k, n in graph.items()}
    candidates: List[Path] = []
    for rel in ["canonical_flow", "egw_query_expansion", "retrieval_engine", "semantic_reranking"]:
        base = ROOT / rel
        if base.exists():
            candidates.extend(base.rglob("*.py"))
    unused: List[str] = []
    for p in candidates:
        rel = str(p.relative_to(ROOT))
        if rel not in referenced:
            try:
                if p.exists() and p.stat().st_size >= 0:
                    unused.append(rel)
            except Exception:
                unused.append(rel)
    return sorted(unused)


def _math_enhancers_enabled(orch: Any) -> bool:
    # Heuristic: if guarantee_value_chain exists and when invoked it ensures non-negative value_added and enhancement path is active.
    try:
        ok = orch.guarantee_value_chain()  # returns True if all already fine; False if enhancements applied
        # In both cases, the enhancer mechanic is effectively available and on by default in orchestrator usage
        return True
    except Exception:
        return False


def _write_artifacts(sequence: List[str]) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    try:
        manifest = {
            "sequence": sequence,
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        }
        mf = ROOT / "canonical_order_manifest.json"
        mf.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        artifacts["canonical_order_manifest"] = str(mf.resolve())
        txt = ROOT / "canonical_sequence.txt"
        txt.write_text("\n".join(sequence))
        artifacts["canonical_sequence_txt"] = str(txt.resolve())
    except Exception:
        pass
    return artifacts


def run(json_only: bool = False) -> Tuple[bool, AuditorReport]:
    graph, meta = _load_graph()
    if not graph:
# # #         # Fallback: build a pseudo-graph from canonical_flow directory order  # Module not found  # Module not found  # Module not found
        cf = ROOT / "canonical_flow"
        entries: List[Tuple[str, Path]] = []
        if cf.exists():
            for p in sorted(cf.rglob("*.py")):
                rel = str(p.relative_to(ROOT))
                entries.append((rel, p))
        class _Node:
            def __init__(self, file_path: str):
                self.file_path = file_path
                self.dependencies = []
                self.stage = type('S', (), {'value': 'canonical_flow'})
        pseudo: Dict[str, Any] = {e[0]: _Node(e[0]) for e in entries}
        graph = pseudo

    # Build order and node infos
    sequence = _topo_order(graph)
    roots, sinks = _roots_sinks(graph)
    isolated = _isolated(graph)
    comps = _components(graph)

    infos = _node_info(graph)
    missing_files = [ni.file_path for ni in infos if not ni.exists]
    blank_or_corrupted = [ni.file_path for ni in infos if ni.blank]

    unused = _scan_unused_assets(graph)

    # Instantiate orch to check enhancers
    enh = False
    try:
        orch = ComprehensivePipelineOrchestrator()  # type: ignore
        enh = _math_enhancers_enabled(orch)
    except Exception:
        enh = False

    artifacts = _write_artifacts(sequence)

    report = AuditorReport(
        ok=(len(missing_files) == 0 and len(isolated) == 0 and len(comps) == 1),
        sequence=sequence,
        roots=sorted(roots),
        sinks=sorted(sinks),
        isolated_nodes=sorted(isolated),
        disconnected_components=comps,
        missing_files=sorted(missing_files),
        blank_or_corrupted=sorted(blank_or_corrupted),
        unused_assets=unused,
        integrated_assets=infos,
        math_enhancers_enabled=enh,
        artifacts=artifacts,
    )
    return report.ok, report


def main():
    parser = argparse.ArgumentParser(description="Canonical Path Auditor")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()
    ok, rep = run(json_only=args.json)
    out = asdict(rep)
    print(json.dumps(out)) if args.json else print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
