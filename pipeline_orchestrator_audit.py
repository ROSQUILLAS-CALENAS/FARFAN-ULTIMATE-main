"""
Pipeline Orchestrator Audit

This helper analyzes PIPELINEORCHESTRATOR.py without importing it (since that file
uses illustrative REPL markers). It extracts the declared process graph entries
(file_path keys and dependencies) and reports:
- Objective of the orchestrator (as described).
- List of required modules/files with existence flags.
- Validation report: missing files, dangling dependencies, cycles.

Run:
  python3 pipeline_orchestrator_audit.py > orchestrator_audit.json
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import json
import os
import re
# # # from typing import Any, Dict, List, Set, Tuple  # Module not found  # Module not found  # Module not found

ROOT = os.getcwd()
PIPELINE_FILE = os.path.join(ROOT, "PIPELINEORCHESTRATOR.py")


def get_objective() -> str:
    return (
        "Chain all pipeline modules (ingestion ➜ context ➜ knowledge ➜ analysis ➜ "
        "classification ➜ routing ➜ retrieval ➜ orchestration ➜ validation ➜ synthesis ➜ integration) "
        "as a single DAG, execute them in topological order, ensure each step adds value, and provide a full trace."
    )


def _parse_graph_from_source(src: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Parse PIPELINEORCHESTRATOR.py text to extract node names, file_path, and dependencies.
    Returns (nodes, errors)
    nodes[node_name] = {"file_path": str, "dependencies": [str]}
    """
    errors: List[str] = []
    nodes: Dict[str, Dict[str, Any]] = {}

    # Heuristic: entries look like
    #    "filename.py": ProcessNode(
    #        file_path="filename.py",
    #        ...
    #        dependencies=["a.py", "b.py"],
    #
    entry_pattern = re.compile(r"\s*\"(?P<key>[^\"]+)\"\s*:\s*ProcessNode\(.*?\)", re.S)

    for m in entry_pattern.finditer(src):
        block = m.group(0)
        key = m.group("key")
        # file_path
        fp_match = re.search(r"file_path\s*=\s*\"([^\"]+)\"", block)
        file_path = fp_match.group(1) if fp_match else key
        # dependencies
        deps_match = re.search(r"dependencies\s*=\s*\[([^\]]*)\]", block)
        deps: List[str] = []
        if deps_match:
            raw = deps_match.group(1).strip()
            if raw:
                # split on commas and strip quotes/spaces
                for tok in raw.split(","):
                    tok = tok.strip()
                    if tok.startswith('"') and tok.endswith('"'):
                        deps.append(tok[1:-1])
        nodes[key] = {"file_path": file_path, "dependencies": deps}

    if not nodes:
# # #         errors.append("No nodes parsed from PIPELINEORCHESTRATOR.py")  # Module not found  # Module not found  # Module not found
    return nodes, errors


def _detect_cycles(nodes: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {k: WHITE for k in nodes}
    cycles: List[Tuple[str, str]] = []

    def dfs(u: str):
        color[u] = GRAY
        for v in nodes[u]["dependencies"]:
            if v not in nodes:
                continue
            if color[v] == WHITE:
                dfs(v)
            elif color[v] == GRAY:
                cycles.append((u, v))
        color[u] = BLACK

    for k in nodes:
        if color[k] == WHITE:
            dfs(k)
    return cycles


def list_required_modules_and_functions(project_root: str = ROOT) -> Dict[str, Any]:
    with open(PIPELINE_FILE, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()
    nodes, parse_errors = _parse_graph_from_source(src)

    items = []
    for node_name, info in nodes.items():
        fp = info["file_path"]
        candidate = fp if fp.endswith(".py") else fp
        exists = os.path.isfile(os.path.join(project_root, candidate))
        items.append(
            {
                "node": node_name,
                "file_path": candidate,
                "exists": bool(exists),
                "expected_functions": ["process", "main"],
                "dependencies": list(info["dependencies"]),
            }
        )

    return {
        "objective": get_objective(),
        "modules": items,
        "parse_errors": parse_errors,
    }


def validate(project_root: str = ROOT) -> Dict[str, Any]:
    with open(PIPELINE_FILE, "r", encoding="utf-8", errors="ignore") as f:
        src = f.read()
    nodes, parse_errors = _parse_graph_from_source(src)

    report: Dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "summary": {},
        "parse_errors": parse_errors,
    }

    # Missing files
    missing = []
    for node_name, info in nodes.items():
        fp = info["file_path"]
        if not os.path.isfile(os.path.join(project_root, fp)):
            missing.append({"node": node_name, "file_path": fp})
    if missing:
        report["errors"].append({"type": "missing_files", "details": missing})

    # Dangling dependencies
    dangling = []
    node_keys = set(nodes.keys())
    for node_name, info in nodes.items():
        for dep in info["dependencies"]:
            if dep not in node_keys:
                dangling.append({"node": node_name, "dependency": dep})
    if dangling:
        report["errors"].append({"type": "dangling_dependencies", "details": dangling})

    # Cycles
    cycles = _detect_cycles(nodes)
    if cycles:
        report["errors"].append({"type": "cycles", "details": cycles})

    report["summary"] = {
        "total_nodes": len(nodes),
        "missing_files": len(missing),
        "dangling_deps": len(dangling),
        "cycles": len(cycles),
    }
    return report


if __name__ == "__main__":
    audit = {
        "objective": get_objective(),
        "required": list_required_modules_and_functions(),
        "validation": validate(),
    }
    print(json.dumps(audit, indent=2, ensure_ascii=False))
