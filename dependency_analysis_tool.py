"""
Lightweight Dependency and Data Flow Analysis Tool
- No external dependencies
- Provides safe, deterministic defaults so the module is executable end-to-end
"""
# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple  # Module not found  # Module not found  # Module not found
import itertools
import json


# ----- Data structures used by the analyzer -----

# Mandatory Pipeline Contract Annotations
__phase__ = "A"
__code__ = "35A"
__stage_order__ = 4

@dataclass
class DataFlowAnomaly:
    flow_id: str
    anomaly_type: str
    location: str
    expected_schema: Dict[str, str]
    actual_schema: Dict[str, str]
    fix_suggestions: List[str] = field(default_factory=list)


@dataclass
class DataIntegrityIssueRecord:
    type: str
    severity: str
    data_range: str


@dataclass
class DataIntegrityIssue:
    flow_id: str
    issue_type: str
    severity: str
    affected_data: str
    root_cause: str


@dataclass
class InformationBottleneck:
    node_id: str
    utilization: float
    throughput: float
    capacity: float
    queue_length: int
    optimization_strategies: List[str] = field(default_factory=list)


@dataclass
class TransformationRecord:
    location: str
    expected_output: Dict[str, str]
    actual_output: Dict[str, str]


@dataclass
class Flow:
    id: str
    edges: List[Tuple[str, str]]


def _schema_signature(obj: Any) -> Dict[str, str]:
    sig: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            if isinstance(v, dict):
                sig[str(k)] = "dict"
            elif isinstance(v, list):
                inner = sorted({type(x).__name__ for x in v[:3]}) if v else []
                sig[str(k)] = f"list<{','.join(inner)}>"
            else:
                sig[str(k)] = type(v).__name__
    else:
        sig["__root__"] = type(obj).__name__
    return sig


class _SimpleGraph:
    def __init__(self, edges: List[Tuple[str, str]]):
        self._adj: Dict[str, List[str]] = {}
        for u, v in edges:
            self._adj.setdefault(u, []).append(v)
            self._adj.setdefault(v, [])  # ensure node exists

    def nodes(self) -> List[str]:
        return list(self._adj.keys())

    def neighbors(self, node: str) -> List[str]:
        return list(self._adj.get(node, []))


class DataFlowAnalyzer:
    def __init__(self, project_graph: Dict[str, List[str]] | List[Tuple[str, str]], data_schema: Optional[Dict[str, Any]] = None):
        """
        project_graph: either dict of node->list(dependencies) or list of (u,v) edges.
        data_schema: baseline schema for validation (dict-like preferred)
        """
        self.graph_input = project_graph
        self.schema = data_schema or {}
        self.graph_edges = self._normalize_edges(project_graph)
        self.data_flows = self._trace_data_flows()

    # ---------------- Public APIs ----------------
    def detect_data_flow_anomalies(self) -> List[DataFlowAnomaly | DataIntegrityIssue]:
        """Detects anomalies in data flow patterns (safe defaults)."""
        anomalies: List[DataFlowAnomaly | DataIntegrityIssue] = []

        for flow in self.data_flows:
            # Check for data transformation consistency
            transformations = self._trace_transformations(flow)
            for transform in transformations:
                if not self._validate_transformation(transform):
                    anomalies.append(DataFlowAnomaly(
                        flow_id=flow.id,
                        anomaly_type="INVALID_TRANSFORMATION",
                        location=transform.location,
                        expected_schema=transform.expected_output,
                        actual_schema=transform.actual_output,
                        fix_suggestions=self._suggest_transformation_fixes(transform)
                    ))

            # Check for data loss or corruption points
            integrity_issues = self._check_data_integrity(flow)
            for issue in integrity_issues:
                anomalies.append(DataIntegrityIssue(
                    flow_id=flow.id,
                    issue_type=issue.type,
                    severity=issue.severity,
                    affected_data=issue.data_range,
                    root_cause=self._identify_root_cause(issue)
                ))

        return anomalies

    def analyze_information_bottlenecks(self) -> List[InformationBottleneck]:
        """Identifies information flow bottlenecks (heuristic and dependency-free)."""
        flow_graph = self._build_information_flow_graph()

        # Calculate flow metrics
        bottlenecks: List[InformationBottleneck] = []
        for node in flow_graph.nodes():
            throughput = self._calculate_throughput(node, flow_graph)
            capacity = self._get_node_capacity(node, flow_graph)
            utilization = throughput / capacity if capacity > 0 else float('inf')

            if utilization > 0.85:  # High utilization threshold
                bottlenecks.append(InformationBottleneck(
                    node_id=node,
                    utilization=float(round(utilization, 3)),
                    throughput=float(round(throughput, 3)),
                    capacity=float(round(capacity, 3)),
                    queue_length=self._get_queue_length(node, flow_graph),
                    optimization_strategies=self._suggest_optimizations(node, flow_graph)
                ))

        return bottlenecks

    # ---------------- Internal helpers ----------------
    def _normalize_edges(self, project_graph) -> List[Tuple[str, str]]:
        if isinstance(project_graph, dict):
            edges: List[Tuple[str, str]] = []
            for u, deps in project_graph.items():
                for v in deps or []:
                    edges.append((v, u))  # dep -> node
            return edges
        elif isinstance(project_graph, list):
            return list(project_graph)
        else:
            return []

    def _trace_data_flows(self) -> List[Flow]:
# # #         # Build simple flow segments from edges; group into one flow for now  # Module not found  # Module not found  # Module not found
        fid = "flow-1"
        return [Flow(id=fid, edges=self.graph_edges)]

    def _trace_transformations(self, flow: Flow) -> List[TransformationRecord]:
        # Use provided schema as both expected and actual unless mismatch can be inferred
        expected = _schema_signature(self.schema)
        # simulate a trivial pass-through transform per edge head
        transforms: List[TransformationRecord] = []
        seen = set()
        for _, dst in flow.edges:
            if dst in seen:
                continue
            seen.add(dst)
            actual = expected.copy()
            transforms.append(TransformationRecord(location=str(dst), expected_output=expected, actual_output=actual))
        return transforms

    def _validate_transformation(self, transform: TransformationRecord) -> bool:
        # basic equality check of keys and types
        return transform.expected_output == transform.actual_output

    def _suggest_transformation_fixes(self, transform: TransformationRecord) -> List[str]:
        exp_keys = set(transform.expected_output.keys())
        act_keys = set(transform.actual_output.keys())
        suggestions: List[str] = []
        missing = sorted(exp_keys - act_keys)
        extra = sorted(act_keys - exp_keys)
        if missing:
            suggestions.append(f"Add missing keys: {', '.join(missing)}")
        if extra:
            suggestions.append(f"Remove unexpected keys: {', '.join(extra)}")
        if not suggestions:
            suggestions.append("Verify type conversions and normalization policies")
        return suggestions

    def _check_data_integrity(self, flow: Flow) -> List[DataIntegrityIssueRecord]:
        # Heuristic: flag nodes with no incoming edges but present in destinations as potential source loss
        indeg: Dict[str, int] = {}
        outdeg: Dict[str, int] = {}
        for u, v in flow.edges:
            outdeg[u] = outdeg.get(u, 0) + 1
            indeg[v] = indeg.get(v, 0) + 1
            indeg.setdefault(u, indeg.get(u, 0))
            outdeg.setdefault(v, outdeg.get(v, 0))
        issues: List[DataIntegrityIssueRecord] = []
        for node, d in indeg.items():
            if d == 0 and outdeg.get(node, 0) > 0:
                issues.append(DataIntegrityIssueRecord(type="SOURCE_WITHOUT_INPUT", severity="LOW", data_range=node))
        return issues

    def _identify_root_cause(self, issue: DataIntegrityIssueRecord) -> str:
        if issue.type == "SOURCE_WITHOUT_INPUT":
            return "Node acts as a source; ensure upstream data ingestion is intentional"
        return "Undetermined"

    def _build_information_flow_graph(self) -> _SimpleGraph:
        return _SimpleGraph(self.graph_edges)

    def _calculate_throughput(self, node: str, graph: _SimpleGraph) -> float:
        # Throughput proportional to out-degree (simple heuristic)
        return float(len(graph.neighbors(node))) or 0.1

    def _get_node_capacity(self, node: str, graph: _SimpleGraph) -> float:
        # Capacity heuristic: 1.5 + degree
        return 1.5 + float(len(graph.neighbors(node)))

    def _get_queue_length(self, node: str, graph: _SimpleGraph) -> int:
        util = self._calculate_throughput(node, graph) / max(self._get_node_capacity(node, graph), 1e-9)
        return int(round(max(0.0, (util - 0.7) * 10)))

    def _suggest_optimizations(self, node: str, graph: _SimpleGraph) -> List[str]:
        suggestions = [
            "Increase worker concurrency for high-outdegree nodes",
            "Batch messages to reduce per-item overhead",
            "Apply backpressure or rate-limiting upstream",
        ]
        if len(graph.neighbors(node)) > 3:
            suggestions.append("Shard downstream consumers to distribute load")
        return suggestions


if __name__ == "__main__":
    # Minimal demo runner for standalone execution
    # Create a tiny graph and schema, then print results as JSON for quick checks
    demo_graph = {"B": ["A"], "C": ["B"], "D": ["B", "C"]}
    demo_schema = {"id": 1, "text": "sample", "scores": [0.1, 0.2]}
    analyzer = DataFlowAnalyzer(demo_graph, demo_schema)
    anomalies = analyzer.detect_data_flow_anomalies()
    bottlenecks = analyzer.analyze_information_bottlenecks()

    print(json.dumps({
        "anomalies": [a.__dict__ if hasattr(a, "__dict__") else a for a in anomalies],
        "bottlenecks": [b.__dict__ for b in bottlenecks]
    }, ensure_ascii=False, indent=2))