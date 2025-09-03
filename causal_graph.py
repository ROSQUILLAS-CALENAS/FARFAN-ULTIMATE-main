"""
NetworkX-based directed graph implementation for representing causal relationships (08K).
Refactored for robustness, efficiency and advanced causal analysis capabilities.
"""

import copy
import json
import logging
import warnings
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple, Union  # Module not found  # Module not found  # Module not found

import networkx as nx

# Import total ordering base
try:
# # #     from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found
    TOTAL_ORDERING_AVAILABLE = True
except ImportError:
    TOTAL_ORDERING_AVAILABLE = False
    # Create mock base class
    class TotalOrderingBase:
        pass

# JSON Schema validation
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    warnings.warn("jsonschema library not available. Install with: pip install jsonschema")

# Alias metadata
alias_code = "08K"
alias_stage = "knowledge_extraction"
component_name = "Causal Graph Builder"

# JSON Schemas for validation
CAUSAL_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "provenance": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "document_stem": {"type": "string"},
                "processing_status": {"type": "string", "enum": ["success", "failed", "partial"]}
            },
            "required": ["component_id", "timestamp", "document_stem", "processing_status"]
        },
        "causal_graph": {
            "type": "object",
            "properties": {
                "nodes": {"type": "array"},
                "edges": {"type": "array"},
                "metadata": {"type": "object"},
                "statistics": {"type": "object"}
            }
        },
        "causal_analysis": {
            "type": "object",
            "properties": {
                "causal_chains": {"type": "array"},
                "causal_strength": {"type": "number"},
                "confounding_factors": {"type": "array"},
                "causal_metrics": {"type": "object"}
            }
        }
    },
    "required": ["provenance", "causal_graph"]
}

def _write_knowledge_artifact(data: Dict[str, Any], document_stem: str, suffix: str, processing_status: str = "success") -> bool:
    """
    Write knowledge artifact to canonical_flow/knowledge/ directory with standardized naming and validation.
    
    Args:
        data: The data to write
        document_stem: Document identifier stem
        suffix: Component-specific suffix
        processing_status: Processing status (success/failed/partial)
    
    Returns:
        bool: True if write was successful
    """
    try:
        # Create output directory
        output_dir = Path("canonical_flow/knowledge")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standardized filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{document_stem}_{alias_code}_{suffix}_{timestamp}.json"
        output_path = output_dir / filename
        
        # Add provenance metadata
        artifact = {
            "provenance": {
                "component_id": alias_code,
                "timestamp": datetime.now().isoformat(),
                "document_stem": document_stem,
                "processing_status": processing_status
            },
            **data
        }
        
        # Validate JSON schema if available
        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(artifact, CAUSAL_GRAPH_SCHEMA)
                logging.info(f"JSON schema validation passed for {filename}")
            except jsonschema.exceptions.ValidationError as e:
                logging.warning(f"JSON schema validation failed for {filename}: {e}")
                # Continue with writing despite validation failure for debugging
        
        # Write JSON with UTF-8 encoding and standardized formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Successfully wrote knowledge artifact: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to write knowledge artifact for {document_stem}_{suffix}: {e}")
        # Try to write a minimal artifact for debugging purposes
        try:
            error_artifact = {
                "provenance": {
                    "component_id": alias_code,
                    "timestamp": datetime.now().isoformat(),
                    "document_stem": document_stem,
                    "processing_status": "failed"
                },
                "error": str(e),
                "attempted_data": str(data)[:1000]  # Truncate for safety
            }
            
            error_filename = f"{document_stem}_{alias_code}_{suffix}_error_{timestamp}.json"
            error_path = output_dir / error_filename
            
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_artifact, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Wrote error artifact for debugging: {error_path}")
        except Exception as inner_e:
            logging.error(f"Failed to write error artifact: {inner_e}")
        
        return False


class CausalRelationType(Enum):
    """Types of causal relationships."""

    DIRECT_CAUSE = "direct_cause"
    ENABLING_CONDITION = "enabling_condition"
    INHIBITING_CONDITION = "inhibiting_condition"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CORRELATION = "correlation"


@dataclass(frozen=True)
class CausalNode:
    """Represents an entity/concept in the causal graph. Immutable for safety."""

    id: str
    text: str
    entity_type: str
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Enhanced attributes for probabilistic modeling
    cpt: Optional[Dict[str, Any]] = None  # Conditional Probability Table

    def __hash__(self):
        return hash(self.id)


@dataclass(frozen=True)
class CausalEdge:
    """Represents a causal relationship between nodes. Immutable for safety."""

    source: str
    target: str
    relation_type: CausalRelationType
    confidence: float
    evidence_spans: List[Tuple[int, int]] = field(default_factory=list)
    linguistic_patterns: List[str] = field(default_factory=list)
    # Enhanced attributes for LLM integration
    evidence_text: str = ""
    context: str = ""
    metadata_llm: Dict[str, Any] = field(default_factory=dict)
    # Probabilistic attributes
    strength: float = 1.0  # Influence strength (e.g., regression coefficient)

    def __hash__(self):
        return hash((self.source, self.target, self.relation_type))


class CausalGraph(TotalOrderingBase):
    """NetworkX-based MultiDiGraph for sophisticated causal relationships analysis."""

    def __init__(self):
        super().__init__("CausalGraph")
        # Single source of truth: NetworkX MultiDiGraph for multiple edge types
        self.graph = nx.MultiDiGraph()
        # Internal version for cache invalidation
        self._version = 0
        self._cache: Dict[str, Any] = {}
        
        # Update state hash for deterministic processing
        self.update_state_hash(self._get_initial_state())

    def _get_comparison_key(self) -> Tuple[str, ...]:
        """Return comparison key for deterministic ordering"""
        return (
            self.component_id,
            str(self._version),
            str(self.graph.number_of_nodes()),
            str(self.graph.number_of_edges()),
            str(self._state_hash or "")
        )
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "version": self._version,
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
        }

    def _bump_version(self) -> None:
        """Increment version to invalidate caches."""
        self._version += 1
        self._cache.clear()
        # Update state hash when structure changes
        self.update_state_hash(self._get_initial_state())

    def add_node(self, node: CausalNode) -> None:
        """Add a causal node to the graph."""
        self.graph.add_node(
            node.id,
            node_obj=node,
            text=node.text,
            entity_type=node.entity_type,
            confidence=node.confidence,
            attributes=node.attributes,
            cpt=node.cpt,
        )
        self._bump_version()

    def add_edge(self, edge: CausalEdge, key: Optional[str] = None) -> str:
        """Add a causal relationship edge to the graph."""
        if not self.graph.has_node(edge.source) or not self.graph.has_node(edge.target):
            raise ValueError("Source and target nodes must exist before adding edge")

        # Generate unique key if not provided
        if key is None:
            key = f"{edge.relation_type.value}_{len(self.graph[edge.source].get(edge.target, {}))}"

        edge_key = self.graph.add_edge(
            edge.source,
            edge.target,
            key=key,
            edge_obj=edge,
            relation_type=edge.relation_type,
            confidence=edge.confidence,
            evidence_spans=edge.evidence_spans,
            linguistic_patterns=edge.linguistic_patterns,
            evidence_text=edge.evidence_text,
            context=edge.context,
            metadata_llm=edge.metadata_llm,
            strength=edge.strength,
        )

        self._bump_version()
        return edge_key

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges safely."""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            self._bump_version()

    def remove_edge(self, source: str, target: str, key: Optional[str] = None) -> None:
        """Remove an edge safely."""
        if key is None:
            # Remove all edges between source and target
            if self.graph.has_edge(source, target):
                self.graph.remove_edge(source, target)
                self._bump_version()
        else:
            # Remove specific edge by key
            if self.graph.has_edge(source, target, key):
                self.graph.remove_edge(source, target, key)
                self._bump_version()

    def get_node(self, node_id: str) -> Optional[CausalNode]:
        """Retrieve a node by ID."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id].get("node_obj")
        return None

    def get_edge(
        self, source: str, target: str, key: Optional[str] = None
    ) -> Optional[CausalEdge]:
        """Retrieve an edge by source, target and optionally key."""
        if not self.graph.has_edge(source, target):
            return None

        edges = self.graph[source][target]
        if key is None:
            # Return first edge if no key specified
            if edges:
                first_key = next(iter(edges))
                return edges[first_key].get("edge_obj")
        else:
            # Return specific edge by key
            if key in edges:
                return edges[key].get("edge_obj")
        return None

    def get_causal_chains(
        self, min_length: int = 2, start_node: Optional[str] = None
    ) -> List[List[str]]:
        """Extract causal chains efficiently. Optimized to avoid N×N loops."""
        chains = []

        if start_node is not None:
# # #             # Find chains starting from specific node  # Module not found  # Module not found  # Module not found
            if not self.graph.has_node(start_node):
                return chains

            for target in self.graph.nodes():
                if start_node != target:
                    try:
                        paths = list(
                            nx.all_simple_paths(
                                self.graph, start_node, target, cutoff=10
                            )
                        )
                        chains.extend(
                            [path for path in paths if len(path) >= min_length]
                        )
                    except nx.NetworkXNoPath:
                        continue
        else:
            # Find all chains - use efficient approach
# # #             # Only start from nodes that could begin a chain (have outgoing edges)  # Module not found  # Module not found  # Module not found
            potential_starts = [
                n for n in self.graph.nodes() if self.graph.out_degree(n) > 0
            ]

            for source in potential_starts:
# # #                 # Only target nodes that could end a chain (reachable from source)  # Module not found  # Module not found  # Module not found
                reachable = nx.descendants(self.graph, source)
                for target in reachable:
                    try:
                        paths = list(
                            nx.all_simple_paths(self.graph, source, target, cutoff=10)
                        )
                        chains.extend(
                            [path for path in paths if len(path) >= min_length]
                        )
                    except nx.NetworkXNoPath:
                        continue

        return chains

    def get_root_causes(self) -> List[str]:
        """Identify root cause nodes (nodes with no incoming edges)."""
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

    def get_terminal_effects(self) -> List[str]:
        """Identify terminal effect nodes (nodes with no outgoing edges)."""
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]

    def get_strongly_connected_components(self) -> List[Set[str]]:
        """Identify circular causal relationships."""
        return list(nx.strongly_connected_components(self.graph))

    def calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for nodes."""
        centrality = {}

        # Convert to simple graph for centrality measures that don't support multigraph
        simple_graph = nx.DiGraph(self.graph)

        centrality["betweenness"] = nx.betweenness_centrality(simple_graph)
        centrality["closeness"] = nx.closeness_centrality(simple_graph)
        centrality["pagerank"] = nx.pagerank(simple_graph)
        centrality["in_degree"] = dict(self.graph.in_degree())
        centrality["out_degree"] = dict(self.graph.out_degree())

        return centrality

    def filter_by_confidence(self, min_confidence: float) -> "CausalGraph":
        """Create a filtered graph containing only high-confidence relationships."""
        filtered_graph = CausalGraph()

        # Add nodes with sufficient confidence
        for node_id in self.graph.nodes():
            node = self.get_node(node_id)
            if node and node.confidence >= min_confidence:
                filtered_graph.add_node(node)

        # Add edges with sufficient confidence
        for u, v, key, edge_data in self.graph.edges(data=True, keys=True):
            edge = edge_data.get("edge_obj")
            if (
                edge
                and edge.confidence >= min_confidence
                and filtered_graph.graph.has_node(u)
                and filtered_graph.graph.has_node(v)
            ):
                filtered_graph.add_edge(edge, key)

        return filtered_graph

    # === PHASE 2: Advanced Causal Analysis ===

    def intervene(self, interventions: Dict[str, Any]) -> "CausalGraph":
        """Simulate do-calculus intervention by removing incoming edges to intervened nodes."""
        intervened_graph = CausalGraph()

        # Copy all nodes
        for node_id in self.graph.nodes():
            node = self.get_node(node_id)
            if node:
                intervened_graph.add_node(node)

        # Copy edges, but skip incoming edges to intervened nodes
        for u, v, key, edge_data in self.graph.edges(data=True, keys=True):
            edge = edge_data.get("edge_obj")
            if edge:
                # Skip edges pointing to intervened nodes
                if v not in interventions:
                    intervened_graph.add_edge(edge, key)

        return intervened_graph

    def find_mediators(self, source: str, target: str) -> List[str]:
        """Find nodes that mediate the relationship between source and target."""
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return []

        try:
            # Find all paths between source and target
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=5))

            # Collect all intermediate nodes
            mediators = set()
            for path in paths:
                if len(path) > 2:  # Has intermediate nodes
                    mediators.update(path[1:-1])  # Exclude source and target

            return list(mediators)
        except nx.NetworkXNoPath:
            return []

    def find_colliders(self, source: str, target: str) -> List[str]:
        """Find collider nodes in paths between source and target."""
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return []

        colliders = []

        # A collider is a node where two paths converge
        for node in self.graph.nodes():
            if node in [source, target]:
                continue

# # #             # Check if there are paths from source to node and from target to node  # Module not found  # Module not found  # Module not found
            has_path_from_source = nx.has_path(self.graph, source, node)
            has_path_from_target = nx.has_path(self.graph, target, node)

            if has_path_from_source and has_path_from_target:
                colliders.append(node)

        return colliders

    def find_confounders(self, node_a: str, node_b: str) -> List[str]:
        """Find common causes (confounders) of node_a and node_b."""
        if not self.graph.has_node(node_a) or not self.graph.has_node(node_b):
            return []

        # Find all ancestors of both nodes
        ancestors_a = set(nx.ancestors(self.graph, node_a))
        ancestors_b = set(nx.ancestors(self.graph, node_b))

        # Common ancestors are potential confounders
        confounders = ancestors_a.intersection(ancestors_b)
        return list(confounders)

    def propagate_probability(self, evidence: Dict[str, Any]) -> Dict[str, float]:
        """Basic probability propagation given evidence (simplified implementation)."""
        # This is a simplified version - for full Bayesian inference, use pgmpy
        probabilities = {}

        # Initialize with evidence
        for node_id, value in evidence.items():
            probabilities[node_id] = 1.0 if value else 0.0

        # Simple propagation for nodes without evidence
        for node_id in self.graph.nodes():
            if node_id not in probabilities:
                node = self.get_node(node_id)
                if node and node.cpt:
                    # Use CPT if available
                    probabilities[node_id] = node.confidence  # Simplified
                else:
                    # Default probability based on confidence
                    probabilities[node_id] = node.confidence if node else 0.5

        return probabilities

    # === PHASE 3: LLM and ML Integration ===

    def explain_path(self, path: List[str]) -> str:
        """Generate natural language explanation for a causal path."""
        if len(path) < 2:
            return "Path too short to explain."

        explanations = []

        for i in range(len(path) - 1):
            source_node = self.get_node(path[i])
            target_node = self.get_node(path[i + 1])
            edge = self.get_edge(path[i], path[i + 1])

            if source_node and target_node:
                source_text = source_node.text
                target_text = target_node.text

                if edge:
                    relation_map = {
                        CausalRelationType.DIRECT_CAUSE: "caused",
                        CausalRelationType.ENABLING_CONDITION: "enabled",
                        CausalRelationType.INHIBITING_CONDITION: "inhibited",
                        CausalRelationType.TEMPORAL_SEQUENCE: "led to",
                        CausalRelationType.CORRELATION: "was associated with",
                    }

                    relation_text = relation_map.get(edge.relation_type, "influenced")
                    explanations.append(f"{source_text} {relation_text} {target_text}")
                else:
                    explanations.append(f"{source_text} influenced {target_text}")

        return ", which in turn ".join(explanations) + "."

    def to_pyg(self):
        """Convert to PyTorch Geometric format for GNN processing."""
        try:
            import torch
# # #             from torch_geometric.data import Data  # Module not found  # Module not found  # Module not found

            # Create node mapping
            node_mapping = {
                node_id: idx for idx, node_id in enumerate(self.graph.nodes())
            }

            # Create edge index
            edge_index = []
            edge_attr = []

            for u, v, key, edge_data in self.graph.edges(data=True, keys=True):
                edge_index.append([node_mapping[u], node_mapping[v]])
                edge_attr.append(
                    [edge_data.get("confidence", 0.5), edge_data.get("strength", 1.0)]
                )

            # Create node features
            node_features = []
            for node_id in self.graph.nodes():
                node = self.get_node(node_id)
                if node:
                    features = [node.confidence, len(node.text), len(node.attributes)]
                    node_features.append(features)
                else:
                    node_features.append([0.0, 0.0, 0.0])

            # Convert to tensors
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 2), dtype=torch.float)

            x = torch.tensor(node_features, dtype=torch.float)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        except ImportError:
            raise ImportError(
                "PyTorch Geometric not available. Install with: pip install torch-geometric"
            )

    def calculate_node_embeddings(self, method: str = "node2vec", **kwargs):
        """Calculate node embeddings using specified method."""
        try:
            if method == "node2vec":
# # #                 from node2vec import Node2Vec  # Module not found  # Module not found  # Module not found

                # Convert to simple graph for node2vec
                simple_graph = nx.Graph(self.graph.to_undirected())

                # Initialize Node2Vec
                node2vec = Node2Vec(
                    simple_graph,
                    dimensions=kwargs.get("dimensions", 64),
                    walk_length=kwargs.get("walk_length", 30),
                    num_walks=kwargs.get("num_walks", 200),
                    workers=kwargs.get("workers", 4),
                )

                # Fit model
                model = node2vec.fit(window=10, min_count=1, batch_words=4)

                # Get embeddings
                embeddings = {}
                for node_id in self.graph.nodes():
                    if str(node_id) in model.wv:
                        embeddings[node_id] = model.wv[str(node_id)]

                return embeddings

        except ImportError:
            raise ImportError(f"Required library for {method} not available")

    def export_to_gephi(self, filename: str) -> None:
        """Export graph to GEXF format for Gephi visualization."""
        # Add visualization attributes
        for node_id in self.graph.nodes():
            node = self.get_node(node_id)
            if node:
                self.graph.nodes[node_id]["label"] = node.text[
                    :50
                ]  # Truncate for visualization
                self.graph.nodes[node_id]["size"] = node.confidence * 10

        nx.write_gexf(self.graph, filename)

    def export_to_graphml(self, filename: str) -> None:
        """Export graph to GraphML format."""
        nx.write_graphml(self.graph, filename)

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation with all new attributes."""
        nodes = []
        edges = []

        for node_id in self.graph.nodes():
            node = self.get_node(node_id)
            if node:
                nodes.append(
                    {
                        "id": node.id,
                        "text": node.text,
                        "entity_type": node.entity_type,
                        "confidence": node.confidence,
                        "attributes": node.attributes,
                        "cpt": node.cpt,
                    }
                )

        for u, v, key, edge_data in self.graph.edges(data=True, keys=True):
            edge = edge_data.get("edge_obj")
            if edge:
                edges.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "key": key,
                        "relation_type": edge.relation_type.value,
                        "confidence": edge.confidence,
                        "evidence_spans": edge.evidence_spans,
                        "linguistic_patterns": edge.linguistic_patterns,
                        "evidence_text": edge.evidence_text,
                        "context": edge.context,
                        "metadata_llm": edge.metadata_llm,
                        "strength": edge.strength,
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "graph_type": "MultiDiGraph",
                "version": self._version,
                "is_acyclic": self.is_acyclic(),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalGraph":
# # #         """Create graph from dictionary representation with all attributes."""  # Module not found  # Module not found  # Module not found
        graph = cls()

        # Add nodes
        for node_data in data["nodes"]:
            node = CausalNode(
                id=node_data["id"],
                text=node_data["text"],
                entity_type=node_data["entity_type"],
                confidence=node_data["confidence"],
                attributes=node_data.get("attributes", {}),
                cpt=node_data.get("cpt"),
            )
            graph.add_node(node)

        # Add edges
        for edge_data in data["edges"]:
            edge = CausalEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                relation_type=CausalRelationType(edge_data["relation_type"]),
                confidence=edge_data["confidence"],
                evidence_spans=edge_data.get("evidence_spans", []),
                linguistic_patterns=edge_data.get("linguistic_patterns", []),
                evidence_text=edge_data.get("evidence_text", ""),
                context=edge_data.get("context", ""),
                metadata_llm=edge_data.get("metadata_llm", {}),
                strength=edge_data.get("strength", 1.0),
            )
            key = edge_data.get("key")
            graph.add_edge(edge, key)

        return graph

    # === PHASE 4: Expert Causal Inference Enhancements ===

    def is_acyclic(self) -> bool:
        """Check if the causal graph is acyclic (a DAG)."""
        try:
            cycles = list(nx.find_cycle(self.graph, orientation="original"))
            return len(cycles) == 0
        except nx.NetworkXNoCycle:
            return True

    def enforce_acyclic(self) -> List[Tuple[str, str, str]]:
        """Remove a minimal set of low-confidence edges to break cycles.
        Returns a list of removed edges as (u, v, key)."""
        removed = []
        try:
            while True:
                cycle = list(nx.find_cycle(self.graph, orientation="original"))
                # Choose edge with smallest confidence on the cycle
                min_edge = None
                min_conf = float("inf")
                for u, v, k, _ in cycle:
                    # MultiDiGraph may not return key here; get min over keys
                    if self.graph.has_edge(u, v):
                        for key, data in self.graph[u][v].items():
                            conf = data.get("confidence", 1.0)
                            if conf < min_conf:
                                min_conf = conf
                                min_edge = (u, v, key)
                if min_edge is None:
                    break
                self.remove_edge(*min_edge)
                removed.append(min_edge)
        except nx.NetworkXNoCycle:
            pass
        return removed

    def identify_backdoor_adjustment_set(self, source: str, target: str) -> List[str]:
        """Heuristic backdoor adjustment set using ancestor-based rule.
# # #         Returns a set of nodes Z that d-separate all backdoor paths from source to target.  # Module not found  # Module not found  # Module not found
        Note: Simplified; for exact identification use do-calculus libraries.
        """
        if not (self.graph.has_node(source) and self.graph.has_node(target)):
            return []
        # Ancestors of source and target
        anc_s = nx.ancestors(self.graph, source)
        anc_t = nx.ancestors(self.graph, target)
        candidates = (anc_s | anc_t) - {source, target}
        # Exclude descendants of source
        desc_s = nx.descendants(self.graph, source)
        Z = [z for z in candidates if z not in desc_s]
        # Remove colliders (heuristic): nodes with indegree >=2 on paths to target
        result = []
        for z in Z:
            if self.graph.in_degree(z) >= 2:
                # Keep z only if we plan to condition on its descendants (we don't), so skip
                continue
            result.append(z)
        return result

    def compute_effect_strength(
        self, source: str, target: str, method: str = "max_path"
    ) -> float:
        """Aggregate causal strength between two nodes via path strengths.
        - 'max_path': maximum product of edge strengths over all simple paths
        - 'sum_paths': sum of products with exponential decay by path length
        """
        if not (self.graph.has_node(source) and self.graph.has_node(target)):
            return 0.0
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=6))
        except nx.NetworkXNoPath:
            return 0.0
        if not paths:
            return 0.0

        def path_strength(p: List[str]) -> float:
            prod = 1.0
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                # Use strongest edge between u and v
                strengths = [
                    data.get("strength", 1.0)
                    for _, _, data in self.graph.edges(u, v, data=True)
                ]
                prod *= max(strengths) if strengths else 1.0
            return prod

        vals = [path_strength(p) for p in paths]
        if method == "max_path":
            return max(vals)
        elif method == "sum_paths":
            # decay by length to avoid overcounting many long paths
            total = 0.0
            for p, s in zip(paths, vals):
                decay = 0.9 ** (len(p) - 1)
                total += s * decay
            return total
        else:
            return max(vals)

    def counterfactual_qualitative(
        self,
        outcome: str,
        do: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Qualitative counterfactual reasoning via abduction-action-prediction sketch.
        Returns a dictionary with the intervened graph and qualitative verdict.
        """
        evidence = evidence or {}
        if outcome not in self.graph.nodes:
            return {"verdict": "unknown", "reason": "outcome node missing"}
        # Abduction: record current influential parents of outcome
        parents = list(self.graph.predecessors(outcome))
        influential = [
            p for p in parents if self.compute_effect_strength(p, outcome) > 0.1
        ]
        # Action: intervene on specified nodes
        g_do = self.intervene(do)
# # #         # Prediction: check if outcome remains reachable from influential parents  # Module not found  # Module not found  # Module not found
        reachable = any(
            nx.has_path(g_do.graph, p, outcome)
            for p in influential
            if g_do.graph.has_node(p)
        )
        verdict = "changed" if not reachable else "unchanged"
        return {
            "verdict": verdict,
            "influential_parents": influential,
            "interventions": do,
            "outcome": outcome,
        }


# === DEMONSTRATION AND TESTING ===

if __name__ == "__main__":
    # Create a sample causal graph to demonstrate new capabilities
    graph = CausalGraph()

    # Add nodes with enhanced attributes
    investment_node = CausalNode(
        id="investment",
        text="increased government investment in technology",
        entity_type="policy",
        confidence=0.9,
        attributes={"sector": "technology", "amount": "high"},
        cpt={"high": 0.3, "medium": 0.5, "low": 0.2},
    )

    tech_dev_node = CausalNode(
        id="tech_development",
        text="accelerated technological development",
        entity_type="outcome",
        confidence=0.85,
        attributes={"innovation_rate": "high"},
    )

    market_expansion_node = CausalNode(
        id="market_expansion",
        text="expansion of domestic technology market",
        entity_type="economic_effect",
        confidence=0.8,
        attributes={"market_size_change": "+25%"},
    )

    # Add nodes to graph
    for node in [investment_node, tech_dev_node, market_expansion_node]:
        graph.add_node(node)

    # Add edges with enhanced metadata
    edge1 = CausalEdge(
        source="investment",
        target="tech_development",
        relation_type=CausalRelationType.DIRECT_CAUSE,
        confidence=0.9,
        evidence_spans=[(10, 45)],
        linguistic_patterns=["X led to Y"],
        evidence_text="The increased investment led to rapid technological advancement",
        context="Government policy analysis section discussing innovation outcomes",
        metadata_llm={"certainty": "high", "modality": "strong_causal"},
        strength=0.85,
    )

    edge2 = CausalEdge(
        source="tech_development",
        target="market_expansion",
        relation_type=CausalRelationType.ENABLING_CONDITION,
        confidence=0.8,
        evidence_spans=[(60, 95)],
        linguistic_patterns=["X enabled Y"],
        evidence_text="Technological advances enabled market growth",
        context="Economic impact section of the development report",
        metadata_llm={"certainty": "medium", "modality": "enabling"},
        strength=0.75,
    )

    # Add edges to graph
    graph.add_edge(edge1, "policy_to_tech")
    graph.add_edge(edge2, "tech_to_market")

    print("=== ADVANCED CAUSAL GRAPH DEMONSTRATION ===\n")

    # Demonstrate causal chain analysis
    chains = graph.get_causal_chains(min_length=2)
    print(f"Found {len(chains)} causal chains:")
    for chain in chains:
        print(f"  Chain: {' -> '.join(chain)}")
        explanation = graph.explain_path(chain)
        print(f"  Explanation: {explanation}")
    print()

    # Demonstrate intervention analysis
    print("=== INTERVENTION ANALYSIS ===")
    interventions = {"tech_development": "high"}
    intervened_graph = graph.intervene(interventions)
    print(f"Original graph edges: {intervened_graph.graph.number_of_edges()}")
    print("Simulated do-calculus intervention on tech_development")
    print()

    # Demonstrate mediator analysis
    mediators = graph.find_mediators("investment", "market_expansion")
    print(f"Mediators between investment and market expansion: {mediators}")
    print()

    # Demonstrate confounders analysis
    confounders = graph.find_confounders("tech_development", "market_expansion")
    print(f"Confounders between tech development and market expansion: {confounders}")
    print()

    # Demonstrate probability propagation
    evidence = {"investment": True}
    probabilities = graph.propagate_probability(evidence)
    print("Probability propagation with evidence {'investment': True}:")
    for node_id, prob in probabilities.items():
        print(f"  {node_id}: {prob:.3f}")
    print()

    # Demonstrate centrality measures
    centrality = graph.calculate_centrality_measures()
    print("Node centrality measures:")
    for node_id in graph.graph.nodes():
        print(f"  {node_id}:")
        print(f"    PageRank: {centrality['pagerank'].get(node_id, 0):.3f}")
        print(f"    Betweenness: {centrality['betweenness'].get(node_id, 0):.3f}")
    print()

    # Demonstrate serialization with new attributes
    graph_dict = graph.to_dict()
    print("Serialization test:")
    print(f"  Nodes: {len(graph_dict['nodes'])}")
    print(f"  Edges: {len(graph_dict['edges'])}")
    print(f"  Metadata: {graph_dict['metadata']}")

    # Test round-trip serialization
    reconstructed_graph = CausalGraph.from_dict(graph_dict)
    print(
        f"  Round-trip test: {reconstructed_graph.graph.number_of_nodes()} nodes, {reconstructed_graph.graph.number_of_edges()} edges"
    )

    print("\n=== DEMONSTRATION COMPLETE ===")

def process_causal_analysis(knowledge_graph: Any, 
                           document_stem: str,
                           entities: Optional[List[Dict[str, Any]]] = None,
                           relations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Main processing function for causal graph analysis stage (08K).
    
    Args:
# # #         knowledge_graph: Input knowledge graph from previous stage  # Module not found  # Module not found  # Module not found
        document_stem: Document identifier for output naming
        entities: Optional list of extracted entities
        relations: Optional list of extracted relations
        
    Returns:
        Dict containing causal graph and analysis results
    """
    try:
        logging.info(f"Starting causal analysis for document: {document_stem}")
        
# # #         # Create causal graph from knowledge graph  # Module not found  # Module not found  # Module not found
        causal_graph = CausalGraph()
        
# # #         # Add nodes from entities or knowledge graph  # Module not found  # Module not found  # Module not found
        if entities:
            for entity in entities:
                node = CausalNode(
                    id=entity.get('id', entity.get('text', str(hash(str(entity))))),
                    text=entity.get('text', ''),
                    entity_type=entity.get('type', 'unknown'),
                    confidence=entity.get('confidence', 0.5),
                    attributes=entity.get('attributes', {})
                )
                causal_graph.add_node(node)
        elif hasattr(knowledge_graph, 'nodes'):
            for node_id, node_data in knowledge_graph.nodes(data=True):
                node = CausalNode(
                    id=node_id,
                    text=node_data.get('text', node_id),
                    entity_type=node_data.get('type', 'unknown'),
                    confidence=node_data.get('confidence', 0.5),
                    attributes=node_data
                )
                causal_graph.add_node(node)
        
# # #         # Add edges from relations or knowledge graph  # Module not found  # Module not found  # Module not found
        if relations:
            for relation in relations:
                edge = CausalEdge(
                    source=relation.get('source', ''),
                    target=relation.get('target', ''),
                    relation_type=CausalRelationType(relation.get('type', 'correlation')),
                    confidence=relation.get('confidence', 0.5),
                    evidence_spans=relation.get('evidence_spans', [])
                )
                causal_graph.add_edge(edge)
        elif hasattr(knowledge_graph, 'edges'):
            for source, target, edge_data in knowledge_graph.edges(data=True):
                edge = CausalEdge(
                    source=source,
                    target=target,
                    relation_type=CausalRelationType(edge_data.get('relation_type', 'correlation')),
                    confidence=edge_data.get('confidence', 0.5),
                    evidence_spans=edge_data.get('evidence_spans', [])
                )
                causal_graph.add_edge(edge)
        
        # Perform causal analysis
        causal_chains = causal_graph.get_causal_chains(min_length=2)
        centrality = causal_graph.calculate_centrality_measures()
        
        # Calculate causal strength
        causal_strength = 0.0
        if causal_graph.graph.number_of_edges() > 0:
            edge_weights = [data.get('weight', 0.5) for _, _, data in causal_graph.graph.edges(data=True)]
            causal_strength = sum(edge_weights) / len(edge_weights)
        
        # Find potential confounding factors
        confounding_factors = []
        nodes = list(causal_graph.graph.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                confounders = causal_graph.find_confounders(node1, node2)
                if confounders:
                    confounding_factors.extend([{
                        "source": node1,
                        "target": node2, 
                        "confounders": confounders
                    }])
        
        # Prepare output data
        graph_data = {
            "causal_graph": {
                "nodes": [
                    {
                        "id": node.id,
                        "text": node.text,
                        "entity_type": node.entity_type,
                        "confidence": node.confidence,
                        "attributes": node.attributes
                    }
                    for node in causal_graph.nodes.values()
                ],
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "relation_type": edge.relation_type.value,
                        "confidence": edge.confidence,
                        "evidence_spans": edge.evidence_spans
                    }
                    for edge in causal_graph.edges
                ],
                "metadata": {
                    "num_nodes": causal_graph.graph.number_of_nodes(),
                    "num_edges": causal_graph.graph.number_of_edges(),
                    "density": nx.density(causal_graph.graph) if causal_graph.graph.number_of_nodes() > 1 else 0.0
                },
                "statistics": {
                    "centrality_measures": centrality,
                    "graph_properties": {
                        "is_dag": nx.is_directed_acyclic_graph(causal_graph.graph),
                        "weakly_connected": nx.is_weakly_connected(causal_graph.graph) if causal_graph.graph.number_of_nodes() > 0 else False
                    }
                }
            }
        }
        
        analysis_data = {
            "causal_analysis": {
                "causal_chains": [list(chain) for chain in causal_chains],
                "causal_strength": float(causal_strength),
                "confounding_factors": confounding_factors,
                "causal_metrics": {
                    "avg_path_length": nx.average_shortest_path_length(causal_graph.graph.to_undirected()) if nx.is_connected(causal_graph.graph.to_undirected()) else 0.0,
                    "transitivity": nx.transitivity(causal_graph.graph.to_undirected()),
                    "num_causal_chains": len(causal_chains)
                }
            }
        }
        
        # Write artifacts to canonical_flow/knowledge/
        _write_knowledge_artifact(graph_data, document_stem, "causal_graph")
        _write_knowledge_artifact(analysis_data, document_stem, "causal_analysis")
        
        logging.info(f"Completed causal analysis for document: {document_stem}")
        return {**graph_data, **analysis_data}
        
    except Exception as e:
        logging.error(f"Error in causal analysis for {document_stem}: {e}")
        
        # Write error artifacts for debugging
        error_data = {"error": str(e), "input_type": type(knowledge_graph).__name__}
        _write_knowledge_artifact(error_data, document_stem, "causal_graph", "failed")
        _write_knowledge_artifact(error_data, document_stem, "causal_analysis", "failed")
        
        return {
            "causal_graph": {"nodes": [], "edges": [], "metadata": {}, "statistics": {}},
            "causal_analysis": {"causal_chains": [], "causal_strength": 0.0, "confounding_factors": [], "causal_metrics": {}},
            "error": str(e)
        }

# Main execution for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run the original demonstration
    print("Running causal graph demonstration...")
    
    # Test the causal analysis process
    mock_knowledge_graph = nx.DiGraph()
    mock_knowledge_graph.add_node("economy", text="Economic conditions", type="concept", confidence=0.8)
    mock_knowledge_graph.add_node("employment", text="Employment rates", type="metric", confidence=0.9)
    mock_knowledge_graph.add_edge("economy", "employment", relation_type="causes", confidence=0.7)
    
    result = process_causal_analysis(mock_knowledge_graph, "test_document")
    
    print(f"\nCausal analysis results:")
    print(f"Nodes: {len(result['causal_graph']['nodes'])}")
    print(f"Edges: {len(result['causal_graph']['edges'])}")
    print(f"Causal strength: {result['causal_analysis']['causal_strength']:.3f}")
    print(f"Causal chains: {len(result['causal_analysis']['causal_chains'])}")
    
    print("="*80)
    print("CAUSAL GRAPH BUILDER COMPLETED")
    print("="*80)
    print(f"Component: {component_name} ({alias_code})")
    print(f"Stage: {alias_stage}")
    print(f"Artifacts written to: canonical_flow/knowledge/")
    print("="*80)



def process(data=None, context=None) -> Dict[str, Any]:
    """
    Process method with audit integration for 09K component.
    
    Args:
        data: Input data containing entities and relationships
        context: Additional context information
        
    Returns:
        Dict containing processed causal graph and analysis
    """
# # #     from canonical_flow.knowledge.knowledge_audit_system import audit_component_execution  # Module not found  # Module not found  # Module not found
    
    @audit_component_execution("09K", metadata={"component": "causal_graph"})
    def _process_with_audit(data, context):
        if data is None:
            return {"error": "No input data provided"}
        
        try:
            # Initialize causal graph
            graph = CausalGraph()
            
# # #             # Extract entities and relationships from data  # Module not found  # Module not found  # Module not found
            entities = []
            relationships = []
            
            if isinstance(data, dict):
                entities = data.get("entities", [])
                relationships = data.get("relationships", [])
                if "text" in data and not entities:
                    # If no entities provided but text is available, create simple entities
                    text_content = data["text"]
                    entities = [{"id": "entity_0", "text": text_content[:100], "type": "concept"}]
            
            # Add nodes to graph
            processed_nodes = []
            for entity in entities:
                if isinstance(entity, dict):
                    node = CausalNode(
                        id=entity.get("id", f"node_{len(processed_nodes)}"),
                        text=entity.get("text", entity.get("name", "")),
                        entity_type=entity.get("type", entity.get("entity_type", "concept")),
                        confidence=entity.get("confidence", 0.8),
                        attributes=entity.get("attributes", {})
                    )
                    graph.add_node(node)
                    processed_nodes.append(node.id)
            
            # Add edges to graph  
            processed_edges = []
            for relation in relationships:
                if isinstance(relation, dict) and "source" in relation and "target" in relation:
                    edge = CausalEdge(
                        source=relation["source"],
                        target=relation["target"],
                        relation_type=CausalRelationType(relation.get("type", "direct_cause")),
                        confidence=relation.get("confidence", 0.8),
                        evidence_spans=relation.get("evidence_spans", [])
                    )
                    edge_id = f"{relation['source']}_to_{relation['target']}"
                    graph.add_edge(edge, edge_id)
                    processed_edges.append(edge_id)
            
            # Perform causal analysis
            analysis = {}
            
            if len(processed_nodes) > 1:
                # Get causal chains
                chains = graph.get_causal_chains(min_length=2)
                analysis["causal_chains"] = [list(chain) for chain in chains]
                
                # Calculate centrality measures
                centrality = graph.calculate_centrality_measures()
                analysis["centrality"] = centrality
                
                # Get topological insights
                analysis["topological_order"] = graph.get_topological_order()
                analysis["strongly_connected_components"] = [list(comp) for comp in graph.get_strongly_connected_components()]
            
            # Prepare result
            result = {
                "graph": {
                    "nodes": processed_nodes,
                    "edges": processed_edges,
                    "node_count": len(processed_nodes),
                    "edge_count": len(processed_edges)
                },
                "analysis": analysis,
                "graph_dict": graph.to_dict()
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Error in causal_graph process: {e}")
    
    return _process_with_audit(data, context)
