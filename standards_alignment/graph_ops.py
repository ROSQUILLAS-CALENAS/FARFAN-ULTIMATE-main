"""Graph operations for standards and document representations."""

# # # from typing import Any, Dict  # Module not found  # Module not found  # Module not found

import networkx as nx
import numpy as np

# # # from .patterns import Criterion, PatternSpec, Requirement  # Module not found  # Module not found  # Module not found


class StandardsGraph:
    """Graph representation of standards with dimensions, subdimensions, and points."""

    def __init__(self):
        self.G = nx.DiGraph()
        self.dimension_patterns = {}
        self.point_requirements = {}
        self.verification_criteria = {}

    def add_dimension(self, dimension: str, patterns: Dict[str, PatternSpec]):
        """Add a dimension node with associated patterns."""
        self.G.add_node(dimension, node_type="dimension")
        self.dimension_patterns[dimension] = patterns

    def add_subdimension(
        self, dimension: str, subdimension: str, criteria: Dict[str, Criterion]
    ):
        """Add subdimension connected to dimension."""
        self.G.add_node(subdimension, node_type="subdimension")
        self.G.add_edge(dimension, subdimension, relation_type="contains")
        self.verification_criteria[(dimension, subdimension)] = criteria

    def add_point(
        self, subdimension: str, point: int, requirements: Dict[str, Requirement]
    ):
        """Add point connected to subdimension."""
        point_id = f"{subdimension}_point_{point}"
        self.G.add_node(point_id, node_type="point", point_number=point)
        self.G.add_edge(subdimension, point_id, relation_type="contains")
        self.point_requirements[point] = requirements

    def get_distance_matrix(self) -> np.ndarray:
        """Compute distance matrix for GW alignment."""
        nodes = list(self.G.nodes())
        n = len(nodes)
        distances = np.zeros((n, n))

        # Use shortest path distances in the graph
        path_lengths = dict(nx.all_pairs_shortest_path_length(self.G.to_undirected()))

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                distances[i, j] = path_lengths.get(node_i, {}).get(node_j, float("inf"))

        # Replace inf with max finite distance + 1
        max_dist = np.max(distances[np.isfinite(distances)])
        distances[np.isinf(distances)] = max_dist + 1

        return distances


class DocumentGraph:
    """Graph representation of documents with sections, paragraphs, tables."""

    def __init__(self):
        self.G = nx.DiGraph()
        self.node_content = {}
        self.node_features = {}

    def add_section(self, section_id: str, title: str, content: str):
        """Add section node."""
        self.G.add_node(section_id, node_type="section")
        self.node_content[section_id] = {"title": title, "content": content}
        self.node_features[section_id] = self._extract_features(title, content)

    def add_paragraph(self, section_id: str, paragraph_id: str, content: str):
        """Add paragraph under section."""
        self.G.add_node(paragraph_id, node_type="paragraph")
        self.G.add_edge(section_id, paragraph_id, relation_type="contains")
        self.node_content[paragraph_id] = {"content": content}
        self.node_features[paragraph_id] = self._extract_features("", content)

    def add_table(self, parent_id: str, table_id: str, data: Dict[str, Any]):
        """Add table under parent node."""
        self.G.add_node(table_id, node_type="table")
        self.G.add_edge(parent_id, table_id, relation_type="contains")
        self.node_content[table_id] = {"data": data}
        self.node_features[table_id] = self._extract_table_features(data)

    def _extract_features(self, title: str, content: str) -> Dict[str, float]:
# # #         """Extract basic features from text."""  # Module not found  # Module not found  # Module not found
        text = (title + " " + content).lower()
        return {
            "word_count": len(text.split()),
            "char_count": len(text),
            "has_numbers": float(any(c.isdigit() for c in text)),
            "has_caps": float(any(c.isupper() for c in title + content)),
            "sentence_count": float(
                text.count(".") + text.count("!") + text.count("?")
            ),
        }

    def _extract_table_features(self, data: Dict[str, Any]) -> Dict[str, float]:
# # #         """Extract features from table data."""  # Module not found  # Module not found  # Module not found
        if not isinstance(data, dict):
            return {"row_count": 0.0, "col_count": 0.0}

        rows = data.get("rows", [])
        if not rows:
            return {"row_count": 0.0, "col_count": 0.0}

        return {
            "row_count": float(len(rows)),
            "col_count": float(len(rows[0]) if rows else 0),
            "has_headers": float("headers" in data),
        }

    def get_distance_matrix(self) -> np.ndarray:
        """Compute distance matrix for GW alignment."""
        nodes = list(self.G.nodes())
        n = len(nodes)
        distances = np.zeros((n, n))

        # Combine structural and feature-based distances
        path_lengths = dict(nx.all_pairs_shortest_path_length(self.G.to_undirected()))

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                # Structural distance
                struct_dist = path_lengths.get(node_i, {}).get(node_j, float("inf"))

                # Feature distance
                feat_i = self.node_features.get(node_i, {})
                feat_j = self.node_features.get(node_j, {})
                feat_dist = self._feature_distance(feat_i, feat_j)

                # Combine distances (weighted)
                distances[i, j] = 0.7 * min(struct_dist, 10) + 0.3 * feat_dist

        return distances

    def _feature_distance(
        self, feat_i: Dict[str, float], feat_j: Dict[str, float]
    ) -> float:
        """Compute distance between feature vectors."""
        if not feat_i or not feat_j:
            return 1.0

        common_keys = set(feat_i.keys()) & set(feat_j.keys())
        if not common_keys:
            return 1.0

        dist = 0.0
        for key in common_keys:
            v1, v2 = feat_i[key], feat_j[key]
            if v1 == 0 and v2 == 0:
                continue
            max_val = max(abs(v1), abs(v2), 1.0)
            dist += abs(v1 - v2) / max_val

        return dist / len(common_keys)
