"""
09K Causal Graph Constructor Module

# # # Inherits from TotalOrderingBase and implements the standardized process() API to consume   # Module not found  # Module not found  # Module not found
# # # evidence and knowledge graph artifacts from previous stages. Constructs directed acyclic   # Module not found  # Module not found  # Module not found
graphs for each dimension DE-1 through DE-4 using extracted causal relationships with 
evidence-linked edges and validity scoring.

Generates four separate causal_DE1.json through causal_DE4.json artifacts in the 
canonical_flow/knowledge/ directory with proper edge validation, deterministic node 
ordering, and validity scoring based on evidence strength and confidence metrics.
"""

import json
import logging
# # # from collections import defaultdict, OrderedDict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
import networkx as nx

# # # # Import TotalOrderingBase from project root  # Module not found  # Module not found  # Module not found
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
# # # from total_ordering_base import TotalOrderingBase  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


class DimensionType(Enum):
    """Enumeration of dimension types for causal graph construction."""
    DE1 = "DE-1"
    DE2 = "DE-2" 
    DE3 = "DE-3"
    DE4 = "DE-4"


@dataclass(frozen=True)
class EvidenceReference:
    """Immutable reference to supporting evidence with page numbers."""
    source_id: str
    page_numbers: List[int] = field(default_factory=list)
    text_snippet: str = ""
    confidence: float = 0.0
    
    def __hash__(self):
        return hash((self.source_id, tuple(self.page_numbers), self.text_snippet))


@dataclass(frozen=True)  
class CausalRelationship:
    """Immutable causal relationship with evidence anchoring."""
    source_node: str
    target_node: str
    relationship_type: str
    confidence: float
    evidence_strength: float
    evidence_references: Tuple[EvidenceReference, ...] = field(default_factory=tuple)
    validity_score: float = 0.0
    
    def __hash__(self):
        return hash((self.source_node, self.target_node, self.relationship_type))


@dataclass
class CausalGraphArtifact:
    """Structured output artifact for causal graphs."""
    dimension: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    validity_statistics: Dict[str, float]


class CausalGraphConstructor(TotalOrderingBase):
    """
    09K Causal Graph Constructor that builds dimension-specific directed acyclic graphs
    with evidence-linked edges and validity scoring.
    """
    
    def __init__(self):
        super().__init__(component_name="CausalGraphConstructor_09K")
        self.output_directory = Path("canonical_flow/knowledge")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize dimension graphs
        self.dimension_graphs: Dict[DimensionType, nx.DiGraph] = {
            dim: nx.DiGraph() for dim in DimensionType
        }
        
        # Evidence and relationship storage
        self.evidence_registry: Dict[str, Dict[str, Any]] = {}
        self.causal_relationships: Dict[DimensionType, List[CausalRelationship]] = {
            dim: [] for dim in DimensionType
        }
        
        # Configuration thresholds
        self.min_evidence_strength = 0.3
        self.min_confidence_threshold = 0.4
        self.validity_score_weights = {
            "evidence_strength": 0.4,
            "confidence": 0.3,
            "evidence_count": 0.2,
            "evidence_diversity": 0.1
        }
        
        logger.info(f"Initialized {self.component_name} with output directory: {self.output_directory}")
    
    def process(self, evidence_artifacts: Dict[str, Any], knowledge_graph_artifacts: Dict[str, Any], 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, CausalGraphArtifact]:
        """
        Standardized process() API implementation.
        
        Args:
# # #             evidence_artifacts: Evidence data from ingestion/preparation stages  # Module not found  # Module not found  # Module not found
# # #             knowledge_graph_artifacts: Knowledge graph data from previous K stages  # Module not found  # Module not found  # Module not found
            context: Optional processing context
            
        Returns:
            Dictionary of causal graph artifacts keyed by dimension
        """
        operation_id = self.generate_operation_id(
            "process_causal_graphs", 
            {"evidence": evidence_artifacts, "knowledge_graphs": knowledge_graph_artifacts}
        )
        
        logger.info(f"Processing causal graphs with operation ID: {operation_id}")
        
        try:
            # Step 1: Load and register evidence
            self._load_evidence_artifacts(evidence_artifacts)
            
# # #             # Step 2: Extract causal relationships from knowledge graphs  # Module not found  # Module not found  # Module not found
            self._extract_causal_relationships(knowledge_graph_artifacts)
            
            # Step 3: Build dimension-specific graphs
            artifacts = {}
            for dimension in DimensionType:
                artifact = self._build_dimension_graph(dimension)
                artifacts[dimension.value] = artifact
                
            # Step 4: Generate output files
            self._generate_output_files(artifacts)
            
            # Update state
            self.update_state_hash(artifacts)
            
            logger.info(f"Successfully processed causal graphs for {len(artifacts)} dimensions")
            return artifacts
            
        except Exception as e:
            logger.error(f"Error processing causal graphs: {str(e)}")
            return self._generate_sparse_artifacts()
    
    def _load_evidence_artifacts(self, evidence_artifacts: Dict[str, Any]) -> None:
# # #         """Load and register evidence from ingestion artifacts."""  # Module not found  # Module not found  # Module not found
        if not evidence_artifacts:
            logger.warning("No evidence artifacts provided - will generate sparse graphs")
            return
            
# # #         # Extract evidence from various sources  # Module not found  # Module not found  # Module not found
        evidence_count = 0
        
        # Process document evidence
        if "documents" in evidence_artifacts:
            for doc_id, doc_data in evidence_artifacts["documents"].items():
                evidence_id = self.generate_stable_id(doc_data, prefix="evidence")
                self.evidence_registry[evidence_id] = {
                    "source_id": doc_id,
                    "type": "document",
                    "content": doc_data.get("content", ""),
                    "pages": doc_data.get("pages", []),
                    "metadata": doc_data.get("metadata", {})
                }
                evidence_count += 1
        
        # Process structured evidence
        if "structured_evidence" in evidence_artifacts:
            for evidence in evidence_artifacts["structured_evidence"]:
                evidence_id = self.generate_stable_id(evidence, prefix="evidence")
                self.evidence_registry[evidence_id] = evidence
                evidence_count += 1
                
        logger.info(f"Loaded {evidence_count} evidence artifacts")
    
    def _extract_causal_relationships(self, knowledge_graph_artifacts: Dict[str, Any]) -> None:
# # #         """Extract causal relationships from knowledge graph artifacts."""  # Module not found  # Module not found  # Module not found
        if not knowledge_graph_artifacts:
            logger.warning("No knowledge graph artifacts provided")
            return
            
        relationship_count = 0
        
# # #         # Extract from graph structures  # Module not found  # Module not found  # Module not found
        if "graphs" in knowledge_graph_artifacts:
            for graph_data in knowledge_graph_artifacts["graphs"]:
                relationships = self._extract_from_graph_structure(graph_data)
                for rel in relationships:
                    dimension = self._classify_relationship_dimension(rel)
                    if dimension:
                        self.causal_relationships[dimension].append(rel)
                        relationship_count += 1
        
# # #         # Extract from entity relationships  # Module not found  # Module not found  # Module not found
        if "entity_relationships" in knowledge_graph_artifacts:
            for rel_data in knowledge_graph_artifacts["entity_relationships"]:
                relationships = self._extract_from_entity_relationships(rel_data)
                for rel in relationships:
                    dimension = self._classify_relationship_dimension(rel)
                    if dimension:
                        self.causal_relationships[dimension].append(rel)
                        relationship_count += 1
                        
        logger.info(f"Extracted {relationship_count} causal relationships")
    
    def _extract_from_graph_structure(self, graph_data: Dict[str, Any]) -> List[CausalRelationship]:
# # #         """Extract causal relationships from graph structure data."""  # Module not found  # Module not found  # Module not found
        relationships = []
        
        if "edges" not in graph_data:
            return relationships
            
        for edge in graph_data["edges"]:
            if self._is_causal_edge(edge):
                rel = self._create_causal_relationship_from_edge(edge)
                if rel:
                    relationships.append(rel)
                    
        return relationships
    
    def _extract_from_entity_relationships(self, rel_data: Dict[str, Any]) -> List[CausalRelationship]:
# # #         """Extract causal relationships from entity relationship data."""  # Module not found  # Module not found  # Module not found
        relationships = []
        
        # Look for causal indicators in relationship types
        causal_types = ["causes", "enables", "prevents", "influences", "leads_to", "results_in"]
        
        if rel_data.get("type", "").lower() in causal_types:
            evidence_refs = self._create_evidence_references(rel_data.get("evidence", []))
            
            rel = CausalRelationship(
                source_node=str(rel_data.get("source", "")),
                target_node=str(rel_data.get("target", "")),
                relationship_type=rel_data.get("type", ""),
                confidence=float(rel_data.get("confidence", 0.5)),
                evidence_strength=self._calculate_evidence_strength(evidence_refs),
                evidence_references=tuple(evidence_refs),
                validity_score=0.0  # Will be calculated later
            )
            
            relationships.append(rel)
            
        return relationships
    
    def _is_causal_edge(self, edge: Dict[str, Any]) -> bool:
        """Determine if an edge represents a causal relationship."""
        causal_indicators = [
            "cause", "effect", "influence", "impact", "lead", "result", 
            "trigger", "enable", "prevent", "facilitate"
        ]
        
        edge_type = edge.get("type", "").lower()
        edge_label = edge.get("label", "").lower()
        
        return any(indicator in edge_type or indicator in edge_label 
                  for indicator in causal_indicators)
    
    def _create_causal_relationship_from_edge(self, edge: Dict[str, Any]) -> Optional[CausalRelationship]:
# # #         """Create a causal relationship from an edge structure."""  # Module not found  # Module not found  # Module not found
        try:
            evidence_refs = self._create_evidence_references(edge.get("evidence", []))
            
            return CausalRelationship(
                source_node=str(edge.get("source", "")),
                target_node=str(edge.get("target", "")),
                relationship_type=edge.get("type", "causal"),
                confidence=float(edge.get("confidence", 0.5)),
                evidence_strength=self._calculate_evidence_strength(evidence_refs),
                evidence_references=tuple(evidence_refs),
                validity_score=0.0
            )
        except (KeyError, ValueError, TypeError) as e:
# # #             logger.warning(f"Failed to create causal relationship from edge: {e}")  # Module not found  # Module not found  # Module not found
            return None
    
    def _create_evidence_references(self, evidence_list: List[Dict[str, Any]]) -> List[EvidenceReference]:
# # #         """Create evidence references from evidence data."""  # Module not found  # Module not found  # Module not found
        references = []
        
        for evidence in evidence_list:
            try:
                ref = EvidenceReference(
                    source_id=str(evidence.get("source_id", "")),
                    page_numbers=list(evidence.get("page_numbers", [])),
                    text_snippet=str(evidence.get("text_snippet", "")),
                    confidence=float(evidence.get("confidence", 0.5))
                )
                references.append(ref)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to create evidence reference: {e}")
                continue
                
        return references
    
    def _calculate_evidence_strength(self, evidence_refs: List[EvidenceReference]) -> float:
# # #         """Calculate aggregate evidence strength from references."""  # Module not found  # Module not found  # Module not found
        if not evidence_refs:
            return 0.0
            
        # Weighted average of evidence confidences with diversity bonus
        total_confidence = sum(ref.confidence for ref in evidence_refs)
        avg_confidence = total_confidence / len(evidence_refs)
        
        # Diversity bonus based on unique sources
        unique_sources = len(set(ref.source_id for ref in evidence_refs))
        diversity_bonus = min(0.2, unique_sources * 0.05)
        
        return min(1.0, avg_confidence + diversity_bonus)
    
    def _classify_relationship_dimension(self, relationship: CausalRelationship) -> Optional[DimensionType]:
        """Classify causal relationship into appropriate dimension."""
        # Simple heuristic classification - can be enhanced with ML models
        source = relationship.source_node.lower()
        target = relationship.target_node.lower()
        rel_type = relationship.relationship_type.lower()
        
        # DE-1: Rights and freedoms
        de1_keywords = ["right", "freedom", "liberty", "civil", "political", "expression"]
        if any(keyword in source or keyword in target or keyword in rel_type 
               for keyword in de1_keywords):
            return DimensionType.DE1
            
        # DE-2: Social and economic
        de2_keywords = ["social", "economic", "poverty", "education", "health", "employment"]
        if any(keyword in source or keyword in target or keyword in rel_type 
               for keyword in de2_keywords):
            return DimensionType.DE2
            
        # DE-3: Environmental and territorial
        de3_keywords = ["environment", "territorial", "land", "resource", "climate", "ecosystem"]
        if any(keyword in source or keyword in target or keyword in rel_type 
               for keyword in de3_keywords):
            return DimensionType.DE3
            
        # DE-4: Institutional and governance
        de4_keywords = ["institutional", "governance", "policy", "administration", "government"]
        if any(keyword in source or keyword in target or keyword in rel_type 
               for keyword in de4_keywords):
            return DimensionType.DE4
        
        # Default to DE-1 if no classification
        return DimensionType.DE1
    
    def _build_dimension_graph(self, dimension: DimensionType) -> CausalGraphArtifact:
        """Build causal graph for specific dimension with cycle prevention."""
        logger.info(f"Building causal graph for dimension {dimension.value}")
        
        relationships = self.causal_relationships[dimension]
        
        if not relationships:
            return self._generate_sparse_artifact(dimension)
        
        # Filter relationships by thresholds
        filtered_relationships = [
            rel for rel in relationships 
            if (rel.confidence >= self.min_confidence_threshold and 
                rel.evidence_strength >= self.min_evidence_strength)
        ]
        
        if not filtered_relationships:
            return self._generate_sparse_artifact(dimension)
        
        # Build graph with cycle prevention
        graph = self.dimension_graphs[dimension]
        graph.clear()
        
        # Add nodes with deterministic ordering
        nodes = set()
        for rel in filtered_relationships:
            nodes.add(rel.source_node)
            nodes.add(rel.target_node)
        
        for node in sorted(nodes):
            graph.add_node(node, node_id=node)
        
        # Add edges with cycle prevention
        valid_edges = []
        for rel in filtered_relationships:
            if self._would_create_cycle(graph, rel.source_node, rel.target_node):
                logger.warning(f"Skipping edge {rel.source_node} -> {rel.target_node} to prevent cycle")
                continue
                
            # Calculate validity score
            validity_score = self._calculate_validity_score(rel)
            
            graph.add_edge(
                rel.source_node,
                rel.target_node,
                relationship=rel,
                validity_score=validity_score,
                weight=validity_score
            )
            valid_edges.append(rel)
        
        # Generate artifact
        return self._create_graph_artifact(dimension, graph, valid_edges)
    
    def _would_create_cycle(self, graph: nx.DiGraph, source: str, target: str) -> bool:
        """Check if adding an edge would create a cycle."""
        if not graph.has_node(source) or not graph.has_node(target):
            return False
            
# # #         # Check if there's already a path from target to source  # Module not found  # Module not found  # Module not found
        try:
            return nx.has_path(graph, target, source)
        except nx.NetworkXError:
            return False
    
    def _calculate_validity_score(self, relationship: CausalRelationship) -> float:
        """Calculate validity score for a causal relationship."""
        weights = self.validity_score_weights
        
        # Evidence strength component
        evidence_score = relationship.evidence_strength * weights["evidence_strength"]
        
        # Confidence component  
        confidence_score = relationship.confidence * weights["confidence"]
        
        # Evidence count component
        evidence_count = len(relationship.evidence_references)
        count_score = min(1.0, evidence_count / 3.0) * weights["evidence_count"]
        
        # Evidence diversity component
        unique_sources = len(set(ref.source_id for ref in relationship.evidence_references))
        diversity_score = min(1.0, unique_sources / 2.0) * weights["evidence_diversity"]
        
        total_score = evidence_score + confidence_score + count_score + diversity_score
        return min(1.0, max(0.0, total_score))
    
    def _create_graph_artifact(self, dimension: DimensionType, graph: nx.DiGraph, 
                             relationships: List[CausalRelationship]) -> CausalGraphArtifact:
# # #         """Create structured artifact from dimension graph."""  # Module not found  # Module not found  # Module not found
        # Generate nodes with deterministic ordering
        nodes = []
        for node_id in sorted(graph.nodes()):
            nodes.append({
                "id": node_id,
                "label": node_id,
                "degree_centrality": nx.degree_centrality(graph).get(node_id, 0.0),
                "betweenness_centrality": nx.betweenness_centrality(graph).get(node_id, 0.0),
                "in_degree": graph.in_degree(node_id),
                "out_degree": graph.out_degree(node_id)
            })
        
        # Generate edges with deterministic ordering
        edges = []
        for source, target, edge_data in sorted(graph.edges(data=True)):
            relationship = edge_data.get("relationship")
            if relationship:
                edges.append({
                    "source": source,
                    "target": target,
                    "relationship_type": relationship.relationship_type,
                    "confidence": relationship.confidence,
                    "evidence_strength": relationship.evidence_strength,
                    "validity_score": edge_data.get("validity_score", 0.0),
                    "evidence_references": [
                        {
                            "source_id": ref.source_id,
                            "page_numbers": list(ref.page_numbers),
                            "text_snippet": ref.text_snippet,
                            "confidence": ref.confidence
                        }
                        for ref in relationship.evidence_references
                    ]
                })
        
        # Calculate validity statistics
        validity_scores = [edge["validity_score"] for edge in edges]
        validity_statistics = {
            "mean_validity": sum(validity_scores) / len(validity_scores) if validity_scores else 0.0,
            "min_validity": min(validity_scores) if validity_scores else 0.0,
            "max_validity": max(validity_scores) if validity_scores else 0.0,
            "edge_count": len(edges),
            "node_count": len(nodes),
            "graph_density": nx.density(graph) if graph.number_of_nodes() > 0 else 0.0
        }
        
        # Generate metadata
        metadata = {
            "dimension": dimension.value,
            "generation_timestamp": self._get_deterministic_timestamp(),
            "component_id": self.component_id,
            "operation_id": self._last_operation_id,
            "is_acyclic": nx.is_directed_acyclic_graph(graph),
            "connected_components": nx.number_weakly_connected_components(graph),
            "processing_thresholds": {
                "min_confidence": self.min_confidence_threshold,
                "min_evidence_strength": self.min_evidence_strength
            }
        }
        
        return CausalGraphArtifact(
            dimension=dimension.value,
            nodes=nodes,
            edges=edges,
            metadata=metadata,
            validity_statistics=validity_statistics
        )
    
    def _generate_sparse_artifact(self, dimension: DimensionType) -> CausalGraphArtifact:
        """Generate sparse artifact when insufficient evidence is available."""
        metadata = {
            "dimension": dimension.value,
            "generation_timestamp": self._get_deterministic_timestamp(),
            "component_id": self.component_id,
            "operation_id": self._last_operation_id,
            "is_sparse": True,
            "reason": "Insufficient causal evidence available"
        }
        
        validity_statistics = {
            "mean_validity": 0.0,
            "min_validity": 0.0,
            "max_validity": 0.0,
            "edge_count": 0,
            "node_count": 0,
            "graph_density": 0.0
        }
        
        return CausalGraphArtifact(
            dimension=dimension.value,
            nodes=[],
            edges=[],
            metadata=metadata,
            validity_statistics=validity_statistics
        )
    
    def _generate_sparse_artifacts(self) -> Dict[str, CausalGraphArtifact]:
        """Generate sparse artifacts for all dimensions when processing fails."""
        artifacts = {}
        for dimension in DimensionType:
            artifacts[dimension.value] = self._generate_sparse_artifact(dimension)
        return artifacts
    
    def _generate_output_files(self, artifacts: Dict[str, CausalGraphArtifact]) -> None:
        """Generate output JSON files for each dimension."""
        for dimension_key, artifact in artifacts.items():
            filename = f"causal_{dimension_key}.json"
            filepath = self.output_directory / filename
            
            # Create serializable dictionary
            output_data = {
                "dimension": artifact.dimension,
                "nodes": artifact.nodes,
                "edges": artifact.edges,
                "metadata": artifact.metadata,
                "validity_statistics": artifact.validity_statistics
            }
            
            # Write with canonical JSON formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.canonical_json_dumps(output_data, indent=2))
            
            logger.info(f"Generated causal graph artifact: {filepath}")


def main():
    """Standalone execution for testing."""
    constructor = CausalGraphConstructor()
    
    # Sample test data
    evidence_artifacts = {
        "documents": {
            "doc1": {
                "content": "Rights violations lead to social unrest",
                "pages": [1, 2],
                "metadata": {"type": "report"}
            }
        }
    }
    
    knowledge_graph_artifacts = {
        "entity_relationships": [
            {
                "source": "rights_violations",
                "target": "social_unrest", 
                "type": "causes",
                "confidence": 0.8,
                "evidence": [
                    {
                        "source_id": "doc1",
                        "page_numbers": [1],
                        "text_snippet": "Rights violations lead to social unrest",
                        "confidence": 0.9
                    }
                ]
            }
        ]
    }
    
    # Process
    results = constructor.process(evidence_artifacts, knowledge_graph_artifacts)
    
    print(f"Generated {len(results)} causal graph artifacts")
    for dim, artifact in results.items():
        print(f"{dim}: {artifact.validity_statistics['node_count']} nodes, "
              f"{artifact.validity_statistics['edge_count']} edges")


if __name__ == "__main__":
    main()