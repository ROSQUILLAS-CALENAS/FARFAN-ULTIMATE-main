"""
knowledge_validator.py - Knowledge Integrity Validation Module

Canonical Flow Module: K_knowledge_extraction/knowledge_validator.py
Phase: K - Knowledge Extraction
Code: K

This module implements integrity validation logic for triples and graph structures,
validating knowledge artifacts for completeness, consistency, and structural integrity.
Provides pre-condition checks for knowledge artifacts and ensures compliance with
ontological constraints and semantic consistency requirements.

Implements the standard process(data, context) -> Dict[str, Any] interface for
seamless integration with the canonical pipeline orchestrator.
"""

# Phase metadata annotations for canonical naming convention
__phase__ = "K_knowledge_extraction"
__code__ = "K"
__stage_order__ = 11

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import networkx as nx

# Configure logging
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"

class KnowledgeArtifactType(Enum):
    """Types of knowledge artifacts to validate."""
    TRIPLES = "triples"
    GRAPH = "graph"
    ONTOLOGY = "ontology"
    CONCEPTS = "concepts"
    RELATIONS = "relations"

@dataclass
class ValidationResult:
    """Result of knowledge validation check."""
    level: ValidationLevel
    artifact_type: KnowledgeArtifactType
    message: str
    details: Dict[str, Any]
    passed: bool

@dataclass
class Triple:
    """RDF-style triple structure."""
    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None
    source: Optional[str] = None

class KnowledgeValidator:
    """
    Validates knowledge artifacts for integrity, consistency, and completeness.
    
    Implements comprehensive validation for:
    - Triple structure and semantics
    - Graph connectivity and consistency
    - Ontological constraints
    - Concept definitions and relationships
    - Cross-reference integrity
    """
    
    def __init__(self, validation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize knowledge validator.
        
        Args:
            validation_config: Configuration for validation thresholds and rules
        """
        self.config = validation_config or self._get_default_config()
        self.validation_results: List[ValidationResult] = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            "min_confidence_threshold": 0.3,
            "max_triples_per_entity": 1000,
            "required_relation_types": ["is_a", "part_of", "related_to"],
            "min_graph_connectivity": 0.1,
            "max_orphaned_nodes": 0.05,
            "enable_semantic_checks": True,
            "enable_consistency_checks": True,
            "enable_completeness_checks": True
        }
    
    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing interface for knowledge validation.
        
        Args:
            data: Knowledge artifacts to validate
            context: Processing context and metadata
            
        Returns:
            Validation results and integrity report
        """
        logger.info("Starting knowledge integrity validation")
        
        # Reset validation results
        self.validation_results = []
        
        # Validate pre-conditions
        precondition_results = self._validate_preconditions(data, context)
        
        # Validate different artifact types
        validation_results = {}
        
        if "triples" in data:
            validation_results["triples"] = self._validate_triples(data["triples"])
            
        if "graph" in data:
            validation_results["graph"] = self._validate_graph(data["graph"])
            
        if "ontology" in data:
            validation_results["ontology"] = self._validate_ontology(data["ontology"])
            
        if "concepts" in data:
            validation_results["concepts"] = self._validate_concepts(data["concepts"])
            
        if "relations" in data:
            validation_results["relations"] = self._validate_relations(data["relations"])
        
        # Generate integrity report
        integrity_report = self._generate_integrity_report()
        
        return {
            "validation_status": "completed",
            "precondition_checks": precondition_results,
            "validation_results": validation_results,
            "integrity_report": integrity_report,
            "total_issues": len([r for r in self.validation_results if not r.passed]),
            "critical_issues": len([r for r in self.validation_results 
                                  if not r.passed and r.level == ValidationLevel.CRITICAL]),
            "validation_summary": self._get_validation_summary()
        }
    
    def _validate_preconditions(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate pre-conditions for knowledge artifacts.
        
        Args:
            data: Input knowledge artifacts
            context: Processing context
            
        Returns:
            Pre-condition validation results
        """
        preconditions = {
            "data_format_valid": True,
            "required_fields_present": True,
            "context_complete": True,
            "dependencies_available": True
        }
        
        issues = []
        
        # Check data format
        if not isinstance(data, dict):
            preconditions["data_format_valid"] = False
            issues.append("Data must be a dictionary")
        
        # Check for at least one artifact type
        artifact_types = ["triples", "graph", "ontology", "concepts", "relations"]
        if not any(artifact_type in data for artifact_type in artifact_types):
            preconditions["required_fields_present"] = False
            issues.append("At least one knowledge artifact type must be present")
        
        # Check context completeness
        required_context_fields = ["source", "timestamp", "processing_stage"]
        for field in required_context_fields:
            if field not in context:
                preconditions["context_complete"] = False
                issues.append(f"Missing required context field: {field}")
        
        return {
            "passed": all(preconditions.values()),
            "checks": preconditions,
            "issues": issues
        }
    
    def _validate_triples(self, triples_data: Union[List[Dict], List[Triple]]) -> Dict[str, Any]:
        """
        Validate RDF-style triples for structure and semantics.
        
        Args:
            triples_data: List of triples to validate
            
        Returns:
            Triple validation results
        """
        if not triples_data:
            self._add_validation_result(
                ValidationLevel.WARNING,
                KnowledgeArtifactType.TRIPLES,
                "No triples provided for validation",
                {"count": 0}
            )
            return {"status": "empty", "issues": [], "statistics": {}}
        
        # Convert to Triple objects if needed
        triples = []
        for triple_data in triples_data:
            if isinstance(triple_data, dict):
                triples.append(Triple(
                    subject=triple_data.get("subject", ""),
                    predicate=triple_data.get("predicate", ""),
                    object=triple_data.get("object", ""),
                    confidence=triple_data.get("confidence"),
                    source=triple_data.get("source")
                ))
            else:
                triples.append(triple_data)
        
        issues = []
        statistics = {
            "total_triples": len(triples),
            "unique_subjects": len(set(t.subject for t in triples)),
            "unique_predicates": len(set(t.predicate for t in triples)),
            "unique_objects": len(set(t.object for t in triples)),
            "with_confidence": len([t for t in triples if t.confidence is not None])
        }
        
        # Validate triple structure
        for i, triple in enumerate(triples):
            if not triple.subject:
                issues.append(f"Triple {i}: Empty subject")
                self._add_validation_result(
                    ValidationLevel.CRITICAL,
                    KnowledgeArtifactType.TRIPLES,
                    f"Empty subject in triple {i}",
                    {"triple_index": i}
                )
            
            if not triple.predicate:
                issues.append(f"Triple {i}: Empty predicate")
                self._add_validation_result(
                    ValidationLevel.CRITICAL,
                    KnowledgeArtifactType.TRIPLES,
                    f"Empty predicate in triple {i}",
                    {"triple_index": i}
                )
            
            if not triple.object:
                issues.append(f"Triple {i}: Empty object")
                self._add_validation_result(
                    ValidationLevel.CRITICAL,
                    KnowledgeArtifactType.TRIPLES,
                    f"Empty object in triple {i}",
                    {"triple_index": i}
                )
            
            # Validate confidence scores
            if triple.confidence is not None:
                if not (0.0 <= triple.confidence <= 1.0):
                    issues.append(f"Triple {i}: Invalid confidence score {triple.confidence}")
                    self._add_validation_result(
                        ValidationLevel.WARNING,
                        KnowledgeArtifactType.TRIPLES,
                        f"Invalid confidence score in triple {i}",
                        {"triple_index": i, "confidence": triple.confidence}
                    )
                elif triple.confidence < self.config["min_confidence_threshold"]:
                    self._add_validation_result(
                        ValidationLevel.INFO,
                        KnowledgeArtifactType.TRIPLES,
                        f"Low confidence score in triple {i}",
                        {"triple_index": i, "confidence": triple.confidence}
                    )
        
        # Check for semantic consistency
        if self.config["enable_semantic_checks"]:
            semantic_issues = self._validate_triple_semantics(triples)
            issues.extend(semantic_issues)
        
        return {
            "status": "completed",
            "issues": issues,
            "statistics": statistics,
            "validation_passed": len(issues) == 0
        }
    
    def _validate_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate graph structure for connectivity and consistency.
        
        Args:
            graph_data: Graph structure to validate
            
        Returns:
            Graph validation results
        """
        if not graph_data:
            self._add_validation_result(
                ValidationLevel.WARNING,
                KnowledgeArtifactType.GRAPH,
                "No graph data provided",
                {}
            )
            return {"status": "empty", "issues": [], "statistics": {}}
        
        issues = []
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        
        # Add nodes and edges
        if "nodes" in graph_data:
            for node in graph_data["nodes"]:
                node_id = node.get("id") if isinstance(node, dict) else str(node)
                G.add_node(node_id)
        
        if "edges" in graph_data:
            for edge in graph_data["edges"]:
                if isinstance(edge, dict):
                    source = edge.get("source")
                    target = edge.get("target")
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    source, target = edge[0], edge[1]
                else:
                    continue
                
                if source and target:
                    G.add_edge(source, target)
        
        # Calculate graph statistics
        statistics = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 0 else 0,
            "connected_components": nx.number_connected_components(G),
            "largest_component_size": len(max(nx.connected_components(G), key=len, default=[])),
            "isolated_nodes": len(list(nx.isolates(G)))
        }
        
        # Validate connectivity
        if statistics["density"] < self.config["min_graph_connectivity"]:
            issues.append(f"Low graph connectivity: {statistics['density']:.3f}")
            self._add_validation_result(
                ValidationLevel.WARNING,
                KnowledgeArtifactType.GRAPH,
                "Low graph connectivity detected",
                {"density": statistics["density"]}
            )
        
        # Check for excessive isolated nodes
        if statistics["total_nodes"] > 0:
            orphaned_ratio = statistics["isolated_nodes"] / statistics["total_nodes"]
            if orphaned_ratio > self.config["max_orphaned_nodes"]:
                issues.append(f"Too many isolated nodes: {orphaned_ratio:.3f}")
                self._add_validation_result(
                    ValidationLevel.WARNING,
                    KnowledgeArtifactType.GRAPH,
                    "Excessive isolated nodes detected",
                    {"orphaned_ratio": orphaned_ratio}
                )
        
        # Validate graph consistency
        if self.config["enable_consistency_checks"]:
            consistency_issues = self._validate_graph_consistency(G, graph_data)
            issues.extend(consistency_issues)
        
        return {
            "status": "completed",
            "issues": issues,
            "statistics": statistics,
            "validation_passed": len(issues) == 0
        }
    
    def _validate_ontology(self, ontology_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ontology structure and constraints.
        
        Args:
            ontology_data: Ontology to validate
            
        Returns:
            Ontology validation results
        """
        issues = []
        statistics = {}
        
        # Check basic ontology structure
        required_fields = ["classes", "properties", "individuals"]
        for field in required_fields:
            if field not in ontology_data:
                issues.append(f"Missing ontology field: {field}")
                self._add_validation_result(
                    ValidationLevel.CRITICAL,
                    KnowledgeArtifactType.ONTOLOGY,
                    f"Missing required ontology field: {field}",
                    {"field": field}
                )
        
        # Validate class hierarchy
        if "classes" in ontology_data:
            class_issues = self._validate_class_hierarchy(ontology_data["classes"])
            issues.extend(class_issues)
            statistics["class_count"] = len(ontology_data["classes"])
        
        # Validate properties
        if "properties" in ontology_data:
            property_issues = self._validate_properties(ontology_data["properties"])
            issues.extend(property_issues)
            statistics["property_count"] = len(ontology_data["properties"])
        
        return {
            "status": "completed",
            "issues": issues,
            "statistics": statistics,
            "validation_passed": len(issues) == 0
        }
    
    def _validate_concepts(self, concepts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate concept definitions and relationships.
        
        Args:
            concepts_data: List of concepts to validate
            
        Returns:
            Concept validation results
        """
        issues = []
        statistics = {
            "total_concepts": len(concepts_data),
            "with_definitions": 0,
            "with_relationships": 0
        }
        
        for i, concept in enumerate(concepts_data):
            if not isinstance(concept, dict):
                issues.append(f"Concept {i}: Invalid format")
                continue
            
            # Check required fields
            if "name" not in concept or not concept["name"]:
                issues.append(f"Concept {i}: Missing or empty name")
                self._add_validation_result(
                    ValidationLevel.CRITICAL,
                    KnowledgeArtifactType.CONCEPTS,
                    f"Missing name in concept {i}",
                    {"concept_index": i}
                )
            
            if "definition" in concept and concept["definition"]:
                statistics["with_definitions"] += 1
            
            if "relationships" in concept and concept["relationships"]:
                statistics["with_relationships"] += 1
        
        return {
            "status": "completed",
            "issues": issues,
            "statistics": statistics,
            "validation_passed": len(issues) == 0
        }
    
    def _validate_relations(self, relations_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate relation definitions and constraints.
        
        Args:
            relations_data: List of relations to validate
            
        Returns:
            Relation validation results
        """
        issues = []
        statistics = {
            "total_relations": len(relations_data),
            "by_type": {}
        }
        
        for i, relation in enumerate(relations_data):
            if not isinstance(relation, dict):
                issues.append(f"Relation {i}: Invalid format")
                continue
            
            # Check required fields
            if "type" not in relation or not relation["type"]:
                issues.append(f"Relation {i}: Missing relation type")
                self._add_validation_result(
                    ValidationLevel.CRITICAL,
                    KnowledgeArtifactType.RELATIONS,
                    f"Missing type in relation {i}",
                    {"relation_index": i}
                )
            else:
                rel_type = relation["type"]
                statistics["by_type"][rel_type] = statistics["by_type"].get(rel_type, 0) + 1
            
            # Validate domain and range if present
            if "domain" in relation and "range" in relation:
                if not relation["domain"] or not relation["range"]:
                    issues.append(f"Relation {i}: Empty domain or range")
        
        return {
            "status": "completed",
            "issues": issues,
            "statistics": statistics,
            "validation_passed": len(issues) == 0
        }
    
    def _validate_triple_semantics(self, triples: List[Triple]) -> List[str]:
        """Validate semantic consistency of triples."""
        issues = []
        
        # Check for contradictory triples
        subject_predicates = {}
        for triple in triples:
            key = (triple.subject, triple.predicate)
            if key not in subject_predicates:
                subject_predicates[key] = []
            subject_predicates[key].append(triple.object)
        
        # Look for functional properties with multiple values
        for (subject, predicate), objects in subject_predicates.items():
            if len(set(objects)) > 1:
                issues.append(f"Multiple objects for {subject} {predicate}: {objects}")
        
        return issues
    
    def _validate_graph_consistency(self, graph: nx.Graph, graph_data: Dict[str, Any]) -> List[str]:
        """Validate internal graph consistency."""
        issues = []
        
        # Check for dangling references
        if "edges" in graph_data:
            for edge in graph_data["edges"]:
                if isinstance(edge, dict):
                    source = edge.get("source")
                    target = edge.get("target")
                    if source not in graph or target not in graph:
                        issues.append(f"Dangling edge reference: {source} -> {target}")
        
        return issues
    
    def _validate_class_hierarchy(self, classes: List[Dict[str, Any]]) -> List[str]:
        """Validate class hierarchy consistency."""
        issues = []
        class_names = set()
        
        for cls in classes:
            if "name" not in cls:
                issues.append("Class missing name field")
                continue
            
            class_names.add(cls["name"])
            
            # Check for circular inheritance
            if "parent" in cls and cls["parent"] == cls["name"]:
                issues.append(f"Circular inheritance: {cls['name']}")
        
        # Check parent references
        for cls in classes:
            if "parent" in cls and cls["parent"] not in class_names:
                issues.append(f"Unknown parent class: {cls['parent']}")
        
        return issues
    
    def _validate_properties(self, properties: List[Dict[str, Any]]) -> List[str]:
        """Validate property definitions."""
        issues = []
        
        for prop in properties:
            if "name" not in prop:
                issues.append("Property missing name field")
                continue
            
            if "type" not in prop:
                issues.append(f"Property {prop['name']} missing type")
        
        return issues
    
    def _add_validation_result(self, level: ValidationLevel, artifact_type: KnowledgeArtifactType, 
                             message: str, details: Dict[str, Any], passed: bool = False):
        """Add a validation result to the results list."""
        self.validation_results.append(ValidationResult(
            level=level,
            artifact_type=artifact_type,
            message=message,
            details=details,
            passed=passed
        ))
    
    def _generate_integrity_report(self) -> Dict[str, Any]:
        """Generate comprehensive integrity report."""
        critical_count = len([r for r in self.validation_results if r.level == ValidationLevel.CRITICAL])
        warning_count = len([r for r in self.validation_results if r.level == ValidationLevel.WARNING])
        info_count = len([r for r in self.validation_results if r.level == ValidationLevel.INFO])
        
        return {
            "overall_status": "pass" if critical_count == 0 else "fail",
            "issue_counts": {
                "critical": critical_count,
                "warning": warning_count,
                "info": info_count
            },
            "recommendations": self._generate_recommendations(),
            "validation_timestamp": hashlib.sha256(str(self.validation_results).encode()).hexdigest()[:16]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        critical_results = [r for r in self.validation_results if r.level == ValidationLevel.CRITICAL]
        if critical_results:
            recommendations.append("Address critical issues before proceeding to next pipeline stage")
        
        warning_results = [r for r in self.validation_results if r.level == ValidationLevel.WARNING]
        if warning_results:
            recommendations.append("Review warnings to improve knowledge quality")
        
        return recommendations
    
    def _get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            "total_checks": len(self.validation_results),
            "passed_checks": len([r for r in self.validation_results if r.passed]),
            "failed_checks": len([r for r in self.validation_results if not r.passed]),
            "artifact_types_checked": list(set(r.artifact_type.value for r in self.validation_results))
        }


def main():
    """Example usage of knowledge validator."""
    # Sample knowledge data
    sample_data = {
        "triples": [
            {"subject": "Paris", "predicate": "is_capital_of", "object": "France", "confidence": 0.95},
            {"subject": "London", "predicate": "is_capital_of", "object": "UK", "confidence": 0.98},
            {"subject": "", "predicate": "located_in", "object": "Europe", "confidence": 0.8}  # Invalid
        ],
        "graph": {
            "nodes": [{"id": "Paris"}, {"id": "France"}, {"id": "Europe"}],
            "edges": [{"source": "Paris", "target": "France"}, {"source": "France", "target": "Europe"}]
        }
    }
    
    context = {
        "source": "knowledge_extraction",
        "timestamp": "2024-01-01T00:00:00Z",
        "processing_stage": "K_knowledge_extraction"
    }
    
    validator = KnowledgeValidator()
    results = validator.process(sample_data, context)
    
    print("Knowledge Validation Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()