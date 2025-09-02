"""
Theory of change validation component that compares detected causal chains 
against expected logical sequences.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from causal_graph import CausalGraph, CausalRelationType


class ValidationResult(Enum):
    """Results of theory of change validation."""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    MISSING_LINK = "missing_link"
    UNEXPECTED_LINK = "unexpected_link"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class TheoryOfChangeStep:
    """Represents a step in a theory of change."""
    id: str
    description: str
    intervention: str
    expected_outcome: str
    indicators: List[str]
    assumptions: List[str]
    timeframe: Optional[str] = None


@dataclass
class ValidationFinding:
    """Represents a validation finding."""
    result: ValidationResult
    expected_step: Optional[TheoryOfChangeStep]
    detected_chain: Optional[List[str]]
    confidence: float
    explanation: str
    evidence_spans: List[Tuple[int, int]]


class TheoryOfChangeValidator:
    """Validates detected causal chains against expected theory of change."""
    
    def __init__(self):
        self.theory_steps: List[TheoryOfChangeStep] = []
        self.validation_threshold = 0.6
    
    def load_theory_of_change(self, theory_data: Dict[str, Any]) -> None:
        """Load theory of change from structured data."""
        self.theory_steps = []
        
        for step_data in theory_data.get('steps', []):
            step = TheoryOfChangeStep(
                id=step_data['id'],
                description=step_data['description'],
                intervention=step_data['intervention'],
                expected_outcome=step_data['expected_outcome'],
                indicators=step_data.get('indicators', []),
                assumptions=step_data.get('assumptions', []),
                timeframe=step_data.get('timeframe')
            )
            self.theory_steps.append(step)
    
    def load_theory_from_file(self, file_path: str) -> None:
        """Load theory of change from JSON file."""
        with open(file_path, 'r') as f:
            theory_data = json.load(f)
        self.load_theory_of_change(theory_data)
    
    def validate_causal_graph(self, graph: CausalGraph) -> List[ValidationFinding]:
        """Validate detected causal graph against theory of change."""
        findings = []
        
        # Check each theory step against detected relationships
        for step in self.theory_steps:
            step_findings = self._validate_theory_step(step, graph)
            findings.extend(step_findings)
        
        # Check for unexpected causal chains
        unexpected_findings = self._detect_unexpected_chains(graph)
        findings.extend(unexpected_findings)
        
        return findings
    
    def _validate_theory_step(self, step: TheoryOfChangeStep, graph: CausalGraph) -> List[ValidationFinding]:
        """Validate a single theory step against the graph."""
        global nx
        findings = []
        
        # Find nodes matching intervention and outcome
        intervention_nodes = self._find_matching_nodes(step.intervention, graph)
        outcome_nodes = self._find_matching_nodes(step.expected_outcome, graph)
        
        if not intervention_nodes:
            finding = ValidationFinding(
                result=ValidationResult.MISSING_LINK,
                expected_step=step,
                detected_chain=None,
                confidence=0.0,
                explanation=f"Intervention '{step.intervention}' not found in detected entities",
                evidence_spans=[]
            )
            findings.append(finding)
            return findings
        
        if not outcome_nodes:
            finding = ValidationFinding(
                result=ValidationResult.MISSING_LINK,
                expected_step=step,
                detected_chain=None,
                confidence=0.0,
                explanation=f"Expected outcome '{step.expected_outcome}' not found in detected entities",
                evidence_spans=[]
            )
            findings.append(finding)
            return findings
        
        # Check for causal paths between intervention and outcome
        paths_found = []
        for intervention_node in intervention_nodes:
            for outcome_node in outcome_nodes:
                try:
                    import networkx as nx
                    paths = list(nx.all_simple_paths(
                        graph.graph, intervention_node, outcome_node, cutoff=5
                    ))
                    paths_found.extend(paths)
                except nx.NetworkXNoPath:
                    continue
        
        if paths_found:
            # Validate path quality
            best_path = self._select_best_path(paths_found, graph)
            path_confidence = self._calculate_path_confidence(best_path, graph)
            
            if path_confidence >= self.validation_threshold:
                finding = ValidationFinding(
                    result=ValidationResult.CONSISTENT,
                    expected_step=step,
                    detected_chain=best_path,
                    confidence=path_confidence,
                    explanation=f"Found consistent causal path from '{step.intervention}' to '{step.expected_outcome}'",
                    evidence_spans=self._get_path_evidence_spans(best_path, graph)
                )
            else:
                finding = ValidationFinding(
                    result=ValidationResult.INSUFFICIENT_EVIDENCE,
                    expected_step=step,
                    detected_chain=best_path,
                    confidence=path_confidence,
                    explanation=f"Causal path found but confidence ({path_confidence:.2f}) below threshold",
                    evidence_spans=self._get_path_evidence_spans(best_path, graph)
                )
            findings.append(finding)
        else:
            finding = ValidationFinding(
                result=ValidationResult.INCONSISTENT,
                expected_step=step,
                detected_chain=None,
                confidence=0.0,
                explanation=f"No causal path found between '{step.intervention}' and '{step.expected_outcome}'",
                evidence_spans=[]
            )
            findings.append(finding)
        
        return findings
    
    def _find_matching_nodes(self, target_text: str, graph: CausalGraph) -> List[str]:
        """Find graph nodes that match the target text."""
        matching_nodes = []
        target_words = set(target_text.lower().split())
        
        for node_id in graph.graph.nodes():
            node = graph.get_node(node_id)
            if node:
                node_words = set(node.text.lower().split())
                
                # Calculate word overlap
                overlap = len(target_words & node_words)
                total_words = len(target_words | node_words)
                
                if total_words > 0:
                    similarity = overlap / total_words
                    if similarity > 0.3:  # Adjust threshold as needed
                        matching_nodes.append(node_id)
        
        return matching_nodes
    
    def _select_best_path(self, paths: List[List[str]], graph: CausalGraph) -> List[str]:
        """Select the best causal path based on confidence and directness."""
        if not paths:
            return []
        
        best_path = None
        best_score = -1
        
        for path in paths:
            # Calculate path score based on length and confidence
            path_confidence = self._calculate_path_confidence(path, graph)
            length_penalty = 1.0 / len(path)  # Shorter paths are preferred
            score = path_confidence * length_penalty
            
            if score > best_score:
                best_score = score
                best_path = path
        
        return best_path or []
    
    def _calculate_path_confidence(self, path: List[str], graph: CausalGraph) -> float:
        """Calculate confidence score for a causal path."""
        if len(path) < 2:
            return 0.0
        
        edge_confidences = []
        for i in range(len(path) - 1):
            edge = graph.get_edge(path[i], path[i + 1])
            if edge:
                edge_confidences.append(edge.confidence)
            else:
                return 0.0  # Missing edge
        
        # Return minimum confidence (weakest link)
        return min(edge_confidences) if edge_confidences else 0.0
    
    def _get_path_evidence_spans(self, path: List[str], graph: CausalGraph) -> List[Tuple[int, int]]:
        """Get evidence spans for all edges in the path."""
        evidence_spans = []
        
        for i in range(len(path) - 1):
            edge = graph.get_edge(path[i], path[i + 1])
            if edge:
                evidence_spans.extend(edge.evidence_spans)
        
        return evidence_spans
    
    def _detect_unexpected_chains(self, graph: CausalGraph) -> List[ValidationFinding]:
        """Detect causal chains that don't match any theory step."""
        findings = []
        
        # Get all causal chains from the graph
        all_chains = graph.get_causal_chains(min_length=2)
        
        # Check each chain against theory steps
        for chain in all_chains:
            if not self._chain_matches_theory(chain, graph):
                chain_confidence = self._calculate_path_confidence(chain, graph)
                
                if chain_confidence >= self.validation_threshold:
                    finding = ValidationFinding(
                        result=ValidationResult.UNEXPECTED_LINK,
                        expected_step=None,
                        detected_chain=chain,
                        confidence=chain_confidence,
                        explanation=f"Detected causal chain not expected in theory of change: {' -> '.join([graph.get_node(node_id).text for node_id in chain])}",
                        evidence_spans=self._get_path_evidence_spans(chain, graph)
                    )
                    findings.append(finding)
        
        return findings
    
    def _chain_matches_theory(self, chain: List[str], graph: CausalGraph) -> bool:
        """Check if a detected chain matches any theory step."""
        if len(chain) < 2:
            return False
        
        source_text = graph.get_node(chain[0]).text
        target_text = graph.get_node(chain[-1]).text
        
        for step in self.theory_steps:
            if (self._text_similarity(source_text, step.intervention) > 0.5 and
                self._text_similarity(target_text, step.expected_outcome) > 0.5):
                return True
        
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        return overlap / total if total > 0 else 0.0
    
    def generate_validation_report(self, findings: List[ValidationFinding]) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        report = {
            'summary': {
                'total_findings': len(findings),
                'consistent': len([f for f in findings if f.result == ValidationResult.CONSISTENT]),
                'inconsistent': len([f for f in findings if f.result == ValidationResult.INCONSISTENT]),
                'missing_links': len([f for f in findings if f.result == ValidationResult.MISSING_LINK]),
                'unexpected_links': len([f for f in findings if f.result == ValidationResult.UNEXPECTED_LINK]),
                'insufficient_evidence': len([f for f in findings if f.result == ValidationResult.INSUFFICIENT_EVIDENCE])
            },
            'findings': [
                {
                    'result': finding.result.value,
                    'confidence': finding.confidence,
                    'explanation': finding.explanation,
                    'expected_step_id': finding.expected_step.id if finding.expected_step else None,
                    'detected_chain': finding.detected_chain,
                    'evidence_spans': finding.evidence_spans
                }
                for finding in findings
            ],
            'recommendations': self._generate_recommendations(findings)
        }
        
        return report
    
    def _generate_recommendations(self, findings: List[ValidationFinding]) -> List[str]:
        """Generate recommendations based on validation findings."""
        recommendations = []
        
        missing_links = [f for f in findings if f.result == ValidationResult.MISSING_LINK]
        if missing_links:
            recommendations.append(
                f"Consider reviewing {len(missing_links)} theory steps with missing causal evidence in the text."
            )
        
        inconsistent = [f for f in findings if f.result == ValidationResult.INCONSISTENT]
        if inconsistent:
            recommendations.append(
                f"Investigate {len(inconsistent)} theory steps with inconsistent causal relationships."
            )
        
        unexpected = [f for f in findings if f.result == ValidationResult.UNEXPECTED_LINK]
        if unexpected:
            recommendations.append(
                f"Review {len(unexpected)} unexpected causal relationships that may indicate theory gaps."
            )
        
        low_evidence = [f for f in findings if f.result == ValidationResult.INSUFFICIENT_EVIDENCE]
        if low_evidence:
            recommendations.append(
                f"Strengthen evidence for {len(low_evidence)} theory steps with low confidence scores."
            )
        
        return recommendations