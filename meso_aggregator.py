"""
Meso Aggregator with Decálogo Cluster-Based Grouping System
# # # - Aggregates results from the four clusters (C1-C4) by computing divergence metrics  # Module not found  # Module not found  # Module not found
- Generates a coverage matrix tracking which Development Plan components have been evaluated
- Implements cluster-based grouping system for the 10 Decálogo points organized into 4 predefined clusters
- Calculates weighted scores for each cluster using individual point scores as inputs
- Generates coverage metrics tracking how many points within each cluster have sufficient evidence
- Adds divergence analysis functionality to identify inconsistencies between points within same cluster
- Implements cross-point evidence linkage detecting when same evidence supports multiple points

Input expectations:
- cluster_audit.micro: {C#: {answers: [{question_id, verdict, score?, evidence_ids[], components?, decalogo_point?}]}}
- Development Plan components referenced in answers
- Decálogo point mappings and evidence items

Output:
- meso_summary with per-question aggregation, divergence metrics, and Decálogo cluster analysis
- coverage_matrix tracking component evaluation across clusters
- decalogo_cluster_analysis with weighted scores, coverage metrics, and evidence linkage
"""

# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import hashlib
import json
import logging
import os
# # # from typing import Any, Dict, List, Set  # Module not found  # Module not found  # Module not found

import numpy as np
# # # from scipy.spatial.distance import cosine  # Module not found  # Module not found  # Module not found
# # # from difflib import SequenceMatcher  # Module not found  # Module not found  # Module not found
# # # from itertools import combinations  # Module not found  # Module not found  # Module not found

# Development Plan Components enum values
DEVELOPMENT_PLAN_COMPONENTS = [
    "OBJECTIVES",
    "STRATEGIES",
    "INDICATORS",
    "TIMELINES",
    "BUDGET",
    "STAKEHOLDERS",
    "RISKS",
    "COMPLIANCE",
    "SUSTAINABILITY",
    "IMPACT",
]

REQUIRED_CLUSTERS = ["C1", "C2", "C3", "C4"]

# Decálogo Points organized into 4 predefined clusters
DECALOGO_CLUSTER_MAPPING = {
    1: {  # Cluster 1: PAZ, SEGURIDAD Y PROTECCIÓN DE DEFENSORES
        "name": "Paz, Seguridad y Protección de Defensores",
        "points": [1, 5, 8],
        "point_descriptions": {
            1: "Prevención de la violencia y protección de la población",
            5: "Derechos de las víctimas y construcción de paz", 
            8: "Líderes y defensores de derechos humanos"
        },
        "weight": 0.30  # Higher weight for security and peace
    },
    2: {  # Cluster 2: DERECHOS SOCIALES FUNDAMENTALES
        "name": "Derechos Sociales Fundamentales",
        "points": [2, 3, 4],
        "point_descriptions": {
            2: "Derecho a la salud",
            3: "Derecho a la educación",
            4: "Derecho a la alimentación"
        },
        "weight": 0.35  # Highest weight for fundamental social rights
    },
    3: {  # Cluster 3: IGUALDAD Y NO DISCRIMINACIÓN
        "name": "Igualdad y No Discriminación",
        "points": [6, 7],
        "point_descriptions": {
            6: "Derechos de las mujeres",
            7: "Derechos de niñas, niños y adolescentes"
        },
        "weight": 0.25  # Important for equality
    },
    4: {  # Cluster 4: DERECHOS TERRITORIALES Y AMBIENTALES
        "name": "Derechos Territoriales y Ambientales",
        "points": [9, 10],
        "point_descriptions": {
            9: "Derechos de los pueblos étnicos",
            10: "Derecho a un ambiente sano"
        },
        "weight": 0.10  # Lower weight but still essential
    }
}

# Evidence similarity threshold for cross-point linkage detection
EVIDENCE_SIMILARITY_THRESHOLD = 0.85
MINIMUM_EVIDENCE_COUNT = 2

# Set up logging
logger = logging.getLogger(__name__)


def canonical_json_dumps(obj: Any) -> str:
    """
    Create deterministic JSON serialization with sorted keys, consistent indentation,
    and UTF-8 encoding to ensure identical output files for the same input data.
    
    Args:
        obj: Object to serialize to JSON
        
    Returns:
        Deterministic JSON string representation
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        separators=(',', ': ')
    )


def generate_evidence_hash_id(evidence: Dict[str, Any]) -> str:
    """
    Generate a stable hash-based ID for evidence items based on their content.
    
    Args:
        evidence: Evidence dictionary containing text, metadata, etc.
        
    Returns:
        Stable hash-based identifier
    """
    # Create canonical representation for hashing
    content_for_hash = {
        'text': evidence.get('text', ''),
        'context': evidence.get('context', ''),
        'question_id': evidence.get('question_id', ''),
        'dimension': evidence.get('dimension', ''),
    }
    
    # Add metadata keys in sorted order
    metadata = evidence.get('metadata', {})
    if isinstance(metadata, dict):
        content_for_hash['metadata'] = {k: metadata[k] for k in sorted(metadata.keys())}
    
# # #     # Create hash from canonical JSON representation  # Module not found  # Module not found  # Module not found
    content_json = canonical_json_dumps(content_for_hash)
    hash_obj = hashlib.sha256(content_json.encode('utf-8'))
    return f"ev_{hash_obj.hexdigest()[:16]}"


def sort_evidence_items(evidence_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort evidence items by their hash-based IDs (or creation timestamps as fallback)
    to guarantee consistent ordering across multiple runs.
    
    Args:
        evidence_items: List of evidence dictionaries
        
    Returns:
        Sorted list of evidence items
    """
    def sort_key(evidence):
        # Primary sort by hash-based ID
        hash_id = generate_evidence_hash_id(evidence)
        
        # Secondary sort by explicit evidence_id if available
        evidence_id = evidence.get('evidence_id', '')
        
        # Tertiary sort by timestamp if available
        timestamp = evidence.get('timestamp', evidence.get('created_at', ''))
        
        # Quaternary sort by content hash as final fallback
        content_hash = hashlib.md5(str(evidence).encode('utf-8')).hexdigest()
        
        return (hash_id, evidence_id, timestamp, content_hash)
    
    return sorted(evidence_items, key=sort_key)


def group_evidence_by_dimension(evidence_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group evidence items by dimension using stable string keys and deterministic grouping.
    
    Args:
        evidence_items: List of evidence dictionaries
        
    Returns:
        Dictionary mapping dimension names to sorted lists of evidence items
    """
    dimension_groups = {}
    
    for evidence in evidence_items:
        # Normalize dimension key
        dimension = evidence.get('dimension', 'unknown')
        dimension_key = str(dimension).strip().lower()
        
        # Create stable string key for dimension
        if dimension_key not in dimension_groups:
            dimension_groups[dimension_key] = []
        
        dimension_groups[dimension_key].append(evidence)
    
    # Sort evidence within each dimension group
    for dimension_key in dimension_groups:
        dimension_groups[dimension_key] = sort_evidence_items(dimension_groups[dimension_key])
    
    # Return with sorted dimension keys for deterministic iteration
    return {k: dimension_groups[k] for k in sorted(dimension_groups.keys())}


def canonicalize_dimensions(dimensions: List[str]) -> List[str]:
    """
    Ensure deterministic dimension grouping by sorting dimensions alphabetically.
    
    Args:
        dimensions: List of dimension names
        
    Returns:
        Alphabetically sorted list of dimension names
    """
    # Normalize and sort dimensions
    normalized_dims = [str(dim).strip() for dim in dimensions if dim]
    return sorted(list(set(normalized_dims)))


def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """Calculate Jensen-Shannon divergence between two probability distributions."""
    if not p or not q or len(p) != len(q):
        return 0.0

    # Convert to numpy arrays and normalize
    p = np.array(p) + 1e-10  # Add small epsilon to avoid log(0)
    q = np.array(q) + 1e-10
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Calculate M = (P + Q) / 2
    m = (p + q) / 2

    # Calculate KL divergences
    kl_p_m = np.sum(p * np.log(p / m))
    kl_q_m = np.sum(q * np.log(q / m))

    # Jensen-Shannon divergence
    js_div = 0.5 * kl_p_m + 0.5 * kl_q_m
    return float(js_div)


def cosine_similarity_score(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    try:
        # Using scipy's cosine distance, then converting to similarity
        cos_dist = cosine(vec1, vec2)
        return 1.0 - cos_dist
    except (ValueError, ZeroDivisionError):
        return 0.0


def extract_components_from_answer(answer: Dict[str, Any]) -> Set[str]:
    """Extract Development Plan components referenced in an answer."""
    components = set()

    # Check explicit components field
    if "components" in answer:
        comps = answer["components"]
        if isinstance(comps, list):
            components.update(str(c).upper() for c in comps)
        elif isinstance(comps, str):
            components.add(comps.upper())

    # Check for component references in text fields
    text_fields = ["question", "verdict", "explanation", "rationale"]
    for field in text_fields:
        text = str(answer.get(field, "")).upper()
        for component in DEVELOPMENT_PLAN_COMPONENTS:
            if component in text:
                components.add(component)

    return components


def build_coverage_matrix(by_question: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build a coverage matrix showing which components are evaluated by which clusters."""
    coverage_matrix = {}

    # Initialize matrix for all components (sorted alphabetically)
    for component in canonicalize_dimensions(DEVELOPMENT_PLAN_COMPONENTS):
        coverage_matrix[component] = {
            "clusters_evaluating": set(),
            "questions_addressing": set(),
            "total_evaluations": 0,
            "coverage_percentage": 0.0,
        }

    # Fill matrix based on answers with deterministic processing
    for question_id in sorted(by_question.keys()):
        slot = by_question[question_id]
        question_components = set()

        # Process clusters in sorted order for deterministic results
        for cluster_id in sorted(slot["by_cluster"].keys()):
            answer = slot["by_cluster"][cluster_id]
            if not isinstance(answer, dict):
                continue

            answer_components = extract_components_from_answer(answer)
            question_components.update(answer_components)

            for component in sorted(answer_components):
                if component in coverage_matrix:
                    coverage_matrix[component]["clusters_evaluating"].add(cluster_id)
                    coverage_matrix[component]["questions_addressing"].add(question_id)
                    coverage_matrix[component]["total_evaluations"] += 1

        # Track which components this question addresses overall
        for component in sorted(question_components):
            if component in coverage_matrix:
                coverage_matrix[component]["questions_addressing"].add(question_id)

    # Calculate coverage percentages and convert sets to sorted lists for JSON serialization
    for component in coverage_matrix:
        clusters_count = len(coverage_matrix[component]["clusters_evaluating"])
        coverage_matrix[component]["coverage_percentage"] = (
            clusters_count / len(REQUIRED_CLUSTERS)
        ) * 100

        # Convert sets to sorted lists for deterministic output
        coverage_matrix[component]["clusters_evaluating"] = sorted(
            list(coverage_matrix[component]["clusters_evaluating"])
        )
        coverage_matrix[component]["questions_addressing"] = sorted(
            list(coverage_matrix[component]["questions_addressing"])
        )

    return coverage_matrix


def calculate_cluster_divergences(
    by_question: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Calculate divergence metrics between cluster responses for each question."""
    question_divergences = {}

    # Process questions in sorted order for deterministic results
    for question_id in sorted(by_question.keys()):
        slot = by_question[question_id]
        cluster_answers = slot["by_cluster"]

        # Extract score vectors for each cluster in sorted order
        cluster_vectors = {}
        cluster_scores = {}

        for cluster_id in sorted(REQUIRED_CLUSTERS):
            answer = cluster_answers.get(cluster_id, {})
            if isinstance(answer, dict):
# # #                 # Create feature vector from answer  # Module not found  # Module not found  # Module not found
                score = answer.get("score", answer.get("confidence", 0.5))
                verdict_score = (
                    1.0
                    if str(answer.get("verdict", "")).upper()
                    in ["YES", "TRUE", "POSITIVE"]
                    else 0.0
                )
                evidence_count = len(answer.get("evidence_ids", []))
                components_count = len(extract_components_from_answer(answer))

                # Ensure all vector elements are numeric
                try:
                    score_val = float(score) if isinstance(score, (int, float, str)) else 0.5
                except (ValueError, TypeError):
                    score_val = 0.5
                
                cluster_vectors[cluster_id] = [
                    score_val,
                    verdict_score,
                    evidence_count,
                    components_count,
                ]
                try:
                    cluster_scores[cluster_id] = float(score)
                except (ValueError, TypeError):
                    # Use default score for invalid values
                    cluster_scores[cluster_id] = 0.5

        # Calculate divergences between cluster pairs
        divergences = {
            "jensen_shannon_max": 0.0,
            "jensen_shannon_avg": 0.0,
            "cosine_similarity_min": 1.0,
            "cosine_similarity_avg": 0.0,
            "score_range": 0.0,
            "score_std": 0.0,
        }

        if len(cluster_vectors) >= 2:
            js_divergences = []
            cos_similarities = []
            scores = list(cluster_scores.values())

            # Calculate pairwise metrics with sorted cluster pairs
            cluster_ids = sorted(cluster_vectors.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    c1, c2 = cluster_ids[i], cluster_ids[j]
                    vec1, vec2 = cluster_vectors[c1], cluster_vectors[c2]

                    # Normalize vectors for probability distribution
                    vec1_norm = [max(0.01, abs(x)) for x in vec1]
                    vec2_norm = [max(0.01, abs(x)) for x in vec2]

                    js_div = jensen_shannon_divergence(vec1_norm, vec2_norm)
                    cos_sim = cosine_similarity_score(vec1, vec2)

                    js_divergences.append(js_div)
                    cos_similarities.append(cos_sim)

            # Aggregate metrics
            if js_divergences:
                divergences["jensen_shannon_max"] = max(js_divergences)
                divergences["jensen_shannon_avg"] = sum(js_divergences) / len(
                    js_divergences
                )

            if cos_similarities:
                divergences["cosine_similarity_min"] = min(cos_similarities)
                divergences["cosine_similarity_avg"] = sum(cos_similarities) / len(
                    cos_similarities
                )

            if scores:
                divergences["score_range"] = max(scores) - min(scores)
                divergences["score_std"] = (
                    float(np.std(scores)) if len(scores) > 1 else 0.0
                )

        question_divergences[question_id] = divergences

    return question_divergences


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()


def match_evidence_by_page_and_similarity(evidence1: Dict[str, Any], evidence2: Dict[str, Any]) -> bool:
    """Check if two evidence items match by page_num and text similarity."""
    # Check page number match
    page1 = evidence1.get('metadata', {}).get('page_num') or evidence1.get('page_num')
    page2 = evidence2.get('metadata', {}).get('page_num') or evidence2.get('page_num')
    
    if page1 and page2 and str(page1) == str(page2):
        # Check text similarity
        text1 = evidence1.get('text', '')
        text2 = evidence2.get('text', '')
        
        similarity = calculate_text_similarity(text1, text2)
        return similarity >= EVIDENCE_SIMILARITY_THRESHOLD
    
    return False


def analyze_decalogo_clusters(by_question: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the 10 Decálogo points organized into 4 predefined clusters.
    
    Returns:
        Dictionary containing cluster analysis results including:
        - Weighted scores for each cluster
        - Coverage metrics for points within each cluster
        - Divergence analysis between points in same cluster
        - Cross-point evidence linkage statistics
    """
    cluster_analysis = {}
    
    # Initialize cluster structures
    for cluster_id in DECALOGO_CLUSTER_MAPPING:
        cluster_info = DECALOGO_CLUSTER_MAPPING[cluster_id]
        cluster_analysis[f"cluster_{cluster_id}"] = {
            "name": cluster_info["name"],
            "points": cluster_info["points"],
            "weight": cluster_info["weight"],
            "point_scores": {},
            "weighted_cluster_score": 0.0,
            "coverage_metrics": {
                "total_points": len(cluster_info["points"]),
                "points_with_evidence": 0,
                "points_with_sufficient_evidence": 0,
                "coverage_percentage": 0.0,
                "evidence_quality_avg": 0.0
            },
            "divergence_analysis": {
                "intra_cluster_divergences": {},
                "inconsistencies": [],
                "compliance_conflicts": 0
            },
            "evidence_linkage": {
                "shared_evidence_pairs": {},
                "evidence_reuse_count": 0,
                "cross_point_references": {}
            }
        }
    
# # #     # Collect point data from answers  # Module not found  # Module not found  # Module not found
    point_data = {}  # {decalogo_point: {question_id: answer_data, ...}}
    
    for question_id, slot in sorted(by_question.items()):
        for cluster_key, answer in sorted(slot["by_cluster"].items()):
            if not isinstance(answer, dict):
                continue
                
# # #             # Extract decálogo point from answer  # Module not found  # Module not found  # Module not found
            decalogo_point = answer.get('decalogo_point')
            if not decalogo_point:
# # #                 # Try to infer from question_id or other fields  # Module not found  # Module not found  # Module not found
                if 'P' in question_id:
                    try:
                        point_part = question_id.split('P')[1].split('_')[0]
                        decalogo_point = int(point_part)
                    except (ValueError, IndexError):
                        continue
                else:
                    continue
            
            decalogo_point = int(decalogo_point)
            if decalogo_point not in range(1, 11):  # Valid Decálogo points 1-10
                continue
                
            if decalogo_point not in point_data:
                point_data[decalogo_point] = {}
            
            point_data[decalogo_point][question_id] = {
                "answer": answer,
                "score": answer.get("score", answer.get("confidence", 0.5)),
                "verdict": answer.get("verdict", ""),
                "evidence_ids": answer.get("evidence_ids", []),
                "evidence_items": answer.get("evidence_items", []),
                "cluster_key": cluster_key
            }
    
    # Process each cluster
    for cluster_id, cluster_info in DECALOGO_CLUSTER_MAPPING.items():
        cluster_key = f"cluster_{cluster_id}"
        cluster_data = cluster_analysis[cluster_key]
        
        point_scores = []
        points_with_evidence = 0
        points_sufficient_evidence = 0
        quality_scores = []
        
        # Process each point in the cluster
        for point_num in cluster_info["points"]:
            if point_num in point_data:
                point_questions = point_data[point_num]
                
                # Calculate aggregated score for this point
                question_scores = []
                total_evidence = 0
                verdicts = []
                
                for question_id, q_data in sorted(point_questions.items()):
                    try:
                        score = float(q_data["score"])
                        question_scores.append(score)
                    except (ValueError, TypeError):
                        question_scores.append(0.5)  # Default score
                    
                    total_evidence += len(q_data["evidence_ids"])
                    verdicts.append(q_data["verdict"].upper())
                
                # Calculate point score
                if question_scores:
                    point_score = sum(question_scores) / len(question_scores)
                    point_scores.append(point_score)
                    cluster_data["point_scores"][point_num] = point_score
                    
                    # Coverage metrics
                    if total_evidence > 0:
                        points_with_evidence += 1
                        if total_evidence >= MINIMUM_EVIDENCE_COUNT:
                            points_sufficient_evidence += 1
                    
                    # Quality based on evidence count and score consistency
                    evidence_quality = min(1.0, total_evidence / 10.0)  # Normalize to [0,1]
                    score_variance = np.var(question_scores) if len(question_scores) > 1 else 0.0
                    quality = evidence_quality * (1.0 - min(0.5, score_variance))
                    quality_scores.append(quality)
        
        # Calculate cluster weighted score
        if point_scores:
            cluster_data["weighted_cluster_score"] = sum(point_scores) / len(point_scores)
        
        # Update coverage metrics
        cluster_data["coverage_metrics"].update({
            "points_with_evidence": points_with_evidence,
            "points_with_sufficient_evidence": points_sufficient_evidence,
            "coverage_percentage": (points_with_evidence / len(cluster_info["points"])) * 100,
            "evidence_quality_avg": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        })
        
        # Analyze divergences between points in same cluster
        cluster_data["divergence_analysis"] = analyze_intra_cluster_divergence(
            cluster_info["points"], point_data
        )
        
        # Analyze evidence linkage within cluster
        cluster_data["evidence_linkage"] = analyze_cross_point_evidence_linkage(
            cluster_info["points"], point_data
        )
    
    # Calculate overall statistics
    overall_stats = {
        "total_clusters": len(DECALOGO_CLUSTER_MAPPING),
        "cluster_weighted_scores": {
            f"cluster_{cid}": cluster_analysis[f"cluster_{cid}"]["weighted_cluster_score"]
            for cid in DECALOGO_CLUSTER_MAPPING.keys()
        },
        "overall_weighted_score": sum(
            cluster_analysis[f"cluster_{cid}"]["weighted_cluster_score"] * 
            DECALOGO_CLUSTER_MAPPING[cid]["weight"]
            for cid in DECALOGO_CLUSTER_MAPPING.keys()
        ),
        "coverage_summary": {
            f"cluster_{cid}": cluster_analysis[f"cluster_{cid}"]["coverage_metrics"]["coverage_percentage"]
            for cid in DECALOGO_CLUSTER_MAPPING.keys()
        }
    }
    
    return {
        "clusters": cluster_analysis,
        "overall_statistics": overall_stats,
        "cluster_mapping": DECALOGO_CLUSTER_MAPPING
    }


def analyze_intra_cluster_divergence(cluster_points: List[int], point_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze divergences and inconsistencies between points within the same cluster."""
    divergence_results = {
        "intra_cluster_divergences": {},
        "inconsistencies": [],
        "compliance_conflicts": 0
    }
    
    # Compare each pair of points within the cluster
    for point1, point2 in combinations(cluster_points, 2):
        if point1 not in point_data or point2 not in point_data:
            continue
        
        pair_key = f"P{point1}_vs_P{point2}"
        
        # Get verdicts and scores for comparison
        point1_data = point_data[point1]
        point2_data = point_data[point2]
        
        # Collect verdicts and scores
        verdicts1 = [q_data["answer"].get("verdict", "").upper() for q_data in point1_data.values()]
        verdicts2 = [q_data["answer"].get("verdict", "").upper() for q_data in point2_data.values()]
        
        scores1 = []
        scores2 = []
        
        for q_data in point1_data.values():
            try:
                scores1.append(float(q_data["score"]))
            except (ValueError, TypeError):
                scores1.append(0.5)
        
        for q_data in point2_data.values():
            try:
                scores2.append(float(q_data["score"]))
            except (ValueError, TypeError):
                scores2.append(0.5)
        
        # Calculate divergence metrics
        if scores1 and scores2:
            # Pad shorter list with average to enable comparison
            max_len = max(len(scores1), len(scores2))
            avg1 = sum(scores1) / len(scores1)
            avg2 = sum(scores2) / len(scores2)
            
            scores1_padded = scores1 + [avg1] * (max_len - len(scores1))
            scores2_padded = scores2 + [avg2] * (max_len - len(scores2))
            
            js_div = jensen_shannon_divergence(scores1_padded, scores2_padded)
            cos_sim = cosine_similarity_score(scores1_padded, scores2_padded)
            
            divergence_results["intra_cluster_divergences"][pair_key] = {
                "jensen_shannon_divergence": js_div,
                "cosine_similarity": cos_sim,
                "score_difference": abs(avg1 - avg2)
            }
            
            # Check for compliance conflicts (contradictory verdicts)
            positive_verdicts1 = sum(1 for v in verdicts1 if v in ["YES", "TRUE", "POSITIVE"])
            positive_verdicts2 = sum(1 for v in verdicts2 if v in ["YES", "TRUE", "POSITIVE"])
            
            # If one point is mostly positive and other mostly negative, it's a conflict
            rate1 = positive_verdicts1 / len(verdicts1) if verdicts1 else 0.5
            rate2 = positive_verdicts2 / len(verdicts2) if verdicts2 else 0.5
            
            if abs(rate1 - rate2) > 0.6:  # Significant disagreement
                divergence_results["compliance_conflicts"] += 1
                divergence_results["inconsistencies"].append({
                    "points": [point1, point2],
                    "conflict_type": "compliance_disagreement",
                    "point1_positive_rate": rate1,
                    "point2_positive_rate": rate2,
                    "severity": abs(rate1 - rate2)
                })
        
        # Check for evidence quality inconsistencies
        evidence_count1 = sum(len(q_data["evidence_ids"]) for q_data in point1_data.values())
        evidence_count2 = sum(len(q_data["evidence_ids"]) for q_data in point2_data.values())
        
        if evidence_count1 > 0 and evidence_count2 > 0:
            evidence_ratio = max(evidence_count1, evidence_count2) / min(evidence_count1, evidence_count2)
            if evidence_ratio > 3.0:  # Significant imbalance in evidence
                divergence_results["inconsistencies"].append({
                    "points": [point1, point2],
                    "conflict_type": "evidence_imbalance",
                    "evidence_counts": [evidence_count1, evidence_count2],
                    "imbalance_ratio": evidence_ratio
                })
    
    return divergence_results


def analyze_cross_point_evidence_linkage(cluster_points: List[int], point_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze cross-point evidence linkage within a cluster."""
    linkage_results = {
        "shared_evidence_pairs": {},
        "evidence_reuse_count": 0,
        "cross_point_references": {}
    }
    
    # Collect all evidence items with their source points
    evidence_by_point = {}
    
    for point_num in cluster_points:
        if point_num not in point_data:
            continue
        
        evidence_by_point[point_num] = []
        
        for question_id, q_data in point_data[point_num].items():
            evidence_items = q_data.get("evidence_items", [])
            for evidence in evidence_items:
                evidence_by_point[point_num].append({
                    "evidence": evidence,
                    "question_id": question_id,
                    "source_point": point_num
                })
    
    # Find shared evidence between points
    shared_evidence_count = 0
    
    for point1, point2 in combinations(cluster_points, 2):
        if point1 not in evidence_by_point or point2 not in evidence_by_point:
            continue
        
        pair_key = f"P{point1}_P{point2}"
        shared_items = []
        
        for ev1_data in evidence_by_point[point1]:
            for ev2_data in evidence_by_point[point2]:
                if match_evidence_by_page_and_similarity(ev1_data["evidence"], ev2_data["evidence"]):
                    shared_items.append({
                        "evidence_1": {
                            "question_id": ev1_data["question_id"],
                            "text": ev1_data["evidence"].get("text", "")[:100] + "...",
                            "page_num": ev1_data["evidence"].get("metadata", {}).get("page_num") or 
                                       ev1_data["evidence"].get("page_num")
                        },
                        "evidence_2": {
                            "question_id": ev2_data["question_id"], 
                            "text": ev2_data["evidence"].get("text", "")[:100] + "...",
                            "page_num": ev2_data["evidence"].get("metadata", {}).get("page_num") or
                                       ev2_data["evidence"].get("page_num")
                        },
                        "similarity_score": calculate_text_similarity(
                            ev1_data["evidence"].get("text", ""),
                            ev2_data["evidence"].get("text", "")
                        )
                    })
                    shared_evidence_count += 1
        
        if shared_items:
            linkage_results["shared_evidence_pairs"][pair_key] = shared_items
    
    linkage_results["evidence_reuse_count"] = shared_evidence_count
    
    # Create cross-reference mappings
    for point_num in cluster_points:
        if point_num in evidence_by_point:
            linkage_results["cross_point_references"][f"P{point_num}"] = []
            
            # Find which other points share evidence with this point
            for other_point in cluster_points:
                if other_point != point_num:
                    pair_key1 = f"P{point_num}_P{other_point}"
                    pair_key2 = f"P{other_point}_P{point_num}"
                    
                    shared_count = 0
                    if pair_key1 in linkage_results["shared_evidence_pairs"]:
                        shared_count += len(linkage_results["shared_evidence_pairs"][pair_key1])
                    if pair_key2 in linkage_results["shared_evidence_pairs"]:
                        shared_count += len(linkage_results["shared_evidence_pairs"][pair_key2])
                    
                    if shared_count > 0:
                        linkage_results["cross_point_references"][f"P{point_num}"].append({
                            "linked_point": other_point,
                            "shared_evidence_count": shared_count
                        })
    
    return linkage_results


def write_meso_artifacts(doc_stem: str, meso_summary: Dict[str, Any], coverage_matrix: Dict[str, Any]) -> None:
    """
    Write meso-level aggregation artifacts to JSON file.
    
    Args:
        doc_stem: Document identifier stem for filename
        meso_summary: Meso-level summary data
        coverage_matrix: Coverage matrix data
    """
    try:
        # Create canonical_flow/aggregation directory if it doesn't exist
        output_dir = "canonical_flow/aggregation"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare the artifact data with required schema fields including Decálogo cluster analysis
        artifact_data = {
            "doc_stem": doc_stem,
            "coverage_metrics": {
                "coverage_matrix": coverage_matrix,
                "component_coverage_summary": meso_summary.get("component_coverage_summary", {})
            },
            "divergence_scores": {
                "divergence_stats": meso_summary.get("divergence_stats", {}),
                "question_divergences": {
                    question_id: item.get("divergence_metrics", {})
                    for question_id, item in meso_summary.get("items", {}).items()
                }
            },
            "cluster_participation": {
                "participation_counts": meso_summary.get("cluster_participation", {}),
                "by_question": {
                    question_id: {
                        "cluster_count": item.get("cluster_count", 0),
                        "participating_clusters": list(item.get("by_cluster", {}).keys())
                    }
                    for question_id, item in meso_summary.get("items", {}).items()
                }
            },
            "dimension_groupings": {
                "by_component": {
                    component: data["questions_addressing"] 
                    for component, data in coverage_matrix.items()
                    if data.get("questions_addressing")
                },
                "by_question": {
                    question_id: item.get("components_addressed", [])
                    for question_id, item in meso_summary.get("items", {}).items()
                }
            },
            "decalogo_cluster_analysis": meso_summary.get("decalogo_cluster_analysis", {}),
            "meso_summary": meso_summary
        }
        
        # Generate filename and write JSON
        filename = f"{doc_stem}_meso.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(artifact_data, f, sort_keys=True, indent=2, ensure_ascii=False)
            
        logger.info(f"Meso aggregation artifact written to {filepath}")
        
    except (OSError, IOError) as e:
        logger.error(f"Failed to write meso aggregation artifact for {doc_stem}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing meso artifact for {doc_stem}: {e}")


def process_macro_alignment(doc_stem: str) -> Dict[str, Any]:
    """
    Process macro alignment calculation using the macro_alignment_calculator.
    
    Args:
        doc_stem: Document identifier stem
        
    Returns:
        Macro alignment results
    """
    try:
# # #         from macro_alignment_calculator import process as macro_process  # Module not found  # Module not found  # Module not found
        result = macro_process(doc_stem)
        logger.info(f"Macro alignment calculation completed for {doc_stem}")
        return result
    except ImportError as e:
        logger.error(f"Could not import macro_alignment_calculator: {e}")
        return {"macro_alignment": {"error": "Macro alignment calculator not available"}}
    except Exception as e:
        logger.error(f"Error processing macro alignment for {doc_stem}: {e}")
        return {"macro_alignment": {"error": str(e)}}


def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Main processing function for meso aggregation with canonicalization system.

    Args:
        data: Input data containing cluster_audit.micro structure
        context: Optional context dictionary

    Returns:
        Enhanced data with meso_summary and coverage_matrix using deterministic ordering
    """
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        out.update(data)

    # Extract micro audit data
    micro = {}
    if isinstance(data, dict):
        cluster_audit = data.get("cluster_audit") or {}
        if isinstance(cluster_audit, dict):
            micro = cluster_audit.get("micro") or {}
        if not isinstance(micro, dict):
            micro = {}

    # Aggregate data by question across clusters with deterministic processing
    by_question: Dict[str, Dict[str, Any]] = {}

    def normalize_question_key(item: Dict[str, Any]) -> str:
        """Create normalized question identifier."""
        return str(
            item.get("question_id") or item.get("question") or f"Q_{hash(str(item))}"
        )

    # Process each cluster's answers in sorted order for deterministic results
    for cluster_id in sorted(micro.keys()):
        payload = micro[cluster_id]
        if not isinstance(payload, dict):
            continue

        answers = payload.get("answers") or []
        if not isinstance(answers, list):
            continue

        # Sort answers for deterministic processing
        sorted_answers = sorted(answers, key=lambda x: normalize_question_key(x) if isinstance(x, dict) else str(x))
        
        for answer in sorted_answers:
            if not isinstance(answer, dict):
                continue

            question_key = normalize_question_key(answer)

            # Initialize question slot if needed
            if question_key not in by_question:
                by_question[question_key] = {
                    "by_cluster": {},
                    "evidence_ids": set(),
                    "evidence_items": [],
                    "scores": [],
                    "components": set(),
                }

            slot = by_question[question_key]
            slot["by_cluster"][cluster_id] = answer

            # Collect scores
            score = answer.get("score") or answer.get("confidence")
            try:
                if isinstance(score, (int, float)):
                    slot["scores"].append(float(score))
                elif isinstance(score, str):
                    # Try to convert string to float
                    float_score = float(score)
                    slot["scores"].append(float_score)
            except (ValueError, TypeError):
                # Skip invalid scores
                pass

            # Collect evidence IDs
            evidence_ids = answer.get("evidence_ids") or []
            if isinstance(evidence_ids, list):
                for eid in evidence_ids:
                    if isinstance(eid, str):
                        slot["evidence_ids"].add(eid)

            # Collect evidence items for canonical processing
            evidence_items = answer.get("evidence_items", [])
            if isinstance(evidence_items, list):
                slot["evidence_items"].extend(evidence_items)

            # Collect components
            answer_components = extract_components_from_answer(answer)
            slot["components"].update(answer_components)

    # Apply canonicalization to evidence items in each question
    for question_key in by_question:
        slot = by_question[question_key]
        
        # Group evidence by dimension and sort
        if slot["evidence_items"]:
            dimension_groups = group_evidence_by_dimension(slot["evidence_items"])
            slot["evidence_by_dimension"] = dimension_groups
            
            # Create flattened sorted evidence list
            sorted_evidence = []
            for dimension in sorted(dimension_groups.keys()):
                sorted_evidence.extend(dimension_groups[dimension])
            slot["evidence_items"] = sorted_evidence

    # Calculate divergence metrics for each question
    question_divergences = calculate_cluster_divergences(by_question)

    # Build coverage matrix with canonical ordering
    coverage_matrix = build_coverage_matrix(by_question)
    
    # Analyze Decálogo clusters
    decalogo_cluster_analysis = analyze_decalogo_clusters(by_question)

    # Build meso summary with deterministic ordering
    meso_summary = {
        "items": {},
        "divergence_stats": {},
        "cluster_participation": {cluster: 0 for cluster in sorted(REQUIRED_CLUSTERS)},
        "component_coverage_summary": {},
    }

    all_js_divergences = []
    all_cos_similarities = []
    all_score_ranges = []

    # Process each question in sorted order
    for question_key in sorted(by_question.keys()):
        slot = by_question[question_key]
        question_div = question_divergences.get(question_key, {})

        # Count cluster participation
        for cluster_id in slot["by_cluster"]:
            if cluster_id in meso_summary["cluster_participation"]:
                meso_summary["cluster_participation"][cluster_id] += 1

        # Collect global divergence stats
        all_js_divergences.append(question_div.get("jensen_shannon_max", 0.0))
        all_cos_similarities.append(question_div.get("cosine_similarity_min", 1.0))
        all_score_ranges.append(question_div.get("score_range", 0.0))

        # Build question summary with canonical evidence ordering
        item_summary = {
            "by_cluster": slot["by_cluster"],
            "divergence_metrics": question_div,
            "evidence_coverage": len(slot["evidence_ids"]),
            "evidence_by_dimension": slot.get("evidence_by_dimension", {}),
            "components_addressed": sorted(list(slot["components"])),
            "cluster_count": len(slot["by_cluster"]),
            "score_summary": {
                "count": len(slot["scores"]),
                "min": min(slot["scores"]) if slot["scores"] else 0.0,
                "max": max(slot["scores"]) if slot["scores"] else 0.0,
                "avg": (
                    sum(slot["scores"]) / len(slot["scores"]) if slot["scores"] else 0.0
                ),
            },
        }

        meso_summary["items"][question_key] = item_summary

    # Calculate global divergence statistics
    if all_js_divergences:
        meso_summary["divergence_stats"] = {
            "jensen_shannon": {
                "max": max(all_js_divergences),
                "min": min(all_js_divergences),
                "avg": sum(all_js_divergences) / len(all_js_divergences),
                "std": float(np.std(all_js_divergences)),
            },
            "cosine_similarity": {
                "max": max(all_cos_similarities),
                "min": min(all_cos_similarities),
                "avg": sum(all_cos_similarities) / len(all_cos_similarities),
                "std": float(np.std(all_cos_similarities)),
            },
            "score_range": {
                "max": max(all_score_ranges),
                "min": min(all_score_ranges),
                "avg": sum(all_score_ranges) / len(all_score_ranges),
                "std": float(np.std(all_score_ranges)),
            },
            "question_count": len(by_question),
        }

    # Component coverage summary with canonical ordering
    total_evaluations = sum(
        coverage_matrix[comp]["total_evaluations"] for comp in coverage_matrix
    )
    fully_covered = sum(
        1
        for comp in coverage_matrix
        if coverage_matrix[comp]["coverage_percentage"] == 100.0
    )
    partially_covered = sum(
        1
        for comp in coverage_matrix
        if 0 < coverage_matrix[comp]["coverage_percentage"] < 100.0
    )
    not_covered = sum(
        1
        for comp in coverage_matrix
        if coverage_matrix[comp]["coverage_percentage"] == 0.0
    )

    # Sort gaps and well-covered lists for deterministic output
    coverage_gaps = sorted([
        comp
        for comp in coverage_matrix
        if coverage_matrix[comp]["coverage_percentage"] < 100.0
    ])
    well_covered = sorted([
        comp
        for comp in coverage_matrix
        if coverage_matrix[comp]["coverage_percentage"] >= 75.0
    ])

    meso_summary["component_coverage_summary"] = {
        "total_components": len(DEVELOPMENT_PLAN_COMPONENTS),
        "fully_covered": fully_covered,
        "partially_covered": partially_covered,
        "not_covered": not_covered,
        "total_evaluations": total_evaluations,
        "coverage_gaps": coverage_gaps,
        "well_covered": well_covered,
    }

    # Add Decálogo cluster analysis to meso summary
    meso_summary["decalogo_cluster_analysis"] = decalogo_cluster_analysis
    
    # Add results to output with canonical JSON serialization method
    out["meso_summary"] = meso_summary
    out["coverage_matrix"] = coverage_matrix
    
    # Remove serialize_canonical function for JSON compatibility
    # Users can call canonical_json_dumps(out) directly if needed

    # Generate artifacts if we have a document identifier
    doc_stem = None
    if context and isinstance(context, dict):
        doc_stem = context.get("doc_stem") or context.get("document_id")
    
    if not doc_stem and isinstance(data, dict):
# # #         # Try to derive doc_stem from data structure  # Module not found  # Module not found  # Module not found
        doc_stem = (
            data.get("doc_stem") or 
            data.get("document_id") or
            data.get("doc_id") or
            data.get("id")
        )
    
    if not doc_stem:
# # #         # Create a fallback doc_stem from available data  # Module not found  # Module not found  # Module not found
        if by_question:
            # Use first question key as basis for doc_stem
            first_question = next(iter(by_question.keys()))
            doc_stem = f"doc_{hash(first_question) % 10000}"
        else:
            doc_stem = "unknown_doc"
    
    # Ensure doc_stem is a valid filename
    doc_stem = str(doc_stem).replace("/", "_").replace("\\", "_").strip()
    
    # Write meso aggregation artifacts
    write_meso_artifacts(doc_stem, meso_summary, coverage_matrix)
    
    # Process macro alignment calculation
    macro_result = process_macro_alignment(doc_stem)
    if "macro_alignment" in macro_result:
        out["macro_alignment"] = macro_result["macro_alignment"]

    return out
