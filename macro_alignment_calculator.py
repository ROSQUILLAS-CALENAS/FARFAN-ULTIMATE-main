"""
Macro Alignment Calculator
# # # - Consumes meso-level cluster aggregations from canonical_flow/aggregation/<doc_stem>_meso.json  # Module not found  # Module not found  # Module not found
- Applies predefined Decálogo weights to compute overall compliance score
- Classifies results into CUMPLE (≥0.75), CUMPLE_PARCIAL (0.5-0.74), or NO_CUMPLE (<0.5) categories
- Generates canonical_flow/aggregation/<doc_stem>_macro.json artifact with weighted score calculation breakdown

Input expectations:
- meso_summary with divergence_scores, coverage_metrics, cluster_participation
- Decálogo weights configuration

Output:
- macro_alignment with weighted compliance score, classification, and supporting metrics
"""

# # # from __future__ import annotations  # Module not found  # Module not found  # Module not found

import json
import logging
import os
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional  # Module not found  # Module not found  # Module not found


# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "107O"
__stage_order__ = 7

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance classification levels"""
    CUMPLE = "CUMPLE"
    CUMPLE_PARCIAL = "CUMPLE_PARCIAL"  
    NO_CUMPLE = "NO_CUMPLE"


# Decálogo weights based on the 10 points across 4 clusters
# Points are weighted by their importance and cluster distribution
DECALOGO_WEIGHTS = {
    # Cluster 1: PAZ, SEGURIDAD Y PROTECCIÓN DE DEFENSORES (3 points: 1, 5, 8)
    "point_1_violence_prevention": 0.12,  # Prevention of violence and protection
    "point_5_victims_rights": 0.10,       # Rights of victims and peace building
    "point_8_human_rights_defenders": 0.08,  # Leaders and defenders of human rights
    
    # Cluster 2: DERECHOS SOCIALES FUNDAMENTALES (3 points: 2, 3, 4)
    "point_2_health": 0.11,               # Right to health
    "point_3_education": 0.11,            # Right to education
    "point_4_food": 0.09,                 # Right to food
    
    # Cluster 3: IGUALDAD Y NO DISCRIMINACIÓN (2 points: 6, 7)
    "point_6_women_rights": 0.10,         # Women's rights
    "point_7_children_rights": 0.10,      # Rights of children and adolescents
    
    # Cluster 4: DERECHOS TERRITORIALES Y AMBIENTALES (2 points: 9, 10)
    "point_9_ethnic_peoples": 0.09,       # Rights of ethnic peoples
    "point_10_environment": 0.10,         # Right to healthy environment
}

# Cluster weights for aggregation
CLUSTER_WEIGHTS = {
    "C1": 0.30,  # PAZ, SEGURIDAD Y PROTECCIÓN (highest weight - peace and security)
    "C2": 0.31,  # DERECHOS SOCIALES (highest weight - fundamental social rights)
    "C3": 0.20,  # IGUALDAD Y NO DISCRIMINACIÓN 
    "C4": 0.19,  # DERECHOS TERRITORIALES Y AMBIENTALES
}

# Component weights for Development Plan elements
COMPONENT_WEIGHTS = {
    "OBJECTIVES": 0.15,
    "STRATEGIES": 0.14,
    "INDICATORS": 0.12,
    "TIMELINES": 0.08,
    "BUDGET": 0.10,
    "STAKEHOLDERS": 0.09,
    "RISKS": 0.07,
    "COMPLIANCE": 0.15,
    "SUSTAINABILITY": 0.05,
    "IMPACT": 0.05,
}


def classify_compliance(score: float) -> ComplianceLevel:
    """
    Classify compliance score into CUMPLE/CUMPLE_PARCIAL/NO_CUMPLE categories.
    
    Args:
        score: Compliance score between 0.0 and 1.0
        
    Returns:
        ComplianceLevel enum value
    """
    if score >= 0.75:
        return ComplianceLevel.CUMPLE
    elif score >= 0.50:
        return ComplianceLevel.CUMPLE_PARCIAL
    else:
        return ComplianceLevel.NO_CUMPLE


def calculate_divergence_penalty(divergence_scores: Dict[str, Any]) -> float:
    """
    Calculate penalty factor based on cluster divergences.
    High divergence indicates inconsistent evaluation across clusters.
    
    Args:
# # #         divergence_scores: Divergence metrics from meso aggregation  # Module not found  # Module not found  # Module not found
        
    Returns:
        Penalty factor between 0.0 and 1.0 (1.0 = no penalty)
    """
    try:
        question_divergences = divergence_scores.get("question_divergences", {})
        if not question_divergences:
            return 1.0
            
        total_js_divergence = 0.0
        total_cos_similarity = 0.0
        question_count = 0
        
        for question_id, metrics in question_divergences.items():
            if isinstance(metrics, dict):
                # Jensen-Shannon divergence (higher = more penalty)
                js_div = metrics.get("jensen_shannon_max", 0.0)
                total_js_divergence += float(js_div)
                
                # Cosine similarity (higher = less penalty)
                cos_sim = metrics.get("cosine_similarity_min", 1.0)
                total_cos_similarity += float(cos_sim)
                
                question_count += 1
        
        if question_count == 0:
            return 1.0
            
        # Calculate average divergence metrics
        avg_js_divergence = total_js_divergence / question_count
        avg_cos_similarity = total_cos_similarity / question_count
        
        # Convert divergence to penalty (higher divergence = lower score)
        js_penalty = max(0.0, 1.0 - (avg_js_divergence * 2.0))  # Scale JS divergence
        cos_penalty = avg_cos_similarity  # Higher similarity = better
        
        # Combined penalty (weighted average)
        penalty = (js_penalty * 0.6) + (cos_penalty * 0.4)
        return max(0.0, min(1.0, penalty))
        
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"Error calculating divergence penalty: {e}")
        return 0.8  # Conservative penalty for missing/invalid data


def calculate_coverage_score(coverage_metrics: Dict[str, Any]) -> float:
    """
    Calculate coverage score based on component coverage across clusters.
    
    Args:
# # #         coverage_metrics: Coverage matrix and summary from meso aggregation  # Module not found  # Module not found  # Module not found
        
    Returns:
        Coverage score between 0.0 and 1.0
    """
    try:
        coverage_matrix = coverage_metrics.get("coverage_matrix", {})
        if not coverage_matrix:
            return 0.0
            
        total_weighted_coverage = 0.0
        total_weights = 0.0
        
        for component, data in coverage_matrix.items():
            if isinstance(data, dict):
                # Get coverage percentage (0-100)
                coverage_percentage = data.get("coverage_percentage", 0.0)
                coverage_score = float(coverage_percentage) / 100.0
                
                # Apply component weight
                component_weight = COMPONENT_WEIGHTS.get(component.upper(), 0.05)
                total_weighted_coverage += coverage_score * component_weight
                total_weights += component_weight
        
        if total_weights == 0:
            return 0.0
            
        return total_weighted_coverage / total_weights
        
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"Error calculating coverage score: {e}")
        return 0.0


def calculate_cluster_participation_score(cluster_participation: Dict[str, Any]) -> float:
    """
    Calculate score based on cluster participation distribution.
    
    Args:
# # #         cluster_participation: Cluster participation data from meso aggregation  # Module not found  # Module not found  # Module not found
        
    Returns:
        Participation score between 0.0 and 1.0
    """
    try:
        participation_counts = cluster_participation.get("participation_counts", {})
        if not participation_counts:
            return 0.0
            
        total_weighted_participation = 0.0
        total_weights = 0.0
        
        for cluster_id, count in participation_counts.items():
            cluster_weight = CLUSTER_WEIGHTS.get(cluster_id, 0.25)
            participation_score = min(1.0, float(count) / 10.0)  # Normalize to [0,1]
            
            total_weighted_participation += participation_score * cluster_weight
            total_weights += cluster_weight
            
        if total_weights == 0:
            return 0.0
            
        return total_weighted_participation / total_weights
        
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"Error calculating participation score: {e}")
        return 0.0


def load_meso_aggregation(doc_stem: str) -> Optional[Dict[str, Any]]:
    """
# # #     Load meso-level aggregation data from canonical_flow/aggregation/<doc_stem>_meso.json  # Module not found  # Module not found  # Module not found
    
    Args:
        doc_stem: Document identifier stem
        
    Returns:
        Meso aggregation data dictionary or None if not found
    """
    try:
        meso_filepath = f"canonical_flow/aggregation/{doc_stem}_meso.json"
        if not os.path.exists(meso_filepath):
            logger.error(f"Meso aggregation file not found: {meso_filepath}")
            return None
            
        with open(meso_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
# # #         logger.info(f"Loaded meso aggregation data from {meso_filepath}")  # Module not found  # Module not found  # Module not found
        return data
        
    except (OSError, IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load meso aggregation for {doc_stem}: {e}")
        return None


def calculate_macro_alignment_score(meso_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate macro alignment score using Decálogo weights and meso aggregation data.
    
    Args:
        meso_data: Meso-level aggregation data
        
    Returns:
        Dictionary with weighted score calculation breakdown
    """
    try:
# # #         # Extract key components from meso data  # Module not found  # Module not found  # Module not found
        coverage_metrics = meso_data.get("coverage_metrics", {})
        divergence_scores = meso_data.get("divergence_scores", {})
        cluster_participation = meso_data.get("cluster_participation", {})
        
        # Calculate component scores
        coverage_score = calculate_coverage_score(coverage_metrics)
        participation_score = calculate_cluster_participation_score(cluster_participation)
        divergence_penalty = calculate_divergence_penalty(divergence_scores)
        
        # Calculate base compliance score
        base_score = (coverage_score * 0.50) + (participation_score * 0.30)
        
        # Apply divergence penalty
        final_score = base_score * divergence_penalty
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))
        
        # Classify compliance level
        compliance_level = classify_compliance(final_score)
        
        return {
            "final_score": final_score,
            "compliance_level": compliance_level.value,
            "score_breakdown": {
                "coverage_score": coverage_score,
                "participation_score": participation_score,
                "base_score": base_score,
                "divergence_penalty": divergence_penalty,
                "final_score": final_score
            },
            "weights_applied": {
                "coverage_weight": 0.50,
                "participation_weight": 0.30,
                "divergence_penalty_factor": divergence_penalty
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating macro alignment score: {e}")
        return {
            "final_score": 0.0,
            "compliance_level": ComplianceLevel.NO_CUMPLE.value,
            "score_breakdown": {},
            "weights_applied": {},
            "error": str(e)
        }


def generate_supporting_metrics(meso_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate supporting metrics for the macro alignment calculation.
    
    Args:
        meso_data: Meso-level aggregation data
        
    Returns:
        Dictionary with supporting metrics
    """
    try:
        coverage_metrics = meso_data.get("coverage_metrics", {})
        divergence_scores = meso_data.get("divergence_scores", {})
        cluster_participation = meso_data.get("cluster_participation", {})
        dimension_groupings = meso_data.get("dimension_groupings", {})
        
        # Coverage analysis
        coverage_matrix = coverage_metrics.get("coverage_matrix", {})
        covered_components = [
            comp for comp, data in coverage_matrix.items() 
            if isinstance(data, dict) and data.get("coverage_percentage", 0) > 0
        ]
        
        # Divergence analysis
        question_divergences = divergence_scores.get("question_divergences", {})
        high_divergence_questions = [
            qid for qid, metrics in question_divergences.items()
            if isinstance(metrics, dict) and metrics.get("jensen_shannon_max", 0) > 0.3
        ]
        
        # Participation analysis
        participation_counts = cluster_participation.get("participation_counts", {})
        total_evaluations = sum(participation_counts.values()) if participation_counts else 0
        
        # Safe calculation of cluster balance score
        cluster_balance_score = 1.0
        if participation_counts:
            participation_values = list(participation_counts.values())
            if participation_values:
                min_participation = min(participation_values)
                max_participation = max(participation_values)
                cluster_balance_score = min_participation / max(max_participation, 1)
        
        return {
            "coverage_analysis": {
                "total_components_evaluated": len(covered_components),
                "components_with_coverage": covered_components,
                "coverage_distribution": {
                    comp: data.get("coverage_percentage", 0)
                    for comp, data in coverage_matrix.items()
                    if isinstance(data, dict)
                }
            },
            "divergence_analysis": {
                "total_questions_analyzed": len(question_divergences),
                "high_divergence_questions": high_divergence_questions,
                "average_divergence": sum(
                    metrics.get("jensen_shannon_avg", 0)
                    for metrics in question_divergences.values()
                    if isinstance(metrics, dict)
                ) / max(1, len(question_divergences))
            },
            "participation_analysis": {
                "total_evaluations": total_evaluations,
                "cluster_distribution": participation_counts,
                "cluster_balance_score": cluster_balance_score
            },
            "dimension_analysis": {
                "components_mapped": len(dimension_groupings.get("by_component", {})),
                "questions_mapped": len(dimension_groupings.get("by_question", {}))
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating supporting metrics: {e}")
        return {
            "coverage_analysis": {},
            "divergence_analysis": {},
            "participation_analysis": {},
            "dimension_analysis": {},
            "error": str(e)
        }


def write_macro_artifacts(doc_stem: str, macro_alignment: Dict[str, Any]) -> None:
    """
    Write macro alignment artifacts to canonical_flow/aggregation/<doc_stem>_macro.json
    
    Args:
        doc_stem: Document identifier stem
        macro_alignment: Macro alignment calculation results
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = "canonical_flow/aggregation"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        filename = f"{doc_stem}_macro.json"
        filepath = os.path.join(output_dir, filename)
        
        # Write JSON artifact
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(macro_alignment, f, sort_keys=True, indent=2, ensure_ascii=False)
            
        logger.info(f"Macro alignment artifact written to {filepath}")
        
    except (OSError, IOError) as e:
        logger.error(f"Failed to write macro alignment artifact for {doc_stem}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing macro artifact for {doc_stem}: {e}")


def process(data: Any, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Main processing function for macro alignment calculation.
    
    Args:
        data: Input data (can contain doc_stem or be the doc_stem directly)
        context: Optional context dictionary
        
    Returns:
        Enhanced data with macro_alignment results
    """
# # #     # Extract doc_stem from input  # Module not found  # Module not found  # Module not found
    doc_stem = None
    if isinstance(data, dict):
        doc_stem = data.get("doc_stem")
    elif isinstance(data, str):
        doc_stem = data
    
    if not doc_stem:
        logger.error("No doc_stem provided for macro alignment calculation")
        return {
            "macro_alignment": {
                "final_score": 0.0,
                "compliance_level": ComplianceLevel.NO_CUMPLE.value,
                "error": "No doc_stem provided"
            }
        }
    
    # Load meso aggregation data
    meso_data = load_meso_aggregation(doc_stem)
    if not meso_data:
        logger.error(f"Could not load meso aggregation data for {doc_stem}")
        return {
            "macro_alignment": {
                "final_score": 0.0,
                "compliance_level": ComplianceLevel.NO_CUMPLE.value,
                "error": "Could not load meso aggregation data"
            }
        }
    
    # Calculate macro alignment score
    alignment_calculation = calculate_macro_alignment_score(meso_data)
    
    # Generate supporting metrics
    supporting_metrics = generate_supporting_metrics(meso_data)
    
    # Combine results
    macro_alignment = {
        "doc_stem": doc_stem,
        "timestamp": str(context.get("timestamp") if context else ""),
        "calculation_results": alignment_calculation,
        "supporting_metrics": supporting_metrics,
        "decalogo_weights": {
            "point_weights": DECALOGO_WEIGHTS,
            "cluster_weights": CLUSTER_WEIGHTS,
            "component_weights": COMPONENT_WEIGHTS
        },
        "classification_thresholds": {
            "CUMPLE": 0.75,
            "CUMPLE_PARCIAL": 0.50,
            "NO_CUMPLE": 0.0
        }
    }
    
    # Write macro artifacts
    write_macro_artifacts(doc_stem, macro_alignment)
    
    # Return enhanced data
    result_data = {}
    if isinstance(data, dict):
        result_data.update(data)
    
    result_data["macro_alignment"] = macro_alignment
    
    return result_data


if __name__ == "__main__":
    # Example usage for testing
    test_doc_stem = "test_document"
    result = process(test_doc_stem)
    print(f"Macro alignment calculation completed for {test_doc_stem}")
    print(f"Final score: {result.get('macro_alignment', {}).get('calculation_results', {}).get('final_score', 0.0)}")
    print(f"Compliance level: {result.get('macro_alignment', {}).get('calculation_results', {}).get('compliance_level', 'NO_CUMPLE')}")