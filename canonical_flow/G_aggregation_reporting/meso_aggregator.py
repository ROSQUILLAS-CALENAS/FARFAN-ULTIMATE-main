#!/usr/bin/env python3
"""
Meso Aggregator for Evidence Analysis
Stage: G_aggregation_reporting
Code: 53G

# # # Aggregates analysis outputs from canonical_flow/analysis/ into meso-level structures  # Module not found  # Module not found  # Module not found
with coverage, divergence, and participation metrics.
"""

import json
import os
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any, Optional, Tuple  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import logging
# # # from collections import defaultdict, Counter  # Module not found  # Module not found  # Module not found

# # # from json_canonicalizer import JSONCanonicalizer  # Module not found  # Module not found  # Module not found

try:
    import numpy as np
except ImportError:
    # Fallback for basic numpy operations
    class NumpyFallback:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        
        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        def max(values):
            return max(values) if values else 0.0
    
    np = NumpyFallback()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process(document_stem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process analysis outputs for a document and generate meso-level aggregation with canonicalization.
    
    Args:
        document_stem: Document stem for file identification
        context: Optional processing context
        
    Returns:
        Dictionary containing meso-level aggregation with canonicalization metadata
    """
    # Initialize canonicalizer
    canonicalizer = JSONCanonicalizer(audit_enabled=True, validation_enabled=True)
    
    # Canonicalize inputs
    input_data = {"document_stem": document_stem}
    canonical_input_json, input_id, input_audit = canonicalizer.canonicalize(
        input_data, {"operation": "process", "stage": "aggregation_reporting", "component": "meso_aggregator"}
    )
    canonical_context_json, context_id, context_audit = canonicalizer.canonicalize(
        context, {"operation": "process", "stage": "aggregation_reporting", "component": "meso_aggregator"}
    )

    def _process_document_stem(stem: str) -> Dict[str, Any]:
    
    Args:
        document_stem: Document identifier stem (e.g., "ACANDI-CHOCO")
        
    Returns:
        Dict with 'success' boolean and 'artifacts' list
    """
    try:
        logger.info(f"Starting meso aggregation for document: {document_stem}")
        
        # Read analysis outputs
        analysis_data = _read_analysis_outputs(document_stem)
        if not analysis_data:
            logger.warning(f"No analysis data found for document: {document_stem}")
            return {
                "success": False,
                "error": "No analysis data found",
                "artifacts": []
            }
        
        # Aggregate evidence by dimensions
        aggregated_evidence = _aggregate_evidence_by_dimensions(analysis_data)
        
        # Calculate coverage metrics
        coverage_metrics = _calculate_coverage_metrics(aggregated_evidence, analysis_data)
        
        # Calculate divergence scores between evidence clusters
        divergence_scores = _calculate_divergence_scores(aggregated_evidence)
        
        # Calculate participation metrics
        participation_metrics = _calculate_participation_metrics(aggregated_evidence, analysis_data)
        
        # Build meso-level structure
        meso_structure = {
            "metadata": {
                "document_stem": document_stem,
                "generated_timestamp": datetime.utcnow().isoformat() + "Z",
                "processing_stage": "G_aggregation_reporting",
                "stage_code": "53G",
                "total_analysis_sources": len(analysis_data),
                "aggregation_version": "1.0"
            },
            "dimensions_aggregation": aggregated_evidence,
            "coverage_metrics": coverage_metrics,
            "divergence_analysis": divergence_scores,
            "participation_metrics": participation_metrics,
            "summary": {
                "total_dimensions": len(aggregated_evidence),
                "total_evidence_items": sum(
                    len(dim_data.get("evidence_items", []))
                    for dim_data in aggregated_evidence.values()
                ),
                "average_coverage_percentage": np.mean(list(coverage_metrics.values())) if coverage_metrics else 0.0,
                "max_divergence_score": max(divergence_scores.values()) if divergence_scores else 0.0
            }
        }
        
        # Write to output file
        output_path = Path("canonical_flow/aggregation") / f"{document_stem}_meso.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(meso_structure, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        logger.info(f"Successfully generated meso aggregation: {output_path}")
        
        return {
            "success": True,
            "artifacts": [str(output_path)],
            "summary": meso_structure["summary"]
        }
        
    except Exception as e:
        logger.error(f"Error processing meso aggregation for {document_stem}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "artifacts": []
        }


def _read_analysis_outputs(document_stem: str) -> List[Dict[str, Any]]:
    """Read all analysis outputs related to the document stem."""
    # Try different possible paths for analysis directory
    possible_paths = [
        Path("canonical_flow/analysis"),
        Path("../analysis"),
        Path("../../canonical_flow/analysis")
    ]
    
    analysis_dir = None
    for path in possible_paths:
        if path.exists():
            analysis_dir = path
            break
    
    analysis_data = []
    
    if not analysis_dir:
        logger.warning("Analysis directory not found in any expected location")
        return analysis_data
    
    # Look for files containing the document stem or relevant analysis outputs
    for file_path in analysis_dir.glob("**/*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if this file relates to our document stem
            if _is_relevant_analysis_file(data, document_stem, file_path.name):
                analysis_data.append({
                    "source_file": str(file_path),
                    "data": data
                })
                logger.info(f"Loaded analysis file: {file_path}")
                
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read analysis file {file_path}: {str(e)}")
            continue
    
    return analysis_data


def _is_relevant_analysis_file(data: Dict[str, Any], document_stem: str, filename: str) -> bool:
    """Check if an analysis file is relevant to the document stem."""
    # Check if document stem appears in the filename
    if document_stem.lower() in filename.lower():
        return True
    
    # Check if document stem appears in the data content
    data_str = json.dumps(data, default=str).lower()
    if document_stem.lower() in data_str:
        return True
    
    # Accept validation and audit files as they contain general evidence
    if any(keyword in filename.lower() for keyword in ['validation', 'audit', 'evidence']):
        return True
    
    return False


def _aggregate_evidence_by_dimensions(analysis_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate evidence by identified dimensions."""
    dimensions = defaultdict(lambda: {
        "evidence_items": [],
        "evidence_types": [],
        "validation_scores": [],
        "dnp_scores": [],
        "severity_levels": [],
        "sources": []
    })
    
    for analysis_source in analysis_data:
        data = analysis_source["data"]
        source_file = analysis_source["source_file"]
        
# # #         # Extract evidence from different analysis file types  # Module not found  # Module not found  # Module not found
        if "results" in data and isinstance(data["results"], list):
            # Evidence validation format
            for result in data["results"]:
                dimension = _extract_dimension_from_evidence(result, data)
                dimensions[dimension]["evidence_items"].append(result)
                dimensions[dimension]["evidence_types"].append(result.get("evidence_id", "unknown"))
                dimensions[dimension]["validation_scores"].append(result.get("validation_score", 0.0))
                dimensions[dimension]["dnp_scores"].append(result.get("dnp_compliance_score", 0.0))
                dimensions[dimension]["severity_levels"].append(result.get("severity_level", "unknown"))
                dimensions[dimension]["sources"].append(source_file)
        
        elif "components" in data:
# # #             # Audit format - extract from component outputs  # Module not found  # Module not found  # Module not found
            for comp_id, comp_data in data["components"].items():
                dimension = f"component_{comp_id}"
                for event in comp_data.get("events", []):
                    if event.get("event_type") == "component_end":
                        output_schema = event.get("output_schema", {})
                        sample_data = output_schema.get("sample_data", {})
                        
                        evidence_item = {
                            "evidence_id": f"{comp_id}_{event.get('event_id', 'unknown')}",
                            "component_name": comp_data.get("component_name", "unknown"),
                            "output_data": sample_data,
                            "performance_metrics": event.get("performance_metrics", {}),
                            "success": event.get("context_data", {}).get("success", False)
                        }
                        
                        dimensions[dimension]["evidence_items"].append(evidence_item)
                        dimensions[dimension]["evidence_types"].append("component_output")
                        dimensions[dimension]["validation_scores"].append(1.0 if evidence_item["success"] else 0.0)
                        dimensions[dimension]["dnp_scores"].append(_calculate_dnp_score_from_metrics(
                            evidence_item.get("performance_metrics", {})
                        ))
                        dimensions[dimension]["severity_levels"].append("info" if evidence_item["success"] else "warning")
                        dimensions[dimension]["sources"].append(source_file)
    
    # Convert defaultdict to regular dict and add metadata
    result = {}
    for dim_name, dim_data in dimensions.items():
        result[dim_name] = {
            "dimension_name": dim_name,
            "total_evidence": len(dim_data["evidence_items"]),
            "evidence_items": dim_data["evidence_items"],
            "evidence_types": list(set(dim_data["evidence_types"])),
            "validation_scores": dim_data["validation_scores"],
            "dnp_scores": dim_data["dnp_scores"],
            "severity_distribution": dict(Counter(dim_data["severity_levels"])),
            "unique_sources": list(set(dim_data["sources"]))
        }
    
    return result


def _extract_dimension_from_evidence(result: Dict[str, Any], data: Dict[str, Any]) -> str:
# # #     """Extract dimension from evidence result."""  # Module not found  # Module not found  # Module not found
# # #     # Try to infer dimension from evidence content  # Module not found  # Module not found  # Module not found
    evidence_id = result.get("evidence_id", "")
    
    # Check for common dimension keywords
    dimension_keywords = {
        "transparency": ["transparent", "disclosure", "public", "open"],
        "governance": ["govern", "administration", "management", "leadership"],
        "compliance": ["compliance", "regulation", "standard", "rule"],
        "performance": ["performance", "efficiency", "effectiveness", "metric"],
        "accountability": ["account", "responsible", "oversight", "audit"]
    }
    
    evidence_text = json.dumps(result, default=str).lower()
    
    for dimension, keywords in dimension_keywords.items():
        if any(keyword in evidence_text for keyword in keywords):
            return dimension
    
    # Default dimension based on evidence type or ID
    if "validation" in evidence_id.lower():
        return "validation_compliance"
    elif "component" in evidence_id.lower():
        return "system_performance"
    else:
        return "general_evidence"


def _calculate_coverage_metrics(aggregated_evidence: Dict[str, Dict[str, Any]], 
                               analysis_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate coverage percentage for each dimension."""
    coverage_metrics = {}
    
    total_sources = len(analysis_data)
    if total_sources == 0:
        return coverage_metrics
    
    for dimension, dim_data in aggregated_evidence.items():
        unique_sources = len(dim_data.get("unique_sources", []))
        coverage_percentage = (unique_sources / total_sources) * 100.0
        coverage_metrics[dimension] = round(coverage_percentage, 2)
    
    return coverage_metrics


def _calculate_divergence_scores(aggregated_evidence: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Calculate divergence scores between evidence clusters."""
    divergence_scores = {}
    dimensions = list(aggregated_evidence.keys())
    
    for i, dim1 in enumerate(dimensions):
        for j, dim2 in enumerate(dimensions[i+1:], i+1):
            # Calculate divergence between two dimensions
            scores1 = aggregated_evidence[dim1].get("validation_scores", [])
            scores2 = aggregated_evidence[dim2].get("validation_scores", [])
            
            if scores1 and scores2:
                # Use coefficient of variation as divergence measure
                mean1, std1 = np.mean(scores1), np.std(scores1)
                mean2, std2 = np.mean(scores2), np.std(scores2)
                
                # Calculate divergence as difference in coefficient of variation
                cv1 = std1 / mean1 if mean1 != 0 else 0
                cv2 = std2 / mean2 if mean2 != 0 else 0
                divergence = abs(cv1 - cv2)
                
                divergence_scores[f"{dim1}_vs_{dim2}"] = round(divergence, 4)
    
    return divergence_scores


def _calculate_participation_metrics(aggregated_evidence: Dict[str, Dict[str, Any]], 
                                   analysis_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calculate participation metrics showing how evidence contributes to thematic groups."""
    participation_metrics = {}
    
    # Define thematic groups based on evidence characteristics
    thematic_groups = {
        "high_quality": {"validation_threshold": 0.7, "dnp_threshold": 0.8},
        "moderate_quality": {"validation_threshold": 0.4, "dnp_threshold": 0.5},
        "improvement_needed": {"validation_threshold": 0.0, "dnp_threshold": 0.0}
    }
    
    for dimension, dim_data in aggregated_evidence.items():
        dimension_metrics = {}
        validation_scores = dim_data.get("validation_scores", [])
        dnp_scores = dim_data.get("dnp_scores", [])
        total_evidence = len(validation_scores)
        
        if total_evidence == 0:
            dimension_metrics = {group: {"count": 0, "percentage": 0.0} for group in thematic_groups}
        else:
            for group_name, thresholds in thematic_groups.items():
                # Count evidence meeting group criteria
                count = 0
                for val_score, dnp_score in zip(validation_scores, dnp_scores):
                    if (val_score >= thresholds["validation_threshold"] and 
                        dnp_score >= thresholds["dnp_threshold"]):
                        count += 1
                
                percentage = (count / total_evidence) * 100.0
                dimension_metrics[group_name] = {
                    "count": count,
                    "percentage": round(percentage, 2)
                }
        
        participation_metrics[dimension] = dimension_metrics
    
    return participation_metrics


def _calculate_dnp_score_from_metrics(performance_metrics: Dict[str, Any]) -> float:
# # #     """Calculate a DNP-style score from performance metrics."""  # Module not found  # Module not found  # Module not found
    if not performance_metrics:
        return 0.5  # Neutral score
    
    # Simple heuristic: good performance = high DNP score
    duration = performance_metrics.get("duration_seconds", 1.0)
    cpu_percent = performance_metrics.get("cpu_percent", 50.0)
    
    # Normalize metrics (lower duration and CPU = higher score)
    duration_score = max(0, 1.0 - min(duration / 10.0, 1.0))  # Cap at 10 seconds
    cpu_score = max(0, 1.0 - min(cpu_percent / 100.0, 1.0))  # Normalize CPU
    
    return (duration_score + cpu_score) / 2.0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python meso_aggregator.py <document_stem>")
        sys.exit(1)
    
    document_stem = sys.argv[1]
    result = process(document_stem)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))