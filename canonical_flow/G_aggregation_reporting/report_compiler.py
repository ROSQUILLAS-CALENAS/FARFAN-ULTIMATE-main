"""
Report Compiler Module

# # # Generates structured narrative reports from meso aggregation data containing:  # Module not found  # Module not found  # Module not found
- Executive summary paragraphs
- Detailed findings sections with evidence references
- Compliance assessment metrics
- Key highlights

Maintains standardized API contract interface with comprehensive error handling.
"""

import json
import logging
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Union  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import uuid

# # # from json_canonicalizer import JSONCanonicalizer  # Module not found  # Module not found  # Module not found

# Configure logging
logger = logging.getLogger(__name__)


class ReportCompilerError(Exception):
    """Custom exception for report compilation errors."""
    pass


class ReportStatus:
    """Status indicators for report generation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    
    WARNINGS = {
        "MISSING_MESO": "Meso aggregation file not found",
        "MALFORMED_DATA": "Malformed input data detected",
        "INSUFFICIENT_EVIDENCE": "Insufficient evidence for complete analysis",
        "PROCESSING_ERROR": "Error during report processing"
    }


def ensure_utf8_encoding(text: str) -> str:
    """Ensure text is properly UTF-8 encoded."""
    if isinstance(text, bytes):
        return text.decode('utf-8', errors='replace')
    return str(text).encode('utf-8', errors='replace').decode('utf-8')


def create_deterministic_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create dictionary with deterministic field ordering."""
    if not isinstance(data, dict):
        return {}
    
    # Sort keys to ensure deterministic ordering
    return {key: data[key] for key in sorted(data.keys())}


def safe_get_nested(data: Dict[str, Any], *keys, default=None) -> Any:
    """Safely get nested dictionary values."""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def generate_executive_summary(meso_data: Dict[str, Any]) -> Dict[str, Any]:
# # #     """Generate executive summary from meso aggregation data."""  # Module not found  # Module not found  # Module not found
    try:
        meso_summary = meso_data.get("meso_summary", {})
        coverage_matrix = meso_data.get("coverage_matrix", {})
        
        # Extract key metrics
        question_count = safe_get_nested(meso_summary, "divergence_stats", "question_count", default=0)
        cluster_participation = meso_summary.get("cluster_participation", {})
        total_clusters = len([c for c in cluster_participation.values() if c > 0])
        
        # Coverage metrics
        coverage_summary = meso_summary.get("component_coverage_summary", {})
        fully_covered = coverage_summary.get("fully_covered", 0)
        total_components = coverage_summary.get("total_components", 0)
        
        # Divergence metrics
        js_stats = safe_get_nested(meso_summary, "divergence_stats", "jensen_shannon", default={})
        avg_divergence = js_stats.get("avg", 0.0)
        
        # Generate summary paragraphs
        overview = ensure_utf8_encoding(
            f"Analysis encompasses {question_count} evaluation questions across "
            f"{total_clusters} operational clusters. Coverage assessment reveals "
            f"{fully_covered} of {total_components} development plan components "
            f"are fully addressed, indicating {'comprehensive' if fully_covered > total_components * 0.8 else 'moderate'} "
            f"implementation scope."
        )
        
        consensus_analysis = ensure_utf8_encoding(
            f"Cross-cluster divergence analysis shows an average Jensen-Shannon "
            f"divergence of {avg_divergence:.3f}, suggesting "
            f"{'high consensus' if avg_divergence < 0.1 else 'moderate alignment' if avg_divergence < 0.3 else 'significant variation'} "
            f"in evaluation perspectives across operational domains."
        )
        
        return create_deterministic_dict({
            "overview": overview,
            "consensus_analysis": consensus_analysis,
            "key_metrics": create_deterministic_dict({
                "questions_evaluated": question_count,
                "clusters_active": total_clusters,
                "components_covered": fully_covered,
                "coverage_percentage": round((fully_covered / max(total_components, 1)) * 100, 1),
                "consensus_score": round(1.0 - min(avg_divergence, 1.0), 3)
            })
        })
        
    except Exception as e:
        logger.warning(f"Error generating executive summary: {e}")
        return create_deterministic_dict({
            "overview": "Executive summary generation encountered processing limitations.",
            "consensus_analysis": "Consensus analysis unavailable due to data constraints.",
            "key_metrics": create_deterministic_dict({
                "questions_evaluated": 0,
                "clusters_active": 0,
                "components_covered": 0,
                "coverage_percentage": 0.0,
                "consensus_score": 0.0
            })
        })


def generate_findings_section(meso_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate detailed findings with evidence references."""
    try:
        meso_summary = meso_data.get("meso_summary", {})
        items = meso_summary.get("items", {})
        
        findings = []
        evidence_references = set()
        
        for question_id, question_data in items.items():
            # Extract cluster responses
            cluster_responses = question_data.get("by_cluster", {})
            components_addressed = question_data.get("components_addressed", [])
            divergence_metrics = question_data.get("divergence_metrics", {})
            
            # Generate finding narrative
            cluster_count = len(cluster_responses)
            component_list = ", ".join(components_addressed[:3])  # Limit for readability
            if len(components_addressed) > 3:
                component_list += f" (and {len(components_addressed) - 3} others)"
            
            js_max = divergence_metrics.get("jensen_shannon_max", 0.0)
            consensus_level = "high" if js_max < 0.1 else "moderate" if js_max < 0.3 else "low"
            
            finding_text = ensure_utf8_encoding(
                f"Evaluation question {question_id} addressed {component_list} "
                f"across {cluster_count} clusters with {consensus_level} consensus "
                f"(divergence: {js_max:.3f}). "
            )
            
            # Add evidence references
            evidence_count = question_data.get("evidence_coverage", 0)
            if evidence_count > 0:
                finding_text += f"Analysis supported by {evidence_count} evidence items. "
                evidence_references.add(f"Q{question_id}_evidence_{evidence_count}")
            
            findings.append(create_deterministic_dict({
                "question_id": question_id,
                "finding": finding_text,
                "components": components_addressed,
                "evidence_count": evidence_count,
                "consensus_metrics": create_deterministic_dict(divergence_metrics)
            }))
        
        return create_deterministic_dict({
            "detailed_findings": findings[:10],  # Limit for report length
            "evidence_references": sorted(list(evidence_references)),
            "total_findings": len(findings),
            "findings_summary": ensure_utf8_encoding(
                f"Comprehensive analysis of {len(findings)} evaluation areas "
                f"with {len(evidence_references)} evidence reference points."
            )
        })
        
    except Exception as e:
        logger.warning(f"Error generating findings section: {e}")
        return create_deterministic_dict({
            "detailed_findings": [],
            "evidence_references": [],
            "total_findings": 0,
            "findings_summary": "Detailed findings unavailable due to processing constraints."
        })


def generate_compliance_assessment(meso_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate compliance assessment metrics."""
    try:
        coverage_matrix = meso_data.get("coverage_matrix", {})
        coverage_summary = safe_get_nested(meso_data, "meso_summary", "component_coverage_summary", default={})
        
        # Component compliance metrics
        total_components = coverage_summary.get("total_components", 0)
        fully_covered = coverage_summary.get("fully_covered", 0)
        partially_covered = coverage_summary.get("partially_covered", 0)
        not_covered = coverage_summary.get("not_covered", 0)
        
        # Calculate compliance scores
        compliance_score = (fully_covered + (partially_covered * 0.5)) / max(total_components, 1)
        
        # Identify high-risk gaps
        coverage_gaps = coverage_summary.get("coverage_gaps", [])
        critical_gaps = [gap for gap in coverage_gaps 
                        if gap in ["COMPLIANCE", "RISKS", "SUSTAINABILITY"]]
        
        # Generate assessment narrative
        assessment_narrative = ensure_utf8_encoding(
            f"Compliance assessment reveals {compliance_score:.1%} coverage across "
            f"development plan components. {fully_covered} components show full "
            f"implementation, while {not_covered} components require attention. "
            f"{'Critical compliance gaps identified.' if critical_gaps else 'No critical gaps detected.'}"
        )
        
        return create_deterministic_dict({
            "overall_compliance_score": round(compliance_score, 3),
            "component_metrics": create_deterministic_dict({
                "total_components": total_components,
                "fully_compliant": fully_covered,
                "partially_compliant": partially_covered,
                "non_compliant": not_covered
            }),
            "risk_assessment": create_deterministic_dict({
                "critical_gaps": critical_gaps,
                "coverage_gaps": coverage_gaps[:5],  # Limit for readability
                "risk_level": "high" if critical_gaps else "moderate" if coverage_gaps else "low"
            }),
            "assessment_narrative": assessment_narrative,
            "recommendations": [
                ensure_utf8_encoding(f"Address gaps in {gap} component")
                for gap in critical_gaps[:3]
            ]
        })
        
    except Exception as e:
        logger.warning(f"Error generating compliance assessment: {e}")
        return create_deterministic_dict({
            "overall_compliance_score": 0.0,
            "component_metrics": create_deterministic_dict({
                "total_components": 0,
                "fully_compliant": 0,
                "partially_compliant": 0,
                "non_compliant": 0
            }),
            "risk_assessment": create_deterministic_dict({
                "critical_gaps": [],
                "coverage_gaps": [],
                "risk_level": "unknown"
            }),
            "assessment_narrative": "Compliance assessment unavailable due to processing constraints.",
            "recommendations": []
        })


def generate_key_highlights(meso_data: Dict[str, Any]) -> Dict[str, Any]:
# # #     """Generate key highlights from the analysis."""  # Module not found  # Module not found  # Module not found
    try:
        meso_summary = meso_data.get("meso_summary", {})
        coverage_matrix = meso_data.get("coverage_matrix", {})
        
        # Identify well-covered components
        well_covered = safe_get_nested(meso_summary, "component_coverage_summary", "well_covered", default=[])
        
        # Find highest consensus questions
        items = meso_summary.get("items", {})
        high_consensus_questions = []
        for q_id, q_data in items.items():
            js_max = safe_get_nested(q_data, "divergence_metrics", "jensen_shannon_max", default=1.0)
            if js_max < 0.1:  # High consensus threshold
                high_consensus_questions.append(q_id)
        
        # Cluster performance analysis
        cluster_participation = meso_summary.get("cluster_participation", {})
        most_active_cluster = max(cluster_participation.items(), key=lambda x: x[1], default=("None", 0))
        
        # Generate highlights
        highlights = []
        
        if well_covered:
            highlights.append(ensure_utf8_encoding(
                f"Strong implementation coverage achieved in {', '.join(well_covered[:3])} components."
            ))
        
        if high_consensus_questions:
            highlights.append(ensure_utf8_encoding(
                f"High evaluation consensus reached on {len(high_consensus_questions)} key questions."
            ))
        
        if most_active_cluster[1] > 0:
            highlights.append(ensure_utf8_encoding(
                f"Cluster {most_active_cluster[0]} demonstrates highest engagement with "
                f"{most_active_cluster[1]} evaluations completed."
            ))
        
        # Divergence insights
        js_stats = safe_get_nested(meso_summary, "divergence_stats", "jensen_shannon", default={})
        if js_stats.get("std", 0) < 0.1:
            highlights.append(ensure_utf8_encoding(
                "Consistent evaluation patterns observed across operational clusters."
            ))
        
        return create_deterministic_dict({
            "key_achievements": highlights[:3],  # Top 3 highlights
            "performance_indicators": create_deterministic_dict({
                "well_covered_components": len(well_covered),
                "consensus_questions": len(high_consensus_questions),
                "most_active_cluster": most_active_cluster[0],
                "evaluation_consistency": "high" if js_stats.get("std", 1.0) < 0.1 else "moderate"
            }),
            "strategic_insights": [
                ensure_utf8_encoding(
                    "Multi-cluster evaluation approach provides comprehensive coverage validation."
                ),
                ensure_utf8_encoding(
                    "Divergence analysis enables identification of evaluation perspective gaps."
                ),
                ensure_utf8_encoding(
                    "Component coverage matrix supports strategic planning alignment assessment."
                )
            ]
        })
        
    except Exception as e:
        logger.warning(f"Error generating key highlights: {e}")
        return create_deterministic_dict({
            "key_achievements": [],
            "performance_indicators": create_deterministic_dict({
                "well_covered_components": 0,
                "consensus_questions": 0,
                "most_active_cluster": "unknown",
                "evaluation_consistency": "unknown"
            }),
            "strategic_insights": [
                "Key highlights unavailable due to processing constraints."
            ]
        })


def process(document_stem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main processing function that generates structured narrative reports with canonicalization.
    
    Args:
        document_stem: Document stem for input/output file naming
        context: Optional context dictionary
        
    Returns:
        Dictionary containing status, warnings, and processing results with canonicalization metadata
    """
    # Initialize canonicalizer
    canonicalizer = JSONCanonicalizer(audit_enabled=True, validation_enabled=True)
    
    # Canonicalize inputs
    input_data = {"document_stem": document_stem}
    canonical_input_json, input_id, input_audit = canonicalizer.canonicalize(
        input_data, {"operation": "process", "stage": "aggregation_reporting", "component": "report_compiler"}
    )
    canonical_context_json, context_id, context_audit = canonicalizer.canonicalize(
        context, {"operation": "process", "stage": "aggregation_reporting", "component": "report_compiler"}
    )
    
    warnings = []
    status = ReportStatus.SUCCESS
    
    try:
        # Construct input file path
        input_path = Path("canonical_flow/aggregation") / f"{document_stem}_meso.json"
        
        # Check if input file exists - fail fast with graceful fallback
        if not input_path.exists():
            logger.warning(f"Meso aggregation file not found: {input_path}")
            warnings.append(ReportStatus.WARNINGS["MISSING_MESO"])
            status = ReportStatus.PARTIAL  # Allow system to continue processing other documents
            
            # Return graceful degradation response with partial status
            return create_deterministic_dict({
                "status": status,
                "document_stem": document_stem,
                "warnings": warnings,
                "timestamp": datetime.now().isoformat(),
                "processing_id": str(uuid.uuid4()),
                "report_data": create_deterministic_dict({
                    "executive_summary": {
                        "overview": "Input file not available for analysis.",
                        "consensus_analysis": "Analysis cannot proceed without meso aggregation data.",
                        "key_metrics": {"error": "No data available"}
                    }
                })
            })
        
        # Load meso aggregation data
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                meso_data = json.load(f)
        except json.JSONDecodeError as e:
            warnings.append(ReportStatus.WARNINGS["MALFORMED_DATA"])
            status = ReportStatus.PARTIAL
            meso_data = {}
        except UnicodeDecodeError:
            warnings.append(ReportStatus.WARNINGS["MALFORMED_DATA"])
            status = ReportStatus.PARTIAL
            meso_data = {}
        
        # Validate meso data structure
        if not isinstance(meso_data, dict):
            warnings.append(ReportStatus.WARNINGS["MALFORMED_DATA"])
            status = ReportStatus.PARTIAL
            meso_data = {}
        
        # Check for minimum required data
        required_keys = ["meso_summary", "coverage_matrix"]
        missing_keys = [key for key in required_keys if key not in meso_data]
        if missing_keys:
            warnings.append(ReportStatus.WARNINGS["INSUFFICIENT_EVIDENCE"])
            if status == ReportStatus.SUCCESS:
                status = ReportStatus.PARTIAL
        
        # Generate report sections with error handling
        processing_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        try:
            executive_summary = generate_executive_summary(meso_data)
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            warnings.append(ReportStatus.WARNINGS["PROCESSING_ERROR"])
            executive_summary = {"error": "Executive summary unavailable"}
            status = ReportStatus.PARTIAL
        
        try:
            findings_section = generate_findings_section(meso_data)
        except Exception as e:
            logger.error(f"Findings section generation failed: {e}")
            warnings.append(ReportStatus.WARNINGS["PROCESSING_ERROR"])
            findings_section = {"error": "Findings section unavailable"}
            status = ReportStatus.PARTIAL
        
        try:
            compliance_assessment = generate_compliance_assessment(meso_data)
        except Exception as e:
            logger.error(f"Compliance assessment generation failed: {e}")
            warnings.append(ReportStatus.WARNINGS["PROCESSING_ERROR"])
            compliance_assessment = {"error": "Compliance assessment unavailable"}
            status = ReportStatus.PARTIAL
        
        try:
            key_highlights = generate_key_highlights(meso_data)
        except Exception as e:
            logger.error(f"Key highlights generation failed: {e}")
            warnings.append(ReportStatus.WARNINGS["PROCESSING_ERROR"])
            key_highlights = {"error": "Key highlights unavailable"}
            status = ReportStatus.PARTIAL
        
        # Compile final report
        report_data = create_deterministic_dict({
            "executive_summary": executive_summary,
            "detailed_findings": findings_section,
            "compliance_assessment": compliance_assessment,
            "key_highlights": key_highlights,
            "metadata": create_deterministic_dict({
                "generation_timestamp": timestamp,
                "processing_id": processing_id,
                "input_file": str(input_path),
                "data_quality": status,
                "source_questions": len(meso_data.get("meso_summary", {}).get("items", {})),
                "processing_warnings": len(warnings)
            })
        })
        
        # Create output structure
        output_data = create_deterministic_dict({
            "status": status,
            "document_stem": document_stem,
            "warnings": warnings,
            "timestamp": timestamp,
            "processing_id": processing_id,
            "report_data": report_data
        })
        
        # Write output file with UTF-8 encoding
        output_path = Path("canonical_flow/aggregation") / f"{document_stem}_report.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, sort_keys=True)
            
            logger.info(f"Report generated successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
            warnings.append(ReportStatus.WARNINGS["PROCESSING_ERROR"])
            output_data["warnings"] = warnings
            
            if status == ReportStatus.SUCCESS:
                status = ReportStatus.PARTIAL
            # Add canonicalization metadata to output
        output_data["canonicalization"] = {
            "input_id": input_id,
            "context_id": context_id,
            "input_hash": input_audit.input_hash,
            "context_hash": context_audit.input_hash,
            "execution_time_ms": input_audit.execution_time_ms + context_audit.execution_time_ms
        }
        
        output_data["status"] = status
        
        # Canonicalize final output
        final_canonical_json, final_id, final_audit = canonicalizer.canonicalize(
            output_data, {"operation": "final_output", "stage": "aggregation_reporting", "component": "report_compiler"}
        )
        
        # Save audit trail to companion file
        output_file = f"report_compiler_{document_stem}_{input_id}.json"
        canonicalizer.save_audit_trail(output_file)
        
        return json.loads(final_canonical_json)
        
    except Exception as e:
        logger.error(f"Report compilation failed: {e}")
        
        # Create error output
        error_output = create_deterministic_dict({
            "status": ReportStatus.FAILED,
            "document_stem": document_stem,
            "warnings": [ReportStatus.WARNINGS["PROCESSING_ERROR"]],
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "processing_id": str(uuid.uuid4()),
            "canonicalization": {
                "input_id": input_id,
                "context_id": context_id,
                "input_hash": input_audit.input_hash,
                "context_hash": context_audit.input_hash if context_audit else "null",
                "execution_time_ms": (input_audit.execution_time_ms + 
                                    (context_audit.execution_time_ms if context_audit else 0))
            },
            "report_data": create_deterministic_dict({
                "executive_summary": {
                    "overview": "Report generation encountered critical errors.",
                    "consensus_analysis": "Processing could not be completed.",
                    "key_metrics": {"error": f"Processing failed: {str(e)}"}
                }
            })
        })
        
        # Canonicalize error output
        error_canonical_json, error_id, error_audit = canonicalizer.canonicalize(
            error_output, {"operation": "error", "stage": "aggregation_reporting", "component": "report_compiler"}
        )
        
        # Save error audit trail
        error_output_file = f"report_compiler_error_{document_stem}_{error_id}.json"
        canonicalizer.save_audit_trail(error_output_file)
        
        return json.loads(error_canonical_json)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        doc_stem = sys.argv[1]
        result = process(doc_stem)
        print(f"Report compilation status: {result['status']}")
        if result['warnings']:
            print(f"Warnings: {', '.join(result['warnings'])}")
    else:
        print("Usage: python report_compiler.py <document_stem>")