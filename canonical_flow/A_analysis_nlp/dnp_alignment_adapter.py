"""
Canonical Flow Alias: 18A
DNP Alignment Adapter with Total Ordering and Deterministic Processing

Source: dnp_alignment_adapter.py
Stage: analysis_nlp
Code: 18A
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import logging

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin

logger = logging.getLogger(__name__)


class DNPAlignmentAdapter(TotalOrderingBase, DeterministicCollectionMixin):
    """
    DNP Alignment Adapter with deterministic processing and total ordering.
    
    Provides consistent DNP compliance results and stable ID generation across runs.
    Serves as a stable, callable entrypoint for the deterministic orchestrator.
    """
    
    def __init__(self):
        super().__init__("DNPAlignmentAdapter")
        
        # Configuration with deterministic defaults
        self.compliance_threshold = 0.7
        self.max_processing_time_ms = 10000
        
        # DNP standards with deterministic ordering
        self.dnp_standards = self._initialize_dnp_standards()
        
        # Compliance tracking statistics
        self.compliance_stats = {
            "documents_processed": 0,
            "compliant_documents": 0,
            "non_compliant_documents": 0,
            "avg_compliance_score": 0.0,
            "last_update": self._get_deterministic_timestamp(),
        }
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_comparison_key(self) -> Tuple[str, ...]:
        """Return comparison key for deterministic ordering"""
        return (
            self.component_id,
            str(self.compliance_threshold),
            str(self.max_processing_time_ms),
            str(len(self.dnp_standards)),
            str(self._state_hash or "")
        )
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "compliance_threshold": self.compliance_threshold,
            "max_processing_time_ms": self.max_processing_time_ms,
            "standards_count": len(self.dnp_standards),
        }
    
    def _initialize_dnp_standards(self) -> Dict[str, Dict[str, Any]]:
        """Initialize DNP standards with deterministic ordering"""
        standards = {
            "budget_allocation": {
                "description": "Proper budget allocation according to DNP guidelines",
                "indicators": ["presupuesto", "asignación", "recursos", "inversión"],
                "required_sections": ["financial_plan", "budget_breakdown"],
                "weight": 0.25,
            },
            "constitutional_compliance": {
                "description": "Compliance with constitutional requirements",
                "indicators": ["constitución", "derechos fundamentales", "principios"],
                "required_sections": ["legal_framework", "constitutional_basis"],
                "weight": 0.20,
            },
            "participatory_planning": {
                "description": "Evidence of participatory planning process",
                "indicators": ["participación", "consulta", "ciudadana", "comunidad"],
                "required_sections": ["participation_plan", "community_engagement"],
                "weight": 0.15,
            },
            "territorial_approach": {
                "description": "Territorial development approach alignment",
                "indicators": ["territorial", "desarrollo", "región", "local"],
                "required_sections": ["territorial_diagnosis", "development_strategy"],
                "weight": 0.20,
            },
            "transparency_accountability": {
                "description": "Transparency and accountability measures",
                "indicators": ["transparencia", "rendición", "cuentas", "control"],
                "required_sections": ["monitoring_plan", "accountability_framework"],
                "weight": 0.20,
            },
        }
        
        return self.sort_dict_by_keys(standards)
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function with deterministic output.
        
        Args:
            data: Input data containing PDT/evaluation data
            context: Processing context
            
        Returns:
            Deterministic DNP alignment results
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract PDT and evaluation data
            pdt_data, eval_results = self._extract_pdt_and_eval_deterministic(canonical_data)
            
            # Perform DNP compliance check
            compliance_results = self._check_dnp_compliance_deterministic(pdt_data, eval_results)
            
            # Generate DNP report
            dnp_report = self._generate_dnp_report_deterministic(compliance_results)
            
            # Create enriched output
            enriched_data = self._create_enriched_output_deterministic(
                canonical_data, compliance_results, dnp_report, operation_id
            )
            
            # Update statistics
            self._update_compliance_stats(compliance_results)
            
            # Update state hash
            self.update_state_hash(enriched_data)
            
            return enriched_data
            
        except Exception as e:
            # Handle errors gracefully to keep pipeline flowing
            logger.error(f"DNP Alignment Adapter error: {e}")
            
            # Return input data with error annotation
            error_output = canonical_data.copy() if isinstance(canonical_data, dict) else {"input": canonical_data}
            error_output.update({
                "component": self.component_name,
                "dnp_guardian": {
                    "enforced": False,
                    "error": str(e),
                    "timestamp": self._get_deterministic_timestamp(),
                },
                "operation_id": operation_id,
                "status": "error_passthrough",
            })
            
            return self.sort_dict_by_keys(error_output)
    
    def _extract_pdt_and_eval_deterministic(self, payload: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Best-effort extraction of PDT data and evaluation results from arbitrary payloads"""
        pdt_data = {}
        eval_results = {}
        
        if isinstance(payload, dict):
            # Common locations for PDT data
            pdt_data = (
                payload.get("pdt_data") or
                payload.get("document") or 
                payload.get("result") or
                payload.get("data") or
                {}
            )
            
            if not isinstance(pdt_data, dict):
                pdt_data = {"raw": pdt_data}
            
            # Common locations for evaluation results
            eval_results = (
                payload.get("evaluation_results") or
                payload.get("results") or
                payload.get("evaluation") or
                payload.get("analysis") or
                {}
            )
            
            if not isinstance(eval_results, dict):
                eval_results = {"raw": eval_results}
        
        elif isinstance(payload, str):
            # Handle text input
            pdt_data = {"text": payload}
            eval_results = {}
        
        else:
            # Handle other types
            pdt_data = {"raw": payload}
            eval_results = {}
        
        return pdt_data, eval_results
    
    def _check_dnp_compliance_deterministic(self, pdt_data: Dict[str, Any], eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check DNP compliance with deterministic scoring"""
        
        # Combine data for analysis
        combined_text = ""
        
        # Extract text from PDT data
        if "text" in pdt_data:
            combined_text += str(pdt_data["text"]) + " "
        if "content" in pdt_data:
            combined_text += str(pdt_data["content"]) + " "
        if "raw" in pdt_data:
            combined_text += str(pdt_data["raw"]) + " "
        
        # Extract text from evaluation results
        if "text" in eval_results:
            combined_text += str(eval_results["text"]) + " "
        if "summary" in eval_results:
            combined_text += str(eval_results["summary"]) + " "
        if "raw" in eval_results:
            combined_text += str(eval_results["raw"]) + " "
        
        combined_text = combined_text.lower().strip()
        
        compliance_results = {
            "overall_compliance_score": 0.0,
            "is_compliant": False,
            "standard_scores": {},
            "missing_elements": [],
            "compliance_issues": [],
            "recommendations": [],
        }
        
        # Check each DNP standard
        total_weighted_score = 0.0
        total_weights = 0.0
        
        for standard_name, standard_config in sorted(self.dnp_standards.items()):
            # Calculate standard score
            standard_score = self._calculate_standard_score_deterministic(
                combined_text, standard_config
            )
            
            compliance_results["standard_scores"][standard_name] = standard_score
            
            # Weight the score
            weight = standard_config["weight"]
            total_weighted_score += standard_score * weight
            total_weights += weight
            
            # Check for missing elements
            if standard_score < 0.5:
                compliance_results["missing_elements"].append({
                    "standard": standard_name,
                    "description": standard_config["description"],
                    "score": standard_score,
                })
            
            # Generate issues for low scores
            if standard_score < 0.3:
                compliance_results["compliance_issues"].append({
                    "issue": f"Low compliance for {standard_name}: {standard_score:.2f}",
                    "severity": "high",
                    "standard": standard_name,
                })
            elif standard_score < 0.5:
                compliance_results["compliance_issues"].append({
                    "issue": f"Moderate compliance for {standard_name}: {standard_score:.2f}",
                    "severity": "medium",
                    "standard": standard_name,
                })
        
        # Calculate overall compliance score
        if total_weights > 0:
            compliance_results["overall_compliance_score"] = total_weighted_score / total_weights
        
        # Determine if compliant
        compliance_results["is_compliant"] = (
            compliance_results["overall_compliance_score"] >= self.compliance_threshold
        )
        
        # Generate recommendations
        compliance_results["recommendations"] = self._generate_recommendations_deterministic(
            compliance_results
        )
        
        # Sort elements for deterministic output
        compliance_results["missing_elements"] = sorted(
            compliance_results["missing_elements"], 
            key=lambda x: x["standard"]
        )
        compliance_results["compliance_issues"] = sorted(
            compliance_results["compliance_issues"],
            key=lambda x: (x["severity"], x["standard"])
        )
        compliance_results["recommendations"] = sorted(compliance_results["recommendations"])
        
        return compliance_results
    
    def _calculate_standard_score_deterministic(self, text: str, standard_config: Dict[str, Any]) -> float:
        """Calculate compliance score for a DNP standard deterministically"""
        
        indicators = standard_config.get("indicators", [])
        required_sections = standard_config.get("required_sections", [])
        
        # Score based on indicator presence
        indicator_score = 0.0
        if indicators:
            matches = 0
            for indicator in sorted(indicators):
                if indicator in text:
                    matches += 1
            indicator_score = matches / len(indicators)
        
        # Score based on required sections (simplified check)
        section_score = 0.0
        if required_sections:
            matches = 0
            for section in sorted(required_sections):
                # Simple check for section-related keywords
                section_keywords = section.replace("_", " ").split()
                if any(keyword in text for keyword in section_keywords):
                    matches += 1
            section_score = matches / len(required_sections)
        
        # Combine scores
        if indicators and required_sections:
            combined_score = (indicator_score * 0.6) + (section_score * 0.4)
        elif indicators:
            combined_score = indicator_score
        elif required_sections:
            combined_score = section_score
        else:
            combined_score = 0.5  # Neutral score
        
        # Apply text length adjustment
        text_length = len(text)
        if text_length > 1000:
            combined_score += 0.1  # Bonus for comprehensive text
        elif text_length < 100:
            combined_score -= 0.2  # Penalty for very short text
        
        return max(0.0, min(1.0, combined_score))
    
    def _generate_recommendations_deterministic(self, compliance_results: Dict[str, Any]) -> list[str]:
        """Generate recommendations based on compliance results"""
        recommendations = []
        
        overall_score = compliance_results["overall_compliance_score"]
        
        # General recommendations based on overall score
        if overall_score < 0.3:
            recommendations.append("Comprehensive review of DNP compliance required")
        elif overall_score < 0.5:
            recommendations.append("Significant improvements needed for DNP compliance")
        elif overall_score < 0.7:
            recommendations.append("Minor adjustments needed for full DNP compliance")
        
        # Specific recommendations based on missing elements
        missing_standards = set()
        for missing in compliance_results.get("missing_elements", []):
            missing_standards.add(missing["standard"])
        
        if "budget_allocation" in missing_standards:
            recommendations.append("Include detailed budget allocation and financial planning")
        
        if "participatory_planning" in missing_standards:
            recommendations.append("Document participatory planning processes and community engagement")
        
        if "territorial_approach" in missing_standards:
            recommendations.append("Strengthen territorial development approach and local focus")
        
        if "transparency_accountability" in missing_standards:
            recommendations.append("Enhance transparency and accountability mechanisms")
        
        if "constitutional_compliance" in missing_standards:
            recommendations.append("Ensure constitutional compliance and legal framework alignment")
        
        return sorted(list(set(recommendations)))  # Remove duplicates and sort
    
    def _generate_dnp_report_deterministic(self, compliance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate DNP compliance report with deterministic structure"""
        
        overall_score = compliance_results["overall_compliance_score"]
        is_compliant = compliance_results["is_compliant"]
        
        # Determine compliance level
        if overall_score >= 0.8:
            compliance_level = "excellent"
        elif overall_score >= 0.7:
            compliance_level = "good"
        elif overall_score >= 0.5:
            compliance_level = "fair"
        elif overall_score >= 0.3:
            compliance_level = "poor"
        else:
            compliance_level = "critical"
        
        # Generate summary
        summary = f"DNP compliance assessment shows {compliance_level} alignment with a score of {overall_score:.2f}."
        if is_compliant:
            summary += " The document meets DNP standards."
        else:
            summary += " The document requires improvements to meet DNP standards."
        
        dnp_report = {
            "compliance_level": compliance_level,
            "compliance_score": overall_score,
            "compliance_status": "compliant" if is_compliant else "non_compliant",
            "detailed_scores": self.sort_dict_by_keys(compliance_results["standard_scores"]),
            "executive_summary": summary,
            "issues_count": len(compliance_results.get("compliance_issues", [])),
            "missing_elements_count": len(compliance_results.get("missing_elements", [])),
            "recommendations": compliance_results.get("recommendations", []),
            "report_timestamp": self._get_deterministic_timestamp(),
        }
        
        return dnp_report
    
    def _create_enriched_output_deterministic(self, original_data: Dict[str, Any], compliance_results: Dict[str, Any], dnp_report: Dict[str, Any], operation_id: str) -> Dict[str, Any]:
        """Create enriched output with DNP compliance data"""
        
        # Start with original data
        enriched_data = original_data.copy() if isinstance(original_data, dict) else {"input": original_data}
        
        # Add DNP compliance information
        enriched_data.update({
            "component": self.component_name,
            "component_id": self.component_id,
            "dnp_compliance": compliance_results,
            "dnp_guardian": {
                "enforced": True,
                "engine": "DNPAlignmentAdapter", 
                "timestamp": self._get_deterministic_timestamp(),
            },
            "dnp_report": dnp_report,
            "metadata": self.get_deterministic_metadata(),
            "operation_id": operation_id,
            "status": "success",
            "timestamp": self._get_deterministic_timestamp(),
        })
        
        return self.sort_dict_by_keys(enriched_data)
    
    def _update_compliance_stats(self, compliance_results: Dict[str, Any]):
        """Update compliance statistics"""
        self.compliance_stats["documents_processed"] += 1
        
        if compliance_results["is_compliant"]:
            self.compliance_stats["compliant_documents"] += 1
        else:
            self.compliance_stats["non_compliant_documents"] += 1
        
        # Update average compliance score
        current_avg = self.compliance_stats["avg_compliance_score"]
        count = self.compliance_stats["documents_processed"]
        new_score = compliance_results["overall_compliance_score"]
        new_avg = (current_avg * (count - 1) + new_score) / count
        self.compliance_stats["avg_compliance_score"] = new_avg
        
        # Update timestamp
        self.compliance_stats["last_update"] = self._get_deterministic_timestamp()
    
    def save_artifact(self, data: Dict[str, Any], document_stem: str, output_dir: str = "canonical_flow/analysis") -> str:
        """
        Save DNP alignment output to canonical_flow/analysis/ directory with standardized naming.
        
        Args:
            data: DNP alignment data to save
            document_stem: Base document identifier (without extension)
            output_dir: Output directory (defaults to canonical_flow/analysis)
            
        Returns:
            Path to saved artifact
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate standardized filename
            filename = f"{document_stem}_dnp.json"
            artifact_path = Path(output_dir) / filename
            
            # Save with consistent JSON formatting
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"DNPAlignmentAdapter artifact saved: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            error_msg = f"Failed to save DNPAlignmentAdapter artifact to {output_dir}/{document_stem}_dnp.json: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Backward compatibility functions
def check_dnp_compliance(data: Any) -> Dict[str, Any]:
    """Check DNP compliance for given data"""
    adapter = DNPAlignmentAdapter()
    return adapter.process(data)


def process(data=None, context=None):
    """Backward compatible process function"""
    adapter = DNPAlignmentAdapter()
    result = adapter.process(data, context)
    
    # Save artifact if document_stem is provided
    if data and isinstance(data, dict) and 'document_stem' in data:
        adapter.save_artifact(result, data['document_stem'])
    
    return result