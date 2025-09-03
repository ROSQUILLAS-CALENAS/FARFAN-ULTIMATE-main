"""
Canonical Flow Alias: GB
ReportCompiler with Total Ordering and Deterministic Processing

Source: G_aggregation_reporting/report_compiler.py
Stage: aggregation_reporting
Code: GB
"""

import json
import logging
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union  # Module not found  # Module not found  # Module not found
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# # # from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Available report output formats"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"


class ReportSection(Enum):
    """Standard report sections"""
    EXECUTIVE_SUMMARY = "executive_summary"
    METHODOLOGY = "methodology"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    FINDINGS = "findings"
    RECOMMENDATIONS = "recommendations"
    APPENDICES = "appendices"


class ReportSeverity(Enum):
    """Report finding severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ReportFinding:
    """Individual finding in the report"""
    finding_id: str
    title: str
    severity: ReportSeverity
    description: str
    evidence_ids: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.evidence_ids = sorted(self.evidence_ids)
        self.recommendations = sorted(self.recommendations)
        if self.supporting_data:
            self.supporting_data = OrderedDict(sorted(self.supporting_data.items()))


@dataclass
class CompiledReport:
    """Complete compiled report structure"""
    report_id: str
    title: str
    format_type: ReportFormat
    sections: Dict[str, Any] = field(default_factory=dict)
    findings: List[ReportFinding] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_timestamp: str = ""
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.sections = OrderedDict(sorted(self.sections.items()))
        self.findings = sorted(self.findings, key=lambda f: f.finding_id)
        if self.metadata:
            self.metadata = OrderedDict(sorted(self.metadata.items()))


class ReportCompiler(TotalOrderingBase, DeterministicCollectionMixin):
    """
    ReportCompiler for comprehensive report generation and compilation.
    
    This component compiles evidence aggregation results into structured,
    comprehensive reports with multiple output formats and standardized sections.
    
    Key Features:
    - Multi-format report generation (JSON, HTML, Markdown, Text, PDF)
    - Standardized report sections and structure
    - Finding categorization and severity assessment
    - Evidence traceability and citation management
    - Template-based report customization
    """
    
    def __init__(self, default_format: ReportFormat = ReportFormat.JSON):
        super().__init__("ReportCompiler")
        
        self.default_format = default_format
        
        # Configuration parameters
        self.include_raw_data = True
        self.include_evidence_citations = True
        self.max_findings_per_section = 50
        self.confidence_threshold = 0.5
        
        # Report templates and formatting
        self.section_templates = self._initialize_section_templates()
        self.formatting_rules = self._initialize_formatting_rules()
        
        # Compilation statistics
        self.compilation_stats = {
            "total_reports_compiled": 0,
            "successful_compilations": 0,
            "failed_compilations": 0,
            "average_findings_per_report": 0.0,
        }
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "default_format": self.default_format.value,
            "configuration": {
                "include_raw_data": self.include_raw_data,
                "include_evidence_citations": self.include_evidence_citations,
                "max_findings_per_section": self.max_findings_per_section,
                "confidence_threshold": self.confidence_threshold,
            }
        }
    
    def _initialize_section_templates(self) -> Dict[str, str]:
        """Initialize section templates for report generation"""
        return {
            ReportSection.EXECUTIVE_SUMMARY.value: "## Executive Summary\n\n{summary_content}\n\n",
            ReportSection.METHODOLOGY.value: "## Methodology\n\n{methodology_content}\n\n",
            ReportSection.EVIDENCE_ANALYSIS.value: "## Evidence Analysis\n\n{evidence_content}\n\n",
            ReportSection.FINDINGS.value: "## Key Findings\n\n{findings_content}\n\n",
            ReportSection.RECOMMENDATIONS.value: "## Recommendations\n\n{recommendations_content}\n\n",
            ReportSection.APPENDICES.value: "## Appendices\n\n{appendices_content}\n\n",
        }
    
    def _initialize_formatting_rules(self) -> Dict[str, Any]:
        """Initialize formatting rules for different report formats"""
        return {
            ReportFormat.JSON.value: {
                "indent": 2,
                "ensure_ascii": False,
                "sort_keys": True,
            },
            ReportFormat.MARKDOWN.value: {
                "heading_prefix": "##",
                "list_marker": "-",
                "emphasis": "**",
            },
            ReportFormat.HTML.value: {
                "doctype": "<!DOCTYPE html>",
                "encoding": "utf-8",
                "style_sheet": "default",
            },
            ReportFormat.TEXT.value: {
                "line_width": 80,
                "section_separator": "=" * 80,
                "subsection_separator": "-" * 40,
            },
        }
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function for report compilation.
        
        Args:
            data: Input data containing aggregation results and evidence
            context: Processing context with compilation parameters
            
        Returns:
            Deterministic report compilation results
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract report requirements
            report_config = self._extract_report_configuration(canonical_context)
            
            # Compile report sections
            compiled_sections = self._compile_report_sections_deterministic(
                canonical_data, report_config
            )
            
            # Generate findings
            findings = self._generate_findings_deterministic(canonical_data, report_config)
            
            # Create compiled report
            compiled_report = self._create_compiled_report(
                compiled_sections, findings, report_config, operation_id
            )
            
            # Format report according to requested format
            formatted_output = self._format_report_output(compiled_report, report_config)
            
            # Generate final output with confidence and quality metrics
            output = self._generate_final_output(formatted_output, compiled_report, operation_id)
            
            # Add confidence and quality metrics to report
# # #             from confidence_quality_metrics import ArtifactMetricsIntegrator  # Module not found  # Module not found  # Module not found
            
            integrator = ArtifactMetricsIntegrator()
# # #             meso_scores = []  # Would be populated from meso-level artifacts    # Module not found  # Module not found  # Module not found
            macro_metrics = integrator.calculator.propagate_to_macro_level(meso_scores, output)
            
            output.update({
                'confidence_score': macro_metrics.confidence_score,
                'quality_score': macro_metrics.quality_score,
                'metrics_metadata': {
                    'aggregation_level': 'macro',
                    'evidence_gaps': macro_metrics.evidence_gaps,
                    'uncertainty_factors': macro_metrics.uncertainty_factors,
                    'calculation_timestamp': macro_metrics.calculation_timestamp,
                }
            })
            
            # Update statistics
            self._update_compilation_stats(compiled_report)
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            error_output = {
                "component": self.component_name,
                "operation_id": operation_id,
                "error": str(e),
                "status": "error",
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _extract_report_configuration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract report configuration with defaults"""
        
        config = {
            "format": context.get("format", self.default_format.value),
            "title": context.get("title", "Evidence Analysis Report"),
            "sections": context.get("sections", [s.value for s in ReportSection]),
            "include_raw_data": context.get("include_raw_data", self.include_raw_data),
            "include_citations": context.get("include_citations", self.include_evidence_citations),
            "max_findings": context.get("max_findings", self.max_findings_per_section),
            "confidence_threshold": context.get("confidence_threshold", self.confidence_threshold),
        }
        
        return self.sort_dict_by_keys(config)
    
    def _compile_report_sections_deterministic(
        self, data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile report sections with deterministic content generation"""
        
        sections = {}
        requested_sections = config.get("sections", [])
        
        for section_name in sorted(requested_sections):
            if section_name == ReportSection.EXECUTIVE_SUMMARY.value:
                sections[section_name] = self._compile_executive_summary(data, config)
            elif section_name == ReportSection.METHODOLOGY.value:
                sections[section_name] = self._compile_methodology(data, config)
            elif section_name == ReportSection.EVIDENCE_ANALYSIS.value:
                sections[section_name] = self._compile_evidence_analysis(data, config)
            elif section_name == ReportSection.FINDINGS.value:
                sections[section_name] = self._compile_findings_section(data, config)
            elif section_name == ReportSection.RECOMMENDATIONS.value:
                sections[section_name] = self._compile_recommendations(data, config)
            elif section_name == ReportSection.APPENDICES.value:
                sections[section_name] = self._compile_appendices(data, config)
        
        return self.sort_dict_by_keys(sections)
    
    def _compile_executive_summary(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile executive summary section"""
        
        # Extract high-level metrics
        total_evidence = data.get("total_evidence", 0)
        avg_confidence = data.get("average_confidence", 0.0)
        key_findings_count = len([f for f in data.get("findings", []) if f.get("severity") in ["critical", "high"]])
        
        summary_content = {
            "overview": f"Analysis of {total_evidence} evidence items with average confidence of {avg_confidence:.2f}",
            "key_metrics": {
                "total_evidence_analyzed": total_evidence,
                "average_confidence_score": avg_confidence,
                "high_priority_findings": key_findings_count,
            },
            "summary_statement": self._generate_summary_statement(data),
        }
        
        return self.sort_dict_by_keys(summary_content)
    
    def _compile_methodology(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile methodology section"""
        
        methodology_content = {
            "approach": "Evidence-based analysis using EGW Query Expansion system",
            "aggregation_strategy": data.get("aggregation_strategy", "weighted_average"),
            "confidence_threshold": config.get("confidence_threshold", 0.5),
            "data_sources": self._extract_data_sources(data),
            "processing_steps": [
                "Evidence collection and validation",
                "Multi-dimensional aggregation",
                "Confidence assessment",
                "Finding generation and prioritization",
            ],
        }
        
        return self.sort_dict_by_keys(methodology_content)
    
    def _compile_evidence_analysis(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile evidence analysis section"""
        
        # Analyze evidence distribution and quality
        evidence_list = data.get("evidence_list", [])
        dimension_distribution = {}
        quality_distribution = {"high": 0, "medium": 0, "low": 0}
        
        for evidence in evidence_list:
            dimension = evidence.get("dimension", "unknown")
            dimension_distribution[dimension] = dimension_distribution.get(dimension, 0) + 1
            
            confidence = evidence.get("confidence_score", 0.5)
            if confidence >= 0.8:
                quality_distribution["high"] += 1
            elif confidence >= 0.6:
                quality_distribution["medium"] += 1
            else:
                quality_distribution["low"] += 1
        
        analysis_content = {
            "evidence_distribution": self.sort_dict_by_keys(dimension_distribution),
            "quality_distribution": quality_distribution,
            "total_evidence_count": len(evidence_list),
            "unique_sources": len(set(e.get("source_id", "") for e in evidence_list if e.get("source_id"))),
        }
        
        return self.sort_dict_by_keys(analysis_content)
    
    def _compile_findings_section(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile findings section"""
        
        findings_list = data.get("findings", [])
        
        # Group findings by severity
        severity_groups = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "info": [],
        }
        
        for finding in findings_list:
            severity = finding.get("severity", "medium")
            if severity in severity_groups:
                severity_groups[severity].append(finding)
        
        # Sort findings within each group
        for severity in severity_groups:
            severity_groups[severity] = sorted(
                severity_groups[severity],
                key=lambda f: f.get("finding_id", "")
            )
        
        findings_content = {
            "findings_by_severity": self.sort_dict_by_keys(severity_groups),
            "total_findings": len(findings_list),
            "severity_summary": {
                severity: len(findings)
                for severity, findings in severity_groups.items()
            },
        }
        
        return self.sort_dict_by_keys(findings_content)
    
    def _compile_recommendations(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile recommendations section"""
        
# # #         # Extract recommendations from findings  # Module not found  # Module not found  # Module not found
        all_recommendations = []
        findings_list = data.get("findings", [])
        
        for finding in findings_list:
            recommendations = finding.get("recommendations", [])
            for rec in recommendations:
                all_recommendations.append({
                    "recommendation": rec,
                    "finding_id": finding.get("finding_id", ""),
                    "severity": finding.get("severity", "medium"),
                })
        
        # Group recommendations by severity
        rec_by_severity = {}
        for rec in all_recommendations:
            severity = rec["severity"]
            if severity not in rec_by_severity:
                rec_by_severity[severity] = []
            rec_by_severity[severity].append(rec["recommendation"])
        
        recommendations_content = {
            "recommendations_by_priority": self.sort_dict_by_keys(rec_by_severity),
            "total_recommendations": len(all_recommendations),
            "actionable_items": len([r for r in all_recommendations if r["severity"] in ["critical", "high"]]),
        }
        
        return self.sort_dict_by_keys(recommendations_content)
    
    def _compile_appendices(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile appendices section"""
        
        appendices_content = {
# # #             "raw_data": data if config.get("include_raw_data", False) else "Raw data excluded from report",  # Module not found  # Module not found  # Module not found
            "evidence_citations": self._generate_evidence_citations(data) if config.get("include_citations", True) else [],
            "processing_metadata": {
                "component": self.component_name,
                "compilation_timestamp": self._get_deterministic_timestamp(),
                "configuration": config,
            },
        }
        
        return self.sort_dict_by_keys(appendices_content)
    
    def _generate_findings_deterministic(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[ReportFinding]:
# # #         """Generate findings from aggregation results"""  # Module not found  # Module not found  # Module not found
        
        findings = []
        aggregation_results = data.get("results", [])
        
        for i, result in enumerate(aggregation_results):
            if not isinstance(result, dict):
                continue
            
            # Determine severity based on aggregated score
            score = result.get("aggregated_score", 0.0)
            severity = self._determine_finding_severity(score)
            
            # Generate finding
            finding = ReportFinding(
                finding_id=self.generate_stable_id(
                    {"result_index": i, "aggregated_score": score},
                    prefix="finding"
                ),
                title=f"Evidence Analysis Result {i+1}",
                severity=severity,
# # #                 description=f"Aggregated evidence score: {score:.3f} from {result.get('evidence_count', 0)} evidence items",  # Module not found  # Module not found  # Module not found
                evidence_ids=result.get("evidence_ids", []),
                confidence_score=result.get("aggregated_score", 0.0),
                supporting_data={
                    "aggregation_id": result.get("aggregation_id", ""),
                    "dimension_scores": result.get("dimension_scores", {}),
                    "summary_metrics": result.get("summary_metrics", {}),
                },
                recommendations=self._generate_finding_recommendations(result, severity),
            )
            
            findings.append(finding)
        
        return findings
    
    def _determine_finding_severity(self, score: float) -> ReportSeverity:
        """Determine finding severity based on aggregated score"""
        if score >= 0.9:
            return ReportSeverity.CRITICAL
        elif score >= 0.8:
            return ReportSeverity.HIGH
        elif score >= 0.6:
            return ReportSeverity.MEDIUM
        elif score >= 0.3:
            return ReportSeverity.LOW
        else:
            return ReportSeverity.INFO
    
    def _generate_finding_recommendations(self, result: Dict[str, Any], severity: ReportSeverity) -> List[str]:
        """Generate recommendations based on finding results"""
        recommendations = []
        
        score = result.get("aggregated_score", 0.0)
        evidence_count = result.get("evidence_count", 0)
        
        if severity in [ReportSeverity.CRITICAL, ReportSeverity.HIGH]:
            recommendations.append("Immediate attention required")
            recommendations.append("Validate with additional sources")
        
        if evidence_count < 3:
            recommendations.append("Collect additional evidence to strengthen findings")
        
        if score < 0.5:
            recommendations.append("Review evidence quality and relevance")
        
        return sorted(recommendations)
    
    def _create_compiled_report(
        self,
        sections: Dict[str, Any],
        findings: List[ReportFinding],
        config: Dict[str, Any],
        operation_id: str,
    ) -> CompiledReport:
        """Create the final compiled report structure"""
        
        report = CompiledReport(
            report_id=self.generate_stable_id(
                {"operation_id": operation_id, "title": config["title"]},
                prefix="report"
            ),
            title=config["title"],
            format_type=ReportFormat(config["format"]),
            sections=sections,
            findings=findings,
            metadata={
                "generation_timestamp": self._get_deterministic_timestamp(),
                "configuration": config,
                "component": self.component_name,
                "operation_id": operation_id,
            },
            generation_timestamp=self._get_deterministic_timestamp(),
        )
        
        return report
    
    def _format_report_output(self, report: CompiledReport, config: Dict[str, Any]) -> str:
        """Format report according to requested output format"""
        
        if report.format_type == ReportFormat.JSON:
            return json.dumps(self._report_to_dict(report), **self.formatting_rules["json"])
        
        elif report.format_type == ReportFormat.MARKDOWN:
            return self._format_as_markdown(report)
        
        elif report.format_type == ReportFormat.HTML:
            return self._format_as_html(report)
        
        elif report.format_type == ReportFormat.TEXT:
            return self._format_as_text(report)
        
        else:
            # Default to JSON
            return json.dumps(self._report_to_dict(report), **self.formatting_rules["json"])
    
    def _report_to_dict(self, report: CompiledReport) -> Dict[str, Any]:
        """Convert CompiledReport to dictionary for serialization"""
        return {
            "report_id": report.report_id,
            "title": report.title,
            "format_type": report.format_type.value,
            "sections": report.sections,
            "findings": [self._finding_to_dict(f) for f in report.findings],
            "metadata": report.metadata,
            "generation_timestamp": report.generation_timestamp,
        }
    
    def _finding_to_dict(self, finding: ReportFinding) -> Dict[str, Any]:
        """Convert ReportFinding to dictionary"""
        return {
            "finding_id": finding.finding_id,
            "title": finding.title,
            "severity": finding.severity.value,
            "description": finding.description,
            "evidence_ids": finding.evidence_ids,
            "confidence_score": finding.confidence_score,
            "supporting_data": finding.supporting_data,
            "recommendations": finding.recommendations,
        }
    
    def _format_as_markdown(self, report: CompiledReport) -> str:
        """Format report as Markdown"""
        md_content = [f"# {report.title}\n"]
        
        for section_name, section_content in report.sections.items():
            md_content.append(f"## {section_name.replace('_', ' ').title()}\n")
            md_content.append(f"{json.dumps(section_content, indent=2)}\n\n")
        
        return "".join(md_content)
    
    def _format_as_html(self, report: CompiledReport) -> str:
        """Format report as HTML"""
        html_content = [
            "<!DOCTYPE html>",
            "<html><head><title>{}</title></head><body>".format(report.title),
            f"<h1>{report.title}</h1>",
        ]
        
        for section_name, section_content in report.sections.items():
            html_content.append(f"<h2>{section_name.replace('_', ' ').title()}</h2>")
            html_content.append(f"<pre>{json.dumps(section_content, indent=2)}</pre>")
        
        html_content.append("</body></html>")
        return "\n".join(html_content)
    
    def _format_as_text(self, report: CompiledReport) -> str:
        """Format report as plain text"""
        text_content = [
            f"{report.title}",
            "=" * len(report.title),
            "",
        ]
        
        for section_name, section_content in report.sections.items():
            text_content.append(section_name.replace("_", " ").title())
            text_content.append("-" * len(section_name))
            text_content.append(json.dumps(section_content, indent=2))
            text_content.append("")
        
        return "\n".join(text_content)
    
    def _generate_summary_statement(self, data: Dict[str, Any]) -> str:
        """Generate high-level summary statement"""
        total_evidence = data.get("total_evidence", 0)
        findings_count = len(data.get("findings", []))
        
        return f"Analysis completed with {total_evidence} evidence items resulting in {findings_count} findings."
    
    def _extract_data_sources(self, data: Dict[str, Any]) -> List[str]:
# # #         """Extract unique data sources from evidence"""  # Module not found  # Module not found  # Module not found
        sources = set()
        evidence_list = data.get("evidence_list", [])
        
        for evidence in evidence_list:
            source = evidence.get("source_id", evidence.get("document_id", "unknown"))
            sources.add(source)
        
        return sorted(list(sources))
    
    def _generate_evidence_citations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate citation list for evidence"""
        citations = []
        evidence_list = data.get("evidence_list", [])
        
        for evidence in evidence_list:
            if "citation" in evidence:
                citations.append({
                    "evidence_id": evidence.get("evidence_id", ""),
                    "citation": evidence["citation"],
                })
        
        return sorted(citations, key=lambda c: c.get("evidence_id", ""))
    
    def _generate_final_output(
        self, formatted_report: str, compiled_report: CompiledReport, operation_id: str
    ) -> Dict[str, Any]:
        """Generate final processing output"""
        
        output = {
            "component": self.component_name,
            "component_id": self.component_id,
            "operation_id": operation_id,
            "report_id": compiled_report.report_id,
            "report_title": compiled_report.title,
            "report_format": compiled_report.format_type.value,
            "formatted_report": formatted_report,
            "compilation_summary": {
                "sections_compiled": len(compiled_report.sections),
                "findings_generated": len(compiled_report.findings),
                "total_report_length": len(formatted_report),
            },
            "metadata": self.get_deterministic_metadata(),
            "status": "success",
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(output)
    
    def _update_compilation_stats(self, report: CompiledReport) -> None:
        """Update internal compilation statistics"""
        self.compilation_stats["total_reports_compiled"] += 1
        self.compilation_stats["successful_compilations"] += 1
        
        findings_count = len(report.findings)
        current_avg = self.compilation_stats["average_findings_per_report"]
        total_reports = self.compilation_stats["total_reports_compiled"]
        
        # Update running average
        self.compilation_stats["average_findings_per_report"] = (
            (current_avg * (total_reports - 1) + findings_count) / total_reports
        )


# Maintain backward compatibility
def process(data=None, context=None):
    """Backward compatible process function"""
    compiler = ReportCompiler()
    return compiler.process(data, context)