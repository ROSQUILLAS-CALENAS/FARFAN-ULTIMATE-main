"""
Calibration Dashboard System

This module implements a standardized calibration reporting system that generates
JSON reports for each pipeline stage (retrieval, confidence, aggregation).
"""

__all__ = ['CalibrationDashboard', 'CalibrationReport']

import json
import logging
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any, Optional, List  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Schema for calibration reports"""
    timestamp: str
    stage_name: str
    quality_score: float
    calibration_decision: str
    coverage_percentage: float
    quality_gate_passed: bool
    
    # Additional metadata
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CalibrationDashboard:
    """
    Calibration Dashboard System that generates standardized JSON reports
    for pipeline stage calibration data.
    """
    
    def __init__(self, output_directory: str = "canonical_flow/calibration"):
        """
        Initialize the calibration dashboard.
        
        Args:
            output_directory: Directory to write calibration reports
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Report filename mappings
        self.report_filenames = {
            "retrieval": "retrieval_calibration.json",
            "confidence": "confidence_calibration.json", 
            "aggregation": "aggregation_calibration.json"
        }
        
    def generate_report(self, stage_name: str, calibration_data: Dict[str, Any]) -> CalibrationReport:
        """
        Generate a calibration report for a given stage.
        
        Args:
            stage_name: Name of the pipeline stage
# # #             calibration_data: Raw calibration data from the stage  # Module not found  # Module not found  # Module not found
            
        Returns:
            CalibrationReport object
        """
        try:
            # Extract required fields with proper error handling
            quality_score = self._extract_quality_score(calibration_data)
            calibration_decision = self._extract_calibration_decision(calibration_data)
            coverage_percentage = self._extract_coverage_percentage(calibration_data)
            quality_gate_passed = self._extract_quality_gate_status(calibration_data)
            
            report = CalibrationReport(
                timestamp=datetime.now().isoformat(),
                stage_name=stage_name,
                quality_score=quality_score,
                calibration_decision=calibration_decision,
                coverage_percentage=coverage_percentage,
                quality_gate_passed=quality_gate_passed,
                metadata=calibration_data.get("metadata", {})
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating calibration report for {stage_name}: {e}")
            
            # Return error report
            return CalibrationReport(
                timestamp=datetime.now().isoformat(),
                stage_name=stage_name,
                quality_score=0.0,
                calibration_decision="error",
                coverage_percentage=0.0,
                quality_gate_passed=False,
                error_message=str(e),
                metadata=calibration_data.get("metadata", {})
            )
    
    def _extract_quality_score(self, data: Dict[str, Any]) -> float:
# # #         """Extract quality score from calibration data with fallbacks"""  # Module not found  # Module not found  # Module not found
        # Try multiple possible field names
        score_fields = ['quality_score', 'score', 'quality', 'metric_score', 'value']
        
        for field in score_fields:
            if field in data:
                score = data[field]
                if isinstance(score, (int, float)):
                    return float(max(0.0, min(1.0, score)))  # Clamp to [0, 1]
                    
        # Try nested structures
        if 'metrics' in data:
            metrics = data['metrics']
            for field in score_fields:
                if field in metrics:
                    score = metrics[field]
                    if isinstance(score, (int, float)):
                        return float(max(0.0, min(1.0, score)))
        
        # Default fallback
        logger.warning(f"No valid quality score found in calibration data, using 0.0")
        return 0.0
    
    def _extract_calibration_decision(self, data: Dict[str, Any]) -> str:
# # #         """Extract calibration decision from data"""  # Module not found  # Module not found  # Module not found
        decision_fields = ['calibration_decision', 'decision', 'action', 'recommendation']
        
        for field in decision_fields:
            if field in data:
                decision = str(data[field]).lower()
                # Normalize decision values
                if decision in ['pass', 'passed', 'accept', 'approved']:
                    return 'pass'
                elif decision in ['fail', 'failed', 'reject', 'denied']:
                    return 'fail'
                elif decision in ['recalibrate', 'adjust', 'tune']:
                    return 'recalibrate'
                else:
                    return decision
                    
        # Check quality gate status as fallback
        if data.get('quality_gate_passed'):
            return 'pass'
        elif data.get('quality_gate_passed') is False:
            return 'fail'
            
        return 'unknown'
    
    def _extract_coverage_percentage(self, data: Dict[str, Any]) -> float:
# # #         """Extract coverage percentage from data"""  # Module not found  # Module not found  # Module not found
        coverage_fields = ['coverage_percentage', 'coverage', 'percent_covered', 'completeness']
        
        for field in coverage_fields:
            if field in data:
                coverage = data[field]
                if isinstance(coverage, (int, float)):
                    return float(max(0.0, min(100.0, coverage)))
                    
        # Try nested metrics
        if 'metrics' in data:
            metrics = data['metrics']
            for field in coverage_fields:
                if field in metrics:
                    coverage = metrics[field]
                    if isinstance(coverage, (int, float)):
                        return float(max(0.0, min(100.0, coverage)))
        
        # Default fallback
        return 0.0
    
    def _extract_quality_gate_status(self, data: Dict[str, Any]) -> bool:
        """Extract quality gate pass/fail status"""
        gate_fields = ['quality_gate_passed', 'passed', 'gate_passed', 'validation_passed']
        
        for field in gate_fields:
            if field in data:
                return bool(data[field])
                
# # #         # Infer from quality score if available  # Module not found  # Module not found  # Module not found
        quality_score = self._extract_quality_score(data)
        threshold = data.get('quality_threshold', 0.8)
        
        return quality_score >= threshold
    
    def write_report(self, stage_name: str, report: CalibrationReport) -> Path:
        """
        Write calibration report to disk in JSON format.
        
        Args:
            stage_name: Name of the pipeline stage
            report: CalibrationReport to write
            
        Returns:
            Path to the written report file
        """
        # Get filename for stage
        filename = self.report_filenames.get(stage_name.lower(), f"{stage_name.lower()}_calibration.json")
        file_path = self.output_directory / filename
        
        try:
            # Convert to dict and ensure UTF-8 encoding
            report_data = asdict(report)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Calibration report written to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to write calibration report to {file_path}: {e}")
            raise
    
    def generate_and_write_report(self, stage_name: str, calibration_data: Dict[str, Any]) -> Path:
        """
        Generate and write a calibration report in one step.
        
        Args:
            stage_name: Name of the pipeline stage
# # #             calibration_data: Raw calibration data from the stage  # Module not found  # Module not found  # Module not found
            
        Returns:
            Path to the written report file
        """
        report = self.generate_report(stage_name, calibration_data)
        return self.write_report(stage_name, report)
    
    def read_report(self, stage_name: str) -> Optional[CalibrationReport]:
        """
# # #         Read an existing calibration report from disk.  # Module not found  # Module not found  # Module not found
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            CalibrationReport object or None if not found
        """
        filename = self.report_filenames.get(stage_name.lower(), f"{stage_name.lower()}_calibration.json")
        file_path = self.output_directory / filename
        
        try:
            if not file_path.exists():
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            return CalibrationReport(**report_data)
            
        except Exception as e:
# # #             logger.error(f"Failed to read calibration report from {file_path}: {e}")  # Module not found  # Module not found  # Module not found
            return None
    
    def list_reports(self) -> List[str]:
        """
        List all available calibration reports.
        
        Returns:
            List of stage names with available reports
        """
        reports = []
        
        for stage_name, filename in self.report_filenames.items():
            file_path = self.output_directory / filename
            if file_path.exists():
                reports.append(stage_name)
                
        return reports
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all calibration reports.
        
        Returns:
            Summary dictionary with aggregate statistics
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_reports": 0,
            "stages": {},
            "overall_quality_gate_passed": True,
            "average_quality_score": 0.0,
            "average_coverage": 0.0
        }
        
        reports = []
        for stage_name in self.list_reports():
            report = self.read_report(stage_name)
            if report:
                reports.append(report)
                summary["stages"][stage_name] = {
                    "quality_score": report.quality_score,
                    "calibration_decision": report.calibration_decision,
                    "coverage_percentage": report.coverage_percentage,
                    "quality_gate_passed": report.quality_gate_passed,
                    "timestamp": report.timestamp
                }
                
        summary["total_reports"] = len(reports)
        
        if reports:
            summary["average_quality_score"] = sum(r.quality_score for r in reports) / len(reports)
            summary["average_coverage"] = sum(r.coverage_percentage for r in reports) / len(reports)
            summary["overall_quality_gate_passed"] = all(r.quality_gate_passed for r in reports)
        
        return summary