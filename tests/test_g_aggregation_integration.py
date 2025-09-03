"""
Integration test suite for G_aggregation_reporting stage.

Tests the complete execution of both meso_aggregator and report_compiler
components sequentially, validating:
- Artifact generation in canonical_flow/aggregation/
- Schema compliance for meso.json and compiled reports
- Deterministic content across multiple runs
- Audit logging completeness
- Status reporting accuracy for success/partial/failure states
"""

import json
import os
import tempfile
import shutil
import pytest
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any, List, Optional  # Module not found  # Module not found  # Module not found
# # # from unittest.mock import patch, MagicMock  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import hashlib

# Import the components under test
# # # from meso_aggregator import process as meso_process  # Module not found  # Module not found  # Module not found
# # # from report_compiler import ReportCompiler, ReportData, ReportType, CompiledReport  # Module not found  # Module not found  # Module not found
# # # from canonical_flow.analysis.audit_logger import AuditLogger, get_audit_logger, set_audit_logger  # Module not found  # Module not found  # Module not found


class TestGAggregationIntegration:
    """Integration tests for G_aggregation_reporting stage execution."""
    
    @pytest.fixture
    def temp_output_dir(self) -> Path:
        """Create temporary output directory for test artifacts."""
        temp_dir = Path(tempfile.mkdtemp())
        aggregation_dir = temp_dir / "canonical_flow" / "aggregation"
        aggregation_dir.mkdir(parents=True)
        yield aggregation_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_cluster_data(self) -> Dict[str, Any]:
# # #         """Sample analysis outputs from four clusters for testing."""  # Module not found  # Module not found  # Module not found
        return {
            "cluster_audit": {
                "micro": {
                    "C1": {
                        "answers": [
                            {
                                "question_id": "Q001",
                                "question": "Does the plan address sustainability objectives?",
                                "verdict": "YES",
                                "score": 0.85,
                                "confidence": 0.8,
                                "evidence_ids": ["E001", "E002"],
                                "components": ["SUSTAINABILITY", "OBJECTIVES"]
                            },
                            {
                                "question_id": "Q002", 
                                "question": "Are stakeholder engagement mechanisms defined?",
                                "verdict": "PARTIAL",
                                "score": 0.65,
                                "confidence": 0.7,
                                "evidence_ids": ["E003"],
                                "components": ["STAKEHOLDERS"]
                            }
                        ]
                    },
                    "C2": {
                        "answers": [
                            {
                                "question_id": "Q001",
                                "question": "Does the plan address sustainability objectives?", 
                                "verdict": "YES",
                                "score": 0.90,
                                "confidence": 0.85,
                                "evidence_ids": ["E001", "E004"],
                                "components": ["SUSTAINABILITY", "OBJECTIVES", "INDICATORS"]
                            },
                            {
                                "question_id": "Q002",
                                "question": "Are stakeholder engagement mechanisms defined?",
                                "verdict": "NO", 
                                "score": 0.35,
                                "confidence": 0.75,
                                "evidence_ids": ["E005"],
                                "components": ["STAKEHOLDERS"]
                            }
                        ]
                    },
                    "C3": {
                        "answers": [
                            {
                                "question_id": "Q001",
                                "question": "Does the plan address sustainability objectives?",
                                "verdict": "YES",
                                "score": 0.80,
                                "confidence": 0.9,
                                "evidence_ids": ["E001", "E006"],
                                "components": ["SUSTAINABILITY", "OBJECTIVES"]
                            }
                        ]
                    },
                    "C4": {
                        "answers": [
                            {
                                "question_id": "Q002",
                                "question": "Are stakeholder engagement mechanisms defined?",
                                "verdict": "PARTIAL",
                                "score": 0.55,
                                "confidence": 0.6,
                                "evidence_ids": ["E007", "E008"],
                                "components": ["STAKEHOLDERS", "COMPLIANCE"]
                            }
                        ]
                    }
                }
            },
            "plan_metadata": {
                "plan_name": "Test Municipal Development Plan",
                "municipality": "Test City",
                "analysis_date": "2024-01-01"
            }
        }
    
    @pytest.fixture
    def audit_logger(self, temp_output_dir: Path) -> AuditLogger:
        """Create isolated audit logger for testing."""
        audit_file = temp_output_dir / "test_audit.json"
        logger = AuditLogger(str(audit_file))
        set_audit_logger(logger)
        return logger
    
    @pytest.fixture
    def sample_report_data(self, sample_cluster_data: Dict[str, Any]) -> ReportData:
        """Sample report data for compiler testing."""
        return ReportData(
            plan_name="Test Municipal Development Plan",
            analysis_results=sample_cluster_data,
            evidence_responses=[],
            scoring_outputs={},
            normative_references=[],
            metadata=sample_cluster_data.get("plan_metadata", {})
        )
    
    def test_meso_aggregator_execution(self, sample_cluster_data: Dict[str, Any], 
                                     temp_output_dir: Path):
        """Test meso_aggregator generates valid meso.json with proper schema."""
        
        # Execute meso aggregation
        result = meso_process(sample_cluster_data)
        
        # Validate core structure
        assert "meso_summary" in result
        assert "coverage_matrix" in result
        
        # Validate meso_summary structure
        meso_summary = result["meso_summary"]
        assert "items" in meso_summary
        assert "divergence_stats" in meso_summary
        assert "cluster_participation" in meso_summary
        assert "component_coverage_summary" in meso_summary
        
        # Validate coverage matrix structure
        coverage_matrix = result["coverage_matrix"]
        expected_components = [
            "OBJECTIVES", "STRATEGIES", "INDICATORS", "TIMELINES", "BUDGET",
            "STAKEHOLDERS", "RISKS", "COMPLIANCE", "SUSTAINABILITY", "IMPACT"
        ]
        for component in expected_components:
            assert component in coverage_matrix
            assert "clusters_evaluating" in coverage_matrix[component]
            assert "questions_addressing" in coverage_matrix[component]
            assert "total_evaluations" in coverage_matrix[component]
            assert "coverage_percentage" in coverage_matrix[component]
        
        # Save meso.json to output directory
        meso_file = temp_output_dir / "meso.json"
        with open(meso_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Validate file exists and is valid JSON
        assert meso_file.exists()
        with open(meso_file, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data == result
    
    def test_report_compiler_execution(self, sample_report_data: ReportData,
                                     temp_output_dir: Path):
        """Test report_compiler generates structured reports."""
        
        compiler = ReportCompiler()
        
        # Test different report types
        for report_type in [ReportType.MACRO, ReportType.MESO, ReportType.MICRO]:
            report = compiler.compile_report(sample_report_data, report_type)
            
            # Validate report structure
            assert isinstance(report, CompiledReport)
            assert report.report_type == report_type
            assert report.plan_name == sample_report_data.plan_name
            assert len(report.sections) > 0
            assert isinstance(report.overall_score, float)
            assert 0.0 <= report.overall_score <= 1.0
            
            # Validate required sections exist
            required_sections = ["executive_summary", "strengths", "weaknesses", "decalogo_alignment"]
            for section_name in required_sections:
                section_key = next((k for k in report.sections.keys() if k.value == section_name), None)
                assert section_key is not None, f"Missing required section: {section_name}"
                assert len(report.sections[section_key]) > 0
            
            # Save report to output directory
            report_file = temp_output_dir / f"{report_type.value}_report.json"
            report_dict = {
                "report_id": report.report_id,
                "report_type": report.report_type.value,
                "plan_name": report.plan_name,
                "sections": {k.value: v for k, v in report.sections.items()},
                "overall_score": report.overall_score,
                "generation_metadata": report.generation_metadata,
                "created_at": report.created_at.isoformat()
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            assert report_file.exists()
    
    def test_sequential_execution_flow(self, sample_cluster_data: Dict[str, Any],
                                     temp_output_dir: Path, audit_logger: AuditLogger):
        """Test complete sequential execution of both components."""
        
        execution_id = f"test_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        audit_logger.start_execution(execution_id)
        
        # Step 1: Execute meso aggregation
        with audit_logger.audit_component_execution("meso_aggregator", sample_cluster_data) as audit_ctx:
            meso_result = meso_process(sample_cluster_data)
            audit_ctx.set_output_data(meso_result)
        
        # Validate meso aggregation artifacts
        meso_file = temp_output_dir / "meso.json"
        with open(meso_file, 'w') as f:
            json.dump(meso_result, f, indent=2)
        
        assert meso_file.exists()
        
# # #         # Step 2: Prepare report data from meso results  # Module not found  # Module not found  # Module not found
        report_data = ReportData(
            plan_name=sample_cluster_data.get("plan_metadata", {}).get("plan_name", "Test Plan"),
            analysis_results=meso_result,
            evidence_responses=[],
            scoring_outputs={},
            normative_references=[],
            metadata=sample_cluster_data.get("plan_metadata", {})
        )
        
        # Step 3: Execute report compilation
        compiler = ReportCompiler()
        
        for report_type in [ReportType.MACRO, ReportType.MESO]:
            with audit_logger.audit_component_execution(f"report_compiler_{report_type.value}", 
                                                       report_data.__dict__) as audit_ctx:
                report = compiler.compile_report(report_data, report_type)
                
                report_dict = {
                    "report_id": report.report_id,
                    "report_type": report.report_type.value,
                    "plan_name": report.plan_name,
                    "sections": {k.value: v for k, v in report.sections.items()},
                    "overall_score": report.overall_score,
                    "generation_metadata": report.generation_metadata,
                    "created_at": report.created_at.isoformat()
                }
                
                audit_ctx.set_output_data(report_dict)
                
                # Save compiled report
                report_file = temp_output_dir / f"compiled_{report_type.value}_report.json"
                with open(report_file, 'w') as f:
                    json.dump(report_dict, f, indent=2)
                
                assert report_file.exists()
        
        # Validate all expected artifacts exist
        expected_files = [
            "meso.json",
            "compiled_macro_report.json", 
            "compiled_meso_report.json"
        ]
        
        for filename in expected_files:
            assert (temp_output_dir / filename).exists(), f"Missing expected file: {filename}"
        
        # Save and validate audit log
        audit_file = audit_logger.save_audit_file()
        assert Path(audit_file).exists()
        
        # Validate audit log contains complete execution trace
        audit_data = audit_logger.serialize_audit_data()
        assert audit_data["execution_id"] == execution_id
        assert len(audit_data["events"]) >= 3  # meso + 2 reports
        
        # Check that all components were audited
        component_names = [event["component_code"] for event in audit_data["events"]]
        assert "meso_aggregator" in component_names
        assert "report_compiler_macro" in component_names
        assert "report_compiler_meso" in component_names
    
    def test_deterministic_content_generation(self, sample_cluster_data: Dict[str, Any]):
        """Test that multiple runs produce deterministic content."""
        
        # Run meso aggregation multiple times
        results = []
        for run_id in range(3):
            result = meso_process(sample_cluster_data.copy())
            results.append(result)
        
        # Validate that core numerical results are identical
        for i in range(1, len(results)):
            # Compare divergence statistics
            stats1 = results[0]["meso_summary"]["divergence_stats"] 
            stats2 = results[i]["meso_summary"]["divergence_stats"]
            
            assert stats1["question_count"] == stats2["question_count"]
            
            # Coverage matrix should be identical
            coverage1 = results[0]["coverage_matrix"]
            coverage2 = results[i]["coverage_matrix"]
            
            for component in coverage1:
                assert coverage1[component]["coverage_percentage"] == coverage2[component]["coverage_percentage"]
                assert coverage1[component]["total_evaluations"] == coverage2[component]["total_evaluations"]
        
        # Test report compilation determinism
        report_data = ReportData(
            plan_name="Test Plan",
            analysis_results=results[0],
            evidence_responses=[],
            scoring_outputs={},
            normative_references=[]
        )
        
        compiler = ReportCompiler()
        reports = []
        
        for run_id in range(3):
            report = compiler.compile_report(report_data, ReportType.MESO)
            reports.append(report)
        
        # Compare report content (excluding timestamps and UUIDs)
        for i in range(1, len(reports)):
            # Overall scores should be identical
            assert reports[0].overall_score == reports[i].overall_score
            
            # Section content should be identical
            for section in reports[0].sections:
                assert reports[0].sections[section] == reports[i].sections[section]
    
    def test_audit_logging_completeness(self, sample_cluster_data: Dict[str, Any],
                                      temp_output_dir: Path, audit_logger: AuditLogger):
        """Test that audit logging produces complete execution traces."""
        
        execution_id = "audit_completeness_test"
        audit_logger.start_execution(execution_id)
        
        # Execute with comprehensive audit logging
        with audit_logger.audit_component_execution("meso_aggregator", sample_cluster_data) as ctx:
            result = meso_process(sample_cluster_data)
            ctx.set_output_data({"summary_keys": list(result.keys())})
            ctx.add_performance_metric("processing_time", 0.5)
            ctx.add_validation_result("schema_valid", True)
        
        with audit_logger.audit_component_execution("report_compiler", {"type": "macro"}) as ctx:
            # Mock report compilation for audit testing
            mock_report = {"status": "completed", "sections": 4}
            ctx.set_output_data(mock_report)
            ctx.add_performance_metric("compilation_time", 1.2)
            ctx.add_validation_result("content_valid", True)
        
        # Validate audit completeness
        audit_data = audit_logger.serialize_audit_data()
        
        # Check execution metadata
        assert audit_data["execution_id"] == execution_id
        assert "start_time" in audit_data
        assert "events" in audit_data
        
        # Check event completeness
        events = audit_data["events"]
        assert len(events) >= 2
        
        # Validate meso_aggregator event
        meso_events = [e for e in events if e["component_code"] == "meso_aggregator"]
        assert len(meso_events) >= 2  # start and end
        
        start_event = next((e for e in meso_events if e["event_type"] == "component_start"), None)
        end_event = next((e for e in meso_events if e["event_type"] == "component_end"), None)
        
        assert start_event is not None
        assert end_event is not None
        assert "performance_metrics" in end_event
        assert end_event["performance_metrics"]["processing_time"] == 0.5
        
        # Validate report_compiler event
        report_events = [e for e in events if e["component_code"] == "report_compiler"]
        assert len(report_events) >= 2
        
        report_end = next((e for e in report_events if e["event_type"] == "component_end"), None)
        assert report_end is not None
        assert report_end["performance_metrics"]["compilation_time"] == 1.2
    
    def test_status_reporting_accuracy(self, sample_cluster_data: Dict[str, Any],
                                     temp_output_dir: Path, audit_logger: AuditLogger):
        """Test status reporting for success/partial/failure states."""
        
        execution_id = "status_reporting_test"
        audit_logger.start_execution(execution_id)
        
        # Test success state
        with audit_logger.audit_component_execution("meso_aggregator", sample_cluster_data) as ctx:
            try:
                result = meso_process(sample_cluster_data)
                ctx.set_output_data(result)
                status = "success"
            except Exception as e:
                ctx.log_error(e)
                status = "failure"
        
        # Test partial success state (simulated)
        partial_data = sample_cluster_data.copy()
        partial_data["cluster_audit"]["micro"].pop("C4")  # Remove one cluster
        
        with audit_logger.audit_component_execution("meso_aggregator_partial", partial_data) as ctx:
            try:
                result = meso_process(partial_data)
                ctx.set_output_data(result)
                # Check if all clusters were processed
                clusters_processed = len(result["meso_summary"]["cluster_participation"])
                expected_clusters = 4
                status = "partial" if clusters_processed < expected_clusters else "success"
                ctx.add_context_data("status", status)
            except Exception as e:
                ctx.log_error(e)
                status = "failure"
        
        # Test failure state (simulated)
        with audit_logger.audit_component_execution("meso_aggregator_failure", {}) as ctx:
            try:
                result = meso_process({})  # Empty data should cause issues
                status = "success"  
            except Exception as e:
                ctx.log_error(e)
                status = "failure"
        
        # Validate status reporting in audit
        audit_data = audit_logger.serialize_audit_data()
        events = audit_data["events"]
        
        # Check success event
        success_events = [e for e in events if e["component_code"] == "meso_aggregator" 
                         and e["event_type"] == "component_end"]
        assert len(success_events) >= 1
        assert success_events[0]["error_details"] is None
        
        # Check partial success event
        partial_events = [e for e in events if e["component_code"] == "meso_aggregator_partial"]
        if partial_events:
            partial_end = next((e for e in partial_events if e["event_type"] == "component_end"), None)
            if partial_end:
                assert "status" in partial_end.get("context_data", {})
        
        # Check failure event
        failure_events = [e for e in events if e["component_code"] == "meso_aggregator_failure" 
                         and e["event_type"] == "component_error"]
        assert len(failure_events) >= 1
        assert failure_events[0]["error_details"] is not None
    
    def test_schema_compliance_validation(self, sample_cluster_data: Dict[str, Any]):
        """Test that generated artifacts comply with expected schemas."""
        
        # Execute meso aggregation
        result = meso_process(sample_cluster_data)
        
        # Define expected meso.json schema
        expected_meso_schema = {
            "meso_summary": {
                "items": dict,
                "divergence_stats": dict, 
                "cluster_participation": dict,
                "component_coverage_summary": dict
            },
            "coverage_matrix": dict
        }
        
        # Validate schema compliance
        self._validate_schema_recursive(result, expected_meso_schema)
        
        # Validate specific field types in meso_summary
        meso_summary = result["meso_summary"]
        assert isinstance(meso_summary["divergence_stats"].get("question_count", 0), int)
        
        # Validate coverage matrix structure
        coverage_matrix = result["coverage_matrix"]
        for component, data in coverage_matrix.items():
            assert isinstance(data["coverage_percentage"], float)
            assert 0.0 <= data["coverage_percentage"] <= 100.0
            assert isinstance(data["total_evaluations"], int)
            assert data["total_evaluations"] >= 0
            assert isinstance(data["clusters_evaluating"], list)
            assert isinstance(data["questions_addressing"], list)
    
    def _validate_schema_recursive(self, data: Any, schema: Dict[str, Any]):
        """Recursively validate data against schema."""
        for key, expected_type in schema.items():
            assert key in data, f"Missing required key: {key}"
            
            if isinstance(expected_type, dict):
                assert isinstance(data[key], dict), f"Expected dict for {key}, got {type(data[key])}"
                if expected_type:  # Not empty dict
                    self._validate_schema_recursive(data[key], expected_type)
            elif isinstance(expected_type, type):
                assert isinstance(data[key], expected_type), f"Expected {expected_type} for {key}, got {type(data[key])}"
    
    def test_error_handling_and_recovery(self, temp_output_dir: Path, audit_logger: AuditLogger):
        """Test error handling and recovery mechanisms."""
        
        execution_id = "error_handling_test"
        audit_logger.start_execution(execution_id)
        
        # Test with invalid input data
        invalid_data = {"invalid": "structure"}
        
        with audit_logger.audit_component_execution("meso_aggregator_error", invalid_data) as ctx:
            try:
                result = meso_process(invalid_data)
                # Should not reach here with invalid data
                assert False, "Expected exception with invalid data"
            except Exception as e:
                ctx.log_error(e)
                # Verify error was logged properly
                audit_data = audit_logger.serialize_audit_data()
                error_events = [ev for ev in audit_data["events"] 
                              if ev["event_type"] == "component_error"]
                assert len(error_events) >= 1
                assert error_events[0]["error_details"]["error_type"] is not None
                assert len(error_events[0]["error_details"]["stack_trace"]) > 0
    
    def test_concurrent_execution_safety(self, sample_cluster_data: Dict[str, Any]):
        """Test that components can handle concurrent execution safely."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def run_meso_aggregation(data: Dict[str, Any], run_id: int):
            try:
                result = meso_process(data.copy())
                results_queue.put((run_id, result))
            except Exception as e:
                errors_queue.put((run_id, str(e)))
        
        # Run multiple concurrent executions
        threads = []
        num_concurrent = 3
        
        for i in range(num_concurrent):
            thread = threading.Thread(
                target=run_meso_aggregation,
                args=(sample_cluster_data, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Validate results
        assert errors_queue.empty(), f"Concurrent execution errors: {list(errors_queue.queue)}"
        assert results_queue.qsize() == num_concurrent, f"Expected {num_concurrent} results, got {results_queue.qsize()}"
        
        # Check that all results are valid
        results = []
        while not results_queue.empty():
            run_id, result = results_queue.get()
            results.append(result)
            
            # Validate basic structure
            assert "meso_summary" in result
            assert "coverage_matrix" in result
        
        # Results should be identical for same input
        for i in range(1, len(results)):
            assert results[0]["meso_summary"]["divergence_stats"]["question_count"] == \
                   results[i]["meso_summary"]["divergence_stats"]["question_count"]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])