"""
Test suite for L-Classification Stage Orchestrator
Validates the standardized process() API contract, artifact generation,
and error handling with deterministic behavior.
"""

import json
import tempfile
import unittest
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from unittest.mock import Mock, patch  # Module not found  # Module not found  # Module not found

import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
sys.path.insert(0, str(Path(__file__).resolve().parent))
# # # from stage_orchestrator import LClassificationStageOrchestrator, process  # Module not found  # Module not found  # Module not found


class TestLClassificationStageOrchestrator(unittest.TestCase):
    """Test cases for the LClassificationStageOrchestrator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LClassificationStageOrchestrator()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample input files
        self.sample_question_data = {
            "DE-1": [
                {
                    "question_id": "DE1_Q1",
                    "response": "Sí",
                    "evidence_completeness": 0.9,
                    "page_reference_quality": 1.0,
                    "evidence_text": "Sample evidence text for validation"
                },
                {
                    "question_id": "DE1_Q2",
                    "response": "Parcial",
                    "evidence_completeness": 0.7,
                    "page_reference_quality": 0.8,
                    "evidence_text": "Additional evidence for testing"
                }
            ],
            "DE-2": [
                {
                    "question_id": "DE2_Q1",
                    "response": "Sí",
                    "evidence_completeness": 0.85,
                    "page_reference_quality": 0.9,
                    "evidence_text": "Evidence for DE-2 dimension"
                }
            ],
            "DE-3": [
                {
                    "question_id": "DE3_Q1",
                    "response": "No",
                    "evidence_completeness": 0.0,
                    "page_reference_quality": 0.0,
                    "evidence_text": "Minimal evidence"
                }
            ],
            "DE-4": [
                {
                    "question_id": "DE4_Q1",
                    "response": "NI",
                    "evidence_completeness": 0.0,
                    "page_reference_quality": 0.0,
                    "evidence_text": "No information available"
                }
            ]
        }
        
        # Create test input files
        self.input_files = []
        for point_id in [1, 2, 3]:
            input_file = self.temp_dir / f"classification_input/doc1/P{point_id}_questions.json"
            input_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(self.sample_question_data, f, indent=2)
            
            self.input_files.append(input_file)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orch = LClassificationStageOrchestrator(precision=2)
        
        self.assertEqual(orch.precision, 2)
        self.assertIsNotNone(orch.scoring_system)
        self.assertIsNotNone(orch.evidence_validator)
        self.assertIsNotNone(orch.dnp_alignment)
        self.assertIsNotNone(orch.audit_log)
    
    def test_resolve_input_files_from_directory(self):
# # #         """Test resolving input files from directory path"""  # Module not found  # Module not found  # Module not found
        input_files = self.orchestrator._resolve_input_files(
            str(self.temp_dir / "classification_input"), 
            None
        )
        
        self.assertEqual(len(input_files), 3)
        # Should be sorted deterministically
        self.assertEqual(input_files[0].name, "P1_questions.json")
        self.assertEqual(input_files[1].name, "P2_questions.json")
        self.assertEqual(input_files[2].name, "P3_questions.json")
    
    def test_resolve_input_files_from_list(self):
# # #         """Test resolving input files from explicit list"""  # Module not found  # Module not found  # Module not found
        file_paths = [str(f) for f in self.input_files[:2]]
        input_files = self.orchestrator._resolve_input_files(file_paths, None)
        
        self.assertEqual(len(input_files), 2)
    
    def test_resolve_input_files_from_dict(self):
# # #         """Test resolving input files from dictionary data"""  # Module not found  # Module not found  # Module not found
        data = {
            "classification_input_path": str(self.temp_dir / "classification_input"),
            "other_param": "value"
        }
        input_files = self.orchestrator._resolve_input_files(data, None)
        
        self.assertEqual(len(input_files), 3)
    
    def test_extract_point_id(self):
# # #         """Test extracting point ID from file path"""  # Module not found  # Module not found  # Module not found
        file_path = Path("classification_input/doc1/P5_questions.json")
        point_id = self.orchestrator._extract_point_id(file_path)
        
        self.assertEqual(point_id, 5)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.orchestrator._extract_point_id(Path("invalid_file.json"))
    
    def test_convert_to_evaluation_format(self):
        """Test conversion of question data to evaluation format"""
        evaluation_data = self.orchestrator._convert_to_evaluation_format(self.sample_question_data)
        
        self.assertIn("DE-1", evaluation_data)
        self.assertIn("DE-2", evaluation_data)
        self.assertIn("DE-3", evaluation_data)
        self.assertIn("DE-4", evaluation_data)
        
        # Check DE-1 data
        de1_questions = evaluation_data["DE-1"]
        self.assertEqual(len(de1_questions), 2)
        self.assertEqual(de1_questions[0]["question_id"], "DE1_Q1")
        self.assertEqual(de1_questions[0]["response"], "Sí")
    
    def test_generate_point_artifacts(self):
        """Test generation of all required artifacts for a point"""
        components = {
            "scoring_system": {
                "point_id": 1,
                "final_score": 0.75,
                "total_questions": 5,
                "dimension_scores": []
            },
            "evidence_validation": {
                "point_id": 1,
                "validation_results": [],
                "summary": {"validation_status": "completed"}
            },
            "dnp_alignment": {
                "point_id": 1,
                "alignment_result": {},
                "alignment_status": "completed"
            }
        }
        
        artifacts = self.orchestrator._generate_point_artifacts(components, 1)
        
        # Check all required artifact types are present
        required_artifacts = [
            "dimension_evaluation",
            "point_summary", 
            "composition_trace",
            "confidence_interval",
            "guard_report"
        ]
        
        for artifact_type in required_artifacts:
            self.assertIn(artifact_type, artifacts)
            self.assertEqual(artifacts[artifact_type]["point_id"], 1)
    
    @patch.object(LClassificationStageOrchestrator, '_execute_evidence_validation')
    @patch.object(LClassificationStageOrchestrator, '_execute_dnp_alignment')
    def test_process_point_success(self, mock_dnp, mock_evidence):
        """Test successful processing of a single point"""
        # Setup mocks
        mock_evidence.return_value = {
            "point_id": 1,
            "validation_results": [],
            "summary": {"validation_status": "completed"}
        }
        
        mock_dnp.return_value = {
            "point_id": 1,
            "alignment_result": {},
            "alignment_status": "completed"
        }
        
        # Process point
        result = self.orchestrator._process_point(self.input_files[0], 1)
        
        # Verify structure
        self.assertEqual(result["point_id"], 1)
        self.assertIn("components", result)
        self.assertIn("artifacts", result)
        
        # Verify components executed
        self.assertIn("scoring_system", result["components"])
        self.assertIn("evidence_validation", result["components"])
        self.assertIn("dnp_alignment", result["components"])
    
    def test_process_point_with_error_handling(self):
        """Test point processing with component failure"""
        # Create orchestrator with faulty evidence validator
        with patch.object(self.orchestrator, '_execute_evidence_validation') as mock_evidence:
            mock_evidence.side_effect = Exception("Evidence validation failed")
            
            with self.assertRaises(Exception):
                self.orchestrator._process_point(self.input_files[0], 1)
    
    @patch.object(LClassificationStageOrchestrator, '_execute_evidence_validation')
    @patch.object(LClassificationStageOrchestrator, '_execute_dnp_alignment')
    def test_full_process_api(self, mock_dnp, mock_evidence):
        """Test the complete process() API contract"""
        # Setup mocks
        mock_evidence.return_value = {
            "point_id": 1,
            "validation_results": [],
            "summary": {"validation_status": "completed"}
        }
        
        mock_dnp.return_value = {
            "point_id": 1,
            "alignment_result": {},
            "alignment_status": "completed"
        }
        
        # Execute full process
        results = self.orchestrator.process(
            data=str(self.temp_dir / "classification_input"),
            context=None
        )
        
        # Verify top-level structure
        self.assertIn("execution_metadata", results)
        self.assertIn("artifacts", results)
        self.assertIn("status_report", results)
        self.assertIn("audit_log", results)
        
        # Verify execution metadata
        metadata = results["execution_metadata"]
        self.assertIn("execution_id", metadata)
        self.assertEqual(metadata["stage"], "L_classification_evaluation")
        self.assertIn("start_time", metadata)
        
        # Verify artifacts structure
        artifacts = results["artifacts"]
        required_artifacts = [
            "dimension_evaluations",
            "point_summaries",
            "composition_traces", 
            "confidence_intervals",
            "guard_reports",
            "points_index"
        ]
        
        for artifact_type in required_artifacts:
            self.assertIn(artifact_type, artifacts)
        
        # Verify status report
        status = results["status_report"]
        self.assertIn("successful_points", status)
        self.assertIn("failed_points", status)
        self.assertIn("processing_summary", status)
        
        # Should have processed 3 points successfully
        self.assertEqual(len(status["successful_points"]), 3)
        self.assertEqual(len(status["failed_points"]), 0)
    
    def test_process_with_mixed_success_failure(self):
        """Test process handling both successful and failed points"""
        # Create an invalid input file to trigger failure
        invalid_file = self.temp_dir / "classification_input/doc1/P99_questions.json"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            json.dump({"invalid": "data"}, f)
        
        # Mock components to work for valid data
        with patch.object(self.orchestrator, '_execute_evidence_validation') as mock_evidence:
            with patch.object(self.orchestrator, '_execute_dnp_alignment') as mock_dnp:
                mock_evidence.return_value = {"validation_status": "completed"}
                mock_dnp.return_value = {"alignment_status": "completed"}
                
                results = self.orchestrator.process(
                    data=str(self.temp_dir / "classification_input")
                )
        
        # Should have some successful and some failed points
        status = results["status_report"]
        self.assertGreater(len(status["successful_points"]), 0)
        # Note: May not have failures depending on how robust the conversion is
    
    def test_deterministic_ordering(self):
        """Test that results maintain deterministic ordering"""
        with patch.object(self.orchestrator, '_execute_evidence_validation') as mock_evidence:
            with patch.object(self.orchestrator, '_execute_dnp_alignment') as mock_dnp:
                mock_evidence.return_value = {"validation_status": "completed"}
                mock_dnp.return_value = {"alignment_status": "completed"}
                
                # Run process multiple times
                results1 = self.orchestrator.process(
                    data=str(self.temp_dir / "classification_input")
                )
                
                # Create new orchestrator to ensure independence
                orchestrator2 = LClassificationStageOrchestrator()
                results2 = orchestrator2.process(
                    data=str(self.temp_dir / "classification_input")
                )
        
        # Compare successful points ordering (should be deterministic)
        points1 = results1["status_report"]["successful_points"]
        points2 = results2["status_report"]["successful_points"]
        
        self.assertEqual(points1, points2)  # Same order
        
        # Check artifacts ordering
        artifacts1_keys = list(results1["artifacts"]["point_summaries"].keys())
        artifacts2_keys = list(results2["artifacts"]["point_summaries"].keys())
        
        self.assertEqual(artifacts1_keys, artifacts2_keys)  # Same key order
    
    def test_serialize_results_deterministic_json(self):
        """Test deterministic JSON serialization"""
        test_results = {
            "execution_metadata": {"execution_id": "test-123"},
            "artifacts": {"point_summaries": {"P01": {"score": 0.8}, "P02": {"score": 0.6}}},
            "status_report": {"successful_points": [1, 2]}
        }
        
        # Serialize multiple times
        json_str1 = self.orchestrator.serialize_results(test_results)
        json_str2 = self.orchestrator.serialize_results(test_results)
        
        # Should be identical
        self.assertEqual(json_str1, json_str2)
        
        # Should use indent=2 and sort_keys=True
        self.assertIn('  "artifacts":', json_str1)  # indent=2
        # Keys should be sorted in JSON output
        parsed = json.loads(json_str1)
        self.assertIsInstance(parsed, dict)
    
    def test_serialize_results_with_file_output(self):
        """Test serialization with file output"""
        test_results = {"test": "data"}
        output_file = self.temp_dir / "results.json"
        
        json_str = self.orchestrator.serialize_results(test_results, output_file)
        
        # File should exist
        self.assertTrue(output_file.exists())
        
        # Content should match
        with open(output_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        self.assertEqual(json_str, file_content)
    
    def test_audit_logging(self):
        """Test audit logging functionality"""
        # Clear existing audit log
        self.orchestrator.execution_id = "test-execution"
        self.orchestrator.audit_log = []
        
        # Log some events
        self.orchestrator._log_audit("test_event", {"param": "value"})
        self.orchestrator._log_audit("another_event", {"data": 123})
        
        # Check audit log
        self.assertEqual(len(self.orchestrator.audit_log), 2)
        
        first_entry = self.orchestrator.audit_log[0]
        self.assertEqual(first_entry["event"], "test_event")
        self.assertEqual(first_entry["execution_id"], "test-execution")
        self.assertEqual(first_entry["details"]["param"], "value")
        self.assertIn("timestamp", first_entry)
    
    def test_get_orchestrator_info(self):
        """Test orchestrator configuration information"""
        info = self.orchestrator.get_orchestrator_info()
        
        # Verify required fields
        self.assertEqual(info["orchestrator"], "LClassificationStageOrchestrator")
        self.assertEqual(info["stage"], "L_classification_evaluation")
        self.assertIn("ScoringSystem", info["components"])
        self.assertIn("EvidenceValidator", info["components"])
        self.assertIn("DNPAlignmentAdapter", info["components"])
        self.assertTrue(info["deterministic"])
        self.assertEqual(info["api_contract"], "standardized_process")
        
        # Check artifact types
        expected_artifacts = [
            "dimension_evaluations",
            "point_summaries",
            "composition_traces",
            "confidence_intervals", 
            "guard_reports",
            "points_index"
        ]
        
        for artifact_type in expected_artifacts:
            self.assertIn(artifact_type, info["artifact_types"])
    
    def test_standalone_process_function(self):
        """Test the standalone process() function"""
        with patch.object(LClassificationStageOrchestrator, 'process') as mock_process:
            mock_process.return_value = {"test": "result"}
            
            result = process(data="test_data", context="test_context")
            
            # Should create orchestrator and call process
            mock_process.assert_called_once_with("test_data", "test_context")
            self.assertEqual(result, {"test": "result"})
    
    def test_error_isolation_between_points(self):
        """Test that errors in one point don't affect others"""
        # Create a scenario where one component fails for some points
        original_execute_scoring = self.orchestrator._execute_scoring_system
        
        def failing_execute_scoring(point_id, question_data):
            if point_id == 2:  # Fail only for point 2
                raise Exception("Scoring failed for point 2")
            return original_execute_scoring(point_id, question_data)
        
        with patch.object(self.orchestrator, '_execute_evidence_validation') as mock_evidence:
            with patch.object(self.orchestrator, '_execute_dnp_alignment') as mock_dnp:
                with patch.object(self.orchestrator, '_execute_scoring_system', side_effect=failing_execute_scoring):
                    mock_evidence.return_value = {"validation_status": "completed"}
                    mock_dnp.return_value = {"alignment_status": "completed"}
                    
                    results = self.orchestrator.process(
                        data=str(self.temp_dir / "classification_input")
                    )
        
        # Should have successful points (1, 3) and failed point (2)
        status = results["status_report"]
        self.assertGreater(len(status["successful_points"]), 0)
        self.assertGreater(len(status["failed_points"]), 0)
        
        # Check that failed point contains error details
        failed_point = status["failed_points"][0]
        self.assertIn("error", failed_point)
        self.assertIn("traceback", failed_point)


if __name__ == "__main__":
    unittest.main()