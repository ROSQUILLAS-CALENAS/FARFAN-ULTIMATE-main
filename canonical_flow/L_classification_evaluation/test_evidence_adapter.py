"""
Test suite for Evidence Adapter Module
Tests deterministic quality metric calculations and QuestionEvalInput transformations.
"""

import unittest
import json
from evidence_adapter import (
    EvidenceAdapter, 
    QuestionEvalInput, 
    EvidenceQualityMetrics,
    create_evidence_adapter
)


class TestEvidenceAdapter(unittest.TestCase):
    """Test cases for EvidenceAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adapter = create_evidence_adapter(precision=4)
        
        # Sample evidence data with various completeness scenarios
        self.sample_evidence_data = {
            "processed_evidences": [
                {
                    "evidence_id": "ev_001",
                    "processed_text": "Complete evidence with page and text",
                    "source_metadata": {
                        "page_number": 15,
                        "document_id": "doc_001"
                    },
                    "evidence_chunk": {
                        "text": "Complete evidence with page and text",
                        "raw_text": "Complete evidence with page and text"
                    }
                },
                {
                    "evidence_id": "ev_002",
                    "processed_text": "Evidence missing page number",
                    "source_metadata": {
                        "document_id": "doc_001"
                        # Missing page_number
                    },
                    "evidence_chunk": {
                        "text": "Evidence missing page number"
                    }
                },
                {
                    "evidence_id": "ev_003",
                    "processed_text": "",  # Empty evidence
                    "source_metadata": {
                        "page_number": 20,
                        "document_id": "doc_002"
                    }
                },
                {
                    "evidence_id": "ev_004",
                    "processed_text": "Evidence with page but no exact text",
                    "source_metadata": {
                        "page_number": 25,
                        "document_id": "doc_003"
                    }
                    # Missing evidence_chunk
                }
            ]
        }
        
        self.sample_responses = {
            "DE1_Q1": "Sí",
            "DE1_Q2": "Parcial",
            "DE2_Q1": "No", 
            "DE2_Q2": "NI"
        }
    
    def test_evidence_completeness_calculation(self):
        """Test evidence completeness calculation."""
        eval_inputs, quality_metrics = self.adapter.transform_evidence_to_eval_inputs(
            self.sample_evidence_data, self.sample_responses
        )
        
        # Should be 3/4 = 0.75 (3 questions with valid evidence out of 4 total)
        expected_completeness = 3.0 / 4.0
        self.assertEqual(quality_metrics.evidence_completeness, expected_completeness)
        self.assertEqual(quality_metrics.total_questions, 4)
        self.assertEqual(quality_metrics.questions_with_valid_evidence, 3)
    
    def test_page_reference_quality_calculation(self):
        """Test page reference quality calculation."""
        eval_inputs, quality_metrics = self.adapter.transform_evidence_to_eval_inputs(
            self.sample_evidence_data, self.sample_responses
        )
        
        # Only ev_001 has both page_num and exact_text, out of 4 items = 1/4 = 0.25
        expected_quality = 1.0 / 4.0
        self.assertEqual(quality_metrics.page_reference_quality, expected_quality)
        self.assertEqual(quality_metrics.total_evidence_items, 4)
        self.assertEqual(quality_metrics.evidence_items_with_complete_references, 1)
    
    def test_question_eval_input_creation(self):
        """Test QuestionEvalInput object creation."""
        eval_inputs, quality_metrics = self.adapter.transform_evidence_to_eval_inputs(
            self.sample_evidence_data, self.sample_responses
        )
        
        self.assertEqual(len(eval_inputs), 4)
        
        # Check first eval input
        first_input = eval_inputs[0]
        self.assertEqual(first_input.question_id, "DE1_Q1")
        self.assertEqual(first_input.response, "Sí")
        self.assertEqual(first_input.evidence_completeness, 0.75)
        self.assertEqual(first_input.page_reference_quality, 0.25)
        self.assertIsInstance(first_input.evidence_metadata, dict)
        self.assertIsNotNone(first_input.processing_timestamp)
    
    def test_deterministic_json_serialization(self):
        """Test deterministic JSON serialization."""
        eval_inputs, _ = self.adapter.transform_evidence_to_eval_inputs(
            self.sample_evidence_data, self.sample_responses
        )
        
        # Serialize multiple times and verify consistency
        json1 = self.adapter.serialize_eval_inputs(eval_inputs)
        json2 = self.adapter.serialize_eval_inputs(eval_inputs)
        
        # Parse and re-serialize to handle timestamp differences
        parsed1 = json.loads(json1)
        parsed2 = json.loads(json2)
        
        # Clear timestamps for comparison
        for item in parsed1:
            item.pop('processing_timestamp', None)
        for item in parsed2:
            item.pop('processing_timestamp', None)
        
        json1_clean = json.dumps(parsed1, sort_keys=True)
        json2_clean = json.dumps(parsed2, sort_keys=True)
        
        self.assertEqual(json1_clean, json2_clean)
    
    def test_empty_evidence_data(self):
        """Test handling of empty evidence data."""
        empty_data = {"processed_evidences": []}
        
        eval_inputs, quality_metrics = self.adapter.transform_evidence_to_eval_inputs(
            empty_data, self.sample_responses
        )
        
        self.assertEqual(quality_metrics.evidence_completeness, 0.0)
        self.assertEqual(quality_metrics.page_reference_quality, 0.0)
        self.assertEqual(len(eval_inputs), 4)  # Still creates inputs for all questions
    
    def test_no_questions(self):
        """Test handling of no questions."""
        eval_inputs, quality_metrics = self.adapter.transform_evidence_to_eval_inputs(
            self.sample_evidence_data, {}
        )
        
        self.assertEqual(quality_metrics.evidence_completeness, 0.0)
        self.assertEqual(quality_metrics.total_questions, 0)
        self.assertEqual(len(eval_inputs), 0)
    
    def test_alternative_evidence_structure(self):
        """Test handling of alternative evidence data structures."""
        alt_structure = {
            "evidences": [
                {
                    "id": "alt_001",
                    "text": "Alternative structure evidence",
                    "metadata": {
                        "page_num": 10
                    },
                    "exact_text": "Alternative structure evidence"
                }
            ]
        }
        
        eval_inputs, quality_metrics = self.adapter.transform_evidence_to_eval_inputs(
            alt_structure, {"Q1": "Sí"}
        )
        
        self.assertEqual(quality_metrics.evidence_completeness, 1.0)  # 1/1
        self.assertEqual(quality_metrics.page_reference_quality, 1.0)  # 1/1
    
    def test_precision_rounding(self):
        """Test precision rounding in calculations."""
        # Create adapter with different precisions
        adapter_2 = create_evidence_adapter(precision=2)
        
        # Use data that would produce non-terminating decimals
        test_data = {
            "processed_evidences": [
                {"evidence_id": "test1", "processed_text": "Evidence 1"},
                {"evidence_id": "test2", "processed_text": "Evidence 2"},
                {"evidence_id": "test3", "processed_text": "Evidence 3"}
            ]
        }
        test_responses = {
            "Q1": "Sí",
            "Q2": "Parcial", 
            "Q3": "No"
        }  # 3 questions, 3 evidence items with valid content
        
        _, quality_metrics = adapter_2.transform_evidence_to_eval_inputs(
            test_data, test_responses
        )
        
        # Evidence completeness should be 1.0 (3/3 questions with valid evidence)
        # Page reference quality should be 0.0 (0/3 complete references)
        self.assertEqual(quality_metrics.evidence_completeness, 1.0)
        self.assertEqual(quality_metrics.page_reference_quality, 0.0)
    
    def test_processing_stats(self):
        """Test processing statistics tracking."""
        initial_stats = self.adapter.get_processing_stats()
        
        self.adapter.transform_evidence_to_eval_inputs(
            self.sample_evidence_data, self.sample_responses
        )
        
        final_stats = self.adapter.get_processing_stats()
        
        self.assertEqual(final_stats["total_adaptations"], initial_stats["total_adaptations"] + 1)
        self.assertEqual(final_stats["valid_evidence_count"], initial_stats["valid_evidence_count"] + 3)
        self.assertEqual(final_stats["invalid_evidence_count"], initial_stats["invalid_evidence_count"] + 1)
        self.assertEqual(final_stats["complete_reference_count"], initial_stats["complete_reference_count"] + 1)
    
    def test_quality_metrics_to_dict(self):
        """Test EvidenceQualityMetrics dictionary conversion."""
        _, quality_metrics = self.adapter.transform_evidence_to_eval_inputs(
            self.sample_evidence_data, self.sample_responses
        )
        
        metrics_dict = quality_metrics.to_dict()
        
        required_keys = [
            "total_questions", "questions_with_valid_evidence",
            "total_evidence_items", "evidence_items_with_complete_references",
            "evidence_completeness", "page_reference_quality"
        ]
        
        for key in required_keys:
            self.assertIn(key, metrics_dict)
        
        # Test deterministic ordering
        self.assertEqual(list(metrics_dict.keys()), required_keys)
    
    def test_question_eval_input_to_dict(self):
        """Test QuestionEvalInput dictionary conversion."""
        eval_inputs, _ = self.adapter.transform_evidence_to_eval_inputs(
            self.sample_evidence_data, self.sample_responses
        )
        
        input_dict = eval_inputs[0].to_dict()
        
        required_keys = [
            "question_id", "response", "evidence_completeness",
            "page_reference_quality", "evidence_metadata", "processing_timestamp"
        ]
        
        for key in required_keys:
            self.assertIn(key, input_dict)
        
        # Test deterministic ordering
        self.assertEqual(list(input_dict.keys()), required_keys)
    
    def test_evidence_extraction_edge_cases(self):
        """Test edge cases in evidence extraction."""
        edge_cases = [
            {"evidence": {"text": "Single evidence"}},
            {"some_key": [{"text": "Evidence in arbitrary list"}]},
            {"nested": {"data": {"processed_text": "Deeply nested"}}},
            {}  # Completely empty
        ]
        
        for case_data in edge_cases:
            try:
                eval_inputs, quality_metrics = self.adapter.transform_evidence_to_eval_inputs(
                    case_data, {"Q1": "Sí"}
                )
                # Should not raise exception
                self.assertIsInstance(eval_inputs, list)
                self.assertIsInstance(quality_metrics, EvidenceQualityMetrics)
            except Exception as e:
                self.fail(f"Evidence extraction failed for case {case_data}: {e}")


class TestEvidenceQualityCalculations(unittest.TestCase):
    """Test specific quality calculation scenarios."""
    
    def test_perfect_evidence_quality(self):
        """Test scenario with perfect evidence quality."""
        perfect_data = {
            "processed_evidences": [
                {
                    "evidence_id": "perfect1",
                    "processed_text": "Perfect evidence 1",
                    "source_metadata": {"page_number": 1},
                    "evidence_chunk": {"text": "Perfect evidence 1"}
                },
                {
                    "evidence_id": "perfect2", 
                    "processed_text": "Perfect evidence 2",
                    "source_metadata": {"page_number": 2},
                    "evidence_chunk": {"text": "Perfect evidence 2"}
                }
            ]
        }
        
        responses = {"Q1": "Sí", "Q2": "Sí"}
        
        adapter = create_evidence_adapter()
        _, quality_metrics = adapter.transform_evidence_to_eval_inputs(perfect_data, responses)
        
        self.assertEqual(quality_metrics.evidence_completeness, 1.0)  # 2/2
        self.assertEqual(quality_metrics.page_reference_quality, 1.0)  # 2/2
    
    def test_zero_evidence_quality(self):
        """Test scenario with zero evidence quality."""
        zero_data = {
            "processed_evidences": [
                {
                    "evidence_id": "empty1",
                    "processed_text": "",  # Empty text
                    "source_metadata": {}  # No page number
                },
                {
                    "evidence_id": "empty2",
                    "processed_text": "",
                    "source_metadata": {}
                }
            ]
        }
        
        responses = {"Q1": "No", "Q2": "NI"}
        
        adapter = create_evidence_adapter()
        _, quality_metrics = adapter.transform_evidence_to_eval_inputs(zero_data, responses)
        
        self.assertEqual(quality_metrics.evidence_completeness, 0.0)  # 0/2
        self.assertEqual(quality_metrics.page_reference_quality, 0.0)  # 0/2


if __name__ == "__main__":
    unittest.main()