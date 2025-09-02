#!/usr/bin/env python3
"""
Unit tests for comprehensive artifact generation system
"""

import unittest
import json
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from canonical_flow.analysis.artifact_generator import (
    ArtifactGenerator, QuestionEvaluation, DimensionSummary, PointSummary,
    MesoClusterAnalysis, MacroAlignment, EvidenceReference, create_sample_data
)


class TestArtifactGenerator(unittest.TestCase):
    """Unit tests for ArtifactGenerator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ArtifactGenerator(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ArtifactGenerator initialization."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(str(self.generator.output_dir), self.temp_dir)
    
    def test_write_json_artifact(self):
        """Test JSON artifact writing."""
        data = {"test": "data", "number": 42}
        filepath = Path(self.temp_dir) / "test.json"
        
        self.generator.write_json_artifact(data, filepath)
        
        self.assertTrue(filepath.exists())
        
        # Verify content
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, data)
    
    def test_comprehensive_artifact_generation(self):
        """Test comprehensive artifact generation."""
        questions, dimensions, points, clusters, macro = create_sample_data("TEST-DOC")
        
        artifacts = self.generator.generate_comprehensive_artifacts(
            "TEST-DOC", questions, dimensions, points, clusters, macro
        )
        
        # Check all artifact types were generated
        expected_types = ["questions", "dimensions", "points", "meso", "macro"]
        self.assertEqual(set(artifacts.keys()), set(expected_types))
        
        # Check files exist
        for artifact_type, filepath in artifacts.items():
            self.assertTrue(Path(filepath).exists(), f"Missing artifact: {artifact_type}")
    
    def test_deterministic_ordering(self):
        """Test deterministic ordering of elements."""
        # Create evidence with different IDs
        evidence_refs = [
            EvidenceReference("E003", "doc", "p.5", "Third", 0.75),
            EvidenceReference("E001", "doc", "p.1", "First", 0.85), 
            EvidenceReference("E002", "doc", "p.2", "Second", 0.80)
        ]
        
        question = QuestionEvaluation(
            "TEST-Q1", "Test question?", "Sí", 1.0, 0.8, 0.9, 0.95, evidence_refs
        )
        
        question_dict = question.to_dict()
        evidence_ids = [ref["evidence_id"] for ref in question_dict["evidence_references"]]
        
        # Should be sorted by evidence_id
        self.assertEqual(evidence_ids, ["E001", "E002", "E003"])
    
    def test_validation(self):
        """Test artifact validation."""
        questions, dimensions, points, clusters, macro = create_sample_data("VALIDATE-TEST")
        
        # Generate artifacts
        self.generator.generate_comprehensive_artifacts(
            "VALIDATE-TEST", questions, dimensions, points, clusters, macro
        )
        
        # Validate artifacts
        validation = self.generator.validate_artifacts("VALIDATE-TEST")
        
        # All should be valid
        for artifact_type, is_valid in validation.items():
            self.assertTrue(is_valid, f"Invalid artifact: {artifact_type}")
    
    def test_discovery(self):
        """Test artifact discovery."""
        # Generate artifacts for multiple documents
        for doc_stem in ["DOC-A", "DOC-B"]:
            questions, dimensions, points, clusters, macro = create_sample_data(doc_stem)
            self.generator.generate_comprehensive_artifacts(
                doc_stem, questions, dimensions, points, clusters, macro
            )
        
        discovered = self.generator.discover_artifacts()
        
        # Should find both documents
        self.assertIn("DOC-A", discovered)
        self.assertIn("DOC-B", discovered)
        
        # Each should have all artifact types
        for doc_stem in ["DOC-A", "DOC-B"]:
            artifact_types = discovered[doc_stem]
            expected = ["dimensions", "macro", "meso", "points", "questions"]
            self.assertEqual(sorted(artifact_types), expected)
    
    def test_utf8_encoding(self):
        """Test UTF-8 encoding with special characters."""
        evidence = [
            EvidenceReference("E001", "doc", "p.1", "Población, atención, evaluación", 0.85)
        ]
        
        question = QuestionEvaluation(
            "UTF8-Q1", "¿Se garantiza la atención?", "Sí", 1.0, 0.8, 0.9, 0.95, evidence
        )
        
        self.generator.generate_question_artifacts("UTF8-TEST", [question])
        
        # Read back and verify UTF-8 characters preserved
        filepath = Path(self.temp_dir) / "UTF8-TEST_questions.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("atención", content)
        self.assertIn("Población", content)
    
    def test_filename_conventions(self):
        """Test filename conventions."""
        questions, dimensions, points, clusters, macro = create_sample_data("FILENAME-TEST")
        
        artifacts = self.generator.generate_comprehensive_artifacts(
            "FILENAME-TEST", questions, dimensions, points, clusters, macro
        )
        
        # Check filename patterns
        expected_suffixes = ["questions", "dimensions", "points", "meso", "macro"]
        for suffix in expected_suffixes:
            expected_filename = f"FILENAME-TEST_{suffix}.json"
            self.assertTrue(any(expected_filename in path for path in artifacts.values()),
                          f"Missing expected filename pattern: {expected_filename}")


class TestDataClasses(unittest.TestCase):
    """Test data classes and their serialization."""
    
    def test_evidence_reference_to_dict(self):
        """Test EvidenceReference to_dict method."""
        evidence = EvidenceReference("E001", "document", "p. 15", "Sample text", 0.85)
        result = evidence.to_dict()
        
        # Should be OrderedDict with specific order
        expected_keys = ["evidence_id", "source_type", "page_reference", "text_excerpt", "confidence_score"]
        self.assertEqual(list(result.keys()), expected_keys)
        
        # Check values
        self.assertEqual(result["evidence_id"], "E001")
        self.assertEqual(result["confidence_score"], 0.85)
    
    def test_question_evaluation_to_dict(self):
        """Test QuestionEvaluation to_dict method."""
        evidence = [EvidenceReference("E001", "doc", "p.1", "Text", 0.8)]
        question = QuestionEvaluation("Q1", "Question?", "Sí", 1.0, 0.8, 0.9, 0.95, evidence)
        result = question.to_dict()
        
        # Check structure
        self.assertIn("question_id", result)
        self.assertIn("evidence_references", result)
        self.assertIsInstance(result["evidence_references"], list)
        
        # Check evidence is properly serialized
        evidence_ref = result["evidence_references"][0]
        self.assertEqual(evidence_ref["evidence_id"], "E001")


if __name__ == "__main__":
    unittest.main()