"""
Test suite for the evidence processing system.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from evidence_processor import (
    Citation,
    CitationFormatter,
    ConfidenceLevel,
    EvidenceChunk,
    EvidenceProcessor,
    EvidenceScoringSystem,
    EvidenceType,
    ScoringMetrics,
    SourceMetadata,
    StructuredEvidence,
)


class TestEvidenceChunk(unittest.TestCase):
    """Test cases for EvidenceChunk class."""

    def test_evidence_chunk_creation(self):
        chunk = EvidenceChunk(
            chunk_id="test_001",
            text="This is a test evidence chunk.",
            context_before="Before context",
            context_after="After context",
        )

        self.assertEqual(chunk.chunk_id, "test_001")
        self.assertEqual(chunk.text, "This is a test evidence chunk.")
        self.assertEqual(chunk.context_before, "Before context")
        self.assertEqual(chunk.context_after, "After context")


class TestSourceMetadata(unittest.TestCase):
    """Test cases for SourceMetadata class."""

    def test_source_metadata_creation(self):
        metadata = SourceMetadata(
            document_id="doc_001",
            title="Test Document",
            author="Test Author",
            publication_date=datetime(2023, 1, 1),
            page_number=10,
        )

        self.assertEqual(metadata.document_id, "doc_001")
        self.assertEqual(metadata.title, "Test Document")
        self.assertEqual(metadata.author, "Test Author")
        self.assertEqual(metadata.page_number, 10)


class TestCitationFormatter(unittest.TestCase):
    """Test cases for CitationFormatter class."""

    def setUp(self):
        self.formatter = CitationFormatter()
        self.sample_metadata = SourceMetadata(
            document_id="doc_001",
            title="Sample Document",
            author="John Doe",
            publication_date=datetime(2023, 5, 15),
            page_number=25,
            section_header="Introduction",
        )

    def test_create_citation(self):
        citation = self.formatter.create_citation(self.sample_metadata)

        self.assertIsInstance(citation, Citation)
        self.assertTrue(citation.citation_id.startswith("cite_"))
        self.assertIn("John Doe", citation.formatted_reference)
        self.assertIn("Sample Document", citation.formatted_reference)

    def test_apa_format(self):
        citation = self.formatter.create_citation(self.sample_metadata)
        apa = citation.to_apa_format()

        self.assertIn("John Doe", apa)
        self.assertIn("(2023)", apa)
        self.assertIn("Sample Document", apa)

    def test_mla_format(self):
        citation = self.formatter.create_citation(self.sample_metadata)
        mla = citation.to_mla_format()

        self.assertIn("John Doe", mla)
        self.assertIn("Sample Document", mla)
        self.assertIn("2023", mla)

    def test_inline_citation(self):
        citation = self.formatter.create_citation(self.sample_metadata)

        self.assertEqual(citation.inline_citation, "(Doe, 2023, p. 25)")

    def test_short_reference(self):
        citation = self.formatter.create_citation(self.sample_metadata)

        self.assertEqual(citation.short_reference, "Doe 2023, p. 25")


class TestEvidenceScoringSystem(unittest.TestCase):
    """Test cases for EvidenceScoringSystem class."""

    def setUp(self):
        self.scoring_system = EvidenceScoringSystem()
        self.sample_chunk = EvidenceChunk(
            chunk_id="test_001",
            text="Machine learning accuracy has improved significantly with recent advances.",
            context_before="In recent studies",
            context_after="compared to traditional methods",
        )
        self.sample_metadata = SourceMetadata(
            document_id="doc_001",
            title="AI Research Journal",
            author="Dr. Jane Smith",
            publication_date=datetime(2023, 1, 1),
            document_type="academic_journal",
            doi="10.1234/test",
        )

    def test_score_evidence(self):
        scoring = self.scoring_system.score_evidence(
            self.sample_chunk, self.sample_metadata, "q_001", "accuracy"
        )

        self.assertIsInstance(scoring, ScoringMetrics)
        self.assertGreaterEqual(scoring.overall_score, 0.0)
        self.assertLessEqual(scoring.overall_score, 1.0)
        self.assertIn(
            scoring.confidence_level,
            [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW],
        )
        
        # Test new scoring metrics
        self.assertGreaterEqual(scoring.relevance_score, 0.0)
        self.assertLessEqual(scoring.relevance_score, 1.0)
        self.assertGreaterEqual(scoring.credibility_score, 0.0)
        self.assertLessEqual(scoring.credibility_score, 1.0)
        self.assertGreaterEqual(scoring.recency_score, 0.0)
        self.assertLessEqual(scoring.recency_score, 1.0)
        self.assertGreaterEqual(scoring.authority_score, 0.0)
        self.assertLessEqual(scoring.authority_score, 1.0)
        
        # Test quality tag
        self.assertIn(scoring.quality_tag, ["high", "medium", "low"])
        
        # Test that overall score follows weighted formula
        expected_score = (
            scoring.relevance_score * 0.4 + 
            scoring.credibility_score * 0.3 + 
            scoring.recency_score * 0.2 + 
            scoring.authority_score * 0.1
        )
        self.assertAlmostEqual(scoring.overall_score, expected_score, places=5)

    def test_relevance_score_calculation(self):
        score = self.scoring_system._calculate_relevance_score(
            "This text mentions accuracy metrics", "q_001", "accuracy"
        )

        self.assertGreater(score, 0.5)  # Should be higher due to keyword match

    def test_credibility_score_calculation(self):
        score = self.scoring_system._calculate_credibility_score(self.sample_metadata)

        self.assertGreater(score, 0.5)  # Should be higher due to DOI and author

    def test_recency_score_calculation(self):
        recent_date = datetime(2023, 12, 1)
        old_date = datetime(2010, 1, 1)

        recent_score = self.scoring_system._calculate_recency_score(recent_date)
        old_score = self.scoring_system._calculate_recency_score(old_date)

        self.assertGreater(recent_score, old_score)

    def test_classify_evidence(self):
        labels = self.scoring_system._classify_evidence(
            "Statistics show 95% accuracy in this study", "accuracy"
        )

        self.assertIn("quantitative", labels)
        self.assertIn("accuracy", labels)


class TestEvidenceProcessor(unittest.TestCase):
    """Test cases for EvidenceProcessor class."""

    def setUp(self):
        self.processor = EvidenceProcessor()
        self.sample_chunks = [
            EvidenceChunk(
                chunk_id="chunk_001",
                text="AI systems demonstrate high accuracy in classification tasks.",
                context_before="Recent evaluations show that",
                context_after="across multiple datasets.",
            ),
            EvidenceChunk(
                chunk_id="chunk_002",
                text="Machine learning models require extensive validation.",
                context_before="It is important to note that",
                context_after="before deployment in production.",
            ),
        ]

        self.sample_metadata = [
            SourceMetadata(
                document_id="doc_001",
                title="AI Classification Systems",
                author="Dr. Alice Johnson",
                publication_date=datetime(2023, 3, 15),
                page_number=12,
            ),
            SourceMetadata(
                document_id="doc_002",
                title="ML Validation Techniques",
                author="Prof. Bob Smith",
                publication_date=datetime(2023, 4, 20),
                page_number=8,
            ),
        ]

    def test_process_evidence_chunks(self):
        evidence_list = self.processor.process_evidence_chunks(
            chunks=self.sample_chunks,
            metadata_list=self.sample_metadata,
            question_id="q_test",
            dimension="accuracy",
            evidence_type=EvidenceType.DIRECT_QUOTE,
        )

        self.assertEqual(len(evidence_list), 2)

        for evidence in evidence_list:
            self.assertIsInstance(evidence, StructuredEvidence)
            self.assertEqual(evidence.question_id, "q_test")
            self.assertEqual(evidence.dimension, "accuracy")
            self.assertEqual(evidence.evidence_type, EvidenceType.DIRECT_QUOTE)
            self.assertTrue(evidence.evidence_id.startswith("ev_"))
            self.assertIsInstance(evidence.citation, Citation)
            self.assertIsInstance(evidence.scoring, ScoringMetrics)
            self.assertGreater(len(evidence.audit_trail), 0)

    def test_mismatched_chunks_metadata_raises_error(self):
        with self.assertRaises(ValueError):
            self.processor.process_evidence_chunks(
                chunks=self.sample_chunks,
                metadata_list=[
                    self.sample_metadata[0]
                ],  # Only one metadata for two chunks
                question_id="q_test",
                dimension="accuracy",
            )

    def test_build_context_text(self):
        chunk = self.sample_chunks[0]
        context = self.processor._build_context_text(chunk)

        self.assertIn("Recent evaluations show that", context)
        self.assertIn(
            "**AI systems demonstrate high accuracy in classification tasks.**", context
        )
        self.assertIn("across multiple datasets.", context)

    def test_aggregate_evidence_by_dimension(self):
        evidence_list = self.processor.process_evidence_chunks(
            chunks=self.sample_chunks,
            metadata_list=self.sample_metadata,
            question_id="q_test",
            dimension="accuracy",
        )

        # Add evidence with different dimension
        different_evidence = self.processor.process_evidence_chunks(
            chunks=[self.sample_chunks[0]],
            metadata_list=[self.sample_metadata[0]],
            question_id="q_test",
            dimension="reliability",
        )

        all_evidence = evidence_list + different_evidence
        aggregated = self.processor.aggregate_evidence_by_dimension(all_evidence)

        self.assertEqual(len(aggregated["accuracy"]), 2)
        self.assertEqual(len(aggregated["reliability"]), 1)

    def test_generate_evidence_summary(self):
        evidence_list = self.processor.process_evidence_chunks(
            chunks=self.sample_chunks,
            metadata_list=self.sample_metadata,
            question_id="q_test",
            dimension="accuracy",
        )

        summary = self.processor.generate_evidence_summary(evidence_list)

        self.assertEqual(summary["total_evidence"], 2)
        self.assertIn("accuracy", summary["dimensions_covered"])
        self.assertIn("direct_quote", summary["evidence_types"])
        self.assertGreaterEqual(summary["average_overall_score"], 0.0)
        self.assertLessEqual(summary["average_overall_score"], 1.0)
        self.assertEqual(summary["unique_sources"], 2)


class TestStructuredEvidence(unittest.TestCase):
    """Test cases for StructuredEvidence class."""

    def setUp(self):
        self.chunk = EvidenceChunk(chunk_id="chunk_test", text="Test evidence text")

        self.metadata = SourceMetadata(document_id="doc_test", title="Test Document")

        self.citation = Citation(
            citation_id="cite_test",
            formatted_reference="Test Reference",
            short_reference="Test 2023",
            inline_citation="(Test, 2023)",
            metadata=self.metadata,
        )

        self.scoring = ScoringMetrics(
            overall_score=0.75, 
            quality_tag="medium",
            confidence_level=ConfidenceLevel.HIGH
        )

        self.evidence = StructuredEvidence(
            evidence_id="ev_test",
            question_id="q_test",
            dimension="test_dimension",
            evidence_type=EvidenceType.DIRECT_QUOTE,
            chunk=self.chunk,
            citation=self.citation,
            scoring=self.scoring,
            exact_text="Test evidence text",
            context_text="Context for test evidence text",
        )

    def test_add_audit_entry(self):
        initial_count = len(self.evidence.audit_trail)

        self.evidence.add_audit_entry("test_action", {"detail": "test_detail"})

        self.assertEqual(len(self.evidence.audit_trail), initial_count + 1)

        latest_entry = self.evidence.audit_trail[-1]
        self.assertEqual(latest_entry["action"], "test_action")
        self.assertEqual(latest_entry["details"]["detail"], "test_detail")
        self.assertIn("timestamp", latest_entry)

    def test_get_traceability_path(self):
        path = self.evidence.get_traceability_path()

        expected_path = {
            "document_id": "doc_test",
            "chunk_id": "chunk_test",
            "evidence_id": "ev_test",
            "question_id": "q_test",
            "dimension": "test_dimension",
        }

        self.assertEqual(path, expected_path)


class TestProcessMethod(unittest.TestCase):
    """Test cases for the new standardized process() method."""
    
    def setUp(self):
        self.processor = EvidenceProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_process_raw_evidence_basic(self):
        """Test basic processing of raw evidence candidates."""
        raw_evidence = [
            {
                "text": "The study shows 95% accuracy in classification tasks.",
                "question_id": "q_001",
                "dimension": "accuracy",
                "evidence_type": "statistical",
                "document_id": "doc_001",
                "title": "ML Research Paper",
                "author": "Dr. Smith",
                "publication_date": "2023-01-15"
            }
        ]
        
        result = self.processor.process(raw_evidence, output_dir=self.temp_dir)
        
        # Check result structure
        self.assertIn("structured_evidence", result)
        self.assertIn("processing_metadata", result)
        self.assertIn("summary", result)
        
        # Check structured evidence
        evidence_list = result["structured_evidence"]
        self.assertEqual(len(evidence_list), 1)
        
        evidence = evidence_list[0]
        self.assertTrue(evidence.evidence_id.startswith("ev_hash_"))
        self.assertEqual(evidence.exact_text, "The study shows 95% accuracy in classification tasks.")
        self.assertEqual(evidence.question_id, "q_001")
        self.assertEqual(evidence.dimension, "accuracy")
    
    def test_process_hash_based_ids(self):
        """Test that deterministic hash-based IDs are generated."""
        raw_evidence = [
            {
                "text": "Test evidence",
                "question_id": "q_001",
                "dimension": "test"
            }
        ]
        
        # Process the same evidence twice
        result1 = self.processor.process(raw_evidence, output_dir=self.temp_dir)
        result2 = self.processor.process(raw_evidence, output_dir=self.temp_dir)
        
        # IDs should be the same (deterministic)
        id1 = result1["structured_evidence"][0].evidence_id
        id2 = result2["structured_evidence"][0].evidence_id
        self.assertEqual(id1, id2)
    
    def test_process_with_validation_hooks(self):
        """Test processing with validation hooks."""
        def custom_validator(evidence, raw_evidence):
            if "accuracy" in evidence.exact_text.lower():
                return {"validation": "passed", "reason": "contains accuracy metric"}
            return {"validation": "failed", "reason": "no accuracy metric"}
        
        def length_validator(evidence, raw_evidence):
            return {"length": len(evidence.exact_text), "valid_length": len(evidence.exact_text) > 10}
        
        validation_hooks = {
            "accuracy_check": custom_validator,
            "length_check": length_validator
        }
        
        raw_evidence = [
            {
                "text": "The study shows 95% accuracy in classification tasks.",
                "question_id": "q_001",
                "dimension": "accuracy"
            }
        ]
        
        result = self.processor.process(raw_evidence, validation_hooks=validation_hooks, output_dir=self.temp_dir)
        
        # Check hooks were applied
        evidence = result["structured_evidence"][0]
        audit_entries = [entry for entry in evidence.audit_trail if "validation_hook" in entry["action"]]
        self.assertEqual(len(audit_entries), 2)
        
        # Check metadata
        metadata = result["processing_metadata"]
        self.assertIn("accuracy_check", metadata["validation_hooks_applied"])
        self.assertIn("length_check", metadata["validation_hooks_applied"])
    
    def test_process_output_file_creation(self):
        """Test that output files are created in canonical_flow/analysis/."""
        raw_evidence = [{"text": "Test evidence", "question_id": "q_001", "dimension": "test"}]
        
        result = self.processor.process(raw_evidence, output_dir=self.temp_dir)
        
        # Check output file was created
        output_file = result["processing_metadata"]["output_file"]
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(output_file.endswith("_evidence.json"))
        
        # Check file content
        with open(output_file, 'r') as f:
            file_data = json.load(f)
        
        self.assertIn("structured_evidence", file_data)
        self.assertIn("processing_metadata", file_data)
    
    def test_process_error_handling(self):
        """Test error handling in process method."""
        # Invalid raw evidence that should cause errors
        raw_evidence = [
            {"text": "Valid evidence", "question_id": "q_001", "dimension": "test"},
            {"invalid": "structure"}  # Missing required fields
        ]
        
        result = self.processor.process(raw_evidence, output_dir=self.temp_dir)
        
        # The process method now handles missing fields by providing defaults,
        # so both evidence items should be processed successfully
        self.assertEqual(len(result["structured_evidence"]), 2)
        
        # Check that both evidence items have valid structure
        for evidence in result["structured_evidence"]:
            self.assertTrue(hasattr(evidence, 'evidence_id'))
            self.assertTrue(hasattr(evidence, 'scoring'))
            self.assertTrue(hasattr(evidence.scoring, 'quality_tag'))
            self.assertIn(evidence.scoring.quality_tag, ['high', 'medium', 'low'])
    
    def test_process_default_output_dir(self):
        """Test that default output directory is used."""
        raw_evidence = [{"text": "Test evidence", "question_id": "q_001", "dimension": "test"}]
        
        # Create canonical_flow/analysis directory for test
        os.makedirs("canonical_flow/analysis", exist_ok=True)
        
        result = self.processor.process(raw_evidence)
        
        # Check default output directory
        metadata = result["processing_metadata"]
        self.assertEqual(metadata["output_directory"], "canonical_flow/analysis")
    
    def test_process_field_schema_consistency(self):
        """Test that structured evidence has consistent field schemas."""
        raw_evidence = [
            {
                "text": "Evidence 1",
                "question_id": "q_001",
                "dimension": "test",
                "context_before": "Before 1",
                "context_after": "After 1",
                "supporting_snippets": ["support1", "support2"]
            },
            {
                "text": "Evidence 2", 
                "question_id": "q_002",
                "dimension": "test"
                # Missing optional fields
            }
        ]
        
        result = self.processor.process(raw_evidence, output_dir=self.temp_dir)
        evidence_list = result["structured_evidence"]
        
        # Both evidence objects should have same schema
        for evidence in evidence_list:
            # Required fields
            self.assertTrue(hasattr(evidence, 'evidence_id'))
            self.assertTrue(hasattr(evidence, 'question_id'))
            self.assertTrue(hasattr(evidence, 'dimension'))
            self.assertTrue(hasattr(evidence, 'exact_text'))
            self.assertTrue(hasattr(evidence, 'context_text'))
            self.assertTrue(hasattr(evidence, 'chunk'))
            self.assertTrue(hasattr(evidence, 'citation'))
            self.assertTrue(hasattr(evidence, 'scoring'))
            self.assertTrue(hasattr(evidence, 'audit_trail'))
            
            # Optional fields should exist even if empty
            self.assertTrue(hasattr(evidence, 'supporting_snippets'))
            self.assertTrue(hasattr(evidence, 'contradicting_snippets'))
    
    def test_process_summary_generation(self):
        """Test that process summary is correctly generated."""
        raw_evidence = [
            {"text": "Evidence 1", "question_id": "q_001", "dimension": "accuracy"},
            {"text": "Evidence 2", "question_id": "q_001", "dimension": "reliability"},
            {"text": "Evidence 3", "question_id": "q_002", "dimension": "accuracy"}
        ]
        
        result = self.processor.process(raw_evidence, output_dir=self.temp_dir)
        summary = result["summary"]
        
        self.assertEqual(summary["total_processed"], 3)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["unique_dimensions"], 2)  # accuracy, reliability
        self.assertEqual(summary["unique_questions"], 2)   # q_001, q_002
        self.assertIn("average_score", summary)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete evidence processing pipeline."""

    def test_end_to_end_processing(self):
        """Test the complete pipeline from chunks to structured evidence."""
        processor = EvidenceProcessor()

        # Create test data
        chunks = [
            EvidenceChunk(
                chunk_id="integration_001",
                text="Statistical analysis reveals 92% effectiveness rate.",
                context_before="The comprehensive study conducted over 12 months",
                context_after="when compared to existing methodologies.",
            )
        ]

        metadata_list = [
            SourceMetadata(
                document_id="integration_doc_001",
                title="Comprehensive Statistical Analysis of Methodologies",
                author="Dr. Research Team",
                publication_date=datetime(2023, 8, 1),
                page_number=145,
                section_header="Results",
                subsection_header="Effectiveness Metrics",
                document_type="academic_journal",
                doi="10.1234/integration.test",
            )
        ]

        # Process evidence
        evidence_list = processor.process_evidence_chunks(
            chunks=chunks,
            metadata_list=metadata_list,
            question_id="integration_q_001",
            dimension="effectiveness",
            evidence_type=EvidenceType.STATISTICAL,
        )

        # Verify complete processing
        self.assertEqual(len(evidence_list), 1)

        evidence = evidence_list[0]

        # Check all components are properly created
        self.assertIsInstance(evidence.citation, Citation)
        self.assertIsInstance(evidence.scoring, ScoringMetrics)
        self.assertEqual(evidence.evidence_type, EvidenceType.STATISTICAL)

        # Check citation formatting
        self.assertIn("Dr. Research Team", evidence.citation.formatted_reference)
        self.assertIn("2023", evidence.citation.formatted_reference)

        # Check scoring
        self.assertGreater(evidence.scoring.overall_score, 0)
        self.assertIn("quantitative", evidence.scoring.classification_labels)

        # Check traceability
        path = evidence.get_traceability_path()
        self.assertEqual(path["document_id"], "integration_doc_001")
        self.assertEqual(path["question_id"], "integration_q_001")

        # Check audit trail
        self.assertGreater(len(evidence.audit_trail), 0)
        self.assertEqual(evidence.audit_trail[0]["action"], "created")


if __name__ == "__main__":
    unittest.main()
