"""
Test suite for CausalGraphConstructor module
"""

import json
import pytest
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
import tempfile
import shutil

# # # from causal_graph_constructor import (  # Module not found  # Module not found  # Module not found
    CausalGraphConstructor, 
    DimensionType,
    EvidenceReference,
    CausalRelationship
)


class TestCausalGraphConstructor:
    """Test suite for the 09K CausalGraphConstructor component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constructor = CausalGraphConstructor()
        self.temp_dir = tempfile.mkdtemp()
        self.constructor.output_directory = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization of the constructor."""
        assert self.constructor.component_name == "CausalGraphConstructor_09K"
        assert len(self.constructor.dimension_graphs) == 4
        assert all(dim in self.constructor.dimension_graphs for dim in DimensionType)
        
    def test_process_with_valid_data(self):
        """Test processing with valid evidence and knowledge graph artifacts."""
        evidence_artifacts = {
            "documents": {
                "doc1": {
                    "content": "Rights violations lead to social unrest",
                    "pages": [1, 2],
                    "metadata": {"type": "report"}
                }
            }
        }
        
        knowledge_graph_artifacts = {
            "entity_relationships": [
                {
                    "source": "rights_violations",
                    "target": "social_unrest",
                    "type": "causes",
                    "confidence": 0.8,
                    "evidence": [
                        {
                            "source_id": "doc1",
                            "page_numbers": [1],
                            "text_snippet": "Rights violations lead to social unrest",
                            "confidence": 0.9
                        }
                    ]
                }
            ]
        }
        
        results = self.constructor.process(evidence_artifacts, knowledge_graph_artifacts)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) == 4
        assert all(dim.value in results for dim in DimensionType)
        
        # Check DE-1 has content
        de1_artifact = results["DE-1"]
        assert de1_artifact.dimension == "DE-1"
        assert len(de1_artifact.nodes) >= 2
        assert len(de1_artifact.edges) >= 1
        
        # Check edge has proper evidence references
        edge = de1_artifact.edges[0]
        assert "evidence_references" in edge
        assert len(edge["evidence_references"]) > 0
        assert "page_numbers" in edge["evidence_references"][0]
        
    def test_process_with_empty_data(self):
        """Test processing with empty input data."""
        results = self.constructor.process({}, {})
        
        # Should generate sparse artifacts
        assert isinstance(results, dict)
        assert len(results) == 4
        
        # All artifacts should be sparse
        for artifact in results.values():
            assert len(artifact.nodes) == 0
            assert len(artifact.edges) == 0
            assert artifact.metadata.get("is_sparse", False) == True
            
    def test_cycle_prevention(self):
        """Test that cycles are prevented in graph construction."""
        knowledge_graph_artifacts = {
            "entity_relationships": [
                {
                    "source": "A",
                    "target": "B", 
                    "type": "causes",
                    "confidence": 0.8,
                    "evidence": [{"source_id": "doc1", "confidence": 0.9}]
                },
                {
                    "source": "B",
                    "target": "C",
                    "type": "causes", 
                    "confidence": 0.8,
                    "evidence": [{"source_id": "doc1", "confidence": 0.9}]
                },
                {
                    "source": "C",
                    "target": "A",  # Would create cycle
                    "type": "causes",
                    "confidence": 0.8,
                    "evidence": [{"source_id": "doc1", "confidence": 0.9}]
                }
            ]
        }
        
        results = self.constructor.process({}, knowledge_graph_artifacts)
        
        # Check that resulting graph is acyclic
        for artifact in results.values():
            if artifact.edges:
                assert artifact.metadata.get("is_acyclic", False) == True
                
    def test_dimension_classification(self):
        """Test that relationships are correctly classified by dimension."""
        knowledge_graph_artifacts = {
            "entity_relationships": [
                {
                    "source": "civil_rights",
                    "target": "freedom_expression",
                    "type": "enables",
                    "confidence": 0.8,
                    "evidence": [{"source_id": "doc1", "confidence": 0.9}]
                },
                {
                    "source": "poverty",
                    "target": "health_outcomes",
                    "type": "influences",
                    "confidence": 0.7,
                    "evidence": [{"source_id": "doc2", "confidence": 0.8}]
                }
            ]
        }
        
        results = self.constructor.process({}, knowledge_graph_artifacts)
        
        # DE-1 should have the rights-related relationship
        de1_artifact = results["DE-1"]
        assert len(de1_artifact.edges) > 0
        
        # DE-2 should have the social/economic relationship  
        de2_artifact = results["DE-2"]
        assert len(de2_artifact.edges) > 0
        
    def test_validity_scoring(self):
        """Test that validity scores are calculated properly."""
        evidence_refs = [
            EvidenceReference(
                source_id="doc1",
                page_numbers=[1, 2],
                text_snippet="Test evidence",
                confidence=0.9
            )
        ]
        
        relationship = CausalRelationship(
            source_node="A",
            target_node="B",
            relationship_type="causes",
            confidence=0.8,
            evidence_strength=0.85,
            evidence_references=tuple(evidence_refs)
        )
        
        validity_score = self.constructor._calculate_validity_score(relationship)
        
        assert 0.0 <= validity_score <= 1.0
        assert validity_score > 0.5  # Should be high given good inputs
        
    def test_output_files_generated(self):
        """Test that output JSON files are properly generated."""
        evidence_artifacts = {"documents": {"doc1": {"content": "test"}}}
        knowledge_graph_artifacts = {
            "entity_relationships": [
                {
                    "source": "rights_violations",
                    "target": "social_unrest",
                    "type": "causes",
                    "confidence": 0.8,
                    "evidence": [{"source_id": "doc1", "confidence": 0.9}]
                }
            ]
        }
        
        results = self.constructor.process(evidence_artifacts, knowledge_graph_artifacts)
        
        # Check that output files exist
        for dimension in DimensionType:
            filename = f"causal_{dimension.value}.json"
            filepath = self.constructor.output_directory / filename
            assert filepath.exists()
            
            # Verify file contains valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert "dimension" in data
                assert "nodes" in data
                assert "edges" in data
                assert "metadata" in data
                assert "validity_statistics" in data
                
    def test_evidence_strength_calculation(self):
        """Test evidence strength calculation with multiple sources."""
        evidence_refs = [
            EvidenceReference("doc1", [1], "snippet1", 0.8),
            EvidenceReference("doc2", [2], "snippet2", 0.9),
            EvidenceReference("doc1", [3], "snippet3", 0.7)  # Same source
        ]
        
        strength = self.constructor._calculate_evidence_strength(evidence_refs)
        
        assert 0.0 <= strength <= 1.0
        # Should include diversity bonus for multiple sources
        assert strength > 0.8
        
    def test_threshold_filtering(self):
        """Test that relationships below thresholds are filtered out."""
        # Set high thresholds
        self.constructor.min_confidence_threshold = 0.9
        self.constructor.min_evidence_strength = 0.9
        
        knowledge_graph_artifacts = {
            "entity_relationships": [
                {
                    "source": "low_confidence",
                    "target": "result",
                    "type": "causes",
                    "confidence": 0.5,  # Below threshold
                    "evidence": [{"source_id": "doc1", "confidence": 0.5}]
                }
            ]
        }
        
        results = self.constructor.process({}, knowledge_graph_artifacts)
        
        # All artifacts should be sparse due to filtering
        for artifact in results.values():
            assert len(artifact.edges) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])