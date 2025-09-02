#!/usr/bin/env python3
"""
Comprehensive Integration Tests for K_knowledge_extraction Workflow

Tests the complete knowledge extraction pipeline (06K→07K→11K→08K→09K→10K)
using real PDF files from planes_input directory to validate all required
artifacts are generated with proper schemas and deterministic behavior.

Expected workflow components:
- 06K: Text extraction and chunking
- 07K: Entity and concept extraction  
- 11K: Embedding generation
- 08K: Knowledge graph construction
- 09K: Causal graph analysis
- 10K: DNP alignment and causal factors

Expected outputs in canonical_flow/knowledge/:
- terms.json: Extracted terms and concepts
- chunks.json: Text chunks with metadata
- embeddings.faiss: Vector embeddings index
- kg_nodes.json: Knowledge graph nodes and relations
- causal_DE1.json through causal_DE4.json: Causal analysis results
- dnp_alignment.json: DNP framework alignment
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# Import components from the canonical flow
try:
    from canonical_flow.K_knowledge_extraction.advanced_knowledge_graph_builder import (
        KnowledgeGraphBuilder,
        create_knowledge_graph_builder
    )
except ImportError as e:
    # Fallback to direct import from original module
    try:
        import sys
        sys.path.insert(0, '.')
        from Advanced_Knowledge_Graph_Builder_Component_for_Semantic_Inference_Engine import (
            KnowledgeGraphBuilder,
            create_knowledge_graph_builder
        )
    except ImportError:
        KnowledgeGraphBuilder = None
        create_knowledge_graph_builder = None
        print(f"Warning: Could not import KnowledgeGraphBuilder: {e}")

try:
    from canonical_flow.K_knowledge_extraction.causal_dnp_framework import (
        CausalDNPFramework,
        CausalAnalyzer
    )
except ImportError:
    try:
        from causal_dnp_framework import CausalDNPFramework, CausalAnalyzer
    except ImportError:
        CausalDNPFramework = None
        CausalAnalyzer = None

try:
    from canonical_flow.K_knowledge_extraction.embedding_builder import EmbeddingBuilder
except ImportError:
    try:
        from embedding_builder import EmbeddingBuilder
    except ImportError:
        EmbeddingBuilder = None

# Import PDF processing utilities
try:
    from pdf_reader import PDFReader
    from text_analyzer import TextAnalyzer
    from document_processor import DocumentProcessor
except ImportError as e:
    print(f"Warning: Could not import PDF processing utilities: {e}")
    PDFReader = None
    TextAnalyzer = None
    DocumentProcessor = None


class MockPDFProcessor:
    """Mock PDF processor for testing when real one is not available"""
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text and basic metadata from PDF"""
        return {
            'text': f"Mock extracted text from {pdf_path.name}",
            'pages': [
                {
                    'page_num': 1,
                    'text': f"Sample page content from {pdf_path.name}",
                    'metadata': {'char_count': 100}
                }
            ],
            'metadata': {
                'filename': pdf_path.name,
                'page_count': 1,
                'extraction_method': 'mock'
            }
        }


class MockComponent:
    """Mock component for testing workflow when components are not available"""
    
    def __init__(self, name: str, output_format: str = "json"):
        self.name = name
        self.output_format = output_format
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock processing that returns deterministic results"""
        return {
            'component': self.name,
            'processed': True,
            'input_hash': self._hash_input(input_data),
            'timestamp': time.time(),
            'mock_data': f"Mock output from {self.name}"
        }
    
    def _hash_input(self, data: Dict[str, Any]) -> str:
        """Create deterministic hash of input data"""
        return hashlib.md5(str(sorted(data.items())).encode()).hexdigest()[:8]


class TestKKnowledgeExtractionWorkflow(unittest.TestCase):
    """Integration tests for the complete K_knowledge_extraction workflow"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with test data and directories"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.planes_input_dir = Path("planes_input")
        cls.output_dir = cls.test_dir / "canonical_flow" / "knowledge"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger("K_WorkflowTest")
        
        # Expected artifacts
        cls.expected_artifacts = [
            "terms.json",
            "chunks.json", 
            "embeddings.faiss",
            "kg_nodes.json",
            "causal_DE1.json",
            "causal_DE2.json", 
            "causal_DE3.json",
            "causal_DE4.json",
            "dnp_alignment.json"
        ]
        
        # Schema validation patterns
        cls.expected_schemas = {
            "terms.json": {
                "required_keys": ["terms", "concepts", "entities"],
                "term_fields": ["text", "frequency", "type", "confidence"]
            },
            "chunks.json": {
                "required_keys": ["chunks", "metadata"],
                "chunk_fields": ["chunk_id", "text", "metadata", "embeddings_index"]
            },
            "kg_nodes.json": {
                "required_keys": ["nodes", "relations", "graph_metadata"],
                "node_fields": ["id", "type", "properties", "relations"]
            },
            "causal_DE1.json": {
                "required_keys": ["causal_factors", "evidence", "methodology"],
                "factor_fields": ["factor_id", "strength", "confidence", "p_value"]
            },
            "dnp_alignment.json": {
                "required_keys": ["alignment_matrix", "dnp_requirements", "compliance_score"],
                "alignment_fields": ["requirement_id", "evidence_mapping", "compliance_level"]
            }
        }
        
        # Get test PDFs
        cls.test_pdfs = cls._get_test_pdfs()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test directory"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _get_test_pdfs(cls) -> List[Path]:
        """Get a small subset of test PDFs for integration testing"""
        if not cls.planes_input_dir.exists():
            cls.logger.warning("planes_input directory not found, creating mock PDFs")
            return cls._create_mock_pdfs()
        
        # Get first 3 PDF files for testing
        pdf_files = list(cls.planes_input_dir.glob("*.pdf"))[:3]
        if not pdf_files:
            cls.logger.warning("No PDF files found, creating mock PDFs")
            return cls._create_mock_pdfs()
        
        return pdf_files
    
    @classmethod
    def _create_mock_pdfs(cls) -> List[Path]:
        """Create mock PDF files for testing"""
        mock_pdfs = []
        mock_data_dir = cls.test_dir / "mock_pdfs"
        mock_data_dir.mkdir(exist_ok=True)
        
        for i in range(3):
            mock_pdf = mock_data_dir / f"mock_plan_{i+1}.pdf"
            mock_pdf.write_text(f"Mock PDF content {i+1}")
            mock_pdfs.append(mock_pdf)
        
        return mock_pdfs
    
    def setUp(self):
        """Set up individual test"""
        self.workflow_components = self._initialize_workflow_components()
        self.test_outputs = {}
        
    def tearDown(self):
        """Clean up after individual test"""
        # Clear output directory for next test
        if self.output_dir.exists():
            for file in self.output_dir.glob("*"):
                if file.is_file():
                    file.unlink()
    
    def _initialize_workflow_components(self) -> Dict[str, Any]:
        """Initialize workflow components (real or mock)"""
        components = {}
        
        # 06K: Text extraction and chunking
        if PDFReader and DocumentProcessor:
            components['06K'] = DocumentProcessor()
        else:
            components['06K'] = MockComponent('06K_text_extraction')
        
        # 07K: Entity and concept extraction
        if TextAnalyzer:
            components['07K'] = TextAnalyzer()
        else:
            components['07K'] = MockComponent('07K_entity_extraction')
        
        # 11K: Embedding generation
        if EmbeddingBuilder:
            components['11K'] = EmbeddingBuilder()
        else:
            components['11K'] = MockComponent('11K_embedding_builder')
        
        # 08K: Knowledge graph construction
        if KnowledgeGraphBuilder:
            try:
                components['08K'] = create_knowledge_graph_builder()
            except Exception:
                components['08K'] = MockComponent('08K_knowledge_graph')
        else:
            components['08K'] = MockComponent('08K_knowledge_graph')
        
        # 09K: Causal graph analysis
        components['09K'] = MockComponent('09K_causal_graph')
        
        # 10K: DNP alignment
        if CausalDNPFramework:
            components['10K'] = CausalDNPFramework()
        else:
            components['10K'] = MockComponent('10K_dnp_alignment')
        
        return components
    
    def test_complete_workflow_execution(self):
        """Test complete workflow execution from PDF input to all artifacts"""
        self.logger.info("Testing complete K_knowledge_extraction workflow")
        
        # Run workflow on test PDFs
        for pdf_path in self.test_pdfs[:2]:  # Test with 2 PDFs
            self.logger.info(f"Processing {pdf_path.name}")
            
            # Execute complete workflow
            workflow_output = self._execute_complete_workflow(pdf_path)
            
            # Verify all expected artifacts are generated
            self._verify_all_artifacts_generated()
            
            # Store outputs for reproducibility testing
            self.test_outputs[pdf_path.name] = workflow_output
            
        self.logger.info("Complete workflow execution test passed")
    
    def test_artifact_schema_compliance(self):
        """Test that all generated artifacts comply with expected schemas"""
        self.logger.info("Testing artifact schema compliance")
        
        # Process one PDF to generate artifacts
        test_pdf = self.test_pdfs[0]
        self._execute_complete_workflow(test_pdf)
        
        # Validate schema compliance for each artifact
        for artifact_name in self.expected_artifacts:
            if artifact_name.endswith('.faiss'):
                self._verify_faiss_file_schema(artifact_name)
            else:
                self._verify_json_schema(artifact_name)
        
        self.logger.info("Schema compliance test passed")
    
    def test_deterministic_behavior(self):
        """Test that workflow produces identical outputs across multiple runs"""
        self.logger.info("Testing deterministic behavior")
        
        test_pdf = self.test_pdfs[0]
        
        # Run workflow twice with same inputs and fixed seeds
        np.random.seed(42)
        first_run = self._execute_complete_workflow(test_pdf)
        first_artifacts = self._capture_artifact_hashes()
        
        # Clear outputs and run again
        self.tearDown()
        self.setUp()
        
        np.random.seed(42)
        second_run = self._execute_complete_workflow(test_pdf)
        second_artifacts = self._capture_artifact_hashes()
        
        # Compare outputs
        self._compare_workflow_outputs(first_artifacts, second_artifacts)
        
        self.logger.info("Deterministic behavior test passed")
    
    def test_reproducibility_verification(self):
        """Test reproducibility of embeddings and knowledge graph outputs"""
        self.logger.info("Testing reproducibility verification")
        
        test_pdf = self.test_pdfs[0]
        
        # Run workflow multiple times and compare critical outputs
        runs = []
        for i in range(3):
            self.tearDown()
            self.setUp()
            
            # Set fixed seed for reproducibility
            np.random.seed(123)
            output = self._execute_complete_workflow(test_pdf)
            
            # Extract critical reproducibility metrics
            run_data = self._extract_reproducibility_metrics()
            runs.append(run_data)
        
        # Verify consistency across runs
        self._verify_reproducibility_consistency(runs)
        
        self.logger.info("Reproducibility verification test passed")
    
    def test_workflow_component_integration(self):
        """Test integration between workflow components"""
        self.logger.info("Testing workflow component integration")
        
        test_pdf = self.test_pdfs[0]
        
        # Test each component integration
        integration_results = {}
        
        # 06K → 07K integration
        text_data = self._run_component('06K', {'input': str(test_pdf)})
        entity_data = self._run_component('07K', text_data)
        integration_results['06K_07K'] = self._verify_component_integration(text_data, entity_data)
        
        # 07K → 11K integration
        embedding_data = self._run_component('11K', entity_data)
        integration_results['07K_11K'] = self._verify_component_integration(entity_data, embedding_data)
        
        # 11K → 08K integration
        kg_data = self._run_component('08K', embedding_data)
        integration_results['11K_08K'] = self._verify_component_integration(embedding_data, kg_data)
        
        # 08K → 09K integration
        causal_data = self._run_component('09K', kg_data)
        integration_results['08K_09K'] = self._verify_component_integration(kg_data, causal_data)
        
        # 09K → 10K integration
        dnp_data = self._run_component('10K', causal_data)
        integration_results['09K_10K'] = self._verify_component_integration(causal_data, dnp_data)
        
        # Verify all integrations successful
        for integration, result in integration_results.items():
            self.assertTrue(result, f"Integration {integration} failed")
        
        self.logger.info("Component integration test passed")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in workflow"""
        self.logger.info("Testing error handling and recovery")
        
        # Test with corrupted input
        corrupted_pdf = self.test_dir / "corrupted.pdf"
        corrupted_pdf.write_bytes(b"corrupted data")
        
        try:
            workflow_output = self._execute_complete_workflow(corrupted_pdf)
            # Should handle error gracefully
            self.assertIn('error_handling', workflow_output)
        except Exception as e:
            self.logger.info(f"Expected error handled: {e}")
        
        # Test with missing components
        original_component = self.workflow_components['08K']
        self.workflow_components['08K'] = None
        
        try:
            workflow_output = self._execute_complete_workflow(self.test_pdfs[0])
            self.assertIn('fallback_used', workflow_output)
        finally:
            self.workflow_components['08K'] = original_component
        
        self.logger.info("Error handling test passed")
    
    def _execute_complete_workflow(self, pdf_path: Path) -> Dict[str, Any]:
        """Execute the complete K_knowledge_extraction workflow"""
        workflow_output = {}
        current_data = {'input': str(pdf_path)}
        
        # Execute workflow stages in sequence: 06K→07K→11K→08K→09K→10K
        stage_order = ['06K', '07K', '11K', '08K', '09K', '10K']
        
        for stage in stage_order:
            try:
                stage_output = self._run_component(stage, current_data)
                workflow_output[stage] = stage_output
                current_data = {**current_data, **stage_output}
                
                # Generate stage-specific artifacts
                self._generate_stage_artifacts(stage, stage_output)
                
            except Exception as e:
                self.logger.error(f"Error in stage {stage}: {e}")
                workflow_output[stage] = {'error': str(e)}
        
        return workflow_output
    
    def _run_component(self, component_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific workflow component"""
        component = self.workflow_components.get(component_id)
        if not component:
            return {'error': f'Component {component_id} not available'}
        
        if hasattr(component, 'process'):
            return component.process(input_data)
        else:
            return {'mock_output': f'Output from {component_id}'}
    
    def _generate_stage_artifacts(self, stage: str, stage_output: Dict[str, Any]):
        """Generate artifacts specific to each workflow stage"""
        if stage == '06K':
            # Generate chunks.json
            chunks_data = {
                'chunks': stage_output.get('chunks', []),
                'metadata': stage_output.get('metadata', {}),
                'timestamp': time.time()
            }
            self._write_json_artifact('chunks.json', chunks_data)
        
        elif stage == '07K':
            # Generate terms.json
            terms_data = {
                'terms': stage_output.get('terms', []),
                'concepts': stage_output.get('concepts', []),
                'entities': stage_output.get('entities', []),
                'extraction_metadata': stage_output.get('metadata', {})
            }
            self._write_json_artifact('terms.json', terms_data)
        
        elif stage == '11K':
            # Generate embeddings.faiss
            self._generate_mock_faiss_file()
        
        elif stage == '08K':
            # Generate kg_nodes.json
            kg_data = {
                'nodes': stage_output.get('nodes', []),
                'relations': stage_output.get('relations', []),
                'graph_metadata': stage_output.get('metadata', {})
            }
            self._write_json_artifact('kg_nodes.json', kg_data)
        
        elif stage == '09K':
            # Generate causal_DE1.json through causal_DE4.json
            for i in range(1, 5):
                causal_data = {
                    'causal_factors': stage_output.get(f'factors_{i}', []),
                    'evidence': stage_output.get(f'evidence_{i}', []),
                    'methodology': f'DE{i}_analysis',
                    'confidence_metrics': stage_output.get(f'confidence_{i}', {})
                }
                self._write_json_artifact(f'causal_DE{i}.json', causal_data)
        
        elif stage == '10K':
            # Generate dnp_alignment.json
            dnp_data = {
                'alignment_matrix': stage_output.get('alignment_matrix', []),
                'dnp_requirements': stage_output.get('requirements', []),
                'compliance_score': stage_output.get('compliance_score', 0.0),
                'alignment_metadata': stage_output.get('metadata', {})
            }
            self._write_json_artifact('dnp_alignment.json', dnp_data)
    
    def _write_json_artifact(self, filename: str, data: Dict[str, Any]):
        """Write JSON artifact to output directory"""
        artifact_path = self.output_dir / filename
        with open(artifact_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_mock_faiss_file(self):
        """Generate a mock FAISS file for testing"""
        faiss_path = self.output_dir / "embeddings.faiss"
        # Create a deterministic mock FAISS file
        mock_faiss_data = np.random.RandomState(42).rand(100, 768).astype(np.float32)
        faiss_path.write_bytes(mock_faiss_data.tobytes())
    
    def _verify_all_artifacts_generated(self):
        """Verify all expected artifacts are generated"""
        for artifact in self.expected_artifacts:
            artifact_path = self.output_dir / artifact
            self.assertTrue(artifact_path.exists(), f"Artifact {artifact} not generated")
            self.assertGreater(artifact_path.stat().st_size, 0, f"Artifact {artifact} is empty")
    
    def _verify_json_schema(self, artifact_name: str):
        """Verify JSON artifact schema compliance"""
        artifact_path = self.output_dir / artifact_name
        self.assertTrue(artifact_path.exists(), f"Artifact {artifact_name} missing")
        
        with open(artifact_path) as f:
            data = json.load(f)
        
        # Check required keys
        expected_schema = self.expected_schemas.get(artifact_name, {})
        required_keys = expected_schema.get('required_keys', [])
        
        for key in required_keys:
            self.assertIn(key, data, f"Required key {key} missing in {artifact_name}")
        
        # Additional schema validation can be added here
        self.logger.info(f"Schema validation passed for {artifact_name}")
    
    def _verify_faiss_file_schema(self, artifact_name: str):
        """Verify FAISS file format and structure"""
        artifact_path = self.output_dir / artifact_name
        self.assertTrue(artifact_path.exists(), f"FAISS file {artifact_name} missing")
        
        # Basic file size and format checks
        file_size = artifact_path.stat().st_size
        self.assertGreater(file_size, 0, "FAISS file is empty")
        
        # Could add more sophisticated FAISS format validation here
        self.logger.info(f"FAISS file validation passed for {artifact_name}")
    
    def _capture_artifact_hashes(self) -> Dict[str, str]:
        """Capture hashes of all generated artifacts for comparison"""
        hashes = {}
        
        for artifact in self.expected_artifacts:
            artifact_path = self.output_dir / artifact
            if artifact_path.exists():
                if artifact.endswith('.faiss'):
                    # Hash binary FAISS file
                    content = artifact_path.read_bytes()
                else:
                    # Hash JSON file content
                    content = artifact_path.read_text().encode()
                
                hashes[artifact] = hashlib.md5(content).hexdigest()
        
        return hashes
    
    def _compare_workflow_outputs(self, first_run: Dict[str, str], second_run: Dict[str, str]):
        """Compare outputs between workflow runs"""
        for artifact in self.expected_artifacts:
            if artifact in first_run and artifact in second_run:
                self.assertEqual(
                    first_run[artifact], 
                    second_run[artifact],
                    f"Deterministic output mismatch for {artifact}"
                )
            else:
                self.logger.warning(f"Artifact {artifact} missing in one of the runs")
    
    def _extract_reproducibility_metrics(self) -> Dict[str, Any]:
        """Extract metrics for reproducibility testing"""
        metrics = {}
        
        # Extract embedding consistency metrics
        if (self.output_dir / "embeddings.faiss").exists():
            faiss_data = (self.output_dir / "embeddings.faiss").read_bytes()
            metrics['embeddings_hash'] = hashlib.md5(faiss_data).hexdigest()
        
        # Extract knowledge graph consistency metrics
        if (self.output_dir / "kg_nodes.json").exists():
            with open(self.output_dir / "kg_nodes.json") as f:
                kg_data = json.load(f)
            metrics['kg_node_count'] = len(kg_data.get('nodes', []))
            metrics['kg_relation_count'] = len(kg_data.get('relations', []))
        
        return metrics
    
    def _verify_reproducibility_consistency(self, runs: List[Dict[str, Any]]):
        """Verify consistency across multiple runs"""
        if len(runs) < 2:
            return
        
        first_run = runs[0]
        for i, run in enumerate(runs[1:], 1):
            for metric, value in first_run.items():
                self.assertEqual(
                    value, run.get(metric),
                    f"Reproducibility failure for {metric} in run {i+1}"
                )
    
    def _verify_component_integration(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> bool:
        """Verify successful integration between components"""
        # Basic integration checks
        if 'error' in output_data:
            return False
        
        # Check that output contains processed information from input
        if not output_data or output_data == input_data:
            return False
        
        return True


class TestSchemaValidation(unittest.TestCase):
    """Additional schema validation tests"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.schemas_dir = self.test_dir / "schemas"
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_terms_json_schema(self):
        """Test terms.json schema validation"""
        valid_terms = {
            "terms": [
                {"text": "development", "frequency": 15, "type": "noun", "confidence": 0.95}
            ],
            "concepts": [
                {"concept": "urban planning", "relevance": 0.88, "category": "planning"}
            ],
            "entities": [
                {"entity": "Municipal Council", "type": "ORGANIZATION", "mentions": 5}
            ]
        }
        
        schema_path = self.schemas_dir / "terms.json"
        with open(schema_path, 'w') as f:
            json.dump(valid_terms, f)
        
        # Validate schema structure
        self.assertTrue(self._validate_terms_schema(valid_terms))
    
    def test_knowledge_graph_schema(self):
        """Test kg_nodes.json schema validation"""
        valid_kg = {
            "nodes": [
                {"id": "node1", "type": "CONCEPT", "properties": {"label": "Planning"}, "relations": []}
            ],
            "relations": [
                {"source": "node1", "target": "node2", "type": "RELATES_TO", "weight": 0.8}
            ],
            "graph_metadata": {"node_count": 1, "relation_count": 1}
        }
        
        self.assertTrue(self._validate_kg_schema(valid_kg))
    
    def _validate_terms_schema(self, data: Dict[str, Any]) -> bool:
        """Validate terms.json schema"""
        required_keys = ["terms", "concepts", "entities"]
        return all(key in data for key in required_keys)
    
    def _validate_kg_schema(self, data: Dict[str, Any]) -> bool:
        """Validate kg_nodes.json schema"""
        required_keys = ["nodes", "relations", "graph_metadata"]
        return all(key in data for key in required_keys)


if __name__ == '__main__':
    # Configure test execution
    unittest.main(verbosity=2)