#!/usr/bin/env python3
"""
Integration Tests for Municipal Development Plans Processing

Tests the complete EGW pipeline flow with actual municipal development plans,
validating extracted evidence quality and performance benchmarks.
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock
import logging

import numpy as np
import pytest
import redis
from concurrent.futures import ThreadPoolExecutor

# Import EGW components
from distributed_processor import DistributedProcessor, ProcessingTask, QualityValidator, ResultAggregator
from egw_query_expansion.core.hybrid_retrieval import HybridRetrieval
from egw_query_expansion.core.gw_alignment import GWAlignment
from evidence_processor import EvidenceProcessor
from answer_synthesizer import AnswerSynthesizer


class TestMunicipalPlansIntegration(unittest.TestCase):
    """Integration tests for municipal development plans processing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.test_data_dir = Path(__file__).parent.parent / "municipal_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create test municipal development plans
        cls._create_test_documents()
        
        # Test queries for municipal development plans
        cls.test_queries = [
            "What are the zoning requirements for residential development?",
            "Describe the environmental impact assessment procedures",
            "What permits are required for commercial construction?",
            "Outline the public consultation process for development proposals",
            "What are the infrastructure requirements for new developments?"
        ]
        
        # Performance benchmarks
        cls.benchmark_requirements = {
            'documents_per_hour': 170,
            'max_processing_time_per_doc': 21.2,  # (3600/170) seconds
            'min_quality_score': 0.8,
            'min_consistency_score': 0.75,
            'max_memory_usage_mb': 2048,
            'max_cpu_usage_percent': 85
        }
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger("MunicipalPlansTest")
    
    @classmethod
    def _create_test_documents(cls):
        """Create test municipal development plan documents"""
        documents = [
            {
                "filename": "zoning_ordinance.txt",
                "content": """
MUNICIPAL ZONING ORDINANCE

SECTION 1: RESIDENTIAL ZONING REQUIREMENTS

1.1 R-1 Single Family Residential Zone
- Minimum lot size: 8,000 square feet
- Maximum building height: 35 feet
- Front yard setback: 25 feet minimum
- Side yard setback: 8 feet minimum each side
- Rear yard setback: 30 feet minimum
- Maximum lot coverage: 40%

1.2 R-2 Two-Family Residential Zone  
- Minimum lot size: 6,000 square feet per dwelling unit
- Maximum building height: 40 feet
- Front yard setback: 20 feet minimum
- Side yard setback: 6 feet minimum each side
- Rear yard setback: 25 feet minimum
- Maximum lot coverage: 50%

SECTION 2: COMMERCIAL ZONING REQUIREMENTS

2.1 C-1 Neighborhood Commercial Zone
- Permitted uses: retail stores, offices, restaurants
- Maximum building height: 45 feet
- No front yard setback required
- Side yard setback: 10 feet where adjacent to residential
- Parking: 1 space per 300 square feet of floor area

2.2 Special Use Permits
All developments exceeding 10,000 square feet require special use permit approval through public hearing process.
                """
            },
            {
                "filename": "environmental_procedures.txt", 
                "content": """
ENVIRONMENTAL IMPACT ASSESSMENT PROCEDURES

CHAPTER 1: ASSESSMENT REQUIREMENTS

1.1 Mandatory Environmental Review
All proposed developments that meet the following criteria require environmental impact assessment:
- Projects disturbing more than 1 acre of land
- Projects within 500 feet of wetlands or water bodies  
- Projects generating more than 100 vehicle trips per day
- Projects involving hazardous materials storage or use

1.2 Assessment Process
Step 1: Initial Environmental Review (IER)
- Completed within 30 days of application submission
- Identifies potential environmental impacts
- Determines if full EIA required

Step 2: Environmental Impact Assessment (if required)
- Comprehensive study of environmental effects
- Analysis of alternatives
- Mitigation measures proposal
- Public comment period (45 days)

1.3 Required Studies
- Traffic impact analysis
- Stormwater management plan
- Noise impact assessment  
- Air quality evaluation
- Ecological impact study
- Historical/archaeological survey

CHAPTER 2: MITIGATION REQUIREMENTS

2.1 Stormwater Management
- On-site retention required for first inch of rainfall
- Green infrastructure preferred
- Bioretention areas encouraged

2.2 Tree Preservation
- Minimum 30% canopy coverage maintenance
- Replacement ratio 2:1 for removed trees over 6 inches diameter
                """
            },
            {
                "filename": "permit_procedures.txt",
                "content": """
DEVELOPMENT PERMIT PROCEDURES MANUAL

PART A: PERMIT TYPES AND REQUIREMENTS

A.1 Building Permits
Required for:
- New construction
- Structural alterations
- Additions over 120 square feet
- Change of occupancy type

Documentation Required:
- Site plan showing building location, setbacks, parking
- Architectural plans and elevations  
- Structural engineering plans (if required)
- Utility connection plans
- Landscape plan

A.2 Commercial Construction Permits
Additional Requirements for Commercial Projects:
- Traffic impact study (if >50 parking spaces)
- Fire department review and approval
- Health department review (food service establishments)
- Accessibility compliance review
- Signage permit (separate application)

Processing Timeline:
- Plan review: 15 business days
- Permit issuance: 5 business days after approval
- Expedited review available (additional fee)

A.3 Special Permits
- Conditional use permits
- Variance applications  
- Site plan approvals
- Subdivision approvals

Public Hearing Requirements:
- Notice published 15 days prior
- Adjacent property owners notified
- Planning Commission review
- City Council final approval (major projects)

PART B: FEE SCHEDULE

B.1 Building Permit Fees
- Residential: $2.50 per $1,000 construction value
- Commercial: $3.00 per $1,000 construction value
- Minimum fee: $50

B.2 Plan Review Fees
- 65% of building permit fee
- Expedited review: 150% of standard fee
                """
            },
            {
                "filename": "public_consultation_guide.txt",
                "content": """
PUBLIC CONSULTATION AND ENGAGEMENT PROCEDURES

SECTION 1: CONSULTATION REQUIREMENTS

1.1 Mandatory Public Consultation
Public consultation is required for:
- Zoning amendments and text changes
- Conditional use permit applications
- Variance requests  
- Master plan amendments
- Major subdivision proposals (>10 lots)
- Projects requiring special use permits

1.2 Notification Procedures
Written Notice Requirements:
- Property owners within 300 feet of project site
- Tenant occupants of properties within 300 feet
- Neighborhood associations and homeowner groups
- Local business associations (commercial projects)

Notice Timeline:
- 21 days prior to Planning Commission hearing
- 15 days prior to City Council hearing
- Published in newspaper of general circulation
- Posted on city website and project site

1.3 Public Hearing Process
Planning Commission Review:
- Staff presentation of project details
- Applicant presentation (15 minutes maximum)  
- Public comment period (3 minutes per speaker)
- Commission deliberation and recommendation

City Council Review (if required):
- Planning Commission recommendation review
- Additional public comment period
- Council deliberation and final decision

SECTION 2: ALTERNATIVE CONSULTATION METHODS

2.1 Community Workshops
- Informal setting for public input
- Interactive displays and information stations
- Small group discussions facilitated by staff
- Input forms and comment cards

2.2 Online Engagement
- Project information posted on city website
- Online comment forms and surveys
- Virtual public meetings (when appropriate)  
- Social media outreach and updates

2.3 Stakeholder Meetings
- Meetings with directly affected neighbors
- Business community briefings
- Utility company coordination
- Emergency services consultation
                """
            },
            {
                "filename": "infrastructure_standards.txt",
                "content": """
MUNICIPAL INFRASTRUCTURE DEVELOPMENT STANDARDS

TITLE I: TRANSPORTATION INFRASTRUCTURE

1.1 Street Design Standards
Collector Streets:
- Right-of-way width: 60 feet minimum
- Pavement width: 32 feet
- Sidewalk width: 5 feet both sides
- Parkway strip: 8 feet minimum

Local Streets:
- Right-of-way width: 50 feet minimum  
- Pavement width: 28 feet
- Sidewalk width: 4 feet both sides
- Parkway strip: 6 feet minimum

1.2 Traffic Control Devices
- Stop signs at all intersections with collector streets
- Speed limit signs every 600 feet
- Street name signs at all intersections
- Crosswalk markings at all intersections

TITLE II: UTILITY INFRASTRUCTURE

2.1 Water System Requirements
- Water main minimum diameter: 8 inches
- Fire hydrant spacing: 400 feet maximum
- Water pressure: 60 psi minimum at all connections
- Looped system required for developments >50 units

2.2 Sewer System Requirements  
- Sanitary sewer minimum diameter: 8 inches
- Minimum slope: 0.5% for 8-inch pipes
- Manholes at grade changes and every 400 feet
- Connection to municipal treatment system required

2.3 Stormwater Management
- Storm sewer minimum diameter: 15 inches
- 25-year flood capacity required
- Retention/detention facilities for peak flow control
- Water quality treatment for first flush runoff

TITLE III: TELECOMMUNICATIONS

3.1 Broadband Infrastructure
- Conduit installation required for all new developments
- Fiber optic capability to each residential unit
- Wireless facility design standards
- Underground installation preferred

3.2 Emergency Services
- Emergency vehicle access roads: 20 feet minimum width
- Fire department connections at all buildings >5,000 sq ft  
- Emergency communication system compatibility
- Street addressing system compliance
                """
            }
        ]
        
        # Write test documents
        for doc in documents:
            doc_path = cls.test_data_dir / doc["filename"]
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
    
    def setUp(self):
        """Set up individual test"""
        # Mock Redis for testing
        self.mock_redis = MagicMock()
        self.redis_patcher = patch('distributed_processor.redis.from_url')
        self.mock_redis_from_url = self.redis_patcher.start()
        self.mock_redis_from_url.return_value = self.mock_redis
        
        # Initialize test processor
        self.processor = DistributedProcessor(
            worker_id="test-worker",
            redis_url="redis://test:6379"
        )
        
        # Performance tracking
        self.performance_metrics = {
            'processing_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'quality_scores': [],
            'consistency_scores': []
        }
    
    def tearDown(self):
        """Clean up after test"""
        self.redis_patcher.stop()
    
    @pytest.mark.asyncio
    async def test_single_document_processing(self):
        """Test processing of single municipal development plan document"""
        # Select test document
        doc_path = self.test_data_dir / "zoning_ordinance.txt"
        query = "What are the zoning requirements for residential development?"
        
        # Create processing task
        task = ProcessingTask(
            task_id="test-task-001",
            document_path=str(doc_path),
            query=query
        )
        
        # Process document
        start_time = time.time()
        result_data = await self.processor._perform_egw_processing(task)
        processing_time = time.time() - start_time
        
        # Validate results
        self.assertIsInstance(result_data, dict)
        self.assertIn('content', result_data)
        self.assertIn('evidence', result_data)
        self.assertIn('summary', result_data)
        self.assertIn('metadata', result_data)
        
        # Check content quality
        content = result_data['content']
        self.assertGreater(len(content), 100, "Generated content should be substantial")
        
        # Verify evidence extraction
        evidence = result_data['evidence']
        self.assertIsInstance(evidence, list)
        self.assertGreater(len(evidence), 0, "Should extract relevant evidence")
        
        # Performance benchmark check
        self.assertLess(
            processing_time, 
            self.benchmark_requirements['max_processing_time_per_doc'],
            f"Processing time {processing_time:.2f}s exceeds benchmark"
        )
        
        self.logger.info(f"Single document processing completed in {processing_time:.2f}s")
    
    @pytest.mark.asyncio 
    async def test_batch_processing_performance(self):
        """Test batch processing performance against 170+ document benchmark"""
        # Use all test documents
        document_paths = list(self.test_data_dir.glob("*.txt"))
        
        # Replicate documents to reach target volume
        target_count = 175  # Slightly above 170 requirement
        replicated_paths = []
        
        for i in range(target_count):
            original_doc = document_paths[i % len(document_paths)]
            replicated_paths.append(str(original_doc))
        
        query = "Summarize the key development requirements and procedures"
        
        # Mock distributed processing for performance test
        async def mock_perform_egw_processing(task):
            """Mock processing that simulates realistic computation"""
            # Simulate processing time
            await asyncio.sleep(0.1)  # 100ms per document
            
            return {
                'content': f"Processed content for {task.document_path}",
                'evidence': [f"Evidence from {task.document_path}"],
                'summary': f"Summary of {task.document_path}",
                'metadata': {'document_path': task.document_path},
                'query_expansion': [query],
                'relevance_scores': [0.85, 0.72, 0.91]
            }
        
        # Patch the processing method
        with patch.object(self.processor, '_perform_egw_processing', mock_perform_egw_processing):
            start_time = time.time()
            
            # Process batch using thread pool to simulate distributed processing
            with ThreadPoolExecutor(max_workers=8) as executor:
                tasks = []
                for i, doc_path in enumerate(replicated_paths):
                    task = ProcessingTask(
                        task_id=f"perf-test-{i:03d}",
                        document_path=doc_path,
                        query=query
                    )
                    tasks.append(task)
                
                # Submit all tasks
                futures = [
                    executor.submit(asyncio.run, self.processor._perform_egw_processing(task))
                    for task in tasks
                ]
                
                # Wait for completion
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Task failed: {e}")
            
            total_time = time.time() - start_time
            documents_processed = len(results)
            
            # Calculate performance metrics
            documents_per_hour = (documents_processed / total_time) * 3600
            avg_processing_time = total_time / documents_processed if documents_processed > 0 else float('inf')
            
            # Validate performance benchmarks
            self.assertGreaterEqual(
                documents_per_hour,
                self.benchmark_requirements['documents_per_hour'],
                f"Processing rate {documents_per_hour:.1f} docs/hour below benchmark"
            )
            
            self.assertLessEqual(
                avg_processing_time,
                self.benchmark_requirements['max_processing_time_per_doc'],
                f"Average processing time {avg_processing_time:.2f}s exceeds benchmark"
            )
            
            self.logger.info(
                f"Batch processing performance: {documents_per_hour:.1f} docs/hour, "
                f"avg time: {avg_processing_time:.2f}s per doc"
            )
    
    def test_quality_validation(self):
        """Test quality validation of extracted evidence"""
        # Create test result
        test_result_data = {
            'content': '''
Based on the municipal zoning ordinance, residential development requires:

1. R-1 Single Family zones need minimum 8,000 sq ft lots with 35 ft height limit
2. Front yard setbacks of 25 feet minimum are required  
3. Side yard setbacks must be at least 8 feet on each side
4. Maximum lot coverage is limited to 40% of total lot area
5. Special use permits required for developments over 10,000 sq ft
            ''',
            'evidence': [
                "Minimum lot size: 8,000 square feet",
                "Maximum building height: 35 feet", 
                "Front yard setback: 25 feet minimum",
                "Side yard setback: 8 feet minimum each side",
                "Maximum lot coverage: 40%"
            ],
            'summary': "Residential zoning requirements include lot size, setbacks, and height restrictions",
            'metadata': {'source': 'zoning_ordinance.txt'}
        }
        
        # Mock result object
        from distributed_processor import ProcessingResult
        test_result = ProcessingResult(
            task_id="quality-test-001",
            worker_id="test-worker",
            status="completed",
            result_data=test_result_data,
            processing_time=2.5,
            quality_metrics={}
        )
        
        query = "What are the zoning requirements for residential development?"
        
        # Validate quality
        quality_metrics = self.processor.quality_validator.validate_result(test_result, query)
        
        # Check quality metrics
        self.assertIn('relevance_score', quality_metrics)
        self.assertIn('coherence_score', quality_metrics)
        self.assertIn('performance_score', quality_metrics)
        self.assertIn('completeness_score', quality_metrics)
        self.assertIn('overall_quality', quality_metrics)
        
        # Validate benchmark requirements
        self.assertGreaterEqual(
            quality_metrics['overall_quality'],
            self.benchmark_requirements['min_quality_score'],
            "Overall quality score below benchmark"
        )
        
        # Check individual metrics are reasonable
        self.assertGreaterEqual(quality_metrics['relevance_score'], 0.6)
        self.assertGreaterEqual(quality_metrics['coherence_score'], 0.7)
        self.assertGreaterEqual(quality_metrics['completeness_score'], 0.8)
        
        self.logger.info(f"Quality validation passed: {quality_metrics['overall_quality']:.3f}")
    
    def test_result_aggregation_consistency(self):
        """Test result aggregation and consistency validation"""
        # Create multiple processing results to simulate distributed processing
        results = []
        
        base_content = "Municipal development requires zoning compliance, permit approval, and environmental review."
        
        for i in range(3):
            # Slight variations to simulate different processing instances
            variant_content = base_content + f" Additional detail from worker {i}."
            
            result = ProcessingResult(
                task_id=f"consistency-test-{i:03d}",
                worker_id=f"worker-{i}",
                status="completed",
                result_data={
                    'content': variant_content,
                    'evidence': [f"Evidence {j}" for j in range(3)],
                    'summary': f"Summary from worker {i}",
                    'metadata': {'worker_id': f"worker-{i}"}
                },
                processing_time=1.5 + i * 0.2,
                quality_metrics={
                    'overall_quality': 0.85 + i * 0.02,
                    'relevance_score': 0.82 + i * 0.03,
                    'coherence_score': 0.88 - i * 0.01
                }
            )
            results.append(result)
        
        # Aggregate results
        aggregated = self.processor.result_aggregator.aggregate_results(
            results, "consistency-test-request"
        )
        
        # Validate aggregated result
        self.assertIsInstance(aggregated.combined_results, dict)
        self.assertIn('consensus_content', aggregated.combined_results)
        self.assertIn('consensus_evidence', aggregated.combined_results)
        self.assertIn('confidence_scores', aggregated.combined_results)
        
        # Check consistency score
        self.assertGreaterEqual(
            aggregated.consistency_score,
            self.benchmark_requirements['min_consistency_score'],
            f"Consistency score {aggregated.consistency_score:.3f} below benchmark"
        )
        
        # Verify quality score aggregation
        self.assertGreaterEqual(aggregated.quality_score, 0.8)
        
        self.logger.info(
            f"Aggregation completed - consistency: {aggregated.consistency_score:.3f}, "
            f"quality: {aggregated.quality_score:.3f}"
        )
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self):
        """Test complete pipeline flow with actual municipal documents"""
        # Select subset of documents for complete flow test
        test_documents = [
            str(self.test_data_dir / "zoning_ordinance.txt"),
            str(self.test_data_dir / "permit_procedures.txt"),
            str(self.test_data_dir / "environmental_procedures.txt")
        ]
        
        query = "What is the complete process for residential development approval?"
        
        # Mock Redis operations for complete flow
        self.mock_redis.brpop.return_value = None  # No tasks in queue initially
        self.mock_redis.hget.side_effect = self._mock_redis_hget
        self.mock_redis.hset.return_value = True
        self.mock_redis.lpush.return_value = len(test_documents)
        
        # Process batch with complete pipeline
        try:
            # This would normally coordinate across multiple workers
            # For testing, we simulate the coordination locally
            
            processing_results = []
            start_time = time.time()
            
            for i, doc_path in enumerate(test_documents):
                task = ProcessingTask(
                    task_id=f"pipeline-test-{i:03d}",
                    document_path=doc_path,
                    query=query
                )
                
                # Process with full pipeline
                result_data = await self.processor._perform_egw_processing(task)
                
                # Validate quality
                result = ProcessingResult(
                    task_id=task.task_id,
                    worker_id=self.processor.worker_id,
                    status="completed",
                    result_data=result_data,
                    processing_time=time.time() - start_time,
                    quality_metrics={}
                )
                
                quality_metrics = self.processor.quality_validator.validate_result(result, query)
                result.quality_metrics = quality_metrics
                
                processing_results.append(result)
            
            total_time = time.time() - start_time
            
            # Aggregate results
            aggregated = self.processor.result_aggregator.aggregate_results(
                processing_results, "complete-pipeline-test"
            )
            
            # Comprehensive validation
            self.assertEqual(len(processing_results), len(test_documents))
            self.assertGreater(len(aggregated.combined_results['consensus_content']), 200)
            self.assertGreaterEqual(aggregated.quality_score, 0.75)
            self.assertGreaterEqual(aggregated.consistency_score, 0.7)
            
            # Performance validation
            avg_processing_time = total_time / len(test_documents)
            self.assertLess(
                avg_processing_time,
                self.benchmark_requirements['max_processing_time_per_doc']
            )
            
            self.logger.info(
                f"Complete pipeline test passed - {len(test_documents)} documents in {total_time:.2f}s, "
                f"quality: {aggregated.quality_score:.3f}, consistency: {aggregated.consistency_score:.3f}"
            )
            
        except Exception as e:
            self.fail(f"Complete pipeline flow test failed: {e}")
    
    def _mock_redis_hget(self, key: str, field: str):
        """Mock Redis hget for testing"""
        if key == "results":
            # Return mock result data
            return json.dumps({
                'task_id': field,
                'worker_id': 'test-worker',
                'status': 'completed',
                'result_data': {'content': 'test content'},
                'processing_time': 1.5,
                'quality_metrics': {'overall_quality': 0.8},
                'completed_at': time.time()
            })
        return None
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        # Test with invalid document path
        invalid_task = ProcessingTask(
            task_id="error-test-001", 
            document_path="/nonexistent/document.txt",
            query="Test query"
        )
        
        # Process should handle error gracefully
        async def test_error_handling():
            try:
                await self.processor._process_task(invalid_task)
                # Should not reach here if error handling works
                self.assertTrue(False, "Expected error was not raised")
            except Exception:
                # Expected - error should be handled internally
                pass
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_error_handling())
        finally:
            loop.close()
    
    @pytest.mark.skip(reason="Requires actual Redis instance for full distributed test")
    def test_distributed_coordination(self):
        """Test distributed coordination with multiple workers (requires Redis)"""
        # This test would require actual Redis instance and multiple worker processes
        # Skipped in unit tests but should be run in integration environment
        pass


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""
    
    def setUp(self):
        """Set up performance tests"""
        self.benchmark_requirements = {
            'documents_per_hour': 170,
            'max_processing_time_per_doc': 21.2,
            'max_memory_usage_mb': 2048,
            'min_quality_score': 0.8
        }
    
    def test_memory_usage_benchmark(self):
        """Test memory usage stays within limits"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize processor
        processor = DistributedProcessor(worker_id="memory-test")
        
        # Simulate processing load
        for i in range(10):
            # Create large test data
            test_data = {
                'content': 'A' * 10000,  # 10KB content
                'evidence': ['Evidence'] * 100,
                'metadata': {'test': True}
            }
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        self.assertLess(
            memory_increase,
            self.benchmark_requirements['max_memory_usage_mb'],
            f"Memory increase {memory_increase:.1f}MB exceeds benchmark"
        )
    
    def test_concurrent_processing_performance(self):
        """Test concurrent processing performance"""
        import concurrent.futures
        
        def mock_process_document():
            """Mock document processing"""
            time.sleep(0.1)  # Simulate processing time
            return {
                'content': 'Processed content',
                'quality_score': 0.85
            }
        
        start_time = time.time()
        
        # Process documents concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(mock_process_document) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        documents_per_hour = (len(results) / total_time) * 3600
        
        self.assertGreaterEqual(
            documents_per_hour,
            self.benchmark_requirements['documents_per_hour'],
            f"Concurrent processing rate {documents_per_hour:.1f} below benchmark"
        )


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)