#!/usr/bin/env python3
"""
Demo script showing integration of standardized schema contracts
with classification and scoring components.
"""

import logging
# # # from typing import Dict, Any, List  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
# # #     from contracts.schemas import (  # Module not found  # Module not found  # Module not found
        QuestionEvalInput,
        DimensionEvalOutput, 
        PointEvalOutput,
        StageMeta,
        ComplianceStatus,
        ConfidenceLevel,
        ProcessingStatus,
        validate_process_schemas,
        enforce_required_fields,
        create_stage_meta,
        SchemaValidationError,
    )
    
    class MockClassificationEngine:
        """Mock classification engine with schema validation"""
        
        @enforce_required_fields("doc_id", "page_num")
        @validate_process_schemas(input_schema=QuestionEvalInput)
        def process(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
            """Process classification with validated schemas"""
            logger.info(f"Processing classification for doc_id: {data.doc_id}, page: {data.page_num}")
            
            start_time = datetime.now()
            
            # Mock processing logic
            mock_score = 0.75 + hash(data.question_text) % 100 / 1000
            mock_score = min(1.0, max(0.0, mock_score))
            
            # Create dimension evaluation output
            dimension_output = DimensionEvalOutput(
                doc_id=data.doc_id,
                page_num=data.page_num,
                dimension_id="DE1",
                dimension_name="Institutional Dimension",
                score=mock_score,
                compliance_status=ComplianceStatus.CUMPLE if mock_score >= 0.75 else ComplianceStatus.CUMPLE_PARCIAL,
                confidence_level=ConfidenceLevel.HIGH,
                evidence_count=5,
                sub_scores={"governance": mock_score + 0.1, "transparency": mock_score - 0.1},
                recommendations=["Enhance documentation", "Improve evidence quality"] if mock_score < 0.8 else []
            )
            
            # Create stage metadata
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            stage_meta = create_stage_meta(
                stage_name="mock_classification_engine",
                processing_status=ProcessingStatus.SUCCESS,
                stage_version="2.0.0",
                execution_time_ms=execution_time,
                performance_metrics={"classification_score": mock_score}
            )
            
            return {
                "doc_id": data.doc_id,
                "page_num": data.page_num,
                "status": "success",
                "dimension_evaluation": dimension_output.dict(),
                "stage_metadata": stage_meta.dict(),
                "deterministic_id": data.get_deterministic_id()
            }
    
    
    class MockScoringEngine:
        """Mock scoring engine with schema validation"""
        
        @enforce_required_fields("doc_id", "page_num")
        def process(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
            """Process scoring with validated schemas"""
            logger.info(f"Processing scoring for doc_id: {data['doc_id']}, page: {data['page_num']}")
            
            start_time = datetime.now()
            
            # Mock multiple point evaluations
            point_outputs = []
            for i in range(1, 6):  # P1-P5
                point_id = f"P{i}"
                mock_score = 0.6 + (hash(f"{data['doc_id']}_{point_id}") % 100) / 250
                mock_score = min(1.0, max(0.0, mock_score))
                
                point_output = PointEvalOutput(
                    doc_id=data["doc_id"],
                    page_num=data["page_num"],
                    point_id=point_id,
                    point_title=f"Human Rights Point {i}",
                    score=mock_score,
                    compliance_status=ComplianceStatus.CUMPLE if mock_score >= 0.75 else 
                                    ComplianceStatus.CUMPLE_PARCIAL if mock_score >= 0.5 else
                                    ComplianceStatus.NO_CUMPLE,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    evidence_count=3 + i,
                    dimension_alignment="DE2" if i <= 2 else "DE3",
                    key_findings=[f"Finding {i}.1", f"Finding {i}.2"],
                    gap_analysis=[f"Gap {i}"] if mock_score < 0.7 else []
                )
                point_outputs.append(point_output.dict())
            
            # Create stage metadata
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            stage_meta = create_stage_meta(
                stage_name="mock_scoring_engine",
                processing_status=ProcessingStatus.SUCCESS,
                stage_version="2.0.0",
                execution_time_ms=execution_time,
                performance_metrics={
                    "points_evaluated": len(point_outputs),
                    "average_score": sum(p["score"] for p in point_outputs) / len(point_outputs)
                }
            )
            
            return {
                "doc_id": data["doc_id"],
                "page_num": data["page_num"],
                "status": "success",
                "point_evaluations": point_outputs,
                "stage_metadata": stage_meta.dict(),
                "global_score": sum(p["score"] for p in point_outputs) / len(point_outputs)
            }
    
    
    def demo_pipeline_integration():
        """Demonstrate full pipeline integration with schema validation"""
        logger.info("Starting schema integration demo...")
        
        # Create mock engines
        classifier = MockClassificationEngine()
        scorer = MockScoringEngine()
        
        # Test data
        test_cases = [
            {
                "doc_id": "PDT_2024_001",
                "page_num": 15,
                "question_text": "¿Cómo se garantizan los derechos humanos en este plan de desarrollo?",
                "context": {"municipality": "Bogotá", "year": 2024},
                "evaluation_criteria": ["human_rights", "institutional_capacity"],
                "priority": 1
            },
            {
                "doc_id": "PDT_2024_002", 
                "page_num": 42,
                "question_text": "¿Qué medidas se toman para la protección ambiental?",
                "context": {"municipality": "Medellín", "year": 2024},
                "evaluation_criteria": ["environmental_protection", "sustainability"],
                "priority": 2
            }
        ]
        
        results = []
        
        for i, test_data in enumerate(test_cases):
            logger.info(f"\n--- Test Case {i+1} ---")
            
            try:
                # Create validated input
                question_input = QuestionEvalInput(**test_data)
                logger.info(f"Created valid input with ID: {question_input.get_deterministic_id()}")
                
                # Process through classification
                classification_result = classifier.process(question_input)
                logger.info(f"Classification result: {classification_result['status']}")
                
                # Process through scoring  
                scoring_result = scorer.process({
                    "doc_id": test_data["doc_id"],
                    "page_num": test_data["page_num"]
                })
                logger.info(f"Scoring result: {scoring_result['status']}")
                
                # Combine results
                combined_result = {
                    "input": question_input.dict(),
                    "classification": classification_result,
                    "scoring": scoring_result,
                    "pipeline_status": "success"
                }
                results.append(combined_result)
                
                logger.info(f"✓ Test case {i+1} completed successfully")
                
            except SchemaValidationError as e:
                logger.error(f"Schema validation failed for test case {i+1}: {e}")
                results.append({
                    "input": test_data,
                    "error": str(e),
                    "pipeline_status": "schema_validation_failed"
                })
            except Exception as e:
                logger.error(f"Processing failed for test case {i+1}: {e}")
                results.append({
                    "input": test_data,
                    "error": str(e),
                    "pipeline_status": "processing_failed"
                })
        
        # Test error cases
        logger.info("\n--- Testing Error Handling ---")
        
        # Missing required fields
        try:
            invalid_input = {"question_text": "Missing doc_id and page_num"}
            classifier.process(invalid_input)
            logger.error("Should have failed validation!")
        except SchemaValidationError as e:
            logger.info(f"✓ Correctly caught validation error: {e}")
        
        # Invalid field values
        try:
            invalid_data = {
                "doc_id": "test",
                "page_num": 0,  # Invalid: must be >= 1
                "question_text": "Test question"
            }
            QuestionEvalInput(**invalid_data)
            logger.error("Should have failed validation!")
        except Exception as e:
            logger.info(f"✓ Correctly caught field validation error: {type(e).__name__}")
        
        # Unknown fields rejection
        try:
            invalid_data = {
                "doc_id": "test",
                "page_num": 1,
                "question_text": "Test",
                "unknown_field": "should fail"
            }
            QuestionEvalInput(**invalid_data)
            logger.error("Should have failed validation!")
        except Exception as e:
            logger.info(f"✓ Correctly rejected unknown field: {type(e).__name__}")
        
        logger.info("\n--- Demo Summary ---")
        successful = sum(1 for r in results if r.get("pipeline_status") == "success")
        logger.info(f"Processed {len(results)} test cases: {successful} successful, {len(results)-successful} failed")
        logger.info("Schema validation integration demo completed ✓")
        
        return results
    
    
    if __name__ == "__main__":
        demo_results = demo_pipeline_integration()
        print(f"\nDemo completed with {len(demo_results)} test cases processed.")

except ImportError as e:
    print(f"Schema contracts not available: {e}")
    print("Please ensure pydantic is installed: pip install pydantic")
    
    # Fallback demo without schemas
    def fallback_demo():
        print("Running fallback demo without schema validation...")
        
        # Mock data processing without validation
        test_data = {
            "doc_id": "test_doc",
            "page_num": 1,
            "question_text": "Test question without validation"
        }
        
        print(f"Processing: {test_data}")
        print("✓ Processed without schema validation (not recommended for production)")
    
    if __name__ == "__main__":
        fallback_demo()