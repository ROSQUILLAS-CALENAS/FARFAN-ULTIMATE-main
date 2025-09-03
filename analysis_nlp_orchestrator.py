"""
Analysis NLP Orchestrator with Total Ordering

This orchestrator coordinates all 9 analysis_nlp components, ensuring they work together
with deterministic processing, consistent sorting, and canonical JSON serialization.
"""

import json
import logging
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union, Tuple, Callable  # Module not found  # Module not found  # Module not found
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found

# Import total ordering base
# # # from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin  # Module not found  # Module not found  # Module not found

# Import all analysis_nlp components
# # # from canonical_flow.A_analysis_nlp import (  # Module not found  # Module not found  # Module not found

# Mandatory Pipeline Contract Annotations
__phase__ = "A"
__code__ = "25A"
__stage_order__ = 4

    AdaptiveAnalyzer,
    QuestionAnalyzer,
    QuestionDecalogoMapper,
    ExtractorEvidenciasContextual,
    EvidenceProcessor,
    EvidenceValidationModel,
    DNPAlignmentAdapter,
    EvaluationDrivenProcessor,
)

logger = logging.getLogger(__name__)


class AnalysisNLPOrchestrator(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Main orchestrator for analysis_nlp components with total ordering.
    
    Coordinates all 9 components to work together deterministically:
    1. AdaptiveAnalyzer - System analysis and adaptation
    2. QuestionAnalyzer - Question analysis and requirements
    3. QuestionDecalogoMapper - Question to decálogo mapping
    4. ExtractorEvidenciasContextual - Contextual evidence extraction
    5. EvidenceProcessor - Evidence processing and structuring
    6. EvidenceValidationModel - Evidence validation and scoring
    7. DNPAlignmentAdapter - DNP compliance checking
    8. EvaluationDrivenProcessor - Integrated evaluation processing
    
    Plus the orchestrator itself as the 9th component.
    """
    
    def __init__(self) -> None:
        super().__init__("AnalysisNLPOrchestrator")
        
        # Initialize all components
        self.components: Dict[str, Any] = self._initialize_components()
        
        # Processing configuration
        self.processing_order: List[str] = [
            "adaptive_analyzer",
            "question_analyzer", 
            "implementacion_mapeo",
            "extractor_evidencias_contextual",
            "evidence_processor",
            "evidence_validation_model",
            "dnp_alignment_adapter",
            "evaluation_driven_processor"
        ]
        
        # Orchestrator statistics
        self.orchestrator_stats: Dict[str, Any] = {
            "components_initialized": len(self.components),
            "processing_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "last_run_timestamp": None,
        }
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "components_count": len(self.components),
            "processing_order": self.processing_order,
            "component_names": sorted(self.components.keys()),
        }
    
    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize all analysis_nlp components"""
        components = {}
        
        try:
            components["adaptive_analyzer"] = AdaptiveAnalyzer()
            logger.info("Initialized AdaptiveAnalyzer")
        except Exception as e:
            logger.error(f"Failed to initialize AdaptiveAnalyzer: {e}")
            components["adaptive_analyzer"] = None
        
        try:
            components["question_analyzer"] = QuestionAnalyzer()
            logger.info("Initialized QuestionAnalyzer")
        except Exception as e:
            logger.error(f"Failed to initialize QuestionAnalyzer: {e}")
            components["question_analyzer"] = None
        
        try:
            components["implementacion_mapeo"] = QuestionDecalogoMapper()
            logger.info("Initialized QuestionDecalogoMapper")
        except Exception as e:
            logger.error(f"Failed to initialize QuestionDecalogoMapper: {e}")
            components["implementacion_mapeo"] = None
        
        try:
            components["extractor_evidencias_contextual"] = ExtractorEvidenciasContextual()
            logger.info("Initialized ExtractorEvidenciasContextual")
        except Exception as e:
            logger.error(f"Failed to initialize ExtractorEvidenciasContextual: {e}")
            components["extractor_evidencias_contextual"] = None
        
        try:
            components["evidence_processor"] = EvidenceProcessor()
            logger.info("Initialized EvidenceProcessor")
        except Exception as e:
            logger.error(f"Failed to initialize EvidenceProcessor: {e}")
            components["evidence_processor"] = None
        
        try:
            components["evidence_validation_model"] = EvidenceValidationModel()
            logger.info("Initialized EvidenceValidationModel")
        except Exception as e:
            logger.error(f"Failed to initialize EvidenceValidationModel: {e}")
            components["evidence_validation_model"] = None
        
        try:
            components["dnp_alignment_adapter"] = DNPAlignmentAdapter()
            logger.info("Initialized DNPAlignmentAdapter")
        except Exception as e:
            logger.error(f"Failed to initialize DNPAlignmentAdapter: {e}")
            components["dnp_alignment_adapter"] = None
        
        try:
            components["evaluation_driven_processor"] = EvaluationDrivenProcessor()
            logger.info("Initialized EvaluationDrivenProcessor")
        except Exception as e:
            logger.error(f"Failed to initialize EvaluationDrivenProcessor: {e}")
            components["evaluation_driven_processor"] = None
        
        return self.sort_dict_by_keys(components)
    
    def process(self, data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Main orchestrated processing function with deterministic output.
        
        Args:
            data: Input data for analysis
            context: Processing context
            
        Returns:
# # #             Deterministic orchestrated results from all components  # Module not found  # Module not found  # Module not found
        """
        operation_id = self.generate_operation_id("orchestrate", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Track processing
            self.orchestrator_stats["processing_runs"] += 1
            
            # Process through each component in order
            component_results = {}
            processing_errors = {}
            
            current_data = canonical_data
            
            for component_name in self.processing_order:
                component = self.components.get(component_name)
                
                if component is None:
                    processing_errors[component_name] = "Component not initialized"
                    continue
                
                try:
                    # Process with current component
                    component_result = component.process(current_data, canonical_context)
                    component_results[component_name] = component_result
                    
                    # Update current_data for next component (if the component provides enriched data)
                    if isinstance(component_result, dict) and "results" in component_result:
                        current_data = component_result["results"]
                    elif isinstance(component_result, dict):
                        current_data = component_result
                    
                    logger.debug(f"Successfully processed with {component_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing with {component_name}: {e}")
                    processing_errors[component_name] = str(e)
                    # Continue processing with other components
                    continue
            
            # Generate comprehensive orchestrated output
            output = self._generate_orchestrated_output(
                canonical_data, canonical_context, component_results, processing_errors, operation_id
            )
            
            # Update statistics
            if processing_errors:
                self.orchestrator_stats["failed_runs"] += 1
            else:
                self.orchestrator_stats["successful_runs"] += 1
            
            self.orchestrator_stats["last_run_timestamp"] = self._get_deterministic_timestamp()
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            self.orchestrator_stats["failed_runs"] += 1
            self.orchestrator_stats["last_run_timestamp"] = self._get_deterministic_timestamp()
            
            error_output = {
                "component": self.component_name,
                "error": str(e),
                "operation_id": operation_id,
                "status": "orchestrator_error",
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _generate_orchestrated_output(
        self, 
        original_data: Dict[str, Any], 
        context: Dict[str, Any], 
        component_results: Dict[str, Any], 
        processing_errors: Dict[str, Any], 
        operation_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive orchestrated output"""
        
# # #         # Aggregate results from all components  # Module not found  # Module not found  # Module not found
        aggregated_results = {}
        
        for component_name, result in sorted(component_results.items()):
            if isinstance(result, dict):
# # #                 # Extract the main results from each component  # Module not found  # Module not found  # Module not found
                if "results" in result:
                    aggregated_results[component_name] = result["results"]
                else:
                    aggregated_results[component_name] = result
            else:
                aggregated_results[component_name] = result
        
        # Generate comprehensive analysis summary
        analysis_summary = self._generate_analysis_summary(component_results)
        
# # #         # Generate recommendations from all components  # Module not found  # Module not found  # Module not found
        orchestrated_recommendations = self._aggregate_recommendations(component_results)
        
        # Generate quality metrics
        quality_metrics = self._calculate_quality_metrics(component_results, processing_errors)
        
        # Create final orchestrated output
        orchestrated_output = {
            "aggregated_results": self.sort_dict_by_keys(aggregated_results),
            "analysis_summary": analysis_summary,
            "component": self.component_name,
            "component_id": self.component_id,
            "component_results": self.sort_dict_by_keys(component_results),
            "metadata": self.get_deterministic_metadata(),
            "operation_id": operation_id,
            "orchestrator_stats": self.sort_dict_by_keys(self.orchestrator_stats),
            "processing_errors": self.sort_dict_by_keys(processing_errors),
            "quality_metrics": quality_metrics,
            "recommendations": orchestrated_recommendations,
            "status": "success" if not processing_errors else "partial_success",
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(orchestrated_output)
    
    def _generate_analysis_summary(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
# # #         """Generate comprehensive analysis summary from all components"""  # Module not found  # Module not found  # Module not found
        
        summary = {
            "components_processed": len(component_results),
            "total_components": len(self.components),
            "processing_success_rate": 0.0,
            "key_findings": [],
            "overall_confidence": 0.0,
        }
        
        # Calculate success rate
        if len(self.components) > 0:
            summary["processing_success_rate"] = len(component_results) / len(self.components)
        
# # #         # Aggregate key findings from components  # Module not found  # Module not found  # Module not found
        confidence_scores = []
        
        for component_name, result in sorted(component_results.items()):
            if isinstance(result, dict):
                # Extract key findings
                if "summary" in result:
                    summary["key_findings"].append(f"{component_name}: {result['summary']}")
                
                # Extract confidence scores
                if "confidence" in result:
                    confidence_scores.append(result["confidence"])
                elif isinstance(result.get("results"), dict) and "confidence" in result["results"]:
                    confidence_scores.append(result["results"]["confidence"])
        
        # Calculate overall confidence
        if confidence_scores:
            summary["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        return summary
    
    def _aggregate_recommendations(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
# # #         """Aggregate recommendations from all components"""  # Module not found  # Module not found  # Module not found
        
        orchestrated_recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "component_specific": {},
        }
        
        for component_name, result in sorted(component_results.items()):
            if isinstance(result, dict):
                component_recommendations = []
                
# # #                 # Extract recommendations from different possible locations  # Module not found  # Module not found  # Module not found
                if "recommendations" in result:
                    recs = result["recommendations"]
                    if isinstance(recs, list):
                        component_recommendations.extend(recs)
                    elif isinstance(recs, dict):
                        if "recommendations" in recs:
                            component_recommendations.extend(recs["recommendations"])
                        else:
                            component_recommendations.append(str(recs))
                
# # #                 # Extract from results  # Module not found  # Module not found  # Module not found
                if isinstance(result.get("results"), dict):
                    results = result["results"]
                    if "recommendations" in results:
                        recs = results["recommendations"]
                        if isinstance(recs, list):
                            component_recommendations.extend(recs)
                        elif isinstance(recs, dict) and "recommendations" in recs:
                            component_recommendations.extend(recs["recommendations"])
                
                # Store component-specific recommendations
                if component_recommendations:
                    orchestrated_recommendations["component_specific"][component_name] = sorted(component_recommendations)
                
                # Categorize by priority (simplified heuristic)
                for rec in component_recommendations:
                    rec_str = str(rec).lower()
                    if any(term in rec_str for term in ["critical", "urgent", "important"]):
                        orchestrated_recommendations["high_priority"].append(f"{component_name}: {rec}")
                    elif any(term in rec_str for term in ["moderate", "should", "consider"]):
                        orchestrated_recommendations["medium_priority"].append(f"{component_name}: {rec}")
                    else:
                        orchestrated_recommendations["low_priority"].append(f"{component_name}: {rec}")
        
        # Sort all recommendations for deterministic output
        orchestrated_recommendations["high_priority"] = sorted(orchestrated_recommendations["high_priority"])
        orchestrated_recommendations["medium_priority"] = sorted(orchestrated_recommendations["medium_priority"])
        orchestrated_recommendations["low_priority"] = sorted(orchestrated_recommendations["low_priority"])
        orchestrated_recommendations["component_specific"] = self.sort_dict_by_keys(orchestrated_recommendations["component_specific"])
        
        return orchestrated_recommendations
    
    def _calculate_quality_metrics(self, component_results: Dict[str, Any], processing_errors: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the orchestrated processing"""
        
        quality_metrics = {
            "component_success_count": len(component_results),
            "component_error_count": len(processing_errors),
            "overall_processing_quality": 0.0,
            "data_consistency_score": 0.0,
            "deterministic_reproducibility": True,
        }
        
        # Calculate overall processing quality
        total_components = len(self.components)
        if total_components > 0:
            quality_metrics["overall_processing_quality"] = len(component_results) / total_components
        
        # Calculate data consistency score (all components producing structured output)
        consistent_components = 0
        for result in component_results.values():
            if isinstance(result, dict) and "component_id" in result:
                consistent_components += 1
        
        if component_results:
            quality_metrics["data_consistency_score"] = consistent_components / len(component_results)
        
        return quality_metrics
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        
        component_status = {}
        
        for component_name, component in sorted(self.components.items()):
            if component is None:
                component_status[component_name] = {
                    "initialized": False,
                    "status": "not_available",
                }
            else:
                component_status[component_name] = {
                    "component_id": getattr(component, "component_id", "unknown"),
                    "component_name": getattr(component, "component_name", component_name),
                    "initialized": True,
                    "status": "available",
                }
        
        return self.sort_dict_by_keys(component_status)
    
    def validate_deterministic_output(self, output1: Dict[str, Any], output2: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that two outputs are deterministically identical"""
        
        # Serialize both outputs canonically
        json1 = self.canonical_json_dumps(output1)
        json2 = self.canonical_json_dumps(output2)
        
        # Compare
        are_identical = json1 == json2
        
        validation_result = {
            "are_identical": are_identical,
            "output1_hash": self.generate_stable_id(output1),
            "output2_hash": self.generate_stable_id(output2),
            "json1_length": len(json1),
            "json2_length": len(json2),
        }
        
        if not are_identical:
            # Try to identify differences (simplified)
            validation_result["difference_detected"] = True
            validation_result["hash_mismatch"] = validation_result["output1_hash"] != validation_result["output2_hash"]
        
        return validation_result


# Standalone functions for backward compatibility
def orchestrate_analysis_nlp(data: Optional[Any] = None, context: Optional[Any] = None) -> Dict[str, Any]:
    """Orchestrate all analysis_nlp components"""
    orchestrator = AnalysisNLPOrchestrator()
    return orchestrator.process(data, context)


def get_all_component_status() -> Dict[str, Any]:
    """Get status of all analysis_nlp components"""
    orchestrator = AnalysisNLPOrchestrator()
    return orchestrator.get_component_status()


def validate_deterministic_processing(data: Optional[Any] = None) -> Dict[str, Any]:
    """Validate that processing is deterministic by running twice"""
    orchestrator = AnalysisNLPOrchestrator()
    
    # Run processing twice with same inputs
    result1 = orchestrator.process(data)
    result2 = orchestrator.process(data)
    
    # Validate deterministic output
    return orchestrator.validate_deterministic_output(result1, result2)


# Main process function for compatibility
def process(data=None, context=None):
    """Main process function for the orchestrator"""
    return orchestrate_analysis_nlp(data, context)