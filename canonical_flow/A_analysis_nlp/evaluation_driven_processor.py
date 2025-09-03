"""
Canonical Flow Alias: 19A
Evaluation Driven Processor with Total Ordering and Deterministic Processing

Source: evaluation_driven_processor.py
Stage: analysis_nlp
Code: 19A
"""

import logging
import time
import sys
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any, Optional, Tuple  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# # # from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import adaptive_scoring_engine
except ImportError:
    adaptive_scoring_engine = None

try:
# # #     from intelligent_recommendation_engine import IntelligentRecommendationEngine  # Module not found  # Module not found  # Module not found
except ImportError:
    IntelligentRecommendationEngine = None

try:
# # #     from models import DocumentPackage, PDTContext, AdaptiveScoringResults, IntelligentRecommendations  # Module not found  # Module not found  # Module not found
except ImportError:
    DocumentPackage = None
    PDTContext = None
    AdaptiveScoringResults = None
    IntelligentRecommendations = None


class EvaluationDrivenProcessor(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Evaluation Driven Processor with deterministic processing and total ordering.
    
    Implements integrated evaluation flow:
    1. Initial Evaluation (Fundamental Weighted Calculation)
    2. Satisfiability Determination (Conjunctive Logic)
    3. Adaptive Refinement (AdaptiveScoringEngine)  
    4. Intelligent Recommendations (IntelligentRecommendationEngine)
    
    Provides consistent evaluation results and stable ID generation across runs.
    """
    
    def __init__(self, models_path: str = "models/adaptive_scoring", recommendations_path: str = "data/recommendations"):
        super().__init__("EvaluationDrivenProcessor")
        
        # Initialize engines if available
        self.adaptive_scoring_engine = None
        self.intelligent_recommendation_engine = None
        
        if adaptive_scoring_engine:
            try:
                self.adaptive_scoring_engine = adaptive_scoring_engine.AdaptiveScoringEngine(models_path)
            except Exception as e:
                logger.warning(f"Failed to initialize AdaptiveScoringEngine: {e}")
        
        if IntelligentRecommendationEngine:
            try:
                self.intelligent_recommendation_engine = IntelligentRecommendationEngine(recommendations_path)
            except Exception as e:
                logger.warning(f"Failed to initialize IntelligentRecommendationEngine: {e}")
        
        # Configuration with deterministic defaults (Phase 1)
        self.question_weights = self._initialize_question_weights()
        self.dimension_weights = self._initialize_dimension_weights()
        self.decalogo_weights = self._initialize_decalogo_weights()
        
        # Configuration for thresholds (Phase 2)
        self.satisfaction_thresholds = self._initialize_satisfaction_thresholds()
        
        # Processing metrics
        self.processing_metrics = {
            "documents_processed": 0,
            "avg_evaluation_score": 0.0,
            "high_scoring_documents": 0,
            "low_scoring_documents": 0,
            "recommendations_generated": 0,
            "last_update": self._get_deterministic_timestamp(),
        }
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "dimension_weights": self.dimension_weights,
            "has_adaptive_engine": self.adaptive_scoring_engine is not None,
            "has_recommendation_engine": self.intelligent_recommendation_engine is not None,
            "question_weights_count": len(self.question_weights),
            "satisfaction_thresholds": self.satisfaction_thresholds,
        }
    
    def _initialize_question_weights(self) -> Dict[str, float]:
        """Initialize question weights with deterministic ordering"""
        weights = {
            "DE1_critical_questions": 1.0,
            "DE1_high_questions": 0.8,
            "DE1_medium_questions": 0.6,
            "DE1_standard_questions": 0.4,
            "DE2_critical_questions": 0.9,
            "DE2_high_questions": 0.7,
            "DE2_medium_questions": 0.5,
            "DE2_standard_questions": 0.3,
            "DE3_critical_questions": 0.9,
            "DE3_high_questions": 0.7,
            "DE3_medium_questions": 0.5,
            "DE3_standard_questions": 0.3,
            "DE4_critical_questions": 0.8,
            "DE4_high_questions": 0.6,
            "DE4_medium_questions": 0.4,
            "DE4_standard_questions": 0.2,
        }
        return self.sort_dict_by_keys(weights)
    
    def _initialize_dimension_weights(self) -> Dict[str, float]:
        """Initialize dimension weights with deterministic values"""
        return self.sort_dict_by_keys({
            "DE1": 0.30,
            "DE2": 0.25,
            "DE3": 0.25,
            "DE4": 0.20
        })
    
    def _initialize_decalogo_weights(self) -> Dict[str, float]:
        """Initialize decálogo weights with deterministic values"""
        return self.sort_dict_by_keys({
            "P1": 0.12, "P2": 0.11, "P3": 0.10, "P4": 0.09, "P5": 0.08,
            "P6": 0.10, "P7": 0.09, "P8": 0.11, "P9": 0.10, "P10": 0.10
        })
    
    def _initialize_satisfaction_thresholds(self) -> Dict[str, float]:
        """Initialize satisfaction thresholds with deterministic values"""
        return self.sort_dict_by_keys({
            "critical_threshold": 0.8,
            "high_threshold": 0.7,
            "medium_threshold": 0.6,
            "minimum_threshold": 0.5,
        })
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function with deterministic output.
        
        Args:
            data: Input data containing evaluation data
            context: Processing context
            
        Returns:
            Deterministic evaluation results
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract evaluation data
            evaluation_data = self._extract_evaluation_data_deterministic(canonical_data)
            
            # Phase 1: Initial Evaluation (Weighted Calculation)
            initial_evaluation = self._perform_initial_evaluation_deterministic(evaluation_data)
            
            # Phase 2: Satisfiability Determination
            satisfiability_analysis = self._determine_satisfiability_deterministic(initial_evaluation)
            
            # Phase 3: Adaptive Refinement (if available)
            adaptive_results = self._perform_adaptive_refinement_deterministic(
                evaluation_data, initial_evaluation
            )
            
            # Phase 4: Intelligent Recommendations (if available)
            recommendations = self._generate_intelligent_recommendations_deterministic(
                evaluation_data, satisfiability_analysis, adaptive_results
            )
            
            # Generate final integrated output
            output = self._generate_deterministic_output(
                evaluation_data, initial_evaluation, satisfiability_analysis,
                adaptive_results, recommendations, operation_id
            )
            
            # Update metrics
            self._update_processing_metrics(output)
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            error_output = {
                "component": self.component_name,
                "error": str(e),
                "operation_id": operation_id,
                "status": "error",
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _extract_evaluation_data_deterministic(self, data: Dict[str, Any]) -> Dict[str, Any]:
# # #         """Extract evaluation data from input with stable ordering"""  # Module not found  # Module not found  # Module not found
        
        evaluation_data = {
            "document_content": "",
            "evaluation_questions": [],
            "metadata": {},
            "raw_data": data,
        }
        
        # Extract document content
        if "document" in data:
            doc = data["document"]
            if isinstance(doc, str):
                evaluation_data["document_content"] = doc
            elif isinstance(doc, dict):
                evaluation_data["document_content"] = doc.get("content", doc.get("text", ""))
                evaluation_data["metadata"] = doc.get("metadata", {})
        
        # Extract evaluation questions
        if "questions" in data:
            questions = data["questions"]
            if isinstance(questions, list):
                evaluation_data["evaluation_questions"] = questions
            elif isinstance(questions, dict):
                evaluation_data["evaluation_questions"] = list(questions.values())
        
        # Extract text directly
        if "text" in data:
            evaluation_data["document_content"] = str(data["text"])
        
        # Extract content directly
        if "content" in data:
            evaluation_data["document_content"] = str(data["content"])
        
        return evaluation_data
    
    def _perform_initial_evaluation_deterministic(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform initial evaluation with deterministic weighted calculation"""
        
        document_content = evaluation_data.get("document_content", "")
        questions = evaluation_data.get("evaluation_questions", [])
        
        # Initialize scoring structure
        dimension_scores = {}
        question_scores = {}
        total_weighted_score = 0.0
        total_weights = 0.0
        
        # If no questions provided, generate default evaluation
        if not questions:
            questions = self._generate_default_questions_deterministic()
        
        # Evaluate each question
        for question in questions:
            question_id = self._get_question_id(question)
            question_text = self._get_question_text(question)
            
            # Calculate question score
            score = self._evaluate_question_against_content_deterministic(
                question_text, document_content
            )
            
            question_scores[question_id] = score
            
            # Determine dimension and weight
            dimension = self._extract_dimension_from_question(question_id)
            weight = self._get_question_weight_deterministic(question_id, dimension)
            
            # Accumulate weighted scores by dimension
            if dimension not in dimension_scores:
                dimension_scores[dimension] = {"total_score": 0.0, "total_weight": 0.0}
            
            dimension_scores[dimension]["total_score"] += score * weight
            dimension_scores[dimension]["total_weight"] += weight
            
            total_weighted_score += score * weight
            total_weights += weight
        
        # Calculate dimension averages
        for dimension in dimension_scores:
            if dimension_scores[dimension]["total_weight"] > 0:
                dimension_scores[dimension]["avg_score"] = (
                    dimension_scores[dimension]["total_score"] / 
                    dimension_scores[dimension]["total_weight"]
                )
            else:
                dimension_scores[dimension]["avg_score"] = 0.0
        
        # Calculate overall score
        overall_score = total_weighted_score / total_weights if total_weights > 0 else 0.0
        
        initial_evaluation = {
            "dimension_scores": self.sort_dict_by_keys(dimension_scores),
            "overall_score": overall_score,
            "question_scores": self.sort_dict_by_keys(question_scores),
            "total_questions": len(questions),
            "evaluation_timestamp": self._get_deterministic_timestamp(),
        }
        
        return initial_evaluation
    
    def _determine_satisfiability_deterministic(self, initial_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Determine satisfiability using conjunctive logic with deterministic thresholds"""
        
        overall_score = initial_evaluation["overall_score"]
        dimension_scores = initial_evaluation["dimension_scores"]
        
        # Check overall satisfiability
        is_satisfiable = overall_score >= self.satisfaction_thresholds["minimum_threshold"]
        
        # Determine satisfaction level
        if overall_score >= self.satisfaction_thresholds["critical_threshold"]:
            satisfaction_level = "excellent"
        elif overall_score >= self.satisfaction_thresholds["high_threshold"]:
            satisfaction_level = "good"
        elif overall_score >= self.satisfaction_thresholds["medium_threshold"]:
            satisfaction_level = "fair"
        elif overall_score >= self.satisfaction_thresholds["minimum_threshold"]:
            satisfaction_level = "poor"
        else:
            satisfaction_level = "unsatisfactory"
        
        # Check dimension-level satisfiability
        dimension_satisfiability = {}
        for dimension, dim_data in dimension_scores.items():
            dim_score = dim_data.get("avg_score", 0.0)
            dimension_satisfiability[dimension] = {
                "is_satisfiable": dim_score >= self.satisfaction_thresholds["minimum_threshold"],
                "satisfaction_level": self._get_satisfaction_level(dim_score),
                "score": dim_score,
            }
        
        # Generate conjunctive analysis
        conjunctive_conditions = []
        unsatisfied_dimensions = []
        
        for dimension, satisfiability in dimension_satisfiability.items():
            if satisfiability["is_satisfiable"]:
                conjunctive_conditions.append(f"{dimension}: SATISFIED")
            else:
                conjunctive_conditions.append(f"{dimension}: NOT_SATISFIED")
                unsatisfied_dimensions.append(dimension)
        
        # Overall conjunctive result
        all_satisfied = len(unsatisfied_dimensions) == 0
        
        satisfiability_analysis = {
            "all_dimensions_satisfied": all_satisfied,
            "conjunctive_conditions": sorted(conjunctive_conditions),
            "dimension_satisfiability": self.sort_dict_by_keys(dimension_satisfiability),
            "is_satisfiable": is_satisfiable,
            "satisfaction_level": satisfaction_level,
            "unsatisfied_dimensions": sorted(unsatisfied_dimensions),
        }
        
        return satisfiability_analysis
    
    def _perform_adaptive_refinement_deterministic(self, evaluation_data: Dict[str, Any], initial_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Perform adaptive refinement using AdaptiveScoringEngine if available"""
        
        if not self.adaptive_scoring_engine:
            return {
                "adaptive_refinement_available": False,
                "message": "AdaptiveScoringEngine not available",
            }
        
        try:
            # Prepare data for adaptive scoring
            adaptive_input = {
                "document": evaluation_data.get("document_content", ""),
                "initial_scores": initial_evaluation["question_scores"],
                "dimension_scores": initial_evaluation["dimension_scores"],
            }
            
            # Perform adaptive scoring
            adaptive_results = self.adaptive_scoring_engine.process(adaptive_input)
            
            return {
                "adaptive_refinement_available": True,
                "adaptive_results": adaptive_results,
            }
            
        except Exception as e:
            logger.warning(f"Adaptive refinement failed: {e}")
            return {
                "adaptive_refinement_available": False,
                "error": str(e),
            }
    
    def _generate_intelligent_recommendations_deterministic(self, evaluation_data: Dict[str, Any], satisfiability_analysis: Dict[str, Any], adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent recommendations using IntelligentRecommendationEngine if available"""
        
        if not self.intelligent_recommendation_engine:
            return {
                "intelligent_recommendations_available": False,
                "message": "IntelligentRecommendationEngine not available",
            }
        
        try:
            # Prepare data for recommendation generation
            recommendation_input = {
                "document": evaluation_data.get("document_content", ""),
                "satisfiability": satisfiability_analysis,
                "adaptive_results": adaptive_results,
                "unsatisfied_dimensions": satisfiability_analysis.get("unsatisfied_dimensions", []),
            }
            
            # Generate recommendations
            recommendations = self.intelligent_recommendation_engine.process(recommendation_input)
            
            return {
                "intelligent_recommendations_available": True,
                "recommendations": recommendations,
            }
            
        except Exception as e:
            logger.warning(f"Intelligent recommendations failed: {e}")
            return {
                "intelligent_recommendations_available": False,
                "error": str(e),
            }
    
    def _generate_default_questions_deterministic(self) -> List[Dict[str, Any]]:
        """Generate default evaluation questions with deterministic ordering"""
        
        default_questions = [
            {"id": "DE1_Q1", "text": "¿El documento define productos medibles?", "dimension": "DE1"},
            {"id": "DE1_Q2", "text": "¿Las metas incluyen responsables institucionales?", "dimension": "DE1"},
            {"id": "DE2_Q1", "text": "¿Existe diagnóstico con línea base?", "dimension": "DE2"},
            {"id": "DE2_Q2", "text": "¿Se identifica la población objetivo?", "dimension": "DE2"},
            {"id": "DE3_Q1", "text": "¿Se especifica el presupuesto requerido?", "dimension": "DE3"},
            {"id": "DE3_Q2", "text": "¿Se identifican fuentes de financiamiento?", "dimension": "DE3"},
            {"id": "DE4_Q1", "text": "¿Se definen indicadores de resultado?", "dimension": "DE4"},
            {"id": "DE4_Q2", "text": "¿Se especifican metas cuantitativas?", "dimension": "DE4"},
        ]
        
        return sorted(default_questions, key=lambda x: x["id"])
    
    def _get_question_id(self, question: Any) -> str:
        """Extract question ID deterministically"""
        if isinstance(question, dict):
            return question.get("id", self.generate_stable_id(str(question), prefix="q"))
        else:
            return self.generate_stable_id(str(question), prefix="q")
    
    def _get_question_text(self, question: Any) -> str:
        """Extract question text deterministically"""
        if isinstance(question, dict):
            return question.get("text", question.get("question", str(question)))
        else:
            return str(question)
    
    def _evaluate_question_against_content_deterministic(self, question_text: str, document_content: str) -> float:
        """Evaluate a question against document content deterministically"""
        
        question_lower = question_text.lower()
        content_lower = document_content.lower()
        
# # #         # Extract key terms from question  # Module not found  # Module not found  # Module not found
        key_terms = self._extract_key_terms_from_question(question_lower)
        
        # Check for term presence in content
        matches = 0
        for term in key_terms:
            if term in content_lower:
                matches += 1
        
        # Calculate base score
        base_score = matches / len(key_terms) if key_terms else 0.0
        
        # Adjust based on content length and quality indicators
        if len(document_content) > 1000:
            base_score += 0.1  # Bonus for comprehensive content
        elif len(document_content) < 100:
            base_score -= 0.2  # Penalty for very short content
        
        # Check for quality indicators
        quality_indicators = ["específico", "detallado", "medible", "cuantificable"]
        quality_count = sum(1 for indicator in quality_indicators if indicator in content_lower)
        base_score += quality_count * 0.05
        
        return max(0.0, min(1.0, base_score))
    
    def _extract_key_terms_from_question(self, question_text: str) -> List[str]:
# # #         """Extract key terms from question deterministically"""  # Module not found  # Module not found  # Module not found
        
        # Remove question words and common terms
        stop_words = {"qué", "cuál", "cómo", "dónde", "cuándo", "por", "para", "con", "sin", "el", "la", "los", "las", "un", "una", "de", "del", "se", "es", "son", "está", "están"}
        
        # Simple tokenization
        import re
        words = re.findall(r'\b[a-záéíóúñ]+\b', question_text.lower())
        
        # Filter and sort
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return sorted(list(set(key_terms)))
    
    def _extract_dimension_from_question(self, question_id: str) -> str:
# # #         """Extract dimension from question ID"""  # Module not found  # Module not found  # Module not found
        for dim in ["DE1", "DE2", "DE3", "DE4"]:
            if dim in question_id:
                return dim
        return "DE1"  # Default
    
    def _get_question_weight_deterministic(self, question_id: str, dimension: str) -> float:
        """Get question weight deterministically"""
        
        # Check for specific question weight
        if question_id in self.question_weights:
            return self.question_weights[question_id]
        
        # Check for pattern-based weights
        for weight_key, weight_value in self.question_weights.items():
            if dimension in weight_key and any(term in question_id.lower() for term in ["critical", "high", "medium", "standard"]):
                return weight_value
        
        # Default weight based on dimension
        return self.dimension_weights.get(dimension, 0.5)
    
    def _get_satisfaction_level(self, score: float) -> str:
        """Get satisfaction level for a score"""
        if score >= self.satisfaction_thresholds["critical_threshold"]:
            return "excellent"
        elif score >= self.satisfaction_thresholds["high_threshold"]:
            return "good"
        elif score >= self.satisfaction_thresholds["medium_threshold"]:
            return "fair"
        elif score >= self.satisfaction_thresholds["minimum_threshold"]:
            return "poor"
        else:
            return "unsatisfactory"
    
    def _update_processing_metrics(self, output: Dict[str, Any]):
        """Update processing metrics"""
        self.processing_metrics["documents_processed"] += 1
        
# # #         # Extract overall score from results  # Module not found  # Module not found  # Module not found
        results = output.get("results", {})
        initial_eval = results.get("initial_evaluation", {})
        overall_score = initial_eval.get("overall_score", 0.0)
        
        # Update average score
        current_avg = self.processing_metrics["avg_evaluation_score"]
        count = self.processing_metrics["documents_processed"]
        new_avg = (current_avg * (count - 1) + overall_score) / count
        self.processing_metrics["avg_evaluation_score"] = new_avg
        
        # Update score categories
        if overall_score >= 0.7:
            self.processing_metrics["high_scoring_documents"] += 1
        elif overall_score < 0.5:
            self.processing_metrics["low_scoring_documents"] += 1
        
        # Update recommendations count
        recommendations = results.get("recommendations", {})
        if recommendations.get("intelligent_recommendations_available", False):
            self.processing_metrics["recommendations_generated"] += 1
        
        # Update timestamp
        self.processing_metrics["last_update"] = self._get_deterministic_timestamp()
    
    def _generate_deterministic_output(self, evaluation_data: Dict[str, Any], initial_evaluation: Dict[str, Any], satisfiability_analysis: Dict[str, Any], adaptive_results: Dict[str, Any], recommendations: Dict[str, Any], operation_id: str) -> Dict[str, Any]:
        """Generate deterministic output structure"""
        
        output = {
            "component": self.component_name,
            "component_id": self.component_id,
            "metadata": self.get_deterministic_metadata(),
            "operation_id": operation_id,
            "processing_metrics": self.sort_dict_by_keys(self.processing_metrics),
            "results": {
                "adaptive_refinement": adaptive_results,
                "evaluation_data": evaluation_data,
                "initial_evaluation": initial_evaluation,
                "recommendations": recommendations,
                "satisfiability_analysis": satisfiability_analysis,
            },
            "status": "success",
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(output)
    
    def save_artifact(self, data: Dict[str, Any], document_stem: str, output_dir: str = "canonical_flow/analysis") -> str:
        """
        Save evaluation processor output to canonical_flow/analysis/ directory with standardized naming.
        
        Args:
            data: Evaluation data to save
            document_stem: Base document identifier (without extension)
            output_dir: Output directory (defaults to canonical_flow/analysis)
            
        Returns:
            Path to saved artifact
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate standardized filename
            filename = f"{document_stem}_evaluation.json"
            artifact_path = Path(output_dir) / filename
            
            # Save with consistent JSON formatting
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"EvaluationDrivenProcessor artifact saved: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            error_msg = f"Failed to save EvaluationDrivenProcessor artifact to {output_dir}/{document_stem}_evaluation.json: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Backward compatibility functions
def process_evaluation(document: str, questions: List[Dict] = None) -> Dict[str, Any]:
    """Process evaluation for document and questions"""
    processor = EvaluationDrivenProcessor()
    data = {"document": document}
    if questions:
        data["questions"] = questions
    return processor.process(data)


def process(data=None, context=None):
    """Backward compatible process function"""
    processor = EvaluationDrivenProcessor()
    result = processor.process(data, context)
    
    # Save artifact if document_stem is provided
    if data and isinstance(data, dict) and 'document_stem' in data:
        processor.save_artifact(result, data['document_stem'])
    
    return result