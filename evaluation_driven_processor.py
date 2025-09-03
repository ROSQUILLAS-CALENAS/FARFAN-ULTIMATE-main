"""
Evaluation Driven Processor
Standardized process() method with bounded metrics (0-1) and deterministic aggregation
"""

import json
import logging
import math
import time
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found
# # # from decimal import Decimal, getcontext, ROUND_HALF_UP  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any, Optional, Union, Tuple  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)

# Set decimal context for consistent floating-point precision
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_UP


class EvaluationDrivenProcessor:
    """
    Standardized evaluation processor with bounded metrics and deterministic aggregation.
    
    This processor implements a standardized process() method that takes structured evidence
    as input and returns evaluation metrics with scores bounded between 0 and 1. It ensures
    deterministic aggregation through stable sorting and consistent floating-point precision.
    
    Evaluation Formulas Documentation:
    =================================
    
    1. Weighted Score Calculation:
       weighted_score = raw_score * weight_factor
       Formula: ws = s * w, where s ∈ [0,1] and w ∈ [0,∞)
    
    2. Bounded Normalization:
       normalized_score = min(1.0, max(0.0, score))
       Formula: ns = max(0, min(1, s))
    
    3. Dimension Aggregation:
       dimension_score = Σ(weighted_scores) / Σ(weights)
       Formula: ds = (Σᵢ sᵢ * wᵢ) / Σᵢ wᵢ
    
    4. Confidence Score:
       confidence = 1.0 - (standard_deviation / mean_score)
       Formula: c = 1 - (σ / μ), bounded to [0,1]
    
    5. Completeness Score:
       completeness = valid_evidence_count / total_expected_evidence
       Formula: comp = |V| / |E|, where V ⊆ E
    
    6. Consistency Score:
       consistency = 1.0 - (variance_across_dimensions / max_possible_variance)
       Formula: cons = 1 - (Var(D) / Var_max)
    
    7. Global Aggregation (Harmonic Mean):
       global_score = n / Σ(1/scoreᵢ)
       Formula: G = n / Σᵢ (1/sᵢ), where sᵢ > 0
    """
    
    def __init__(self, output_path: str = "canonical_flow/analysis"):
        """
        Initialize processor with standardized configuration.
        
        Args:
            output_path: Path for JSON artifact generation
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Standardized weights with deterministic ordering
        self.dimension_weights = OrderedDict([
            ('completeness', Decimal('0.25')),
            ('consistency', Decimal('0.25')),  
            ('reliability', Decimal('0.30')),
            ('validity', Decimal('0.20'))
        ])
        
        self.metric_weights = OrderedDict([
            ('accuracy', Decimal('0.30')),
            ('precision', Decimal('0.25')),
            ('recall', Decimal('0.25')),
            ('f1_score', Decimal('0.20'))
        ])
        
        # Processing statistics for deterministic reporting
        self.processing_stats = {
            'total_processed': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_processing_time': 0.0
        }
        
    def process(self, structured_evidence: Dict[str, Any], 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Standardized process method with bounded metrics (0-1) and deterministic aggregation.
        
        This method implements the core evaluation pipeline with the following guarantees:
        - All output scores are bounded between 0.0 and 1.0
        - Aggregation is deterministic with stable sorting
        - Consistent floating-point precision handling
        - Comprehensive error handling for edge cases
        
        Args:
            structured_evidence: Dictionary containing evidence data with structure:
                {
                    'evidence_items': List[Dict],
                    'metadata': Dict,
                    'quality_indicators': Dict (optional)
                }
            context: Optional processing context and parameters
            
        Returns:
            Dict containing evaluation metrics with deterministic structure:
                {
                    'scores': Dict[str, float],  # All values ∈ [0,1]
                    'metrics': Dict[str, float],  # All values ∈ [0,1] 
                    'metadata': Dict[str, Any],
                    'confidence': float,  # ∈ [0,1]
                    'processing_info': Dict
                }
            
        Raises:
            ValueError: For invalid evidence structure
            TypeError: For incompatible data types
        """
        # Audit logging for component execution
        audit_logger = get_audit_logger() if get_audit_logger else None
        input_data = {
            "municipality_name": pdt_context.municipality_name,
            "department": pdt_context.department,
            "documents_count": len(document_package.documents),
            "questions_count": len(evaluation_questions)
        }
        
        if audit_logger:
            with audit_logger.audit_component_execution("19A", input_data) as audit_ctx:
                result = self._process_pdt_evaluation_internal(document_package, pdt_context, evaluation_questions)
                audit_ctx.set_output({
                    "processing_success": True,
                    "adaptive_scores_calculated": hasattr(result, 'adaptive_scores'),
                    "recommendations_generated": hasattr(result, 'intelligent_recommendations')
                })
                return result
        else:
            return self._process_pdt_evaluation_internal(document_package, pdt_context, evaluation_questions)

    def _process_pdt_evaluation_internal(self, 
                                        document_package: DocumentPackage,
                                        pdt_context: PDTContext,
                                        evaluation_questions: List[Dict[str, Any]]) -> DocumentPackage:
        """Internal implementation of PDT evaluation processing."""
        start_time = time.time()
        processing_id = self._generate_processing_id()
        
        logger.info(f"Starting standardized evaluation process {processing_id}")
        
        try:
            # Validate input structure
            self._validate_evidence_structure(structured_evidence)
            
            # Extract and normalize evidence items
            evidence_items = structured_evidence.get('evidence_items', [])
            metadata = structured_evidence.get('metadata', {})
            quality_indicators = structured_evidence.get('quality_indicators', {})
            
            # Handle edge case: empty evidence
            if not evidence_items:
                return self._create_empty_result(processing_id, start_time)
                
            # Core evaluation pipeline
            dimension_scores = self._calculate_dimension_scores(evidence_items, quality_indicators)
            metric_scores = self._calculate_metric_scores(evidence_items)
            confidence_score = self._calculate_confidence_score(dimension_scores, metric_scores)
            
            # Create standardized result with deterministic ordering
            result = self._create_standardized_result(
                processing_id=processing_id,
                dimension_scores=dimension_scores,
                metric_scores=metric_scores,
                confidence_score=confidence_score,
                metadata=metadata,
                processing_time=time.time() - start_time
            )
            
            # Generate JSON artifact
            self._generate_json_artifact(result, processing_id)
            
            # Update processing statistics
            self._update_processing_stats(time.time() - start_time, success=True)
            
            logger.info(f"Evaluation process {processing_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation process {processing_id} failed: {str(e)}")
            self._update_processing_stats(time.time() - start_time, success=False)
            
            # Return error result with bounded values
            return self._create_error_result(processing_id, str(e), start_time)
    
    def _validate_evidence_structure(self, structured_evidence: Dict[str, Any]) -> None:
        """
        Validate evidence structure and handle malformed input.
        
        Validation Rules:
        - Must be a dictionary
        - Must contain 'evidence_items' key
        - 'evidence_items' must be a list
        - Each evidence item must be a dictionary with required fields
        
        Args:
            structured_evidence: Input evidence to validate
            
        Raises:
            ValueError: If structure is invalid
            TypeError: If types are incompatible
        """
        if not isinstance(structured_evidence, dict):
            raise TypeError("structured_evidence must be a dictionary")
            
        if 'evidence_items' not in structured_evidence:
            raise ValueError("structured_evidence must contain 'evidence_items' key")
            
        evidence_items = structured_evidence['evidence_items']
        if not isinstance(evidence_items, list):
            raise TypeError("'evidence_items' must be a list")
            
        # Validate each evidence item
        for i, item in enumerate(evidence_items):
            if not isinstance(item, dict):
                raise TypeError(f"Evidence item {i} must be a dictionary")
                
            # Check for required fields
            required_fields = ['score', 'weight']
            for field in required_fields:
                if field not in item:
                    logger.warning(f"Evidence item {i} missing field '{field}', using default")
                    
    def _calculate_dimension_scores(self, evidence_items: List[Dict], 
                                  quality_indicators: Dict[str, Any]) -> OrderedDict:
        """
        Calculate dimension scores with bounded outputs and deterministic aggregation.
        
        Implements weighted aggregation formula:
        dimension_score = Σ(evidence_score * weight * quality_factor) / Σ(weight * quality_factor)
        
        Where:
        - evidence_score ∈ [0,1]
        - weight > 0
# # #         - quality_factor ∈ [0,1] (derived from quality indicators)  # Module not found  # Module not found  # Module not found
        
        Args:
            evidence_items: List of evidence dictionaries
            quality_indicators: Quality assessment data
            
        Returns:
            OrderedDict with dimension scores, all values ∈ [0,1]
        """
        dimension_scores = OrderedDict()
        
        # Group evidence by dimension with stable sorting
        evidence_by_dimension = self._group_evidence_by_dimension(evidence_items)
        
        for dimension_name in sorted(self.dimension_weights.keys()):
            dimension_evidence = evidence_by_dimension.get(dimension_name, [])
            
            if not dimension_evidence:
                # Handle missing evidence case
                dimension_scores[dimension_name] = 0.0
                logger.warning(f"No evidence found for dimension '{dimension_name}'")
                continue
                
            # Calculate weighted score with precision handling
            weighted_sum = Decimal('0')
            weight_sum = Decimal('0')
            
            for evidence in dimension_evidence:
                # Extract and normalize scores
                raw_score = self._extract_numeric_value(evidence.get('score', 0))
                weight = self._extract_numeric_value(evidence.get('weight', 1))
                
                # Apply quality factor
                quality_factor = self._calculate_quality_factor(evidence, quality_indicators)
                
                # Bounded normalization
                normalized_score = max(0, min(1, raw_score))
                effective_weight = max(Decimal('0.001'), Decimal(str(weight))) * Decimal(str(quality_factor))
                
                weighted_sum += Decimal(str(normalized_score)) * effective_weight
                weight_sum += effective_weight
            
            # Calculate dimension score with division by zero protection
            if weight_sum > 0:
                dimension_score = float(weighted_sum / weight_sum)
            else:
                dimension_score = 0.0
                
            # Ensure bounded output
            dimension_scores[dimension_name] = max(0.0, min(1.0, dimension_score))
            
        return dimension_scores
    
    def _calculate_metric_scores(self, evidence_items: List[Dict]) -> OrderedDict:
        """
        Calculate standardized metric scores with statistical validation.
        
        Implements the following metric calculations:
        1. Accuracy = (TP + TN) / (TP + TN + FP + FN)
        2. Precision = TP / (TP + FP)  
        3. Recall = TP / (TP + FN)
        4. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        
# # #         Where TP, TN, FP, FN are derived from evidence classification.  # Module not found  # Module not found  # Module not found
        
        Args:
            evidence_items: List of evidence dictionaries
            
        Returns:
            OrderedDict with metric scores, all values ∈ [0,1]
        """
        metric_scores = OrderedDict()
        
# # #         # Extract classification statistics from evidence  # Module not found  # Module not found  # Module not found
        tp, tn, fp, fn = self._extract_classification_stats(evidence_items)
        
        # Calculate accuracy with division by zero protection
        total = tp + tn + fp + fn
        if total > 0:
            accuracy = (tp + tn) / total
        else:
            accuracy = 0.0
            
        # Calculate precision with division by zero protection  
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
            
        # Calculate recall with division by zero protection
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
            
        # Calculate F1-score with division by zero protection
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Store with deterministic ordering and bounded values
        metric_scores['accuracy'] = max(0.0, min(1.0, accuracy))
        metric_scores['precision'] = max(0.0, min(1.0, precision)) 
        metric_scores['recall'] = max(0.0, min(1.0, recall))
        metric_scores['f1_score'] = max(0.0, min(1.0, f1_score))
        
        return metric_scores
    
    def _calculate_confidence_score(self, dimension_scores: OrderedDict, 
                                  metric_scores: OrderedDict) -> float:
        """
        Calculate overall confidence score using statistical measures.
        
        Confidence formula:
        confidence = (1 - coefficient_of_variation) * completeness_factor * consistency_factor
        
        Where:
        - coefficient_of_variation = std_dev / mean
        - completeness_factor = non_zero_scores / total_scores  
        - consistency_factor = 1 - variance_penalty
        
        Args:
            dimension_scores: Calculated dimension scores
            metric_scores: Calculated metric scores
            
        Returns:
            Confidence score ∈ [0,1]
        """
        all_scores = list(dimension_scores.values()) + list(metric_scores.values())
        
        if not all_scores:
            return 0.0
            
        # Calculate statistical measures with precision handling
        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
        std_dev = math.sqrt(variance)
        
        # Coefficient of variation (bounded)
        if mean_score > 0:
            cv = std_dev / mean_score
        else:
            cv = 1.0  # Maximum uncertainty for zero mean
            
        cv_factor = max(0.0, 1.0 - cv)
        
        # Completeness factor
        non_zero_count = sum(1 for score in all_scores if score > 0.001)
        completeness_factor = non_zero_count / len(all_scores)
        
        # Consistency factor (penalize high variance)
        max_possible_variance = 0.25  # Maximum variance for uniform distribution over [0,1]
        consistency_factor = max(0.0, 1.0 - (variance / max_possible_variance))
        
        # Combined confidence score
        confidence = cv_factor * completeness_factor * consistency_factor
        
        return max(0.0, min(1.0, confidence))
    
    def _group_evidence_by_dimension(self, evidence_items: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group evidence items by dimension with stable sorting.
        
        Args:
            evidence_items: List of evidence dictionaries
            
        Returns:
            Dictionary mapping dimension names to evidence lists
        """
        grouped = {}
        
        for evidence in evidence_items:
            dimension = evidence.get('dimension', 'validity')  # Default dimension
            
            if dimension not in grouped:
                grouped[dimension] = []
                
            grouped[dimension].append(evidence)
        
        # Sort each group for deterministic processing
        for dimension in grouped:
            grouped[dimension].sort(key=lambda x: (
                x.get('timestamp', ''),
                x.get('id', ''),
                str(x)  # Fallback for complete determinism
            ))
            
        return grouped
    
    def _extract_classification_stats(self, evidence_items: List[Dict]) -> Tuple[int, int, int, int]:
        """
# # #         Extract classification statistics (TP, TN, FP, FN) from evidence.  # Module not found  # Module not found  # Module not found
        
        Args:
            evidence_items: List of evidence dictionaries
            
        Returns:
            Tuple of (true_positives, true_negatives, false_positives, false_negatives)
        """
        tp = tn = fp = fn = 0
        
        for evidence in evidence_items:
            classification = evidence.get('classification', {})
            
            tp += int(classification.get('true_positive', 0))
            tn += int(classification.get('true_negative', 0))
            fp += int(classification.get('false_positive', 0))
            fn += int(classification.get('false_negative', 0))
        
        return tp, tn, fp, fn
    
    def _calculate_quality_factor(self, evidence: Dict, quality_indicators: Dict[str, Any]) -> float:
        """
        Calculate quality factor for evidence item.
        
        Quality formula:
        quality_factor = (source_reliability + data_freshness + validation_score) / 3
        
        Args:
            evidence: Single evidence dictionary
            quality_indicators: Overall quality assessment data
            
        Returns:
            Quality factor ∈ [0,1]
        """
        # Extract quality components
        source_reliability = quality_indicators.get('source_reliability', 0.8)
        
        # Data freshness based on timestamp
        timestamp = evidence.get('timestamp')
        if timestamp:
            freshness = self._calculate_freshness_score(timestamp)
        else:
            freshness = 0.5  # Neutral score for missing timestamp
            
# # #         # Validation score from evidence  # Module not found  # Module not found  # Module not found
        validation_score = evidence.get('validation_score', 0.7)
        
        # Weighted combination
        quality_factor = (
            0.4 * source_reliability +
            0.3 * freshness +
            0.3 * validation_score
        )
        
        return max(0.0, min(1.0, quality_factor))
    
    def _calculate_freshness_score(self, timestamp: str) -> float:
        """
        Calculate data freshness score based on timestamp.
        
        Freshness formula:
        freshness_score = exp(-age_in_days / decay_constant)
        
        Where decay_constant determines how quickly freshness degrades.
        
        Args:
            timestamp: ISO format timestamp string
            
        Returns:
            Freshness score ∈ [0,1]
        """
        try:
            evidence_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            current_time = datetime.now(evidence_time.tzinfo)
            
            age_days = (current_time - evidence_time).total_seconds() / (24 * 3600)
            decay_constant = 30.0  # 30-day half-life
            
            freshness_score = math.exp(-age_days / decay_constant)
            
            return max(0.0, min(1.0, freshness_score))
            
        except (ValueError, TypeError):
            return 0.5  # Neutral score for invalid timestamp
    
    def _extract_numeric_value(self, value: Any) -> float:
        """
        Extract numeric value with error handling.
        
        Args:
            value: Input value (int, float, string, or other)
            
        Returns:
            Numeric value with fallback to 0.0
        """
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                logger.warning(f"Cannot convert {type(value)} to float, using 0.0")
                return 0.0
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric value '{value}', using 0.0")
            return 0.0
    
    def _create_standardized_result(self, processing_id: str, dimension_scores: OrderedDict,
                                  metric_scores: OrderedDict, confidence_score: float,
                                  metadata: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """
        Create standardized result structure with deterministic ordering.
        
        Args:
            processing_id: Unique processing identifier
            dimension_scores: Calculated dimension scores
            metric_scores: Calculated metric scores  
            confidence_score: Overall confidence score
            metadata: Input metadata
            processing_time: Processing duration in seconds
            
        Returns:
            Standardized result dictionary with bounded values
        """
        # Calculate global score using harmonic mean for balanced aggregation
        all_scores = list(dimension_scores.values()) + list(metric_scores.values())
        non_zero_scores = [score for score in all_scores if score > 0.001]
        
        if non_zero_scores:
            harmonic_mean = len(non_zero_scores) / sum(1/score for score in non_zero_scores)
            global_score = max(0.0, min(1.0, harmonic_mean))
        else:
            global_score = 0.0
        
        return OrderedDict([
            ('processing_id', processing_id),
            ('scores', OrderedDict([
                ('dimensions', dimension_scores),
                ('metrics', metric_scores), 
                ('global', global_score)
            ])),
            ('confidence', confidence_score),
            ('metadata', OrderedDict([
                ('input_metadata', metadata),
                ('processing_timestamp', datetime.now().isoformat()),
                ('processing_time_seconds', round(processing_time, 6)),
                ('processor_version', '2.0.0'),
                ('evaluation_method', 'standardized_bounded_aggregation')
            ])),
            ('processing_info', OrderedDict([
                ('total_evidence_items', metadata.get('evidence_count', 0)),
                ('dimensions_evaluated', len(dimension_scores)),
                ('metrics_calculated', len(metric_scores)),
                ('bounded_scores', True),
                ('deterministic_aggregation', True)
            ]))
        ])
    
    def _create_empty_result(self, processing_id: str, start_time: float) -> Dict[str, Any]:
        """
        Create result for empty evidence input.
        
        Args:
            processing_id: Unique processing identifier
            start_time: Processing start timestamp
            
        Returns:
            Empty result with zero scores
        """
        empty_dimensions = OrderedDict([(dim, 0.0) for dim in self.dimension_weights.keys()])
        empty_metrics = OrderedDict([(metric, 0.0) for metric in self.metric_weights.keys()])
        
        return self._create_standardized_result(
            processing_id=processing_id,
            dimension_scores=empty_dimensions,
            metric_scores=empty_metrics,
            confidence_score=0.0,
            metadata={'evidence_count': 0, 'status': 'empty_input'},
            processing_time=time.time() - start_time
        )
    
    def _create_error_result(self, processing_id: str, error_message: str, 
                           start_time: float) -> Dict[str, Any]:
        """
        Create result for error cases.
        
        Args:
            processing_id: Unique processing identifier
            error_message: Error description
            start_time: Processing start timestamp
            
        Returns:
            Error result with bounded zero scores
        """
        error_dimensions = OrderedDict([(dim, 0.0) for dim in self.dimension_weights.keys()])
        error_metrics = OrderedDict([(metric, 0.0) for metric in self.metric_weights.keys()])
        
        return self._create_standardized_result(
            processing_id=processing_id,
            dimension_scores=error_dimensions,
            metric_scores=error_metrics,
            confidence_score=0.0,
            metadata={
                'evidence_count': 0, 
                'status': 'error',
                'error_message': error_message
            },
            processing_time=time.time() - start_time
        )
    
    def _generate_json_artifact(self, result: Dict[str, Any], processing_id: str) -> None:
        """
        Generate JSON artifact with deterministic ordering.
        
        Args:
            result: Evaluation result to serialize
            processing_id: Unique processing identifier
        """
        try:
            output_file = self.output_path / "evaluation_driven_processor_evaluation.json"
            
            # Load existing results if file exists
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {'evaluations': []}
            
            # Add new result with timestamp-based sorting
            existing_data['evaluations'].append(result)
            
            # Sort by processing timestamp for deterministic ordering
            existing_data['evaluations'].sort(
                key=lambda x: x.get('metadata', {}).get('processing_timestamp', '')
            )
            
            # Add summary statistics
            existing_data['summary'] = self._generate_summary_statistics(existing_data['evaluations'])
            
            # Write with consistent formatting
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, sort_keys=True, ensure_ascii=False)
                
            logger.info(f"JSON artifact generated: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate JSON artifact: {str(e)}")
    
    def _generate_summary_statistics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for deterministic reporting.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Summary statistics dictionary
        """
        if not evaluations:
            return {'total_evaluations': 0}
        
        # Extract global scores for statistics
        global_scores = []
        confidence_scores = []
        
        for eval_result in evaluations:
            scores = eval_result.get('scores', {})
            global_score = scores.get('global', 0.0)
            confidence = eval_result.get('confidence', 0.0)
            
            global_scores.append(global_score)
            confidence_scores.append(confidence)
        
        # Calculate statistics with precision handling
        return OrderedDict([
            ('total_evaluations', len(evaluations)),
            ('global_score_stats', OrderedDict([
                ('mean', round(sum(global_scores) / len(global_scores), 6)),
                ('min', round(min(global_scores), 6)),
                ('max', round(max(global_scores), 6)),
                ('std_dev', round(math.sqrt(sum((x - sum(global_scores)/len(global_scores))**2 
                                              for x in global_scores) / len(global_scores)), 6))
            ])),
            ('confidence_stats', OrderedDict([
                ('mean', round(sum(confidence_scores) / len(confidence_scores), 6)),
                ('min', round(min(confidence_scores), 6)),
                ('max', round(max(confidence_scores), 6))
            ])),
            ('last_updated', datetime.now().isoformat())
        ])
    
    def _generate_processing_id(self) -> str:
        """
        Generate unique processing identifier.
        
        Returns:
            Unique processing ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"eval_proc_{timestamp}"
    
    def _update_processing_stats(self, processing_time: float, success: bool) -> None:
        """
        Update processing statistics.
        
        Args:
            processing_time: Processing duration in seconds
            success: Whether processing was successful
        """
        self.processing_stats['total_processed'] += 1
        
        if success:
            self.processing_stats['successful_evaluations'] += 1
        else:
            self.processing_stats['failed_evaluations'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_processed']
        current_avg = self.processing_stats['average_processing_time']
        self.processing_stats['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        total = self.processing_stats['total_processed']
        success_rate = (
            self.processing_stats['successful_evaluations'] / total
            if total > 0 else 0.0
        )
        
        return OrderedDict([
            ('total_processed', total),
            ('successful_evaluations', self.processing_stats['successful_evaluations']),
            ('failed_evaluations', self.processing_stats['failed_evaluations']),
            ('success_rate', round(success_rate, 6)),
            ('average_processing_time', round(self.processing_stats['average_processing_time'], 6))
        ])