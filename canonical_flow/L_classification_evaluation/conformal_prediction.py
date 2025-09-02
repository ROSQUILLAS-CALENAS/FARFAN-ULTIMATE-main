"""
Conformal Prediction Module for Calibrated Confidence Intervals

Implements split conformal prediction with bootstrapped variance estimation for 
dimension and point-level scores, generating calibrated confidence intervals
at configurable confidence levels (90%, 95%, 99%) for the Decálogo scoring system.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
import numpy as np
from scipy import stats
import collections

from .decalogo_scoring_system import ScoringSystem, PointScore, DimensionScore

logger = logging.getLogger(__name__)


@dataclass
class ConformalInterval:
    """Conformal prediction interval with bounds and metadata"""
    lower_bound: float
    upper_bound: float
    prediction_interval: float  # upper - lower
    coverage_probability: float
    confidence_level: float
    nonconformity_score: float


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics for conformal predictions"""
    empirical_coverage: float
    expected_coverage: float
    coverage_gap: float  # |empirical - expected|
    interval_width_mean: float
    interval_width_std: float
    nonconformity_scores_mean: float
    nonconformity_scores_std: float
    n_calibration_samples: int


@dataclass
class ConformalPrediction:
    """Complete conformal prediction result for a dimension or point"""
    prediction_id: str
    predicted_score: float
    confidence_intervals: Dict[str, ConformalInterval]  # "90%", "95%", "99%"
    calibration_metrics: CalibrationMetrics
    bootstrap_estimates: Dict[str, float]  # variance estimates


@dataclass
class DocumentConformalResults:
    """Complete conformal prediction results for a document"""
    document_id: str
    point_predictions: Dict[str, ConformalPrediction]  # "P1", "P2", etc.
    dimension_predictions: Dict[str, ConformalPrediction]  # "P1_DE-1", etc.
    aggregated_predictions: Dict[str, ConformalPrediction]  # "overall_point_score"
    generation_timestamp: str
    calibration_dataset_size: int
    confidence_levels: List[float]


class ConformalPredictor:
    """
    Split conformal prediction implementation for Decálogo scoring with
    bootstrapped variance estimation and calibrated confidence intervals.
    """
    
    CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
    BOOTSTRAP_SAMPLES = 1000
    DEFAULT_CALIBRATION_SPLIT = 0.3
    
    def __init__(
        self,
        scoring_system: Optional[ScoringSystem] = None,
        calibration_split: float = DEFAULT_CALIBRATION_SPLIT,
        bootstrap_samples: int = BOOTSTRAP_SAMPLES,
        confidence_levels: Optional[List[float]] = None,
        random_seed: int = 42
    ):
        """
        Initialize conformal predictor.
        
        Args:
            scoring_system: Decálogo scoring system instance
            calibration_split: Fraction of data to use for calibration
            bootstrap_samples: Number of bootstrap samples for variance estimation
            confidence_levels: List of confidence levels to generate intervals for
            random_seed: Random seed for reproducibility
        """
        self.scoring_system = scoring_system or ScoringSystem()
        self.calibration_split = calibration_split
        self.bootstrap_samples = bootstrap_samples
        self.confidence_levels = confidence_levels or self.CONFIDENCE_LEVELS
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
    def compute_nonconformity_score(
        self,
        predicted_score: float,
        true_score: float,
        score_type: str = "absolute"
    ) -> float:
        """
        Compute nonconformity score between predicted and true scores.
        
        Args:
            predicted_score: Model prediction
            true_score: Ground truth score
            score_type: Type of nonconformity score ("absolute", "normalized")
            
        Returns:
            Nonconformity score
        """
        if score_type == "absolute":
            return abs(predicted_score - true_score)
        elif score_type == "normalized":
            # Normalized by the predicted score (avoid division by zero)
            if predicted_score == 0:
                return abs(true_score)
            return abs(predicted_score - true_score) / abs(predicted_score)
        else:
            raise ValueError(f"Unknown nonconformity score type: {score_type}")
    
    def fit_calibration_data(
        self,
        calibration_scores: List[Tuple[float, float]]
    ) -> List[float]:
        """
        Fit calibration data and compute nonconformity scores.
        
        Args:
            calibration_scores: List of (predicted_score, true_score) pairs
            
        Returns:
            List of nonconformity scores
        """
        nonconformity_scores = []
        
        for predicted, true in calibration_scores:
            nonconformity = self.compute_nonconformity_score(predicted, true)
            nonconformity_scores.append(nonconformity)
        
        return nonconformity_scores
    
    def compute_quantile(
        self,
        nonconformity_scores: List[float],
        confidence_level: float
    ) -> float:
        """
        Compute conformal quantile for given confidence level.
        
        Args:
            nonconformity_scores: Calibration nonconformity scores
            confidence_level: Target confidence level
            
        Returns:
            Quantile value for prediction intervals
        """
        n = len(nonconformity_scores)
        if n == 0:
            logger.warning("No calibration data available")
            return 0.0
        
        # Conformal quantile: ceil((n+1) * confidence_level) / n
        rank = int(np.ceil((n + 1) * confidence_level))
        rank = min(rank, n)  # Cap at maximum available rank
        
        sorted_scores = sorted(nonconformity_scores)
        return sorted_scores[rank - 1]  # Convert to 0-based indexing
    
    def bootstrap_variance_estimation(
        self,
        scores: List[float],
        n_bootstrap: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Bootstrap variance estimation for score uncertainty.
        
        Args:
            scores: List of scores for bootstrap sampling
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with variance statistics
        """
        n_bootstrap = n_bootstrap or self.bootstrap_samples
        if len(scores) < 2:
            return {
                "mean": scores[0] if scores else 0.0,
                "std": 0.0,
                "var": 0.0,
                "ci_lower": scores[0] if scores else 0.0,
                "ci_upper": scores[0] if scores else 0.0
            }
        
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        return {
            "mean": float(np.mean(bootstrap_means)),
            "std": float(np.std(bootstrap_means)),
            "var": float(np.var(bootstrap_means)),
            "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
            "ci_upper": float(np.percentile(bootstrap_means, 97.5))
        }
    
    def generate_prediction_intervals(
        self,
        predicted_score: float,
        nonconformity_scores: List[float],
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, ConformalInterval]:
        """
        Generate conformal prediction intervals at multiple confidence levels.
        
        Args:
            predicted_score: Point prediction
            nonconformity_scores: Calibration nonconformity scores
            confidence_levels: Target confidence levels
            
        Returns:
            Dictionary mapping confidence levels to intervals
        """
        confidence_levels = confidence_levels or self.confidence_levels
        intervals = {}
        
        for level in confidence_levels:
            quantile = self.compute_quantile(nonconformity_scores, level)
            
            lower_bound = max(0.0, predicted_score - quantile)  # Scores >= 0
            upper_bound = min(1.0, predicted_score + quantile)  # Scores <= 1
            
            interval = ConformalInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                prediction_interval=upper_bound - lower_bound,
                coverage_probability=level,
                confidence_level=level,
                nonconformity_score=quantile
            )
            
            intervals[f"{int(level * 100)}%"] = interval
        
        return intervals
    
    def evaluate_calibration_quality(
        self,
        test_predictions: List[Tuple[float, float, Dict[str, ConformalInterval]]],
        confidence_levels: Optional[List[float]] = None
    ) -> CalibrationMetrics:
        """
        Evaluate calibration quality on test data.
        
        Args:
            test_predictions: List of (predicted, true, intervals) tuples
            confidence_levels: Confidence levels to evaluate
            
        Returns:
            Calibration quality metrics
        """
        confidence_levels = confidence_levels or self.confidence_levels
        
        coverage_results = {}
        interval_widths = {f"{int(level * 100)}%": [] for level in confidence_levels}
        nonconformity_scores = []
        
        for predicted, true, intervals in test_predictions:
            nonconformity_scores.append(abs(predicted - true))
            
            for level in confidence_levels:
                level_key = f"{int(level * 100)}%"
                if level_key in intervals:
                    interval = intervals[level_key]
                    is_covered = interval.lower_bound <= true <= interval.upper_bound
                    
                    if level_key not in coverage_results:
                        coverage_results[level_key] = []
                    coverage_results[level_key].append(is_covered)
                    interval_widths[level_key].append(interval.prediction_interval)
        
        # Compute empirical coverage for primary confidence level (95%)
        primary_level = "95%"
        empirical_coverage = (
            np.mean(coverage_results[primary_level]) if primary_level in coverage_results
            else 0.0
        )
        expected_coverage = 0.95
        coverage_gap = abs(empirical_coverage - expected_coverage)
        
        # Compute interval width statistics
        all_widths = []
        for widths in interval_widths.values():
            all_widths.extend(widths)
        
        return CalibrationMetrics(
            empirical_coverage=empirical_coverage,
            expected_coverage=expected_coverage,
            coverage_gap=coverage_gap,
            interval_width_mean=float(np.mean(all_widths)) if all_widths else 0.0,
            interval_width_std=float(np.std(all_widths)) if all_widths else 0.0,
            nonconformity_scores_mean=float(np.mean(nonconformity_scores)),
            nonconformity_scores_std=float(np.std(nonconformity_scores)),
            n_calibration_samples=len(test_predictions)
        )
    
    def predict_with_intervals(
        self,
        predicted_score: float,
        calibration_data: List[Tuple[float, float]],
        prediction_id: str
    ) -> ConformalPrediction:
        """
        Generate conformal prediction with intervals and calibration metrics.
        
        Args:
            predicted_score: Point prediction
            calibration_data: Calibration dataset as (predicted, true) pairs
            prediction_id: Unique identifier for this prediction
            
        Returns:
            Complete conformal prediction result
        """
        # Fit nonconformity scores
        nonconformity_scores = self.fit_calibration_data(calibration_data)
        
        # Generate prediction intervals
        intervals = self.generate_prediction_intervals(
            predicted_score, nonconformity_scores
        )
        
        # Bootstrap variance estimation
        calibration_scores = [true for _, true in calibration_data]
        bootstrap_estimates = self.bootstrap_variance_estimation(calibration_scores)
        
        # Evaluate calibration on calibration data (placeholder metrics)
        test_predictions = [
            (pred, true, intervals) for pred, true in calibration_data
        ]
        calibration_metrics = self.evaluate_calibration_quality(test_predictions)
        
        return ConformalPrediction(
            prediction_id=prediction_id,
            predicted_score=predicted_score,
            confidence_intervals=intervals,
            calibration_metrics=calibration_metrics,
            bootstrap_estimates=bootstrap_estimates
        )
    
    def process_document_predictions(
        self,
        document_id: str,
        point_scores: Dict[str, float],  # "P1": score, "P2": score, etc.
        dimension_scores: Dict[str, float],  # "P1_DE-1": score, etc.
        calibration_dataset: Dict[str, List[Tuple[float, float]]],
        output_dir: str = "canonical_flow/classification"
    ) -> DocumentConformalResults:
        """
        Process conformal predictions for all dimensions and points in a document.
        
        Args:
            document_id: Document identifier
            point_scores: Dictionary of point-level scores
            dimension_scores: Dictionary of dimension-level scores  
            calibration_dataset: Calibration data for each prediction type
            output_dir: Output directory for results
            
        Returns:
            Complete document conformal results
        """
        # Initialize results
        point_predictions = {}
        dimension_predictions = {}
        aggregated_predictions = {}
        
        # Process point-level predictions
        for point_id, score in point_scores.items():
            calibration_data = calibration_dataset.get(f"point_{point_id}", [])
            if calibration_data:
                prediction = self.predict_with_intervals(
                    score, calibration_data, f"{document_id}_{point_id}"
                )
                point_predictions[point_id] = prediction
        
        # Process dimension-level predictions
        for dim_id, score in dimension_scores.items():
            calibration_data = calibration_dataset.get(f"dimension_{dim_id}", [])
            if calibration_data:
                prediction = self.predict_with_intervals(
                    score, calibration_data, f"{document_id}_{dim_id}"
                )
                dimension_predictions[dim_id] = prediction
        
        # Process aggregated predictions
        if point_scores:
            overall_score = np.mean(list(point_scores.values()))
            calibration_data = calibration_dataset.get("overall", [])
            if calibration_data:
                prediction = self.predict_with_intervals(
                    overall_score, calibration_data, f"{document_id}_overall"
                )
                aggregated_predictions["overall_point_score"] = prediction
        
        # Create results object
        results = DocumentConformalResults(
            document_id=document_id,
            point_predictions=point_predictions,
            dimension_predictions=dimension_predictions,
            aggregated_predictions=aggregated_predictions,
            generation_timestamp=self._get_timestamp(),
            calibration_dataset_size=sum(len(data) for data in calibration_dataset.values()),
            confidence_levels=self.confidence_levels
        )
        
        # Save to file
        self.save_results(results, output_dir)
        
        return results
    
    def save_results(
        self,
        results: DocumentConformalResults,
        output_dir: str
    ) -> str:
        """
        Save conformal prediction results to JSON file with deterministic ordering.
        
        Args:
            results: Document conformal results
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        doc_dir = os.path.join(output_dir, results.document_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Convert to serializable format with ordered keys
        serializable_results = self._make_serializable(results)
        
        # Save with deterministic point numbering
        for point_id in sorted(results.point_predictions.keys()):
            point_num = point_id.replace("P", "")
            filename = f"P{point_num}_confidence.json"
            filepath = os.path.join(doc_dir, filename)
            
            # Extract point-specific data
            point_data = {
                "document_id": results.document_id,
                "point_id": point_id,
                "generation_timestamp": results.generation_timestamp,
                "confidence_levels": results.confidence_levels,
                "point_prediction": serializable_results["point_predictions"][point_id],
                "dimension_predictions": {},
                "calibration_dataset_size": results.calibration_dataset_size
            }
            
            # Add relevant dimension predictions
            for dim_id, prediction in serializable_results["dimension_predictions"].items():
                if dim_id.startswith(point_id + "_"):
                    point_data["dimension_predictions"][dim_id] = prediction
            
            # Add aggregated prediction if available
            if "overall_point_score" in serializable_results["aggregated_predictions"]:
                point_data["aggregated_prediction"] = (
                    serializable_results["aggregated_predictions"]["overall_point_score"]
                )
            
            # Write with pretty formatting and sorted keys
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(point_data, f, indent=2, sort_keys=True, ensure_ascii=False)
            
            logger.info(f"Saved conformal predictions to: {filepath}")
        
        return doc_dir
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert dataclass objects to JSON-serializable dictionaries."""
        if isinstance(obj, (DocumentConformalResults, ConformalPrediction, 
                           ConformalInterval, CalibrationMetrics)):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# Helper functions for integration with existing scoring system

def generate_sample_calibration_data(
    n_samples: int = 100,
    score_range: Tuple[float, float] = (0.0, 1.0),
    noise_level: float = 0.1,
    random_seed: int = 42
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Generate sample calibration data for testing conformal predictions.
    
    Args:
        n_samples: Number of calibration samples per type
        score_range: Range of scores to generate
        noise_level: Amount of noise to add to predictions
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with calibration data for different prediction types
    """
    np.random.seed(random_seed)
    
    calibration_data = {}
    
    # Generate data for different prediction types
    prediction_types = [
        "point_P1", "point_P2", "point_P3", "point_P4", "point_P5",
        "dimension_P1_DE-1", "dimension_P1_DE-2", "dimension_P1_DE-3", "dimension_P1_DE-4",
        "dimension_P2_DE-1", "dimension_P2_DE-2", "dimension_P2_DE-3", "dimension_P2_DE-4",
        "overall"
    ]
    
    for pred_type in prediction_types:
        samples = []
        for _ in range(n_samples):
            # Generate true score
            true_score = np.random.uniform(score_range[0], score_range[1])
            
            # Generate predicted score with noise
            noise = np.random.normal(0, noise_level)
            predicted_score = np.clip(true_score + noise, score_range[0], score_range[1])
            
            samples.append((predicted_score, true_score))
        
        calibration_data[pred_type] = samples
    
    return calibration_data


def demo_conformal_prediction():
    """Demonstrate conformal prediction functionality."""
    # Initialize predictor
    predictor = ConformalPredictor()
    
    # Generate sample data
    calibration_data = generate_sample_calibration_data()
    
    # Sample document scores
    point_scores = {
        "P1": 0.75,
        "P2": 0.68,
        "P3": 0.82,
        "P4": 0.71,
        "P5": 0.79
    }
    
    dimension_scores = {
        "P1_DE-1": 0.78,
        "P1_DE-2": 0.72,
        "P1_DE-3": 0.76,
        "P1_DE-4": 0.74,
        "P2_DE-1": 0.69,
        "P2_DE-2": 0.67,
        "P2_DE-3": 0.70,
        "P2_DE-4": 0.66
    }
    
    # Process document predictions
    results = predictor.process_document_predictions(
        document_id="SAMPLE_DOC",
        point_scores=point_scores,
        dimension_scores=dimension_scores,
        calibration_dataset=calibration_data
    )
    
    print("Conformal prediction demo completed successfully!")
    print(f"Generated predictions for document: {results.document_id}")
    print(f"Point predictions: {len(results.point_predictions)}")
    print(f"Dimension predictions: {len(results.dimension_predictions)}")
    
    return results


if __name__ == "__main__":
    demo_conformal_prediction()