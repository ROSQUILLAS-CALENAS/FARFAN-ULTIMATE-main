"""
Test suite for conformal prediction module.
"""

import unittest
import tempfile
import shutil
import json
import os
import numpy as np
from decimal import Decimal

from .conformal_prediction import (
    ConformalPredictor,
    ConformalInterval,
    ConformalPrediction,
    CalibrationMetrics,
    DocumentConformalResults,
    generate_sample_calibration_data
)
from .decalogo_scoring_system import ScoringSystem


class TestConformalPredictor(unittest.TestCase):
    """Test cases for ConformalPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = ConformalPredictor(random_seed=42)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_nonconformity_score_computation(self):
        """Test nonconformity score computation."""
        # Test absolute nonconformity
        score = self.predictor.compute_nonconformity_score(0.8, 0.7, "absolute")
        self.assertAlmostEqual(score, 0.1, places=6)
        
        # Test normalized nonconformity
        score = self.predictor.compute_nonconformity_score(0.8, 0.7, "normalized")
        self.assertAlmostEqual(score, 0.1 / 0.8, places=6)
        
        # Test edge case: zero predicted score
        score = self.predictor.compute_nonconformity_score(0.0, 0.5, "normalized")
        self.assertAlmostEqual(score, 0.5, places=6)
    
    def test_calibration_data_fitting(self):
        """Test calibration data fitting."""
        calibration_data = [
            (0.8, 0.7),
            (0.6, 0.65),
            (0.9, 0.85),
            (0.4, 0.5)
        ]
        
        nonconformity_scores = self.predictor.fit_calibration_data(calibration_data)
        
        expected_scores = [0.1, 0.05, 0.05, 0.1]
        self.assertEqual(len(nonconformity_scores), len(expected_scores))
        
        for i, expected in enumerate(expected_scores):
            self.assertAlmostEqual(nonconformity_scores[i], expected, places=6)
    
    def test_quantile_computation(self):
        """Test conformal quantile computation."""
        nonconformity_scores = [0.05, 0.1, 0.15, 0.2, 0.25]
        
        # Test 90% confidence level
        quantile = self.predictor.compute_quantile(nonconformity_scores, 0.9)
        self.assertAlmostEqual(quantile, 0.2, places=6)
        
        # Test 95% confidence level
        quantile = self.predictor.compute_quantile(nonconformity_scores, 0.95)
        self.assertAlmostEqual(quantile, 0.25, places=6)
        
        # Test edge case: empty scores
        quantile = self.predictor.compute_quantile([], 0.95)
        self.assertEqual(quantile, 0.0)
    
    def test_bootstrap_variance_estimation(self):
        """Test bootstrap variance estimation."""
        scores = [0.7, 0.8, 0.75, 0.85, 0.72, 0.78]
        
        bootstrap_results = self.predictor.bootstrap_variance_estimation(scores)
        
        # Check all expected keys are present
        expected_keys = ["mean", "std", "var", "ci_lower", "ci_upper"]
        for key in expected_keys:
            self.assertIn(key, bootstrap_results)
        
        # Check that mean is reasonable
        self.assertGreater(bootstrap_results["mean"], 0.7)
        self.assertLess(bootstrap_results["mean"], 0.9)
        
        # Check that std is non-negative
        self.assertGreaterEqual(bootstrap_results["std"], 0)
        
        # Check that confidence interval is reasonable
        self.assertLessEqual(bootstrap_results["ci_lower"], bootstrap_results["mean"])
        self.assertGreaterEqual(bootstrap_results["ci_upper"], bootstrap_results["mean"])
        
        # Test edge case: single score
        single_score_results = self.predictor.bootstrap_variance_estimation([0.8])
        self.assertEqual(single_score_results["mean"], 0.8)
        self.assertEqual(single_score_results["std"], 0.0)
    
    def test_prediction_intervals_generation(self):
        """Test prediction interval generation."""
        predicted_score = 0.75
        nonconformity_scores = [0.05, 0.1, 0.15, 0.2]
        
        intervals = self.predictor.generate_prediction_intervals(
            predicted_score, nonconformity_scores
        )
        
        # Check that all confidence levels are present
        for level in self.predictor.confidence_levels:
            level_key = f"{int(level * 100)}%"
            self.assertIn(level_key, intervals)
            
            interval = intervals[level_key]
            self.assertIsInstance(interval, ConformalInterval)
            
            # Check bounds are reasonable
            self.assertGreaterEqual(interval.lower_bound, 0.0)
            self.assertLessEqual(interval.upper_bound, 1.0)
            self.assertLessEqual(interval.lower_bound, interval.upper_bound)
            
            # Check interval width calculation
            expected_width = interval.upper_bound - interval.lower_bound
            self.assertAlmostEqual(interval.prediction_interval, expected_width, places=6)
    
    def test_calibration_quality_evaluation(self):
        """Test calibration quality evaluation."""
        # Create test data with known coverage
        test_predictions = []
        
        # Create intervals that should have good coverage
        for i in range(20):
            predicted = 0.75
            true = 0.75 + np.random.normal(0, 0.05)  # Small noise
            
            intervals = {
                "90%": ConformalInterval(
                    lower_bound=0.65, upper_bound=0.85,
                    prediction_interval=0.2, coverage_probability=0.9,
                    confidence_level=0.9, nonconformity_score=0.1
                ),
                "95%": ConformalInterval(
                    lower_bound=0.6, upper_bound=0.9,
                    prediction_interval=0.3, coverage_probability=0.95,
                    confidence_level=0.95, nonconformity_score=0.15
                ),
                "99%": ConformalInterval(
                    lower_bound=0.5, upper_bound=0.95,
                    prediction_interval=0.45, coverage_probability=0.99,
                    confidence_level=0.99, nonconformity_score=0.25
                )
            }
            
            test_predictions.append((predicted, true, intervals))
        
        calibration_metrics = self.predictor.evaluate_calibration_quality(test_predictions)
        
        # Check metrics structure
        self.assertIsInstance(calibration_metrics, CalibrationMetrics)
        self.assertGreaterEqual(calibration_metrics.empirical_coverage, 0.0)
        self.assertLessEqual(calibration_metrics.empirical_coverage, 1.0)
        self.assertGreater(calibration_metrics.interval_width_mean, 0.0)
        self.assertEqual(calibration_metrics.n_calibration_samples, 20)
    
    def test_predict_with_intervals(self):
        """Test complete prediction with intervals."""
        predicted_score = 0.8
        calibration_data = [
            (0.75, 0.7), (0.85, 0.8), (0.8, 0.85), (0.9, 0.88)
        ]
        prediction_id = "test_prediction"
        
        prediction = self.predictor.predict_with_intervals(
            predicted_score, calibration_data, prediction_id
        )
        
        # Check prediction structure
        self.assertIsInstance(prediction, ConformalPrediction)
        self.assertEqual(prediction.prediction_id, prediction_id)
        self.assertEqual(prediction.predicted_score, predicted_score)
        
        # Check intervals
        self.assertIn("90%", prediction.confidence_intervals)
        self.assertIn("95%", prediction.confidence_intervals)
        self.assertIn("99%", prediction.confidence_intervals)
        
        # Check bootstrap estimates
        self.assertIn("mean", prediction.bootstrap_estimates)
        self.assertIn("std", prediction.bootstrap_estimates)
        
        # Check calibration metrics
        self.assertIsInstance(prediction.calibration_metrics, CalibrationMetrics)
    
    def test_sample_calibration_data_generation(self):
        """Test sample calibration data generation."""
        calibration_data = generate_sample_calibration_data(
            n_samples=50, random_seed=42
        )
        
        # Check that all expected prediction types are present
        expected_types = [
            "point_P1", "point_P2", "point_P3", "point_P4", "point_P5",
            "dimension_P1_DE-1", "dimension_P1_DE-2", "dimension_P1_DE-3", 
            "dimension_P1_DE-4", "overall"
        ]
        
        for pred_type in expected_types:
            self.assertIn(pred_type, calibration_data)
            self.assertEqual(len(calibration_data[pred_type]), 50)
            
            # Check that all samples are valid tuples
            for predicted, true in calibration_data[pred_type]:
                self.assertGreaterEqual(predicted, 0.0)
                self.assertLessEqual(predicted, 1.0)
                self.assertGreaterEqual(true, 0.0)
                self.assertLessEqual(true, 1.0)
    
    def test_document_predictions_processing(self):
        """Test complete document prediction processing."""
        document_id = "TEST_DOC"
        point_scores = {"P1": 0.75, "P2": 0.68}
        dimension_scores = {"P1_DE-1": 0.78, "P1_DE-2": 0.72, "P2_DE-1": 0.69}
        
        # Generate calibration data
        calibration_dataset = {
            "point_P1": [(0.75, 0.73), (0.8, 0.77)],
            "point_P2": [(0.68, 0.7), (0.65, 0.66)],
            "dimension_P1_DE-1": [(0.78, 0.76), (0.8, 0.79)],
            "dimension_P1_DE-2": [(0.72, 0.74), (0.7, 0.71)],
            "dimension_P2_DE-1": [(0.69, 0.67), (0.71, 0.7)],
            "overall": [(0.715, 0.72), (0.725, 0.73)]
        }
        
        results = self.predictor.process_document_predictions(
            document_id=document_id,
            point_scores=point_scores,
            dimension_scores=dimension_scores,
            calibration_dataset=calibration_dataset,
            output_dir=self.temp_dir
        )
        
        # Check results structure
        self.assertIsInstance(results, DocumentConformalResults)
        self.assertEqual(results.document_id, document_id)
        self.assertEqual(len(results.point_predictions), 2)
        self.assertEqual(len(results.dimension_predictions), 3)
        self.assertIn("overall_point_score", results.aggregated_predictions)
        
        # Check that output files were created
        doc_dir = os.path.join(self.temp_dir, document_id)
        self.assertTrue(os.path.exists(doc_dir))
        
        # Check individual point files
        for point_id in point_scores.keys():
            point_num = point_id.replace("P", "")
            filename = f"P{point_num}_confidence.json"
            filepath = os.path.join(doc_dir, filename)
            self.assertTrue(os.path.exists(filepath))
            
            # Verify JSON structure
            with open(filepath, 'r') as f:
                point_data = json.load(f)
            
            self.assertEqual(point_data["document_id"], document_id)
            self.assertEqual(point_data["point_id"], point_id)
            self.assertIn("point_prediction", point_data)
            self.assertIn("dimension_predictions", point_data)
            self.assertIn("confidence_levels", point_data)
    
    def test_json_serialization(self):
        """Test JSON serialization of results."""
        # Create a simple prediction
        predicted_score = 0.8
        calibration_data = [(0.75, 0.7), (0.85, 0.8)]
        prediction_id = "serialization_test"
        
        prediction = self.predictor.predict_with_intervals(
            predicted_score, calibration_data, prediction_id
        )
        
        # Test serialization
        serializable_data = self.predictor._make_serializable(prediction)
        
        # Try to convert to JSON (should not raise exception)
        json_str = json.dumps(serializable_data, indent=2)
        self.assertIsInstance(json_str, str)
        
        # Verify round-trip
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["prediction_id"], prediction_id)
        self.assertEqual(deserialized["predicted_score"], predicted_score)
    
    def test_deterministic_output_ordering(self):
        """Test that outputs have deterministic ordering."""
        document_id = "ORDERING_TEST"
        point_scores = {"P3": 0.8, "P1": 0.7, "P2": 0.75}  # Intentionally unordered
        
        calibration_dataset = {
            "point_P1": [(0.7, 0.72)],
            "point_P2": [(0.75, 0.73)],
            "point_P3": [(0.8, 0.78)],
            "overall": [(0.75, 0.74)]
        }
        
        results = self.predictor.process_document_predictions(
            document_id=document_id,
            point_scores=point_scores,
            dimension_scores={},
            calibration_dataset=calibration_dataset,
            output_dir=self.temp_dir
        )
        
        # Check that files are created with proper ordering
        doc_dir = os.path.join(self.temp_dir, document_id)
        files = sorted(os.listdir(doc_dir))
        
        expected_files = ["P1_confidence.json", "P2_confidence.json", "P3_confidence.json"]
        self.assertEqual(files, expected_files)
        
        # Check JSON key ordering within files
        for filename in files:
            with open(os.path.join(doc_dir, filename), 'r') as f:
                data = json.load(f)
                
            # Keys should be in alphabetical order due to sort_keys=True
            json_str = json.dumps(data, sort_keys=True)
            reloaded = json.loads(json_str)
            
            # Should be identical after sort_keys round-trip
            self.assertEqual(data, reloaded)


class TestIntegrationWithScoringSystem(unittest.TestCase):
    """Integration tests with the existing scoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scoring_system = ScoringSystem()
        self.predictor = ConformalPredictor(scoring_system=self.scoring_system)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integration_with_point_scoring(self):
        """Test integration with PointScore objects."""
        # Create sample evaluation data
        evaluation_data = {
            "DE-1": [
                {
                    "question_id": "Q1.1",
                    "response": "Sí",
                    "evidence_completeness": 0.9,
                    "page_reference_quality": 0.8
                }
            ],
            "DE-2": [
                {
                    "question_id": "Q1.2", 
                    "response": "Parcial",
                    "evidence_completeness": 0.7,
                    "page_reference_quality": 0.6
                }
            ],
            "DE-3": [
                {
                    "question_id": "Q1.3",
                    "response": "Sí",
                    "evidence_completeness": 0.85,
                    "page_reference_quality": 0.9
                }
            ],
            "DE-4": [
                {
                    "question_id": "Q1.4",
                    "response": "No",
                    "evidence_completeness": 0.3,
                    "page_reference_quality": 0.4
                }
            ]
        }
        
        # Process with scoring system
        point_score = self.scoring_system.process_point_evaluation(1, evaluation_data)
        
        # Extract scores for conformal prediction
        point_scores = {"P1": point_score.final_score}
        dimension_scores = {
            f"P1_{dim.dimension_id}": dim.weighted_average 
            for dim in point_score.dimension_scores
        }
        
        # Generate calibration data
        calibration_dataset = generate_sample_calibration_data(n_samples=20)
        
        # Process conformal predictions
        results = self.predictor.process_document_predictions(
            document_id="INTEGRATION_TEST",
            point_scores=point_scores,
            dimension_scores=dimension_scores,
            calibration_dataset=calibration_dataset,
            output_dir=self.temp_dir
        )
        
        # Verify results
        self.assertEqual(results.document_id, "INTEGRATION_TEST")
        self.assertIn("P1", results.point_predictions)
        
        # Check that dimension predictions match the scoring system output
        for dim_score in point_score.dimension_scores:
            dim_key = f"P1_{dim_score.dimension_id}"
            if dim_key in results.dimension_predictions:
                prediction = results.dimension_predictions[dim_key]
                self.assertAlmostEqual(
                    prediction.predicted_score, dim_score.weighted_average, places=4
                )


if __name__ == '__main__':
    unittest.main()