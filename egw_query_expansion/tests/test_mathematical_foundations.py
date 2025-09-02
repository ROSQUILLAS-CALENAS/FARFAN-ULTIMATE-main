"""
Tests for mathematical foundations module

Tests information-theoretic measures, semantic similarity metrics,
and Bayesian inference components.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from egw_query_expansion.mathematical_foundations import (
    InformationTheory,
    SemanticSimilarity,
    BayesianInference,
    QuestionClassificationMath,
    EntropyMeasures,
    SimilarityResult,
    BayesianResult,
    compute_question_entropy_features,
    bootstrap_similarity_test
)


class TestInformationTheory:
    """Test information-theoretic measures"""
    
    def test_shannon_entropy_uniform(self):
        """Test Shannon entropy for uniform distribution"""
        # Uniform distribution should have maximum entropy
        uniform_prob = np.ones(4) / 4  # [0.25, 0.25, 0.25, 0.25]
        entropy = InformationTheory.shannon_entropy(uniform_prob, base='2')
        
        # Maximum entropy for 4 categories is log2(4) = 2
        assert abs(entropy - 2.0) < 1e-10
    
    def test_shannon_entropy_deterministic(self):
        """Test Shannon entropy for deterministic distribution"""
        # Deterministic distribution should have zero entropy
        deterministic_prob = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = InformationTheory.shannon_entropy(deterministic_prob)
        
        assert abs(entropy - 0.0) < 1e-10
    
    def test_mutual_information_independent(self):
        """Test mutual information for independent variables"""
        # Independent variables should have zero mutual information
        marginal_x = np.array([0.5, 0.5])
        marginal_y = np.array([0.5, 0.5])
        # Joint distribution for independent variables: P(X,Y) = P(X)P(Y)
        joint_prob = np.outer(marginal_x, marginal_y)
        
        mi = InformationTheory.mutual_information(joint_prob, marginal_x, marginal_y)
        
        assert abs(mi) < 1e-10
    
    def test_kl_divergence_identical(self):
        """Test KL divergence between identical distributions"""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.3, 0.4, 0.3])
        
        kl_div = InformationTheory.kl_divergence(p, q)
        
        assert abs(kl_div) < 1e-10
    
    def test_compute_all_measures(self):
        """Test computation of all measures together"""
        marginal_x = np.array([0.6, 0.4])
        marginal_y = np.array([0.7, 0.3])
        joint_prob = np.array([[0.5, 0.1], [0.2, 0.2]])
        
        measures = InformationTheory.compute_all_measures(joint_prob, marginal_x, marginal_y)
        
        assert isinstance(measures, EntropyMeasures)
        assert measures.shannon_entropy >= 0
        assert measures.conditional_entropy >= 0
        assert measures.mutual_information >= 0
        assert measures.kl_divergence >= 0
        assert measures.cross_entropy >= 0


class TestSemanticSimilarity:
    """Test semantic similarity measures"""
    
    def test_cosine_similarity_with_ci_identical(self):
        """Test cosine similarity with identical embeddings"""
        embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        result = SemanticSimilarity.cosine_similarity_with_ci(
            embeddings, embeddings, bootstrap_samples=100
        )
        
        assert isinstance(result, SimilarityResult)
        assert abs(result.similarity - 1.0) < 0.1  # Should be close to 1
        assert result.confidence_lower <= result.confidence_upper
        assert 0 <= result.confidence_lower <= 1
        assert 0 <= result.confidence_upper <= 1
    
    def test_cosine_similarity_with_ci_orthogonal(self):
        """Test cosine similarity with orthogonal embeddings"""
        embeddings1 = np.array([[1, 0]])
        embeddings2 = np.array([[0, 1]])
        
        result = SemanticSimilarity.cosine_similarity_with_ci(
            embeddings1, embeddings2, bootstrap_samples=50
        )
        
        assert abs(result.similarity) < 0.1  # Should be close to 0
    
    def test_distributional_similarity_wasserstein(self):
        """Test distributional similarity using Wasserstein distance"""
        embeddings1 = np.random.normal(0, 1, (10, 5))
        embeddings2 = np.random.normal(0, 1, (10, 5))
        
        similarity = SemanticSimilarity.distributional_similarity(
            embeddings1, embeddings2, method='wasserstein'
        )
        
        assert 0 <= similarity <= 1
    
    def test_semantic_coherence_intra_cluster(self):
        """Test semantic coherence measurement"""
        # Create coherent embeddings (similar vectors)
        base_vector = np.array([1, 0, 0])
        coherent_embeddings = np.array([
            base_vector + 0.1 * np.random.normal(0, 0.1, 3) for _ in range(5)
        ])
        
        coherence = SemanticSimilarity.semantic_coherence(
            coherent_embeddings, method='intra_cluster'
        )
        
        assert coherence > 0.5  # Should be reasonably coherent


class TestBayesianInference:
    """Test Bayesian inference components"""
    
    def test_beta_binomial_posterior_uninformative(self):
        """Test Beta-Binomial posterior with uninformative prior"""
        # 7 successes out of 10 trials with uniform prior
        result = BayesianInference.beta_binomial_posterior(
            successes=7, trials=10, alpha_prior=1.0, beta_prior=1.0
        )
        
        assert isinstance(result, BayesianResult)
        # Posterior mean should be (7+1)/(10+2) = 8/12 ≈ 0.67
        assert abs(result.posterior_mean - 8/12) < 0.01
        assert result.posterior_std > 0
        assert result.credible_interval[0] < result.credible_interval[1]
        assert len(result.posterior_samples) == 1000
    
    def test_gaussian_posterior(self):
        """Test Gaussian posterior inference"""
        # Generate some data around mean=2
        data = np.array([1.8, 2.1, 2.0, 1.9, 2.2])
        
        result = BayesianInference.gaussian_posterior(
            data, prior_mean=0.0, prior_precision=1.0, noise_precision=1.0
        )
        
        assert isinstance(result, BayesianResult)
        # Posterior mean should be between prior and data mean
        assert 0 < result.posterior_mean < np.mean(data)
        assert result.posterior_std > 0
        assert len(result.posterior_samples) == 1000
    
    def test_multinomial_dirichlet_posterior(self):
        """Test Multinomial-Dirichlet posterior"""
        counts = np.array([5, 3, 2])  # Observed counts for 3 classes
        
        result = BayesianInference.multinomial_dirichlet_posterior(counts)
        
        assert 'posterior_mean' in result
        assert 'posterior_samples' in result
        assert 'credible_intervals' in result
        
        # Check that posterior means sum to 1
        assert abs(np.sum(result['posterior_mean']) - 1.0) < 1e-10
        
        # Check that credible intervals are proper
        assert len(result['credible_intervals']) == 3
        for lower, upper in result['credible_intervals']:
            assert 0 <= lower <= upper <= 1


class TestQuestionClassificationMath:
    """Test integrated question classification mathematics"""
    
    @pytest.fixture
    def classification_math(self):
        return QuestionClassificationMath(alpha=0.05)
    
    def test_entropy_based_categorization(self, classification_math):
        """Test entropy-based question categorization"""
        # Create mock embeddings
        question_embeddings = np.random.normal(0, 1, (3, 10))
        category_embeddings = np.random.normal(0, 1, (2, 10))
        category_labels = ["causal", "associational"]
        
        scores = classification_math.entropy_based_categorization(
            question_embeddings, category_embeddings, category_labels
        )
        
        assert isinstance(scores, dict)
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores.values())
    
    def test_similarity_with_confidence(self, classification_math):
        """Test similarity computation with confidence intervals"""
        question_embedding = np.random.normal(0, 1, (1, 10))
        reference_embeddings = np.random.normal(0, 1, (2, 10))
        reference_labels = ["label1", "label2"]
        
        results = classification_math.similarity_with_confidence(
            question_embedding, reference_embeddings, reference_labels
        )
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert all(isinstance(result, SimilarityResult) for result in results.values())
    
    def test_bayesian_classification_confidence(self, classification_math):
        """Test Bayesian confidence computation"""
        # Mock classification history
        history = [("class_a", True), ("class_a", True), ("class_a", False), ("class_b", True)]
        
        result = classification_math.bayesian_classification_confidence(
            history, "class_a"
        )
        
        assert isinstance(result, BayesianResult)
        assert 0 <= result.posterior_mean <= 1
        assert result.posterior_std >= 0
    
    def test_uncertainty_quantification(self, classification_math):
        """Test uncertainty quantification methods"""
        # Create mock predictions
        predictions = np.array([0.7, 0.2, 0.1])
        
        uncertainty = classification_math.uncertainty_quantification(
            predictions, method='entropy'
        )
        
        assert isinstance(uncertainty, dict)
        assert 'predictive_entropy' in uncertainty
        assert all(isinstance(value, (int, float)) for value in uncertainty.values())


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_compute_question_entropy_features(self):
        """Test question entropy feature computation"""
        question_text = "What is the effect of X on Y?"
        embeddings = np.random.normal(0, 1, 50)
        
        features = compute_question_entropy_features(question_text, embeddings)
        
        assert isinstance(features, dict)
        expected_keys = [
            'shannon_entropy', 'max_entropy', 'entropy_ratio',
            'effective_vocab_size', 'concentration_index'
        ]
        for key in expected_keys:
            assert key in features
            assert isinstance(features[key], float)
    
    def test_bootstrap_similarity_test(self):
        """Test bootstrap similarity testing"""
        embeddings1 = np.random.normal(0, 1, (10, 5))
        embeddings2 = np.random.normal(0, 1, (10, 5))
        
        result = bootstrap_similarity_test(
            embeddings1, embeddings2, n_bootstrap=100
        )
        
        assert isinstance(result, dict)
        expected_keys = [
            'observed_similarity', 'p_value', 'confidence_interval',
            'effect_size', 'bootstrap_mean', 'bootstrap_std'
        ]
        for key in expected_keys:
            assert key in result


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for testing"""
    mock = Mock()
    # Mock encode method to return random embeddings
    mock.encode.return_value = np.random.normal(0, 1, (1, 384))
    return mock


class TestIntegrationWithQuestionAnalyzer:
    """Integration tests with QuestionAnalyzer"""
    
    @patch('question_analyzer.SentenceTransformer')
    @patch('question_analyzer.AutoTokenizer')
    def test_question_analyzer_mathematical_integration(self, mock_tokenizer, mock_transformer):
        """Test that QuestionAnalyzer integrates properly with mathematical foundations"""
        from question_analyzer import QuestionAnalyzer
        
        # Mock the models
        mock_transformer.return_value.encode.return_value = np.random.normal(0, 1, (1, 384))
        mock_tokenizer.from_pretrained.return_value.tokenize.return_value = ['what', 'is', 'effect']
        
        analyzer = QuestionAnalyzer()
        
        # Test mathematical analysis
        question = "What is the effect of education on income?"
        analysis = analyzer.get_mathematical_analysis(question)
        
        assert isinstance(analysis, dict)
        assert 'entropy_features' in analysis
        assert 'similarity_results' in analysis
        assert 'uncertainty_measures' in analysis
        assert 'bayesian_confidence' in analysis
        assert 'classification_quality_score' in analysis
        
        # Test routing confidence
        confidence = analyzer.get_routing_confidence(question, "causal_analysis_route")
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])