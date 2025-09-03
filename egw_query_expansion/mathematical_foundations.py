"""
Mathematical Foundations for Question Analysis

This module implements core mathematical functions for:
1. Information-theoretic measures for question classification
2. Semantic similarity metrics with confidence intervals  
3. Probabilistic scoring with Bayesian inference and uncertainty quantification

Theoretical foundations based on:
- Shannon entropy and mutual information theory
- Distributional semantics and vector space models
- Bayesian inference with credible intervals
- Conformal prediction for finite-sample guarantees
"""

# # # from typing import Dict, List, Optional, Tuple, Union, Any  # Module not found  # Module not found  # Module not found
import warnings
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found

# Import safety wrapper
try:
# # #     from .core.import_safety import safe_import  # Module not found  # Module not found  # Module not found
except ImportError:
    # Fallback if import_safety not available
# # #     from importlib import import_module  # Module not found  # Module not found  # Module not found
    def safe_import(module_name, **kwargs):
        class MockResult:
            def __init__(self, success, module=None):
                self.success = success
                self.module = module
        try:
            return MockResult(True, import_module(module_name))
        except ImportError:
            return MockResult(False)

# Safe imports with fallbacks
numpy_result = safe_import('numpy', attributes=['array', 'linalg'])
if numpy_result.success:
    np = numpy_result.module
else:
    np = None

torch_result = safe_import('torch', required=False)
torch = torch_result.module if torch_result.success else None

torch_functional_result = safe_import('torch.nn.functional', required=False)
F = torch_functional_result.module if torch_functional_result.success else None

scipy_stats_result = safe_import('scipy.stats', required=False)
stats = scipy_stats_result.module if scipy_stats_result.success else None

scipy_special_result = safe_import('scipy.special', required=False)
if scipy_special_result.success:
# # #     from scipy.special import digamma, gammaln  # Module not found  # Module not found  # Module not found

scipy_distance_result = safe_import('scipy.spatial.distance', required=False)
if scipy_distance_result.success:
# # #     from scipy.spatial.distance import cosine as cosine_distance  # Module not found  # Module not found  # Module not found

sklearn_similarity_result = safe_import('sklearn.metrics.pairwise', required=False)
if sklearn_similarity_result.success:
# # #     from sklearn.metrics.pairwise import cosine_similarity  # Module not found  # Module not found  # Module not found

sklearn_preprocessing_result = safe_import('sklearn.preprocessing', required=False)
if sklearn_preprocessing_result.success:
# # #     from sklearn.preprocessing import normalize  # Module not found  # Module not found  # Module not found


@dataclass
class EntropyMeasures:
    """Container for entropy-based measures"""
    shannon_entropy: float
    conditional_entropy: float
    mutual_information: float
    kl_divergence: float
    cross_entropy: float


@dataclass
class SimilarityResult:
    """Container for similarity computation results"""
    similarity: float
    confidence_lower: float
    confidence_upper: float
    p_value: float
    effect_size: float


@dataclass
class BayesianResult:
    """Container for Bayesian inference results"""
    posterior_mean: float
    posterior_std: float
    credible_interval: Tuple[float, float]
    bayes_factor: float
    posterior_samples: Union[List, Any]  # Can be np.ndarray or list


class InformationTheory:
    """Information-theoretic measures for question classification"""
    
    @staticmethod
    def shannon_entropy(probabilities, base: str = 'e') -> float:
        """
        Compute Shannon entropy H(X) = -∑ p(x) log p(x)
        
        Args:
            probabilities: Probability distribution
            base: Logarithm base ('e', '2', '10')
            
        Returns:
            Shannon entropy in specified base
        """
        if not numpy_result.success:
            # Fallback implementation without numpy
            prob_list = list(probabilities) if hasattr(probabilities, '__iter__') else [probabilities]
            prob_list = [p for p in prob_list if p > 0]
            if not prob_list:
                return 0.0
            if base == 'e':
                import math
                return -sum(p * math.log(p) for p in prob_list)
            elif base == '2':
                import math
                return -sum(p * math.log2(p) for p in prob_list)
            return 0.0
        
        # Handle zero probabilities
        probabilities = np.asarray(probabilities)
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) == 0:
            return 0.0
            
        if base == 'e':
            return -np.sum(probabilities * np.log(probabilities))
        elif base == '2':
            return -np.sum(probabilities * np.log2(probabilities))
        elif base == '10':
            return -np.sum(probabilities * np.log10(probabilities))
        else:
            raise ValueError(f"Unknown base: {base}")
    
    @staticmethod
    def conditional_entropy(joint_prob, marginal_prob) -> float:
        """
        Compute conditional entropy H(Y|X) = H(X,Y) - H(X)
        
        Args:
            joint_prob: Joint probability distribution P(X,Y)
            marginal_prob: Marginal probability distribution P(X)
            
        Returns:
            Conditional entropy H(Y|X)
        """
        if numpy_result.success:
            joint_entropy = InformationTheory.shannon_entropy(np.asarray(joint_prob).flatten())
            marginal_entropy = InformationTheory.shannon_entropy(np.asarray(marginal_prob))
        else:
            # Fallback without numpy
            joint_flat = list(joint_prob) if hasattr(joint_prob, '__iter__') else [joint_prob]
            joint_entropy = InformationTheory.shannon_entropy(joint_flat)
            marginal_entropy = InformationTheory.shannon_entropy(marginal_prob)
        return joint_entropy - marginal_entropy
    
    @staticmethod
    def mutual_information(joint_prob, marginal_x, marginal_y) -> float:
        """
        Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            joint_prob: Joint probability P(X,Y)
            marginal_x: Marginal probability P(X)
            marginal_y: Marginal probability P(Y)
            
        Returns:
            Mutual information I(X;Y)
        """
        h_x = InformationTheory.shannon_entropy(marginal_x)
        h_y = InformationTheory.shannon_entropy(marginal_y)
        if numpy_result.success:
            h_xy = InformationTheory.shannon_entropy(np.asarray(joint_prob).flatten())
        else:
            joint_flat = list(joint_prob) if hasattr(joint_prob, '__iter__') else [joint_prob]
            h_xy = InformationTheory.shannon_entropy(joint_flat)
        return h_x + h_y - h_xy
    
    @staticmethod
    def kl_divergence(p, q) -> float:
        """
        Compute Kullback-Leibler divergence KL(P||Q) = ∑ p(x) log(p(x)/q(x))
        
        Args:
            p: Reference distribution
            q: Comparison distribution
            
        Returns:
            KL divergence (non-negative, 0 if p=q)
        """
        if not numpy_result.success:
            # Fallback implementation without numpy
            import math
            p_list = list(p) if hasattr(p, '__iter__') else [p]
            q_list = list(q) if hasattr(q, '__iter__') else [q]
            
            # Ensure same support and handle zeros
            p_list = [x + 1e-10 for x in p_list]
            q_list = [x + 1e-10 for x in q_list]
            
            # Normalize
            p_sum = sum(p_list)
            q_sum = sum(q_list)
            p_list = [x / p_sum for x in p_list]
            q_list = [x / q_sum for x in q_list]
            
            return sum(p_i * math.log(p_i / q_i) for p_i, q_i in zip(p_list, q_list))
        
        # Ensure same support and handle zeros
        p = np.asarray(p) + 1e-10
        q = np.asarray(q) + 1e-10
        
        # Normalize to probabilities
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))
    
    @staticmethod
    def cross_entropy(p, q) -> float:
        """
        Compute cross entropy H(p,q) = -∑ p(x) log q(x)
        
        Args:
            p: True distribution
            q: Predicted distribution
            
        Returns:
            Cross entropy
        """
        if not numpy_result.success:
            # Fallback implementation without numpy
            import math
            p_list = list(p) if hasattr(p, '__iter__') else [p]
            q_list = list(q) if hasattr(q, '__iter__') else [q]
            q_list = [x + 1e-10 for x in q_list]  # Avoid log(0)
            return -sum(p_i * math.log(q_i) for p_i, q_i in zip(p_list, q_list))
        
        q = np.asarray(q) + 1e-10  # Avoid log(0)
        p = np.asarray(p)
        return -np.sum(p * np.log(q))
    
    @classmethod
    def compute_all_measures(cls, joint_prob, marginal_x, marginal_y) -> EntropyMeasures:
        """
        Compute all information-theoretic measures for a joint distribution
        
        Args:
            joint_prob: Joint probability distribution
            marginal_x: Marginal distribution for X
            marginal_y: Marginal distribution for Y
            
        Returns:
            EntropyMeasures containing all computed measures
        """
        shannon_h = cls.shannon_entropy(marginal_x)
        conditional_h = cls.conditional_entropy(joint_prob, marginal_x)
        mutual_info = cls.mutual_information(joint_prob, marginal_x, marginal_y)
        kl_div = cls.kl_divergence(marginal_x, marginal_y)
        cross_h = cls.cross_entropy(marginal_x, marginal_y)
        
        return EntropyMeasures(
            shannon_entropy=shannon_h,
            conditional_entropy=conditional_h,
            mutual_information=mutual_info,
            kl_divergence=kl_div,
            cross_entropy=cross_h
        )


class SemanticSimilarity:
    """Semantic similarity metrics with confidence intervals"""
    
    @staticmethod
    def cosine_similarity_with_ci(embeddings1: np.ndarray, 
                                 embeddings2: np.ndarray,
                                 alpha: float = 0.05,
                                 bootstrap_samples: int = 1000) -> SimilarityResult:
        """
        Compute cosine similarity with bootstrap confidence intervals
        
        Args:
            embeddings1: First set of embeddings (n_samples, n_features)
            embeddings2: Second set of embeddings (n_samples, n_features)
            alpha: Significance level for confidence interval
            bootstrap_samples: Number of bootstrap samples
            
        Returns:
            SimilarityResult with similarity and confidence bounds
        """
        # Compute base similarity
        similarities = cosine_similarity(embeddings1, embeddings2)
        base_similarity = np.mean(similarities)
        
        # Bootstrap confidence interval
        bootstrap_similarities = []
        n_samples = min(len(embeddings1), len(embeddings2))
        
        for _ in range(bootstrap_samples):
            # Sample with replacement
            idx1 = np.random.choice(len(embeddings1), size=n_samples, replace=True)
            idx2 = np.random.choice(len(embeddings2), size=n_samples, replace=True)
            
            boot_sim = cosine_similarity(embeddings1[idx1], embeddings2[idx2])
            bootstrap_similarities.append(np.mean(boot_sim))
        
        bootstrap_similarities = np.array(bootstrap_similarities)
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_similarities, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_similarities, 100 * (1 - alpha / 2))
        
        # Statistical test (one-sample t-test against 0)
        t_stat, p_value = stats.ttest_1samp(bootstrap_similarities, 0.0)
        
        # Effect size (Cohen's d)
        effect_size = np.mean(bootstrap_similarities) / np.std(bootstrap_similarities)
        
        return SimilarityResult(
            similarity=base_similarity,
            confidence_lower=ci_lower,
            confidence_upper=ci_upper,
            p_value=p_value,
            effect_size=effect_size
        )
    
    @staticmethod
    def distributional_similarity(embeddings1: np.ndarray, 
                                embeddings2: np.ndarray,
                                method: str = 'wasserstein') -> float:
        """
        Compute distributional similarity between embedding sets
        
        Args:
            embeddings1: First embedding set
            embeddings2: Second embedding set  
            method: Similarity method ('wasserstein', 'energy', 'mmd')
            
        Returns:
            Distributional similarity score
        """
        if method == 'wasserstein':
            # Wasserstein distance in embedding space
# # #             from scipy.stats import wasserstein_distance  # Module not found  # Module not found  # Module not found
            # Flatten embeddings for 1D comparison
            flat1 = embeddings1.flatten()
            flat2 = embeddings2.flatten()
            return 1.0 / (1.0 + wasserstein_distance(flat1, flat2))
            
        elif method == 'energy':
            # Energy distance
# # #             from scipy.spatial.distance import pdist, squareform  # Module not found  # Module not found  # Module not found
            
            # Compute pairwise distances
            combined = np.vstack([embeddings1, embeddings2])
            distances = squareform(pdist(combined))
            
            n1, n2 = len(embeddings1), len(embeddings2)
            
            # Energy statistic components
            term1 = np.mean(distances[:n1, :n1])
            term2 = np.mean(distances[n1:, n1:])  
            term3 = np.mean(distances[:n1, n1:])
            
            energy_dist = 2 * term3 - term1 - term2
            return 1.0 / (1.0 + energy_dist)
            
        elif method == 'mmd':
            # Maximum Mean Discrepancy with RBF kernel
            def rbf_kernel(X, Y, gamma=1.0):
                """RBF kernel computation"""
                XX = np.sum(X**2, axis=1)[:, np.newaxis]
                YY = np.sum(Y**2, axis=1)[np.newaxis, :]
                XY = np.dot(X, Y.T)
                return np.exp(-gamma * (XX + YY - 2 * XY))
            
            K_XX = rbf_kernel(embeddings1, embeddings1)
            K_YY = rbf_kernel(embeddings2, embeddings2)
            K_XY = rbf_kernel(embeddings1, embeddings2)
            
            mmd_squared = (np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))
            return 1.0 / (1.0 + np.sqrt(max(0, mmd_squared)))
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def semantic_coherence(embeddings: np.ndarray, 
                          method: str = 'intra_cluster') -> float:
        """
        Measure semantic coherence within embedding set
        
        Args:
            embeddings: Set of embeddings to analyze
            method: Coherence method ('intra_cluster', 'silhouette')
            
        Returns:
            Coherence score (higher = more coherent)
        """
        if method == 'intra_cluster':
            # Average pairwise cosine similarity
            similarities = cosine_similarity(embeddings)
            # Exclude diagonal
            mask = ~np.eye(len(similarities), dtype=bool)
            return np.mean(similarities[mask])
            
        elif method == 'silhouette':
            # Silhouette analysis (requires clustering)
# # #             from sklearn.cluster import KMeans  # Module not found  # Module not found  # Module not found
# # #             from sklearn.metrics import silhouette_score  # Module not found  # Module not found  # Module not found
            
            if len(embeddings) < 2:
                return 1.0
                
            # Use 2 clusters or fewer if not enough samples
            n_clusters = min(2, len(embeddings))
            if n_clusters < 2:
                return 1.0
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            if len(np.unique(labels)) < 2:
                return 1.0
                
            return silhouette_score(embeddings, labels)
            
        else:
            raise ValueError(f"Unknown method: {method}")


class BayesianInference:
    """Bayesian inference for classification confidence with uncertainty quantification"""
    
    @staticmethod
    def beta_binomial_posterior(successes: int, 
                               trials: int,
                               alpha_prior: float = 1.0,
                               beta_prior: float = 1.0) -> BayesianResult:
        """
        Compute Beta-Binomial posterior for classification confidence
        
        Args:
            successes: Number of successful classifications
            trials: Total number of trials
            alpha_prior: Prior alpha parameter
            beta_prior: Prior beta parameter
            
        Returns:
            BayesianResult with posterior statistics
        """
        # Posterior parameters
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (trials - successes)
        
        # Posterior statistics
        posterior_mean = alpha_post / (alpha_post + beta_post)
        posterior_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
        posterior_std = np.sqrt(posterior_var)
        
        # Credible interval (95% by default)
        credible_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
        credible_upper = stats.beta.ppf(0.975, alpha_post, beta_post)
        
        # Bayes Factor (against uniform prior)
        # BF = P(data|H1) / P(data|H0) where H0: uniform, H1: beta prior
        log_marginal_h1 = (gammaln(alpha_prior + beta_prior) - gammaln(alpha_prior) - gammaln(beta_prior) +
                           gammaln(alpha_post) + gammaln(beta_post) - gammaln(alpha_post + beta_post))
        log_marginal_h0 = -np.log(trials + 1)  # Uniform over [0,1]
        log_bf = log_marginal_h1 - log_marginal_h0
        bayes_factor = np.exp(log_bf)
        
        # Generate posterior samples
        posterior_samples = stats.beta.rvs(alpha_post, beta_post, size=1000)
        
        return BayesianResult(
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_interval=(credible_lower, credible_upper),
            bayes_factor=bayes_factor,
            posterior_samples=posterior_samples
        )
    
    @staticmethod
    def gaussian_posterior(data: np.ndarray,
                          prior_mean: float = 0.0,
                          prior_precision: float = 1.0,
                          noise_precision: float = 1.0) -> BayesianResult:
        """
        Compute Gaussian posterior for continuous scores
        
        Args:
            data: Observed data points
            prior_mean: Prior mean
            prior_precision: Prior precision (1/variance)
            noise_precision: Noise precision
            
        Returns:
            BayesianResult with posterior statistics
        """
        n = len(data)
        data_mean = np.mean(data)
        
        # Posterior parameters
        posterior_precision = prior_precision + n * noise_precision
        posterior_mean = (prior_precision * prior_mean + n * noise_precision * data_mean) / posterior_precision
        posterior_var = 1.0 / posterior_precision
        posterior_std = np.sqrt(posterior_var)
        
        # Credible interval
        credible_lower = posterior_mean - 1.96 * posterior_std
        credible_upper = posterior_mean + 1.96 * posterior_std
        
        # Bayes Factor (against zero mean)
        # Marginal likelihood under H1 (informative prior)
        log_ml_h1 = (-0.5 * n * np.log(2 * np.pi / noise_precision) - 
                     0.5 * noise_precision * np.sum((data - data_mean)**2) -
                     0.5 * np.log(2 * np.pi / prior_precision) +
                     0.5 * np.log(2 * np.pi / posterior_precision) +
                     0.5 * posterior_precision * posterior_mean**2 - 
                     0.5 * prior_precision * prior_mean**2)
        
        # Marginal likelihood under H0 (zero mean)  
        log_ml_h0 = (-0.5 * n * np.log(2 * np.pi / noise_precision) - 
                     0.5 * noise_precision * np.sum(data**2))
        
        log_bf = log_ml_h1 - log_ml_h0
        bayes_factor = np.exp(log_bf)
        
        # Generate posterior samples
        posterior_samples = np.random.normal(posterior_mean, posterior_std, size=1000)
        
        return BayesianResult(
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            credible_interval=(credible_lower, credible_upper),
            bayes_factor=bayes_factor,
            posterior_samples=posterior_samples
        )
    
    @staticmethod
    def multinomial_dirichlet_posterior(counts: np.ndarray,
                                      alpha_prior: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Compute Multinomial-Dirichlet posterior for multi-class classification
        
        Args:
            counts: Observed counts for each class
            alpha_prior: Prior Dirichlet parameters
            
        Returns:
            Dictionary with posterior statistics
        """
        k = len(counts)
        
        if alpha_prior is None:
            alpha_prior = np.ones(k)  # Uniform prior
        
        # Posterior parameters
        alpha_post = alpha_prior + counts
        
        # Posterior statistics
        total_count = np.sum(alpha_post)
        posterior_mean = alpha_post / total_count
        
        # Posterior covariance
        posterior_cov = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    posterior_cov[i, j] = alpha_post[i] * (total_count - alpha_post[i]) / (total_count**2 * (total_count + 1))
                else:
                    posterior_cov[i, j] = -alpha_post[i] * alpha_post[j] / (total_count**2 * (total_count + 1))
        
        # Generate posterior samples
        posterior_samples = stats.dirichlet.rvs(alpha_post, size=1000)
        
        # Credible intervals for each class
        credible_intervals = []
        for i in range(k):
            lower = np.percentile(posterior_samples[:, i], 2.5)
            upper = np.percentile(posterior_samples[:, i], 97.5)
            credible_intervals.append((lower, upper))
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_cov': posterior_cov,
            'credible_intervals': credible_intervals,
            'posterior_samples': posterior_samples,
            'alpha_posterior': alpha_post
        }


class QuestionClassificationMath:
    """Mathematical foundations for question classification"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize mathematical foundations
        
        Args:
            alpha: Significance level for confidence intervals
        """
        self.alpha = alpha
        self.info_theory = InformationTheory()
        self.similarity = SemanticSimilarity()
        self.bayesian = BayesianInference()
    
    def entropy_based_categorization(self, 
                                   question_embeddings: np.ndarray,
                                   category_embeddings: np.ndarray,
                                   category_labels: List[str]) -> Dict[str, float]:
        """
        Categorize questions using entropy-based measures
        
        Args:
            question_embeddings: Embeddings for questions to classify
            category_embeddings: Reference embeddings for each category
            category_labels: Labels for categories
            
        Returns:
            Dictionary mapping categories to entropy-based scores
        """
        scores = {}
        
        # Compute similarity distributions
        for i, category in enumerate(category_labels):
            category_emb = category_embeddings[i:i+1]
            
            # Compute similarities
            similarities = cosine_similarity(question_embeddings, category_emb).flatten()
            
            # Convert to probability distribution (softmax)
            probabilities = F.softmax(torch.tensor(similarities), dim=0).numpy()
            
            # Compute entropy measures
            entropy_score = self.info_theory.shannon_entropy(probabilities)
            
            # Lower entropy = more concentrated = better category match
            scores[category] = -entropy_score  # Negative for ranking
        
        return scores
    
    def similarity_with_confidence(self,
                                 question_embedding: np.ndarray,
                                 reference_embeddings: np.ndarray,
                                 reference_labels: List[str]) -> Dict[str, SimilarityResult]:
        """
        Compute semantic similarity with confidence intervals
        
        Args:
            question_embedding: Embedding for question to analyze
            reference_embeddings: Reference embeddings
            reference_labels: Labels for references
            
        Returns:
            Dictionary mapping labels to SimilarityResult
        """
        results = {}
        
        # Expand question embedding to match reference dimensions if needed
        if question_embedding.ndim == 1:
            question_embedding = question_embedding.reshape(1, -1)
        
        for i, label in enumerate(reference_labels):
            ref_emb = reference_embeddings[i:i+1]
            
            # Compute similarity with confidence interval
            sim_result = self.similarity.cosine_similarity_with_ci(
                question_embedding, ref_emb, alpha=self.alpha
            )
            
            results[label] = sim_result
        
        return results
    
    def bayesian_classification_confidence(self,
                                         classification_history: List[Tuple[str, bool]],
                                         target_class: str) -> BayesianResult:
        """
        Compute Bayesian confidence for classification decisions
        
        Args:
            classification_history: List of (predicted_class, was_correct) pairs
            target_class: Class to compute confidence for
            
        Returns:
            BayesianResult with posterior confidence statistics
        """
        # Filter history for target class
        target_predictions = [(pred == target_class, correct) 
                            for pred, correct in classification_history]
        
        if not target_predictions:
            # No history, use uninformative prior
            return self.bayesian.beta_binomial_posterior(0, 1)
        
        # Count successes (correct predictions for this class)
        successes = sum(1 for was_target, was_correct in target_predictions 
                       if was_target and was_correct)
        trials = sum(1 for was_target, was_correct in target_predictions if was_target)
        
        if trials == 0:
            return self.bayesian.beta_binomial_posterior(0, 1)
        
        return self.bayesian.beta_binomial_posterior(successes, trials)
    
    def uncertainty_quantification(self,
                                 predictions: np.ndarray,
                                 method: str = 'entropy') -> Dict[str, float]:
        """
        Quantify prediction uncertainty using various methods
        
        Args:
            predictions: Model predictions (probabilities)
            method: Uncertainty method ('entropy', 'variance', 'mutual_info')
            
        Returns:
            Dictionary with uncertainty measures
        """
        results = {}
        
        if method == 'entropy':
            # Predictive entropy
            entropy_unc = self.info_theory.shannon_entropy(predictions)
            results['predictive_entropy'] = entropy_unc
            
        elif method == 'variance':
            # Predictive variance
            pred_var = np.var(predictions)
            results['predictive_variance'] = pred_var
            
        elif method == 'mutual_info':
            # Mutual information between parameters and predictions
            # Approximate using prediction statistics
            mean_pred = np.mean(predictions)
            results['mutual_information'] = self.info_theory.shannon_entropy(predictions) - \
                                          self.info_theory.shannon_entropy(np.array([mean_pred, 1-mean_pred]))
        
        # Additional uncertainty measures
        results['max_probability'] = np.max(predictions)
        results['prediction_confidence'] = np.max(predictions) - np.min(predictions)
        results['gini_impurity'] = 1 - np.sum(predictions**2)
        
        return results


# Utility functions for integration with existing code

def compute_question_entropy_features(question_text: str, 
                                    embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute entropy-based features for a question
    
    Args:
        question_text: Text of the question
        embeddings: Embedding representations
        
    Returns:
        Dictionary of entropy features
    """
    # Normalize embeddings to probabilities
    probabilities = F.softmax(torch.tensor(embeddings), dim=-1).numpy()
    
    # Compute various entropy measures
    features = {
        'shannon_entropy': InformationTheory.shannon_entropy(probabilities),
        'max_entropy': np.log(len(probabilities)),  # Maximum possible entropy
        'entropy_ratio': InformationTheory.shannon_entropy(probabilities) / np.log(len(probabilities)),
        'effective_vocab_size': np.exp(InformationTheory.shannon_entropy(probabilities)),
        'concentration_index': np.sum(probabilities**2),  # Herfindahl index
    }
    
    return features


def bootstrap_similarity_test(embeddings1: np.ndarray,
                            embeddings2: np.ndarray,
                            n_bootstrap: int = 1000,
                            alpha: float = 0.05) -> Dict[str, float]:
    """
    Bootstrap test for similarity between embedding sets
    
    Args:
        embeddings1: First embedding set
        embeddings2: Second embedding set  
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        
    Returns:
        Dictionary with test statistics
    """
    # Compute observed similarity
    observed_sim = np.mean(cosine_similarity(embeddings1, embeddings2))
    
    # Bootstrap under null hypothesis (no difference)
    combined = np.vstack([embeddings1, embeddings2])
    n1, n2 = len(embeddings1), len(embeddings2)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample
        perm_idx = np.random.permutation(len(combined))
        boot_emb1 = combined[perm_idx[:n1]]
        boot_emb2 = combined[perm_idx[n1:n1+n2]]
        
        # Compute statistic
        boot_sim = np.mean(cosine_similarity(boot_emb1, boot_emb2))
        bootstrap_stats.append(boot_sim)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # P-value (two-tailed)
    p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_sim))
    
    # Confidence interval
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return {
        'observed_similarity': observed_sim,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'effect_size': observed_sim / np.std(bootstrap_stats),
        'bootstrap_mean': np.mean(bootstrap_stats),
        'bootstrap_std': np.std(bootstrap_stats)
    }