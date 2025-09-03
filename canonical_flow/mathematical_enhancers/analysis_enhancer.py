"""
Math Stage 4 Analysis Enhancer

Information-theoretic measures module for analyzing questions and processing evidence
within the pipeline's stage 4 analysis phase. Implements Shannon entropy and mutual
information calculations to quantify semantic relationships between question-evidence pairs.

Integrates with the existing evidence processing workflow by providing mathematical
validation functions that compute entropy measures on question text and evidence content.

Theoretical Foundations:
- Shannon Entropy: H(X) = -∑ p(x) log p(x) for information content quantification
- Mutual Information: I(X;Y) = ∑∑ p(x,y) log(p(x,y)/(p(x)p(y))) for dependency measurement
- Conditional Entropy: H(X|Y) = H(X,Y) - H(Y) for information reduction analysis
- Cross-Entropy: H(X,Y) = -∑ p(x) log q(y) for distribution comparison

Integrates with cluster execution controller during four-cluster questionnaire application.
"""

import logging
import math
import numpy as np
# # # from collections import Counter, defaultdict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Union, Callable  # Module not found  # Module not found  # Module not found
# Optional sklearn with fallbacks
try:
# # #     from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore  # Module not found  # Module not found  # Module not found
# # #     from sklearn.metrics.pairwise import cosine_similarity  # type: ignore  # Module not found  # Module not found  # Module not found
except Exception:
    class TfidfVectorizer:  # minimal fallback
        def __init__(self, ngram_range=(1, 1), min_df=1, max_features=None, lowercase=True, stop_words=None):
            self.ngram_range = ngram_range
            self.min_df = min_df
            self.max_features = max_features
            self.lowercase = lowercase
            self.stop_words = set(stop_words) if stop_words else None
            self.vocabulary_ = {}
        def _tokenize(self, text: str):
            if self.lowercase:
                text = text.lower()
            tokens = text.split()
            if self.stop_words:
                tokens = [t for t in tokens if t not in self.stop_words]
            return tokens
        def fit(self, corpus):
# # #             from collections import Counter  # Module not found  # Module not found  # Module not found
            df = Counter()
            for doc in corpus:
                for t in set(self._tokenize(doc)):
                    df[t] += 1
            items = list(df.items())
            if self.max_features:
                items = sorted(items, key=lambda x: (-x[1], x[0]))[: self.max_features]
            self.vocabulary_ = {t: i for i, (t, _) in enumerate(sorted(items))}
            return self
        def transform(self, corpus):
            import numpy as np
            X = np.zeros((len(corpus), len(self.vocabulary_)), dtype=np.float32)
            for i, doc in enumerate(corpus):
                for t in self._tokenize(doc):
                    j = self.vocabulary_.get(t)
                    if j is not None:
                        X[i, j] += 1.0
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            return X / norms
        def fit_transform(self, corpus):
            return self.fit(corpus).transform(corpus)
    def cosine_similarity(A, B):
        import numpy as np
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return A_norm @ B_norm.T
# # # from scipy.spatial.distance import pdist, squareform  # Module not found  # Module not found  # Module not found
# # # from scipy.sparse import csr_matrix  # Module not found  # Module not found  # Module not found
# # # from scipy.linalg import eigvals, norm  # Module not found  # Module not found  # Module not found
# # # from itertools import combinations, chain  # Module not found  # Module not found  # Module not found

# Import existing pipeline components for integration
try:
# # #     from evidence_processor import EvidenceProcessor, StructuredEvidence, EvidenceType  # Module not found  # Module not found  # Module not found
# # #     from question_analyzer import QuestionAnalyzer, CausalPosture  # Module not found  # Module not found  # Module not found
# # #     from adaptive_analyzer import AdaptiveAnalyzer, SystemState  # Module not found  # Module not found  # Module not found
except ImportError:
    # Fallbacks for missing dependencies
    EvidenceProcessor = None
    StructuredEvidence = None
    EvidenceType = None
    QuestionAnalyzer = None
    CausalPosture = None
    AdaptiveAnalyzer = None
    SystemState = None

logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """Analysis modes for information-theoretic processing"""
    ENTROPY_ONLY = "entropy_only"
    MUTUAL_INFO_ONLY = "mutual_info_only"
    FULL_ANALYSIS = "full_analysis"
    VALIDATION_FOCUSED = "validation_focused"


class InformationMetric(Enum):
    """Information-theoretic metrics supported by the enhancer"""
    SHANNON_ENTROPY = "shannon_entropy"
    CONDITIONAL_ENTROPY = "conditional_entropy"
    MUTUAL_INFORMATION = "mutual_information"
    NORMALIZED_MUTUAL_INFO = "normalized_mutual_info"
    CROSS_ENTROPY = "cross_entropy"
    KL_DIVERGENCE = "kl_divergence"


@dataclass
class EntropyAnalysis:
# # #     """Results from entropy analysis of text content"""  # Module not found  # Module not found  # Module not found
    text_id: str
    text_content: str
    shannon_entropy: float
    conditional_entropy: Optional[float] = None
    character_entropy: float = 0.0
    word_entropy: float = 0.0
    semantic_entropy: float = 0.0
    complexity_score: float = 0.0
    information_density: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MutualInformationAnalysis:
# # #     """Results from mutual information analysis between question-evidence pairs"""  # Module not found  # Module not found  # Module not found
    question_id: str
    evidence_id: str
    mutual_information: float
    normalized_mi: float
    question_entropy: float
    evidence_entropy: float
    joint_entropy: float
    semantic_similarity: float
    information_gain: float
    relevance_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationMetrics:
    """Mathematical validation metrics for question-evidence relationships"""
    pair_id: str
    entropy_consistency: float
    information_coherence: float
    semantic_alignment: float
    causal_strength: Optional[float] = None
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    validation_passed: bool = False
    validation_reasons: List[str] = field(default_factory=list)


@dataclass
class CategoricalFunctor:
    """Represents a categorical functor for topology-preserving transformations"""
    source_category: str
    target_category: str
    object_mapping: Dict[str, str]
    morphism_mapping: Dict[Tuple[str, str], Tuple[str, str]]
    composition_preserving: bool = True
    identity_preserving: bool = True


@dataclass
class NaturalTransformation:
    """Natural transformation between functors with component mappings"""
    source_functor: CategoricalFunctor
    target_functor: CategoricalFunctor
    component_maps: Dict[str, Callable]
    naturality_condition: bool = True
    transformation_id: str = ""


@dataclass
class HomologyInvariant:
    """Homological invariant for path space analysis"""
    dimension: int
    betti_numbers: List[int]
    torsion_coefficients: List[int]
    persistence_intervals: List[Tuple[float, float]]
    critical_values: List[float]
    homology_rank: int = 0


@dataclass
class SimplexComplex:
    """Simplicial complex representation for topological analysis"""
    vertices: List[str]
    simplices: Dict[int, List[List[str]]]  # dimension -> list of simplices
    boundary_maps: Dict[int, np.ndarray]
    vertex_weights: Dict[str, float] = field(default_factory=dict)
    simplex_weights: Dict[Tuple[str, ...], float] = field(default_factory=dict)


class DirectedAlgebraicTopologyOptimizer:
    """
    Directed algebraic topology optimizer using higher category theory for spectral analysis.

    Implements categorical functors and natural transformations to optimize information
    flow through the pipeline by analyzing the topological structure of processing paths.
    """

    def __init__(self, spectral_threshold: float = 0.85, functor_composition_depth: int = 3):
        self.spectral_threshold = spectral_threshold
        self.functor_composition_depth = functor_composition_depth
        self.category_registry: Dict[str, List[str]] = {}
        self.functor_chain: List[CategoricalFunctor] = []
        self.natural_transformations: List[NaturalTransformation] = []

    def create_spectral_category(self, analysis_space: Dict[str, Any]) -> str:
        """Create categorical representation of spectral analysis space"""
        category_id = f"spec_cat_{len(self.category_registry)}"

# # #         # Extract objects from analysis space  # Module not found  # Module not found  # Module not found
        objects = []
        for key, value in analysis_space.items():
            if isinstance(value, (dict, list)):
                objects.extend([f"{key}_{i}" for i in range(len(value) if isinstance(value, list) else len(value))])
            else:
                objects.append(key)

        self.category_registry[category_id] = objects
        return category_id

    def construct_transition_functor(self, source_cat: str, target_cat: str,
                                   transition_rules: Dict[str, str]) -> CategoricalFunctor:
        """Construct categorical functor for topology-preserving transitions"""
        source_objects = self.category_registry.get(source_cat, [])
        target_objects = self.category_registry.get(target_cat, [])

        # Object mapping based on transition rules
        object_mapping = {}
        for src_obj in source_objects:
            # Apply transition rules or default mapping
            target_obj = transition_rules.get(src_obj,
                                            target_objects[hash(src_obj) % len(target_objects)] if target_objects else src_obj)
            object_mapping[src_obj] = target_obj

        # Morphism mapping (simplified - identity on mapped objects)
        morphism_mapping = {}
        for src1, src2 in combinations(source_objects, 2):
            tgt1, tgt2 = object_mapping[src1], object_mapping[src2]
            morphism_mapping[(src1, src2)] = (tgt1, tgt2)

        return CategoricalFunctor(
            source_category=source_cat,
            target_category=target_cat,
            object_mapping=object_mapping,
            morphism_mapping=morphism_mapping
        )

    def compute_natural_transformation(self, functor1: CategoricalFunctor,
                                     functor2: CategoricalFunctor) -> NaturalTransformation:
        """Compute natural transformation between compatible functors"""
        if functor1.target_category != functor2.source_category:
            raise ValueError("Functors not composable")

        component_maps = {}
        for obj in functor1.object_mapping:
            intermediate = functor1.object_mapping[obj]
            final = functor2.object_mapping.get(intermediate, intermediate)
            component_maps[obj] = lambda x, f=final: f  # Identity transformation

        return NaturalTransformation(
            source_functor=functor1,
            target_functor=functor2,
            component_maps=component_maps,
            transformation_id=f"nat_trans_{len(self.natural_transformations)}"
        )

    def optimize_spectral_transitions(self, analysis_pipeline: List[Dict[str, Any]]) -> List[CategoricalFunctor]:
        """Optimize spectral analysis transitions using categorical composition"""
        optimized_functors = []

        for i, stage in enumerate(analysis_pipeline):
            category_id = self.create_spectral_category(stage)

            if i > 0:
                prev_category = f"spec_cat_{i-1}"
                transition_rules = self._infer_transition_rules(analysis_pipeline[i-1], stage)
                functor = self.construct_transition_functor(prev_category, category_id, transition_rules)
                optimized_functors.append(functor)

        return optimized_functors

    def _infer_transition_rules(self, prev_stage: Dict[str, Any],
                               curr_stage: Dict[str, Any]) -> Dict[str, str]:
        """Infer optimal transition rules between analysis stages"""
        rules = {}
        prev_keys = set(prev_stage.keys())
        curr_keys = set(curr_stage.keys())

        # Map common keys directly
        common_keys = prev_keys.intersection(curr_keys)
        for key in common_keys:
            rules[key] = key

        # Map remaining keys by semantic similarity
        remaining_prev = prev_keys - common_keys
        remaining_curr = curr_keys - common_keys

        for prev_key in remaining_prev:
            # Simple heuristic: map to most similar key name
            best_match = min(remaining_curr,
                           key=lambda x: sum(c1 != c2 for c1, c2 in zip(prev_key, x)),
                           default=prev_key)
            rules[prev_key] = best_match

        return rules


class PathSpaceHomologyAnalyzer:
    """
    Path space homology analyzer for computing homological invariants of pipeline execution paths.

    Uses persistent homology and simplicial complex analysis to identify optimal
    information-theoretic processing routes through topological analysis.
    """

    def __init__(self, max_dimension: int = 3, persistence_threshold: float = 0.1):
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold
        self.execution_paths: List[List[str]] = []
        self.path_complexes: Dict[str, SimplexComplex] = {}
        self.homology_cache: Dict[str, HomologyInvariant] = {}

    def construct_path_simplex(self, execution_path: List[str],
                              path_weights: Dict[str, float]) -> SimplexComplex:
# # #         """Construct simplicial complex from execution path"""  # Module not found  # Module not found  # Module not found
        vertices = execution_path
        simplices = {0: [[v] for v in vertices]}  # 0-simplices (vertices)

        # 1-simplices (edges between consecutive path elements)
        simplices[1] = [[vertices[i], vertices[i+1]] for i in range(len(vertices)-1)]

        # Higher-dimensional simplices for overlapping path segments
        if len(vertices) >= 3:
            simplices[2] = [[vertices[i], vertices[i+1], vertices[i+2]]
                           for i in range(len(vertices)-2)]

        if len(vertices) >= 4 and self.max_dimension >= 3:
            simplices[3] = [[vertices[i], vertices[i+1], vertices[i+2], vertices[i+3]]
                           for i in range(len(vertices)-3)]

        # Compute boundary maps
        boundary_maps = self._compute_boundary_maps(simplices, vertices)

        return SimplexComplex(
            vertices=vertices,
            simplices=simplices,
            boundary_maps=boundary_maps,
            vertex_weights=path_weights,
            simplex_weights=self._compute_simplex_weights(simplices, path_weights)
        )

    def _compute_boundary_maps(self, simplices: Dict[int, List[List[str]]],
                              vertices: List[str]) -> Dict[int, np.ndarray]:
        """Compute boundary maps for simplicial complex"""
        boundary_maps = {}
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}

        for dim in range(1, min(self.max_dimension + 1, max(simplices.keys()) + 1)):
            if dim not in simplices or (dim-1) not in simplices:
                continue

            curr_simplices = simplices[dim]
            prev_simplices = simplices[dim-1]

            # Create boundary matrix
            boundary_matrix = np.zeros((len(prev_simplices), len(curr_simplices)))

            for j, simplex in enumerate(curr_simplices):
                # Each face of the simplex
                for i, vertex in enumerate(simplex):
                    face = simplex[:i] + simplex[i+1:]  # Remove vertex i
                    if face in prev_simplices:
                        face_idx = prev_simplices.index(face)
                        boundary_matrix[face_idx, j] = (-1) ** i

            boundary_maps[dim] = boundary_matrix

        return boundary_maps

    def _compute_simplex_weights(self, simplices: Dict[int, List[List[str]]],
                               vertex_weights: Dict[str, float]) -> Dict[Tuple[str, ...], float]:
        """Compute weights for simplices based on vertex weights"""
        simplex_weights = {}

        for dim, simplex_list in simplices.items():
            for simplex in simplex_list:
                # Weight as geometric mean of vertex weights
                weights = [vertex_weights.get(v, 1.0) for v in simplex]
                if weights:
                    simplex_weight = np.prod(weights) ** (1.0 / len(weights))
                    simplex_weights[tuple(sorted(simplex))] = simplex_weight

        return simplex_weights

    def compute_persistent_homology(self, simplex_complex: SimplexComplex) -> HomologyInvariant:
        """Compute persistent homology of simplicial complex"""
        betti_numbers = []
        persistence_intervals = []
        critical_values = []

        # Simplified persistent homology computation
        for dim in range(self.max_dimension + 1):
            if dim in simplex_complex.boundary_maps:
                boundary_matrix = simplex_complex.boundary_maps[dim]

                # Compute rank of boundary matrix
                rank = np.linalg.matrix_rank(boundary_matrix)

                # Betti number = dimension of kernel / dimension of image
                if dim == 0:
                    betti_dim = len(simplex_complex.vertices) - rank
                else:
                    prev_boundary = simplex_complex.boundary_maps.get(dim-1, np.array([]))
                    prev_rank = np.linalg.matrix_rank(prev_boundary) if prev_boundary.size > 0 else 0
                    betti_dim = boundary_matrix.shape[0] - rank - prev_rank

                betti_numbers.append(max(0, betti_dim))

                # Persistence intervals (simplified)
                if boundary_matrix.size > 0:
                    eigenvals = np.real(eigvals(boundary_matrix @ boundary_matrix.T))
                    eigenvals = eigenvals[eigenvals > 1e-10]  # Filter near-zero eigenvals

                    for i, val in enumerate(eigenvals):
                        birth = i * self.persistence_threshold
                        death = birth + 1.0 / (val + 1e-10)
                        if death - birth > self.persistence_threshold:
                            persistence_intervals.append((birth, death))
                            critical_values.extend([birth, death])
            else:
                betti_numbers.append(0)

        return HomologyInvariant(
            dimension=self.max_dimension,
            betti_numbers=betti_numbers,
            torsion_coefficients=[],  # Simplified
            persistence_intervals=persistence_intervals,
            critical_values=sorted(set(critical_values)),
            homology_rank=sum(betti_numbers)
        )

    def analyze_execution_paths(self, paths: List[List[str]],
                              path_metrics: Dict[str, float]) -> Dict[str, HomologyInvariant]:
        """Analyze homological invariants of multiple execution paths"""
        path_invariants = {}

        for i, path in enumerate(paths):
            path_id = f"path_{i}"

            # Create weights for this path
            path_weights = {node: path_metrics.get(f"{path_id}_{node}", 1.0) for node in path}

            # Construct simplicial complex
            simplex_complex = self.construct_path_simplex(path, path_weights)
            self.path_complexes[path_id] = simplex_complex

            # Compute homological invariants
            invariant = self.compute_persistent_homology(simplex_complex)
            path_invariants[path_id] = invariant
            self.homology_cache[path_id] = invariant

        return path_invariants

    def identify_optimal_routes(self, path_invariants: Dict[str, HomologyInvariant]) -> List[str]:
        """Identify optimal information-theoretic processing routes"""
        optimal_paths = []

        # Score paths based on homological properties
        path_scores = {}
        for path_id, invariant in path_invariants.items():
            # Score based on:
            # 1. Homology rank (higher is more connected)
            # 2. Number of persistent features
            # 3. Total persistence (sum of interval lengths)

            homology_score = invariant.homology_rank * 2.0
            persistence_score = len(invariant.persistence_intervals) * 1.5
            interval_score = sum(death - birth for birth, death in invariant.persistence_intervals)

            total_score = homology_score + persistence_score + interval_score
            path_scores[path_id] = total_score

        # Select top paths
        sorted_paths = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)
        optimal_threshold = np.percentile([score for _, score in sorted_paths], 75) if sorted_paths else 0

        optimal_paths = [path_id for path_id, score in sorted_paths
                        if score >= optimal_threshold]

        return optimal_paths


class MathStage4AnalysisEnhancer:
    """
    Information-theoretic analysis enhancer for stage 4 pipeline processing.

    Provides mathematical validation functions using Shannon entropy and mutual
    information to analyze questions and evidence content. Integrates with the
    cluster execution controller for four-cluster questionnaire processing.

    Enhanced with DirectedAlgebraicTopologyOptimizer and PathSpaceHomologyAnalyzer
    for advanced topological analysis of information flow patterns.
    """

    def __init__(
        self,
        mode: AnalysisMode = AnalysisMode.FULL_ANALYSIS,
        entropy_threshold: float = 2.0,
        mi_threshold: float = 0.1,
        validation_threshold: float = 0.5,
        alpha: float = 0.05,
        enable_topology_analysis: bool = True,
        topology_auto_activate: bool = True
    ):
        """
        Initialize the mathematical analysis enhancer.

        Args:
            mode: Analysis mode configuration
            entropy_threshold: Minimum entropy for content significance
            mi_threshold: Minimum mutual information for relationship significance
            validation_threshold: Threshold for validation metric acceptance
            alpha: Significance level for statistical tests
            enable_topology_analysis: Enable directed algebraic topology optimization
            topology_auto_activate: Automatically activate topology analysis
        """
        self.mode = mode
        self.entropy_threshold = entropy_threshold
        self.mi_threshold = mi_threshold
        self.validation_threshold = validation_threshold
        self.alpha = alpha
        self.enable_topology_analysis = enable_topology_analysis
        self.topology_auto_activate = topology_auto_activate

        # Internal state
        self.analysis_history: List[Dict[str, Any]] = []
        self.validation_cache: Dict[str, ValidationMetrics] = {}
        self.entropy_cache: Dict[str, EntropyAnalysis] = {}
        self.mi_cache: Dict[str, MutualInformationAnalysis] = {}

        # TF-IDF vectorizer for semantic analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # Integration components
        self.evidence_processor = None
        self.question_analyzer = None
        self.adaptive_analyzer = None

        # Topological analysis components
        if self.enable_topology_analysis:
            self.topology_optimizer = DirectedAlgebraicTopologyOptimizer()
            self.homology_analyzer = PathSpaceHomologyAnalyzer()
            self.execution_paths: List[List[str]] = []
            self.path_metrics: Dict[str, float] = {}
            self.optimal_routes: List[str] = []

        self._initialize_integration()

        logger.info(f"Math Stage 4 Analysis Enhancer initialized with mode: {mode}, "
                   f"topology analysis: {enable_topology_analysis}")

    def _initialize_integration(self):
        """Initialize integration with existing pipeline components"""
        try:
            if EvidenceProcessor:
                self.evidence_processor = EvidenceProcessor()
            if QuestionAnalyzer:
                self.question_analyzer = QuestionAnalyzer()
            if AdaptiveAnalyzer:
                self.adaptive_analyzer = AdaptiveAnalyzer()

            # Auto-activate topology analysis if enabled
            if self.enable_topology_analysis and self.topology_auto_activate:
                self._auto_activate_topology_analysis()

        except Exception as e:
            logger.warning(f"Integration initialization failed: {e}")

    def _auto_activate_topology_analysis(self):
        """Automatically activate topology analysis without manual configuration"""
        try:
            # Initialize with default execution paths based on analysis modes
            default_paths = [
                ["entropy_analysis", "mutual_information", "validation", "cluster_processing"],
                ["question_analysis", "evidence_analysis", "relationship_validation", "output"],
                ["preprocessing", "feature_extraction", "information_theory", "optimization"],
                ["input_processing", "semantic_analysis", "topological_optimization", "results"]
            ]

            # Set up path metrics based on analysis thresholds
            path_metrics = {
                "entropy_analysis": self.entropy_threshold,
                "mutual_information": self.mi_threshold,
                "validation": self.validation_threshold,
                "cluster_processing": 1.0,
                "question_analysis": self.entropy_threshold,
                "evidence_analysis": self.entropy_threshold,
                "relationship_validation": self.validation_threshold,
                "output": 1.0,
                "preprocessing": 0.5,
                "feature_extraction": 0.7,
                "information_theory": self.mi_threshold,
                "optimization": 0.9,
                "input_processing": 0.6,
                "semantic_analysis": 0.8,
                "topological_optimization": 1.0,
                "results": 1.0
            }

            self.execution_paths = default_paths
            self.path_metrics = path_metrics

            # Perform initial homology analysis
            path_invariants = self.homology_analyzer.analyze_execution_paths(
                self.execution_paths, self.path_metrics
            )

            # Identify optimal routes
            self.optimal_routes = self.homology_analyzer.identify_optimal_routes(path_invariants)

            logger.info(f"Topology analysis auto-activated with {len(self.optimal_routes)} optimal routes identified")

        except Exception as e:
            logger.warning(f"Auto-activation of topology analysis failed: {e}")

    def compute_shannon_entropy(self, text: str, level: str = "character") -> float:
        """
        Compute Shannon entropy H(X) = -∑ p(x) log p(x) for text content.

        Args:
            text: Input text content
            level: Analysis level ("character", "word", "semantic")

        Returns:
            Shannon entropy value in bits
        """
        if not text:
            return 0.0

        if level == "character":
            # Character-level entropy
            char_counts = Counter(text.lower())
            total_chars = len(text)

            if total_chars == 0:
                return 0.0

            entropy = 0.0
            for count in char_counts.values():
                probability = count / total_chars
                if probability > 0:
                    entropy -= probability * math.log2(probability)

        elif level == "word":
            # Word-level entropy
            words = text.lower().split()
            word_counts = Counter(words)
            total_words = len(words)

            if total_words == 0:
                return 0.0

            entropy = 0.0
            for count in word_counts.values():
                probability = count / total_words
                if probability > 0:
                    entropy -= probability * math.log2(probability)

        elif level == "semantic":
            # Semantic entropy based on TF-IDF features
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
                feature_probs = tfidf_matrix.toarray()[0]
                feature_probs = feature_probs / np.sum(feature_probs) if np.sum(feature_probs) > 0 else feature_probs

                entropy = 0.0
                for prob in feature_probs:
                    if prob > 0:
                        entropy -= prob * math.log2(prob)
            except Exception as e:
                logger.warning(f"Semantic entropy computation failed: {e}")
                return self.compute_shannon_entropy(text, "word")
        else:
            raise ValueError(f"Unsupported entropy level: {level}")

        return entropy

    def compute_mutual_information(
        self,
        text1: str,
        text2: str,
        normalize: bool = True
    ) -> Tuple[float, float, float, float]:
        """
        Compute mutual information I(X;Y) = ∑∑ p(x,y) log(p(x,y)/(p(x)p(y))).

        Args:
            text1: First text (typically question)
            text2: Second text (typically evidence)
            normalize: Whether to normalize by joint entropy

        Returns:
            Tuple of (mutual_info, normalized_mi, text1_entropy, text2_entropy)
        """
        if not text1 or not text2:
            return 0.0, 0.0, 0.0, 0.0

        try:
            # Vectorize texts using TF-IDF
            texts = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

            # Convert to probability distributions
            vec1 = tfidf_matrix[0].toarray()[0]
            vec2 = tfidf_matrix[1].toarray()[0]

            # Normalize to probabilities
            vec1 = vec1 / np.sum(vec1) if np.sum(vec1) > 0 else vec1
            vec2 = vec2 / np.sum(vec2) if np.sum(vec2) > 0 else vec2

            # Compute individual entropies
            h1 = -np.sum([p * math.log2(p) for p in vec1 if p > 0])
            h2 = -np.sum([p * math.log2(p) for p in vec2 if p > 0])

            # Create joint distribution (simplified approach using element-wise minimum)
            joint_prob = np.minimum(vec1, vec2)
            joint_prob = joint_prob / np.sum(joint_prob) if np.sum(joint_prob) > 0 else joint_prob

            # Compute joint entropy
            joint_entropy = -np.sum([p * math.log2(p) for p in joint_prob if p > 0])

            # Compute mutual information
            mutual_info = h1 + h2 - joint_entropy

            # Normalize if requested
            normalized_mi = mutual_info / joint_entropy if joint_entropy > 0 and normalize else mutual_info

            return mutual_info, normalized_mi, h1, h2

        except Exception as e:
            logger.warning(f"Mutual information computation failed: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def analyze_question_entropy(self, question_text: str, question_id: str) -> EntropyAnalysis:
        """
        Comprehensive entropy analysis of question text.

        Args:
            question_text: The question content
            question_id: Unique identifier for the question

        Returns:
            EntropyAnalysis results
        """
        if question_id in self.entropy_cache:
            return self.entropy_cache[question_id]

        # Compute different entropy levels
        char_entropy = self.compute_shannon_entropy(question_text, "character")
        word_entropy = self.compute_shannon_entropy(question_text, "word")
        semantic_entropy = self.compute_shannon_entropy(question_text, "semantic")

        # Overall Shannon entropy (average of levels)
        shannon_entropy = (char_entropy + word_entropy + semantic_entropy) / 3

        # Complexity and information density
        complexity_score = semantic_entropy / max(word_entropy, 1e-6)
        information_density = shannon_entropy / max(len(question_text), 1)

        analysis = EntropyAnalysis(
            text_id=question_id,
            text_content=question_text,
            shannon_entropy=shannon_entropy,
            character_entropy=char_entropy,
            word_entropy=word_entropy,
            semantic_entropy=semantic_entropy,
            complexity_score=complexity_score,
            information_density=information_density
        )

        self.entropy_cache[question_id] = analysis
        return analysis

    def analyze_evidence_entropy(self, evidence_text: str, evidence_id: str) -> EntropyAnalysis:
        """
        Comprehensive entropy analysis of evidence content.

        Args:
            evidence_text: The evidence content
            evidence_id: Unique identifier for the evidence

        Returns:
            EntropyAnalysis results
        """
        return self.analyze_question_entropy(evidence_text, evidence_id)

    def compute_question_evidence_mi(
        self,
        question_text: str,
        evidence_text: str,
        question_id: str,
        evidence_id: str
    ) -> MutualInformationAnalysis:
        """
        Compute mutual information between question and evidence content.

        Args:
            question_text: Question content
            evidence_text: Evidence content
            question_id: Question identifier
            evidence_id: Evidence identifier

        Returns:
            MutualInformationAnalysis results
        """
        pair_key = f"{question_id}_{evidence_id}"
        if pair_key in self.mi_cache:
            return self.mi_cache[pair_key]

        # Compute mutual information
        mi, normalized_mi, q_entropy, e_entropy = self.compute_mutual_information(
            question_text, evidence_text
        )

        # Compute semantic similarity using cosine similarity
        try:
            texts = [question_text, evidence_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0]
        except Exception:
            similarity = 0.0

        # Joint entropy
        joint_entropy = q_entropy + e_entropy - mi

        # Information gain (mutual information normalized by question entropy)
        info_gain = mi / max(q_entropy, 1e-6)

        # Relevance score combining MI and similarity
        relevance_score = (normalized_mi + similarity) / 2

        analysis = MutualInformationAnalysis(
            question_id=question_id,
            evidence_id=evidence_id,
            mutual_information=mi,
            normalized_mi=normalized_mi,
            question_entropy=q_entropy,
            evidence_entropy=e_entropy,
            joint_entropy=joint_entropy,
            semantic_similarity=similarity,
            information_gain=info_gain,
            relevance_score=relevance_score
        )

        self.mi_cache[pair_key] = analysis
        return analysis

    def validate_question_evidence_relationship(
        self,
        question_text: str,
        evidence_text: str,
        question_id: str,
        evidence_id: str
    ) -> ValidationMetrics:
        """
        Mathematical validation of question-evidence relationship using information theory.

        Args:
            question_text: Question content
            evidence_text: Evidence content
            question_id: Question identifier
            evidence_id: Evidence identifier

        Returns:
            ValidationMetrics with comprehensive analysis
        """
        pair_key = f"{question_id}_{evidence_id}"
        if pair_key in self.validation_cache:
            return self.validation_cache[pair_key]

        # Get entropy analyses
        q_entropy = self.analyze_question_entropy(question_text, question_id)
        e_entropy = self.analyze_evidence_entropy(evidence_text, evidence_id)

        # Get mutual information analysis
        mi_analysis = self.compute_question_evidence_mi(
            question_text, evidence_text, question_id, evidence_id
        )

        # Compute validation metrics

        # 1. Entropy consistency: Similar complexity levels
        entropy_diff = abs(q_entropy.shannon_entropy - e_entropy.shannon_entropy)
        max_entropy = max(q_entropy.shannon_entropy, e_entropy.shannon_entropy)
        entropy_consistency = 1.0 - (entropy_diff / max(max_entropy, 1e-6))

        # 2. Information coherence: High mutual information
        information_coherence = mi_analysis.normalized_mi

        # 3. Semantic alignment: Cosine similarity
        semantic_alignment = mi_analysis.semantic_similarity

        # 4. Overall validation score
        validation_score = (
            entropy_consistency * 0.3 +
            information_coherence * 0.4 +
            semantic_alignment * 0.3
        )

        # Validation criteria
        validation_passed = (
            validation_score >= self.validation_threshold and
            mi_analysis.mutual_information >= self.mi_threshold and
            q_entropy.shannon_entropy >= self.entropy_threshold
        )

        # Validation reasons
        reasons = []
        if validation_score < self.validation_threshold:
            reasons.append(f"Low validation score: {validation_score:.3f}")
        if mi_analysis.mutual_information < self.mi_threshold:
            reasons.append(f"Low mutual information: {mi_analysis.mutual_information:.3f}")
        if q_entropy.shannon_entropy < self.entropy_threshold:
            reasons.append(f"Low question entropy: {q_entropy.shannon_entropy:.3f}")

        # Confidence interval (simplified)
        ci_lower = max(0.0, validation_score - 0.1)
        ci_upper = min(1.0, validation_score + 0.1)

        metrics = ValidationMetrics(
            pair_id=pair_key,
            entropy_consistency=entropy_consistency,
            information_coherence=information_coherence,
            semantic_alignment=semantic_alignment,
            confidence_interval=(ci_lower, ci_upper),
            validation_passed=validation_passed,
            validation_reasons=reasons
        )

        self.validation_cache[pair_key] = metrics

        # Apply topological optimization if enabled
        if self.enable_topology_analysis and hasattr(self, 'topology_optimizer'):
            self._apply_topological_enhancement(metrics, question_text, evidence_text)

        return metrics

    def process_cluster_analysis(
        self,
        cluster_data: Dict[str, Any],
        cluster_id: str
    ) -> Dict[str, Any]:
        """
        Process information-theoretic analysis for a specific cluster in the four-cluster workflow.

        Args:
            cluster_data: Data for the cluster processing
            cluster_id: Cluster identifier (C1, C2, C3, C4)

        Returns:
            Enhanced cluster data with mathematical analysis
        """
        analysis_results = {
            "cluster_id": cluster_id,
            "entropy_analyses": {},
            "mi_analyses": {},
            "validation_metrics": {},
            "cluster_summary": {}
        }

# # #         # Extract questions and evidence from cluster data  # Module not found  # Module not found  # Module not found
        questions = cluster_data.get("questions", [])
        evidence_items = cluster_data.get("evidence", [])

        # Analyze questions
        for question in questions:
            if isinstance(question, dict):
                q_id = question.get("question_id", question.get("id"))
                q_text = question.get("text", question.get("question"))

                if q_id and q_text:
                    entropy_analysis = self.analyze_question_entropy(q_text, q_id)
                    analysis_results["entropy_analyses"][q_id] = entropy_analysis

        # Analyze evidence
        for evidence in evidence_items:
            if isinstance(evidence, dict):
                e_id = evidence.get("evidence_id", evidence.get("id"))
                e_text = evidence.get("content", evidence.get("text"))

                if e_id and e_text:
                    entropy_analysis = self.analyze_evidence_entropy(e_text, e_id)
                    analysis_results["entropy_analyses"][e_id] = entropy_analysis

        # Cross-analysis: Question-Evidence relationships
        for question in questions:
            if not isinstance(question, dict):
                continue

            q_id = question.get("question_id", question.get("id"))
            q_text = question.get("text", question.get("question"))

            if not (q_id and q_text):
                continue

            for evidence in evidence_items:
                if not isinstance(evidence, dict):
                    continue

                e_id = evidence.get("evidence_id", evidence.get("id"))
                e_text = evidence.get("content", evidence.get("text"))

                if not (e_id and e_text):
                    continue

                # Mutual information analysis
                mi_analysis = self.compute_question_evidence_mi(q_text, e_text, q_id, e_id)
                pair_key = f"{q_id}_{e_id}"
                analysis_results["mi_analyses"][pair_key] = mi_analysis

                # Validation metrics
                validation = self.validate_question_evidence_relationship(
                    q_text, e_text, q_id, e_id
                )
                analysis_results["validation_metrics"][pair_key] = validation

        # Cluster summary statistics
        entropy_values = [
            analysis.shannon_entropy
            for analysis in analysis_results["entropy_analyses"].values()
        ]
        mi_values = [
            analysis.mutual_information
            for analysis in analysis_results["mi_analyses"].values()
        ]
        validation_scores = [
            (metrics.entropy_consistency + metrics.information_coherence + metrics.semantic_alignment) / 3
            for metrics in analysis_results["validation_metrics"].values()
        ]

        analysis_results["cluster_summary"] = {
            "total_entropy_analyses": len(analysis_results["entropy_analyses"]),
            "total_mi_analyses": len(analysis_results["mi_analyses"]),
            "total_validations": len(analysis_results["validation_metrics"]),
            "avg_entropy": np.mean(entropy_values) if entropy_values else 0.0,
            "avg_mutual_information": np.mean(mi_values) if mi_values else 0.0,
            "avg_validation_score": np.mean(validation_scores) if validation_scores else 0.0,
            "validation_pass_rate": sum(
                1 for m in analysis_results["validation_metrics"].values() if m.validation_passed
            ) / max(len(analysis_results["validation_metrics"]), 1)
        }

        # Store in history
        self.analysis_history.append({
            "timestamp": datetime.now(),
            "cluster_id": cluster_id,
            "analysis_results": analysis_results
        })

        return analysis_results

    def _apply_topological_enhancement(self, metrics: ValidationMetrics,
                                     question_text: str, evidence_text: str):
        """Apply topological enhancement to validation metrics"""
        try:
            # Create analysis pipeline for this specific validation
            pipeline = [
                {"question": question_text, "entropy": metrics.entropy_consistency},
                {"evidence": evidence_text, "coherence": metrics.information_coherence},
                {"validation": metrics.semantic_alignment, "result": metrics.validation_passed}
            ]

            # Optimize using categorical functors
            functors = self.topology_optimizer.optimize_spectral_transitions(pipeline)

            # Update path metrics based on optimization
            if functors:
                enhancement_factor = len([f for f in functors if f.composition_preserving]) / len(functors)

                # Apply enhancement to validation score
                original_score = (metrics.entropy_consistency + metrics.information_coherence +
                                metrics.semantic_alignment) / 3
                enhanced_score = min(1.0, original_score * (1.0 + enhancement_factor * 0.1))

                # Update confidence interval
                ci_lower, ci_upper = metrics.confidence_interval
                enhancement_width = (ci_upper - ci_lower) * (1.0 - enhancement_factor * 0.05)
                metrics.confidence_interval = (
                    max(0.0, enhanced_score - enhancement_width/2),
                    min(1.0, enhanced_score + enhancement_width/2)
                )

        except Exception as e:
            logger.warning(f"Topological enhancement failed: {e}")

    def get_topology_optimization_results(self) -> Dict[str, Any]:
# # #         """Get results from topology optimization analysis"""  # Module not found  # Module not found  # Module not found
        if not self.enable_topology_analysis:
            return {"topology_analysis_enabled": False}

        results = {
            "topology_analysis_enabled": True,
            "optimal_routes": self.optimal_routes,
            "execution_paths": len(self.execution_paths),
            "path_complexes": len(self.homology_analyzer.path_complexes),
            "homology_invariants": {}
        }

        # Add homology invariant summaries
        for path_id, invariant in self.homology_analyzer.homology_cache.items():
            results["homology_invariants"][path_id] = {
                "dimension": invariant.dimension,
                "betti_numbers": invariant.betti_numbers,
                "homology_rank": invariant.homology_rank,
                "persistence_intervals": len(invariant.persistence_intervals),
                "critical_values": len(invariant.critical_values)
            }

        return results

    def update_execution_paths(self, new_paths: List[List[str]],
                             new_metrics: Dict[str, float]):
        """Update execution paths and recompute optimal routes"""
        if not self.enable_topology_analysis:
            logger.warning("Topology analysis not enabled")
            return

        self.execution_paths.extend(new_paths)
        self.path_metrics.update(new_metrics)

        # Recompute homology analysis
        path_invariants = self.homology_analyzer.analyze_execution_paths(
            new_paths, new_metrics
        )

        # Update optimal routes
        all_invariants = {**self.homology_analyzer.homology_cache, **path_invariants}
        self.optimal_routes = self.homology_analyzer.identify_optimal_routes(all_invariants)

        logger.info(f"Updated execution paths: {len(self.execution_paths)} total paths, "
                   f"{len(self.optimal_routes)} optimal routes")

    def apply_categorical_functor_optimization(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply categorical functor optimization to analysis data"""
        if not self.enable_topology_analysis:
            return analysis_data

        try:
            # Create spectral category for the analysis data
            category_id = self.topology_optimizer.create_spectral_category(analysis_data)

            # If we have previous analysis, create transition functor
            if hasattr(self, '_previous_category_id'):
                transition_rules = self.topology_optimizer._infer_transition_rules(
                    self._previous_analysis_data, analysis_data
                )
                functor = self.topology_optimizer.construct_transition_functor(
                    self._previous_category_id, category_id, transition_rules
                )

                # Apply functor transformation (simplified)
                optimized_data = analysis_data.copy()
                optimization_factor = 1.0 + (0.1 if functor.composition_preserving else -0.05)

                # Apply optimization to numerical values
                for key, value in optimized_data.items():
                    if isinstance(value, (int, float)):
                        optimized_data[key] = value * optimization_factor
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)):
                                optimized_data[key][subkey] = subvalue * optimization_factor

                analysis_data = optimized_data

            self._previous_category_id = category_id
            self._previous_analysis_data = analysis_data.copy()

        except Exception as e:
            logger.warning(f"Categorical functor optimization failed: {e}")

        return analysis_data

    def integrate_with_cluster_controller(
        self,
        controller_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integration hook for cluster execution controller during four-cluster processing.

        Args:
# # #             controller_data: Data from cluster execution controller  # Module not found  # Module not found  # Module not found
            context: Processing context

        Returns:
            Enhanced data with mathematical analysis
        """
        enhanced_data = controller_data.copy()

        # Process each cluster if present
        cluster_processing = controller_data.get("cluster_processing", {})
        mathematical_analysis = {}

        for cluster_id, cluster_data in cluster_processing.items():
            if cluster_id in ["C1", "C2", "C3", "C4"]:
                cluster_analysis = self.process_cluster_analysis(cluster_data, cluster_id)
                mathematical_analysis[cluster_id] = cluster_analysis

        # Add mathematical analysis to enhanced data
        enhanced_data["mathematical_analysis"] = mathematical_analysis

        # Compute cross-cluster statistics
        if mathematical_analysis:
            cross_cluster_stats = self._compute_cross_cluster_statistics(mathematical_analysis)
            enhanced_data["cross_cluster_mathematical_summary"] = cross_cluster_stats

        return enhanced_data

    def _compute_cross_cluster_statistics(self, mathematical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics across all clusters"""
        all_entropy_values = []
        all_mi_values = []
        all_validation_scores = []
        cluster_summaries = {}

        for cluster_id, analysis in mathematical_analysis.items():
            summary = analysis.get("cluster_summary", {})
            cluster_summaries[cluster_id] = summary

            if summary.get("avg_entropy"):
                all_entropy_values.append(summary["avg_entropy"])
            if summary.get("avg_mutual_information"):
                all_mi_values.append(summary["avg_mutual_information"])
            if summary.get("avg_validation_score"):
                all_validation_scores.append(summary["avg_validation_score"])

        return {
            "total_clusters_analyzed": len(mathematical_analysis),
            "overall_avg_entropy": np.mean(all_entropy_values) if all_entropy_values else 0.0,
            "overall_avg_mi": np.mean(all_mi_values) if all_mi_values else 0.0,
            "overall_avg_validation": np.mean(all_validation_scores) if all_validation_scores else 0.0,
            "entropy_std": np.std(all_entropy_values) if len(all_entropy_values) > 1 else 0.0,
            "mi_std": np.std(all_mi_values) if len(all_mi_values) > 1 else 0.0,
            "validation_std": np.std(all_validation_scores) if len(all_validation_scores) > 1 else 0.0,
            "cluster_summaries": cluster_summaries
        }

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses performed"""
        return {
            "total_analyses": len(self.analysis_history),
            "cached_entropy_analyses": len(self.entropy_cache),
            "cached_mi_analyses": len(self.mi_cache),
            "cached_validations": len(self.validation_cache),
            "mode": self.mode.value,
            "thresholds": {
                "entropy": self.entropy_threshold,
                "mutual_information": self.mi_threshold,
                "validation": self.validation_threshold
            },
            "last_analysis": self.analysis_history[-1]["timestamp"] if self.analysis_history else None
        }


def process(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main processing function for integration with comprehensive pipeline orchestrator.

    Args:
# # #         data: Input data from pipeline  # Module not found  # Module not found  # Module not found
        context: Processing context

    Returns:
        Enhanced data with mathematical analysis
    """
    ctx = context or {}
    enhancer = MathStage4AnalysisEnhancer(
        mode=AnalysisMode.FULL_ANALYSIS,
        entropy_threshold=ctx.get("entropy_threshold", 2.0),
        mi_threshold=ctx.get("mi_threshold", 0.1),
        validation_threshold=ctx.get("validation_threshold", 0.5)
    )

    if isinstance(data, dict):
        # Check if this is cluster controller data
        if "cluster_audit" in data or "cluster_processing" in data:
            return enhancer.integrate_with_cluster_controller(data, ctx)

        # General processing
        enhanced_data = data.copy()
        enhanced_data["math_stage4_analysis"] = enhancer.get_analysis_summary()
        return enhanced_data

    return {"math_stage4_analysis": "No valid data provided", "original_data": data}


if __name__ == "__main__":
    # Demo usage
    enhancer = MathStage4AnalysisEnhancer()

    # Sample question and evidence
    question = "¿Se garantiza el derecho a la vida y seguridad de la población?"
    evidence = "El plan nacional de desarrollo incluye programas específicos para la protección de la vida y la seguridad ciudadana."

    # Analyze
    q_entropy = enhancer.analyze_question_entropy(question, "Q1")
    e_entropy = enhancer.analyze_evidence_entropy(evidence, "E1")
    mi_analysis = enhancer.compute_question_evidence_mi(question, evidence, "Q1", "E1")
    validation = enhancer.validate_question_evidence_relationship(question, evidence, "Q1", "E1")

    print(f"Question Entropy: {q_entropy.shannon_entropy:.3f}")
    print(f"Evidence Entropy: {e_entropy.shannon_entropy:.3f}")
    print(f"Mutual Information: {mi_analysis.mutual_information:.3f}")
    print(f"Validation Passed: {validation.validation_passed}")
