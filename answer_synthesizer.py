"""
AdvancedAnswerSynthesizer: A sophisticated, self-contained evidence synthesis system
with neural-inspired scoring, multi-modal reasoning, adaptive confidence estimation,
and comprehensive audit trails—completely offline and dependency-free.

Key Enhancements:
- Neural-inspired attention mechanisms for evidence weighting
- Multi-modal reasoning with structured and unstructured evidence
- Adaptive confidence estimation with Bayesian updating
- Hierarchical synthesis with sub-question decomposition
- Comprehensive audit trails with causality tracking
- Self-validating integrity checks with cryptographic hashing
- Contextual reasoning with temporal and domain awareness
"""

import json
import math
import hashlib
import hmac
import time
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable
from enum import Enum
from collections import defaultdict, Counter
import re


class ConfidenceMethod(Enum):
    """Different confidence estimation approaches."""
    FREQUENTIST = "frequentist"
    BAYESIAN = "bayesian"
    CONFORMAL = "conformal"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


class ReasoningMode(Enum):
    """Different reasoning approaches for synthesis."""
    INDUCTIVE = "inductive"
    DEDUCTIVE = "deductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


class EvidenceType(Enum):
    """Types of evidence for specialized handling."""
    TEXTUAL = "textual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    STRUCTURED = "structured"
    METADATA = "metadata"


@dataclass
class AttentionWeight:
    """Neural-inspired attention weight for evidence pieces."""
    value: float
    reasoning: str
    context_sensitivity: float = 1.0
    temporal_decay: float = 1.0
    domain_relevance: float = 1.0


@dataclass
class CausalLink:
    """Represents causal relationships between evidence pieces."""
    source_id: str
    target_id: str
    strength: float
    direction: str  # "forward", "backward", "bidirectional"
    confidence: float
    mechanism: Optional[str] = None


@dataclass
class AdvancedPremise:
    """Enhanced premise with multi-dimensional analysis."""
    text: str
    evidence_id: Optional[str] = None
    evidence_type: EvidenceType = EvidenceType.TEXTUAL
    citation: Optional[Dict[str, Any]] = None

    # Advanced scoring
    semantic_score: float = 0.0
    syntactic_score: float = 0.0
    pragmatic_score: float = 0.0
    attention_weight: Optional[AttentionWeight] = None

    # Contextual features
    temporal_context: Optional[Dict[str, Any]] = None
    domain_context: Optional[str] = None
    certainty_markers: List[str] = field(default_factory=list)

    # Causal and logical structure
    causal_links: List[CausalLink] = field(default_factory=list)
    logical_operators: List[str] = field(default_factory=list)

    # Quality indicators
    source_credibility: float = 0.5
    information_completeness: float = 0.5
    consistency_score: float = 1.0

    # Metadata
    extraction_timestamp: float = field(default_factory=time.time)
    processing_flags: Set[str] = field(default_factory=set)


@dataclass
class ReasoningTrace:
    """Tracks the reasoning process for explainability."""
    step_id: str
    operation: str
    input_premises: List[str]
    output_conclusion: str
    confidence_delta: float
    reasoning_mode: ReasoningMode
    evidence_synthesis: Dict[str, Any]
    temporal_context: Optional[Dict[str, Any]] = None


@dataclass
class AdvancedSynthesizedAnswer:
    """Sophisticated answer with comprehensive metadata and audit trails."""
    question: str
    verdict: str
    rationale: str

    # Enhanced premises and structure
    premises: List[AdvancedPremise] = field(default_factory=list)
    sub_questions: List['AdvancedSynthesizedAnswer'] = field(default_factory=list)

    # Advanced confidence estimation
    confidence: float = 0.0
    confidence_method: ConfidenceMethod = ConfidenceMethod.ADAPTIVE
    confidence_interval: Optional[Tuple[float, float]] = None
    confidence_distribution: Optional[Dict[str, float]] = None
    uncertainty_sources: List[str] = field(default_factory=list)

    # Reasoning and causality
    reasoning_trace: List[ReasoningTrace] = field(default_factory=list)
    causal_graph: List[CausalLink] = field(default_factory=list)
    alternative_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    # Quality and validation
    synthesis_quality_score: float = 0.0
    internal_consistency: float = 1.0
    completeness_score: float = 0.0

    # Citations and references
    citations: List[Dict[str, Any]] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)

    # Compliance and standards
    unmet_requirements: List[Dict[str, Any]] = field(default_factory=list)
    compliance_score: float = 1.0

    # Metadata and audit
    synthesis_timestamp: float = field(default_factory=time.time)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    # Cryptographic integrity
    _integrity_hash: Optional[str] = field(default=None, init=False)
    _audit_trail: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._compute_integrity_hash()
        self._audit_trail.append({
            "operation": "initialization",
            "timestamp": time.time(),
            "hash": self._integrity_hash
        })

    def _compute_integrity_hash(self) -> None:
        """Compute cryptographic hash for integrity verification."""
        content = {
            "question": self.question,
            "verdict": self.verdict,
            "premises_count": len(self.premises),
            "confidence": self.confidence,
            "timestamp": self.synthesis_timestamp
        }
        content_str = json.dumps(content, sort_keys=True)
        self._integrity_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify the integrity of the synthesized answer."""
        current_hash = self._integrity_hash
        self._compute_integrity_hash()
        is_valid = current_hash == self._integrity_hash
        self._integrity_hash = current_hash  # Restore original
        return is_valid

    def add_audit_entry(self, operation: str, details: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        self._audit_trail.append({
            "operation": operation,
            "timestamp": time.time(),
            "details": details,
            "hash_before": self._integrity_hash
        })
        self._compute_integrity_hash()


class AdvancedSemanticAnalyzer:
    """Sophisticated semantic analysis without external dependencies."""

    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'would', 'there',
            'we', 'when', 'your', 'can', 'said', 'them', 'who', 'may', 'been'
        }
        self.certainty_markers = {
            'high': ['definitely', 'certainly', 'absolutely', 'clearly', 'undoubtedly'],
            'medium': ['likely', 'probably', 'generally', 'typically', 'usually'],
            'low': ['possibly', 'maybe', 'perhaps', 'might', 'could', 'potentially']
        }

    def extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive semantic features from text."""
        words = self._tokenize(text.lower())
        word_count = Counter(words)

        # Basic statistics
        features = {
            "word_count": len(words),
            "unique_words": len(word_count),
            "lexical_diversity": len(word_count) / max(len(words), 1),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1)
        }

        # Certainty analysis
        certainty_score = 0.0
        certainty_count = 0
        for level, markers in self.certainty_markers.items():
            count = sum(text.lower().count(marker) for marker in markers)
            if level == 'high':
                certainty_score += count * 1.0
            elif level == 'medium':
                certainty_score += count * 0.6
            else:
                certainty_score += count * 0.3
            certainty_count += count

        features["certainty_score"] = certainty_score / max(certainty_count, 1)
        features["certainty_markers"] = certainty_count

        # Numeric content analysis
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        features["numeric_density"] = len(numbers) / max(len(words), 1)
        features["has_percentages"] = any('%' in n for n in numbers)

        # Temporal markers
        temporal_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(?:q1|q2|q3|q4)\b',  # Quarters
            r'\b(?:by|until|after|before|during)\s+\d{4}\b'
        ]
        temporal_matches = sum(len(re.findall(pattern, text.lower())) for pattern in temporal_patterns)
        features["temporal_density"] = temporal_matches / max(len(words), 1)

        return features

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Advanced semantic similarity without external models."""
        # Tokenize and clean
        words1 = set(self._tokenize(text1.lower())) - self.stop_words
        words2 = set(self._tokenize(text2.lower())) - self.stop_words

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / max(union, 1)

        # N-gram overlap (bigrams)
        bigrams1 = self._extract_ngrams(text1.lower(), 2)
        bigrams2 = self._extract_ngrams(text2.lower(), 2)
        bigram_overlap = len(bigrams1 & bigrams2) / max(len(bigrams1 | bigrams2), 1)

        # Character-level similarity (for handling typos)
        char_sim = self._character_similarity(text1.lower(), text2.lower())

        # Weighted combination
        return 0.5 * jaccard + 0.3 * bigram_overlap + 0.2 * char_sim

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text)

    def _extract_ngrams(self, text: str, n: int) -> Set[str]:
        """Extract n-grams from text."""
        words = self._tokenize(text)
        return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}

    def _character_similarity(self, s1: str, s2: str) -> float:
        """Character-level similarity using longest common subsequence."""
        if not s1 or not s2:
            return 0.0

        # Simplified LCS
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return 2 * lcs_length / (m + n)


class AttentionMechanism:
    """Neural-inspired attention mechanism for evidence weighting."""

    def __init__(self, context_window: int = 5):
        self.context_window = context_window
        self.semantic_analyzer = AdvancedSemanticAnalyzer()

    def compute_attention_weights(
            self,
            question: str,
            premises: List[AdvancedPremise]
    ) -> List[AttentionWeight]:
        """Compute attention weights for premises given the question."""
        weights = []
        question_features = self.semantic_analyzer.extract_semantic_features(question)

        for i, premise in enumerate(premises):
            # Base semantic similarity
            semantic_sim = self.semantic_analyzer.compute_semantic_similarity(
                question, premise.text
            )

            # Context sensitivity (nearby premises influence)
            context_boost = self._compute_context_boost(i, premises, question)

            # Temporal decay (more recent evidence weighted higher)
            temporal_factor = self._compute_temporal_factor(premise)

            # Domain relevance
            domain_factor = self._compute_domain_relevance(question_features, premise)

            # Combine factors
            attention_value = (
                    semantic_sim * 0.4 +
                    context_boost * 0.2 +
                    temporal_factor * 0.2 +
                    domain_factor * 0.2
            )

            weights.append(AttentionWeight(
                value=min(max(attention_value, 0.0), 1.0),
                reasoning=f"Semantic: {semantic_sim:.3f}, Context: {context_boost:.3f}, "
                          f"Temporal: {temporal_factor:.3f}, Domain: {domain_factor:.3f}",
                context_sensitivity=context_boost,
                temporal_decay=temporal_factor,
                domain_relevance=domain_factor
            ))

        return weights

    def _compute_context_boost(
            self,
            index: int,
            premises: List[AdvancedPremise],
            question: str
    ) -> float:
        """Compute context boost based on nearby premises."""
        if len(premises) <= 1:
            return 0.5

        start = max(0, index - self.context_window // 2)
        end = min(len(premises), index + self.context_window // 2 + 1)

        context_premises = [p for i, p in enumerate(premises[start:end]) if i != index - start]
        if not context_premises:
            return 0.5

        # Average similarity to context
        context_similarity = sum(
            self.semantic_analyzer.compute_semantic_similarity(
                premises[index].text, p.text
            ) for p in context_premises
        ) / len(context_premises)

        return context_similarity

    def _compute_temporal_factor(self, premise: AdvancedPremise) -> float:
        """Compute temporal relevance factor."""
        if premise.temporal_context is None:
            return 0.5

        # Simple exponential decay based on age
        age_hours = (time.time() - premise.extraction_timestamp) / 3600
        decay_rate = 0.01  # 1% per hour
        return math.exp(-decay_rate * age_hours)

    def _compute_domain_relevance(
            self,
            question_features: Dict[str, Any],
            premise: AdvancedPremise
    ) -> float:
        """Compute domain-specific relevance."""
        premise_features = self.semantic_analyzer.extract_semantic_features(premise.text)

        # Compare feature distributions
        similarity_factors = []

        # Numeric content alignment
        if question_features.get("has_percentages") and premise_features.get("has_percentages"):
            similarity_factors.append(0.8)
        elif question_features.get("numeric_density", 0) > 0.1 and premise_features.get("numeric_density", 0) > 0.1:
            similarity_factors.append(0.6)

        # Temporal content alignment
        if question_features.get("temporal_density", 0) > 0.05 and premise_features.get("temporal_density", 0) > 0.05:
            similarity_factors.append(0.7)

        # Certainty alignment
        certainty_diff = abs(
            question_features.get("certainty_score", 0.5) -
            premise_features.get("certainty_score", 0.5)
        )
        similarity_factors.append(1.0 - certainty_diff)

        return sum(similarity_factors) / max(len(similarity_factors), 1) if similarity_factors else 0.5


class BayesianConfidenceEstimator:
    """Bayesian approach to confidence estimation with prior knowledge."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha  # Prior successes
        self.prior_beta = prior_beta  # Prior failures
        self.evidence_history: List[Tuple[float, bool]] = []

    def update_posterior(self, evidence_scores: List[float], ground_truth: Optional[bool] = None) -> Tuple[
        float, float]:
        """Update posterior distribution based on evidence."""
        # Convert evidence scores to binary outcomes (simplified)
        mean_score = sum(evidence_scores) / max(len(evidence_scores), 1)

        if ground_truth is not None:
            self.evidence_history.append((mean_score, ground_truth))

            # Update hyperparameters based on evidence
            if ground_truth:
                self.prior_alpha += mean_score
            else:
                self.prior_beta += (1.0 - mean_score)

        return self.prior_alpha, self.prior_beta

    def estimate_confidence(self, evidence_scores: List[float]) -> Tuple[float, Tuple[float, float]]:
        """Estimate confidence using Beta distribution."""
        alpha, beta = self.update_posterior(evidence_scores)

        # Point estimate (posterior mean)
        confidence = alpha / (alpha + beta)

        # Credible interval (simplified)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std_dev = math.sqrt(variance)

        # 95% credible interval approximation
        lower = max(0.0, confidence - 1.96 * std_dev)
        upper = min(1.0, confidence + 1.96 * std_dev)

        return confidence, (lower, upper)

    def get_uncertainty_sources(self, evidence_scores: List[float]) -> List[str]:
        """Identify sources of uncertainty."""
        sources = []

        if len(evidence_scores) < 3:
            sources.append("Insufficient evidence quantity")

        if len(evidence_scores) > 0:
            score_variance = statistics.variance(evidence_scores)
            if score_variance > 0.1:
                sources.append("High evidence score variance")

            mean_score = statistics.mean(evidence_scores)
            if 0.3 < mean_score < 0.7:
                sources.append("Evidence scores in ambiguous range")

        if len(self.evidence_history) < 10:
            sources.append("Limited calibration history")

        return sources


class HierarchicalReasoning:
    """Hierarchical reasoning for complex question decomposition."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.semantic_analyzer = AdvancedSemanticAnalyzer()

    def decompose_question(self, question: str) -> List[str]:
        """Decompose complex question into sub-questions."""
        # Simple heuristic-based decomposition
        sub_questions = []

        # Look for conjunctions
        if ' and ' in question.lower():
            parts = question.lower().split(' and ')
            sub_questions.extend([part.strip() + '?' for part in parts if len(part.strip()) > 10])

        # Look for multiple clauses
        clauses = re.split(r'[,;]', question)
        if len(clauses) > 1:
            sub_questions.extend([clause.strip() + '?' for clause in clauses if len(clause.strip()) > 10])

        # Look for temporal elements
        temporal_match = re.search(r'\b(by|until|after|before|during)\s+(\d{4}|\w+\s+\d{4})\b', question.lower())
        if temporal_match:
            temporal_part = f"What is the timeline mentioned in: {question}?"
            sub_questions.append(temporal_part)

        # Look for quantitative elements
        if re.search(r'\d+%|\d+\s*percent|measurable|target|goal', question.lower()):
            quantitative_part = f"What are the quantitative aspects of: {question}?"
            sub_questions.append(quantitative_part)

        return sub_questions[:3]  # Limit to 3 sub-questions

    def synthesize_hierarchical(
            self,
            main_answer: 'AdvancedSynthesizedAnswer',
            sub_answers: List['AdvancedSynthesizedAnswer']
    ) -> 'AdvancedSynthesizedAnswer':
        """Synthesize hierarchical answer from main and sub-answers."""
        # Aggregate confidences
        all_confidences = [main_answer.confidence] + [sa.confidence for sa in sub_answers]
        hierarchical_confidence = sum(all_confidences) / len(all_confidences)

        # Combine rationales
        combined_rationale = main_answer.rationale
        if sub_answers:
            combined_rationale += "\n\nSub-question analysis:"
            for i, sa in enumerate(sub_answers, 1):
                combined_rationale += f"\n{i}. {sa.question} -> {sa.verdict.upper()}"
                combined_rationale += f"\n   Rationale: {sa.rationale[:200]}..."

        # Update main answer
        main_answer.sub_questions = sub_answers
        main_answer.confidence = hierarchical_confidence
        main_answer.rationale = combined_rationale
        main_answer.add_audit_entry("hierarchical_synthesis", {
            "sub_question_count": len(sub_answers),
            "aggregated_confidence": hierarchical_confidence
        })

        return main_answer


class AdvancedAnswerSynthesizer:
    """
    The main synthesizer with all advanced capabilities integrated.
    """

    def __init__(
            self,
            confidence_method: ConfidenceMethod = ConfidenceMethod.ADAPTIVE,
            reasoning_mode: ReasoningMode = ReasoningMode.ABDUCTIVE,
            enable_hierarchical: bool = True,
            attention_mechanism: bool = True,
            random_seed: int = 42
    ):
        self.confidence_method = confidence_method
        self.reasoning_mode = reasoning_mode
        self.enable_hierarchical = enable_hierarchical
        self.attention_mechanism = attention_mechanism

        # Initialize components
        self.semantic_analyzer = AdvancedSemanticAnalyzer()
        self.attention = AttentionMechanism() if attention_mechanism else None
        self.bayesian_estimator = BayesianConfidenceEstimator()
        self.hierarchical_reasoner = HierarchicalReasoning() if enable_hierarchical else None

        # State tracking
        self.synthesis_history: List[AdvancedSynthesizedAnswer] = []
        self.performance_metrics: Dict[str, Any] = defaultdict(list)

        # Random state
        self.random_seed = random_seed

    def synthesize_answer(
            self,
            question: str,
            evidence: List[Any],
            standards: Optional[Dict[str, Any]] = None,
            context: Optional[Dict[str, Any]] = None
    ) -> AdvancedSynthesizedAnswer:
        """
        Main synthesis method with full advanced capabilities.
        """
        start_time = time.time()

        # Convert evidence to AdvancedPremises
        premises = self._convert_to_premises(evidence)

        # Apply attention mechanism if enabled
        if self.attention:
            attention_weights = self.attention.compute_attention_weights(question, premises)
            for premise, weight in zip(premises, attention_weights):
                premise.attention_weight = weight

        # Compute semantic scores
        for premise in premises:
            premise.semantic_score = self.semantic_analyzer.compute_semantic_similarity(
                question, premise.text
            )
            premise.syntactic_score = self._compute_syntactic_alignment(question, premise.text)
            premise.pragmatic_score = self._compute_pragmatic_score(question, premise)

        # Build causal graph
        causal_graph = self._build_causal_graph(premises)

        # Generate reasoning trace
        reasoning_trace = self._generate_reasoning_trace(question, premises)

        # Determine verdict using advanced logic
        verdict = self._determine_advanced_verdict(question, premises, reasoning_trace)

        # Build comprehensive rationale
        rationale = self._build_advanced_rationale(question, verdict, premises, reasoning_trace)

        # Create base answer
        answer = AdvancedSynthesizedAnswer(
            question=question,
            verdict=verdict,
            rationale=rationale,
            premises=premises,
            reasoning_trace=reasoning_trace,
            causal_graph=causal_graph,
            processing_metadata={
                "synthesis_duration": time.time() - start_time,
                "premise_count": len(premises),
                "confidence_method": self.confidence_method.value,
                "reasoning_mode": self.reasoning_mode.value,
                "context": context or {}
            }
        )

        # Apply standards checking
        if standards:
            answer = self._apply_advanced_standards(answer, standards)

        # Hierarchical reasoning if enabled
        if self.hierarchical_reasoner:
            sub_questions = self.hierarchical_reasoner.decompose_question(question)
            if sub_questions:
                sub_answers = []
                for sq in sub_questions:
                    # Recursive synthesis for sub-questions (limited depth)
                    if len(self.synthesis_history) < 10:  # Prevent infinite recursion
                        sub_answer = self.synthesize_answer(sq, evidence, None, context)
                        sub_answers.append(sub_answer)

                if sub_answers:
                    answer = self.hierarchical_reasoner.synthesize_hierarchical(answer, sub_answers)

        # Advanced confidence estimation
        confidence_result = self._estimate_advanced_confidence(answer)
        answer.confidence = confidence_result["point_estimate"]
        answer.confidence_interval = confidence_result["interval"]
        answer.confidence_distribution = confidence_result["distribution"]
        answer.uncertainty_sources = confidence_result["uncertainty_sources"]

        # Quality assessment
        answer.synthesis_quality_score = self._assess_synthesis_quality(answer)
        answer.internal_consistency = self._assess_internal_consistency(answer)
        answer.completeness_score = self._assess_completeness(answer, question)

        # Store in history
        self.synthesis_history.append(answer)

        # Update performance metrics
        self._update_performance_metrics(answer)

        return answer

    def _convert_to_premises(self, evidence: List[Any]) -> List[AdvancedPremise]:
        """Convert raw evidence to AdvancedPremise objects."""
        premises = []

        for i, ev in enumerate(evidence):
            text_val = None
            ev_id = None
            citation = None
            evidence_type = EvidenceType.TEXTUAL

            # Extract text and metadata
            if hasattr(ev, "chunk") and hasattr(ev.chunk, "text"):
                text_val = ev.chunk.text
                ev_id = getattr(ev, "evidence_id", f"ev_{i}")
                if hasattr(ev, "citation"):
                    citation = self._extract_citation_metadata(ev.citation)
            elif isinstance(ev, dict):
                text_val = ev.get("text")
                ev_id = ev.get("evidence_id", f"ev_{i}")
                citation = ev.get("citation")
                evidence_type = EvidenceType(ev.get("type", "textual"))
            elif hasattr(ev, "text"):
                text_val = ev.text
                ev_id = getattr(ev, "evidence_id", f"ev_{i}")
                citation = getattr(ev, "citation", None)

            if text_val:
                # Extract advanced features
                features = self.semantic_analyzer.extract_semantic_features(text_val)
                certainty_markers = self._extract_certainty_markers(text_val)
                temporal_context = self._extract_temporal_context(text_val)

                premise = AdvancedPremise(
                    text=text_val,
                    evidence_id=ev_id,
                    evidence_type=evidence_type,
                    citation=citation,
                    certainty_markers=certainty_markers,
                    temporal_context=temporal_context,
                    source_credibility=self._assess_source_credibility(citation),
                    information_completeness=min(features.get("word_count", 0) / 50.0, 1.0)
                )

                premises.append(premise)

        return premises

    def _extract_citation_metadata(self, citation) -> Dict[str, Any]:
        """Extract metadata from citation objects."""
        if hasattr(citation, "metadata"):
            meta = citation.metadata
            return {
                "document_id": getattr(meta, "document_id", None),
                "title": getattr(meta, "title", None),
                "author": getattr(meta, "author", None),
                "page_number": getattr(meta, "page_number", None),
                "publication_date": getattr(meta, "publication_date", None),
                "source_type": getattr(meta, "source_type", "unknown"),
                "inline_citation": getattr(citation, "inline_citation", None)
            }
        elif isinstance(citation, dict):
            return citation
        return {}

    def _extract_certainty_markers(self, text: str) -> List[str]:
        """Extract certainty markers from text."""
        markers = []
        text_lower = text.lower()

        for level, marker_list in self.semantic_analyzer.certainty_markers.items():
            for marker in marker_list:
                if marker in text_lower:
                    markers.append(f"{level}:{marker}")

        # Additional uncertainty markers
        uncertainty_markers = ['uncertain', 'unclear', 'ambiguous', 'disputed', 'contested']
        for marker in uncertainty_markers:
            if marker in text_lower:
                markers.append(f"uncertainty:{marker}")

        return markers

    def _extract_temporal_context(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract temporal context from text."""
        temporal_info = {}

        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            temporal_info["years"] = list(set(years))

        # Extract quarters
        quarters = re.findall(r'\b[Qq][1-4]\b', text)
        if quarters:
            temporal_info["quarters"] = list(set(quarters))

        # Extract months
        months = re.findall(
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
            text.lower())
        if months:
            temporal_info["months"] = list(set(months))

        # Extract temporal indicators
        temporal_indicators = re.findall(r'\b(?:by|until|after|before|during|from|to)\s+(?:\d{4}|\w+\s+\d{4})\b',
                                         text.lower())
        if temporal_indicators:
            temporal_info["indicators"] = temporal_indicators

        return temporal_info if temporal_info else None

    def _assess_source_credibility(self, citation: Optional[Dict[str, Any]]) -> float:
        """Assess source credibility based on citation metadata."""
        if not citation:
            return 0.5

        credibility_score = 0.5

        # Author credibility
        if citation.get("author"):
            credibility_score += 0.1

        # Publication type
        source_type = citation.get("source_type", "").lower()
        if source_type in ["journal", "academic", "peer_reviewed"]:
            credibility_score += 0.2
        elif source_type in ["government", "official"]:
            credibility_score += 0.15
        elif source_type in ["news", "media"]:
            credibility_score += 0.1

        # Publication date (more recent = more credible for certain domains)
        pub_date = citation.get("publication_date")
        if pub_date:
            # Simple heuristic: recent publications get slight boost
            credibility_score += 0.05

        return min(credibility_score, 1.0)

    def _compute_syntactic_alignment(self, question: str, text: str) -> float:
        """Compute syntactic alignment between question and text."""
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        t_words = set(re.findall(r'\b\w+\b', text.lower()))

        # Direct word overlap
        overlap = len(q_words & t_words)
        total = len(q_words | t_words)
        word_overlap = overlap / max(total, 1)

        # Question word types (what, how, when, etc.)
        question_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which'}
        q_question_words = q_words & question_words

        # Syntactic patterns alignment
        q_patterns = self._extract_syntactic_patterns(question)
        t_patterns = self._extract_syntactic_patterns(text)

        pattern_overlap = len(set(q_patterns) & set(t_patterns)) / max(len(set(q_patterns) | set(t_patterns)), 1)

        return 0.7 * word_overlap + 0.3 * pattern_overlap

    def _extract_syntactic_patterns(self, text: str) -> List[str]:
        """Extract syntactic patterns from text."""
        patterns = []

        # Simple POS-like patterns
        words = re.findall(r'\b\w+\b', text.lower())

        # Verb patterns (simplified)
        verb_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'would', 'can', 'could', 'should']
        for word in words:
            if word in verb_indicators:
                patterns.append(f"VERB:{word}")

        # Number patterns
        if re.search(r'\d', text):
            patterns.append("NUMERIC")

        # Temporal patterns
        if re.search(r'\b\d{4}\b', text):
            patterns.append("YEAR")

        # Question patterns
        if text.strip().endswith('?'):
            patterns.append("QUESTION")

        return patterns

    def _compute_pragmatic_score(self, question: str, premise: AdvancedPremise) -> float:
        """Compute pragmatic alignment considering context and intent."""
        score = 0.5

        # Intent matching
        question_intent = self._extract_question_intent(question)
        text_content = premise.text.lower()

        if question_intent == "factual" and any(
                marker in text_content for marker in ['according to', 'studies show', 'data indicates']):
            score += 0.2
        elif question_intent == "temporal" and premise.temporal_context:
            score += 0.3
        elif question_intent == "quantitative" and re.search(r'\d+%|\d+\s*percent', text_content):
            score += 0.25

        # Certainty alignment
        question_certainty = self._extract_question_certainty(question)
        premise_certainty = len(premise.certainty_markers) > 0

        if question_certainty and premise_certainty:
            score += 0.1
        elif not question_certainty and not premise_certainty:
            score += 0.05

        # Source credibility impact
        score = score * (0.5 + 0.5 * premise.source_credibility)

        return min(score, 1.0)

    def _extract_question_intent(self, question: str) -> str:
        """Extract the intent of the question."""
        q_lower = question.lower()

        if any(word in q_lower for word in ['when', 'by', 'until', 'timeline']):
            return "temporal"
        elif any(word in q_lower for word in ['how much', 'how many', 'percentage', 'target', 'goal']):
            return "quantitative"
        elif any(word in q_lower for word in ['what', 'which', 'describe', 'explain']):
            return "descriptive"
        elif any(word in q_lower for word in ['does', 'is', 'has', 'will']):
            return "factual"
        else:
            return "general"

    def _extract_question_certainty(self, question: str) -> bool:
        """Determine if question seeks certain/definitive information."""
        certainty_indicators = ['definitely', 'certainly', 'specific', 'exact', 'precise']
        return any(indicator in question.lower() for indicator in certainty_indicators)

    def _build_causal_graph(self, premises: List[AdvancedPremise]) -> List[CausalLink]:
        """Build causal relationship graph between premises."""
        causal_links = []

        for i, premise_a in enumerate(premises):
            for j, premise_b in enumerate(premises[i + 1:], i + 1):
                # Simple causal relationship detection
                link_strength, direction = self._detect_causal_relationship(premise_a, premise_b)

                if link_strength > 0.3:
                    causal_links.append(CausalLink(
                        source_id=premise_a.evidence_id or f"premise_{i}",
                        target_id=premise_b.evidence_id or f"premise_{j}",
                        strength=link_strength,
                        direction=direction,
                        confidence=min(premise_a.source_credibility, premise_b.source_credibility),
                        mechanism=self._identify_causal_mechanism(premise_a.text, premise_b.text)
                    ))

        return causal_links

    def _detect_causal_relationship(self, premise_a: AdvancedPremise, premise_b: AdvancedPremise) -> Tuple[float, str]:
        """Detect causal relationship between two premises."""
        text_a = premise_a.text.lower()
        text_b = premise_b.text.lower()

        # Causal indicators
        causal_indicators = [
            ('because', 'forward'), ('due to', 'backward'), ('results in', 'forward'),
            ('leads to', 'forward'), ('causes', 'forward'), ('therefore', 'forward'),
            ('consequently', 'forward'), ('as a result', 'forward')
        ]

        strength = 0.0
        direction = "forward"

        # Check for explicit causal language
        for indicator, dir_hint in causal_indicators:
            if indicator in text_a and self._texts_related(text_a, text_b):
                strength += 0.4
                direction = dir_hint
            elif indicator in text_b and self._texts_related(text_b, text_a):
                strength += 0.4
                direction = "backward" if dir_hint == "forward" else "forward"

        # Temporal relationships can suggest causality
        temporal_a = premise_a.temporal_context
        temporal_b = premise_b.temporal_context

        if temporal_a and temporal_b:
            if self._is_temporally_before(temporal_a, temporal_b):
                strength += 0.2
                direction = "forward"
            elif self._is_temporally_before(temporal_b, temporal_a):
                strength += 0.2
                direction = "backward"

        # Semantic similarity can suggest relationship
        semantic_sim = self.semantic_analyzer.compute_semantic_similarity(text_a, text_b)
        if semantic_sim > 0.5:
            strength += 0.1

        return min(strength, 1.0), direction

    def _texts_related(self, text1: str, text2: str) -> bool:
        """Check if two texts are semantically related."""
        return self.semantic_analyzer.compute_semantic_similarity(text1, text2) > 0.3

    def _is_temporally_before(self, temporal_a: Dict[str, Any], temporal_b: Dict[str, Any]) -> bool:
        """Check if temporal context A is before B."""
        years_a = temporal_a.get("years", [])
        years_b = temporal_b.get("years", [])

        if years_a and years_b:
            min_year_a = min(int(year) for year in years_a)
            min_year_b = min(int(year) for year in years_b)
            return min_year_a < min_year_b

        return False

    def _identify_causal_mechanism(self, text_a: str, text_b: str) -> Optional[str]:
        """Identify the mechanism of causal relationship."""
        combined_text = (text_a + " " + text_b).lower()

        mechanisms = {
            'policy': ['policy', 'regulation', 'law', 'rule'],
            'economic': ['market', 'economic', 'financial', 'cost', 'price'],
            'technological': ['technology', 'innovation', 'digital', 'automated'],
            'social': ['behavior', 'culture', 'social', 'community'],
            'temporal': ['time', 'schedule', 'deadline', 'timeline']
        }

        for mechanism, keywords in mechanisms.items():
            if any(keyword in combined_text for keyword in keywords):
                return mechanism

        return None

    def _generate_reasoning_trace(self, question: str, premises: List[AdvancedPremise]) -> List[ReasoningTrace]:
        """Generate detailed reasoning trace."""
        trace = []

        # Initial premise evaluation
        trace.append(ReasoningTrace(
            step_id="premise_evaluation",
            operation="evaluate_premises",
            input_premises=[p.evidence_id or f"premise_{i}" for i, p in enumerate(premises)],
            output_conclusion=f"Evaluated {len(premises)} premises for relevance to question",
            confidence_delta=0.0,
            reasoning_mode=ReasoningMode.INDUCTIVE,
            evidence_synthesis={"premise_count": len(premises),
                                "avg_semantic_score": sum(p.semantic_score for p in premises) / max(len(premises), 1)}
        ))

        # Attention weighting if enabled
        if self.attention:
            high_attention_premises = [p for p in premises if p.attention_weight and p.attention_weight.value > 0.6]
            trace.append(ReasoningTrace(
                step_id="attention_weighting",
                operation="apply_attention_mechanism",
                input_premises=[p.evidence_id or f"premise_{premises.index(p)}" for p in high_attention_premises],
                output_conclusion=f"Identified {len(high_attention_premises)} high-attention premises",
                confidence_delta=0.1 if high_attention_premises else -0.1,
                reasoning_mode=ReasoningMode.ABDUCTIVE,
                evidence_synthesis={"high_attention_count": len(high_attention_premises)}
            ))

        # Semantic synthesis
        semantic_scores = [p.semantic_score for p in premises]
        avg_semantic = sum(semantic_scores) / max(len(semantic_scores), 1)

        trace.append(ReasoningTrace(
            step_id="semantic_synthesis",
            operation="synthesize_semantic_evidence",
            input_premises=[p.evidence_id or f"premise_{i}" for i, p in enumerate(premises)],
            output_conclusion=f"Average semantic alignment: {avg_semantic:.3f}",
            confidence_delta=avg_semantic - 0.5,
            reasoning_mode=ReasoningMode.INDUCTIVE,
            evidence_synthesis={"avg_semantic_score": avg_semantic,
                                "score_variance": statistics.variance(semantic_scores) if len(
                                    semantic_scores) > 1 else 0.0}
        ))

        return trace

    def _determine_advanced_verdict(self, question: str, premises: List[AdvancedPremise],
                                    reasoning_trace: List[ReasoningTrace]) -> str:
        """Determine verdict using advanced multi-factor analysis."""
        factors = []

        # Semantic factor
        semantic_scores = [p.semantic_score for p in premises]
        avg_semantic = sum(semantic_scores) / max(len(semantic_scores), 1)
        factors.append(("semantic", avg_semantic, 0.3))

        # Attention factor
        if self.attention and premises:
            attention_scores = [p.attention_weight.value if p.attention_weight else 0.5 for p in premises]
            avg_attention = sum(attention_scores) / len(attention_scores)
            factors.append(("attention", avg_attention, 0.2))

        # Source credibility factor
        credibility_scores = [p.source_credibility for p in premises]
        avg_credibility = sum(credibility_scores) / max(len(credibility_scores), 1)
        factors.append(("credibility", avg_credibility, 0.15))

        # Certainty factor
        certainty_scores = []
        for p in premises:
            cert_score = 0.5
            if p.certainty_markers:
                high_cert = sum(1 for m in p.certainty_markers if m.startswith('high:'))
                low_cert = sum(1 for m in p.certainty_markers if m.startswith('low:'))
                cert_score = 0.8 if high_cert > low_cert else 0.3 if low_cert > high_cert else 0.5
            certainty_scores.append(cert_score)
        avg_certainty = sum(certainty_scores) / max(len(certainty_scores), 1)
        factors.append(("certainty", avg_certainty, 0.1))

        # Completeness factor
        completeness_scores = [p.information_completeness for p in premises]
        avg_completeness = sum(completeness_scores) / max(len(completeness_scores), 1)
        factors.append(("completeness", avg_completeness, 0.1))

        # Temporal relevance factor
        temporal_relevance = 0.5
        question_intent = self._extract_question_intent(question)
        if question_intent == "temporal":
            temporal_premises = [p for p in premises if p.temporal_context]
            temporal_relevance = len(temporal_premises) / max(len(premises), 1)
        factors.append(("temporal", temporal_relevance, 0.15))

        # Weighted combination
        weighted_score = sum(score * weight for name, score, weight in factors)

        # Dynamic thresholds based on question complexity
        complexity = self._assess_question_complexity(question)
        yes_threshold = 0.6 + (complexity * 0.1)  # Higher threshold for complex questions
        no_threshold = 0.4 - (complexity * 0.1)  # Lower threshold for complex questions

        if weighted_score >= yes_threshold:
            return "yes"
        elif weighted_score <= no_threshold:
            return "no"
        else:
            return "unknown"

    def _assess_question_complexity(self, question: str) -> float:
        """Assess the complexity of the question."""
        complexity_score = 0.0

        # Length factor
        word_count = len(re.findall(r'\b\w+\b', question))
        if word_count > 15:
            complexity_score += 0.3
        elif word_count > 10:
            complexity_score += 0.2

        # Multiple clauses
        clause_count = len(re.split(r'[,;]', question))
        if clause_count > 2:
            complexity_score += 0.2

        # Conjunction complexity
        if ' and ' in question.lower() or ' or ' in question.lower():
            complexity_score += 0.2

        # Technical terms (simplified heuristic)
        technical_indicators = ['target', 'baseline', 'metric', 'indicator', 'compliance', 'standard']
        tech_count = sum(1 for term in technical_indicators if term in question.lower())
        complexity_score += min(tech_count * 0.1, 0.3)

        return min(complexity_score, 1.0)

    def _build_advanced_rationale(self, question: str, verdict: str, premises: List[AdvancedPremise],
                                  reasoning_trace: List[ReasoningTrace]) -> str:
        """Build comprehensive rationale with multi-layered explanations."""
        rationale_parts = []

        # Question analysis
        rationale_parts.append(f"QUESTION ANALYSIS: {question}")
        rationale_parts.append(f"Verdict: {verdict.upper()}")
        rationale_parts.append("")

        # Evidence summary
        rationale_parts.append("EVIDENCE SYNTHESIS:")
        if premises:
            # Top premises by combined score
            combined_scores = []
            for i, p in enumerate(premises):
                combined_score = (
                        p.semantic_score * 0.4 +
                        (p.attention_weight.value if p.attention_weight else 0.5) * 0.3 +
                        p.source_credibility * 0.2 +
                        p.information_completeness * 0.1
                )
                combined_scores.append((i, p, combined_score))

            # Sort by combined score
            combined_scores.sort(key=lambda x: x[2], reverse=True)

            for rank, (idx, premise, score) in enumerate(combined_scores[:5], 1):
                rationale_parts.append(f"  {rank}. [{premise.evidence_id or f'P{idx + 1}'}] {premise.text[:150]}...")
                rationale_parts.append(
                    f"     → Combined Score: {score:.3f} (Semantic: {premise.semantic_score:.3f}, Credibility: {premise.source_credibility:.3f})")

                if premise.certainty_markers:
                    rationale_parts.append(f"     → Certainty Markers: {', '.join(premise.certainty_markers[:3])}")

                if premise.temporal_context:
                    rationale_parts.append(f"     → Temporal Context: {premise.temporal_context}")
                rationale_parts.append("")

        # Reasoning process
        rationale_parts.append("REASONING PROCESS:")
        for trace in reasoning_trace:
            rationale_parts.append(f"  • {trace.operation}: {trace.output_conclusion}")
            if trace.confidence_delta != 0:
                rationale_parts.append(f"    Confidence Impact: {trace.confidence_delta:+.3f}")
        rationale_parts.append("")

        # Factor analysis
        rationale_parts.append("DECISION FACTORS:")
        semantic_avg = sum(p.semantic_score for p in premises) / max(len(premises), 1)
        rationale_parts.append(f"  • Semantic Alignment: {semantic_avg:.3f}")

        if self.attention and premises:
            attention_avg = sum(p.attention_weight.value if p.attention_weight else 0.5 for p in premises) / len(
                premises)
            rationale_parts.append(f"  • Attention-Weighted Relevance: {attention_avg:.3f}")

        credibility_avg = sum(p.source_credibility for p in premises) / max(len(premises), 1)
        rationale_parts.append(f"  • Source Credibility: {credibility_avg:.3f}")

        return "\n".join(rationale_parts)

    def _apply_advanced_standards(self, answer: AdvancedSynthesizedAnswer,
                                  standards: Dict[str, Any]) -> AdvancedSynthesizedAnswer:
        """Apply advanced standards checking with detailed analysis."""
        unmet_requirements = []
        compliance_factors = []

        # Gather all text for analysis
        all_text = (answer.rationale + "\n" + "\n".join(p.text for p in answer.premises)).lower()

        # Mandatory indicators check
        mandatory_indicators = standards.get("mandatory_indicators", [])
        for indicator in mandatory_indicators:
            if str(indicator).lower() not in all_text:
                unmet_requirements.append({
                    "type": "mandatory_indicator",
                    "requirement": str(indicator),
                    "severity": "high",
                    "description": f"Required indicator '{indicator}' not found in evidence"
                })
                compliance_factors.append(0.0)
            else:
                compliance_factors.append(1.0)

        # Rules evaluation
        rules = standards.get("rules", [])
        for rule in rules:
            rule_satisfied = self._evaluate_advanced_rule(rule, all_text, answer.premises)
            if not rule_satisfied:
                unmet_requirements.append({
                    "type": "rule_violation",
                    "rule_id": rule.get("id"),
                    "requirement": rule.get("description"),
                    "severity": rule.get("severity", "medium"),
                    "description": f"Rule '{rule.get('id')}' not satisfied"
                })
                compliance_factors.append(0.0)
            else:
                compliance_factors.append(1.0)

        # Quality thresholds
        quality_standards = standards.get("quality_thresholds", {})
        min_confidence = quality_standards.get("min_confidence", 0.0)
        min_evidence_count = quality_standards.get("min_evidence_count", 0)

        if answer.confidence < min_confidence:
            unmet_requirements.append({
                "type": "quality_threshold",
                "requirement": f"Minimum confidence {min_confidence}",
                "severity": "medium",
                "description": f"Confidence {answer.confidence:.3f} below required {min_confidence}"
            })
            compliance_factors.append(0.0)

        if len(answer.premises) < min_evidence_count:
            unmet_requirements.append({
                "type": "quality_threshold",
                "requirement": f"Minimum evidence count {min_evidence_count}",
                "severity": "medium",
                "description": f"Evidence count {len(answer.premises)} below required {min_evidence_count}"
            })
            compliance_factors.append(0.0)

        # Calculate compliance score
        compliance_score = sum(compliance_factors) / max(len(compliance_factors), 1)

        # Update answer
        answer.unmet_requirements = unmet_requirements
        answer.compliance_score = compliance_score

        # Verdict adjustment based on compliance
        if unmet_requirements and answer.verdict == "yes":
            high_severity_unmet = [r for r in unmet_requirements if r.get("severity") == "high"]
            if high_severity_unmet:
                answer.verdict = "unknown"
                answer.rationale += f"\n\nNOTE: Verdict adjusted to 'unknown' due to {len(high_severity_unmet)} high-severity compliance issues."

        answer.add_audit_entry("standards_compliance", {
            "unmet_count": len(unmet_requirements),
            "compliance_score": compliance_score,
            "high_severity_issues": len([r for r in unmet_requirements if r.get("severity") == "high"])
        })

        return answer

    def _evaluate_advanced_rule(self, rule: Dict[str, Any], text: str, premises: List[AdvancedPremise]) -> bool:
        """Evaluate a single rule with advanced logic."""
        # Basic pattern matching
        pattern = rule.get("pattern")
        if pattern and pattern.lower() not in text:
            return False

        # Any-of logic
        any_of = rule.get("any_of", [])
        if any_of:
            if not any(str(item).lower() in text for item in any_of):
                return False

        # All-of logic
        all_of = rule.get("all_of", [])
        if all_of:
            if not all(str(item).lower() in text for item in all_of):
                return False

        # Context-aware rules
        context_rule = rule.get("context_requirements")
        if context_rule:
            return self._evaluate_context_rule(context_rule, premises)

        # Temporal rules
        temporal_rule = rule.get("temporal_requirements")
        if temporal_rule:
            return self._evaluate_temporal_rule(temporal_rule, premises)

        # Quantitative rules
        quantitative_rule = rule.get("quantitative_requirements")
        if quantitative_rule:
            return self._evaluate_quantitative_rule(quantitative_rule, text, premises)

        return True

    def _evaluate_context_rule(self, context_rule: Dict[str, Any], premises: List[AdvancedPremise]) -> bool:
        """Evaluate context-aware rules."""
        required_context = context_rule.get("required_context_type")
        min_premises = context_rule.get("min_premises_with_context", 1)

        matching_premises = 0
        for premise in premises:
            if required_context == "temporal" and premise.temporal_context:
                matching_premises += 1
            elif required_context == "numerical" and re.search(r'\d', premise.text):
                matching_premises += 1
            elif required_context == "high_credibility" and premise.source_credibility > 0.7:
                matching_premises += 1

        return matching_premises >= min_premises

    def _evaluate_temporal_rule(self, temporal_rule: Dict[str, Any], premises: List[AdvancedPremise]) -> bool:
        """Evaluate temporal-specific rules."""
        required_year = temporal_rule.get("required_year")
        min_temporal_premises = temporal_rule.get("min_temporal_premises", 1)

        temporal_premises = [p for p in premises if p.temporal_context]

        if len(temporal_premises) < min_temporal_premises:
            return False

        if required_year:
            year_found = any(
                required_year in p.temporal_context.get("years", [])
                for p in temporal_premises
                if p.temporal_context
            )
            return year_found

        return True

    def _evaluate_quantitative_rule(self, quant_rule: Dict[str, Any], text: str,
                                    premises: List[AdvancedPremise]) -> bool:
        """Evaluate quantitative requirements."""
        requires_percentages = quant_rule.get("requires_percentages", False)
        min_numeric_premises = quant_rule.get("min_numeric_premises", 0)

        if requires_percentages:
            if not re.search(r'\d+%|\d+\s*percent', text):
                return False

        if min_numeric_premises > 0:
            numeric_premises = sum(1 for p in premises if re.search(r'\d', p.text))
            if numeric_premises < min_numeric_premises:
                return False

        return True

    def _estimate_advanced_confidence(self, answer: AdvancedSynthesizedAnswer) -> Dict[str, Any]:
        """Estimate confidence using the configured method."""
        evidence_scores = []

        for premise in answer.premises:
            # Combine multiple score dimensions
            combined_score = (
                    premise.semantic_score * 0.3 +
                    premise.syntactic_score * 0.2 +
                    premise.pragmatic_score * 0.2 +
                    (premise.attention_weight.value if premise.attention_weight else 0.5) * 0.15 +
                    premise.source_credibility * 0.15
            )
            evidence_scores.append(combined_score)

        if self.confidence_method == ConfidenceMethod.BAYESIAN:
            confidence, interval = self.bayesian_estimator.estimate_confidence(evidence_scores)
            uncertainty_sources = self.bayesian_estimator.get_uncertainty_sources(evidence_scores)

            return {
                "point_estimate": confidence,
                "interval": interval,
                "distribution": {"type": "beta", "alpha": self.bayesian_estimator.prior_alpha,
                                 "beta": self.bayesian_estimator.prior_beta},
                "uncertainty_sources": uncertainty_sources,
                "method": "bayesian"
            }

        elif self.confidence_method == ConfidenceMethod.ENSEMBLE:
            # Multiple estimation methods
            methods = []

            # Simple average
            simple_avg = sum(evidence_scores) / max(len(evidence_scores), 1)
            methods.append(("simple_average", simple_avg))

            # Weighted by credibility
            if answer.premises:
                weights = [p.source_credibility for p in answer.premises]
                weighted_avg = sum(s * w for s, w in zip(evidence_scores, weights)) / max(sum(weights), 1e-10)
                methods.append(("credibility_weighted", weighted_avg))

            # Attention-weighted
            if self.attention and answer.premises:
                attention_weights = [p.attention_weight.value if p.attention_weight else 0.5 for p in answer.premises]
                attention_avg = sum(s * w for s, w in zip(evidence_scores, attention_weights)) / max(
                    sum(attention_weights), 1e-10)
                methods.append(("attention_weighted", attention_avg))

            # Conservative (min)
            conservative = min(evidence_scores) if evidence_scores else 0.0
            methods.append(("conservative", conservative))

            # Ensemble average
            ensemble_confidence = sum(conf for _, conf in methods) / len(methods)
            ensemble_variance = sum((conf - ensemble_confidence) ** 2 for _, conf in methods) / len(methods)

            # Confidence interval based on ensemble variance
            std_dev = math.sqrt(ensemble_variance)
            lower = max(0.0, ensemble_confidence - 1.96 * std_dev)
            upper = min(1.0, ensemble_confidence + 1.96 * std_dev)

            uncertainty_sources = []
            if ensemble_variance > 0.05:
                uncertainty_sources.append("High variance between estimation methods")
            if len(evidence_scores) < 3:
                uncertainty_sources.append("Limited evidence quantity")

            return {
                "point_estimate": ensemble_confidence,
                "interval": (lower, upper),
                "distribution": {
                    "type": "ensemble",
                    "methods": {name: conf for name, conf in methods},
                    "variance": ensemble_variance
                },
                "uncertainty_sources": uncertainty_sources,
                "method": "ensemble"
            }

        else:  # ADAPTIVE or fallback
            # Adaptive method that selects best approach based on data characteristics
            data_variance = statistics.variance(evidence_scores) if len(evidence_scores) > 1 else 0.0
            evidence_count = len(evidence_scores)

            if evidence_count >= 5 and data_variance < 0.1:
                # High confidence in simple average for stable, sufficient data
                confidence = sum(evidence_scores) / len(evidence_scores)
                uncertainty_sources = ["Stable evidence pattern"]
            elif evidence_count >= 3:
                # Use weighted approach for moderate data
                weights = [p.source_credibility * (p.attention_weight.value if p.attention_weight else 0.5)
                           for p in answer.premises]
                total_weight = sum(weights) or 1.0
                confidence = sum(s * w for s, w in zip(evidence_scores, weights)) / total_weight
                uncertainty_sources = []
                if data_variance > 0.2:
                    uncertainty_sources.append("High evidence variance")
            else:
                # Conservative approach for limited data
                confidence = min(evidence_scores) if evidence_scores else 0.0
                uncertainty_sources = ["Limited evidence quantity", "Conservative estimation"]

            # Dynamic confidence interval
            base_uncertainty = 0.1 + (data_variance * 0.5) + (1.0 / max(evidence_count, 1)) * 0.2
            lower = max(0.0, confidence - base_uncertainty)
            upper = min(1.0, confidence + base_uncertainty)

            return {
                "point_estimate": confidence,
                "interval": (lower, upper),
                "distribution": {
                    "type": "adaptive",
                    "data_variance": data_variance,
                    "evidence_count": evidence_count,
                    "base_uncertainty": base_uncertainty
                },
                "uncertainty_sources": uncertainty_sources,
                "method": "adaptive"
            }

    def _assess_synthesis_quality(self, answer: AdvancedSynthesizedAnswer) -> float:
        """Assess the overall quality of the synthesis."""
        quality_factors = []

        # Evidence diversity
        if answer.premises:
            evidence_types = set(p.evidence_type for p in answer.premises)
            type_diversity = len(evidence_types) / len(EvidenceType)
            quality_factors.append(("evidence_diversity", type_diversity, 0.15))

        # Source credibility
        if answer.premises:
            avg_credibility = sum(p.source_credibility for p in answer.premises) / len(answer.premises)
            quality_factors.append(("source_credibility", avg_credibility, 0.2))

        # Reasoning trace completeness
        trace_completeness = min(len(answer.reasoning_trace) / 3.0, 1.0)
        quality_factors.append(("reasoning_completeness", trace_completeness, 0.15))

        # Causal analysis depth
        causal_depth = min(len(answer.causal_graph) / max(len(answer.premises), 1), 1.0)
        quality_factors.append(("causal_analysis", causal_depth, 0.1))

        # Confidence calibration
        confidence_calibration = 1.0 - abs(answer.confidence - 0.7)  # Prefer moderate confidence
        quality_factors.append(("confidence_calibration", confidence_calibration, 0.1))

        # Rationale comprehensiveness
        rationale_length = len(answer.rationale.split())
        rationale_score = min(rationale_length / 200.0, 1.0)  # Prefer detailed rationales
        quality_factors.append(("rationale_comprehensiveness", rationale_score, 0.15))

        # Standards compliance
        compliance_score = answer.compliance_score
        quality_factors.append(("standards_compliance", compliance_score, 0.15))

        # Weighted combination
        total_quality = sum(score * weight for _, score, weight in quality_factors)
        return min(total_quality, 1.0)

    def _assess_internal_consistency(self, answer: AdvancedSynthesizedAnswer) -> float:
        """Assess internal consistency of the answer."""
        consistency_checks = []

        # Verdict-confidence consistency
        verdict_confidence_map = {"yes": 0.7, "no": 0.3, "unknown": 0.5}
        expected_confidence = verdict_confidence_map.get(answer.verdict, 0.5)
        confidence_consistency = 1.0 - abs(answer.confidence - expected_confidence) / 0.5
        consistency_checks.append(confidence_consistency)

        # Premise-verdict consistency
        if answer.premises:
            premise_scores = [p.semantic_score for p in answer.premises]
            avg_premise_score = sum(premise_scores) / len(premise_scores)

            if answer.verdict == "yes" and avg_premise_score < 0.4:
                consistency_checks.append(0.3)  # Low consistency
            elif answer.verdict == "no" and avg_premise_score > 0.6:
                consistency_checks.append(0.3)  # Low consistency
            else:
                consistency_checks.append(0.9)  # High consistency

        # Reasoning trace consistency
        if answer.reasoning_trace:
            confidence_deltas = [t.confidence_delta for t in answer.reasoning_trace]
            if confidence_deltas:
                total_delta = sum(confidence_deltas)
                # Check if reasoning trace changes align with final confidence
                if abs(total_delta) > 0.5:  # Large confidence changes
                    if answer.confidence < 0.5 and total_delta > 0:
                        consistency_checks.append(0.4)  # Inconsistent
                    elif answer.confidence > 0.5 and total_delta < 0:
                        consistency_checks.append(0.4)  # Inconsistent
                    else:
                        consistency_checks.append(0.8)
                else:
                    consistency_checks.append(0.9)  # Stable reasoning

        return sum(consistency_checks) / max(len(consistency_checks), 1)

    def _assess_completeness(self, answer: AdvancedSynthesizedAnswer, question: str) -> float:
        """Assess completeness of the answer relative to the question."""
        completeness_factors = []

        # Question type coverage
        question_intent = self._extract_question_intent(question)

        if question_intent == "temporal":
            temporal_coverage = sum(1 for p in answer.premises if p.temporal_context) / max(len(answer.premises), 1)
            completeness_factors.append(temporal_coverage)
        elif question_intent == "quantitative":
            numeric_coverage = sum(1 for p in answer.premises if re.search(r'\d', p.text)) / max(len(answer.premises),
                                                                                                 1)
            completeness_factors.append(numeric_coverage)
        elif question_intent == "factual":
            credible_coverage = sum(1 for p in answer.premises if p.source_credibility > 0.6) / max(
                len(answer.premises), 1)
            completeness_factors.append(credible_coverage)

        # Evidence sufficiency
        evidence_sufficiency = min(len(answer.premises) / 3.0, 1.0)  # Prefer at least 3 pieces of evidence
        completeness_factors.append(evidence_sufficiency)

        # Citation coverage
        citation_coverage = sum(1 for p in answer.premises if p.citation) / max(len(answer.premises), 1)
        completeness_factors.append(citation_coverage)

        # Sub-question coverage (if hierarchical)
        if answer.sub_questions:
            sub_coverage = min(len(answer.sub_questions) / 2.0, 1.0)
            completeness_factors.append(sub_coverage)
        else:
            completeness_factors.append(0.5)  # Neutral if no hierarchical analysis

        return sum(completeness_factors) / len(completeness_factors)

    def _update_performance_metrics(self, answer: AdvancedSynthesizedAnswer) -> None:
        """Update performance tracking metrics."""
        self.performance_metrics["synthesis_times"].append(
            answer.processing_metadata.get("synthesis_duration", 0.0)
        )
        self.performance_metrics["confidence_scores"].append(answer.confidence)
        self.performance_metrics["quality_scores"].append(answer.synthesis_quality_score)
        self.performance_metrics["premise_counts"].append(len(answer.premises))
        self.performance_metrics["compliance_scores"].append(answer.compliance_score)

        # Keep only recent metrics (last 100)
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 100:
                self.performance_metrics[key] = self.performance_metrics[key][-100:]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        if not self.performance_metrics["synthesis_times"]:
            return {"status": "no_data", "message": "No synthesis operations recorded"}

        report = {
            "synthesis_statistics": {
                "total_syntheses": len(self.synthesis_history),
                "avg_synthesis_time": statistics.mean(self.performance_metrics["synthesis_times"]),
                "avg_confidence": statistics.mean(self.performance_metrics["confidence_scores"]),
                "avg_quality": statistics.mean(self.performance_metrics["quality_scores"]),
                "avg_premise_count": statistics.mean(self.performance_metrics["premise_counts"]),
                "avg_compliance": statistics.mean(self.performance_metrics["compliance_scores"])
            },
            "quality_trends": {
                "confidence_trend": self._calculate_trend(self.performance_metrics["confidence_scores"]),
                "quality_trend": self._calculate_trend(self.performance_metrics["quality_scores"]),
                "performance_stability": statistics.stdev(self.performance_metrics["synthesis_times"])
            },
            "configuration": {
                "confidence_method": self.confidence_method.value,
                "reasoning_mode": self.reasoning_mode.value,
                "hierarchical_enabled": self.enable_hierarchical,
                "attention_enabled": self.attention_mechanism
            }
        }

        return report

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend
        n = len(values)
        x_vals = list(range(n))

        # Calculate slope
        x_mean = sum(x_vals) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

    def calibrate_confidence(self, validation_pairs: List[Tuple[str, List[Any], bool]]) -> Dict[str, Any]:
        """Calibrate confidence estimation using ground truth data."""
        calibration_results = []

        for question, evidence, ground_truth in validation_pairs:
            answer = self.synthesize_answer(question, evidence)
            predicted_confidence = answer.confidence
            actual_correctness = 1.0 if (
                    (ground_truth and answer.verdict == "yes") or
                    (not ground_truth and answer.verdict == "no")
            ) else 0.0

            calibration_results.append({
                "predicted_confidence": predicted_confidence,
                "actual_correctness": actual_correctness,
                "absolute_error": abs(predicted_confidence - actual_correctness)
            })

        # Calculate calibration metrics
        if calibration_results:
            avg_confidence = statistics.mean(r["predicted_confidence"] for r in calibration_results)
            avg_accuracy = statistics.mean(r["actual_correctness"] for r in calibration_results)
            avg_abs_error = statistics.mean(r["absolute_error"] for r in calibration_results)

            # Update Bayesian estimator
            for result in calibration_results:
                self.bayesian_estimator.update_posterior(
                    [result["predicted_confidence"]],
                    result["actual_correctness"] > 0.5
                )

            return {
                "calibration_score": 1.0 - avg_abs_error,
                "average_confidence": avg_confidence,
                "average_accuracy": avg_accuracy,
                "sample_count": len(calibration_results),
                "calibration_status": "calibrated"
            }

        return {"calibration_status": "no_data"}

    def export_synthesis_pipeline(self) -> Dict[str, Any]:
        """Export the complete synthesis pipeline configuration and state."""
        return {
            "version": "1.0.0",
            "configuration": {
                "confidence_method": self.confidence_method.value,
                "reasoning_mode": self.reasoning_mode.value,
                "hierarchical_reasoning": self.enable_hierarchical,
                "attention_mechanism": self.attention_mechanism,
                "random_seed": self.random_seed
            },
            "calibration_state": {
                "bayesian_alpha": self.bayesian_estimator.prior_alpha,
                "bayesian_beta": self.bayesian_estimator.prior_beta,
                "evidence_history_count": len(self.bayesian_estimator.evidence_history)
            },
            "performance_metrics": dict(self.performance_metrics),
            "synthesis_history_count": len(self.synthesis_history),
            "export_timestamp": time.time()
        }


# ----------------------------------------------------------------------
# Convenience Functions and Factory Methods
# ----------------------------------------------------------------------

def create_advanced_synthesizer(
        confidence_method: str = "adaptive",
        reasoning_mode: str = "abductive",
        enable_hierarchical: bool = True,
        attention_mechanism: bool = True
) -> AdvancedAnswerSynthesizer:
    """Factory function to create configured synthesizer."""
    return AdvancedAnswerSynthesizer(
        confidence_method=ConfidenceMethod(confidence_method),
        reasoning_mode=ReasoningMode(reasoning_mode),
        enable_hierarchical=enable_hierarchical,
        attention_mechanism=attention_mechanism
    )


def synthesize_advanced_answer(
        question: str,
        evidence: List[Any],
        standards: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        **synthesizer_kwargs
) -> AdvancedSynthesizedAnswer:
    """Convenience function for one-shot synthesis with advanced features."""
    synthesizer = create_advanced_synthesizer(**synthesizer_kwargs)
    return synthesizer.synthesize_answer(question, evidence, standards, context)


def format_advanced_response(answer: AdvancedSynthesizedAnswer) -> Dict[str, Any]:
    """Format advanced answer for JSON serialization."""
    return {
        "question": answer.question,
        "verdict": answer.verdict,
        "confidence": answer.confidence,
        "confidence_method": answer.confidence_method.value,
        "confidence_interval": answer.confidence_interval,
        "confidence_distribution": answer.confidence_distribution,
        "rationale": answer.rationale,
        "premises": [
            {
                "text": p.text,
                "evidence_id": p.evidence_id,
                "evidence_type": p.evidence_type.value,
                "semantic_score": p.semantic_score,
                "attention_weight": p.attention_weight.value if p.attention_weight else None,
                "source_credibility": p.source_credibility,
                "certainty_markers": p.certainty_markers,
                "temporal_context": p.temporal_context,
                "citation": p.citation
            }
            for p in answer.premises
        ],
        "sub_questions": [
            {
                "question": sq.question,
                "verdict": sq.verdict,
                "confidence": sq.confidence
            }
            for sq in answer.sub_questions
        ],
        "reasoning_trace": [
            {
                "step_id": trace.step_id,
                "operation": trace.operation,
                "conclusion": trace.output_conclusion,
                "confidence_delta": trace.confidence_delta,
                "reasoning_mode": trace.reasoning_mode.value
            }
            for trace in answer.reasoning_trace
        ],
        "causal_graph": [
            {
                "source": link.source_id,
                "target": link.target_id,
                "strength": link.strength,
                "direction": link.direction,
                "mechanism": link.mechanism
            }
            for link in answer.causal_graph
        ],
        "quality_assessment": {
            "synthesis_quality_score": answer.synthesis_quality_score,
            "internal_consistency": answer.internal_consistency,
            "completeness_score": answer.completeness_score
        },
        "compliance": {
            "compliance_score": answer.compliance_score,
            "unmet_requirements": answer.unmet_requirements
        },
        "metadata": {
            "synthesis_timestamp": answer.synthesis_timestamp,
            "processing_metadata": answer.processing_metadata,
            "uncertainty_sources": answer.uncertainty_sources,
            "integrity_hash": answer._integrity_hash,
            "integrity_verified": answer.verify_integrity()
        }
    }


# ----------------------------------------------------------------------
# Advanced Demo and Testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Comprehensive demonstration
    demo_evidence = [
        {
            "text": "The 2025 strategic plan establishes a target of 25% improvement in coverage metrics by Q4 2025, with baseline measurements taken in 2023.",
            "evidence_id": "strategic_plan_2025",
            "type": "structured",
            "citation": {
                "document_id": "SP-2025-001",
                "title": "Strategic Planning Document 2025",
                "author": "Strategy Team",
                "publication_date": "2024-01-15",
                "source_type": "official",
                "page_number": 12
            }
        },
        {
            "text": "Historical analysis shows that similar coverage initiatives typically achieve 15-30% improvements when properly implemented with adequate baseline data.",
            "evidence_id": "historical_analysis",
            "type": "textual",
            "citation": {
                "document_id": "HA-2024-003",
                "title": "Historical Performance Analysis",
                "author": "Analytics Department",
                "source_type": "academic",
                "page_number": 45
            }
        },
        {
            "text": "The baseline measurement framework was established in Q2 2023, providing comprehensive metrics across all operational domains.",
            "evidence_id": "baseline_framework",
            "type": "temporal",
            "citation": {
                "document_id": "BF-2023-012",
                "title": "Baseline Framework Implementation",
                "source_type": "government"
            }
        },
        {
            "text": "Quarterly progress reports indicate consistent advancement toward the 25% target, with current achievement at 18% as of Q2 2024.",
            "evidence_id": "progress_report",
            "type": "numerical",
            "citation": {
                "document_id": "PR-Q2-2024",
                "title": "Q2 2024 Progress Report",
                "source_type": "official"
            }
        }
    ]

    # Advanced standards
    comprehensive_standards = {
        "mandatory_indicators": ["2025", "25%", "baseline", "target"],
        "rules": [
            {
                "id": "temporal_requirement",
                "description": "Must reference 2025 timeline",
                "pattern": "2025",
                "severity": "high"
            },
            {
                "id": "quantitative_requirement",
                "description": "Must include percentage targets",
                "any_of": ["%", "percent"],
                "quantitative_requirements": {
                    "requires_percentages": True,
                    "min_numeric_premises": 2
                },
                "severity": "high"
            },
            {
                "id": "baseline_requirement",
                "description": "Must reference baseline data",
                "all_of": ["baseline"],
                "context_requirements": {
                    "required_context_type": "temporal",
                    "min_premises_with_context": 1
                },
                "severity": "medium"
            }
        ],
        "quality_thresholds": {
            "min_confidence": 0.6,
            "min_evidence_count": 3
        }
    }

    # Create advanced synthesizer with full capabilities
    synthesizer = create_advanced_synthesizer(
        confidence_method="ensemble",
        reasoning_mode="causal",
        enable_hierarchical=True,
        attention_mechanism=True
    )

    # Advanced synthesis
    question = "Does the plan include measurable targets for 2025 with established baselines?"
    context = {
        "domain": "strategic_planning",
        "urgency": "high",
        "stakeholders": ["executive_team", "strategy_team"]
    }

    print("🔬 ADVANCED ANSWER SYNTHESIZER DEMONSTRATION")
    print("=" * 60)

    result = synthesizer.synthesize_answer(
        question=question,
        evidence=demo_evidence,
        standards=comprehensive_standards,
        context=context
    )

    # Format and display results
    formatted_result = format_advanced_response(result)

    print(f"Question: {result.question}")
    print(f"Verdict: {result.verdict.upper()}")
    print(f"Confidence: {result.confidence:.3f} ({result.confidence_method.value})")
    print(f"Quality Score: {result.synthesis_quality_score:.3f}")
    print(f"Compliance Score: {result.compliance_score:.3f}")
    print("\nRationale:")
    print("-" * 40)
    print(result.rationale[:500] + "..." if len(result.rationale) > 500 else result.rationale)

    if result.sub_questions:
        print(f"\nHierarchical Analysis ({len(result.sub_questions)} sub-questions):")
        for i, sq in enumerate(result.sub_questions, 1):
            print(f"  {i}. {sq.question} → {sq.verdict.upper()} (conf: {sq.confidence:.3f})")

    if result.unmet_requirements:
        print(f"\nCompliance Issues ({len(result.unmet_requirements)}):")
        for req in result.unmet_requirements:
            print(f"  • [{req['severity'].upper()}] {req['description']}")

    print("\nPerformance Report:")
    print("-" * 40)
    perf_report = synthesizer.get_performance_report()
    print(json.dumps(perf_report, indent=2))

    print(f"\nIntegrity Verification: {'✓ PASSED' if result.verify_integrity() else '✗ FAILED'}")
    print(f"Processing Time: {result.processing_metadata.get('synthesis_duration', 0):.3f}s")
    print(f"Synthesis Hash: {result._integrity_hash}")

    print("\n" + "=" * 60)
    print("✨ Advanced synthesis complete with full audit trail and validation!")