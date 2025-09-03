"""
Core modules for EGW Query Expansion system.

This package contains the core processing and orchestration components
for the Entropic Gromov-Wasserstein query expansion system.
"""

try:
    from .safety_controller import (
        MathEnhancementSafetyController,
        EnhancementStatus,
        StabilityLevel,
        EnhancementConfig,
        StabilityGuard,
        SafeComputationContext,
        NumericalInstabilityError,
        IterationLimitExceededError,
        ConvergenceError,
        safe_enhancement,
    )
except ImportError:
    pass

try:
    from .confluent_orchestrator import ConfluentOrchestrator, NodeType, TaskNode
except ImportError:
    pass

try:
    from .conformal_risk_control import (
        CertificationResult,
        ClassificationScore,
        ConformalRiskController,
        QuantileRegressionScore,
        RegressionScore,
        RiskControlConfig,
        create_conformal_system,
    )
except ImportError:
    pass

try:
    from .context_adapter import ContextAdapter
except ImportError:
    pass

try:
    from .deterministic_router import (
        ActionType,
        DeterministicRouter,
        ImmutableConfig,
        RoutingContext,
        RoutingDecision,
        create_deterministic_router,
    )
except ImportError:
    pass

try:
    from .gw_alignment import GromovWassersteinAligner
except ImportError:
    pass

try:
    from .hybrid_retrieval import HybridRetriever
except ImportError:
    pass

try:
    from .pattern_matcher import PatternMatcher
except ImportError:
    pass

try:
    from .query_generator import QueryGenerator
except ImportError:
    pass

try:
    from .import_safety import (
        ImportSafety,
        ImportResult,
        safe_import,
        get_import_report,
        register_fallback,
        register_mock,
        import_with_fallback,
        safe_import_numpy,
        safe_import_scipy,
        safe_import_torch,
        safe_import_sklearn,
        safe_import_faiss,
        safe_import_transformers,
        safe_import_sentence_transformers,
        safe_import_pot,
        check_dependencies,
        log_import_summary,
    )
except ImportError:
    pass

# Import hash policies
try:
    from .hash_policies import (
        HashPolicy,
        CanonicalHashPolicy,
        FastHashPolicy,
        SecureHashPolicy,
        ContextHasher,
        SynthesisHasher,
        PipelineHashValidator,
        DEFAULT_HASH_POLICY,
        DEFAULT_CONTEXT_HASHER,
        DEFAULT_SYNTHESIS_HASHER,
        hash_object,
        create_pipeline_validator,
    )
except ImportError:
    pass

# Import immutable context
try:
    from .immutable_context import (
        QuestionContext,
        ImmutableDict,
        DerivationDAG,
        create_question_context,
        create_expanded_context,
        is_valid_context,
        assert_linear_reference,
    )
except ImportError:
    pass