"""
Core modules for EGW Query Expansion system.

This package contains the core processing and orchestration components
for the Entropic Gromov-Wasserstein query expansion system.
"""

# # # from .safety_controller import (  # Module not found  # Module not found  # Module not found
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

try:
# # #     from .confluent_orchestrator import ConfluentOrchestrator, NodeType, TaskNode  # Module not found  # Module not found  # Module not found
except ImportError:
    pass

try:
# # #     from .conformal_risk_control import (  # Module not found  # Module not found  # Module not found
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
# # #     from .context_adapter import ContextAdapter  # Module not found  # Module not found  # Module not found
except ImportError:
    pass

try:
# # #     from .deterministic_router import (  # Module not found  # Module not found  # Module not found
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
# # #     from .gw_alignment import GromovWassersteinAligner  # Module not found  # Module not found  # Module not found
except ImportError:
    pass

try:
# # #     from .hybrid_retrieval import HybridRetriever  # Module not found  # Module not found  # Module not found
except ImportError:
    pass

try:
# # #     from .pattern_matcher import PatternMatcher  # Module not found  # Module not found  # Module not found
except ImportError:
    pass

try:
# # #     from .query_generator import QueryGenerator  # Module not found  # Module not found  # Module not found
except ImportError:
    pass

try:
# # #     from .import_safety import (  # Module not found  # Module not found  # Module not found
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