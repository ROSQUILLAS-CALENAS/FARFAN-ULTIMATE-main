"""
Core modules for EGW Query Expansion system.

This package contains the core processing and orchestration components
for the Entropic Gromov-Wasserstein query expansion system.
"""

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