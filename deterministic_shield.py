"""
Deterministic Guard/Shield System for Safe Synthesis

Implements a deterministic, declarative guard system inspired by safety-shield synthesis
(Bloem et al., 2024, "Shield Synthesis for Safe Reinforcement Learning", FMSD).

Key properties implemented here:
- Enumerated preconditions ("compuertas") with identifiable clause IDs and descriptions.
- Deterministic evaluation of guards in canonical order, producing a full evaluation trace.
- Typed, canonical rejection responses with minimal-fact justifications.
- Identical violations yield identical rejection artifacts via canonical serialization + fingerprinting.
- GuardedSynthesizer wrapper enforces that no synthesis occurs beyond failed guards.

This module is self-contained and does not alter existing pipelines. It can be
adopted by wrapping any synthesizer function/callable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

# ----------------------------- Canonicalization ----------------------------- #


def _to_canonical(obj: Any) -> Any:
    """Recursively convert data into canonical, JSON-serializable structures.
    - Dicts: keys sorted; values canonicalized.
    - Sets/Tuples: converted to sorted lists by canonical string form.
    - Lists: values canonicalized.
    - Primitives left unchanged.
    """
    if isinstance(obj, dict):
        return {k: _to_canonical(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (set, tuple, list)):
        canon_list = [_to_canonical(x) for x in obj]
        # Convert to list and sort by deterministic string representation
        return sorted(
            canon_list,
            key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")),
        )
    return obj


def canonical_json(data: Mapping[str, Any]) -> str:
    """Return a canonical JSON string of the given mapping.
    Ensures identical data -> identical string (and thus identical fingerprints).
    """
    canon = _to_canonical(dict(data))
    return json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(data: Mapping[str, Any]) -> str:
    """Compute a stable SHA-256 fingerprint over canonical JSON of the data."""
    cj = canonical_json(data)
    return hashlib.sha256(cj.encode("utf-8")).hexdigest()


# --------------------------------- Models ---------------------------------- #


class GuardSeverity(Enum):
    SAFETY = "safety"
    RUNTIME = "runtime"
    POLICY = "policy"


Predicate = Callable[[Any], "GuardResult"]


@dataclass(frozen=True)
class GuardClause:
    """Declarative precondition (compuerta) definition.
    - clause_id: stable identifier used for determinism and traceability.
    - description: human-readable explanation of the clause.
    - predicate: function that inspects context and returns GuardResult.
    - severity: used to classify type of constraint (default: SAFETY).
    """

    clause_id: str
    description: str
    predicate: Predicate
    severity: GuardSeverity = GuardSeverity.SAFETY


@dataclass
class GuardResult:
    """Evaluation result of a single guard clause.
    - clause_id: the evaluated clause.
    - satisfied: True if guard is met; False otherwise.
    - minimal_facts: minimal sufficient facts used to justify the decision.
    - reason: short, identifiable reason string (stable wording recommended).
    """

    clause_id: str
    satisfied: bool
    minimal_facts: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "satisfied": bool(self.satisfied),
            "minimal_facts": _to_canonical(self.minimal_facts),
            "reason": self.reason or "",
        }


@dataclass
class PreconditionCatalog:
    """Catalog of guard clauses with deterministic iteration order.
    Clauses are iterated in canonical order by clause_id.
    """

    clauses: Dict[str, GuardClause] = field(default_factory=dict)

    def add(self, clause: GuardClause) -> None:
        if clause.clause_id in self.clauses:
            raise ValueError(f"Duplicate clause_id: {clause.clause_id}")
        self.clauses[clause.clause_id] = clause

    def add_many(self, clauses: Iterable[GuardClause]) -> None:
        for c in clauses:
            self.add(c)

    def get(self, clause_id: str) -> GuardClause:
        return self.clauses[clause_id]

    def ordered(self) -> List[GuardClause]:
        return [self.clauses[k] for k in sorted(self.clauses.keys())]


@dataclass
class RejectionResponse:
    """Typed rejection artifact emitted by the shield.
    Designed to be identical for identical violations (no dynamic fields).
    """

    mechanism: str
    violation_ids: List[str]
    reasons: Dict[str, str]
    minimal_facts: Dict[str, Any]
    trace: List[Dict[str, Any]]
    canonical_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        # Provide a deterministic dict representation (no timestamps)
        payload = {
            "mechanism": self.mechanism,
            "violation_ids": list(sorted(self.violation_ids)),
            "reasons": _to_canonical(self.reasons),
            "minimal_facts": _to_canonical(self.minimal_facts),
            "trace": [_to_canonical(t) for t in self.trace],
        }
        payload["canonical_fingerprint"] = fingerprint(payload)
        return payload


# ------------------------------ Evaluation --------------------------------- #


@dataclass
class EvaluationReport:
    """Enhanced evaluation report with L_classification_evaluation validation capabilities.
    
    Validates evaluation stage outputs by checking:
    - Exact 47-question count per point
    - Score bounds between 0 and 1.2 for all dimension and point scores  
    - Dimension weighted averages match documented formulas
    - Composition trace mathematical correctness
    
    Implements first_failure reporting and canonical trace artifact generation.
    """
    satisfied: bool
    results: List[GuardResult]
    first_failure: Optional[GuardResult]
    canonical_trace: List[Dict[str, Any]]
    validation_metadata: Dict[str, Any]

    def __init__(self, satisfied: bool, results: List[GuardResult], 
                 first_failure: Optional[GuardResult] = None):
        self.satisfied = satisfied
        self.results = results
        self.first_failure = first_failure
        self.canonical_trace = self._generate_canonical_trace()
        self.validation_metadata = self._generate_validation_metadata()

    def to_trace(self) -> List[Dict[str, Any]]:
        return [r.to_canonical_dict() for r in self.results]
    
    def _generate_canonical_trace(self) -> List[Dict[str, Any]]:
        """Generate canonical trace showing all validation steps."""
        trace = []
        for result in self.results:
            step = {
                "step_id": result.clause_id,
                "validation_rule": result.clause_id,
                "satisfied": result.satisfied,
                "minimal_facts": _to_canonical(result.minimal_facts),
                "reason": result.reason or "",
                "timestamp_canonical": "deterministic",  # No actual timestamps for determinism
                "evaluation_order": len(trace) + 1
            }
            trace.append(step)
        return trace
    
    def _generate_validation_metadata(self) -> Dict[str, Any]:
        """Generate metadata about validation process."""
        failed_rules = [r for r in self.results if not r.satisfied]
        return {
            "total_rules_evaluated": len(self.results),
            "rules_passed": len(self.results) - len(failed_rules),
            "rules_failed": len(failed_rules),
            "first_failure_rule": self.first_failure.clause_id if self.first_failure else None,
            "first_failure_reason": self.first_failure.reason if self.first_failure else None,
            "validation_complete": True,
            "deterministic_execution": True
        }

    @classmethod
    def validate_l_classification_evaluation(cls, evaluation_data: Dict[str, Any]) -> 'EvaluationReport':
        """Validate L_classification_evaluation stage outputs with deterministic shield.
        
        Args:
            evaluation_data: Dictionary containing point evaluations with structure:
                {
                    "point_1": {
                        "dimensions": {
                            "DE-1": {"questions": [...], "weighted_average": float, "total_questions": int},
                            "DE-2": {"questions": [...], "weighted_average": float, "total_questions": int},
                            "DE-3": {"questions": [...], "weighted_average": float, "total_questions": int}, 
                            "DE-4": {"questions": [...], "weighted_average": float, "total_questions": int}
                        },
                        "final_score": float,
                        "total_questions": int,
                        "composition_trace": {...}
                    },
                    ...
                }
        
        Returns:
            EvaluationReport with validation results and canonical trace
        """
        results: List[GuardResult] = []
        first_failure: Optional[GuardResult] = None
        
        # Define validation rules in canonical order
        validation_rules = [
            ("QUESTION_COUNT_47", "Each point must have exactly 47 questions"),
            ("SCORE_BOUNDS_DIMENSION", "All dimension scores must be between 0 and 1.2"),
            ("SCORE_BOUNDS_POINT", "All point scores must be between 0 and 1.2"), 
            ("WEIGHTED_AVERAGES_FORMULA", "Dimension weighted averages must match documented formulas"),
            ("COMPOSITION_TRACE_CORRECTNESS", "Composition trace mathematical correctness validation")
        ]
        
        for rule_id, rule_description in validation_rules:
            try:
                if rule_id == "QUESTION_COUNT_47":
                    result = cls._validate_question_count_47(evaluation_data)
                elif rule_id == "SCORE_BOUNDS_DIMENSION":
                    result = cls._validate_dimension_score_bounds(evaluation_data)
                elif rule_id == "SCORE_BOUNDS_POINT":
                    result = cls._validate_point_score_bounds(evaluation_data)
                elif rule_id == "WEIGHTED_AVERAGES_FORMULA":
                    result = cls._validate_weighted_averages_formula(evaluation_data)
                elif rule_id == "COMPOSITION_TRACE_CORRECTNESS":
                    result = cls._validate_composition_trace_correctness(evaluation_data)
                else:
                    result = GuardResult(
                        clause_id=rule_id,
                        satisfied=False,
                        minimal_facts={"error": "Unknown validation rule"},
                        reason="unknown_rule"
                    )
                
                results.append(result)
                
                # First failure detection
                if not result.satisfied and first_failure is None:
                    first_failure = result
                    
            except Exception as e:
                error_result = GuardResult(
                    clause_id=rule_id,
                    satisfied=False,
                    minimal_facts={"exception": str(type(e).__name__), "message": str(e)},
                    reason="validation_exception"
                )
                results.append(error_result)
                if first_failure is None:
                    first_failure = error_result
        
        all_satisfied = all(r.satisfied for r in results)
        return cls(satisfied=all_satisfied, results=results, first_failure=first_failure)
    
    @staticmethod
    def _validate_question_count_47(evaluation_data: Dict[str, Any]) -> GuardResult:
        """Validate exact 47-question count per point."""
        violations = []
        
        for point_id, point_data in evaluation_data.items():
            total_questions = point_data.get("total_questions", 0)
            if total_questions != 47:
                violations.append({
                    "point_id": point_id,
                    "actual_count": total_questions,
                    "expected_count": 47
                })
        
        if violations:
            return GuardResult(
                clause_id="QUESTION_COUNT_47",
                satisfied=False,
                minimal_facts={
                    "violations": violations,
                    "total_violations": len(violations)
                },
                reason="incorrect_question_count"
            )
        
        return GuardResult(
            clause_id="QUESTION_COUNT_47",
            satisfied=True,
            minimal_facts={
                "points_validated": len(evaluation_data),
                "questions_per_point": 47
            },
            reason="question_count_valid"
        )
    
    @staticmethod
    def _validate_dimension_score_bounds(evaluation_data: Dict[str, Any]) -> GuardResult:
        """Validate dimension scores are between 0 and 1.2."""
        violations = []
        
        for point_id, point_data in evaluation_data.items():
            dimensions = point_data.get("dimensions", {})
            
            for dimension_id, dimension_data in dimensions.items():
                weighted_average = dimension_data.get("weighted_average", 0.0)
                
                if not (0.0 <= weighted_average <= 1.2):
                    violations.append({
                        "point_id": point_id,
                        "dimension_id": dimension_id,
                        "actual_score": weighted_average,
                        "valid_range": [0.0, 1.2]
                    })
                
                # Also validate individual question scores if available
                questions = dimension_data.get("questions", [])
                for question in questions:
                    final_score = question.get("final_score", 0.0)
                    if not (0.0 <= final_score <= 1.2):
                        violations.append({
                            "point_id": point_id,
                            "dimension_id": dimension_id,
                            "question_id": question.get("question_id", "unknown"),
                            "actual_score": final_score,
                            "valid_range": [0.0, 1.2],
                            "violation_type": "question_score"
                        })
        
        if violations:
            return GuardResult(
                clause_id="SCORE_BOUNDS_DIMENSION",
                satisfied=False,
                minimal_facts={
                    "violations": violations,
                    "total_violations": len(violations)
                },
                reason="dimension_score_out_of_bounds"
            )
        
        return GuardResult(
            clause_id="SCORE_BOUNDS_DIMENSION",
            satisfied=True,
            minimal_facts={
                "dimensions_validated": sum(len(pd.get("dimensions", {})) for pd in evaluation_data.values()),
                "valid_range": [0.0, 1.2]
            },
            reason="dimension_scores_valid"
        )
    
    @staticmethod 
    def _validate_point_score_bounds(evaluation_data: Dict[str, Any]) -> GuardResult:
        """Validate point scores are between 0 and 1.2."""
        violations = []
        
        for point_id, point_data in evaluation_data.items():
            final_score = point_data.get("final_score", 0.0)
            
            if not (0.0 <= final_score <= 1.2):
                violations.append({
                    "point_id": point_id,
                    "actual_score": final_score,
                    "valid_range": [0.0, 1.2]
                })
        
        if violations:
            return GuardResult(
                clause_id="SCORE_BOUNDS_POINT",
                satisfied=False,
                minimal_facts={
                    "violations": violations,
                    "total_violations": len(violations)
                },
                reason="point_score_out_of_bounds"
            )
        
        return GuardResult(
            clause_id="SCORE_BOUNDS_POINT",
            satisfied=True,
            minimal_facts={
                "points_validated": len(evaluation_data),
                "valid_range": [0.0, 1.2]
            },
            reason="point_scores_valid"
        )
    
    @staticmethod
    def _validate_weighted_averages_formula(evaluation_data: Dict[str, Any]) -> GuardResult:
        """Validate dimension weighted averages match documented formulas."""
        violations = []
        
        for point_id, point_data in evaluation_data.items():
            dimensions = point_data.get("dimensions", {})
            
            for dimension_id, dimension_data in dimensions.items():
                questions = dimension_data.get("questions", [])
                reported_average = dimension_data.get("weighted_average", 0.0)
                
                if not questions:
                    continue
                
                # Calculate expected average (simple arithmetic mean)
                total_score = sum(q.get("final_score", 0.0) for q in questions)
                expected_average = total_score / len(questions) if questions else 0.0
                
                # Allow small floating point tolerance
                tolerance = 1e-6
                if abs(reported_average - expected_average) > tolerance:
                    violations.append({
                        "point_id": point_id,
                        "dimension_id": dimension_id,
                        "reported_average": reported_average,
                        "expected_average": expected_average,
                        "difference": abs(reported_average - expected_average),
                        "tolerance": tolerance,
                        "question_count": len(questions)
                    })
        
        if violations:
            return GuardResult(
                clause_id="WEIGHTED_AVERAGES_FORMULA",
                satisfied=False,
                minimal_facts={
                    "violations": violations,
                    "total_violations": len(violations)
                },
                reason="weighted_average_formula_mismatch"
            )
        
        return GuardResult(
            clause_id="WEIGHTED_AVERAGES_FORMULA",
            satisfied=True,
            minimal_facts={
                "dimensions_validated": sum(len(pd.get("dimensions", {})) for pd in evaluation_data.values()),
                "formula_type": "arithmetic_mean",
                "tolerance": 1e-6
            },
            reason="weighted_averages_valid"
        )
    
    @staticmethod
    def _validate_composition_trace_correctness(evaluation_data: Dict[str, Any]) -> GuardResult:
        """Validate composition trace mathematical correctness."""
        violations = []
        
        # Decálogo weights from the scoring system
        DECALOGO_WEIGHTS = {
            "DE-1": 0.30,
            "DE-2": 0.25,
            "DE-3": 0.25,
            "DE-4": 0.20
        }
        
        for point_id, point_data in evaluation_data.items():
            dimensions = point_data.get("dimensions", {})
            reported_final_score = point_data.get("final_score", 0.0)
            composition_trace = point_data.get("composition_trace", {})
            
            # Calculate expected final score using Decálogo weights
            weighted_sum = 0.0
            total_weight = 0.0
            
            dimension_scores_used = {}
            for dimension_id in ["DE-1", "DE-2", "DE-3", "DE-4"]:
                if dimension_id in dimensions:
                    dimension_score = dimensions[dimension_id].get("weighted_average", 0.0)
                    weight = DECALOGO_WEIGHTS.get(dimension_id, 0.0)
                    
                    weighted_sum += dimension_score * weight
                    total_weight += weight
                    dimension_scores_used[dimension_id] = {
                        "score": dimension_score,
                        "weight": weight,
                        "contribution": dimension_score * weight
                    }
            
            expected_final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Validate composition trace if present
            trace_violations = []
            if composition_trace:
                trace_weighted_sum = composition_trace.get("weighted_sum", 0.0)
                trace_total_weight = composition_trace.get("total_weight", 0.0)
                trace_final_score = composition_trace.get("final_score", 0.0)
                
                tolerance = 1e-6
                if abs(trace_weighted_sum - weighted_sum) > tolerance:
                    trace_violations.append("weighted_sum_mismatch")
                if abs(trace_total_weight - total_weight) > tolerance:
                    trace_violations.append("total_weight_mismatch")
                if abs(trace_final_score - expected_final_score) > tolerance:
                    trace_violations.append("final_score_calculation_mismatch")
            
            # Check if reported final score matches expected
            tolerance = 1e-6
            if abs(reported_final_score - expected_final_score) > tolerance or trace_violations:
                violations.append({
                    "point_id": point_id,
                    "reported_final_score": reported_final_score,
                    "expected_final_score": expected_final_score,
                    "difference": abs(reported_final_score - expected_final_score),
                    "tolerance": tolerance,
                    "dimension_scores_used": dimension_scores_used,
                    "composition_calculation": {
                        "weighted_sum": weighted_sum,
                        "total_weight": total_weight,
                        "formula": "weighted_sum / total_weight"
                    },
                    "trace_violations": trace_violations
                })
        
        if violations:
            return GuardResult(
                clause_id="COMPOSITION_TRACE_CORRECTNESS",
                satisfied=False,
                minimal_facts={
                    "violations": violations,
                    "total_violations": len(violations),
                    "decalogo_weights": DECALOGO_WEIGHTS
                },
                reason="composition_trace_mathematical_error"
            )
        
        return GuardResult(
            clause_id="COMPOSITION_TRACE_CORRECTNESS",
            satisfied=True,
            minimal_facts={
                "points_validated": len(evaluation_data),
                "decalogo_weights": DECALOGO_WEIGHTS,
                "composition_formula": "weighted_sum / total_weight",
                "tolerance": 1e-6
            },
            reason="composition_trace_valid"
        )


class GuardEvaluator:
    """Deterministic guard evaluator.
    Evaluates catalog clauses in canonical (sorted by clause_id) order.
    """

    def __init__(self, catalog: PreconditionCatalog):
        self.catalog = catalog

    def evaluate(self, context: Any) -> EvaluationReport:
        ordered = self.catalog.ordered()
        results: List[GuardResult] = []
        first_failure: Optional[GuardResult] = None

        for clause in ordered:
            try:
                res = clause.predicate(context)
            except Exception as e:
                # Treat predicate errors as failures with stable reason
                res = GuardResult(
                    clause_id=clause.clause_id,
                    satisfied=False,
                    minimal_facts={"exception": str(type(e).__name__)},
                    reason="predicate_exception",
                )
            # Ensure clause_id correctness even if predicate forgot to set it
            if res.clause_id != clause.clause_id:
                res.clause_id = clause.clause_id  # type: ignore[attr-defined]
            results.append(res)
            if not res.satisfied and first_failure is None:
                first_failure = res
                # We still continue to evaluate all guards for complete traceability.

        all_ok = all(r.satisfied for r in results)
        return EvaluationReport(
            satisfied=all_ok, results=results, first_failure=first_failure
        )


# ------------------------------ Rendering ---------------------------------- #


class RejectionRenderer:
    """Produces canonical, typed rejection responses from evaluation results."""

    def __init__(self, mechanism: str = "deterministic_shield"):
        self.mechanism = mechanism

    def render(self, report: EvaluationReport) -> RejectionResponse:
        if report.satisfied:
            raise ValueError("Cannot render rejection for a satisfied report")

        failed = [r for r in report.results if not r.satisfied]
        violation_ids = [r.clause_id for r in failed]
        reasons = {r.clause_id: (r.reason or "unspecified_reason") for r in failed}

        # Aggregate minimal facts deterministically: union by key, prefer earlier (by clause_id order)
        facts: Dict[str, Any] = {}
        for r in sorted(failed, key=lambda x: x.clause_id):
            for k, v in _to_canonical(r.minimal_facts).items():
                if k not in facts:
                    facts[k] = v

        trace = report.to_trace()
        payload = {
            "mechanism": self.mechanism,
            "violation_ids": violation_ids,
            "reasons": reasons,
            "minimal_facts": facts,
            "trace": trace,
        }
        fp = fingerprint(payload)
        return RejectionResponse(
            mechanism=self.mechanism,
            violation_ids=violation_ids,
            reasons=reasons,
            minimal_facts=facts,
            trace=trace,
            canonical_fingerprint=fp,
        )


# --------------------------- Guarded Synthesizer ---------------------------- #


@dataclass
class SynthesisResponse:
    kind: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        payload = {"kind": self.kind, "value": _to_canonical(self.value)}
        return {
            **payload,
            "canonical_fingerprint": fingerprint(payload),
        }


class GuardedSynthesizer:
    """Wrapper that enforces guard satisfaction before invoking a synthesizer.

    Usage:
        catalog = PreconditionCatalog()
        catalog.add(GuardClause(
            clause_id="SAFETY.NONEMPTY_QUERY",
            description="The query string must be non-empty.",
            predicate=lambda ctx: GuardResult(
                clause_id="SAFETY.NONEMPTY_QUERY",
                satisfied=bool(getattr(ctx, "query", None)),
                minimal_facts={"query_present": bool(getattr(ctx, "query", None))},
                reason="empty_query" if not getattr(ctx, "query", None) else None,
            ),
        ))

        evaluator = GuardEvaluator(catalog)
        renderer = RejectionRenderer()
        guarded = GuardedSynthesizer(actual_synthesizer, evaluator, renderer)

        result = guarded.synthesize(ctx)
        if isinstance(result, RejectionResponse):
            # Handle rejection deterministically
            ...
        else:
            # Use synthesized response
            ...
    """

    def __init__(
        self,
        synthesizer: Callable[[Any], Any],
        evaluator: GuardEvaluator,
        renderer: RejectionRenderer,
    ):
        self._synthesizer = synthesizer
        self._evaluator = evaluator
        self._renderer = renderer

    def synthesize(self, context: Any) -> "SynthesisResponse | RejectionResponse":
        report = self._evaluator.evaluate(context)
        if not report.satisfied:
            # Prohibit any synthesis beyond failed guards
            return self._renderer.render(report)
        # All guards satisfied: run synthesis deterministically (wrapper does not alter RNG/state)
        value = self._synthesizer(context)
        return SynthesisResponse(kind="synthesis", value=value)


# ------------------------------- Quick Demo -------------------------------- #

if __name__ == "__main__":
    # Minimal, self-contained smoke test demonstrating identical rejections.
    class Ctx:
        def __init__(self, query: str, user_role: str):
            self.query = query
            self.user_role = user_role

    # Define catalog
    catalog = PreconditionCatalog()
    catalog.add(
        GuardClause(
            clause_id="SAFETY.NONEMPTY_QUERY",
            description="The query string must be non-empty.",
            predicate=lambda ctx: GuardResult(
                clause_id="SAFETY.NONEMPTY_QUERY",
                satisfied=bool(getattr(ctx, "query", None)),
                minimal_facts={"query_present": bool(getattr(ctx, "query", None))},
                reason="empty_query" if not getattr(ctx, "query", None) else None,
            ),
        )
    )
    catalog.add(
        GuardClause(
            clause_id="POLICY.ALLOWED_ROLE",
            description="User role must be in allowed set {analyst, admin}.",
            predicate=lambda ctx: GuardResult(
                clause_id="POLICY.ALLOWED_ROLE",
                satisfied=getattr(ctx, "user_role", None) in {"analyst", "admin"},
                minimal_facts={"user_role": getattr(ctx, "user_role", None)},
                reason=(
                    "role_not_allowed"
                    if getattr(ctx, "user_role", None) not in {"analyst", "admin"}
                    else None
                ),
            ),
            severity=GuardSeverity.POLICY,
        )
    )

    evaluator = GuardEvaluator(catalog)
    renderer = RejectionRenderer()

    def dummy_synth(ctx: Ctx) -> Dict[str, Any]:
        # Pretend to build an answer
        return {"answer": f"Echo: {ctx.query}", "role": ctx.user_role}

    guarded = GuardedSynthesizer(dummy_synth, evaluator, renderer)

    ctx1 = Ctx(query="", user_role="guest")
    rej1 = guarded.synthesize(ctx1)  # Expect rejection
    ctx2 = Ctx(query="", user_role="guest")
    rej2 = guarded.synthesize(ctx2)  # Identical rejection

    if isinstance(rej1, RejectionResponse) and isinstance(rej2, RejectionResponse):
        d1 = rej1.to_dict()
        d2 = rej2.to_dict()
        # Print fingerprints to illustrate determinism
        print("rej1 fp:", d1["canonical_fingerprint"])  # noqa: T201
        print("rej2 fp:", d2["canonical_fingerprint"])  # noqa: T201
        assert d1 == d2

    ctx_ok = Ctx(query="hello", user_role="analyst")
    ok = guarded.synthesize(ctx_ok)
    if not isinstance(ok, RejectionResponse):
        print("synthesized:", ok.to_dict())  # noqa: T201

    # Demonstrate L_classification_evaluation validation
    print("\n=== L_classification_evaluation Validation Demo ===")  # noqa: T201
    
    # Create valid test data
    valid_evaluation_data = {
        "point_1": {
            "dimensions": {
                "DE-1": {
                    "questions": [
                        {"question_id": f"DE1_Q{i}", "final_score": 0.8} for i in range(12)
                    ],
                    "weighted_average": 0.8,
                    "total_questions": 12
                },
                "DE-2": {
                    "questions": [
                        {"question_id": f"DE2_Q{i}", "final_score": 0.6} for i in range(12)
                    ],
                    "weighted_average": 0.6,
                    "total_questions": 12
                },
                "DE-3": {
                    "questions": [
                        {"question_id": f"DE3_Q{i}", "final_score": 0.7} for i in range(12)
                    ],
                    "weighted_average": 0.7,
                    "total_questions": 12
                },
                "DE-4": {
                    "questions": [
                        {"question_id": f"DE4_Q{i}", "final_score": 0.9} for i in range(11)
                    ],
                    "weighted_average": 0.9,
                    "total_questions": 11
                }
            },
            "final_score": 0.745,  # 0.8*0.3 + 0.6*0.25 + 0.7*0.25 + 0.9*0.2
            "total_questions": 47,
            "composition_trace": {
                "weighted_sum": 0.745,
                "total_weight": 1.0,
                "final_score": 0.745
            }
        }
    }
    
    # Test valid data
    valid_report = EvaluationReport.validate_l_classification_evaluation(valid_evaluation_data)
    print(f"Valid data - Satisfied: {valid_report.satisfied}")  # noqa: T201
    print(f"Validation metadata: {valid_report.validation_metadata}")  # noqa: T201
    
    # Create invalid test data (wrong question count)
    invalid_evaluation_data = {
        "point_1": {
            "dimensions": {
                "DE-1": {
                    "questions": [
                        {"question_id": f"DE1_Q{i}", "final_score": 0.8} for i in range(10)  # Too few
                    ],
                    "weighted_average": 0.8,
                    "total_questions": 10
                }
            },
            "final_score": 0.8,
            "total_questions": 10  # Should be 47
        }
    }
    
    # Test invalid data - should fail at first rule
    invalid_report = EvaluationReport.validate_l_classification_evaluation(invalid_evaluation_data)
    print(f"Invalid data - Satisfied: {invalid_report.satisfied}")  # noqa: T201
    print(f"First failure: {invalid_report.first_failure.clause_id if invalid_report.first_failure else None}")  # noqa: T201
    print(f"Canonical trace length: {len(invalid_report.canonical_trace)}")  # noqa: T201
