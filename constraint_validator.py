"""
Constraint Validation System with Conformal Risk Control and Proximal Causal Inference
Implements monotone risk bounds using CRC and Horn clause SMT checks for conjunctive conditions.
"""

import hashlib
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Optional z3-solver for Horn/SMT checks
try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    warnings.warn(
        "z3-solver not available. Horn clause checks will use simplified logic."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatisfiabilityLevel(str, Enum):
    """Satisfiability outcomes for constraint validation"""

    SATISFIED = "SATISFIED"
    PARTIAL = "PARTIAL"
    UNSATISFIED = "UNSATISFIED"


@dataclass
class ProofObject:
    """Proof object containing passed rules and counterexamples"""

    passed_rules: List[str] = field(default_factory=list)
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    confidence_level: float = 0.0
    crc_risk_bound: float = 1.0

    def add_passed_rule(self, rule_name: str):
        """Add a rule that passed validation"""
        self.passed_rules.append(rule_name)

    def add_counterexample(self, rule_name: str, context: Dict[str, Any]):
        """Add a counterexample for failed validation"""
        self.counterexamples.append(
            {
                "rule": rule_name,
                "context": context,
                "timestamp": __import__("time").time(),
            }
        )


@dataclass
class CausalBridgeTest:
    """Bridge assumption test for proximal causal inference"""

    assumption_name: str
    test_statistic: float
    p_value: float
    passed: bool
    proxy_variables: List[str] = field(default_factory=list)

    def is_valid(self, alpha: float = 0.05) -> bool:
        """Check if bridge assumption holds at significance level alpha"""
        return self.passed and self.p_value < alpha


class ConstraintValidator:
    """
    Constraint validation system with CRC bounds and causal inference checks
    Treats satisfiability scoring as monotone risk bounded by Conformal Risk Control
    """

    def __init__(self, alpha: float = 0.1, delta: float = 0.05):
        """
        Initialize constraint validator

        Args:
            alpha: Coverage level for CRC (1-alpha coverage target)
            delta: Risk tolerance for bridge assumption tests
        """
        self.alpha = alpha
        self.delta = delta
        self._risk_history: List[float] = []
        self._bridge_tests: Dict[str, CausalBridgeTest] = {}
        self._dimension_cache: Dict[str, bool] = {}

    def validate_dimension_requirements(
        self, dimension: str, evidence: List[Dict[str, Any]]
    ) -> Tuple[bool, ProofObject]:
        """
        Validate dimension requirements with CRC risk bounds

        Args:
            dimension: Dimension identifier
            evidence: List of evidence objects

        Returns:
            Tuple of (validation_result, proof_object)
        """
        proof = ProofObject()

        if not evidence:
            proof.add_counterexample(
                "empty_evidence", {"dimension": dimension, "evidence_count": 0}
            )
            proof.crc_risk_bound = 1.0
            return False, proof

        # Extract scores for CRC calculation
        scores = [
            e.get("score", 0.0)
            for e in evidence
            if isinstance(e.get("score"), (int, float))
        ]

        if not scores:
            proof.add_counterexample(
                "no_valid_scores",
                {"dimension": dimension, "evidence_items": len(evidence)},
            )
            proof.crc_risk_bound = 1.0
            return False, proof

        # Calculate CRC risk bound using Jackknife+ methodology
        risk_bound, coverage = self._calculate_crc_risk_bound(np.array(scores))
        proof.crc_risk_bound = risk_bound
        proof.confidence_level = coverage

        # Dimension-specific validation rules
        validation_passed = True

        # Rule 1: Minimum evidence threshold
        min_evidence = 3
        if len(evidence) < min_evidence:
            proof.add_counterexample(
                "insufficient_evidence",
                {
                    "dimension": dimension,
                    "required": min_evidence,
                    "found": len(evidence),
                },
            )
            validation_passed = False
        else:
            proof.add_passed_rule("minimum_evidence_threshold")

        # Rule 2: Score quality threshold
        mean_score = np.mean(scores)
        score_threshold = 0.5
        if mean_score < score_threshold:
            proof.add_counterexample(
                "low_score_quality",
                {
                    "dimension": dimension,
                    "mean_score": mean_score,
                    "threshold": score_threshold,
                },
            )
            validation_passed = False
        else:
            proof.add_passed_rule("score_quality_threshold")

        # Rule 3: Risk bound acceptable
        risk_threshold = 0.2
        if risk_bound > risk_threshold:
            proof.add_counterexample(
                "high_risk_bound",
                {
                    "dimension": dimension,
                    "risk_bound": risk_bound,
                    "threshold": risk_threshold,
                },
            )
            validation_passed = False
        else:
            proof.add_passed_rule("crc_risk_bound_acceptable")

        # Cache result for efficiency
        cache_key = f"{dimension}_{len(evidence)}_{hash(str(sorted(scores)))}"
        self._dimension_cache[cache_key] = validation_passed

        self._risk_history.append(risk_bound)

        logger.info(
            f"Dimension validation for {dimension}: {validation_passed}, risk_bound: {risk_bound:.4f}"
        )

        return validation_passed, proof

    def check_conjunctive_conditions(
        self, point: str, scores: Dict[str, float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check conjunctive conditions using Horn clauses with SMT solver

        Args:
            point: Point identifier (e.g., "P1", "P2")
            scores: Dictionary of dimension scores

        Returns:
            Tuple of (satisfies_conditions, explanation_dict)
        """
        explanation = {
            "point": point,
            "total_conditions": 0,
            "satisfied_conditions": 0,
            "failed_literals": [],
            "z3_available": Z3_AVAILABLE,
        }

        if not scores:
            explanation["failed_literals"].append("empty_scores")
            return False, explanation

        # Define conjunctive conditions as Horn clauses
        conditions = self._get_horn_clauses_for_point(point)
        explanation["total_conditions"] = len(conditions)

        if Z3_AVAILABLE:
            return self._check_conditions_with_z3(conditions, scores, explanation)
        else:
            return self._check_conditions_simplified(conditions, scores, explanation)

    def verify_mandatory_indicators(
        self, point: str, evidence: List[Dict[str, Any]]
    ) -> Tuple[bool, float]:
        """
        Verify mandatory indicators with CRC risk bound for missed indicators

        Args:
            point: Point identifier
            evidence: Evidence list

        Returns:
            Tuple of (all_mandatory_present, crc_risk_bound)
        """
        mandatory_indicators = self._get_mandatory_indicators(point)

        if not mandatory_indicators:
            return True, 0.0

        # Check presence of mandatory indicators in evidence
        present_indicators = set()
        indicator_scores = []

        for ev in evidence:
            if isinstance(ev, dict):
                # Check for indicator presence in evidence metadata
                ev_indicators = ev.get("indicators", [])
                if isinstance(ev_indicators, list):
                    present_indicators.update(ev_indicators)

                # Collect scores for risk calculation
                if "score" in ev and isinstance(ev["score"], (int, float)):
                    indicator_scores.append(ev["score"])

        missing_indicators = set(mandatory_indicators) - present_indicators
        all_present = len(missing_indicators) == 0

        # Calculate risk bound for missed indicators
        if indicator_scores:
            risk_bound, _ = self._calculate_crc_risk_bound(np.array(indicator_scores))

            # Adjust risk based on missing indicators
            if missing_indicators:
                missing_ratio = len(missing_indicators) / len(mandatory_indicators)
                risk_bound = min(1.0, risk_bound * (1 + missing_ratio))
        else:
            risk_bound = 1.0 if missing_indicators else 0.0

        logger.info(
            f"Mandatory indicators for {point}: {len(present_indicators)}/{len(mandatory_indicators)} present, risk: {risk_bound:.4f}"
        )

        return all_present, risk_bound

    def calculate_satisfiability(
        self, scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate satisfiability with RCPS/α summary

        Args:
            scores: Nested dict {point: {dimension: score}}

        Returns:
            Dict with satisfiability result and RCPS/α summary
        """
        if not scores:
            return {
                "satisfiability": SatisfiabilityLevel.UNSATISFIED,
                "rcps_alpha_summary": {"coverage": 0.0, "risk_bound": 1.0},
                "details": {"error": "empty_scores"},
            }

        # Flatten scores for analysis
        all_scores = []
        point_satisfiability = {}

        for point, dim_scores in scores.items():
            if not isinstance(dim_scores, dict):
                continue

            point_scores = list(dim_scores.values())
            if point_scores:
                all_scores.extend(point_scores)
                mean_score = np.mean(point_scores)

                # Determine point satisfiability
                if mean_score >= 0.8:
                    point_satisfiability[point] = SatisfiabilityLevel.SATISFIED
                elif mean_score >= 0.5:
                    point_satisfiability[point] = SatisfiabilityLevel.PARTIAL
                else:
                    point_satisfiability[point] = SatisfiabilityLevel.UNSATISFIED

        # Overall satisfiability
        if not point_satisfiability:
            overall = SatisfiabilityLevel.UNSATISFIED
        else:
            satisfied_count = sum(
                1
                for s in point_satisfiability.values()
                if s == SatisfiabilityLevel.SATISFIED
            )
            partial_count = sum(
                1
                for s in point_satisfiability.values()
                if s == SatisfiabilityLevel.PARTIAL
            )
            total_points = len(point_satisfiability)

            if satisfied_count == total_points:
                overall = SatisfiabilityLevel.SATISFIED
            elif satisfied_count + partial_count >= total_points * 0.7:
                overall = SatisfiabilityLevel.PARTIAL
            else:
                overall = SatisfiabilityLevel.UNSATISFIED

        # RCPS/α summary
        if all_scores:
            risk_bound, coverage = self._calculate_crc_risk_bound(np.array(all_scores))
        else:
            risk_bound, coverage = 1.0, 0.0

        rcps_summary = {
            "coverage": coverage,
            "risk_bound": risk_bound,
            "alpha": self.alpha,
            "target_coverage": 1 - self.alpha,
            "sample_size": len(all_scores),
        }

        return {
            "satisfiability": overall,
            "rcps_alpha_summary": rcps_summary,
            "point_satisfiability": point_satisfiability,
            "total_points": len(point_satisfiability),
            "mean_score": np.mean(all_scores) if all_scores else 0.0,
        }

    def run_bridge_assumption_test(
        self,
        assumption_name: str,
        proxy_data: np.ndarray,
        outcome_data: np.ndarray,
        confounders: Optional[np.ndarray] = None,
    ) -> CausalBridgeTest:
        """
        Run proximal causal inference bridge assumption test

        Args:
            assumption_name: Name of the bridge assumption
            proxy_data: Proxy variable data
            outcome_data: Outcome variable data
            confounders: Optional confounder data

        Returns:
            CausalBridgeTest result
        """
        if len(proxy_data) != len(outcome_data):
            raise ValueError("Proxy and outcome data must have same length")

        # Simplified bridge test using correlation
        # In practice, this would use more sophisticated causal inference methods
        correlation = np.corrcoef(proxy_data, outcome_data)[0, 1]

        # Test statistic and p-value (simplified)
        n = len(proxy_data)
        test_stat = abs(correlation) * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)

        # Approximate p-value using normal distribution
        from scipy.stats import norm

        p_value = 2 * (1 - norm.cdf(abs(test_stat)))

        # Bridge assumption passes if correlation is significant and positive
        passed = correlation > 0.1 and p_value < self.delta

        bridge_test = CausalBridgeTest(
            assumption_name=assumption_name,
            test_statistic=test_stat,
            p_value=p_value,
            passed=passed,
            proxy_variables=[
                f"proxy_{i}"
                for i in range(proxy_data.shape[0] if proxy_data.ndim > 1 else 1)
            ],
        )

        self._bridge_tests[assumption_name] = bridge_test

        logger.info(
            f"Bridge test {assumption_name}: passed={passed}, p_value={p_value:.4f}"
        )

        return bridge_test

    def audit_system_with_adversarial_cases(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Audit system with adversarial cases to test downgrading behavior

        Args:
            n_trials: Number of adversarial trials

        Returns:
            Dict with audit results
        """
        audit_results = {
            "n_trials": n_trials,
            "downgrade_cases": 0,
            "risk_violations": 0,
            "bridge_failures": 0,
            "average_risk": 0.0,
        }

        total_risk = 0.0

        for trial in range(n_trials):
            # Generate adversarial case
            adversarial_scores = np.random.beta(0.3, 0.7, size=10)  # Low quality scores

            # Test dimension validation
            evidence = [
                {"score": s, "indicators": [f"ind_{i}"]}
                for i, s in enumerate(adversarial_scores)
            ]
            passed, proof = self.validate_dimension_requirements(
                f"test_dim_{trial}", evidence
            )

            if not passed:
                audit_results["downgrade_cases"] += 1

            # Check if CRC risk bound is violated
            if proof.crc_risk_bound > 1 - self.alpha:
                audit_results["risk_violations"] += 1

            total_risk += proof.crc_risk_bound

            # Test bridge assumption with adversarial proxy
            proxy_data = np.random.normal(0, 1, 50)
            outcome_data = np.random.normal(0, 1, 50)  # Uncorrelated

            bridge_test = self.run_bridge_assumption_test(
                f"adv_test_{trial}", proxy_data, outcome_data
            )
            if not bridge_test.passed:
                audit_results["bridge_failures"] += 1

        audit_results["average_risk"] = total_risk / n_trials
        audit_results["downgrade_rate"] = audit_results["downgrade_cases"] / n_trials
        audit_results["risk_violation_rate"] = (
            audit_results["risk_violations"] / n_trials
        )
        audit_results["bridge_failure_rate"] = (
            audit_results["bridge_failures"] / n_trials
        )

        logger.info(
            f"Adversarial audit: {audit_results['downgrade_rate']:.2%} downgrade rate, "
            f"{audit_results['average_risk']:.4f} avg risk"
        )

        return audit_results

    def _calculate_crc_risk_bound(self, scores: np.ndarray) -> Tuple[float, float]:
        """Calculate Conformal Risk Control bound using Jackknife+ methodology"""
        if len(scores) == 0:
            return 1.0, 0.0

        n = len(scores)

        # Jackknife+ conformity scores
        conformity_scores = np.zeros(n)
        for i in range(n):
            # Leave-one-out residual (simplified)
            loo_mean = np.mean(np.concatenate([scores[:i], scores[i + 1 :]]))
            conformity_scores[i] = abs(scores[i] - loo_mean)

        # Quantile for coverage
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = np.clip(q_level, 0.0, 1.0)

        threshold = np.quantile(conformity_scores, q_level)
        coverage = np.mean(conformity_scores <= threshold)

        # Risk bound is 1 - coverage
        risk_bound = max(0.0, 1.0 - coverage)

        return risk_bound, coverage

    def _get_horn_clauses_for_point(self, point: str) -> List[Dict[str, Any]]:
        """Get Horn clause conditions for a specific point"""
        # Example Horn clauses - in practice, these would be loaded from configuration
        clauses = {
            "P1": [
                {
                    "condition": "diagnostic_quality >= 0.7",
                    "dimensions": ["diagnostic"],
                },
                {
                    "condition": "participation_score >= 0.6",
                    "dimensions": ["participation"],
                },
                {
                    "condition": "diagnostic_quality * participation_score >= 0.5",
                    "dimensions": ["diagnostic", "participation"],
                },
            ],
            "P2": [
                {"condition": "programs_coverage >= 0.8", "dimensions": ["programs"]},
                {"condition": "budget_alignment >= 0.7", "dimensions": ["budget"]},
                {
                    "condition": "programs_coverage + budget_alignment >= 1.3",
                    "dimensions": ["programs", "budget"],
                },
            ],
        }

        return clauses.get(point, [])

    def _get_mandatory_indicators(self, point: str) -> List[str]:
        """Get mandatory indicators for a specific point"""
        indicators = {
            "P1": [
                "diagnostic_completeness",
                "stakeholder_participation",
                "territorial_analysis",
            ],
            "P2": ["program_definition", "budget_allocation", "timeline_specification"],
            "P3": ["monitoring_framework", "kpi_definition", "reporting_structure"],
        }

        return indicators.get(point, [])

    def _check_conditions_with_z3(
        self,
        conditions: List[Dict[str, Any]],
        scores: Dict[str, float],
        explanation: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check conditions using Z3 SMT solver"""
        if not Z3_AVAILABLE:
            return self._check_conditions_simplified(conditions, scores, explanation)

        solver = z3.Solver()
        z3_vars = {}

        # Create Z3 variables for scores
        for dim in scores:
            z3_vars[dim] = z3.Real(dim)
            solver.add(z3_vars[dim] >= 0.0)
            solver.add(z3_vars[dim] <= 1.0)
            solver.add(z3_vars[dim] == scores[dim])

        satisfied_count = 0

        for condition in conditions:
            condition_expr = condition.get("condition", "")

            try:
                # Parse and add condition to solver
                # This is a simplified version - full implementation would need proper parsing
                if ">=" in condition_expr:
                    parts = condition_expr.split(">=")
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        threshold = float(parts[1].strip())

                        if var_name in z3_vars:
                            constraint = z3_vars[var_name] >= threshold
                            solver.push()
                            solver.add(z3.Not(constraint))

                            if solver.check() == z3.unsat:
                                satisfied_count += 1
                            else:
                                explanation["failed_literals"].append(condition_expr)

                            solver.pop()

            except Exception as e:
                logger.warning(f"Z3 condition parsing failed: {e}")
                explanation["failed_literals"].append(f"parse_error: {condition_expr}")

        explanation["satisfied_conditions"] = satisfied_count
        return satisfied_count == len(conditions), explanation

    def _check_conditions_simplified(
        self,
        conditions: List[Dict[str, Any]],
        scores: Dict[str, float],
        explanation: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check conditions using simplified logic (when Z3 not available)"""
        satisfied_count = 0

        for condition in conditions:
            condition_expr = condition.get("condition", "")
            dimensions = condition.get("dimensions", [])

            try:
                # Simple condition evaluation
                if ">=" in condition_expr:
                    parts = condition_expr.split(">=")
                    if len(parts) == 2:
                        left_expr = parts[0].strip()
                        threshold = float(parts[1].strip())

                        # Evaluate left side
                        if left_expr in scores:
                            value = scores[left_expr]
                        elif "*" in left_expr:
                            # Handle multiplication
                            mult_parts = [p.strip() for p in left_expr.split("*")]
                            value = 1.0
                            for part in mult_parts:
                                if part in scores:
                                    value *= scores[part]
                                else:
                                    value = 0.0
                                    break
                        elif "+" in left_expr:
                            # Handle addition
                            add_parts = [p.strip() for p in left_expr.split("+")]
                            value = 0.0
                            for part in add_parts:
                                if part in scores:
                                    value += scores[part]
                        else:
                            value = 0.0

                        if value >= threshold:
                            satisfied_count += 1
                        else:
                            explanation["failed_literals"].append(
                                f"{left_expr}={value:.3f} < {threshold}"
                            )

            except Exception as e:
                logger.warning(f"Condition evaluation failed: {e}")
                explanation["failed_literals"].append(f"eval_error: {condition_expr}")

        explanation["satisfied_conditions"] = satisfied_count
        return satisfied_count == len(conditions), explanation


# Factory function for easy instantiation
def create_constraint_validator(
    alpha: float = 0.1, delta: float = 0.05
) -> ConstraintValidator:
    """
    Create a constraint validator instance with specified parameters

    Args:
        alpha: Coverage level for CRC (1-alpha coverage target)
        delta: Risk tolerance for bridge assumption tests

    Returns:
        ConstraintValidator instance
    """
    return ConstraintValidator(alpha=alpha, delta=delta)
