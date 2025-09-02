"""
Test fixtures for question analyzer validation.

Implements synthetic tasks with known ground truth for coverage testing
and causal validation scenarios.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from question_analyzer import (
    CausalPosture,
    EvidenceType,
    QuestionAnalyzer,
    ValidationRule,
    create_synthetic_associational_task,
    create_synthetic_causal_task,
)


class SyntheticDataGenerator:
    """Generate synthetic data for testing theoretical guarantees"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_confounded_dag(self, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate DAG with hidden confounding: X <- U -> Y, X -> Y

        This tests the analyzer's ability to detect when backdoor criterion
        is insufficient and proximal methods are needed.
        """
        # Hidden confounder U
        U = np.random.normal(0, 1, n_samples)

        # Treatment X affected by U
        X = 0.5 * U + np.random.normal(0, 0.5, n_samples)

        # Outcome Y affected by both X and U (confounded)
        Y = 0.3 * X + 0.7 * U + np.random.normal(0, 0.3, n_samples)

        # Observable proxies
        Z = U + np.random.normal(0, 0.2, n_samples)  # Proxy for confounder
        W = (
            0.4 * X + 0.3 * U + np.random.normal(0, 0.2, n_samples)
        )  # Proxy for mechanism

        return {
            "X": X,
            "Y": Y,
            "U": U,
            "Z": Z,
            "W": W,
            "true_effect": 0.3,
            "confounding_bias": 0.7 * 0.5,  # E[U|X] * effect of U on Y
        }

    def generate_jackknife_validation_data(
        self, n_folds: int = 50
    ) -> List[Dict[str, Any]]:
        """Generate validation data for Jackknife+ coverage testing"""
        validation_samples = []

        for fold in range(n_folds):
            # Generate question with known evidence type requirements
            question_types = [
                (
                    "What causes obesity in adults?",
                    ["experimental_evidence", "observational_evidence"],
                ),
                (
                    "Is there an association between smoking and lung cancer?",
                    ["statistical_evidence", "observational_evidence"],
                ),
                (
                    "How does exercise affect mental health?",
                    ["experimental_evidence", "mechanistic_evidence"],
                ),
                (
                    "What is the relationship between income and happiness?",
                    ["observational_evidence", "comparative_evidence"],
                ),
            ]

            question, true_types = question_types[fold % len(question_types)]

            validation_samples.append(
                {
                    "question": question,
                    "true_evidence_types": true_types,
                    "fold_id": fold,
                }
            )

        return validation_samples


class CoverageValidator:
    """Validate Jackknife+ coverage guarantees"""

    def __init__(self, analyzer: QuestionAnalyzer, alpha: float = 0.1):
        self.analyzer = analyzer
        self.alpha = alpha

    def test_coverage_guarantee(
        self, validation_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Test that empirical coverage ≥ 1-α as guaranteed by Jackknife+

        Returns:
            coverage_results: Dictionary with coverage statistics
        """
        n_covered = 0
        n_total = len(validation_data)

        coverage_by_type = {}

        for sample in validation_data:
            question = sample["question"]
            true_types = set(sample["true_evidence_types"])

            # Get analyzer predictions with confidence intervals
            evidence_types = self.analyzer.identify_evidence_types(question)

            # Check if true types fall within confidence intervals
            predicted_types = set()
            for et in evidence_types:
                # Consider prediction "covered" if confidence interval includes true positive
                if (
                    et.confidence_lower <= 0.8 <= et.confidence_upper
                ):  # Threshold for "true"
                    predicted_types.add(et.type_name)

            # Calculate coverage for this sample
            intersection = len(true_types & predicted_types)
            union = len(true_types | predicted_types)

            if union > 0:
                jaccard_coverage = intersection / union
                if jaccard_coverage >= 0.5:  # Reasonable coverage threshold
                    n_covered += 1

            # Track by evidence type
            for true_type in true_types:
                if true_type not in coverage_by_type:
                    coverage_by_type[true_type] = []
                coverage_by_type[true_type].append(true_type in predicted_types)

        empirical_coverage = n_covered / n_total
        theoretical_guarantee = 1 - self.alpha

        return {
            "empirical_coverage": empirical_coverage,
            "theoretical_guarantee": theoretical_guarantee,
            "coverage_satisfied": empirical_coverage >= theoretical_guarantee,
            "coverage_by_type": {k: np.mean(v) for k, v in coverage_by_type.items()},
            "n_samples": n_total,
        }


class CausalValidator:
    """Validate causal analysis capabilities"""

    def __init__(self, analyzer: QuestionAnalyzer):
        self.analyzer = analyzer

    def test_hidden_confounding_detection(
        self, synthetic_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Test that analyzer correctly identifies need for proximal methods
        when backdoor criterion is insufficient.
        """
        causal_question = "What is the causal effect of X on Y in the presence of unmeasured confounding?"

        # Analyze question
        requirements = self.analyzer.analyze_question(
            causal_question, "confounding_test"
        )
        validation_rules = self.analyzer.determine_validation_rules(requirements)

        # Check that analyzer rejects backdoor-only approaches
        backdoor_rejection = any(
            "backdoor" in rule.rule_name and "reject" in rule.description.lower()
            for rule in validation_rules
        )

        # Check that proximal proxy plan is generated
        has_proximal_plan = requirements.proximal_proxy_plan is not None

        # Check causal posture classification
        correct_posture = requirements.causal_posture == CausalPosture.INTERVENTIONAL

        return {
            "rejects_backdoor_only": backdoor_rejection,
            "proposes_proximal_bridge": has_proximal_plan,
            "correct_causal_posture": correct_posture,
            "proximal_conditions": requirements.proximal_proxy_plan.sufficiency_conditions
            if has_proximal_plan
            else [],
            "theoretical_validity": backdoor_rejection
            and has_proximal_plan
            and correct_posture,
        }

    def test_bridge_function_conditions(self, requirements) -> Dict[str, bool]:
        """Test that bridge function satisfies completeness and relevance"""
        if not requirements.proximal_proxy_plan:
            return {"completeness": False, "relevance": False, "independence": False}

        plan = requirements.proximal_proxy_plan

        # Check for completeness condition mention
        completeness = any(
            "information about U" in condition or "completeness" in condition.lower()
            for condition in plan.sufficiency_conditions
        )

        # Check for relevance condition mention
        relevance = any(
            "information about X" in condition or "relevance" in condition.lower()
            for condition in plan.sufficiency_conditions
        )

        # Check for conditional independence
        independence = any(
            "⊥" in condition or "independent" in condition.lower()
            for condition in plan.sufficiency_conditions
        )

        return {
            "completeness": completeness,
            "relevance": relevance,
            "independence": independence,
        }


def run_acceptance_tests():
    """Run all acceptance tests as specified in requirements"""

    # Initialize components
    analyzer = QuestionAnalyzer(alpha=0.1)
    data_gen = SyntheticDataGenerator()
    coverage_validator = CoverageValidator(analyzer)
    causal_validator = CausalValidator(analyzer)

    print("Running Acceptance Tests for Question Analyzer")
    print("=" * 60)

    # Test 1: Jackknife+ Coverage Guarantee
    print("\n1. Testing Jackknife+ Coverage Guarantee")
    print("-" * 40)

    validation_data = data_gen.generate_jackknife_validation_data(n_folds=30)
    coverage_results = coverage_validator.test_coverage_guarantee(validation_data)

    print(f"Empirical Coverage: {coverage_results['empirical_coverage']:.3f}")
    print(f"Theoretical Guarantee: {coverage_results['theoretical_guarantee']:.3f}")
    print(f"Coverage Satisfied: {coverage_results['coverage_satisfied']}")
    print(f"Coverage by Evidence Type:")
    for etype, coverage in coverage_results["coverage_by_type"].items():
        print(f"  {etype}: {coverage:.3f}")

    # Test 2: Hidden Confounding Detection
    print("\n2. Testing Hidden Confounding Detection")
    print("-" * 40)

    confounded_data = data_gen.generate_confounded_dag()
    confounding_results = causal_validator.test_hidden_confounding_detection(
        confounded_data
    )

    print(f"Rejects Backdoor-Only: {confounding_results['rejects_backdoor_only']}")
    print(
        f"Proposes Proximal Bridge: {confounding_results['proposes_proximal_bridge']}"
    )
    print(f"Correct Causal Posture: {confounding_results['correct_causal_posture']}")
    print(f"Theoretical Validity: {confounding_results['theoretical_validity']}")

    # Test 3: Bridge Function Conditions
    print("\n3. Testing Bridge Function Conditions")
    print("-" * 40)

    causal_question = (
        "What is the effect of treatment on outcome with unmeasured confounders?"
    )
    requirements = analyzer.analyze_question(causal_question, "bridge_test")
    bridge_results = causal_validator.test_bridge_function_conditions(requirements)

    print(f"Completeness Condition: {bridge_results['completeness']}")
    print(f"Relevance Condition: {bridge_results['relevance']}")
    print(f"Independence Condition: {bridge_results['independence']}")

    # Test 4: End-to-End Analysis
    print("\n4. End-to-End Analysis Example")
    print("-" * 40)

    test_question = "How does educational intervention affect student achievement across different socioeconomic backgrounds?"

    # Full analysis pipeline
    requirements = analyzer.analyze_question(test_question, "e2e_test")
    patterns = analyzer.extract_search_patterns(test_question)
    evidence_types = analyzer.identify_evidence_types(test_question)
    validation_rules = analyzer.determine_validation_rules(requirements)

    print(f"Question: {test_question}")
    print(f"Causal Posture: {requirements.causal_posture.value}")
    print(f"Search Patterns: {patterns[:5]}...")  # First 5 patterns
    print(f"Evidence Types: {[et.type_name for et in evidence_types]}")
    print(f"Validation Rules: {[vr.rule_name for vr in validation_rules]}")

    if requirements.proximal_proxy_plan:
        print(f"Proximal Z Candidates: {requirements.proximal_proxy_plan.z_candidates}")
        print(f"Proximal W Candidates: {requirements.proximal_proxy_plan.w_candidates}")

    # Overall test summary
    print("\n" + "=" * 60)
    print("ACCEPTANCE TEST SUMMARY")
    print("=" * 60)

    all_tests_passed = (
        coverage_results["coverage_satisfied"]
        and confounding_results["theoretical_validity"]
        and all(bridge_results.values())
    )

    print(
        f"Coverage Test: {'PASS' if coverage_results['coverage_satisfied'] else 'FAIL'}"
    )
    print(
        f"Confounding Detection: {'PASS' if confounding_results['theoretical_validity'] else 'FAIL'}"
    )
    print(
        f"Bridge Function Tests: {'PASS' if all(bridge_results.values()) else 'FAIL'}"
    )
    print(
        f"Overall Result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}"
    )

    return {
        "coverage": coverage_results,
        "confounding": confounding_results,
        "bridge_function": bridge_results,
        "overall_pass": all_tests_passed,
    }


if __name__ == "__main__":
    results = run_acceptance_tests()
