"""
Example usage of the Constraint Validation System
Demonstrates CRC bounds, Horn clause checking, and adversarial testing
"""

import logging

import numpy as np

# # # from constraint_validator import SatisfiabilityLevel, create_constraint_validator  # Module not found  # Module not found  # Module not found

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Demonstrate constraint validation system capabilities"""

    print("=" * 60)
    print("Constraint Validation System - Example Usage")
    print("=" * 60)

    # Initialize validator
    validator = create_constraint_validator(alpha=0.1, delta=0.05)
    print(f"Initialized validator with α={validator.alpha}, δ={validator.delta}")
    print()

    # Example 1: Dimension Requirement Validation
    print("1. DIMENSION REQUIREMENT VALIDATION")
    print("-" * 40)

    # Good evidence example
    good_evidence = [
        {
            "score": 0.85,
            "indicators": ["diagnostic_completeness", "stakeholder_participation"],
        },
        {
            "score": 0.90,
            "indicators": ["territorial_analysis", "stakeholder_participation"],
        },
        {
            "score": 0.75,
            "indicators": ["diagnostic_completeness", "territorial_analysis"],
        },
        {
            "score": 0.82,
            "indicators": [
                "diagnostic_completeness",
                "stakeholder_participation",
                "territorial_analysis",
            ],
        },
    ]

    passed, proof = validator.validate_dimension_requirements(
        "territorial_planning", good_evidence
    )
    print(f"Good evidence validation: {'PASSED' if passed else 'FAILED'}")
    print(f"  CRC Risk Bound: {proof.crc_risk_bound:.4f}")
    print(f"  Confidence Level: {proof.confidence_level:.4f}")
    print(f"  Passed Rules: {len(proof.passed_rules)}")
    print(f"  Counterexamples: {len(proof.counterexamples)}")
    print()

    # Poor evidence example
    poor_evidence = [
        {"score": 0.25, "indicators": ["diagnostic_completeness"]},
        {"score": 0.30, "indicators": ["stakeholder_participation"]},
    ]

    passed_poor, proof_poor = validator.validate_dimension_requirements(
        "low_quality_dimension", poor_evidence
    )
    print(f"Poor evidence validation: {'PASSED' if passed_poor else 'FAILED'}")
    print(f"  CRC Risk Bound: {proof_poor.crc_risk_bound:.4f}")
    print(f"  Counterexamples: {[ce['rule'] for ce in proof_poor.counterexamples]}")
    print()

    # Example 2: Conjunctive Conditions (Horn Clauses)
    print("2. CONJUNCTIVE CONDITIONS (HORN CLAUSES)")
    print("-" * 40)

    # Test P1 conditions
    p1_scores = {
        "diagnostic_quality": 0.80,
        "participation_score": 0.70,
        "programs_coverage": 0.60,
        "budget_alignment": 0.75,
    }

    p1_passed, p1_explanation = validator.check_conjunctive_conditions("P1", p1_scores)
    print(f"P1 Conjunctive Conditions: {'SATISFIED' if p1_passed else 'FAILED'}")
    print(f"  Total Conditions: {p1_explanation['total_conditions']}")
    print(f"  Satisfied: {p1_explanation['satisfied_conditions']}")
    if p1_explanation["failed_literals"]:
        print(f"  Failed Literals: {p1_explanation['failed_literals']}")
    print(f"  Z3 SMT Solver Available: {p1_explanation['z3_available']}")
    print()

    # Example 3: Mandatory Indicators
    print("3. MANDATORY INDICATORS VERIFICATION")
    print("-" * 40)

    complete_evidence = [
        {
            "indicators": ["diagnostic_completeness", "stakeholder_participation"],
            "score": 0.8,
        },
        {"indicators": ["territorial_analysis"], "score": 0.7},
        {
            "indicators": ["diagnostic_completeness", "territorial_analysis"],
            "score": 0.85,
        },
    ]

    all_present, risk_bound = validator.verify_mandatory_indicators(
        "P1", complete_evidence
    )
    print(f"P1 Mandatory Indicators: {'COMPLETE' if all_present else 'INCOMPLETE'}")
    print(f"  CRC Risk Bound for missed indicators: {risk_bound:.4f}")
    print()

    # Example 4: Satisfiability Calculation
    print("4. SATISFIABILITY CALCULATION WITH RCPS/α SUMMARY")
    print("-" * 40)

    multi_point_scores = {
        "P1": {"diagnostic": 0.85, "participation": 0.75, "territorial": 0.80},
        "P2": {"programs": 0.70, "budget": 0.65, "implementation": 0.60},
        "P3": {"monitoring": 0.55, "evaluation": 0.50, "reporting": 0.45},
        "P4": {"environmental": 0.90, "social": 0.85, "economic": 0.80},
    }

    satisfiability_result = validator.calculate_satisfiability(multi_point_scores)

    print(f"Overall Satisfiability: {satisfiability_result['satisfiability']}")
    print(f"Mean Score: {satisfiability_result['mean_score']:.3f}")
    print(f"Total Points Evaluated: {satisfiability_result['total_points']}")
    print()

    print("RCPS/α Summary:")
    rcps = satisfiability_result["rcps_alpha_summary"]
    print(f"  Coverage: {rcps['coverage']:.4f}")
    print(f"  Risk Bound: {rcps['risk_bound']:.4f}")
    print(f"  Target Coverage (1-α): {rcps['target_coverage']:.4f}")
    print(f"  Sample Size: {rcps['sample_size']}")
    print()

    print("Point-level Satisfiability:")
    for point, status in satisfiability_result["point_satisfiability"].items():
        print(f"  {point}: {status}")
    print()

    # Example 5: Proximal Causal Inference Bridge Tests
    print("5. PROXIMAL CAUSAL INFERENCE BRIDGE TESTS")
    print("-" * 40)

    # Valid bridge assumption (correlated proxy and outcome)
    np.random.seed(42)
    valid_proxy = np.random.normal(0, 1, 100)
    valid_outcome = 0.6 * valid_proxy + np.random.normal(0, 0.4, 100)

    valid_bridge = validator.run_bridge_assumption_test(
        "territorial_planning_proxy", valid_proxy, valid_outcome
    )

    print(f"Valid Bridge Test: {'PASSED' if valid_bridge.passed else 'FAILED'}")
    print(f"  Test Statistic: {valid_bridge.test_statistic:.4f}")
    print(f"  P-value: {valid_bridge.p_value:.6f}")
    print(f"  Valid at α=0.05: {valid_bridge.is_valid(0.05)}")
    print()

    # Invalid bridge assumption (uncorrelated)
    invalid_proxy = np.random.normal(0, 1, 100)
    invalid_outcome = np.random.normal(0, 1, 100)

    invalid_bridge = validator.run_bridge_assumption_test(
        "weak_proxy_assumption", invalid_proxy, invalid_outcome
    )

    print(f"Invalid Bridge Test: {'PASSED' if invalid_bridge.passed else 'FAILED'}")
    print(f"  Test Statistic: {invalid_bridge.test_statistic:.4f}")
    print(f"  P-value: {invalid_bridge.p_value:.6f}")
    print(f"  Valid at α=0.05: {invalid_bridge.is_valid(0.05)}")
    print()

    # Example 6: Adversarial Testing and System Audit
    print("6. ADVERSARIAL TESTING AND SYSTEM AUDIT")
    print("-" * 40)

    audit_results = validator.audit_system_with_adversarial_cases(n_trials=50)

    print("Adversarial Audit Results:")
    print(f"  Total Trials: {audit_results['n_trials']}")
    print(f"  Downgrade Rate: {audit_results['downgrade_rate']:.1%}")
    print(f"  Risk Violation Rate: {audit_results['risk_violation_rate']:.1%}")
    print(f"  Bridge Failure Rate: {audit_results['bridge_failure_rate']:.1%}")
    print(f"  Average Risk: {audit_results['average_risk']:.4f}")
    print()

    # Evaluation of audit results
    if audit_results["downgrade_rate"] > 0.3:
        print("✓ System properly downgrades under adversarial conditions")
    else:
        print("⚠ System may be too permissive under adversarial conditions")

    if audit_results["average_risk"] <= (1 - validator.alpha):
        print("✓ Average risk within CRC bounds")
    else:
        print("⚠ Average risk exceeds CRC bounds")
    print()

    # Example 7: Integration with Evidence System
    print("7. INTEGRATION DEMONSTRATION")
    print("-" * 40)

    # Simulate a complete validation workflow
    dimensions_to_validate = ["territorial", "diagnostic", "participatory"]
    all_validations_passed = True
    total_risk = 0.0

    for i, dimension in enumerate(dimensions_to_validate):
        # Generate synthetic evidence
        n_evidence = np.random.randint(3, 8)
        evidence = []

        for j in range(n_evidence):
            score = np.random.beta(3, 1)  # Tend toward higher scores
            indicators = [f"indicator_{j}", f"mandatory_{dimension}"]
            evidence.append({"score": score, "indicators": indicators})

        # Validate dimension
        passed, proof = validator.validate_dimension_requirements(dimension, evidence)
        total_risk += proof.crc_risk_bound

        if not passed:
            all_validations_passed = False

        print(
            f"  {dimension}: {'PASS' if passed else 'FAIL'} "
            f"(risk: {proof.crc_risk_bound:.3f})"
        )

    average_risk = total_risk / len(dimensions_to_validate)

    print(f"\nIntegration Summary:")
    print(f"  All Dimensions Valid: {all_validations_passed}")
    print(f"  Average Risk Bound: {average_risk:.4f}")
    print(f"  Within CRC Target: {average_risk <= (1 - validator.alpha)}")

    # Final system health check
    print()
    print("=" * 60)
    print("SYSTEM HEALTH CHECK")
    print("=" * 60)

    health_metrics = {
        "CRC Bounds Respected": average_risk <= (1 - validator.alpha),
        "Adversarial Downgrade": audit_results["downgrade_rate"] > 0.2,
        "Bridge Tests Available": len(validator._bridge_tests) > 0,
        "Risk History Tracking": len(validator._risk_history) > 0,
    }

    for metric, status in health_metrics.items():
        status_icon = "✓" if status else "✗"
        print(f"{status_icon} {metric}")

    all_healthy = all(health_metrics.values())
    print(f"\nOverall System Health: {'HEALTHY' if all_healthy else 'NEEDS ATTENTION'}")

    print("\n" + "=" * 60)
    print("Constraint validation demonstration completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
