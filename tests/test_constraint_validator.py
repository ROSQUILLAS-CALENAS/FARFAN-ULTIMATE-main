"""
Test suite for constraint validation system
Tests CRC bounds, Horn clause checking, and adversarial downgrading behavior
"""

import numpy as np
import pytest

from constraint_validator import (
    CausalBridgeTest,
    ConstraintValidator,
    ProofObject,
    SatisfiabilityLevel,
)


class TestConstraintValidator:
    """Test suite for ConstraintValidator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.validator = ConstraintValidator(alpha=0.1, delta=0.05)

        # Sample evidence data
        self.good_evidence = [
            {"score": 0.8, "indicators": ["ind_1", "ind_2"]},
            {"score": 0.9, "indicators": ["ind_1", "ind_3"]},
            {"score": 0.7, "indicators": ["ind_2", "ind_4"]},
            {"score": 0.85, "indicators": ["ind_1", "ind_2", "ind_3"]},
        ]

        self.poor_evidence = [
            {"score": 0.3, "indicators": ["ind_1"]},
            {"score": 0.2, "indicators": ["ind_2"]},
        ]

        self.sample_scores = {
            "P1": {"diagnostic": 0.8, "participation": 0.7},
            "P2": {"programs": 0.9, "budget": 0.6},
            "P3": {"monitoring": 0.5, "evaluation": 0.4},
        }

    def test_validate_dimension_requirements_good_evidence(self):
        """Test dimension validation with good evidence"""
        passed, proof = self.validator.validate_dimension_requirements(
            "test_dimension", self.good_evidence
        )

        assert passed is True
        assert isinstance(proof, ProofObject)
        assert len(proof.passed_rules) > 0
        assert len(proof.counterexamples) == 0
        assert proof.crc_risk_bound < 0.5
        assert proof.confidence_level > 0.7

    def test_validate_dimension_requirements_poor_evidence(self):
        """Test dimension validation with poor evidence"""
        passed, proof = self.validator.validate_dimension_requirements(
            "test_dimension", self.poor_evidence
        )

        assert passed is False
        assert isinstance(proof, ProofObject)
        assert len(proof.counterexamples) > 0
        assert proof.crc_risk_bound > 0.2

    def test_validate_dimension_requirements_empty_evidence(self):
        """Test dimension validation with empty evidence"""
        passed, proof = self.validator.validate_dimension_requirements(
            "test_dimension", []
        )

        assert passed is False
        assert proof.crc_risk_bound == 1.0
        assert any("empty_evidence" in ce["rule"] for ce in proof.counterexamples)

    def test_check_conjunctive_conditions_satisfied(self):
        """Test conjunctive conditions with satisfying scores"""
        scores = {"diagnostic_quality": 0.8, "participation_score": 0.7}

        passed, explanation = self.validator.check_conjunctive_conditions("P1", scores)

        assert isinstance(passed, bool)
        assert isinstance(explanation, dict)
        assert "point" in explanation
        assert "total_conditions" in explanation
        assert "satisfied_conditions" in explanation
        assert "failed_literals" in explanation

    def test_check_conjunctive_conditions_empty_scores(self):
        """Test conjunctive conditions with empty scores"""
        passed, explanation = self.validator.check_conjunctive_conditions("P1", {})

        assert passed is False
        assert "empty_scores" in explanation["failed_literals"]

    def test_verify_mandatory_indicators_complete(self):
        """Test mandatory indicator verification with complete evidence"""
        complete_evidence = [
            {"indicators": ["diagnostic_completeness", "stakeholder_participation"]},
            {"indicators": ["territorial_analysis"], "score": 0.8},
        ]

        all_present, risk_bound = self.validator.verify_mandatory_indicators(
            "P1", complete_evidence
        )

        assert all_present is True
        assert risk_bound < 0.5

    def test_verify_mandatory_indicators_incomplete(self):
        """Test mandatory indicator verification with incomplete evidence"""
        incomplete_evidence = [
            {"indicators": ["diagnostic_completeness"], "score": 0.6}
        ]

        all_present, risk_bound = self.validator.verify_mandatory_indicators(
            "P1", incomplete_evidence
        )

        assert all_present is False
        assert risk_bound > 0.0

    def test_calculate_satisfiability_satisfied(self):
        """Test satisfiability calculation with high scores"""
        high_scores = {
            "P1": {"diagnostic": 0.9, "participation": 0.8},
            "P2": {"programs": 0.85, "budget": 0.9},
        }

        result = self.validator.calculate_satisfiability(high_scores)

        assert result["satisfiability"] == SatisfiabilityLevel.SATISFIED
        assert "rcps_alpha_summary" in result
        assert result["rcps_alpha_summary"]["alpha"] == self.validator.alpha
        assert result["mean_score"] > 0.8

    def test_calculate_satisfiability_partial(self):
        """Test satisfiability calculation with mixed scores"""
        mixed_scores = {
            "P1": {"diagnostic": 0.7, "participation": 0.6},
            "P2": {"programs": 0.5, "budget": 0.4},
        }

        result = self.validator.calculate_satisfiability(mixed_scores)

        assert result["satisfiability"] in [
            SatisfiabilityLevel.PARTIAL,
            SatisfiabilityLevel.UNSATISFIED,
        ]
        assert result["mean_score"] < 0.8

    def test_calculate_satisfiability_unsatisfied(self):
        """Test satisfiability calculation with low scores"""
        low_scores = {
            "P1": {"diagnostic": 0.3, "participation": 0.2},
            "P2": {"programs": 0.1, "budget": 0.4},
        }

        result = self.validator.calculate_satisfiability(low_scores)

        assert result["satisfiability"] == SatisfiabilityLevel.UNSATISFIED
        assert result["mean_score"] < 0.5

    def test_calculate_satisfiability_empty_scores(self):
        """Test satisfiability calculation with empty scores"""
        result = self.validator.calculate_satisfiability({})

        assert result["satisfiability"] == SatisfiabilityLevel.UNSATISFIED
        assert result["rcps_alpha_summary"]["risk_bound"] == 1.0

    def test_bridge_assumption_test_valid(self):
        """Test bridge assumption test with valid proxy"""
        # Create correlated proxy and outcome data
        np.random.seed(42)
        proxy_data = np.random.normal(0, 1, 100)
        outcome_data = 0.7 * proxy_data + np.random.normal(0, 0.3, 100)

        bridge_test = self.validator.run_bridge_assumption_test(
            "test_assumption", proxy_data, outcome_data
        )

        assert isinstance(bridge_test, CausalBridgeTest)
        assert bridge_test.assumption_name == "test_assumption"
        assert bridge_test.passed is True
        assert bridge_test.p_value < 0.05
        assert bridge_test.test_statistic > 0

    def test_bridge_assumption_test_invalid(self):
        """Test bridge assumption test with invalid proxy"""
        # Create uncorrelated data
        np.random.seed(42)
        proxy_data = np.random.normal(0, 1, 100)
        outcome_data = np.random.normal(0, 1, 100)

        bridge_test = self.validator.run_bridge_assumption_test(
            "test_assumption_invalid", proxy_data, outcome_data
        )

        assert isinstance(bridge_test, CausalBridgeTest)
        assert bridge_test.passed is False

    def test_bridge_assumption_test_dimension_mismatch(self):
        """Test bridge assumption test with mismatched dimensions"""
        proxy_data = np.random.normal(0, 1, 50)
        outcome_data = np.random.normal(0, 1, 100)

        with pytest.raises(ValueError, match="same length"):
            self.validator.run_bridge_assumption_test(
                "test_mismatch", proxy_data, outcome_data
            )

    def test_adversarial_audit_downgrading(self):
        """Test adversarial audit shows proper downgrading behavior"""
        audit_results = self.validator.audit_system_with_adversarial_cases(n_trials=20)

        assert "n_trials" in audit_results
        assert audit_results["n_trials"] == 20
        assert "downgrade_rate" in audit_results
        assert "average_risk" in audit_results
        assert "bridge_failure_rate" in audit_results

        # System should downgrade under adversarial conditions
        assert audit_results["downgrade_rate"] > 0.3
        assert audit_results["average_risk"] > 0.1

    def test_crc_risk_bound_calculation(self):
        """Test CRC risk bound calculation"""
        scores = np.array([0.8, 0.7, 0.9, 0.6, 0.85])

        risk_bound, coverage = self.validator._calculate_crc_risk_bound(scores)

        assert 0.0 <= risk_bound <= 1.0
        assert 0.0 <= coverage <= 1.0
        assert coverage == 1.0 - risk_bound or abs(coverage - (1.0 - risk_bound)) < 0.1

    def test_crc_risk_bound_empty_scores(self):
        """Test CRC risk bound with empty scores"""
        risk_bound, coverage = self.validator._calculate_crc_risk_bound(np.array([]))

        assert risk_bound == 1.0
        assert coverage == 0.0

    def test_horn_clauses_retrieval(self):
        """Test Horn clause retrieval for different points"""
        p1_clauses = self.validator._get_horn_clauses_for_point("P1")
        p2_clauses = self.validator._get_horn_clauses_for_point("P2")
        unknown_clauses = self.validator._get_horn_clauses_for_point("P999")

        assert isinstance(p1_clauses, list)
        assert isinstance(p2_clauses, list)
        assert isinstance(unknown_clauses, list)
        assert len(unknown_clauses) == 0

        if len(p1_clauses) > 0:
            clause = p1_clauses[0]
            assert "condition" in clause
            assert "dimensions" in clause

    def test_mandatory_indicators_retrieval(self):
        """Test mandatory indicator retrieval"""
        p1_indicators = self.validator._get_mandatory_indicators("P1")
        unknown_indicators = self.validator._get_mandatory_indicators("P999")

        assert isinstance(p1_indicators, list)
        assert isinstance(unknown_indicators, list)
        assert len(unknown_indicators) == 0

    def test_proof_object_operations(self):
        """Test ProofObject operations"""
        proof = ProofObject()

        # Test adding passed rules
        proof.add_passed_rule("rule_1")
        proof.add_passed_rule("rule_2")

        assert len(proof.passed_rules) == 2
        assert "rule_1" in proof.passed_rules
        assert "rule_2" in proof.passed_rules

        # Test adding counterexamples
        proof.add_counterexample("failed_rule", {"context": "test"})

        assert len(proof.counterexamples) == 1
        assert proof.counterexamples[0]["rule"] == "failed_rule"
        assert "timestamp" in proof.counterexamples[0]

    def test_causal_bridge_test_validity(self):
        """Test CausalBridgeTest validity checking"""
        # Valid bridge test
        valid_test = CausalBridgeTest(
            assumption_name="valid", test_statistic=3.5, p_value=0.001, passed=True
        )

        assert valid_test.is_valid(alpha=0.05) is True

        # Invalid bridge test (high p-value)
        invalid_test = CausalBridgeTest(
            assumption_name="invalid", test_statistic=0.5, p_value=0.8, passed=False
        )

        assert invalid_test.is_valid(alpha=0.05) is False

    def test_validator_parameters(self):
        """Test validator parameter settings"""
        custom_validator = ConstraintValidator(alpha=0.05, delta=0.01)

        assert custom_validator.alpha == 0.05
        assert custom_validator.delta == 0.01
        assert len(custom_validator._risk_history) == 0
        assert len(custom_validator._bridge_tests) == 0

    def test_integration_full_workflow(self):
        """Test full constraint validation workflow"""
        # Step 1: Validate dimensions
        passed_dim, proof = self.validator.validate_dimension_requirements(
            "integration_test", self.good_evidence
        )

        # Step 2: Check conjunctive conditions
        scores = {"diagnostic_quality": 0.8, "participation_score": 0.7}
        passed_conj, explanation = self.validator.check_conjunctive_conditions(
            "P1", scores
        )

        # Step 3: Verify mandatory indicators
        all_present, risk_bound = self.validator.verify_mandatory_indicators(
            "P1", self.good_evidence
        )

        # Step 4: Calculate satisfiability
        satisfiability_result = self.validator.calculate_satisfiability(
            self.sample_scores
        )

        # Step 5: Run bridge assumption test
        np.random.seed(42)
        proxy = np.random.normal(0, 1, 50)
        outcome = 0.6 * proxy + np.random.normal(0, 0.4, 50)
        bridge_test = self.validator.run_bridge_assumption_test(
            "integration", proxy, outcome
        )

        # Verify all steps completed successfully
        assert isinstance(passed_dim, bool)
        assert isinstance(proof, ProofObject)
        assert isinstance(passed_conj, bool)
        assert isinstance(explanation, dict)
        assert isinstance(all_present, bool)
        assert isinstance(risk_bound, float)
        assert isinstance(satisfiability_result, dict)
        assert isinstance(bridge_test, CausalBridgeTest)

        # Check satisfiability result structure
        assert "satisfiability" in satisfiability_result
        assert "rcps_alpha_summary" in satisfiability_result
        assert satisfiability_result["satisfiability"] in [
            SatisfiabilityLevel.SATISFIED,
            SatisfiabilityLevel.PARTIAL,
            SatisfiabilityLevel.UNSATISFIED,
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
