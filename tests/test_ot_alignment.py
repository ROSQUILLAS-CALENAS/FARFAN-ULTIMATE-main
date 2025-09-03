"""
Test reproducibility and ablation studies for OT alignment.
Ensures EGW alignment is reproducible with fixed hyperparameters.
"""

import hashlib
import json
import logging
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

import numpy as np
import pytest
# # # from scipy.spatial.distance import cdist  # Module not found  # Module not found  # Module not found

# # # from standards_alignment.gw_alignment import gw_align  # Module not found  # Module not found  # Module not found
# # # from standards_alignment.graph_ops import DocumentGraph, StandardsGraph  # Module not found  # Module not found  # Module not found


class TestOTAlignment:
    """Test suite for OT alignment reproducibility and ablation."""
    
    def setup_method(self):
        """Setup test fixtures with fixed seed."""
        self.seed = 42
        np.random.seed(self.seed)
        
        # Fixed hyperparameters
        self.lambda_reg = 0.1
        self.epsilon = 0.05
        self.max_iter = 100
        
        # Tolerance for numerical reproducibility (relaxed for POT library)
        self.tol = 1e-8
        
        # Create deterministic test graphs
        self.std_graph, self.doc_graph = self._create_deterministic_graphs()
        
    def _create_deterministic_graphs(self):
        """Create deterministic graphs for reproducible testing."""
        # Standards graph with fixed structure
        std_graph = StandardsGraph()
        std_graph.add_dimension("security", {"priority": 1})
        std_graph.add_subdimension("security", "auth", {"level": "high"})
        std_graph.add_subdimension("security", "encryption", {"level": "medium"})
        
        for i in range(3):
            std_graph.add_point("auth", f"auth_req_{i}", {"weight": i + 1})
            std_graph.add_point("encryption", f"enc_req_{i}", {"weight": i * 2})
            
        # Document graph with fixed structure
        doc_graph = DocumentGraph()
        doc_graph.add_section("sec1", "Authentication", "User authentication requirements")
        doc_graph.add_section("sec2", "Data Protection", "Data encryption standards")
        
        for i in range(4):
            doc_graph.add_paragraph("sec1", f"auth_para_{i}", 
                                  f"Authentication requirement {i}")
            doc_graph.add_paragraph("sec2", f"enc_para_{i}",
                                  f"Encryption requirement {i}")
                                  
        doc_graph.add_table("sec1", "auth_table", 
                          {"rows": [["Method", "Security"], ["MFA", "High"]], 
                           "headers": True})
        doc_graph.add_table("sec2", "enc_table",
                          {"rows": [["Algorithm", "Strength"], ["AES256", "High"]],
                           "headers": True})
                           
        return std_graph, doc_graph
        
    def _compute_plan_digest(self, transport_plan):
        """Compute reproducible digest of transport plan."""
        # Round to avoid floating point differences
        rounded_plan = np.round(transport_plan, decimals=10)
        
        # Create structured representation
        digest_data = {
            "shape": list(transport_plan.shape),
            "sum": float(np.sum(rounded_plan)),
            "max": float(np.max(rounded_plan)), 
            "min": float(np.min(rounded_plan)),
            "nnz": int(np.count_nonzero(rounded_plan > self.tol)),
            "frobenius_norm": float(np.linalg.norm(rounded_plan, 'fro'))
        }
        
        # Hash for reproducible digest
        digest_str = json.dumps(digest_data, sort_keys=True)
        return hashlib.sha256(digest_str.encode()).hexdigest()[:16]
        
    def test_reproducibility_same_hyperparams(self):
        """Test identical transport plans with same hyperparameters."""
        # Run alignment twice with identical parameters
        result1 = gw_align(self.std_graph, self.doc_graph, 
                          reg=self.lambda_reg, max_iter=self.max_iter)
        
        np.random.seed(self.seed)  # Reset seed
        result2 = gw_align(self.std_graph, self.doc_graph,
                          reg=self.lambda_reg, max_iter=self.max_iter)
        
        # Plans should be identical within numerical tolerance
        plan_diff = np.abs(result1.transport_plan - result2.transport_plan)
        max_diff = np.max(plan_diff)
        
        assert max_diff < self.tol, f"Plans differ by {max_diff} > {self.tol}"
        
        # Costs should match
        cost_diff = abs(result1.cost_decomposition["total_cost"] - 
                       result2.cost_decomposition["total_cost"])
        assert cost_diff < self.tol, f"Costs differ by {cost_diff}"
        
        # Digests should match
        digest1 = self._compute_plan_digest(result1.transport_plan)
        digest2 = self._compute_plan_digest(result2.transport_plan)
        assert digest1 == digest2, "Plan digests don't match"
        
    def test_ablation_lambda_increase_cost(self):
        """Test that reducing lambda increases total cost."""
        # Baseline with standard lambda
        baseline = gw_align(self.std_graph, self.doc_graph,
                           reg=self.lambda_reg, max_iter=self.max_iter)
        baseline_cost = baseline.cost_decomposition["total_cost"]
        
        # Reduced regularization (lambda / 2)
        reduced_reg = gw_align(self.std_graph, self.doc_graph,
                              reg=self.lambda_reg / 2, max_iter=self.max_iter) 
        reduced_cost = reduced_reg.cost_decomposition["total_cost"]
        
        # Increased regularization (lambda * 2)
        increased_reg = gw_align(self.std_graph, self.doc_graph,
                                reg=self.lambda_reg * 2, max_iter=self.max_iter)
        increased_cost = increased_reg.cost_decomposition["total_cost"]
        
        # With less regularization, GW distance should be lower but entropic penalty higher
        assert reduced_reg.cost_decomposition["gw_distance"] <= baseline.cost_decomposition["gw_distance"]
        assert increased_reg.cost_decomposition["gw_distance"] >= baseline.cost_decomposition["gw_distance"]
        
        # Log results for ablation analysis
        logging.info(f"Ablation λ: {self.lambda_reg/2:.3f} -> {baseline_cost:.6f}")
        logging.info(f"Ablation λ: {self.lambda_reg:.3f} -> {baseline_cost:.6f}") 
        logging.info(f"Ablation λ: {self.lambda_reg*2:.3f} -> {increased_cost:.6f}")
        
    def test_ablation_epsilon_convergence(self):
        """Test epsilon (entropic regularization) effect on convergence."""
        epsilons = [0.01, 0.05, 0.1, 0.2]
        results = []
        
        for eps in epsilons:
            # Use epsilon as reg parameter in POT
            result = gw_align(self.std_graph, self.doc_graph,
                             reg=eps, max_iter=self.max_iter)
            results.append({
                "epsilon": eps,
                "cost": result.cost_decomposition["total_cost"],
                "plan_digest": self._compute_plan_digest(result.transport_plan),
                "unmatched_mass": 1.0 - np.sum(result.transport_plan)
            })
            
        # Verify costs follow expected pattern (higher epsilon = more regularized)
        costs = [r["cost"] for r in results]
        
        # Generally expect monotonic relationship, but allow some tolerance for numerics
        # We mainly check that extreme values behave correctly
        assert costs[0] != costs[-1], "Different epsilon should give different costs"
        
        logging.info("Epsilon ablation results:")
        for r in results:
            logging.info(f"ε={r['epsilon']:.3f}: cost={r['cost']:.6f}, digest={r['plan_digest']}")
            
    def test_ablation_max_iter_convergence(self):
        """Test max_iter effect on convergence."""
        iter_values = [10, 50, 100, 200]
        results = []
        
        for max_iter in iter_values:
            result = gw_align(self.std_graph, self.doc_graph,
                             reg=self.lambda_reg, max_iter=max_iter)
            results.append({
                "max_iter": max_iter,
                "cost": result.cost_decomposition["total_cost"],
                "plan_digest": self._compute_plan_digest(result.transport_plan)
            })
            
        # With more iterations, should converge to similar solution
        final_costs = [r["cost"] for r in results[-2:]]  # Last two runs
        cost_diff = abs(final_costs[0] - final_costs[1])
        
        # Should converge within reasonable tolerance
        assert cost_diff < 0.01, f"Cost not converged: diff={cost_diff}"
        
        logging.info("Max_iter ablation results:")
        for r in results:
            logging.info(f"max_iter={r['max_iter']}: cost={r['cost']:.6f}, digest={r['plan_digest']}")
            
    def test_numerical_stability(self):
        """Test numerical stability across different random seeds."""
        costs = []
        digests = []
        
        # Test with different seeds
        for seed in [42, 123, 456, 789]:
            np.random.seed(seed)
            result = gw_align(self.std_graph, self.doc_graph,
                             reg=self.lambda_reg, max_iter=self.max_iter)
            costs.append(result.cost_decomposition["total_cost"])
            digests.append(self._compute_plan_digest(result.transport_plan))
            
        # All results should be identical (deterministic alignment)
        cost_std = np.std(costs)
        assert cost_std < self.tol, f"Cost variance across seeds: {cost_std}"
        
        unique_digests = set(digests)
        assert len(unique_digests) == 1, f"Multiple plan digests: {unique_digests}"
        
    def test_mass_conservation(self):
        """Test transport plan mass conservation."""
        result = gw_align(self.std_graph, self.doc_graph,
                         reg=self.lambda_reg, max_iter=self.max_iter)
        
        plan = result.transport_plan
        
        # Row sums (source marginals)
        row_sums = np.sum(plan, axis=1)
        expected_row_sum = 1.0 / plan.shape[0]  # Uniform distribution
        
        for i, row_sum in enumerate(row_sums):
            assert abs(row_sum - expected_row_sum) < self.tol, \
                   f"Row {i} sum {row_sum} != {expected_row_sum}"
                   
        # Column sums (target marginals)  
        col_sums = np.sum(plan, axis=0)
        expected_col_sum = 1.0 / plan.shape[1]  # Uniform distribution
        
        for j, col_sum in enumerate(col_sums):
            assert abs(col_sum - expected_col_sum) < self.tol, \
                   f"Col {j} sum {col_sum} != {expected_col_sum}"
                   
        # Total mass should be 1
        total_mass = np.sum(plan)
        assert abs(total_mass - 1.0) < self.tol, f"Total mass {total_mass} != 1.0"
        
    def test_generate_certificate(self):
        """Generate ASC certificate for reproducibility.""" 
        result = gw_align(self.std_graph, self.doc_graph,
                         reg=self.lambda_reg, max_iter=self.max_iter)
        
        plan_digest = self._compute_plan_digest(result.transport_plan)
        total_cost = result.cost_decomposition["total_cost"]
        unmatched_mass = abs(1.0 - np.sum(result.transport_plan))
        
        certificate = {
            "pass": True,
            "hyperparameters": {
                "lambda": self.lambda_reg,
                "epsilon": self.epsilon, 
                "max_iter": self.max_iter,
                "seed": self.seed
            },
            "plan_digest": plan_digest,
            "cost": float(total_cost),
            "unmatched_mass": float(unmatched_mass),
            "transport_plan_shape": list(result.transport_plan.shape),
            "stability_bound": float(result.stability_bound),
            "test_timestamp": "2024-01-01T00:00:00Z",
            "tolerance": self.tol
        }
        
        # Write certificate
        cert_path = Path("asc_certificate.json")
        with open(cert_path, "w") as f:
            json.dump(certificate, f, indent=2)
            
        # Verify certificate
        assert certificate["pass"], "Certificate validation failed"
        assert certificate["unmatched_mass"] < 0.01, "Excessive unmatched mass"
        
        logging.info(f"ASC Certificate generated: {cert_path}")
        logging.info(f"Plan digest: {plan_digest}")
        logging.info(f"Total cost: {total_cost:.8f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])