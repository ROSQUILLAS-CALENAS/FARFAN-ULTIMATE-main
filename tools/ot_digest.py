#!/usr/bin/env python3
"""
OT Digest Tool - Print hyperparameters and results for EGW alignment.
Provides {λ, ε, max_iter, cost, unmatched_mass, plan_digest} for reproducibility.
"""

import argparse
import hashlib
import json
import logging
import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

import numpy as np
import networkx as nx

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# # # from standards_alignment.gw_alignment import gw_align  # Module not found  # Module not found  # Module not found
# # # from standards_alignment.graph_ops import DocumentGraph, StandardsGraph  # Module not found  # Module not found  # Module not found


class OTDigester:
    """Generate reproducible digests for OT alignment results."""
    
    def __init__(self, tolerance=1e-10):
        """
        Initialize digester.
        
        Args:
            tolerance: Numerical tolerance for reproducibility
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
    
    def compute_plan_digest(self, transport_plan):
        """
        Compute reproducible digest of transport plan.
        
        Args:
            transport_plan: Transport plan matrix
            
        Returns:
            Hexadecimal digest string
        """
        # Round for numerical stability
        rounded_plan = np.round(transport_plan, decimals=10)
        
        # Structured representation for hashing
        digest_data = {
            "shape": list(transport_plan.shape),
            "sum": float(np.sum(rounded_plan)),
            "max": float(np.max(rounded_plan)),
            "min": float(np.min(rounded_plan)), 
            "nnz": int(np.count_nonzero(rounded_plan > self.tolerance)),
            "frobenius_norm": float(np.linalg.norm(rounded_plan, 'fro')),
            "trace": float(np.trace(rounded_plan @ rounded_plan.T))
        }
        
        # Deterministic JSON serialization
        digest_str = json.dumps(digest_data, sort_keys=True)
        return hashlib.sha256(digest_str.encode()).hexdigest()[:16]
    
    def compute_unmatched_mass(self, transport_plan):
        """
# # #         Compute unmatched mass (deviation from perfect transport).  # Module not found  # Module not found  # Module not found
        
        Args:
            transport_plan: Transport plan matrix
            
        Returns:
            Unmatched mass value
        """
        return abs(1.0 - np.sum(transport_plan))
    
    def create_demo_graphs(self):
        """Create demonstration graphs for testing."""
        # Standards graph
        std_graph = StandardsGraph()
        std_graph.add_dimension("security", {"priority": 1})
        std_graph.add_subdimension("security", "authentication", {"level": "high"})
        std_graph.add_subdimension("security", "encryption", {"level": "medium"})
        
        for i in range(3):
            std_graph.add_point("authentication", f"auth_req_{i}", {"weight": i + 1})
            std_graph.add_point("encryption", f"enc_req_{i}", {"weight": i * 2})
        
        # Document graph
        doc_graph = DocumentGraph()
        doc_graph.add_section("sec1", "Authentication Requirements", 
                             "This section covers user authentication.")
        doc_graph.add_section("sec2", "Data Protection", 
                             "This section covers data encryption.")
        
        for i in range(4):
            doc_graph.add_paragraph("sec1", f"auth_para_{i}",
                                  f"Authentication requirement {i}: Strong passwords required.")
            doc_graph.add_paragraph("sec2", f"enc_para_{i}",
                                  f"Encryption requirement {i}: Use AES-256 encryption.")
        
        # Add tables for structure
        doc_graph.add_table("sec1", "auth_methods",
                          {"rows": [["Method", "Security Level"], 
                                   ["Password", "Medium"],
                                   ["MFA", "High"],
                                   ["Biometric", "Very High"]], 
                           "headers": True})
        
        doc_graph.add_table("sec2", "enc_algorithms", 
                          {"rows": [["Algorithm", "Key Length"],
                                   ["AES", "256"],
                                   ["RSA", "2048"]], 
                           "headers": True})
        
        return std_graph, doc_graph
    
    def run_alignment(self, lambda_reg=0.1, epsilon=0.05, max_iter=100, seed=42):
        """
        Run EGW alignment with specified parameters.
        
        Args:
            lambda_reg: Regularization parameter λ  
            epsilon: Entropic regularization ε
            max_iter: Maximum iterations
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with alignment results and digest
        """
        # Set reproducible seed
        np.random.seed(seed)
        
        # Create graphs
        std_graph, doc_graph = self.create_demo_graphs()
        
        self.logger.info(f"Running EGW alignment:")
        self.logger.info(f"  λ (lambda_reg): {lambda_reg}")
        self.logger.info(f"  ε (epsilon): {epsilon}")
        self.logger.info(f"  max_iter: {max_iter}")
        self.logger.info(f"  seed: {seed}")
        
        # Run alignment (using reg parameter as epsilon in POT)
        result = gw_align(std_graph, doc_graph, reg=epsilon, max_iter=max_iter)
        
        # Compute derived metrics
        plan_digest = self.compute_plan_digest(result.transport_plan)
        unmatched_mass = self.compute_unmatched_mass(result.transport_plan)
        total_cost = result.cost_decomposition["total_cost"]
        
        # Package results
        digest_result = {
            "hyperparameters": {
                "lambda": lambda_reg,
                "epsilon": epsilon,
                "max_iter": max_iter,
                "seed": seed
            },
            "results": {
                "cost": float(total_cost),
                "unmatched_mass": float(unmatched_mass),
                "plan_digest": plan_digest,
                "stability_bound": float(result.stability_bound),
                "regularization": float(result.regularization)
            },
            "graph_info": {
                "standards_nodes": len(std_graph.G.nodes()),
                "document_nodes": len(doc_graph.G.nodes()),
                "transport_shape": list(result.transport_plan.shape)
            },
            "cost_decomposition": {
                k: float(v) for k, v in result.cost_decomposition.items()
            }
        }
        
        return digest_result
    
    def print_digest(self, digest_result):
        """Print formatted digest output."""
        print("=" * 60)
        print("OT ALIGNMENT DIGEST")
        print("=" * 60)
        
        # Hyperparameters
        hyp = digest_result["hyperparameters"]
        print(f"Hyperparameters:")
        print(f"  λ (lambda):     {hyp['lambda']:.6f}")
        print(f"  ε (epsilon):    {hyp['epsilon']:.6f}")
        print(f"  max_iter:       {hyp['max_iter']}")
        print(f"  seed:           {hyp['seed']}")
        print()
        
        # Main results
        res = digest_result["results"]
        print(f"Results:")
        print(f"  cost:           {res['cost']:.8f}")
        print(f"  unmatched_mass: {res['unmatched_mass']:.8f}")
        print(f"  plan_digest:    {res['plan_digest']}")
        print(f"  stability_bound: {res['stability_bound']:.6f}")
        print()
        
        # Cost breakdown
        costs = digest_result["cost_decomposition"]
        print(f"Cost Decomposition:")
        for key, value in costs.items():
            print(f"  {key:15s}: {value:.8f}")
        print()
        
        # Graph info
        info = digest_result["graph_info"]
        print(f"Graph Structure:")
        print(f"  standards_nodes: {info['standards_nodes']}")
        print(f"  document_nodes:  {info['document_nodes']}")
        print(f"  transport_shape: {info['transport_shape']}")
        
        print("=" * 60)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate OT alignment digest for reproducibility testing"
    )
    
    parser.add_argument("--lambda", "-l", type=float, default=0.1,
                       help="Lambda regularization parameter (default: 0.1)")
    parser.add_argument("--epsilon", "-e", type=float, default=0.05,
                       help="Epsilon entropic regularization (default: 0.05)")
    parser.add_argument("--max-iter", "-m", type=int, default=100,
                       help="Maximum iterations (default: 100)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--json", "-j", action="store_true",
                       help="Output as JSON instead of formatted text")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file path (default: stdout)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Create digester
    digester = OTDigester()
    
    try:
        # Run alignment and generate digest
        digest_result = digester.run_alignment(
            lambda_reg=getattr(args, 'lambda'),
            epsilon=args.epsilon,
            max_iter=args.max_iter,
            seed=args.seed
        )
        
        # Output results
        if args.json:
            output_text = json.dumps(digest_result, indent=2)
        else:
            # Capture formatted output
            import io
# # #             from contextlib import redirect_stdout  # Module not found  # Module not found  # Module not found
            
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                digester.print_digest(digest_result)
            output_text = output_buffer.getvalue()
        
        # Write to file or stdout
        if args.output:
            Path(args.output).write_text(output_text)
            print(f"Digest written to: {args.output}", file=sys.stderr)
        else:
            print(output_text)
            
    except Exception as e:
        logging.error(f"Failed to generate digest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()