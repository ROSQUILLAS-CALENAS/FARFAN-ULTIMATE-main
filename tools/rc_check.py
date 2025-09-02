#!/usr/bin/env python3
"""
Routing Contract CLI Checker
Command-line tool for verifying routing contract compliance

Usage:
    python tools/rc_check.py --question "test query" --steps "step1,step2,step3"
    python tools/rc_check.py --config config.json --verify-route route.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import blake3
    hash_func = blake3.blake3
    HASH_NAME = "blake3"
except ImportError:
    import hashlib
    hash_func = hashlib.sha256
    HASH_NAME = "sha256"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from egw_query_expansion.core.deterministic_router import (
    DeterministicRouter,
    ImmutableConfig,
    RoutingContext,
)
from egw_query_expansion.core.immutable_context import QuestionContext


class RoutingContractCLI:
    """CLI interface for routing contract verification"""
    
    def __init__(self, config: ImmutableConfig = None):
        self.config = config or ImmutableConfig()
        self.router = DeterministicRouter(self.config)
    
    def compute_route_hash(self, route: List[str]) -> str:
        """Compute cryptographic hash of route"""
        route_bytes = json.dumps(route, sort_keys=True).encode('utf-8')
        return hash_func(route_bytes).hexdigest()
    
    def create_routing_context(self, question: str, mode: str = "hybrid", corpus_size: int = 1000) -> RoutingContext:
        """Create routing context from question"""
        return RoutingContext.from_query(
            query=question,
            embedding=[0.1, 0.2, 0.3, 0.4],  # Dummy embedding for CLI
            corpus_size=corpus_size,
            mode=mode
        )
    
    def check_route(self, question: str, steps: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Check routing for given question and steps"""
        # Create context
        context = self.create_routing_context(question)
        
        # Create steps format
        step_objects = [
            {"step_id": step, "content_hash": hash_func(step.encode()).hexdigest()}
            for step in steps
        ]
        
        # Execute routing
        route = self.router.routing_fn(context, step_objects)
        route_hash = self.compute_route_hash(route)
        
        # Compute inputs hash
        inputs_data = {
            "question": question,
            "steps": step_objects,
            "context_hash": context.query_hash,
            "config_hash": self.config.config_hash
        }
        inputs_hash = hash_func(
            json.dumps(inputs_data, sort_keys=True).encode()
        ).hexdigest()
        
        result = {
            "question": question,
            "context_hash": context.query_hash,
            "route": route,
            "route_hash": route_hash,
            "inputs_hash": inputs_hash,
            "step_count": len(steps),
            "config_hash": self.config.config_hash,
            "hash_algorithm": HASH_NAME
        }
        
        if verbose:
            result["step_details"] = step_objects
            result["routing_config"] = {
                "tolerance": self.config.projection_tolerance,
                "max_iterations": self.config.max_iterations,
                "convergence_threshold": self.config.convergence_threshold,
                "tie_breaker": self.config.lexicographic_tie_breaker
            }
        
        return result
    
    def verify_determinism(self, question: str, steps: List[str], iterations: int = 10) -> Dict[str, Any]:
        """Verify deterministic routing across multiple iterations"""
        context = self.create_routing_context(question)
        step_objects = [
            {"step_id": step, "content_hash": hash_func(step.encode()).hexdigest()}
            for step in steps
        ]
        
        routes = []
        route_hashes = []
        
        for i in range(iterations):
            route = self.router.routing_fn(context, step_objects)
            route_hash = self.compute_route_hash(route)
            routes.append(route)
            route_hashes.append(route_hash)
        
        # Check consistency
        unique_routes = set(tuple(r) for r in routes)
        unique_hashes = set(route_hashes)
        
        return {
            "deterministic": len(unique_routes) == 1,
            "iterations": iterations,
            "unique_routes": len(unique_routes),
            "unique_hashes": len(unique_hashes),
            "route": routes[0] if routes else [],
            "route_hash": route_hashes[0] if route_hashes else "",
            "all_hashes_identical": len(unique_hashes) == 1
        }
    
    def print_route_info(self, result: Dict[str, Any]):
        """Print formatted route information"""
        print("=" * 60)
        print("ROUTING CONTRACT VERIFICATION")
        print("=" * 60)
        print(f"Question: {result['question']}")
        print(f"Context Hash: {result['context_hash'][:16]}...")
        print(f"Config Hash: {result['config_hash'][:16]}...")
        print()
        print("ROUTE:")
        for i, step in enumerate(result['route']):
            print(f"  {i+1}. {step}")
        print()
        print(f"Route Hash ({HASH_NAME}): {result['route_hash']}")
        print(f"Inputs Hash ({HASH_NAME}): {result['inputs_hash']}")
        print(f"Step Count: {result['step_count']}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Routing Contract CLI Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/rc_check.py --question "What is machine learning?" --steps "sparse,dense,colbert"
  python tools/rc_check.py -q "test" -s "a,b,c" --verify-determinism --iterations 20
  python tools/rc_check.py -q "example" -s "x,y" --verbose --output result.json
        """
    )
    
    parser.add_argument(
        "-q", "--question",
        required=True,
        help="Input question for routing"
    )
    
    parser.add_argument(
        "-s", "--steps",
        required=True,
        help="Comma-separated list of steps to route"
    )
    
    parser.add_argument(
        "--mode",
        choices=["sparse", "dense", "colbert", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)"
    )
    
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=1000,
        help="Corpus size for routing context (default: 1000)"
    )
    
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Verify deterministic routing across multiple iterations"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for determinism verification (default: 10)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for JSON results"
    )
    
    parser.add_argument(
        "--config",
        help="JSON config file for router settings"
    )
    
    args = parser.parse_args()
    
    # Parse steps
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    
    if not steps:
        print("Error: No valid steps provided", file=sys.stderr)
        sys.exit(1)
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                config = ImmutableConfig(**config_data)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Create CLI checker
    cli = RoutingContractCLI(config)
    
    try:
        if args.verify_determinism:
            # Verify determinism
            result = cli.verify_determinism(args.question, steps, args.iterations)
            
            print("DETERMINISM VERIFICATION")
            print("=" * 40)
            print(f"Iterations: {result['iterations']}")
            print(f"Deterministic: {result['deterministic']}")
            print(f"Unique Routes: {result['unique_routes']}")
            print(f"All Hashes Identical: {result['all_hashes_identical']}")
            
            if result['deterministic']:
                print("✅ ROUTING CONTRACT VERIFIED - DETERMINISTIC")
            else:
                print("❌ ROUTING CONTRACT VIOLATED - NON-DETERMINISTIC")
                sys.exit(1)
            
            if args.verbose and result['route']:
                print(f"\nFinal Route: {' → '.join(result['route'])}")
                print(f"Route Hash: {result['route_hash']}")
        
        else:
            # Single route check
            result = cli.check_route(args.question, steps, args.verbose)
            cli.print_route_info(result)
        
        # Output to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()