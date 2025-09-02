#!/usr/bin/env python3
"""
PIC Probe Tool - Permutation Invariant Checking diagnostic utility.

Takes a multiset and displays a digest of the aggregated result, useful for debugging
and validating permutation invariant aggregation operations.

Usage:
    python pic_probe.py --multiset "[[1,2,3], [1,2], [3]]" --aggregation sum
    python pic_probe.py --file multiset_data.json --aggregation mean --verbose
"""

import argparse
import hashlib
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

try:
    from egw_query_expansion.core.permutation_invariant_processor import (
        AggregationType,
        DeterministicPooler,
    )
except ImportError:
    print(
        "Error: Cannot import egw_query_expansion modules. "
        "Please ensure package is installed."
    )
    sys.exit(1)


class MultisetDigest:
    """
    Generates comprehensive digests for multiset aggregations.
    """

    def __init__(self, aggregation_type: AggregationType = AggregationType.SUM):
        self.aggregation_type = aggregation_type
        self.pooler = DeterministicPooler.get_pooler(aggregation_type)

    def compute_digest(
        self,
        multiset_data: Union[List[List[float]], torch.Tensor],
        multiplicities: Optional[List[float]] = None,
        include_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive digest of multiset aggregation.

        Args:
            multiset_data: Input multiset as nested list or tensor
            multiplicities: Optional element multiplicities
            include_intermediate: Whether to include intermediate computation steps

        Returns:
            Dictionary containing digest information
        """
        # Convert input to tensor format
        if isinstance(multiset_data, list):
            # Convert nested list to tensor
            if isinstance(multiset_data[0], list):
                # Already in [set_size, feature_dim] format
                tensor_data = torch.tensor(
                    multiset_data, dtype=torch.float32
                ).unsqueeze(0)
            else:
                # Flat list, treat as single-element multiset
                tensor_data = (
                    torch.tensor(multiset_data, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
        else:
            tensor_data = multiset_data.clone()

        if len(tensor_data.shape) == 2:
            tensor_data = tensor_data.unsqueeze(0)  # Add batch dimension

        batch_size, set_size, feature_dim = tensor_data.shape

        # Process multiplicities
        mult_tensor = None
        if multiplicities is not None:
            mult_tensor = torch.tensor(multiplicities, dtype=torch.float32).unsqueeze(0)

        # Compute basic statistics
        element_norms = torch.norm(tensor_data, dim=-1)
        total_norm = torch.norm(tensor_data)

        # Element-wise transformation (ψ function)
        psi_x = torch.tanh(tensor_data) + 0.1 * torch.sin(tensor_data)

        # Apply multiplicities if provided
        if mult_tensor is not None:
            psi_x = psi_x * mult_tensor.unsqueeze(-1)

        # Aggregation step
        aggregated = self.pooler(psi_x, dim=1)

        # Final transformation (φ function)
        final_result = torch.sigmoid(aggregated) * aggregated

        # Generate content-based hash
        content_hash = self._generate_content_hash(tensor_data, mult_tensor)

        # Create digest
        digest = {
            "content_hash": content_hash,
            "aggregation_type": self.aggregation_type.value,
            "multiset_shape": {
                "batch_size": batch_size,
                "set_size": set_size,
                "feature_dim": feature_dim,
            },
            "statistics": {
                "total_elements": batch_size * set_size,
                "unique_elements": set_size,  # Simplified assumption
                "total_norm": total_norm.item(),
                "mean_element_norm": element_norms.mean().item(),
                "max_element_norm": element_norms.max().item(),
                "min_element_norm": element_norms.min().item(),
            },
            "aggregated_result": {
                "shape": list(final_result.shape),
                "values": final_result.squeeze(0).tolist(),
                "norm": torch.norm(final_result).item(),
                "mean": final_result.mean().item(),
                "std": final_result.std().item(),
            },
            "multiplicities": {
                "provided": multiplicities is not None,
                "values": multiplicities if multiplicities is not None else None,
                "sum": sum(multiplicities) if multiplicities is not None else set_size,
                "max": max(multiplicities) if multiplicities is not None else 1.0,
            },
        }

        # Add intermediate computations if requested
        if include_intermediate:
            digest["intermediate"] = {
                "psi_statistics": {
                    "norm": torch.norm(psi_x).item(),
                    "mean": psi_x.mean().item(),
                    "std": psi_x.std().item(),
                },
                "aggregated_statistics": {
                    "norm": torch.norm(aggregated).item(),
                    "mean": aggregated.mean().item(),
                    "std": aggregated.std().item(),
                },
                "raw_aggregation": aggregated.squeeze(0).tolist(),
            }

        return digest

    def _generate_content_hash(
        self, tensor_data: torch.Tensor, multiplicities: Optional[torch.Tensor] = None
    ) -> str:
        """Generate a hash based on multiset content (order-invariant)."""
        # Convert to canonical form for hashing
        batch_size, set_size, feature_dim = tensor_data.shape

        # Flatten and sort for canonical representation
        flattened = tensor_data.view(-1, feature_dim)

        # Create a deterministic hash based on sorted elements
        # Note: This is simplified - in practice you'd want more robust canonicalization
        sorted_tensor, _ = torch.sort(flattened.sum(dim=-1))

        # Include multiplicities in hash if provided
        hash_input = sorted_tensor.detach().numpy().tobytes()
        if multiplicities is not None:
            sorted_mult, _ = torch.sort(multiplicities.view(-1))
            hash_input += sorted_mult.detach().numpy().tobytes()

        return hashlib.sha256(hash_input).hexdigest()[:16]


class PICProbe:
    """Main probe class for multiset analysis."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("PICProbe")
        handler = logging.StreamHandler()

        if self.verbose:
            logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            logger.setLevel(logging.INFO)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(levelname)s: %(message)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def probe_multiset(
        self,
        multiset_data: Union[str, List, Dict],
        aggregation_type: AggregationType = AggregationType.SUM,
        multiplicities: Optional[List[float]] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Probe a multiset and generate comprehensive digest.

        Args:
            multiset_data: Multiset data (string, list, or loaded from file)
            aggregation_type: Type of aggregation to use
            multiplicities: Optional multiplicities
            output_file: Optional file to write results

        Returns:
            Complete digest dictionary
        """
        self.logger.info(f"Probing multiset with {aggregation_type.value} aggregation")

        # Parse input data
        if isinstance(multiset_data, str):
            try:
                parsed_data = json.loads(multiset_data)
            except json.JSONDecodeError:
                try:
                    # Try to evaluate as Python literal
                    import ast

                    parsed_data = ast.literal_eval(multiset_data)
                except (ValueError, SyntaxError):
                    raise ValueError(f"Cannot parse multiset data: {multiset_data}")
        else:
            parsed_data = multiset_data

        # Create digest generator
        digest_gen = MultisetDigest(aggregation_type)

        # Generate digest
        digest = digest_gen.compute_digest(
            parsed_data,
            multiplicities=multiplicities,
            include_intermediate=self.verbose,
        )

        # Add metadata
        digest["metadata"] = {
            "probe_version": "1.0.0",
            "aggregation_type": aggregation_type.value,
            "input_type": type(multiset_data).__name__,
            "processed_at": str(torch.utils.data.get_worker_info() or "main_thread"),
        }

        # Log key information
        self.logger.info(f"Content hash: {digest['content_hash']}")
        self.logger.info(f"Multiset shape: {digest['multiset_shape']}")
        self.logger.info(f"Result norm: {digest['aggregated_result']['norm']:.6f}")

        if self.verbose:
            self.logger.debug(
                f"Raw aggregated values: {digest['aggregated_result']['values']}"
            )
            if "intermediate" in digest:
                self.logger.debug(
                    f"Intermediate ψ norm: {digest['intermediate']['psi_statistics']['norm']:.6f}"
                )

        # Write to file if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(digest, f, indent=2)
            self.logger.info(f"Digest written to {output_file}")

        return digest

    def compare_digests(
        self, digest1: Dict[str, Any], digest2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two multiset digests for equivalence."""
        comparison = {
            "content_hashes_match": digest1["content_hash"] == digest2["content_hash"],
            "aggregation_types_match": digest1["aggregation_type"]
            == digest2["aggregation_type"],
            "shapes_match": digest1["multiset_shape"] == digest2["multiset_shape"],
            "results_match": np.allclose(
                digest1["aggregated_result"]["values"],
                digest2["aggregated_result"]["values"],
                atol=1e-6,
            ),
            "result_difference": np.max(
                np.abs(
                    np.array(digest1["aggregated_result"]["values"])
                    - np.array(digest2["aggregated_result"]["values"])
                )
            ),
        }

        comparison["overall_match"] = all(
            [comparison["content_hashes_match"], comparison["results_match"]]
        )

        return comparison


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="PIC Probe - Analyze multiset aggregations for permutation invariance"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--multiset",
        type=str,
        help="Multiset data as JSON string (e.g., '[[1,2], [3,4]]')",
    )
    input_group.add_argument(
        "--file", type=str, help="Load multiset data from JSON file"
    )

    # Configuration options
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["sum", "mean", "log_sum_exp", "max", "min"],
        default="sum",
        help="Aggregation type to use",
    )

    parser.add_argument(
        "--multiplicities",
        type=str,
        help="Element multiplicities as JSON list (e.g., '[1.0, 2.0, 1.5]')",
    )

    parser.add_argument("--output", type=str, help="Output file for digest results")

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with intermediate computations",
    )

    parser.add_argument("--compare", type=str, help="Compare with another digest file")

    args = parser.parse_args()

    # Map aggregation string to enum
    agg_type_map = {
        "sum": AggregationType.SUM,
        "mean": AggregationType.MEAN,
        "log_sum_exp": AggregationType.LOG_SUM_EXP,
        "max": AggregationType.MAX,
        "min": AggregationType.MIN,
    }

    agg_type = agg_type_map[args.aggregation]

    # Parse multiplicities if provided
    multiplicities = None
    if args.multiplicities:
        try:
            multiplicities = json.loads(args.multiplicities)
        except json.JSONDecodeError:
            print(f"Error: Cannot parse multiplicities: {args.multiplicities}")
            sys.exit(1)

    # Load input data
    if args.file:
        try:
            with open(args.file, "r") as f:
                multiset_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading file {args.file}: {e}")
            sys.exit(1)
    else:
        multiset_data = args.multiset

    # Create probe and analyze
    probe = PICProbe(verbose=args.verbose)

    try:
        digest = probe.probe_multiset(
            multiset_data,
            aggregation_type=agg_type,
            multiplicities=multiplicities,
            output_file=args.output,
        )

        # Pretty print digest
        print("\n" + "=" * 60)
        print("MULTISET DIGEST")
        print("=" * 60)
        print(f"Content Hash: {digest['content_hash']}")
        print(f"Aggregation: {digest['aggregation_type']}")
        print(f"Shape: {digest['multiset_shape']}")
        print(f"Total Norm: {digest['statistics']['total_norm']:.6f}")
        print(f"Result Norm: {digest['aggregated_result']['norm']:.6f}")
        print(f"Result Values: {digest['aggregated_result']['values']}")

        if multiplicities:
            print(f"Multiplicities Sum: {digest['multiplicities']['sum']}")

        # Comparison if requested
        if args.compare:
            try:
                with open(args.compare, "r") as f:
                    other_digest = json.load(f)

                comparison = probe.compare_digests(digest, other_digest)

                print("\n" + "=" * 60)
                print("COMPARISON RESULTS")
                print("=" * 60)
                print(f"Overall Match: {comparison['overall_match']}")
                print(f"Content Hashes Match: {comparison['content_hashes_match']}")
                print(f"Results Match: {comparison['results_match']}")
                print(f"Max Difference: {comparison['result_difference']:.8f}")

            except Exception as e:
                print(f"Error comparing with {args.compare}: {e}")

        print("=" * 60)

    except Exception as e:
        print(f"Error probing multiset: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
