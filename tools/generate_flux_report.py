#!/usr/bin/env python3
"""
Generate Deterministic Flux Report

This CLI utility generates a granular, stage-organized explanation of the
canonical deterministic pipeline ("flux") and writes it to DETERMINISTIC_FLUX.md
by default, or to a custom path if provided.

Usage:
  python -m tools.generate_flux_report [OUTPUT_PATH] [--no-values]

- OUTPUT_PATH: Optional path for the markdown report (default: DETERMINISTIC_FLUX.md)
- --no-values: Exclude value-chain contribution metrics from the report
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comprehensive_pipeline_orchestrator import (
    write_deterministic_flux_report,
    generate_deterministic_flux_markdown,
)


def main(argv: list[str]) -> int:
    out_path = "DETERMINISTIC_FLUX.md"
    include_values = True

    # Simple arg parsing
    args = [a for a in argv if a]
    for a in list(args):
        if a == "--no-values":
            include_values = False
            args.remove(a)
    if args:
        out_path = args[0]

    # Write report
    path = write_deterministic_flux_report(out_path, include_value_chain=include_values)
    print(f"Deterministic flux report written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
