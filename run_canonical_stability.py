#!/usr/bin/env python3
"""
Canonical Flow Stability Tests (dependency-free runner)

Purpose: Validate "no surprises" properties of the canonical PDF text flow.
- Verifies failure semantics for missing files (artifact written with status="failed").
- Verifies success path schema and artifact creation for an existing PDF (if available and PyPDF2 installed).
- Verifies idempotency/determinism: re-running produces identical artifact JSON.
- Verifies batch consistency on first N PDFs.

Usage:
  python run_canonical_stability.py [-n MAX_FILES]

Exit codes:
  0 - All checks passed (or gracefully skipped when environment lacks inputs/deps)
  1 - One or more checks failed

Notes:
- This runner uses pdf_text_reader.process (text-only, no OCR) and does not rely on pytest.
- It only inspects canonical_flow/<stem>.json artifacts for determinism. It intentionally
  ignores files like canonical_flow/execution_summary.json which contain timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project-local imports (minimal canonical component)
try:
    from pdf_text_reader import process as pdf_text_process
except Exception as e:
    pdf_text_process = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parent
PLANES_INPUT = PROJECT_ROOT / "planes_input"
CANONICAL_DIR = PROJECT_ROOT / "canonical_flow"


@dataclass
class TestResult:
    name: str
    success: bool
    message: str = ""


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_missing_file_semantics() -> TestResult:
    name = "missing_file_failure_semantics"
    if pdf_text_process is None:
        return TestResult(name, True, f"SKIP: pdf_text_reader import failed: {_IMPORT_ERROR}")

    missing = PLANES_INPUT / "__definitely_missing__.pdf"
    try:
        res = pdf_text_process({"pdf_path": str(missing)})
        artifact = CANONICAL_DIR / f"{missing.stem}.json"
        if not artifact.exists():
            return TestResult(name, False, f"Expected failure artifact not written: {artifact}")
        if res.get("status") != "failed":
            return TestResult(name, False, f"Expected status='failed', got: {res.get('status')}")
        err = res.get("error", "")
        if "File not found" not in err:
            return TestResult(name, False, f"Error does not contain 'File not found': {err}")
        return TestResult(name, True, f"OK: failure artifact present and error message correct")
    except Exception as e:
        return TestResult(name, False, f"Unexpected exception: {e}")


def _first_existing_pdf() -> Optional[Path]:
    if not PLANES_INPUT.exists():
        return None
    for p in sorted(PLANES_INPUT.glob("*.pdf")):
        if p.is_file():
            return p
    return None


def check_success_and_idempotency(pdf_path: Path) -> TestResult:
    name = f"success_and_idempotency:{pdf_path.name}"
    if pdf_text_process is None:
        return TestResult(name, True, f"SKIP: pdf_text_reader import failed: {_IMPORT_ERROR}")

    try:
        # First run
        res1 = pdf_text_process({"pdf_path": str(pdf_path)})
        artifact = CANONICAL_DIR / f"{pdf_path.stem}.json"
        if res1.get("status") not in {"success", "failed"}:
            return TestResult(name, False, f"Invalid status: {res1.get('status')}")
        if not artifact.exists():
            return TestResult(name, False, f"Artifact not written: {artifact}")

        # Basic schema check
        data1 = _load_json(artifact)
        for key in ("file", "text", "pages", "status"):
            if key not in data1:
                return TestResult(name, False, f"Missing key '{key}' in artifact")
        if not isinstance(data1.get("pages"), int):
            return TestResult(name, False, "'pages' must be integer")
        if data1.get("status") == "success" and data1.get("pages", 0) < 0:
            return TestResult(name, False, "'pages' invalid when success")

        # Second run
        res2 = pdf_text_process({"pdf_path": str(pdf_path)})
        data2 = _load_json(artifact)

        # Determinism: JSON equality and byte-wise hash equality
        if data1 != data2:
            return TestResult(name, False, "Artifact JSON changed across runs (non-deterministic)")
        h1 = _hash_file(artifact)
        h2 = _hash_file(artifact)
        if h1 != h2:
            return TestResult(name, False, "Artifact hash changed across immediate re-read (unexpected)")

        return TestResult(name, True, "OK: schema valid and artifact is idempotent")
    except Exception as e:
        return TestResult(name, False, f"Unexpected exception: {e}")


def check_batch_consistency(max_files: int) -> TestResult:
    name = f"batch_consistency_first_{max_files}"
    if pdf_text_process is None:
        return TestResult(name, True, f"SKIP: pdf_text_reader import failed: {_IMPORT_ERROR}")
    if not PLANES_INPUT.exists():
        return TestResult(name, True, "SKIP: planes_input does not exist")
    pdfs = sorted(PLANES_INPUT.glob("*.pdf"))[:max_files]
    if not pdfs:
        return TestResult(name, True, "SKIP: no PDFs found in planes_input")

    try:
        snapshots: List[Tuple[Path, Dict[str, Any], str]] = []
        # First pass
        for p in pdfs:
            pdf_text_process({"pdf_path": str(p)})
            artifact = CANONICAL_DIR / f"{p.stem}.json"
            if not artifact.exists():
                return TestResult(name, False, f"Artifact missing after first pass: {artifact}")
            data = _load_json(artifact)
            h = _hash_file(artifact)
            snapshots.append((artifact, data, h))
        # Second pass
        for (artifact, data_first, h_first) in snapshots:
            stem = artifact.stem
            src = PLANES_INPUT / f"{stem}.pdf"
            if src.exists():
                pdf_text_process({"pdf_path": str(src)})
            if not artifact.exists():
                return TestResult(name, False, f"Artifact disappeared on second pass: {artifact}")
            data_second = _load_json(artifact)
            h_second = _hash_file(artifact)
            if data_first != data_second:
                return TestResult(name, False, f"Artifact JSON changed for {artifact.name}")
            if h_first != h_second:
                return TestResult(name, False, f"Artifact hash changed for {artifact.name}")
        return TestResult(name, True, "OK: batch artifacts stable and idempotent")
    except Exception as e:
        return TestResult(name, False, f"Unexpected exception: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Canonical Flow Stability Test Runner")
    parser.add_argument("-n", "--max-files", type=int, default=3, help="Maximum PDFs to include in batch test")
    args = parser.parse_args(argv)

    results: List[TestResult] = []

    # 1) Missing-file semantics must be stable
    results.append(check_missing_file_semantics())

    # 2) Success + idempotency on the first available PDF (if any and deps present)
    candidate = _first_existing_pdf()
    if candidate is not None:
        results.append(check_success_and_idempotency(candidate))
    else:
        results.append(TestResult("success_and_idempotency", True, "SKIP: no PDFs found in planes_input"))

    # 3) Batch-level consistency
    results.append(check_batch_consistency(args.max_files))

    # Report
    any_fail = False
    print("\nCanonical Flow Stability Report")
    print("=" * 34)
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"- {r.name}: {status} - {r.message}")
        if not r.success:
            any_fail = True

    if any_fail:
        print("\nResult: FAIL - One or more checks failed.")
        return 1
    print("\nResult: PASS - All checks passed (or were skipped appropriately).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
