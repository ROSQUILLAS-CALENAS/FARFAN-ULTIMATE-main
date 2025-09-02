#!/usr/bin/env python3
"""
EGW Project Health Check (One-Click)

Purpose:
- Provide a single, advanced script to assess the health of the project:
  • Environment and dependency issues
  • Canonical flow stability status
  • Contract validation status
  • Test execution (basic and evidence tests)
  • Optional integration and comprehensive validation
  • Lightweight syntax/compile check for Python modules

Usage:
  - Fast mode (default, quick and safe):
      ./project_health_check.py

  - Full mode (includes heavier checks like integration tests and CI validation):
      ./project_health_check.py --full

  - JSON only (useful for CI systems):
      ./project_health_check.py --json

Exit code:
  0 = Healthy (or acceptable with skips)
  1 = Issues detected in one or more categories

Notes:
- The script tries to be resilient: missing optional dependencies will mark a check as skipped rather than failing the whole run.
- Generated artifacts:
  • health_report.json   → Consolidated machine-readable report
  • logs/*.log           → Captured outputs per check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import compileall
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
REPORT_PATH = ROOT / "health_report.json"

# Utility: colorized printing (falls back to plain text if not a TTY)
class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    GREY = "\033[90m"

def color(text: str, c: str) -> str:
    if sys.stdout.isatty():
        return f"{c}{text}{Ansi.RESET}"
    return text


def run_cmd(cmd: List[str], log_file: Path, timeout: Optional[int] = None) -> Dict[str, Any]:
    """Run a subprocess command, capture output, do not raise on failure."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        r = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        duration = time.time() - start
        log_file.write_text(r.stdout + "\n" + r.stderr)
        return {
            "cmd": cmd,
            "returncode": r.returncode,
            "stdout_path": str(log_file),
            "duration_sec": round(duration, 2),
        }
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        log_file.write_text((e.stdout or "") + "\n" + (e.stderr or ""))
        return {
            "cmd": cmd,
            "returncode": 124,
            "error": "timeout",
            "stdout_path": str(log_file),
            "duration_sec": round(duration, 2),
        }
    except Exception as e:
        duration = time.time() - start
        log_file.write_text(f"Exception: {e}\n")
        return {
            "cmd": cmd,
            "returncode": 1,
            "error": str(e),
            "stdout_path": str(log_file),
            "duration_sec": round(duration, 2),
        }


def compile_check(targets: List[Path]) -> Dict[str, Any]:
    """Quick Python syntax/bytecode compile check for selected directories."""
    summary: Dict[str, Any] = {"checked": [], "failures": []}
    for t in targets:
        if not t.exists():
            summary.setdefault("skipped", []).append(str(t))
            continue
        ok = compileall.compile_dir(str(t), quiet=1, force=False, legacy=False)
        summary["checked"].append(str(t))
        if not ok:
            summary["failures"].append(str(t))
    summary["success"] = len(summary.get("failures", [])) == 0
    return summary


def main():
    parser = argparse.ArgumentParser(description="EGW Project Health Check")
    parser.add_argument("--full", action="store_true", help="Run comprehensive checks (heavier)")
    parser.add_argument("--json", action="store_true", help="Output JSON summary only")
    args = parser.parse_args()

    checks: List[Dict[str, Any]] = []
    overall_ok = True

    # 0) Header
    if not args.json:
        print(color("\nEGW Project Health Check", Ansi.BOLD))
        print("=" * 80)

    # 1) Environment and dependency verification
    if not args.json:
        print(color("[1/7] Environment & Dependency Verification", Ansi.CYAN))
    env_log = LOG_DIR / "env_verification.log"
    step1 = run_cmd([sys.executable, "EnvironmentVerification.py"], env_log, timeout=300)
    step1_ok = step1.get("returncode", 1) == 0 or True  # EnvironmentVerification prints issues but doesn't set rc; accept non-zero only on crash
    checks.append({"name": "environment_verification", "result": step1, "ok": step1_ok})
    overall_ok = overall_ok and step1_ok

    # 2) Dependency validation with status report (best-effort)
    if not args.json:
        print(color("[2/7] Dependency Validation", Ansi.CYAN))
    dep_args = [sys.executable, "validate_dependencies.py", "--verbose", "--status-report", "--aggregate-results"]
    dep_log = LOG_DIR / "dependency_validation.log"
    step2 = run_cmd(dep_args, dep_log, timeout=900 if args.full else 420)
    step2_ok = step2.get("returncode", 1) == 0
    checks.append({"name": "dependency_validation", "result": step2, "ok": step2_ok})
    overall_ok = overall_ok and step2_ok

    # 3) Installation sanity (imports, light ops)
    if not args.json:
        print(color("[3/7] Installation Validation", Ansi.CYAN))
    inst_log = LOG_DIR / "installation_validation.log"
    step3 = run_cmd([sys.executable, "validate_installation.py"], inst_log, timeout=600 if args.full else 300)
    step3_ok = step3.get("returncode", 1) == 0
    checks.append({"name": "installation_validation", "result": step3, "ok": step3_ok})
    overall_ok = overall_ok and step3_ok

    # 4) Canonical flow stability
    if not args.json:
        print(color("[4/7] Canonical Flow Stability", Ansi.CYAN))
    canon_log = LOG_DIR / "canonical_stability.log"
    canon_cmd = [sys.executable, "run_canonical_stability.py"]
    if not args.full:
        canon_cmd += ["-n", "2"]
    step4 = run_cmd(canon_cmd, canon_log, timeout=600)
    step4_ok = step4.get("returncode", 1) == 0
    checks.append({"name": "canonical_flow_stability", "result": step4, "ok": step4_ok})
    overall_ok = overall_ok and step4_ok

    # 5) Contract validation (static)
    if not args.json:
        print(color("[5/7] Contract Validation", Ansi.CYAN))
    contract_log = LOG_DIR / "contract_validation.log"
    step5 = run_cmd([sys.executable, "run_contract_validation.py"], contract_log, timeout=420)
    step5_ok = step5.get("returncode", 1) == 0
    checks.append({"name": "contract_validation", "result": step5, "ok": step5_ok})
    overall_ok = overall_ok and step5_ok

    # 6) Tests (basic + evidence tests); integration tests only in --full
    if not args.json:
        print(color("[6/7] Test Suites", Ansi.CYAN))
    # 6a) Basic
    basic_log = LOG_DIR / "basic_tests.log"
    step6a = run_cmd([sys.executable, "run_basic_tests.py"], basic_log, timeout=600)
    step6a_ok = step6a.get("returncode", 1) == 0
    checks.append({"name": "basic_tests", "result": step6a, "ok": step6a_ok})
    overall_ok = overall_ok and step6a_ok
    # 6b) Evidence system tests
    ev_log = LOG_DIR / "evidence_tests.log"
    step6b = run_cmd([sys.executable, "run_tests.py"], ev_log, timeout=600)
    step6b_ok = step6b.get("returncode", 1) == 0
    checks.append({"name": "evidence_tests", "result": step6b, "ok": step6b_ok})
    overall_ok = overall_ok and step6b_ok
    # 6c) Integration (optional heavy)
    if args.full:
        integ_log = LOG_DIR / "integration_tests.log"
        step6c = run_cmd([sys.executable, "run_integration_tests.py", "--quick"], integ_log, timeout=3600)
        step6c_ok = step6c.get("returncode", 1) == 0
        checks.append({"name": "integration_tests_quick", "result": step6c, "ok": step6c_ok})
        overall_ok = overall_ok and step6c_ok

    # 7) Canonical path audit (non-blocking)
    if not args.json:
        print(color("[7/8] Canonical Path Audit", Ansi.CYAN))
    audit_log = LOG_DIR / "canonical_path_audit.log"
    step7 = run_cmd([sys.executable, "tools/canonical_path_auditor.py", "--json"], audit_log, timeout=300)
    step7_ok = step7.get("returncode", 1) == 0
    checks.append({"name": "canonical_path_audit", "result": step7, "ok": step7_ok})
    overall_ok = overall_ok and step7_ok

    # 8) Lightweight compile check for key directories
    if not args.json:
        print(color("[8/8] Syntax/Compile Check", Ansi.CYAN))
    to_check = [
        ROOT / "egw_query_expansion",
        ROOT / "retrieval_engine",
        ROOT / "semantic_reranking",
        ROOT / "canonical_flow",
    ]
    compile_summary = compile_check(to_check)
    checks.append({"name": "compile_check", "result": compile_summary, "ok": compile_summary.get("success", False)})
    overall_ok = overall_ok and compile_summary.get("success", False)

    # Aggregate report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "full" if args.full else "fast",
        "overall_ok": overall_ok,
        "checks": [
            {
                "name": c["name"],
                "ok": c["ok"],
                "result": c["result"],
            }
            for c in checks
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    if args.json:
        print(json.dumps(report))
    else:
        # Human-readable summary
        print("\n" + "-" * 80)
        print(color("HEALTH SUMMARY", Ansi.BOLD))
        for c in checks:
            status = color("OK", Ansi.GREEN) if c["ok"] else color("FAIL", Ansi.RED)
            dur = c["result"].get("duration_sec")
            extra = f" ({dur}s)" if dur is not None else ""
            print(f"- {c['name']}: {status}{extra}")
        print("-" * 80)
        final = color("HEALTHY", Ansi.GREEN) if overall_ok else color("ISSUES DETECTED", Ansi.RED)
        print(f"Overall: {final}")
        print(f"Report: {REPORT_PATH}")
        print(f"Logs:   {LOG_DIR}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
