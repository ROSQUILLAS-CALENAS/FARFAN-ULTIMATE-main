#!/usr/bin/env python3
"""
Local development script to run the full strict static analysis suite.
Provides the same checks as CI but for local development workflow.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n🔍 {description}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path.cwd()
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED")
            print(f"Output:\n{output}")
            
        return success, output
    except FileNotFoundError:
        print(f"❌ {description} - COMMAND NOT FOUND")
        return False, f"Command not found: {' '.join(cmd)}"


def main() -> int:
    """Run the complete strict static analysis suite."""
    print("🚀 Running Strict Static Analysis Suite")
    print("=" * 50)
    
    # Ensure we're in the right directory
    if not Path("egw_query_expansion").exists():
        print("❌ Error: Must be run from repository root (egw_query_expansion/ not found)")
        return 1
    
    # Create reports directory
    reports_dir = Path("analysis_reports")
    reports_dir.mkdir(exist_ok=True)
    
    checks = []
    all_passed = True
    
    # 1. MyPy type checking
    success, output = run_command([
        "mypy", "--strict", "--config-file", "mypy.ini", 
        "--show-error-codes", "egw_query_expansion/"
    ], "MyPy Type Checking")
    checks.append(("MyPy", success, output))
    all_passed = all_passed and success
    
    # 2. Ruff linting
    success, output = run_command([
        "ruff", "check", "--config", "pyproject.toml", "egw_query_expansion/"
    ], "Ruff Linting")
    checks.append(("Ruff", success, output))
    all_passed = all_passed and success
    
    # 3. Star imports check
    py_files = [str(f) for f in Path("egw_query_expansion").rglob("*.py")]
    success, output = run_command([
        "python3", "scripts/check_star_imports.py"
    ] + py_files, "Star Import Detection")
    checks.append(("Star Imports", success, output))
    all_passed = all_passed and success
    
    # 4. Circular imports check
    success, output = run_command([
        "python3", "scripts/check_circular_imports.py"
    ] + py_files, "Circular Import Detection")
    checks.append(("Circular Imports", success, output))
    all_passed = all_passed and success
    
    # 5. Type checking imports validation
    success, output = run_command([
        "python3", "scripts/validate_type_imports.py"
    ] + py_files, "Type Import Validation")
    checks.append(("Type Imports", success, output))
    all_passed = all_passed and success
    
    # 6. Import organization
    success, output = run_command([
        "ruff", "check", "--select", "I", "--config", "pyproject.toml", "egw_query_expansion/"
    ], "Import Organization")
    checks.append(("Import Order", success, output))
    all_passed = all_passed and success
    
    # 7. Black formatting check
    success, output = run_command([
        "black", "--check", "--diff", "egw_query_expansion/"
    ], "Black Formatting")
    checks.append(("Black Format", success, output))
    all_passed = all_passed and success
    
    # Generate summary report
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    for check_name, passed, _ in checks:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check_name:20s} : {status}")
    
    total_checks = len(checks)
    passed_checks = sum(1 for _, passed, _ in checks if passed)
    
    print(f"\nResults: {passed_checks}/{total_checks} checks passed")
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Code is ready for commit.")
        return 0
    else:
        print(f"\n🚨 {total_checks - passed_checks} CHECK(S) FAILED!")
        print("Please fix the issues above before committing.")
        
        # Save detailed reports
        for check_name, passed, output in checks:
            if not passed and output:
                report_file = reports_dir / f"{check_name.lower().replace(' ', '_')}_report.txt"
                report_file.write_text(output, encoding='utf-8')
                print(f"Detailed report saved: {report_file}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())