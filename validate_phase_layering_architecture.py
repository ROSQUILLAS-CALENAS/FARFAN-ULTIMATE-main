#!/usr/bin/env python3
"""
Phase Layering Architecture Validation Script

This script provides a comprehensive validation suite for the canonical phase
sequence architecture enforcement system.
"""

import subprocess
import sys
from pathlib import Path
import json
import time
from typing import Dict, Any


def run_command(command: list, description: str) -> Dict[str, Any]:
    """Run a command and return structured results."""
    print(f"🔍 {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            cwd=Path.cwd()
        )
        
        execution_time = time.time() - start_time
        
        return {
            "description": description,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": execution_time
        }
        
    except FileNotFoundError as e:
        return {
            "description": description,
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command not found: {e}",
            "execution_time": 0
        }


def validate_import_linter() -> Dict[str, Any]:
    """Validate using import-linter contracts."""
    return run_command(
        ["lint-imports"], 
        "Import-linter contract validation"
    )


def validate_fitness_functions() -> Dict[str, Any]:
    """Validate using architecture fitness functions."""
    return run_command(
        ["python", "architecture_fitness_functions.py", "--fail-on-violations", "--run-import-linter"],
        "Architecture fitness functions validation"
    )


def validate_fitness_functions_json() -> Dict[str, Any]:
    """Get JSON results from fitness functions."""
    return run_command(
        ["python", "architecture_fitness_functions.py", "--json", "--run-import-linter"],
        "Architecture fitness functions JSON output"
    )


def check_dependencies() -> Dict[str, Any]:
    """Check if required dependencies are installed."""
    results = []
    
    # Check import-linter
    lint_result = run_command(
        ["lint-imports", "--version"],
        "Check import-linter installation"
    )
    results.append(lint_result)
    
    # Check Python version
    python_result = run_command(
        ["python", "--version"],
        "Check Python version"
    )
    results.append(python_result)
    
    return {
        "description": "Dependency checks",
        "results": results,
        "success": all(r["success"] for r in results)
    }


def print_results_summary(results: Dict[str, Any]):
    """Print a formatted summary of all validation results."""
    print("\n" + "=" * 80)
    print("PHASE LAYERING ARCHITECTURE VALIDATION SUMMARY")
    print("=" * 80)
    
    total_checks = 0
    passed_checks = 0
    
    for check_name, result in results.items():
        if check_name == "dependency_check":
            continue
            
        total_checks += 1
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"  {result['description']}: {status}")
        
        if result["success"]:
            passed_checks += 1
        else:
            if result.get("stderr"):
                print(f"    Error: {result['stderr'][:200]}...")
    
    print(f"\nOVERALL RESULT: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 ALL ARCHITECTURE VALIDATIONS PASSED")
        print("The canonical phase sequence I→X→K→A→L→R→O→G→T→S is properly enforced.")
    else:
        print("⚠️  ARCHITECTURE VIOLATIONS DETECTED")
        print("Please review the validation reports and fix any backward dependencies.")
    
    print("=" * 80)


def main():
    """Main validation orchestrator."""
    print("🏗️  Phase Layering Architecture Validation Suite")
    print("Canonical Sequence: I→X→K→A→L→R→O→G→T→S")
    print("-" * 60)
    
    # Create reports directory
    reports_dir = Path("validation_reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Run all validation checks
    results = {}
    
    # 1. Check dependencies
    results["dependency_check"] = check_dependencies()
    
    # 2. Run import-linter validation
    results["import_linter"] = validate_import_linter()
    
    # 3. Run fitness functions validation
    results["fitness_functions"] = validate_fitness_functions()
    
    # 4. Get JSON results for detailed analysis
    results["fitness_json"] = validate_fitness_functions_json()
    
    # Save detailed results
    with open(reports_dir / "phase_layering_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_results_summary(results)
    
    # Generate combined report
    with open(reports_dir / "phase_layering_validation_report.txt", "w") as f:
        f.write("PHASE LAYERING ARCHITECTURE VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Canonical Phase Sequence: I→X→K→A→L→R→O→G→T→S\n\n")
        
        for check_name, result in results.items():
            if check_name == "dependency_check":
                continue
                
            status = "PASSED" if result["success"] else "FAILED"
            f.write(f"{result['description']}: {status}\n")
            
            if not result["success"] and result.get("stderr"):
                f.write(f"  Error: {result['stderr']}\n")
            
            f.write(f"  Execution time: {result.get('execution_time', 0):.2f}s\n\n")
    
    # Determine exit code
    validation_passed = (
        results.get("import_linter", {}).get("success", False) and
        results.get("fitness_functions", {}).get("success", False)
    )
    
    if not validation_passed:
        print("\n❌ Validation failed - exiting with error code")
        sys.exit(1)
    else:
        print("\n✅ All validations passed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()