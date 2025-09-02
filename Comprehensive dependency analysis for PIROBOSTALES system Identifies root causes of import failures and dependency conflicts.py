#!/usr/bin/env python3
"""
Comprehensive dependency analysis for PIROBOSTALES system
Identifies root causes of import failures and dependency conflicts
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path


def check_python_environment():
    """Analyze Python environment basics"""
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": sys.platform,
        "architecture": sys.maxsize > 2 ** 32 and "64-bit" or "32-bit"
    }


def check_core_dependencies():
    """Test core mathematical and ML dependencies"""
    dependencies = {
        "numpy": {"required": True, "min_version": "1.24.0"},
        "scipy": {"required": True, "min_version": "1.11.0"},
        "scikit-learn": {"required": True, "min_version": "1.3.0"},
        "torch": {"required": False, "min_version": "2.0.0"},
        "transformers": {"required": False, "min_version": "4.35.0"},
        "sentence-transformers": {"required": False, "min_version": "2.2.2"},
        "faiss-cpu": {"required": False, "min_version": "1.7.4"},
        "spacy": {"required": False, "min_version": "3.7.0"}
    }

    results = {}

    for package, config in dependencies.items():
        try:
            module = importlib.import_module(package.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")

            results[package] = {
                "status": "installed",
                "version": version,
                "required": config["required"],
                "min_version": config["min_version"],
                "location": getattr(module, "__file__", "unknown")
            }

        except ImportError as e:
            results[package] = {
                "status": "missing",
                "error": str(e),
                "required": config["required"],
                "min_version": config["min_version"]
            }
        except Exception as e:
            results[package] = {
                "status": "error",
                "error": str(e),
                "required": config["required"]
            }

    return results


def check_system_packages():
    """Check system-level dependencies"""
    system_deps = ["tesseract", "git", "curl"]
    results = {}

    for dep in system_deps:
        try:
            result = subprocess.run(
                ["which", dep],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                results[dep] = {
                    "status": "installed",
                    "location": result.stdout.strip()
                }
            else:
                results[dep] = {"status": "missing"}
        except Exception as e:
            results[dep] = {"status": "error", "error": str(e)}

    return results


def analyze_import_failures():
    """Test specific import patterns that cause startup failures"""
    critical_imports = [
        "canonical_flow.mathematical_enhancers.mathematical_pipeline_coordinator",
        "egw_query_expansion.core.import_safety",
        "egw_query_expansion.mathematical_foundations",
        "comprehensive_pipeline_orchestrator"
    ]

    results = {}

    for module_name in critical_imports:
        try:
            importlib.import_module(module_name)
            results[module_name] = {"status": "success"}
        except ImportError as e:
            results[module_name] = {
                "status": "import_error",
                "error": str(e),
                "error_type": type(e).__name__
            }
        except Exception as e:
            results[module_name] = {
                "status": "other_error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    return results


def check_requirements_files():
    """Analyze requirements.txt files for inconsistencies"""
    req_files = [
        "requirements.txt",
        "requirements-minimal.txt",
        "requirements_minimal.txt",
        "requirements-core.txt"
    ]

    results = {}

    for req_file in req_files:
        path = Path(req_file)
        if path.exists():
            try:
                with open(path) as f:
                    content = f.read()
                    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]

                results[req_file] = {
                    "exists": True,
                    "packages": lines,
                    "count": len(lines)
                }
            except Exception as e:
                results[req_file] = {
                    "exists": True,
                    "error": str(e)
                }
        else:
            results[req_file] = {"exists": False}

    return results


def generate_recommendations(dependency_results, import_results):
    """Generate specific recommendations based on analysis"""
    recommendations = []

    # Check for missing critical dependencies
    missing_critical = [
        pkg for pkg, info in dependency_results.items()
        if info.get("required", False) and info.get("status") == "missing"
    ]

    if missing_critical:
        recommendations.append({
            "priority": "CRITICAL",
            "issue": f"Missing required dependencies: {', '.join(missing_critical)}",
            "solution": f"pip install {' '.join(missing_critical)}"
        })

    # Check for import failures
    failed_imports = [
        module for module, info in import_results.items()
        if info.get("status") != "success"
    ]

    if failed_imports:
        recommendations.append({
            "priority": "HIGH",
            "issue": "Critical modules failing to import",
            "solution": "Check module dependencies and fix import chains"
        })

    # Check for version conflicts
    installed_deps = {
        pkg: info for pkg, info in dependency_results.items()
        if info.get("status") == "installed"
    }

    if len(installed_deps) < 3:
        recommendations.append({
            "priority": "HIGH",
            "issue": "Insufficient ML stack installed",
            "solution": "Install full requirements: pip install -r requirements.txt"
        })

    return recommendations


def main():
    print("PIROBOSTALES Dependency Analysis")
    print("=" * 50)

    analysis = {
        "timestamp": str(subprocess.check_output(["date"], text=True).strip()),
        "environment": check_python_environment(),
        "dependencies": check_core_dependencies(),
        "system_packages": check_system_packages(),
        "import_analysis": analyze_import_failures(),
        "requirements_files": check_requirements_files()
    }

    # Generate recommendations
    recommendations = generate_recommendations(
        analysis["dependencies"],
        analysis["import_analysis"]
    )
    analysis["recommendations"] = recommendations

    # Print summary
    print("\nCRITICAL DEPENDENCIES:")
    for pkg, info in analysis["dependencies"].items():
        if info.get("required", False):
            status = info.get("status", "unknown")
            print(f"  {pkg}: {status}")
            if status == "missing":
                print(f"    ERROR: {info.get('error', 'Unknown error')}")

    print(f"\nIMPORT FAILURES:")
    for module, info in analysis["import_analysis"].items():
        if info.get("status") != "success":
            print(f"  {module}: {info.get('status')}")
            print(f"    ERROR: {info.get('error', 'Unknown error')}")

    print(f"\nRECOMMENDATIONS:")
    for rec in recommendations:
        print(f"  [{rec['priority']}] {rec['issue']}")
        print(f"    Solution: {rec['solution']}")

    # Save detailed report
    with open("dependency_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nDetailed report saved to: dependency_analysis.json")

    # Exit with appropriate code
    critical_failures = any(
        info.get("required", False) and info.get("status") == "missing"
        for info in analysis["dependencies"].values()
    )

    return 1 if critical_failures else 0


if __name__ == "__main__":
    sys.exit(main())