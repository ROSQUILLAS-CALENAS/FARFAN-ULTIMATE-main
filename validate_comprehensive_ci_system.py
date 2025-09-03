#!/usr/bin/env python3
"""
Comprehensive Validation and Continuous Integration System Validator

This script executes the comprehensive validation test suite and generates detailed reports
for continuous integration pipelines. It validates the DecalogoQuestionRegistry, evidence
artifacts, score bounds, aggregation consistency, and determinism verification.

Usage:
    python validate_comprehensive_ci_system.py [--output-dir DIR] [--verbose]
"""

import argparse
import json
import logging
import os
import sys
import time
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List  # Module not found  # Module not found  # Module not found

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import test modules
try:
# # #     from tests.test_comprehensive_validation_ci import ComprehensiveValidationCI  # Module not found  # Module not found  # Module not found
    import unittest
    TEST_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing test modules: {e}")
    TEST_IMPORTS_AVAILABLE = False


def setup_logging(verbose: bool = False):
    """Configure logging for the validation system."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('comprehensive_validation.log')
        ]
    )


def run_validation_tests(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run the comprehensive validation test suite.
    
    Args:
        output_dir: Directory for output files
        verbose: Enable verbose logging
        
    Returns:
        Validation results summary
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive validation test suite")
    
    if not TEST_IMPORTS_AVAILABLE:
        return {
            "status": "error",
            "message": "Test modules not available",
            "tests_run": 0,
            "failures": 1,
            "errors": 1
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(ComprehensiveValidationCI)
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, stream=sys.stdout)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Generate results summary
    summary = {
        "status": "success" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "execution_time_seconds": round(end_time - start_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "details": {
            "failure_details": [
                {
                    "test": str(test),
                    "traceback": traceback
                } for test, traceback in result.failures
            ],
            "error_details": [
                {
                    "test": str(test),
                    "traceback": traceback
                } for test, traceback in result.errors
            ]
        }
    }
    
    # Write summary to file
    summary_path = os.path.join(output_dir, "validation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Validation summary written to {summary_path}")
    
    return summary


def generate_ci_artifacts(output_dir: str, summary: Dict[str, Any]) -> List[str]:
    """
    Generate CI/CD artifacts for pipeline integration.
    
    Args:
        output_dir: Directory for output files
        summary: Validation results summary
        
    Returns:
        List of generated artifact paths
    """
    logger = logging.getLogger(__name__)
    artifacts = []
    
    try:
        # Generate JUnit XML report for CI integration
        junit_path = os.path.join(output_dir, "junit-results.xml")
        junit_xml = generate_junit_xml(summary)
        
        with open(junit_path, 'w', encoding='utf-8') as f:
            f.write(junit_xml)
        
        artifacts.append(junit_path)
        logger.info(f"JUnit XML report written to {junit_path}")
        
        # Generate status badge data
        badge_path = os.path.join(output_dir, "status_badge.json")
        badge_data = generate_status_badge_data(summary)
        
        with open(badge_path, 'w', encoding='utf-8') as f:
            json.dump(badge_data, f, indent=2, ensure_ascii=False)
        
        artifacts.append(badge_path)
        logger.info(f"Status badge data written to {badge_path}")
        
        # Generate metrics for monitoring
        metrics_path = os.path.join(output_dir, "validation_metrics.json")
        metrics_data = generate_metrics_data(summary)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        artifacts.append(metrics_path)
        logger.info(f"Validation metrics written to {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error generating CI artifacts: {e}")
    
    return artifacts


def generate_junit_xml(summary: Dict[str, Any]) -> str:
    """Generate JUnit XML format for CI/CD integration."""
    xml_template = '''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="ComprehensiveValidationCI" 
           tests="{tests_run}" 
           failures="{failures}" 
           errors="{errors}" 
           time="{execution_time}"
           timestamp="{timestamp}">
{test_cases}
</testsuite>'''
    
    test_cases = ""
    
    # Add failure test cases
    for failure in summary.get("details", {}).get("failure_details", []):
        test_cases += f'''
    <testcase classname="ComprehensiveValidationCI" name="{failure['test']}" time="0">
        <failure message="Test failed">{failure['traceback']}</failure>
    </testcase>'''
    
    # Add error test cases
    for error in summary.get("details", {}).get("error_details", []):
        test_cases += f'''
    <testcase classname="ComprehensiveValidationCI" name="{error['test']}" time="0">
        <error message="Test error">{error['traceback']}</error>
    </testcase>'''
    
    # Add successful test cases (approximate)
    successful_tests = summary["tests_run"] - summary["failures"] - summary["errors"]
    for i in range(successful_tests):
        test_cases += f'''
    <testcase classname="ComprehensiveValidationCI" name="test_success_{i}" time="0"/>'''
    
    return xml_template.format(
        tests_run=summary["tests_run"],
        failures=summary["failures"],
        errors=summary["errors"],
        execution_time=summary.get("execution_time_seconds", 0),
        timestamp=summary.get("timestamp", "unknown"),
        test_cases=test_cases
    )


def generate_status_badge_data(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Generate status badge data for README display."""
    if summary["status"] == "success":
        color = "brightgreen"
        message = f"passing ({summary['tests_run']}/{summary['tests_run']})"
    else:
        color = "red"
        failed_count = summary["failures"] + summary["errors"]
        message = f"failing ({summary['tests_run'] - failed_count}/{summary['tests_run']})"
    
    return {
        "schemaVersion": 1,
        "label": "validation",
        "message": message,
        "color": color,
        "namedLogo": "github-actions"
    }


def generate_metrics_data(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Generate metrics data for monitoring systems."""
    return {
        "validation_metrics": {
            "test_success_rate": {
                "value": (summary["tests_run"] - summary["failures"] - summary["errors"]) / summary["tests_run"] if summary["tests_run"] > 0 else 0,
                "unit": "percent"
            },
            "execution_time": {
                "value": summary.get("execution_time_seconds", 0),
                "unit": "seconds"
            },
            "total_tests": {
                "value": summary["tests_run"],
                "unit": "count"
            },
            "failed_tests": {
                "value": summary["failures"] + summary["errors"],
                "unit": "count"
            }
        },
        "timestamp": summary.get("timestamp", "unknown"),
        "status": summary["status"]
    }


def main():
    """Main entry point for the validation system."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive validation and CI system"
    )
    parser.add_argument(
        "--output-dir", 
        default="validation_reports",
        help="Output directory for reports and artifacts"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Run validation tests
        logger.info("Starting comprehensive validation system")
        summary = run_validation_tests(args.output_dir, args.verbose)
        
        # Generate CI artifacts
        artifacts = generate_ci_artifacts(args.output_dir, summary)
        
        # Print summary
        print(f"\n{'='*50}")
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Status: {summary['status'].upper()}")
        print(f"Tests Run: {summary['tests_run']}")
        print(f"Failures: {summary['failures']}")
        print(f"Errors: {summary['errors']}")
        print(f"Execution Time: {summary['execution_time_seconds']}s")
        print(f"Output Directory: {args.output_dir}")
        print(f"Artifacts Generated: {len(artifacts)}")
        
        # Exit with appropriate code for CI systems
        if summary["status"] == "success":
            logger.info("All validation tests passed successfully")
            sys.exit(0)
        else:
            logger.error("Some validation tests failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Critical error in validation system: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()