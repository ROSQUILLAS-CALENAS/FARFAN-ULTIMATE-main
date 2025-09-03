#!/usr/bin/env python3
"""
Validate Dependencies Script

This script validates project dependencies by:
1. Importing all project modules to detect import failures
2. Checking for version conflicts using importlib.metadata
3. Validating that required classes and functions are accessible
4. Generating comprehensive library status reports with fallback tracking
5. Performing behavioral consistency validation across EGW components
6. Providing detailed error messages for any issues found

Usage:
    python validate_dependencies.py [--ci] [--verbose] [--status-report] [--aggregate-results]
    
Options:
    --ci                CI mode with minimal output (exit 1 on any failure)
    --verbose           Verbose mode with detailed dependency information
    --status-report     Generate comprehensive library status report
    --aggregate-results Aggregate results across all EGW components
"""

import argparse
import importlib
import importlib.metadata
import sys
import traceback
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Tuple, Any, Optional  # Module not found  # Module not found  # Module not found
import warnings

# Import the LibraryStatusReporter
try:
# # #     from canonical_flow.mathematical_enhancers.mathematical_compatibility_matrix import (  # Module not found  # Module not found  # Module not found
        LibraryStatusReporter, MathematicalCompatibilityMatrix
    )
    STATUS_REPORTER_AVAILABLE = True
except ImportError:
    STATUS_REPORTER_AVAILABLE = False

# Suppress warnings during validation unless in verbose mode
warnings.filterwarnings("ignore")

class DependencyValidator:
    """Main validator class for checking project dependencies and imports."""
    
    def __init__(self, verbose: bool = False, ci_mode: bool = False, 
                 enable_status_report: bool = False, aggregate_results: bool = False):
        self.verbose = verbose
        self.ci_mode = ci_mode
        self.enable_status_report = enable_status_report
        self.aggregate_results = aggregate_results
        self.errors = []
        self.warnings = []
        self.status_reporter = None
        self.aggregated_results = {}
        
# # #         # Expected core components from egw_query_expansion  # Module not found  # Module not found  # Module not found
        self.expected_components = {
            "egw_query_expansion": [
                "GromovWassersteinAligner",
                "QueryGenerator", 
                "HybridRetriever",
                "PatternMatcher",
                "DeterministicRouter",
                "ConformalRiskController",
                "PermutationInvariantProcessor",
                "MonotonicTaskSelector",
            ],
            "egw_query_expansion.core": [
                "QuestionContext",
                "ImmutableContextManager",
                "create_question_context",
                "linear_context_scope",
            ],
            "egw_query_expansion.mathematical_foundations": [
                "InformationTheory",
                "SemanticSimilarity",
                "BayesianInference",
                "compute_question_entropy_features",
            ]
        }
        
# # #         # Critical dependencies from requirements.txt  # Module not found  # Module not found  # Module not found
        self.critical_dependencies = {
            "torch": ">=2.0.0",
            "transformers": ">=4.35.0", 
            "faiss-cpu": ">=1.7.4",
            "sentence-transformers": ">=2.2.2",
            "numpy": ">=1.24.0",
            "scipy": ">=1.11.0",
            "POT": ">=0.9.1",
            "scikit-learn": ">=1.3.0",
            "datasets": ">=2.14.0",
            "pandas": ">=1.5.0",
            "beir": ">=2.0.0"
        }

    def log_error(self, message: str):
        """Log an error message."""
        self.errors.append(message)
        if not self.ci_mode:
            print(f"❌ ERROR: {message}")

    def log_warning(self, message: str):
        """Log a warning message."""
        self.warnings.append(message)
        if self.verbose and not self.ci_mode:
            print(f"⚠️  WARNING: {message}")

    def log_success(self, message: str):
        """Log a success message."""
        if self.verbose and not self.ci_mode:
            print(f"✅ SUCCESS: {message}")

    def log_info(self, message: str):
        """Log an info message."""
        if self.verbose and not self.ci_mode:
            print(f"ℹ️  INFO: {message}")

    def check_package_version(self, package_name: str, required_version: str) -> bool:
        """Check if a package meets version requirements."""
        try:
            installed_version = importlib.metadata.version(package_name)
            # Simple version comparison - assumes semantic versioning
            if required_version.startswith(">="):
                required = required_version[2:]
                # Basic version comparison (works for major.minor.patch)
                installed_parts = [int(x) for x in installed_version.split('.')]
                required_parts = [int(x) for x in required.split('.')]
                
                # Pad with zeros if needed
                max_len = max(len(installed_parts), len(required_parts))
                installed_parts.extend([0] * (max_len - len(installed_parts)))
                required_parts.extend([0] * (max_len - len(required_parts)))
                
                meets_requirement = installed_parts >= required_parts
                
                if meets_requirement:
                    self.log_success(f"{package_name} {installed_version} meets requirement {required_version}")
                else:
                    self.log_error(f"{package_name} {installed_version} does not meet requirement {required_version}")
                
                return meets_requirement
                
        except importlib.metadata.PackageNotFoundError:
            self.log_error(f"Package {package_name} is not installed")
            return False
        except Exception as e:
            self.log_warning(f"Could not check version for {package_name}: {e}")
            return True  # Don't fail on version check errors
            
        return True

    def validate_critical_dependencies(self) -> bool:
        """Validate that critical dependencies are installed with correct versions."""
        self.log_info("Validating critical dependencies...")
        
        all_valid = True
        for package, version_req in self.critical_dependencies.items():
            if not self.check_package_version(package, version_req):
                all_valid = False
                
        return all_valid

    def import_module_safely(self, module_name: str) -> Tuple[bool, Optional[Any], Optional[str]]:
        """Safely import a module and return success status, module, and error."""
        try:
            module = importlib.import_module(module_name)
            return True, module, None
        except Exception as e:
            error_msg = f"Failed to import {module_name}: {str(e)}"
            if self.verbose:
                error_msg += f"\n{traceback.format_exc()}"
            return False, None, error_msg

    def validate_project_modules(self) -> bool:
        """Validate that all project modules can be imported."""
        self.log_info("Validating project module imports...")
        
        project_modules = [
            "egw_query_expansion",
            "egw_query_expansion.core",
            "egw_query_expansion.mathematical_foundations",
            "egw_query_expansion.core.gw_alignment",
            "egw_query_expansion.core.query_generator", 
            "egw_query_expansion.core.hybrid_retrieval",
            "egw_query_expansion.core.pattern_matcher",
            "egw_query_expansion.core.deterministic_router",
            "egw_query_expansion.core.conformal_risk_control",
            "egw_query_expansion.core.immutable_context",
            "egw_query_expansion.core.permutation_invariant_processor",
            "egw_query_expansion.core.submodular_task_selector",
        ]
        
        all_valid = True
        for module_name in project_modules:
            success, module, error = self.import_module_safely(module_name)
            if success:
                self.log_success(f"Successfully imported {module_name}")
            else:
                self.log_error(f"Import failed for {module_name}: {error}")
                all_valid = False
                
        return all_valid

    def validate_expected_components(self) -> bool:
        """Validate that expected classes and functions are accessible."""
        self.log_info("Validating expected components accessibility...")
        
        all_valid = True
        for module_name, components in self.expected_components.items():
            success, module, error = self.import_module_safely(module_name)
            if not success:
                self.log_error(f"Cannot validate components in {module_name}: {error}")
                all_valid = False
                continue
                
            for component_name in components:
                if hasattr(module, component_name):
                    component = getattr(module, component_name)
                    self.log_success(f"Found {component_name} in {module_name}")
                    
                    # Additional validation for classes
                    if hasattr(component, '__init__'):
                        self.log_info(f"{component_name} is a class")
                    elif callable(component):
                        self.log_info(f"{component_name} is a function")
                else:
                    self.log_warning(f"{component_name} not found in {module_name} (may be optional)")
                    
        return all_valid

    def detect_version_conflicts(self) -> bool:
        """Detect potential version conflicts between packages."""
        self.log_info("Checking for potential version conflicts...")
        
        # Check for common conflicts
        conflicts_found = False
        
        try:
            # Check PyTorch ecosystem compatibility
            torch_version = importlib.metadata.version("torch")
            transformers_version = importlib.metadata.version("transformers")
            
            # Check if versions are compatible (basic heuristic)
            torch_major = int(torch_version.split('.')[0])
            if torch_major < 2:
                self.log_warning("PyTorch version < 2.0 may have compatibility issues")
                
        except importlib.metadata.PackageNotFoundError:
            pass
            
        # Check for FAISS CPU/GPU conflicts
        try:
            faiss_cpu = importlib.metadata.version("faiss-cpu")
            try:
                faiss_gpu = importlib.metadata.version("faiss-gpu") 
                self.log_warning("Both faiss-cpu and faiss-gpu are installed - this may cause conflicts")
                conflicts_found = True
            except importlib.metadata.PackageNotFoundError:
                pass
        except importlib.metadata.PackageNotFoundError:
            pass
            
        return not conflicts_found

    def initialize_status_reporter(self) -> bool:
        """Initialize the status reporter if available and enabled."""
        if not self.enable_status_report or not STATUS_REPORTER_AVAILABLE:
            return False
        
        try:
            compatibility_matrix = MathematicalCompatibilityMatrix()
            self.status_reporter = LibraryStatusReporter(
                matrix=compatibility_matrix,
                report_path="library_status_report.json"
            )
            return True
        except Exception as e:
            self.log_warning(f"Could not initialize status reporter: {e}")
            return False

    def run_comprehensive_status_reporting(self) -> Dict[str, Any]:
        """Run comprehensive status reporting including behavioral consistency tests."""
        if not self.status_reporter:
            return {"error": "Status reporter not available"}
        
        self.log_info("Running comprehensive library status reporting...")
        
        try:
            # Generate the comprehensive report
            report_summary = self.status_reporter.invoke_status_reporting()
            
            if report_summary['success']:
                self.log_success(f"Status report generated: {report_summary['report_path']}")
                self.log_info(f"System health: {report_summary['system_health']['health_classification']}")
                self.log_info(f"Risk level: {report_summary['risk_level']}")
                
                if report_summary['critical_issues']:
                    self.log_error("Critical issues detected in library status")
                
                # Track library calls during validation
                for lib_name in self.status_reporter.matrix.library_specs.keys():
                    self.status_reporter.track_library_call(lib_name, is_mock=False)
                
            return report_summary
            
        except Exception as e:
            error_msg = f"Status reporting failed: {str(e)}"
            self.log_error(error_msg)
            return {"error": error_msg}

    def aggregate_component_results(self) -> Dict[str, Any]:
        """Aggregate results across all EGW components."""
        if not self.aggregate_results:
            return {}
        
        self.log_info("Aggregating results across EGW components...")
        
        aggregated = {
            'total_components_checked': 0,
            'components_with_issues': 0,
            'common_missing_libraries': {},
            'stage_readiness_summary': {},
            'overall_system_health': {},
            'component_details': {}
        }
        
        # Define EGW component stages to check
        egw_stages = [
            'differential_geometry',
            'category_theory', 
            'topological_data_analysis',
            'information_theory',
            'optimal_transport',
            'spectral_methods',
            'control_theory',
            'measure_theory',
            'optimization_theory',
            'algebraic_topology',
            'functional_analysis',
            'statistical_learning'
        ]
        
        if self.status_reporter:
            try:
# # #                 # Get comprehensive report from status reporter  # Module not found  # Module not found  # Module not found
                full_report = self.status_reporter.generate_comprehensive_report()
                
                aggregated['total_components_checked'] = len(egw_stages)
                aggregated['overall_system_health'] = full_report['system_health']
                aggregated['stage_readiness_summary'] = full_report['stage_readiness']
                
                # Count components with issues
                for stage, readiness in full_report['stage_readiness'].items():
                    if readiness['status'] != 'READY':
                        aggregated['components_with_issues'] += 1
                    
                    aggregated['component_details'][stage] = {
                        'status': readiness['status'],
                        'missing_deps': readiness['missing_dependencies'],
                        'degraded_deps': readiness['degraded_dependencies']
                    }
                
                # Track common missing libraries across components
                for stage, readiness in full_report['stage_readiness'].items():
                    for missing_lib in readiness['missing_dependencies']:
                        if missing_lib not in aggregated['common_missing_libraries']:
                            aggregated['common_missing_libraries'][missing_lib] = []
                        aggregated['common_missing_libraries'][missing_lib].append(stage)
                
                self.aggregated_results = aggregated
                
            except Exception as e:
                self.log_error(f"Component aggregation failed: {str(e)}")
                
        return aggregated

    def run_validation(self) -> bool:
        """Run full validation and return success status."""
        if not self.ci_mode:
            print("🔍 Starting comprehensive dependency validation...")
            print("=" * 50)
        
        # Initialize status reporter if requested
        status_reporter_initialized = self.initialize_status_reporter()
        
        # Run all validation checks
        results = [
            self.validate_critical_dependencies(),
            self.validate_project_modules(), 
            self.validate_expected_components(),
            self.detect_version_conflicts()
        ]
        
        all_passed = all(results)
        
        # Run comprehensive status reporting if enabled
        status_report_summary = {}
        if status_reporter_initialized:
            status_report_summary = self.run_comprehensive_status_reporting()
            if not status_report_summary.get('success', False):
                all_passed = False
        
        # Run component aggregation if requested
        aggregated_results = {}
        if self.aggregate_results:
            aggregated_results = self.aggregate_component_results()
        
        # Print summary
        if not self.ci_mode:
            print("\n" + "=" * 50)
            print("📊 VALIDATION SUMMARY")
            print(f"Errors: {len(self.errors)}")
            print(f"Warnings: {len(self.warnings)}")
            
            if status_report_summary.get('success'):
                print(f"\n📋 STATUS REPORT GENERATED")
                print(f"Report Path: {status_report_summary['report_path']}")
                print(f"System Health: {status_report_summary['system_health']['health_classification']}")
                print(f"Risk Level: {status_report_summary['risk_level']}")
                print(f"Recommendations: {status_report_summary['recommendations_count']}")
            
            if aggregated_results:
                print(f"\n🔄 COMPONENT AGGREGATION")
                print(f"Components Checked: {aggregated_results['total_components_checked']}")
                print(f"Components with Issues: {aggregated_results['components_with_issues']}")
                if aggregated_results['common_missing_libraries']:
                    print(f"Common Missing Libraries: {list(aggregated_results['common_missing_libraries'].keys())}")
            
            if self.errors:
                print("\n❌ ERRORS:")
                for error in self.errors:
                    print(f"  • {error}")
                    
            if self.warnings and self.verbose:
                print("\n⚠️  WARNINGS:")
                for warning in self.warnings:
                    print(f"  • {warning}")
                    
            if all_passed and not self.errors:
                print("\n🎉 All validations passed!")
            else:
                print(f"\n💥 Validation failed with {len(self.errors)} errors")
        elif self.errors:
            # In CI mode, only print errors
            for error in self.errors:
                print(f"ERROR: {error}")
                
        return all_passed and not self.errors

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate project dependencies and imports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--ci", 
        action="store_true", 
        help="CI mode with minimal output (exit 1 on failure)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose mode with detailed dependency information"
    )
    
    parser.add_argument(
        "--status-report",
        action="store_true",
        help="Generate comprehensive library status report"
    )
    
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Aggregate results across all EGW components"
    )
    
    args = parser.parse_args()
    
    # Enable warnings in verbose mode
    if args.verbose:
        warnings.filterwarnings("default")
    
    validator = DependencyValidator(
        verbose=args.verbose, 
        ci_mode=args.ci,
        enable_status_report=args.status_report,
        aggregate_results=args.aggregate_results
    )
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()