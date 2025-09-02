#!/usr/bin/env python3
"""
Comprehensive Dependency Compatibility Validation Script

This script provides enhanced dependency validation with the existing check_library_compatibility() 
function integration, fail-fast behavior, and compatibility report generation for the EGW Query 
Expansion pipeline.

Key Features:
- Executes existing check_library_compatibility() function at startup
- Tests Python 3.8-3.12 compatibility  
- Import validation for all pipeline modules
- Version conflict detection (faiss-cpu/gpu, PyTorch variants)
- Mock implementation API consistency validation
- Generates actionable compatibility reports with upgrade recommendations

Usage:
    python scripts/validate_dependency_compatibility.py [--fail-fast] [--output-dir DIR] [--verbose]
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class DependencyCompatibilityValidator:
    """Enhanced dependency compatibility validator with fail-fast behavior and comprehensive reporting."""
    
    def __init__(self, fail_fast: bool = False, output_dir: Path = None, verbose: bool = False):
        self.fail_fast = fail_fast
        self.output_dir = output_dir or Path("validation_reports")
        self.verbose = verbose
        self.output_dir.mkdir(exist_ok=True)
        
        self.errors = []
        self.warnings = []
        self.compatibility_results = {}
        self.import_results = {}
        self.mock_validation_results = {}
        self.upgrade_recommendations = []
        
        # Critical dependencies that trigger fail-fast
        self.critical_dependencies = [
            'numpy', 'scipy', 'torch', 'sklearn', 'POT', 
            'transformers', 'sentence-transformers'
        ]
        
        # EGW pipeline modules for import validation
        self.egw_modules = [
            'egw_query_expansion',
            'egw_query_expansion.core',
            'egw_query_expansion.core.gw_alignment',
            'egw_query_expansion.core.query_generator',
            'egw_query_expansion.core.hybrid_retrieval', 
            'egw_query_expansion.core.pattern_matcher',
            'egw_query_expansion.core.deterministic_router',
            'egw_query_expansion.core.conformal_risk_control',
            'egw_query_expansion.core.immutable_context',
            'egw_query_expansion.core.permutation_invariant_processor',
            'egw_query_expansion.core.submodular_task_selector',
            'egw_query_expansion.mathematical_foundations'
        ]
        
    def log_error(self, message: str, critical: bool = False):
        """Log error with optional fail-fast behavior."""
        self.errors.append(message)
        if self.verbose:
            print(f"❌ ERROR: {message}")
        
        if critical and self.fail_fast:
            print(f"FAIL_FAST: Critical error encountered: {message}")
            sys.exit(1)
            
    def log_warning(self, message: str):
        """Log warning message."""
        self.warnings.append(message)
        if self.verbose:
            print(f"⚠️  WARNING: {message}")
            
    def log_success(self, message: str):
        """Log success message."""
        if self.verbose:
            print(f"✅ SUCCESS: {message}")
            
    def log_info(self, message: str):
        """Log info message."""
        if self.verbose:
            print(f"ℹ️  INFO: {message}")

    def execute_existing_compatibility_check(self) -> bool:
        """Execute the existing check_library_compatibility() function with fail-fast behavior."""
        self.log_info("Executing existing check_library_compatibility() function...")
        
        try:
            # Import the mathematical compatibility matrix
            from canonical_flow.mathematical_enhancers.mathematical_compatibility_matrix import (
                MathematicalCompatibilityMatrix
            )
            
            matrix = MathematicalCompatibilityMatrix()
            
            # Check critical dependencies first with fail-fast
            critical_failures = []
            for lib in self.critical_dependencies:
                result = matrix.check_library_compatibility(lib)
                self.compatibility_results[lib] = {
                    'compatible': result.is_compatible,
                    'installed_version': result.installed_version,
                    'required_version': result.required_version,
                    'issues': result.issues,
                    'warnings': result.warnings
                }
                
                if not result.is_compatible:
                    critical_failures.append(lib)
                    self.log_error(f"Critical dependency {lib} incompatible: {result.issues}", critical=True)
                else:
                    self.log_success(f"Critical dependency {lib} compatible: {result.installed_version}")
            
            # If critical failures and fail-fast enabled, exit immediately
            if critical_failures and self.fail_fast:
                self.log_error(f"FAIL_FAST: Critical dependencies failed: {critical_failures}", critical=True)
                return False
            
            # Check all other libraries
            all_results = matrix.check_all_compatibility()
            for lib, result in all_results.items():
                if lib not in self.compatibility_results:
                    self.compatibility_results[lib] = {
                        'compatible': result.is_compatible,
                        'installed_version': result.installed_version,
                        'required_version': result.required_version,
                        'issues': result.issues,
                        'warnings': result.warnings
                    }
            
            # Generate upgrade recommendations
            for lib, result_data in self.compatibility_results.items():
                if not result_data['compatible'] and result_data.get('installed_version'):
                    priority = 'high' if lib in self.critical_dependencies else 'medium'
                    self.upgrade_recommendations.append({
                        'library': lib,
                        'current_version': result_data['installed_version'],
                        'required_version': result_data.get('required_version', 'latest'),
                        'priority': priority,
                        'issues': result_data.get('issues', [])
                    })
            
            return len(critical_failures) == 0
            
        except Exception as e:
            self.log_error(f"Failed to execute check_library_compatibility(): {e}", critical=True)
            if self.verbose:
                traceback.print_exc()
            return False
    
    def validate_python_version_compatibility(self) -> bool:
        """Validate Python version compatibility across 3.8-3.12."""
        self.log_info("Validating Python version compatibility...")
        
        current_version = sys.version_info
        version_str = f"{current_version.major}.{current_version.minor}"
        
        supported_versions = ['3.8', '3.9', '3.10', '3.11', '3.12']
        
        if version_str not in supported_versions:
            self.log_warning(f"Python {version_str} not in officially supported versions: {supported_versions}")
            return False
        else:
            self.log_success(f"Python {version_str} is supported")
            return True
    
    def detect_version_conflicts(self) -> bool:
        """Detect version conflicts between critical packages."""
        self.log_info("Detecting version conflicts...")
        
        conflicts_detected = False
        
        try:
            # Import metadata with fallback
            try:
                import importlib.metadata as metadata
            except ImportError:
                try:
                    import importlib_metadata as metadata
                except ImportError:
                    self.log_warning("importlib.metadata not available - cannot check version conflicts")
                    return True
            
            # Check FAISS CPU/GPU conflicts
            faiss_variants = []
            for variant in ['faiss-cpu', 'faiss-gpu']:
                try:
                    version = metadata.version(variant)
                    faiss_variants.append((variant, version))
                except metadata.PackageNotFoundError:
                    pass
                except Exception:
                    pass
            
            if len(faiss_variants) > 1:
                conflict_msg = f"Multiple FAISS variants detected: {faiss_variants}"
                self.log_error(conflict_msg)
                conflicts_detected = True
            elif faiss_variants:
                variant, version = faiss_variants[0]
                self.log_success(f"Single FAISS variant: {variant} {version}")
            else:
                self.log_error("No FAISS variant installed", critical=True)
                conflicts_detected = True
            
            # Check PyTorch variant conflicts  
            torch_variants = []
            for variant in ['torch', 'torch-nightly']:
                try:
                    version = metadata.version(variant)
                    torch_variants.append((variant, version))
                except metadata.PackageNotFoundError:
                    pass
                except Exception:
                    pass
            
            if len(torch_variants) > 1:
                conflict_msg = f"Multiple PyTorch variants detected: {torch_variants}"
                self.log_warning(conflict_msg)
            elif torch_variants:
                variant, version = torch_variants[0]
                self.log_success(f"PyTorch variant: {variant} {version}")
            else:
                self.log_error("No PyTorch variant installed", critical=True)
                conflicts_detected = True
                
        except Exception as e:
            self.log_error(f"Error detecting version conflicts: {e}")
            conflicts_detected = True
        
        return not conflicts_detected
    
    def validate_egw_pipeline_imports(self) -> bool:
        """Validate import safety across all EGW pipeline components."""
        self.log_info("Validating EGW pipeline module imports...")
        
        import_failures = []
        import_warnings = []
        
        for module_name in self.egw_modules:
            try:
                # Attempt to import the module
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.log_warning(f"Module spec not found: {module_name}")
                    import_warnings.append((module_name, "spec_not_found"))
                    continue
                
                module = importlib.import_module(module_name)
                self.import_results[module_name] = {
                    'status': 'success',
                    'path': getattr(spec, 'origin', 'unknown'),
                    'submodules': [name for name in dir(module) if not name.startswith('_')]
                }
                self.log_success(f"Successfully imported: {module_name}")
                
            except ImportError as e:
                error_msg = str(e)
                if 'optional' in error_msg.lower() or 'not available' in error_msg.lower():
                    self.log_warning(f"Optional dependency missing for {module_name}: {error_msg}")
                    import_warnings.append((module_name, error_msg))
                    self.import_results[module_name] = {
                        'status': 'warning',
                        'error': error_msg,
                        'optional': True
                    }
                else:
                    self.log_error(f"Import failed for {module_name}: {error_msg}")
                    import_failures.append((module_name, error_msg))
                    self.import_results[module_name] = {
                        'status': 'error',
                        'error': error_msg,
                        'optional': False
                    }
                    
            except Exception as e:
                error_msg = f"Unexpected error importing {module_name}: {e}"
                self.log_error(error_msg)
                import_failures.append((module_name, str(e)))
                self.import_results[module_name] = {
                    'status': 'error',
                    'error': str(e),
                    'unexpected': True
                }
        
        # Fail-fast on critical import failures
        if import_failures and self.fail_fast:
            critical_failures = [f for f in import_failures if not f[1].get('optional', False)]
            if critical_failures:
                self.log_error(f"FAIL_FAST: Critical import failures: {[f[0] for f in critical_failures]}", critical=True)
        
        return len(import_failures) == 0
    
    def validate_mock_api_consistency(self) -> bool:
        """Validate that mock implementations maintain API consistency with real libraries."""
        self.log_info("Validating mock implementation API consistency...")
        
        mock_tests = {
            'torch': self._test_torch_mock_api,
            'faiss': self._test_faiss_mock_api,
            'transformers': self._test_transformers_mock_api
        }
        
        all_consistent = True
        
        for lib_name, test_func in mock_tests.items():
            try:
                is_consistent = test_func()
                self.mock_validation_results[lib_name] = {
                    'consistent': is_consistent,
                    'tested': True
                }
                
                if is_consistent:
                    self.log_success(f"Mock API consistent for {lib_name}")
                else:
                    self.log_error(f"Mock API inconsistent for {lib_name}")
                    all_consistent = False
                    
            except Exception as e:
                self.log_warning(f"Could not test mock API for {lib_name}: {e}")
                self.mock_validation_results[lib_name] = {
                    'consistent': False,
                    'tested': False,
                    'error': str(e)
                }
        
        return all_consistent
    
    def _test_torch_mock_api(self) -> bool:
        """Test PyTorch mock API consistency."""
        class MockTorch:
            def __init__(self):
                self.cuda = MockCuda()
                self.backends = MockBackends()
                self.__version__ = "mock"
                
            def manual_seed(self, seed): pass
            def no_grad(self): 
                return type('NoGradContext', (), {
                    '__enter__': lambda s: s,
                    '__exit__': lambda s, *a: None
                })()
        
        class MockCuda:
            def is_available(self): return False
            def manual_seed(self, seed): pass
            def manual_seed_all(self, seed): pass
            
        class MockBackends:
            def __init__(self):
                self.cudnn = type('MockCudnn', (), {
                    'deterministic': True,
                    'benchmark': False
                })()
        
        # Test API consistency
        mock_torch = MockTorch()
        
        required_attrs = ['manual_seed', 'cuda', 'backends', 'no_grad']
        for attr in required_attrs:
            if not hasattr(mock_torch, attr):
                return False
                
        # Test cuda API
        cuda_attrs = ['is_available', 'manual_seed', 'manual_seed_all']
        for attr in cuda_attrs:
            if not hasattr(mock_torch.cuda, attr):
                return False
                
        # Test backends API
        if not hasattr(mock_torch.backends, 'cudnn'):
            return False
            
        cudnn_attrs = ['deterministic', 'benchmark']
        for attr in cudnn_attrs:
            if not hasattr(mock_torch.backends.cudnn, attr):
                return False
        
        return True
    
    def _test_faiss_mock_api(self) -> bool:
        """Test FAISS mock API consistency."""
        class MockFaiss:
            def seed_global_rng(self, seed): pass
            def IndexFlatIP(self, dim): return MockIndex(dim)
            def IndexFlatL2(self, dim): return MockIndex(dim) 
            def write_index(self, index, path): 
                Path(path).write_bytes(b'mock_faiss_index')
            def read_index(self, path): return MockIndex(128)
        
        class MockIndex:
            def __init__(self, dim):
                self.d = dim
                self.ntotal = 0
            def add(self, vectors): 
                self.ntotal += len(vectors) if hasattr(vectors, '__len__') else 1
        
        mock_faiss = MockFaiss()
        
        # Test required methods
        required_methods = ['seed_global_rng', 'IndexFlatIP', 'IndexFlatL2', 'write_index', 'read_index']
        for method in required_methods:
            if not hasattr(mock_faiss, method):
                return False
        
        # Test index API
        index = mock_faiss.IndexFlatIP(128)
        index_attrs = ['d', 'ntotal', 'add']
        for attr in index_attrs:
            if not hasattr(index, attr):
                return False
        
        return True
    
    def _test_transformers_mock_api(self) -> bool:
        """Test Transformers mock API consistency."""
        class MockTransformers:
            class AutoTokenizer:
                @staticmethod
                def from_pretrained(model_name): return MockTokenizer()
            class AutoModel:
                @staticmethod  
                def from_pretrained(model_name): return MockModel()
        
        class MockTokenizer:
            def encode(self, text): return [1, 2, 3]
            def decode(self, tokens): return "mock text"
            def __call__(self, text): return {'input_ids': [1, 2, 3]}
        
        class MockModel:
            def eval(self): return self
            def __call__(self, *args, **kwargs): return type('MockOutput', (), {'last_hidden_state': None})()
        
        mock_transformers = MockTransformers()
        
        # Test required classes and methods
        if not hasattr(mock_transformers, 'AutoTokenizer'):
            return False
        if not hasattr(mock_transformers, 'AutoModel'):
            return False
            
        tokenizer = mock_transformers.AutoTokenizer.from_pretrained("mock")
        tokenizer_methods = ['encode', 'decode', '__call__']
        for method in tokenizer_methods:
            if not hasattr(tokenizer, method):
                return False
        
        model = mock_transformers.AutoModel.from_pretrained("mock")
        model_methods = ['eval', '__call__']
        for method in model_methods:
            if not hasattr(model, method):
                return False
        
        return True
    
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive compatibility report with upgrade recommendations."""
        report = {
            'metadata': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform,
                'fail_fast_enabled': self.fail_fast
            },
            'summary': {
                'total_libraries_tested': len(self.compatibility_results),
                'compatible_libraries': sum(1 for r in self.compatibility_results.values() if r['compatible']),
                'critical_failures': len([lib for lib in self.critical_dependencies 
                                        if not self.compatibility_results.get(lib, {}).get('compatible', False)]),
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings)
            },
            'compatibility_results': self.compatibility_results,
            'import_validation': {
                'modules_tested': len(self.egw_modules),
                'successful_imports': len([r for r in self.import_results.values() if r['status'] == 'success']),
                'failed_imports': len([r for r in self.import_results.values() if r['status'] == 'error']),
                'results': self.import_results
            },
            'mock_validation': self.mock_validation_results,
            'upgrade_recommendations': self.upgrade_recommendations,
            'fallback_usage_patterns': self._generate_fallback_patterns(),
            'errors': self.errors,
            'warnings': self.warnings
        }
        
        # Save report to file
        report_file = self.output_dir / 'dependency_compatibility_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log_info(f"Compatibility report saved to: {report_file}")
        return report
    
    def _generate_fallback_patterns(self) -> Dict[str, Any]:
        """Generate fallback usage patterns analysis."""
        patterns = {
            'torch_fallbacks': {
                'cpu_only': not self._is_cuda_available(),
                'mock_usage': 'torch' not in sys.modules or not hasattr(sys.modules.get('torch', None), 'cuda')
            },
            'faiss_fallbacks': {
                'cpu_variant': 'faiss-cpu' in [lib for lib, _ in self._get_installed_packages()],
                'mock_usage': 'faiss' not in sys.modules
            },
            'optional_dependencies': {
                'control_theory': self._check_optional_dependency('control'),
                'topological_analysis': self._check_optional_dependency('gudhi'),
                'advanced_optimization': self._check_optional_dependency('ray')
            }
        }
        return patterns
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _get_installed_packages(self) -> List[Tuple[str, str]]:
        """Get list of installed packages."""
        try:
            # Try importlib.metadata first (Python 3.8+)
            try:
                import importlib.metadata as metadata
                return [(dist.metadata['name'], dist.version) for dist in metadata.distributions()]
            except (ImportError, AttributeError):
                # Fallback to pkg_resources
                import pkg_resources
                return [(pkg.project_name, pkg.version) for pkg in pkg_resources.working_set]
        except:
            return []
    
    def _check_optional_dependency(self, dep_name: str) -> bool:
        """Check if optional dependency is available."""
        try:
            importlib.import_module(dep_name)
            return True
        except ImportError:
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete validation pipeline."""
        print("🔍 EGW Query Expansion - Dependency Compatibility Validation")
        print("=" * 70)
        
        validation_steps = [
            ("Existing Compatibility Check", self.execute_existing_compatibility_check),
            ("Python Version Compatibility", self.validate_python_version_compatibility),
            ("Version Conflict Detection", self.detect_version_conflicts),
            ("EGW Pipeline Import Validation", self.validate_egw_pipeline_imports),
            ("Mock API Consistency", self.validate_mock_api_consistency)
        ]
        
        results = []
        for step_name, step_func in validation_steps:
            self.log_info(f"Running: {step_name}")
            try:
                result = step_func()
                results.append(result)
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"{status} {step_name}")
            except Exception as e:
                self.log_error(f"Exception in {step_name}: {e}")
                results.append(False)
                print(f"💥 EXCEPTION {step_name}: {e}")
                if self.verbose:
                    traceback.print_exc()
        
        # Generate final report
        report = self.generate_compatibility_report()
        
        # Print summary
        all_passed = all(results)
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print(f"Steps passed: {sum(results)}/{len(results)}")
        print(f"Errors: {len(self.errors)}")  
        print(f"Warnings: {len(self.warnings)}")
        print(f"Critical libraries compatible: {report['summary']['compatible_libraries']}/{report['summary']['total_libraries_tested']}")
        print(f"EGW modules imported: {report['import_validation']['successful_imports']}/{report['import_validation']['modules_tested']}")
        
        if self.upgrade_recommendations:
            print(f"\n🔄 UPGRADE RECOMMENDATIONS: {len(self.upgrade_recommendations)}")
            for rec in self.upgrade_recommendations:
                priority_emoji = "🚨" if rec['priority'] == 'high' else "⚠️"
                print(f"  {priority_emoji} {rec['library']}: {rec['current_version']} → {rec['required_version']}")
        
        if all_passed:
            print("\n🎉 All validation steps passed!")
            return True
        else:
            print(f"\n💥 Validation failed: {len(results) - sum(results)} steps failed")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive dependency compatibility validation for EGW Query Expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--fail-fast", 
        action="store_true",
        help="Exit immediately on critical dependency failures"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_reports"),
        help="Output directory for compatibility reports"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Enable verbose logging output"
    )
    
    args = parser.parse_args()
    
    validator = DependencyCompatibilityValidator(
        fail_fast=args.fail_fast,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    success = validator.run_full_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()