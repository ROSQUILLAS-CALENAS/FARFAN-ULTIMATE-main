#!/usr/bin/env python3

# Mandatory Pipeline Contract Annotations
__phase__ = "O"
__code__ = "98O"
__stage_order__ = 7

"""
Validation script for EGW Query Expansion installation with comprehensive pre-installation checks
"""

import importlib
import os
import platform
import shutil
import subprocess
import sys
import traceback
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Optional, Tuple  # Module not found  # Module not found  # Module not found


@dataclass
class ValidationIssue:
    """Represents a validation issue with remediation steps."""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'python', 'venv', 'system', 'disk', 'dependencies'
    message: str
    remediation: List[str]


class PreInstallationValidator:
    """Validates environment before installation begins."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.min_python_version = (3, 8)
        self.max_python_version = (3, 12)
        self.required_disk_space_gb = 5.0  # Minimum disk space in GB
        
    def validate_python_version(self) -> bool:
        """Validate Python version compatibility."""
        print("🐍 Checking Python version...")
        
        current_version = sys.version_info[:2]
        version_str = f"{current_version[0]}.{current_version[1]}"
        
        if current_version < self.min_python_version:
            self.issues.append(ValidationIssue(
                severity='critical',
                category='python',
                message=f"Python {version_str} is too old (minimum: {self.min_python_version[0]}.{self.min_python_version[1]})",
                remediation=[
                    f"Install Python {self.min_python_version[0]}.{self.min_python_version[1]} or newer",
                    "Visit https://python.org/downloads/ to download the latest version",
                    "On Ubuntu/Debian: sudo apt update && sudo apt install python3.9 python3.9-venv python3.9-pip",
                    "On macOS with Homebrew: brew install python@3.9",
# # #                     "On Windows: Download from python.org and run the installer",  # Module not found  # Module not found  # Module not found
                    "After installation, create a new virtual environment with the updated Python version"
                ]
            ))
            return False
            
        elif current_version > self.max_python_version:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='python',
                message=f"Python {version_str} may be too new (tested up to: {self.max_python_version[0]}.{self.max_python_version[1]})",
                remediation=[
                    f"Consider using Python {self.max_python_version[0]}.{self.max_python_version[1]} for best compatibility",
                    "If you encounter issues, install an older Python version",
                    "Use pyenv to manage multiple Python versions: pyenv install 3.11.0 && pyenv local 3.11.0"
                ]
            ))
            
        print(f"✅ Python {version_str} is compatible")
        return True
        
    def validate_virtual_environment(self) -> bool:
        """Validate virtual environment status and configuration."""
        print("🏠 Checking virtual environment...")
        
        is_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if not is_venv:
            self.issues.append(ValidationIssue(
                severity='critical',
                category='venv',
                message="Not running in a virtual environment",
                remediation=[
                    "Create a virtual environment: python -m venv venv",
                    "Activate it:",
                    "  - Linux/macOS: source venv/bin/activate",
                    "  - Windows: venv\\Scripts\\activate",
                    "Verify activation: which python (should show path in venv directory)",
                    "Re-run this script after activation"
                ]
            ))
            return False
            
        # Check if venv directory exists and is properly structured
        venv_path = Path(sys.prefix)
        if not venv_path.exists():
            self.issues.append(ValidationIssue(
                severity='critical',
                category='venv',
                message=f"Virtual environment path {venv_path} does not exist",
                remediation=[
                    "Recreate the virtual environment: python -m venv venv",
                    "Activate it and try again"
                ]
            ))
            return False
            
        # Check for proper venv structure
        required_venv_dirs = ['lib', 'bin' if os.name != 'nt' else 'Scripts']
        for dir_name in required_venv_dirs:
            if not (venv_path / dir_name).exists():
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='venv',
                    message=f"Virtual environment appears incomplete (missing {dir_name})",
                    remediation=[
                        "Recreate the virtual environment: python -m venv venv --clear",
                        "Activate it: source venv/bin/activate",
                        "Upgrade pip: python -m pip install --upgrade pip"
                    ]
                ))
                
        print(f"✅ Running in virtual environment: {sys.prefix}")
        return True
        
    def validate_system_prerequisites(self) -> bool:
        """Validate system-level prerequisites."""
        print("🔧 Checking system prerequisites...")
        
        system = platform.system().lower()
        issues_found = False
        
        # Check for essential system commands
        if system in ['linux', 'darwin']:  # Linux or macOS
            essential_commands = ['gcc', 'g++', 'make']
            for cmd in essential_commands:
                if not shutil.which(cmd):
                    self.issues.append(ValidationIssue(
                        severity='critical',
                        category='system',
                        message=f"Missing essential build tool: {cmd}",
                        remediation=[
                            "Install build essentials:",
                            "  - Ubuntu/Debian: sudo apt update && sudo apt install build-essential",
                            "  - CentOS/RHEL: sudo yum groupinstall 'Development Tools'",
                            "  - macOS: xcode-select --install",
# # #                             "  - Or install Xcode from the App Store"  # Module not found  # Module not found  # Module not found
                        ]
                    ))
                    issues_found = True
                    
        # Check Python development headers
        if system == 'linux':
            try:
                import distutils.util
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                # This is a heuristic check - not perfect but catches common issues
                python_dev_paths = [
                    f"/usr/include/python{python_version}",
                    f"/usr/local/include/python{python_version}",
                ]
                if not any(os.path.exists(path) for path in python_dev_paths):
                    self.issues.append(ValidationIssue(
                        severity='warning',
                        category='system',
                        message="Python development headers may be missing",
                        remediation=[
                            f"Install Python dev headers: sudo apt install python{python_version}-dev",
                            f"Or: sudo yum install python{python_version.replace('.', '')}-devel",
                            "This may be needed for compiling some dependencies"
                        ]
                    ))
            except ImportError:
                pass
                
        if not issues_found:
            print("✅ System prerequisites look good")
            
        return not issues_found
        
    def validate_disk_space(self) -> bool:
        """Validate available disk space."""
        print("💾 Checking disk space...")
        
        try:
            if hasattr(shutil, 'disk_usage'):
                total, used, free = shutil.disk_usage('.')
                free_gb = free / (1024**3)
                
                if free_gb < self.required_disk_space_gb:
                    self.issues.append(ValidationIssue(
                        severity='critical',
                        category='disk',
                        message=f"Insufficient disk space: {free_gb:.1f}GB available, {self.required_disk_space_gb}GB required",
                        remediation=[
                            "Free up disk space by:",
                            "  - Deleting unnecessary files and folders",
                            "  - Clearing temporary files: rm -rf /tmp/* (Linux/macOS)",
                            "  - Emptying trash/recycle bin",
                            "  - Removing old virtual environments",
                            "  - Using disk cleanup tools",
                            f"Need at least {self.required_disk_space_gb}GB free for installation"
                        ]
                    ))
                    return False
                    
                print(f"✅ Sufficient disk space: {free_gb:.1f}GB available")
                return True
        except Exception as e:
            print(f"⚠️  Could not check disk space: {e}")
            
        return True
        
    def validate_pip_configuration(self) -> bool:
        """Validate pip installation and configuration."""
        print("📦 Checking pip configuration...")
        
        try:
            # Check if pip is available
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode != 0:
                self.issues.append(ValidationIssue(
                    severity='critical',
                    category='dependencies',
                    message="pip is not available or not working",
                    remediation=[
                        "Install/upgrade pip:",
                        "python -m ensurepip --upgrade",
                        "Or download get-pip.py and run: python get-pip.py",
                        "Verify: python -m pip --version"
                    ]
                ))
                return False
                
            # Check pip version (warn if very old)
            pip_version = result.stdout.strip()
            print(f"✅ pip is available: {pip_version}")
            
            # Try to upgrade pip
            upgrade_result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                                         capture_output=True, text=True)
            if upgrade_result.returncode == 0:
                print("✅ pip upgraded to latest version")
            else:
                self.issues.append(ValidationIssue(
                    severity='warning',
                    category='dependencies',
                    message="Could not upgrade pip",
                    remediation=[
                        "Try manually: python -m pip install --upgrade pip",
                        "If behind corporate firewall, use: --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org"
                    ]
                ))
                
            return True
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity='critical',
                category='dependencies',
                message=f"pip validation failed: {e}",
                remediation=[
                    "Reinstall pip: python -m ensurepip --upgrade",
# # #                     "Or download get-pip.py from https://bootstrap.pypa.io/get-pip.py"  # Module not found  # Module not found  # Module not found
                ]
            ))
            return False
            
    def validate_network_connectivity(self) -> bool:
        """Validate network connectivity for package downloads."""
        print("🌐 Checking network connectivity...")
        
        try:
            import urllib.request
            urllib.request.urlopen('https://pypi.org', timeout=10)
            print("✅ Network connectivity to PyPI is working")
            return True
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity='warning',
                category='system',
                message=f"Network connectivity issue: {e}",
                remediation=[
                    "Check internet connection",
                    "If behind corporate firewall:",
                    "  - Configure pip with proxy: pip config set global.proxy http://proxy:port",
                    "  - Or use trusted hosts: --trusted-host pypi.org --trusted-host files.pythonhosted.org",
                    "Try: pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy"
                ]
            ))
            return False
            
    def run_all_validations(self) -> bool:
        """Run all pre-installation validations."""
        print("🔍 Running Pre-Installation Environment Validation")
        print("=" * 60)
        
        validations = [
            self.validate_python_version,
            self.validate_virtual_environment,
            self.validate_system_prerequisites,
            self.validate_disk_space,
            self.validate_pip_configuration,
            self.validate_network_connectivity
        ]
        
        all_passed = True
        for validation in validations:
            try:
                if not validation():
                    all_passed = False
            except Exception as e:
                print(f"❌ Validation error: {e}")
                all_passed = False
                
        return all_passed
        
    def print_issues_and_remediation(self) -> None:
        """Print detailed issues and remediation steps."""
        if not self.issues:
            return
            
        print("\n" + "=" * 60)
        print("🚨 ISSUES DETECTED - REMEDIATION REQUIRED")
        print("=" * 60)
        
        critical_issues = [i for i in self.issues if i.severity == 'critical']
        warning_issues = [i for i in self.issues if i.severity == 'warning']
        
        if critical_issues:
            print("\n🛑 CRITICAL ISSUES (must be fixed before proceeding):")
            for i, issue in enumerate(critical_issues, 1):
                print(f"\n{i}. {issue.message}")
                print("   Remediation steps:")
                for step in issue.remediation:
                    print(f"   • {step}")
                    
        if warning_issues:
            print("\n⚠️  WARNINGS (recommended to fix):")
            for i, issue in enumerate(warning_issues, 1):
                print(f"\n{i}. {issue.message}")
                print("   Remediation steps:")
                for step in issue.remediation:
                    print(f"   • {step}")


def test_imports():
    """Test importing core modules."""
    print("\n🔍 Testing module imports...")

    try:
        # Test numpy/scipy
        import numpy as np
        import scipy
        print("✅ NumPy and SciPy imported successfully")

        # Test PyTorch
        import torch
        print("✅ PyTorch imported successfully")

        # Test FAISS
        import faiss
        print("✅ FAISS imported successfully")

        # Test POT
        import ot
        print("✅ POT (Python Optimal Transport) imported successfully")

        # Test transformers and sentence-transformers
        import sentence_transformers
        import transformers
        print("✅ Transformers and Sentence-Transformers imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 If this is your first run, dependencies may not be installed yet.")
        print("   The pre-installation validation above should help identify any issues.")
        return False


def test_basic_functionality():
    """Test basic functionality without heavy computation."""
    print("\n🧪 Testing basic functionality...")

    try:
        import numpy as np

        # Test basic array operations
        a = np.random.rand(10, 5)
        b = np.random.rand(5, 8)
        c = np.dot(a, b)
        print("✅ Basic NumPy operations work")

        # Test FAISS basic operations
        import faiss
        index = faiss.IndexFlatL2(128)
        vectors = np.random.random((10, 128)).astype("float32")
        index.add(vectors)
        print("✅ Basic FAISS operations work")

        # Test POT basic operations
        import ot
        a = np.ones(10) / 10
        b = np.ones(10) / 10
        M = np.random.rand(10, 10)
        transport = ot.emd(a, b, M)
        print("✅ Basic POT operations work")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_egw_modules():
    """Test EGW-specific module imports."""
    print("\n🎯 Testing EGW module imports...")

    try:
        sys.path.append(".")

        # Test conformal risk control module
# # #         from egw_query_expansion.core.conformal_risk_control import (  # Module not found  # Module not found  # Module not found
            ConformalRiskController,
            RiskControlConfig,
            create_conformal_system,
        )
        print("✅ Conformal Risk Control module imported successfully")

        # Test other core modules if they exist
        try:
# # #             from egw_query_expansion.core.gw_alignment import GromovWassersteinAligner  # Module not found  # Module not found  # Module not found
            print("✅ GromovWassersteinAligner imported successfully")
        except ImportError:
            print("ℹ️  GromovWassersteinAligner not available (module not implemented)")

        try:
# # #             from egw_query_expansion.core.query_generator import QueryGenerator  # Module not found  # Module not found  # Module not found
            print("✅ QueryGenerator imported successfully")
        except ImportError:
            print("ℹ️  QueryGenerator not available (module not implemented)")

        try:
# # #             from egw_query_expansion.core.hybrid_retrieval import HybridRetriever  # Module not found  # Module not found  # Module not found
            print("✅ HybridRetriever imported successfully")
        except ImportError:
            print("ℹ️  HybridRetriever not available (module not implemented)")

        try:
# # #             from egw_query_expansion.core.pattern_matcher import PatternMatcher  # Module not found  # Module not found  # Module not found
            print("✅ PatternMatcher imported successfully")
        except ImportError:
            print("ℹ️  PatternMatcher not available (module not implemented)")

        # Test main package import
        import egw_query_expansion
        print("✅ Main package imported successfully")

        return True

    except ImportError as e:
        print(f"❌ EGW module import error: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")

    try:
        import yaml
        config_path = "egw_query_expansion/configs/default_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print("✅ Configuration file loaded successfully")
            print(f"   - Models configured: {list(config.get('models', {}).keys())}")
            print(f"   - GW epsilon: {config.get('gw_alignment', {}).get('epsilon', 'N/A')}")
            return True
        else:
            print(f"❌ Configuration file not found: {config_path}")
            return False

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_conformal_functionality():
    """Test conformal risk control functionality."""
    print("\n🔬 Testing Conformal Risk Control functionality...")

    try:
        import numpy as np
# # #         from egw_query_expansion.core.conformal_risk_control import (  # Module not found  # Module not found  # Module not found
            ClassificationScore,
            RegressionScore,
            create_conformal_system,
        )

        # Test score functions
        score_fn = ClassificationScore()
        y_pred = np.array([[0.1, 0.9], [0.7, 0.3]])
        y_true = np.array([1, 0])
        scores = score_fn(y_pred, y_true)
        print("✅ Classification scoring function works")

        reg_score_fn = RegressionScore()
        y_pred_reg = np.array([1.5, 2.8])
        y_true_reg = np.array([1.0, 3.0])
        reg_scores = reg_score_fn(y_pred_reg, y_true_reg)
        print("✅ Regression scoring function works")

        # Test system creation
        controller, score_fn = create_conformal_system(
            task_type="classification", alpha=0.1, seed=42
        )
        print("✅ Conformal system creation works")

        return True

    except Exception as e:
        print(f"❌ Conformal functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests with pre-installation checks."""
    print("🚀 EGW Query Expansion with Conformal Risk Control - Comprehensive Validation")
    print("=" * 80)

    # Step 1: Pre-installation environment validation
    validator = PreInstallationValidator()
    pre_install_passed = validator.run_all_validations()
    
    if validator.issues:
        validator.print_issues_and_remediation()
        
    critical_issues = [i for i in validator.issues if i.severity == 'critical']
    if critical_issues:
        print(f"\n❌ VALIDATION FAILED: {len(critical_issues)} critical issues must be resolved before proceeding.")
        print("\nPlease address the critical issues above and re-run this script.")
        return 1
        
    if not pre_install_passed:
        print("\n⚠️  Some validation warnings were found. Consider addressing them for optimal performance.")
    else:
        print("\n✅ Pre-installation validation completed successfully!")

    # Step 2: Dependency and functionality tests (only if dependencies might be available)
    print("\n" + "=" * 60)
    print("🧪 RUNNING DEPENDENCY AND FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("EGW Modules", test_egw_modules),
        ("Conformal Functionality", test_conformal_functionality),
        ("Configuration", test_configuration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 Validation Summary")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nDependency Tests: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! EGW Query Expansion is ready to use.")
        return 0
    else:
        failed = total - passed
        if failed == total:
            print(f"\n💡 All {failed} dependency tests failed - this is normal if dependencies aren't installed yet.")
            print("   Install dependencies with: pip install -r requirements.txt")
            print("   Then re-run this script to verify the installation.")
        else:
            print(f"⚠️  {failed}/{total} tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())