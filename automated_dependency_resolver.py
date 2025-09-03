#!/usr/bin/env python3
"""
Automated Dependency Resolver

This script detects the current Python environment, validates core dependencies,
and installs any missing packages with specific version constraints.

Features:
- Virtual environment detection and warnings
- Core dependency validation (numpy, scipy, scikit-learn)
- Automatic installation with version constraints
- Comprehensive error handling and informative messages
"""

import sys
import subprocess
import importlib
import importlib.util
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Tuple, Optional  # Module not found  # Module not found  # Module not found
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Core dependencies with version constraints
CORE_DEPENDENCIES = {
    'numpy': '>=1.21.0',
    'scipy': '>=1.7.0', 
    'scikit-learn': '>=1.0.0'
}

class EnvironmentDetector:
    """Detects Python environment and virtual environment status."""
    
    def __init__(self):
        self.python_executable = sys.executable
        self.base_prefix = getattr(sys, 'base_prefix', sys.prefix)
        self.prefix = sys.prefix
        
    def is_virtual_environment(self) -> bool:
        """Check if running in a virtual environment."""
        return (
            hasattr(sys, 'real_prefix') or  # virtualenv
            (hasattr(sys, 'base_prefix') and self.base_prefix != self.prefix)  # venv
        )
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get detailed environment information."""
        venv_name = "Unknown"
        if self.is_virtual_environment():
            venv_path = Path(self.prefix)
            venv_name = venv_path.name
            
        return {
            'python_executable': self.python_executable,
            'python_version': sys.version,
            'is_virtual_env': str(self.is_virtual_environment()),
            'virtual_env_name': venv_name,
            'sys_prefix': self.prefix,
            'base_prefix': self.base_prefix
        }

class DependencyValidator:
    """Validates and manages core dependencies."""
    
    def __init__(self):
        self.missing_deps = []
        self.installed_deps = {}
        
    def check_package(self, package_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a package is installed and get its version."""
        try:
            # Try importing the package
            if package_name == 'scikit-learn':
                import sklearn
                version = sklearn.__version__
            else:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'unknown')
            
            return True, version
        except ImportError:
            return False, None
        except Exception as e:
            logger.warning(f"Error checking {package_name}: {e}")
            return False, None
    
    def validate_version(self, package_name: str, installed_version: str, 
                        required_version: str) -> bool:
        """Validate if installed version meets requirements."""
        try:
# # #             from packaging import version as pkg_version  # Module not found  # Module not found  # Module not found
            
            # Parse version constraint (e.g., ">=1.21.0")
            if required_version.startswith('>='):
                min_version = required_version[2:]
                return pkg_version.parse(installed_version) >= pkg_version.parse(min_version)
            elif required_version.startswith('>'):
                min_version = required_version[1:]
                return pkg_version.parse(installed_version) > pkg_version.parse(min_version)
            elif required_version.startswith('=='):
                exact_version = required_version[2:]
                return pkg_version.parse(installed_version) == pkg_version.parse(exact_version)
            else:
                # Assume exact match if no operator
                return installed_version == required_version
                
        except Exception as e:
            logger.warning(f"Version comparison failed for {package_name}: {e}")
            return False
    
    def validate_dependencies(self) -> Dict[str, Dict]:
        """Validate all core dependencies."""
        results = {}
        
        for package, version_constraint in CORE_DEPENDENCIES.items():
            is_installed, installed_version = self.check_package(package)
            
            status = {
                'installed': is_installed,
                'version': installed_version,
                'required': version_constraint,
                'meets_requirement': False
            }
            
            if is_installed and installed_version:
                status['meets_requirement'] = self.validate_version(
                    package, installed_version, version_constraint
                )
                
                if status['meets_requirement']:
                    self.installed_deps[package] = installed_version
                    logger.info(f"✓ {package} {installed_version} (meets {version_constraint})")
                else:
                    self.missing_deps.append(package)
                    logger.warning(f"⚠ {package} {installed_version} (needs {version_constraint})")
            else:
                self.missing_deps.append(package)
                logger.warning(f"✗ {package} not found (needs {version_constraint})")
            
            results[package] = status
            
        return results

class DependencyInstaller:
    """Handles installation of missing dependencies."""
    
    def __init__(self, python_executable: str):
        self.python_executable = python_executable
        
    def install_package(self, package_name: str, version_constraint: str) -> bool:
        """Install a single package with version constraint."""
        package_spec = f"{package_name}{version_constraint}"
        
        try:
            logger.info(f"Installing {package_spec}...")
            
            # Use subprocess to call pip
            cmd = [
                self.python_executable, '-m', 'pip', 'install',
                '--upgrade', package_spec
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            logger.info(f"✓ Successfully installed {package_spec}")
            logger.debug(f"Installation output: {result.stdout}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Installation of {package_spec} timed out")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {package_spec}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error installing {package_spec}: {e}")
            return False
    
    def install_missing_dependencies(self, missing_deps: List[str]) -> Dict[str, bool]:
        """Install all missing dependencies."""
        results = {}
        
        if not missing_deps:
            logger.info("No missing dependencies to install")
            return results
            
        logger.info(f"Installing {len(missing_deps)} missing dependencies...")
        
        for package in missing_deps:
            version_constraint = CORE_DEPENDENCIES[package]
            success = self.install_package(package, version_constraint)
            results[package] = success
            
        return results

def print_virtual_env_guidance():
    """Print guidance for creating a virtual environment."""
    print("\n" + "="*60)
    print("VIRTUAL ENVIRONMENT SETUP GUIDANCE")
    print("="*60)
    print("\nIt's recommended to use a virtual environment to avoid conflicts.")
    print("Here's how to create and activate one:")
    print("\n1. Create virtual environment:")
    print("   python -m venv venv")
    print("\n2. Activate virtual environment:")
    print("   # On Linux/macOS:")
    print("   source venv/bin/activate")
    print("   # On Windows:")
    print("   venv\\Scripts\\activate")
    print("\n3. Then re-run this script:")
    print("   python automated_dependency_resolver.py")
    print("\n" + "="*60)

def main():
    """Main execution function."""
    print("="*60)
    print("AUTOMATED DEPENDENCY RESOLVER")
    print("="*60)
    
    # 1. Detect environment
    env_detector = EnvironmentDetector()
    env_info = env_detector.get_environment_info()
    
    print(f"\nPython Environment Detection:")
    print(f"  Python Executable: {env_info['python_executable']}")
    print(f"  Python Version: {env_info['python_version'].split()[0]}")
    print(f"  Virtual Environment: {env_info['is_virtual_env']}")
    
    if env_detector.is_virtual_environment():
        print(f"  Virtual Env Name: {env_info['virtual_env_name']}")
        logger.info("✓ Running in virtual environment")
    else:
        print("  ⚠️  WARNING: Not running in a virtual environment!")
        logger.warning("Not running in virtual environment")
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print_virtual_env_guidance()
            return 1
    
    # 2. Validate dependencies
    print(f"\nValidating Core Dependencies:")
    validator = DependencyValidator()
    validation_results = validator.validate_dependencies()
    
    # 3. Install missing dependencies if needed
    if validator.missing_deps:
        print(f"\nMissing Dependencies: {len(validator.missing_deps)}")
        for dep in validator.missing_deps:
            print(f"  - {dep} {CORE_DEPENDENCIES[dep]}")
            
        response = input(f"\nInstall missing dependencies? (Y/n): ").strip().lower()
        if response not in ['n', 'no']:
            installer = DependencyInstaller(env_detector.python_executable)
            install_results = installer.install_missing_dependencies(validator.missing_deps)
            
            # Re-validate after installation
            print(f"\nRe-validating dependencies after installation...")
            validator = DependencyValidator()
            validation_results = validator.validate_dependencies()
            
            # Report installation results
            print(f"\nInstallation Results:")
            for package, success in install_results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {package}")
        else:
            print("Skipping installation.")
    else:
        print("✓ All core dependencies are satisfied!")
    
    # 4. Final summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Environment: {'Virtual Env' if env_detector.is_virtual_environment() else 'System Python'}")
    print(f"Dependencies checked: {len(CORE_DEPENDENCIES)}")
    print(f"Dependencies satisfied: {len(validator.installed_deps)}")
    print(f"Dependencies missing: {len(validator.missing_deps)}")
    
    if not validator.missing_deps:
        print("\n✅ All dependencies are ready!")
        return 0
    else:
        print(f"\n⚠️  {len(validator.missing_deps)} dependencies still need attention")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)