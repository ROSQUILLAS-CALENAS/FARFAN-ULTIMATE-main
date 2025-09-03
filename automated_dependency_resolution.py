#!/usr/bin/env python3
"""
Automated Dependency Resolution Script for EGW Query Expansion System
Handles dependency conflicts and installation issues across different environments.
"""

import subprocess
import sys
import platform
import json
import os
# # # from typing import List, Dict, Any, Optional, Tuple  # Module not found  # Module not found  # Module not found
import importlib
import pkg_resources


class DependencyResolver:
    """Automated dependency resolution system."""
    
    def __init__(self):
        self.python_version = platform.python_version()
        self.system = platform.system()
        self.machine = platform.machine()
        
        # Core dependencies with version constraints by Python version
        self.core_dependencies = {
            'numpy': {
                '3.8': 'numpy>=1.20.0,<1.25.0',
                '3.9': 'numpy>=1.21.0,<1.26.0', 
                '3.10': 'numpy>=1.21.0,<1.27.0',
                '3.11': 'numpy>=1.23.0,<1.27.0',
                'default': 'numpy>=1.24.0'
            },
            'scipy': {
                '3.8': 'scipy>=1.7.0,<1.12.0',
                '3.9': 'scipy>=1.8.0,<1.12.0',
                '3.10': 'scipy>=1.9.0,<1.13.0', 
                '3.11': 'scipy>=1.9.0,<1.13.0',
                'default': 'scipy>=1.11.0'
            },
            'scikit-learn': {
                '3.8': 'scikit-learn>=1.1.0,<1.4.0',
                '3.9': 'scikit-learn>=1.2.0,<1.4.0',
                '3.10': 'scikit-learn>=1.3.0,<1.5.0',
                '3.11': 'scikit-learn>=1.3.0,<1.5.0',
                'default': 'scikit-learn>=1.3.0'
            },
            'torch': {
                '3.8': 'torch>=1.13.0,<2.1.0',
                '3.9': 'torch>=2.0.0,<2.2.0',
                '3.10': 'torch>=2.0.0,<2.2.0',
                '3.11': 'torch>=2.0.0,<2.2.0', 
                'default': 'torch>=2.0.0'
            },
        }
        
        # Platform-specific package mappings
        self.platform_packages = {
            'faiss': {
                'Windows': 'faiss-cpu',
                'Darwin': 'faiss-cpu',
                'Linux': 'faiss-cpu',  # or faiss-gpu if CUDA available
            }
        }
    
    def get_python_major_minor(self) -> str:
        """Get major.minor Python version."""
        version_parts = self.python_version.split('.')
        return f"{version_parts[0]}.{version_parts[1]}"
    
    def resolve_package_version(self, package: str) -> str:
        """Resolve package version based on Python version."""
        if package in self.core_dependencies:
            versions = self.core_dependencies[package]
            py_version = self.get_python_major_minor()
            return versions.get(py_version, versions['default'])
        
        # Platform-specific resolution
        if package in self.platform_packages:
            platform_map = self.platform_packages[package]
            return platform_map.get(self.system, package)
        
        return package
    
    def check_package_installed(self, package: str) -> Tuple[bool, Optional[str]]:
        """Check if a package is installed and get its version."""
        try:
            if package == 'faiss-cpu':
                # Special handling for FAISS
                import faiss
                return True, getattr(faiss, '__version__', 'unknown')
            
            module_name = package.replace('-', '_')
            if module_name == 'scikit_learn':
                module_name = 'sklearn'
            elif module_name == 'POT':
                module_name = 'ot'
            
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            return True, version
        except ImportError:
            return False, None
    
    def install_package(self, package_spec: str, timeout: int = 300) -> Tuple[bool, str]:
        """Install a package with proper error handling."""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', package_spec, '--no-cache-dir']
            
            # Add extra index URL for PyTorch if needed
            if 'torch' in package_spec:
                cmd.extend(['--extra-index-url', 'https://download.pytorch.org/whl/cpu'])
            
            print(f"Installing: {package_spec}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Installation timeout ({timeout}s)"
        except Exception as e:
            return False, str(e)
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version."""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except:
            return False
    
    def install_core_dependencies(self) -> Dict[str, Any]:
        """Install core dependencies with conflict resolution."""
        results = {}
        
        # First upgrade pip
        print("🔧 Upgrading pip...")
        pip_upgraded = self.upgrade_pip()
        results['pip_upgrade'] = pip_upgraded
        
        # Install core packages in order
        core_packages = ['numpy', 'scipy', 'scikit-learn', 'torch']
        
        for package in core_packages:
            package_spec = self.resolve_package_version(package)
            installed, version = self.check_package_installed(package)
            
            if not installed:
                success, output = self.install_package(package_spec)
                results[package] = {
                    'status': 'installed' if success else 'failed',
                    'package_spec': package_spec,
                    'output': output[:500] if output else None  # Truncate long output
                }
            else:
                results[package] = {
                    'status': 'already_installed',
                    'version': version,
                    'package_spec': package_spec
                }
        
        # Handle FAISS separately (platform-specific)
        faiss_package = self.resolve_package_version('faiss')
        installed, version = self.check_package_installed('faiss-cpu')
        
        if not installed:
            success, output = self.install_package(faiss_package)
            results['faiss'] = {
                'status': 'installed' if success else 'failed',
                'package_spec': faiss_package,
                'output': output[:500] if output else None
            }
        else:
            results['faiss'] = {
                'status': 'already_installed',
                'version': version,
                'package_spec': faiss_package
            }
        
        return results
    
    def install_additional_dependencies(self) -> Dict[str, Any]:
# # #         """Install additional dependencies from requirements."""  # Module not found  # Module not found  # Module not found
        additional_packages = [
            'transformers>=4.35.0',
            'sentence-transformers>=2.2.2',
            'POT>=0.9.1',
            'datasets>=2.14.0',
            'pandas>=1.5.0',
            'pyyaml>=5.1',
            'tqdm>=4.66.0',
        ]
        
        results = {}
        
        for package_spec in additional_packages:
            package_name = package_spec.split('>=')[0].split('==')[0]
            installed, version = self.check_package_installed(package_name)
            
            if not installed:
                success, output = self.install_package(package_spec)
                results[package_name] = {
                    'status': 'installed' if success else 'failed',
                    'package_spec': package_spec,
                    'output': output[:200] if output else None
                }
            else:
                results[package_name] = {
                    'status': 'already_installed',
                    'version': version,
                    'package_spec': package_spec
                }
        
        return results
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect package conflicts."""
        conflicts = []
        
        try:
            working_set = pkg_resources.working_set
            
            # Check for version conflicts
            for dist in working_set:
                try:
                    dist.check_version_conflict()
                except pkg_resources.VersionConflict as e:
                    conflicts.append({
                        'type': 'version_conflict',
                        'package': dist.project_name,
                        'error': str(e)
                    })
                except Exception as e:
                    conflicts.append({
                        'type': 'unknown_conflict',
                        'package': dist.project_name,
                        'error': str(e)
                    })
        except Exception:
            # If pkg_resources fails, skip conflict detection
            pass
        
        return conflicts
    
    def resolve_dependencies(self) -> Dict[str, Any]:
        """Run complete dependency resolution."""
        print("🚀 Starting automated dependency resolution...")
        print(f"Python {self.python_version} on {self.system} {self.machine}")
        
        results = {
            'system_info': {
                'python_version': self.python_version,
                'system': self.system,
                'machine': self.machine,
            },
            'core_installation': self.install_core_dependencies(),
            'additional_installation': self.install_additional_dependencies(),
            'conflicts': self.detect_conflicts(),
        }
        
        return results
    
    def generate_troubleshooting_report(self, results: Dict[str, Any]) -> str:
        """Generate troubleshooting report."""
        report = []
        report.append("=" * 70)
        report.append("Automated Dependency Resolution Report")
        report.append("=" * 70)
        
        # System info
        info = results['system_info']
        report.append(f"\n🖥️  System: Python {info['python_version']} on {info['system']} {info['machine']}")
        
        # Core installation results
        report.append("\n📦 Core Dependencies:")
        core_success = 0
        core_total = 0
        
        for package, result in results['core_installation'].items():
            if package == 'pip_upgrade':
                continue
            core_total += 1
            if result['status'] in ['installed', 'already_installed']:
                core_success += 1
                status = "✅"
            else:
                status = "❌"
            
            report.append(f"   {status} {package}: {result['status']}")
            if result['status'] == 'failed' and result.get('output'):
                report.append(f"      Error: {result['output'][:100]}...")
        
        # Additional dependencies
        additional_success = 0
        additional_total = 0
        
        for package, result in results['additional_installation'].items():
            additional_total += 1
            if result['status'] in ['installed', 'already_installed']:
                additional_success += 1
        
        report.append(f"\n📚 Additional Dependencies: {additional_success}/{additional_total} successful")
        
        # Conflicts
        conflicts = results.get('conflicts', [])
        if conflicts:
            report.append(f"\n⚠️  Detected {len(conflicts)} potential conflicts:")
            for conflict in conflicts[:3]:  # Show first 3
                report.append(f"   - {conflict['package']}: {conflict['error'][:50]}...")
        
        # Summary
        report.append(f"\n📊 Summary:")
        report.append(f"   Core Dependencies: {core_success}/{core_total}")
        report.append(f"   Additional Dependencies: {additional_success}/{additional_total}")
        
        if core_success == core_total:
            report.append("\n🎉 Core dependencies resolved successfully!")
        else:
            report.append(f"\n⚠️  {core_total - core_success} core dependencies failed. Check errors above.")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    resolver = DependencyResolver()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        # JSON output for CI/CD
        results = resolver.resolve_dependencies()
        print(json.dumps(results, indent=2, default=str))
    else:
        # Human-readable report
        results = resolver.resolve_dependencies()
        report = resolver.generate_troubleshooting_report(results)
        print(report)
        
        # Check if core dependencies were successful
        core_results = results['core_installation']
        failed_core = []
        for package, result in core_results.items():
            if package != 'pip_upgrade' and result['status'] == 'failed':
                failed_core.append(package)
        
        if failed_core:
            print(f"\n❌ Failed to install core dependencies: {', '.join(failed_core)}")
            return 1
        else:
            return 0


if __name__ == "__main__":
    sys.exit(main())