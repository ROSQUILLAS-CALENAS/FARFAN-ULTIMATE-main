#!/usr/bin/env python3
"""
Installation Troubleshooting Tools for EGW Query Expansion System
Provides troubleshooting utilities for common installation issues.
"""

import os
import sys
import subprocess
import platform
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
import importlib


class InstallationTroubleshooter:
    """Installation troubleshooting system."""
    
    def __init__(self):
        self.system = platform.system()
        self.python_version = platform.python_version()
        self.architecture = platform.architecture()[0]
        
        # Common issues and their solutions
        self.known_issues = {
            'faiss_import_error': {
                'symptoms': ['ImportError: No module named faiss', 'ModuleNotFoundError: No module named \'faiss\''],
                'solutions': [
                    'pip install faiss-cpu',
                    'pip install --upgrade faiss-cpu',
                    'conda install -c conda-forge faiss-cpu'
                ],
                'description': 'FAISS library not installed or corrupted'
            },
            'torch_cuda_mismatch': {
                'symptoms': ['CUDA out of memory', 'No CUDA GPUs are available'],
                'solutions': [
                    'pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu',
                    'Set CUDA_VISIBLE_DEVICES="" to force CPU usage'
                ],
                'description': 'PyTorch CUDA configuration issues'
            },
            'numpy_version_conflict': {
                'symptoms': ['numpy.dtype size changed', 'ValueError: numpy.dtype size changed'],
                'solutions': [
                    'pip install --upgrade numpy',
                    'pip install --force-reinstall numpy',
                    'pip uninstall numpy && pip install numpy'
                ],
                'description': 'NumPy version compatibility issues'
            },
            'transformers_tokenizer_parallelism': {
                'symptoms': ['huggingface/tokenizers: The current process', 'tokenizers parallelism'],
                'solutions': [
                    'export TOKENIZERS_PARALLELISM=false',
                    'os.environ["TOKENIZERS_PARALLELISM"] = "false"'
                ],
                'description': 'HuggingFace tokenizers parallelism warning'
            },
            'memory_issues': {
                'symptoms': ['MemoryError', 'Out of memory', 'Killed'],
                'solutions': [
                    'Increase virtual memory/swap',
                    'Use smaller batch sizes',
                    'Enable gradient checkpointing',
                    'Use CPU instead of GPU for large models'
                ],
                'description': 'Insufficient memory for operations'
            }
        }
    
    def check_pip_configuration(self) -> Dict[str, Any]:
        """Check pip configuration and health."""
        result = {'status': 'unknown', 'details': {}}
        
        try:
            # Check pip version
            pip_result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                      capture_output=True, text=True, timeout=30)
            result['pip_version'] = pip_result.stdout.strip() if pip_result.returncode == 0 else 'unknown'
            
            # Check pip configuration
            config_result = subprocess.run([sys.executable, '-m', 'pip', 'config', 'list'], 
                                         capture_output=True, text=True, timeout=30)
            result['pip_config'] = config_result.stdout.strip() if config_result.returncode == 0 else 'default'
            
            # Check if we can reach PyPI
            list_result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                       capture_output=True, text=True, timeout=60)
            result['pip_list_works'] = list_result.returncode == 0
            
            result['status'] = 'healthy' if result['pip_list_works'] else 'issues'
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment health."""
        env_info = {
            'python_executable': sys.executable,
            'python_version': self.python_version,
            'python_path': sys.path[:5],  # First 5 entries
            'site_packages': [],
            'virtual_env': os.environ.get('VIRTUAL_ENV'),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        }
        
        # Find site-packages directories
        try:
            import site
            env_info['site_packages'] = site.getsitepackages()
        except:
            pass
        
        # Check if we're in a virtual environment
        env_info['in_virtualenv'] = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        return env_info
    
    def check_system_dependencies(self) -> Dict[str, Any]:
        """Check system-level dependencies."""
        system_deps = {}
        
        if self.system == 'Linux':
            # Check for common Linux dependencies
            commands = {
                'gcc': 'gcc --version',
                'g++': 'g++ --version', 
                'make': 'make --version',
                'cmake': 'cmake --version',
                'pkg-config': 'pkg-config --version',
            }
            
            for name, cmd in commands.items():
                try:
                    result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
                    system_deps[name] = {
                        'available': result.returncode == 0,
                        'version': result.stdout.split('\n')[0] if result.returncode == 0 else None
                    }
                except:
                    system_deps[name] = {'available': False}
        
        elif self.system == 'Windows':
            # Check for Windows build tools
            system_deps['visual_cpp'] = {'available': 'unknown', 'note': 'Check Visual Studio Build Tools'}
        
        elif self.system == 'Darwin':
            # Check for Xcode command line tools
            try:
                result = subprocess.run(['xcode-select', '--print-path'], capture_output=True, text=True, timeout=10)
                system_deps['xcode_tools'] = {
                    'available': result.returncode == 0,
                    'path': result.stdout.strip() if result.returncode == 0 else None
                }
            except:
                system_deps['xcode_tools'] = {'available': False}
        
        return system_deps
    
    def test_import_sequence(self) -> Dict[str, Any]:
        """Test importing packages in the correct sequence."""
        import_sequence = [
            'numpy',
            'scipy',
            'sklearn',
            'torch', 
            'transformers',
            'sentence_transformers',
            'faiss',
            'ot',  # POT
            'datasets',
            'pandas',
            'yaml',
        ]
        
        results = {}
        
        for package in import_sequence:
            try:
                module = importlib.import_module(package)
                results[package] = {
                    'status': 'success',
                    'version': getattr(module, '__version__', 'unknown'),
                    'location': getattr(module, '__file__', 'unknown')
                }
            except ImportError as e:
                results[package] = {
                    'status': 'import_error',
                    'error': str(e)
                }
            except Exception as e:
                results[package] = {
                    'status': 'other_error', 
                    'error': str(e)
                }
        
        return results
    
    def test_functionality_sequence(self) -> Dict[str, Any]:
        """Test basic functionality in sequence."""
        tests = {}
        
        # Test 1: NumPy basic operations
        try:
            import numpy as np
            a = np.array([1, 2, 3])
            b = np.array([4, 5, 6])
            c = np.dot(a, b)
            tests['numpy_basic'] = {'status': 'success', 'result': c}
        except Exception as e:
            tests['numpy_basic'] = {'status': 'error', 'error': str(e)}
        
        # Test 2: PyTorch tensor operations
        try:
            import torch
            x = torch.tensor([1.0, 2.0, 3.0])
            y = torch.tensor([4.0, 5.0, 6.0])
            z = torch.dot(x, y)
            tests['torch_basic'] = {'status': 'success', 'result': z.item()}
        except Exception as e:
            tests['torch_basic'] = {'status': 'error', 'error': str(e)}
        
        # Test 3: FAISS index creation
        try:
            import faiss
            import numpy as np
            d = 64
            index = faiss.IndexFlatL2(d)
            vectors = np.random.random((10, d)).astype('float32')
            index.add(vectors)
            tests['faiss_basic'] = {'status': 'success', 'index_size': index.ntotal}
        except Exception as e:
            tests['faiss_basic'] = {'status': 'error', 'error': str(e)}
        
        # Test 4: Transformers tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
            tokens = tokenizer.tokenize("test")
            tests['transformers_basic'] = {'status': 'success', 'tokens': tokens}
        except Exception as e:
            tests['transformers_basic'] = {'status': 'error', 'error': str(e)}
        
        return tests
    
    def diagnose_issue(self, error_message: str) -> List[Dict[str, Any]]:
        """Diagnose specific issue based on error message."""
        suggestions = []
        
        for issue_name, issue_info in self.known_issues.items():
            for symptom in issue_info['symptoms']:
                if symptom.lower() in error_message.lower():
                    suggestions.append({
                        'issue': issue_name,
                        'description': issue_info['description'],
                        'solutions': issue_info['solutions']
                    })
                    break
        
        if not suggestions:
            # Generic troubleshooting steps
            suggestions.append({
                'issue': 'generic_error',
                'description': 'Unknown error - try general troubleshooting steps',
                'solutions': [
                    'pip install --upgrade pip',
                    'pip install --upgrade setuptools wheel',
                    'pip cache purge',
                    'Restart Python environment',
                    'Check system dependencies'
                ]
            })
        
        return suggestions
    
    def run_comprehensive_troubleshooting(self) -> Dict[str, Any]:
        """Run comprehensive troubleshooting analysis."""
        print("🔧 Running comprehensive troubleshooting analysis...")
        
        analysis = {
            'system_info': {
                'platform': self.system,
                'python_version': self.python_version,
                'architecture': self.architecture
            },
            'pip_health': self.check_pip_configuration(),
            'python_env': self.check_python_environment(),
            'system_deps': self.check_system_dependencies(),
            'import_tests': self.test_import_sequence(),
            'functionality_tests': self.test_functionality_sequence(),
        }
        
        return analysis
    
    def generate_troubleshooting_report(self, analysis: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive troubleshooting report."""
        if analysis is None:
            analysis = self.run_comprehensive_troubleshooting()
        
        report = []
        report.append("=" * 70)
        report.append("Installation Troubleshooting Report")
        report.append("=" * 70)
        
        # System info
        sys_info = analysis['system_info']
        report.append(f"\n🖥️  System: {sys_info['platform']} {sys_info['architecture']}")
        report.append(f"   Python: {sys_info['python_version']}")
        
        # Environment info
        env = analysis['python_env']
        report.append(f"\n🐍 Python Environment:")
        report.append(f"   Executable: {env['python_executable']}")
        report.append(f"   Virtual Environment: {'Yes' if env['in_virtualenv'] else 'No'}")
        if env['virtual_env']:
            report.append(f"   VIRTUAL_ENV: {env['virtual_env']}")
        
        # Pip health
        pip = analysis['pip_health']
        pip_status = "✅" if pip['status'] == 'healthy' else "❌"
        report.append(f"\n📦 Pip Health: {pip_status} {pip['status']}")
        
        # Import test results
        report.append(f"\n📚 Import Tests:")
        imports = analysis['import_tests']
        success_count = sum(1 for result in imports.values() if result['status'] == 'success')
        total_count = len(imports)
        
        for package, result in imports.items():
            if result['status'] == 'success':
                report.append(f"   ✅ {package}: {result.get('version', 'unknown')}")
            else:
                report.append(f"   ❌ {package}: {result.get('error', 'unknown error')}")
        
        # Functionality tests
        report.append(f"\n🧪 Functionality Tests:")
        func_tests = analysis['functionality_tests'] 
        func_success = sum(1 for result in func_tests.values() if result['status'] == 'success')
        func_total = len(func_tests)
        
        for test, result in func_tests.items():
            status = "✅" if result['status'] == 'success' else "❌"
            report.append(f"   {status} {test}")
        
        # Summary and recommendations
        report.append(f"\n📊 Summary:")
        report.append(f"   Import Tests: {success_count}/{total_count}")
        report.append(f"   Functionality Tests: {func_success}/{func_total}")
        
        # Generate recommendations based on failures
        failed_imports = [pkg for pkg, result in imports.items() if result['status'] != 'success']
        if failed_imports:
            report.append(f"\n🔧 Recommendations:")
            report.append(f"   Failed imports: {', '.join(failed_imports)}")
            report.append(f"   1. Try: pip install --upgrade {' '.join(failed_imports)}")
            report.append(f"   2. Check requirements.txt compatibility")
            report.append(f"   3. Consider using conda for problematic packages")
        
        if success_count == total_count and func_success == func_total:
            report.append(f"\n🎉 All tests passed! Installation appears healthy.")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    troubleshooter = InstallationTroubleshooter()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--json':
            # JSON output for CI/CD
            analysis = troubleshooter.run_comprehensive_troubleshooting()
            print(json.dumps(analysis, indent=2, default=str))
            return
        elif sys.argv[1] == '--diagnose':
            # Diagnose specific error
            if len(sys.argv) > 2:
                error_msg = ' '.join(sys.argv[2:])
                suggestions = troubleshooter.diagnose_issue(error_msg)
                
                print("🔍 Diagnostic Suggestions:")
                for suggestion in suggestions:
                    print(f"\nIssue: {suggestion['issue']}")
                    print(f"Description: {suggestion['description']}")
                    print("Solutions:")
                    for i, solution in enumerate(suggestion['solutions'], 1):
                        print(f"  {i}. {solution}")
            else:
                print("Usage: python installation_troubleshooting.py --diagnose 'error message'")
            return
    
    # Default: comprehensive report
    report = troubleshooter.generate_troubleshooting_report()
    print(report)
    
    # Return appropriate exit code
    analysis = troubleshooter.run_comprehensive_troubleshooting()
    import_success = sum(1 for result in analysis['import_tests'].values() if result['status'] == 'success')
    import_total = len(analysis['import_tests'])
    
    if import_success < import_total:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())