#!/usr/bin/env python3
"""
Installation Diagnostics Module for EGW Query Expansion System
Provides comprehensive diagnostics for installation validation across environments.
"""

import importlib
import platform
import sys
import subprocess
import json
from typing import Dict, List, Any, Tuple, Optional
import os


class InstallationDiagnostics:
    """Comprehensive installation diagnostics system."""
    
    def __init__(self):
        self.results = {}
        self.system_info = self._collect_system_info()
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'architecture': platform.architecture(),
            'node': platform.node(),
        }
    
    def check_core_dependencies(self) -> Dict[str, Any]:
        """Check core dependencies installation."""
        core_deps = {
            'numpy': 'numpy',
            'scipy': 'scipy.sparse',
            'scikit-learn': 'sklearn',
            'torch': 'torch',
            'faiss': 'faiss',
            'transformers': 'transformers',
            'sentence_transformers': 'sentence_transformers',
            'POT': 'ot',
            'datasets': 'datasets',
            'pandas': 'pandas',
            'pyyaml': 'yaml',
        }
        
        results = {}
        for name, module in core_deps.items():
            try:
                mod = importlib.import_module(module)
                version = getattr(mod, '__version__', 'unknown')
                results[name] = {
                    'status': 'success',
                    'version': version,
                    'module': module,
                    'location': getattr(mod, '__file__', 'unknown')
                }
            except ImportError as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'module': module
                }
        
        return results
    
    def check_optional_dependencies(self) -> Dict[str, Any]:
        """Check optional dependencies."""
        optional_deps = {
            'spacy': 'spacy',
            'nltk': 'nltk',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'beir': 'beir',
            'z3': 'z3',
            'redis': 'redis',
            'ray': 'ray',
        }
        
        results = {}
        for name, module in optional_deps.items():
            try:
                mod = importlib.import_module(module)
                version = getattr(mod, '__version__', 'unknown')
                results[name] = {
                    'status': 'success',
                    'version': version,
                    'module': module
                }
            except ImportError:
                results[name] = {
                    'status': 'missing',
                    'module': module
                }
        
        return results
    
    def test_functionality(self) -> Dict[str, Any]:
        """Test basic functionality of core libraries."""
        tests = {}
        
        # Test NumPy
        try:
            import numpy as np
            a = np.random.rand(100, 50)
            b = np.random.rand(50, 25)
            c = np.dot(a, b)
            tests['numpy_operations'] = {'status': 'success', 'shape_result': c.shape}
        except Exception as e:
            tests['numpy_operations'] = {'status': 'error', 'error': str(e)}
        
        # Test PyTorch
        try:
            import torch
            x = torch.randn(10, 5)
            y = torch.mm(x, torch.randn(5, 3))
            tests['pytorch_operations'] = {
                'status': 'success', 
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except Exception as e:
            tests['pytorch_operations'] = {'status': 'error', 'error': str(e)}
        
        # Test FAISS
        try:
            import faiss
            import numpy as np
            d = 64
            nb = 100
            xb = np.random.random((nb, d)).astype('float32')
            index = faiss.IndexFlatL2(d)
            index.add(xb)
            tests['faiss_operations'] = {
                'status': 'success',
                'index_size': index.ntotal,
                'gpu_available': hasattr(faiss, 'StandardGpuResources')
            }
        except Exception as e:
            tests['faiss_operations'] = {'status': 'error', 'error': str(e)}
        
        # Test Transformers
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
            tokens = tokenizer.tokenize("Hello world")
            tests['transformers_operations'] = {
                'status': 'success',
                'tokenizer_vocab_size': tokenizer.vocab_size,
                'sample_tokens': tokens[:5]
            }
        except Exception as e:
            tests['transformers_operations'] = {'status': 'error', 'error': str(e)}
        
        return tests
    
    def check_environment_variables(self) -> Dict[str, Any]:
        """Check relevant environment variables."""
        env_vars = [
            'CUDA_VISIBLE_DEVICES',
            'PYTORCH_CUDA_ALLOC_CONF',
            'TOKENIZERS_PARALLELISM',
            'OMP_NUM_THREADS',
            'FAISS_OPT_LEVEL',
            'TRANSFORMERS_CACHE',
            'HF_HOME',
        ]
        
        return {var: os.environ.get(var, 'not_set') for var in env_vars}
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        import shutil
        
        paths = ['.', '/tmp', os.path.expanduser('~/.cache')]
        results = {}
        
        for path in paths:
            if os.path.exists(path):
                try:
                    usage = shutil.disk_usage(path)
                    results[path] = {
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'free_percent': round((usage.free / usage.total) * 100, 1)
                    }
                except Exception as e:
                    results[path] = {'error': str(e)}
        
        return results
    
    def check_memory(self) -> Dict[str, Any]:
        """Check available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent,
                'free_gb': round((memory.total - memory.used) / (1024**3), 2)
            }
        except ImportError:
            # Fallback without psutil
            return {'status': 'psutil_not_available'}
    
    def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic tests."""
        print("🔍 Running comprehensive installation diagnostics...")
        
        diagnostics = {
            'system_info': self.system_info,
            'core_dependencies': self.check_core_dependencies(),
            'optional_dependencies': self.check_optional_dependencies(),
            'functionality_tests': self.test_functionality(),
            'environment_variables': self.check_environment_variables(),
            'disk_space': self.check_disk_space(),
            'memory': self.check_memory(),
        }
        
        return diagnostics
    
    def generate_report(self) -> str:
        """Generate a human-readable diagnostic report."""
        diagnostics = self.run_comprehensive_diagnostics()
        
        report = []
        report.append("=" * 70)
        report.append("EGW Query Expansion - Installation Diagnostics Report")
        report.append("=" * 70)
        
        # System Info
        report.append("\n🖥️  System Information:")
        for key, value in diagnostics['system_info'].items():
            report.append(f"   {key}: {value}")
        
        # Core Dependencies
        report.append("\n📦 Core Dependencies:")
        core_success = 0
        core_total = 0
        for name, info in diagnostics['core_dependencies'].items():
            core_total += 1
            if info['status'] == 'success':
                core_success += 1
                report.append(f"   ✅ {name}: {info['version']}")
            else:
                report.append(f"   ❌ {name}: {info.get('error', 'unknown error')}")
        
        # Functionality Tests
        report.append("\n🧪 Functionality Tests:")
        func_success = 0
        func_total = 0
        for test, result in diagnostics['functionality_tests'].items():
            func_total += 1
            if result['status'] == 'success':
                func_success += 1
                report.append(f"   ✅ {test}")
            else:
                report.append(f"   ❌ {test}: {result.get('error', 'unknown error')}")
        
        # Memory and Disk
        if 'total_gb' in diagnostics['memory']:
            mem = diagnostics['memory']
            report.append(f"\n💾 Memory: {mem['available_gb']:.1f}GB available of {mem['total_gb']:.1f}GB total")
        
        # Summary
        report.append(f"\n📊 Summary:")
        report.append(f"   Core Dependencies: {core_success}/{core_total}")
        report.append(f"   Functionality Tests: {func_success}/{func_total}")
        
        if core_success == core_total and func_success == func_total:
            report.append("\n🎉 Installation appears to be working correctly!")
        else:
            report.append(f"\n⚠️  Issues detected. Please check failed components above.")
        
        return "\n".join(report)


def main():
    """Main entry point for standalone execution."""
    diagnostics = InstallationDiagnostics()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        # Output JSON for CI/CD consumption
        result = diagnostics.run_comprehensive_diagnostics()
        print(json.dumps(result, indent=2, default=str))
    else:
        # Output human-readable report
        report = diagnostics.generate_report()
        print(report)
        
        # Return exit code based on core dependencies
        result = diagnostics.run_comprehensive_diagnostics()
        core_deps = result['core_dependencies']
        failed_core = [name for name, info in core_deps.items() if info['status'] != 'success']
        
        if failed_core:
            print(f"\n❌ Failed core dependencies: {', '.join(failed_core)}")
            return 1
        else:
            return 0


if __name__ == "__main__":
    sys.exit(main())