"""
Installation diagnostics module for EGW Query Expansion.

This module provides environment analysis capabilities that can be leveraged
by CLI tools and other diagnostic utilities.
"""

import importlib
import os
import platform
import subprocess
import sys
import warnings
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found

try:
    import pkg_resources
except ImportError:
    pkg_resources = None


class EnvironmentDiagnostics:
    """Core environment diagnostic functionality."""
    
    def __init__(self):
        self.system_info = self._gather_system_info()
        self.python_info = self._gather_python_info()
        
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }
        
    def _gather_python_info(self) -> Dict[str, Any]:
        """Gather Python environment information."""
        return {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": sys.version_info,
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, 'base_prefix', None),
            "path": sys.path,
            "modules": list(sys.modules.keys()),
        }
        
    def check_package_availability(self, package_name: str) -> Dict[str, Any]:
        """Check if a package is available and get its information."""
        result = {
            "name": package_name,
            "available": False,
            "version": None,
            "location": None,
            "dependencies": [],
            "import_error": None,
        }
        
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            result["available"] = True
            result["version"] = getattr(module, "__version__", "unknown")
            result["location"] = getattr(module, "__file__", None)
            
# # #             # Try to get package info from pkg_resources  # Module not found  # Module not found  # Module not found
            if pkg_resources:
                try:
                    dist = pkg_resources.get_distribution(package_name)
                    result["version"] = dist.version
                    result["location"] = dist.location
                    result["dependencies"] = [str(req) for req in dist.requires()]
                except pkg_resources.DistributionNotFound:
                    pass
                
        except ImportError as e:
            result["import_error"] = str(e)
            
        return result
        
    def check_core_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check all core EGW dependencies."""
        core_packages = [
            "numpy", "scipy", "torch", "faiss", "transformers", 
            "sentence_transformers", "ot", "sklearn", "datasets",
            "pandas", "yaml", "tqdm", "matplotlib", "seaborn",
            "plotly", "beir", "pytest", "jupyter", "z3",
            "msgspec", "pydantic", "orjson", "psutil", "httpx"
        ]
        
        results = {}
        for package in core_packages:
            # Handle special cases
            if package == "faiss":
                # Try both faiss-cpu and faiss-gpu
                faiss_result = self.check_package_availability("faiss")
                if not faiss_result["available"]:
                    # Also check for faiss-cpu specifically
                    try:
                        import faiss
                        faiss_result["available"] = True
                        faiss_result["version"] = getattr(faiss, "__version__", "unknown")
                    except ImportError:
                        pass
                results[package] = faiss_result
            elif package == "ot":
                # POT package imports as 'ot'
                results["POT"] = self.check_package_availability("ot")
            elif package == "sklearn":
                # scikit-learn imports as 'sklearn'
                results["scikit-learn"] = self.check_package_availability("sklearn")
            elif package == "yaml":
                # PyYAML imports as 'yaml'
                results["PyYAML"] = self.check_package_availability("yaml")
            else:
                results[package] = self.check_package_availability(package)
                
        return results
        
    def check_virtual_environment(self) -> Dict[str, Any]:
        """Check virtual environment status and details."""
        in_venv = (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        venv_info = {
            "in_virtual_env": in_venv,
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, 'base_prefix', None),
            "real_prefix": getattr(sys, 'real_prefix', None),
        }
        
        # Try to detect virtual environment type
        if in_venv:
            if os.path.exists(os.path.join(sys.prefix, "pyvenv.cfg")):
                venv_info["type"] = "venv"
            elif "conda" in sys.prefix or "anaconda" in sys.prefix:
                venv_info["type"] = "conda"
            elif hasattr(sys, 'real_prefix'):
                venv_info["type"] = "virtualenv"
            else:
                venv_info["type"] = "unknown"
        else:
            venv_info["type"] = "none"
            
        return venv_info
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system-level requirements."""
        system = platform.system().lower()
        requirements = {
            "system": system,
            "build_tools": {"available": False, "details": []},
            "python_dev": {"available": False, "details": []},
        }
        
        if system == "linux":
            # Check for GCC
            try:
                result = subprocess.run(
                    ["gcc", "--version"], 
                    capture_output=True, text=True, check=True
                )
                requirements["build_tools"]["available"] = True
                requirements["build_tools"]["details"].append(f"GCC: {result.stdout.split()[2]}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                requirements["build_tools"]["details"].append("GCC not found")
                
            # Check for Python development headers
            try:
                subprocess.run(
                    ["python3-config", "--includes"], 
                    capture_output=True, check=True
                )
                requirements["python_dev"]["available"] = True
                requirements["python_dev"]["details"].append("Python dev headers available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                requirements["python_dev"]["details"].append("Python dev headers not found")
                
        elif system == "darwin":  # macOS
            try:
                result = subprocess.run(
                    ["xcode-select", "--print-path"], 
                    capture_output=True, text=True, check=True
                )
                requirements["build_tools"]["available"] = True
                requirements["build_tools"]["details"].append(f"Xcode path: {result.stdout.strip()}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                requirements["build_tools"]["details"].append("Xcode command line tools not found")
                
        elif system == "windows":
            # Check for Visual Studio Build Tools
            vs_paths = [
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools",
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools", 
                "C:\\Program Files\\Microsoft Visual Studio\\2019\\Community",
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community",
            ]
            found_vs = [p for p in vs_paths if Path(p).exists()]
            if found_vs:
                requirements["build_tools"]["available"] = True
                requirements["build_tools"]["details"].extend(found_vs)
            else:
                requirements["build_tools"]["details"].append("Visual Studio Build Tools not found")
                
        return requirements
        
    def check_gpu_support(self) -> Dict[str, Any]:
        """Check GPU and CUDA support."""
        gpu_info = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_devices": [],
            "nvidia_driver": None,
        }
        
        # Check PyTorch CUDA support
        try:
            import torch
            gpu_info["cuda_available"] = torch.cuda.is_available()
            if gpu_info["cuda_available"]:
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["gpu_count"] = torch.cuda.device_count()
                gpu_info["gpu_devices"] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(torch.cuda.device_count())
                ]
        except ImportError:
            pass
            
        # Check nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split('\n')
            if lines:
                gpu_info["nvidia_driver"] = lines[0].split(',')[1].strip()
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return gpu_info
        
    def check_memory_disk_space(self) -> Dict[str, Any]:
        """Check memory and disk space availability."""
        resources = {}
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            resources["memory"] = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent,
            }
        except ImportError:
            resources["memory"] = {"error": "psutil not available"}
            
        # Disk space check
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            resources["disk"] = {
                "total_gb": total / (1024**3),
                "free_gb": free / (1024**3), 
                "used_gb": used / (1024**3),
            }
        except Exception as e:
            resources["disk"] = {"error": str(e)}
            
        return resources
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive diagnostic report."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return {
                "system_info": self.system_info,
                "python_info": self.python_info,
                "virtual_environment": self.check_virtual_environment(),
                "system_requirements": self.check_system_requirements(),
                "core_dependencies": self.check_core_dependencies(),
                "gpu_support": self.check_gpu_support(),
                "resources": self.check_memory_disk_space(),
            }


def diagnose_environment() -> Dict[str, Any]:
    """Convenience function to run full environment diagnostics."""
    diagnostics = EnvironmentDiagnostics()
    return diagnostics.generate_comprehensive_report()


def check_installation_readiness() -> Tuple[bool, List[str]]:
    """Check if the environment is ready for EGW installation."""
    diagnostics = EnvironmentDiagnostics()
    report = diagnostics.generate_comprehensive_report()
    
    issues = []
    
    # Check Python version
    version = sys.version_info
    if version.major != 3 or version.minor < 8 or version.minor > 12:
        issues.append(f"Unsupported Python version: {version.major}.{version.minor}")
        
    # Check virtual environment
    if not report["virtual_environment"]["in_virtual_env"]:
        issues.append("Not running in a virtual environment")
        
    # Check critical dependencies
    deps = report["core_dependencies"]
    critical = ["numpy", "torch"]
    for dep in critical:
        if dep in deps and not deps[dep]["available"]:
            issues.append(f"Critical dependency {dep} not available")
            
    # Check system requirements
    sys_req = report["system_requirements"]
    if not sys_req["build_tools"]["available"]:
        issues.append("Build tools not available (may cause compilation issues)")
        
    # Check resources
    resources = report["resources"]
    if "memory" in resources and "available_gb" in resources["memory"]:
        if resources["memory"]["available_gb"] < 2:
            issues.append("Low memory (< 2GB available)")
            
    if "disk" in resources and "free_gb" in resources["disk"]:
        if resources["disk"]["free_gb"] < 3:
            issues.append("Low disk space (< 3GB free)")
            
    return len(issues) == 0, issues


# Compatibility with CLI tool
def get_diagnostics() -> Dict[str, Any]:
    """Get diagnostic information (compatibility function)."""
    return diagnose_environment()