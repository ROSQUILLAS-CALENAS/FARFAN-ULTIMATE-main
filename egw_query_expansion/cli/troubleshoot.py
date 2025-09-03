#!/usr/bin/env python3
"""
CLI troubleshooting tool for EGW Query Expansion installation.

This tool provides interactive guidance for diagnosing and resolving
installation issues, leveraging environment analysis capabilities.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import the installation diagnostics module
try:
    from ..installation_diagnostics import (
        EnvironmentDiagnostics as BaseEnvironmentDiagnostics,
        diagnose_environment,
        check_installation_readiness,
    )
    HAS_DIAGNOSTICS_MODULE = True
except ImportError:
    HAS_DIAGNOSTICS_MODULE = False


class Colors:
    """ANSI color codes for terminal output."""
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class DiagnosticResult:
    """Result of a diagnostic check."""
    
    def __init__(
        self,
        name: str,
        passed: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        remediation: Optional[str] = None,
    ):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.remediation = remediation
        self.timestamp = datetime.now().isoformat()


class EnvironmentAnalyzer:
    """Analyzes the current environment and identifies issues."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[DiagnosticResult] = []
        
        # Initialize base diagnostics if available
        self.base_diagnostics = None
        if HAS_DIAGNOSTICS_MODULE:
            try:
                self.base_diagnostics = BaseEnvironmentDiagnostics()
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not initialize base diagnostics: {e}")
        
    def log(self, message: str, color: str = Colors.WHITE) -> None:
        """Log a message with optional color."""
        if self.verbose:
            print(f"{color}{message}{Colors.RESET}")
            
    def add_result(self, result: DiagnosticResult) -> None:
        """Add a diagnostic result."""
        self.results.append(result)
        
    def check_python_version(self) -> DiagnosticResult:
        """Check Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        # Supported versions: 3.8 - 3.12
        if version.major != 3 or version.minor < 8 or version.minor > 12:
            return DiagnosticResult(
                name="Python Version",
                passed=False,
                message=f"Unsupported Python version: {version_str}",
                details={"version": version_str, "required": "3.8-3.12"},
                remediation="""Install a supported Python version:
                
1. Using pyenv:
   curl https://pyenv.run | bash
   pyenv install 3.11.7
   pyenv global 3.11.7

2. Using conda:
   conda create -n egw-env python=3.11
   conda activate egw-env

3. System package manager (varies by OS)"""
            )
            
        elif version.minor >= 13:
            return DiagnosticResult(
                name="Python Version",
                passed=False,
                message=f"Python {version_str} has limited package support",
                details={"version": version_str, "issues": ["Some dependencies unavailable"]},
                remediation="Consider downgrading to Python 3.11 for full compatibility"
            )
        else:
            return DiagnosticResult(
                name="Python Version",
                passed=True,
                message=f"Python version {version_str} is supported",
                details={"version": version_str}
            )
            
    def check_virtual_environment(self) -> DiagnosticResult:
        """Check virtual environment status."""
        in_venv = (
            hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )
        
        if not in_venv:
            return DiagnosticResult(
                name="Virtual Environment",
                passed=False,
                message="Not running in a virtual environment",
                details={"prefix": sys.prefix, "base_prefix": getattr(sys, 'base_prefix', None)},
                remediation="""Create and activate a virtual environment:

1. Create environment:
   python -m venv venv

2. Activate (Linux/Mac):
   source venv/bin/activate
   
3. Activate (Windows):
   venv\\Scripts\\activate

4. Verify:
   which python
   which pip"""
            )
        else:
            return DiagnosticResult(
                name="Virtual Environment",
                passed=True,
                message=f"Running in virtual environment: {sys.prefix}",
                details={"prefix": sys.prefix}
            )
            
    def check_pip_version(self) -> DiagnosticResult:
        """Check pip version and functionality."""
        try:
            import pip
            pip_version = pip.__version__
            
            # Check if pip is recent enough (>= 20.0 for dependency resolver)
            version_parts = [int(x) for x in pip_version.split('.')]
            if version_parts[0] < 20:
                return DiagnosticResult(
                    name="Pip Version",
                    passed=False,
                    message=f"Pip version {pip_version} is too old",
                    details={"version": pip_version, "required": ">=20.0"},
                    remediation="Upgrade pip: python -m pip install --upgrade pip"
                )
                
            return DiagnosticResult(
                name="Pip Version", 
                passed=True,
                message=f"Pip version {pip_version} is adequate",
                details={"version": pip_version}
            )
            
        except ImportError:
            return DiagnosticResult(
                name="Pip Version",
                passed=False,
                message="Pip is not available",
                remediation="Install pip: python -m ensurepip --upgrade"
            )
            
    def check_system_dependencies(self) -> DiagnosticResult:
        """Check system-level dependencies."""
        system = platform.system().lower()
        issues = []
        suggestions = []
        
        if system == "linux":
            # Check for build tools
            try:
                subprocess.run(["gcc", "--version"], 
                              capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                issues.append("GCC compiler not found")
                suggestions.append("Install build tools: sudo apt-get install build-essential")
                
            # Check for Python dev headers
            try:
                subprocess.run(["python3-config", "--includes"], 
                              capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                issues.append("Python development headers not found")
                suggestions.append("Install Python dev: sudo apt-get install python3-dev")
                
        elif system == "darwin":  # macOS
            try:
                subprocess.run(["xcode-select", "--print-path"], 
                              capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                issues.append("Xcode command line tools not found")
                suggestions.append("Install: xcode-select --install")
                
        elif system == "windows":
            # Check for Visual C++ build tools
            vs_paths = [
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools",
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools",
                "C:\\Program Files\\Microsoft Visual Studio\\2019\\Community",
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community",
            ]
            if not any(Path(p).exists() for p in vs_paths):
                issues.append("Microsoft Visual C++ Build Tools not found")
                suggestions.append("Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
                
        if issues:
            return DiagnosticResult(
                name="System Dependencies",
                passed=False,
                message=f"Missing system dependencies: {', '.join(issues)}",
                details={"issues": issues, "platform": system},
                remediation="\n".join(suggestions)
            )
        else:
            return DiagnosticResult(
                name="System Dependencies",
                passed=True,
                message="System dependencies appear adequate",
                details={"platform": system}
            )
            
    def check_core_packages(self) -> DiagnosticResult:
        """Check core package installations."""
        # Use base diagnostics if available
        if self.base_diagnostics:
            try:
                deps_info = self.base_diagnostics.check_core_dependencies()
                
                issues = []
                available_count = 0
                total_count = len(deps_info)
                
                for package, info in deps_info.items():
                    if info["available"]:
                        available_count += 1
                    else:
                        if info.get("import_error"):
                            issues.append(f"{package}: {info['import_error']}")
                        else:
                            issues.append(f"{package} not available")
                
                if issues:
                    return DiagnosticResult(
                        name="Core Packages",
                        passed=False,
                        message=f"{available_count}/{total_count} packages available",
                        details={"packages": deps_info, "issues": issues},
                        remediation="""Install missing core packages:

pip install numpy>=1.24.0 scipy>=1.11.0
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu>=1.7.4
pip install transformers>=4.35.0
pip install POT>=0.9.1

Or install all requirements:
pip install -r requirements.txt"""
                    )
                else:
                    return DiagnosticResult(
                        name="Core Packages",
                        passed=True,
                        message=f"All {total_count} core packages are available",
                        details={"packages": deps_info}
                    )
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Base diagnostics check failed: {e}")
                
        # Fallback to manual checking
        core_packages = {
            "numpy": ">=1.24.0",
            "scipy": ">=1.11.0", 
            "torch": ">=2.0.0",
            "faiss": ">=1.7.4",
            "transformers": ">=4.35.0",
            "POT": ">=0.9.1",
        }
        
        results = {}
        issues = []
        
        for package, min_version in core_packages.items():
            try:
                if package == "faiss":
                    # FAISS can be faiss-cpu or faiss-gpu
                    try:
                        import faiss
                        results[package] = "installed"
                    except ImportError:
                        results[package] = "missing"
                        issues.append(f"{package} not installed")
                elif package == "POT":
                    import ot
                    results[package] = "installed"
                else:
                    module = __import__(package)
                    version = getattr(module, "__version__", "unknown")
                    results[package] = version
                    
            except ImportError:
                results[package] = "missing"
                issues.append(f"{package} not installed")
            except Exception as e:
                results[package] = f"error: {str(e)}"
                issues.append(f"{package} import error: {str(e)}")
                
        if issues:
            return DiagnosticResult(
                name="Core Packages",
                passed=False,
                message=f"Package issues found: {len(issues)}",
                details={"packages": results, "issues": issues},
                remediation="""Install missing core packages:

pip install numpy>=1.24.0 scipy>=1.11.0
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu>=1.7.4
pip install transformers>=4.35.0
pip install POT>=0.9.1"""
            )
        else:
            return DiagnosticResult(
                name="Core Packages", 
                passed=True,
                message="All core packages are available",
                details={"packages": results}
            )
            
    def check_gpu_support(self) -> DiagnosticResult:
        """Check GPU and CUDA support."""
        gpu_info = {"cuda_available": False, "gpu_devices": []}
        
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
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            gpu_info["nvidia_smi"] = result.stdout.strip().split('\n')
        except (subprocess.CalledProcessError, FileNotFoundError):
            gpu_info["nvidia_smi"] = None
            
        return DiagnosticResult(
            name="GPU Support",
            passed=True,  # This is informational, not a failure
            message=f"CUDA available: {gpu_info['cuda_available']}",
            details=gpu_info
        )
        
    def check_memory_availability(self) -> DiagnosticResult:
        """Check available system memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            # Warn if less than 4GB available
            if available_gb < 4:
                return DiagnosticResult(
                    name="Memory",
                    passed=False,
                    message=f"Low memory: {available_gb:.1f}GB available",
                    details={"available_gb": available_gb, "total_gb": total_gb, "percent": memory.percent},
                    remediation="""Increase available memory:

1. Close other applications
2. Add swap space (Linux):
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile

3. Use lighter package alternatives:
   pip install --only-binary=all -r requirements.txt"""
                )
            else:
                return DiagnosticResult(
                    name="Memory",
                    passed=True,
                    message=f"Adequate memory: {available_gb:.1f}GB available",
                    details={"available_gb": available_gb, "total_gb": total_gb}
                )
                
        except ImportError:
            return DiagnosticResult(
                name="Memory",
                passed=True,
                message="Memory check skipped (psutil not available)",
                details={}
            )
            
    def check_disk_space(self) -> DiagnosticResult:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            if free_gb < 5:
                return DiagnosticResult(
                    name="Disk Space",
                    passed=False,
                    message=f"Low disk space: {free_gb:.1f}GB free",
                    details={"free_gb": free_gb, "total_gb": total / (1024**3)},
                    remediation="Free up disk space or install to a different location"
                )
            else:
                return DiagnosticResult(
                    name="Disk Space",
                    passed=True,
                    message=f"Adequate disk space: {free_gb:.1f}GB free",
                    details={"free_gb": free_gb}
                )
                
        except Exception as e:
            return DiagnosticResult(
                name="Disk Space",
                passed=True,
                message=f"Disk space check failed: {str(e)}",
                details={}
            )
            
    def check_network_connectivity(self) -> DiagnosticResult:
        """Check network connectivity to PyPI."""
        try:
            import urllib.request
            
            with urllib.request.urlopen("https://pypi.org", timeout=10):
                pass
                
            return DiagnosticResult(
                name="Network",
                passed=True,
                message="Network connectivity to PyPI is working",
                details={}
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="Network",
                passed=False,
                message=f"Network connectivity issue: {str(e)}",
                details={"error": str(e)},
                remediation="""Check network connectivity:

1. Test internet connection
2. Check firewall settings
3. Try alternative index:
   pip install -i https://pypi.douban.com/simple/ <package>
4. Use offline installation if needed"""
            )
            
    def run_all_checks(self) -> List[DiagnosticResult]:
        """Run all diagnostic checks."""
        checks = [
            self.check_python_version,
            self.check_virtual_environment,
            self.check_pip_version,
            self.check_system_dependencies,
            self.check_core_packages,
            self.check_gpu_support,
            self.check_memory_availability,
            self.check_disk_space,
            self.check_network_connectivity,
        ]
        
        self.results = []
        for check in checks:
            try:
                result = check()
                self.add_result(result)
                
                # Display result immediately if verbose
                if self.verbose:
                    status = f"{Colors.GREEN}✅ PASS" if result.passed else f"{Colors.RED}❌ FAIL"
                    print(f"{status}{Colors.RESET} {result.name}: {result.message}")
                    
            except Exception as e:
                error_result = DiagnosticResult(
                    name=check.__name__.replace("check_", "").replace("_", " ").title(),
                    passed=False,
                    message=f"Check failed with error: {str(e)}",
                    details={"error": str(e), "traceback": traceback.format_exc()}
                )
                self.add_result(error_result)
                
        return self.results


class InteractiveGuide:
    """Provides interactive guidance for resolving issues."""
    
    def __init__(self, results: List[DiagnosticResult]):
        self.results = results
        self.failed_results = [r for r in results if not r.passed]
        
    def show_summary(self) -> None:
        """Show diagnostic summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print(f"\n{Colors.BOLD}🔍 Diagnostic Summary{Colors.RESET}")
        print("=" * 50)
        print(f"Total checks: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        
        # Show installation readiness assessment if available
        if HAS_DIAGNOSTICS_MODULE:
            try:
                ready, issues = check_installation_readiness()
                readiness_status = f"{Colors.GREEN}✅ READY" if ready else f"{Colors.RED}❌ NOT READY"
                print(f"Installation readiness: {readiness_status}{Colors.RESET}")
                if not ready and issues:
                    print(f"Critical issues: {', '.join(issues[:3])}")
                    if len(issues) > 3:
                        print(f"... and {len(issues) - 3} more")
            except Exception:
                pass
        
        if failed == 0:
            print(f"\n{Colors.GREEN}🎉 All checks passed! Your environment looks good.{Colors.RESET}")
            return
            
        print(f"\n{Colors.YELLOW}⚠️  Found {failed} issue(s) that need attention:{Colors.RESET}")
        
        for i, result in enumerate(self.failed_results, 1):
            print(f"\n{Colors.RED}{i}. {result.name}{Colors.RESET}")
            print(f"   Issue: {result.message}")
            
    def show_detailed_issues(self) -> None:
        """Show detailed information about failed checks."""
        if not self.failed_results:
            return
            
        print(f"\n{Colors.BOLD}📋 Detailed Issue Analysis{Colors.RESET}")
        print("=" * 50)
        
        for i, result in enumerate(self.failed_results, 1):
            print(f"\n{Colors.RED}{Colors.BOLD}Issue {i}: {result.name}{Colors.RESET}")
            print(f"Problem: {result.message}")
            
            if result.details:
                print(f"Details:")
                for key, value in result.details.items():
                    print(f"  • {key}: {value}")
                    
            if result.remediation:
                print(f"{Colors.CYAN}Recommended Solution:{Colors.RESET}")
                print(result.remediation)
                
            print("-" * 50)
            
    def interactive_resolution(self) -> None:
        """Provide interactive resolution guidance."""
        if not self.failed_results:
            return
            
        print(f"\n{Colors.BOLD}🛠️  Interactive Issue Resolution{Colors.RESET}")
        print("=" * 50)
        
        for i, result in enumerate(self.failed_results, 1):
            print(f"\n{Colors.YELLOW}Issue {i}/{len(self.failed_results)}: {result.name}{Colors.RESET}")
            print(f"Problem: {result.message}")
            
            if result.remediation:
                print(f"\n{Colors.CYAN}Suggested fix:{Colors.RESET}")
                print(result.remediation)
                
                try:
                    response = input(f"\nWould you like to see more details for this issue? (y/N): ").strip().lower()
                    if response in ['y', 'yes']:
                        if result.details:
                            print(f"\n{Colors.BLUE}Additional details:{Colors.RESET}")
                            print(json.dumps(result.details, indent=2))
                            
                    input("\nPress Enter to continue to next issue...")
                    
                except KeyboardInterrupt:
                    print(f"\n{Colors.YELLOW}Interrupted by user.{Colors.RESET}")
                    break
                    
        print(f"\n{Colors.GREEN}💡 Tips:{Colors.RESET}")
        print("• Address issues in the order shown (some fixes may resolve multiple issues)")
        print("• Re-run the diagnostic after making changes")
        print("• Check the troubleshooting guide in docs/INSTALLATION_TROUBLESHOOTING.md")
        print("• Use 'python -m egw_query_expansion.cli.troubleshoot --export-report' to save detailed diagnostics")


def export_report(results: List[DiagnosticResult], filename: Optional[str] = None) -> str:
    """Export diagnostic results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"egw_diagnostic_report_{timestamp}.json"
        
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "working_directory": os.getcwd(),
        },
        "diagnostic_results": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "details": r.details,
                "remediation": r.remediation,
                "timestamp": r.timestamp,
            }
            for r in results
        ],
        "summary": {
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results if r.passed),
            "failed_checks": sum(1 for r in results if not r.passed),
        }
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        return filename
    except Exception as e:
        print(f"{Colors.RED}Failed to export report: {str(e)}{Colors.RESET}")
        return ""


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EGW Query Expansion Installation Troubleshooting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m egw_query_expansion.cli.troubleshoot
  python -m egw_query_expansion.cli.troubleshoot --verbose
  python -m egw_query_expansion.cli.troubleshoot --export-report
  python -m egw_query_expansion.cli.troubleshoot --quick
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--export-report",
        action="store_true", 
        help="Export diagnostic report to JSON file"
    )
    
    parser.add_argument(
        "--report-file",
        type=str,
        help="Specify custom report filename"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential checks (faster)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Enable interactive guidance (default)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive guidance"
    )
    
    args = parser.parse_args()
    
    # Print header
    print(f"{Colors.BOLD}{Colors.CYAN}🔧 EGW Query Expansion - Installation Troubleshooter{Colors.RESET}")
    print(f"{Colors.BLUE}Analyzing your environment...{Colors.RESET}\n")
    
    # Run diagnostics
    analyzer = EnvironmentAnalyzer(verbose=args.verbose)
    results = analyzer.run_all_checks()
    
    # Create guide and show results
    guide = InteractiveGuide(results)
    guide.show_summary()
    
    # Export report if requested
    if args.export_report:
        filename = export_report(results, args.report_file)
        if filename:
            print(f"\n{Colors.GREEN}📄 Report exported to: {filename}{Colors.RESET}")
    
    # Show detailed issues
    if any(not r.passed for r in results):
        guide.show_detailed_issues()
        
        # Interactive resolution unless disabled
        if not args.no_interactive and (args.interactive or not args.quick):
            guide.interactive_resolution()
    
    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.passed)
    if failed_count > 0:
        print(f"\n{Colors.YELLOW}⚠️  Found {failed_count} issue(s). See recommendations above.{Colors.RESET}")
        print(f"{Colors.BLUE}💡 For detailed troubleshooting steps, see: docs/INSTALLATION_TROUBLESHOOTING.md{Colors.RESET}")
        sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}✅ Environment looks good! You should be able to install EGW Query Expansion.{Colors.RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()