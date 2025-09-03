"""
Mathematical Compatibility Matrix for EGW Query Expansion

This module provides comprehensive version compatibility checking and numerical stability
validation for mathematical libraries used across 12 stage enhancers in the EGW system.

Stage enhancers covered:
1. Differential Geometry - torch, numpy, scipy
2. Category Theory - networkx, numpy, scipy  
3. Topological Data Analysis - sklearn, numpy, scipy
4. Information Theory - numpy, scipy.stats, sklearn
5. Optimal Transport - POT, torch, numpy, scipy
6. Spectral Methods - scipy.linalg, numpy, sklearn
7. Control Theory - scipy.optimize, numpy, control (optional)
8. Measure Theory - scipy.stats, numpy, torch
9. Optimization Theory - scipy.optimize, sklearn, torch
10. Algebraic Topology - networkx, numpy, scipy
11. Functional Analysis - torch, numpy, scipy
12. Statistical Learning - sklearn, scipy, numpy, statsmodels

Python version support: 3.8 through 3.12
"""

import sys
import warnings
import platform
import re
# # # from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import importlib
import importlib.util
import subprocess
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from packaging import version  # Module not found  # Module not found  # Module not found
# # # from concurrent.futures import ThreadPoolExecutor, as_completed  # Module not found  # Module not found  # Module not found
import threading
import traceback
import json
import os

# Import numpy safely
try:
    import numpy as np
except ImportError:
    np = None


class PythonVersion(Enum):
    """Supported Python versions"""
    PYTHON_38 = "3.8"
    PYTHON_39 = "3.9"
    PYTHON_310 = "3.10"
    PYTHON_311 = "3.11"
    PYTHON_312 = "3.12"


class StageEnhancer(Enum):
    """12 Mathematical stage enhancers"""
    DIFFERENTIAL_GEOMETRY = "differential_geometry"
    CATEGORY_THEORY = "category_theory"
    TOPOLOGICAL_DATA_ANALYSIS = "topological_data_analysis"
    INFORMATION_THEORY = "information_theory"
    OPTIMAL_TRANSPORT = "optimal_transport"
    SPECTRAL_METHODS = "spectral_methods"
    CONTROL_THEORY = "control_theory"
    MEASURE_THEORY = "measure_theory"
    OPTIMIZATION_THEORY = "optimization_theory"
    ALGEBRAIC_TOPOLOGY = "algebraic_topology"
    FUNCTIONAL_ANALYSIS = "functional_analysis"
    STATISTICAL_LEARNING = "statistical_learning"


@dataclass
class VersionConstraint:
    """Version constraint specification"""
    min_version: str
    max_version: Optional[str] = None
    excluded_versions: List[str] = field(default_factory=list)
    notes: str = ""
    
    def is_version_compatible(self, ver: str) -> bool:
        """Check if a version satisfies this constraint"""
        try:
            v = version.parse(ver)
            min_v = version.parse(self.min_version)
            
            # Check minimum version
            if v < min_v:
                return False
                
            # Check maximum version if specified
            if self.max_version:
                max_v = version.parse(self.max_version)
                if v > max_v:
                    return False
            
            # Check excluded versions
            if ver in self.excluded_versions:
                return False
                
            return True
        except Exception:
            return False


@dataclass
class PlatformConstraint:
    """Platform-specific constraints"""
    supported_platforms: List[str] = field(default_factory=lambda: ["Windows", "Linux", "Darwin"])
    unsupported_platforms: List[str] = field(default_factory=list)
    platform_specific_notes: Dict[str, str] = field(default_factory=dict)
    
    def is_platform_supported(self, platform_name: str) -> bool:
        """Check if platform is supported"""
        if platform_name in self.unsupported_platforms:
            return False
        return platform_name in self.supported_platforms or not self.supported_platforms


@dataclass 
class LibrarySpec:
    """Library specification with version constraints"""
    name: str
    import_name: str
    constraints: Dict[PythonVersion, VersionConstraint]
    platform_constraints: PlatformConstraint = field(default_factory=PlatformConstraint)
    is_optional: bool = False
    fallback_libraries: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    install_command: Optional[str] = None


@dataclass
class CompatibilityResult:
    """Result of compatibility check"""
    is_compatible: bool
    installed_version: Optional[str] = None
    required_version: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    platform_issues: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class NumericalStabilityResult:
    """Result of numerical stability validation"""
    is_stable: bool
    max_error: float
    mean_error: float
    failed_operations: List[str] = field(default_factory=list)
    precision_warnings: List[str] = field(default_factory=list)


@dataclass
class CrossPlatformResult:
    """Result of cross-platform compatibility test"""
    platform: str
    python_version: str
    is_compatible: bool
    library_results: Dict[str, CompatibilityResult] = field(default_factory=dict)
    platform_specific_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass  
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: str
    python_version: str
    platform: str
    library_compatibility: Dict[str, CompatibilityResult] = field(default_factory=dict)
    stage_compatibility: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    numerical_stability: Optional[NumericalStabilityResult] = None
    cross_platform_results: List[CrossPlatformResult] = field(default_factory=list)
    faiss_conflicts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)


class MathematicalCompatibilityMatrix:
    """
    Central compatibility matrix for all mathematical libraries used in EGW system.
    Provides version checking, numerical stability validation, and compatibility resolution.
    """

    def __init__(self):
        """Initialize the compatibility matrix with all library specifications"""
        self.current_python_version = self._detect_python_version()
        self.current_platform = platform.system()
        self.platform_details = self._get_platform_details()
        self.library_specs = self._initialize_library_specs()
        self.stage_dependencies = self._initialize_stage_dependencies()
        self._lock = threading.Lock()
        
    def _detect_python_version(self) -> PythonVersion:
        """Detect current Python version"""
        version_info = sys.version_info
        version_str = f"{version_info.major}.{version_info.minor}"
        
        try:
            return PythonVersion(version_str)
        except ValueError:
            # Fallback to closest supported version
            if version_info.minor < 8:
                return PythonVersion.PYTHON_38
            elif version_info.minor > 12:
                return PythonVersion.PYTHON_312
            else:
                return PythonVersion.PYTHON_310  # Safe default
                
    def _get_platform_details(self) -> Dict[str, str]:
        """Get detailed platform information"""
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "release": platform.release(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version()
        }
    
    def _initialize_library_specs(self) -> Dict[str, LibrarySpec]:
        """Initialize all library specifications with version constraints"""
        
        specs = {}
        
        # Core numerical libraries
        specs["numpy"] = LibrarySpec(
            name="numpy",
            import_name="numpy",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("1.19.0", "1.26.4", notes="Stable for all operations"),
                PythonVersion.PYTHON_39: VersionConstraint("1.20.0", "1.26.4", notes="Enhanced performance"),
                PythonVersion.PYTHON_310: VersionConstraint("1.21.0", "1.26.4", notes="Full compatibility"),
                PythonVersion.PYTHON_311: VersionConstraint("1.23.0", "1.26.4", notes="Optimized builds"),
                PythonVersion.PYTHON_312: VersionConstraint("1.24.0", notes="Latest stable")
            },
            install_command="pip install numpy>=1.24.0"
        )
        
        specs["scipy"] = LibrarySpec(
            name="scipy",
            import_name="scipy",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("1.7.0", "1.12.0", notes="Core stability"),
                PythonVersion.PYTHON_39: VersionConstraint("1.8.0", "1.12.0", notes="Performance improvements"),
                PythonVersion.PYTHON_310: VersionConstraint("1.9.0", "1.12.0", notes="Full feature set"),
                PythonVersion.PYTHON_311: VersionConstraint("1.10.0", "1.12.0", notes="Enhanced algorithms"),
                PythonVersion.PYTHON_312: VersionConstraint("1.11.0", notes="Latest features")
            }
        )
        
        specs["torch"] = LibrarySpec(
            name="torch",
            import_name="torch",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("1.12.0", "2.1.2", notes="CUDA 11.x compatibility"),
                PythonVersion.PYTHON_39: VersionConstraint("1.13.0", "2.1.2", notes="Improved performance"),
                PythonVersion.PYTHON_310: VersionConstraint("1.13.0", "2.1.2", notes="Full PyTorch features"),
                PythonVersion.PYTHON_311: VersionConstraint("2.0.0", "2.1.2", notes="PyTorch 2.x optimizations"),
                PythonVersion.PYTHON_312: VersionConstraint("2.1.0", notes="Latest PyTorch")
            },
            platform_constraints=PlatformConstraint(
                platform_specific_notes={
                    "Windows": "Consider CUDA version compatibility",
                    "Linux": "Optimal platform for PyTorch",
                    "Darwin": "MPS acceleration available on Apple Silicon"
                }
            ),
            install_command="pip install torch>=2.0.0"
        )
        
        specs["sklearn"] = LibrarySpec(
            name="scikit-learn",
            import_name="sklearn",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("1.1.0", "1.4.0", notes="Stable algorithms"),
                PythonVersion.PYTHON_39: VersionConstraint("1.2.0", "1.4.0", notes="Enhanced features"),
                PythonVersion.PYTHON_310: VersionConstraint("1.2.0", "1.4.0", notes="Full compatibility"),
                PythonVersion.PYTHON_311: VersionConstraint("1.3.0", "1.4.0", notes="Performance optimizations"),
                PythonVersion.PYTHON_312: VersionConstraint("1.3.0", notes="Latest sklearn")
            },
            install_command="pip install scikit-learn>=1.3.0"
        )
        
        # Optimal transport library
        specs["POT"] = LibrarySpec(
            name="POT",
            import_name="ot",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("0.8.2", "0.9.3", notes="Basic OT functionality"),
                PythonVersion.PYTHON_39: VersionConstraint("0.9.0", "0.9.3", notes="Enhanced algorithms"),
                PythonVersion.PYTHON_310: VersionConstraint("0.9.0", "0.9.3", notes="Full feature support"),
                PythonVersion.PYTHON_311: VersionConstraint("0.9.1", "0.9.3", notes="Latest optimizations"),
                PythonVersion.PYTHON_312: VersionConstraint("0.9.2", notes="Python 3.12 support")
            }
        )
        
        # Graph and network libraries
        specs["networkx"] = LibrarySpec(
            name="networkx",
            import_name="networkx",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("2.6.0", "3.2.1", notes="Graph algorithms"),
                PythonVersion.PYTHON_39: VersionConstraint("2.8.0", "3.2.1", notes="Enhanced performance"),
                PythonVersion.PYTHON_310: VersionConstraint("2.8.0", "3.2.1", notes="Full compatibility"),
                PythonVersion.PYTHON_311: VersionConstraint("3.0.0", "3.2.1", notes="NetworkX 3.x features"),
                PythonVersion.PYTHON_312: VersionConstraint("3.1.0", notes="Latest NetworkX")
            }
        )
        
        # Statistical libraries
        specs["statsmodels"] = LibrarySpec(
            name="statsmodels",
            import_name="statsmodels",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("0.13.0", "0.14.1", notes="Core statistics"),
                PythonVersion.PYTHON_39: VersionConstraint("0.13.5", "0.14.1", notes="Enhanced models"),
                PythonVersion.PYTHON_310: VersionConstraint("0.13.5", "0.14.1", notes="Full compatibility"),
                PythonVersion.PYTHON_311: VersionConstraint("0.14.0", "0.14.1", notes="Latest features"),
                PythonVersion.PYTHON_312: VersionConstraint("0.14.0", notes="Python 3.12 support")
            }
        )
        
        # Control theory (optional)
        specs["control"] = LibrarySpec(
            name="control",
            import_name="control",
            is_optional=True,
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("0.9.0", "0.9.4", notes="Basic control systems"),
                PythonVersion.PYTHON_39: VersionConstraint("0.9.2", "0.9.4", notes="Enhanced functionality"),
                PythonVersion.PYTHON_310: VersionConstraint("0.9.2", "0.9.4", notes="Full features"),
                PythonVersion.PYTHON_311: VersionConstraint("0.9.3", "0.9.4", notes="Latest control"),
                PythonVersion.PYTHON_312: VersionConstraint("0.9.4", notes="Python 3.12 support")
            },
            fallback_libraries=["scipy.optimize", "numpy"]
        )
        
        # FAISS libraries with conflict detection
        specs["faiss-cpu"] = LibrarySpec(
            name="faiss-cpu",
            import_name="faiss",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("1.7.4", notes="CPU-only FAISS"),
                PythonVersion.PYTHON_39: VersionConstraint("1.7.4", notes="CPU-only FAISS"),
                PythonVersion.PYTHON_310: VersionConstraint("1.7.4", notes="CPU-only FAISS"),
                PythonVersion.PYTHON_311: VersionConstraint("1.7.4", notes="CPU-only FAISS"),
                PythonVersion.PYTHON_312: VersionConstraint("1.7.4", notes="CPU-only FAISS")
            },
            conflicts_with=["faiss-gpu"],
            install_command="pip install faiss-cpu>=1.7.4"
        )
        
        specs["faiss-gpu"] = LibrarySpec(
            name="faiss-gpu",
            import_name="faiss",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("1.7.4", notes="GPU-enabled FAISS"),
                PythonVersion.PYTHON_39: VersionConstraint("1.7.4", notes="GPU-enabled FAISS"),
                PythonVersion.PYTHON_310: VersionConstraint("1.7.4", notes="GPU-enabled FAISS"),
                PythonVersion.PYTHON_311: VersionConstraint("1.7.4", notes="GPU-enabled FAISS"),
                PythonVersion.PYTHON_312: VersionConstraint("1.7.4", notes="GPU-enabled FAISS")
            },
            platform_constraints=PlatformConstraint(
                platform_specific_notes={
                    "Windows": "CUDA toolkit required",
                    "Linux": "CUDA toolkit and drivers required",
                    "Darwin": "Not officially supported on macOS"
                }
            ),
            conflicts_with=["faiss-cpu"],
            install_command="pip install faiss-gpu>=1.7.4"
        )
        
        # Sentence transformers
        specs["sentence-transformers"] = LibrarySpec(
            name="sentence-transformers",
            import_name="sentence_transformers",
            constraints={
                PythonVersion.PYTHON_38: VersionConstraint("2.2.2", notes="Stable sentence embeddings"),
                PythonVersion.PYTHON_39: VersionConstraint("2.2.2", notes="Enhanced models"),
                PythonVersion.PYTHON_310: VersionConstraint("2.2.2", notes="Full compatibility"),
                PythonVersion.PYTHON_311: VersionConstraint("2.2.2", notes="Latest features"),
                PythonVersion.PYTHON_312: VersionConstraint("2.2.2", notes="Python 3.12 support")
            },
            install_command="pip install sentence-transformers>=2.2.2"
        )
        
        return specs
    
    def _initialize_stage_dependencies(self) -> Dict[StageEnhancer, List[str]]:
        """Initialize dependencies for each stage enhancer"""
        return {
            StageEnhancer.DIFFERENTIAL_GEOMETRY: ["torch", "numpy", "scipy"],
            StageEnhancer.CATEGORY_THEORY: ["networkx", "numpy", "scipy"],
            StageEnhancer.TOPOLOGICAL_DATA_ANALYSIS: ["sklearn", "numpy", "scipy", "faiss-cpu"],
            StageEnhancer.INFORMATION_THEORY: ["numpy", "scipy", "sklearn"],
            StageEnhancer.OPTIMAL_TRANSPORT: ["POT", "torch", "numpy", "scipy"],
            StageEnhancer.SPECTRAL_METHODS: ["scipy", "numpy", "sklearn"],
            StageEnhancer.CONTROL_THEORY: ["scipy", "numpy", "control"],
            StageEnhancer.MEASURE_THEORY: ["scipy", "numpy", "torch"],
            StageEnhancer.OPTIMIZATION_THEORY: ["scipy", "sklearn", "torch"],
            StageEnhancer.ALGEBRAIC_TOPOLOGY: ["networkx", "numpy", "scipy"],
            StageEnhancer.FUNCTIONAL_ANALYSIS: ["torch", "numpy", "scipy", "sentence-transformers"],
            StageEnhancer.STATISTICAL_LEARNING: ["sklearn", "scipy", "numpy", "statsmodels"]
        }
    
    def check_library_compatibility(self, library_name: str) -> CompatibilityResult:
        """Check if a specific library is compatible with current Python version and platform"""
        
        if library_name not in self.library_specs:
            return CompatibilityResult(
                is_compatible=False,
                issues=[f"Library {library_name} not in compatibility matrix"]
            )
        
        spec = self.library_specs[library_name]
        result = CompatibilityResult(is_compatible=False)
        
        # Check platform compatibility
        if not spec.platform_constraints.is_platform_supported(self.current_platform):
            result.platform_issues.append(f"Platform {self.current_platform} not supported")
            if self.current_platform in spec.platform_constraints.unsupported_platforms:
                result.issues.append(f"Library {library_name} explicitly unsupported on {self.current_platform}")
                return result
        
        # Add platform-specific notes
        if self.current_platform in spec.platform_constraints.platform_specific_notes:
            note = spec.platform_constraints.platform_specific_notes[self.current_platform]
            result.warnings.append(f"Platform note: {note}")
        
        # Check for conflicts with other libraries
        conflicts = self._detect_library_conflicts(library_name, spec)
        if conflicts:
            result.conflicts.extend(conflicts)
            if conflicts and not spec.is_optional:
                result.issues.append(f"Conflicts detected: {', '.join(conflicts)}")
        
        # Check if library is installed
        try:
            import_name = spec.import_name
            module = importlib.import_module(import_name)
            
            # Get installed version
            installed_version = getattr(module, '__version__', 'unknown')
            result.installed_version = installed_version
            
            # Version validation
            if self.current_python_version not in spec.constraints:
                result.issues.append(f"No version constraints defined for Python {self.current_python_version.value}")
                return result
            
            constraint = spec.constraints[self.current_python_version]
            result.required_version = f">={constraint.min_version}"
            if constraint.max_version:
                result.required_version += f",<={constraint.max_version}"
            
            # Check semantic version compatibility
            if installed_version != 'unknown':
                if constraint.is_version_compatible(installed_version):
                    result.is_compatible = True
                    if constraint.notes:
                        result.warnings.append(f"Version note: {constraint.notes}")
                else:
                    result.issues.append(f"Installed version {installed_version} incompatible with required {result.required_version}")
            else:
                result.warnings.append("Unable to determine installed version")
                result.is_compatible = True  # Assume compatible if we can't check version
                
        except ImportError:
            if spec.is_optional:
                result.is_compatible = True
                result.warnings.append(f"Optional library {library_name} not installed")
                if spec.fallback_libraries:
                    result.recommendations.append(f"Consider fallback libraries: {', '.join(spec.fallback_libraries)}")
            else:
                result.issues.append(f"Required library {library_name} not installed")
                if spec.install_command:
                    result.recommendations.append(f"Install with: {spec.install_command}")
        
        return result
        
    def _detect_library_conflicts(self, library_name: str, spec: LibrarySpec) -> List[str]:
        """Detect conflicts with other installed libraries"""
        conflicts = []
        
        for conflict_lib in spec.conflicts_with:
            try:
                # Check if conflicting library is installed
                conflict_spec = self.library_specs.get(conflict_lib)
                if conflict_spec:
                    importlib.import_module(conflict_spec.import_name)
                    conflicts.append(f"{library_name} conflicts with {conflict_lib}")
            except ImportError:
                continue  # No conflict if library not installed
                
        return conflicts
    
    def check_stage_compatibility(self, stage: StageEnhancer) -> Dict[str, CompatibilityResult]:
        """Check compatibility of all libraries required for a stage enhancer"""
        
        if stage not in self.stage_dependencies:
            return {}
        
        results = {}
        for library in self.stage_dependencies[stage]:
            results[library] = self.check_library_compatibility(library)
        
        return results
    
    def check_all_compatibility(self) -> Dict[str, CompatibilityResult]:
        """Check compatibility of all libraries in the matrix with parallel execution"""
        results = {}
        
        def check_library(lib_name):
            return lib_name, self.check_library_compatibility(lib_name)
        
        with ThreadPoolExecutor(max_workers=min(len(self.library_specs), 10)) as executor:
            future_to_lib = {executor.submit(check_library, lib): lib for lib in self.library_specs}
            
            for future in as_completed(future_to_lib):
                try:
                    lib_name, result = future.result()
                    results[lib_name] = result
                except Exception as exc:
                    lib_name = future_to_lib[future]
                    results[lib_name] = CompatibilityResult(
                        is_compatible=False,
                        issues=[f"Exception during compatibility check: {exc}"]
                    )
        
        return results
    
    def validate_numerical_stability(self, 
                                   operations: List[str] = None,
                                   tolerance: float = 1e-10) -> NumericalStabilityResult:
        """
        Validate numerical stability across library combinations.
        
        Args:
            operations: List of operations to test ('matmul', 'svd', 'eig', 'ot', etc.)
            tolerance: Numerical tolerance for stability validation
            
        Returns:
            NumericalStabilityResult with stability assessment
        """
        
        if operations is None:
            operations = ['matmul', 'svd', 'eigendecomp', 'ot_basic', 'optimization']
        
        failed_operations = []
        precision_warnings = []
        errors = []
        
        # Test basic numpy operations
        if 'matmul' in operations:
            try:
                # Test matrix multiplication precision
                A = np.random.randn(100, 100).astype(np.float64)
                B = np.random.randn(100, 100).astype(np.float64)
                
                C1 = A @ B
                C2 = np.matmul(A, B)
                
                error = np.max(np.abs(C1 - C2))
                errors.append(error)
                
                if error > tolerance:
                    failed_operations.append("matmul_precision")
                    
            except Exception as e:
                failed_operations.append(f"matmul_error: {str(e)}")
        
        # Test SVD stability
        if 'svd' in operations:
            try:
                # Test SVD reconstruction
                A = np.random.randn(50, 50).astype(np.float64)
                U, s, Vt = np.linalg.svd(A)
                A_reconstructed = U @ np.diag(s) @ Vt
                
                error = np.max(np.abs(A - A_reconstructed))
                errors.append(error)
                
                if error > tolerance * 100:  # SVD typically has higher tolerance
                    failed_operations.append("svd_reconstruction")
                    
            except Exception as e:
                failed_operations.append(f"svd_error: {str(e)}")
        
        # Test eigendecomposition
        if 'eigendecomp' in operations:
            try:
                # Test symmetric eigendecomposition
                A = np.random.randn(30, 30)
                A = A + A.T  # Make symmetric
                
                eigvals, eigvecs = np.linalg.eigh(A)
                A_reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                error = np.max(np.abs(A - A_reconstructed))
                errors.append(error)
                
                if error > tolerance * 100:
                    failed_operations.append("eigen_reconstruction")
                    
            except Exception as e:
                failed_operations.append(f"eigen_error: {str(e)}")
        
        # Test optimal transport if POT is available
        if 'ot_basic' in operations:
            try:
                import ot
                
                # Test basic OT computation
                n = 100
                a = np.random.rand(n)
                a = a / a.sum()
                b = np.random.rand(n)
                b = b / b.sum()
                
                M = np.random.rand(n, n)
                
                # Test that transport plan sums correctly
                T = ot.emd(a, b, M)
                
                error_a = np.abs(T.sum(axis=1) - a).max()
                error_b = np.abs(T.sum(axis=0) - b).max()
                
                errors.extend([error_a, error_b])
                
                if max(error_a, error_b) > tolerance * 10:
                    failed_operations.append("ot_marginal_constraints")
                    
            except ImportError:
                precision_warnings.append("POT not available for OT stability testing")
            except Exception as e:
                failed_operations.append(f"ot_error: {str(e)}")
        
        # Test scipy optimization
        if 'optimization' in operations:
            try:
# # #                 from scipy.optimize import minimize  # Module not found  # Module not found  # Module not found
                
                # Test simple quadratic optimization
                def objective(x):
                    return np.sum((x - 1)**2)
                
                result = minimize(objective, np.zeros(5), method='BFGS')
                
                error = np.max(np.abs(result.x - 1.0))
                errors.append(error)
                
                if error > tolerance * 100:
                    failed_operations.append("optimization_convergence")
                    
            except Exception as e:
                failed_operations.append(f"optimization_error: {str(e)}")
        
        # Calculate summary statistics
        if errors:
            max_error = max(errors)
            mean_error = np.mean(errors)
        else:
            max_error = 0.0
            mean_error = 0.0
        
        is_stable = len(failed_operations) == 0 and max_error < tolerance * 1000
        
        return NumericalStabilityResult(
            is_stable=is_stable,
            max_error=max_error,
            mean_error=mean_error,
            failed_operations=failed_operations,
            precision_warnings=precision_warnings
        )
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive compatibility report"""
        
        report = {
            "python_version": self.current_python_version.value,
            "library_compatibility": {},
            "stage_compatibility": {},
            "numerical_stability": {},
            "recommendations": []
        }
        
        # Check all library compatibility
        lib_results = self.check_all_compatibility()
        report["library_compatibility"] = {
            name: {
                "compatible": result.is_compatible,
                "installed_version": result.installed_version,
                "required_version": result.required_version,
                "issues": result.issues or [],
                "warnings": result.warnings or []
            }
            for name, result in lib_results.items()
        }
        
        # Check stage compatibility
        for stage in StageEnhancer:
            stage_results = self.check_stage_compatibility(stage)
            report["stage_compatibility"][stage.value] = {
                name: result.is_compatible
                for name, result in stage_results.items()
            }
        
        # Numerical stability validation
        stability_result = self.validate_numerical_stability()
        report["numerical_stability"] = {
            "is_stable": stability_result.is_stable,
            "max_error": stability_result.max_error,
            "mean_error": stability_result.mean_error,
            "failed_operations": stability_result.failed_operations or [],
            "warnings": stability_result.precision_warnings or []
        }
        
        # Generate recommendations
        incompatible_libs = [
            name for name, result in lib_results.items()
            if not result.is_compatible
        ]
        
        if incompatible_libs:
            report["recommendations"].append(
                f"Install/upgrade incompatible libraries: {', '.join(incompatible_libs)}"
            )
        
        if not stability_result.is_stable:
            report["recommendations"].append(
                "Review numerical stability issues and consider library version updates"
            )
        
        return report


class LibraryStatusReporter:
    """
    Comprehensive library status reporter with version detection, fallback tracking,
    behavioral consistency validation, and upgrade recommendations.
    """
    
    def __init__(self, matrix: MathematicalCompatibilityMatrix = None, 
                 report_path: str = "library_status_report.json"):
        """
        Initialize the library status reporter.
        
        Args:
            matrix: Mathematical compatibility matrix instance
            report_path: Path to save the status report JSON
        """
        self.matrix = matrix or MathematicalCompatibilityMatrix()
        self.report_path = report_path
        self.fallback_counters = {}
        self.consistency_test_cache = {}
        self.behavioral_validators = {}
        
        # Initialize fallback counters for all libraries
        for lib_name in self.matrix.library_specs.keys():
            self.fallback_counters[lib_name] = {
                'mock_calls': 0,
                'real_calls': 0,
                'fallback_triggered': 0,
                'consistency_failures': 0
            }
    
    def detect_library_versions(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect installed versions of all libraries with detailed metadata.
        
        Returns:
            Dict mapping library names to version information
        """
        detected_versions = {}
        
        for lib_name, spec in self.matrix.library_specs.items():
            version_info = {
                'library_name': lib_name,
                'import_name': spec.import_name,
                'is_optional': spec.is_optional,
                'installed': False,
                'version': None,
                'install_path': None,
                'size_mb': None,
                'dependencies': [],
                'capabilities': {},
                'health_score': 0.0
            }
            
            try:
                # Attempt to import the library
                import importlib
                module = importlib.import_module(spec.import_name)
                version_info['installed'] = True
                
                # Get version information
                version_info['version'] = getattr(module, '__version__', 'unknown')
                version_info['install_path'] = getattr(module, '__file__', 'unknown')
                
                # Calculate library size (rough estimate)
                if version_info['install_path'] and version_info['install_path'] != 'unknown':
                    try:
                        import os
                        lib_path = os.path.dirname(version_info['install_path'])
                        total_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(lib_path)
                            for filename in filenames
                        )
                        version_info['size_mb'] = round(total_size / (1024 * 1024), 2)
                    except Exception:
                        version_info['size_mb'] = None
                
                # Detect library capabilities
                version_info['capabilities'] = self._detect_library_capabilities(lib_name, module)
                
                # Get dependencies using importlib.metadata if available
                try:
                    import importlib.metadata
                    dist = importlib.metadata.distribution(lib_name)
                    if dist.requires:
                        version_info['dependencies'] = [str(req) for req in dist.requires]
                except Exception:
                    pass
                
                # Calculate health score based on various factors
                version_info['health_score'] = self._calculate_library_health_score(
                    lib_name, spec, version_info
                )
                
            except ImportError:
                version_info['installed'] = False
                if not spec.is_optional:
                    version_info['health_score'] = 0.0
                else:
                    version_info['health_score'] = 0.5  # Optional library missing
            except Exception as e:
                version_info['error'] = str(e)
                version_info['health_score'] = 0.1
            
            detected_versions[lib_name] = version_info
        
        return detected_versions
    
    def _detect_library_capabilities(self, lib_name: str, module) -> Dict[str, bool]:
        """Detect specific capabilities of a library."""
        capabilities = {}
        
        if lib_name == 'torch':
            capabilities['cuda_available'] = hasattr(module, 'cuda') and module.cuda.is_available()
            capabilities['mps_available'] = hasattr(module.backends, 'mps') and module.backends.mps.is_available() if hasattr(module, 'backends') else False
            capabilities['distributed'] = hasattr(module, 'distributed')
            capabilities['autograd'] = hasattr(module, 'autograd')
            capabilities['jit'] = hasattr(module, 'jit')
        
        elif lib_name == 'numpy':
            capabilities['blas_available'] = bool(hasattr(module, 'linalg') and hasattr(module.linalg, 'lapack_lite'))
            capabilities['mkl_available'] = 'mkl' in str(module.linalg._umath_linalg).lower()
            capabilities['simd'] = hasattr(module, '__config__')
        
        elif lib_name == 'scipy':
            capabilities['sparse_available'] = hasattr(module, 'sparse')
            capabilities['optimize_available'] = hasattr(module, 'optimize')
            capabilities['linalg_available'] = hasattr(module, 'linalg')
            capabilities['stats_available'] = hasattr(module, 'stats')
        
        elif lib_name == 'sklearn':
            capabilities['parallel_backend'] = hasattr(module, 'utils') and hasattr(module.utils, 'parallel_backend')
            capabilities['gpu_support'] = any('gpu' in str(est).lower() for est in dir(module) if hasattr(module, est))
        
        elif lib_name == 'POT':
            capabilities['gpu_support'] = hasattr(module, 'gpu')
            capabilities['backend_available'] = hasattr(module, 'backend')
        
        return capabilities
    
    def _calculate_library_health_score(self, lib_name: str, spec: LibrarySpec, 
                                      version_info: Dict[str, Any]) -> float:
        """Calculate a health score (0-1) for the library installation."""
        if not version_info['installed']:
            return 0.0 if not spec.is_optional else 0.5
        
        score = 0.7  # Base score for being installed
        
        # Version compatibility check
        compat_result = self.matrix.check_library_compatibility(lib_name)
        if compat_result.is_compatible:
            score += 0.2
        
        # Capability bonuses
        capabilities = version_info.get('capabilities', {})
        if capabilities:
            capability_score = sum(1 for cap in capabilities.values() if cap) / len(capabilities)
            score += 0.1 * capability_score
        
        # Size penalty for extremely large libraries
        size_mb = version_info.get('size_mb', 0) or 0
        if size_mb > 1000:  # Libraries over 1GB
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def increment_fallback_counter(self, lib_name: str, counter_type: str):
        """Increment a fallback counter for tracking library usage patterns."""
        if lib_name in self.fallback_counters and counter_type in self.fallback_counters[lib_name]:
            self.fallback_counters[lib_name][counter_type] += 1
    
    def track_library_call(self, lib_name: str, is_mock: bool = False, 
                          fallback_triggered: bool = False):
        """Track a library call for usage statistics."""
        if lib_name not in self.fallback_counters:
            return
        
        if is_mock:
            self.increment_fallback_counter(lib_name, 'mock_calls')
        else:
            self.increment_fallback_counter(lib_name, 'real_calls')
        
        if fallback_triggered:
            self.increment_fallback_counter(lib_name, 'fallback_triggered')
    
    def validate_behavioral_consistency(self, lib_name: str, 
                                      test_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate behavioral consistency between mock and real library implementations.
        
        Args:
            lib_name: Name of library to test
            test_cases: Custom test cases, or None for default tests
            
        Returns:
            Dict with consistency test results
        """
        if lib_name in self.consistency_test_cache:
            return self.consistency_test_cache[lib_name]
        
        result = {
            'library_name': lib_name,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'consistency_score': 0.0,
            'failed_operations': [],
            'warnings': [],
            'test_details': []
        }
        
        # Check if library is available
        if lib_name not in self.matrix.library_specs:
            result['warnings'].append(f"Library {lib_name} not in compatibility matrix")
            self.consistency_test_cache[lib_name] = result
            return result
        
        spec = self.matrix.library_specs[lib_name]
        
        try:
            module = importlib.import_module(spec.import_name)
        except ImportError:
            result['warnings'].append(f"Library {lib_name} not installed - skipping consistency tests")
            self.consistency_test_cache[lib_name] = result
            return result
        
        # Run library-specific consistency tests
        if test_cases is None:
            test_cases = self._generate_default_test_cases(lib_name, module)
        
        for test_case in test_cases:
            test_result = self._run_consistency_test(lib_name, module, test_case)
            result['test_details'].append(test_result)
            result['tests_run'] += 1
            
            if test_result['passed']:
                result['tests_passed'] += 1
            else:
                result['tests_failed'] += 1
                result['failed_operations'].append(test_result['operation'])
        
        # Calculate consistency score
        if result['tests_run'] > 0:
            result['consistency_score'] = result['tests_passed'] / result['tests_run']
        
        self.consistency_test_cache[lib_name] = result
        return result
    
    def _generate_default_test_cases(self, lib_name: str, module) -> List[Dict[str, Any]]:
        """Generate default test cases for a library."""
        test_cases = []
        
        if lib_name == 'numpy':
            test_cases = [
                {
                    'operation': 'basic_array_creation',
                    'function': lambda: np.array([1, 2, 3, 4, 5]),
                    'validator': lambda result: isinstance(result, np.ndarray) and len(result) == 5
                },
                {
                    'operation': 'matrix_multiplication',
                    'function': lambda: np.random.randn(10, 10) @ np.random.randn(10, 10),
                    'validator': lambda result: isinstance(result, np.ndarray) and result.shape == (10, 10)
                },
                {
                    'operation': 'linalg_svd',
                    'function': lambda: np.linalg.svd(np.random.randn(5, 5)),
                    'validator': lambda result: len(result) == 3 and all(isinstance(x, np.ndarray) for x in result)
                }
            ]
        
        elif lib_name == 'torch':
            test_cases = [
                {
                    'operation': 'tensor_creation',
                    'function': lambda: module.tensor([1, 2, 3, 4, 5]),
                    'validator': lambda result: hasattr(result, 'shape') and result.shape[0] == 5
                },
                {
                    'operation': 'basic_autograd',
                    'function': lambda: module.tensor([1.0, 2.0], requires_grad=True).sum().backward(),
                    'validator': lambda result: True  # Just check it doesn't crash
                }
            ]
        
        elif lib_name == 'POT':
            test_cases = [
                {
                    'operation': 'wasserstein_distance',
                    'function': lambda: module.wasserstein_1d(np.array([0., 1.]), np.array([0., 1.])),
                    'validator': lambda result: isinstance(result, float) and result >= 0
                }
            ]
        
        return test_cases
    
    def _run_consistency_test(self, lib_name: str, module, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single consistency test."""
        result = {
            'operation': test_case['operation'],
            'passed': False,
            'error': None,
            'execution_time_ms': 0.0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Execute the test function
            test_result = test_case['function']()
            
            # Validate the result
            if test_case.get('validator'):
                result['passed'] = test_case['validator'](test_result)
            else:
                result['passed'] = True
            
            result['execution_time_ms'] = (time.time() - start_time) * 1000
            
        except Exception as e:
            result['error'] = str(e)
            result['passed'] = False
        
        return result
    
    def generate_upgrade_recommendations(self, detected_versions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate automated upgrade recommendations based on compatibility matrix."""
        recommendations = []
        
        for lib_name, version_info in detected_versions.items():
            if not version_info['installed']:
                if not self.matrix.library_specs[lib_name].is_optional:
                    recommendations.append({
                        'library': lib_name,
                        'priority': 'HIGH',
                        'action': 'INSTALL',
                        'current_version': None,
                        'recommended_version': self._get_recommended_version(lib_name),
                        'reason': 'Required library not installed',
                        'urgency_score': 1.0
                    })
                continue
            
            current_version = version_info['version']
            if current_version == 'unknown':
                recommendations.append({
                    'library': lib_name,
                    'priority': 'MEDIUM',
                    'action': 'INVESTIGATE',
                    'current_version': current_version,
                    'recommended_version': self._get_recommended_version(lib_name),
                    'reason': 'Cannot determine library version',
                    'urgency_score': 0.6
                })
                continue
            
            # Check compatibility
            compat_result = self.matrix.check_library_compatibility(lib_name)
            if not compat_result.is_compatible:
                recommendations.append({
                    'library': lib_name,
                    'priority': 'HIGH',
                    'action': 'UPGRADE',
                    'current_version': current_version,
                    'recommended_version': self._get_recommended_version(lib_name),
                    'reason': f'Compatibility issues: {"; ".join(compat_result.issues or [])}',
                    'urgency_score': 0.9
                })
            
            # Check health score
            health_score = version_info.get('health_score', 0.0)
            if health_score < 0.5:
                recommendations.append({
                    'library': lib_name,
                    'priority': 'MEDIUM',
                    'action': 'INVESTIGATE' if health_score > 0.2 else 'REINSTALL',
                    'current_version': current_version,
                    'recommended_version': self._get_recommended_version(lib_name),
                    'reason': f'Low health score ({health_score:.2f}) - potential installation issues',
                    'urgency_score': health_score
                })
        
        # Sort by urgency score (descending)
        recommendations.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        return recommendations
    
    def _get_recommended_version(self, lib_name: str) -> str:
        """Get recommended version for a library based on current Python version."""
        spec = self.matrix.library_specs.get(lib_name)
        if not spec:
            return "latest"
        
        constraint = spec.constraints.get(self.matrix.current_python_version)
        if not constraint:
            return "latest"
        
        return f">={constraint.min_version}" + (f",<={constraint.max_version}" if constraint.max_version else "")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive library status report."""
        # Detect all library versions
        detected_versions = self.detect_library_versions()
        
        # Run consistency validation for all libraries
        consistency_results = {}
        for lib_name in self.matrix.library_specs.keys():
            consistency_results[lib_name] = self.validate_behavioral_consistency(lib_name)
        
        # Generate upgrade recommendations
        recommendations = self.generate_upgrade_recommendations(detected_versions)
        
        # Calculate overall system health
        overall_health = self._calculate_overall_system_health(detected_versions)
        
        # Compile the comprehensive report
        report = {
            'metadata': {
                'generated_timestamp': datetime.now().isoformat(),
                'python_version': self.matrix.current_python_version.value,
                'report_version': '1.0',
                'total_libraries': len(self.matrix.library_specs)
            },
            'detected_versions': detected_versions,
            'fallback_statistics': dict(self.fallback_counters),
            'consistency_test_results': consistency_results,
            'upgrade_recommendations': recommendations,
            'system_health': overall_health,
            'stage_readiness': self._assess_stage_readiness(detected_versions),
            'risk_assessment': self._perform_risk_assessment(detected_versions, consistency_results)
        }
        
        return report
    
    def _calculate_overall_system_health(self, detected_versions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        total_libraries = len(detected_versions)
        installed_libraries = sum(1 for v in detected_versions.values() if v['installed'])
        required_missing = sum(
            1 for lib_name, v in detected_versions.items() 
            if not v['installed'] and not self.matrix.library_specs[lib_name].is_optional
        )
        
        total_health_score = sum(v.get('health_score', 0.0) for v in detected_versions.values())
        average_health_score = total_health_score / total_libraries if total_libraries > 0 else 0.0
        
        return {
            'overall_score': average_health_score,
            'installation_rate': installed_libraries / total_libraries if total_libraries > 0 else 0.0,
            'required_missing_count': required_missing,
            'total_libraries': total_libraries,
            'installed_libraries': installed_libraries,
            'health_classification': self._classify_health_score(average_health_score),
            'critical_issues': required_missing > 0
        }
    
    def _classify_health_score(self, score: float) -> str:
        """Classify health score into categories."""
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.7:
            return "GOOD"
        elif score >= 0.5:
            return "FAIR"
        elif score >= 0.3:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _assess_stage_readiness(self, detected_versions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Assess readiness of each stage enhancer based on library availability."""
        stage_readiness = {}
        
        for stage, dependencies in self.matrix.stage_dependencies.items():
            total_deps = len(dependencies)
            available_deps = 0
            missing_deps = []
            degraded_deps = []
            
            for dep in dependencies:
                if dep in detected_versions and detected_versions[dep]['installed']:
                    health_score = detected_versions[dep].get('health_score', 0.0)
                    if health_score >= 0.7:
                        available_deps += 1
                    else:
                        available_deps += 0.5
                        degraded_deps.append(dep)
                else:
                    missing_deps.append(dep)
            
            readiness_score = available_deps / total_deps if total_deps > 0 else 1.0
            
            stage_readiness[stage.value] = {
                'readiness_score': readiness_score,
                'status': 'READY' if readiness_score >= 0.9 else ('DEGRADED' if readiness_score >= 0.5 else 'NOT_READY'),
                'total_dependencies': total_deps,
                'available_dependencies': available_deps,
                'missing_dependencies': missing_deps,
                'degraded_dependencies': degraded_deps
            }
        
        return stage_readiness
    
    def _perform_risk_assessment(self, detected_versions: Dict[str, Dict[str, Any]], 
                                consistency_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform risk assessment based on library status and consistency."""
        risks = []
        risk_score = 0.0
        
        # Check for missing critical libraries
        for lib_name, spec in self.matrix.library_specs.items():
            if not spec.is_optional and not detected_versions[lib_name]['installed']:
                risks.append({
                    'type': 'MISSING_CRITICAL_LIBRARY',
                    'library': lib_name,
                    'severity': 'HIGH',
                    'impact': 'System functionality will be severely impacted',
                    'mitigation': f'Install {lib_name} using pip or conda'
                })
                risk_score += 0.3
        
        # Check for consistency failures
        for lib_name, consistency in consistency_results.items():
            if consistency['consistency_score'] < 0.8 and consistency['tests_run'] > 0:
                risks.append({
                    'type': 'BEHAVIORAL_INCONSISTENCY',
                    'library': lib_name,
                    'severity': 'MEDIUM',
                    'impact': f'Inconsistent behavior in {len(consistency["failed_operations"])} operations',
                    'mitigation': 'Review library version compatibility and consider upgrade'
                })
                risk_score += 0.1 * (1 - consistency['consistency_score'])
        
        # Check for excessive fallback usage
        for lib_name, counters in self.fallback_counters.items():
            total_calls = counters['mock_calls'] + counters['real_calls']
            if total_calls > 100 and counters['mock_calls'] / total_calls > 0.5:
                risks.append({
                    'type': 'EXCESSIVE_FALLBACK_USAGE',
                    'library': lib_name,
                    'severity': 'LOW',
                    'impact': f'High fallback rate: {counters["mock_calls"]}/{total_calls} calls are mocked',
                    'mitigation': 'Ensure proper library installation and configuration'
                })
                risk_score += 0.05
        
        risk_level = 'LOW' if risk_score < 0.2 else ('MEDIUM' if risk_score < 0.5 else 'HIGH')
        
        return {
            'overall_risk_score': min(1.0, risk_score),
            'risk_level': risk_level,
            'identified_risks': risks,
            'risk_count': len(risks),
            'recommendations': [
                'Review and address high severity risks immediately',
                'Monitor consistency test results regularly',
                'Keep libraries updated within compatibility constraints',
                'Implement proper fallback monitoring'
            ]
        }
    
    def save_report(self, report: Dict[str, Any] = None) -> str:
        """Save the comprehensive report to JSON file."""
        if report is None:
            report = self.generate_comprehensive_report()
        
        import json
# # #         from pathlib import Path  # Module not found  # Module not found  # Module not found
        
        report_path = Path(self.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(report_path)
    
    def invoke_status_reporting(self) -> Dict[str, Any]:
        """
        Main entry point for automated status reporting during pipeline startup.
        
        Returns:
            Dict with report summary and file location
        """
        try:
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Save to file
            report_path = self.save_report(report)
            
            # Return summary
            return {
                'success': True,
                'report_path': report_path,
                'system_health': report['system_health'],
                'critical_issues': report['system_health']['critical_issues'],
                'risk_level': report['risk_assessment']['risk_level'],
                'recommendations_count': len(report['upgrade_recommendations']),
                'timestamp': report['metadata']['generated_timestamp']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def detect_faiss_conflicts(self) -> List[str]:
        """Detect conflicts between faiss-cpu and faiss-gpu installations"""
        conflicts = []
        
        try:
            import faiss
            
            # Check if both CPU and GPU variants might be installed
            has_gpu_support = False
            try:
                # Try to access GPU resources
                if hasattr(faiss, 'get_num_gpus'):
                    gpu_count = faiss.get_num_gpus()
                    has_gpu_support = gpu_count > 0
            except Exception:
                pass
            
            # Check installation method via pip
            try:
                result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=10)
                pip_output = result.stdout.lower()
                
                has_faiss_cpu = 'faiss-cpu' in pip_output
                has_faiss_gpu = 'faiss-gpu' in pip_output
                
                if has_faiss_cpu and has_faiss_gpu:
                    conflicts.append("Both faiss-cpu and faiss-gpu are installed - this can cause conflicts")
                    
                if has_faiss_gpu and not has_gpu_support:
                    conflicts.append("faiss-gpu installed but no GPU support detected")
                    
                if has_faiss_cpu and has_gpu_support:
                    conflicts.append("Using faiss-cpu but GPU support is available - consider faiss-gpu for better performance")
                    
            except subprocess.TimeoutExpired:
                conflicts.append("Timeout checking FAISS installation via pip")
            except Exception as e:
                conflicts.append(f"Error checking FAISS installation: {e}")
                
        except ImportError:
            conflicts.append("FAISS not installed")
            
        return conflicts
    
    def validate_cross_platform_compatibility(self, target_platforms: List[str] = None, 
                                            target_python_versions: List[str] = None) -> List[CrossPlatformResult]:
        """
        Validate compatibility across different platforms and Python versions.
        
        Note: This simulates cross-platform testing based on known constraints.
        For actual cross-platform testing, this would need to run on multiple systems.
        """
        if target_platforms is None:
            target_platforms = ["Windows", "Linux", "Darwin"]
            
        if target_python_versions is None:
            target_python_versions = ["3.8", "3.9", "3.10", "3.11", "3.12"]
        
        results = []
        
        for platform_name in target_platforms:
            for py_version in target_python_versions:
                try:
                    py_ver_enum = PythonVersion(py_version)
                except ValueError:
                    continue
                
                platform_result = CrossPlatformResult(
                    platform=platform_name,
                    python_version=py_version,
                    is_compatible=True
                )
                
                # Check each library for this platform/Python combination
                for lib_name, lib_spec in self.library_specs.items():
                    lib_compatible = True
                    issues = []
                    
                    # Platform compatibility
                    if not lib_spec.platform_constraints.is_platform_supported(platform_name):
                        lib_compatible = False
                        issues.append(f"Platform {platform_name} not supported")
                    
                    # Python version compatibility
                    if py_ver_enum not in lib_spec.constraints:
                        lib_compatible = False
                        issues.append(f"Python {py_version} not supported")
                    
                    # Special platform-specific logic
                    if lib_name == "faiss-gpu" and platform_name == "Darwin":
                        lib_compatible = False
                        issues.append("faiss-gpu not officially supported on macOS")
                    
                    platform_result.library_results[lib_name] = CompatibilityResult(
                        is_compatible=lib_compatible,
                        issues=issues
                    )
                    
                    if not lib_compatible and not lib_spec.is_optional:
                        platform_result.is_compatible = False
                        platform_result.platform_specific_issues.extend(issues)
                
                # Generate recommendations
                if not platform_result.is_compatible:
                    if platform_name == "Darwin" and "faiss-gpu" in [str(issue) for issue in platform_result.platform_specific_issues]:
                        platform_result.recommendations.append("Use faiss-cpu instead of faiss-gpu on macOS")
                    
                    if py_version in ["3.8", "3.9"]:
                        platform_result.recommendations.append("Consider upgrading to Python 3.10+ for better library support")
                
                results.append(platform_result)
        
        return results
    
    def generate_unified_validation_report(self) -> ValidationReport:
        """Generate comprehensive unified validation report"""
# # #         from datetime import datetime  # Module not found  # Module not found  # Module not found
        
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            python_version=self.current_python_version.value,
            platform=self.current_platform
        )
        
        # Library compatibility
        report.library_compatibility = self.check_all_compatibility()
        
        # Stage compatibility
        for stage in StageEnhancer:
            stage_results = self.check_stage_compatibility(stage)
            report.stage_compatibility[stage.value] = {
                name: result.is_compatible
                for name, result in stage_results.items()
            }
        
        # Numerical stability
        try:
            report.numerical_stability = self.validate_numerical_stability()
        except Exception as e:
            report.critical_issues.append(f"Failed to validate numerical stability: {e}")
        
        # Cross-platform results
        try:
            report.cross_platform_results = self.validate_cross_platform_compatibility()
        except Exception as e:
            report.critical_issues.append(f"Failed cross-platform validation: {e}")
        
        # FAISS conflicts
        report.faiss_conflicts = self.detect_faiss_conflicts()
        
        # Generate comprehensive recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: ValidationReport):
        """Generate comprehensive recommendations based on validation results"""
        
        # Critical library issues
        for lib_name, result in report.library_compatibility.items():
            if not result.is_compatible:
                if result.recommendations:
                    report.recommendations.extend(result.recommendations)
                elif not self.library_specs[lib_name].is_optional:
                    install_cmd = self.library_specs[lib_name].install_command
                    if install_cmd:
                        report.recommendations.append(install_cmd)
                    else:
                        report.recommendations.append(f"Install {lib_name}")
        
        # FAISS conflict resolution
        if report.faiss_conflicts:
            if "Both faiss-cpu and faiss-gpu are installed" in str(report.faiss_conflicts):
                report.recommendations.append("Uninstall one FAISS variant: 'pip uninstall faiss-cpu faiss-gpu' then reinstall preferred version")
            
            if "faiss-gpu installed but no GPU support detected" in str(report.faiss_conflicts):
                report.recommendations.append("Switch to faiss-cpu: 'pip uninstall faiss-gpu && pip install faiss-cpu'")
        
        # Platform-specific recommendations  
        if report.platform == "Darwin":
            report.recommendations.append("macOS users: Consider using Conda for better package compatibility")
            
        if report.platform == "Windows":
            report.recommendations.append("Windows users: Ensure Visual C++ Redistributable is installed for compiled packages")
        
        # Python version recommendations
        if report.python_version in ["3.8", "3.9"]:
            report.recommendations.append("Consider upgrading to Python 3.10+ for better library support and performance")
        
        # Numerical stability recommendations
        if report.numerical_stability and not report.numerical_stability.is_stable:
            report.recommendations.append("Numerical instability detected - verify library versions and consider updating NumPy/SciPy")
        
        # Remove duplicates
        report.recommendations = list(dict.fromkeys(report.recommendations))
        
    def save_validation_report(self, report: ValidationReport, filepath: str):
        """Save validation report to JSON file"""
        
        # Convert dataclasses to dict for JSON serialization
        def to_dict(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if isinstance(value, list):
                        result[key] = [to_dict(item) for item in value]
                    elif hasattr(value, '__dict__'):
                        result[key] = to_dict(value)
                    else:
                        result[key] = value
                return result
            return obj
        
        report_dict = to_dict(report)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
    
    def unified_pipeline_startup_validation(self) -> Tuple[bool, ValidationReport]:
        """
        Unified validation entry point for pipeline startup.
        
        Returns:
            (is_valid, validation_report): Tuple indicating if pipeline can start safely
        """
        report = self.generate_unified_validation_report()
        
        # Check critical requirements
        critical_failures = []
        
        # Check required libraries for core stages
        core_stages = [
            StageEnhancer.OPTIMAL_TRANSPORT,
            StageEnhancer.INFORMATION_THEORY, 
            StageEnhancer.STATISTICAL_LEARNING
        ]
        
        for stage in core_stages:
            if stage.value in report.stage_compatibility:
                stage_compatible = all(report.stage_compatibility[stage.value].values())
                if not stage_compatible:
                    critical_failures.append(f"Core stage {stage.value} has incompatible libraries")
        
        # Check numerical stability
        if report.numerical_stability and not report.numerical_stability.is_stable:
            if report.numerical_stability.failed_operations:
                critical_failures.append("Critical numerical operations failed")
        
        # Check for blocking FAISS conflicts
        if report.faiss_conflicts:
            for conflict in report.faiss_conflicts:
                if "Both faiss-cpu and faiss-gpu are installed" in conflict:
                    critical_failures.append("FAISS installation conflict detected")
        
        report.critical_issues.extend(critical_failures)
        is_valid = len(critical_failures) == 0
        
        return is_valid, report
    
    def validate_stage_enhancer(self, stage: StageEnhancer) -> bool:
        """
        Validate that a stage enhancer has all required compatible libraries.
        
        Args:
            stage: Stage enhancer to validate
            
        Returns:
            True if all required libraries are compatible, False otherwise
        """
        
        stage_results = self.check_stage_compatibility(stage)
        
        for library, result in stage_results.items():
            spec = self.library_specs.get(library, None)
            
            # Skip optional libraries that aren't installed
            if spec and spec.is_optional and not result.is_compatible:
                continue
                
            # Fail if required library is incompatible
            if not result.is_compatible:
                return False
        
        return True


# Global compatibility matrix instance
_compatibility_matrix = None

def get_compatibility_matrix() -> MathematicalCompatibilityMatrix:
    """Get global compatibility matrix instance (singleton pattern)"""
    global _compatibility_matrix
    if _compatibility_matrix is None:
        _compatibility_matrix = MathematicalCompatibilityMatrix()
    return _compatibility_matrix


def check_stage_enhancer_compatibility(stage: StageEnhancer) -> bool:
    """
    Quick compatibility check for a stage enhancer.
    
    Args:
        stage: Stage enhancer to check
        
    Returns:
        True if compatible, False otherwise
    """
    matrix = get_compatibility_matrix()
    return matrix.validate_stage_enhancer(stage)


def validate_mathematical_environment() -> ValidationReport:
    """
    Comprehensive validation of the mathematical environment.
    
    Returns:
        Complete validation report
    """
    matrix = get_compatibility_matrix()
    return matrix.generate_unified_validation_report()


def ensure_numerical_stability(tolerance: float = 1e-10) -> bool:
    """
    Ensure numerical stability across all mathematical operations.
    
    Args:
        tolerance: Numerical tolerance for stability validation
        
    Returns:
        True if numerically stable, False otherwise
    """
    matrix = get_compatibility_matrix()
    result = matrix.validate_numerical_stability(tolerance=tolerance)
    return result.is_stable


# Convenience imports for stage enhancers
def validate_differential_geometry() -> bool:
    """Validate differential geometry stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.DIFFERENTIAL_GEOMETRY)

def validate_category_theory() -> bool:
    """Validate category theory stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.CATEGORY_THEORY)

def validate_topological_data_analysis() -> bool:
    """Validate topological data analysis stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.TOPOLOGICAL_DATA_ANALYSIS)

def validate_information_theory() -> bool:
    """Validate information theory stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.INFORMATION_THEORY)

def validate_optimal_transport() -> bool:
    """Validate optimal transport stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.OPTIMAL_TRANSPORT)

def validate_spectral_methods() -> bool:
    """Validate spectral methods stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.SPECTRAL_METHODS)

def validate_control_theory() -> bool:
    """Validate control theory stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.CONTROL_THEORY)

def validate_measure_theory() -> bool:
    """Validate measure theory stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.MEASURE_THEORY)

def validate_optimization_theory() -> bool:
    """Validate optimization theory stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.OPTIMIZATION_THEORY)

def validate_algebraic_topology() -> bool:
    """Validate algebraic topology stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.ALGEBRAIC_TOPOLOGY)

def validate_functional_analysis() -> bool:
    """Validate functional analysis stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.FUNCTIONAL_ANALYSIS)

def validate_statistical_learning() -> bool:
    """Validate statistical learning stage enhancer compatibility"""
    return check_stage_enhancer_compatibility(StageEnhancer.STATISTICAL_LEARNING)


# New unified validation functions
def startup_validation() -> Tuple[bool, ValidationReport]:
    """
    Unified validation entry point for pipeline startup.
    
    Returns:
        (is_valid, report): Boolean indicating if pipeline can start safely and detailed report
    """
    matrix = get_compatibility_matrix()
    return matrix.unified_pipeline_startup_validation()


def check_faiss_installation() -> List[str]:
    """Check FAISS installation for conflicts"""
    matrix = get_compatibility_matrix()
    return matrix.detect_faiss_conflicts()


def validate_cross_platform(platforms: List[str] = None, 
                           python_versions: List[str] = None) -> List[CrossPlatformResult]:
    """Validate cross-platform compatibility"""
    matrix = get_compatibility_matrix()
    return matrix.validate_cross_platform_compatibility(platforms, python_versions)


def generate_compatibility_report(output_file: str = None) -> ValidationReport:
    """Generate and optionally save comprehensive compatibility report"""
    matrix = get_compatibility_matrix()
    report = matrix.generate_unified_validation_report()
    
    if output_file:
        matrix.save_validation_report(report, output_file)
        
    return report


if __name__ == "__main__":
    # Demo usage with comprehensive validation
    print("Mathematical Compatibility Matrix - Unified Validation System")
    print("=" * 60)
    
    try:
        # Run startup validation
        is_valid, report = startup_validation()
        
        print(f"Pipeline Startup Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Platform: {report.platform}")  
        print(f"Python Version: {report.python_version}")
        
        if report.numerical_stability:
            print(f"Numerical Stability: {'✅' if report.numerical_stability.is_stable else '❌'}")
            if not report.numerical_stability.is_stable:
                print(f"  Max Error: {report.numerical_stability.max_error:.2e}")
                print(f"  Failed Operations: {report.numerical_stability.failed_operations}")
        
        print("\nLibrary Compatibility:")
        for lib, result in report.library_compatibility.items():
            icon = "✅" if result.is_compatible else "❌"
            version = result.installed_version or 'Not installed'
            print(f"  {icon} {lib}: {version}")
            
            if result.issues:
                for issue in result.issues[:2]:  # Show first 2 issues
                    print(f"    ⚠️  {issue}")
            
            if result.warnings:
                for warning in result.warnings[:1]:  # Show first warning
                    print(f"    ℹ️  {warning}")
        
        print("\nStage Enhancer Compatibility:")
        for stage, libs in report.stage_compatibility.items():
            all_compatible = all(libs.values()) if libs else False
            icon = "✅" if all_compatible else "❌"
            print(f"  {icon} {stage.replace('_', ' ').title()}")
        
        # FAISS conflicts
        if report.faiss_conflicts:
            print("\nFAISS Installation Issues:")
            for conflict in report.faiss_conflicts:
                print(f"  ⚠️  {conflict}")
        
        # Critical issues
        if report.critical_issues:
            print("\nCritical Issues:")
            for issue in report.critical_issues:
                print(f"  🚨 {issue}")
        
        # Recommendations
        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations[:10]:  # Show first 10 recommendations
                print(f"  • {rec}")
        
        # Cross-platform summary
        if report.cross_platform_results:
            print(f"\nCross-Platform Compatibility: {len([r for r in report.cross_platform_results if r.is_compatible])}/{len(report.cross_platform_results)} platform/version combinations supported")
        
        # Save detailed report
        report_file = "compatibility_validation_report.json"
        matrix = get_compatibility_matrix()
        matrix.save_validation_report(report, report_file)
        print(f"\nDetailed report saved to: {report_file}")
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        traceback.print_exc()