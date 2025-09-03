"""
Comprehensive Pre-Flight Validator for EGW Query Expansion System

This module provides comprehensive startup validation for all critical dependencies
with fail-fast behavior for missing libraries and graceful degradation for optional ones.

Main entry point: check_library_compatibility()

Integration points:
- ComprehensivePipelineOrchestrator: Called during __init__
- canonical_web_server: Called before server startup

Key features:
- Version checking with compatibility matrix lookups
- Mock fallback activation for optional dependencies
- Comprehensive logging of validation status
- SystemExit for critical missing dependencies
- Graceful degradation for optional dependencies
"""

import sys
import warnings
import logging
# # # from typing import Dict, List, Optional, Tuple, Any, Set  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
import importlib.util
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Import the existing compatibility matrix
# # # from .mathematical_compatibility_matrix import (  # Module not found  # Module not found  # Module not found
    MathematicalCompatibilityMatrix,
    PythonVersion,
    StageEnhancer,
    VersionConstraint,
    LibrarySpec,
    CompatibilityResult
)

# Configure logging
logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Classification of dependency criticality"""
    CRITICAL = "critical"
    IMPORTANT = "important"
    OPTIONAL = "optional"


@dataclass
class ValidationResult:
    """Result of complete system validation"""
    success: bool
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    activated_fallbacks: List[str] = field(default_factory=list)
    library_status: Dict[str, CompatibilityResult] = field(default_factory=dict)
    validation_time: float = 0.0


@dataclass
class DependencyConfig:
    """Configuration for dependency validation"""
    name: str
    import_names: List[str]
    dependency_type: DependencyType
    fallback_modules: List[str] = field(default_factory=list)
    version_check_attribute: str = "__version__"
    critical_for_stages: List[StageEnhancer] = field(default_factory=list)


class MockFallbackRegistry:
    """Registry for mock fallback implementations"""
    
    def __init__(self):
        self.active_mocks: Set[str] = set()
        self.mock_implementations: Dict[str, Any] = {}
    
    def activate_faiss_fallback(self):
        """Activate FAISS fallback using sklearn NearestNeighbors"""
        if "faiss" not in self.active_mocks:
            try:
                import numpy as np
# # #                 from sklearn.neighbors import NearestNeighbors  # Module not found  # Module not found  # Module not found
                
                class MockFAISS:
                    """Mock FAISS implementation using sklearn"""
                    
                    @staticmethod
                    def IndexFlatL2(dim):
                        return MockFAISSIndex(dim)
                    
                    @staticmethod
                    def IndexFlatIP(dim):
                        return MockFAISSIndex(dim, metric='cosine')
                
                class MockFAISSIndex:
                    def __init__(self, dim, metric='l2'):
                        self.dim = dim
                        self.metric = metric
                        self.vectors = None
                        self.nn_model = None
                        self.ntotal = 0
                    
                    def add(self, vectors):
                        if self.vectors is None:
                            self.vectors = vectors
                        else:
                            self.vectors = np.vstack([self.vectors, vectors])
                        
                        self.ntotal = len(self.vectors)
                        # Rebuild nearest neighbors model
                        metric = 'cosine' if self.metric == 'cosine' else 'euclidean'
                        self.nn_model = NearestNeighbors(
                            n_neighbors=min(10, self.ntotal),
                            metric=metric
                        ).fit(self.vectors)
                    
                    def search(self, query_vectors, k):
                        if self.nn_model is None:
                            return np.array([]), np.array([[]])
                        
                        k = min(k, self.ntotal)
                        distances, indices = self.nn_model.kneighbors(query_vectors, n_neighbors=k)
                        return distances, indices
                
                self.mock_implementations["faiss"] = MockFAISS()
                self.active_mocks.add("faiss")
                logger.warning("FAISS fallback activated using sklearn NearestNeighbors")
                
            except ImportError:
                logger.error("Cannot activate FAISS fallback: sklearn not available")
    
    def activate_pytorch_fallback(self):
        """Activate PyTorch fallback using numpy"""
        if "torch" not in self.active_mocks:
            try:
                import numpy as np
                
                class MockTorch:
                    """Basic PyTorch fallback using numpy"""
                    
                    @staticmethod
                    def tensor(data):
                        return np.array(data)
                    
                    @staticmethod
                    def zeros(*shape):
                        return np.zeros(shape)
                    
                    @staticmethod
                    def ones(*shape):
                        return np.ones(shape)
                    
                    @staticmethod
                    def randn(*shape):
                        return np.random.randn(*shape)
                    
                    class nn:
                        class Module:
                            def __init__(self):
                                pass
                            
                            def forward(self, x):
                                return x
                
                self.mock_implementations["torch"] = MockTorch()
                self.active_mocks.add("torch")
                logger.warning("PyTorch fallback activated using numpy")
                
            except ImportError:
                logger.error("Cannot activate PyTorch fallback: numpy not available")
    
    def activate_sentence_transformers_fallback(self):
        """Activate sentence-transformers fallback"""
        if "sentence-transformers" not in self.active_mocks:
            try:
                import numpy as np
                
                class MockSentenceTransformer:
                    """Mock sentence transformer using random embeddings"""
                    
                    def __init__(self, model_name):
                        self.model_name = model_name
                        self.embedding_dim = 384  # Common dimension
                    
                    def encode(self, sentences, **kwargs):
                        if isinstance(sentences, str):
                            sentences = [sentences]
                        # Return random embeddings (for testing only)
                        return np.random.randn(len(sentences), self.embedding_dim)
                
                self.mock_implementations["sentence_transformers"] = {
                    "SentenceTransformer": MockSentenceTransformer
                }
                self.active_mocks.add("sentence-transformers")
                logger.warning("sentence-transformers fallback activated with random embeddings")
                
            except ImportError:
                logger.error("Cannot activate sentence-transformers fallback: numpy not available")


class PreFlightValidator:
    """Comprehensive pre-flight validation system"""
    
    def __init__(self):
        self.compatibility_matrix = MathematicalCompatibilityMatrix()
        self.fallback_registry = MockFallbackRegistry()
        self.dependency_configs = self._initialize_dependency_configs()
        
    def _initialize_dependency_configs(self) -> Dict[str, DependencyConfig]:
        """Initialize dependency configurations"""
        configs = {}
        
        # Critical dependencies - system cannot function without these
        configs["numpy"] = DependencyConfig(
            name="numpy",
            import_names=["numpy"],
            dependency_type=DependencyType.CRITICAL,
            critical_for_stages=list(StageEnhancer)  # Required by all stages
        )
        
        configs["scipy"] = DependencyConfig(
            name="scipy", 
            import_names=["scipy"],
            dependency_type=DependencyType.CRITICAL,
            critical_for_stages=[
                StageEnhancer.DIFFERENTIAL_GEOMETRY,
                StageEnhancer.INFORMATION_THEORY,
                StageEnhancer.SPECTRAL_METHODS,
                StageEnhancer.MEASURE_THEORY,
                StageEnhancer.OPTIMIZATION_THEORY
            ]
        )
        
        configs["scikit-learn"] = DependencyConfig(
            name="scikit-learn",
            import_names=["sklearn"],
            dependency_type=DependencyType.CRITICAL,
            critical_for_stages=[
                StageEnhancer.TOPOLOGICAL_DATA_ANALYSIS,
                StageEnhancer.INFORMATION_THEORY,
                StageEnhancer.SPECTRAL_METHODS,
                StageEnhancer.STATISTICAL_LEARNING
            ]
        )
        
        # Important dependencies - enable core functionality
        configs["pytorch"] = DependencyConfig(
            name="pytorch",
            import_names=["torch"],
            dependency_type=DependencyType.IMPORTANT,
            fallback_modules=["numpy"],
            critical_for_stages=[
                StageEnhancer.DIFFERENTIAL_GEOMETRY,
                StageEnhancer.OPTIMAL_TRANSPORT,
                StageEnhancer.FUNCTIONAL_ANALYSIS
            ]
        )
        
        configs["pandas"] = DependencyConfig(
            name="pandas",
            import_names=["pandas"],
            dependency_type=DependencyType.IMPORTANT,
            fallback_modules=["numpy"]
        )
        
        configs["faiss"] = DependencyConfig(
            name="faiss",
            import_names=["faiss"],
            dependency_type=DependencyType.IMPORTANT,
            fallback_modules=["sklearn.neighbors"]
        )
        
        configs["sentence-transformers"] = DependencyConfig(
            name="sentence-transformers",
            import_names=["sentence_transformers"],
            dependency_type=DependencyType.IMPORTANT,
            fallback_modules=["transformers", "torch"]
        )
        
        # Optional dependencies - enhance functionality but not critical
        configs["transformers"] = DependencyConfig(
            name="transformers",
            import_names=["transformers"],
            dependency_type=DependencyType.OPTIONAL
        )
        
        configs["POT"] = DependencyConfig(
            name="POT",
            import_names=["ot"],
            dependency_type=DependencyType.OPTIONAL,
            critical_for_stages=[StageEnhancer.OPTIMAL_TRANSPORT]
        )
        
        return configs
    
    def _check_import_availability(self, import_names: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if any of the import names can be imported"""
        for import_name in import_names:
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                return True, import_name, version
            except ImportError:
                continue
        return False, None, None
    
    def _validate_single_dependency(self, config: DependencyConfig) -> CompatibilityResult:
        """Validate a single dependency"""
        available, import_name, version = self._check_import_availability(config.import_names)
        
        if not available:
            if config.dependency_type == DependencyType.CRITICAL:
                return CompatibilityResult(
                    is_compatible=False,
                    issues=[f"Critical dependency {config.name} not available"]
                )
            else:
                return CompatibilityResult(
                    is_compatible=True,  # Optional, so compatible even if missing
                    warnings=[f"Optional dependency {config.name} not available"]
                )
        
        # Check version compatibility using the mathematical compatibility matrix
        matrix_result = self.compatibility_matrix.check_library_compatibility(config.name)
        
        return CompatibilityResult(
            is_compatible=matrix_result.is_compatible,
            installed_version=version,
            required_version=matrix_result.required_version,
            issues=matrix_result.issues or [],
            warnings=matrix_result.warnings or []
        )
    
    def _activate_fallbacks(self, failed_dependencies: List[str]) -> List[str]:
        """Activate fallback implementations for failed dependencies"""
        activated = []
        
        for dep_name in failed_dependencies:
            if dep_name == "faiss":
                self.fallback_registry.activate_faiss_fallback()
                activated.append("faiss -> sklearn.neighbors")
            elif dep_name == "pytorch":
                self.fallback_registry.activate_pytorch_fallback()
                activated.append("pytorch -> numpy")
            elif dep_name == "sentence-transformers":
                self.fallback_registry.activate_sentence_transformers_fallback()
                activated.append("sentence-transformers -> mock embeddings")
        
        return activated
    
    def check_library_compatibility(self) -> ValidationResult:
        """
        Main entry point for comprehensive library compatibility validation.
        
        Returns:
            ValidationResult with complete validation status
            
        Raises:
            SystemExit: If critical dependencies are missing and cannot be resolved
        """
        start_time = time.time()
        logger.info("Starting comprehensive library compatibility validation...")
        
        result = ValidationResult(success=True)
        critical_failures = []
        important_failures = []
        
        # Validate each dependency
        for name, config in self.dependency_configs.items():
            logger.debug(f"Validating dependency: {name}")
            validation = self._validate_single_dependency(config)
            result.library_status[name] = validation
            
            if not validation.is_compatible:
                if config.dependency_type == DependencyType.CRITICAL:
                    critical_failures.append(name)
                    result.critical_failures.extend(validation.issues or [])
                elif config.dependency_type == DependencyType.IMPORTANT:
                    important_failures.append(name)
            
            if validation.warnings:
                result.warnings.extend(validation.warnings)
        
        # Handle critical failures - these cause system exit
        if critical_failures:
            logger.error(f"Critical dependencies missing: {critical_failures}")
            for failure in result.critical_failures:
                logger.error(f"CRITICAL: {failure}")
            
            print("\n" + "="*80)
            print("CRITICAL DEPENDENCY VALIDATION FAILURE")
            print("="*80)
            print(f"Missing critical dependencies: {', '.join(critical_failures)}")
            print("\nRequired actions:")
            print("1. Activate your virtual environment: source venv/bin/activate")
            print("2. Install missing dependencies: pip install -r requirements.txt")
            print("3. Run validation: python validate_installation.py")
            print("="*80)
            
            # Exit with error code 1 for critical failures
            raise SystemExit(1)
        
        # Handle important failures with graceful degradation
        if important_failures:
            logger.warning(f"Important dependencies missing, activating fallbacks: {important_failures}")
            activated_fallbacks = self._activate_fallbacks(important_failures)
            result.activated_fallbacks = activated_fallbacks
            
            for fallback in activated_fallbacks:
                logger.warning(f"Activated fallback: {fallback}")
        
        # Additional validation checks
        self._validate_version_compatibility(result)
        self._validate_stage_dependencies(result)
        
        result.validation_time = time.time() - start_time
        
        # Log summary
        if result.success and not critical_failures:
            logger.info(f"✓ Library compatibility validation completed in {result.validation_time:.2f}s")
            if result.activated_fallbacks:
                logger.warning(f"Running with {len(result.activated_fallbacks)} fallback implementations")
            if result.warnings:
                logger.warning(f"Validation completed with {len(result.warnings)} warnings")
        
        return result
    
    def _validate_version_compatibility(self, result: ValidationResult):
        """Additional version compatibility validation"""
        python_version = self.compatibility_matrix.current_python_version
        
        # Check if Python version is supported
        if python_version == PythonVersion.PYTHON_38:
            result.warnings.append("Running on Python 3.8 - consider upgrading for better performance")
        
        # Check for specific version conflicts
        numpy_result = result.library_status.get("numpy")
        scipy_result = result.library_status.get("scipy")
        
        if (numpy_result and numpy_result.installed_version and 
            scipy_result and scipy_result.installed_version):
            # Add specific numpy/scipy compatibility checks if needed
            pass
    
    def _validate_stage_dependencies(self, result: ValidationResult):
        """Validate dependencies for each mathematical stage enhancer"""
        stage_issues = []
        
        for stage in StageEnhancer:
            stage_deps = self.compatibility_matrix.stage_dependencies.get(stage, [])
            missing_deps = []
            
            for dep in stage_deps:
                dep_result = result.library_status.get(dep)
                if dep_result and not dep_result.is_compatible:
                    missing_deps.append(dep)
            
            if missing_deps:
                stage_issues.append(f"Stage {stage.value} missing dependencies: {missing_deps}")
        
        if stage_issues:
            result.warnings.extend(stage_issues)
    
    def get_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """Generate a comprehensive validation summary"""
        return {
            "validation_success": result.success,
            "validation_time": result.validation_time,
            "python_version": self.compatibility_matrix.current_python_version.value,
            "total_dependencies": len(self.dependency_configs),
            "critical_failures": len(result.critical_failures),
            "activated_fallbacks": len(result.activated_fallbacks),
            "total_warnings": len(result.warnings),
            "dependency_status": {
                name: {
                    "available": status.is_compatible,
                    "version": status.installed_version,
                    "type": self.dependency_configs[name].dependency_type.value
                }
                for name, status in result.library_status.items()
            },
            "fallback_implementations": result.activated_fallbacks,
            "recommendations": self._generate_recommendations(result)
        }
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if result.activated_fallbacks:
            recommendations.append(
                "Install missing important dependencies for full functionality: "
                f"pip install {' '.join([fb.split(' -> ')[0] for fb in result.activated_fallbacks])}"
            )
        
        if any("PyTorch" in warning for warning in result.warnings):
            recommendations.append("Consider installing PyTorch with CUDA support for GPU acceleration")
        
        if len(result.warnings) > 5:
            recommendations.append("Multiple warnings detected - review system configuration")
        
        return recommendations


# Global validator instance
_validator_instance = None


def get_validator() -> PreFlightValidator:
    """Get singleton validator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = PreFlightValidator()
    return _validator_instance


def check_library_compatibility() -> ValidationResult:
    """
    Main entry point for library compatibility validation.
    
    This function should be called at system startup before any processing begins.
    
    Returns:
        ValidationResult with validation status
        
    Raises:
        SystemExit: If critical dependencies are missing
    """
    validator = get_validator()
    return validator.check_library_compatibility()


def validate_and_configure_system() -> Tuple[ValidationResult, Dict[str, Any]]:
    """
    Complete system validation and configuration.
    
    Returns:
        Tuple of (ValidationResult, configuration summary)
    """
    result = check_library_compatibility()
    validator = get_validator()
    summary = validator.get_validation_summary(result)
    
    return result, summary


# Import time to avoid circular imports
import time

if __name__ == "__main__":
    # Standalone validation when run directly
    print("EGW Query Expansion - Pre-Flight Validation")
    print("=" * 50)
    
    try:
        result, summary = validate_and_configure_system()
        
        print(f"Validation completed in {result.validation_time:.2f}s")
        print(f"Python version: {summary['python_version']}")
        print(f"Dependencies checked: {summary['total_dependencies']}")
        
        if result.success:
            print("✓ All critical dependencies available")
        
        if result.activated_fallbacks:
            print(f"⚠ {len(result.activated_fallbacks)} fallback implementations activated")
            for fb in result.activated_fallbacks:
                print(f"  - {fb}")
        
        if result.warnings:
            print(f"⚠ {len(result.warnings)} warnings:")
            for warning in result.warnings[:5]:  # Show first 5 warnings
                print(f"  - {warning}")
        
        print("\nValidation successful!")
        
    except SystemExit:
        print("\nValidation failed - system cannot start with missing critical dependencies")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected validation error: {e}")
        sys.exit(1)