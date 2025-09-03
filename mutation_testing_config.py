"""
Mutation Testing Configuration

This module defines mutation testing configuration for validator modules and 
context/synthesis merger components, specifying target directories, test command 
patterns, and mutation score thresholds.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class MutationTool(Enum):
    """Supported mutation testing tools."""
    MUTMUT = "mutmut"
    COSMIC_RAY = "cosmic-ray"

class ComponentType(Enum):
    """Types of components for mutation testing."""
    VALIDATOR = "validator"
    CONTEXT_SYNTHESIS = "context_synthesis"
    CRITICAL_INVARIANT = "critical_invariant"

@dataclass
class MutationTestTarget:
    """Configuration for a mutation testing target."""
    name: str
    paths: List[str]
    test_patterns: List[str]
    component_type: ComponentType
    min_mutation_score: float
    max_runtime_minutes: int = 30
    tool: MutationTool = MutationTool.MUTMUT
    exclude_patterns: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = []

class MutationTestingConfig:
    """Central configuration for mutation testing."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.targets = self._initialize_targets()
        
    def _initialize_targets(self) -> Dict[str, MutationTestTarget]:
        """Initialize mutation testing targets."""
        return {
            # Validator modules - critical for ensuring data integrity
            "validators": MutationTestTarget(
                name="validator_modules",
                paths=[
                    "canonical_flow/I_ingestion_preparation/gate_validation_system.py",
                    "canonical_flow/I_ingestion_preparation/preflight_validation.py", 
                    "canonical_flow/K_knowledge_extraction/gate_validator.py",
                    "canonical_flow/L_classification_evaluation/schemas.py",
                    "canonical_flow/analysis/audit_validation.py",
                    "canonical_flow/A_analysis_nlp/evidence_validation_model.py",
                    "egw_query_expansion/core/early_error_detector.py",
                    "egw_query_expansion/core/conformal_risk_control.py",
                    "constraint_validator.py",
                    "contract_validator.py",
                    "normative_validator.py",
                    "rubric_validator.py",
                    "validator.py"
                ],
                test_patterns=[
                    "tests/**/test_*validation*.py",
                    "tests/**/test_*validator*.py", 
                    "canonical_flow/*/tests/test_*validation*.py",
                    "egw_query_expansion/tests/test_*validator*.py",
                    "egw_query_expansion/tests/test_early_error_detector.py"
                ],
                component_type=ComponentType.VALIDATOR,
                min_mutation_score=85.0,  # High threshold for validators
                max_runtime_minutes=45
            ),
            
            # Context and synthesis operations - critical for output quality
            "context_synthesis": MutationTestTarget(
                name="context_synthesis_components", 
                paths=[
                    "canonical_flow/mathematical_enhancers/context_enhancer.py",
                    "canonical_flow/mathematical_enhancers/synthesis_enhancer.py",
                    "egw_query_expansion/core/context_adapter.py",
                    "egw_query_expansion/core/immutable_context.py",
                    "canonical_flow/X_context_construction/context_orchestrator.py",
                    "canonical_flow/S_synthesis_output/synthesis_orchestrator.py",
                    "answer_synthesizer.py",
                    "context_adapter.py"
                ],
                test_patterns=[
                    "tests/**/test_*context*.py",
                    "tests/**/test_*synthesis*.py",
                    "canonical_flow/X_context_construction/tests/test_*.py",
                    "canonical_flow/S_synthesis_output/tests/test_*.py",
                    "egw_query_expansion/tests/test_context_adapter.py"
                ],
                component_type=ComponentType.CONTEXT_SYNTHESIS,
                min_mutation_score=80.0,  # Standard threshold for synthesis
                max_runtime_minutes=35
            ),
            
            # Critical invariant code - highest threshold
            "critical_invariants": MutationTestTarget(
                name="critical_invariant_code",
                paths=[
                    "canonical_flow/mathematical_enhancers/mathematical_compatibility_matrix.py",
                    "canonical_flow/mathematical_enhancers/mathematical_pipeline_coordinator.py",
                    "egw_query_expansion/core/mathematical_safety_controller.py",
                    "egw_query_expansion/core/linear_type_enforcer.py",
                    "egw_query_expansion/core/total_ordering.py",
                    "deterministic_pipeline_validator.py",
                    "deterministic_shield.py"
                ],
                test_patterns=[
                    "tests/**/test_*mathematical*.py",
                    "tests/**/test_*deterministic*.py",
                    "tests/**/test_*invariant*.py",
                    "egw_query_expansion/tests/test_*mathematical*.py",
                    "egw_query_expansion/tests/test_*linear*.py"
                ],
                component_type=ComponentType.CRITICAL_INVARIANT,
                min_mutation_score=90.0,  # Highest threshold for invariants
                max_runtime_minutes=60,
                exclude_patterns=[
                    "**/test_*.py",  # Don't mutate test files themselves
                    "**/__pycache__/**",
                    "**/.*"
                ]
            )
        }
    
    def get_target(self, name: str) -> Optional[MutationTestTarget]:
        """Get mutation testing target by name."""
        return self.targets.get(name)
    
    def get_all_targets(self) -> List[MutationTestTarget]:
        """Get all mutation testing targets."""
        return list(self.targets.values())
    
    def get_targets_by_type(self, component_type: ComponentType) -> List[MutationTestTarget]:
        """Get targets filtered by component type."""
        return [target for target in self.targets.values() 
                if target.component_type == component_type]
    
    def get_mutmut_config(self, target: MutationTestTarget) -> Dict[str, Union[str, List[str], float]]:
        """Generate mutmut configuration for a target."""
        # Filter paths that exist
        existing_paths = []
        for path in target.paths:
            full_path = self.project_root / path
            if full_path.exists():
                existing_paths.append(str(full_path))
        
        config = {
            "paths_to_mutate": existing_paths,
            "tests_dir": "tests/",
            "test_command": f"python -m pytest {' '.join(target.test_patterns)} -x",
            "runner": "python_unittest",
            "mutation_threshold": target.min_mutation_score,
            "timeout": target.max_runtime_minutes * 60
        }
        
        if target.exclude_patterns:
            config["exclude"] = target.exclude_patterns
            
        return config
    
    def get_cosmic_ray_config(self, target: MutationTestTarget) -> Dict[str, Union[str, List[str], float]]:
        """Generate cosmic-ray configuration for a target."""
        # Filter paths that exist
        existing_paths = []
        for path in target.paths:
            full_path = self.project_root / path
            if full_path.exists():
                existing_paths.append(str(full_path))
        
        config = {
            "module": existing_paths,
            "test_command": f"python -m pytest {' '.join(target.test_patterns)}",
            "timeout": target.max_runtime_minutes,
            "baseline": 10,  # Number of baseline executions
            "mutation_threshold": target.min_mutation_score
        }
        
        return config

# Global configuration instance
MUTATION_CONFIG = MutationTestingConfig()

# Convenience functions for common operations
def get_validator_targets() -> List[MutationTestTarget]:
    """Get all validator mutation testing targets."""
    return MUTATION_CONFIG.get_targets_by_type(ComponentType.VALIDATOR)

def get_context_synthesis_targets() -> List[MutationTestTarget]:
    """Get all context/synthesis mutation testing targets.""" 
    return MUTATION_CONFIG.get_targets_by_type(ComponentType.CONTEXT_SYNTHESIS)

def get_critical_invariant_targets() -> List[MutationTestTarget]:
    """Get all critical invariant mutation testing targets."""
    return MUTATION_CONFIG.get_targets_by_type(ComponentType.CRITICAL_INVARIANT)

def get_minimum_thresholds() -> Dict[str, float]:
    """Get minimum mutation score thresholds for each target type."""
    return {
        "validators": 85.0,
        "context_synthesis": 80.0, 
        "critical_invariants": 90.0
    }

if __name__ == "__main__":
    # Quick test of configuration
    config = MUTATION_CONFIG
    print(f"Loaded {len(config.targets)} mutation testing targets:")
    for name, target in config.targets.items():
        print(f"  {name}: {target.component_type.value} "
              f"(threshold: {target.min_mutation_score}%)")