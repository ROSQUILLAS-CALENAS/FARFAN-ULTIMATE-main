"""
Phase layering enforcement tests using architecture fitness functions.

These tests analyze the import graph to detect backward dependencies
that violate the canonical phase flow: I → X → K → A → L → R → O → G → T → S
"""

import ast
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest


class PhaseViolationError(Exception):
    """Raised when a phase layering violation is detected."""
    pass


class ImportAnalyzer:
    """Analyzes Python modules for import dependencies."""
    
    def __init__(self, root_path: str = "canonical_flow"):
        self.root_path = Path(root_path)
        self.phase_order = [
            "I_ingestion_preparation",
            "X_context_construction", 
            "K_knowledge_extraction",
            "A_analysis_nlp",
            "L_classification_evaluation",
            "R_search_retrieval",
            "O_orchestration_control",
            "G_aggregation_reporting",
            "T_integration_storage",
            "S_synthesis_output"
        ]
        self.phase_index = {phase: i for i, phase in enumerate(self.phase_order)}
    
    def get_imports_from_file(self, file_path: Path) -> List[str]:
        """Extract import statements from a Python file."""
        if not file_path.exists() or file_path.suffix != '.py':
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return imports
        except (SyntaxError, UnicodeDecodeError):
            return []
    
    def get_phase_from_module(self, module_name: str) -> str:
        """Extract phase name from module path."""
        if not module_name.startswith("canonical_flow."):
            return None
        
        parts = module_name.split('.')
        if len(parts) >= 2:
            potential_phase = parts[1]
            if potential_phase in self.phase_index:
                return potential_phase
        
        return None
    
    def analyze_phase_dependencies(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Analyze all modules and return violations of phase ordering.
        
        Returns:
            Dict mapping phase names to list of violations.
            Each violation is (source_file, target_module, target_phase).
        """
        violations = {}
        
        for phase in self.phase_order:
            phase_path = self.root_path / phase
            if not phase_path.exists():
                continue
            
            violations[phase] = []
            
            for py_file in phase_path.glob("**/*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                imports = self.get_imports_from_file(py_file)
                
                for import_name in imports:
                    target_phase = self.get_phase_from_module(import_name)
                    
                    if target_phase and target_phase in self.phase_index:
                        source_phase_idx = self.phase_index[phase]
                        target_phase_idx = self.phase_index[target_phase]
                        
                        # Violation: importing from earlier phase (backward dependency)
                        if target_phase_idx <= source_phase_idx:
                            violations[phase].append((
                                str(py_file.relative_to(self.root_path)),
                                import_name,
                                target_phase
                            ))
        
        return violations


@pytest.fixture
def import_analyzer():
    """Create an ImportAnalyzer instance."""
    return ImportAnalyzer()


@pytest.mark.architecture
@pytest.mark.phase_enforcement
class TestPhaseLayerEnforcement:
    """Architecture fitness functions for phase layering enforcement."""
    
    def test_no_backward_dependencies_ingestion_to_context(self, import_analyzer):
        """Test that X_context_construction doesn't import from I_ingestion_preparation."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        x_violations = [
            v for v in violations.get("X_context_construction", [])
            if v[2] == "I_ingestion_preparation"
        ]
        
        if x_violations:
            error_msg = f"Phase X_context_construction has backward dependencies to I_ingestion_preparation:\n"
            for file_path, import_name, target_phase in x_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_no_backward_dependencies_knowledge_to_earlier_phases(self, import_analyzer):
        """Test that K_knowledge_extraction doesn't import from I or X phases."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        k_violations = [
            v for v in violations.get("K_knowledge_extraction", [])
            if v[2] in ["I_ingestion_preparation", "X_context_construction"]
        ]
        
        if k_violations:
            error_msg = f"Phase K_knowledge_extraction has backward dependencies:\n"
            for file_path, import_name, target_phase in k_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_no_backward_dependencies_analysis_to_earlier_phases(self, import_analyzer):
        """Test that A_analysis_nlp doesn't import from I, X, or K phases."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        a_violations = [
            v for v in violations.get("A_analysis_nlp", [])
            if v[2] in ["I_ingestion_preparation", "X_context_construction", "K_knowledge_extraction"]
        ]
        
        if a_violations:
            error_msg = f"Phase A_analysis_nlp has backward dependencies:\n"
            for file_path, import_name, target_phase in a_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_no_backward_dependencies_classification_to_earlier_phases(self, import_analyzer):
        """Test that L_classification_evaluation doesn't import from I, X, K, or A phases."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        l_violations = [
            v for v in violations.get("L_classification_evaluation", [])
            if v[2] in ["I_ingestion_preparation", "X_context_construction", 
                       "K_knowledge_extraction", "A_analysis_nlp"]
        ]
        
        if l_violations:
            error_msg = f"Phase L_classification_evaluation has backward dependencies:\n"
            for file_path, import_name, target_phase in l_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_no_backward_dependencies_retrieval_to_earlier_phases(self, import_analyzer):
        """Test that R_search_retrieval doesn't import from I, X, K, A, or L phases."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        r_violations = [
            v for v in violations.get("R_search_retrieval", [])
            if v[2] in ["I_ingestion_preparation", "X_context_construction", 
                       "K_knowledge_extraction", "A_analysis_nlp", "L_classification_evaluation"]
        ]
        
        if r_violations:
            error_msg = f"Phase R_search_retrieval has backward dependencies:\n"
            for file_path, import_name, target_phase in r_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_no_backward_dependencies_orchestration_to_earlier_phases(self, import_analyzer):
        """Test that O_orchestration_control doesn't import from I, X, K, A, L, or R phases."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        o_violations = [
            v for v in violations.get("O_orchestration_control", [])
            if v[2] in ["I_ingestion_preparation", "X_context_construction", 
                       "K_knowledge_extraction", "A_analysis_nlp", "L_classification_evaluation",
                       "R_search_retrieval"]
        ]
        
        if o_violations:
            error_msg = f"Phase O_orchestration_control has backward dependencies:\n"
            for file_path, import_name, target_phase in o_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_no_backward_dependencies_aggregation_to_earlier_phases(self, import_analyzer):
        """Test that G_aggregation_reporting doesn't import from earlier phases."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        g_violations = [
            v for v in violations.get("G_aggregation_reporting", [])
            if v[2] in ["I_ingestion_preparation", "X_context_construction", 
                       "K_knowledge_extraction", "A_analysis_nlp", "L_classification_evaluation",
                       "R_search_retrieval", "O_orchestration_control"]
        ]
        
        if g_violations:
            error_msg = f"Phase G_aggregation_reporting has backward dependencies:\n"
            for file_path, import_name, target_phase in g_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_no_backward_dependencies_integration_to_earlier_phases(self, import_analyzer):
        """Test that T_integration_storage doesn't import from earlier phases."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        t_violations = [
            v for v in violations.get("T_integration_storage", [])
            if v[2] in ["I_ingestion_preparation", "X_context_construction", 
                       "K_knowledge_extraction", "A_analysis_nlp", "L_classification_evaluation",
                       "R_search_retrieval", "O_orchestration_control", "G_aggregation_reporting"]
        ]
        
        if t_violations:
            error_msg = f"Phase T_integration_storage has backward dependencies:\n"
            for file_path, import_name, target_phase in t_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_no_backward_dependencies_synthesis_to_earlier_phases(self, import_analyzer):
        """Test that S_synthesis_output doesn't import from any earlier phases."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        s_violations = [
            v for v in violations.get("S_synthesis_output", [])
            if v[2] in ["I_ingestion_preparation", "X_context_construction", 
                       "K_knowledge_extraction", "A_analysis_nlp", "L_classification_evaluation",
                       "R_search_retrieval", "O_orchestration_control", "G_aggregation_reporting",
                       "T_integration_storage"]
        ]
        
        if s_violations:
            error_msg = f"Phase S_synthesis_output has backward dependencies:\n"
            for file_path, import_name, target_phase in s_violations:
                error_msg += f"  - {file_path} imports {import_name} from {target_phase}\n"
            raise PhaseViolationError(error_msg)
    
    def test_comprehensive_phase_ordering_validation(self, import_analyzer):
        """Comprehensive test to validate all phase ordering constraints."""
        violations = import_analyzer.analyze_phase_dependencies()
        
        all_violations = []
        for phase, phase_violations in violations.items():
            all_violations.extend([(phase, v) for v in phase_violations])
        
        if all_violations:
            error_msg = f"Found {len(all_violations)} phase layering violations:\n\n"
            
            for source_phase, (file_path, import_name, target_phase) in all_violations:
                error_msg += f"VIOLATION: {source_phase} → {target_phase}\n"
                error_msg += f"  File: {file_path}\n"
                error_msg += f"  Import: {import_name}\n"
                error_msg += f"  Expected flow: I → X → K → A → L → R → O → G → T → S\n\n"
            
            raise PhaseViolationError(error_msg)


@pytest.mark.architecture
def test_phase_directories_exist():
    """Test that all expected phase directories exist."""
    expected_phases = [
        "I_ingestion_preparation",
        "X_context_construction", 
        "K_knowledge_extraction",
        "A_analysis_nlp",
        "L_classification_evaluation",
        "R_search_retrieval",
        "O_orchestration_control",
        "G_aggregation_reporting",
        "T_integration_storage",
        "S_synthesis_output"
    ]
    
    root_path = Path("canonical_flow")
    missing_phases = []
    
    for phase in expected_phases:
        phase_path = root_path / phase
        if not phase_path.exists():
            missing_phases.append(phase)
    
    if missing_phases:
        pytest.fail(f"Missing phase directories: {missing_phases}")


if __name__ == "__main__":
    # Allow running this file directly for debugging
    analyzer = ImportAnalyzer()
    violations = analyzer.analyze_phase_dependencies()
    
    total_violations = sum(len(v) for v in violations.values())
    print(f"Found {total_violations} total violations:")
    
    for phase, phase_violations in violations.items():
        if phase_violations:
            print(f"\n{phase}:")
            for file_path, import_name, target_phase in phase_violations:
                print(f"  {file_path} imports {import_name} from {target_phase}")