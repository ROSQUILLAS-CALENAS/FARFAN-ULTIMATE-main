"""
Pipeline Contract Annotations System
Defines mandatory static contract annotations for all canonical pipeline components.
"""

from enum import Enum
from typing import Dict, Any, Set
import inspect
import importlib
import ast
from pathlib import Path


class PipelinePhase(Enum):
    """Canonical pipeline phase sequence I→X→K→A→L→R→O→G→T→S"""
    INGESTION = "I"              # I_ingestion_preparation
    CONTEXT = "X"                # X_context_construction  
    KNOWLEDGE = "K"              # K_knowledge_extraction
    ANALYSIS = "A"               # A_analysis_nlp
    CLASSIFICATION = "L"         # L_classification_evaluation
    RETRIEVAL = "R"              # R_search_retrieval
    ORCHESTRATION = "O"          # O_orchestration_control
    AGGREGATION = "G"            # G_aggregation_reporting
    INTEGRATION = "T"            # T_integration_storage
    SYNTHESIS = "S"              # S_synthesis_output


class ComponentAnnotations:
    """Mandatory static contract annotations for pipeline components"""
    
    # Phase mapping for existing components based on canonical flow structure
    PHASE_MAPPING = {
        "I_ingestion_preparation": PipelinePhase.INGESTION,
        "X_context_construction": PipelinePhase.CONTEXT,
        "K_knowledge_extraction": PipelinePhase.KNOWLEDGE,
        "A_analysis_nlp": PipelinePhase.ANALYSIS,
        "L_classification_evaluation": PipelinePhase.CLASSIFICATION,
        "R_search_retrieval": PipelinePhase.RETRIEVAL,
        "O_orchestration_control": PipelinePhase.ORCHESTRATION,
        "G_aggregation_reporting": PipelinePhase.AGGREGATION,
        "T_integration_storage": PipelinePhase.INTEGRATION,
        "S_synthesis_output": PipelinePhase.SYNTHESIS,
    }
    
    # Stage ordering within the I→X→K→A→L→R→O→G→T→S sequence
    STAGE_ORDER = {
        PipelinePhase.INGESTION: 1,
        PipelinePhase.CONTEXT: 2,
        PipelinePhase.KNOWLEDGE: 3,
        PipelinePhase.ANALYSIS: 4,
        PipelinePhase.CLASSIFICATION: 5,
        PipelinePhase.RETRIEVAL: 6,
        PipelinePhase.ORCHESTRATION: 7,
        PipelinePhase.AGGREGATION: 8,
        PipelinePhase.INTEGRATION: 9,
        PipelinePhase.SYNTHESIS: 10,
    }
    
    @staticmethod
    def extract_phase_from_path(file_path: str) -> PipelinePhase:
        """Extract phase from canonical flow directory structure"""
        path_parts = Path(file_path).parts
        
        for part in path_parts:
            if part.startswith(("I_", "X_", "K_", "A_", "L_", "R_", "O_", "G_", "T_", "S_")):
                return ComponentAnnotations.PHASE_MAPPING.get(part, None)
        
        # Fallback mapping for components outside canonical structure
        if "ingestion" in file_path.lower():
            return PipelinePhase.INGESTION
        elif "context" in file_path.lower():
            return PipelinePhase.CONTEXT
        elif "knowledge" in file_path.lower():
            return PipelinePhase.KNOWLEDGE
        elif "analysis" in file_path.lower() or "nlp" in file_path.lower():
            return PipelinePhase.ANALYSIS
        elif "classification" in file_path.lower() or "evaluation" in file_path.lower():
            return PipelinePhase.CLASSIFICATION
        elif "retrieval" in file_path.lower() or "search" in file_path.lower():
            return PipelinePhase.RETRIEVAL
        elif "orchestration" in file_path.lower() or "orchestrator" in file_path.lower():
            return PipelinePhase.ORCHESTRATION
        elif "aggregation" in file_path.lower() or "reporting" in file_path.lower():
            return PipelinePhase.AGGREGATION
        elif "integration" in file_path.lower() or "storage" in file_path.lower():
            return PipelinePhase.INTEGRATION
        elif "synthesis" in file_path.lower() or "output" in file_path.lower():
            return PipelinePhase.SYNTHESIS
        else:
            # Default unknown components to ORCHESTRATION as they likely coordinate between phases
            return PipelinePhase.ORCHESTRATION
    
    @staticmethod
    def generate_component_code(phase: PipelinePhase, sequence_number: int) -> str:
        """Generate component code in format: {NN}{PHASE}"""
        return f"{sequence_number:02d}{phase.value}"
    
    @staticmethod
    def validate_annotations(module) -> Dict[str, Any]:
        """Validate that a module has required annotations"""
        missing = []
        invalid = []
        
        if not hasattr(module, '__phase__'):
            missing.append('__phase__')
        elif not isinstance(module.__phase__, (str, PipelinePhase)):
            invalid.append('__phase__ must be string or PipelinePhase enum')
            
        if not hasattr(module, '__code__'):
            missing.append('__code__')
        elif not isinstance(module.__code__, str):
            invalid.append('__code__ must be string')
            
        if not hasattr(module, '__stage_order__'):
            missing.append('__stage_order__')
        elif not isinstance(module.__stage_order__, int):
            invalid.append('__stage_order__ must be integer')
            
        return {
            'valid': len(missing) == 0 and len(invalid) == 0,
            'missing': missing,
            'invalid': invalid
        }


def add_contract_annotations(file_path: str, phase: PipelinePhase = None, component_code: str = None):
    """Add mandatory contract annotations to a Python file"""
    
    if phase is None:
        phase = ComponentAnnotations.extract_phase_from_path(file_path)
    
    if component_code is None:
        # Generate next available code for this phase
        component_code = f"XX{phase.value}"  # Placeholder, should be assigned proper sequence
    
    stage_order = ComponentAnnotations.STAGE_ORDER[phase]
    
    annotations = f'''
# Mandatory Pipeline Contract Annotations
__phase__ = "{phase.value}"
__code__ = "{component_code}"
__stage_order__ = {stage_order}
'''
    
    # Read existing file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    
    # Check if annotations already exist
    if '__phase__' in content and '__code__' in content and '__stage_order__' in content:
        print(f"Annotations already exist in {file_path}")
        return False
    
    # Insert annotations after docstring/imports but before main code
    lines = content.split('\n')
    insert_index = 0
    
    # Skip docstring
    if content.strip().startswith('"""') or content.strip().startswith("'''"):
        in_docstring = True
        quote_type = '"""' if '"""' in content else "'''"
        quote_count = 1
        
        for i, line in enumerate(lines[1:], 1):
            if quote_type in line:
                quote_count += line.count(quote_type)
                if quote_count >= 2:
                    insert_index = i + 1
                    break
    
    # Skip imports
    for i, line in enumerate(lines[insert_index:], insert_index):
        stripped = line.strip()
        if not (stripped.startswith('import ') or stripped.startswith('from ') or 
                stripped.startswith('#') or stripped == ''):
            insert_index = i
            break
    
    # Insert annotations
    lines.insert(insert_index, annotations)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return True


class ComponentScanner:
    """Scans and validates pipeline components for required annotations"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.components_found = []
        self.components_missing_annotations = []
        self.components_invalid_annotations = []
    
    def scan_components(self) -> Dict[str, Any]:
        """Scan all Python files for pipeline components"""
        
        for py_file in self.root_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            # Skip test files for now
            if "test_" in py_file.name or py_file.name.startswith("test_"):
                continue
            
            try:
                # Read file content to check for process() method or component patterns
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if this looks like a pipeline component
                if self._is_pipeline_component(content):
                    self.components_found.append(str(py_file))
                    
                    # Try to import and validate annotations
                    validation = self._validate_file_annotations(py_file)
                    
                    if not validation['valid']:
                        if validation['missing']:
                            self.components_missing_annotations.append({
                                'file': str(py_file),
                                'missing': validation['missing'],
                                'invalid': validation['invalid']
                            })
                        
            except Exception as e:
                print(f"Error scanning {py_file}: {e}")
        
        return {
            'total_components': len(self.components_found),
            'missing_annotations': len(self.components_missing_annotations),
            'components_found': self.components_found,
            'components_missing': self.components_missing_annotations
        }
    
    def _is_pipeline_component(self, content: str) -> bool:
        """Check if file contains pipeline component patterns"""
        patterns = [
            'def process(',
            'class.*Processor',
            'class.*Engine',
            'class.*Analyzer',
            'class.*Router',
            'class.*Orchestrator',
            'class.*Generator',
            'class.*Extractor',
            'class.*Validator',
            'class.*Builder',
            'class.*Manager'
        ]
        
        import re
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _validate_file_annotations(self, file_path: Path) -> Dict[str, Any]:
        """Validate annotations in a specific file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find module-level constants
            tree = ast.parse(content)
            
            found_annotations = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id in ['__phase__', '__code__', '__stage_order__']:
                                if isinstance(node.value, ast.Constant):
                                    found_annotations[target.id] = node.value.value
                                elif isinstance(node.value, ast.Str):  # Python < 3.8
                                    found_annotations[target.id] = node.value.s
            
            missing = []
            invalid = []
            
            required = ['__phase__', '__code__', '__stage_order__']
            for req in required:
                if req not in found_annotations:
                    missing.append(req)
                else:
                    # Validate types
                    if req == '__phase__' and not isinstance(found_annotations[req], str):
                        invalid.append(f'{req} must be string')
                    elif req == '__code__' and not isinstance(found_annotations[req], str):
                        invalid.append(f'{req} must be string')
                    elif req == '__stage_order__' and not isinstance(found_annotations[req], int):
                        invalid.append(f'{req} must be integer')
            
            return {
                'valid': len(missing) == 0 and len(invalid) == 0,
                'missing': missing,
                'invalid': invalid,
                'found': found_annotations
            }
            
        except Exception as e:
            return {
                'valid': False,
                'missing': ['__phase__', '__code__', '__stage_order__'],
                'invalid': [],
                'error': str(e)
            }