"""
Updated Component Scanning System
Enhanced to validate mandatory pipeline annotations and ensure conformance
"""

import ast
import re
import os
import json
from typing import Dict, List, Any, Set, Optional
from pathlib import Path
from enum import Enum

class ValidationResult(Enum):
    VALID = "valid"
    MISSING_ANNOTATIONS = "missing_annotations"
    INVALID_ANNOTATIONS = "invalid_annotations"
    NOT_PIPELINE_COMPONENT = "not_pipeline_component"

class ComponentRegistry:
    """Updated component registry with annotation validation"""
    
    VALID_PHASES = {'I', 'X', 'K', 'A', 'L', 'R', 'O', 'G', 'T', 'S'}
    PHASE_NAMES = {
        'I': 'ingestion_preparation',
        'X': 'context_construction', 
        'K': 'knowledge_extraction',
        'A': 'analysis_nlp',
        'L': 'classification_evaluation',
        'R': 'search_retrieval',
        'O': 'orchestration_control',
        'G': 'aggregation_reporting',
        'T': 'integration_storage',
        'S': 'synthesis_output'
    }
    STAGE_ORDER = {
        'I': 1, 'X': 2, 'K': 3, 'A': 4, 'L': 5,
        'R': 6, 'O': 7, 'G': 8, 'T': 9, 'S': 10
    }
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.registered_components: Dict[str, Dict[str, Any]] = {}
        self.registered_codes: Set[str] = set()
        self.validation_errors: List[Dict[str, Any]] = []
    
    def is_pipeline_component(self, content: str) -> bool:
        """Check if file contains pipeline component patterns"""
        patterns = [
            r'def process\(',
            r'class.*Processor',
            r'class.*Engine', 
            r'class.*Analyzer',
            r'class.*Router',
            r'class.*Orchestrator',
            r'class.*Generator',
            r'class.*Extractor',
            r'class.*Validator',
            r'class.*Builder',
            r'class.*Manager',
            r'class.*Adapter',
            r'class.*Controller'
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)
    
    def extract_annotations(self, file_path: Path) -> Dict[str, Any]:
        """Extract module-level annotations from Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find module-level constants
            tree = ast.parse(content)
            annotations = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id in ['__phase__', '__code__', '__stage_order__']:
                                if isinstance(node.value, ast.Constant):
                                    annotations[target.id] = node.value.value
                                elif isinstance(node.value, ast.Str):  # Python < 3.8
                                    annotations[target.id] = node.value.s
                                elif isinstance(node.value, ast.Num):  # Python < 3.8
                                    annotations[target.id] = node.value.n
            
            return annotations
            
        except Exception as e:
            return {'error': str(e)}
    
    def validate_component_annotations(self, file_path: Path, annotations: Dict[str, Any]) -> ValidationResult:
        """Validate component annotations against canonical requirements"""
        
        # Check for required annotations
        required = ['__phase__', '__code__', '__stage_order__']
        missing = [req for req in required if req not in annotations]
        
        if missing:
            self.validation_errors.append({
                'file': str(file_path),
                'error': 'missing_annotations',
                'missing': missing
            })
            return ValidationResult.MISSING_ANNOTATIONS
        
        # Validate phase
        phase = annotations.get('__phase__')
        if not isinstance(phase, str) or phase not in self.VALID_PHASES:
            self.validation_errors.append({
                'file': str(file_path),
                'error': 'invalid_phase',
                'value': phase,
                'expected': list(self.VALID_PHASES)
            })
            return ValidationResult.INVALID_ANNOTATIONS
        
        # Validate code format
        code = annotations.get('__code__')
        if not isinstance(code, str) or not re.match(r'^\d{2}[IXKALROGTS]$', code):
            self.validation_errors.append({
                'file': str(file_path),
                'error': 'invalid_code_format', 
                'value': code,
                'expected': 'Format: NNX (e.g., 01I, 25A)'
            })
            return ValidationResult.INVALID_ANNOTATIONS
        
        # Check for duplicate codes
        if code in self.registered_codes:
            self.validation_errors.append({
                'file': str(file_path),
                'error': 'duplicate_code',
                'code': code
            })
            return ValidationResult.INVALID_ANNOTATIONS
        
        # Validate stage order
        stage_order = annotations.get('__stage_order__')
        expected_order = self.STAGE_ORDER.get(phase)
        if stage_order != expected_order:
            self.validation_errors.append({
                'file': str(file_path),
                'error': 'invalid_stage_order',
                'value': stage_order,
                'expected': expected_order
            })
            return ValidationResult.INVALID_ANNOTATIONS
        
        return ValidationResult.VALID
    
    def scan_and_register_component(self, file_path: Path) -> bool:
        """Scan single component and register if valid"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it's a pipeline component
            if not self.is_pipeline_component(content):
                return False
            
            # Extract annotations
            annotations = self.extract_annotations(file_path)
            if 'error' in annotations:
                self.validation_errors.append({
                    'file': str(file_path),
                    'error': 'parse_error',
                    'details': annotations['error']
                })
                return False
            
            # Validate annotations
            validation_result = self.validate_component_annotations(file_path, annotations)
            
            if validation_result == ValidationResult.VALID:
                # Register component
                component_info = {
                    'file_path': str(file_path),
                    'phase': annotations['__phase__'],
                    'code': annotations['__code__'],
                    'stage_order': annotations['__stage_order__'],
                    'phase_name': self.PHASE_NAMES[annotations['__phase__']]
                }
                
                self.registered_components[annotations['__code__']] = component_info
                self.registered_codes.add(annotations['__code__'])
                return True
            
            return False
            
        except Exception as e:
            self.validation_errors.append({
                'file': str(file_path),
                'error': 'scan_error',
                'details': str(e)
            })
            return False
    
    def scan_repository(self) -> Dict[str, Any]:
        """Scan entire repository for pipeline components"""
        
        print("🔍 Scanning repository for pipeline components...")
        
        total_files = 0
        components_found = 0
        components_registered = 0
        
        for root, dirs, files in os.walk(self.root_path):
            # Skip certain directories
            if any(skip in root for skip in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']):
                continue
            
            for file in files:
                if file.endswith('.py') and file != '__init__.py' and not file.startswith('test_'):
                    file_path = Path(root) / file
                    total_files += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if self.is_pipeline_component(content):
                            components_found += 1
                            if self.scan_and_register_component(file_path):
                                components_registered += 1
                                
                    except Exception:
                        continue
        
        result = {
            'total_files_scanned': total_files,
            'pipeline_components_found': components_found,
            'components_registered': components_registered,
            'validation_errors': len(self.validation_errors),
            'registered_components': dict(sorted(self.registered_components.items())),
            'errors': self.validation_errors
        }
        
        print(f"📊 Scan completed:")
        print(f"   Files scanned: {total_files}")
        print(f"   Pipeline components found: {components_found}")
        print(f"   Components registered: {components_registered}")
        print(f"   Validation errors: {len(self.validation_errors)}")
        
        return result
    
    def validate_canonical_sequence(self) -> Dict[str, Any]:
        """Validate that components follow canonical I→X→K→A→L→R→O→G→T→S sequence"""
        
        sequence_validation = {
            'valid': True,
            'issues': [],
            'phase_coverage': {}
        }
        
        # Group components by phase
        by_phase = {}
        for code, component in self.registered_components.items():
            phase = component['phase']
            if phase not in by_phase:
                by_phase[phase] = []
            by_phase[phase].append(component)
        
        # Check phase coverage
        for phase_char, phase_name in self.PHASE_NAMES.items():
            count = len(by_phase.get(phase_char, []))
            sequence_validation['phase_coverage'][phase_name] = {
                'phase': phase_char,
                'component_count': count,
                'components': [c['code'] for c in by_phase.get(phase_char, [])]
            }
            
            # Flag phases with no components (may need attention)
            if count == 0:
                sequence_validation['issues'].append({
                    'type': 'empty_phase',
                    'phase': phase_char,
                    'phase_name': phase_name
                })
        
        # Validate code sequences within each phase
        for phase, components in by_phase.items():
            codes = sorted([int(c['code'][:2]) for c in components])
            for i, code_num in enumerate(codes):
                if i > 0 and code_num != codes[i-1] + 1:
                    sequence_validation['valid'] = False
                    sequence_validation['issues'].append({
                        'type': 'sequence_gap',
                        'phase': phase,
                        'gap_after': f"{codes[i-1]:02d}{phase}",
                        'gap_before': f"{code_num:02d}{phase}"
                    })
        
        return sequence_validation
    
    def get_registration_index(self) -> Dict[str, Any]:
        """Generate component registration index"""
        return {
            'metadata': {
                'total_components': len(self.registered_components),
                'phases_covered': len(set(c['phase'] for c in self.registered_components.values())),
                'validation_status': 'VALID' if not self.validation_errors else 'ERRORS'
            },
            'components': self.registered_components,
            'sequence_validation': self.validate_canonical_sequence()
        }


def main():
    """Main entry point for component scanning"""
    
    print("🚀 Pipeline Component Scanner v2.0")
    print("   Enhanced with annotation validation")
    
    registry = ComponentRegistry(".")
    scan_result = registry.scan_repository()
    
    # Generate registration index
    index = registry.get_registration_index()
    
    # Save results
    with open('pipeline_component_registry.json', 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\n💾 Saved component registry to: pipeline_component_registry.json")
    
    # Report validation issues
    if registry.validation_errors:
        print(f"\n⚠️  Validation Issues Found:")
        for error in registry.validation_errors[:10]:  # Show first 10
            print(f"   - {error['file']}: {error['error']}")
            if 'missing' in error:
                print(f"     Missing: {', '.join(error['missing'])}")
        
        if len(registry.validation_errors) > 10:
            print(f"   ... and {len(registry.validation_errors) - 10} more errors")
    
    # Summary
    sequence_validation = index['sequence_validation']
    print(f"\n📋 Phase Coverage Summary:")
    for phase_name, info in sequence_validation['phase_coverage'].items():
        print(f"   {info['phase']}: {info['component_count']} components")
    
    if sequence_validation['issues']:
        print(f"\n🔍 Sequence Issues: {len(sequence_validation['issues'])}")
    
    print(f"\n✅ Component scanning complete")
    return index


if __name__ == "__main__":
    main()