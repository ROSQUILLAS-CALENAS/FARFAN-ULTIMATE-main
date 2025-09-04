"""
Updated Component Scanning System
Enhanced to validate mandatory pipeline annotations and ensure conformance
"""

import ast
import re
import os
import json
import subprocess
from typing import Dict, List, Any, Set, Optional
from pathlib import Path
from enum import Enum
from datetime import datetime
from integration_layer import IntegrationLayer, ComponentMetadata, LifecycleState

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
        
        # Initialize integration layer for SQL registry
        self.integration_layer = IntegrationLayer()
        
        # Git analysis for owner information
        self.git_available = self._check_git_availability()
    
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
    
    def _check_git_availability(self) -> bool:
        """Check if git is available for blame analysis"""
        try:
            result = subprocess.run(['git', 'status'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5,
                                  cwd=self.root_path)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_file_owner_from_git(self, file_path: Path) -> str:
        """Get file owner using git blame analysis"""
        if not self.git_available:
            return "unknown"
        
        try:
            # Get relative path from repo root
            rel_path = file_path.relative_to(self.root_path)
            
            # Use git blame to find most frequent committer
            result = subprocess.run([
                'git', 'blame', '--line-porcelain', str(rel_path)
            ], capture_output=True, text=True, timeout=10, cwd=self.root_path)
            
            if result.returncode == 0:
                authors = []
                for line in result.stdout.split('\n'):
                    if line.startswith('author '):
                        authors.append(line[7:])  # Remove 'author ' prefix
                
                if authors:
                    # Return most frequent author
                    from collections import Counter
                    return Counter(authors).most_common(1)[0][0]
            
            return "unknown"
            
        except Exception as e:
            print(f"Warning: Could not get git blame for {file_path}: {e}")
            return "unknown"
    
    def _calculate_evidence_score(self, file_path: Path, annotations: Dict[str, Any]) -> float:
        """Calculate canonical evidence score based on various factors"""
        score = 0.0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Base score for having proper annotations
            score += 30.0
            
            # Documentation score
            if '"""' in content or "'''" in content:
                score += 20.0
            
            # Type hints score  
            if 'typing' in content or '->' in content or ': str' in content:
                score += 15.0
            
            # Error handling score
            if 'try:' in content and 'except' in content:
                score += 10.0
            
            # Logging score
            if 'logger' in content or 'logging' in content:
                score += 10.0
            
            # Test coverage bonus (if corresponding test file exists)
            test_file = self.root_path / f"tests/test_{file_path.stem}.py"
            if test_file.exists():
                score += 15.0
            
            # Ensure score doesn't exceed 100
            return min(score, 100.0)
            
        except Exception:
            return 0.0
    
    def _determine_lifecycle_state(self, evidence_score: float, file_path: Path) -> LifecycleState:
        """Determine initial lifecycle state based on evidence score and other factors"""
        
        # Check for experimental markers
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            if any(marker in content for marker in ['experimental', 'prototype', 'draft']):
                return LifecycleState.EXPERIMENTAL
            
            if any(marker in content for marker in ['deprecated', 'obsolete']):
                return LifecycleState.DEPRECATED
                
        except Exception:
            pass
        
        # Use evidence score to determine state
        if evidence_score >= 80:
            return LifecycleState.ACTIVE
        elif evidence_score >= 60:
            return LifecycleState.MAINTENANCE
        else:
            return LifecycleState.EXPERIMENTAL
    
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
                # Get owner information from git blame
                owner = self._get_file_owner_from_git(file_path)
                
                # Calculate evidence score
                evidence_score = self._calculate_evidence_score(file_path, annotations)
                
                # Determine lifecycle state
                lifecycle_state = self._determine_lifecycle_state(evidence_score, file_path)
                
                # Register component in memory
                component_info = {
                    'file_path': str(file_path),
                    'phase': annotations['__phase__'],
                    'code': annotations['__code__'],
                    'stage_order': annotations['__stage_order__'],
                    'phase_name': self.PHASE_NAMES[annotations['__phase__']],
                    'owner': owner,
                    'evidence_score': evidence_score,
                    'lifecycle_state': lifecycle_state.value
                }
                
                self.registered_components[annotations['__code__']] = component_info
                self.registered_codes.add(annotations['__code__'])
                
                # Register in SQL registry
                try:
                    # Determine alias path based on canonical structure
                    stage_name = self.PHASE_NAMES[annotations['__phase__']]
                    alias_path = f"canonical_flow/{annotations['__phase__']}_{stage_name}/{annotations['__code__']}_{file_path.stem}.py"
                    
                    metadata = ComponentMetadata(
                        code=annotations['__code__'],
                        stage=stage_name,
                        alias_path=alias_path,
                        original_path=str(file_path),
                        owner=owner,
                        lifecycle_state=lifecycle_state,
                        evidence_score=evidence_score
                    )
                    
                    success = self.integration_layer.registry.register_component(metadata)
                    if success:
                        print(f"✓ Registered {annotations['__code__']} in SQL registry (score: {evidence_score:.1f})")
                    else:
                        print(f"⚠ Failed to register {annotations['__code__']} in SQL registry")
                        
                except Exception as e:
                    print(f"⚠ SQL registry error for {annotations['__code__']}: {e}")
                
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
    
    print("🚀 Pipeline Component Scanner v3.0")
    print("   Enhanced with SQL registry integration and git blame analysis")
    
    registry = ComponentRegistry(".")
    scan_result = registry.scan_repository()
    
    # Perform bidirectional sync with canonical_flow index.json
    print("\n🔄 Performing registry synchronization...")
    sync_result = registry.integration_layer.bidirectional_sync()
    
    if sync_result["errors"]:
        print("⚠️ Synchronization errors:")
        for error in sync_result["errors"]:
            print(f"   - {error}")
    else:
        print("✅ Registry synchronization completed successfully")
    
    # Check for inconsistencies
    print("\n🔍 Checking for inconsistencies...")
    inconsistencies = registry.integration_layer.get_inconsistencies()
    
    total_issues = (len(inconsistencies.get("registry_only", [])) + 
                   len(inconsistencies.get("index_only", [])) + 
                   len(inconsistencies.get("metadata_mismatches", [])))
    
    if total_issues > 0:
        print(f"⚠️ Found {total_issues} inconsistencies")
        if inconsistencies.get("registry_only"):
            print(f"   Registry-only components: {len(inconsistencies['registry_only'])}")
        if inconsistencies.get("index_only"):
            print(f"   Index-only components: {len(inconsistencies['index_only'])}")
        if inconsistencies.get("metadata_mismatches"):
            print(f"   Metadata mismatches: {len(inconsistencies['metadata_mismatches'])}")
    else:
        print("✅ No inconsistencies found")
    
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
    
    # SQL Registry Statistics
    sql_components = registry.integration_layer.registry.list_components()
    print(f"\n📊 SQL Registry Statistics:")
    print(f"   Total registered components: {len(sql_components)}")
    
    # Lifecycle state summary
    state_counts = {}
    for comp in sql_components:
        state = comp.lifecycle_state.value
        state_counts[state] = state_counts.get(state, 0) + 1
    
    for state, count in state_counts.items():
        print(f"   {state.capitalize()}: {count}")
    
    # Average evidence score
    if sql_components:
        avg_score = sum(comp.evidence_score for comp in sql_components) / len(sql_components)
        print(f"   Average evidence score: {avg_score:.1f}")
    
    print(f"\n✅ Component scanning and registry integration complete")
    
    # Clean up
    registry.integration_layer.close()
    
    return index


if __name__ == "__main__":
    main()