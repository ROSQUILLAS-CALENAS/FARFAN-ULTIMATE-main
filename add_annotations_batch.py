#!/usr/bin/env python3
"""
Batch add pipeline contract annotations to all components
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import json

# Phase mapping based on file location and function
PHASE_MAPPING = {
    'I': 'I',  # Ingestion
    'X': 'X',  # Context
    'K': 'K',  # Knowledge
    'A': 'A',  # Analysis
    'L': 'L',  # Classification
    'R': 'R',  # Retrieval
    'O': 'O',  # Orchestration
    'G': 'G',  # Aggregation
    'T': 'T',  # Integration
    'S': 'S'   # Synthesis
}

STAGE_ORDER = {
    'I': 1, 'X': 2, 'K': 3, 'A': 4, 'L': 5,
    'R': 6, 'O': 7, 'G': 8, 'T': 9, 'S': 10
}

def extract_phase_from_path(file_path: str) -> str:
    """Extract phase from file path and content patterns"""
    
    # Check canonical_flow directory structure first
    if 'canonical_flow/' in file_path:
        if '/I_ingestion_preparation/' in file_path:
            return 'I'
        elif '/X_context_construction/' in file_path:
            return 'X'
        elif '/K_knowledge_extraction/' in file_path:
            return 'K'
        elif '/A_analysis_nlp/' in file_path:
            return 'A'
        elif '/L_classification_evaluation/' in file_path:
            return 'L'
        elif '/R_search_retrieval/' in file_path:
            return 'R'
        elif '/O_orchestration_control/' in file_path:
            return 'O'
        elif '/G_aggregation_reporting/' in file_path:
            return 'G'
        elif '/T_integration_storage/' in file_path:
            return 'T'
        elif '/S_synthesis_output/' in file_path:
            return 'S'
    
    # Fallback to filename/content analysis
    filename = Path(file_path).stem.lower()
    
    # Ingestion patterns
    if any(term in filename for term in ['pdf_reader', 'loader', 'feature_extractor', 'normative', 'ingestion', 'raw_data']):
        return 'I'
    
    # Context patterns  
    if any(term in filename for term in ['context', 'lineage', 'immutable']):
        return 'X'
    
    # Knowledge patterns
    if any(term in filename for term in ['knowledge', 'graph', 'embedding', 'causal', 'dnp', 'entity', 'chunking']):
        return 'K'
    
    # Analysis patterns
    if any(term in filename for term in ['analyzer', 'analysis', 'question', 'evidence', 'mapeo', 'nlp']):
        return 'A'
    
    # Classification/Evaluation patterns
    if any(term in filename for term in ['scoring', 'score', 'classification', 'evaluation', 'conformal', 'adaptive_scoring']):
        return 'L'
    
    # Retrieval patterns
    if any(term in filename for term in ['retrieval', 'search', 'index', 'lexical', 'vector', 'hybrid', 'reranker', 'recommendation']):
        return 'R'
    
    # Orchestration patterns (default for many control/management files)
    if any(term in filename for term in ['orchestrator', 'router', 'engine', 'controller', 'manager', 'validator', 'monitor', 'telemetry', 'circuit', 'alert']):
        return 'O'
    
    # Aggregation/Reporting patterns  
    if any(term in filename for term in ['aggregat', 'report', 'compiler', 'meso', 'audit_logger']):
        return 'G'
    
    # Integration/Storage patterns
    if any(term in filename for term in ['metrics', 'analytics', 'feedback', 'compensation', 'optimization', 'integration', 'storage']):
        return 'T'
    
    # Synthesis/Output patterns
    if any(term in filename for term in ['synthesis', 'answer', 'formatter', 'output']):
        return 'S'
    
    # Default to Orchestration for unknown components
    return 'O'

def is_pipeline_component(content: str) -> bool:
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
        r'class.*Manager'
    ]
    
    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False

def has_annotations(content: str) -> bool:
    """Check if file already has required annotations"""
    return '__phase__' in content and '__code__' in content and '__stage_order__' in content

def find_insertion_point(lines: List[str]) -> int:
    """Find the best place to insert annotations"""
    insert_index = 0
    
    # Skip docstring
    if len(lines) > 0:
        first_line = lines[0].strip()
        if first_line.startswith('"""') or first_line.startswith("'''"):
            quote_type = '"""' if '"""' in first_line else "'''"
            
            # Single-line docstring
            if first_line.count(quote_type) >= 2:
                insert_index = 1
            else:
                # Multi-line docstring
                for i, line in enumerate(lines[1:], 1):
                    if quote_type in line:
                        insert_index = i + 1
                        break
    
    # Skip imports and comments
    for i, line in enumerate(lines[insert_index:], insert_index):
        stripped = line.strip()
        if stripped == '' or stripped.startswith('#'):
            continue
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue
        insert_index = i
        break
    
    return insert_index

def add_annotations_to_file(file_path: str, phase: str, component_code: str) -> bool:
    """Add contract annotations to a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if has_annotations(content):
            print(f"✓ {file_path} already has annotations")
            return True
        
        lines = content.split('\n')
        insert_index = find_insertion_point(lines)
        
        stage_order = STAGE_ORDER[phase]
        annotations = f'''
# Mandatory Pipeline Contract Annotations
__phase__ = "{phase}"
__code__ = "{component_code}"
__stage_order__ = {stage_order}
'''
        
        lines.insert(insert_index, annotations)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Added annotations to {file_path} [{component_code}:{phase}]")
        return True
        
    except Exception as e:
        print(f"❌ Error adding annotations to {file_path}: {e}")
        return False

def load_existing_codes() -> Dict[str, int]:
    """Load existing component codes to avoid conflicts"""
    existing_codes = set()
    sequence_counters = {phase: 1 for phase in PHASE_MAPPING.keys()}
    
    # Load from index.json if available
    index_file = Path("canonical_flow/index.json")
    if index_file.exists():
        try:
            with open(index_file, 'r') as f:
                components = json.load(f)
                
            for comp in components:
                code = comp.get('code', '')
                if code and len(code) >= 3:
                    existing_codes.add(code)
                    try:
                        seq_num = int(code[:2])
                        phase_char = code[2:]
                        if phase_char in sequence_counters:
                            sequence_counters[phase_char] = max(sequence_counters[phase_char], seq_num + 1)
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Warning: Could not load existing codes: {e}")
    
    return sequence_counters

def generate_component_code(phase: str, sequence_counters: Dict[str, int], existing_codes: set) -> str:
    """Generate next available component code for phase"""
    while True:
        code = f"{sequence_counters[phase]:02d}{phase}"
        if code not in existing_codes:
            existing_codes.add(code)
            sequence_counters[phase] += 1
            return code
        sequence_counters[phase] += 1

def main():
    """Main function to add annotations to all components"""
    print("🔍 Finding pipeline components...")
    
    components = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']):
            continue
            
        for file in files:
            if file.endswith('.py') and file != '__init__.py' and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if is_pipeline_component(content) and not has_annotations(content):
                        phase = extract_phase_from_path(file_path)
                        components.append((file_path, phase))
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    print(f"Found {len(components)} components needing annotations")
    
    # Load existing codes and initialize counters
    sequence_counters = load_existing_codes()
    existing_codes = set()
    
    # Sort components by phase to maintain order
    components.sort(key=lambda x: STAGE_ORDER[x[1]])
    
    success_count = 0
    for file_path, phase in components:
        component_code = generate_component_code(phase, sequence_counters, existing_codes)
        
        if add_annotations_to_file(file_path, phase, component_code):
            success_count += 1
    
    print(f"\n✅ Successfully added annotations to {success_count}/{len(components)} components")

if __name__ == "__main__":
    main()