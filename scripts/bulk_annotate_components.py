#!/usr/bin/env python3
"""
Bulk Annotation Script for Pipeline Components
Adds mandatory static contract annotations to components in batches.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Phase and stage mappings
PHASE_MAPPING = {
    'I': 1, 'X': 2, 'K': 3, 'A': 4, 'L': 5,
    'R': 6, 'O': 7, 'G': 8, 'T': 9, 'S': 10
}

def extract_phase_from_path(file_path: str) -> str:
    """Extract phase from file path and name patterns"""
    
    # Canonical flow directory structure
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
    
    # Pattern-based classification
    filename = Path(file_path).stem.lower()
    
    if any(term in filename for term in ['pdf_reader', 'loader', 'feature_extractor', 'normative', 'ingestion', 'raw_data']):
        return 'I'
    elif any(term in filename for term in ['context', 'lineage', 'immutable']):
        return 'X'
    elif any(term in filename for term in ['knowledge', 'graph', 'embedding', 'causal', 'dnp', 'entity', 'chunking']):
        return 'K'
    elif any(term in filename for term in ['analyzer', 'analysis', 'question', 'evidence', 'mapeo', 'nlp']):
        return 'A'
    elif any(term in filename for term in ['scoring', 'score', 'classification', 'evaluation', 'conformal']):
        return 'L'
    elif any(term in filename for term in ['retrieval', 'search', 'index', 'lexical', 'vector', 'hybrid', 'reranker']):
        return 'R'
    elif any(term in filename for term in ['synthesis', 'answer', 'formatter', 'output']):
        return 'S'
    elif any(term in filename for term in ['aggregat', 'report', 'compiler', 'meso', 'audit_logger']):
        return 'G'
    elif any(term in filename for term in ['metrics', 'analytics', 'feedback', 'compensation', 'optimization']):
        return 'T'
    else:
        return 'O'  # Default to orchestration

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
        r'class.*Manager',
        r'class.*Adapter',
        r'class.*Controller'
    ]
    
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)

def has_annotations(content: str) -> bool:
    """Check if file has required annotations"""
    return '__phase__' in content and '__code__' in content and '__stage_order__' in content

def add_annotations_to_file(file_path: str, phase: str, code: str) -> bool:
    """Add annotations to a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if has_annotations(content):
            return True
        
        # Find insertion point
        lines = content.split('\n')
        insert_index = 0
        
        # Skip docstring
        if lines and (lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''")):
            quote_type = '"""' if '"""' in lines[0] else "'''"
            if lines[0].count(quote_type) < 2:  # Multi-line docstring
                for i, line in enumerate(lines[1:], 1):
                    if quote_type in line:
                        insert_index = i + 1
                        break
            else:
                insert_index = 1
        
        # Skip imports
        for i, line in enumerate(lines[insert_index:], insert_index):
            stripped = line.strip()
            if not (stripped.startswith('import ') or stripped.startswith('from ') or 
                   stripped.startswith('#') or stripped == ''):
                insert_index = i
                break
        
        # Add annotations
        stage_order = PHASE_MAPPING[phase]
        annotation_block = f'''
# Mandatory Pipeline Contract Annotations
__phase__ = "{phase}"
__code__ = "{code}"
__stage_order__ = {stage_order}
'''
        
        lines.insert(insert_index, annotation_block)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Annotated {file_path} with {code}:{phase}")
        return True
        
    except Exception as e:
        print(f"❌ Error annotating {file_path}: {e}")
        return False

def process_in_batches(batch_size: int = 50):
    """Process components in batches to add annotations"""
    
    # Find all components needing annotations
    components = []
    for root, dirs, files in os.walk('.'):
        if any(skip in root for skip in ['.git', '__pycache__', '.venv', 'venv']):
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
                        
                except Exception:
                    continue
    
    print(f"Found {len(components)} components needing annotations")
    
    # Load existing codes to avoid conflicts
    existing_codes: Set[str] = set()
    sequence_counters = {phase: 1 for phase in 'IXKALROGTS'}
    
    try:
        with open('canonical_flow/index.json', 'r') as f:
            index_data = json.load(f)
            for item in index_data:
                code = item.get('code', '')
                if code and len(code) >= 3:
                    existing_codes.add(code)
                    try:
                        seq_num = int(code[:2])
                        phase_char = code[2:]
                        if phase_char in sequence_counters:
                            sequence_counters[phase_char] = max(sequence_counters[phase_char], seq_num + 1)
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    
    # Process in batches
    annotated_count = 0
    total_batches = (len(components) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(components))
        batch = components[start_idx:end_idx]
        
        print(f"\n📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch)} components)")
        
        for file_path, phase in batch:
            # Generate unique code
            while True:
                code = f"{sequence_counters[phase]:02d}{phase}"
                if code not in existing_codes:
                    existing_codes.add(code)
                    sequence_counters[phase] += 1
                    break
                sequence_counters[phase] += 1
            
            if add_annotations_to_file(file_path, phase, code):
                annotated_count += 1
    
    print(f"\n✅ Successfully annotated {annotated_count}/{len(components)} components")
    return annotated_count

if __name__ == "__main__":
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"🚀 Starting bulk annotation with batch size {batch_size}")
    annotated = process_in_batches(batch_size)
    print(f"📊 Total components annotated: {annotated}")