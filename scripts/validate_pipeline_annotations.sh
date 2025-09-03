#!/bin/bash
# CI Pipeline Component Annotation Validator
# Automatically rejects commits with missing pipeline component annotations

set -e

echo "🔍 Validating pipeline component annotations..."

# Check for Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not available for validation"
    exit 1
fi

# Run annotation validation
python3 -c "
import sys
import os
import re
from pathlib import Path

def is_pipeline_component(content):
    '''Check if file contains pipeline component patterns'''
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
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)

def has_annotations(content):
    '''Check if file has required annotations'''
    return '__phase__' in content and '__code__' in content and '__stage_order__' in content

# Find components missing annotations
components_missing = []
total_components = 0

for root, dirs, files in os.walk('.'):
    if any(skip in root for skip in ['.git', '__pycache__', '.venv', 'venv']):
        continue
        
    for file in files:
        if file.endswith('.py') and file != '__init__.py' and not file.startswith('test_'):
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if is_pipeline_component(content):
                    total_components += 1
                    if not has_annotations(content):
                        components_missing.append(file_path)
                        
            except Exception:
                continue

if components_missing:
    print(f'❌ Found {len(components_missing)} components missing required annotations')
    print('Components missing annotations:')
    for comp in components_missing[:20]:  # Show first 20
        print(f'  - {comp}')
    if len(components_missing) > 20:
        print(f'  ... and {len(components_missing) - 20} more')
    
    print()
    print('Required annotations for all pipeline components:')
    print('  __phase__ = \"X\"  # Pipeline phase (I, X, K, A, L, R, O, G, T, S)')
    print('  __code__ = \"XXX\"  # Component code (e.g., \"01I\", \"15A\")')  
    print('  __stage_order__ = N  # Stage order in pipeline sequence')
    print()
    print('To fix, run: python3 scripts/bulk_annotate_components.py')
    sys.exit(1)
else:
    print(f'✅ All {total_components} pipeline components have required annotations')
"

echo "✅ Pipeline component annotations validation passed"