#!/usr/bin/env python3
"""
Count pipeline components that need annotations
"""

import os
import re
from pathlib import Path

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

def count_components():
    """Count all Python files that look like pipeline components"""
    components = []
    missing_annotations = []
    
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
                    
                    if is_pipeline_component(content):
                        components.append(file_path)
                        
                        # Check if has annotations
                        if '__phase__' not in content or '__code__' not in content or '__stage_order__' not in content:
                            missing_annotations.append(file_path)
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    print(f"📊 Component Analysis:")
    print(f"   Total pipeline components: {len(components)}")
    print(f"   Missing annotations: {len(missing_annotations)}")
    print(f"   Already annotated: {len(components) - len(missing_annotations)}")
    
    if missing_annotations:
        print(f"\n📋 First 20 components missing annotations:")
        for comp in missing_annotations[:20]:
            print(f"   - {comp}")
        
        if len(missing_annotations) > 20:
            print(f"   ... and {len(missing_annotations) - 20} more")
    
    return {
        'total': len(components),
        'missing': len(missing_annotations),
        'components': components,
        'missing_list': missing_annotations
    }

if __name__ == "__main__":
    count_components()