#!/usr/bin/env python3
"""
Pre-commit Hook for Pipeline Component Annotations
Blocks commits that add components without required annotations
"""

import os
import re
import sys
import subprocess
from typing import List, Dict, Any

def get_staged_python_files() -> List[str]:
    """Get Python files in git staging area"""
    try:
        result = subprocess.run(['git', 'diff', '--cached', '--name-only', '--diff-filter=A'], 
                              capture_output=True, text=True)
        files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return [f for f in files if f.endswith('.py') and f != '__init__.py' and not f.startswith('test_')]
    except subprocess.CalledProcessError:
        return []

def is_pipeline_component(content: str) -> bool:
    """Check if file is a pipeline component"""
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

def has_required_annotations(content: str) -> bool:
    """Check if content has all required annotations"""
    required_annotations = ['__phase__', '__code__', '__stage_order__']
    return all(annotation in content for annotation in required_annotations)

def validate_annotation_format(content: str) -> List[str]:
    """Validate annotation format and return issues"""
    issues = []
    
    # Check phase format
    phase_match = re.search(r'__phase__\s*=\s*["\']([IXKALROGTS])["\']', content)
    if not phase_match:
        issues.append("__phase__ must be one of: I, X, K, A, L, R, O, G, T, S")
    
    # Check code format  
    code_match = re.search(r'__code__\s*=\s*["\'](\d{2}[IXKALROGTS])["\']', content)
    if not code_match:
        issues.append("__code__ must be in format NNX (e.g., '01I', '25A')")
    
    # Check stage order format
    order_match = re.search(r'__stage_order__\s*=\s*(\d+)', content)
    if not order_match:
        issues.append("__stage_order__ must be an integer")
    
    # Validate consistency between phase and code
    if phase_match and code_match:
        phase = phase_match.group(1)
        code = code_match.group(1)
        if not code.endswith(phase):
            issues.append(f"Phase '{phase}' must match code suffix in '{code}'")
    
    return issues

def main() -> int:
    """Main pre-commit hook logic"""
    
    print("🔍 Checking pipeline component annotations...")
    
    staged_files = get_staged_python_files()
    if not staged_files:
        print("✅ No Python files staged for commit")
        return 0
    
    violations = []
    
    for file_path in staged_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if is_pipeline_component(content):
                if not has_required_annotations(content):
                    violations.append({
                        'file': file_path,
                        'error': 'missing_annotations',
                        'message': 'Pipeline component missing required annotations'
                    })
                else:
                    format_issues = validate_annotation_format(content)
                    if format_issues:
                        violations.append({
                            'file': file_path,
                            'error': 'invalid_format',
                            'issues': format_issues
                        })
                        
        except Exception as e:
            print(f"Warning: Could not analyze {file_path}: {e}")
    
    if violations:
        print(f"\n❌ COMMIT BLOCKED - Pipeline annotation violations found:")
        
        for violation in violations:
            print(f"\n📄 {violation['file']}")
            if violation['error'] == 'missing_annotations':
                print("   Missing required annotations:")
                print("   - __phase__ = \"X\"  # Pipeline phase (I,X,K,A,L,R,O,G,T,S)")
                print("   - __code__ = \"NNX\"  # Component code (e.g., '01I', '25A')")
                print("   - __stage_order__ = N  # Stage order (1-10)")
            
            elif violation['error'] == 'invalid_format':
                print("   Invalid annotation format:")
                for issue in violation['issues']:
                    print(f"   - {issue}")
        
        print(f"\n🔧 To fix these issues:")
        print(f"   1. Add missing annotations to the files above")
        print(f"   2. Or run: python3 scripts/bulk_annotate_components.py")
        print(f"   3. Then re-stage and commit your changes")
        
        return 1
    
    print(f"✅ All staged pipeline components have valid annotations")
    return 0

if __name__ == "__main__":
    sys.exit(main())