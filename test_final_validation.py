#!/usr/bin/env python3
"""
Final validation test for pipeline component annotations
"""

import os
import re
import json
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
    return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)

def has_annotations(content: str) -> bool:
    """Check if file has required annotations"""
    return '__phase__' in content and '__code__' in content and '__stage_order__' in content

def validate_annotations(content: str) -> dict:
    """Validate annotation values"""
    issues = []
    
    # Extract phase
    phase_match = re.search(r'__phase__\s*=\s*["\']([IXKALROGTS])["\']', content)
    if not phase_match:
        issues.append("Invalid or missing __phase__ annotation")
    
    # Extract code
    code_match = re.search(r'__code__\s*=\s*["\'](\d{2}[IXKALROGTS])["\']', content)
    if not code_match:
        issues.append("Invalid or missing __code__ annotation")
    
    # Extract stage order
    order_match = re.search(r'__stage_order__\s*=\s*(\d+)', content)
    if not order_match:
        issues.append("Invalid or missing __stage_order__ annotation")
    
    # Validate consistency
    if phase_match and code_match:
        phase = phase_match.group(1)
        code = code_match.group(1)
        if not code.endswith(phase):
            issues.append(f"Phase {phase} doesn't match code {code}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'phase': phase_match.group(1) if phase_match else None,
        'code': code_match.group(1) if code_match else None,
        'stage_order': int(order_match.group(1)) if order_match else None
    }

def main():
    """Main validation function"""
    print("🔍 Final Pipeline Component Annotation Validation")
    
    components = []
    annotated = []
    missing_annotations = []
    invalid_annotations = []
    codes_seen = set()
    phase_counts = {}
    
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
                        components.append(file_path)
                        
                        if has_annotations(content):
                            validation = validate_annotations(content)
                            
                            if validation['valid']:
                                annotated.append({
                                    'file': file_path,
                                    'phase': validation['phase'],
                                    'code': validation['code'],
                                    'stage_order': validation['stage_order']
                                })
                                
                                # Track codes and phases
                                code = validation['code']
                                if code in codes_seen:
                                    print(f"⚠️  Duplicate code {code} in {file_path}")
                                codes_seen.add(code)
                                
                                phase = validation['phase']
                                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                                
                            else:
                                invalid_annotations.append({
                                    'file': file_path,
                                    'issues': validation['issues']
                                })
                        else:
                            missing_annotations.append(file_path)
                            
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Results
    print(f"\n📊 Validation Results:")
    print(f"   Total pipeline components: {len(components)}")
    print(f"   Properly annotated: {len(annotated)}")
    print(f"   Missing annotations: {len(missing_annotations)}")
    print(f"   Invalid annotations: {len(invalid_annotations)}")
    
    if missing_annotations:
        print(f"\n❌ Components missing annotations:")
        for comp in missing_annotations[:10]:
            print(f"   - {comp}")
        if len(missing_annotations) > 10:
            print(f"   ... and {len(missing_annotations) - 10} more")
    
    if invalid_annotations:
        print(f"\n⚠️  Components with invalid annotations:")
        for comp in invalid_annotations[:5]:
            print(f"   - {comp['file']}: {', '.join(comp['issues'])}")
        if len(invalid_annotations) > 5:
            print(f"   ... and {len(invalid_annotations) - 5} more")
    
    # Phase distribution
    print(f"\n📋 Phase Distribution:")
    for phase in sorted(phase_counts.keys()):
        phase_names = {
            'I': 'Ingestion', 'X': 'Context', 'K': 'Knowledge', 'A': 'Analysis', 'L': 'Classification',
            'R': 'Retrieval', 'O': 'Orchestration', 'G': 'Aggregation', 'T': 'Integration', 'S': 'Synthesis'
        }
        print(f"   {phase} ({phase_names.get(phase, 'Unknown')}): {phase_counts[phase]} components")
    
    # Success criteria
    success = len(missing_annotations) == 0 and len(invalid_annotations) == 0
    
    if success:
        print(f"\n✅ SUCCESS: All {len(components)} pipeline components have valid annotations!")
        print(f"   Total unique codes: {len(codes_seen)}")
        print(f"   Phases covered: {len(phase_counts)}/10")
    else:
        print(f"\n❌ VALIDATION FAILED")
        print(f"   Components needing fixes: {len(missing_annotations) + len(invalid_annotations)}")
    
    # Save detailed report
    report = {
        'validation_status': 'SUCCESS' if success else 'FAILED',
        'total_components': len(components),
        'annotated_components': len(annotated),
        'missing_annotations': len(missing_annotations),
        'invalid_annotations': len(invalid_annotations),
        'phase_distribution': phase_counts,
        'annotated_files': annotated[:50],  # Sample
        'missing_files': missing_annotations,
        'invalid_files': invalid_annotations
    }
    
    with open('final_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: final_validation_report.json")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)