#!/usr/bin/env python3
"""
Add Pipeline Contract Annotations Tool
Automatically adds mandatory static contract annotations to all pipeline components.
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Any
import ast
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_flow.pipeline_contract_annotations import (
    ComponentAnnotations, 
    ComponentScanner, 
    PipelinePhase,
    add_contract_annotations
)


class PipelineAnnotator:
    """Tool to add mandatory annotations to all pipeline components"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.scanner = ComponentScanner(root_path)
        self.sequence_counters = {phase: 1 for phase in PipelinePhase}
        self.existing_codes = set()
        
        # Load existing component codes from index.json if available
        self._load_existing_codes()
    
    def _load_existing_codes(self):
        """Load existing component codes from canonical_flow/index.json"""
        index_file = self.root_path / "canonical_flow" / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    components = json.load(f)
                    
                for comp in components:
                    code = comp.get('code', '')
                    if code:
                        self.existing_codes.add(code)
                        # Update sequence counters based on existing codes
                        if len(code) >= 3:
                            try:
                                seq_num = int(code[:2])
                                phase_char = code[2:]
                                for phase in PipelinePhase:
                                    if phase.value == phase_char:
                                        self.sequence_counters[phase] = max(self.sequence_counters[phase], seq_num + 1)
                                        break
                            except ValueError:
                                pass
                                
            except Exception as e:
                print(f"Warning: Could not load existing codes from {index_file}: {e}")
    
    def scan_and_annotate_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """Scan all components and add missing annotations"""
        
        print("🔍 Scanning for pipeline components...")
        scan_result = self.scanner.scan_components()
        
        print(f"Found {scan_result['total_components']} pipeline components")
        print(f"Components missing annotations: {scan_result['missing_annotations']}")
        
        if dry_run:
            print("\n🔍 DRY RUN - No files will be modified")
        
        annotated_files = []
        errors = []
        
        for missing in scan_result['components_missing']:
            file_path = Path(missing['file'])
            
            try:
                # Determine phase from file path
                phase = ComponentAnnotations.extract_phase_from_path(str(file_path))
                
                # Check if component already has a code assigned
                existing_code = self._find_existing_code(file_path)
                
                if existing_code:
                    component_code = existing_code
                else:
                    # Generate new component code
                    component_code = self._generate_next_code(phase)
                
                if not dry_run:
                    # Add annotations to file
                    success = self._add_annotations_to_file(file_path, phase, component_code)
                    
                    if success:
                        annotated_files.append({
                            'file': str(file_path),
                            'phase': phase.value,
                            'code': component_code,
                            'stage_order': ComponentAnnotations.STAGE_ORDER[phase]
                        })
                    else:
                        errors.append(f"Failed to annotate {file_path}")
                else:
                    annotated_files.append({
                        'file': str(file_path),
                        'phase': phase.value,
                        'code': component_code,
                        'stage_order': ComponentAnnotations.STAGE_ORDER[phase],
                        'dry_run': True
                    })
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        return {
            'components_scanned': scan_result['total_components'],
            'components_annotated': len(annotated_files),
            'annotated_files': annotated_files,
            'errors': errors,
            'dry_run': dry_run
        }
    
    def _find_existing_code(self, file_path: Path) -> str:
        """Check if component already has a code in index.json"""
        index_file = self.root_path / "canonical_flow" / "index.json"
        if not index_file.exists():
            return None
            
        try:
            with open(index_file, 'r') as f:
                components = json.load(f)
                
            file_str = str(file_path)
            for comp in components:
                original_path = comp.get('original_path', '')
                if original_path and (original_path in file_str or file_str.endswith(original_path)):
                    return comp.get('code', '')
                    
        except Exception:
            pass
            
        return None
    
    def _generate_next_code(self, phase: PipelinePhase) -> str:
        """Generate next available component code for phase"""
        while True:
            code = ComponentAnnotations.generate_component_code(phase, self.sequence_counters[phase])
            if code not in self.existing_codes:
                self.existing_codes.add(code)
                self.sequence_counters[phase] += 1
                return code
            self.sequence_counters[phase] += 1
    
    def _add_annotations_to_file(self, file_path: Path, phase: PipelinePhase, component_code: str) -> bool:
        """Add contract annotations to a specific file"""
        try:
            stage_order = ComponentAnnotations.STAGE_ORDER[phase]
            
            # Read existing file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if annotations already exist
            if '__phase__' in content and '__code__' in content and '__stage_order__' in content:
                print(f"✓ Annotations already exist in {file_path}")
                return True
            
            # Prepare annotation block
            annotations = f'''
# Mandatory Pipeline Contract Annotations
__phase__ = "{phase.value}"
__code__ = "{component_code}"  
__stage_order__ = {stage_order}
'''
            
            # Find insertion point after docstring/imports
            lines = content.split('\n')
            insert_index = self._find_annotation_insertion_point(lines)
            
            # Insert annotations
            lines.insert(insert_index, annotations)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print(f"✅ Added annotations to {file_path} [{component_code}:{phase.value}]")
            return True
            
        except Exception as e:
            print(f"❌ Error adding annotations to {file_path}: {e}")
            return False
    
    def _find_annotation_insertion_point(self, lines: List[str]) -> int:
        """Find the best place to insert annotations in the file"""
        insert_index = 0
        
        # Skip initial docstring
        if len(lines) > 0:
            first_line = lines[0].strip()
            if first_line.startswith('"""') or first_line.startswith("'''"):
                quote_type = '"""' if '"""' in first_line else "'''"
                
                # Handle single-line docstring
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
            # Stop at first non-import, non-comment line
            insert_index = i
            break
        
        return insert_index
    
    def generate_ci_validation_rules(self) -> str:
        """Generate CI validation rules for pipeline annotations"""
        
        ci_script = """#!/bin/bash
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
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path('.').resolve()))

try:
    from canonical_flow.pipeline_contract_annotations import ComponentScanner
    
    scanner = ComponentScanner('.')
    result = scanner.scan_components()
    
    if result['missing_annotations'] > 0:
        print(f'❌ Found {result[\"missing_annotations\"]} components missing required annotations')
        print('Components missing annotations:')
        for comp in result['components_missing']:
            print(f'  - {comp[\"file\"]}: missing {comp[\"missing\"]}')
        
        print()
        print('Required annotations for all pipeline components:')
        print('  __phase__ = \"X\"  # Pipeline phase (I, X, K, A, L, R, O, G, T, S)')
        print('  __code__ = \"XXX\"  # Component code (e.g., \"01I\", \"15A\")')  
        print('  __stage_order__ = N  # Stage order in pipeline sequence')
        print()
        print('To fix, run: python3 tools/add_pipeline_annotations.py --annotate')
        sys.exit(1)
    else:
        print(f'✅ All {result[\"total_components\"]} pipeline components have required annotations')
        
except ImportError as e:
    print(f'❌ Failed to import validation modules: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Validation failed: {e}')
    sys.exit(1)
"

echo "✅ Pipeline component annotations validation passed"
"""
        
        return ci_script


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add mandatory pipeline contract annotations')
    parser.add_argument('--scan', action='store_true', help='Scan components without modifying')
    parser.add_argument('--annotate', action='store_true', help='Add missing annotations to components')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--generate-ci', action='store_true', help='Generate CI validation rules')
    parser.add_argument('--root', default='.', help='Root directory to scan (default: current)')
    
    args = parser.parse_args()
    
    if args.generate_ci:
        annotator = PipelineAnnotator(args.root)
        ci_script = annotator.generate_ci_validation_rules()
        
        ci_file = Path('.github/workflows/validate-annotations.yml')
        ci_file.parent.mkdir(parents=True, exist_ok=True)
        
        github_workflow = f"""name: Validate Pipeline Annotations

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  validate-annotations:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Validate Pipeline Component Annotations
      run: |
{chr(10).join('        ' + line for line in ci_script.strip().split(chr(10)))}
"""
        
        with open(ci_file, 'w') as f:
            f.write(github_workflow)
        
        # Also create standalone script
        script_file = Path('scripts/validate_pipeline_annotations.sh')
        script_file.parent.mkdir(exist_ok=True)
        with open(script_file, 'w') as f:
            f.write(ci_script)
        script_file.chmod(0o755)
        
        print(f"✅ Generated CI validation rules:")
        print(f"   - GitHub Workflow: {ci_file}")
        print(f"   - Standalone script: {script_file}")
        return
    
    annotator = PipelineAnnotator(args.root)
    
    if args.scan:
        print("🔍 Scanning pipeline components...")
        result = annotator.scanner.scan_components()
        
        print(f"\n📊 Scan Results:")
        print(f"   Total components found: {result['total_components']}")
        print(f"   Missing annotations: {result['missing_annotations']}")
        
        if result['components_missing']:
            print(f"\n📋 Components missing annotations:")
            for comp in result['components_missing']:
                print(f"   - {comp['file']}: missing {', '.join(comp['missing'])}")
        
    elif args.annotate or args.dry_run:
        result = annotator.scan_and_annotate_all(dry_run=args.dry_run)
        
        print(f"\n📊 Annotation Results:")
        print(f"   Components scanned: {result['components_scanned']}")
        print(f"   Components annotated: {result['components_annotated']}")
        
        if result['errors']:
            print(f"   Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"     - {error}")
        
        if result['annotated_files']:
            print(f"\n📋 {'Would annotate' if args.dry_run else 'Annotated'} files:")
            for comp in result['annotated_files']:
                print(f"   - {comp['file']} [{comp['code']}:{comp['phase']}]")
        
        if not args.dry_run and result['components_annotated'] > 0:
            print(f"\n✅ Successfully added annotations to {result['components_annotated']} components")
            print("   Run validation with: python3 scripts/validate_pipeline_annotations.sh")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()