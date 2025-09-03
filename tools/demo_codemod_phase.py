#!/usr/bin/env python3
"""
Demo script for CodeMod Phase tool.

Shows how to use the refactoring tool for canonical naming cleanup.
"""

import sys
from pathlib import Path

# Import our codemod tool
sys.path.append(str(Path(__file__).parent))
from codemod_phase import CodemodPhase


def main():
    """Demo the codemod tool."""
    print("🔧 CodeMod Phase Demo")
    print("=" * 50)
    
    # Initialize the tool in dry-run mode
    codemod = CodemodPhase(
        project_root=".",
        dry_run=True,  # Safe mode - no actual changes
        verbose=True
    )
    
    print("📋 Discovering refactoring opportunities...")
    operations = codemod.discover_refactoring_opportunities()
    
    print(f"\n✅ Found {len(operations)} opportunities:")
    
    for i, op in enumerate(operations, 1):
        print(f"\n{i}. {op.operation_type.upper()}")
        print(f"   Source: {op.source_path}")
        print(f"   Target: {op.target_path}")
        if op.metadata.get('original_prefix'):
            print(f"   Cleaning up: {op.metadata['original_prefix']} pattern")
        if op.metadata.get('content_cleanup'):
            print(f"   Action: Clean up old naming references in content")
    
    print(f"\n🔍 Import mappings that would be generated:")
    mappings = codemod.generate_import_mappings(operations)
    for old, new in mappings.items():
        print(f"   {old} → {new}")
    
    print("\n⚠️  This is a DRY RUN - no changes were made!")
    print("💡 To actually apply changes:")
    print("   1. Install dependencies: pip install libcst bowler mypy ruff")
    print("   2. Run: python3 tools/codemod_phase.py")
    print("   3. Review changes and validate with tests")
    

if __name__ == '__main__':
    main()