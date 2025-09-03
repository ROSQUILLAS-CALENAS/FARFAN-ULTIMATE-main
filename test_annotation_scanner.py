#!/usr/bin/env python3
"""
Test the pipeline annotation scanner to find components missing annotations
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from canonical_flow.pipeline_contract_annotations import ComponentScanner, PipelinePhase

def main():
    print("🔍 Scanning for pipeline components...")
    scanner = ComponentScanner(".")
    result = scanner.scan_components()
    
    print(f"\n📊 Scan Results:")
    print(f"   Total components found: {result['total_components']}")
    print(f"   Missing annotations: {result['missing_annotations']}")
    
    if result['components_missing']:
        print(f"\n📋 Components missing annotations:")
        for comp in result['components_missing']:
            print(f"   - {comp['file']}: missing {', '.join(comp['missing'])}")
    
    # Show sample of found components
    if result['components_found']:
        print(f"\n📁 Sample components found:")
        for comp in result['components_found'][:10]:  # Show first 10
            print(f"   - {comp}")
        
        if len(result['components_found']) > 10:
            print(f"   ... and {len(result['components_found']) - 10} more")
    
    print(f"\n✅ Scan complete")
    return result

if __name__ == "__main__":
    main()