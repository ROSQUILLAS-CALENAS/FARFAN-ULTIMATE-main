#!/usr/bin/env python3
"""
Pipeline Index Validation Script

This script is designed to be integrated into CI/CD pipelines to ensure
the index.json file accurately reflects the current codebase state.
"""

import sys
import json
import subprocess
from pathlib import Path
from pipeline_index_system import PipelineIndexSystem

def main():
    """Main validation entry point for CI/CD integration"""
    
    print("🔍 Pipeline Index Validation Starting...")
    
    system = PipelineIndexSystem()
    
    # Step 1: Reconcile filesystem with index
    print("📊 Reconciling filesystem with index.json...")
    components, reconciliation = system.reconcile_index()
    
    # Step 2: Validate consistency
    print("✅ Validating index consistency...")
    validation = system.validate_index_consistency(components)
    
    # Step 3: Report results
    print("\n" + "="*60)
    print("PIPELINE INDEX VALIDATION REPORT")
    print("="*60)
    
    print(f"\n📈 Component Statistics:")
    print(f"  Total Components: {len(components)}")
    print(f"  Added: {len(reconciliation['added'])}")
    print(f"  Modified: {len(reconciliation['modified'])}")
    print(f"  Deleted: {len(reconciliation['deleted'])}")
    print(f"  Unchanged: {len(reconciliation['unchanged'])}")
    
    if reconciliation['added']:
        print(f"\n➕ Newly Discovered Components:")
        for path in reconciliation['added']:
            print(f"    + {path}")
    
    if reconciliation['modified']:
        print(f"\n🔄 Modified Components:")
        for path in reconciliation['modified']:
            print(f"    ~ {path}")
    
    if reconciliation['deleted']:
        print(f"\n❌ Missing Components (in index but not on filesystem):")
        for path in reconciliation['deleted']:
            print(f"    - {path}")
    
    # Step 4: Validation results
    print(f"\n🎯 Validation Status: {'✅ PASS' if validation['valid'] else '❌ FAIL'}")
    
    if validation['errors']:
        print(f"\n🚨 Critical Errors:")
        for error in validation['errors']:
            print(f"    ❌ {error}")
    
    if validation['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"    ⚠️  {warning}")
    
    # Step 5: Auto-update index if validation passed with only modifications
    if (validation['valid'] and 
        (reconciliation['added'] or reconciliation['modified']) and
        not reconciliation['deleted']):
        
        print(f"\n🔄 Auto-updating index.json with discovered changes...")
        system.save_index(components)
        print(f"✅ Index.json updated successfully")
        
        # Generate updated visualizations
        print(f"📊 Updating pipeline visualizations...")
        try:
            system.generate_dag_visualization(components)
            print(f"✅ Visualizations updated")
        except Exception as e:
            print(f"⚠️  Visualization update failed: {e}")
    
    # Step 6: Generate validation artifacts
    validation_report_path = Path("validation_reports/pipeline_index_validation.json")
    validation_report_path.parent.mkdir(exist_ok=True)
    
    full_report = {
        "validation": validation,
        "reconciliation": reconciliation,
        "component_count": len(components),
        "stage_distribution": {}
    }
    
    # Add stage distribution
    stage_counts = {}
    for comp in components:
        stage_counts[comp.stage] = stage_counts.get(comp.stage, 0) + 1
    full_report["stage_distribution"] = stage_counts
    
    with open(validation_report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"\n📋 Full validation report saved to: {validation_report_path}")
    
    # Step 7: Return appropriate exit code
    if not validation['valid']:
        print(f"\n❌ BUILD SHOULD FAIL - Index validation errors detected")
        sys.exit(1)
    else:
        print(f"\n✅ BUILD CAN PROCEED - Index validation passed")
        sys.exit(0)

if __name__ == "__main__":
    main()