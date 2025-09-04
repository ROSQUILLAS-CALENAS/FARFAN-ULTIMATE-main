#!/usr/bin/env python3
"""
Demo script showing the pipeline index system in action
"""

from pipeline_index_system import PipelineIndexSystem
from pathlib import Path
import json

def main():
    """Demonstrate the pipeline index system functionality"""
    
    print("🚀 Pipeline Index System Demo")
    print("=" * 50)
    
    # Initialize system
    system = PipelineIndexSystem()
    
    # Step 1: Scan and reconcile
    print("\n1️⃣  Scanning filesystem and reconciling with index...")
    components, reconciliation = system.reconcile_index()
    
    print(f"   📊 Found {len(components)} components")
    print(f"   ➕ Added: {len(reconciliation['added'])}")
    print(f"   🔄 Modified: {len(reconciliation['modified'])}")
    print(f"   ❌ Deleted: {len(reconciliation['deleted'])}")
    print(f"   ✅ Unchanged: {len(reconciliation['unchanged'])}")
    
    # Step 2: Show stage distribution
    print("\n2️⃣  Component distribution by stage:")
    stage_counts = {}
    for comp in components:
        stage_counts[comp.stage] = stage_counts.get(comp.stage, 0) + 1
    
    for stage in system.STAGE_ORDER:
        count = stage_counts.get(stage, 0)
        if count > 0:
            prefix = system.STAGE_PREFIXES[stage]
            print(f"      {prefix} - {stage}: {count} components")
    
    # Step 3: Save updated index
    print(f"\n3️⃣  Saving updated index to {system.index_path}...")
    system.save_index(components)
    
    # Step 4: Validate consistency
    print(f"\n4️⃣  Validating index consistency...")
    validation = system.validate_index_consistency(components)
    
    status = "✅ VALID" if validation['valid'] else "❌ INVALID"
    print(f"      Status: {status}")
    
    if validation['errors']:
        print(f"      Errors: {len(validation['errors'])}")
    if validation['warnings']:
        print(f"      Warnings: {len(validation['warnings'])}")
    
    # Step 5: Generate visualizations
    print(f"\n5️⃣  Generating visualizations...")
    try:
        system.generate_dag_visualization(components)
        print(f"      ✅ Visualizations generated in pipeline_visualizations/")
    except Exception as e:
        print(f"      ⚠️  Visualization error: {e}")
    
    # Step 6: Show sample dependencies
    print(f"\n6️⃣  Sample component dependencies:")
    dependencies = system.calculate_dependencies(components)
    
    count = 0
    for comp_code, deps in dependencies.items():
        if deps and count < 5:  # Show first 5 with dependencies
            comp = next(c for c in components if c.code == comp_code)
            print(f"      {comp_code} ({comp.stage}) -> {', '.join(deps)}")
            count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total components: {len(components)}")
    print(f"   Total dependencies: {sum(len(deps) for deps in dependencies.values())}")
    print(f"   Components with deps: {len([d for d in dependencies.values() if d])}")
    print(f"   Index file: {system.index_path}")
    
    print(f"\n✨ Demo completed successfully!")

if __name__ == "__main__":
    main()