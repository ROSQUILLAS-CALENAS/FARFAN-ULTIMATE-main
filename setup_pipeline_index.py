#!/usr/bin/env python3
"""
Pipeline Index Setup Script

This script sets up the complete pipeline index system including:
1. Initial index generation
2. Git hooks installation  
3. Validation framework setup
4. Visualization generation
"""

import os
import sys
from pathlib import Path
from pipeline_index_system import PipelineIndexSystem

def main():
    """Setup the complete pipeline index system"""
    
    print("🚀 Pipeline Index System Setup")
    print("="*50)
    
    system = PipelineIndexSystem()
    
    # Step 1: Initial scan and index generation
    print("\n1️⃣  Performing initial filesystem scan...")
    components, reconciliation = system.reconcile_index()
    
    print(f"   📊 Discovered {len(components)} components")
    print(f"   📈 Stage distribution:")
    
    stage_counts = {}
    for comp in components:
        stage_counts[comp.stage] = stage_counts.get(comp.stage, 0) + 1
    
    for stage in system.STAGE_ORDER:
        count = stage_counts.get(stage, 0)
        if count > 0:
            print(f"      {system.STAGE_PREFIXES[stage]} - {stage}: {count}")
    
    # Step 2: Save initial index
    print(f"\n2️⃣  Generating canonical index.json...")
    system.save_index(components)
    print(f"   ✅ Index saved to {system.index_path}")
    
    # Step 3: Setup git hooks
    print(f"\n3️⃣  Setting up git hooks...")
    try:
        system.setup_git_hooks()
        print(f"   ✅ Git hooks installed")
        print(f"      - pre-commit: Index validation")
        print(f"      - post-commit: Visualization updates")
    except Exception as e:
        print(f"   ⚠️  Git hooks setup failed: {e}")
    
    # Step 4: Generate initial visualizations
    print(f"\n4️⃣  Generating pipeline visualizations...")
    try:
        system.generate_dag_visualization(components)
        print(f"   ✅ Visualizations generated in pipeline_visualizations/")
        print(f"      - pipeline_dag.png")
        print(f"      - pipeline_dag.svg") 
        print(f"      - pipeline_dag_networkx.png")
        print(f"      - pipeline_dag_networkx.svg")
    except Exception as e:
        print(f"   ⚠️  Visualization generation failed: {e}")
        print(f"      This may require additional dependencies (graphviz, matplotlib)")
    
    # Step 5: Initial validation
    print(f"\n5️⃣  Performing initial validation...")
    validation = system.validate_index_consistency(components)
    
    if validation['valid']:
        print(f"   ✅ Initial validation passed")
    else:
        print(f"   ⚠️  Initial validation found issues:")
        for error in validation['errors']:
            print(f"      ❌ {error}")
    
    # Step 6: Create validation reports directory
    print(f"\n6️⃣  Setting up validation infrastructure...")
    reports_dir = Path("validation_reports")
    reports_dir.mkdir(exist_ok=True)
    
    viz_dir = Path("pipeline_visualizations") 
    viz_dir.mkdir(exist_ok=True)
    
    print(f"   ✅ Created directories:")
    print(f"      - {reports_dir}/ (validation reports)")
    print(f"      - {viz_dir}/ (DAG visualizations)")
    
    # Step 7: Generate usage documentation
    print(f"\n7️⃣  Generating usage documentation...")
    
    readme_content = """# Pipeline Index System

This directory contains the comprehensive pipeline index system that maintains a single source of truth for all pipeline components.

## Files

- `index.json` - Canonical component registry
- `pipeline_index_system.py` - Main system implementation
- `validate_pipeline_index.py` - CI/CD validation script
- `setup_pipeline_index.py` - Initial setup script

## Usage

### Manual Operations

```bash
# Scan and reconcile components
python pipeline_index_system.py --reconcile

# Validate index consistency  
python pipeline_index_system.py --validate

# Generate visualizations
python pipeline_index_system.py --visualize

# Setup git hooks
python pipeline_index_system.py --setup-hooks
```

### CI/CD Integration

Add to your build pipeline:

```bash
# Validate index before build
python validate_pipeline_index.py
```

This will:
- ✅ Validate index matches filesystem
- ❌ Fail build if critical mismatches found
- 🔄 Auto-update index for safe changes
- 📊 Generate validation reports

## Git Hooks

The system automatically installs git hooks:

- **pre-commit**: Validates index consistency before commits
- **post-commit**: Updates visualizations after successful commits

## Visualizations

The system generates several DAG visualizations showing component dependencies:

- `pipeline_visualizations/pipeline_dag.png` - Graphviz DAG
- `pipeline_visualizations/pipeline_dag.svg` - Graphviz DAG (vector)
- `pipeline_visualizations/pipeline_dag_networkx.png` - NetworkX DAG
- `pipeline_visualizations/pipeline_dag_networkx.svg` - NetworkX DAG (vector)

## Stage Order

Components are organized in this canonical order:

1. **I** - Ingestion/Preparation
2. **X** - Context Construction  
3. **K** - Knowledge Extraction
4. **A** - Analysis/NLP
5. **L** - Classification/Evaluation
6. **O** - Orchestration/Control
7. **R** - Search/Retrieval
8. **S** - Synthesis/Output
9. **G** - Aggregation/Reporting
10. **T** - Integration/Storage

## Validation Rules

The system enforces these constraints:

1. All filesystem components must be in index
2. All index components must exist on filesystem  
3. Component dependencies must respect stage order
4. Component codes must be sequential within stages
5. File hashes must match between index and filesystem

Violations will fail the build and must be resolved before proceeding.
"""
    
    readme_path = system.canonical_dir / "README_INDEX_SYSTEM.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"   ✅ Documentation generated: {readme_path}")
    
    # Step 8: Summary
    print(f"\n" + "="*50)
    print(f"🎉 PIPELINE INDEX SYSTEM SETUP COMPLETE")
    print(f"="*50)
    
    print(f"\n📋 Summary:")
    print(f"   Components indexed: {len(components)}")
    print(f"   Git hooks: {'✅ Installed' if Path('.git/hooks/pre-commit').exists() else '❌ Failed'}")
    print(f"   Visualizations: {'✅ Generated' if viz_dir.exists() else '❌ Failed'}")
    print(f"   Validation: {'✅ Passed' if validation['valid'] else '❌ Issues found'}")
    
    print(f"\n🔧 Next Steps:")
    print(f"   1. Review generated index.json")
    print(f"   2. Check pipeline visualizations")
    print(f"   3. Integrate validate_pipeline_index.py into CI/CD")
    print(f"   4. Commit changes to activate git hooks")
    
    print(f"\n✨ The pipeline index system is now active!")

if __name__ == "__main__":
    main()