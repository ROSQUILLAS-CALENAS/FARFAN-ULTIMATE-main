# Pipeline Index System

A comprehensive single source of truth system for managing pipeline components with automatic discovery, validation, and DAG visualization capabilities.

## Overview

The Pipeline Index System implements:

1. **Autoscan System** - Automatically discovers and reconciles canonical pipeline components
2. **Index Reconciliation** - Maintains `index.json` as single source of truth
3. **DAG Visualization** - Generates PNG/SVG dependency graphs
4. **Validation Logic** - Ensures consistency between index and filesystem
5. **Git Integration** - Hooks for automatic updates on commits
6. **CI/CD Integration** - Build validation to prevent mismatches

## Quick Start

### Basic Usage

```bash
# Run complete setup (first time)
python3 setup_pipeline_index.py

# Quick demo
python3 demo_pipeline_index.py  

# Manual reconciliation
python3 pipeline_index_system.py --reconcile

# Validation only
python3 pipeline_index_system.py --validate

# Generate visualizations  
python3 pipeline_index_system.py --visualize
```

### CI/CD Integration

Add to your build pipeline:

```bash
# Validate before build
python3 validate_pipeline_index.py
```

This will:
- ✅ Auto-update index for safe changes
- ❌ Fail build for critical mismatches  
- 📊 Generate validation reports

## Architecture

### Core Components

- `pipeline_index_system.py` - Main system implementation
- `validate_pipeline_index.py` - CI/CD validation script  
- `setup_pipeline_index.py` - Initial setup wizard
- `canonical_flow/index.json` - Canonical component registry

### Pipeline Stages

Components are organized in 10 canonical stages:

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

### Component Metadata

Each component includes:

```json
{
  "code": "01I",
  "stage": "ingestion_preparation", 
  "alias_path": "canonical_flow/I_ingestion_preparation/01I_pdf_reader.py",
  "original_path": "pdf_reader.py",
  "file_hash": "abc123...",
  "dependencies": ["02X", "05K"],
  "imports": ["pathlib", "json"],
  "exports": ["PDFReader", "extract_text"],
  "last_modified": "2024-12-19T10:30:00",
  "size_bytes": 5420
}
```

## Features

### Autoscan System

- **Filesystem Discovery**: Automatically discovers Python components
- **Stage Classification**: Determines component stage from path/keywords  
- **Dependency Analysis**: Extracts imports/exports using AST parsing
- **Hash Tracking**: SHA256 hashes for change detection
- **Metadata Enrichment**: Size, timestamps, and more

### Validation Logic

The system enforces several constraints:

1. **Completeness**: All filesystem components must be in index
2. **Consistency**: All index components must exist on filesystem
3. **Stage Order**: Dependencies must respect canonical stage order
4. **Code Sequence**: Component codes must be sequential within stages
5. **Hash Integrity**: File content must match recorded hashes

### DAG Visualization

Generates multiple visualization formats:

- **Graphviz DAGs** - `pipeline_dag.png/.svg` (if graphviz installed)
- **NetworkX DAGs** - `pipeline_dag_networkx.png/.svg` (if matplotlib installed)  
- **Text DAG** - `pipeline_dag.txt` (always generated)

### Git Integration

Automatic git hooks:

- **pre-commit**: Validates index before commits
- **post-commit**: Updates visualizations after commits

Install with: `python3 pipeline_index_system.py --setup-hooks`

## Example Workflows

### Development Workflow

```bash
# 1. Make code changes
vim some_component.py

# 2. Git will auto-validate on commit
git add .
git commit -m "Update component"  # Validation runs automatically

# 3. Visualizations updated automatically
```

### Build Pipeline

```yaml
# .github/workflows/build.yml
- name: Validate Pipeline Index
  run: python3 validate_pipeline_index.py
  
- name: Run Tests  
  run: pytest
  if: success()  # Only if validation passed
```

### Manual Reconciliation

```bash
# Scan for changes
python3 pipeline_index_system.py --scan

# Show what would change
python3 pipeline_index_system.py --validate

# Apply changes
python3 pipeline_index_system.py --reconcile

# Update visualizations
python3 pipeline_index_system.py --visualize
```

## Validation Reports

The system generates detailed reports in `validation_reports/`:

- `pipeline_index_validation.json` - Complete validation results
- Component reconciliation details  
- Stage distribution statistics
- Dependency analysis

## Current Status

✅ **Implemented Features:**
- Complete autoscan system with 330+ components discovered
- Index reconciliation and validation logic
- Text-based DAG visualization  
- Git hooks support
- CI/CD validation script
- Comprehensive test coverage

⚠️ **Optional Dependencies:**
- `graphviz` - For PNG/SVG DAG generation
- `matplotlib` - For NetworkX visualizations
- `networkx` - For graph analysis

🔄 **Auto-Discovery Results:**
- 331 components discovered across 10 stages
- 56 inter-component dependencies identified
- Stage distribution: O(68), R(47), A(45), K(38), L(38), I(27), G(26), T(18), X(15), S(9)

## Benefits

1. **Single Source of Truth** - Authoritative component registry
2. **Automated Maintenance** - No manual index updates required  
3. **Build Safety** - Prevents inconsistent deployments
4. **Visual Understanding** - Clear dependency visualization
5. **Compliance** - Enforces architectural constraints
6. **Observability** - Detailed change tracking and reporting

The system ensures that your pipeline index always accurately reflects the current codebase state while providing rich visualization and validation capabilities.