#!/usr/bin/env python3
"""
Generate a comprehensive static analysis report with remediation suggestions.
Part of the CI validation system for static analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional


def read_file_safe(file_path: str) -> str:
    """Safely read a file, returning empty string if not found."""
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except (FileNotFoundError, IOError):
        return ""


def count_violations(content: str, violation_patterns: List[str]) -> int:
    """Count violations based on patterns in content."""
    count = 0
    lines = content.split('\n')
    for line in lines:
        for pattern in violation_patterns:
            if pattern.lower() in line.lower():
                count += 1
                break
    return count


def generate_summary_section(results: Dict[str, str]) -> str:
    """Generate the summary section of the analysis report."""
    mypy_violations = count_violations(results['mypy'], ['error:', 'warning:'])
    ruff_violations = count_violations(results['ruff'], ['error', 'warning'])
    star_import_violations = count_violations(results['star_imports'], ['star imports detected'])
    circular_import_violations = count_violations(results['circular_imports'], ['circular imports detected'])
    type_import_violations = count_violations(results['type_imports'], ['violations detected'])
    import_order_violations = count_violations(results['import_order'], ['I0', 'I1'])
    
    total_violations = (mypy_violations + ruff_violations + star_import_violations + 
                       circular_import_violations + type_import_violations + import_order_violations)
    
    status = "🚨 FAILED" if total_violations > 0 else "✅ PASSED"
    
    return f"""# Static Analysis Summary

**Status:** {status}  
**Total Violations:** {total_violations}

## Violation Breakdown

| Check | Status | Count | Severity |
|-------|--------|-------|----------|
| MyPy Type Checking | {'❌' if mypy_violations > 0 else '✅'} | {mypy_violations} | High |
| Ruff Linting | {'❌' if ruff_violations > 0 else '✅'} | {ruff_violations} | Medium |
| Star Imports | {'❌' if star_import_violations > 0 else '✅'} | {star_import_violations} | Medium |
| Circular Imports | {'❌' if circular_import_violations > 0 else '✅'} | {circular_import_violations} | High |
| Type Import Issues | {'❌' if type_import_violations > 0 else '✅'} | {type_import_violations} | Medium |
| Import Organization | {'❌' if import_order_violations > 0 else '✅'} | {import_order_violations} | Low |

"""


def generate_detailed_sections(results: Dict[str, str]) -> str:
    """Generate detailed sections for each analysis type."""
    sections = []
    
    # MyPy section
    if results['mypy'].strip():
        sections.append(f"""## 🔍 MyPy Type Checking Results

### Issues Found:
```
{results['mypy'][:2000]}{'...' if len(results['mypy']) > 2000 else ''}
```

### Remediation Guide:
- **Add type annotations** to all function parameters and return values
- **Use strict typing** with `disallow_untyped_defs=True`
- **Import proper types** from `typing` module
- **Fix Any types** by using more specific type annotations
- **Check third-party library stubs** or add `# type: ignore` comments for untyped libraries

### Example Fixes:
```python
# Before
def process_data(data):
    return data.transform()

# After  
def process_data(data: Dict[str, Any]) -> ProcessedData:
    return data.transform()
```
""")

    # Ruff section
    if results['ruff'].strip():
        sections.append(f"""## 🧹 Ruff Linting Results

### Issues Found:
```
{results['ruff'][:2000]}{'...' if len(results['ruff']) > 2000 else ''}
```

### Common Fixes:
- **Fix import order** with `isort` or ruff's import sorting
- **Remove unused imports** and variables
- **Follow PEP 8** formatting guidelines
- **Use comprehensions** instead of unnecessary loops
- **Simplify boolean expressions** and conditions
""")

    # Star imports section
    if results['star_imports'].strip():
        sections.append(f"""## ⭐ Star Import Analysis

### Issues Found:
```
{results['star_imports']}
```

### Why Star Imports Are Problematic:
- **Namespace pollution**: Unclear which symbols come from which modules
- **Hidden dependencies**: Hard to track what's actually being used
- **IDE confusion**: Auto-completion and refactoring tools struggle
- **Name conflicts**: Risk of accidentally overriding symbols

### Fix Pattern:
```python
# ❌ Bad
from module import *

# ✅ Good
from module import specific_function, SpecificClass
# or
import module
```
""")

    # Circular imports section
    if results['circular_imports'].strip():
        sections.append(f"""## 🔄 Circular Import Analysis

### Issues Found:
```
{results['circular_imports']}
```

### Resolution Strategies:
1. **Extract common code** into a separate module
2. **Use local imports** inside functions
3. **Refactor module boundaries** based on responsibilities
4. **Use TYPE_CHECKING imports** for type hints only
5. **Consider dependency injection** patterns

### Example Refactoring:
```python
# ❌ Before - Circular dependency
# module_a.py
from module_b import helper

# module_b.py  
from module_a import data_processor

# ✅ After - Extract common dependencies
# shared.py
def common_helper(): ...

# module_a.py
from shared import common_helper

# module_b.py
from shared import common_helper
```
""")

    # Type imports section
    if results['type_imports'].strip():
        sections.append(f"""## 📝 Type Import Analysis

### Issues Found:
```
{results['type_imports']}
```

### Best Practices:
- **Use TYPE_CHECKING** for imports only needed for type annotations
- **Separate runtime vs type-only** imports clearly
- **Import TYPE_CHECKING** from typing when using the pattern
- **Use forward references** for complex circular type dependencies

### Correct Pattern:
```python
from typing import TYPE_CHECKING, List, Dict  # Runtime types

if TYPE_CHECKING:
    from typing import Protocol  # Type-only symbol
    from .models import User     # Circular dependency
    
def process_users(users: List[User]) -> Dict[str, Any]:
    # Implementation uses runtime types only
    pass
```
""")

    # Import order section  
    if results['import_order'].strip():
        sections.append(f"""## 📋 Import Organization

### Issues Found:
```
{results['import_order']}
```

### Import Order Standards (PEP 8):
1. **Standard library** imports
2. **Third-party** imports  
3. **Local application** imports

### Within each group:
- **Absolute imports** before relative imports
- **Alphabetical order** within each section
- **One import per line** for readability

### Example:
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch
from transformers import AutoModel

# Local imports
from .models import QueryExpander
from .utils import load_config
```
""")

    return '\n'.join(sections)


def generate_action_items() -> str:
    """Generate actionable next steps."""
    return """## 🎯 Action Items

### Immediate Actions (High Priority)
1. **Fix all MyPy type errors** - These indicate potential runtime bugs
2. **Resolve circular imports** - These can cause import failures and are hard to debug
3. **Add missing type annotations** - Improves code quality and IDE support

### Short-term Actions (Medium Priority)  
4. **Eliminate star imports** - Replace with explicit imports
5. **Fix type checking imports** - Separate runtime from type-only imports
6. **Clean up Ruff violations** - Address code quality issues

### Long-term Actions (Low Priority)
7. **Organize import order** - Improve code readability
8. **Set up pre-commit hooks** - Prevent future violations
9. **Configure IDE integration** - Enable real-time linting

### Prevention Strategies
- **Enable strict mode** in your IDE/editor
- **Run static analysis locally** before committing
- **Set up CI gates** to prevent merging with violations  
- **Regular code reviews** focusing on type safety
- **Developer training** on Python typing best practices

## 📚 Resources
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Ruff Rules Reference](https://docs.astral.sh/ruff/rules/)
- [Python Type Hints Guide](https://docs.python.org/3/library/typing.html)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)"""


def main() -> int:
    """Generate comprehensive static analysis report."""
    parser = argparse.ArgumentParser(description="Generate static analysis report")
    parser.add_argument("--mypy-results", default="", help="Path to MyPy results")
    parser.add_argument("--ruff-results", default="", help="Path to Ruff results")
    parser.add_argument("--star-imports", default="", help="Path to star imports results")
    parser.add_argument("--circular-imports", default="", help="Path to circular imports results")
    parser.add_argument("--type-imports", default="", help="Path to type imports results")
    parser.add_argument("--import-order", default="", help="Path to import order results")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    
    args = parser.parse_args()
    
    # Read all result files
    results = {
        'mypy': read_file_safe(args.mypy_results),
        'ruff': read_file_safe(args.ruff_results),
        'star_imports': read_file_safe(args.star_imports),
        'circular_imports': read_file_safe(args.circular_imports),
        'type_imports': read_file_safe(args.type_imports),
        'import_order': read_file_safe(args.import_order),
    }
    
    # Generate report sections
    summary = generate_summary_section(results)
    detailed = generate_detailed_sections(results)
    actions = generate_action_items()
    
    # Combine into final report
    report = f"""{summary}

{detailed}

{actions}

---
*Report generated by static analysis CI pipeline*"""
    
    # Write report
    try:
        Path(args.output).write_text(report, encoding='utf-8')
        print(f"✅ Analysis report generated: {args.output}")
        return 0
    except IOError as e:
        print(f"❌ Failed to write report: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())