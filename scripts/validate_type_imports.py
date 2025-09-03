#!/usr/bin/env python3
"""
Static analysis tool to validate proper type checking imports.
Ensures TYPE_CHECKING imports are used correctly and runtime imports are appropriate.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class TypeCheckingImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze type checking imports."""
    
    def __init__(self) -> None:
        self.type_checking_imports: Set[str] = set()
        self.runtime_imports: Set[str] = set()
        self.type_checking_block_active = False
        self.violations: List[Tuple[int, str, str]] = []
        self.type_checking_imported = False
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Analyze ImportFrom nodes."""
        if node.module == "typing":
            for alias in node.names:
                if alias.name == "TYPE_CHECKING":
                    self.type_checking_imported = True
                elif not self.type_checking_block_active:
                    # Runtime typing import - check if it should be in TYPE_CHECKING
                    if alias.name in self._get_type_only_symbols():
                        self.violations.append((
                            node.lineno,
                            f"Runtime import of type-only symbol '{alias.name}' from typing",
                            "Move to TYPE_CHECKING block if only used for type annotations"
                        ))
        
        # Track all imports
        if node.module:
            if self.type_checking_block_active:
                self.type_checking_imports.add(node.module)
            else:
                self.runtime_imports.add(node.module)
                
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Analyze Import nodes."""
        for alias in node.names:
            if self.type_checking_block_active:
                self.type_checking_imports.add(alias.name)
            else:
                self.runtime_imports.add(alias.name)
        self.generic_visit(node)
        
    def visit_If(self, node: ast.If) -> None:
        """Detect TYPE_CHECKING blocks."""
        if (isinstance(node.test, ast.Name) and 
            node.test.id == "TYPE_CHECKING"):
            old_state = self.type_checking_block_active
            self.type_checking_block_active = True
            for child in node.body:
                self.visit(child)
            self.type_checking_block_active = old_state
        else:
            self.generic_visit(node)
    
    def _get_type_only_symbols(self) -> Set[str]:
        """Get symbols that should only be used in TYPE_CHECKING blocks."""
        return {
            "Protocol", "TypedDict", "Literal", "Final", "ClassVar",
            "TypeAlias", "TypeGuard", "NotRequired", "Required",
            "Self", "LiteralString", "Never", "reveal_type"
        }


class RuntimeUsageDetector(ast.NodeVisitor):
    """Detect if imports are used at runtime vs type-checking only."""
    
    def __init__(self, imports_to_check: Set[str]) -> None:
        self.imports_to_check = imports_to_check
        self.runtime_usage: Set[str] = set()
        self.in_type_annotation = False
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions for type annotations."""
        # Check return type annotation
        old_state = self.in_type_annotation
        if node.returns:
            self.in_type_annotation = True
            self.visit(node.returns)
            
        # Check argument type annotations
        for arg in node.args.args:
            if arg.annotation:
                self.in_type_annotation = True
                self.visit(arg.annotation)
                
        self.in_type_annotation = old_state
        
        # Check function body
        for stmt in node.body:
            self.visit(stmt)
            
    def visit_Name(self, node: ast.Name) -> None:
        """Track name usage."""
        if node.id in self.imports_to_check and not self.in_type_annotation:
            self.runtime_usage.add(node.id)
        self.generic_visit(node)
        
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Track attribute access."""
        # Extract base name for imported modules
        if isinstance(node.value, ast.Name):
            if node.value.id in self.imports_to_check and not self.in_type_annotation:
                self.runtime_usage.add(node.value.id)
        self.generic_visit(node)


def analyze_type_imports(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Analyze a file for type checking import violations.
    
    Args:
        file_path: Path to Python file to analyze
        
    Returns:
        List of (line_number, violation, suggestion) tuples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content, filename=str(file_path))
        analyzer = TypeCheckingImportAnalyzer()
        analyzer.visit(tree)
        
        violations = analyzer.violations.copy()
        
        # Check for TYPE_CHECKING usage without import
        if analyzer.type_checking_imports and not analyzer.type_checking_imported:
            violations.append((
                1,
                "TYPE_CHECKING block used without importing TYPE_CHECKING",
                "Add: from typing import TYPE_CHECKING"
            ))
            
        # Check if imports in TYPE_CHECKING are actually used at runtime
        if analyzer.type_checking_imports:
            usage_detector = RuntimeUsageDetector(analyzer.type_checking_imports)
            usage_detector.visit(tree)
            
            for module in usage_detector.runtime_usage:
                violations.append((
                    1,
                    f"Module '{module}' in TYPE_CHECKING block but used at runtime",
                    f"Move '{module}' import outside TYPE_CHECKING block"
                ))
        
        return violations
        
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []


def generate_type_import_remediation(violation: str, suggestion: str) -> str:
    """
    Generate detailed remediation for type import violations.
    
    Args:
        violation: Description of the violation
        suggestion: Suggested fix
        
    Returns:
        Formatted remediation guide
    """
    return f"""
TYPE IMPORT REMEDIATION:
- Issue: {violation}
- Fix: {suggestion}

BEST PRACTICES:
1. **Use TYPE_CHECKING for type-only imports**:
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from .other_module import SomeClass
   ```

2. **Runtime vs Type-only imports**:
   - Runtime: Classes/functions used in actual code execution
   - Type-only: Classes/functions used only in type annotations
   
3. **Common type-only symbols from typing**:
   - Protocol, TypedDict, Literal, Final, ClassVar
   - TypeAlias, TypeGuard, Self, Never
   
4. **Forward references for circular dependencies**:
   ```python
   def function(param: 'ForwardRef') -> 'ReturnType':
       pass
   ```

EXAMPLE CORRECT USAGE:
```python
from typing import TYPE_CHECKING, List, Dict  # Runtime types
if TYPE_CHECKING:
    from typing import Protocol              # Type-only
    from .models import User                # Avoid circular imports
```
"""


def main() -> int:
    """
    Main function to validate type checking imports.
    
    Returns:
        Exit code: 0 if no violations found, 1 if found
    """
    if len(sys.argv) < 2:
        print("Usage: validate_type_imports.py <file1.py> [file2.py ...]", file=sys.stderr)
        return 1
        
    total_violations = 0
    
    for file_arg in sys.argv[1:]:
        file_path = Path(file_arg)
        
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist", file=sys.stderr)
            continue
            
        if not file_path.suffix == '.py':
            continue
            
        violations = analyze_type_imports(file_path)
        
        if violations:
            total_violations += len(violations)
            print(f"\n❌ TYPE IMPORT VIOLATIONS in {file_path}:")
            
            for line_num, violation, suggestion in violations:
                print(f"  Line {line_num}: {violation}")
                print(generate_type_import_remediation(violation, suggestion))
                
    if total_violations > 0:
        print(f"\n🚨 TOTAL VIOLATIONS: {total_violations} type import violation(s) found")
        print("❌ PRE-COMMIT HOOK FAILED: Please fix type import issues before committing")
        return 1
    else:
        print("✅ No type import violations detected. Good job!")
        return 0


if __name__ == "__main__":
    sys.exit(main())