# comprehensive_audit.py
# Static audit:
# 1) Find unguarded, top-level imports of heavy libs in startup-critical files
# 2) Detect circular dependencies among internal modules
#
# No external packages required. Safe in PyCharm, CLI, and IPython/Jupyter.

import ast
import os
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, DefaultDict
from collections import defaultdict
import sys

# ---------------- CONFIGURATION ----------------

HEAVY_DEPENDENCIES: Set[str] = {
    "numpy", "scipy", "pandas", "torch", "sklearn", "transformers",
    "sentence_transformers", "faiss", "tensorflow", "jax", "matplotlib",
    "seaborn", "plotly"
}

STARTUP_CRITICAL_PATTERNS: List[str] = [
    "*server.py", "*app.py", "*wsgi.py", "*asgi.py", "main.py",
    "*coordinator.py", "*manager.py", "*builder.py", "*registry.py",
    "*entrypoint.py", "validate_installation.py", "settings.py", "config.py"
]

IGNORED_DIRECTORIES: Set[str] = {
    ".venv", "venv", "env", ".git", "__pycache__", ".pytest_cache",
    "build", "dist", "docs", "node_modules", ".idea", ".vscode"
}

# ---------------- IMPLEMENTATION ----------------

class ImportVisitor(ast.NodeVisitor):
    """Collect imports and flag unguarded heavy imports at module top level."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.issues: List[Dict[str, Any]] = []
        self.imports: Set[str] = set()
        self._stack: List[ast.AST] = []

    def visit(self, node: ast.AST):
        self._stack.append(node)
        super().visit(node)
        self._stack.pop()

    def _is_guarded(self) -> bool:
        return any(isinstance(n, ast.Try) for n in self._stack[:-1])

    def _is_top_level(self) -> bool:
        return len(self._stack) > 1 and isinstance(self._stack[-2], ast.Module)

    def _record_issue(self, node: ast.AST, msg: str):
        try:
            stmt = ast.unparse(node).strip()
        except Exception:
            stmt = "<unparseable>"
        self.issues.append({
            "file": self.file_path,
            "line": getattr(node, "lineno", 0),
            "type": "Unguarded Heavy Import",
            "message": msg,
            "statement": stmt
        })

    def _check_heavy(self, module_name: str, node: ast.AST):
        root = (module_name or "").split(".")[0]
        if root in HEAVY_DEPENDENCIES and self._is_top_level() and not self._is_guarded():
            self._record_issue(node, f"Unguarded, top-level import of heavy dependency '{root}'")

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
            self._check_heavy(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        # Represent relative imports like '..utils' as dotted prefix
        if node.level and mod:
            dotted = "." * node.level + mod
        elif node.level and not mod:
            dotted = "." * node.level
        else:
            dotted = mod
        if dotted:
            self.imports.add(dotted)
            self._check_heavy(mod, node)
        self.generic_visit(node)


class ComprehensiveAuditor:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.heavy_import_issues: List[Dict[str, Any]] = []
        self.circular_cycles: List[List[str]] = []
        self.module_deps: Dict[str, Set[str]] = {}
        self.internal_modules: Set[str] = set()
        self.startup_critical: Set[Path] = set()
        self.all_py_files: List[Path] = []

    def run(self):
        print("=" * 80)
        print("PART 1: STATIC CODE AUDIT (Imports + Cycles)")
        print("=" * 80)

        self.all_py_files = self._find_python_files()
        self.internal_modules = self._discover_internal_modules(self.all_py_files)
        self.startup_critical = {p for p in self.all_py_files if self._is_startup_critical(p)}

        print(f"📄 Python files found: {len(self.all_py_files)}")
        print(f"🏷️  Internal modules discovered: {len(self.internal_modules)}")
        print(f"🚀 Startup-critical files: {len(self.startup_critical)}")

        for file_path in self.all_py_files:
            self._analyze_file(file_path)

        self._detect_circular_dependencies()
        self._print_report()

    def _find_python_files(self) -> List[Path]:
        files: List[Path] = []
        for root, dirs, filenames in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]
            for fname in filenames:
                if fname.endswith(".py"):
                    files.append(Path(root) / fname)
        return files

    def _module_name_from_path(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.project_root)
        parts = list(rel.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        return ".".join(p for p in parts if p and p != "__init__")

    def _discover_internal_modules(self, files: List[Path]) -> Set[str]:
        return {self._module_name_from_path(p) for p in files}

    def _is_startup_critical(self, file_path: Path) -> bool:
        return any(fnmatch.fnmatch(file_path.name, pat) for pat in STARTUP_CRITICAL_PATTERNS)

    def _normalize_import(self, importer: str, imported: str) -> Optional[str]:
        """Normalize imported name to internal module if applicable."""
        if not imported:
            return None

        # Handle relative import like '..utils.helpers'
        if imported.startswith("."):
            level = len(imported) - len(imported.lstrip("."))
            tail = imported[level:]  # strip leading dots
            importer_parts = importer.split(".")
            base_parts = importer_parts[:-level] if level <= len(importer_parts) else []
            norm_parts = base_parts + ([p for p in tail.split(".") if p] if tail else [])
            cand = ".".join(norm_parts).strip(".")
        else:
            cand = imported

        # Reduce prefixes to see if any matches an internal module
        segments = cand.split(".")
        for i in range(len(segments), 0, -1):
            prefix = ".".join(segments[:i])
            if prefix in self.internal_modules:
                return prefix
        return None

    def _analyze_file(self, file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src, filename=str(file_path))
        except Exception as e:
            print(f"⚠️  Could not parse {file_path}: {e}")
            return

        visitor = ImportVisitor(file_path)
        visitor.visit(tree)

        mod_name = self._module_name_from_path(file_path)
        deps: Set[str] = set()
        for imp in visitor.imports:
            norm = self._normalize_import(mod_name, imp)
            if norm:
                deps.add(norm)
        self.module_deps[mod_name] = deps

        if file_path in self.startup_critical:
            self.heavy_import_issues.extend(visitor.issues)

    def _detect_circular_dependencies(self):
        graph = self.module_deps
        temp_stack: Set[str] = set()
        visited: Set[str] = set()
        cycle_set: Set[tuple] = set()

        def dfs(node: str, path: List[str]):
            if node in temp_stack:
                # cycle found
                if node in path:
                    i = path.index(node)
                    cycle = path[i:] + [node]
                else:
                    cycle = path + [node]
                core = cycle[:-1]
                rotations = [tuple(core[j:] + core[:j]) for j in range(len(core))]
                canon = min(rotations)
                cycle_set.add(canon)
                return

            if node in visited:
                return

            temp_stack.add(node)
            visited.add(node)
            for nb in graph.get(node, set()):
                dfs(nb, path + [node])
            temp_stack.remove(node)

        for n in list(graph.keys()):
            if n not in visited:
                dfs(n, [])

        self.circular_cycles = [list(c) + [c[0]] for c in cycle_set]

    def _print_report(self):
        print("\n--- STATIC AUDIT REPORT ---")

        if self.heavy_import_issues:
            print(f"\n🚨 Unguarded heavy imports in startup-critical files: {len(self.heavy_import_issues)}")
            by_file: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
            for iss in self.heavy_import_issues:
                rel = iss["file"].resolve().relative_to(self.project_root)
                by_file[str(rel)].append(iss)
            for file_rel, issues in by_file.items():
                print(f"  📄 {file_rel}")
                for iss in sorted(issues, key=lambda x: x["line"]):
                    print(f"    - L{iss['line']}: {iss['statement']}")
        else:
            print("\n✅ No unguarded heavy imports found in startup-critical files.")

        if self.circular_cycles:
            print(f"\n🚨 Circular dependency cycles detected: {len(self.circular_cycles)}")
            for i, cyc in enumerate(sorted(self.circular_cycles), 1):
                print(f"  #{i}: " + " -> ".join(cyc))
        else:
            print("\n✅ No circular dependencies detected.")

        print("\n✅ Static analysis complete.")

def _resolve_project_root() -> Path:
    try:
        return Path(__file__).parent.resolve()
    except NameError:
        # IPython/Jupyter: __file__ not defined
        print("⚠️  '__file__' not defined. Using current working directory as project root.")
        return Path.cwd().resolve()

if __name__ == "__main__":
    # Optional argument to override project root
    project_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _resolve_project_root()
    auditor = ComprehensiveAuditor(project_root)
    auditor.run()
