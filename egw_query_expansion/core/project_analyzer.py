#!/usr/bin/env python3
"""
Project Structure Analyzer
Analyzes a software project to identify modules, dependencies, missing connections,
and provides actionable insights for project compilation and implementation.
"""

import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class ProjectAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.modules = {}
        self.imports = defaultdict(set)
        self.exports = defaultdict(set)
        self.missing_imports = []
        self.file_types = defaultdict(list)
        self.config_files = {}
        self.dependencies = {}
        self.api_endpoints = []
        self.database_schemas = []
        self.environment_vars = set()
        self.todos = []
        self.errors = []

    def analyze(self):
        """Main analysis function"""
        print("🔍 Starting Project Analysis...")
        print("=" * 60)

        self.scan_directory_structure()
        self.analyze_package_files()
        self.analyze_source_files()
        self.analyze_config_files()
        self.check_environment_setup()
        self.analyze_documentation()
        self.generate_report()

    def scan_directory_structure(self):
        """Scan and categorize all files in the project"""
        print("\n📁 Scanning directory structure...")

        for root, dirs, files in os.walk(self.project_root):
            # Skip common directories to ignore
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {
                    ".git",
                    "node_modules",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "env",
                    ".env",
                    "dist",
                    "build",
                }
            ]

            rel_path = Path(root).relative_to(self.project_root)

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                self.file_types[ext].append(str(rel_path / file))

                # Identify module type based on file extension and location
                if ext in [
                    ".py",
                    ".js",
                    ".ts",
                    ".jsx",
                    ".tsx",
                    ".java",
                    ".cpp",
                    ".c",
                    ".go",
                    ".rs",
                ]:
                    self.modules[str(rel_path / file)] = {
                        "type": self.get_module_type(file_path),
                        "language": self.get_language(ext),
                        "size": os.path.getsize(file_path),
                    }

    def get_module_type(self, file_path: Path) -> str:
        """Determine module type based on path and content"""
        path_str = str(file_path).lower()

        if "test" in path_str or "spec" in path_str:
            return "test"
        elif "model" in path_str or "schema" in path_str:
            return "model"
        elif "controller" in path_str or "handler" in path_str or "route" in path_str:
            return "controller"
        elif "service" in path_str or "business" in path_str:
            return "service"
        elif "util" in path_str or "helper" in path_str:
            return "utility"
        elif "config" in path_str or "setting" in path_str:
            return "config"
        elif "component" in path_str or "view" in path_str or "page" in path_str:
            return "ui"
        elif "api" in path_str or "endpoint" in path_str:
            return "api"
        else:
            return "general"

    def get_language(self, ext: str) -> str:
        """Map file extension to language"""
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript-react",
            ".tsx": "typescript-react",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
        }
        return lang_map.get(ext, "unknown")

    def analyze_package_files(self):
        """Analyze package.json, requirements.txt, pom.xml, etc."""
        print("\n📦 Analyzing package files...")

        # Python projects
        req_files = [
            "requirements.txt",
            "requirements.in",
            "Pipfile",
            "pyproject.toml",
            "setup.py",
        ]
        for req_file in req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                self.analyze_python_deps(file_path)

        # Node.js projects
        package_json = self.project_root / "package.json"
        if package_json.exists():
            self.analyze_node_deps(package_json)

        # Java projects
        pom_xml = self.project_root / "pom.xml"
        if pom_xml.exists():
            self.analyze_maven_deps(pom_xml)

        gradle_file = self.project_root / "build.gradle"
        if gradle_file.exists():
            self.analyze_gradle_deps(gradle_file)

    def analyze_python_deps(self, file_path: Path):
        """Analyze Python dependencies"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            if file_path.name == "requirements.txt":
                deps = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        deps.append(line)
                self.dependencies["python"] = deps

            elif file_path.name == "pyproject.toml":
                # Basic TOML parsing for dependencies
                import toml

                data = toml.load(file_path)
                if "tool" in data and "poetry" in data["tool"]:
                    self.dependencies["python"] = list(
                        data["tool"]["poetry"].get("dependencies", {}).keys()
                    )
                elif "project" in data:
                    self.dependencies["python"] = data["project"].get(
                        "dependencies", []
                    )

        except Exception as e:
            self.errors.append(f"Error analyzing {file_path}: {e}")

    def analyze_node_deps(self, file_path: Path):
        """Analyze Node.js dependencies"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            deps = []
            if "dependencies" in data:
                deps.extend(data["dependencies"].keys())
            if "devDependencies" in data:
                deps.extend(data["devDependencies"].keys())

            self.dependencies["node"] = deps
            self.config_files["package.json"] = data

        except Exception as e:
            self.errors.append(f"Error analyzing {file_path}: {e}")

    def analyze_maven_deps(self, file_path: Path):
        """Analyze Maven dependencies"""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            deps = []
            for dep in root.findall(".//{http://maven.apache.org/POM/4.0.0}dependency"):
                group_id = dep.find("{http://maven.apache.org/POM/4.0.0}groupId")
                artifact_id = dep.find("{http://maven.apache.org/POM/4.0.0}artifactId")
                if group_id is not None and artifact_id is not None:
                    deps.append(f"{group_id.text}:{artifact_id.text}")

            self.dependencies["maven"] = deps

        except Exception as e:
            self.errors.append(f"Error analyzing {file_path}: {e}")

    def analyze_gradle_deps(self, file_path: Path):
        """Analyze Gradle dependencies"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Basic regex to find dependencies
            deps = re.findall(
                r"(?:implementation|compile|api|testImplementation)\s+['\"]([^'\"]+)['\"]",
                content,
            )
            self.dependencies["gradle"] = deps

        except Exception as e:
            self.errors.append(f"Error analyzing {file_path}: {e}")

    def analyze_source_files(self):
        """Analyze source code files for imports, exports, and connections"""
        print("\n🔗 Analyzing source code connections...")

        for module_path, module_info in self.modules.items():
            file_path = self.project_root / module_path

            if module_info["language"].startswith("python"):
                self.analyze_python_file(file_path, module_path)
            elif module_info["language"].startswith("javascript") or module_info[
                "language"
            ].startswith("typescript"):
                self.analyze_js_file(file_path, module_path)
            elif module_info["language"] == "java":
                self.analyze_java_file(file_path, module_path)

    def analyze_python_file(self, file_path: Path, module_path: str):
        """Analyze Python file for imports and definitions"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            for node in ast.walk(tree):
                # Find imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports[module_path].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.imports[module_path].add(node.module)

                # Find class and function definitions (exports)
                elif isinstance(node, ast.ClassDef):
                    self.exports[module_path].add(f"class:{node.name}")
                elif isinstance(node, ast.FunctionDef):
                    self.exports[module_path].add(f"function:{node.name}")

                    # Check for API endpoints (Flask/FastAPI patterns)
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if hasattr(decorator.func, "attr"):
                                if decorator.func.attr in [
                                    "route",
                                    "get",
                                    "post",
                                    "put",
                                    "delete",
                                    "patch",
                                ]:
                                    self.api_endpoints.append(
                                        {
                                            "file": module_path,
                                            "function": node.name,
                                            "method": decorator.func.attr,
                                        }
                                    )

            # Find TODOs and FIXMEs
            todos = re.findall(r"#\s*(TODO|FIXME|XXX|HACK|NOTE):\s*(.+)", content)
            for todo_type, todo_text in todos:
                self.todos.append(
                    {"file": module_path, "type": todo_type, "text": todo_text.strip()}
                )

            # Find environment variables
            env_vars = re.findall(r'os\.environ\.get\([\'"]([^\'"]+)[\'"]', content)
            self.environment_vars.update(env_vars)
            env_vars = re.findall(r'os\.getenv\([\'"]([^\'"]+)[\'"]', content)
            self.environment_vars.update(env_vars)

        except Exception as e:
            self.errors.append(f"Error analyzing Python file {file_path}: {e}")

    def analyze_js_file(self, file_path: Path, module_path: str):
        """Analyze JavaScript/TypeScript file for imports and exports"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find imports
            imports = re.findall(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]', content)
            imports.extend(re.findall(r'require\([\'"]([^\'"]+)[\'"]', content))
            self.imports[module_path].update(imports)

            # Find exports
            exports = re.findall(
                r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)",
                content,
            )
            self.exports[module_path].update(exports)

            # Find API endpoints (Express patterns)
            endpoints = re.findall(
                r'app\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]', content
            )
            for method, path in endpoints:
                self.api_endpoints.append(
                    {"file": module_path, "method": method, "path": path}
                )

            # Find TODOs
            todos = re.findall(r"//\s*(TODO|FIXME|XXX|HACK|NOTE):\s*(.+)", content)
            for todo_type, todo_text in todos:
                self.todos.append(
                    {"file": module_path, "type": todo_type, "text": todo_text.strip()}
                )

            # Find environment variables
            env_vars = re.findall(r"process\.env\.(\w+)", content)
            self.environment_vars.update(env_vars)

        except Exception as e:
            self.errors.append(f"Error analyzing JS file {file_path}: {e}")

    def analyze_java_file(self, file_path: Path, module_path: str):
        """Analyze Java file for imports and class definitions"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find imports
            imports = re.findall(r"import\s+([\w.]+);", content)
            self.imports[module_path].update(imports)

            # Find class definitions
            classes = re.findall(r"(?:public\s+)?class\s+(\w+)", content)
            for cls in classes:
                self.exports[module_path].add(f"class:{cls}")

            # Find Spring REST endpoints
            endpoints = re.findall(
                r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping)\([\'"]([^\'"]+)[\'"]',
                content,
            )
            for method, path in endpoints:
                self.api_endpoints.append(
                    {
                        "file": module_path,
                        "method": method.replace("Mapping", "").lower(),
                        "path": path,
                    }
                )

        except Exception as e:
            self.errors.append(f"Error analyzing Java file {file_path}: {e}")

    def analyze_config_files(self):
        """Analyze configuration files"""
        print("\n⚙️ Analyzing configuration files...")

        config_patterns = {
            ".env": self.analyze_env_file,
            "docker-compose.yml": self.analyze_docker_compose,
            "docker-compose.yaml": self.analyze_docker_compose,
            "Dockerfile": self.analyze_dockerfile,
            ".gitignore": lambda x: None,  # Just note existence
            "Makefile": self.analyze_makefile,
            "webpack.config.js": lambda x: None,
            "tsconfig.json": self.analyze_tsconfig,
            "jest.config.js": lambda x: None,
            ".eslintrc.json": lambda x: None,
            ".prettierrc": lambda x: None,
        }

        for pattern, analyzer in config_patterns.items():
            file_path = self.project_root / pattern
            if file_path.exists():
                self.config_files[pattern] = True
                if analyzer:
                    analyzer(file_path)

    def analyze_env_file(self, file_path: Path):
        """Analyze .env file"""
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key = line.split("=")[0].strip()
                        self.environment_vars.add(key)
        except Exception as e:
            self.errors.append(f"Error analyzing .env file: {e}")

    def analyze_docker_compose(self, file_path: Path):
        """Analyze docker-compose file"""
        try:
            import yaml

            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            if "services" in data:
                services = list(data["services"].keys())
                self.config_files["docker_services"] = services

        except Exception as e:
            self.errors.append(f"Error analyzing docker-compose: {e}")

    def analyze_dockerfile(self, file_path: Path):
        """Analyze Dockerfile"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Find base image
            base_image = re.search(r"FROM\s+([^\s]+)", content)
            if base_image:
                self.config_files["docker_base_image"] = base_image.group(1)

        except Exception as e:
            self.errors.append(f"Error analyzing Dockerfile: {e}")

    def analyze_makefile(self, file_path: Path):
        """Analyze Makefile for targets"""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Find targets
            targets = re.findall(r"^([a-zA-Z_-]+):", content, re.MULTILINE)
            self.config_files["makefile_targets"] = targets

        except Exception as e:
            self.errors.append(f"Error analyzing Makefile: {e}")

    def analyze_tsconfig(self, file_path: Path):
        """Analyze TypeScript configuration"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            self.config_files["typescript"] = {
                "target": data.get("compilerOptions", {}).get("target"),
                "module": data.get("compilerOptions", {}).get("module"),
                "strict": data.get("compilerOptions", {}).get("strict", False),
            }

        except Exception as e:
            self.errors.append(f"Error analyzing tsconfig: {e}")

    def check_environment_setup(self):
        """Check for missing environment variables and setup issues"""
        print("\n🔍 Checking environment setup...")

        # Check if required env vars are defined
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"

        if env_example.exists() and not env_file.exists():
            self.errors.append("⚠️ .env.example exists but .env is missing")

        # Check for missing imports (imports that don't match any exports)
        for module, imports in self.imports.items():
            for imp in imports:
                # Check if it's an internal import
                if not imp.startswith("."):
                    continue

                # Try to resolve internal import
                found = False
                for exp_module, exports in self.exports.items():
                    if imp in str(exp_module):
                        found = True
                        break

                if not found and not self.is_external_package(imp):
                    self.missing_imports.append({"module": module, "import": imp})

    def is_external_package(self, import_name: str) -> bool:
        """Check if an import is an external package"""
        # Check in dependencies
        for lang, deps in self.dependencies.items():
            if import_name in deps or any(
                import_name.startswith(dep.split("==")[0]) for dep in deps
            ):
                return True

        # Common standard library modules
        python_stdlib = {
            "os",
            "sys",
            "json",
            "datetime",
            "collections",
            "itertools",
            "re",
            "math",
            "random",
        }
        js_builtin = {
            "fs",
            "path",
            "http",
            "https",
            "crypto",
            "util",
            "stream",
            "events",
        }

        return import_name in python_stdlib or import_name in js_builtin

    def analyze_documentation(self):
        """Check for documentation files"""
        print("\n📚 Checking documentation...")

        doc_files = [
            "README.md",
            "README.rst",
            "README.txt",
            "CONTRIBUTING.md",
            "LICENSE",
            "CHANGELOG.md",
            "docs/",
        ]

        for doc in doc_files:
            doc_path = self.project_root / doc
            if doc_path.exists():
                self.config_files[f"doc_{doc}"] = True

    def check_compilation_readiness(self) -> Dict[str, Any]:
        """Check if project is ready to compile/run"""
        issues = []
        ready = True

        # Check for main entry point
        entry_points = [
            "main.py",
            "app.py",
            "index.js",
            "index.ts",
            "Main.java",
            "main.go",
        ]
        has_entry = any((self.project_root / ep).exists() for ep in entry_points)

        if not has_entry:
            issues.append(
                "No clear entry point found (main.py, app.py, index.js, etc.)"
            )
            ready = False

        # Check for package management files
        if not self.dependencies:
            issues.append("No dependency management file found")
            ready = False

        # Check for missing imports
        if self.missing_imports:
            issues.append(f"Found {len(self.missing_imports)} missing imports")
            ready = False

        # Check for configuration
        if not self.config_files:
            issues.append("No configuration files found")

        # Check for tests
        has_tests = any("test" in str(m).lower() for m in self.modules.keys())
        if not has_tests:
            issues.append("No test files found")

        return {
            "ready": ready,
            "issues": issues,
            "has_entry_point": has_entry,
            "has_dependencies": bool(self.dependencies),
            "has_tests": has_tests,
            "has_docker": "Dockerfile" in self.config_files,
            "has_ci": any(
                ci in self.file_types
                for ci in [".github", ".gitlab-ci.yml", ".travis.yml"]
            ),
        }

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 60)
        print("📊 PROJECT ANALYSIS REPORT")
        print("=" * 60)

        # Project Overview
        print("\n🏗️ PROJECT OVERVIEW:")
        print(f"  Root: {self.project_root}")
        print(f"  Total Files: {sum(len(files) for files in self.file_types.values())}")
        print(f"  Modules: {len(self.modules)}")

        # Language Distribution
        print("\n💻 LANGUAGE DISTRIBUTION:")
        lang_counts = defaultdict(int)
        for module in self.modules.values():
            lang_counts[module["language"]] += 1

        for lang, count in sorted(
            lang_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {lang}: {count} files")

        # Module Types
        print("\n📦 MODULE TYPES:")
        type_counts = defaultdict(int)
        for module in self.modules.values():
            type_counts[module["type"]] += 1

        for mtype, count in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {mtype}: {count} modules")

        # Dependencies
        if self.dependencies:
            print("\n📚 DEPENDENCIES:")
            for lang, deps in self.dependencies.items():
                print(f"  {lang}: {len(deps)} packages")
                if len(deps) <= 5:
                    for dep in deps[:5]:
                        print(f"    - {dep}")

        # API Endpoints
        if self.api_endpoints:
            print(f"\n🌐 API ENDPOINTS: {len(self.api_endpoints)} found")
            for ep in self.api_endpoints[:5]:
                print(
                    f"  {ep['method'].upper()}: {ep.get('path', 'N/A')} ({ep['file']})"
                )

        # Configuration
        print("\n⚙️ CONFIGURATION FILES:")
        for config, value in self.config_files.items():
            if isinstance(value, bool) and value:
                print(f"  ✓ {config}")
            elif isinstance(value, list):
                print(f"  ✓ {config}: {len(value)} items")

        # Environment Variables
        if self.environment_vars:
            print(f"\n🔐 ENVIRONMENT VARIABLES: {len(self.environment_vars)} found")
            for var in sorted(list(self.environment_vars)[:10]):
                print(f"  - {var}")

        # Missing Imports
        if self.missing_imports:
            print(f"\n⚠️ MISSING IMPORTS: {len(self.missing_imports)}")
            for miss in self.missing_imports[:5]:
                print(f"  {miss['module']}: {miss['import']}")

        # TODOs
        if self.todos:
            print(f"\n📝 TODOs/FIXMEs: {len(self.todos)} found")
            for todo in self.todos[:5]:
                print(f"  [{todo['type']}] {todo['file']}: {todo['text'][:50]}...")

        # Compilation Readiness
        readiness = self.check_compilation_readiness()
        print("\n✅ COMPILATION READINESS:")
        print(f"  Status: {'READY' if readiness['ready'] else 'NOT READY'}")
        print(f"  Entry Point: {'✓' if readiness['has_entry_point'] else '✗'}")
        print(f"  Dependencies: {'✓' if readiness['has_dependencies'] else '✗'}")
        print(f"  Tests: {'✓' if readiness['has_tests'] else '✗'}")
        print(f"  Docker: {'✓' if readiness['has_docker'] else '✗'}")
        print(f"  CI/CD: {'✓' if readiness['has_ci'] else '✗'}")

        if readiness["issues"]:
            print("\n  Issues to Address:")
            for issue in readiness["issues"]:
                print(f"    - {issue}")

        # Errors
        if self.errors:
            print(f"\n❌ ERRORS ENCOUNTERED: {len(self.errors)}")
            for error in self.errors[:5]:
                print(f"  - {error}")

        # Next Steps
        print("\n🚀 RECOMMENDED NEXT STEPS:")
        steps = self.generate_next_steps(readiness)
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

        # Export to JSON
        self.export_json_report()

    def generate_next_steps(self, readiness: Dict) -> List[str]:
        """Generate actionable next steps based on analysis"""
        steps = []

        if not readiness["has_entry_point"]:
            steps.append("Create a main entry point file (main.py, index.js, etc.)")

        if not readiness["has_dependencies"]:
            steps.append(
                "Create a dependency management file (requirements.txt, package.json, etc.)"
            )

        if self.missing_imports:
            steps.append(f"Resolve {len(self.missing_imports)} missing imports")

        if not self.config_files.get(".env") and self.environment_vars:
            steps.append("Create .env file with required environment variables")

        if not readiness["has_tests"]:
            steps.append("Add unit tests for core modules")

        if not readiness["has_docker"] and len(self.modules) > 10:
            steps.append("Consider adding Docker configuration for deployment")

        if not self.config_files.get("doc_README.md"):
            steps.append("Create README.md with project documentation")

        if self.todos:
            steps.append(f"Address {len(self.todos)} TODO/FIXME comments in code")

        if "python" in self.dependencies and not self.config_files.get(".gitignore"):
            steps.append("Add .gitignore file to exclude unnecessary files")

        if readiness["ready"]:
            steps.append("Run build/compilation command for your project type")
            if "python" in self.dependencies:
                steps.append("Run: pip install -r requirements.txt && python main.py")
            elif "node" in self.dependencies:
                steps.append("Run: npm install && npm start")

        return steps

    def export_json_report(self):
        """Export analysis results to JSON file"""
        report = {
            "project_root": str(self.project_root),
            "summary": {
                "total_files": sum(len(files) for files in self.file_types.values()),
                "total_modules": len(self.modules),
                "languages": dict(self.count_languages()),
                "module_types": dict(self.count_module_types()),
            },
            "modules": self.modules,
            "dependencies": self.dependencies,
            "imports": {k: list(v) for k, v in self.imports.items()},
            "exports": {k: list(v) for k, v in self.exports.items()},
            "missing_imports": self.missing_imports,
            "api_endpoints": self.api_endpoints,
            "environment_variables": list(self.environment_vars),
            "todos": self.todos,
            "config_files": self.config_files,
            "errors": self.errors,
            "readiness": self.check_compilation_readiness(),
        }

        output_file = self.project_root / "project_analysis_report.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Full report exported to: {output_file}")

    def count_languages(self):
        lang_counts = defaultdict(int)
        for module in self.modules.values():
            lang_counts[module["language"]] += 1
        return lang_counts

    def count_module_types(self):
        type_counts = defaultdict(int)
        for module in self.modules.values():
            type_counts[module["type"]] += 1
        return type_counts


def main():
    """Main function to run the analyzer"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze project structure and provide actionable insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_analyzer.py                    # Analyze current directory
  python project_analyzer.py /path/to/project   # Analyze specific project
  python project_analyzer.py --verbose          # Show detailed output
        """,
    )

    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to project root directory (default: current directory)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including all files",
    )

    parser.add_argument(
        "--output", "-o", help="Output report to specified file (JSON format)"
    )

    parser.add_argument(
        "--ignore",
        nargs="+",
        default=[],
        help="Additional directories to ignore during scan",
    )

    args = parser.parse_args()

    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"❌ Error: Project path '{project_path}' does not exist")
        sys.exit(1)

    if not project_path.is_dir():
        print(f"❌ Error: '{project_path}' is not a directory")
        sys.exit(1)

    print(f"🚀 Analyzing project at: {project_path}")
    print("=" * 60)

    # Run analyzer
    analyzer = ProjectAnalyzer(str(project_path))

    # Add custom ignores if specified
    if args.ignore:
        print(f"📝 Ignoring additional directories: {', '.join(args.ignore)}")

    try:
        analyzer.analyze()

        # Custom output file if specified
        if args.output:
            output_path = Path(args.output)
            report = {
                "project_root": str(analyzer.project_root),
                "timestamp": str(Path.ctime(Path.cwd())),
                "summary": {
                    "total_files": sum(
                        len(files) for files in analyzer.file_types.values()
                    ),
                    "total_modules": len(analyzer.modules),
                    "languages": dict(analyzer.count_languages()),
                    "module_types": dict(analyzer.count_module_types()),
                },
                "modules": analyzer.modules,
                "dependencies": analyzer.dependencies,
                "imports": {k: list(v) for k, v in analyzer.imports.items()},
                "exports": {k: list(v) for k, v in analyzer.exports.items()},
                "missing_imports": analyzer.missing_imports,
                "api_endpoints": analyzer.api_endpoints,
                "environment_variables": list(analyzer.environment_vars),
                "todos": analyzer.todos,
                "config_files": analyzer.config_files,
                "errors": analyzer.errors,
                "readiness": analyzer.check_compilation_readiness(),
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            print(f"\n💾 Custom report saved to: {output_path}")

        print("\n✨ Analysis complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during analysis: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
