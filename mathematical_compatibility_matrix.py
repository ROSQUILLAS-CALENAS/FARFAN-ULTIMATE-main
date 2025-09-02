"""
Mathematical Compatibility Matrix Validator

This module provides comprehensive version constraint validation using semantic
versioning and mathematical set theory operations to detect incompatible package
combinations and version conflicts.
"""

import json
import re
import sys
import subprocess
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
from pathlib import Path


class ConflictSeverity(Enum):
    """Severity levels for package conflicts"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VersionOperator(Enum):
    """Semantic version operators"""
    EQ = "=="
    GE = ">="
    LE = "<="
    GT = ">"
    LT = "<"
    NE = "!="
    COMPATIBLE = "~="
    ARBITRARY = "==="


@dataclass
class VersionConstraint:
    """Represents a semantic version constraint"""
    operator: VersionOperator
    version: str
    
    def __post_init__(self):
        self.major, self.minor, self.patch = self._parse_version(self.version)
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string into components"""
        # Remove any pre-release/build metadata
        clean_version = re.split(r'[+\-]', version)[0]
        parts = clean_version.split('.')
        
        # Pad with zeros if needed
        while len(parts) < 3:
            parts.append('0')
        
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError as e:
            raise ValueError(f"Invalid version format: {version}") from e
    
    def satisfies(self, version: str) -> bool:
        """Check if a version satisfies this constraint"""
        try:
            v_major, v_minor, v_patch = self._parse_version(version)
            constraint_version = (self.major, self.minor, self.patch)
            test_version = (v_major, v_minor, v_patch)
            
            if self.operator == VersionOperator.EQ:
                return test_version == constraint_version
            elif self.operator == VersionOperator.GE:
                return test_version >= constraint_version
            elif self.operator == VersionOperator.LE:
                return test_version <= constraint_version
            elif self.operator == VersionOperator.GT:
                return test_version > constraint_version
            elif self.operator == VersionOperator.LT:
                return test_version < constraint_version
            elif self.operator == VersionOperator.NE:
                return test_version != constraint_version
            elif self.operator == VersionOperator.COMPATIBLE:
                # Compatible release (~=): allows patch-level changes
                return (test_version[:2] == constraint_version[:2] and 
                       test_version >= constraint_version)
            elif self.operator == VersionOperator.ARBITRARY:
                return version == self.version
            else:
                return False
        except ValueError:
            return False


@dataclass
class PackageInfo:
    """Information about an installed package"""
    name: str
    version: str
    location: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ConflictReport:
    """Report of a detected package conflict"""
    severity: ConflictSeverity
    conflict_type: str
    packages: List[str]
    message: str
    remediation: str
    python_version: str


class CompatibilityMatrixValidator:
    """Validates package compatibility using the version matrix"""
    
    def __init__(self, matrix_path: str = "version_compatibility_matrix.json"):
        self.matrix_path = Path(matrix_path)
        self.matrix_data = self._load_matrix()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.installed_packages = self._get_installed_packages()
    
    def _load_matrix(self) -> Dict[str, Any]:
        """Load the compatibility matrix from JSON"""
        try:
            with open(self.matrix_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Compatibility matrix not found: {self.matrix_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in compatibility matrix: {e}")
    
    def _get_installed_packages(self) -> Dict[str, PackageInfo]:
        """Get list of currently installed packages"""
        packages = {}
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'list', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            pip_list = json.loads(result.stdout)
            for pkg in pip_list:
                packages[pkg['name'].lower()] = PackageInfo(
                    name=pkg['name'],
                    version=pkg['version']
                )
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Warning: Could not retrieve installed packages: {e}")
        
        return packages
    
    def _parse_version_spec(self, spec: str) -> List[VersionConstraint]:
        """Parse a version specification string into constraints"""
        constraints = []
        
        # Handle multiple constraints separated by commas
        for constraint_str in spec.split(','):
            constraint_str = constraint_str.strip()
            
            # Match operator and version
            match = re.match(r'([><=!~]+)([0-9]+(?:\.[0-9]+)*(?:[-+][a-zA-Z0-9.]+)?)', constraint_str)
            if match:
                op_str, version = match.groups()
                
                # Map operator string to enum
                operator_map = {
                    '==': VersionOperator.EQ,
                    '>=': VersionOperator.GE,
                    '<=': VersionOperator.LE,
                    '>': VersionOperator.GT,
                    '<': VersionOperator.LT,
                    '!=': VersionOperator.NE,
                    '~=': VersionOperator.COMPATIBLE,
                    '===': VersionOperator.ARBITRARY
                }
                
                operator = operator_map.get(op_str)
                if operator:
                    constraints.append(VersionConstraint(operator, version))
        
        return constraints
    
    def _get_python_version_spec(self) -> Optional[Dict[str, Any]]:
        """Get the specification for the current Python version"""
        return self.matrix_data.get('python_versions', {}).get(self.python_version)
    
    def _check_faiss_conflicts(self) -> List[ConflictReport]:
        """Check for FAISS CPU/GPU conflicts"""
        conflicts = []
        faiss_packages = ['faiss-cpu', 'faiss-gpu', 'faiss']
        installed_faiss = [pkg for pkg in faiss_packages 
                          if pkg.lower() in self.installed_packages]
        
        if len(installed_faiss) > 1:
            conflicts.append(ConflictReport(
                severity=ConflictSeverity.CRITICAL,
                conflict_type="faiss_mutual_exclusion",
                packages=installed_faiss,
                message=f"Multiple FAISS variants detected: {', '.join(installed_faiss)}. "
                       "Only one FAISS variant should be installed.",
                remediation=f"Uninstall conflicting packages: pip uninstall {' '.join(installed_faiss[1:])}",
                python_version=self.python_version
            ))
        
        return conflicts
    
    def _check_version_constraints(self) -> List[ConflictReport]:
        """Check installed packages against version constraints"""
        conflicts = []
        python_spec = self._get_python_version_spec()
        
        if not python_spec:
            conflicts.append(ConflictReport(
                severity=ConflictSeverity.HIGH,
                conflict_type="unsupported_python_version",
                packages=[],
                message=f"Python {self.python_version} is not explicitly supported",
                remediation=f"Use a supported Python version: {', '.join(self.matrix_data['python_versions'].keys())}",
                python_version=self.python_version
            ))
            return conflicts
        
        # Check each package category
        for category_name, category_packages in python_spec.items():
            if category_name in ['status', 'excluded_packages', 'exclusion_reasons']:
                continue
            
            if isinstance(category_packages, dict):
                for package_name, version_spec in category_packages.items():
                    if isinstance(version_spec, str):
                        self._validate_package_version(
                            package_name, version_spec, conflicts
                        )
                    elif isinstance(version_spec, dict):
                        # Handle special cases like FAISS variants
                        self._validate_complex_package(
                            package_name, version_spec, conflicts
                        )
        
        return conflicts
    
    def _validate_package_version(self, package_name: str, version_spec: str, 
                                 conflicts: List[ConflictReport]):
        """Validate a single package version against constraints"""
        installed_pkg = self.installed_packages.get(package_name.lower())
        
        if not installed_pkg:
            conflicts.append(ConflictReport(
                severity=ConflictSeverity.MEDIUM,
                conflict_type="missing_package",
                packages=[package_name],
                message=f"Required package '{package_name}' is not installed",
                remediation=f"Install package: pip install '{package_name}{version_spec}'",
                python_version=self.python_version
            ))
            return
        
        # Parse and check version constraints
        constraints = self._parse_version_spec(version_spec)
        version_satisfies = all(
            constraint.satisfies(installed_pkg.version) 
            for constraint in constraints
        )
        
        if not version_satisfies:
            conflicts.append(ConflictReport(
                severity=ConflictSeverity.HIGH,
                conflict_type="version_mismatch",
                packages=[package_name],
                message=f"Package '{package_name}' version {installed_pkg.version} "
                       f"does not satisfy constraint {version_spec}",
                remediation=f"Update package: pip install '{package_name}{version_spec}'",
                python_version=self.python_version
            ))
    
    def _validate_complex_package(self, package_name: str, spec: Dict[str, Any], 
                                 conflicts: List[ConflictReport]):
        """Validate complex package specifications (like FAISS variants)"""
        if package_name == "faiss":
            # Handle FAISS CPU/GPU variants
            cpu_spec = spec.get('cpu', {})
            gpu_spec = spec.get('gpu', {})
            
            # Check if any FAISS variant is installed
            faiss_cpu = self.installed_packages.get('faiss-cpu')
            faiss_gpu = self.installed_packages.get('faiss-gpu')
            faiss_generic = self.installed_packages.get('faiss')
            
            installed_variants = []
            if faiss_cpu:
                installed_variants.append('faiss-cpu')
                # Validate CPU version constraints
                if 'version' in cpu_spec:
                    self._validate_package_version('faiss-cpu', cpu_spec['version'], conflicts)
            if faiss_gpu:
                installed_variants.append('faiss-gpu')
                # Validate GPU version constraints
                if 'version' in gpu_spec:
                    self._validate_package_version('faiss-gpu', gpu_spec['version'], conflicts)
            if faiss_generic:
                installed_variants.append('faiss')
            
            if not installed_variants:
                conflicts.append(ConflictReport(
                    severity=ConflictSeverity.HIGH,
                    conflict_type="missing_package",
                    packages=["faiss-cpu", "faiss-gpu"],
                    message="No FAISS variant is installed",
                    remediation="Install CPU variant: pip install faiss-cpu "
                              "OR GPU variant: pip install faiss-gpu",
                    python_version=self.python_version
                ))
    
    def _check_excluded_packages(self) -> List[ConflictReport]:
        """Check for packages that should be excluded for this Python version"""
        conflicts = []
        python_spec = self._get_python_version_spec()
        
        if not python_spec:
            return conflicts
        
        excluded = python_spec.get('excluded_packages', [])
        exclusion_reasons = python_spec.get('exclusion_reasons', {})
        
        for package_name in excluded:
            if package_name.lower() in self.installed_packages:
                reason = exclusion_reasons.get(package_name, "Not compatible")
                conflicts.append(ConflictReport(
                    severity=ConflictSeverity.MEDIUM,
                    conflict_type="excluded_package",
                    packages=[package_name],
                    message=f"Package '{package_name}' should not be installed on "
                           f"Python {self.python_version}: {reason}",
                    remediation=f"Uninstall package: pip uninstall {package_name}",
                    python_version=self.python_version
                ))
        
        return conflicts
    
    def validate_version_constraints(self) -> Dict[str, List[ConflictReport]]:
        """
        Main validation function that checks all version constraints
        
        Returns:
            Dictionary with conflict categories and their reports
        """
        all_conflicts = {
            'faiss_conflicts': self._check_faiss_conflicts(),
            'version_mismatches': self._check_version_constraints(),
            'excluded_packages': self._check_excluded_packages()
        }
        
        return all_conflicts
    
    def generate_validation_report(self, conflicts: Dict[str, List[ConflictReport]]) -> str:
        """Generate a comprehensive validation report"""
        report_lines = [
            "=== EGW Query Expansion - Version Compatibility Report ===",
            f"Python Version: {self.python_version}",
            f"Matrix Version: {self.matrix_data.get('matrix_version', 'unknown')}",
            f"Total Installed Packages: {len(self.installed_packages)}",
            ""
        ]
        
        total_conflicts = sum(len(conflict_list) for conflict_list in conflicts.values())
        
        if total_conflicts == 0:
            report_lines.extend([
                "✅ All packages are compatible!",
                "No version conflicts detected.",
                ""
            ])
        else:
            report_lines.extend([
                f"❌ {total_conflicts} conflicts detected:",
                ""
            ])
            
            severity_counts = {}
            for category_name, conflict_list in conflicts.items():
                if not conflict_list:
                    continue
                
                report_lines.append(f"## {category_name.replace('_', ' ').title()}")
                report_lines.append("")
                
                for conflict in conflict_list:
                    severity_counts[conflict.severity] = severity_counts.get(conflict.severity, 0) + 1
                    
                    severity_emoji = {
                        ConflictSeverity.CRITICAL: "🔴",
                        ConflictSeverity.HIGH: "🟠",
                        ConflictSeverity.MEDIUM: "🟡",
                        ConflictSeverity.LOW: "🔵"
                    }
                    
                    report_lines.extend([
                        f"{severity_emoji[conflict.severity]} **{conflict.severity.value.upper()}**: {conflict.message}",
                        f"   Packages: {', '.join(conflict.packages)}",
                        f"   Remediation: {conflict.remediation}",
                        ""
                    ])
        
        # Add summary
        if total_conflicts > 0:
            report_lines.extend([
                "## Summary",
                f"- Critical: {severity_counts.get(ConflictSeverity.CRITICAL, 0)}",
                f"- High: {severity_counts.get(ConflictSeverity.HIGH, 0)}",
                f"- Medium: {severity_counts.get(ConflictSeverity.MEDIUM, 0)}",
                f"- Low: {severity_counts.get(ConflictSeverity.LOW, 0)}",
                ""
            ])
        
        # Add Python version specific recommendations
        python_spec = self._get_python_version_spec()
        if python_spec and python_spec.get('status') == 'supported':
            report_lines.extend([
                "## Recommended Installation Commands",
                f"For clean installation on Python {self.python_version}:",
                "",
                "```bash",
                "# Remove conflicting packages",
                "pip uninstall faiss-cpu faiss-gpu faiss -y",
                "",
                "# Install CPU variant (recommended for most users)",
                "pip install faiss-cpu>=1.7.4,<1.8.0",
                "",
                "# OR install GPU variant (requires CUDA)",
                "# pip install faiss-gpu>=1.7.4,<1.8.0",
                "",
                "# Install other core dependencies",
                f"pip install torch>=2.0.0 transformers>=4.35.0 sentence-transformers>=2.2.2",
                "```"
            ])
        
        return "\n".join(report_lines)


def validate_version_constraints(matrix_path: str = "version_compatibility_matrix.json") -> None:
    """
    Main function to validate version constraints and print report
    
    Args:
        matrix_path: Path to the compatibility matrix JSON file
    """
    try:
        validator = CompatibilityMatrixValidator(matrix_path)
        conflicts = validator.validate_version_constraints()
        report = validator.generate_validation_report(conflicts)
        
        print(report)
        
        # Exit with error code if critical conflicts exist
        critical_conflicts = any(
            conflict.severity == ConflictSeverity.CRITICAL
            for conflict_list in conflicts.values()
            for conflict in conflict_list
        )
        
        if critical_conflicts:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate package version constraints")
    parser.add_argument(
        "--matrix", "-m",
        default="version_compatibility_matrix.json",
        help="Path to compatibility matrix JSON file"
    )
    
    args = parser.parse_args()
    validate_version_constraints(args.matrix)