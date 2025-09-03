#!/usr/bin/env python3
"""
Standalone script to validate phase enforcement constraints.

This script can be run independently to check for phase layering violations
and provides detailed reports on any issues found.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from architecture_tests.test_phase_enforcement import ImportAnalyzer, PhaseViolationError
    from architecture_tests.test_import_graph_analysis import ImportGraphAnalyzer
except ImportError as e:
    print(f"Error importing architecture tests: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


class PhaseEnforcementValidator:
    """Comprehensive phase enforcement validation."""
    
    def __init__(self, root_path: str = "canonical_flow"):
        self.import_analyzer = ImportAnalyzer(root_path)
        self.graph_analyzer = ImportGraphAnalyzer(root_path)
        self.results = {
            'phase_violations': {},
            'circular_dependencies': [],
            'connectivity_matrix': {},
            'summary': {}
        }
    
    def validate_all(self) -> bool:
        """Run all validation checks and return overall success status."""
        success = True
        
        print("🔍 Analyzing phase enforcement constraints...")
        print(f"Canonical Phase Flow: {' → '.join(self.import_analyzer.phase_order)}")
        print()
        
        # Check phase violations
        success &= self._validate_phase_violations()
        
        # Check circular dependencies
        success &= self._validate_circular_dependencies()
        
        # Generate connectivity matrix
        self._generate_connectivity_matrix()
        
        # Generate summary
        self._generate_summary()
        
        return success
    
    def _validate_phase_violations(self) -> bool:
        """Validate phase layering constraints."""
        print("📋 Checking phase layering violations...")
        
        violations = self.import_analyzer.analyze_phase_dependencies()
        self.results['phase_violations'] = violations
        
        total_violations = sum(len(v) for v in violations.values())
        
        if total_violations == 0:
            print("✅ No phase layering violations found!")
            return True
        
        print(f"❌ Found {total_violations} phase layering violations:")
        print()
        
        for phase, phase_violations in violations.items():
            if phase_violations:
                print(f"  {phase} ({len(phase_violations)} violations):")
                for file_path, import_name, target_phase in phase_violations:
                    print(f"    • {file_path}")
                    print(f"      imports {import_name} from {target_phase}")
                print()
        
        return False
    
    def _validate_circular_dependencies(self) -> bool:
        """Validate circular dependencies."""
        print("🔄 Checking circular dependencies...")
        
        cycles = self.graph_analyzer.find_circular_dependencies()
        self.results['circular_dependencies'] = cycles
        
        if not cycles:
            print("✅ No circular dependencies found!")
            return True
        
        print(f"❌ Found {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles[:10]):  # Show first 10
            cycle_str = ' → '.join(cycle) + f' → {cycle[0]}'
            print(f"  {i+1}. {cycle_str}")
        
        if len(cycles) > 10:
            print(f"  ... and {len(cycles) - 10} more cycles")
        
        print()
        return False
    
    def _generate_connectivity_matrix(self):
        """Generate phase connectivity matrix."""
        print("🔗 Generating phase connectivity matrix...")
        
        connectivity = self.graph_analyzer.analyze_phase_connectivity()
        self.results['connectivity_matrix'] = connectivity
        
        # Display matrix
        phases = self.import_analyzer.phase_order
        
        print("    ", end="")
        for target_phase in phases:
            print(f"{target_phase[:3]:>4}", end="")
        print()
        
        for source_phase in phases:
            print(f"{source_phase[:3]:>3} ", end="")
            for target_phase in phases:
                count = connectivity.get(source_phase, {}).get(target_phase, 0)
                if count > 0:
                    print(f"{count:>4}", end="")
                else:
                    print("   .", end="")
            print()
        print()
    
    def _generate_summary(self):
        """Generate validation summary."""
        violations = self.results['phase_violations']
        cycles = self.results['circular_dependencies']
        
        total_violations = sum(len(v) for v in violations.values())
        
        summary = {
            'total_phase_violations': total_violations,
            'total_circular_dependencies': len(cycles),
            'phases_with_violations': len([p for p, v in violations.items() if v]),
            'overall_status': 'PASS' if total_violations == 0 and len(cycles) == 0 else 'FAIL'
        }
        
        self.results['summary'] = summary
        
        print("📊 Validation Summary:")
        print(f"  • Phase violations: {summary['total_phase_violations']}")
        print(f"  • Circular dependencies: {summary['total_circular_dependencies']}")
        print(f"  • Phases with violations: {summary['phases_with_violations']}")
        print(f"  • Overall status: {summary['overall_status']}")
        print()
    
    def save_report(self, output_path: str):
        """Save detailed report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {output_path}")
    
    def generate_remediation_suggestions(self) -> List[str]:
        """Generate suggestions for fixing violations."""
        suggestions = []
        
        violations = self.results['phase_violations']
        total_violations = sum(len(v) for v in violations.values())
        
        if total_violations > 0:
            suggestions.extend([
                "Phase Layering Violation Remediation:",
                "1. Remove direct imports from later phases to earlier phases",
                "2. Use dependency injection patterns instead of direct imports",
                "3. Implement event-driven communication between phases",
                "4. Move shared utilities to a common module outside phase hierarchy",
                "5. Consider refactoring phase boundaries if violations are widespread"
            ])
        
        cycles = self.results['circular_dependencies']
        if cycles:
            suggestions.extend([
                "",
                "Circular Dependency Remediation:",
                "1. Extract common interfaces or abstract base classes",
                "2. Use dependency injection to break circular imports",
                "3. Implement observer pattern for loose coupling",
                "4. Move shared code to separate modules",
                "5. Consider architectural refactoring to eliminate cycles"
            ])
        
        return suggestions


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate phase enforcement constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_phase_enforcement.py
  python scripts/validate_phase_enforcement.py --output report.json
  python scripts/validate_phase_enforcement.py --root-path custom_flow
        """
    )
    
    parser.add_argument(
        '--root-path',
        default='canonical_flow',
        help='Root path for phase analysis (default: canonical_flow)'
    )
    
    parser.add_argument(
        '--output',
        help='Output JSON report file path'
    )
    
    parser.add_argument(
        '--remediation',
        action='store_true',
        help='Show remediation suggestions'
    )
    
    args = parser.parse_args()
    
    print("🚀 Phase Enforcement Validation")
    print("=" * 50)
    
    validator = PhaseEnforcementValidator(args.root_path)
    
    try:
        success = validator.validate_all()
        
        if args.output:
            validator.save_report(args.output)
        
        if args.remediation or not success:
            suggestions = validator.generate_remediation_suggestions()
            if suggestions:
                print("💡 Remediation Suggestions:")
                for suggestion in suggestions:
                    print(f"   {suggestion}")
                print()
        
        if success:
            print("🎉 All phase enforcement constraints are satisfied!")
            return 0
        else:
            print("❌ Phase enforcement violations detected!")
            print("Please address the issues above before proceeding.")
            return 1
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())