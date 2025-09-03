"""
Validation script for the complete phase implementation.
"""

import json
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path.cwd()))

from phases._validation import validate_phase_boundaries


def main():
    print("Validating Phase Implementation...")
    print("=" * 50)
    
    # Run validation
    try:
        report = validate_phase_boundaries(install_hook=False)
        
        print("\nPhase API Compliance Report:")
        print("-" * 30)
        
        for phase, result in report['phase_api_compliance'].items():
            status = "✓ PASS" if result['compliant'] else "✗ FAIL"
            print(f"Phase {phase}: {status}")
            
            if not result['compliant']:
                for violation in result['violations'][:2]:
                    print(f"  - {violation}")
        
        print(f"\nImport Violations Summary:")
        print(f"Total violations found: {report['summary']['total_violations']}")
        
        if report['summary']['critical_issues']:
            print("\nCritical Issues:")
            for issue in report['summary']['critical_issues'][:3]:
                print(f"  - {issue}")
        else:
            print("No critical issues found!")
        
        # Save detailed report
        with open('phase_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nCompliant phases: {report['summary']['compliant_phases']}/10")
        
        if report['summary']['compliant_phases'] >= 8:
            print("\n✓ Phase implementation is SUCCESSFUL!")
            return 0
        else:
            print("\n✗ Phase implementation needs improvement.")
            return 1
            
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)