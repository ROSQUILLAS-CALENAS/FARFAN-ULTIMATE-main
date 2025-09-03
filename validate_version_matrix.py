#!/usr/bin/env python3
"""
Comprehensive validation script for the EGW Query Expansion version compatibility matrix.
This script validates the current environment against the version matrix and provides
detailed remediation steps.
"""

import sys
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from mathematical_compatibility_matrix import validate_version_constraints  # Module not found  # Module not found  # Module not found

def main():
    """Main entry point for version validation"""
    print("🔍 EGW Query Expansion - Version Compatibility Validator")
    print("=" * 60)
    
    matrix_file = "version_compatibility_matrix.json"
    
    # Check if matrix file exists
    if not Path(matrix_file).exists():
        print(f"❌ Error: Version compatibility matrix not found: {matrix_file}")
        print("Please ensure the matrix file is in the current directory.")
        sys.exit(1)
    
    # Load and display matrix info
    try:
        with open(matrix_file, 'r') as f:
            matrix_data = json.load(f)
        
        print(f"📋 Matrix Version: {matrix_data.get('matrix_version', 'unknown')}")
        print(f"🐍 Current Python: {sys.version_info.major}.{sys.version_info.minor}")
        
        supported_versions = list(matrix_data.get('python_versions', {}).keys())
        print(f"✅ Supported Python versions: {', '.join(supported_versions)}")
        print()
        
    except Exception as e:
        print(f"❌ Error reading matrix file: {e}")
        sys.exit(1)
    
    # Run validation
    try:
        validate_version_constraints(matrix_file)
        print("\n🎉 Validation completed successfully!")
        
    except SystemExit as e:
        if e.code == 1:
            print("\n⚠️  Critical conflicts detected. Please resolve them before proceeding.")
            print("\nFor detailed installation instructions, see:")
            print("- README.md")
            print("- TONKOTSU.md")
            print("- requirements.txt")
        sys.exit(e.code)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()