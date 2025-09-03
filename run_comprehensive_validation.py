#!/usr/bin/env python3
"""
Quick runner script for comprehensive validation system.
Provides a simple interface to execute all validation tests locally.
"""

import subprocess
import sys
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

def main():
    """Run comprehensive validation with proper environment setup."""
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🚀 Starting Comprehensive Validation System")
    print("=" * 50)
    
    # Check if virtual environment exists
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("   pip install -r requirements.txt")
        return 1
    
    # Run validation system
    try:
        cmd = [
            sys.executable,
            "validate_comprehensive_ci_system.py",
            "--output-dir", "validation_reports",
            "--verbose"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("-" * 50)
        
        result = subprocess.run(cmd, check=False)
        
        print("-" * 50)
        if result.returncode == 0:
            print("✅ All validation tests passed!")
        else:
            print("❌ Some validation tests failed.")
            print("📋 Check validation_reports/ directory for detailed results.")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())