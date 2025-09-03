#!/usr/bin/env python3
"""
Test script for environment_setup_automation.py
"""

import sys
import tempfile
from pathlib import Path

# Add current directory to path to import the module
sys.path.insert(0, str(Path.cwd()))

try:
    from environment_setup_automation import (
        EnvironmentSetupAutomation, 
        EnvironmentBackup,
        EnvironmentSetupError
    )
    print("✅ Module imports successful")
    
    # Test basic instantiation
    temp_dir = Path(tempfile.mkdtemp())
    setup = EnvironmentSetupAutomation(project_dir=temp_dir, venv_name="test_venv")
    print("✅ Class instantiation successful")
    
    # Test backup manager
    backup_manager = EnvironmentBackup(temp_dir / ".backups")
    print("✅ Backup manager instantiation successful")
    
    # Test path methods
    python_exe = setup.get_python_executable()
    pip_exe = setup.get_pip_executable()
    print(f"✅ Python executable: {python_exe}")
    print(f"✅ Pip executable: {pip_exe}")
    
    # Test requirements file detection
    req_files = setup.find_requirements_files()
    print(f"✅ Requirements files found: {len(req_files)}")
    
    print("\n🎉 All basic tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test error: {e}")
    sys.exit(1)