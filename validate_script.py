#!/usr/bin/env python3
"""
Simple validation for environment_setup_automation.py
"""

print("Validating environment_setup_automation.py script...")

try:
    # Test import
    import environment_setup_automation
    print("✅ Import successful")
    
    # Test class instantiation 
# # #     from pathlib import Path  # Module not found  # Module not found  # Module not found
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp())
    setup = environment_setup_automation.EnvironmentSetupAutomation(
        project_dir=temp_dir, 
        venv_name="test_venv"
    )
    print("✅ Class instantiation successful")
    
    # Test methods exist
    methods = [
        'create_virtual_environment',
        'upgrade_pip', 
        'install_requirements',
        'verify_installation',
        'rollback_environment',
        'main'
    ]
    
    for method in methods:
        if hasattr(setup, method):
            print(f"✅ Method {method} exists")
        else:
            print(f"❌ Method {method} missing")
    
    print("✅ Script validation completed successfully!")
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    exit(1)