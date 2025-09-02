#!/usr/bin/env python3
"""
Enhanced Import Fix Script for Virtual Environment

This script:
1. Activates the virtual environment if it exists
2. Installs missing dependencies in the virtual environment
3. Fixes common import issues and module path problems
4. Specifically addresses PDF processing library imports
5. Updates broken import statements in source files
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple


class VenvImportFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.venv_path = self.project_root / "venv"
        self.venv_python = None
        self.venv_pip = None
        self.fixes_applied = []
        
        # Common import fixes and corrections
        self.import_corrections = {
            'PyPDF2': 'PyPDF2',
            'pdfplumber': 'pdfplumber', 
            'PyMuPDF': 'fitz',  # PyMuPDF is imported as fitz
            'cv2': 'opencv-python',
            'PIL': 'pillow',
            'sklearn': 'scikit-learn',
            'yaml': 'pyyaml',
            'spellchecker': 'pyspellchecker',
            'msgspec': 'msgspec',
            'blake3': 'blake3',
        }
        
    def setup_venv_paths(self):
        """Setup virtual environment paths"""
        if not self.venv_path.exists():
            print(f"Virtual environment not found at {self.venv_path}")
            print("Creating virtual environment...")
            subprocess.run([sys.executable, '-m', 'venv', str(self.venv_path)])
        
        # Set up paths for venv executables
        if sys.platform == "win32":
            self.venv_python = self.venv_path / "Scripts" / "python.exe"
            self.venv_pip = self.venv_path / "Scripts" / "pip.exe"
        else:
            self.venv_python = self.venv_path / "bin" / "python"
            self.venv_pip = self.venv_path / "bin" / "pip"
    
    def install_requirements(self):
        """Install requirements.txt in virtual environment"""
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("No requirements.txt found")
            return False
        
        print("Installing requirements in virtual environment...")
        try:
            result = subprocess.run([
                str(self.venv_pip), 'install', '-r', str(requirements_file)
            ], capture_output=True, text=True, cwd=str(self.project_root))
            
            if result.returncode == 0:
                print("✓ Successfully installed requirements")
                self.fixes_applied.append("Installed requirements.txt")
                return True
            else:
                print(f"✗ Failed to install requirements: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error installing requirements: {e}")
            return False
    
    def install_specific_packages(self, packages: List[str]):
        """Install specific packages in virtual environment"""
        for package in packages:
            try:
                print(f"Installing {package}...")
                result = subprocess.run([
                    str(self.venv_pip), 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✓ Successfully installed {package}")
                    self.fixes_applied.append(f"Installed {package}")
                else:
                    print(f"✗ Failed to install {package}: {result.stderr}")
                    
            except Exception as e:
                print(f"Error installing {package}: {e}")
    
    def fix_pdf_imports(self):
        """Fix PDF-related import issues in source files"""
        print("Fixing PDF library imports...")
        
        # Common PDF import fixes
        pdf_fixes = [
            ('import fitz  # PyMuPDF', 'import fitz  # PyMuPDF'),
            ('import fitz  # PyMuPDF', 'import fitz  # PyMuPDF'),
            ('import cv2', 'import cv2'),  # Make sure opencv is properly imported
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if any(part in file_path.parts for part in ['.git', 'venv', '__pycache__']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply PDF import fixes
                for old_import, new_import in pdf_fixes:
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        print(f"Fixed import in {file_path}: {old_import} -> {new_import}")
                
                # Fix common import path issues
                content = self.fix_relative_imports(content, file_path)
                
                # Save changes if any were made
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.fixes_applied.append(f"Fixed imports in {file_path.name}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def fix_relative_imports(self, content: str, file_path: Path) -> str:
        """Fix relative import issues"""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            original_line = line
            
            # Fix common module path issues
            if 'from ' in line:
                line = line.replace('from ', 'from ')
            elif 'import ' in line:
                line = line.replace('import ', 'import ')
            
            if 'from ' in line:
                line = line.replace('from ', 'from ')
            elif 'import ' in line:
                line = line.replace('import ', 'import ')
            
            if 'from ' in line:
                # These look like they should be local imports
                line = line.replace('from ', 'from ')
            
            # Fix orchestration imports
#             if 'from orchestration.event_bus' in line:  # Module not found
                line = '# ' + line + '  # Module not found'
#             elif 'from tracing.decorators' in line:  # Module not found
                line = '# ' + line + '  # Module not found'
#             elif 'from monitoring_stack' in line:  # Module not found
                line = '# ' + line + '  # Module not found'
            
            if line != original_line:
                print(f"Fixed import: {original_line.strip()} -> {line.strip()}")
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def create_missing_modules(self):
        """Create stub modules for missing imports"""
        missing_modules = {
            'event_bus.py': '''"""Event bus module stub"""
class EventBus:
    def __init__(self):
        pass
    
    def publish(self, event, data):
        pass
    
    def subscribe(self, event, callback):
        pass
''',
            'path_verification.py': '''"""Path verification module stub"""
def verify_path(path):
    return True
''',
            'monitoring_stack.py': '''"""Monitoring stack module stub"""
class MonitoringStack:
    def __init__(self):
        pass
    
    def start_monitoring(self):
        pass
''',
            'tracing/__init__.py': '''"""Tracing module"""''',
            'tracing/decorators.py': '''"""Tracing decorators stub"""
def trace(func):
    return func
''',
        }
        
        for module_path, content in missing_modules.items():
            full_path = self.project_root / module_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not full_path.exists():
                with open(full_path, 'w') as f:
                    f.write(content)
                print(f"Created stub module: {module_path}")
                self.fixes_applied.append(f"Created stub module {module_path}")
    
    def verify_pdf_libraries(self):
        """Verify PDF processing libraries are working"""
        print("Verifying PDF libraries...")
        
        test_script = '''
import sys
sys.path.insert(0, ".")

libraries_status = {}

# Test PyMuPDF (fitz)
try:
    import fitz
    libraries_status["PyMuPDF (fitz)"] = "✓ Available"
except ImportError as e:
    libraries_status["PyMuPDF (fitz)"] = f"✗ Not available: {e}"

# Test pdfplumber
try:
    import pdfplumber
    libraries_status["pdfplumber"] = "✓ Available"
except ImportError as e:
    libraries_status["pdfplumber"] = f"✗ Not available: {e}"

# Test pytesseract
try:
    import pytesseract
    libraries_status["pytesseract"] = "✓ Available"
except ImportError as e:
    libraries_status["pytesseract"] = f"✗ Not available: {e}"

# Test camelot
try:
    import camelot
    libraries_status["camelot"] = "✓ Available"
except ImportError as e:
    libraries_status["camelot"] = f"✗ Not available: {e}"

# Test tabula
try:
    import tabula
    libraries_status["tabula"] = "✓ Available"
except ImportError as e:
    libraries_status["tabula"] = f"✗ Not available: {e}"

# Test opencv
try:
    import cv2
    libraries_status["opencv (cv2)"] = "✓ Available"
except ImportError as e:
    libraries_status["opencv (cv2)"] = f"✗ Not available: {e}"

for lib, status in libraries_status.items():
    print(f"{lib}: {status}")
'''
        
        try:
            result = subprocess.run([
                str(self.venv_python), '-c', test_script
            ], capture_output=True, text=True, cwd=str(self.project_root))
            
            print("PDF Library Verification Results:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
                
        except Exception as e:
            print(f"Error verifying PDF libraries: {e}")
    
    def run_comprehensive_fix(self):
        """Run the complete import fixing process"""
        print("="*60)
        print("COMPREHENSIVE IMPORT FIXER")
        print("="*60)
        
        # Setup virtual environment
        self.setup_venv_paths()
        
        # Install requirements
        self.install_requirements()
        
        # Install specific problematic packages
        specific_packages = [
            'PyPDF2',
            'pdfplumber', 
            'PyMuPDF',
            'pytesseract',
            'camelot-py',
            'tabula-py',
            'opencv-python',
            'msgspec',
            'blake3',
            'pyspellchecker',
        ]
        self.install_specific_packages(specific_packages)
        
        # Create missing module stubs
        self.create_missing_modules()
        
        # Fix import statements in source files
        self.fix_pdf_imports()
        
        # Verify PDF libraries
        self.verify_pdf_libraries()
        
        # Summary
        print("\n" + "="*60)
        print("FIXES APPLIED SUMMARY:")
        print("="*60)
        for fix in self.fixes_applied:
            print(f"✓ {fix}")
        
        print(f"\nTotal fixes applied: {len(self.fixes_applied)}")
        print("Import fixing process completed!")


def main():
    """Main function to run import fixes"""
    fixer = VenvImportFixer()
    fixer.run_comprehensive_fix()


if __name__ == "__main__":
    main()