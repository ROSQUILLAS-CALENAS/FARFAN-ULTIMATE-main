#!/usr/bin/env python3
"""
Recovery Scripts for EGW Query Expansion System

This module provides comprehensive recovery functionality to diagnose and repair
failed or incomplete installations of the EGW system. It includes functions to:
- detect_partial_installation(): Scan for incomplete package installations
- clean_corrupted_environment(): Remove broken packages and clear caches  
- reinstall_dependencies_ordered(): Reinstall packages in dependency order

Can be executed as a standalone script for automated recovery workflow.
"""

import sys
import os
import subprocess
import importlib
import json
import logging
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Tuple, Set, Optional  # Module not found  # Module not found  # Module not found
import tempfile
import shutil
# # # from packaging import version  # Module not found  # Module not found  # Module not found


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('recovery_scripts.log')
    ]
)
logger = logging.getLogger(__name__)


def detect_partial_installation() -> Dict[str, any]:
    """
    Scan for incomplete package installations by checking import failures 
    and version mismatches.
    
    Returns:
        dict: Recovery report with failed imports, version issues, and recommendations
    """
    logger.info("Starting partial installation detection...")
    
    report = {
        'failed_imports': [],
        'version_mismatches': [],
        'missing_packages': [],
        'corrupted_installs': [],
        'recommendations': []
    }
    
# # #     # Core dependencies from requirements.txt  # Module not found  # Module not found  # Module not found
    core_packages = [
        ('faiss', 'faiss-cpu>=1.7.4'),
        ('transformers', 'transformers>=4.35.0'),
        ('sentence_transformers', 'sentence-transformers>=2.2.2'),
        ('torch', 'torch>=2.0.0'),
        ('numpy', 'numpy>=1.24.0'),
        ('scipy', 'scipy>=1.11.0'),
        ('ot', 'POT>=0.9.1'),
        ('sklearn', 'scikit-learn>=1.3.0'),
        ('datasets', 'datasets>=2.14.0'),
        ('pandas', 'pandas>=1.5.0'),
        ('yaml', 'pyyaml>=5.1'),
        ('tqdm', 'tqdm>=4.66.0'),
        ('spacy', 'spacy>=3.7.0'),
        ('nltk', 'nltk>=3.8.0'),
        ('matplotlib', 'matplotlib>=3.7.0'),
        ('pytest', 'pytest>=7.0.0'),
        ('jupyter', 'jupyter>=1.0.0'),
        ('z3', 'z3-solver>=4.12.0'),
        ('msgspec', 'msgspec>=0.18.0'),
        ('pydantic', 'pydantic>=2.0.0'),
        ('orjson', 'orjson>=3.8.0')
    ]
    
    # Test imports and version compatibility
    for module_name, package_spec in core_packages:
        try:
            # Attempt import
            module = importlib.import_module(module_name)
            logger.info(f"Successfully imported {module_name}")
            
            # Check version if available
            if hasattr(module, '__version__'):
                current_version = module.__version__
# # #                 # Extract minimum version from spec  # Module not found  # Module not found  # Module not found
                min_version = package_spec.split('>=')[1] if '>=' in package_spec else None
                
                if min_version and version.parse(current_version) < version.parse(min_version):
                    report['version_mismatches'].append({
                        'package': module_name,
                        'current_version': current_version,
                        'required_version': min_version,
                        'package_spec': package_spec
                    })
                    logger.warning(f"Version mismatch: {module_name} {current_version} < {min_version}")
                    
        except ImportError as e:
            report['failed_imports'].append({
                'package': module_name,
                'error': str(e),
                'package_spec': package_spec
            })
            logger.error(f"Failed to import {module_name}: {e}")
        except Exception as e:
            report['corrupted_installs'].append({
                'package': module_name,
                'error': str(e),
                'package_spec': package_spec
            })
            logger.error(f"Corrupted installation detected for {module_name}: {e}")
    
    # Test EGW-specific components
    egw_modules = [
        'egw_query_expansion.core',
        'egw_query_expansion.core.hybrid_retrieval',
        'egw_query_expansion.core.gw_alignment'
    ]
    
    for module_name in egw_modules:
        try:
            importlib.import_module(module_name)
            logger.info(f"Successfully imported EGW module: {module_name}")
        except ImportError as e:
            report['failed_imports'].append({
                'package': module_name,
                'error': str(e),
                'package_spec': 'EGW core component'
            })
            logger.error(f"Failed to import EGW module {module_name}: {e}")
    
    # Generate recommendations
    if report['failed_imports']:
        report['recommendations'].append("Run clean_corrupted_environment() to remove broken packages")
    if report['version_mismatches']:
        report['recommendations'].append("Run reinstall_dependencies_ordered() to update packages")
    if report['corrupted_installs']:
        report['recommendations'].append("Complete environment cleanup and reinstall required")
    
    logger.info(f"Detection complete. Found {len(report['failed_imports'])} import failures, "
               f"{len(report['version_mismatches'])} version mismatches")
    
    return report


def clean_corrupted_environment() -> bool:
    """
    Remove broken packages and clear pip cache to prepare for clean reinstallation.
    
    Returns:
        bool: True if cleanup successful, False otherwise
    """
    logger.info("Starting corrupted environment cleanup...")
    
    try:
        # Clear pip cache
        logger.info("Clearing pip cache...")
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                      check=True, capture_output=True, text=True)
        
        # Get list of installed packages
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                               capture_output=True, text=True, check=True)
        installed_packages = json.loads(result.stdout)
        
# # #         # Identify potentially corrupted packages from requirements.txt  # Module not found  # Module not found  # Module not found
        requirements_file = Path("requirements.txt")
        corrupted_packages = []
        
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                requirements = f.readlines()
            
# # #             # Extract package names from requirements  # Module not found  # Module not found  # Module not found
            req_packages = set()
            for line in requirements:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle package name extraction
                    package_name = line.split('>=')[0].split('==')[0].split('[')[0]
                    req_packages.add(package_name)
            
            # Find installed packages that match requirements but may be corrupted
            for package in installed_packages:
                if package['name'] in req_packages:
                    # Test if package is importable
                    try:
                        importlib.import_module(package['name'].replace('-', '_'))
                    except ImportError:
                        corrupted_packages.append(package['name'])
                        logger.info(f"Marking {package['name']} for removal (import failed)")
        
        # Uninstall corrupted packages
        if corrupted_packages:
            logger.info(f"Uninstalling {len(corrupted_packages)} corrupted packages...")
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y"] + corrupted_packages
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Clean up Python cache files
        logger.info("Cleaning Python cache files...")
        for root, dirs, files in os.walk('.'):
            # Remove __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for d in dirs.copy():
                if d == '__pycache__':
                    shutil.rmtree(os.path.join(root, d))
                    dirs.remove(d)
            
            # Remove .pyc files
            for file in files:
                if file.endswith('.pyc'):
                    os.remove(os.path.join(root, file))
        
        # Clear any temporary build directories
        build_dirs = ['build', 'dist', '*.egg-info']
        for pattern in build_dirs:
            for path in Path('.').glob(pattern):
                if path.is_dir():
                    logger.info(f"Removing build directory: {path}")
                    shutil.rmtree(path)
        
        logger.info("Environment cleanup completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Cleanup failed: {e}")
        logger.error(f"Command output: {e.stdout if hasattr(e, 'stdout') else 'N/A'}")
        logger.error(f"Command stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {e}")
        return False


def reinstall_dependencies_ordered() -> bool:
    """
# # #     Reinstall packages following dependency hierarchy from requirements.txt.  # Module not found  # Module not found  # Module not found
    
    Returns:
        bool: True if reinstallation successful, False otherwise
    """
    logger.info("Starting ordered dependency reinstallation...")
    
    try:
        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            return False
        
        # Define installation order based on dependency hierarchy
        install_order = [
            # Core system dependencies first
            ['setuptools>=68.0.0', 'packaging>=21.0'],
            
            # Basic numerical and scientific computing
            ['numpy>=1.24.0', 'scipy>=1.11.0'],
            
            # Core ML/AI frameworks
            ['torch>=2.0.0'],
            
            # Higher-level ML libraries
            ['scikit-learn>=1.3.0', 'transformers>=4.35.0', 'sentence-transformers>=2.2.2'],
            
            # Specialized ML/NLP libraries
            ['faiss-cpu>=1.7.4', 'POT>=0.9.1'],
            
            # Data handling
            ['pandas>=1.5.0', 'datasets>=2.14.0', 'pyyaml>=5.1', 'tqdm>=4.66.0'],
            
            # NLP libraries
            ['spacy[en_core_web_sm]>=3.7.0', 'nltk>=3.8.0'],
            
            # Statistical and mathematical
            ['pingouin>=0.5.0', 'statsmodels>=0.14.0', 'networkx>=3.1.0'],
            
            # Visualization
            ['matplotlib>=3.7.0', 'seaborn>=0.12.0', 'plotly>=5.15.0'],
            
            # Evaluation and benchmarking  
            ['beir>=2.0.0'],
            
            # Development and testing
            ['pytest>=7.0.0', 'pytest-cov>=4.0.0', 'jupyter>=1.0.0', 'pre-commit>=3.6.0'],
            
            # Constraint solving and optimization
            ['z3-solver>=4.12.0'],
            
            # Data serialization
            ['msgspec>=0.18.0', 'pydantic>=2.0.0', 'orjson>=3.8.0', 'dill>=0.3.7', 'cloudpickle>=2.2.0'],
            
            # Security and utilities
            ['blake3>=0.3.3', 'psutil>=5.9.0', 'GitPython>=3.1.0'],
            
            # Cloud services
            ['google-cloud-storage>=2.10.0', 'google-cloud-pubsub>=2.18.0', 
             'google-cloud-bigquery>=3.11.0', 'google-cloud-vision>=3.4.0'],
            
            # Async and HTTP
            ['aioredis>=2.0.0', 'httpx>=0.24.0', 'redis>=4.5.0'],
            
            # Document processing
            ['PyMuPDF>=1.23.0', 'pdfplumber>=0.9.0', 'PyPDF2>=3.0.1'],
            
            # Additional text processing
            ['pytesseract>=0.3.10', 'easyocr>=1.7.0', 'opencv-python>=4.8.0'],
            
            # Remaining dependencies
            ['python-consul>=1.1.0', 'pm4py>=2.7.0', 'deap>=1.4.0', 'whoosh>=2.7.4',
             'ray>=2.7.0', 'torch-geometric>=2.3.0', 'elasticsearch>=8.9.0',
             'joblib>=1.3.0', 'dask>=2023.8.0', 'sqlalchemy>=2.0.0', 'lark>=1.1.0',
             'jsonschema>=4.19.0', 'toml>=0.10.2', 'prometheus-client>=0.17.0',
             'opentelemetry-api>=1.20.0', 'opentelemetry-sdk>=1.20.0', 'loguru>=0.7.0',
             'rich>=13.5.0', 'tabula-py>=2.8.0', 'fuzzywuzzy>=0.18.0', 
             'python-Levenshtein>=0.21.0', 'kubernetes>=27.2.0', 'kafka-python>=2.0.2',
             'hypothesis>=6.82.0', 'pillow>=10.0.0']
        ]
        
        # Upgrade pip first
        logger.info("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                      check=True, capture_output=True, text=True)
        
        # Install packages in order
        for batch_idx, batch in enumerate(install_order):
            logger.info(f"Installing batch {batch_idx + 1}/{len(install_order)}: {batch}")
            
            for package in batch:
                try:
                    cmd = [sys.executable, "-m", "pip", "install", package]
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    logger.info(f"Successfully installed: {package}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {package}: {e}")
                    logger.warning(f"Stderr: {e.stderr}")
                    # Continue with other packages
                    continue
        
        # Install EGW package in development mode
        logger.info("Installing EGW package in development mode...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                      check=True, capture_output=True, text=True)
        
        logger.info("Dependency reinstallation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during dependency reinstallation: {e}")
        return False


def run_recovery_workflow() -> bool:
    """
    Main recovery workflow that orchestrates detection, cleanup, and reinstallation.
    
    Returns:
        bool: True if recovery successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("EGW SYSTEM RECOVERY WORKFLOW STARTING")
    logger.info("=" * 60)
    
    try:
        # Step 1: Detect issues
        logger.info("STEP 1: Detecting partial installations...")
        detection_report = detect_partial_installation()
        
        # Log detection results
        logger.info(f"Detection Summary:")
        logger.info(f"- Failed imports: {len(detection_report['failed_imports'])}")
        logger.info(f"- Version mismatches: {len(detection_report['version_mismatches'])}")
        logger.info(f"- Corrupted installs: {len(detection_report['corrupted_installs'])}")
        
        # Save detection report
        with open('recovery_detection_report.json', 'w') as f:
            json.dump(detection_report, f, indent=2)
        logger.info("Detection report saved to recovery_detection_report.json")
        
        # Step 2: Clean environment if issues detected
        needs_cleanup = (detection_report['failed_imports'] or 
                        detection_report['version_mismatches'] or 
                        detection_report['corrupted_installs'])
        
        if needs_cleanup:
            logger.info("STEP 2: Cleaning corrupted environment...")
            if not clean_corrupted_environment():
                logger.error("Environment cleanup failed")
                return False
            logger.info("Environment cleanup completed successfully")
        else:
            logger.info("STEP 2: No cleanup needed - environment appears healthy")
        
        # Step 3: Reinstall dependencies if cleanup was performed
        if needs_cleanup:
            logger.info("STEP 3: Reinstalling dependencies in proper order...")
            if not reinstall_dependencies_ordered():
                logger.error("Dependency reinstallation failed")
                return False
            logger.info("Dependencies reinstalled successfully")
        else:
            logger.info("STEP 3: No reinstallation needed")
        
        # Step 4: Final validation
        logger.info("STEP 4: Running final validation...")
        final_report = detect_partial_installation()
        
        remaining_issues = (len(final_report['failed_imports']) + 
                          len(final_report['version_mismatches']) + 
                          len(final_report['corrupted_installs']))
        
        if remaining_issues == 0:
            logger.info("✓ RECOVERY SUCCESSFUL - All issues resolved")
            return True
        else:
            logger.warning(f"⚠ PARTIAL RECOVERY - {remaining_issues} issues remaining")
            logger.warning("Check recovery_detection_report.json for details")
            return False
            
    except Exception as e:
        logger.error(f"Recovery workflow failed with unexpected error: {e}")
        return False
    finally:
        logger.info("=" * 60)
        logger.info("EGW SYSTEM RECOVERY WORKFLOW COMPLETE")
        logger.info("=" * 60)


if __name__ == "__main__":
    """
    Execute recovery workflow as standalone script
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="EGW System Recovery Tool")
    parser.add_argument("--detect-only", action="store_true", 
                       help="Only run detection, don't perform recovery")
    parser.add_argument("--clean-only", action="store_true",
                       help="Only clean environment, don't reinstall")
    parser.add_argument("--reinstall-only", action="store_true", 
                       help="Only reinstall dependencies, don't clean")
    
    args = parser.parse_args()
    
    if args.detect_only:
        report = detect_partial_installation()
        print(json.dumps(report, indent=2))
    elif args.clean_only:
        success = clean_corrupted_environment()
        sys.exit(0 if success else 1)
    elif args.reinstall_only:
        success = reinstall_dependencies_ordered()
        sys.exit(0 if success else 1)
    else:
        # Run full recovery workflow
        success = run_recovery_workflow()
        sys.exit(0 if success else 1)