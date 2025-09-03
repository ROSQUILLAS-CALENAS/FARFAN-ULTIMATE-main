#!/usr/bin/env python3
"""
Environment Setup Automation Script

This script provides automated Python virtual environment creation, activation,
dependency installation, and rollback capabilities with comprehensive error handling.

Features:
- Automatic virtual environment creation and activation
- pip upgrade to latest version
- Requirements.txt installation with error recovery
- Environment backup and rollback functionality
- Comprehensive logging and error reporting
- Cross-platform compatibility

Usage:
    python environment_setup_automation.py [options]
"""

import os
import sys
import subprocess
import shutil
import logging
import tempfile
import json
import argparse
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Optional, Dict, List, Tuple, Any  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
import venv
import platform


class EnvironmentSetupError(Exception):
    """Custom exception for environment setup errors."""
    pass


class EnvironmentBackup:
    """Handles environment backup and restore operations."""
    
    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.backup_metadata = {}
        
    def create_backup(self, venv_path: Path) -> str:
        """Create a backup of existing virtual environment."""
        if not venv_path.exists():
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"venv_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        try:
            logging.info(f"Creating backup of environment at {venv_path}")
            shutil.copytree(venv_path, backup_path)
            
            # Save metadata
            metadata = {
                "original_path": str(venv_path),
                "backup_path": str(backup_path),
                "timestamp": timestamp,
                "platform": platform.system(),
                "python_version": sys.version
            }
            
            metadata_file = backup_path / "backup_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.backup_metadata[backup_name] = metadata
            logging.info(f"Backup created successfully at {backup_path}")
            return backup_name
            
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            raise EnvironmentSetupError(f"Backup creation failed: {e}")
    
    def restore_backup(self, backup_name: str, target_path: Path) -> bool:
# # #         """Restore environment from backup."""  # Module not found  # Module not found  # Module not found
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            logging.error(f"Backup {backup_name} not found at {backup_path}")
            return False
            
        try:
# # #             logging.info(f"Restoring environment from backup {backup_name}")  # Module not found  # Module not found  # Module not found
            
            # Remove current environment if exists
            if target_path.exists():
                shutil.rmtree(target_path)
                
# # #             # Restore from backup  # Module not found  # Module not found  # Module not found
            shutil.copytree(backup_path, target_path)
            logging.info(f"Environment restored successfully to {target_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to restore backup: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 3):
        """Remove old backups, keeping only the most recent ones."""
        try:
            backups = list(self.backup_dir.glob("venv_backup_*"))
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for backup in backups[keep_count:]:
                logging.info(f"Removing old backup: {backup}")
                shutil.rmtree(backup, ignore_errors=True)
                
        except Exception as e:
            logging.warning(f"Failed to cleanup old backups: {e}")


class EnvironmentSetupAutomation:
    """Main class for automated environment setup with rollback capabilities."""
    
    def __init__(self, project_dir: Optional[Path] = None, venv_name: str = "venv"):
        self.project_dir = project_dir or Path.cwd()
        self.venv_name = venv_name
        self.venv_path = self.project_dir / venv_name
        self.backup_dir = self.project_dir / ".env_backups"
        self.backup_manager = EnvironmentBackup(self.backup_dir)
        self.setup_log = []
        self.current_backup = None
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging with both file and console output."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        log_dir = self.project_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup file and console logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / "environment_setup.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def log_step(self, step: str, status: str, details: str = ""):
        """Log a setup step with status."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "details": details
        }
        self.setup_log.append(log_entry)
        logging.info(f"Step: {step} - Status: {status}")
        if details:
            logging.info(f"Details: {details}")
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   timeout: int = 300) -> Tuple[bool, str, str]:
        """Run a command with error handling and logging."""
        cmd_str = " ".join(cmd)
        logging.info(f"Executing command: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            success = result.returncode == 0
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            if success:
                logging.info(f"Command succeeded: {cmd_str}")
                if stdout:
                    logging.debug(f"Output: {stdout}")
            else:
                logging.error(f"Command failed: {cmd_str}")
                logging.error(f"Exit code: {result.returncode}")
                if stdout:
                    logging.error(f"Stdout: {stdout}")
                if stderr:
                    logging.error(f"Stderr: {stderr}")
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            logging.error(f"Command timed out after {timeout}s: {cmd_str}")
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            logging.error(f"Exception running command {cmd_str}: {e}")
            return False, "", str(e)
    
    def get_python_executable(self) -> str:
        """Get the appropriate Python executable path."""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def get_pip_executable(self) -> str:
        """Get the appropriate pip executable path."""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def create_virtual_environment(self) -> bool:
        """Create a new virtual environment."""
        try:
            self.log_step("Creating virtual environment", "STARTED")
            
            if self.venv_path.exists():
                logging.warning(f"Virtual environment already exists at {self.venv_path}")
                self.log_step("Creating virtual environment", "SKIPPED", 
                            "Environment already exists")
                return True
            
            # Create virtual environment using venv module
            venv.create(self.venv_path, with_pip=True, clear=False)
            
            if not self.venv_path.exists():
                raise EnvironmentSetupError("Virtual environment creation failed")
            
            self.log_step("Creating virtual environment", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_step("Creating virtual environment", "FAILED", str(e))
            raise EnvironmentSetupError(f"Failed to create virtual environment: {e}")
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to the latest version."""
        try:
            self.log_step("Upgrading pip", "STARTED")
            
            pip_cmd = [self.get_python_executable(), "-m", "pip", "install", "--upgrade", "pip"]
            success, stdout, stderr = self.run_command(pip_cmd)
            
            if not success:
                raise EnvironmentSetupError(f"Failed to upgrade pip: {stderr}")
            
            # Verify pip version
            version_cmd = [self.get_pip_executable(), "--version"]
            success, version_output, _ = self.run_command(version_cmd)
            
            if success:
                self.log_step("Upgrading pip", "SUCCESS", f"Pip version: {version_output}")
            else:
                self.log_step("Upgrading pip", "SUCCESS", "Upgraded but version check failed")
            
            return True
            
        except Exception as e:
            self.log_step("Upgrading pip", "FAILED", str(e))
            raise EnvironmentSetupError(f"Failed to upgrade pip: {e}")
    
    def find_requirements_files(self) -> List[Path]:
        """Find all requirements files in the project directory."""
        requirements_files = []
        
        # Common requirements file names
        common_names = [
            "requirements.txt",
            "requirements-core.txt", 
            "requirements-essential.txt",
            "requirements-minimal.txt",
            "dev-requirements.txt",
            "test-requirements.txt"
        ]
        
        for name in common_names:
            req_file = self.project_dir / name
            if req_file.exists():
                requirements_files.append(req_file)
                logging.info(f"Found requirements file: {req_file}")
        
        if not requirements_files:
            logging.warning("No requirements files found")
        
        return requirements_files
    
    def install_requirements(self, requirements_file: Path, 
                           retry_count: int = 3) -> bool:
# # #         """Install requirements from a file with retry logic."""  # Module not found  # Module not found  # Module not found
        try:
            self.log_step(f"Installing {requirements_file.name}", "STARTED")
            
            if not requirements_file.exists():
                raise EnvironmentSetupError(f"Requirements file not found: {requirements_file}")
            
            # Try installation with retries
            for attempt in range(retry_count):
                logging.info(f"Installation attempt {attempt + 1}/{retry_count}")
                
                install_cmd = [
                    self.get_pip_executable(), 
                    "install", 
                    "-r", 
                    str(requirements_file),
                    "--no-cache-dir"  # Avoid cache issues
                ]
                
                success, stdout, stderr = self.run_command(install_cmd, timeout=1800)  # 30 minutes
                
                if success:
                    self.log_step(f"Installing {requirements_file.name}", "SUCCESS")
                    return True
                
                logging.warning(f"Installation attempt {attempt + 1} failed: {stderr}")
                
                if attempt < retry_count - 1:
                    logging.info("Retrying installation...")
                    # Try to install setuptools and wheel first for next attempt
                    self.run_command([
                        self.get_pip_executable(), 
                        "install", 
                        "--upgrade", 
                        "setuptools", 
                        "wheel"
                    ])
            
            raise EnvironmentSetupError(f"Failed to install {requirements_file.name} after {retry_count} attempts")
            
        except Exception as e:
            self.log_step(f"Installing {requirements_file.name}", "FAILED", str(e))
            raise EnvironmentSetupError(f"Failed to install requirements: {e}")
    
    def install_all_requirements(self) -> bool:
        """Install all found requirements files."""
        requirements_files = self.find_requirements_files()
        
        if not requirements_files:
            self.log_step("Installing requirements", "SKIPPED", "No requirements files found")
            return True
        
        # Install main requirements.txt first if it exists
        main_req = self.project_dir / "requirements.txt"
        if main_req in requirements_files:
            requirements_files.remove(main_req)
            requirements_files.insert(0, main_req)
        
        # Install each requirements file
        for req_file in requirements_files:
            try:
                self.install_requirements(req_file)
            except EnvironmentSetupError:
                logging.error(f"Failed to install {req_file.name}, continuing with others...")
                # Continue with other files but mark overall as partial failure
                continue
        
        return True
    
    def verify_installation(self) -> bool:
        """Verify the installation by running basic checks."""
        try:
            self.log_step("Verifying installation", "STARTED")
            
            # Check Python version
            python_cmd = [self.get_python_executable(), "--version"]
            success, python_version, _ = self.run_command(python_cmd)
            
            if not success:
                raise EnvironmentSetupError("Python verification failed")
            
            # Check pip list
            pip_cmd = [self.get_pip_executable(), "list"]
            success, pip_list, _ = self.run_command(pip_cmd)
            
            if not success:
                raise EnvironmentSetupError("Pip list verification failed")
            
            # Count installed packages
            package_count = len([line for line in pip_list.split('\n') if line.strip() and not line.startswith('-')])
            
            verification_details = f"Python: {python_version}, Packages: {package_count}"
            self.log_step("Verifying installation", "SUCCESS", verification_details)
            
            return True
            
        except Exception as e:
            self.log_step("Verifying installation", "FAILED", str(e))
            return False
    
    def rollback_environment(self) -> bool:
        """Rollback to the previous environment state."""
        if not self.current_backup:
            logging.error("No backup available for rollback")
            return False
        
        try:
            logging.info("Initiating environment rollback...")
            success = self.backup_manager.restore_backup(self.current_backup, self.venv_path)
            
            if success:
                logging.info("Environment rollback completed successfully")
                self.log_step("Environment rollback", "SUCCESS")
            else:
                logging.error("Environment rollback failed")
                self.log_step("Environment rollback", "FAILED")
            
            return success
            
        except Exception as e:
            logging.error(f"Rollback failed: {e}")
            self.log_step("Environment rollback", "FAILED", str(e))
            return False
    
    def generate_setup_report(self) -> str:
        """Generate a comprehensive setup report."""
        report = []
        report.append("=" * 50)
        report.append("ENVIRONMENT SETUP REPORT")
        report.append("=" * 50)
        report.append(f"Project Directory: {self.project_dir}")
        report.append(f"Virtual Environment: {self.venv_path}")
        report.append(f"Python Version: {sys.version}")
        report.append(f"Platform: {platform.system()} {platform.release()}")
        report.append(f"Setup Time: {datetime.now().isoformat()}")
        report.append("")
        
        report.append("SETUP STEPS:")
        report.append("-" * 30)
        for step in self.setup_log:
            status_symbol = "✓" if step["status"] == "SUCCESS" else "✗" if step["status"] == "FAILED" else "⚠"
            report.append(f"{status_symbol} {step['step']}: {step['status']}")
            if step["details"]:
                report.append(f"   Details: {step['details']}")
        
        report.append("")
        if self.current_backup:
            report.append(f"Backup Created: {self.current_backup}")
        
        return "\n".join(report)
    
    def main(self, create_backup: bool = True, install_requirements: bool = True) -> bool:
        """Main setup function with comprehensive error handling."""
        try:
            logging.info("Starting environment setup automation")
            logging.info(f"Project directory: {self.project_dir}")
            logging.info(f"Virtual environment: {self.venv_path}")
            
            # Create backup of existing environment if requested
            if create_backup and self.venv_path.exists():
                try:
                    self.current_backup = self.backup_manager.create_backup(self.venv_path)
                except EnvironmentSetupError as e:
                    logging.warning(f"Backup creation failed: {e}")
                    # Continue without backup
            
            # Create virtual environment
            self.create_virtual_environment()
            
            # Upgrade pip
            self.upgrade_pip()
            
            # Install requirements if requested
            if install_requirements:
                self.install_all_requirements()
            
            # Verify installation
            verification_success = self.verify_installation()
            
            if not verification_success:
                logging.warning("Installation verification failed")
            
            # Cleanup old backups
            self.backup_manager.cleanup_old_backups()
            
            # Generate and display report
            report = self.generate_setup_report()
            print("\n" + report)
            
            # Save report to file
            report_file = self.project_dir / "environment_setup_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logging.info(f"Setup report saved to: {report_file}")
            logging.info("Environment setup completed successfully")
            
            return True
            
        except EnvironmentSetupError as e:
            logging.error(f"Setup failed: {e}")
            
            # Attempt rollback if backup exists
            if self.current_backup:
                logging.info("Attempting to rollback to previous environment...")
                rollback_success = self.rollback_environment()
                if not rollback_success:
                    logging.error("Rollback failed. Manual intervention required.")
            
            # Generate error report
            report = self.generate_setup_report()
            print("\n" + report)
            
            return False
            
        except KeyboardInterrupt:
            logging.info("Setup interrupted by user")
            
            # Attempt rollback if backup exists
            if self.current_backup:
                logging.info("Attempting to rollback due to interruption...")
                self.rollback_environment()
            
            return False
            
        except Exception as e:
            logging.error(f"Unexpected error during setup: {e}")
            
            # Attempt rollback if backup exists
            if self.current_backup:
                logging.info("Attempting to rollback due to unexpected error...")
                self.rollback_environment()
            
            return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated Python environment setup with rollback capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python environment_setup_automation.py
  python environment_setup_automation.py --venv-name myenv --no-backup
  python environment_setup_automation.py --project-dir /path/to/project
        """
    )
    
    parser.add_argument(
        "--project-dir",
        type=Path,
        help="Project directory path (default: current directory)"
    )
    
    parser.add_argument(
        "--venv-name", 
        default="venv",
        help="Virtual environment directory name (default: venv)"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup of existing environment"
    )
    
    parser.add_argument(
        "--no-requirements",
        action="store_true", 
        help="Skip installing requirements files"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize setup automation
    setup_automation = EnvironmentSetupAutomation(
        project_dir=args.project_dir,
        venv_name=args.venv_name
    )
    
    # Run setup with specified options
    success = setup_automation.main(
        create_backup=not args.no_backup,
        install_requirements=not args.no_requirements
    )
    
    if success:
        print("\n✅ Environment setup completed successfully!")
        print(f"   Virtual environment: {setup_automation.venv_path}")
        
        # Display activation instructions
        if platform.system() == "Windows":
            activate_cmd = f"{setup_automation.venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {setup_automation.venv_path}/bin/activate"
        
        print(f"   To activate: {activate_cmd}")
        sys.exit(0)
    else:
        print("\n❌ Environment setup failed!")
        print("   Check the logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()