#!/usr/bin/env python3
"""
Setup script for automated nightly deletion system verification.
Creates cron jobs and systemd timers for automated execution.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List


class CronSetup:
    """Setup cron jobs for the deletion system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.script_path = project_root / 'tools' / 'controlled_deletion_system.py'
    
    def install_cron_jobs(self) -> bool:
        """Install cron jobs for nightly verification."""
        cron_entries = [
            # Nightly verification at 2 AM
            f"0 2 * * * cd {self.project_root} && python {self.script_path} --verify",
            # Weekly scan and embargo initiation on Sundays at 1 AM
            f"0 1 * * 0 cd {self.project_root} && python {self.script_path} --scan --embargo",
            # Safe deletion execution on first day of month at 3 AM
            f"0 3 1 * * cd {self.project_root} && python {self.script_path} --delete"
        ]
        
        try:
            # Get current crontab
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            current_crontab = result.stdout if result.returncode == 0 else ""
            
            # Add our entries if not already present
            new_entries = []
            for entry in cron_entries:
                if 'controlled_deletion_system.py' not in current_crontab or entry.split('cd')[1] not in current_crontab:
                    new_entries.append(entry)
            
            if new_entries:
                updated_crontab = current_crontab + '\n' + '\n'.join(new_entries) + '\n'
                
                # Install updated crontab
                process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
                process.communicate(input=updated_crontab)
                
                if process.returncode == 0:
                    print(f"Added {len(new_entries)} cron job(s)")
                    return True
                else:
                    print("Failed to install cron jobs")
                    return False
            else:
                print("Cron jobs already installed")
                return True
                
        except FileNotFoundError:
            print("cron not available on this system")
            return False
        except Exception as e:
            print(f"Failed to install cron jobs: {e}")
            return False


class SystemdSetup:
    """Setup systemd timers for the deletion system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.script_path = project_root / 'tools' / 'controlled_deletion_system.py'
        self.systemd_user_dir = Path.home() / '.config' / 'systemd' / 'user'
    
    def install_systemd_timers(self) -> bool:
        """Install systemd user timers."""
        try:
            self.systemd_user_dir.mkdir(parents=True, exist_ok=True)
            
            # Create service files
            services = {
                'deletion-verify': {
                    'description': 'Controlled Deletion System Nightly Verification',
                    'command': f'--verify',
                    'timer_spec': 'daily',
                    'timer_time': '02:00:00'
                },
                'deletion-scan': {
                    'description': 'Controlled Deletion System Weekly Scan',
                    'command': '--scan --embargo',
                    'timer_spec': 'weekly',
                    'timer_time': 'Sun 01:00:00'
                },
                'deletion-execute': {
                    'description': 'Controlled Deletion System Monthly Execution',
                    'command': '--delete',
                    'timer_spec': 'monthly',
                    'timer_time': '03:00:00'
                }
            }
            
            for service_name, config in services.items():
                # Create service file
                service_content = f"""[Unit]
Description={config['description']}
After=network.target

[Service]
Type=oneshot
WorkingDirectory={self.project_root}
ExecStart={sys.executable} {self.script_path} {config['command']}
User={os.getenv('USER', 'user')}
Environment=PYTHONPATH={self.project_root}
"""
                
                service_file = self.systemd_user_dir / f'{service_name}.service'
                with open(service_file, 'w') as f:
                    f.write(service_content)
                
                # Create timer file
                timer_content = f"""[Unit]
Description=Timer for {config['description']}
Requires={service_name}.service

[Timer]
OnCalendar={config['timer_time']}
Persistent=true

[Install]
WantedBy=timers.target
"""
                
                timer_file = self.systemd_user_dir / f'{service_name}.timer'
                with open(timer_file, 'w') as f:
                    f.write(timer_content)
            
            # Reload systemd user configuration
            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
            
            # Enable and start timers
            for service_name in services.keys():
                subprocess.run(['systemctl', '--user', 'enable', f'{service_name}.timer'], check=True)
                subprocess.run(['systemctl', '--user', 'start', f'{service_name}.timer'], check=True)
            
            print(f"Installed {len(services)} systemd timers")
            return True
            
        except Exception as e:
            print(f"Failed to install systemd timers: {e}")
            return False


class SetupScript:
    """Main setup script for deletion system automation."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.cron_setup = CronSetup(self.project_root)
        self.systemd_setup = SystemdSetup(self.project_root)
    
    def setup_automation(self, method: str = 'auto') -> bool:
        """Setup automation using specified method."""
        if method == 'auto':
            # Try systemd first, fall back to cron
            if self._has_systemd():
                method = 'systemd'
            elif self._has_cron():
                method = 'cron'
            else:
                print("Neither systemd nor cron available")
                return False
        
        if method == 'systemd':
            return self.systemd_setup.install_systemd_timers()
        elif method == 'cron':
            return self.cron_setup.install_cron_jobs()
        else:
            print(f"Unknown automation method: {method}")
            return False
    
    def _has_systemd(self) -> bool:
        """Check if systemd is available."""
        try:
            subprocess.run(['systemctl', '--version'], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    def _has_cron(self) -> bool:
        """Check if cron is available."""
        try:
            subprocess.run(['crontab', '-l'], capture_output=True)
            return True
        except FileNotFoundError:
            return False
    
    def create_monitoring_script(self):
        """Create monitoring script for deletion system."""
        monitoring_script = self.project_root / 'tools' / 'deletion_monitor.py'
        
        monitoring_content = '''#!/usr/bin/env python3
"""
Monitoring script for the controlled deletion system.
Provides status checks and alerts for the deletion process.
"""

import json
import datetime
from pathlib import Path
from controlled_deletion_system import ControlledDeletionSystem


def check_system_health(project_root: str = None) -> dict:
    """Check health of the deletion system."""
    system = ControlledDeletionSystem(project_root)
    
    health_status = {
        'timestamp': datetime.datetime.now().isoformat(),
        'embargo_records_count': 0,
        'ready_for_deletion': 0,
        'overdue_deletions': 0,
        'system_errors': [],
        'status': 'healthy'
    }
    
    try:
        # Load embargo records
        records = system._load_embargo_records()
        health_status['embargo_records_count'] = len(records)
        
        now = datetime.datetime.now()
        
        for record in records:
            if record.directory.status.value == 'scheduled_for_deletion':
                health_status['ready_for_deletion'] += 1
            
            # Check for overdue deletions
            if now > record.expected_deletion + datetime.timedelta(days=7):
                health_status['overdue_deletions'] += 1
        
        # Check system status
        if health_status['overdue_deletions'] > 0:
            health_status['status'] = 'warning'
            health_status['system_errors'].append(
                f"{health_status['overdue_deletions']} overdue deletions"
            )
        
    except Exception as e:
        health_status['status'] = 'error'
        health_status['system_errors'].append(str(e))
    
    return health_status


def generate_status_report(project_root: str = None):
    """Generate and save status report."""
    health = check_system_health(project_root)
    
    system = ControlledDeletionSystem(project_root)
    report_path = system.data_dir / f"health_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(health, f, indent=2)
    
    print(f"Health report saved to: {report_path}")
    return health


if __name__ == '__main__':
    import sys
    project_root = sys.argv[1] if len(sys.argv) > 1 else None
    
    health = generate_status_report(project_root)
    
    print(f"System Status: {health['status']}")
    print(f"Embargo Records: {health['embargo_records_count']}")
    print(f"Ready for Deletion: {health['ready_for_deletion']}")
    
    if health['system_errors']:
        print("Errors:")
        for error in health['system_errors']:
            print(f"  - {error}")
'''
        
        with open(monitoring_script, 'w') as f:
            f.write(monitoring_content)
        
        # Make executable
        monitoring_script.chmod(0o755)
        
        print(f"Created monitoring script: {monitoring_script}")
    
    def create_config_files(self):
        """Create configuration files for the deletion system."""
        config_dir = self.project_root / '.controlled_deletion'
        config_dir.mkdir(exist_ok=True)
        
        # Main configuration
        config = {
            'embargo_days': 30,
            'scan_frequency': 'weekly',
            'verification_frequency': 'daily',
            'deletion_frequency': 'monthly',
            'backup_retention_days': 90,
            'excluded_paths': [
                '.git',
                '.venv',
                'venv',
                '__pycache__',
                '.pytest_cache',
                'node_modules',
                '.idea',
                '.vscode',
                'logs',
                'data'
            ],
            'canonical_path_prefixes': [
                'egw_query_expansion/',
                'canonical_flow/',
                'src/'
            ],
            'notification_email': None,
            'dry_run_by_default': True
        }
        
        config_file = config_dir / 'deletion_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created configuration file: {config_file}")
        
        # Import linter configuration
        pyproject_content = '''[tool.importlinter]
root_package = "egw_query_expansion"

[[tool.importlinter.contracts]]
name = "Deprecated module bans"
type = "forbidden"
source_modules = ["*"]
forbidden_modules = []

[[tool.importlinter.contracts]]
name = "Canonical import enforcement"
type = "layers"
layers = [
    "egw_query_expansion.core",
    "egw_query_expansion.tests",
    "canonical_flow"
]
'''
        
        pyproject_file = self.project_root / 'pyproject.toml'
        if not pyproject_file.exists():
            with open(pyproject_file, 'w') as f:
                f.write(pyproject_content)
            print(f"Created pyproject.toml with import linter config")


def main():
    """Main entry point for setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Controlled Deletion System Automation')
    parser.add_argument('--method', choices=['auto', 'systemd', 'cron'], default='auto',
                       help='Automation method to use')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--no-automation', action='store_true', 
                       help='Skip automation setup, only create scripts')
    
    args = parser.parse_args()
    
    setup = SetupScript(args.project_root)
    
    # Create monitoring and config files
    setup.create_monitoring_script()
    setup.create_config_files()
    
    if not args.no_automation:
        # Setup automation
        success = setup.setup_automation(args.method)
        
        if success:
            print("Deletion system automation setup complete!")
            print("The system will now:")
            print("  - Run weekly scans for duplicate directories")
            print("  - Perform nightly verification of embargoed directories")
            print("  - Execute safe deletions monthly")
        else:
            print("Failed to setup automation. Manual execution required.")
    else:
        print("Scripts created. Setup automation manually or run:")
        print(f"python {setup.project_root}/tools/controlled_deletion_system.py --help")


if __name__ == '__main__':
    main()