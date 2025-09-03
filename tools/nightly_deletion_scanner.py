#!/usr/bin/env python3
"""
Nightly Deletion Scanner

Automated service that runs nightly to:
1. Scan embargoed directories for dead code using vulture and custom analysis
2. Generate reports on remaining usage
3. Safely remove directories with zero external references after embargo expiry
4. Send notifications and update CI status
"""

import asyncio
import datetime
import json
import logging
import smtplib
import subprocess
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional
try:
    import schedule
except ImportError:
    schedule = None
import time

try:
    from .controlled_deletion_system import ControlledDeletionManager
    from .import_linter_config import CIIntegrationHelper
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from controlled_deletion_system import ControlledDeletionManager
    from import_linter_config import CIIntegrationHelper


class DeadCodeAnalyzer:
    """Advanced dead code analysis using multiple tools"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def run_vulture_analysis(self, directory: Path) -> Dict:
        """Run vulture dead code detection"""
        try:
            cmd = [
                sys.executable, '-m', 'vulture', 
                str(directory),
                '--min-confidence', '60',
                '--sort-by-size'
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            # Parse vulture output
            dead_code = {
                'unused_functions': [],
                'unused_classes': [],
                'unused_variables': [],
                'unused_imports': [],
                'unused_attributes': [],
                'total_lines': 0
            }
            
            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                
                if 'unused function' in line:
                    dead_code['unused_functions'].append(line.strip())
                elif 'unused class' in line:
                    dead_code['unused_classes'].append(line.strip())
                elif 'unused variable' in line:
                    dead_code['unused_variables'].append(line.strip())
                elif 'unused import' in line:
                    dead_code['unused_imports'].append(line.strip())
                elif 'unused attribute' in line:
                    dead_code['unused_attributes'].append(line.strip())
            
            # Count total lines in directory
            total_lines = 0
            for py_file in directory.rglob('*.py'):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except:
                    pass
            
            dead_code['total_lines'] = total_lines
            dead_code['confidence_score'] = self._calculate_confidence_score(dead_code)
            
            return dead_code
            
        except Exception as e:
            logging.error(f"Vulture analysis failed for {directory}: {e}")
            return {'error': str(e)}
    
    def run_custom_reference_analysis(self, directory: Path) -> Dict:
        """Custom reference analysis using AST parsing"""
        import ast
        from collections import defaultdict
        
        defined_symbols = defaultdict(list)  # symbol -> [definition_locations]
        referenced_symbols = defaultdict(list)  # symbol -> [reference_locations]
        
        # Analyze all Python files in directory
        for py_file in directory.rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content, filename=str(py_file))
                
                # Find definitions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        defined_symbols[node.name].append(f"{py_file}:{node.lineno}")
                    elif isinstance(node, ast.ClassDef):
                        defined_symbols[node.name].append(f"{py_file}:{node.lineno}")
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                defined_symbols[target.id].append(f"{py_file}:{node.lineno}")
                
                # Find references (simplified)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        referenced_symbols[node.id].append(f"{py_file}:{node.lineno}")
                    elif isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                        referenced_symbols[node.func.id].append(f"{py_file}:{node.lineno}")
            
            except Exception as e:
                logging.warning(f"Failed to analyze {py_file}: {e}")
        
        # Find unreferenced symbols
        unreferenced = {}
        for symbol, definitions in defined_symbols.items():
            if symbol not in referenced_symbols:
                unreferenced[symbol] = definitions
        
        return {
            'defined_symbols': len(defined_symbols),
            'referenced_symbols': len(referenced_symbols), 
            'unreferenced_symbols': unreferenced,
            'unreferenced_count': len(unreferenced)
        }
    
    def _calculate_confidence_score(self, dead_code: Dict) -> float:
        """Calculate confidence score for dead code analysis"""
        total_items = (
            len(dead_code.get('unused_functions', [])) +
            len(dead_code.get('unused_classes', [])) +
            len(dead_code.get('unused_variables', [])) +
            len(dead_code.get('unused_imports', [])) +
            len(dead_code.get('unused_attributes', []))
        )
        
        total_lines = dead_code.get('total_lines', 1)
        
        # Higher ratio of unused items to total lines suggests more confidence
        ratio = total_items / max(total_lines / 10, 1)  # Normalize per 10 lines
        return min(1.0, ratio)


class NotificationService:
    """Service for sending notifications about embargo status"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.smtp_config = config.get('notifications', {}).get('smtp', {})
        self.webhook_config = config.get('notifications', {}).get('webhook', {})
    
    def send_email_notification(self, subject: str, body: str, recipients: List[str]):
        """Send email notification"""
        if not self.smtp_config or not recipients:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html' if '<html>' in body else 'plain'))
            
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            if self.smtp_config.get('use_tls', True):
                server.starttls()
            
            if 'username' in self.smtp_config:
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email notification sent to {recipients}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")
            return False
    
    def send_webhook_notification(self, payload: Dict):
        """Send webhook notification (e.g., to Slack, Teams)"""
        if not self.webhook_config.get('url'):
            return False
        
        try:
            import requests
            
            response = requests.post(
                self.webhook_config['url'],
                json=payload,
                timeout=30,
                headers=self.webhook_config.get('headers', {})
            )
            response.raise_for_status()
            
            logging.info("Webhook notification sent successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {e}")
            return False


class NightlyDeletionScanner:
    """Main nightly scanner service"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.manager = ControlledDeletionManager(config_path)
        self.dead_code_analyzer = DeadCodeAnalyzer(self.manager.project_root)
        self.ci_helper = CIIntegrationHelper(self.manager)
        self.notification_service = NotificationService(self.manager.config)
        
        # Setup logging
        log_path = self.manager.reports_dir / 'nightly_scanner.log'
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
    
    async def scan_single_directory(self, directory: str, record) -> Dict:
        """Scan a single embargoed directory"""
        dir_path = Path(directory)
        
        logging.info(f"Scanning embargoed directory: {directory}")
        
        scan_result = {
            'directory': directory,
            'status': record.status,
            'days_remaining': record.days_remaining,
            'scan_timestamp': datetime.datetime.now().isoformat(),
            'external_references': [],
            'dead_code_analysis': {},
            'reference_analysis': {},
            'recommendations': []
        }
        
        # Find external references
        external_refs = self.manager.static_analyzer.find_external_references(directory)
        scan_result['external_references'] = external_refs
        
        # Run dead code analysis
        scan_result['dead_code_analysis'] = self.dead_code_analyzer.run_vulture_analysis(dir_path)
        
        # Run custom reference analysis
        scan_result['reference_analysis'] = self.dead_code_analyzer.run_custom_reference_analysis(dir_path)
        
        # Generate recommendations
        scan_result['recommendations'] = self._generate_recommendations(scan_result, record)
        
        # Update record
        record.external_references = external_refs
        record.last_scan = datetime.datetime.now()
        
        # Update status based on scan results
        if not external_refs and record.is_expired:
            record.status = "ready_for_deletion"
        elif external_refs and record.is_expired:
            record.status = "warning"
        
        return scan_result
    
    def _generate_recommendations(self, scan_result: Dict, record) -> List[str]:
        """Generate recommendations based on scan results"""
        recommendations = []
        
        external_refs = scan_result['external_references']
        dead_code = scan_result['dead_code_analysis']
        reference_analysis = scan_result['reference_analysis']
        
        # Recommendations based on external references
        if external_refs:
            recommendations.append(
                f"Found {len(external_refs)} external references. "
                "Review and migrate before deletion."
            )
            
            # Group references by importing file
            ref_files = set(ref.split(':')[0] for ref in external_refs)
            if len(ref_files) <= 3:
                recommendations.append(
                    f"External references found in {len(ref_files)} files: {list(ref_files)[:3]}"
                )
        else:
            recommendations.append("No external references found - safe for deletion")
        
        # Recommendations based on dead code analysis
        if 'confidence_score' in dead_code:
            confidence = dead_code['confidence_score']
            if confidence > 0.8:
                recommendations.append(
                    f"High confidence ({confidence:.1%}) that directory contains mostly dead code"
                )
            elif confidence < 0.3:
                recommendations.append(
                    f"Low confidence ({confidence:.1%}) - directory may contain active code"
                )
        
        # Recommendations based on reference analysis
        if reference_analysis.get('unreferenced_count', 0) > 0:
            unreferenced = reference_analysis['unreferenced_count']
            total = reference_analysis.get('defined_symbols', 1)
            ratio = unreferenced / total
            
            if ratio > 0.7:
                recommendations.append(
                    f"High ratio ({ratio:.1%}) of unreferenced symbols suggests dead code"
                )
        
        # Time-based recommendations
        if record.is_expired:
            recommendations.append("Embargo period has expired - ready for action")
        else:
            recommendations.append(
                f"Embargo expires in {record.days_remaining} days"
            )
        
        return recommendations
    
    async def run_full_scan(self) -> Dict:
        """Run full nightly scan of all embargoed directories"""
        scan_start = datetime.datetime.now()
        logging.info("Starting nightly deletion scan")
        
        # Scan all embargoed directories
        scan_results = []
        for directory, record in self.manager.embargo_registry.items():
            if record.status in ['embargoed', 'warning', 'ready_for_deletion']:
                result = await self.scan_single_directory(directory, record)
                scan_results.append(result)
        
        # Save updated registry
        self.manager._save_embargo_registry()
        
        # Generate summary
        summary = {
            'scan_timestamp': scan_start.isoformat(),
            'duration_seconds': (datetime.datetime.now() - scan_start).total_seconds(),
            'directories_scanned': len(scan_results),
            'ready_for_deletion': len([r for r in scan_results if r.get('status') == 'ready_for_deletion']),
            'with_external_refs': len([r for r in scan_results if r.get('external_references')]),
            'total_external_refs': sum(len(r.get('external_references', [])) for r in scan_results)
        }
        
        # Run import checks
        import_check_passed, import_results = self.ci_helper.run_full_check()
        
        # Generate reports
        full_report = {
            'summary': summary,
            'scan_results': scan_results,
            'import_check_results': import_results,
            'import_check_passed': import_check_passed
        }
        
        # Save detailed report
        report_path = self.manager.reports_dir / f'nightly_scan_{scan_start.strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        logging.info(f"Nightly scan completed. Report saved to {report_path}")
        
        return full_report
    
    def perform_safe_deletions(self, dry_run: bool = True) -> Dict:
        """Perform safe deletions of ready directories"""
        deletion_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'dry_run': dry_run,
            'deletions_attempted': 0,
            'deletions_successful': 0,
            'deletions_failed': 0,
            'deleted_directories': [],
            'failed_deletions': []
        }
        
        for directory, record in list(self.manager.embargo_registry.items()):
            if (record.status == 'ready_for_deletion' and 
                not record.external_references and 
                record.is_expired):
                
                deletion_results['deletions_attempted'] += 1
                
                if dry_run:
                    logging.info(f"DRY RUN: Would delete {directory}")
                    deletion_results['deleted_directories'].append({
                        'directory': directory,
                        'reason': 'dry_run'
                    })
                    deletion_results['deletions_successful'] += 1
                else:
                    try:
                        import shutil
                        
                        # Create backup before deletion
                        backup_path = self.manager.reports_dir / 'backups' / f"{directory.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
                        backup_path.parent.mkdir(exist_ok=True)
                        
                        # Create tar backup
                        subprocess.run([
                            'tar', '-czf', str(backup_path), directory
                        ], check=True)
                        
                        # Delete directory
                        shutil.rmtree(directory)
                        
                        # Update record
                        record.status = 'deleted'
                        record.metadata['deletion_date'] = datetime.datetime.now().isoformat()
                        record.metadata['backup_path'] = str(backup_path)
                        
                        deletion_results['deleted_directories'].append({
                            'directory': directory,
                            'backup_path': str(backup_path)
                        })
                        deletion_results['deletions_successful'] += 1
                        
                        logging.info(f"Successfully deleted {directory} (backup: {backup_path})")
                        
                    except Exception as e:
                        deletion_results['failed_deletions'].append({
                            'directory': directory,
                            'error': str(e)
                        })
                        deletion_results['deletions_failed'] += 1
                        logging.error(f"Failed to delete {directory}: {e}")
        
        if not dry_run:
            self.manager._save_embargo_registry()
        
        return deletion_results
    
    def send_notifications(self, scan_report: Dict, deletion_report: Optional[Dict] = None):
        """Send notifications about scan results"""
        summary = scan_report['summary']
        
        # Email notification
        email_body = f"""
        <html>
        <body>
        <h2>🗑️ Nightly Deletion Scan Report</h2>
        
        <h3>Summary</h3>
        <ul>
        <li>Directories scanned: {summary['directories_scanned']}</li>
        <li>Ready for deletion: {summary['ready_for_deletion']}</li>
        <li>With external references: {summary['with_external_refs']}</li>
        <li>Total external references: {summary['total_external_refs']}</li>
        <li>Import check passed: {scan_report['import_check_passed']}</li>
        </ul>
        
        <h3>Action Required</h3>
        """
        
        action_items = []
        
        # Find directories needing attention
        for result in scan_report['scan_results']:
            if result['external_references']:
                action_items.append(f"📋 {result['directory']} has {len(result['external_references'])} external references")
            if result['status'] == 'ready_for_deletion':
                action_items.append(f"🗑️ {result['directory']} is ready for deletion")
        
        if action_items:
            email_body += "<ul>"
            for item in action_items:
                email_body += f"<li>{item}</li>"
            email_body += "</ul>"
        else:
            email_body += "<p>✅ No immediate action required</p>"
        
        if deletion_report:
            email_body += f"""
            <h3>Deletion Results</h3>
            <ul>
            <li>Deletions attempted: {deletion_report['deletions_attempted']}</li>
            <li>Successful: {deletion_report['deletions_successful']}</li>
            <li>Failed: {deletion_report['deletions_failed']}</li>
            </ul>
            """
        
        email_body += """
        <p><em>This is an automated report from the Controlled Deletion System.</em></p>
        </body>
        </html>
        """
        
        recipients = self.manager.config.get('notification_emails', [])
        if recipients:
            self.notification_service.send_email_notification(
                subject=f"🗑️ Deletion Scan Report - {summary['ready_for_deletion']} directories ready",
                body=email_body,
                recipients=recipients
            )
        
        # Webhook notification (e.g., Slack)
        webhook_payload = {
            'text': f"🗑️ Nightly deletion scan completed",
            'attachments': [
                {
                    'color': 'warning' if summary['ready_for_deletion'] > 0 else 'good',
                    'fields': [
                        {'title': 'Directories Scanned', 'value': summary['directories_scanned'], 'short': True},
                        {'title': 'Ready for Deletion', 'value': summary['ready_for_deletion'], 'short': True},
                        {'title': 'With External Refs', 'value': summary['with_external_refs'], 'short': True},
                        {'title': 'Import Check', 'value': '✅ Passed' if scan_report['import_check_passed'] else '❌ Failed', 'short': True}
                    ]
                }
            ]
        }
        
        self.notification_service.send_webhook_notification(webhook_payload)
    
    async def run_nightly_job(self):
        """Main nightly job that runs the complete scan and cleanup process"""
        logging.info("=== Starting Nightly Deletion Job ===")
        
        try:
            # Run full scan
            scan_report = await self.run_full_scan()
            
            # Perform safe deletions (with dry-run option)
            perform_deletions = self.manager.config.get('auto_delete', False)
            deletion_report = self.perform_safe_deletions(dry_run=not perform_deletions)
            
            # Send notifications
            self.send_notifications(scan_report, deletion_report)
            
            # Update CI status if configured
            if self.manager.config.get('ci_integration', {}).get('generate_reports', True):
                ci_report_path = self.manager.project_root / '.github' / 'deletion_status.json'
                ci_status = {
                    'last_scan': scan_report['summary']['scan_timestamp'],
                    'import_check_passed': scan_report['import_check_passed'],
                    'directories_ready': scan_report['summary']['ready_for_deletion'],
                    'external_refs_count': scan_report['summary']['total_external_refs']
                }
                
                ci_report_path.parent.mkdir(exist_ok=True)
                with open(ci_report_path, 'w') as f:
                    json.dump(ci_status, f, indent=2)
            
            logging.info("=== Nightly Deletion Job Completed Successfully ===")
            
        except Exception as e:
            logging.error(f"Nightly job failed: {e}")
            
            # Send error notification
            error_payload = {
                'text': f"🚨 Nightly deletion scan failed: {str(e)}"
            }
            self.notification_service.send_webhook_notification(error_payload)


def setup_scheduler(scanner: NightlyDeletionScanner):
    """Setup scheduled jobs"""
    if not schedule:
        logging.error("Schedule module not available. Install with: pip install schedule")
        return False
        
    # Schedule nightly scan at 2 AM
    schedule.every().day.at("02:00").do(
        lambda: asyncio.run(scanner.run_nightly_job())
    )
    
    # Optional: More frequent checks during business hours
    schedule.every().hour.do(
        lambda: asyncio.run(scanner.run_full_scan())
    ).tag('business_hours')
    
    logging.info("Scheduler setup complete")
    return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nightly Deletion Scanner")
    parser.add_argument('command', choices=['scan', 'delete', 'daemon', 'test'])
    parser.add_argument('--dry-run', action='store_true', help="Dry run mode")
    parser.add_argument('--config', help="Config file path")
    
    args = parser.parse_args()
    
    scanner = NightlyDeletionScanner(Path(args.config) if args.config else None)
    
    if args.command == 'scan':
        # Run single scan
        report = asyncio.run(scanner.run_full_scan())
        print(json.dumps(report, indent=2))
    
    elif args.command == 'delete':
        # Perform deletions
        deletion_report = scanner.perform_safe_deletions(dry_run=args.dry_run)
        print(json.dumps(deletion_report, indent=2))
    
    elif args.command == 'daemon':
        # Run as daemon with scheduler
        if not setup_scheduler(scanner):
            print("Failed to setup scheduler - install schedule module")
            sys.exit(1)
        
        logging.info("Starting nightly deletion scanner daemon")
        while True:
            if schedule:
                schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    elif args.command == 'test':
        # Test run
        logging.info("Running test scan")
        report = asyncio.run(scanner.run_nightly_job())


if __name__ == '__main__':
    main()