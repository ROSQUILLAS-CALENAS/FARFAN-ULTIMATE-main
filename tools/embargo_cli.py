#!/usr/bin/env python3
"""
Embargo CLI - Command Line Interface for Controlled Deletion System

Provides convenient commands for managing the embargo and deletion process
for duplicate non-canonical directories.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

try:
    from .controlled_deletion_system import ControlledDeletionManager
    from .import_linter_config import CIIntegrationHelper
    from .nightly_deletion_scanner import NightlyDeletionScanner
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from controlled_deletion_system import ControlledDeletionManager
    from import_linter_config import CIIntegrationHelper
    from nightly_deletion_scanner import NightlyDeletionScanner


class EmbargoCLI:
    """Command line interface for embargo operations"""
    
    def __init__(self):
        self.manager = ControlledDeletionManager()
        self.ci_helper = CIIntegrationHelper(self.manager)
        self.scanner = NightlyDeletionScanner()
    
    def embargo_directory(self, args):
        """Embargo a directory for deletion"""
        success = self.manager.embargo_directory(
            args.directory,
            args.reason,
            args.grace_period
        )
        
        if success:
            record = self.manager.embargo_registry[args.directory]
            print(f"✅ Directory '{args.directory}' placed under embargo")
            print(f"📅 Expiry date: {record.expiry_date.strftime('%Y-%m-%d')}")
            print(f"⏳ Grace period: {record.grace_period_days} days")
            print(f"📝 Reason: {record.reason}")
            
            # Show canonical mapping if available
            canonical = self.manager.config['canonical_mapping'].get(args.directory)
            if canonical:
                print(f"🎯 Canonical path: {canonical}")
        else:
            print(f"❌ Failed to embargo directory '{args.directory}'")
            sys.exit(1)
    
    def list_embargoed(self, args):
        """List all embargoed directories"""
        if not self.manager.embargo_registry:
            print("📝 No directories are currently under embargo")
            return
        
        print(f"📋 Embargoed Directories ({len(self.manager.embargo_registry)} total)")
        print("=" * 80)
        
        for directory, record in sorted(self.manager.embargo_registry.items()):
            status_emoji = {
                'embargoed': '🟡',
                'warning': '🟠',
                'ready_for_deletion': '🔴',
                'deleted': '✅'
            }.get(record.status, '❓')
            
            print(f"{status_emoji} {directory}")
            print(f"   Status: {record.status}")
            print(f"   Expires: {record.expiry_date.strftime('%Y-%m-%d')} ({record.days_remaining} days remaining)")
            print(f"   Reason: {record.reason}")
            
            if record.external_references:
                print(f"   ⚠️  External refs: {len(record.external_references)}")
            else:
                print(f"   ✅ No external references")
            
            if record.last_scan:
                print(f"   Last scan: {record.last_scan.strftime('%Y-%m-%d %H:%M')}")
            
            print()
    
    def scan_directory(self, args):
        """Scan a specific directory or all embargoed directories"""
        if args.directory:
            if args.directory not in self.manager.embargo_registry:
                print(f"❌ Directory '{args.directory}' is not under embargo")
                sys.exit(1)
            
            record = self.manager.embargo_registry[args.directory]
            print(f"🔍 Scanning {args.directory}...")
            
            # Find external references
            external_refs = self.manager.static_analyzer.find_external_references(args.directory)
            record.external_references = external_refs
            record.last_scan = datetime.now()
            
            print(f"📊 Scan Results:")
            print(f"   External references: {len(external_refs)}")
            
            if external_refs:
                print("   References found in:")
                for ref in external_refs[:10]:  # Show first 10
                    print(f"     • {ref}")
                if len(external_refs) > 10:
                    print(f"     ... and {len(external_refs) - 10} more")
            else:
                print("   ✅ No external references - safe for deletion")
            
            # Update status
            if not external_refs and record.is_expired:
                record.status = "ready_for_deletion"
                print("   🔴 Status updated to: ready_for_deletion")
            
            self.manager._save_embargo_registry()
            
        else:
            # Scan all directories
            print("🔍 Scanning all embargoed directories...")
            scan_results = self.manager.scan_embargoed_directories()
            
            for directory, results in scan_results.items():
                print(f"\n📁 {directory}:")
                print(f"   Status: {results['status']}")
                print(f"   External refs: {len(results['external_references'])}")
                print(f"   Days remaining: {results['days_remaining']}")
    
    def check_imports(self, args):
        """Check for deprecated imports"""
        print("🔍 Checking for deprecated imports...")
        
        passed, results = self.ci_helper.run_full_check()
        
        if passed:
            print("✅ No deprecated imports found")
        else:
            print("❌ Deprecated imports detected!")
            
            # Show import linter results
            if not results['import_linter']['passed']:
                print("\n📋 Import Linter violations:")
                for violation in results['import_linter']['violations']:
                    print(f"   • {violation}")
            
            # Show AST checker results
            if not results['ast_checker']['passed']:
                print("\n🔍 AST Checker violations:")
                for violation in results['ast_checker']['violations']:
                    if isinstance(violation, dict):
                        print(f"   • {violation['file']}:{violation['line']} - {violation['message']}")
        
        if args.fail_on_violations and not passed:
            sys.exit(1)
    
    def generate_report(self, args):
        """Generate embargo status report"""
        summary = self.manager.generate_summary_report()
        
        if args.format == 'json':
            print(json.dumps(summary, indent=2))
        else:
            print("📊 Embargo Status Report")
            print("=" * 40)
            print(f"Total embargoed directories: {summary['total_embargoed']}")
            print(f"Ready for deletion: {summary['ready_for_deletion']}")
            print(f"With external references: {summary['with_external_refs']}")
            print(f"Expired embargoes: {summary['expired_embargoes']}")
            print(f"Generated: {summary['generated_at']}")
            
            print("\n📁 Directory Details:")
            for dir_info in summary['directories']:
                status_emoji = {
                    'embargoed': '🟡',
                    'warning': '🟠', 
                    'ready_for_deletion': '🔴',
                    'deleted': '✅'
                }.get(dir_info['status'], '❓')
                
                print(f"   {status_emoji} {dir_info['directory']} - {dir_info['status']} ({dir_info['days_remaining']} days)")
    
    def delete_ready(self, args):
        """Delete directories that are ready for deletion"""
        deleted = self.manager.safe_delete_ready_directories(dry_run=args.dry_run)
        
        if args.dry_run:
            if deleted:
                print(f"🧪 Dry run - would delete {len(deleted)} directories:")
                for directory in deleted:
                    print(f"   • {directory}")
            else:
                print("🧪 Dry run - no directories ready for deletion")
        else:
            if deleted:
                print(f"🗑️ Successfully deleted {len(deleted)} directories:")
                for directory in deleted:
                    print(f"   • {directory}")
            else:
                print("📝 No directories were ready for deletion")
    
    def remove_embargo(self, args):
        """Remove embargo from a directory"""
        if args.directory not in self.manager.embargo_registry:
            print(f"❌ Directory '{args.directory}' is not under embargo")
            sys.exit(1)
        
        record = self.manager.embargo_registry.pop(args.directory)
        self.manager._save_embargo_registry()
        
        print(f"✅ Removed embargo from '{args.directory}'")
        print(f"📝 Was scheduled for deletion on: {record.expiry_date.strftime('%Y-%m-%d')}")
    
    def extend_embargo(self, args):
        """Extend embargo period for a directory"""
        if args.directory not in self.manager.embargo_registry:
            print(f"❌ Directory '{args.directory}' is not under embargo")
            sys.exit(1)
        
        record = self.manager.embargo_registry[args.directory]
        old_expiry = record.expiry_date
        
        # Extend the grace period
        record.grace_period_days += args.days
        new_expiry = record.expiry_date
        
        self.manager._save_embargo_registry()
        
        print(f"⏰ Extended embargo for '{args.directory}'")
        print(f"   Old expiry: {old_expiry.strftime('%Y-%m-%d')}")
        print(f"   New expiry: {new_expiry.strftime('%Y-%m-%d')}")
        print(f"   Extended by: {args.days} days")
    
    def run_nightly_scan(self, args):
        """Run the nightly scan process"""
        import asyncio
        
        print("🌙 Running nightly scan...")
        results = asyncio.run(self.scanner.run_nightly_job())
        print("✅ Nightly scan completed")
    
    def show_config(self, args):
        """Show current configuration"""
        config = self.manager.config
        
        print("⚙️  Current Configuration")
        print("=" * 30)
        print(f"Default grace period: {config['default_grace_period_days']} days")
        print(f"Auto deletion: {'enabled' if config.get('auto_delete', False) else 'disabled'}")
        print(f"Vulture analysis: {'enabled' if config['enable_vulture'] else 'disabled'}")
        print(f"Import linter: {'enabled' if config['enable_import_linter'] else 'disabled'}")
        
        if config.get('notification_emails'):
            print(f"Notification emails: {len(config['notification_emails'])} configured")
        
        if config.get('canonical_mapping'):
            print(f"\n🎯 Canonical Mappings:")
            for old_path, new_path in config['canonical_mapping'].items():
                print(f"   {old_path} → {new_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Embargo CLI - Controlled Deletion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embargo a directory
  embargo-cli embargo analysis_nlp "Duplicate of canonical_flow/A_analysis_nlp" --grace-period 45

  # List all embargoed directories
  embargo-cli list

  # Scan for external references
  embargo-cli scan --directory analysis_nlp

  # Check for deprecated imports
  embargo-cli check-imports --fail-on-violations

  # Generate status report
  embargo-cli report --format json

  # Delete ready directories (dry run)
  embargo-cli delete --dry-run
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Embargo command
    embargo_parser = subparsers.add_parser('embargo', help='Place directory under embargo')
    embargo_parser.add_argument('directory', help='Directory to embargo')
    embargo_parser.add_argument('reason', help='Reason for embargo')
    embargo_parser.add_argument('--grace-period', type=int, help='Grace period in days')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List embargoed directories')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for external references')
    scan_parser.add_argument('--directory', help='Specific directory to scan')
    
    # Check imports command
    check_parser = subparsers.add_parser('check-imports', help='Check for deprecated imports')
    check_parser.add_argument('--fail-on-violations', action='store_true', 
                             help='Exit with error code if violations found')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate status report')
    report_parser.add_argument('--format', choices=['text', 'json'], default='text')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete ready directories')
    delete_parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    # Remove embargo command
    remove_parser = subparsers.add_parser('remove-embargo', help='Remove embargo from directory')
    remove_parser.add_argument('directory', help='Directory to remove embargo from')
    
    # Extend embargo command
    extend_parser = subparsers.add_parser('extend-embargo', help='Extend embargo period')
    extend_parser.add_argument('directory', help='Directory to extend embargo for')
    extend_parser.add_argument('--days', type=int, required=True, help='Days to extend')
    
    # Nightly scan command
    nightly_parser = subparsers.add_parser('nightly-scan', help='Run nightly scan process')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = EmbargoCLI()
    
    # Map commands to methods
    command_map = {
        'embargo': cli.embargo_directory,
        'list': cli.list_embargoed,
        'scan': cli.scan_directory,
        'check-imports': cli.check_imports,
        'report': cli.generate_report,
        'delete': cli.delete_ready,
        'remove-embargo': cli.remove_embargo,
        'extend-embargo': cli.extend_embargo,
        'nightly-scan': cli.run_nightly_scan,
        'config': cli.show_config
    }
    
    try:
        command_map[args.command](args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()