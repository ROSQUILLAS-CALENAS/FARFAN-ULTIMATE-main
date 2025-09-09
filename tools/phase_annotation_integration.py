#!/usr/bin/env python3
"""
Phase Annotation CI/CD Integration

This module provides integration utilities for incorporating phase annotation
compliance checks into CI/CD pipelines and development workflows.

Features:
- Pre-commit hook integration
- CI/CD pipeline scripts
- Automated blocking of non-compliant merges
- Integration with code review tools
- Slack/Teams notifications for violations

Usage:
    python tools/phase_annotation_integration.py --setup-hooks
    python tools/phase_annotation_integration.py --ci-check
    python tools/phase_annotation_integration.py --generate-configs
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class IntegrationConfig:
    """Configuration for CI/CD integration."""
    block_on_errors: bool = True
    block_on_warnings: bool = False
    auto_fix_enabled: bool = True
    notification_channels: List[str] = None
    report_formats: List[str] = None

class PhaseAnnotationIntegration:
    """Main integration system for CI/CD pipelines."""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.config = IntegrationConfig()
        
    def setup_pre_commit_hooks(self) -> bool:
        """Set up pre-commit hooks for phase annotation validation."""
        print("🔧 Setting up pre-commit hooks for phase annotation validation...")
        
        try:
            # Create pre-commit config if it doesn't exist
            precommit_config = self.root_dir / '.pre-commit-config.yaml'
            
            if precommit_config.exists():
                with open(precommit_config, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = {'repos': []}
            
            # Add our hooks
            phase_annotation_repo = {
                'repo': 'local',
                'hooks': [
                    {
                        'id': 'phase-annotation-validation',
                        'name': 'Phase Annotation Validation',
                        'entry': 'python tools/phase_annotation_validator.py --validate --ci-mode',
                        'language': 'system',
                        'files': r'\.py$',
                        'pass_filenames': False
                    },
                    {
                        'id': 'phase-annotation-auto-fix',
                        'name': 'Phase Annotation Auto-fix',
                        'entry': 'python tools/phase_annotation_refactor.py --fix --dry-run',
                        'language': 'system',
                        'files': r'\.py$',
                        'pass_filenames': False,
                        'stages': ['manual']
                    }
                ]
            }
            
            # Check if our hooks already exist
            existing_repos = config.get('repos', [])
            has_phase_hooks = any(
                repo.get('repo') == 'local' and 
                any(hook.get('id') == 'phase-annotation-validation' for hook in repo.get('hooks', []))
                for repo in existing_repos
            )
            
            if not has_phase_hooks:
                config['repos'].append(phase_annotation_repo)
                
                with open(precommit_config, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                print("✅ Pre-commit hooks configured successfully!")
                print("Run 'pre-commit install' to activate hooks.")
            else:
                print("✅ Pre-commit hooks already configured.")
                
            return True
            
        except Exception as e:
            print(f"❌ Error setting up pre-commit hooks: {str(e)}")
            return False
    
    def setup_git_hooks(self) -> bool:
        """Set up Git hooks for phase annotation validation."""
        print("🔧 Setting up Git hooks...")
        
        try:
            git_hooks_dir = self.root_dir / '.git' / 'hooks'
            git_hooks_dir.mkdir(exist_ok=True)
            
            # Pre-commit hook
            pre_commit_hook = git_hooks_dir / 'pre-commit'
            pre_commit_script = """#!/bin/bash
# Phase Annotation Validation Pre-commit Hook

echo "🔍 Running phase annotation validation..."

# Run validation
python tools/phase_annotation_validator.py --validate --ci-mode

if [ $? -ne 0 ]; then
    echo "❌ Phase annotation validation failed!"
    echo "Run 'python tools/phase_annotation_refactor.py --fix --apply' to auto-fix issues."
    exit 1
fi

echo "✅ Phase annotation validation passed!"
"""
            
            with open(pre_commit_hook, 'w') as f:
                f.write(pre_commit_script)
            pre_commit_hook.chmod(0o755)
            
            # Pre-push hook for comprehensive validation
            pre_push_hook = git_hooks_dir / 'pre-push'
            pre_push_script = """#!/bin/bash
# Phase Annotation Comprehensive Validation Pre-push Hook

echo "🔍 Running comprehensive phase annotation validation before push..."

# Generate detailed report
python tools/phase_annotation_validator.py --validate --report --output phase_validation_pre_push.json

if [ $? -ne 0 ]; then
    echo "❌ Phase annotation validation failed!"
    echo "See phase_validation_pre_push.json for detailed report."
    exit 1
fi

echo "✅ Comprehensive phase annotation validation passed!"
"""
            
            with open(pre_push_hook, 'w') as f:
                f.write(pre_push_script)
            pre_push_hook.chmod(0o755)
            
            print("✅ Git hooks installed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up Git hooks: {str(e)}")
            return False
    
    def generate_ci_configs(self) -> Dict[str, str]:
        """Generate CI/CD configuration files for popular platforms."""
        print("📝 Generating CI/CD configuration files...")
        
        configs = {}
        
        # GitHub Actions
        configs['github_actions'] = self._generate_github_actions_config()
        
        # GitLab CI
        configs['gitlab_ci'] = self._generate_gitlab_ci_config()
        
        # Jenkins
        configs['jenkins'] = self._generate_jenkins_config()
        
        # Azure DevOps
        configs['azure_devops'] = self._generate_azure_devops_config()
        
        # Circle CI
        configs['circleci'] = self._generate_circleci_config()
        
        return configs
    
    def _generate_github_actions_config(self) -> str:
        """Generate GitHub Actions workflow configuration."""
        return """# Generated GitHub Actions configuration for Phase Annotation Compliance
name: Phase Annotation Compliance

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
    paths: ['**/*.py']

jobs:
  validate-phase-annotations:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Validate Phase Annotations
      run: |
        python tools/phase_annotation_validator.py --validate --ci-mode
        python tools/phase_annotation_validator.py --report --output validation_report.json
    - name: Upload Report
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: phase-validation-report
        path: validation_report.json
"""
    
    def _generate_gitlab_ci_config(self) -> str:
        """Generate GitLab CI configuration."""
        return """# Generated GitLab CI configuration for Phase Annotation Compliance
phases_annotation_validation:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - python tools/phase_annotation_validator.py --validate --ci-mode
    - python tools/phase_annotation_validator.py --report --output validation_report.json
  artifacts:
    reports:
      junit: validation_report.json
    paths:
      - validation_report.json
    expire_in: 1 week
  only:
    changes:
      - "**/*.py"
"""
    
    def _generate_jenkins_config(self) -> str:
        """Generate Jenkins pipeline configuration."""
        return """// Generated Jenkins pipeline for Phase Annotation Compliance
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Phase Annotation Validation') {
            steps {
                sh 'python tools/phase_annotation_validator.py --validate --ci-mode'
                sh 'python tools/phase_annotation_validator.py --report --output validation_report.json'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'validation_report.json', fingerprint: true
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: '.',
                        reportFiles: 'validation_report.json',
                        reportName: 'Phase Annotation Report'
                    ])
                }
            }
        }
    }
}"""
    
    def _generate_azure_devops_config(self) -> str:
        """Generate Azure DevOps pipeline configuration."""
        return """# Generated Azure DevOps pipeline for Phase Annotation Compliance
trigger:
  branches:
    include:
    - main
    - develop
  paths:
    include:
    - '**/*.py'

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.11'

- script: |
    pip install -r requirements.txt
  displayName: 'Install dependencies'

- script: |
    python tools/phase_annotation_validator.py --validate --ci-mode
  displayName: 'Validate Phase Annotations'

- script: |
    python tools/phase_annotation_validator.py --report --output validation_report.json
  displayName: 'Generate Validation Report'

- task: PublishTestResults@2
  condition: always()
  inputs:
    testResultsFiles: 'validation_report.json'
    testRunTitle: 'Phase Annotation Validation'
"""
    
    def _generate_circleci_config(self) -> str:
        """Generate Circle CI configuration."""
        return """# Generated Circle CI configuration for Phase Annotation Compliance
version: 2.1

jobs:
  validate-phase-annotations:
    docker:
      - image: python:3.11
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: pip install -r requirements.txt
      - run:
          name: Validate Phase Annotations
          command: python tools/phase_annotation_validator.py --validate --ci-mode
      - run:
          name: Generate Report
          command: python tools/phase_annotation_validator.py --report --output validation_report.json
      - store_artifacts:
          path: validation_report.json
          destination: validation-report

workflows:
  version: 2
  test:
    jobs:
      - validate-phase-annotations:
          filters:
            branches:
              only: /main|develop/
"""
    
    def run_ci_check(self) -> bool:
        """Run CI validation check."""
        print("🔍 Running CI phase annotation validation check...")
        
        try:
            # Run validator in CI mode
            result = subprocess.run([
                sys.executable, 
                'tools/phase_annotation_validator.py',
                '--validate',
                '--ci-mode'
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            if result.returncode != 0:
                print("❌ Phase annotation validation failed in CI mode!")
                return False
            
            print("✅ Phase annotation validation passed in CI mode!")
            return True
            
        except Exception as e:
            print(f"❌ Error running CI check: {str(e)}")
            return False
    
    def setup_commit_msg_hook(self) -> bool:
        """Set up commit message hook to include phase annotation info."""
        print("🔧 Setting up commit message hook...")
        
        try:
            git_hooks_dir = self.root_dir / '.git' / 'hooks'
            git_hooks_dir.mkdir(exist_ok=True)
            
            commit_msg_hook = git_hooks_dir / 'commit-msg'
            commit_msg_script = """#!/bin/bash
# Phase Annotation Commit Message Hook

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat $COMMIT_MSG_FILE)

# Check if commit involves Python files
PYTHON_FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep '\.py$' | wc -l)

if [ $PYTHON_FILES -gt 0 ]; then
    # Run quick validation
    python tools/phase_annotation_validator.py --validate --ci-mode > /dev/null 2>&1
    VALIDATION_RESULT=$?
    
    if [ $VALIDATION_RESULT -ne 0 ]; then
        echo "" >> $COMMIT_MSG_FILE
        echo "" >> $COMMIT_MSG_FILE
        echo "⚠️  Phase annotation validation issues detected" >> $COMMIT_MSG_FILE
        echo "Run: python tools/phase_annotation_refactor.py --fix --apply" >> $COMMIT_MSG_FILE
    fi
fi
"""
            
            with open(commit_msg_hook, 'w') as f:
                f.write(commit_msg_script)
            commit_msg_hook.chmod(0o755)
            
            print("✅ Commit message hook installed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up commit message hook: {str(e)}")
            return False
    
    def create_makefile_targets(self) -> str:
        """Generate Makefile targets for phase annotation tasks."""
        makefile_content = """
# Phase Annotation Validation Targets

.PHONY: validate-annotations fix-annotations setup-phase-hooks

validate-annotations:
\t@echo "🔍 Validating phase annotations..."
\t@python tools/phase_annotation_validator.py --validate --ci-mode

validate-annotations-report:
\t@echo "📊 Generating phase annotation validation report..."
\t@python tools/phase_annotation_validator.py --validate --report --output phase_validation_report.json
\t@echo "Report saved to phase_validation_report.json"

fix-annotations:
\t@echo "🔧 Auto-fixing phase annotation issues..."
\t@python tools/phase_annotation_refactor.py --fix --apply

fix-annotations-dry-run:
\t@echo "🔧 Simulating phase annotation fixes..."
\t@python tools/phase_annotation_refactor.py --fix --dry-run

setup-phase-hooks:
\t@echo "⚙️  Setting up phase annotation hooks..."
\t@python tools/phase_annotation_integration.py --setup-hooks

phase-annotation-status:
\t@echo "📋 Phase annotation compliance status:"
\t@python -c "\\
import subprocess, json; \\
result = subprocess.run(['python', 'tools/phase_annotation_validator.py', '--validate', '--report', '--output', 'temp_report.json'], capture_output=True); \\
report = json.load(open('temp_report.json')); \\
summary = report['validation_summary']; \\
print(f'Files: {summary[\"files_with_annotations\"]}/{summary[\"total_files_scanned\"]}'); \\
print(f'Violations: {summary[\"total_violations\"]}'); \\
print(f'Compliance: {summary[\"compliance_score\"]}%')"

clean-phase-reports:
\t@echo "🧹 Cleaning up phase annotation reports..."
\t@rm -f phase_validation_report.json temp_report.json phase_validation_pre_push.json
"""
        return makefile_content
    
    def install_all_integrations(self) -> bool:
        """Install all available integrations."""
        print("🚀 Installing all phase annotation integrations...")
        
        success = True
        
        # Pre-commit hooks
        if not self.setup_pre_commit_hooks():
            success = False
        
        # Git hooks
        if not self.setup_git_hooks():
            success = False
        
        # Commit message hook
        if not self.setup_commit_msg_hook():
            success = False
        
        # Generate CI configs
        configs = self.generate_ci_configs()
        ci_dir = self.root_dir / 'ci_configs'
        ci_dir.mkdir(exist_ok=True)
        
        for platform, config in configs.items():
            config_file = ci_dir / f'{platform}.yml'
            with open(config_file, 'w') as f:
                f.write(config)
            print(f"📝 Generated {platform} configuration: {config_file}")
        
        # Generate Makefile targets
        makefile_content = self.create_makefile_targets()
        makefile_path = self.root_dir / 'Makefile.phase_annotations'
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
        print(f"📝 Generated Makefile targets: {makefile_path}")
        
        if success:
            print("✅ All integrations installed successfully!")
            print("\nNext steps:")
            print("1. Run 'pre-commit install' to activate pre-commit hooks")
            print("2. Copy appropriate CI config to your CI/CD platform")
            print("3. Include Makefile.phase_annotations in your main Makefile")
            print("4. Test with 'make validate-annotations'")
        else:
            print("⚠️  Some integrations failed to install. Check errors above.")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Phase Annotation CI/CD Integration")
    parser.add_argument('--setup-hooks', action='store_true', help='Set up pre-commit and Git hooks')
    parser.add_argument('--ci-check', action='store_true', help='Run CI validation check')
    parser.add_argument('--generate-configs', action='store_true', help='Generate CI/CD configuration files')
    parser.add_argument('--install-all', action='store_true', help='Install all integrations')
    parser.add_argument('--root', type=str, default='.', help='Root directory')
    
    args = parser.parse_args()
    
    if not any([args.setup_hooks, args.ci_check, args.generate_configs, args.install_all]):
        parser.print_help()
        return
    
    integration = PhaseAnnotationIntegration(args.root)
    
    success = True
    
    if args.setup_hooks:
        success &= integration.setup_pre_commit_hooks()
        success &= integration.setup_git_hooks()
    
    if args.ci_check:
        success &= integration.run_ci_check()
    
    if args.generate_configs:
        configs = integration.generate_ci_configs()
        print(f"Generated {len(configs)} CI/CD configurations")
    
    if args.install_all:
        success &= integration.install_all_integrations()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()