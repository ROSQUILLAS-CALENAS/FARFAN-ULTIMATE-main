#!/usr/bin/env python3
"""
CI/CD Integration for Continuous Canonical Compliance (CCC) Validator

Provides CI/CD pipeline integration with configurable failure thresholds
and automated report generation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import the CCC Validator
try:
    from .ccc_validator import CCCValidator
except ImportError:
    from ccc_validator import CCCValidator


class CCCCIIntegration:
    """CI/CD integration wrapper for CCC Validator."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.repo_root = Path(self.config.get('repo_root', '.'))
        self.output_dir = Path(self.config.get('output_dir', 'ccc_reports'))
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load CI configuration."""
        default_config = {
            'repo_root': '.',
            'output_dir': 'ccc_reports',
            'failure_thresholds': {
                'file_naming': {'max_violations': 0},
                'index_sync': {'max_violations': 0},
                'signature_validation': {'max_violations': 5},
                'phase_layering': {'max_violations': 0},
                'dag_validation': {'max_violations': 0}
            },
            'ci_settings': {
                'fail_fast': True,
                'generate_artifacts': True,
                'upload_reports': False,
                'notify_on_failure': False
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def run_validation(self) -> Dict[str, Any]:
        """Run CCC validation with CI/CD specific handling."""
        print("🚀 Starting CCC validation in CI mode...")
        
        # Initialize validator
        validator = CCCValidator(self.repo_root, self.config)
        
        # Run validation
        report = validator.validate_all()
        
        # Apply failure thresholds
        threshold_results = self._apply_failure_thresholds(report)
        
        # Generate artifacts
        if self.config['ci_settings']['generate_artifacts']:
            artifacts = validator.export_artifacts(self.output_dir)
            report['ci_artifacts'] = artifacts
        
        # Add CI-specific metadata
        report['ci_metadata'] = {
            'thresholds_applied': threshold_results,
            'should_fail': threshold_results['should_fail'],
            'pipeline_compatible': True
        }
        
        return report
    
    def _apply_failure_thresholds(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configurable failure thresholds."""
        thresholds = self.config['failure_thresholds']
        threshold_results = {
            'gate_threshold_status': {},
            'should_fail': False,
            'threshold_violations': []
        }
        
        for gate_result in report['gate_results']:
            gate_name = gate_result['gate']
            gate_config = thresholds.get(gate_name, {})
            max_violations = gate_config.get('max_violations', 0)
            
            # Count violations for this gate
            violation_count = 0
            if 'details' in gate_result and 'violations' in gate_result['details']:
                violation_count = len(gate_result['details']['violations'])
            elif 'details' in gate_result and 'issues' in gate_result['details']:
                violation_count = len(gate_result['details']['issues'])
            elif 'details' in gate_result and 'cycles' in gate_result['details']:
                violation_count = len(gate_result['details']['cycles'])
            
            # Check threshold
            exceeds_threshold = violation_count > max_violations
            threshold_results['gate_threshold_status'][gate_name] = {
                'violation_count': violation_count,
                'max_allowed': max_violations,
                'exceeds_threshold': exceeds_threshold,
                'status': 'FAIL' if exceeds_threshold else 'PASS'
            }
            
            if exceeds_threshold:
                threshold_results['should_fail'] = True
                threshold_results['threshold_violations'].append({
                    'gate': gate_name,
                    'violations': violation_count,
                    'threshold': max_violations
                })
        
        return threshold_results
    
    def generate_ci_summary(self, report: Dict[str, Any]) -> str:
        """Generate CI-friendly summary output."""
        summary = report['summary']
        ci_meta = report.get('ci_metadata', {})
        
        output = []
        output.append("=" * 60)
        output.append("🔍 CONTINUOUS CANONICAL COMPLIANCE REPORT")
        output.append("=" * 60)
        
        # Overall status
        status_icon = "✅" if summary['overall_status'] == 'PASS' else "❌"
        output.append(f"{status_icon} Overall Status: {summary['overall_status']}")
        output.append(f"📊 Success Rate: {summary['success_rate']:.1%}")
        output.append(f"🚦 Gates: {summary['passed_gates']}/{summary['total_gates']} passed")
        output.append(f"📦 Components: {len(report['components'])}")
        output.append("")
        
        # Gate results with thresholds
        output.append("🚦 VALIDATION GATES:")
        output.append("-" * 40)
        
        threshold_status = ci_meta.get('thresholds_applied', {}).get('gate_threshold_status', {})
        
        for gate_result in report['gate_results']:
            gate_name = gate_result['gate']
            gate_icon = "✅" if gate_result['status'] == 'PASS' else "❌"
            
            threshold_info = threshold_status.get(gate_name, {})
            threshold_text = ""
            if threshold_info:
                violations = threshold_info['violation_count']
                max_allowed = threshold_info['max_allowed']
                threshold_text = f" ({violations}/{max_allowed} violations)"
            
            output.append(f"{gate_icon} {gate_name.replace('_', ' ').title()}{threshold_text}")
            output.append(f"    {gate_result['message']}")
            
        output.append("")
        
        # Threshold violations
        if ci_meta.get('thresholds_applied', {}).get('threshold_violations'):
            output.append("⚠️  THRESHOLD VIOLATIONS:")
            output.append("-" * 40)
            for violation in ci_meta['thresholds_applied']['threshold_violations']:
                output.append(f"❌ {violation['gate']}: {violation['violations']} > {violation['threshold']}")
            output.append("")
        
        # Artifacts
        if 'ci_artifacts' in report:
            output.append("📄 GENERATED ARTIFACTS:")
            output.append("-" * 40)
            artifacts = report['ci_artifacts']
            output.append(f"📊 HTML Report: {artifacts['html_report']}")
            output.append(f"📋 JSON Report: {artifacts['json_report']}")
            output.append(f"📁 Output Directory: {artifacts['output_directory']}")
            output.append("")
        
        # Final recommendation
        should_fail = ci_meta.get('thresholds_applied', {}).get('should_fail', False)
        if should_fail:
            output.append("🚨 RECOMMENDATION: PIPELINE SHOULD FAIL")
            output.append("   Threshold violations detected above configured limits.")
        else:
            output.append("✅ RECOMMENDATION: PIPELINE CAN CONTINUE")
            output.append("   All violations within acceptable thresholds.")
        
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def set_ci_outputs(self, report: Dict[str, Any]):
        """Set CI system outputs (GitHub Actions, etc.)."""
        summary = report['summary']
        ci_meta = report.get('ci_metadata', {})
        
        # GitHub Actions outputs
        if os.getenv('GITHUB_ACTIONS'):
            self._set_github_outputs(report)
        
        # GitLab CI outputs  
        if os.getenv('GITLAB_CI'):
            self._set_gitlab_outputs(report)
        
        # Jenkins outputs
        if os.getenv('JENKINS_URL'):
            self._set_jenkins_outputs(report)
    
    def _set_github_outputs(self, report: Dict[str, Any]):
        """Set GitHub Actions outputs."""
        summary = report['summary']
        ci_meta = report.get('ci_metadata', {})
        
        github_output = os.getenv('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                f.write(f"ccc_status={summary['overall_status']}\n")
                f.write(f"ccc_success_rate={summary['success_rate']:.3f}\n")
                f.write(f"ccc_gates_passed={summary['passed_gates']}\n")
                f.write(f"ccc_gates_total={summary['total_gates']}\n")
                f.write(f"ccc_should_fail={str(ci_meta.get('thresholds_applied', {}).get('should_fail', False)).lower()}\n")
                
                if 'ci_artifacts' in report:
                    artifacts = report['ci_artifacts']
                    f.write(f"ccc_html_report={artifacts['html_report']}\n")
                    f.write(f"ccc_json_report={artifacts['json_report']}\n")
    
    def _set_gitlab_outputs(self, report: Dict[str, Any]):
        """Set GitLab CI outputs."""
        # GitLab uses environment variables for outputs
        summary = report['summary']
        os.environ['CCC_STATUS'] = summary['overall_status']
        os.environ['CCC_SUCCESS_RATE'] = f"{summary['success_rate']:.3f}"
        os.environ['CCC_GATES_PASSED'] = str(summary['passed_gates'])
        os.environ['CCC_GATES_TOTAL'] = str(summary['total_gates'])
    
    def _set_jenkins_outputs(self, report: Dict[str, Any]):
        """Set Jenkins outputs."""
        # Jenkins can use build properties file
        summary = report['summary']
        with open('ccc_build.properties', 'w') as f:
            f.write(f"CCC_STATUS={summary['overall_status']}\n")
            f.write(f"CCC_SUCCESS_RATE={summary['success_rate']:.3f}\n")
            f.write(f"CCC_GATES_PASSED={summary['passed_gates']}\n")
            f.write(f"CCC_GATES_TOTAL={summary['total_gates']}\n")


def create_default_ci_config():
    """Create a default CI configuration file."""
    config = {
        "repo_root": ".",
        "output_dir": "ccc_reports",
        "failure_thresholds": {
            "file_naming": {
                "max_violations": 0,
                "description": "File naming must follow canonical conventions strictly"
            },
            "index_sync": {
                "max_violations": 0,
                "description": "Index must be perfectly synchronized with filesystem"
            },
            "signature_validation": {
                "max_violations": 5,
                "description": "Allow up to 5 components without proper process signatures"
            },
            "phase_layering": {
                "max_violations": 0,
                "description": "No backward dependencies allowed in canonical flow"
            },
            "dag_validation": {
                "max_violations": 0,
                "description": "No circular dependencies allowed"
            }
        },
        "ci_settings": {
            "fail_fast": True,
            "generate_artifacts": True,
            "upload_reports": False,
            "notify_on_failure": False
        },
        "artifact_retention": {
            "days": 30,
            "max_reports": 100
        }
    }
    
    config_path = Path("ccc_ci_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📄 Created default CI configuration: {config_path}")
    return config_path


def main():
    """CLI interface for CI integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CCC Validator CI/CD Integration")
    parser.add_argument("--config", type=Path, 
                       help="CI configuration file path")
    parser.add_argument("--create-config", action="store_true",
                       help="Create default CI configuration file")
    parser.add_argument("--summary-only", action="store_true",
                       help="Print summary only (no detailed output)")
    parser.add_argument("--set-ci-outputs", action="store_true",
                       help="Set CI system outputs")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_ci_config()
        return
    
    # Initialize CI integration
    ci = CCCCIIntegration(args.config)
    
    # Run validation
    report = ci.run_validation()
    
    # Generate output
    if args.summary_only:
        print(ci.generate_ci_summary(report))
    else:
        # Full output for debugging
        print(json.dumps(report, indent=2))
    
    # Set CI outputs if requested
    if args.set_ci_outputs:
        ci.set_ci_outputs(report)
    
    # Exit with appropriate code
    should_fail = report.get('ci_metadata', {}).get('thresholds_applied', {}).get('should_fail', False)
    if should_fail and ci.config['ci_settings']['fail_fast']:
        print("\n🚨 Pipeline failure recommended due to threshold violations")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()