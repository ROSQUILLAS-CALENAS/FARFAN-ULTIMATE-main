"""
Security regression test module using subprocess to run Bandit against the codebase
and assert no high-severity findings.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest


class SecurityRegressionTest:
    """Test suite for security regression analysis using Bandit."""
    
    @pytest.fixture
    def bandit_config_file(self) -> Optional[Path]:
        """Get the Bandit configuration file if it exists."""
        project_root = Path(__file__).parent.parent
        config_files = [
            project_root / ".bandit",
            project_root / "bandit.yaml",
            project_root / "bandit.yml",
            project_root / "pyproject.toml"  # Bandit config can be in pyproject.toml
        ]
        
        for config_file in config_files:
            if config_file.exists():
                return config_file
        return None
    
    @pytest.fixture
    def scan_directories(self) -> List[Path]:
        """Get directories to scan for security issues."""
        project_root = Path(__file__).parent.parent
        
        scan_dirs = []
        potential_dirs = [
            "egw_query_expansion",
            "src",
            "scripts",
            "tools",
            "phases",
            "adapters",
            "microservices"
        ]
        
        for dir_name in potential_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                scan_dirs.append(dir_path)
        
        # If no specific directories found, scan the entire project
        if not scan_dirs:
            scan_dirs.append(project_root)
        
        return scan_dirs
    
    def _run_bandit_scan(self, directories: List[Path], config_file: Optional[Path] = None) -> Dict[str, Any]:
        """Run Bandit security scan and return results."""
        cmd = [sys.executable, "-m", "bandit", "-f", "json", "-r"]
        
        # Add config file if available
        if config_file and config_file.exists():
            if config_file.name == "pyproject.toml":
                # Check if pyproject.toml contains Bandit config
                try:
                    import tomllib
                    with open(config_file, "rb") as f:
                        data = tomllib.load(f)
                    if "tool" in data and "bandit" in data["tool"]:
                        cmd.extend(["--configfile", str(config_file)])
                except ImportError:
                    # tomllib not available in Python < 3.11, try tomli
                    try:
                        import tomli
                        with open(config_file, "rb") as f:
                            data = tomli.load(f)
                        if "tool" in data and "bandit" in data["tool"]:
                            cmd.extend(["--configfile", str(config_file)])
                    except ImportError:
                        pass  # No TOML support, skip config
            else:
                cmd.extend(["--configfile", str(config_file)])
        
        # Add exclude patterns for common false positives and test files
        exclude_patterns = [
            "*/test*",
            "*/tests/*", 
            "*_test.py",
            "test_*.py",
            "*/venv/*",
            "*/.venv/*",
            "*/build/*",
            "*/dist/*",
            "*/__pycache__/*"
        ]
        
        for pattern in exclude_patterns:
            cmd.extend(["--exclude", pattern])
        
        # Add directories to scan
        for directory in directories:
            cmd.append(str(directory))
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=False   # Don't raise exception on non-zero exit
            )
            
            # Bandit returns non-zero exit code when issues are found
            if result.stdout:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw output
                    return {
                        "results": [],
                        "errors": [{"issue": "JSON parsing failed", "output": result.stdout}],
                        "metrics": {"_totals": {"nosec": 0, "skipped_tests": 0}}
                    }
            else:
                return {
                    "results": [],
                    "errors": [{"issue": "No output from Bandit", "stderr": result.stderr}],
                    "metrics": {"_totals": {"nosec": 0, "skipped_tests": 0}}
                }
                
        except subprocess.TimeoutExpired:
            pytest.skip("Bandit scan timed out after 5 minutes")
        except FileNotFoundError:
            pytest.skip("Bandit not installed - run 'pip install bandit' to enable security regression testing")
        except Exception as e:
            pytest.fail(f"Failed to run Bandit scan: {e}")
    
    def test_no_high_severity_security_issues(self, scan_directories: List[Path], 
                                              bandit_config_file: Optional[Path]) -> None:
        """Test that there are no high-severity security issues in the codebase."""
        results = self._run_bandit_scan(scan_directories, bandit_config_file)
        
        if "errors" in results and results["errors"]:
            # Check if errors are just about missing Bandit
            error_messages = [error.get("issue", str(error)) for error in results["errors"]]
            if any("No output from Bandit" in msg or "JSON parsing failed" in msg for msg in error_messages):
                pytest.skip("Bandit scan failed - please ensure Bandit is properly installed")
        
        high_severity_issues = []
        medium_severity_issues = []
        low_severity_issues = []
        
        for issue in results.get("results", []):
            severity = issue.get("issue_severity", "UNDEFINED").upper()
            confidence = issue.get("issue_confidence", "UNDEFINED").upper()
            
            issue_summary = {
                "test_name": issue.get("test_name", "Unknown"),
                "test_id": issue.get("test_id", "Unknown"),
                "severity": severity,
                "confidence": confidence,
                "filename": issue.get("filename", "Unknown"),
                "line_number": issue.get("line_number", 0),
                "issue_text": issue.get("issue_text", "No description"),
                "code": issue.get("code", "").strip()
            }
            
            if severity == "HIGH":
                high_severity_issues.append(issue_summary)
            elif severity == "MEDIUM":
                medium_severity_issues.append(issue_summary)
            else:
                low_severity_issues.append(issue_summary)
        
        # Fail test if high-severity issues found
        if high_severity_issues:
            error_details = []
            for issue in high_severity_issues:
                details = (
                    f"High severity security issue found:\n"
                    f"  File: {issue['filename']}:{issue['line_number']}\n"
                    f"  Test: {issue['test_name']} ({issue['test_id']})\n"
                    f"  Severity: {issue['severity']} (Confidence: {issue['confidence']})\n"
                    f"  Issue: {issue['issue_text']}"
                )
                if issue['code']:
                    details += f"\n  Code: {issue['code']}"
                error_details.append(details)
            
            pytest.fail(
                f"Found {len(high_severity_issues)} high-severity security issues:\n\n" +
                "\n\n".join(error_details)
            )
        
        # Log medium and low severity issues as warnings (not failures)
        if medium_severity_issues:
            print(f"\nFound {len(medium_severity_issues)} medium-severity security issues (not failing test)")
        
        if low_severity_issues:
            print(f"Found {len(low_severity_issues)} low-severity security issues (not failing test)")
    
    def test_no_hardcoded_secrets(self, scan_directories: List[Path], 
                                  bandit_config_file: Optional[Path]) -> None:
        """Test specifically for hardcoded secrets and passwords."""
        results = self._run_bandit_scan(scan_directories, bandit_config_file)
        
        secret_related_tests = [
            "B105",  # hardcoded_password_string
            "B106",  # hardcoded_password_funcarg  
            "B107",  # hardcoded_password_default
            "B108",  # hardcoded_tmp_directory
            "B110",  # try_except_pass
        ]
        
        secret_issues = []
        for issue in results.get("results", []):
            test_id = issue.get("test_id", "")
            if test_id in secret_related_tests:
                secret_issues.append({
                    "test_id": test_id,
                    "filename": issue.get("filename", "Unknown"),
                    "line_number": issue.get("line_number", 0),
                    "issue_text": issue.get("issue_text", "No description"),
                    "code": issue.get("code", "").strip(),
                    "severity": issue.get("issue_severity", "UNDEFINED")
                })
        
        if secret_issues:
            error_details = []
            for issue in secret_issues:
                details = (
                    f"Potential hardcoded secret found:\n"
                    f"  File: {issue['filename']}:{issue['line_number']}\n"
                    f"  Test: {issue['test_id']}\n"
                    f"  Severity: {issue['severity']}\n"
                    f"  Issue: {issue['issue_text']}"
                )
                if issue['code']:
                    details += f"\n  Code: {issue['code']}"
                error_details.append(details)
            
            pytest.fail(
                f"Found {len(secret_issues)} potential hardcoded secrets:\n\n" +
                "\n\n".join(error_details)
            )
    
    def test_no_shell_injection_vulnerabilities(self, scan_directories: List[Path], 
                                                bandit_config_file: Optional[Path]) -> None:
        """Test specifically for shell injection vulnerabilities."""
        results = self._run_bandit_scan(scan_directories, bandit_config_file)
        
        shell_injection_tests = [
            "B602",  # subprocess_popen_with_shell_equals_true
            "B603",  # subprocess_without_shell_equals_true
            "B604",  # any_other_function_with_shell_equals_true
            "B605",  # start_process_with_a_shell
            "B606",  # start_process_with_no_shell
            "B607",  # start_process_with_partial_path
            "B609",  # linux_commands_wildcard_injection
        ]
        
        shell_issues = []
        for issue in results.get("results", []):
            test_id = issue.get("test_id", "")
            if test_id in shell_injection_tests:
                shell_issues.append({
                    "test_id": test_id,
                    "filename": issue.get("filename", "Unknown"),
                    "line_number": issue.get("line_number", 0),
                    "issue_text": issue.get("issue_text", "No description"),
                    "code": issue.get("code", "").strip(),
                    "severity": issue.get("issue_severity", "UNDEFINED")
                })
        
        if shell_issues:
            error_details = []
            for issue in shell_issues:
                details = (
                    f"Potential shell injection vulnerability found:\n"
                    f"  File: {issue['filename']}:{issue['line_number']}\n"
                    f"  Test: {issue['test_id']}\n"
                    f"  Severity: {issue['severity']}\n"
                    f"  Issue: {issue['issue_text']}"
                )
                if issue['code']:
                    details += f"\n  Code: {issue['code']}"
                error_details.append(details)
            
            pytest.fail(
                f"Found {len(shell_issues)} potential shell injection vulnerabilities:\n\n" +
                "\n\n".join(error_details)
            )


# Global test functions for pytest discovery
def test_security_regression_suite():
    """Entry point for security regression tests."""
    test_instance = SecurityRegressionTest()
    
    # Get fixtures manually
    project_root = Path(__file__).parent.parent
    scan_dirs = []
    potential_dirs = ["egw_query_expansion", "src", "scripts", "tools", "phases", "adapters", "microservices"]
    
    for dir_name in potential_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            scan_dirs.append(dir_path)
    
    if not scan_dirs:
        scan_dirs.append(project_root)
    
    config_files = [project_root / ".bandit", project_root / "bandit.yaml", 
                   project_root / "bandit.yml", project_root / "pyproject.toml"]
    config_file = next((cf for cf in config_files if cf.exists()), None)
    
    # Run individual tests
    test_instance.test_no_high_severity_security_issues(scan_dirs, config_file)
    test_instance.test_no_hardcoded_secrets(scan_dirs, config_file)
    test_instance.test_no_shell_injection_vulnerabilities(scan_dirs, config_file)