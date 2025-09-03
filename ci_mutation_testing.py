#!/usr/bin/env python3
"""
CI Mutation Testing Integration Script

This script runs mutation tests on validators and context/synthesis operations 
with threshold-based build failure when mutation scores fall below configured minimums.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mutation_testing_config import (
    MUTATION_CONFIG, 
    MutationTestTarget, 
    MutationTool,
    ComponentType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('mutation_testing.log')
    ]
)
logger = logging.getLogger(__name__)

class MutationTestResult:
    """Result of a mutation testing run."""
    
    def __init__(self, target_name: str, mutation_score: float, 
                 total_mutants: int, killed_mutants: int, 
                 survived_mutants: int, runtime_seconds: float,
                 success: bool, error_message: Optional[str] = None):
        self.target_name = target_name
        self.mutation_score = mutation_score
        self.total_mutants = total_mutants
        self.killed_mutants = killed_mutants
        self.survived_mutants = survived_mutants
        self.runtime_seconds = runtime_seconds
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'target_name': self.target_name,
            'mutation_score': self.mutation_score,
            'total_mutants': self.total_mutants,
            'killed_mutants': self.killed_mutants,
            'survived_mutants': self.survived_mutants,
            'runtime_seconds': self.runtime_seconds,
            'success': self.success,
            'error_message': self.error_message
        }

class MutationTestRunner:
    """Orchestrates mutation testing for configured targets."""
    
    def __init__(self, tool: MutationTool = MutationTool.MUTMUT, 
                 parallel: bool = False, max_workers: int = 2):
        self.tool = tool
        self.parallel = parallel
        self.max_workers = max_workers
        self.results: List[MutationTestResult] = []
        
    def run_mutmut_test(self, target: MutationTestTarget) -> MutationTestResult:
        """Run mutation test using mutmut."""
        logger.info(f"Running mutmut on target: {target.name}")
        start_time = time.time()
        
        try:
            # Filter existing paths
            existing_paths = []
            for path in target.paths:
                full_path = Path(path)
                if full_path.exists():
                    existing_paths.append(str(full_path))
                else:
                    logger.warning(f"Path does not exist: {path}")
            
            if not existing_paths:
                logger.warning(f"No existing paths found for target {target.name}")
                return MutationTestResult(
                    target.name, 0.0, 0, 0, 0, 
                    time.time() - start_time, False, 
                    "No existing paths to test"
                )
            
            # Create temporary configuration
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
                f.write("[mutmut]\n")
                f.write(f"paths_to_mutate = {','.join(existing_paths)}\n")
                f.write(f"tests_dir = tests\n")
                f.write(f"runner = python_unittest\n")
                temp_config = f.name
            
            try:
                # Run mutmut
                cmd = [
                    'python', '-m', 'mutmut', 'run',
                    '--paths-to-mutate', ','.join(existing_paths),
                    '--tests-dir', 'tests',
                    '--runner', 'python'
                ]
                
                logger.info(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=target.max_runtime_minutes * 60,
                    cwd=Path(__file__).parent
                )
                
                # Parse mutmut results
                if result.returncode == 0:
                    # Get mutation results
                    status_cmd = ['python', '-m', 'mutmut', 'results']
                    status_result = subprocess.run(
                        status_cmd, capture_output=True, text=True,
                        cwd=Path(__file__).parent
                    )
                    
                    mutation_score, total, killed, survived = self._parse_mutmut_output(
                        status_result.stdout
                    )
                    
                    success = mutation_score >= target.min_mutation_score
                    
                    return MutationTestResult(
                        target.name, mutation_score, total, killed, survived,
                        time.time() - start_time, success
                    )
                else:
                    logger.error(f"Mutmut failed: {result.stderr}")
                    return MutationTestResult(
                        target.name, 0.0, 0, 0, 0,
                        time.time() - start_time, False, 
                        f"Mutmut execution failed: {result.stderr}"
                    )
                    
            finally:
                # Cleanup temporary config
                if os.path.exists(temp_config):
                    os.unlink(temp_config)
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Mutmut timed out for target {target.name}")
            return MutationTestResult(
                target.name, 0.0, 0, 0, 0,
                time.time() - start_time, False, "Timeout"
            )
        except Exception as e:
            logger.error(f"Error running mutmut for {target.name}: {e}")
            return MutationTestResult(
                target.name, 0.0, 0, 0, 0,
                time.time() - start_time, False, str(e)
            )
    
    def _parse_mutmut_output(self, output: str) -> Tuple[float, int, int, int]:
        """Parse mutmut results output."""
        # Default values
        mutation_score = 0.0
        total_mutants = 0
        killed_mutants = 0
        survived_mutants = 0
        
        try:
            lines = output.strip().split('\n')
            for line in lines:
                if 'killed' in line.lower():
                    # Try to parse mutation score from output
                    if '%' in line:
                        import re
                        match = re.search(r'(\d+\.?\d*)%', line)
                        if match:
                            mutation_score = float(match.group(1))
                    
                    # Parse counts from line like "Killed: 15, Survived: 3, Total: 18"
                    killed_match = re.search(r'killed:?\s*(\d+)', line, re.IGNORECASE)
                    if killed_match:
                        killed_mutants = int(killed_match.group(1))
                    
                    survived_match = re.search(r'survived:?\s*(\d+)', line, re.IGNORECASE)
                    if survived_match:
                        survived_mutants = int(survived_match.group(1))
                    
                    total_match = re.search(r'total:?\s*(\d+)', line, re.IGNORECASE)
                    if total_match:
                        total_mutants = int(total_match.group(1))
                    
        except Exception as e:
            logger.warning(f"Error parsing mutmut output: {e}")
        
        # Calculate score if not found in output
        if mutation_score == 0.0 and total_mutants > 0:
            mutation_score = (killed_mutants / total_mutants) * 100
        
        return mutation_score, total_mutants, killed_mutants, survived_mutants
    
    def run_cosmic_ray_test(self, target: MutationTestTarget) -> MutationTestResult:
        """Run mutation test using cosmic-ray."""
        logger.info(f"Running cosmic-ray on target: {target.name}")
        start_time = time.time()
        
        try:
            # Filter existing paths
            existing_paths = []
            for path in target.paths:
                full_path = Path(path)
                if full_path.exists():
                    existing_paths.append(str(full_path))
            
            if not existing_paths:
                return MutationTestResult(
                    target.name, 0.0, 0, 0, 0, 
                    time.time() - start_time, False, 
                    "No existing paths to test"
                )
            
            # Create cosmic-ray config
            config = MUTATION_CONFIG.get_cosmic_ray_config(target)
            config_file = f"cosmic_ray_{target.name}.toml"
            
            with open(config_file, 'w') as f:
                f.write(f'[cosmic-ray]\n')
                f.write(f'module-path = "{existing_paths[0]}"\n')
                f.write(f'python-path = ""\n')
                f.write(f'test-command = "python -m pytest {" ".join(target.test_patterns)}"\n')
                
            try:
                # Initialize cosmic-ray
                init_cmd = ['cosmic-ray', 'init', config_file, f"{target.name}.sqlite"]
                subprocess.run(init_cmd, check=True, capture_output=True, text=True)
                
                # Run cosmic-ray
                exec_cmd = ['cosmic-ray', 'exec', f"{target.name}.sqlite"]
                result = subprocess.run(
                    exec_cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=target.max_runtime_minutes * 60
                )
                
                # Get results
                report_cmd = ['cosmic-ray', 'report', f"{target.name}.sqlite"]
                report_result = subprocess.run(report_cmd, capture_output=True, text=True)
                
                mutation_score, total, killed, survived = self._parse_cosmic_ray_output(
                    report_result.stdout
                )
                
                success = mutation_score >= target.min_mutation_score
                
                return MutationTestResult(
                    target.name, mutation_score, total, killed, survived,
                    time.time() - start_time, success
                )
                
            finally:
                # Cleanup
                for cleanup_file in [config_file, f"{target.name}.sqlite"]:
                    if os.path.exists(cleanup_file):
                        os.unlink(cleanup_file)
                        
        except Exception as e:
            logger.error(f"Error running cosmic-ray for {target.name}: {e}")
            return MutationTestResult(
                target.name, 0.0, 0, 0, 0,
                time.time() - start_time, False, str(e)
            )
    
    def _parse_cosmic_ray_output(self, output: str) -> Tuple[float, int, int, int]:
        """Parse cosmic-ray results output."""
        # Simple parsing - cosmic-ray output format varies
        mutation_score = 0.0
        total_mutants = 0
        killed_mutants = 0
        survived_mutants = 0
        
        try:
            import re
            lines = output.strip().split('\n')
            for line in lines:
                if 'score' in line.lower() and '%' in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        mutation_score = float(match.group(1))
        except Exception as e:
            logger.warning(f"Error parsing cosmic-ray output: {e}")
            
        return mutation_score, total_mutants, killed_mutants, survived_mutants
    
    def run_target_test(self, target: MutationTestTarget) -> MutationTestResult:
        """Run mutation test for a single target."""
        if self.tool == MutationTool.MUTMUT:
            return self.run_mutmut_test(target)
        elif self.tool == MutationTool.COSMIC_RAY:
            return self.run_cosmic_ray_test(target)
        else:
            raise ValueError(f"Unsupported mutation tool: {self.tool}")
    
    def run_all_tests(self, target_names: Optional[List[str]] = None) -> List[MutationTestResult]:
        """Run mutation tests for all configured targets."""
        targets = MUTATION_CONFIG.get_all_targets()
        
        if target_names:
            targets = [t for t in targets if t.name in target_names]
        
        logger.info(f"Running mutation tests for {len(targets)} targets")
        
        results = []
        for target in targets:
            logger.info(f"Testing target: {target.name}")
            result = self.run_target_test(target)
            results.append(result)
            self.results.append(result)
            
            # Log intermediate results
            if result.success:
                logger.info(
                    f"✓ {target.name}: {result.mutation_score:.1f}% "
                    f"(threshold: {target.min_mutation_score}%)"
                )
            else:
                logger.error(
                    f"✗ {target.name}: {result.mutation_score:.1f}% "
                    f"(threshold: {target.min_mutation_score}%) - {result.error_message}"
                )
        
        return results
    
    def generate_report(self, output_file: str = 'mutation_test_report.json') -> Dict:
        """Generate comprehensive mutation testing report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tool': self.tool.value,
            'total_targets': len(self.results),
            'successful_targets': sum(1 for r in self.results if r.success),
            'failed_targets': sum(1 for r in self.results if not r.success),
            'overall_success': all(r.success for r in self.results),
            'results': [r.to_dict() for r in self.results],
            'summary_by_type': self._generate_type_summary()
        }
        
        # Write report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Mutation testing report written to: {output_file}")
        return report
    
    def _generate_type_summary(self) -> Dict:
        """Generate summary by component type."""
        summary = {}
        
        for component_type in ComponentType:
            type_targets = MUTATION_CONFIG.get_targets_by_type(component_type)
            type_results = [r for r in self.results 
                          if any(t.name == r.target_name for t in type_targets)]
            
            if type_results:
                summary[component_type.value] = {
                    'total': len(type_results),
                    'passed': sum(1 for r in type_results if r.success),
                    'failed': sum(1 for r in type_results if not r.success),
                    'avg_score': sum(r.mutation_score for r in type_results) / len(type_results),
                    'min_threshold': min(t.min_mutation_score for t in type_targets),
                    'results': [r.target_name for r in type_results]
                }
        
        return summary

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='CI Mutation Testing Integration')
    parser.add_argument(
        '--tool', 
        choices=['mutmut', 'cosmic-ray'],
        default='mutmut',
        help='Mutation testing tool to use'
    )
    parser.add_argument(
        '--targets',
        nargs='*',
        help='Specific targets to test (default: all)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=2,
        help='Maximum parallel workers'
    )
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first failure'
    )
    parser.add_argument(
        '--report-file',
        default='mutation_test_report.json',
        help='Output file for results report'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create runner
    tool = MutationTool.MUTMUT if args.tool == 'mutmut' else MutationTool.COSMIC_RAY
    runner = MutationTestRunner(tool, args.parallel, args.max_workers)
    
    try:
        # Run mutation tests
        results = runner.run_all_tests(args.targets)
        
        # Generate report
        report = runner.generate_report(args.report_file)
        
        # Print summary
        print("\n" + "="*60)
        print("MUTATION TESTING SUMMARY")
        print("="*60)
        print(f"Tool: {tool.value}")
        print(f"Total Targets: {report['total_targets']}")
        print(f"Successful: {report['successful_targets']}")
        print(f"Failed: {report['failed_targets']}")
        print(f"Overall Success: {report['overall_success']}")
        
        print("\nResults by Target:")
        for result in results:
            status = "✓" if result.success else "✗"
            print(f"  {status} {result.target_name}: {result.mutation_score:.1f}%")
            if result.error_message:
                print(f"    Error: {result.error_message}")
        
        print(f"\nDetailed report: {args.report_file}")
        
        # Set exit code based on overall success
        if not report['overall_success']:
            logger.error("Mutation testing failed - scores below threshold")
            sys.exit(1)
        else:
            logger.info("All mutation tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Mutation testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()