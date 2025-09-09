#!/usr/bin/env python3
"""
Performance Monitoring Script for EGW Query Expansion System Remediation

This script captures, analyzes, and reports on key performance metrics during
remediation work to establish baselines and detect regressions.

Usage:
    python scripts/performance_monitor.py --capture-baseline --save baseline.json
    python scripts/performance_monitor.py --benchmark --compare-baseline baseline.json
    python scripts/performance_monitor.py --continuous-monitor --duration 3600
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import optional dependencies with fallbacks
try:
    import psutil
except ImportError:
    print("Warning: psutil not installed. Using fallback system monitoring.")
    psutil = None

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not installed. Using fallback statistics.")
    np = None


class PerformanceMonitor:
    """Comprehensive performance monitoring for remediation work."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance monitor with configuration."""
        self.config = config or self._load_default_config()
        self.start_time = time.time()
        self.metrics: Dict[str, List[float]] = {
            'cpu_percent': [],
            'memory_mb': [], 
            'memory_percent': [],
            'disk_io_read_mb': [],
            'disk_io_write_mb': [],
            'network_sent_mb': [],
            'network_recv_mb': [],
            'test_execution_times': [],
            'build_durations': [],
            'response_times': []
        }
        self.process_metrics: Dict[int, Dict[str, List[float]]] = {}
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default monitoring configuration."""
        return {
            'sampling_interval': 1.0,
            'memory_threshold_mb': 1500,
            'cpu_threshold_percent': 80.0,
            'response_time_threshold_ms': 2000,
            'test_timeout_seconds': 1800,
            'build_timeout_seconds': 300,
            'thresholds': {
                'response_time_p95': 2.0,
                'memory_usage_peak': 1.5e9,
                'cpu_utilization_avg': 0.7,
                'error_rate': 0.001,
                'throughput_min': 100,
                'test_execution_time': 1800,
                'build_time': 300
            }
        }
    
    def capture_system_metrics(self) -> Dict[str, float]:
        """Capture current system performance metrics."""
        try:
            if psutil:
                # Full metrics with psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 ** 2)
                memory_percent = memory.percent
                
                disk_io = psutil.disk_io_counters()
                disk_read_mb = disk_io.read_bytes / (1024 ** 2) if disk_io else 0
                disk_write_mb = disk_io.write_bytes / (1024 ** 2) if disk_io else 0
                
                network_io = psutil.net_io_counters()
                network_sent_mb = network_io.bytes_sent / (1024 ** 2) if network_io else 0
                network_recv_mb = network_io.bytes_recv / (1024 ** 2) if network_io else 0
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': memory_percent,
                    'disk_io_read_mb': disk_read_mb,
                    'disk_io_write_mb': disk_write_mb,
                    'network_sent_mb': network_sent_mb,
                    'network_recv_mb': network_recv_mb,
                    'available_memory_mb': memory.available / (1024 ** 2),
                    'disk_usage_percent': psutil.disk_usage('/').percent
                }
            else:
                # Fallback metrics without psutil
                metrics = self._get_fallback_metrics()
            
            # Store metrics for trend analysis
            for key, value in metrics.items():
                if key != 'timestamp' and key in self.metrics:
                    self.metrics[key].append(value)
            
            return metrics
            
        except Exception as e:
            print(f"Error capturing system metrics: {e}")
            return self._get_fallback_metrics()
    
    def _get_fallback_metrics(self) -> Dict[str, float]:
        """Get basic system metrics without psutil."""
        import platform
        import resource
        
        try:
            # Get basic memory info from /proc/meminfo on Linux
            memory_info = {}
            if platform.system() == 'Linux':
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2:
                                key = parts[0].rstrip(':')
                                value = int(parts[1]) * 1024  # Convert KB to bytes
                                memory_info[key] = value
                except:
                    pass
            
            # Get process resource usage
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': 0.0,  # Cannot determine without psutil
                'memory_mb': rusage.ru_maxrss / 1024.0 if platform.system() != 'Darwin' else rusage.ru_maxrss / (1024 * 1024),
                'memory_percent': 0.0,  # Cannot determine without psutil
                'disk_io_read_mb': 0.0,
                'disk_io_write_mb': 0.0, 
                'network_sent_mb': 0.0,
                'network_recv_mb': 0.0,
                'available_memory_mb': memory_info.get('MemAvailable', 0) / (1024 ** 2),
                'disk_usage_percent': 0.0
            }
            
            return metrics
        except:
            return {
                'timestamp': time.time(),
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'memory_percent': 0.0,
                'disk_io_read_mb': 0.0,
                'disk_io_write_mb': 0.0,
                'network_sent_mb': 0.0, 
                'network_recv_mb': 0.0,
                'available_memory_mb': 0.0,
                'disk_usage_percent': 0.0
            }
    
    def monitor_test_execution(self, test_command: str) -> Dict[str, Any]:
        """Monitor performance during test execution."""
        print(f"Monitoring test execution: {test_command}")
        
        start_time = time.time()
        metrics_before = self.capture_system_metrics()
        
        # Start monitoring in background
        monitoring_active = True
        test_metrics = []
        
        def background_monitor():
            while monitoring_active:
                test_metrics.append(self.capture_system_metrics())
                time.sleep(self.config['sampling_interval'])
        
        # Start background monitoring
        import threading
        monitor_thread = threading.Thread(target=background_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # Execute test command
            result = subprocess.run(
                test_command.split(),
                capture_output=True,
                text=True,
                timeout=self.config['test_timeout_seconds']
            )
            
            execution_time = time.time() - start_time
            monitoring_active = False
            monitor_thread.join(timeout=5)
            
            metrics_after = self.capture_system_metrics()
            
            # Calculate performance deltas
            memory_delta = metrics_after.get('memory_mb', 0) - metrics_before.get('memory_mb', 0)
            cpu_max = max([m.get('cpu_percent', 0) for m in test_metrics]) if test_metrics else 0
            memory_max = max([m.get('memory_mb', 0) for m in test_metrics]) if test_metrics else 0
            
            self.metrics['test_execution_times'].append(execution_time)
            
            return {
                'command': test_command,
                'execution_time': execution_time,
                'return_code': result.returncode,
                'memory_delta_mb': memory_delta,
                'peak_cpu_percent': cpu_max,
                'peak_memory_mb': memory_max,
                'metrics_timeline': test_metrics,
                'stdout_lines': len(result.stdout.splitlines()),
                'stderr_lines': len(result.stderr.splitlines()),
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            monitoring_active = False
            execution_time = time.time() - start_time
            return {
                'command': test_command,
                'execution_time': execution_time,
                'timeout': True,
                'success': False,
                'error': 'Test execution timeout'
            }
        except Exception as e:
            monitoring_active = False
            return {
                'command': test_command,
                'error': str(e),
                'success': False
            }
    
    def monitor_build_process(self, build_command: str) -> Dict[str, Any]:
        """Monitor performance during build process."""
        print(f"Monitoring build process: {build_command}")
        
        start_time = time.time()
        metrics_before = self.capture_system_metrics()
        
        try:
            result = subprocess.run(
                build_command.split(),
                capture_output=True,
                text=True,
                timeout=self.config['build_timeout_seconds']
            )
            
            build_time = time.time() - start_time
            metrics_after = self.capture_system_metrics()
            
            self.metrics['build_durations'].append(build_time)
            
            return {
                'command': build_command,
                'build_time': build_time,
                'return_code': result.returncode,
                'memory_delta_mb': metrics_after.get('memory_mb', 0) - metrics_before.get('memory_mb', 0),
                'success': result.returncode == 0,
                'stdout_lines': len(result.stdout.splitlines()),
                'stderr_lines': len(result.stderr.splitlines())
            }
            
        except subprocess.TimeoutExpired:
            build_time = time.time() - start_time
            return {
                'command': build_command,
                'build_time': build_time,
                'timeout': True,
                'success': False,
                'error': 'Build timeout'
            }
        except Exception as e:
            return {
                'command': build_command,
                'error': str(e),
                'success': False
            }
    
    def run_performance_suite(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite."""
        suite_results = {
            'suite_start_time': datetime.now(timezone.utc).isoformat(),
            'system_info': self._get_system_info(),
            'baseline_metrics': self.capture_system_metrics(),
            'test_results': [],
            'build_results': [],
            'overall_metrics': {}
        }
        
        # Test commands to monitor
        test_commands = [
            "python -m pytest egw_query_expansion/tests/unit/ -v --tb=short",
            "python -m pytest egw_query_expansion/tests/integration/ -v",
            "python validate_installation.py",
            "python scripts/health_check.py --full-system"
        ]
        
        # Build commands to monitor  
        build_commands = [
            "python -m pip install -e .",
            "python -m flake8 egw_query_expansion/",
            "python -m mypy egw_query_expansion/"
        ]
        
        print("Starting comprehensive performance monitoring suite...")
        
        # Monitor test executions
        for test_cmd in test_commands:
            if self._command_exists(test_cmd.split()[0]):
                result = self.monitor_test_execution(test_cmd)
                suite_results['test_results'].append(result)
                print(f"Test completed: {test_cmd[:50]}... ({result.get('execution_time', 0):.2f}s)")
            else:
                print(f"Skipping test (command not found): {test_cmd}")
        
        # Monitor build processes
        for build_cmd in build_commands:
            if self._command_exists(build_cmd.split()[0]):
                result = self.monitor_build_process(build_cmd)
                suite_results['build_results'].append(result)
                print(f"Build completed: {build_cmd[:50]}... ({result.get('build_time', 0):.2f}s)")
            else:
                print(f"Skipping build (command not found): {build_cmd}")
        
        # Calculate overall metrics
        suite_results['overall_metrics'] = self._calculate_summary_metrics()
        suite_results['suite_end_time'] = datetime.now(timezone.utc).isoformat()
        suite_results['total_duration'] = time.time() - self.start_time
        
        return suite_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for context."""
        if psutil:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024 ** 3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024 ** 3),
                'python_version': sys.version,
                'platform': sys.platform,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        else:
            import os
            import platform
            return {
                'cpu_count': os.cpu_count() or 1,
                'memory_total_gb': 0.0,  # Cannot determine without psutil
                'disk_total_gb': 0.0,    # Cannot determine without psutil
                'python_version': sys.version,
                'platform': platform.system(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in the system."""
        try:
            subprocess.run([command, '--version'], capture_output=True, timeout=10)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _calculate_summary_metrics(self) -> Dict[str, float]:
        """Calculate summary statistics from collected metrics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                if np:
                    # Use numpy for accurate statistics
                    summary[f"{metric_name}_avg"] = float(np.mean(values))
                    summary[f"{metric_name}_max"] = float(np.max(values))
                    summary[f"{metric_name}_min"] = float(np.min(values))
                    summary[f"{metric_name}_p95"] = float(np.percentile(values, 95))
                    summary[f"{metric_name}_std"] = float(np.std(values))
                else:
                    # Fallback to basic Python statistics
                    summary[f"{metric_name}_avg"] = sum(values) / len(values)
                    summary[f"{metric_name}_max"] = max(values)
                    summary[f"{metric_name}_min"] = min(values)
                    # Simple approximation for 95th percentile
                    sorted_values = sorted(values)
                    p95_index = int(0.95 * len(sorted_values))
                    summary[f"{metric_name}_p95"] = sorted_values[min(p95_index, len(sorted_values) - 1)]
                    # Simple standard deviation approximation
                    avg = summary[f"{metric_name}_avg"]
                    variance = sum((x - avg) ** 2 for x in values) / len(values)
                    summary[f"{metric_name}_std"] = variance ** 0.5
        
        return summary
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
                
            current_metrics = self._calculate_summary_metrics()
            baseline_metrics = baseline_data.get('overall_metrics', {})
            
            comparison = {
                'baseline_file': baseline_file,
                'comparison_time': datetime.now(timezone.utc).isoformat(),
                'metrics_comparison': {},
                'regressions': [],
                'improvements': [],
                'overall_status': 'unknown'
            }
            
            for metric, current_value in current_metrics.items():
                if metric in baseline_metrics:
                    baseline_value = baseline_metrics[metric]
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    
                    comparison['metrics_comparison'][metric] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'change_percent': change_percent,
                        'change_absolute': current_value - baseline_value
                    }
                    
                    # Identify regressions (performance got worse)
                    if self._is_regression(metric, change_percent):
                        comparison['regressions'].append({
                            'metric': metric,
                            'change_percent': change_percent,
                            'severity': self._get_regression_severity(metric, change_percent)
                        })
                    
                    # Identify improvements  
                    elif self._is_improvement(metric, change_percent):
                        comparison['improvements'].append({
                            'metric': metric,
                            'change_percent': change_percent
                        })
            
            # Overall status assessment
            critical_regressions = [r for r in comparison['regressions'] if r['severity'] == 'critical']
            high_regressions = [r for r in comparison['regressions'] if r['severity'] == 'high']
            
            if critical_regressions:
                comparison['overall_status'] = 'critical_regression'
            elif high_regressions:
                comparison['overall_status'] = 'regression_detected'
            elif comparison['regressions']:
                comparison['overall_status'] = 'minor_regression'
            else:
                comparison['overall_status'] = 'acceptable'
            
            return comparison
            
        except Exception as e:
            return {
                'error': f"Failed to compare with baseline: {e}",
                'baseline_file': baseline_file
            }
    
    def _is_regression(self, metric: str, change_percent: float) -> bool:
        """Determine if a metric change represents a regression."""
        # Metrics where increase is bad
        increase_bad_metrics = [
            'memory_mb', 'memory_percent', 'cpu_percent', 
            'test_execution_times', 'build_durations', 'response_times'
        ]
        
        # Metrics where decrease is bad
        decrease_bad_metrics = ['throughput']
        
        if any(bad_metric in metric for bad_metric in increase_bad_metrics):
            return change_percent > 5  # 5% increase is regression
        elif any(bad_metric in metric for bad_metric in decrease_bad_metrics):
            return change_percent < -5  # 5% decrease is regression
        
        return False
    
    def _is_improvement(self, metric: str, change_percent: float) -> bool:
        """Determine if a metric change represents an improvement."""
        # Metrics where decrease is good
        decrease_good_metrics = [
            'memory_mb', 'memory_percent', 'cpu_percent',
            'test_execution_times', 'build_durations', 'response_times'
        ]
        
        # Metrics where increase is good
        increase_good_metrics = ['throughput']
        
        if any(good_metric in metric for good_metric in decrease_good_metrics):
            return change_percent < -5  # 5% decrease is improvement
        elif any(good_metric in metric for good_metric in increase_good_metrics):
            return change_percent > 5  # 5% increase is improvement
        
        return False
    
    def _get_regression_severity(self, metric: str, change_percent: float) -> str:
        """Determine severity of performance regression."""
        if abs(change_percent) > 50:
            return 'critical'
        elif abs(change_percent) > 25:
            return 'high'
        elif abs(change_percent) > 10:
            return 'medium'
        else:
            return 'low'
    
    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate human-readable performance report."""
        report_lines = [
            "# Performance Monitoring Report",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Duration: {results.get('total_duration', 0):.2f} seconds",
            ""
        ]
        
        # System Information
        system_info = results.get('system_info', {})
        report_lines.extend([
            "## System Information",
            f"- CPU Cores: {system_info.get('cpu_count', 'unknown')}",
            f"- Total Memory: {system_info.get('memory_total_gb', 0):.2f} GB", 
            f"- Platform: {system_info.get('platform', 'unknown')}",
            f"- Python: {system_info.get('python_version', 'unknown')}",
            ""
        ])
        
        # Test Results Summary
        test_results = results.get('test_results', [])
        total_test_time = sum(t.get('execution_time', 0) for t in test_results)
        if test_results:
            successful_tests = [t for t in test_results if t.get('success', False)]
            
            report_lines.extend([
                "## Test Execution Summary",
                f"- Total Tests: {len(test_results)}",
                f"- Successful: {len(successful_tests)}",
                f"- Failed: {len(test_results) - len(successful_tests)}",
                f"- Total Test Time: {total_test_time:.2f} seconds",
                f"- Average Test Time: {total_test_time / len(test_results):.2f} seconds",
                ""
            ])
        else:
            report_lines.extend([
                "## Test Execution Summary",
                "- No test results available (commands not found or skipped)",
                ""
            ])
        
        # Build Results Summary  
        build_results = results.get('build_results', [])
        total_build_time = sum(b.get('build_time', 0) for b in build_results)
        if build_results:
            successful_builds = [b for b in build_results if b.get('success', False)]
            
            report_lines.extend([
                "## Build Process Summary", 
                f"- Total Builds: {len(build_results)}",
                f"- Successful: {len(successful_builds)}",
                f"- Failed: {len(build_results) - len(successful_builds)}",
                f"- Total Build Time: {total_build_time:.2f} seconds",
                f"- Average Build Time: {total_build_time / len(build_results):.2f} seconds",
                ""
            ])
        else:
            report_lines.extend([
                "## Build Process Summary", 
                "- No build results available (commands not found or skipped)",
                ""
            ])
        
        # Performance Metrics Summary
        overall_metrics = results.get('overall_metrics', {})
        if overall_metrics:
            report_lines.extend([
                "## Performance Metrics",
                f"- Peak Memory Usage: {overall_metrics.get('memory_mb_max', 0):.2f} MB",
                f"- Average CPU Usage: {overall_metrics.get('cpu_percent_avg', 0):.2f}%",
                f"- Peak CPU Usage: {overall_metrics.get('cpu_percent_max', 0):.2f}%",
                ""
            ])
        
        # Threshold Validation
        thresholds = self.config['thresholds']
        violations = []
        
        if overall_metrics.get('memory_mb_max', 0) * 1024 * 1024 > thresholds['memory_usage_peak']:
            violations.append("Memory usage exceeded threshold")
        
        if overall_metrics.get('cpu_percent_avg', 0) / 100 > thresholds['cpu_utilization_avg']:
            violations.append("CPU utilization exceeded threshold")
        
        if total_test_time > thresholds['test_execution_time']:
            violations.append("Test execution time exceeded threshold")
        
        if violations:
            report_lines.extend([
                "## ⚠️ Threshold Violations",
                *[f"- {violation}" for violation in violations],
                ""
            ])
        else:
            report_lines.extend([
                "## ✅ All Thresholds Met",
                "No performance threshold violations detected.",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"Report saved to: {output_file}")
        
        return report_content


def main():
    """Main entry point for performance monitoring."""
    parser = argparse.ArgumentParser(
        description="Performance monitoring for EGW Query Expansion System remediation"
    )
    parser.add_argument('--capture-baseline', action='store_true',
                        help='Capture baseline performance metrics')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark suite')
    parser.add_argument('--compare-baseline', type=str,
                        help='Compare current performance with baseline file')
    parser.add_argument('--continuous-monitor', action='store_true',
                        help='Run continuous monitoring')
    parser.add_argument('--duration', type=int, default=3600,
                        help='Duration for continuous monitoring (seconds)')
    parser.add_argument('--save', type=str,
                        help='Save results to JSON file')
    parser.add_argument('--report', type=str,
                        help='Generate report to file')
    parser.add_argument('--config', type=str,
                        help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    monitor = PerformanceMonitor(config)
    
    try:
        if args.capture_baseline or args.benchmark:
            print("Running performance benchmark suite...")
            results = monitor.run_performance_suite()
            
            # Save results if requested
            if args.save:
                os.makedirs(os.path.dirname(args.save) if os.path.dirname(args.save) else '.', exist_ok=True)
                with open(args.save, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Results saved to: {args.save}")
            
            # Generate report if requested
            if args.report:
                monitor.generate_report(results, args.report)
            else:
                print("\n" + monitor.generate_report(results))
                
        elif args.compare_baseline:
            if not os.path.exists(args.compare_baseline):
                print(f"Baseline file not found: {args.compare_baseline}")
                sys.exit(1)
                
            print("Running current performance suite for comparison...")
            current_results = monitor.run_performance_suite()
            
            print("Comparing with baseline...")
            comparison = monitor.compare_with_baseline(args.compare_baseline)
            
            # Print comparison summary
            print(f"\n# Baseline Comparison Results")
            print(f"Status: {comparison['overall_status']}")
            print(f"Regressions found: {len(comparison['regressions'])}")
            print(f"Improvements found: {len(comparison['improvements'])}")
            
            if comparison['regressions']:
                print("\n## Regressions:")
                for reg in comparison['regressions']:
                    print(f"- {reg['metric']}: {reg['change_percent']:.2f}% ({reg['severity']})")
            
            if comparison['improvements']:
                print("\n## Improvements:")
                for imp in comparison['improvements']:
                    print(f"- {imp['metric']}: {imp['change_percent']:.2f}%")
            
            # Save comparison if requested
            if args.save:
                comparison_file = args.save.replace('.json', '_comparison.json')
                with open(comparison_file, 'w') as f:
                    json.dump(comparison, f, indent=2, default=str)
                print(f"Comparison saved to: {comparison_file}")
                
        elif args.continuous_monitor:
            print(f"Starting continuous monitoring for {args.duration} seconds...")
            end_time = time.time() + args.duration
            
            samples = []
            while time.time() < end_time:
                sample = monitor.capture_system_metrics()
                samples.append(sample)
                print(f"Sample: CPU {sample.get('cpu_percent', 0):.1f}%, "
                      f"Memory {sample.get('memory_mb', 0):.1f}MB", end='\r')
                time.sleep(monitor.config['sampling_interval'])
            
            # Generate summary
            results = {
                'monitoring_type': 'continuous',
                'duration': args.duration,
                'samples': samples,
                'sample_count': len(samples),
                'overall_metrics': monitor._calculate_summary_metrics()
            }
            
            if args.save:
                with open(args.save, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nContinuous monitoring results saved to: {args.save}")
            
            print(f"\nMonitoring complete. Captured {len(samples)} samples.")
            
        else:
            print("Please specify an operation: --capture-baseline, --benchmark, --compare-baseline, or --continuous-monitor")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()