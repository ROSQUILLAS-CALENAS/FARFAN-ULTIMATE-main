#!/usr/bin/env python3
"""
Integration Test Runner for EGW Query Expansion Pipeline

Runs comprehensive integration tests including municipal development plans
processing, performance benchmarks, and deployment validation.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil

import pytest
import docker
import redis


class IntegrationTestRunner:
    """Comprehensive integration test runner"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.test_results = {}
        self.docker_client = None
        self.redis_client = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("IntegrationTestRunner")
        
        # Test environment setup
        self.test_env = {
            'PYTHONPATH': str(Path.cwd()),
            'REDIS_URL': self.config.get('redis_url', 'redis://localhost:6379'),
            'TEST_DATA_DIR': str(Path.cwd() / 'tests' / 'municipal_data'),
            'BENCHMARK_MODE': 'true'
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test configuration"""
        default_config = {
            'redis_url': 'redis://localhost:6379',
            'docker_compose_file': 'deployment/docker-compose.yml',
            'kubernetes_manifests_dir': 'deployment/kubernetes',
            'benchmark_requirements': {
                'documents_per_hour': 170,
                'max_processing_time': 21.2,
                'min_quality_score': 0.8,
                'min_consistency_score': 0.75
            },
            'test_timeout': 1800,  # 30 minutes
            'parallel_workers': 4
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        self.logger.info("Starting comprehensive integration test suite")
        
        test_suite = [
            ("unit_tests", self.run_unit_tests),
            ("docker_build", self.test_docker_build),
            ("municipal_plans_integration", self.run_municipal_plans_tests),
            ("performance_benchmarks", self.run_performance_benchmarks),
            ("distributed_processing", self.test_distributed_processing),
            ("deployment_validation", self.test_deployment_configurations),
            ("quality_validation", self.test_quality_validation),
            ("consistency_validation", self.test_consistency_validation)
        ]
        
        overall_results = {
            'start_time': time.time(),
            'tests': {},
            'summary': {},
            'errors': []
        }
        
        for test_name, test_func in test_suite:
            self.logger.info(f"Running {test_name}...")
            
            try:
                start_time = time.time()
                result = await test_func()
                execution_time = time.time() - start_time
                
                overall_results['tests'][test_name] = {
                    'status': 'passed' if result.get('success', False) else 'failed',
                    'execution_time': execution_time,
                    'details': result
                }
                
                self.logger.info(f"{test_name}: {'PASSED' if result.get('success', False) else 'FAILED'} ({execution_time:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"{test_name} failed with exception: {e}")
                overall_results['tests'][test_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                overall_results['errors'].append(f"{test_name}: {str(e)}")
        
        # Generate summary
        overall_results['end_time'] = time.time()
        overall_results['total_time'] = overall_results['end_time'] - overall_results['start_time']
        
        passed_tests = [name for name, result in overall_results['tests'].items() 
                       if result['status'] == 'passed']
        failed_tests = [name for name, result in overall_results['tests'].items() 
                       if result['status'] in ['failed', 'error']]
        
        overall_results['summary'] = {
            'total_tests': len(test_suite),
            'passed': len(passed_tests),
            'failed': len(failed_tests),
            'success_rate': len(passed_tests) / len(test_suite),
            'overall_success': len(failed_tests) == 0
        }
        
        return overall_results
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests with pytest"""
        self.logger.info("Running unit tests...")
        
        # Run pytest with coverage
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/',
            '-v',
            '--tb=short',
            '--cov=egw_query_expansion',
            '--cov-report=html:coverage_html',
            '--cov-report=json:coverage.json',
            '--junit-xml=test-results.xml'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config['test_timeout'],
                env={**os.environ, **self.test_env}
            )
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Unit tests timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to run unit tests: {str(e)}"
            }
    
    async def test_docker_build(self) -> Dict[str, Any]:
        """Test Docker build process"""
        self.logger.info("Testing Docker build...")
        
        try:
            self.docker_client = docker.from_env()
            
            # Build Docker image
            image, build_logs = self.docker_client.images.build(
                path='.',
                dockerfile='Dockerfile',
                tag='egw-query-expansion:test',
                rm=True,
                forcerm=True
            )
            
            # Test basic container functionality
            container = self.docker_client.containers.run(
                'egw-query-expansion:test',
                'python --version',
                remove=True,
                detach=False
            )
            
            return {
                'success': True,
                'image_id': image.id,
                'image_size': image.attrs['Size'],
                'build_logs': [log for log in build_logs]
            }
            
        except docker.errors.BuildError as e:
            return {
                'success': False,
                'error': f"Docker build failed: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Docker test failed: {str(e)}"
            }
    
    async def run_municipal_plans_tests(self) -> Dict[str, Any]:
        """Run municipal development plans integration tests"""
        self.logger.info("Running municipal development plans tests...")
        
        try:
            # Run specific municipal plans integration tests
            cmd = [
                sys.executable, '-m', 'pytest',
                'tests/integration/test_municipal_plans_integration.py',
                '-v',
                '--tb=short',
                '-m', 'not skip'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config['test_timeout'],
                env={**os.environ, **self.test_env}
            )
            
            success = result.returncode == 0
            
            # Parse test output for specific metrics
            metrics = self._parse_test_metrics(result.stdout)
            
            return {
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'metrics': metrics
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Municipal plans tests timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Municipal plans tests failed: {str(e)}"
            }
    
    async def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmark tests"""
        self.logger.info("Running performance benchmarks...")
        
        try:
            # Run benchmark tests
            cmd = [
                sys.executable, '-m', 'pytest',
                'tests/performance/test_benchmarks.py',
                '-v',
                '--tb=short',
                '-m', 'benchmark'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config['test_timeout'],
                env={**os.environ, **self.test_env}
            )
            
            success = result.returncode == 0
            
            # Parse benchmark results
            benchmark_results = self._parse_benchmark_results(result.stdout)
            
            # Validate against requirements
            requirements_met = self._validate_benchmark_requirements(benchmark_results)
            
            return {
                'success': success and requirements_met,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'benchmark_results': benchmark_results,
                'requirements_met': requirements_met
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Performance benchmarks failed: {str(e)}"
            }
    
    async def test_distributed_processing(self) -> Dict[str, Any]:
        """Test distributed processing coordination"""
        self.logger.info("Testing distributed processing...")
        
        try:
            # Start Redis for coordination
            await self._start_test_redis()
            
            # Test distributed processor initialization
            from distributed_processor import DistributedProcessor
            
            processor = DistributedProcessor(
                worker_id="integration-test-worker",
                redis_url=self.config['redis_url']
            )
            
            # Test basic coordination functionality
            test_documents = [
                'tests/municipal_data/sample_doc_1.txt',
                'tests/municipal_data/sample_doc_2.txt'
            ]
            
            # Create test documents if they don't exist
            self._create_test_documents(test_documents)
            
            # Test batch processing
            request_id = "integration-test-batch"
            query = "What are the development requirements?"
            
            # This would normally be distributed, but we test coordination logic
            processing_results = []
            
            for i, doc_path in enumerate(test_documents):
                from distributed_processor import ProcessingTask
                task = ProcessingTask(
                    task_id=f"dist-test-{i}",
                    document_path=doc_path,
                    query=query
                )
                
                # Simulate processing
                result_data = await processor._perform_egw_processing(task)
                processing_results.append(result_data)
            
            success = len(processing_results) == len(test_documents)
            
            return {
                'success': success,
                'processed_documents': len(processing_results),
                'results': processing_results[:2]  # Limit output size
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Distributed processing test failed: {str(e)}"
            }
        finally:
            await self._stop_test_redis()
    
    async def test_deployment_configurations(self) -> Dict[str, Any]:
        """Test deployment configurations"""
        self.logger.info("Testing deployment configurations...")
        
        results = {}
        
        # Test Docker Compose configuration
        try:
            compose_file = Path(self.config['docker_compose_file'])
            if compose_file.exists():
                # Validate Docker Compose file
                cmd = ['docker-compose', '-f', str(compose_file), 'config']
                result = subprocess.run(cmd, capture_output=True, text=True)
                results['docker_compose'] = {
                    'valid': result.returncode == 0,
                    'error': result.stderr if result.returncode != 0 else None
                }
            else:
                results['docker_compose'] = {
                    'valid': False,
                    'error': 'Docker Compose file not found'
                }
        except Exception as e:
            results['docker_compose'] = {
                'valid': False,
                'error': str(e)
            }
        
        # Test Kubernetes manifests
        try:
            k8s_dir = Path(self.config['kubernetes_manifests_dir'])
            if k8s_dir.exists():
                yaml_files = list(k8s_dir.glob('*.yaml'))
                k8s_results = {}
                
                for yaml_file in yaml_files:
                    try:
                        cmd = ['kubectl', 'apply', '--dry-run=client', '-f', str(yaml_file)]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        k8s_results[yaml_file.name] = {
                            'valid': result.returncode == 0,
                            'error': result.stderr if result.returncode != 0 else None
                        }
                    except Exception as e:
                        k8s_results[yaml_file.name] = {
                            'valid': False,
                            'error': str(e)
                        }
                
                results['kubernetes'] = k8s_results
            else:
                results['kubernetes'] = {
                    'valid': False,
                    'error': 'Kubernetes manifests directory not found'
                }
        except Exception as e:
            results['kubernetes'] = {
                'valid': False,
                'error': str(e)
            }
        
        # Overall success
        docker_success = results.get('docker_compose', {}).get('valid', False)
        k8s_success = all(
            manifest.get('valid', False) 
            for manifest in results.get('kubernetes', {}).values()
            if isinstance(manifest, dict)
        ) if isinstance(results.get('kubernetes'), dict) else False
        
        return {
            'success': docker_success and k8s_success,
            'details': results
        }
    
    async def test_quality_validation(self) -> Dict[str, Any]:
        """Test quality validation mechanisms"""
        self.logger.info("Testing quality validation...")
        
        try:
            from distributed_processor import QualityValidator, ProcessingResult
            
            validator = QualityValidator(self.config['benchmark_requirements'])
            
            # Test with mock high-quality result
            high_quality_result = ProcessingResult(
                task_id="quality-test-high",
                worker_id="test-worker",
                status="completed",
                result_data={
                    'content': 'Comprehensive municipal development requires compliance with zoning regulations, environmental assessments, and permit procedures.',
                    'evidence': ['Zoning regulation citation', 'Environmental requirement', 'Permit procedure'],
                    'summary': 'Development requires regulatory compliance',
                    'metadata': {'source': 'municipal_code'}
                },
                processing_time=5.0,
                quality_metrics={}
            )
            
            high_quality_metrics = validator.validate_result(
                high_quality_result, 
                "What are municipal development requirements?"
            )
            
            # Test with mock low-quality result
            low_quality_result = ProcessingResult(
                task_id="quality-test-low",
                worker_id="test-worker",
                status="completed",
                result_data={
                    'content': 'Some stuff',
                    'evidence': [],
                    'summary': '',
                    'metadata': {}
                },
                processing_time=1.0,
                quality_metrics={}
            )
            
            low_quality_metrics = validator.validate_result(
                low_quality_result,
                "What are municipal development requirements?"
            )
            
            # Validate that high-quality result passes and low-quality fails
            high_passes = high_quality_metrics.get('quality_passed', False)
            low_passes = low_quality_metrics.get('quality_passed', False)
            
            return {
                'success': high_passes and not low_passes,
                'high_quality_score': high_quality_metrics.get('overall_quality', 0.0),
                'low_quality_score': low_quality_metrics.get('overall_quality', 0.0),
                'validation_working': high_passes != low_passes
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Quality validation test failed: {str(e)}"
            }
    
    async def test_consistency_validation(self) -> Dict[str, Any]:
        """Test result consistency validation"""
        self.logger.info("Testing consistency validation...")
        
        try:
            from distributed_processor import ResultAggregator, ProcessingResult
            
            aggregator = ResultAggregator(self.config['benchmark_requirements'])
            
            # Create multiple consistent results
            consistent_results = []
            base_content = "Municipal development requires zoning compliance and environmental review."
            
            for i in range(3):
                result = ProcessingResult(
                    task_id=f"consistency-test-{i}",
                    worker_id=f"worker-{i}",
                    status="completed",
                    result_data={
                        'content': base_content + f" Additional detail {i}.",
                        'evidence': ['Evidence A', 'Evidence B'],
                        'summary': f"Summary {i}",
                        'metadata': {'worker': i}
                    },
                    processing_time=2.0 + i * 0.1,
                    quality_metrics={
                        'overall_quality': 0.85 + i * 0.01,
                        'relevance_score': 0.8,
                        'coherence_score': 0.9
                    }
                )
                consistent_results.append(result)
            
            # Aggregate consistent results
            aggregated = aggregator.aggregate_results(consistent_results, "consistency-test")
            
            consistency_score = aggregated.consistency_score
            quality_score = aggregated.quality_score
            
            # Validate consistency meets requirements
            consistency_passes = consistency_score >= self.config['benchmark_requirements']['min_consistency_score']
            quality_passes = quality_score >= self.config['benchmark_requirements']['min_quality_score']
            
            return {
                'success': consistency_passes and quality_passes,
                'consistency_score': consistency_score,
                'quality_score': quality_score,
                'aggregated_results_count': len(aggregated.combined_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Consistency validation test failed: {str(e)}"
            }
    
    def _create_test_documents(self, document_paths: List[str]):
        """Create test documents for integration testing"""
        os.makedirs('tests/municipal_data', exist_ok=True)
        
        sample_content = """
        MUNICIPAL DEVELOPMENT REGULATIONS
        
        This document outlines requirements for municipal development including
        zoning regulations, environmental assessments, and permit procedures.
        
        ZONING REQUIREMENTS:
        - R-1 zones require minimum 8,000 sq ft lots
        - Commercial zones allow mixed-use development
        - Special use permits required for large projects
        
        ENVIRONMENTAL PROCEDURES:
        - Environmental impact assessment for projects over 1 acre
        - Stormwater management plans required
        - Tree preservation ordinances apply
        
        PERMIT PROCEDURES:
        - Building permits required for all construction
        - Plan review takes 15 business days
        - Special permits require public hearings
        """
        
        for doc_path in document_paths:
            os.makedirs(os.path.dirname(doc_path), exist_ok=True)
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)
    
    async def _start_test_redis(self):
        """Start Redis for testing"""
        try:
            self.redis_client = redis.from_url(self.config['redis_url'])
            self.redis_client.ping()
            self.logger.info("Connected to test Redis instance")
        except Exception as e:
            self.logger.warning(f"Could not connect to Redis: {e}")
    
    async def _stop_test_redis(self):
        """Stop test Redis connection"""
        if self.redis_client:
            self.redis_client.close()
    
    def _parse_test_metrics(self, test_output: str) -> Dict[str, Any]:
        """Parse test output for metrics"""
        # Simple parsing - in practice you'd use more sophisticated parsing
        metrics = {
            'passed': 'PASSED' in test_output,
            'failed': 'FAILED' in test_output,
            'errors': 'ERROR' in test_output
        }
        return metrics
    
    def _parse_benchmark_results(self, benchmark_output: str) -> Dict[str, Any]:
        """Parse benchmark test output"""
        # Extract benchmark metrics from output
        results = {
            'throughput': 0.0,
            'quality_score': 0.0,
            'consistency_score': 0.0,
            'processing_time': 0.0
        }
        
        # Simple parsing - would be more sophisticated in practice
        for line in benchmark_output.split('\n'):
            if 'documents/hour' in line:
                try:
                    value = float(line.split()[-2])
                    results['throughput'] = value
                except (ValueError, IndexError):
                    pass
            elif 'quality' in line.lower():
                try:
                    value = float(line.split()[-1])
                    results['quality_score'] = value
                except (ValueError, IndexError):
                    pass
        
        return results
    
    def _validate_benchmark_requirements(self, benchmark_results: Dict[str, Any]) -> bool:
        """Validate benchmark results against requirements"""
        requirements = self.config['benchmark_requirements']
        
        throughput_ok = benchmark_results.get('throughput', 0) >= requirements['documents_per_hour']
        quality_ok = benchmark_results.get('quality_score', 0) >= requirements['min_quality_score']
        consistency_ok = benchmark_results.get('consistency_score', 0) >= requirements['min_consistency_score']
        
        return throughput_ok and quality_ok and consistency_ok
    
    def generate_report(self, results: Dict[str, Any], output_path: str = "integration_test_results.json"):
        """Generate comprehensive test report"""
        self.logger.info(f"Generating test report: {output_path}")
        
        # Enhanced report with detailed analysis
        report = {
            'test_run_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_execution_time': results['total_time'],
                'configuration': self.config
            },
            'test_results': results['tests'],
            'summary': results['summary'],
            'benchmark_validation': self._validate_overall_benchmarks(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable summary
        self._print_test_summary(results)
    
    def _validate_overall_benchmarks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall system against benchmarks"""
        benchmark_validation = {
            'documents_per_hour_met': False,
            'quality_requirements_met': False,
            'consistency_requirements_met': False,
            'deployment_ready': False
        }
        
        # Check performance benchmarks
        perf_results = results['tests'].get('performance_benchmarks', {})
        if perf_results.get('status') == 'passed':
            benchmark_results = perf_results.get('details', {}).get('benchmark_results', {})
            benchmark_validation['documents_per_hour_met'] = (
                benchmark_results.get('throughput', 0) >= self.config['benchmark_requirements']['documents_per_hour']
            )
        
        # Check quality validation
        quality_results = results['tests'].get('quality_validation', {})
        if quality_results.get('status') == 'passed':
            benchmark_validation['quality_requirements_met'] = True
        
        # Check consistency validation
        consistency_results = results['tests'].get('consistency_validation', {})
        if consistency_results.get('status') == 'passed':
            benchmark_validation['consistency_requirements_met'] = True
        
        # Check deployment readiness
        deployment_results = results['tests'].get('deployment_validation', {})
        if deployment_results.get('status') == 'passed':
            benchmark_validation['deployment_ready'] = True
        
        return benchmark_validation
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        perf_results = results['tests'].get('performance_benchmarks', {})
        if perf_results.get('status') != 'passed':
            recommendations.append("Consider optimizing processing pipeline for better throughput")
            recommendations.append("Review resource allocation and scaling configuration")
        
        # Quality recommendations
        quality_results = results['tests'].get('quality_validation', {})
        if quality_results.get('status') != 'passed':
            recommendations.append("Review quality validation thresholds and metrics")
            recommendations.append("Consider improving result processing algorithms")
        
        # Deployment recommendations
        deployment_results = results['tests'].get('deployment_validation', {})
        if deployment_results.get('status') != 'passed':
            recommendations.append("Fix deployment configuration issues before production")
            recommendations.append("Validate Kubernetes manifests and Docker configurations")
        
        return recommendations
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print human-readable test summary"""
        print("\n" + "="*60)
        print("EGW QUERY EXPANSION INTEGRATION TEST SUMMARY")
        print("="*60)
        
        summary = results['summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Execution Time: {results['total_time']:.2f} seconds")
        
        print(f"\nOverall Result: {'✅ PASSED' if summary['overall_success'] else '❌ FAILED'}")
        
        if results.get('errors'):
            print("\nErrors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
        
        # Print individual test results
        print(f"\nDetailed Results:")
        for test_name, test_result in results['tests'].items():
            status = test_result['status']
            execution_time = test_result.get('execution_time', 0)
            emoji = "✅" if status == "passed" else "❌"
            print(f"  {emoji} {test_name}: {status.upper()} ({execution_time:.2f}s)")
        
        print("="*60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="EGW Integration Test Runner")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--output", type=str, default="integration_test_results.json",
                       help="Output report path")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test suite (skip performance benchmarks)")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IntegrationTestRunner(config_path=args.config)
    
    # Run tests
    try:
        results = await runner.run_all_tests()
        
        # Generate report
        runner.generate_report(results, args.output)
        
        # Exit with appropriate code
        sys.exit(0 if results['summary']['overall_success'] else 1)
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())