"""
Preflight Validation System
===========================

Performs comprehensive validation before enhancement activation including:
- Input schema verification
- Library version compatibility checks
- Threshold satisfaction validation
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pkg_resources
from packaging import version
from pydantic import BaseModel, ValidationError
import jsonschema
from jsonschema import validate


class ValidationResult(BaseModel):
    """Result of validation check"""
    passed: bool
    score: float
    details: Dict[str, Any]
    errors: List[str] = []
    warnings: List[str] = []
    timestamp: datetime = datetime.now()


class PreflightValidator:
    """Comprehensive preflight validation system"""
    
    def __init__(self, thresholds_path: str = "calibration_safety_governance/thresholds.json"):
        self.logger = logging.getLogger(__name__)
        self.thresholds_path = Path(thresholds_path)
        self.thresholds = self._load_thresholds()
        self.validation_schemas = self._load_validation_schemas()
        
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load unified thresholds configuration"""
        try:
            with open(self.thresholds_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Thresholds file not found: {self.thresholds_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in thresholds file: {e}")
            return {}
    
    def _load_validation_schemas(self) -> Dict[str, Dict]:
        """Load JSON schemas for input validation"""
        schemas = {
            "enhancement_request": {
                "type": "object",
                "properties": {
                    "enhancement_id": {"type": "string"},
                    "enhancement_type": {"type": "string", "enum": ["adaptive_scoring", "dynamic_thresholding", "enhanced_validation"]},
                    "configuration": {"type": "object"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "activation_criteria": {"type": "object"},
                    "metadata": {"type": "object"}
                },
                "required": ["enhancement_id", "enhancement_type", "configuration"],
                "additionalProperties": True
            },
            "system_state": {
                "type": "object",
                "properties": {
                    "performance_metrics": {"type": "object"},
                    "stability_indicators": {"type": "object"},
                    "active_enhancements": {"type": "array"},
                    "resource_utilization": {"type": "object"}
                },
                "required": ["performance_metrics", "stability_indicators"]
            }
        }
        return schemas

    def validate_input_schema(self, input_data: Dict[str, Any], schema_type: str) -> ValidationResult:
        """Validate input data against predefined schemas"""
        errors = []
        warnings = []
        
        try:
            schema = self.validation_schemas.get(schema_type)
            if not schema:
                errors.append(f"Unknown schema type: {schema_type}")
                return ValidationResult(passed=False, score=0.0, details={}, errors=errors)
            
            validate(instance=input_data, schema=schema)
            
            # Calculate compliance score based on field completeness
            required_fields = schema.get("required", [])
            present_fields = [field for field in required_fields if field in input_data]
            
            field_completeness = len(present_fields) / len(required_fields) if required_fields else 1.0
            
            # Check data type accuracy
            type_errors = 0
            total_checks = 0
            
            for field, field_schema in schema.get("properties", {}).items():
                if field in input_data:
                    total_checks += 1
                    expected_type = field_schema.get("type")
                    if expected_type and not self._check_type_match(input_data[field], expected_type):
                        type_errors += 1
                        warnings.append(f"Type mismatch for field {field}: expected {expected_type}")
            
            type_accuracy = (total_checks - type_errors) / total_checks if total_checks > 0 else 1.0
            
            # Calculate overall score
            score = (field_completeness * 0.6 + type_accuracy * 0.4)
            
            thresholds = self.thresholds.get("validation", {}).get("schema_compliance", {})
            min_fields_threshold = thresholds.get("minimum_fields_present", 0.95)
            type_accuracy_threshold = thresholds.get("data_type_accuracy", 0.98)
            
            passed = (field_completeness >= min_fields_threshold and 
                     type_accuracy >= type_accuracy_threshold)
            
            details = {
                "field_completeness": field_completeness,
                "type_accuracy": type_accuracy,
                "total_fields": len(schema.get("properties", {})),
                "present_required_fields": len(present_fields),
                "required_fields": len(required_fields)
            }
            
            return ValidationResult(
                passed=passed,
                score=score,
                details=details,
                errors=errors,
                warnings=warnings
            )
            
        except ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            return ValidationResult(passed=False, score=0.0, details={}, errors=errors)
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
            return ValidationResult(passed=False, score=0.0, details={}, errors=errors)

    def _check_type_match(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid
            
        return isinstance(value, expected_python_type)

    def check_library_compatibility(self, requirements_path: str = "requirements.txt") -> ValidationResult:
        """Check library version compatibility against requirements"""
        errors = []
        warnings = []
        
        try:
            requirements = self._parse_requirements(requirements_path)
            compatibility_issues = 0
            total_dependencies = len(requirements)
            critical_mismatches = 0
            
            for package_name, version_spec in requirements.items():
                try:
                    installed_version = pkg_resources.get_distribution(package_name).version
                    
                    if version_spec and not self._check_version_compatibility(installed_version, version_spec):
                        compatibility_issues += 1
                        severity = self._assess_compatibility_severity(package_name)
                        
                        if severity == "critical":
                            critical_mismatches += 1
                            errors.append(f"Critical version mismatch: {package_name} {installed_version} vs {version_spec}")
                        else:
                            warnings.append(f"Version mismatch: {package_name} {installed_version} vs {version_spec}")
                            
                except pkg_resources.DistributionNotFound:
                    errors.append(f"Required package not installed: {package_name}")
                    compatibility_issues += 1
                    critical_mismatches += 1
                    
            # Calculate compatibility score
            compatibility_score = (total_dependencies - compatibility_issues) / total_dependencies if total_dependencies > 0 else 1.0
            
            thresholds = self.thresholds.get("validation", {}).get("library_compatibility", {})
            tolerance_threshold = thresholds.get("version_mismatch_tolerance", 0.1)
            critical_threshold = thresholds.get("critical_dependency_alignment", 1.0)
            
            passed = (compatibility_score >= (1.0 - tolerance_threshold) and 
                     critical_mismatches == 0)
            
            details = {
                "total_dependencies": total_dependencies,
                "compatibility_issues": compatibility_issues,
                "critical_mismatches": critical_mismatches,
                "compatibility_score": compatibility_score
            }
            
            return ValidationResult(
                passed=passed,
                score=compatibility_score,
                details=details,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            errors.append(f"Library compatibility check failed: {str(e)}")
            return ValidationResult(passed=False, score=0.0, details={}, errors=errors)

    def _parse_requirements(self, requirements_path: str) -> Dict[str, Optional[str]]:
        """Parse requirements.txt file"""
        requirements = {}
        
        try:
            with open(requirements_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package name and version specification
                        match = re.match(r'^([a-zA-Z0-9_-]+[a-zA-Z0-9_.-]*?)([><=!]+.*)?$', line)
                        if match:
                            package_name = match.group(1)
                            version_spec = match.group(2)
                            requirements[package_name] = version_spec
                            
        except FileNotFoundError:
            self.logger.warning(f"Requirements file not found: {requirements_path}")
            
        return requirements

    def _check_version_compatibility(self, installed: str, spec: str) -> bool:
        """Check if installed version satisfies requirement specification"""
        try:
            installed_version = version.parse(installed)
            
            # Parse version specification
            operators = re.findall(r'[><=!]+', spec)
            versions = re.findall(r'[\d.]+', spec)
            
            if not operators or not versions:
                return True  # No specific version required
                
            for op, ver in zip(operators, versions):
                required_version = version.parse(ver)
                
                if op == ">=" and installed_version < required_version:
                    return False
                elif op == ">" and installed_version <= required_version:
                    return False
                elif op == "<=" and installed_version > required_version:
                    return False
                elif op == "<" and installed_version >= required_version:
                    return False
                elif op == "==" and installed_version != required_version:
                    return False
                elif op == "!=" and installed_version == required_version:
                    return False
                    
            return True
            
        except Exception:
            return True  # Assume compatible if parsing fails

    def _assess_compatibility_severity(self, package_name: str) -> str:
        """Assess severity of version mismatch for specific packages"""
        critical_packages = {
            "torch", "transformers", "faiss-cpu", "numpy", "scipy",
            "pydantic", "jsonschema", "z3-solver"
        }
        
        if package_name.lower() in critical_packages:
            return "critical"
        return "warning"

    def validate_threshold_satisfaction(self, current_metrics: Dict[str, Any]) -> ValidationResult:
        """Validate that current system state meets threshold requirements"""
        errors = []
        warnings = []
        
        try:
            threshold_config = self.thresholds.get("validation", {}).get("threshold_satisfaction", {})
            
            validation_results = {}
            total_score = 0.0
            total_weight = 0.0
            
            # Check mandatory clause compliance
            mandatory_compliance = current_metrics.get("mandatory_compliance", 0.0)
            mandatory_threshold = threshold_config.get("mandatory_clause_compliance", 1.0)
            mandatory_passed = mandatory_compliance >= mandatory_threshold
            
            validation_results["mandatory_compliance"] = {
                "value": mandatory_compliance,
                "threshold": mandatory_threshold,
                "passed": mandatory_passed,
                "weight": 0.3
            }
            
            if not mandatory_passed:
                errors.append(f"Mandatory compliance insufficient: {mandatory_compliance} < {mandatory_threshold}")
            
            # Check proxy score minimum
            proxy_score = current_metrics.get("proxy_score", 0.0)
            proxy_threshold = threshold_config.get("proxy_score_minimum", 0.7)
            proxy_passed = proxy_score >= proxy_threshold
            
            validation_results["proxy_score"] = {
                "value": proxy_score,
                "threshold": proxy_threshold,
                "passed": proxy_passed,
                "weight": 0.25
            }
            
            if not proxy_passed:
                errors.append(f"Proxy score insufficient: {proxy_score} < {proxy_threshold}")
            
            # Check confidence alpha
            confidence_alpha = current_metrics.get("confidence_alpha", 0.0)
            alpha_threshold = threshold_config.get("confidence_alpha", 0.95)
            alpha_passed = confidence_alpha >= alpha_threshold
            
            validation_results["confidence_alpha"] = {
                "value": confidence_alpha,
                "threshold": alpha_threshold,
                "passed": alpha_passed,
                "weight": 0.2
            }
            
            if not alpha_passed:
                warnings.append(f"Confidence alpha below threshold: {confidence_alpha} < {alpha_threshold}")
            
            # Check sigma presence
            sigma_presence = current_metrics.get("sigma_presence", 0.0)
            sigma_threshold = threshold_config.get("sigma_presence", 0.05)
            sigma_passed = sigma_presence >= sigma_threshold
            
            validation_results["sigma_presence"] = {
                "value": sigma_presence,
                "threshold": sigma_threshold,
                "passed": sigma_passed,
                "weight": 0.15
            }
            
            # Check governance completeness
            governance_score = current_metrics.get("governance_completeness", 0.0)
            governance_threshold = threshold_config.get("governance_completeness", 0.85)
            governance_passed = governance_score >= governance_threshold
            
            validation_results["governance_completeness"] = {
                "value": governance_score,
                "threshold": governance_threshold,
                "passed": governance_passed,
                "weight": 0.1
            }
            
            # Calculate weighted score
            for result in validation_results.values():
                weight = result["weight"]
                score_component = result["value"] * weight if result["passed"] else 0.0
                total_score += score_component
                total_weight += weight
            
            overall_score = total_score / total_weight if total_weight > 0 else 0.0
            
            # Determine if validation passes overall
            critical_failures = sum(1 for r in validation_results.values() 
                                  if not r["passed"] and r["weight"] >= 0.2)
            
            passed = critical_failures == 0 and overall_score >= 0.7
            
            details = {
                "validation_results": validation_results,
                "overall_score": overall_score,
                "critical_failures": critical_failures
            }
            
            return ValidationResult(
                passed=passed,
                score=overall_score,
                details=details,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            errors.append(f"Threshold validation failed: {str(e)}")
            return ValidationResult(passed=False, score=0.0, details={}, errors=errors)

    def run_comprehensive_validation(
        self, 
        input_data: Dict[str, Any], 
        schema_type: str,
        current_metrics: Dict[str, Any],
        requirements_path: str = "requirements.txt"
    ) -> Dict[str, ValidationResult]:
        """Run all preflight validation checks"""
        
        self.logger.info("Starting comprehensive preflight validation")
        
        results = {}
        
        # Input schema validation
        self.logger.info("Validating input schema")
        results["schema_validation"] = self.validate_input_schema(input_data, schema_type)
        
        # Library compatibility check
        self.logger.info("Checking library compatibility")
        results["library_compatibility"] = self.check_library_compatibility(requirements_path)
        
        # Threshold satisfaction validation
        self.logger.info("Validating threshold satisfaction")
        results["threshold_satisfaction"] = self.validate_threshold_satisfaction(current_metrics)
        
        # Log summary
        passed_checks = sum(1 for result in results.values() if result.passed)
        total_checks = len(results)
        
        self.logger.info(f"Preflight validation complete: {passed_checks}/{total_checks} checks passed")
        
        return results

    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate summary of validation results"""
        total_checks = len(results)
        passed_checks = sum(1 for result in results.values() if result.passed)
        
        overall_score = sum(result.score for result in results.values()) / total_checks if total_checks > 0 else 0.0
        
        all_errors = []
        all_warnings = []
        
        for check_name, result in results.items():
            for error in result.errors:
                all_errors.append(f"{check_name}: {error}")
            for warning in result.warnings:
                all_warnings.append(f"{check_name}: {warning}")
        
        return {
            "overall_passed": passed_checks == total_checks,
            "overall_score": overall_score,
            "checks_passed": passed_checks,
            "total_checks": total_checks,
            "errors": all_errors,
            "warnings": all_warnings,
            "details": {check: result.dict() for check, result in results.items()},
            "timestamp": datetime.now().isoformat()
        }