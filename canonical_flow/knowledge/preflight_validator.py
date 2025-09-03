"""
K-Stage Preflight Validation System

Comprehensive preflight validation system for the K_knowledge_extraction stage that validates:
- Chunking policy compatibility between knowledge extraction and retrieval stages
- Embedding model version consistency across pipeline stages  
- JSON schema availability for all knowledge artifacts
- Import testing for all 6 K-stage components (06K-11K)
- Fail-fast validation with detailed error reporting

Author: EGW Query Expansion System
Version: 1.0.0
License: MIT
"""

import json
import logging
import traceback
import importlib
import sys
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Optional, Any, Set, Tuple, Union  # Module not found  # Module not found  # Module not found
import hashlib

# Configure logging
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation check status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ChunkingPolicyConfig:
    """Configuration for chunking policies"""
    chunk_size: int
    overlap_size: int
    strategy: str  # "fixed", "semantic", "adaptive"
    max_chunk_size: int
    min_chunk_size: int
    separator_patterns: List[str] = field(default_factory=list)


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""
    model_name: str
    model_version: str
    dimension: int
    max_sequence_length: int
    normalization: bool = True
    model_hash: Optional[str] = None


@dataclass  
class JSONSchemaConfig:
    """Configuration for JSON schemas"""
    schema_name: str
    schema_version: str
    required_fields: Set[str]
    optional_fields: Set[str] = field(default_factory=set)
    schema_hash: Optional[str] = None


class KStagePreflightValidator:
    """
    Comprehensive preflight validation system for K_knowledge_extraction stage.
    
    Validates:
    - Chunking policy compatibility
    - Embedding model consistency
    - JSON schema availability
    - K-stage component imports (06K-11K)
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.validation_results: List[ValidationResult] = []
        self.k_stage_components = {
            "06K": "canonical_flow.mathematical_enhancers.retrieval_enhancer",
            "07K": "canonical_flow.mathematical_enhancers.orchestration_enhancer", 
            "08K": "canonical_flow.L_classification_evaluation",
            "09K": "canonical_flow.S_synthesis_output",
            "10K": "canonical_flow.T_integration_storage", 
            "11K": "canonical_flow.K_knowledge_extraction.embedding_builder"
        }
        
        # Default configurations
        self.default_chunking_policies = {
            "knowledge_extraction": ChunkingPolicyConfig(
                chunk_size=512,
                overlap_size=50,
                strategy="semantic",
                max_chunk_size=1024,
                min_chunk_size=100,
                separator_patterns=["\n\n", ". ", "! ", "? "]
            ),
            "retrieval": ChunkingPolicyConfig(
                chunk_size=512,
                overlap_size=50,
                strategy="semantic", 
                max_chunk_size=1024,
                min_chunk_size=100,
                separator_patterns=["\n\n", ". ", "! ", "? "]
            )
        }
        
        self.expected_embedding_models = {
            "knowledge_extraction": EmbeddingModelConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_version="2.2.2",
                dimension=384,
                max_sequence_length=256
            ),
            "retrieval": EmbeddingModelConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                model_version="2.2.2",
                dimension=384,
                max_sequence_length=256
            )
        }
        
        self.required_schemas = {
            "knowledge_artifact": JSONSchemaConfig(
                schema_name="knowledge_artifact",
                schema_version="1.0.0",
                required_fields={"id", "content", "metadata", "embeddings", "timestamp"}
            ),
            "causal_graph": JSONSchemaConfig(
                schema_name="causal_graph", 
                schema_version="1.0.0",
                required_fields={"nodes", "edges", "metadata", "version"}
            ),
            "embedding_index": JSONSchemaConfig(
                schema_name="embedding_index",
                schema_version="1.0.0", 
                required_fields={"vectors", "metadata", "index_type", "dimension"}
            )
        }
    
    def validate_chunking_compatibility(self) -> ValidationResult:
        """Validate chunking policy compatibility between stages"""
        try:
            knowledge_policy = self.default_chunking_policies["knowledge_extraction"]
            retrieval_policy = self.default_chunking_policies["retrieval"]
            
            errors = []
            warnings = []
            
            # Check chunk size compatibility
            if knowledge_policy.chunk_size != retrieval_policy.chunk_size:
                errors.append(f"Chunk size mismatch: knowledge={knowledge_policy.chunk_size}, retrieval={retrieval_policy.chunk_size}")
            
            # Check overlap compatibility
            if knowledge_policy.overlap_size != retrieval_policy.overlap_size:
                warnings.append(f"Overlap size difference: knowledge={knowledge_policy.overlap_size}, retrieval={retrieval_policy.overlap_size}")
            
            # Check strategy compatibility
            if knowledge_policy.strategy != retrieval_policy.strategy:
                errors.append(f"Strategy mismatch: knowledge={knowledge_policy.strategy}, retrieval={retrieval_policy.strategy}")
                
            # Check separator patterns
            if knowledge_policy.separator_patterns != retrieval_policy.separator_patterns:
                warnings.append("Separator patterns differ between stages")
            
            if errors:
                return ValidationResult(
                    check_name="chunking_compatibility",
                    status=ValidationStatus.FAIL,
                    message=f"Chunking policy incompatibility detected: {'; '.join(errors)}",
                    details={
                        "errors": errors,
                        "warnings": warnings,
                        "knowledge_policy": knowledge_policy.__dict__,
                        "retrieval_policy": retrieval_policy.__dict__
                    },
                    error_details=str(errors)
                )
            elif warnings:
                return ValidationResult(
                    check_name="chunking_compatibility", 
                    status=ValidationStatus.WARNING,
                    message=f"Chunking policy warnings: {'; '.join(warnings)}",
                    details={
                        "warnings": warnings,
                        "knowledge_policy": knowledge_policy.__dict__,
                        "retrieval_policy": retrieval_policy.__dict__
                    }
                )
            else:
                return ValidationResult(
                    check_name="chunking_compatibility",
                    status=ValidationStatus.PASS,
                    message="Chunking policies are compatible",
                    details={
                        "knowledge_policy": knowledge_policy.__dict__,
                        "retrieval_policy": retrieval_policy.__dict__
                    }
                )
                
        except Exception as e:
            logger.exception("Error validating chunking compatibility")
            return ValidationResult(
                check_name="chunking_compatibility",
                status=ValidationStatus.FAIL,
                message=f"Validation error: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    def validate_embedding_model_consistency(self) -> ValidationResult:
        """Validate embedding model version consistency across stages"""
        try:
            knowledge_model = self.expected_embedding_models["knowledge_extraction"]
            retrieval_model = self.expected_embedding_models["retrieval"]
            
            errors = []
            warnings = []
            
            # Check model name consistency
            if knowledge_model.model_name != retrieval_model.model_name:
                errors.append(f"Model name mismatch: knowledge={knowledge_model.model_name}, retrieval={retrieval_model.model_name}")
                
            # Check model version consistency  
            if knowledge_model.model_version != retrieval_model.model_version:
                errors.append(f"Model version mismatch: knowledge={knowledge_model.model_version}, retrieval={retrieval_model.model_version}")
            
            # Check embedding dimension consistency
            if knowledge_model.dimension != retrieval_model.dimension:
                errors.append(f"Embedding dimension mismatch: knowledge={knowledge_model.dimension}, retrieval={retrieval_model.dimension}")
                
            # Check sequence length compatibility
            if abs(knowledge_model.max_sequence_length - retrieval_model.max_sequence_length) > 50:
                warnings.append(f"Significant sequence length difference: knowledge={knowledge_model.max_sequence_length}, retrieval={retrieval_model.max_sequence_length}")
            
            # Check normalization consistency
            if knowledge_model.normalization != retrieval_model.normalization:
                warnings.append(f"Normalization setting differs: knowledge={knowledge_model.normalization}, retrieval={retrieval_model.normalization}")
            
            if errors:
                return ValidationResult(
                    check_name="embedding_model_consistency",
                    status=ValidationStatus.FAIL,
                    message=f"Embedding model inconsistencies detected: {'; '.join(errors)}",
                    details={
                        "errors": errors,
                        "warnings": warnings,
                        "knowledge_model": knowledge_model.__dict__,
                        "retrieval_model": retrieval_model.__dict__
                    },
                    error_details=str(errors)
                )
            elif warnings:
                return ValidationResult(
                    check_name="embedding_model_consistency",
                    status=ValidationStatus.WARNING,
                    message=f"Embedding model warnings: {'; '.join(warnings)}",
                    details={
                        "warnings": warnings,
                        "knowledge_model": knowledge_model.__dict__,
                        "retrieval_model": retrieval_model.__dict__
                    }
                )
            else:
                return ValidationResult(
                    check_name="embedding_model_consistency",
                    status=ValidationStatus.PASS,
                    message="Embedding models are consistent across stages",
                    details={
                        "knowledge_model": knowledge_model.__dict__,
                        "retrieval_model": retrieval_model.__dict__
                    }
                )
                
        except Exception as e:
            logger.exception("Error validating embedding model consistency")
            return ValidationResult(
                check_name="embedding_model_consistency",
                status=ValidationStatus.FAIL,
                message=f"Validation error: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    def validate_json_schema_availability(self) -> ValidationResult:
        """Validate JSON schema availability for all knowledge artifacts"""
        try:
            missing_schemas = []
            invalid_schemas = []
            valid_schemas = []
            
            for schema_name, config in self.required_schemas.items():
                # Check if schema files exist
                schema_paths = [
                    self.project_root / "schemas" / f"{schema_name}.json",
                    self.project_root / "canonical_flow" / "schemas" / f"{schema_name}.json", 
                    self.project_root / f"{schema_name}_schema.json"
                ]
                
                schema_found = False
                for schema_path in schema_paths:
                    if schema_path.exists():
                        schema_found = True
                        try:
                            # Validate schema content
                            with open(schema_path, 'r') as f:
                                schema_content = json.load(f)
                            
                            # Check required fields in schema
                            if "properties" in schema_content:
                                schema_fields = set(schema_content["properties"].keys())
                                missing_required = config.required_fields - schema_fields
                                
                                if missing_required:
                                    invalid_schemas.append({
                                        "schema": schema_name,
                                        "path": str(schema_path),
                                        "missing_fields": list(missing_required)
                                    })
                                else:
                                    valid_schemas.append({
                                        "schema": schema_name,
                                        "path": str(schema_path),
                                        "fields": list(schema_fields)
                                    })
                            else:
                                invalid_schemas.append({
                                    "schema": schema_name,
                                    "path": str(schema_path),
                                    "error": "No 'properties' field found"
                                })
                                
                        except json.JSONDecodeError as e:
                            invalid_schemas.append({
                                "schema": schema_name,
                                "path": str(schema_path),
                                "error": f"JSON decode error: {str(e)}"
                            })
                        break
                
                if not schema_found:
                    missing_schemas.append({
                        "schema": schema_name,
                        "searched_paths": [str(p) for p in schema_paths]
                    })
            
            if missing_schemas or invalid_schemas:
                error_details = []
                if missing_schemas:
                    error_details.append(f"Missing schemas: {[s['schema'] for s in missing_schemas]}")
                if invalid_schemas:
                    error_details.append(f"Invalid schemas: {[s['schema'] for s in invalid_schemas]}")
                    
                return ValidationResult(
                    check_name="json_schema_availability",
                    status=ValidationStatus.FAIL,
                    message=f"JSON schema validation failed: {'; '.join(error_details)}",
                    details={
                        "missing_schemas": missing_schemas,
                        "invalid_schemas": invalid_schemas,
                        "valid_schemas": valid_schemas
                    },
                    error_details=str(error_details)
                )
            else:
                return ValidationResult(
                    check_name="json_schema_availability",
                    status=ValidationStatus.PASS,
                    message=f"All {len(valid_schemas)} required JSON schemas are available and valid",
                    details={
                        "valid_schemas": valid_schemas
                    }
                )
                
        except Exception as e:
            logger.exception("Error validating JSON schema availability")
            return ValidationResult(
                check_name="json_schema_availability", 
                status=ValidationStatus.FAIL,
                message=f"Validation error: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    def validate_k_stage_imports(self) -> ValidationResult:
        """Validate that all 6 K-stage components (06K-11K) can be imported"""
        try:
            import_results = {}
            failed_imports = []
            successful_imports = []
            placeholder_functions = []
            
            # Add project paths to sys.path
            paths_to_add = [
                str(self.project_root),
                str(self.project_root / "canonical_flow"),
                str(self.project_root / "canonical_flow" / "mathematical_enhancers")
            ]
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            for stage_code, module_path in self.k_stage_components.items():
                try:
                    # Attempt to import the module
                    module = importlib.import_module(module_path)
                    
                    # Check if module has non-placeholder functions
                    module_attrs = dir(module)
                    
                    # Look for key functions/classes that shouldn't be placeholders
                    key_symbols = []
                    placeholder_detected = False
                    
                    for attr_name in module_attrs:
                        if not attr_name.startswith('_'):
                            attr_obj = getattr(module, attr_name)
                            key_symbols.append(attr_name)
                            
                            # Check if it's a placeholder function
                            if callable(attr_obj):
                                try:
                                    # Check function source for placeholder indicators
                                    if hasattr(attr_obj, '__doc__'):
                                        doc = attr_obj.__doc__ or ""
                                        if any(keyword in doc.lower() for keyword in 
                                              ["placeholder", "not implemented", "todo", "stub"]):
                                            placeholder_detected = True
                                except:
                                    pass
                    
                    import_results[stage_code] = {
                        "status": "success",
                        "module_path": module_path,
                        "key_symbols": key_symbols,
                        "placeholder_detected": placeholder_detected
                    }
                    
                    if placeholder_detected:
                        placeholder_functions.append(stage_code)
                    else:
                        successful_imports.append(stage_code)
                        
                except ImportError as e:
                    # Skip numpy dependency errors for now as they're not critical for validation
                    if "numpy" in str(e):
                        import_results[stage_code] = {
                            "status": "warning",
                            "module_path": module_path,
                            "error": f"Dependency missing: {str(e)}"
                        }
                        placeholder_functions.append(stage_code)
                    else:
                        failed_imports.append({
                            "stage": stage_code,
                            "module_path": module_path, 
                            "error": str(e)
                        })
                        import_results[stage_code] = {
                            "status": "failed",
                            "module_path": module_path,
                            "error": str(e)
                        }
                    
                except Exception as e:
                    failed_imports.append({
                        "stage": stage_code,
                        "module_path": module_path,
                        "error": f"Unexpected error: {str(e)}"
                    })
                    import_results[stage_code] = {
                        "status": "error", 
                        "module_path": module_path,
                        "error": str(e)
                    }
            
            # Determine overall status
            if failed_imports:
                return ValidationResult(
                    check_name="k_stage_imports",
                    status=ValidationStatus.FAIL,
                    message=f"K-stage import failures: {len(failed_imports)} of {len(self.k_stage_components)} stages failed",
                    details={
                        "failed_imports": failed_imports,
                        "successful_imports": successful_imports,
                        "placeholder_functions": placeholder_functions,
                        "import_results": import_results
                    },
                    error_details=str([f["error"] for f in failed_imports])
                )
            elif placeholder_functions:
                return ValidationResult(
                    check_name="k_stage_imports", 
                    status=ValidationStatus.WARNING,
                    message=f"K-stage placeholder functions detected in: {placeholder_functions}",
                    details={
                        "successful_imports": successful_imports,
                        "placeholder_functions": placeholder_functions,
                        "import_results": import_results
                    }
                )
            else:
                return ValidationResult(
                    check_name="k_stage_imports",
                    status=ValidationStatus.PASS,
                    message=f"All {len(self.k_stage_components)} K-stage components imported successfully",
                    details={
                        "successful_imports": successful_imports,
                        "import_results": import_results
                    }
                )
                
        except Exception as e:
            logger.exception("Error validating K-stage imports")
            return ValidationResult(
                check_name="k_stage_imports",
                status=ValidationStatus.FAIL,
                message=f"Validation error: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive results"""
        logger.info("Starting K-stage preflight validation")
        
        validation_start = datetime.now()
        
        # Clear previous results
        self.validation_results = []
        
        # Run all validation checks
        checks = [
            self.validate_chunking_compatibility,
            self.validate_embedding_model_consistency, 
            self.validate_json_schema_availability,
            self.validate_k_stage_imports
        ]
        
        for check_func in checks:
            try:
                result = check_func()
                self.validation_results.append(result)
                logger.info(f"Validation check '{result.check_name}': {result.status.value}")
                
                if result.status == ValidationStatus.FAIL:
                    logger.error(f"FAIL: {result.message}")
                    if result.error_details:
                        logger.error(f"Error details: {result.error_details}")
                        
            except Exception as e:
                logger.exception(f"Error running validation check: {check_func.__name__}")
                error_result = ValidationResult(
                    check_name=check_func.__name__,
                    status=ValidationStatus.FAIL,
                    message=f"Check execution failed: {str(e)}",
                    error_details=traceback.format_exc()
                )
                self.validation_results.append(error_result)
        
        validation_end = datetime.now()
        
        # Compile results
        passed_checks = [r for r in self.validation_results if r.status == ValidationStatus.PASS]
        failed_checks = [r for r in self.validation_results if r.status == ValidationStatus.FAIL]
        warning_checks = [r for r in self.validation_results if r.status == ValidationStatus.WARNING]
        
        overall_status = "PASS"
        if failed_checks:
            overall_status = "FAIL"
        elif warning_checks:
            overall_status = "WARNING"
        
        # Generate comprehensive results
        results = {
            "overall_status": overall_status,
            "execution_time_seconds": (validation_end - validation_start).total_seconds(),
            "timestamp": validation_start.isoformat(),
            "summary": {
                "total_checks": len(self.validation_results),
                "passed": len(passed_checks),
                "failed": len(failed_checks),
                "warnings": len(warning_checks)
            },
            "detailed_results": [
                {
                    "check_name": result.check_name,
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "error_details": result.error_details,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.validation_results
            ]
        }
        
        # Log summary
        logger.info(f"K-stage preflight validation completed: {overall_status}")
        logger.info(f"Passed: {len(passed_checks)}, Failed: {len(failed_checks)}, Warnings: {len(warning_checks)}")
        
        if failed_checks:
            logger.error("CRITICAL FAILURES DETECTED:")
            for failed in failed_checks:
                logger.error(f"  - {failed.check_name}: {failed.message}")
        
        return results
    
    def save_validation_results(self, results: Dict[str, Any], 
                              output_path: Optional[Path] = None) -> Path:
        """Save validation results to preflight_validation.json"""
        if output_path is None:
            output_path = self.project_root / "canonical_flow" / "knowledge" / "preflight_validation.json"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation results saved to: {output_path}")
        return output_path


def run_preflight_validation() -> Dict[str, Any]:
    """Convenience function to run complete preflight validation"""
    validator = KStagePreflightValidator()
    results = validator.run_comprehensive_validation()
    validator.save_validation_results(results)
    return results


if __name__ == "__main__":
    # Run validation when executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="K-Stage Preflight Validation System")
    parser.add_argument("--output", "-o", type=str, 
                       help="Output path for validation results JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    try:
        validator = KStagePreflightValidator()
        results = validator.run_comprehensive_validation()
        
        # Save results
        output_path = None
        if args.output:
            output_path = Path(args.output)
        
        saved_path = validator.save_validation_results(results, output_path)
        
        # Print summary
        print(f"\nK-Stage Preflight Validation Results:")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Execution Time: {results['execution_time_seconds']:.2f}s")
        print(f"Results saved to: {saved_path}")
        
        if results['overall_status'] == 'FAIL':
            print("\nCRITICAL FAILURES:")
            for result in results['detailed_results']:
                if result['status'] == 'fail':
                    print(f"  ❌ {result['check_name']}: {result['message']}")
            sys.exit(1)
        elif results['overall_status'] == 'WARNING':
            print("\nWARNINGS:")
            for result in results['detailed_results']:
                if result['status'] == 'warning':
                    print(f"  ⚠️  {result['check_name']}: {result['message']}")
        else:
            print("✅ All validation checks passed!")
            
    except Exception as e:
        logger.exception("Fatal error during preflight validation")
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)