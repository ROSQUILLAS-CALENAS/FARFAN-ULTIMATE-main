"""
Refusal Matrix: Comprehensive mapping of refusal clauses and triggers
Maps all governance failure conditions to their refusal triggers and recovery paths.
"""

# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any, Optional, Callable  # Module not found  # Module not found  # Module not found
import json


class RefusalClause(Enum):
    """All possible refusal clauses in the governance system"""
    MANDATORY_MISSING = "mandatory_missing"
    PROXY_INSUFFICIENT = "proxy_insufficient" 
    ALPHA_VIOLATED = "alpha_violated"
    SIGMA_ABSENT = "sigma_absent"
    COMPLIANCE_FAILED = "compliance_failed"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class TriggerCondition:
    """Defines what triggers a refusal clause"""
    condition_id: str
    description: str
    check_function: str  # Name of function that checks this condition
    threshold_value: Any = None
    required_fields: List[str] = None
    error_patterns: List[str] = None


@dataclass
class RefusalResponse:
    """Defines the response when refusal is triggered"""
    refusal_type: RefusalClause
    message_template: str
    recovery_instructions: str
    stable: bool = True  # Whether refusal is stable (deterministic)
    severity: str = "HIGH"
    remediation_steps: List[str] = None


@dataclass
class RefusalMapping:
    """Complete mapping of clause to triggers and responses"""
    clause: RefusalClause
    triggers: List[TriggerCondition]
    response: RefusalResponse
    precedence: int  # Order of evaluation (lower = higher priority)
    test_scenarios: List[str] = None


class RefusalMatrix:
    """Complete refusal matrix for governance system"""
    
    def __init__(self):
        self.refusal_mappings = self._initialize_refusal_mappings()
        self.trigger_registry = self._build_trigger_registry()
        self.evaluation_order = sorted(self.refusal_mappings.values(), key=lambda x: x.precedence)
        
    def _initialize_refusal_mappings(self) -> Dict[RefusalClause, RefusalMapping]:
        """Initialize complete refusal clause mappings"""
        
        return {
            # CLAUSE 1: MANDATORY MISSING (Highest Priority)
            RefusalClause.MANDATORY_MISSING: RefusalMapping(
                clause=RefusalClause.MANDATORY_MISSING,
                precedence=1,
                triggers=[
                    TriggerCondition(
                        condition_id="MAND_001",
                        description="Required field completely absent",
                        check_function="check_field_presence",
                        required_fields=["diagnostic", "programs", "budget", "indicators"]
                    ),
                    TriggerCondition(
                        condition_id="MAND_002", 
                        description="Required field has null/empty value",
                        check_function="check_field_validity",
                        required_fields=["diagnostic", "programs", "budget", "indicators"]
                    ),
                    TriggerCondition(
                        condition_id="MAND_003",
                        description="Critical governance components missing",
                        check_function="check_governance_completeness", 
                        required_fields=["compliance_matrix", "regulatory_framework"]
                    )
                ],
                response=RefusalResponse(
                    refusal_type=RefusalClause.MANDATORY_MISSING,
                    message_template="Mandatory criteria missing: {missing_fields}",
                    recovery_instructions="Provide all required fields: {missing_fields}",
                    severity="CRITICAL",
                    stable=True,
                    remediation_steps=[
                        "Identify missing mandatory fields",
                        "Gather required information",
                        "Validate field completeness",
                        "Resubmit complete request"
                    ]
                ),
                test_scenarios=["REF001", "REF002", "REF003", "REF009"]
            ),
            
            # CLAUSE 2: PROXY INSUFFICIENT
            RefusalClause.PROXY_INSUFFICIENT: RefusalMapping(
                clause=RefusalClause.PROXY_INSUFFICIENT,
                precedence=2,
                triggers=[
                    TriggerCondition(
                        condition_id="PROX_001",
                        description="Proxy score below minimum threshold",
                        check_function="check_proxy_threshold",
                        threshold_value=0.7
                    ),
                    TriggerCondition(
                        condition_id="PROX_002",
                        description="Proxy variables insufficient for causal inference",
                        check_function="check_causal_proxy_sufficiency", 
                        threshold_value=0.6
                    ),
                    TriggerCondition(
                        condition_id="PROX_003",
                        description="Bridge assumption test failed",
                        check_function="check_bridge_assumption",
                        threshold_value=0.05  # p-value threshold
                    )
                ],
                response=RefusalResponse(
                    refusal_type=RefusalClause.PROXY_INSUFFICIENT,
                    message_template="Proxy insufficient: {proxy_score:.3f} < {threshold}",
                    recovery_instructions="Increase proxy score to at least {threshold}",
                    severity="HIGH",
                    stable=True,
                    remediation_steps=[
                        "Review proxy variable selection",
                        "Improve proxy measurement quality", 
                        "Validate causal identification assumptions",
                        "Rerun with sufficient proxies"
                    ]
                ),
                test_scenarios=["REF004", "REF005", "REF011"]
            ),
            
            # CLAUSE 3: ALPHA VIOLATED 
            RefusalClause.ALPHA_VIOLATED: RefusalMapping(
                clause=RefusalClause.ALPHA_VIOLATED,
                precedence=3,
                triggers=[
                    TriggerCondition(
                        condition_id="ALPH_001",
                        description="Confidence level below alpha threshold",
                        check_function="check_alpha_confidence",
                        threshold_value=0.95  # 1 - alpha where alpha = 0.05
                    ),
                    TriggerCondition(
                        condition_id="ALPH_002",
                        description="Statistical significance violated",
                        check_function="check_statistical_significance",
                        threshold_value=0.05
                    ),
                    TriggerCondition(
                        condition_id="ALPH_003", 
                        description="Conformal risk control bound exceeded",
                        check_function="check_crc_risk_bound",
                        threshold_value=0.1  # Risk tolerance
                    )
                ],
                response=RefusalResponse(
                    refusal_type=RefusalClause.ALPHA_VIOLATED,
                    message_template="Alpha violated: confidence {confidence:.3f} < {required:.3f}",
                    recovery_instructions="Increase confidence to at least {required:.3f}",
                    severity="HIGH",
                    stable=True,
                    remediation_steps=[
                        "Review statistical methodology",
                        "Increase sample size if needed",
                        "Validate confidence calculations", 
                        "Adjust alpha parameter if appropriate"
                    ]
                ),
                test_scenarios=["REF006", "REF007"]
            ),
            
            # CLAUSE 4: SIGMA ABSENT
            RefusalClause.SIGMA_ABSENT: RefusalMapping(
                clause=RefusalClause.SIGMA_ABSENT,
                precedence=4,
                triggers=[
                    TriggerCondition(
                        condition_id="SIGM_001",
# # #                         description="Sigma parameter missing from request",  # Module not found  # Module not found  # Module not found
                        check_function="check_sigma_presence"
                    ),
                    TriggerCondition(
                        condition_id="SIGM_002",
                        description="Sigma parameter invalid or out of range",
                        check_function="check_sigma_validity",
                        threshold_value=(0.01, 1.0)  # Valid range
                    ),
                    TriggerCondition(
                        condition_id="SIGM_003",
                        description="Uncertainty estimation requires sigma",
                        check_function="check_uncertainty_requirements"
                    )
                ],
                response=RefusalResponse(
                    refusal_type=RefusalClause.SIGMA_ABSENT,
# # #                     message_template="Sigma parameter absent from request",  # Module not found  # Module not found  # Module not found
                    recovery_instructions="Provide sigma parameter (recommended: {default_sigma})",
                    severity="MEDIUM",
                    stable=True,
                    remediation_steps=[
                        "Add sigma parameter to request",
                        "Use recommended default value: 0.1",
                        "Validate sigma is within acceptable range",
                        "Resubmit with complete parameters"
                    ]
                ),
                test_scenarios=["REF008"]
            ),
            
            # CLAUSE 5: COMPLIANCE FAILED
            RefusalClause.COMPLIANCE_FAILED: RefusalMapping(
                clause=RefusalClause.COMPLIANCE_FAILED,
                precedence=5,
                triggers=[
                    TriggerCondition(
                        condition_id="COMP_001",
                        description="DNP compliance standards not met",
                        check_function="check_dnp_compliance",
                        threshold_value=0.85
                    ),
                    TriggerCondition(
                        condition_id="COMP_002",
                        description="Regulatory framework violations",
                        check_function="check_regulatory_compliance",
                        required_fields=["law_152", "law_1454", "decree_1082"]
                    ),
                    TriggerCondition(
                        condition_id="COMP_003",
                        description="Technical standard violations",
                        check_function="check_technical_standards",
                        error_patterns=["MAPEO_VIOLATION", "ALIGNMENT_FAILURE"]
                    )
                ],
                response=RefusalResponse(
                    refusal_type=RefusalClause.COMPLIANCE_FAILED,
                    message_template="Compliance failed: {violations}",
                    recovery_instructions="Address compliance violations: {remediation_actions}",
                    severity="CRITICAL",
                    stable=True,
                    remediation_steps=[
                        "Review compliance requirements",
                        "Address identified violations",
                        "Validate against all standards",
                        "Obtain compliance certification"
                    ]
                ),
                test_scenarios=["COMP_001", "COMP_002"]
            ),
            
            # CLAUSE 6: VALIDATION ERROR
            RefusalClause.VALIDATION_ERROR: RefusalMapping(
                clause=RefusalClause.VALIDATION_ERROR,
                precedence=6,
                triggers=[
                    TriggerCondition(
                        condition_id="VALD_001",
                        description="Data format validation failed",
                        check_function="check_data_format",
                        error_patterns=["INVALID_JSON", "SCHEMA_VIOLATION"]
                    ),
                    TriggerCondition(
                        condition_id="VALD_002", 
                        description="Business logic validation failed",
                        check_function="check_business_rules",
                        error_patterns=["INCONSISTENT_DATA", "LOGICAL_CONTRADICTION"]
                    ),
                    TriggerCondition(
                        condition_id="VALD_003",
                        description="Constraint satisfaction failed",
                        check_function="check_constraint_satisfaction",
                        threshold_value=0.8
                    )
                ],
                response=RefusalResponse(
                    refusal_type=RefusalClause.VALIDATION_ERROR,
                    message_template="Validation error: {error_details}",
                    recovery_instructions="Fix validation errors: {error_list}",
                    severity="MEDIUM",
                    stable=True,
                    remediation_steps=[
                        "Review validation errors",
                        "Fix data format issues",
                        "Resolve business logic conflicts",
                        "Validate constraints are satisfied"
                    ]
                ),
                test_scenarios=["VALD_001", "VALD_002"]
            )
        }
        
    def _build_trigger_registry(self) -> Dict[str, TriggerCondition]:
        """Build registry of all trigger conditions"""
        registry = {}
        
        for mapping in self.refusal_mappings.values():
            for trigger in mapping.triggers:
                registry[trigger.condition_id] = trigger
                
        return registry
        
    def get_refusal_for_condition(self, condition_id: str) -> Optional[RefusalMapping]:
        """Get refusal mapping for a specific trigger condition"""
        for mapping in self.refusal_mappings.values():
            for trigger in mapping.triggers:
                if trigger.condition_id == condition_id:
                    return mapping
        return None
        
    def get_triggers_by_function(self, function_name: str) -> List[TriggerCondition]:
        """Get all triggers that use a specific check function"""
        triggers = []
        for trigger in self.trigger_registry.values():
            if trigger.check_function == function_name:
                triggers.append(trigger)
        return triggers
        
    def get_evaluation_precedence(self) -> List[RefusalClause]:
        """Get refusal clauses in evaluation order (precedence)"""
        return [mapping.clause for mapping in self.evaluation_order]
        
    def generate_trigger_matrix(self) -> Dict[str, Any]:
        """Generate complete trigger matrix for documentation"""
        matrix = {
            "total_clauses": len(self.refusal_mappings),
            "total_triggers": len(self.trigger_registry),
            "evaluation_order": [clause.value for clause in self.get_evaluation_precedence()],
            "clause_details": {}
        }
        
        for clause, mapping in self.refusal_mappings.items():
            matrix["clause_details"][clause.value] = {
                "precedence": mapping.precedence,
                "trigger_count": len(mapping.triggers),
                "triggers": [
                    {
                        "id": trigger.condition_id,
                        "description": trigger.description,
                        "function": trigger.check_function,
                        "threshold": trigger.threshold_value,
                        "required_fields": trigger.required_fields,
                        "error_patterns": trigger.error_patterns
                    }
                    for trigger in mapping.triggers
                ],
                "response": {
                    "message_template": mapping.response.message_template,
                    "severity": mapping.response.severity,
                    "stable": mapping.response.stable,
                    "remediation_steps": mapping.response.remediation_steps
                },
                "test_scenarios": mapping.test_scenarios
            }
            
        return matrix
        
    def validate_matrix_completeness(self) -> Dict[str, Any]:
        """Validate that refusal matrix is complete and consistent"""
        validation = {
            "complete": True,
            "issues": [],
            "coverage": {}
        }
        
        # Check all clauses have triggers
        for clause, mapping in self.refusal_mappings.items():
            if not mapping.triggers:
                validation["complete"] = False
                validation["issues"].append(f"Clause {clause.value} has no triggers")
                
            # Check all triggers have valid functions
            for trigger in mapping.triggers:
                if not trigger.check_function:
                    validation["complete"] = False
                    validation["issues"].append(f"Trigger {trigger.condition_id} missing check function")
                    
            # Check response completeness
            if not mapping.response.message_template:
                validation["complete"] = False
                validation["issues"].append(f"Clause {clause.value} missing response template")
                
        # Calculate coverage metrics
        validation["coverage"] = {
            "mandatory_triggers": len([t for t in self.trigger_registry.values() 
                                     if "mandatory" in t.description.lower()]),
            "threshold_triggers": len([t for t in self.trigger_registry.values() 
                                     if t.threshold_value is not None]),
            "field_triggers": len([t for t in self.trigger_registry.values() 
                                 if t.required_fields is not None]),
            "pattern_triggers": len([t for t in self.trigger_registry.values() 
                                   if t.error_patterns is not None])
        }
        
        return validation
        
    def export_matrix(self, filepath: str = "refusal_matrix.json"):
        """Export complete refusal matrix to JSON"""
        matrix_data = {
            "refusal_matrix": self.generate_trigger_matrix(),
            "validation": self.validate_matrix_completeness(),
            "metadata": {
                "total_clauses": len(RefusalClause),
                "implemented_clauses": len(self.refusal_mappings),
                "coverage_percentage": (len(self.refusal_mappings) / len(RefusalClause)) * 100,
                "precedence_levels": len(set(m.precedence for m in self.refusal_mappings.values()))
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(matrix_data, f, indent=2, default=str)
            
        return matrix_data


# Instantiate global refusal matrix
REFUSAL_MATRIX = RefusalMatrix()


def get_refusal_matrix() -> RefusalMatrix:
    """Get the global refusal matrix instance"""
    return REFUSAL_MATRIX


def check_refusal_condition(condition_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Check if a specific refusal condition is triggered"""
    trigger = REFUSAL_MATRIX.trigger_registry.get(condition_id)
    if not trigger:
        return None
        
    mapping = REFUSAL_MATRIX.get_refusal_for_condition(condition_id)
    if not mapping:
        return None
        
    # This would normally call the actual check function
    # For now, return a mock result structure
    return {
        "condition_id": condition_id,
        "triggered": False,  # Would be determined by actual check
        "clause": mapping.clause.value,
        "response": mapping.response.__dict__
    }


if __name__ == "__main__":
    # Generate and export refusal matrix
    matrix = RefusalMatrix()
    
    print("Refusal Matrix Summary:")
    print(f"Total clauses: {len(matrix.refusal_mappings)}")
    print(f"Total triggers: {len(matrix.trigger_registry)}")
    
    validation = matrix.validate_matrix_completeness()
    print(f"Matrix complete: {validation['complete']}")
    
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
            
    print(f"Coverage: {validation['coverage']}")
    
    # Export matrix
    matrix_data = matrix.export_matrix()
    print(f"Matrix exported with {matrix_data['metadata']['coverage_percentage']:.1f}% coverage")