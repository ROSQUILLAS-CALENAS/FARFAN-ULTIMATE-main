# Comprehensive System Audit – Full Summary

Generated: 2025-09-09T00:55:12.004548

## Summary

```
total_checks: 347
passed_checks: 33
failed_checks: 270
warning_checks: 43
skipped_checks: 1
critical_issues: 1
success_rate: 9.510086455331413
health_score: 0
``

## Recommendations

- CRITICAL: Address 1 critical issues immediately
- • Review and secure potentially dangerous code patterns
- HIGH PRIORITY: Fix 237 high-severity failures
- SECURITY: Review and fix security vulnerabilities
- Fix architectural compliance issues
- System health is concerning - prioritize fixing failed checks

## FAIL (270)

### contract_validation
- status: FAIL
- severity: high
- message: Contract validation failed
<details><summary>details</summary>


```json
{
  "stderr": "  File \"/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_contract_validation.py\", line 96\n    if imports[\"import\"]:\nIndentationError: expected an indented block after 'for' statement on line 92\n"
}
```

</details>

### invalid_phase_05I_raw_data_generator.py
- status: FAIL
- severity: medium
- message: Invalid phase 'I_ingestion_preparation' in 05I_raw_data_generator.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/I_ingestion_preparation/05I_raw_data_generator.py",
  "phase": "I_ingestion_preparation"
}
```

</details>

### invalid_phase_08x_context_construction.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context' in 08x_context_construction.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/08x_context_construction.py",
  "phase": "context"
}
```

</details>

### invalid_phase_09x_context_construction.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context' in 09x_context_construction.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/09x_context_construction.py",
  "phase": "context"
}
```

</details>

### invalid_phase_09x_context_construction_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context_construction' in 09x_context_construction_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/09x_context_construction_component.py",
  "phase": "context_construction"
}
```

</details>

### invalid_phase_10x_context_construction.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context' in 10x_context_construction.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/10x_context_construction.py",
  "phase": "context"
}
```

</details>

### invalid_phase_10x_context_construction_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context_construction' in 10x_context_construction_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/10x_context_construction_component.py",
  "phase": "context_construction"
}
```

</details>

### invalid_phase_11x_context_construction.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context' in 11x_context_construction.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/11x_context_construction.py",
  "phase": "context"
}
```

</details>

### invalid_phase_11x_context_construction_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context_construction' in 11x_context_construction_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/11x_context_construction_component.py",
  "phase": "context_construction"
}
```

</details>

### invalid_phase_12x_context_construction.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context' in 12x_context_construction.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/12x_context_construction.py",
  "phase": "context"
}
```

</details>

### invalid_phase_12x_context_construction_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'context_construction' in 12x_context_construction_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/12x_context_construction_component.py",
  "phase": "context_construction"
}
```

</details>

### invalid_phase_52s_synthesis_output.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis' in 52s_synthesis_output.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/52s_synthesis_output.py",
  "phase": "synthesis"
}
```

</details>

### invalid_phase_52s_synthesis_output_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis_output' in 52s_synthesis_output_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/52s_synthesis_output_component.py",
  "phase": "synthesis_output"
}
```

</details>

### invalid_phase_53s_synthesis_output.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis' in 53s_synthesis_output.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/53s_synthesis_output.py",
  "phase": "synthesis"
}
```

</details>

### invalid_phase_53s_synthesis_output_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis_output' in 53s_synthesis_output_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/53s_synthesis_output_component.py",
  "phase": "synthesis_output"
}
```

</details>

### invalid_phase_54s_synthesis_output.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis' in 54s_synthesis_output.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/54s_synthesis_output.py",
  "phase": "synthesis"
}
```

</details>

### invalid_phase_54s_synthesis_output_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis_output' in 54s_synthesis_output_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/54s_synthesis_output_component.py",
  "phase": "synthesis_output"
}
```

</details>

### invalid_phase_55s_synthesis_output.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis' in 55s_synthesis_output.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/55s_synthesis_output.py",
  "phase": "synthesis"
}
```

</details>

### invalid_phase_55s_synthesis_output_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis_output' in 55s_synthesis_output_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/55s_synthesis_output_component.py",
  "phase": "synthesis_output"
}
```

</details>

### invalid_phase_56s_synthesis_output.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis' in 56s_synthesis_output.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/56s_synthesis_output.py",
  "phase": "synthesis"
}
```

</details>

### invalid_phase_56s_synthesis_output_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis_output' in 56s_synthesis_output_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/56s_synthesis_output_component.py",
  "phase": "synthesis_output"
}
```

</details>

### invalid_phase_57s_synthesis_output.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis' in 57s_synthesis_output.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/57s_synthesis_output.py",
  "phase": "synthesis"
}
```

</details>

### invalid_phase_57s_synthesis_output_component.py
- status: FAIL
- severity: medium
- message: Invalid phase 'synthesis_output' in 57s_synthesis_output_component.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/57s_synthesis_output_component.py",
  "phase": "synthesis_output"
}
```

</details>

### invalid_phase_add_annotations_batch.py
- status: FAIL
- severity: medium
- message: Invalid phase '{phase}' in add_annotations_batch.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/add_annotations_batch.py",
  "phase": "{phase}"
}
```

</details>

### invalid_phase_add_pipeline_annotations.py
- status: FAIL
- severity: medium
- message: Invalid phase '{phase.value}' in add_pipeline_annotations.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/add_pipeline_annotations.py",
  "phase": "{phase.value}"
}
```

</details>

### invalid_phase_bulk_annotate_components.py
- status: FAIL
- severity: medium
- message: Invalid phase '{phase}' in bulk_annotate_components.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/scripts/bulk_annotate_components.py",
  "phase": "{phase}"
}
```

</details>

### invalid_phase_knowledge_validator.py
- status: FAIL
- severity: medium
- message: Invalid phase 'K_knowledge_extraction' in knowledge_validator.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/K_knowledge_extraction/knowledge_validator.py",
  "phase": "K_knowledge_extraction"
}
```

</details>

### invalid_phase_pipeline_contract_annotations.py
- status: FAIL
- severity: medium
- message: Invalid phase '{phase.value}' in pipeline_contract_annotations.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/pipeline_contract_annotations.py",
  "phase": "{phase.value}"
}
```

</details>

### invalid_phase_scaffold_canonical.py
- status: FAIL
- severity: medium
- message: Invalid phase '{phase}' in scaffold_canonical.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/scaffold_canonical.py",
  "phase": "{phase}"
}
```

</details>

### invalid_phase_scaffold_canonical_modules.py
- status: FAIL
- severity: medium
- message: Invalid phase '{{ phase }}' in scaffold_canonical_modules.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/scaffold_canonical_modules.py",
  "phase": "{{ phase }}"
}
```

</details>

### invalid_phase_storage_validator.py
- status: FAIL
- severity: medium
- message: Invalid phase 'T_integration_storage' in storage_validator.py
- remediation: Use valid phase from: I, X, K, A, L, R, O, G, T, S
<details><summary>details</summary>


```json
{
  "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/T_integration_storage/storage_validator.py",
  "phase": "T_integration_storage"
}
```

</details>

### missing_imports
- status: FAIL
- severity: medium
- message: Found 352 missing imports
- remediation: Install missing dependencies or fix import paths
<details><summary>details</summary>


```json
{
  "missing_imports": [
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/evidence_system.py",
      "plotly.graph_objects"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/evidence_system.py",
      "numpy"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/evidence_system.py",
      "matplotlib.pyplot"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/evidence_system.py",
      "msgspec"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/visual_testing_framework.py",
      "numpy"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/module_distributed_processor.py",
      "torch"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/module_distributed_processor.py",
      "numpy"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/airflow_orchestrator.py",
      "airflow"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_ccc_system.py",
      "ccc_validator"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_ccc_system.py",
      "ci_ccc_integration"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/example_constraint_usage.py",
      "numpy"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_owner_assignment_system.py",
      "pytest"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/config_loader.py",
      "jsonschema"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/config_loader.py",
      "yaml"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/config_loader.py",
      "toml"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/config_loader.py",
      "watchgod"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/recovery_system.py",
      "numpy"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/recovery_system.py",
      "redis"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/automated_dependency_resolver.py",
      "sklearn"
    ],
    [
      "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/monitoring_dashboard.py",
      "psutil"
    ]
  ]
}
```

</details>

### security_patterns
- status: FAIL
- severity: critical
- message: Found 62 potential security issues
- remediation: Review and secure potentially dangerous code patterns
<details><summary>details</summary>


```json
{
  "issues": [
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/visual_testing_framework.py",
      "issue": "Unencrypted HTTP URLs",
      "matches": 2
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/module_distributed_processor.py",
      "issue": "Use of eval() function",
      "matches": 1
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/serializable_wrappers.py",
      "issue": "Use of eval() function",
      "matches": 1
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/serializable_wrappers.py",
      "issue": "Use of pickle (potential security risk)",
      "matches": 1
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_decalogo.py",
      "issue": "Use of exec() function",
      "matches": 1
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_atroz_server.py",
      "issue": "Unencrypted HTTP URLs",
      "matches": 8
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/alert_system.py",
      "issue": "Unencrypted HTTP URLs",
      "matches": 2
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/hybrid_retrieval.py",
      "issue": "Use of eval() function",
      "matches": 1
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/hybrid_retrieval.py",
      "issue": "Use of pickle (potential security risk)",
      "matches": 1
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/demo_z3_integration.py",
      "issue": "Use of eval() function",
      "matches": 1
    }
  ]
}
```

</details>

### stage_order_violations
- status: FAIL
- severity: medium
- message: Found 24 stage order violations
- remediation: Correct __stage_order__ values to match phase sequence
<details><summary>details</summary>


```json
{
  "violations": [
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/09x_context_construction.py",
      "phase": "context",
      "actual_stage": 9,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/12x_context_construction.py",
      "phase": "context",
      "actual_stage": 12,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/12x_context_construction_component.py",
      "phase": "context_construction",
      "actual_stage": 12,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/08x_context_construction.py",
      "phase": "context",
      "actual_stage": 8,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/10x_context_construction.py",
      "phase": "context",
      "actual_stage": 10,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/11x_context_construction_component.py",
      "phase": "context_construction",
      "actual_stage": 11,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/09x_context_construction_component.py",
      "phase": "context_construction",
      "actual_stage": 9,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/10x_context_construction_component.py",
      "phase": "context_construction",
      "actual_stage": 10,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/11x_context_construction.py",
      "phase": "context",
      "actual_stage": 11,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/T_integration_storage/storage_validator.py",
      "phase": "T_integration_storage",
      "actual_stage": 20,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/54s_synthesis_output.py",
      "phase": "synthesis",
      "actual_stage": 54,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/54s_synthesis_output_component.py",
      "phase": "synthesis_output",
      "actual_stage": 54,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/52s_synthesis_output_component.py",
      "phase": "synthesis_output",
      "actual_stage": 52,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/55s_synthesis_output.py",
      "phase": "synthesis",
      "actual_stage": 55,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/53s_synthesis_output_component.py",
      "phase": "synthesis_output",
      "actual_stage": 53,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/55s_synthesis_output_component.py",
      "phase": "synthesis_output",
      "actual_stage": 55,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/56s_synthesis_output_component.py",
      "phase": "synthesis_output",
      "actual_stage": 56,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/56s_synthesis_output.py",
      "phase": "synthesis",
      "actual_stage": 56,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/53s_synthesis_output.py",
      "phase": "synthesis",
      "actual_stage": 53,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/57s_synthesis_output.py",
      "phase": "synthesis",
      "actual_stage": 57,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/52s_synthesis_output.py",
      "phase": "synthesis",
      "actual_stage": 52,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/57s_synthesis_output_component.py",
      "phase": "synthesis_output",
      "actual_stage": 57,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/K_knowledge_extraction/knowledge_validator.py",
      "phase": "K_knowledge_extraction",
      "actual_stage": 11,
      "expected_stage": 0
    },
    {
      "file": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/I_ingestion_preparation/05I_raw_data_generator.py",
      "phase": "I_ingestion_preparation",
      "actual_stage": 5,
      "expected_stage": 0
    }
  ]
}
```

</details>

### syntax_error_09x_context_construction_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/09x_context_construction_component.py: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers (<unknown>, line 50)

### syntax_error_11x_context_construction_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/11x_context_construction_component.py: invalid decimal literal (<unknown>, line 50)

### syntax_error_12x_context_construction_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/12x_context_construction_component.py: invalid decimal literal (<unknown>, line 50)

### syntax_error_52s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/52s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 50)

### syntax_error_53s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/53s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 50)

### syntax_error_54s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/54s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 50)

### syntax_error_55s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/55s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 50)

### syntax_error_56s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/56s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 50)

### syntax_error_57s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/57s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 50)

### syntax_error_EXTRACTOR DE EVIDENCIAS CONTEXTUAL.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/EXTRACTOR DE EVIDENCIAS CONTEXTUAL.py: expected an indented block after 'try' statement on line 29 (<unknown>, line 31)

### syntax_error_PIPELINEORCHESTRATOR.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/PIPELINEORCHESTRATOR.py: unexpected indent (<unknown>, line 19)

### syntax_error___init__.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/standards_alignment/__init__.py: unexpected indent (<unknown>, line 9)

### syntax_error___init__.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/contracts/__init__.py: unexpected indent (<unknown>, line 10)

### syntax_error___init__.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/mocks/__init__.py: unexpected indent (<unknown>, line 26)

### syntax_error___init__.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/calibration/__init__.py: unexpected indent (<unknown>, line 10)

### syntax_error___init__.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/__init__.py: unexpected indent (<unknown>, line 4)

### syntax_error___init__.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/knowledge/__init__.py: unexpected indent (<unknown>, line 8)

### syntax_error___init__.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/I_ingestion_preparation/__init__.py: unexpected indent (<unknown>, line 25)

### syntax_error_adaptive_analyzer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/adaptive_analyzer.py: invalid syntax (<unknown>, line 9)

### syntax_error_adaptive_scoring_engine.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/adaptive_scoring_engine.py: unindent does not match any outer indentation level (<unknown>, line 64)

### syntax_error_advanced_loader.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/advanced_loader.py: expected an indented block after 'try' statement on line 303 (<unknown>, line 305)

### syntax_error_analysis_enhancer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/mathematical_enhancers/analysis_enhancer.py: expected an indented block after 'try' statement on line 35 (<unknown>, line 38)

### syntax_error_analysis_nlp_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/analysis_nlp_orchestrator.py: unexpected indent (<unknown>, line 26)

### syntax_error_analytics_enhancement.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/analytics_enhancement.py: unexpected indent (<unknown>, line 22)

### syntax_error_answer_formatter.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/answer_formatter.py: unindent does not match any outer indentation level (<unknown>, line 33)

### syntax_error_answer_synthesizer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/answer_synthesizer.py: unterminated triple-quoted string literal (detected at line 806) (<unknown>, line 749)

### syntax_error_answer_synthesizer_backup.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/answer_synthesizer_backup.py: unterminated triple-quoted string literal (detected at line 807) (<unknown>, line 750)

### syntax_error_calibration_dashboard.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/calibration/calibration_dashboard.py: unmatched ')' (<unknown>, line 33)

### syntax_error_canonical_path_auditor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/canonical_path_auditor.py: unindent does not match any outer indentation level (<unknown>, line 30)

### syntax_error_canonical_web_server.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_web_server.py: f-string: expecting '=', or '!', or ':', or '}' (<unknown>, line 497)

### syntax_error_certificate_generator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/certificate_generator.py: unexpected indent (<unknown>, line 34)

### syntax_error_cluster_execution_controller.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/cluster_execution_controller.py: unindent does not match any outer indentation level (<unknown>, line 28)

### syntax_error_confluent_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/confluent_orchestrator.py: expected an indented block after 'if' statement on line 431 (<unknown>, line 434)

### syntax_error_conformal_risk_certification_demo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/conformal_risk_certification_demo.py: unexpected indent (<unknown>, line 22)

### syntax_error_conformal_risk_demo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/conformal_risk_demo.py: unexpected indent (<unknown>, line 21)

### syntax_error_conformal_risk_demo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/examples/conformal_risk_demo.py: unexpected indent (<unknown>, line 28)

### syntax_error_connection_pool.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/connection_pool.py: expected an indented block after 'try' statement on line 43 (<unknown>, line 45)

### syntax_error_context_adapter.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/context_adapter.py: unexpected indent (<unknown>, line 17)

### syntax_error_context_enhancer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/mathematical_enhancers/context_enhancer.py: expected an indented block after 'try' statement on line 34 (<unknown>, line 36)

### syntax_error_contracts_validation_utility.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/contracts_validation_utility.py: invalid syntax (<unknown>, line 95)

### syntax_error_core_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/core_orchestrator.py: unexpected indent (<unknown>, line 17)

### syntax_error_demo_import_safety.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/demo_import_safety.py: unindent does not match any outer indentation level (<unknown>, line 21)

### syntax_error_demo_integration.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/contracts/demo_integration.py: unindent does not match any outer indentation level (<unknown>, line 34)

### syntax_error_demo_schema_validation.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/demo_schema_validation.py: unexpected indent (<unknown>, line 12)

### syntax_error_dependency_audit.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/dependency_audit.py: unterminated triple-quoted string literal (detected at line 249) (<unknown>, line 154)

### syntax_error_deterministic_embedder.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/src/stages/K_knowledge_extraction/deterministic_embedder.py: expected an indented block after 'except' statement on line 234 (<unknown>, line 237)

### syntax_error_deterministic_router.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/deterministic_router.py: unterminated triple-quoted string literal (detected at line 637) (<unknown>, line 631)

### syntax_error_distributed_processor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/distributed_processor.py: unmatched ')' (<unknown>, line 179)

### syntax_error_dnp_alignment_adapter.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/dnp_alignment_adapter.py: expected an indented block after 'try' statement on line 33 (<unknown>, line 35)

### syntax_error_early_error_detector_demo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/examples/early_error_detector_demo.py: unexpected indent (<unknown>, line 16)

### syntax_error_embedding_builder.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/embedding_builder.py: expected an indented block after 'try' statement on line 30 (<unknown>, line 32)

### syntax_error_enhanced_core_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/enhanced_core_orchestrator.py: unexpected indent (<unknown>, line 33)

### syntax_error_evaluation_driven_processor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/A_analysis_nlp/evaluation_driven_processor.py: expected an indented block after 'try' statement on line 37 (<unknown>, line 39)

### syntax_error_evidence_processor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/evidence_processor.py: expected an indented block after 'try' statement on line 23 (<unknown>, line 25)

### syntax_error_evidence_processor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/A_analysis_nlp/evidence_processor.py: unindent does not match any outer indentation level (<unknown>, line 41)

### syntax_error_evidence_router.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/evidence_router.py: expected an indented block after 'try' statement on line 22 (<unknown>, line 24)

### syntax_error_evidence_validation_model.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/evidence_validation_model.py: expected an indented block after 'try' statement on line 29 (<unknown>, line 31)

### syntax_error_feature_extractor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/feature_extractor.py: unmatched '}' (<unknown>, line 86)

### syntax_error_fix_imports_venv.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/fix_imports_venv.py: invalid syntax (<unknown>, line 152)

### syntax_error_gate_validation_system.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/I_ingestion_preparation/gate_validation_system.py: unindent does not match any outer indentation level (<unknown>, line 42)

### syntax_error_generate_flux_report.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/generate_flux_report.py: unexpected indent (<unknown>, line 24)

### syntax_error_gw_alignment.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/gw_alignment.py: expected an indented block after 'try' statement on line 16 (<unknown>, line 18)

### syntax_error_gw_alignment.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/gw_alignment.py: expected an indented block after 'if' statement on line 27 (<unknown>, line 30)

### syntax_error_handoff_validation_system.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/handoff_validation_system.py: unexpected indent (<unknown>, line 23)

### syntax_error_implementacion_mapeo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/implementacion_mapeo.py: expected an indented block after 'try' statement on line 28 (<unknown>, line 30)

### syntax_error_ingestion_enhancer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/mathematical_enhancers/ingestion_enhancer.py: expected an indented block after 'try' statement on line 35 (<unknown>, line 37)

### syntax_error_ingestion_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/I_ingestion_preparation/ingestion_orchestrator.py: unexpected indent (<unknown>, line 26)

### syntax_error_integration_demo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/analysis/integration_demo.py: unexpected indent (<unknown>, line 17)

### syntax_error_intelligent_recommendation_engine.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/intelligent_recommendation_engine.py: unexpected indent (<unknown>, line 21)

### syntax_error_knowledge_audit_demo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/knowledge/knowledge_audit_demo.py: unexpected indent (<unknown>, line 17)

### syntax_error_knowledge_enhancer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/mathematical_enhancers/knowledge_enhancer.py: expected an indented block after 'try' statement on line 70 (<unknown>, line 72)

### syntax_error_lexical_index.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/retrieval_engine/lexical_index.py: expected an indented block after 'except' statement on line 227 (<unknown>, line 230)

### syntax_error_m_c_c_engine_monotone_compliance_evaluator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/m_c_c_engine_monotone_compliance_evaluator.py: unexpected indent (<unknown>, line 20)

### syntax_error_m_c_c_label_evaluation_engine.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/m_c_c_label_evaluation_engine.py: unexpected indent (<unknown>, line 20)

### syntax_error_mathematical_compatibility_matrix.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/mathematical_compatibility_matrix.py: expected an indented block after 'except' statement on line 67 (<unknown>, line 70)

### syntax_error_mathematical_foundations.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/mathematical_foundations.py: expected an indented block after 'try' statement on line 21 (<unknown>, line 23)

### syntax_error_mathematical_pipeline_coordinator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/mathematical_enhancers/mathematical_pipeline_coordinator.py: unmatched ')' (<unknown>, line 49)

### syntax_error_meso_aggregator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/G_aggregation_reporting/meso_aggregator.py: unterminated triple-quoted string literal (detected at line 427) (<unknown>, line 404)

### syntax_error_metrics_collector.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/metrics_collector.py: unindent does not match any outer indentation level (<unknown>, line 25)

### syntax_error_models.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/models.py: expected an indented block after 'try' statement on line 16 (<unknown>, line 18)

### syntax_error_normative_validator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/normative_validator.py: expected an indented block after 'try' statement on line 1829 (<unknown>, line 1831)

### syntax_error_orchestration_enhancer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/mathematical_enhancers/orchestration_enhancer.py: invalid syntax (<unknown>, line 1029)

### syntax_error_pdf_processing_error_handler.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/pdf_processing_error_handler.py: expected an indented block after 'except' statement on line 550 (<unknown>, line 554)

### syntax_error_pdf_reader.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/pdf_reader.py: expected an indented block after 'try' statement on line 731 (<unknown>, line 733)

### syntax_error_persistence.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/dashboard/persistence.py: expected an indented block after 'except' statement on line 178 (<unknown>, line 181)

### syntax_error_pic_probe.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/pic_probe.py: unindent does not match any outer indentation level (<unknown>, line 27)

### syntax_error_pipeline_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/pipeline_orchestrator.py: expected an indented block after 'try' statement on line 17 (<unknown>, line 19)

### syntax_error_pipeline_orchestrator_audit.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/pipeline_orchestrator_audit.py: expected an indented block after 'if' statement on line 68 (<unknown>, line 70)

### syntax_error_pipeline_state_manager.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/pipeline_state_manager.py: expected an indented block after 'if' statement on line 456 (<unknown>, line 459)

### syntax_error_pipeline_value_analysis_system.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/pipeline_value_analysis_system.py: unindent does not match any outer indentation level (<unknown>, line 52)

### syntax_error_plan_diff.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/plan_diff.py: unexpected indent (<unknown>, line 18)

### syntax_error_pre_flight_validator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/mathematical_enhancers/pre_flight_validator.py: unexpected indent (<unknown>, line 38)

### syntax_error_process_inventory.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/process_inventory.py: expected an indented block after 'try' statement on line 30 (<unknown>, line 32)

### syntax_error_provenance_tracker.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/calibration_safety_governance/provenance_tracker.py: expected an indented block after 'except' statement on line 155 (<unknown>, line 158)

### syntax_error_question_analyzer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/question_analyzer.py: expected an indented block after 'try' statement on line 39 (<unknown>, line 41)

### syntax_error_question_analyzer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/A_analysis_nlp/question_analyzer.py: expected an indented block after 'try' statement on line 45 (<unknown>, line 47)

### syntax_error_question_level_scoring_pipeline.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/question_level_scoring_pipeline.py: expected an indented block after 'try' statement on line 31 (<unknown>, line 33)

### syntax_error_question_registry.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/question_registry.py: unexpected indent (<unknown>, line 14)

### syntax_error_quickstart_notebook.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/quickstart_notebook.py: unmatched ')' (<unknown>, line 47)

### syntax_error_raw_data_generator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/raw_data_generator.py: expected an indented block after 'try' statement on line 35 (<unknown>, line 37)

### syntax_error_rc_check.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/rc_check.py: unexpected indent (<unknown>, line 30)

### syntax_error_report_compiler.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/report_compiler.py: expected an indented block after 'try' statement on line 22 (<unknown>, line 26)

### syntax_error_retrieval_trace.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/retrieval_trace.py: unindent does not match any outer indentation level (<unknown>, line 20)

### syntax_error_run_basic_tests.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_basic_tests.py: unexpected indent (<unknown>, line 15)

### syntax_error_run_canonical_stability.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_canonical_stability.py: expected an indented block after 'try' statement on line 34 (<unknown>, line 36)

### syntax_error_run_contract_validation.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_contract_validation.py: expected an indented block after 'for' statement on line 92 (<unknown>, line 96)

### syntax_error_run_demo_quick.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_demo_quick.py: unexpected indent (<unknown>, line 14)

### syntax_error_run_g_aggregation_tests.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_g_aggregation_tests.py: expected an indented block after 'try' statement on line 135 (<unknown>, line 137)

### syntax_error_run_k_workflow_tests.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_k_workflow_tests.py: unexpected indent (<unknown>, line 27)

### syntax_error_run_l_stage_tests.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_l_stage_tests.py: unexpected indent (<unknown>, line 21)

### syntax_error_run_mcc_tests.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_mcc_tests.py: unexpected indent (<unknown>, line 14)

### syntax_error_run_safety_demo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_safety_demo.py: unindent does not match any outer indentation level (<unknown>, line 24)

### syntax_error_schema_registry.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/schema_registry.py: expected an indented block after 'try' statement on line 16 (<unknown>, line 18)

### syntax_error_schemas.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/contracts/schemas.py: expected an indented block after 'if' statement on line 30 (<unknown>, line 32)

### syntax_error_schemas.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/schemas.py: '(' was never closed (<unknown>, line 594)

### syntax_error_score_calculator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/score_calculator.py: expected an indented block after 'try' statement on line 28 (<unknown>, line 30)

### syntax_error_serializable_wrappers.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/serializable_wrappers.py: expected an indented block after 'try' statement on line 85 (<unknown>, line 91)

### syntax_error_service_discovery.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/service_discovery.py: expected an indented block after 'except' statement on line 279 (<unknown>, line 282)

### syntax_error_simple_g_aggregation_test.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/simple_g_aggregation_test.py: expected an indented block after 'try' statement on line 42 (<unknown>, line 44)

### syntax_error_simple_import_test.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/simple_import_test.py: unexpected indent (<unknown>, line 31)

### syntax_error_simple_integration_test.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/simple_integration_test.py: unindent does not match any outer indentation level (<unknown>, line 216)

### syntax_error_simple_test_decalogo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/simple_test_decalogo.py: unexpected indent (<unknown>, line 7)

### syntax_error_simple_test_math_enhancer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/simple_test_math_enhancer.py: unindent does not match any outer indentation level (<unknown>, line 20)

### syntax_error_snapshot_guard.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/snapshot_guard.py: unexpected indent (<unknown>, line 37)

### syntax_error_sort_sanity.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/sort_sanity.py: unexpected indent (<unknown>, line 19)

### syntax_error_src:nlp_engine:semantic_inference_engine.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/src:nlp_engine:semantic_inference_engine.py: '(' was never closed (<unknown>, line 172)

### syntax_error_stage_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/stage_orchestrator.py: expected an indented block after 'try' statement on line 35 (<unknown>, line 39)

### syntax_error_step_handlers.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/step_handlers.py: unindent does not match any outer indentation level (<unknown>, line 266)

### syntax_error_task_selector_demo.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/task_selector_demo.py: unexpected indent (<unknown>, line 19)

### syntax_error_telemetry_collector.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/telemetry_collector.py: unindent does not match any outer indentation level (<unknown>, line 24)

### syntax_error_test_08x_context_construction.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_08x_context_construction.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_09x_context_construction.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_09x_context_construction.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_10x_context_construction.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_10x_context_construction.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_10x_context_construction_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/test_10x_context_construction_component.py: invalid decimal literal (<unknown>, line 40)

### syntax_error_test_11x_context_construction.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_11x_context_construction.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_11x_context_construction_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/test_11x_context_construction_component.py: invalid decimal literal (<unknown>, line 40)

### syntax_error_test_12x_context_construction.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_12x_context_construction.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_12x_context_construction_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/test_12x_context_construction_component.py: invalid decimal literal (<unknown>, line 40)

### syntax_error_test_52s_synthesis_output.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_52s_synthesis_output.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_53s_synthesis_output.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_53s_synthesis_output.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_54s_synthesis_output.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_54s_synthesis_output.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_54s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/test_54s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 40)

### syntax_error_test_55s_synthesis_output.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_55s_synthesis_output.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_55s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/test_55s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 40)

### syntax_error_test_56s_synthesis_output.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_56s_synthesis_output.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_56s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/test_56s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 40)

### syntax_error_test_57s_synthesis_output.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_57s_synthesis_output.py: invalid decimal literal (<unknown>, line 47)

### syntax_error_test_57s_synthesis_output_component.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/S_synthesis_output/test_57s_synthesis_output_component.py: invalid decimal literal (<unknown>, line 40)

### syntax_error_test_answer_formatter.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_answer_formatter.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_api.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_api.py: unexpected indent (<unknown>, line 11)

### syntax_error_test_artifact_generator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/analysis/test_artifact_generator.py: unexpected indent (<unknown>, line 14)

### syntax_error_test_audit_logger.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/evaluation/test_audit_logger.py: unexpected indent (<unknown>, line 17)

### syntax_error_test_auto_enhancement_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_auto_enhancement_orchestrator.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_basic_functionality.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_basic_functionality.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_beir_evaluation.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_beir_evaluation.py: unmatched ')' (<unknown>, line 74)

### syntax_error_test_bmc.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_bmc.py: unexpected indent (<unknown>, line 12)

### syntax_error_test_calibration_ci.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_calibration_ci.py: expected an indented block after 'try' statement on line 43 (<unknown>, line 45)

### syntax_error_test_causal_graph_constructor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/K_knowledge_extraction/test_causal_graph_constructor.py: unexpected indent (<unknown>, line 12)

### syntax_error_test_code_quality_fixes.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_code_quality_fixes.py: expected an indented block after 'try' statement on line 41 (<unknown>, line 43)

### syntax_error_test_code_quality_fixes_minimal.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_code_quality_fixes_minimal.py: unindent does not match any outer indentation level (<unknown>, line 50)

### syntax_error_test_confluent_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_confluent_orchestrator.py: unexpected indent (<unknown>, line 16)

### syntax_error_test_conformal_prediction.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/test_conformal_prediction.py: unexpected indent (<unknown>, line 14)

### syntax_error_test_conformal_risk_control.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_conformal_risk_control.py: unexpected indent (<unknown>, line 16)

### syntax_error_test_conformal_risk_system.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_conformal_risk_system.py: unindent does not match any outer indentation level (<unknown>, line 24)

### syntax_error_test_constraint_validator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_constraint_validator.py: unexpected indent (<unknown>, line 10)

### syntax_error_test_determinism_verification.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_determinism_verification.py: unexpected indent (<unknown>, line 28)

### syntax_error_test_deterministic_retrieval.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_deterministic_retrieval.py: unexpected indent (<unknown>, line 9)

### syntax_error_test_deterministic_router.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_deterministic_router.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_deterministic_router_simple.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_deterministic_router_simple.py: unindent does not match any outer indentation level (<unknown>, line 22)

### syntax_error_test_distributed_processor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_distributed_processor.py: unindent does not match any outer indentation level (<unknown>, line 64)

### syntax_error_test_early_error_detector.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_early_error_detector.py: unexpected indent (<unknown>, line 12)

### syntax_error_test_environment_automation.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_environment_automation.py: unindent does not match any outer indentation level (<unknown>, line 18)

### syntax_error_test_evidence_adapter.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/test_evidence_adapter.py: unexpected indent (<unknown>, line 9)

### syntax_error_test_evidence_processor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_evidence_processor.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_evidence_validation.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_evidence_validation.py: unmatched ')' (<unknown>, line 93)

### syntax_error_test_ffc.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_ffc.py: unexpected indent (<unknown>, line 33)

### syntax_error_test_fixtures.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_fixtures.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_full_deterministic_router.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_full_deterministic_router.py: unindent does not match any outer indentation level (<unknown>, line 21)

### syntax_error_test_gw_alignment.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_gw_alignment.py: unexpected indent (<unknown>, line 12)

### syntax_error_test_immutable_context.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_immutable_context.py: unexpected indent (<unknown>, line 15)

### syntax_error_test_immutable_context_basic.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_immutable_context_basic.py: unindent does not match any outer indentation level (<unknown>, line 19)

### syntax_error_test_integration_layer.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_integration_layer.py: expected an indented block after 'if' statement on line 199 (<unknown>, line 202)

### syntax_error_test_k_knowledge_extraction_workflow.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/integration/test_k_knowledge_extraction_workflow.py: unindent does not match any outer indentation level (<unknown>, line 45)

### syntax_error_test_l_stage_assertions.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_l_stage_assertions.py: unexpected indent (<unknown>, line 24)

### syntax_error_test_l_stage_determinism.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_l_stage_determinism.py: unexpected indent (<unknown>, line 24)

### syntax_error_test_l_stage_preflight.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_l_stage_preflight.py: unexpected indent (<unknown>, line 23)

### syntax_error_test_library_compatibility.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_library_compatibility.py: expected an indented block after 'try' statement on line 31 (<unknown>, line 33)

### syntax_error_test_lineage_tracker.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_lineage_tracker.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_mathematical_foundations.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_mathematical_foundations.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_mathematical_safety_controller.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_mathematical_safety_controller.py: unexpected indent (<unknown>, line 17)

### syntax_error_test_meso_aggregator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_meso_aggregator.py: unexpected indent (<unknown>, line 120)

### syntax_error_test_orchestration_system.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/calibration_safety_governance/test_orchestration_system.py: unexpected indent (<unknown>, line 19)

### syntax_error_test_pdf_error_handling.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_pdf_error_handling.py: unexpected indent (<unknown>, line 17)

### syntax_error_test_permutation_invariant_processor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_permutation_invariant_processor.py: unexpected indent (<unknown>, line 17)

### syntax_error_test_pic.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_pic.py: unexpected indent (<unknown>, line 18)

### syntax_error_test_rc.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_rc.py: unmatched ')' (<unknown>, line 42)

### syntax_error_test_report_compiler.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_report_compiler.py: unexpected indent (<unknown>, line 84)

### syntax_error_test_retriever_determinism.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_retriever_determinism.py: unindent does not match any outer indentation level (<unknown>, line 45)

### syntax_error_test_routing_contract.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_routing_contract.py: unmatched ')' (<unknown>, line 13)

### syntax_error_test_safety_import.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_safety_import.py: unindent does not match any outer indentation level (<unknown>, line 14)

### syntax_error_test_schemas.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/contracts/test_schemas.py: unexpected indent (<unknown>, line 10)

### syntax_error_test_schemas.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/test_schemas.py: unexpected indent (<unknown>, line 12)

### syntax_error_test_serializable_wrappers.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_serializable_wrappers.py: unexpected indent (<unknown>, line 17)

### syntax_error_test_snapshot.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_snapshot.py: unmatched ')' (<unknown>, line 32)

### syntax_error_test_snapshot_contract.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_snapshot_contract.py: unmatched ')' (<unknown>, line 16)

### syntax_error_test_stable_gw_aligner.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_stable_gw_aligner.py: unexpected indent (<unknown>, line 11)

### syntax_error_test_submodular_selector.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_submodular_selector.py: unexpected indent (<unknown>, line 13)

### syntax_error_test_submodular_task_selector.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_submodular_task_selector.py: unexpected indent (<unknown>, line 15)

### syntax_error_test_toc.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_toc.py: unexpected indent (<unknown>, line 11)

### syntax_error_test_total_ordering.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/test_total_ordering.py: unexpected indent (<unknown>, line 4)

### syntax_error_test_unit_artifact_generator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/analysis/test_unit_artifact_generator.py: unexpected indent (<unknown>, line 18)

### syntax_error_test_visual_framework.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_visual_framework.py: unexpected indent (<unknown>, line 11)

### syntax_error_test_wasserstein_fisher_rao.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_wasserstein_fisher_rao.py: unindent does not match any outer indentation level (<unknown>, line 18)

### syntax_error_troubleshoot.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/cli/troubleshoot.py: invalid syntax (<unknown>, line 29)

### syntax_error_validate_contract_imports.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_contract_imports.py: unindent does not match any outer indentation level (<unknown>, line 23)

### syntax_error_validate_dashboard_implementation.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_dashboard_implementation.py: unindent does not match any outer indentation level (<unknown>, line 18)

### syntax_error_validate_decalogo_registry.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_decalogo_registry.py: unindent does not match any outer indentation level (<unknown>, line 19)

### syntax_error_validate_dependencies.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_dependencies.py: unindent does not match any outer indentation level (<unknown>, line 42)

### syntax_error_validate_dependency_compatibility.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/scripts/validate_dependency_compatibility.py: unindent does not match any outer indentation level (<unknown>, line 116)

### syntax_error_validate_g_aggregation.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_g_aggregation.py: expected an indented block after 'try' statement on line 179 (<unknown>, line 181)

### syntax_error_validate_handoff_system.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_handoff_system.py: unexpected indent (<unknown>, line 13)

### syntax_error_validate_import_safety.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_import_safety.py: unindent does not match any outer indentation level (<unknown>, line 17)

### syntax_error_validate_l_orchestrator.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_l_orchestrator.py: expected an indented block after 'if' statement on line 351 (<unknown>, line 355)

### syntax_error_validate_l_stage_tests.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_l_stage_tests.py: unindent does not match any outer indentation level (<unknown>, line 21)

### syntax_error_validate_mathematical_foundations.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_mathematical_foundations.py: unindent does not match any outer indentation level (<unknown>, line 102)

### syntax_error_validate_mathematical_pipeline.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_mathematical_pipeline.py: unindent does not match any outer indentation level (<unknown>, line 11)

### syntax_error_validate_monitoring.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_monitoring.py: unindent does not match any outer indentation level (<unknown>, line 13)

### syntax_error_validate_operadic_implementation.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_operadic_implementation.py: unexpected indent (<unknown>, line 27)

### syntax_error_validate_parallel_processor.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_parallel_processor.py: unindent does not match any outer indentation level (<unknown>, line 31)

### syntax_error_validate_pipeline_analysis.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_pipeline_analysis.py: unindent does not match any outer indentation level (<unknown>, line 22)

### syntax_error_validate_recovery_system.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_recovery_system.py: unindent does not match any outer indentation level (<unknown>, line 26)

### syntax_error_validate_safety_controller.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_safety_controller.py: unindent does not match any outer indentation level (<unknown>, line 23)

### syntax_error_validate_stage_middleware.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_stage_middleware.py: unexpected indent (<unknown>, line 10)

### syntax_error_vector_index.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/retrieval_engine/vector_index.py: expected an indented block after 'try' statement on line 62 (<unknown>, line 64)

### syntax_error_workflow_engine.py
- status: FAIL
- severity: high
- message: Syntax error in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/workflow_engine.py: unexpected indent (<unknown>, line 23)

### syntax_errors
- status: FAIL
- severity: high
- message: Found 235 files with syntax errors
- remediation: Fix syntax errors in Python files

## WARN (43)

### external_api_dependencies
- status: WARN
- severity: medium
- message: Found 10 files with external API calls
- remediation: Ensure proper error handling and timeouts for external API calls
<details><summary>details</summary>


```json
{
  "files": [
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_atroz_server.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/alert_system.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_installation.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/atroz_api_demo.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/service_discovery.py"
  ]
}
```

</details>

### import_analysis_05I_raw_data_generator.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/I_ingestion_preparation/05I_raw_data_generator.py: name 'Set' is not defined

### import_analysis___init__.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/adapters/__init__.py: name 'Enum' is not defined

### import_analysis___init__.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/__init__.py: name 'Dict' is not defined

### import_analysis___init__.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/phases/I/__init__.py: unexpected indent (__init__.py, line 25)

### import_analysis___init__.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/phases/L/__init__.py: unexpected indent (__init__.py, line 4)

### import_analysis___init__.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/__init__.py: expected an indented block after 'try' statement on line 16 (gw_alignment.py, line 18)

### import_analysis_anti_corruption_adapters.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tools/anti_corruption_adapters.py: name 'Enum' is not defined

### import_analysis_api.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/standards_alignment/api.py: name 'Any' is not defined

### import_analysis_audit_last_execution.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/audit_last_execution.py: expected an indented block after 'if' statement on line 68 (pipeline_orchestrator_audit.py, line 70)

### import_analysis_bridge_analysis_enhancer.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/A_analysis_nlp/bridge_analysis_enhancer.py: expected an indented block after 'try' statement on line 35 (analysis_enhancer.py, line 38)

### import_analysis_bridge_context_enhancer.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/X_context_construction/bridge_context_enhancer.py: expected an indented block after 'try' statement on line 34 (context_enhancer.py, line 36)

### import_analysis_bridge_ingestion_enhancer.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/I_ingestion_preparation/bridge_ingestion_enhancer.py: expected an indented block after 'try' statement on line 35 (ingestion_enhancer.py, line 37)

### import_analysis_bridge_knowledge_enhancer.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/K_knowledge_extraction/bridge_knowledge_enhancer.py: expected an indented block after 'try' statement on line 70 (knowledge_enhancer.py, line 72)

### import_analysis_bridge_orchestration_enhancer.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/O_orchestration_control/bridge_orchestration_enhancer.py: invalid syntax (orchestration_enhancer.py, line 1029)

### import_analysis_canonical_output_auditor.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_output_auditor.py: name 'dataclass' is not defined

### import_analysis_comprehensive_pipeline_orchestrator.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/comprehensive_pipeline_orchestrator.py: invalid syntax (orchestration_enhancer.py, line 1029)

### import_analysis_demo_adapter_usage.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/adapters/demo_adapter_usage.py: name 'Enum' is not defined

### import_analysis_event_bus_demo.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/examples/event_bus_demo.py: expected an indented block after 'if' statement on line 431 (confluent_orchestrator.py, line 434)

### import_analysis_gcp_io.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/gcp_io.py: name 'Any' is not defined

### import_analysis_import_blocker.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/adapters/import_blocker.py: name 'Enum' is not defined

### import_analysis_integration_example.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/adapters/integration_example.py: name 'Enum' is not defined

### import_analysis_packager.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/packager.py: name 'Any' is not defined

### import_analysis_retrieval_analysis_adapter.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/adapters/retrieval_analysis_adapter.py: name 'Enum' is not defined

### import_analysis_run_canonical_audit_demo.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/run_canonical_audit_demo.py: name 'dataclass' is not defined

### import_analysis_runtime_import_guard.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/scripts/runtime_import_guard.py: expected an indented block after 'if' statement on line 431 (confluent_orchestrator.py, line 434)

### import_analysis_simple_test_final_report.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/simple_test_final_report.py: expected an indented block after 'try' statement on line 22 (report_compiler.py, line 26)

### import_analysis_smoke_tests.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/smoke_tests.py: expected an indented block after 'try' statement on line 33 (dnp_alignment_adapter.py, line 35)

### import_analysis_test_event_bus.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_event_bus.py: expected an indented block after 'if' statement on line 431 (confluent_orchestrator.py, line 434)

### import_analysis_test_event_driven_orchestrator.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_event_driven_orchestrator.py: expected an indented block after 'if' statement on line 431 (confluent_orchestrator.py, line 434)

### import_analysis_test_event_schemas.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_event_schemas.py: expected an indented block after 'if' statement on line 431 (confluent_orchestrator.py, line 434)

### import_analysis_test_hash_policies_simple.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_hash_policies_simple.py: expected an indented block after 'if' statement on line 431 (confluent_orchestrator.py, line 434)

### import_analysis_test_ingestion_preparation_dag.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/tests/integration/test_ingestion_preparation_dag.py: name 'Any' is not defined

### import_analysis_test_recovery_scripts.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_recovery_scripts.py: name 'Dict' is not defined

### import_analysis_test_synthesis_integration.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_synthesis_integration.py: unterminated triple-quoted string literal (detected at line 806) (answer_synthesizer.py, line 749)

### import_analysis_test_validator_adapters.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_validator_adapters.py: expected an indented block after 'if' statement on line 431 (confluent_orchestrator.py, line 434)

### import_analysis_validate_advanced_loader.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_advanced_loader.py: expected an indented block after 'try' statement on line 303 (advanced_loader.py, line 305)

### import_analysis_validate_installation.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_installation.py: expected an indented block after 'if' statement on line 431 (confluent_orchestrator.py, line 434)

### import_analysis_validate_mathematical_compatibility_matrix.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_mathematical_compatibility_matrix.py: name 'Enum' is not defined

### import_analysis_validate_script.py
- status: WARN
- severity: medium
- message: Could not analyze imports in /Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_script.py: name 'Path' is not defined

### python_version
- status: WARN
- severity: medium
- message: Python 3.13.5 may have ecosystem gaps
- remediation: Consider downgrading to Python 3.11 or 3.12 for better stability
<details><summary>details</summary>


```json
{
  "version": "3.13.5"
}
```

</details>

### static_analysis_tools
- status: WARN
- severity: medium
- message: No static analysis tools found
- remediation: Install tools like mypy, flake8, black for code quality

### type_hint_coverage
- status: WARN
- severity: low
- message: Type hint coverage: 65.6% (524/799 files)
- remediation: Add type hints for better code clarity

## PASS (33)

### ci_cd_configuration
- status: PASS
- severity: medium
- message: Found CI/CD configuration: .github/workflows/
<details><summary>details</summary>


```json
{
  "files": [
    ".github/workflows/"
  ]
}
```

</details>

### configuration_files
- status: PASS
- severity: medium
- message: Found 1 configuration files
<details><summary>details</summary>


```json
{
  "files": [
    "pyproject.toml"
  ]
}
```

</details>

### contract_files_present
- status: PASS
- severity: medium
- message: Found 4 contract system files
<details><summary>details</summary>


```json
{
  "files": [
    "contract_validator.py",
    "constraint_validator.py",
    "rubric_validator.py",
    "run_contract_validation.py"
  ]
}
```

</details>

### cpu_usage
- status: PASS
- severity: medium
- message: CPU usage: 0.0%
<details><summary>details</summary>


```json
{
  "cpu_percent": 0.0
}
```

</details>

### cpu_usage
- status: PASS
- severity: medium
- message: CPU usage: 0.0%
<details><summary>details</summary>


```json
{
  "cpu_percent": 0.0
}
```

</details>

### critical_file_README.md
- status: PASS
- severity: medium
- message: Critical file README.md exists
<details><summary>details</summary>


```json
{
  "size": 15521
}
```

</details>

### critical_file_pyproject.toml
- status: PASS
- severity: medium
- message: Critical file pyproject.toml exists
<details><summary>details</summary>


```json
{
  "size": 7844
}
```

</details>

### critical_file_requirements-windows.txt
- status: PASS
- severity: medium
- message: Created requirements-windows.txt
- remediation: Create requirements-windows.txt file

### critical_file_requirements.txt
- status: PASS
- severity: medium
- message: Critical file requirements.txt exists
<details><summary>details</summary>


```json
{
  "size": 3555
}
```

</details>

### critical_file_setup.py
- status: PASS
- severity: medium
- message: Critical file setup.py exists
<details><summary>details</summary>


```json
{
  "size": 2292
}
```

</details>

### data_files
- status: PASS
- severity: medium
- message: Found 232 data files (101.3MB total)
<details><summary>details</summary>


```json
{
  "file_count": 232,
  "total_size_mb": 101.31029987335205,
  "extensions": [
    ".csv",
    ".json"
  ]
}
```

</details>

### database_usage
- status: PASS
- severity: medium
- message: Found database usage in 21 files
<details><summary>details</summary>


```json
{
  "files": [
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_recovery_system.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/test_owner_assignment_system.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/recovery_system.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_owner_assignment_system.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/distributed_processor.py"
  ],
  "db_types": [
    "redis\\.",
    "psycopg2",
    "sqlalchemy",
    "sqlite3\\."
  ]
}
```

</details>

### disk_space
- status: PASS
- severity: medium
- message: Adequate disk space: 5.93GB free

### documentation_coverage
- status: PASS
- severity: low
- message: Documentation coverage: 97.7% (781/799 files)
- remediation: Add docstrings to modules and functions

### documentation_files
- status: PASS
- severity: medium
- message: Found documentation files: README.md, docs/
<details><summary>details</summary>


```json
{
  "files": [
    "README.md",
    "docs/"
  ]
}
```

</details>

### file_permissions
- status: PASS
- severity: medium
- message: All Python files are readable

### git_available
- status: PASS
- severity: medium
- message: Git is available: git version 2.39.5 (Apple Git-154)

### hardcoded_secrets
- status: PASS
- severity: medium
- message: No obvious hardcoded secrets found

### logging_configuration
- status: PASS
- severity: medium
- message: Logging configuration found

### memory_management_patterns
- status: PASS
- severity: medium
- message: Memory management patterns: 49.9% of files (399/799)
- remediation: Implement proper memory management with context managers, caching, and cleanup
<details><summary>details</summary>


```json
{
  "percentage": 49.93742177722152
}
```

</details>

### memory_usage
- status: PASS
- severity: low
- message: Memory usage: 50.0%

### memory_usage
- status: PASS
- severity: low
- message: Memory usage: 50.0% (4.0GB used / 8.0GB total)
<details><summary>details</summary>


```json
{
  "percent": 50.0,
  "used_gb": 4.0,
  "total_gb": 8.0,
  "available_gb": 4.0
}
```

</details>

### monitoring_setup
- status: PASS
- severity: medium
- message: Found monitoring setup in 282 files
<details><summary>details</summary>


```json
{
  "files": [
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/evidence_system.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/visual_testing_framework.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/dependency_analysis_module.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/module_distributed_processor.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/airflow_orchestrator.py"
  ],
  "monitoring_types": [
    "prometheus_client",
    "logging\\.",
    "opentelemetry"
  ]
}
```

</details>

### network_connectivity
- status: PASS
- severity: medium
- message: Basic network connectivity checks skipped in this environment

### orchestrator_files
- status: PASS
- severity: medium
- message: Found 29 orchestrator files
<details><summary>details</summary>


```json
{
  "files": [
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/airflow_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/event_driven_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/refactored_pipeline_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_l_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/validate_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/analysis_nlp_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/core_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/pipeline_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/pipeline_orchestrator_audit.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/debug_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/comprehensive_pipeline_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/enhanced_core_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/calibration_safety_governance/orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/event_driven_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/confluent_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/core/auto_enhancement_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_auto_enhancement_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_event_driven_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/egw_query_expansion/tests/test_confluent_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/A_analysis_nlp/refactored_stage_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/O_orchestration_control/airflow_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/O_orchestration_control/core_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/O_orchestration_control/confluent_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/O_orchestration_control/enhanced_core_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/stage_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/demo_stage_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/L_classification_evaluation/test_stage_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/canonical_flow/I_ingestion_preparation/ingestion_orchestrator.py",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/PIPELINEORCHESTRATOR.py"
  ]
}
```

</details>

### phase_annotated_files
- status: PASS
- severity: medium
- message: Found 252 files with phase annotations
<details><summary>details</summary>


```json
{
  "phase_distribution": {
    "O": 109,
    "A": 28,
    "X": 7,
    "I": 13,
    "S": 4,
    "K": 19,
    "G": 9,
    "T": 7,
    "L": 11,
    "{phase}": 3,
    "R": 15,
    "{phase.value}": 2,
    "{{ phase }}": 1,
    "context": 5,
    "context_construction": 4,
    "T_integration_storage": 1,
    "synthesis": 6,
    "synthesis_output": 6,
    "K_knowledge_extraction": 1,
    "I_ingestion_preparation": 1
  }
}
```

</details>

### pip_available
- status: PASS
- severity: medium
- message: pip is available (version 25.2)
<details><summary>details</summary>


```json
{
  "pip_version": "25.2"
}
```

</details>

### python_files_count
- status: PASS
- severity: medium
- message: Found 799 Python files to analyze

### recovery_files
- status: PASS
- severity: medium
- message: Found 5 recovery system files
<details><summary>details</summary>


```json
{
  "files": [
    "validate_recovery_system.py",
    "validate_safety_controller.py",
    "compensation_engine.py",
    "circuit_breaker.py",
    "exception_monitoring.py"
  ]
}
```

</details>

### requirements_files
- status: PASS
- severity: medium
- message: Found 10 requirements files
<details><summary>details</summary>


```json
{
  "files": [
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/requirements.txt",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/requirements-core.txt",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/requirements_visual_testing.txt",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/requirements_minimal.txt",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/requirements-essential.txt",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/requirements-minimal.txt",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/requirements_core.txt",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/requirements-deletion-system.txt",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/pyproject.toml",
    "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/setup.py"
  ]
}
```

</details>

### system_resources
- status: PASS
- severity: medium
- message: System: 8 CPUs, 8.0GB RAM
<details><summary>details</summary>


```json
{
  "cpu_count": 8,
  "memory_total_gb": 8.0,
  "memory_available_gb": 4.0,
  "memory_percent": 50.0
}
```

</details>

### thresholds_configuration
- status: PASS
- severity: medium
- message: Thresholds configuration loaded successfully

### virtual_environment
- status: PASS
- severity: medium
- message: Virtual environment is active
<details><summary>details</summary>


```json
{
  "venv_path": "/Users/recovered/PycharmProjects/FARFAN-ULTIMATE-main/.venv"
}
```

</details>

## SKIP (1)

### dependency_conflicts
- status: SKIP
- severity: medium
- message: Could not check dependency conflicts: No module named 'pkg_resources'
