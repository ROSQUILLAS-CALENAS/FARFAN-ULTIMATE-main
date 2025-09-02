
Component Fix Guide: 04I (normative_validator)
==================================================

Module Path: canonical_flow.I_ingestion_preparation.normative_validator
Expected Methods: process, validate, check_compliance

Common Issues and Fixes:
1. Import Failures:
   - Check if module file exists
   - Verify dependencies are installed
   - Fix syntax errors in module

2. Placeholder Implementations:
   - Replace 'process' method placeholder with real implementation
   - Remove TODO/stub comments
   - Implement actual functionality

3. Missing Methods:
   - Add required methods: process, validate, check_compliance
   - Ensure methods are callable functions
   - Add proper error handling

4. Smoke Test Failures:
   - Test with real PDF documents from planes_input/
   - Handle edge cases and exceptions
   - Return meaningful results (not None or error dicts)

Production Readiness Criteria:
✅ Module imports successfully
✅ All required methods exist and are callable  
✅ No placeholder code detected
✅ Smoke tests pass with real data
✅ Proper error handling implemented
