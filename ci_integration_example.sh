#!/bin/bash
# CI/CD Integration Example for Pipeline Index System
# This script demonstrates how to integrate the pipeline index validation into CI/CD

set -e  # Exit on any error

echo "🔄 CI/CD Pipeline Index Validation"
echo "=================================="

# Step 1: Validate pipeline index consistency
echo "1️⃣  Validating pipeline index..."
python3 validate_pipeline_index.py

# Capture exit code for later use
VALIDATION_EXIT_CODE=$?

echo ""
echo "📊 Validation Results:"
if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    echo "   ✅ Pipeline index validation PASSED"
    echo "   ✅ Build can proceed"
else
    echo "   ❌ Pipeline index validation FAILED"
    echo "   ❌ Build should be aborted"
    
    echo ""
    echo "🔧 Common fixes:"
    echo "   - Run: python3 pipeline_index_system.py --reconcile"
    echo "   - Review validation report in validation_reports/"
    echo "   - Commit updated index.json if changes are legitimate"
    
    exit 1
fi

echo ""
echo "📋 Additional checks that could be added:"
echo "   - Static code analysis"
echo "   - Dependency vulnerability scanning"
echo "   - Component interface validation"
echo "   - Performance regression testing"

echo ""
echo "✨ Pipeline validation completed successfully!"