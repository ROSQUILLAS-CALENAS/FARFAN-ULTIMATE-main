#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from canonical_flow.A_analysis_nlp.decalogo_question_registry import create_decalogo_question_registry

registry = create_decalogo_question_registry()
validation = registry.validate_registry()

print('Registry Validation Results:')
print(f'Valid: {validation.is_valid}')
print(f'Total Questions: {validation.total_questions}')
print(f'Expected Total: 470')
print(f'Questions per Point Distribution: {validation.questions_per_point}')
print(f'Errors: {len(validation.errors)}')
print(f'Warnings: {len(validation.warnings)}')

if validation.errors:
    for error in validation.errors:
        print(f'ERROR: {error}')
        
if validation.warnings:
    for warning in validation.warnings:
        print(f'WARNING: {warning}')

# Test stable ordering
ids1 = registry.get_stable_iteration_order()
registry2 = create_decalogo_question_registry()
ids2 = registry2.get_stable_iteration_order()

print(f'Stable ordering test: {ids1 == ids2}')
print(f'Sample question IDs: {ids1[:5]}')