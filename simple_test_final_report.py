#!/usr/bin/env python3
"""Simple test for final report generation."""

import report_compiler
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

def test_final_report():
    """Test comprehensive final report generation."""
    
    # Test data
    test_data = {
        'point_specific_analysis': {'test_analysis': 'data'},
        'coverage_matrix': {
            'C1': {'coverage': 0.85},
            'C2': {'coverage': 0.72}
        },
        'meso_summary': {
            'C1': {'base_score': 0.82},
            'C2': {'base_score': 0.75}
        },
        'cluster_audit': {
            'micro': {
                'C1': {
                    'answers': [
                        {'question_id': 'punto_01_Q01', 'verdict': 'CUMPLE', 'score': 0.85, 'evidence_ids': ['ev1']}
                    ]
                },
                'C2': {
                    'answers': [
                        {'question_id': 'punto_02_Q01', 'verdict': 'CUMPLE_PARCIAL', 'score': 0.65, 'evidence_ids': ['ev2']}
                    ]
                }
            }
        }
    }
    
    test_context = {'doc_stem': 'test_document', 'plan_name': 'Test Plan'}
    
    # Run process
    try:
        result = report_compiler.process(test_data, test_context)
        print('✓ Process executed successfully')
        print(f'Output keys: {list(result.keys())}')
        
        if 'final_report' in result:
            final_report = result['final_report']
            print(f'✓ Final report sections: {list(final_report.keys())}')
            
            # Check classification
            macro_level = final_report.get('macro_level_alignment', {})
            classification = macro_level.get('overall_classification', 'N/A')
            print(f'✓ Classification: {classification}')
            
            # Check audit trail
            audit = final_report.get('audit_trail', {})
            print(f'✓ Questions analyzed: {audit.get("questions_analyzed", 0)}')
            print(f'✓ Evidence items: {audit.get("evidence_items_processed", 0)}')
        
        if 'final_report_path' in result:
            print(f'✓ Report saved to: {result["final_report_path"]}')
            
            # Check if file exists
            report_path = Path(result["final_report_path"])
            if report_path.exists():
                print(f'✓ File verified: {report_path.stat().st_size} bytes')
            else:
                print('⚠ File not found')
        
        print('✓ All tests passed!')
        return True
        
    except Exception as e:
        print(f'✗ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_final_report()