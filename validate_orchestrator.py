#!/usr/bin/env python3
"""
Validate the comprehensive pipeline orchestrator
"""

# # # from comprehensive_pipeline_orchestrator import ComprehensivePipelineOrchestrator  # Module not found  # Module not found  # Module not found

def main():
    orch = ComprehensivePipelineOrchestrator()
    execution_order = orch._topological_sort()
    
    print('First 10 modules in execution order:')
    for i, module in enumerate(execution_order[:10]):
        print(f'{i+1:2d}. {module}')
    
    print('\nLast 10 modules in execution order:')
    for i, module in enumerate(execution_order[-10:], len(execution_order)-9):
        print(f'{i:2d}. {module}')
    
    print(f'\nTotal modules: {len(execution_order)}')
    
    # Find position of key modules
    key_modules = ['adaptive_controller.py', 'semantic_reranking/reranker.py', 'hybrid_retrieval.py']
    for module in key_modules:
        if module in execution_order:
            pos = execution_order.index(module)
            print(f'{module}: position {pos+1}')

if __name__ == "__main__":
    main()