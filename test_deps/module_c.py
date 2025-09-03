from test_deps.module_a import function_a

def function_c():
    # This creates a circular dependency
    return 'C'