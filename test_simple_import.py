"""Very simple syntax check for import_safety module"""

def check_syntax():
    try:
        with open('egw_query_expansion/core/import_safety.py', 'r') as f:
            code = f.read()
        
        # Basic syntax compilation check
        compile(code, 'egw_query_expansion/core/import_safety.py', 'exec')
        print("✓ Syntax check passed")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == '__main__':
    check_syntax()