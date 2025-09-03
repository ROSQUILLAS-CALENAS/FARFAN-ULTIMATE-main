#!/usr/bin/env python3

# Test the template formatting issue - working version

template = """
Component {code} with {code_lower}
"""

spec_code = "08X"

try:
    result = template.format(code=spec_code, code_lower=spec_code.lower())
    print("Success:", result)
except Exception as e:
    print("Error:", e)
    print("Type of spec_code:", type(spec_code))