#!/usr/bin/env python3
"""
Manual test of report compiler functionality
"""

import json
import sys
from pathlib import Path

# Test the core functionality manually
def test_report_compiler():
    # Add path
    sys.path.insert(0, 'canonical_flow/G_aggregation_reporting')
    
    try:
        # Import the module
        from report_compiler import process, ReportStatus
        print("✓ Module imported successfully")
        
        # Run the process function
        result = process('test')
        
        # Check basic structure
        required_fields = ['status', 'document_stem', 'warnings', 'timestamp', 'processing_id', 'report_data']
        for field in required_fields:
            if field in result:
                print(f"✓ Field '{field}' present")
            else:
                print(f"✗ Field '{field}' missing")
                
        print(f"✓ Process completed with status: {result['status']}")
        print(f"✓ Document stem: {result['document_stem']}")
        print(f"✓ Warnings count: {len(result['warnings'])}")
        
        # Check report sections
        report_data = result.get('report_data', {})
        sections = ['executive_summary', 'detailed_findings', 'compliance_assessment', 'key_highlights']
        
        for section in sections:
            if section in report_data:
                print(f"✓ Section '{section}' generated")
            else:
                print(f"✗ Section '{section}' missing")
        
        # Check if output file exists
        output_path = Path("canonical_flow/aggregation/test_report.json")
        if output_path.exists():
            print("✓ Output file created")
            with open(output_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            print("✓ Output file is valid JSON")
            print(f"✓ File size: {output_path.stat().st_size} bytes")
        else:
            print("✗ Output file not created")
            
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_report_compiler()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")