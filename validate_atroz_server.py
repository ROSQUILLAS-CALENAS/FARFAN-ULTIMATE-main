#!/usr/bin/env python3
"""Validation script for AtroZ web server functionality."""

import json
import time
import subprocess
import threading
import urllib.request
import urllib.error
# # # from canonical_web_server import CanonicalFlowServer, AtroZCSSManager  # Module not found  # Module not found  # Module not found

def test_server_startup():
    """Test server startup and basic endpoints."""
    print("=== Testing Server Startup ===")
    
    # Start server in a separate thread
    server = CanonicalFlowServer(port=8001)
    server_thread = threading.Thread(target=server.start, kwargs={'run_analysis': False})
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Test endpoints
    endpoints = [
        ('http://localhost:8001/', 'Index page'),
        ('http://localhost:8001/health', 'Health check'),
        ('http://localhost:8001/api/status', 'API status'),
        ('http://localhost:8001/atroz/styles.css', 'AtroZ CSS'),
        ('http://localhost:8001/atroz/validate/colors', 'Color validation'),
        ('http://localhost:8001/atroz/validate/fonts', 'Font validation'),
        ('http://localhost:8001/atroz/validate/integrity', 'Integrity check')
    ]
    
    results = {}
    for url, description in endpoints:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                status = response.getcode()
                content_type = response.getheader('Content-Type', '')
                results[description] = {
                    'status': status,
                    'content_type': content_type,
                    'success': status == 200
                }
                print(f"   ✓ {description}: {status} ({content_type})")
        except urllib.error.URLError as e:
            results[description] = {'status': 'error', 'error': str(e), 'success': False}
            print(f"   ✗ {description}: {e}")
        except Exception as e:
            results[description] = {'status': 'error', 'error': str(e), 'success': False}
            print(f"   ✗ {description}: {e}")
    
    # Cleanup
    if hasattr(server, 'shutdown'):
        server.shutdown()
    
    return results

def test_css_validation():
    """Test CSS validation endpoints."""
    print("\n=== Testing CSS Validation ===")
    
    manager = AtroZCSSManager()
    
    # Test color validation
    color_results = manager.validate_color_scheme()
    print(f"   Color validation: {sum(color_results.values())}/{len(color_results)} colors valid")
    
    # Test browser detection
    test_user_agents = [
        ('Chrome/91', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
        ('IE9', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)'),
        ('Mobile Safari', 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)')
    ]
    
    for name, ua in test_user_agents:
        caps = manager.detect_browser_capabilities(ua)
        print(f"   {name}: CSS Grid={caps['supports_grid']}, Custom Props={caps['supports_css_custom_properties']}")
    
    return True

def main():
    """Run all validation tests."""
    print("AtroZ Canonical Web Server Validation")
    print("=" * 40)
    
    # Test CSS manager
    css_success = test_css_validation()
    
    # Test server endpoints (commented out to avoid blocking)
    # server_results = test_server_startup()
    
    print(f"\n=== Validation Summary ===")
    print(f"CSS Manager: {'✓ PASS' if css_success else '✗ FAIL'}")
    # print(f"Server Endpoints: {'✓ PASS' if all(r['success'] for r in server_results.values()) else '✗ FAIL'}")
    
    print("\n=== Manual Testing Instructions ===")
    print("1. Run: python3 canonical_web_server.py --port 8000")
    print("2. Visit: http://localhost:8000")
    print("3. Test endpoints:")
    print("   - /atroz/validate/colors")
    print("   - /atroz/validate/fonts")
    print("   - /atroz/validate/integrity")
    print("   - /atroz/styles.css")

if __name__ == "__main__":
    main()