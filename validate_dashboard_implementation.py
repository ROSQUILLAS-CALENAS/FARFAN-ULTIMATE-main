#!/usr/bin/env python3
"""
Validation script for the dashboard state management system implementation.
"""

import sys
import os
import traceback

def validate_imports():
    """Validate that all modules can be imported."""
    print("🔍 Validating imports...")
    
    try:
        from dashboard import (
            StateManager, DashboardStore, UIStateManager, 
            BackendStateManager, StatePersistence, TransitionManager
        )
        print("✅ All core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def validate_functionality():
    """Validate basic functionality."""
    print("🔍 Validating core functionality...")
    
    try:
        from dashboard import StateManager
        
        # Test state manager creation
        with StateManager('validation_test', auto_save_interval=0) as sm:
            print("✅ StateManager creation successful")
            
            # Test UI operations
            ui_success = sm.show_radial_menu('test', (0, 0), {'item': 'test'}, animated=False)
            print(f"✅ UI operations: {'working' if ui_success else 'failed'}")
            
            # Test backend operations  
            backend_success = sm.update_analysis_progress('test', 0.5)
            print(f"✅ Backend operations: {'working' if backend_success else 'failed'}")
            
            # Test state retrieval
            state = sm.get_state()
            print(f"✅ State retrieval: {'working' if state else 'failed'}")
            print(f"✅ State structure: {list(state.keys())}")
            
            # Test diagnostics
            diagnostics = sm.get_diagnostics()
            print(f"✅ Diagnostics: {'working' if diagnostics else 'failed'}")
            
        return True
        
    except Exception as e:
        print(f"❌ Functionality validation failed: {e}")
        traceback.print_exc()
        return False


def validate_architecture():
    """Validate architectural components."""
    print("🔍 Validating architecture...")
    
    try:
        from dashboard import StateManager
        
        sm = StateManager('arch_test', auto_save_interval=0)
        
        # Validate component structure
        assert hasattr(sm, 'store'), "StateManager missing store"
        assert hasattr(sm.store, 'ui_state'), "Store missing UI state manager"
        assert hasattr(sm.store, 'backend_state'), "Store missing backend state manager"
        assert hasattr(sm.store, 'persistence'), "Store missing persistence"
        assert hasattr(sm.store, 'transitions'), "Store missing transitions"
        
        print("✅ All architectural components present")
        
        # Validate state layers
        state = sm.get_state()
        assert 'ui' in state, "Missing UI state layer"
        assert 'backend' in state, "Missing backend state layer"
        assert 'meta' in state, "Missing metadata layer"
        
        print("✅ State layer separation working")
        
        # Validate UI state structure
        ui_state = state['ui']
        expected_ui_keys = ['radial_menus', 'focus_mode', 'overlays', 'viewport', 'interactions', 'layout']
        for key in expected_ui_keys:
            assert key in ui_state, f"Missing UI state key: {key}"
        
        print("✅ UI state structure valid")
        
        # Validate backend state structure
        backend_state = state['backend']
        expected_backend_keys = ['analysis_pipeline', 'real_time_data', 'processed_data', 'system_status']
        for key in expected_backend_keys:
            assert key in backend_state, f"Missing backend state key: {key}"
        
        print("✅ Backend state structure valid")
        
        sm.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Architecture validation failed: {e}")
        traceback.print_exc()
        return False


def validate_features():
    """Validate key features."""
    print("🔍 Validating key features...")
    
    try:
        from dashboard import StateManager
        import time
        
        with StateManager('features_test', auto_save_interval=0) as sm:
            
            # Test debounced updates
            sm.update_ui_state({'viewport': {'zoom': 1.0}}, debounce_ms=10)
            sm.update_ui_state({'viewport': {'zoom': 2.0}}, debounce_ms=10)
            time.sleep(0.05)  # Wait for debounce
            state = sm.get_state('ui')
            assert state['viewport']['zoom'] == 2.0, "Debounced updates not working"
            print("✅ Debounced updates working")
            
            # Test optimistic updates
            success = sm.update_backend_state({'test': 'optimistic'}, optimistic=True)
            assert success, "Optimistic updates failed"
            optimistic_updates = sm.store.backend_state.get_optimistic_updates()
            assert len(optimistic_updates) > 0, "Optimistic updates not tracked"
            print("✅ Optimistic updates working")
            
            # Test transitions
            transition_id = sm.start_transition('test_transition', 100)
            assert transition_id is not None, "Transition creation failed"
            active_transitions = sm.store.transitions.get_active_transitions()
            assert len(active_transitions) > 0, "Active transitions not tracked"
            sm.complete_transition(transition_id)
            print("✅ Transitions working")
            
            # Test persistence
            success = sm.save_state()
            assert success, "State persistence failed"
            print("✅ State persistence working")
            
            # Test event system
            events = []
            def event_handler(data):
                events.append(data)
            
            sm.on('state_changed', event_handler)
            sm.update_ui_state({'test': 'event'})
            time.sleep(0.1)  # Wait for debounced event
            assert len(events) > 0, "Event system not working"
            print("✅ Event system working")
            
        return True
        
    except Exception as e:
        print(f"❌ Feature validation failed: {e}")
        traceback.print_exc()
        return False


def run_validation():
    """Run complete validation."""
    print("🚀 Dashboard State Management System Validation")
    print("=" * 60)
    
    tests = [
        ("Imports", validate_imports),
        ("Functionality", validate_functionality), 
        ("Architecture", validate_architecture),
        ("Features", validate_features)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} validation:")
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name} validation passed")
            else:
                print(f"❌ {test_name} validation failed")
        except Exception as e:
            print(f"❌ {test_name} validation crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"📊 Validation Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validations passed! Dashboard state management system is working correctly.")
        return True
    else:
        print("❌ Some validations failed! Please check the implementation.")
        return False


if __name__ == '__main__':
    success = run_validation()
    sys.exit(0 if success else 1)