#!/usr/bin/env python3
"""
Startup Time Performance Test
Measures the time taken to initialize Frankenstino AI components
"""

import time
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_startup_time():
    """Test startup time with ComponentFactory"""
    print("=== STARTUP TIME PERFORMANCE TEST ===\n")

    # Record start time
    start_time = time.time()

    try:
        # Import and initialize components
        print("Importing ComponentFactory...")
        from backend.utils.component_factory import ComponentFactory

        print("Initializing components...")
        ComponentFactory.initialize_components()

        # Record end time
        end_time = time.time()
        startup_time = end_time - start_time

        print(f"Total startup time: {startup_time:.2f} seconds")
        print(f"Time to initialize: {startup_time:.2f} seconds")
        # Get initialization status
        status = ComponentFactory.get_initialization_status()
        print(f"Components initialized: {sum(status['components'].values())}/{len(status['components'])}")

        # Test basic functionality
        print("\nTesting basic functionality...")
        mem_manager = ComponentFactory.get_memory_manager()
        stats = mem_manager.get_memory_stats()
        print(f"Memory stats retrieved: {len(stats)} keys")

        llm = ComponentFactory.get_llm_core()
        status = llm.get_model_status()
        print(f"LLM status: {status.get('status', 'unknown')}")

        # Cleanup
        ComponentFactory.shutdown()

        return startup_time

    except Exception as e:
        print(f"Error during startup test: {e}")
        return None

def test_lazy_loading_comparison():
    """Compare with old lazy loading approach (simulated)"""
    print("\n=== LAZY LOADING COMPARISON (ESTIMATED) ===")

    # This would be the old approach - each component lazy loaded on first access
    # Estimated times based on typical lazy loading patterns
    lazy_loading_estimate = 45.0  # seconds (from EFFICIENCY.MD baseline)

    print(f"Estimated lazy loading time: {lazy_loading_estimate:.1f} seconds")
    print("Note: Old lazy loading would initialize components on first API call,")
    print("      causing delays for users. ComponentFactory initializes everything at startup.")

if __name__ == '__main__':
    startup_time = test_startup_time()

    if startup_time is not None:
        test_lazy_loading_comparison()

        print("\n=== TEST RESULTS ===")
        print(f"Actual startup time: {startup_time:.2f} seconds")
        print("✅ Startup time optimization successful!" if startup_time < 30 else "⚠️  Startup time still above target")

        # Save results
        with open('startup_test_results.txt', 'w') as f:
            f.write(f"Actual startup time: {startup_time:.2f} seconds\n")
            f.write("Target: <30 seconds\n")
            f.write(f"Status: {'PASS' if startup_time < 30 else 'REVIEW'}\n")
    else:
        print("❌ Startup test failed")
        sys.exit(1)
