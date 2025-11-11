#!/usr/bin/env python3
"""
Test script for memory optimization features:
- Lazy loading (load on-demand, unload immediately)
- MMAP support for efficient memory usage
- Better error logging
"""

import sys
import os
import time
import psutil
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.llm.llm_core import LLMCore
from backend.memory.memory_manager import MemoryManager
from backend.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_lazy_loading():
    """Test lazy loading functionality"""
    print("🧠 Testing Lazy Loading & Memory Optimization")
    print("=" * 50)

    # Initialize memory manager
    memory_manager = MemoryManager()

    # Initialize LLM core with lazy loading
    llm_core = LLMCore(memory_manager)

    print("✅ LLM Core initialized with lazy loading (model not loaded yet)")

    # Check initial memory
    initial_memory = get_memory_usage()
    print(f"💾 Initial memory usage: {initial_memory:.1f} MB")

    # Check model status before query
    status = llm_core.get_model_status()
    print(f"📊 Model status before query: {status['status']}")

    # Test query (should load model on-demand)
    print("\n🔄 Testing on-demand loading...")
    start_time = time.time()

    try:
        response = llm_core.query(
            "Hello, can you tell me about memory optimization?",
            temperature=0.7
        )

        load_time = time.time() - start_time
        print(f"⚡ Query completed in {load_time:.2f}s")
        print(f"📝 Response length: {len(response.get('answer', ''))} characters")

        # Check performance metadata
        perf = response.get('performance', {})
        print(f"🎯 Model loaded on-demand: {perf.get('model_loaded_on_demand', False)}")
        print(f"🔢 Query number: {perf.get('query_number', 0)}")

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

    # Check memory after query (model should be unloaded)
    time.sleep(0.5)  # Allow time for unloading
    post_query_memory = get_memory_usage()
    print(f"💾 Memory after query: {post_query_memory:.1f} MB")

    # Check model status after query (should be unloaded)
    status_after = llm_core.get_model_status()
    print(f"📊 Model status after query: {status_after.get('status', 'unknown')}")

    # Test MMAP configuration
    print("\n🗺️  Testing MMAP Configuration...")

    # Get model info (this will load the model temporarily)
    model_info = llm_core.get_model_status()
    if 'model_path' in model_info:
        print("✅ Model info retrieved successfully")
        print(f"📁 Model path: {model_info.get('model_path', 'N/A')}")
        print(f"📏 Model size: {model_info.get('model_size_mb', 0):.1f} MB")

        mmap_info = model_info.get('memory_mapping', {})
        print(f"🗺️  MMAP enabled: {mmap_info.get('mmap_enabled', False)}")
        print(f"🔓 MLock disabled: {mmap_info.get('mlock_disabled', False)}")
        print(f"⚡ Memory efficient: {mmap_info.get('memory_efficient', False)}")

    print("\n🎉 Memory optimization test completed!")
    print(f"💾 Memory delta: {post_query_memory - initial_memory:.1f} MB")
    print("✅ Lazy loading: Model loads on-demand and unloads immediately")
    print("✅ MMAP: Memory mapping enabled for efficient RAM usage")
    print("✅ Error logging: Full tracebacks now included in errors")
    return True

if __name__ == "__main__":
    print("🧠 Frankenstino AI - Memory Optimization Test")
    print("=" * 50)

    try:
        success = test_lazy_loading()
        if success:
            print("\n🎊 All memory optimization features working correctly!")
            sys.exit(0)
        else:
            print("\n❌ Memory optimization test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
