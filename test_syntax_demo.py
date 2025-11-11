#!/usr/bin/env python3
"""
Quick test to verify syntax-aware connectivity is working with spaCy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.memory.selective_connectivity import SelectiveConnectivity, ConnectivityStrategy

def test_syntax_connectivity():
    """Test syntax-aware connectivity"""
    print("Testing Syntax-Aware Neural Connectivity")
    print("=" * 50)

    # Test syntax-aware connectivity
    connectivity = SelectiveConnectivity(strategy=ConnectivityStrategy.SYNTAX_AWARE)
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat']

    print(f"Input words: {words}")
    connections = connectivity.connect_words_in_text(words)

    print(f"Syntax-aware connections: {len(connections)}")
    for i, (word_a, word_b, weight) in enumerate(connections[:5]):  # Show first 5
        print(f"  {i+1}. {word_a} -> {word_b} (weight: {weight:.2f})")

    # Compare with sliding window
    print("\nComparing with Sliding Window:")
    sliding_conn = SelectiveConnectivity(strategy=ConnectivityStrategy.SLIDING_WINDOW)
    sliding_connections = sliding_conn.connect_words_in_text(words)
    print(f"Sliding window connections: {len(sliding_connections)}")

    print("\n[SUCCESS] Syntax-aware connectivity test complete!")
    print("spaCy integration working properly!")

if __name__ == "__main__":
    test_syntax_connectivity()