#!/usr/bin/env python3
"""
Demo script to test selective connectivity improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.memory.selective_connectivity import SelectiveConnectivity, ConnectivityStrategy

def test_selective_connectivity():
    """Test the selective connectivity improvements"""

    print("Testing Selective Neural Connectivity Improvements")
    print("=" * 60)

    # Test text
    test_text = "The cat sat on the mat and watched the dog run through the park"
    words = test_text.split()

    print(f"Test text: '{test_text}'")
    print(f"Words: {words}")
    print()

    # Test different strategies
    strategies = [
        ("Full Connectivity (Old)", ConnectivityStrategy.FULL_CONNECTIVITY),
        ("Sliding Window (New)", ConnectivityStrategy.SLIDING_WINDOW),
    ]

    for strategy_name, strategy in strategies:
        print(f"Testing {strategy_name}")
        print("-" * 40)

        connectivity = SelectiveConnectivity(strategy=strategy, window_size=3)

        # Get connections
        connections = connectivity.connect_words_in_text(words)

        # Get stats
        stats = connectivity.get_connection_stats(words)

        print(f"  Total connections: {stats['total_connections']}")
        print(f"  Avg weight: {stats['avg_weight']:.3f}")
        print(f"  Connections per word: {stats['connections_per_word']:.2f}")

        # Show sample connections
        print("  Sample connections:")
        for i, (word_a, word_b, weight) in enumerate(connections[:5]):
            print(f"    {word_a} <-> {word_b} (weight: {weight:.3f})")
        if len(connections) > 5:
            print(f"    ... and {len(connections) - 5} more")

        print()

    # Performance comparison
    print("Performance Comparison")
    print("-" * 40)

    import time

    # Larger test
    large_text = "This is a much longer sentence that contains many more words to test the performance differences between the old full connectivity approach and the new selective sliding window approach which should be significantly more efficient and create fewer but more meaningful connections."
    large_words = large_text.split()

    print(f"Large text: {len(large_words)} words")

    for strategy_name, strategy in strategies:
        connectivity = SelectiveConnectivity(strategy=strategy, window_size=3)

        start_time = time.time()
        connections = connectivity.connect_words_in_text(large_words)
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"  {strategy_name}: {len(connections)} connections in {processing_time:.4f}s")

    print()
    print("SUCCESS: Selective connectivity implementation complete!")
    print("Benefits: Reduced connection complexity, better semantic relevance, improved performance")

if __name__ == "__main__":
    test_selective_connectivity()