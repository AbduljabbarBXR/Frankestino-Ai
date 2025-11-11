#!/usr/bin/env python3
"""
Test script for semantic chunking functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.text_processor import TextProcessor

def test_semantic_chunking():
    """Test the semantic chunking functionality"""

    # Initialize text processor
    processor = TextProcessor()

    # Test text with paragraphs and sentences
    test_text = """
    Artificial Intelligence is transforming our world. Machine learning algorithms can now recognize patterns in data with unprecedented accuracy.

    Neural networks are the backbone of modern AI systems. These computational models are inspired by the human brain's structure and function.

    Deep learning has revolutionized computer vision. Convolutional neural networks can identify objects in images with remarkable precision.

    Natural language processing enables computers to understand human language. Transformer models like BERT and GPT have achieved state-of-the-art results in various NLP tasks.

    The future of AI holds great promise. Continued research and development will unlock new capabilities and applications across all domains.
    """

    print("=== Testing Semantic Chunking ===")
    print(f"Input text length: {len(test_text)} characters")
    print(f"Input text paragraphs: {len([p for p in test_text.split('\n\n') if p.strip()])}")

    # Test semantic chunking
    chunks = processor.semantic_chunk(test_text, max_chunk_size=800, overlap=50)

    print(f"\nGenerated {len(chunks)} chunks:")
    print("-" * 50)

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Type: {chunk['metadata']['chunk_type']}")
        print(f"  Boundary: {chunk['metadata']['semantic_boundary']}")
        print(f"  Characters: {chunk['metadata']['char_count']}")
        print(f"  Sentences: {len(chunk.get('sentences', []))}")
        print(f"  Paragraphs: {len(chunk.get('paragraphs', []))}")
        print(f"  Text preview: {chunk['text'][:150]}...")
        print("-" * 30)

    # Test with different chunk sizes
    print("\n=== Testing Different Chunk Sizes ===")

    for size in [512, 1024, 1536]:
        chunks = processor.semantic_chunk(test_text, max_chunk_size=size)
        print(f"Chunk size {size}: {len(chunks)} chunks, avg {sum(c['metadata']['char_count'] for c in chunks)/len(chunks):.0f} chars")

    print("\n=== Semantic Chunking Test Complete ===")

if __name__ == "__main__":
    test_semantic_chunking()
