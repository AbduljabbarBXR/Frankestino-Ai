#!/usr/bin/env python3
"""
Test script to verify Phase 1 setup
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test basic imports"""
    try:
        from backend.config import settings
        print("✓ Config import successful")
        print(f"  Model path: {settings.model_path}")
        print(f"  Embedding dim: {settings.embedding_dim}")

        import numpy as np
        print("✓ NumPy import successful")

        import faiss
        print("✓ FAISS import successful")

        from backend.memory.vector_store import OptimizedVectorStore
        print("✓ OptimizedVectorStore import successful")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_vector_store():
    """Test basic vector store functionality"""
    try:
        from backend.memory.vector_store import OptimizedVectorStore
        import numpy as np

        # Create test vectors
        dim = 384
        n_vectors = 10
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)

        # Create metadata dictionaries
        metadata = []
        for i in range(n_vectors):
            metadata.append({
                'text': f"Test document {i}",
                'source': f"source_{i}.txt",
                'type': 'test'
            })

        # Initialize store with flat index (no training required)
        store = OptimizedVectorStore(embedding_dim=dim, index_type="flat")

        # Add vectors
        ids = store.add_vectors(vectors, metadata)
        print(f"✓ Added {len(ids)} vectors to store")

        # Search
        query = np.random.randn(dim).astype(np.float32)
        results = store.search(query, top_k=3)
        print(f"✓ Search returned {len(results)} results")

        # Get stats
        stats = store.get_stats()
        print(f"✓ Store stats: {stats}")

        return True
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Frankenstino AI Phase 1 Setup")
    print("=" * 40)

    success = True
    success &= test_imports()
    print()
    success &= test_vector_store()

    print()
    if success:
        print("🎉 All tests passed! Phase 1 foundation is solid.")
        print("Ready to proceed to Phase 2: Core Memory System")
    else:
        print("❌ Some tests failed. Please check the setup.")
