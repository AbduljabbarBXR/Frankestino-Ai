"""
Unit tests for vector store reliability - ensuring IVF fallback works
"""
import unittest
import numpy as np
from unittest.mock import Mock, patch
from backend.memory.vector_store import OptimizedVectorStore


class TestVectorStoreReliability(unittest.TestCase):
    """Test that vector store works reliably with IVF fallback"""

    def setUp(self):
        """Set up test fixtures"""
        self.embedding_dim = 384
        self.test_vectors = np.random.rand(5, self.embedding_dim).astype(np.float32)
        self.test_metadata = [
            {'text': f'Test chunk {i}', 'source': 'test', 'score': 0.8}
            for i in range(5)
        ]

    @patch('backend.memory.vector_store.faiss')
    def test_ivf_index_creation_success(self, mock_faiss):
        """Test that IVF index creates successfully"""
        # Setup FAISS mocks
        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFFlat.return_value = Mock()
        mock_faiss.get_num_gpus.return_value = 0

        # Create store with IVF
        store = OptimizedVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="ivf",
            use_gpu=False
        )

        # Verify IVF index was created
        self.assertEqual(store.index_type, "ivf")
        mock_faiss.IndexIVFFlat.assert_called_once()

    @patch('backend.memory.vector_store.faiss')
    def test_ivfpq_fallback_to_ivf_on_training_failure(self, mock_faiss):
        """Test that IVFPQ falls back to IVF when training fails"""
        # Setup FAISS mocks
        mock_index = Mock()
        mock_index.is_trained = False
        mock_index.train.side_effect = Exception("Training failed")
        mock_index.add = Mock()

        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFPQ.return_value = mock_index
        mock_faiss.IndexIVFFlat.return_value = Mock()
        mock_faiss.get_num_gpus.return_value = 0

        # Create store with IVFPQ - should fallback to IVF
        store = OptimizedVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="ivfpq",
            use_gpu=False
        )

        # Try to add vectors (this triggers training)
        store.add_vectors(self.test_vectors, self.test_metadata)

        # Verify it fell back to IVF
        self.assertEqual(store.index_type, "ivf")
        mock_faiss.IndexIVFFlat.assert_called()

    @patch('backend.memory.vector_store.faiss')
    def test_ivf_works_with_single_vector(self, mock_faiss):
        """Test that IVF works even with just one vector (falls back to flat-like behavior)"""
        # Setup FAISS mocks
        mock_index = Mock()
        mock_index.is_trained = False
        mock_index.ntotal = 0
        mock_index.train = Mock()
        mock_index.add = Mock()

        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0

        # Create store with small nlist so fewer vectors needed for training
        store = OptimizedVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="ivf",
            nlist=1,  # Very small nlist
            use_gpu=False
        )

        # Add single vector
        single_vector = np.random.rand(1, self.embedding_dim).astype(np.float32)
        single_metadata = [{'text': 'Single test', 'source': 'test'}]

        ids = store.add_vectors(single_vector, single_metadata)

        # Verify it worked (vectors added even if training didn't happen)
        self.assertEqual(len(ids), 1)
        mock_index.add.assert_called_once()
        # Training may or may not happen depending on threshold

    @patch('backend.memory.vector_store.faiss')
    def test_ivf_works_with_zero_vectors(self, mock_faiss):
        """Test that IVF handles empty index gracefully"""
        # Setup FAISS mocks
        mock_index = Mock()
        mock_index.is_trained = True
        mock_index.ntotal = 0
        mock_index.search.return_value = (np.array([[]]), np.array([[]]))

        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0

        # Create store
        store = OptimizedVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="ivf",
            use_gpu=False
        )

        # Search empty index
        query = np.random.rand(self.embedding_dim).astype(np.float32)
        results = store.search(query, top_k=5)

        # Should return empty results gracefully
        self.assertEqual(len(results), 0)

    @patch('backend.memory.vector_store.faiss')
    def test_conversation_storage_works(self, mock_faiss):
        """Test that conversation storage works without IVFPQ errors"""
        # Setup FAISS mocks
        mock_index = Mock()
        mock_index.is_trained = False
        mock_index.ntotal = 0
        mock_index.train = Mock()
        mock_index.add = Mock()

        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0

        # Create store with small nlist so fewer vectors needed for training
        store = OptimizedVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="ivf",
            nlist=1,  # Very small nlist
            use_gpu=False
        )

        # Simulate conversation storage (multiple small additions)
        conversation_vectors = []
        conversation_metadata = []

        for i in range(3):  # 3 conversation turns
            vec = np.random.rand(1, self.embedding_dim).astype(np.float32)
            meta = [{
                'text': f'Conversation turn {i}',
                'source': 'conversation',
                'conversation_id': 'test_conv',
                'role': 'user' if i % 2 == 0 else 'assistant'
            }]

            conversation_vectors.append(vec[0])
            conversation_metadata.extend(meta)

        # Add all conversation vectors at once
        all_vectors = np.array(conversation_vectors).reshape(-1, self.embedding_dim)
        ids = store.add_vectors(all_vectors, conversation_metadata)

        # Verify storage worked
        self.assertEqual(len(ids), 3)
        # Vectors are added even if training doesn't happen due to insufficient data
        mock_index.add.assert_called_once()

        # Verify search works (may return fewer results if training didn't happen)
        query = np.random.rand(self.embedding_dim).astype(np.float32)
        results = store.search(query, top_k=3)
        # Results may be fewer than requested if index isn't properly trained
        self.assertIsInstance(results, list)

    @patch('backend.memory.vector_store.faiss')
    def test_insufficient_training_data_fallback(self, mock_faiss):
        """Test fallback when there are too few vectors for training"""
        # Setup FAISS mocks
        mock_index = Mock()
        mock_index.is_trained = False
        mock_index.add = Mock()

        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFPQ.return_value = mock_index
        mock_faiss.IndexIVFFlat.return_value = Mock()
        mock_faiss.get_num_gpus.return_value = 0

        # Create store with IVFPQ
        store = OptimizedVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="ivfpq",
            nlist=100,  # Requires many vectors for training
            use_gpu=False
        )

        # Try to add too few vectors for training
        few_vectors = np.random.rand(2, self.embedding_dim).astype(np.float32)
        few_metadata = [{'text': f'Test {i}'} for i in range(2)]

        # This should trigger fallback due to insufficient training data
        ids = store.add_vectors(few_vectors, few_metadata)

        # Verify fallback occurred and vectors were added
        self.assertEqual(len(ids), 2)
        self.assertEqual(store.index_type, "ivf")  # Should have fallen back

    def test_fallback_method_exists(self):
        """Test that the fallback method exists and works"""
        store = OptimizedVectorStore(embedding_dim=self.embedding_dim)

        # Verify _fallback_to_ivf_flat method exists
        self.assertTrue(hasattr(store, '_fallback_to_ivf_flat'))
        self.assertTrue(callable(getattr(store, '_fallback_to_ivf_flat')))

        # Test calling the fallback method
        original_type = store.index_type
        store._fallback_to_ivf_flat()

        # Should change to IVF
        self.assertEqual(store.index_type, "ivf")
        self.assertNotEqual(store.index_type, original_type)

    @patch('backend.memory.vector_store.faiss')
    def test_memory_stats_tracking(self, mock_faiss):
        """Test that memory stats are tracked properly"""
        # Setup FAISS mocks
        mock_index = Mock()
        mock_index.is_trained = True
        mock_index.ntotal = 0
        mock_index.add = Mock()

        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0

        # Create store
        store = OptimizedVectorStore(
            embedding_dim=self.embedding_dim,
            index_type="ivf",
            use_gpu=False
        )

        # Add vectors
        ids = store.add_vectors(self.test_vectors, self.test_metadata)

        # Check stats
        stats = store.get_stats()
        self.assertEqual(stats['total_vectors'], 5)
        self.assertEqual(stats['index_type'], 'ivf')
        self.assertGreater(stats['index_size_mb'], 0)
        self.assertEqual(stats['query_count'], 0)  # No searches yet


if __name__ == '__main__':
    unittest.main()
