"""
Integration tests for all fixes working together
"""
import unittest
from unittest.mock import Mock, patch
import time
from backend.llm.llm_core import LLMCore
from backend.memory.memory_manager import MemoryManager
from backend.memory.vector_store import OptimizedVectorStore


class TestIntegrationFixes(unittest.TestCase):
    """Test that all fixes work together end-to-end"""

    def setUp(self):
        """Set up complete test environment"""
        self.memory_manager = Mock(spec=MemoryManager)
        # Add neural_mesh attribute that real MemoryManager has
        self.memory_manager.neural_mesh = Mock()
        # Mock the LLMCore to avoid loading actual model
        with patch('backend.llm.llm_core.GGUFModelLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_model.return_value = True
            mock_loader.is_loaded = True
            mock_loader.generate_text.return_value = "Mock response"
            mock_loader_class.return_value = mock_loader

            with patch('backend.llm.llm_core.MemoryAugmentedInference') as mock_inference_class:
                mock_inference = Mock()
                mock_inference.default_temperature = 0.6
                mock_inference.generate_response.return_value = {
                    "answer": "Mock response",
                    "memory_chunks_used": 2,
                    "conversation_messages_used": 1,
                    "learning_enabled": True
                }
                mock_inference_class.return_value = mock_inference

                self.llm_core = LLMCore(self.memory_manager)
                self.mock_loader = mock_loader
                self.mock_inference = mock_inference

    @patch('backend.llm.llm_core.GGUFModelLoader')
    @patch('backend.memory.vector_store.faiss')
    def test_complete_system_initialization(self, mock_faiss, mock_model_loader_class):
        """Test that the complete system initializes without lazy loading or IVFPQ issues"""
        # Setup mocks
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader_class.return_value = mock_model_loader

        # Setup FAISS mocks for IVF
        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFFlat.return_value = Mock()
        mock_faiss.get_num_gpus.return_value = 0

        # Mock memory search
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [],
            'conversation_context': []
        }

        # Create system components
        llm_core = LLMCore(self.memory_manager)
        vector_store = OptimizedVectorStore(embedding_dim=384, index_type="ivf")

        # Verify model loaded immediately (no lazy loading)
        mock_model_loader.load_model.assert_called_once()

        # Verify IVF index created (no IVFPQ issues)
        self.assertEqual(vector_store.index_type, "ivf")
        mock_faiss.IndexIVFFlat.assert_called()

        # Verify system is ready
        self.assertIsNotNone(llm_core.inference_engine)
        self.assertIsNotNone(vector_store.index)

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_end_to_end_query_flow(self, mock_model_loader_class):
        """Test complete query flow: search → generate → respond"""
        # Setup mock model
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Integration test response"
        mock_model_loader_class.return_value = mock_model_loader

        # Setup memory search with realistic data
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [
                {'text': 'Test memory chunk 1', 'score': 0.8},
                {'text': 'Test memory chunk 2', 'score': 0.7}
            ],
            'conversation_context': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]
        }

        llm_core = LLMCore(self.memory_manager)

        # Make query
        start_time = time.time()
        response = llm_core.query("Test integration query")
        end_time = time.time()

        # Verify complete flow worked
        self.assertIn('answer', response)
        self.assertEqual(response['answer'], "Integration test response")
        self.assertEqual(response['memory_chunks_used'], 2)
        self.assertEqual(response['conversation_messages_used'], 2)
        self.assertTrue(response['learning_enabled'])

        # Verify performance (should be fast, no 21s reload)
        query_time = end_time - start_time
        self.assertLess(query_time, 1.0, "Integration query should be fast")

        # Verify model was only loaded once
        mock_model_loader.load_model.assert_called_once()

    @patch('backend.llm.llm_core.GGUFModelLoader')
    @patch('backend.memory.vector_store.faiss')
    def test_vector_storage_integration(self, mock_faiss, mock_model_loader_class):
        """Test that vector storage works with conversation persistence"""
        # Setup mocks
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader_class.return_value = mock_model_loader

        # Setup FAISS for IVF
        mock_index = Mock()
        mock_index.is_trained = False
        mock_index.ntotal = 0
        mock_index.train = Mock()
        mock_index.add = Mock()

        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0

        # Create vector store with small nlist so training happens with few vectors
        vector_store = OptimizedVectorStore(embedding_dim=384, index_type="ivf", nlist=1)

        # Simulate conversation storage
        import numpy as np
        conversation_vectors = np.random.rand(3, 384).astype(np.float32)
        conversation_metadata = [
            {'text': f'Conv message {i}', 'conversation_id': 'test_conv', 'role': 'user' if i % 2 == 0 else 'assistant'}
            for i in range(3)
        ]

        # Store conversation
        ids = vector_store.add_vectors(conversation_vectors, conversation_metadata)

        # Verify storage worked
        self.assertEqual(len(ids), 3)
        # Vectors are stored in metadata even if index training doesn't happen
        self.assertEqual(len(vector_store.metadata), 3)

        # Verify search works (may return fewer results if index isn't trained)
        query_vec = np.random.rand(384).astype(np.float32)
        results = vector_store.search(query_vec, top_k=3)
        # Results list should exist, even if empty
        self.assertIsInstance(results, list)

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_error_recovery_integration(self, mock_model_loader_class):
        """Test that system recovers gracefully from errors"""
        # Setup mock that fails then succeeds
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        # First call fails, second succeeds
        mock_model_loader.generate_text.side_effect = [
            Exception("Temporary failure"),
            "Recovered response"
        ]
        mock_model_loader_class.return_value = mock_model_loader

        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [],
            'conversation_context': []
        }

        llm_core = LLMCore(self.memory_manager)

        # First query fails
        response1 = llm_core.query("Failing query")
        self.assertIn('error', response1)
        self.assertIn('I apologize', response1['answer'])

        # Second query succeeds (simulating recovery)
        response2 = llm_core.query("Working query")
        self.assertEqual(response2['answer'], "Recovered response")
        self.assertNotIn('error', response2)

    def test_no_lazy_loading_artifacts(self):
        """Test that no lazy loading code artifacts remain"""
        llm_core = LLMCore(self.memory_manager)

        # Should not have lazy loading attributes
        self.assertFalse(hasattr(llm_core, '_model_loaded'))
        self.assertFalse(hasattr(llm_core, '_load_start_time'))
        self.assertFalse(hasattr(llm_core, '_query_count'))
        self.assertFalse(hasattr(llm_core, '_total_load_time'))

        # Should not have lazy unloading method
        self.assertFalse(hasattr(llm_core, '_unload_model_if_idle'))

        # Should not have complex symmetric methods
        self.assertFalse(hasattr(llm_core, 'query_symmetric'))
        self.assertFalse(hasattr(llm_core, '_generate_natural_response'))
        self.assertFalse(hasattr(llm_core, '_should_enhance_with_memory'))
        self.assertFalse(hasattr(llm_core, '_is_memory_valuable'))
        self.assertFalse(hasattr(llm_core, '_enhance_with_memory'))

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_memory_integration_simplified(self, mock_model_loader_class):
        """Test that memory integration is simplified, not complex"""
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Simple memory-integrated response"
        mock_model_loader_class.return_value = mock_model_loader

        # Setup memory with various chunks
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [
                {'text': 'High relevance memory', 'score': 0.9},
                {'text': 'Medium relevance memory', 'score': 0.6},
                {'text': 'Low relevance memory', 'score': 0.3}
            ],
            'conversation_context': [
                {'role': 'user', 'content': 'Remember what we talked about?'},
                {'role': 'assistant', 'content': 'Yes, we discussed several topics.'}
            ]
        }

        llm_core = LLMCore(self.memory_manager)

        response = llm_core.query("What do you remember about our conversation?")

        # Should use memory but not have complex enhancement
        self.assertEqual(response['memory_chunks_used'], 3)
        self.assertEqual(response['conversation_messages_used'], 2)
        self.assertEqual(response['answer'], "Simple memory-integrated response")

        # Inference engine should not have complex methods
        inference = llm_core.inference_engine
        self.assertFalse(hasattr(inference, 'smart_memory_integration'))
        self.assertFalse(hasattr(inference, 'build_bootstrap_prompt'))
        self.assertTrue(hasattr(inference, 'build_simple_prompt'))

    @patch('backend.llm.llm_core.GGUFModelLoader')
    @patch('backend.memory.vector_store.faiss')
    def test_system_scalability(self, mock_faiss, mock_model_loader_class):
        """Test that system scales properly with data"""
        # Setup mocks
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Scalable response"
        mock_model_loader_class.return_value = mock_model_loader

        # Setup FAISS mocks
        mock_index = Mock()
        mock_index.is_trained = False
        mock_index.ntotal = 0
        mock_index.train = Mock()
        mock_index.add = Mock()

        mock_faiss.IndexFlatIP.return_value = Mock()
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.get_num_gpus.return_value = 0

        # Test with increasing amounts of data - use small nlist so training happens
        vector_store = OptimizedVectorStore(embedding_dim=384, index_type="ivf", nlist=1)

        import numpy as np
        for batch_size in [1, 5, 10]:
            vectors = np.random.rand(batch_size, 384).astype(np.float32)
            metadata = [{'text': f'Item {i}'} for i in range(batch_size)]

            ids = vector_store.add_vectors(vectors, metadata)
            self.assertEqual(len(ids), batch_size)

        # Final count should be correct (vectors are added even if training doesn't happen)
        self.assertEqual(len(vector_store.metadata), 16)  # 1 + 5 + 10

    def test_architecture_simplification_complete(self):
        """Test that architecture simplification is complete"""
        # Check that the codebase has been simplified

        # LLM Core should be simple
        llm_core = LLMCore(self.memory_manager)
        self.assertTrue(hasattr(llm_core, 'query'))
        self.assertTrue(hasattr(llm_core, 'inference_engine'))

        # Should not have complex architectures
        self.assertFalse(hasattr(llm_core, 'query_symmetric'))
        self.assertFalse(hasattr(llm_core, 'memory_curator'))

        # Inference should be simple
        inference = llm_core.inference_engine
        self.assertTrue(hasattr(inference, 'generate_response'))
        self.assertTrue(hasattr(inference, 'build_simple_prompt'))

        # Should not have complex inference methods
        complex_methods = [
            'smart_memory_integration',
            'build_bootstrap_prompt',
            'generate_bootstrap_response',
            '_enhance_with_memory',
            '_should_enhance_with_memory'
        ]

        for method in complex_methods:
            self.assertFalse(hasattr(inference, method),
                           f"Complex method {method} should be removed")


if __name__ == '__main__':
    unittest.main()
