"""
Unit tests for lazy loading removal - ensuring model persistence
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from backend.llm.llm_core import LLMCore
from backend.memory.memory_manager import MemoryManager


class TestLazyLoadingRemoval(unittest.TestCase):
    """Test that lazy loading has been properly removed"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory_manager = Mock(spec=MemoryManager)
        # Add neural_mesh attribute that real MemoryManager has
        self.memory_manager.neural_mesh = Mock()
        # Don't create LLMCore here - create it in individual tests where patches are applied

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_model_loads_at_initialization(self, mock_model_loader_class):
        """Test that model loads immediately at LLMCore initialization"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader_class.return_value = mock_model_loader

        # Create new instance - should load model immediately
        llm_core = LLMCore(self.memory_manager)

        # Verify model loader was created and load_model was called
        mock_model_loader_class.assert_called_once()
        mock_model_loader.load_model.assert_called_once()

        # Verify inference engine was created
        self.assertIsNotNone(llm_core.inference_engine)

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_model_stays_loaded_across_queries(self, mock_model_loader_class):
        """Test that model stays loaded and doesn't reload on each query"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Test response"
        mock_model_loader_class.return_value = mock_model_loader

        # Mock memory search
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [],
            'conversation_context': []
        }

        # Create instance
        llm_core = LLMCore(self.memory_manager)

        # Make multiple queries
        for i in range(3):
            response = llm_core.query(f"Test query {i}")

            # Verify response structure
            self.assertIn('answer', response)
            self.assertIn('memory_chunks_used', response)
            self.assertIn('learning_enabled', response)

        # Verify model was only loaded once
        mock_model_loader.load_model.assert_called_once()

        # Verify generate_text was called for each query
        self.assertEqual(mock_model_loader.generate_text.call_count, 3)

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_no_lazy_loading_message_in_logs(self, mock_model_loader_class):
        """Test that initialization doesn't mention lazy loading"""
        with patch('backend.llm.llm_core.logger') as mock_logger:
            # Setup mock
            mock_model_loader = Mock()
            mock_model_loader.load_model.return_value = True
            mock_model_loader.is_loaded = True
            mock_model_loader_class.return_value = mock_model_loader

            # Create instance
            LLMCore(self.memory_manager)

            # Check that no lazy loading message was logged
            log_calls = [call for call in mock_logger.info.call_args_list
                        if 'lazy loading' in str(call).lower()]
            self.assertEqual(len(log_calls), 0, "Should not log lazy loading messages")

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_persistent_model_status(self, mock_model_loader_class):
        """Test that model status shows persistent loading"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.get_model_info.return_value = {
            'status': 'loaded',
            'model_path': 'test.gguf',
            'context_length': 32768
        }
        mock_model_loader_class.return_value = mock_model_loader

        llm_core = LLMCore(self.memory_manager)

        # Get model status
        status = llm_core.get_model_status()

        # Verify status indicates loaded state
        self.assertEqual(status['status'], 'loaded')
        self.assertTrue(status['inference_engine_ready'])
        self.assertIsNotNone(status['model_path'])

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_no_unload_method_exists(self, mock_model_loader_class):
        """Test that the unload method has been removed"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader_class.return_value = mock_model_loader

        llm_core = LLMCore(self.memory_manager)

        # Verify _unload_model_if_idle method doesn't exist
        self.assertFalse(hasattr(llm_core, '_unload_model_if_idle'),
                        "Lazy unloading method should be removed")

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_query_performance_no_reload_delay(self, mock_model_loader_class):
        """Test that queries don't have reload delays"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Fast response"
        mock_model_loader_class.return_value = mock_model_loader

        # Mock memory search
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [],
            'conversation_context': []
        }

        llm_core = LLMCore(self.memory_manager)

        start_time = time.time()

        # Make query
        response = llm_core.query("Test query")

        end_time = time.time()
        query_time = end_time - start_time

        # Should be fast (no 21s reload delay)
        self.assertLess(query_time, 1.0, "Query should be fast without reload delay")

        # Verify response
        self.assertEqual(response['answer'], "Fast response")


if __name__ == '__main__':
    unittest.main()
