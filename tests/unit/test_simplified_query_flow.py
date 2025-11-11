"""
Unit tests for simplified query architecture - ensuring direct response flow
"""
import unittest
from unittest.mock import Mock, patch
import time
from backend.llm.llm_core import LLMCore
from backend.llm.inference import MemoryAugmentedInference
from backend.memory.memory_manager import MemoryManager


class TestSimplifiedQueryFlow(unittest.TestCase):
    """Test that simplified query flow works without complex enhancements"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory_manager = Mock(spec=MemoryManager)
        # Add neural_mesh attribute that real MemoryManager has
        self.memory_manager.neural_mesh = Mock()
        # Don't create LLMCore here - create it in individual tests where patches are applied

    @patch('backend.llm.llm_core.GGUFModelLoader')
    @patch('backend.llm.llm_core.MemoryAugmentedInference')
    def test_direct_query_response_flow(self, mock_inference_class, mock_model_loader_class):
        """Test that query goes directly to response generation"""
        # Setup mocks
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader_class.return_value = mock_model_loader

        mock_inference = Mock()
        mock_inference.generate_response.return_value = {
            'query': 'test query',
            'answer': 'Direct response',
            'memory_chunks_used': 2,
            'conversation_messages_used': 1,
            'learning_enabled': True
        }
        mock_inference_class.return_value = mock_inference

        # Mock memory search with proper list-like behavior
        memory_results = {
            'results': [{'text': 'memory chunk', 'score': 0.8}],
            'conversation_context': [{'role': 'user', 'content': 'previous message'}]
        }
        self.memory_manager.hybrid_search_cached.return_value = memory_results

        # Create LLM core
        llm_core = LLMCore(self.memory_manager)

        # Make query
        response = llm_core.query("test query")

        # Verify direct flow: memory search → inference → response
        self.memory_manager.hybrid_search_cached.assert_called_once()
        mock_inference.generate_response.assert_called_once()

        # Verify response structure
        self.assertEqual(response['answer'], 'Direct response')
        self.assertEqual(response['memory_chunks_used'], 2)
        self.assertEqual(response['conversation_messages_used'], 1)
        self.assertTrue(response['learning_enabled'])

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_no_complex_enhancement_logic(self, mock_model_loader_class):
        """Test that complex enhancement logic has been removed"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Simple response"
        mock_model_loader_class.return_value = mock_model_loader

        # Mock memory search
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [],
            'conversation_context': []
        }

        llm_core = LLMCore(self.memory_manager)

        # Verify no complex enhancement methods exist
        inference = llm_core.inference_engine

        # Should not have complex enhancement methods
        self.assertFalse(hasattr(inference, 'smart_memory_integration'),
                        "Complex enhancement logic should be removed")
        self.assertFalse(hasattr(inference, '_enhance_with_memory'),
                        "Memory enhancement methods should be removed")
        self.assertFalse(hasattr(inference, '_should_enhance_with_memory'),
                        "Enhancement decision logic should be removed")

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_memory_search_happens_but_doesnt_block(self, mock_model_loader_class):
        """Test that memory search happens but doesn't create complex flow"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Fast response"
        mock_model_loader_class.return_value = mock_model_loader

        # Mock memory search with some results
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [
                {'text': 'Relevant memory', 'score': 0.9},
                {'text': 'Another memory', 'score': 0.7}
            ],
            'conversation_context': [
                {'role': 'user', 'content': 'Previous question'},
                {'role': 'assistant', 'content': 'Previous answer'}
            ]
        }

        llm_core = LLMCore(self.memory_manager)

        start_time = time.time()
        response = llm_core.query("How does this work?")
        end_time = time.time()

        # Should be fast (no complex processing)
        query_time = end_time - start_time
        self.assertLess(query_time, 1.0, "Query should be fast without complex processing")

        # Memory search should have been called
        self.memory_manager.hybrid_search_cached.assert_called_once()

        # Response should include memory usage info
        self.assertIn('memory_chunks_used', response)
        self.assertIn('conversation_messages_used', response)

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_response_generation_is_direct(self, mock_model_loader_class):
        """Test that response generation doesn't have complex logic"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Direct generated response"
        mock_model_loader_class.return_value = mock_model_loader

        # Mock empty memory search
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [],
            'conversation_context': []
        }

        llm_core = LLMCore(self.memory_manager)

        # Get the inference engine
        inference = llm_core.inference_engine

        # Verify it uses simplified generate_response method
        self.assertTrue(hasattr(inference, 'generate_response'))
        self.assertTrue(hasattr(inference, 'build_simple_prompt'))

        # Should not have complex methods
        self.assertFalse(hasattr(inference, 'build_bootstrap_prompt'))
        self.assertFalse(hasattr(inference, 'generate_bootstrap_response'))

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_error_handling_graceful(self, mock_model_loader_class):
        """Test that errors are handled gracefully without complex fallbacks"""
        # Setup mock that fails
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.side_effect = Exception("Model error")
        mock_model_loader_class.return_value = mock_model_loader

        # Mock memory search
        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [],
            'conversation_context': []
        }

        llm_core = LLMCore(self.memory_manager)

        # Make query that will fail
        response = llm_core.query("test query")

        # Should handle error gracefully
        self.assertIn('error', response)
        self.assertIn('I apologize', response['answer'])
        self.assertEqual(response['memory_chunks_used'], 0)
        self.assertTrue(response['learning_enabled'])

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_memory_chunks_passed_to_inference(self, mock_model_loader_class):
        """Test that memory chunks are passed to inference but don't cause complexity"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Response with memory context"
        mock_model_loader_class.return_value = mock_model_loader

        # Mock memory search with results
        memory_chunks = [
            {'text': 'Important fact 1', 'score': 0.9},
            {'text': 'Important fact 2', 'score': 0.8}
        ]
        conversation_context = [
            {'role': 'user', 'content': 'What do you know?'},
            {'role': 'assistant', 'content': 'I know many things!'}
        ]

        self.memory_manager.hybrid_search_cached.return_value = {
            'results': memory_chunks,
            'conversation_context': conversation_context
        }

        llm_core = LLMCore(self.memory_manager)

        response = llm_core.query("Tell me something important")

        # Verify memory was searched and passed to inference
        self.memory_manager.hybrid_search_cached.assert_called_once()

        # Response should reflect memory usage
        self.assertEqual(response['memory_chunks_used'], 2)
        self.assertEqual(response['conversation_messages_used'], 2)

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_no_symmetric_query_method(self, mock_model_loader_class):
        """Test that the complex symmetric query method has been removed"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader_class.return_value = mock_model_loader

        llm_core = LLMCore(self.memory_manager)

        # Should not have query_symmetric method
        self.assertFalse(hasattr(llm_core, 'query_symmetric'),
                        "Complex symmetric query method should be removed")

        # Should not have complex enhancement methods
        self.assertFalse(hasattr(llm_core, '_generate_natural_response'))
        self.assertFalse(hasattr(llm_core, '_should_enhance_with_memory'))
        self.assertFalse(hasattr(llm_core, '_is_memory_valuable'))
        self.assertFalse(hasattr(llm_core, '_enhance_with_memory'))

    @patch('backend.llm.llm_core.GGUFModelLoader')
    def test_query_method_signature_simplified(self, mock_model_loader_class):
        """Test that query method has simplified signature"""
        # Setup mock
        mock_model_loader = Mock()
        mock_model_loader.load_model.return_value = True
        mock_model_loader.is_loaded = True
        mock_model_loader.generate_text.return_value = "Simple response"
        mock_model_loader_class.return_value = mock_model_loader

        self.memory_manager.hybrid_search_cached.return_value = {
            'results': [],
            'conversation_context': []
        }

        llm_core = LLMCore(self.memory_manager)

        # Query method should exist and be callable
        self.assertTrue(hasattr(llm_core, 'query'))
        self.assertTrue(callable(llm_core.query))

        # Test basic call works
        response = llm_core.query("test")
        self.assertIn('answer', response)
        self.assertIn('memory_chunks_used', response)


if __name__ == '__main__':
    unittest.main()
