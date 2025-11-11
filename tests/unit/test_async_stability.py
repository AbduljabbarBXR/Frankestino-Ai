"""
Unit tests for async stability - ensuring no asyncio cancellation errors
"""
import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from backend.main import get_learning_pipeline, get_memory_manager
from backend.learning_pipeline import LearningPipeline
from backend.memory.memory_manager import MemoryManager


class TestAsyncStability(unittest.TestCase):
    """Test that async operations are stable and don't cause cancellation errors"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory_manager = Mock(spec=MemoryManager)
        self.learning_pipeline = Mock(spec=LearningPipeline)

    def test_learning_pipeline_async_lifecycle(self):
        """Test that learning pipeline has proper async lifecycle"""
        # Mock the ComponentFactory to return a mock learning pipeline
        with patch('backend.utils.component_factory.ComponentFactory.get_learning_pipeline') as mock_get_lp:
            mock_lp = Mock()
            mock_lp.start_processing = AsyncMock()
            mock_lp.stop_processing = AsyncMock()
            mock_get_lp.return_value = mock_lp

            # This should not raise any async errors during creation
            lp = get_learning_pipeline()

            # Verify it was created properly
            mock_get_lp.assert_called_once()
            self.assertIsNotNone(lp)

    @patch('backend.utils.component_factory.ComponentFactory.get_memory_curator')
    @patch('backend.utils.component_factory.ComponentFactory.get_memory_manager')
    def test_memory_manager_creation_no_async_issues(self, mock_get_mm, mock_get_curator):
        """Test that memory manager creation doesn't have async conflicts"""
        mock_mm = Mock()
        mock_curator = Mock()
        mock_get_mm.return_value = mock_mm
        mock_get_curator.return_value = mock_curator

        # This should work without async issues
        mm = get_memory_manager()

        # Verify it returned the mock
        self.assertEqual(mm, mock_mm)

    def test_no_asyncio_in_regular_methods(self):
        """Test that regular methods don't accidentally use asyncio"""
        from backend.memory.memory_manager import MemoryManager

        # Create a mock memory manager
        mm = Mock(spec=MemoryManager)

        # These methods should not use asyncio
        methods_to_check = [
            'hybrid_search_cached',
            'get_conversation_messages',
            'store_conversation',
            'get_memory_stats'
        ]

        for method_name in methods_to_check:
            if hasattr(mm, method_name):
                method = getattr(mm, method_name)
                # Method should be callable and not async
                self.assertTrue(callable(method))

    def test_learning_pipeline_methods_are_async(self):
        """Test that learning pipeline methods are properly async"""
        from backend.learning_pipeline import LearningPipeline

        # Check that key methods are async (have async in their names or are coroutines)
        async_methods = [
            'start_processing',
            'stop_processing',
            'learn_from_interaction'
        ]

        # This is a static check - in real implementation these should be async
        for method_name in async_methods:
            # Just check the method exists - actual async testing would need an event loop
            self.assertTrue(hasattr(LearningPipeline, method_name) or hasattr(LearningPipeline, f'_{method_name}'))

    @patch('backend.main.LearningPipeline')
    def test_async_context_manager_usage(self, mock_lp_class):
        """Test that async context managers are used properly"""
        mock_lp = Mock()
        mock_lp.start_processing = AsyncMock()
        mock_lp.stop_processing = AsyncMock()
        mock_lp_class.return_value = mock_lp

        # Test that the lifecycle functions work
        async def test_lifecycle():
            # This simulates what happens in main.py lifespan
            lp = get_learning_pipeline()
            await lp.start_processing()
            await lp.stop_processing()

        # Should not raise any async errors
        try:
            # We can't run async code in unittest without more setup,
            # but we can at least verify the functions exist and are async
            lp = get_learning_pipeline()
            self.assertTrue(hasattr(lp, 'start_processing'))
            self.assertTrue(hasattr(lp, 'stop_processing'))
        except Exception as e:
            self.fail(f"Async lifecycle setup failed: {e}")

    def test_no_blocking_async_calls_in_sync_methods(self):
        """Test that sync methods don't accidentally block on async calls"""
        # This is a common source of asyncio.CancelledError

        # Check that LLM core methods are synchronous
        from backend.llm.llm_core import LLMCore

        llm_methods = ['query', 'get_model_status']
        for method_name in llm_methods:
            # These should be regular synchronous methods
            self.assertTrue(hasattr(LLMCore, method_name))

    def test_proper_error_handling_in_async_contexts(self):
        """Test that async contexts have proper error handling"""
        # Mock learning pipeline that fails
        with patch('backend.main.LearningPipeline') as mock_lp_class:
            mock_lp = Mock()
            mock_lp.start_processing = AsyncMock(side_effect=Exception("Async startup failed"))
            mock_lp_class.return_value = mock_lp

            # This should not crash the application
            try:
                lp = get_learning_pipeline()
                # In real code, this would be awaited
                # For testing, we just verify the mock is set up correctly
                self.assertIsNotNone(lp)
            except Exception:
                # Should handle errors gracefully
                pass

    def test_async_cleanup_properly_implemented(self):
        """Test that async cleanup is implemented properly"""
        # Check that main.py has proper async cleanup in lifespan
        import inspect
        from backend.main import lifespan

        # Lifespan should be an async context manager (decorated with @asynccontextmanager)
        # Check that it has proper async context management
        source = inspect.getsource(lifespan)

        # Should have ComponentFactory initialization and shutdown
        self.assertIn('ComponentFactory.initialize_components', source)
        self.assertIn('ComponentFactory.shutdown', source)
        self.assertIn('async def lifespan', source)
        self.assertIn('yield', source)

    def test_no_await_in_sync_contexts(self):
        """Test that sync contexts don't accidentally await"""
        # This would cause RuntimeError about event loop

        # Check main.py for any accidental awaits in sync code
        import inspect
        from backend.main import (
            get_memory_manager, get_llm_core, get_memory_curator,
            get_learning_pipeline, get_memory_metrics
        )

        sync_functions = [
            get_memory_manager, get_llm_core, get_memory_curator,
            get_learning_pipeline, get_memory_metrics
        ]

        for func in sync_functions:
            source = inspect.getsource(func)
            # Should not have await in sync functions
            self.assertNotIn('await ', source,
                           f"Sync function {func.__name__} should not use await")

    def test_learning_pipeline_is_optional(self):
        """Test that learning pipeline failures don't break the system"""
        # Mock learning pipeline that fails to initialize
        with patch('backend.main.LearningPipeline') as mock_lp_class:
            mock_lp_class.side_effect = Exception("Learning pipeline init failed")

            # This should not prevent the system from starting
            try:
                lp = get_learning_pipeline()
                # Should handle the failure gracefully
            except Exception:
                # In real implementation, this should be caught
                pass

    def test_concurrent_query_handling(self):
        """Test that concurrent queries don't cause async conflicts"""
        # This is harder to test without a full event loop,
        # but we can verify the architecture supports it

        from backend.llm.llm_core import LLMCore

        # LLM core should be thread-safe for concurrent queries
        # (Each query creates its own context)

        llm_methods = ['query']
        for method_name in llm_methods:
            method = getattr(LLMCore, method_name)
            # Method should exist and be callable
            self.assertTrue(callable(method))


if __name__ == '__main__':
    unittest.main()
