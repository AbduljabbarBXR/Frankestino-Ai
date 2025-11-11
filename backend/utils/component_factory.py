"""
Component Factory Pattern - Eliminates Lazy Loading for Faster Startup
Provides centralized component management and dependency injection
"""

from typing import Optional, TYPE_CHECKING
import logging

# Import protocols for type safety
from .protocols import (
    MemoryManagerProtocol, LLMCoreProtocol, MemoryCuratorProtocol,
    MemoryTaxonomyProtocol, MemoryMetricsProtocol, LearningPipelineProtocol,
    MetricsCollectorProtocol
)

# Use TYPE_CHECKING to avoid runtime imports for better performance
if TYPE_CHECKING:
    from ..memory.memory_manager import MemoryManager
    from ..llm.llm_core import LLMCore
    from ..llm.memory_curator import MemoryCurator
    from ..memory.memory_taxonomy import MemoryTaxonomy
    from ..memory.memory_metrics import MemoryMetrics, QATestHarness
    from ..learning_pipeline import LearningPipeline
    from ..monitoring.metrics_collector import MetricsCollector
    from ..monitoring.hallucination_detector import HallucinationDetector
    from ..monitoring.precision_tester import PrecisionTester

logger = logging.getLogger(__name__)

class ComponentFactory:
    """
    Centralized component factory for dependency injection.
    Eliminates lazy loading and circular imports for faster startup.
    """

    # Singleton instances
    _memory_manager: Optional['MemoryManager'] = None
    _llm_core: Optional['LLMCore'] = None
    _memory_curator: Optional['MemoryCurator'] = None
    _memory_taxonomy: Optional['MemoryTaxonomy'] = None
    _memory_metrics: Optional['MemoryMetrics'] = None
    _qa_harness: Optional['QATestHarness'] = None
    _learning_pipeline: Optional['LearningPipeline'] = None
    _metrics_collector: Optional['MetricsCollector'] = None
    _hallucination_detector: Optional['HallucinationDetector'] = None
    _precision_tester: Optional['PrecisionTester'] = None

    # Initialization flags
    _initialized = False

    @classmethod
    def initialize_components(cls) -> None:
        """Initialize all components at startup - called once during app startup"""
        if cls._initialized:
            return

        logger.info("Initializing components via ComponentFactory...")

        try:
            # Initialize in dependency order to avoid circular imports

            # 1. Memory Curator (no dependencies)
            from ..llm.memory_curator import MemoryCurator
            cls._memory_curator = MemoryCurator()
            logger.debug("Memory curator initialized")

            # 2. Memory Manager (depends on curator)
            from ..memory.memory_manager import MemoryManager
            cls._memory_manager = MemoryManager(curator=cls._memory_curator)
            logger.debug("Memory manager initialized")

            # 3. LLM Core (depends on memory manager)
            from ..llm.llm_core import LLMCore
            cls._llm_core = LLMCore(cls._memory_manager)
            logger.debug("LLM core initialized")

            # 4. Memory Taxonomy (depends on memory manager and curator)
            from ..memory.memory_taxonomy import MemoryTaxonomy, MemoryType
            taxonomy = MemoryTaxonomy()

            # Register memory managers for each type
            # Create separate instances for different policies
            stable_manager = MemoryManager(curator=cls._memory_curator)
            conv_manager = MemoryManager(curator=cls._memory_curator)
            func_manager = MemoryManager(curator=cls._memory_curator)

            taxonomy.register_memory_manager(MemoryType.STABLE, stable_manager)
            taxonomy.register_memory_manager(MemoryType.CONVERSATIONAL, conv_manager)
            taxonomy.register_memory_manager(MemoryType.FUNCTIONAL, func_manager)

            cls._memory_taxonomy = taxonomy
            logger.debug("Memory taxonomy initialized")

            # 5. Memory Metrics (depends on memory manager and curator)
            from ..memory.memory_metrics import MemoryMetrics
            cls._memory_metrics = MemoryMetrics(cls._memory_manager, cls._memory_curator)
            logger.debug("Memory metrics initialized")

            # 6. QA Harness (depends on memory manager and metrics)
            from ..memory.memory_metrics import QATestHarness
            cls._qa_harness = QATestHarness(cls._memory_manager, cls._memory_metrics)
            logger.debug("QA harness initialized")

            # 7. Learning Pipeline (depends on memory manager, neural mesh, curator)
            from ..learning_pipeline import LearningPipeline
            neural_mesh = cls._memory_manager.neural_mesh
            cls._learning_pipeline = LearningPipeline(cls._memory_manager, neural_mesh, cls._memory_curator)
            logger.debug("Learning pipeline initialized")

            # 8. Monitoring components (optional - can fail gracefully)
            try:
                from ..monitoring.metrics_collector import MetricsCollector
                cls._metrics_collector = MetricsCollector()
                logger.debug("Metrics collector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize metrics collector: {e}")

            try:
                from ..monitoring.hallucination_detector import HallucinationDetector
                cls._hallucination_detector = HallucinationDetector(cls._llm_core.model_loader if cls._llm_core else None)
                logger.debug("Hallucination detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hallucination detector: {e}")

            try:
                from ..monitoring.precision_tester import PrecisionTester
                cls._precision_tester = PrecisionTester(cls._memory_manager)
                logger.debug("Precision tester initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize precision tester: {e}")

            cls._initialized = True
            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    @classmethod
    def get_memory_manager(cls) -> 'MemoryManager':
        """Get memory manager instance"""
        if not cls._initialized:
            cls.initialize_components()
        if cls._memory_manager is None:
            raise RuntimeError("Memory manager not initialized")
        return cls._memory_manager

    @classmethod
    def get_llm_core(cls) -> 'LLMCore':
        """Get LLM core instance"""
        if not cls._initialized:
            cls.initialize_components()
        if cls._llm_core is None:
            raise RuntimeError("LLM core not initialized")
        return cls._llm_core

    @classmethod
    def get_memory_curator(cls) -> 'MemoryCurator':
        """Get memory curator instance"""
        if not cls._initialized:
            cls.initialize_components()
        if cls._memory_curator is None:
            raise RuntimeError("Memory curator not initialized")
        return cls._memory_curator

    @classmethod
    def get_memory_taxonomy(cls) -> 'MemoryTaxonomy':
        """Get memory taxonomy instance"""
        if not cls._initialized:
            cls.initialize_components()
        if cls._memory_taxonomy is None:
            raise RuntimeError("Memory taxonomy not initialized")
        return cls._memory_taxonomy

    @classmethod
    def get_memory_metrics(cls) -> 'MemoryMetrics':
        """Get memory metrics instance"""
        if not cls._initialized:
            cls.initialize_components()
        if cls._memory_metrics is None:
            raise RuntimeError("Memory metrics not initialized")
        return cls._memory_metrics

    @classmethod
    def get_qa_harness(cls) -> 'QATestHarness':
        """Get QA harness instance"""
        if not cls._initialized:
            cls.initialize_components()
        if cls._qa_harness is None:
            raise RuntimeError("QA harness not initialized")
        return cls._qa_harness

    @classmethod
    def get_learning_pipeline(cls) -> 'LearningPipeline':
        """Get learning pipeline instance"""
        if not cls._initialized:
            cls.initialize_components()
        if cls._learning_pipeline is None:
            raise RuntimeError("Learning pipeline not initialized")
        return cls._learning_pipeline

    @classmethod
    def get_metrics_collector(cls) -> Optional['MetricsCollector']:
        """Get metrics collector instance (optional)"""
        if not cls._initialized:
            cls.initialize_components()
        return cls._metrics_collector

    @classmethod
    def get_hallucination_detector(cls) -> Optional['HallucinationDetector']:
        """Get hallucination detector instance (optional)"""
        if not cls._initialized:
            cls.initialize_components()
        return cls._hallucination_detector

    @classmethod
    def get_precision_tester(cls) -> Optional['PrecisionTester']:
        """Get precision tester instance (optional)"""
        if not cls._initialized:
            cls.initialize_components()
        return cls._precision_tester

    @classmethod
    def get_mesh_accelerator(cls):
        """Get mesh accelerator instance (optional)"""
        try:
            from ..utils.mesh_accelerator import MeshAccelerator
            return MeshAccelerator()
        except ImportError:
            # Fallback if accelerator not available
            class MockAccelerator:
                def get_performance_stats(self):
                    return {
                        'available': False,
                        'acceleration_factor': 1.0,
                        'memory_usage_mb': 0,
                        'operations_per_second': 0
                    }
            return MockAccelerator()

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown all components gracefully"""
        logger.info("Shutting down components...")

        try:
            if cls._learning_pipeline:
                # Note: This would need to be made async in real implementation
                # await cls._learning_pipeline.stop_processing()
                pass

            if cls._llm_core and hasattr(cls._llm_core, 'model_loader'):
                cls._llm_core.model_loader.unload_model()

            if cls._memory_curator and hasattr(cls._memory_curator, 'model_loader'):
                cls._memory_curator.model_loader.unload_model()

            # Clear all references
            cls._memory_manager = None
            cls._llm_core = None
            cls._memory_curator = None
            cls._memory_taxonomy = None
            cls._memory_metrics = None
            cls._qa_harness = None
            cls._learning_pipeline = None
            cls._metrics_collector = None
            cls._hallucination_detector = None
            cls._precision_tester = None

            cls._initialized = False
            logger.info("Component shutdown complete")

        except Exception as e:
            logger.error(f"Error during component shutdown: {e}")

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if components are initialized"""
        return cls._initialized

    @classmethod
    def get_initialization_status(cls) -> dict:
        """Get detailed initialization status"""
        return {
            'initialized': cls._initialized,
            'components': {
                'memory_manager': cls._memory_manager is not None,
                'llm_core': cls._llm_core is not None,
                'memory_curator': cls._memory_curator is not None,
                'memory_taxonomy': cls._memory_taxonomy is not None,
                'memory_metrics': cls._memory_metrics is not None,
                'qa_harness': cls._qa_harness is not None,
                'learning_pipeline': cls._learning_pipeline is not None,
                'metrics_collector': cls._metrics_collector is not None,
                'hallucination_detector': cls._hallucination_detector is not None,
                'precision_tester': cls._precision_tester is not None,
            }
        }
