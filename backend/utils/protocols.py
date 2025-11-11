"""
Protocol Classes for Type Safety
Separates interface definitions from implementations
"""

from typing import Protocol, Optional, Dict, List, Any, AsyncGenerator
from pathlib import Path

class MemoryManagerProtocol(Protocol):
    """Protocol for memory manager interface"""

    def ingest_document(self, file_path: Path, category: Optional[str] = None) -> Dict[str, Any]:
        """Ingest a document into memory"""
        ...

    def hybrid_search(self, query: str, category: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Perform hybrid search across memory"""
        ...

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        ...

    def store_conversation(self, conversation_id: str, messages: List[Dict], category: Optional[str] = None) -> bool:
        """Store conversation in memory"""
        ...

    def get_conversation_history(self, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get conversation history"""
        ...

    def list_categories(self) -> List[Dict]:
        """List available categories"""
        ...

class LLMCoreProtocol(Protocol):
    """Protocol for LLM core interface"""

    def query(self, query: str, category: Optional[str] = None, temperature: float = 0.7,
              stream: bool = False, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query through the LLM"""
        ...

    def query_substrate(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process query using substrate-only mode"""
        ...

    def query_autonomous(self, query: str, category: Optional[str] = None, temperature: float = 0.7,
                        stream: bool = False, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query using autonomous mode"""
        ...

    def get_model_status(self) -> Dict[str, Any]:
        """Get model status information"""
        ...

    def get_substrate_stats(self) -> Dict[str, Any]:
        """Get substrate processing statistics"""
        ...

    def get_autonomy_stats(self) -> Dict[str, Any]:
        """Get autonomy system statistics"""
        ...

class MemoryCuratorProtocol(Protocol):
    """Protocol for memory curator interface"""

    def is_ready(self) -> bool:
        """Check if curator is ready"""
        ...

    async def summarize_chunk(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Summarize a text chunk"""
        ...

    async def validate_memory_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a memory chunk"""
        ...

    async def process_memory_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of memory chunks"""
        ...

    async def detect_duplicates(self, new_chunk: Dict[str, Any], existing_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect duplicate chunks"""
        ...

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        ...

class MemoryTaxonomyProtocol(Protocol):
    """Protocol for memory taxonomy interface"""

    def route_ingestion_request(self, content: Any, memory_type: Any, source: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Route ingestion request to appropriate memory type"""
        ...

    def query_memory_type(self, query: str, memory_type: Any, source: str,
                         max_results: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query a specific memory type"""
        ...

    def validate_ingestion_request(self, memory_type: Any, source: str, content_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate an ingestion request"""
        ...

    def get_memory_type_stats(self, memory_type: Optional[Any] = None) -> Dict[str, Any]:
        """Get statistics for memory types"""
        ...

    def apply_retention_policies(self) -> None:
        """Apply retention policies to all memory types"""
        ...

class MemoryMetricsProtocol(Protocol):
    """Protocol for memory metrics interface"""

    def record_query_metrics(self, query: str, response: str, memory_results: Dict[str, Any],
                           response_time_ms: float, source: str) -> None:
        """Record query metrics"""
        ...

    def get_quality_report(self) -> Dict[str, Any]:
        """Get quality report"""
        ...

    def detect_regression(self, baseline_window_hours: int = 24) -> Dict[str, Any]:
        """Detect performance regression"""
        ...

class LearningPipelineProtocol(Protocol):
    """Protocol for learning pipeline interface"""

    async def learn_from_interaction(self, query: str, response: str, conversation_id: Optional[str],
                                   context: Optional[Dict] = None) -> None:
        """Learn from user interaction"""
        ...

    def is_running(self) -> bool:
        """Check if pipeline is running"""
        ...

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        ...

class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collector interface"""

    def record_hallucination_score(self, score: float) -> None:
        """Record hallucination score"""
        ...

    def record_retrieval_precision(self, k: int, precision: float) -> None:
        """Record retrieval precision"""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        ...

class ComponentFactoryProtocol(Protocol):
    """Protocol for component factory interface"""

    @classmethod
    def initialize_components(cls) -> None:
        """Initialize all components"""
        ...

    @classmethod
    def get_memory_manager(cls) -> MemoryManagerProtocol:
        """Get memory manager instance"""
        ...

    @classmethod
    def get_llm_core(cls) -> LLMCoreProtocol:
        """Get LLM core instance"""
        ...

    @classmethod
    def get_memory_curator(cls) -> MemoryCuratorProtocol:
        """Get memory curator instance"""
        ...

    @classmethod
    def get_memory_taxonomy(cls) -> MemoryTaxonomyProtocol:
        """Get memory taxonomy instance"""
        ...

    @classmethod
    def get_memory_metrics(cls) -> MemoryMetricsProtocol:
        """Get memory metrics instance"""
        ...

    @classmethod
    def get_learning_pipeline(cls) -> LearningPipelineProtocol:
        """Get learning pipeline instance"""
        ...

    @classmethod
    def get_metrics_collector(cls) -> Optional[MetricsCollectorProtocol]:
        """Get metrics collector instance"""
        ...

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown all components"""
        ...
