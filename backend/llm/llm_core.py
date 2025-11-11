"""
LLM Core - Main Interface for Language Model Operations
Integrates model loading, inference, and memory augmentation
"""
import logging
import time
from typing import Dict, Any, Optional, Generator, List
from .model_loader import GGUFModelLoader
from .inference import MemoryAugmentedInference
from .prompt_builder import PromptBuilder
from .memory_commander import MemoryCommander
from ..memory.memory_manager import MemoryManager
from ..memory.signal_processor import SignalProcessor
from ..memory.autonomy_system import AutonomySystem, OperationMode

logger = logging.getLogger(__name__)


class LLMCore:
    """Main LLM interface for Frankenstino AI with persistent model loading"""

    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize LLM core with persistent model loading (loads immediately)

        Args:
            memory_manager: Initialized memory manager instance
        """
        self.memory_manager = memory_manager
        self.prompt_builder = PromptBuilder()

        # Initialize Memory Commander for direct memory manipulation
        self.memory_commander = MemoryCommander(memory_manager)

        # ===== SCAFFOLDING & SUBSTRATE MODEL: Signal Processor for Phase 2 =====
        self.signal_processor = SignalProcessor(
            neural_mesh=memory_manager.neural_mesh,
            memory_manager=memory_manager,
            llm_interface=self  # Pass self for LLM concept extraction
        )

        # ===== PHASE 3: AUTONOMY SYSTEM - Critical mass detection and hybrid routing =====
        self.autonomy_system = AutonomySystem(
            neural_mesh=memory_manager.neural_mesh,
            memory_manager=memory_manager,
            signal_processor=self.signal_processor,
            llm_core=self
        )

        # UNIFIED MODEL: Single Qwen model for all tasks
        self.model_type = "qwen"

        # Load model immediately at startup
        logger.info(f"LLM Core initializing with persistent loading for {self.model_type} model")
        self._load_model_persistently()

    def _load_model_persistently(self) -> None:
        """Load model immediately and keep it loaded permanently"""
        try:
            logger.info(f"Loading {self.model_type} model persistently...")

            # Create model loader with unified Qwen model
            self.model_loader = GGUFModelLoader(model_type=self.model_type)

            # Load the model immediately
            if self.model_loader.load_model():
                # Create inference engine
                self.inference_engine = MemoryAugmentedInference(self.model_loader)
                logger.info("Model loaded successfully and will remain loaded")
            else:
                raise RuntimeError(f"Failed to load {self.model_type} language model")

        except Exception as e:
            logger.error(f"Persistent model loading failed: {e}")
            raise

    # REMOVED: _ensure_model_loaded - model is now loaded persistently at startup

    # REMOVED: Lazy unloading - model stays loaded permanently for performance
    # def _unload_model_if_idle(self):
    #     """Unload model to free memory (called after each query)"""
    #     pass  # Model persistence for unified architecture

    def query(self, user_query: str, category: str = None,
             temperature: float = 0.7, stream: bool = False,
             conversation_id: str = None, conversation_messages: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query with memory augmentation and conversation context
        Uses lazy loading: model loads on-demand and unloads immediately after

        Args:
            user_query: User's question or request
            category: Optional category to limit search
            temperature: Generation temperature
            stream: Whether to stream the response
            conversation_id: Current conversation ID for context

        Returns:
            Response dictionary with answer and metadata
        """
        query_start_time = time.time()

        try:
            logger.info(f"Processing query: '{user_query}' (category: {category}, conversation: {conversation_id})")

            # Search memory for relevant context using cached hybrid search
            memory_results = self.memory_manager.hybrid_search_cached(
                user_query,
                category=category,
                max_results=5,
                use_mesh=True
            )

            # Get conversation context if conversation_id provided
            conversation_context = []
            if conversation_id:
                conversation_messages = self.memory_manager.get_conversation_messages(
                    conversation_id, max_messages=10  # Get last 10 messages for better context
                )
                logger.info(f"Retrieved {len(conversation_messages)} messages for conversation {conversation_id}")

                if conversation_messages:
                    # Convert to context format - include all messages except current query if it's the last one
                    for i, msg in enumerate(conversation_messages):
                        # Skip if this is the current user query (very last message and matches current query)
                        if (i == len(conversation_messages) - 1 and
                            msg.get('role') == 'user' and
                            msg.get('content', '').strip() == user_query.strip()):
                            logger.info("Skipping current user query from context (already in prompt)")
                            continue

                        if msg.get('role') in ['user', 'assistant']:
                            conversation_context.append({
                                'role': msg['role'],
                                'content': msg.get('content', ''),
                                'type': 'conversation_context'
                            })

                    logger.info(f"Added {len(conversation_context)} conversation context messages")

            # Add conversation context to memory results
            if conversation_context:
                memory_results['conversation_context'] = conversation_context

            # Model is already loaded persistently - generate response directly
            # OPTIMIZE TEMPERATURE FOR QWEN
            if temperature == 0.7:  # Default temperature, optimize for Qwen
                temperature = self.inference_engine.default_temperature  # 0.6 for more focused responses

            response = self.inference_engine.generate_response(
                user_query,
                memory_results,
                temperature=temperature,
                stream=stream
            )

            # Check for memory manipulation commands in AI response
            ai_response_text = response.get('answer', '')
            memory_commands_executed = []

            if ai_response_text:
                # Parse and execute any memory commands in the response
                command = self.memory_commander.parse_command(ai_response_text)
                if command:
                    logger.info(f"Detected memory command in AI response: {command['type']}")
                    execution_result = self.memory_commander.execute_command(command)
                    memory_commands_executed.append({
                        'command': command,
                        'result': execution_result
                    })

                    # Add command execution note to response
                    if execution_result.get('success', False):
                        command_note = f"\n\n*Memory operation completed: {command['type']}*"
                        response['answer'] += command_note
                        logger.info(f"Successfully executed memory command: {command['type']}")
                    else:
                        error_note = f"\n\n*Memory operation failed: {execution_result.get('error', 'Unknown error')}*"
                        response['answer'] += error_note
                        logger.warning(f"Memory command execution failed: {execution_result.get('error', 'Unknown error')}")

            # ===== SCAFFOLDING & SUBSTRATE MODEL: Apply Hebbian learning =====
            # "Nodes that fire together, wire together" - reinforce connections between co-activated memory chunks
            try:
                memory_chunks = memory_results.get('results', [])
                if len(memory_chunks) > 1:
                    # Find corresponding neural mesh node IDs for the retrieved chunks
                    activated_node_ids = []
                    for chunk in memory_chunks:
                        chunk_source = chunk.get('source', '')
                        chunk_text = chunk.get('text', '')

                        # Find mesh node for this chunk
                        for node_id, node in self.memory_manager.neural_mesh.nodes.items():
                            if (node.metadata.get('document_id') == chunk_source and
                                chunk_text.startswith(node.metadata.get('text_preview', '')[:50])):
                                activated_node_ids.append(node_id)
                                break

                    # Apply Hebbian learning to reinforce connections between co-activated nodes
                    if len(activated_node_ids) > 1:
                        self.memory_manager.neural_mesh.reinforce_hebbian_connections(
                            activated_node_ids, reward=0.1
                        )
                        logger.debug(f"Applied Hebbian learning to {len(activated_node_ids)} co-activated memory nodes")

            except Exception as e:
                logger.debug(f"Hebbian learning application failed (non-critical): {e}")

            # Add memory search metadata
            response.update({
                "memory_search": {
                    "total_results": memory_results.get('total_found', 0),
                    "categories_searched": memory_results.get('categories', []),
                    "search_type": memory_results.get('search_type', 'unknown'),
                    "conversation_context_used": len(conversation_context)
                },
                "memory_commands": {
                    "commands_executed": len(memory_commands_executed),
                    "command_details": memory_commands_executed
                },
                "performance": {
                    "model_persistently_loaded": True,
                    "query_processed_efficiently": True
                }
            })

            query_time = time.time() - query_start_time
            logger.info(f"Query processed successfully in {query_time:.2f}s. Response length: {len(response.get('answer', ''))}")

            return response

        except Exception as e:
            import traceback
            logger.error(f"Query processing failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "query": user_query,
                "answer": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "error": str(e),
                "memory_chunks_used": 0,
                "traceback": traceback.format_exc(),  # Include full traceback for debugging
                "performance": {
                    "model_loaded_on_demand": False,
                    "query_failed": True
                }
            }
        # NOTE: Removed lazy unloading - with unified single model, keep it loaded for performance







    def stream_query(self, user_query: str, category: str = None,
                    temperature: float = 0.7) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a query response token by token

        Args:
            user_query: User's question
            category: Optional category filter
            temperature: Generation temperature

        Yields:
            Response chunks
        """
        try:
            # Search memory first using cached hybrid search
            memory_results = self.memory_manager.hybrid_search_cached(
                user_query,
                category=category,
                max_results=5,
                use_mesh=True
            )

            # Stream the response
            for chunk in self.inference_engine.stream_response(
                user_query,
                memory_results,
                temperature=temperature
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield {
                "token": f"Error: {str(e)}",
                "full_response": f"I apologize, but I encountered an error: {str(e)}",
                "finished": True,
                "error": True
            }

    def get_model_status(self) -> Dict[str, Any]:
        """Get model loading and performance status"""
        if not self.model_loader:
            return {"status": "not_initialized"}

        model_info = self.model_loader.get_model_info()
        model_info.update({
            "inference_engine_ready": self.inference_engine is not None,
            "prompt_builder_ready": self.prompt_builder is not None,
            "memory_integration": self.memory_manager is not None
        })

        return model_info

    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        from ..config import settings

        models_info = {
            "qwen": {
                "name": "Qwen2.5-7B-Instruct-1M",
                "path": settings.model_path,
                "active": self.model_type == "qwen",
                "description": "General-purpose instruction-tuned model with large context window"
            }
        }

        # Check if model file exists
        for model_key, model_info in models_info.items():
            model_path = Path(model_info["path"])
            model_info["exists"] = model_path.exists()
            if model_path.exists():
                model_info["size_mb"] = model_path.stat().st_size / (1024 * 1024)

        return models_info

    def estimate_query_cost(self, query: str, category: str = None) -> Dict[str, Any]:
        """
        Estimate computational cost of a query

        Args:
            query: Query to estimate
            category: Optional category filter

        Returns:
            Cost estimation data
        """
        try:
            # Get memory search results
            memory_results = self.memory_manager.search_memory(
                query,
                category=category,
                max_results=5
            )

            # Build sample prompt
            sample_prompt = self.inference_engine.build_prompt(
                query,
                memory_results.get('results', [])
            )

            # Get cost estimation
            cost_estimate = self.inference_engine.estimate_cost(sample_prompt)

            # Add memory search cost
            cost_estimate.update({
                "memory_search_chunks": len(memory_results.get('results', [])),
                "memory_search_type": memory_results.get('search_type', 'unknown')
            })

            return cost_estimate

        except Exception as e:
            logger.error(f"Cost estimation failed: {e}")
            return {
                "error": str(e),
                "estimated_time_seconds": 0,
                "model_loaded": self.model_loader.is_loaded if self.model_loader else False
            }

    def generate_without_memory(self, prompt: str, max_tokens: int = 256,
                               temperature: float = 0.7) -> str:
        """
        Generate text without memory augmentation (direct model access)

        Args:
            prompt: Raw prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature

        Returns:
            Generated text
        """
        if not self.model_loader or not self.model_loader.is_loaded:
            raise RuntimeError("Model not loaded")

        return self.model_loader.generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def query_substrate(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query using substrate-only signal propagation
        Part of Scaffolding & Substrate Model - Phase 2

        Args:
            user_query: Natural language query
            context: Optional context information

        Returns:
            Substrate processing results
        """
        try:
            logger.info(f"Processing substrate query: '{user_query[:50]}...'")

            # Use signal processor for substrate-only processing
            substrate_result = self.signal_processor.process_query_substrate(
                user_query, context
            )

            if substrate_result.get('success', False):
                pattern_result = substrate_result.get('pattern_result', {})

                # Convert answer nodes to text content
                answer_nodes = pattern_result.get('answer_nodes', [])
                answer_texts = []

                for node_id in answer_nodes:
                    if node_id in self.memory_manager.neural_mesh.nodes:
                        node = self.memory_manager.neural_mesh.nodes[node_id]
                        text_content = node.metadata.get('text_preview', '')
                        if text_content:
                            answer_texts.append(text_content)

                # Generate response using retrieved content
                if answer_texts:
                    # Create a simple response from the retrieved content
                    combined_answer = " ".join(answer_texts[:3])  # Use top 3 results

                    response = {
                        'query': user_query,
                        'answer': combined_answer,
                        'method': 'substrate_signal_propagation',
                        'confidence': pattern_result.get('confidence', 0.5),
                        'answer_nodes': len(answer_nodes),
                        'processing_time': substrate_result.get('processing_time', 0),
                        'substrate_details': substrate_result
                    }
                else:
                    response = {
                        'query': user_query,
                        'answer': "I don't have enough information in my substrate to answer this query.",
                        'method': 'substrate_signal_propagation',
                        'confidence': 0.0,
                        'answer_nodes': 0,
                        'processing_time': substrate_result.get('processing_time', 0),
                        'substrate_details': substrate_result
                    }
            else:
                response = {
                    'query': user_query,
                    'answer': "Substrate processing failed. Falling back to standard processing.",
                    'method': 'substrate_failed',
                    'error': substrate_result.get('reason', 'unknown'),
                    'processing_time': substrate_result.get('processing_time', 0),
                    'substrate_details': substrate_result
                }

            logger.info(f"Substrate query completed: success={substrate_result.get('success', False)}, "
                       f"nodes={len(substrate_result.get('pattern_result', {}).get('answer_nodes', []))}")

            return response

        except Exception as e:
            logger.error(f"Substrate query failed: {e}")
            return {
                'query': user_query,
                'answer': f"Substrate processing error: {str(e)}",
                'method': 'substrate_error',
                'error': str(e)
            }

    def get_substrate_stats(self) -> Dict[str, Any]:
        """Get substrate processing statistics"""
        return {
            'signal_processor': self.signal_processor.get_processing_stats(),
            'neural_mesh_nodes': len(self.memory_manager.neural_mesh.nodes),
            'neural_mesh_edges': len(self.memory_manager.neural_mesh.edges),
            'substrate_ready': True
        }

    def query_autonomous(self, user_query: str, category: str = None,
                        temperature: float = 0.7, stream: bool = False,
                        conversation_id: str = None) -> Dict[str, Any]:
        """
        Process a query using Phase 3 autonomy system with intelligent mode selection
        Automatically chooses between hybrid and substrate-only based on maturity and query analysis

        Args:
            user_query: User's question or request
            category: Optional category to limit search
            temperature: Generation temperature
            stream: Whether to stream the response
            conversation_id: Current conversation ID for context

        Returns:
            Response dictionary with answer and autonomy metadata
        """
        try:
            logger.info(f"Processing autonomous query: '{user_query}' (category: {category}, conversation: {conversation_id})")

            # Use autonomy system for intelligent processing with feedback collection
            response = self.autonomy_system.process_query_with_feedback(
                query=user_query,
                context={
                    'category': category,
                    'temperature': temperature,
                    'stream': stream,
                    'conversation_id': conversation_id
                }
            )

            logger.info(f"Autonomous query completed: mode={response.get('autonomy', {}).get('mode_used', 'unknown')}, "
                       f"maturity={response.get('autonomy', {}).get('maturity_score', 0):.3f}")

            return response

        except Exception as e:
            logger.error(f"Autonomous query failed: {e}")
            # Fallback to regular hybrid query
            logger.info("Falling back to regular hybrid query processing")
            return self.query(user_query, category, temperature, stream, conversation_id)

    def get_autonomy_stats(self) -> Dict[str, Any]:
        """Get autonomy system statistics and maturity assessment"""
        return self.autonomy_system.get_system_stats()

    def warmup_model(self):
        """Warm up the model with a small generation task"""
        try:
            logger.info("Warming up model...")
            warmup_prompt = "Hello, this is a test prompt for model warmup."
            self.generate_without_memory(warmup_prompt, max_tokens=10)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")



    def __del__(self):
        """Cleanup on deletion"""
        if self.model_loader:
            self.model_loader.unload_model()
