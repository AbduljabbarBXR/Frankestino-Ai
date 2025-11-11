"""
Learning Pipeline - Asynchronous Learning System for Frankenstino AI
Handles background learning from user interactions without blocking responses
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningPipeline:
    """
    Asynchronous learning system that processes user interactions in the background

    This decouples learning time from response time, making the AI feel faster
    while continuously improving through every interaction.

    Part of Scaffolding & Substrate Model - Phase 1: Background pattern learning
    """

    def __init__(self, memory_manager, neural_mesh, memory_curator=None):
        """
        Initialize the learning pipeline

        Args:
            memory_manager: Memory manager instance
            neural_mesh: Neural mesh instance
            memory_curator: Optional memory curator for advanced processing
        """
        self.memory_manager = memory_manager
        self.neural_mesh = neural_mesh
        self.memory_curator = memory_curator

        # ===== SCAFFOLDING & SUBSTRATE MODEL: Pattern Recognition Engine =====
        try:
            from .memory.pattern_recognition import PatternRecognitionEngine
            self.pattern_recognition = PatternRecognitionEngine(neural_mesh, memory_manager)
            logger.info("Pattern Recognition Engine integrated into Learning Pipeline")
        except ImportError as e:
            logger.warning(f"Could not import Pattern Recognition Engine: {e}")
            self.pattern_recognition = None

        # Learning queue for background processing
        self.learning_queue = asyncio.Queue()
        self.is_running = False
        self.processed_count = 0
        self._processing_task = None  # Store reference to background task
        self.learning_stats = {
            'conversations_stored': 0,
            'knowledge_chunks_extracted': 0,
            'neural_connections_updated': 0,
            'patterns_reinforced': 0,
            'patterns_learned': 0,  # Scaffolding & Substrate addition
            'background_learning_cycles': 0  # Scaffolding & Substrate addition
        }

        logger.info("Learning Pipeline initialized with Scaffolding & Substrate capabilities")

    async def start_processing(self):
        """Start the background learning task"""
        if self.is_running:
            logger.warning("Learning pipeline already running")
            return

        self.is_running = True
        logger.info("Starting background learning pipeline...")

        # Start the background processing task and store reference
        self._processing_task = asyncio.create_task(self._process_learning_queue())

    async def stop_processing(self):
        """Stop the background learning task"""
        if not self.is_running:
            return

        logger.info("Stopping background learning pipeline...")
        self.is_running = False

        # Cancel the background task if it exists
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                logger.info("Background learning task cancelled successfully")
            except Exception as e:
                logger.warning(f"Error cancelling background task: {e}")

        self._processing_task = None

    async def learn_from_interaction(self, query: str, response: str,
                                   conversation_id: str = None,
                                   context: Dict[str, Any] = None) -> bool:
        """
        Queue an interaction for background learning

        Args:
            query: User's query
            response: AI's response
            conversation_id: Optional conversation ID
            context: Additional context information

        Returns:
            True if queued successfully
        """
        try:
            # Start background processing if not already running
            if not self.is_running:
                await self.start_processing()

            interaction = {
                'query': query,
                'response': response,
                'conversation_id': conversation_id,
                'context': context or {},
                'timestamp': time.time(),
                'processed': False
            }

            await self.learning_queue.put(interaction)
            logger.debug(f"Queued interaction for learning: '{query[:50]}...'")
            return True

        except Exception as e:
            logger.error(f"Failed to queue interaction for learning: {e}")
            return False

    async def _process_learning_queue(self):
        """Background task that processes the learning queue"""
        logger.info("Background learning processor started")

        while self.is_running:
            try:
                # Wait for an interaction to process
                interaction = await self.learning_queue.get()

                # Process the interaction
                await self._analyze_and_learn(interaction)

                # Mark as processed
                self.processed_count += 1
                self.learning_queue.task_done()

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in learning queue processor: {e}")
                await asyncio.sleep(1)  # Wait before retrying

    async def _analyze_and_learn(self, interaction: Dict[str, Any]):
        """
        Analyze an interaction and learn from it

        This is the core learning logic that runs in the background
        """
        try:
            query = interaction['query']
            response = interaction['response']
            conversation_id = interaction.get('conversation_id')
            timestamp = interaction['timestamp']

            logger.debug(f"Learning from interaction: '{query[:50]}...'")

            # Step 1: Store conversation for pattern learning
            await self._store_conversation(interaction)

            # Step 2: Extract valuable knowledge chunks
            knowledge_chunks = await self._extract_knowledge(query, response)

            # Step 3: Update neural mesh connections
            if knowledge_chunks:
                await self._update_neural_connections(knowledge_chunks)

            # Step 4: Reinforce learned patterns
            await self._reinforce_patterns(interaction)

            # Update statistics
            self.learning_stats['conversations_stored'] += 1
            self.learning_stats['knowledge_chunks_extracted'] += len(knowledge_chunks) if knowledge_chunks else 0

            logger.debug(f"Successfully learned from interaction, extracted {len(knowledge_chunks) if knowledge_chunks else 0} knowledge chunks")

        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")

    async def _store_conversation(self, interaction: Dict[str, Any]):
        """Store the conversation for future pattern learning"""
        try:
            conversation_id = interaction.get('conversation_id')
            if not conversation_id:
                # Generate a conversation ID if not provided
                conversation_id = f"conv_{int(interaction['timestamp'])}_{hash(interaction['query']) % 10000}"

            messages = [
                {
                    'role': 'user',
                    'content': interaction['query'],
                    'timestamp': interaction['timestamp']
                },
                {
                    'role': 'assistant',
                    'content': interaction['response'],
                    'timestamp': interaction['timestamp'] + 0.1  # Slight delay for assistant
                }
            ]

            # Store in memory system
            success = self.memory_manager.store_conversation(conversation_id, messages)
            if success:
                logger.debug(f"Stored conversation {conversation_id} for learning")
            else:
                logger.warning(f"Failed to store conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")

    async def _extract_knowledge(self, query: str, response: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract valuable knowledge chunks from the interaction

        Uses the memory curator if available, otherwise basic extraction
        """
        try:
            # If we have a memory curator, use it for advanced knowledge extraction
            if self.memory_curator and self.memory_curator.is_ready():
                return await self._extract_with_curator(query, response)
            else:
                return self._extract_basic_knowledge(query, response)

        except Exception as e:
            logger.error(f"Failed to extract knowledge: {e}")
            return None

    async def _extract_with_curator(self, query: str, response: str) -> Optional[List[Dict[str, Any]]]:
        """Use memory curator for advanced knowledge extraction"""
        try:
            # Combine query and response for analysis
            full_text = f"User: {query}\nAssistant: {response}"

            # Use curator to summarize and extract key information
            summary = await self.memory_curator.summarize_chunk(full_text)

            if summary and summary.get('summary'):
                # Create knowledge chunk from summary
                chunk = {
                    'text': summary['summary'],
                    'source': 'conversation_learning',
                    'confidence': summary.get('confidence', 0.5),
                    'metadata': {
                        'original_query': query,
                        'response_length': len(response),
                        'extraction_method': 'curator_summary'
                    }
                }
                return [chunk]

        except Exception as e:
            logger.error(f"Curator-based extraction failed: {e}")
            return self._extract_basic_knowledge(query, response)

    def _extract_basic_knowledge(self, query: str, response: str) -> Optional[List[Dict[str, Any]]]:
        """Basic knowledge extraction without curator"""
        try:
            # Simple heuristics for valuable information
            combined_text = f"{query} {response}".lower()

            # Check if this interaction contains potentially valuable knowledge
            knowledge_indicators = [
                'project', 'task', 'deadline', 'meeting', 'schedule',
                'requirement', 'specification', 'design', 'implementation',
                'problem', 'solution', 'issue', 'fix', 'update',
                'preference', 'like', 'dislike', 'prefer', 'usually',
                'experience', 'background', 'skill', 'expertise'
            ]

            has_valuable_content = any(indicator in combined_text for indicator in knowledge_indicators)

            if has_valuable_content and len(response) > 50:  # Substantial response
                chunk = {
                    'text': f"User query: {query}\nAI response: {response}",
                    'source': 'conversation_learning',
                    'confidence': 0.3,  # Lower confidence for basic extraction
                    'metadata': {
                        'extraction_method': 'basic_heuristics',
                        'response_length': len(response)
                    }
                }
                return [chunk]

            return None

        except Exception as e:
            logger.error(f"Basic knowledge extraction failed: {e}")
            return None

    async def _update_neural_connections(self, knowledge_chunks: List[Dict[str, Any]]):
        """Update neural mesh connections based on new knowledge"""
        try:
            if not knowledge_chunks:
                return

            for chunk in knowledge_chunks:
                # Add the knowledge to the memory system
                # This will automatically update neural mesh connections
                success = self.memory_manager.add_knowledge_chunk(
                    text=chunk['text'],
                    metadata={
                        **chunk.get('metadata', {}),
                        'source': 'learning_pipeline',
                        'confidence': chunk.get('confidence', 0.5)
                    }
                )

                if not success:
                    logger.warning(f"Failed to add knowledge chunk: {chunk['text'][:50]}...")

            self.learning_stats['neural_connections_updated'] += len(knowledge_chunks)
            logger.debug(f"Updated neural mesh with {len(knowledge_chunks)} knowledge chunks")

        except Exception as e:
            logger.error(f"Failed to update neural connections: {e}")

    async def _reinforce_patterns(self, interaction: Dict[str, Any]):
        """
        Reinforce learned patterns based on successful interactions
        Part of Scaffolding & Substrate Model - Phase 1: Background pattern learning
        """
        try:
            # ===== SCAFFOLDING & SUBSTRATE MODEL: Pattern Recognition =====
            if self.pattern_recognition:
                # Analyze the interaction for successful patterns
                query = interaction.get('query', '')
                response = interaction.get('response', '')
                context = interaction.get('context', {})

                # Extract memory search results from context if available
                memory_results = context.get('memory_results', {})
                response_quality = context.get('response_quality', 0.7)  # Default good quality
                execution_time = context.get('execution_time', 1.0)  # Default 1 second

                # Analyze query success and learn patterns
                analysis_result = self.pattern_recognition.analyze_query_success(
                    query=query,
                    memory_results=memory_results,
                    response_quality=response_quality,
                    execution_time=execution_time
                )

                if analysis_result.get('pattern_extracted', False):
                    self.learning_stats['patterns_learned'] += 1
                    logger.debug(f"Learned pattern from interaction: {analysis_result.get('query_type', 'unknown')}")

                # Run background learning periodically (every 10 interactions)
                if self.processed_count % 10 == 0:
                    background_stats = self.pattern_recognition.run_background_learning()
                    if background_stats and not background_stats.get('error'):
                        self.learning_stats['background_learning_cycles'] += 1
                        logger.info(f"Completed background learning cycle: {background_stats}")

            # Legacy pattern reinforcement (keep for compatibility)
            self.learning_stats['patterns_reinforced'] += 1

        except Exception as e:
            logger.error(f"Failed to reinforce patterns: {e}")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning pipeline statistics"""
        return {
            'is_running': self.is_running,
            'queue_size': self.learning_queue.qsize(),
            'processed_count': self.processed_count,
            'learning_stats': self.learning_stats.copy()
        }

    async def force_process_queue(self):
        """Force process all items in the learning queue (for testing)"""
        logger.info("Force processing learning queue...")

        while not self.learning_queue.empty():
            try:
                interaction = await asyncio.wait_for(self.learning_queue.get(), timeout=1.0)
                await self._analyze_and_learn(interaction)
                self.processed_count += 1
                self.learning_queue.task_done()
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error during forced processing: {e}")

        logger.info("Force processing complete")
