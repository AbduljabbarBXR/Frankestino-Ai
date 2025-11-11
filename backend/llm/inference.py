"""
LLM Inference with Memory Augmentation
Handles prompt construction and response generation
"""
import logging
from typing import List, Dict, Any, Optional, Generator
from .model_loader import GGUFModelLoader

logger = logging.getLogger(__name__)


class MemoryAugmentedInference:
    """Handles LLM inference with memory context"""

    def __init__(self, model_loader: GGUFModelLoader):
        """
        Initialize inference engine

        Args:
            model_loader: Loaded GGUF model instance
        """
        self.model_loader = model_loader
        self.max_context_tokens = 30720  # DeepSeek has 32K context, leave room for response
        self.response_max_tokens = 1024  # DeepSeek can handle longer responses

        # OPTIMIZE FOR CONVERSATIONAL RESPONSES
        self.default_temperature = 0.8  # Higher for more natural, conversational flow
        self.reasoning_boost = False  # Disable analytical reasoning mode

    def build_prompt(self, query: str, memory_results: Dict[str, Any],
                    system_prompt: Optional[str] = None) -> str:
        """
        Build a memory-augmented prompt with conversation context

        Args:
            query: User query
            memory_results: Memory search results (chunks + conversations + context)
            system_prompt: Optional system prompt

        Returns:
            Complete prompt for the model
        """
        # CONVERSATIONAL SYSTEM PROMPT FOR DIRECT, FRIENDLY RESPONSES
        if system_prompt is None:
            system_prompt = (
                "You are Frankenstino AI, a helpful and friendly assistant with advanced memory capabilities. "
                "Give direct, conversational answers that feel natural and engaging. "
                "Use the provided context to enhance your responses, but don't show your analytical process. "
                "Be warm, informative, and conversational like a knowledgeable friend."
            )

        # Build structured context with prioritization
        context_parts = []

        # PRIORITY 1: Recent Conversation Context (Highest Priority)
        conversation_context = memory_results.get('conversation_context', [])
        if conversation_context:
            context_parts.append("=== RECENT CONVERSATION HISTORY ===")
            context_parts.append("Use this to personalize your response and maintain conversation flow:")

            for msg in conversation_context[-5:]:  # Last 5 messages for context window efficiency
                role = msg.get('role', 'unknown')
                content = msg.get('content', '').strip()
                if content:
                    context_parts.append(f"{role.title()}: {content}")
            context_parts.append("")  # Spacing

        # PRIORITY 2: Relevant Memory Chunks (Medium Priority)
        memory_chunks = memory_results.get('results', [])
        if memory_chunks:
            # FIXED: Lower threshold from 0.7 to 0.4 and include more chunks
            relevant_chunks = [chunk for chunk in memory_chunks if chunk.get('score', 0) > 0.4][:5]  # Top 5 with score > 0.4

            # If no chunks meet threshold, include the best available chunk
            if not relevant_chunks and memory_chunks:
                relevant_chunks = memory_chunks[:3]  # Fallback to top 3 regardless of score
                logger.info(f"Using fallback memory chunks (best available scores: {[c.get('score', 0) for c in relevant_chunks]})")

            if relevant_chunks:
                context_parts.append("=== RELEVANT KNOWLEDGE FROM MEMORY ===")
                context_parts.append("Use this information to enhance your response when relevant:")

                for i, chunk in enumerate(relevant_chunks, 1):
                    text = chunk.get('text', '').strip()
                    source = chunk.get('source', 'Unknown')
                    score = chunk.get('score', 0)

                    if text and len(text) > 20:  # Lower threshold for substantial chunks
                        context_parts.append(f"[{source} - Relevance: {score:.2f}]")
                        context_parts.append(f"{text[:500]}..." if len(text) > 500 else text)
                        context_parts.append("")

        # PRIORITY 3: Conversation Summaries (Lower Priority)
        conversations = memory_results.get('conversations', [])
        if conversations and not conversation_context:  # Only if no recent context
            context_parts.append("=== CONVERSATION SUMMARIES ===")
            for conv in conversations[:2]:  # Limit summaries
                preview = conv.get('preview', '').strip()
                conv_id = conv.get('conversation_id', 'Unknown')

                if preview:
                    context_parts.append(f"Previous conversation ({conv_id}): {preview}")

        # Combine context
        context = "\n".join(context_parts) if context_parts else "No relevant context available."

        # Enhanced prompt structure for memory utilization
        prompt_parts = []

        # System instructions
        prompt_parts.append(f"System: {system_prompt}")

        # Structured context block
        prompt_parts.append(f"Context Information:\n{context}")

        # Clear user query
        prompt_parts.append(f"Current User Query: {query}")

        # CONVERSATIONAL RESPONSE GUIDELINES
        prompt_parts.append("\n".join([
            "Response Guidelines:",
            "- Give direct, natural answers like a helpful friend would",
            "- Weave in relevant information from memory seamlessly and naturally",
            "- Keep your tone warm, engaging, and conversational",
            "- Don't show your analytical process or reasoning steps",
            "- Be informative but not overwhelming - focus on what matters to the user"
        ]))

        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)



    def build_simple_prompt(self, query: str, memory_results: Dict[str, Any]) -> str:
        """
        Build a simple, reliable prompt for direct response generation

        Args:
            query: User query
            memory_results: Memory search results

        Returns:
            Simple prompt string
        """
        prompt_parts = []

        # Simple system prompt
        prompt_parts.append("You are Frankenstino AI, a helpful conversational AI with memory capabilities.")

        # Add conversation context if available
        conversation_context = memory_results.get('conversation_context', [])
        if conversation_context:
            prompt_parts.append("\nRecent conversation:")
            for msg in conversation_context[-3:]:  # Last 3 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '').strip()
                if content:
                    prompt_parts.append(f"{role.title()}: {content}")

        # Add memory context if available
        memory_chunks = memory_results.get('results', [])
        if memory_chunks:
            prompt_parts.append("\nRelevant information:")
            for chunk in memory_chunks[:2]:  # Top 2 chunks
                text = chunk.get('text', '').strip()
                if text and len(text) > 10:
                    prompt_parts.append(f"• {text[:200]}..." if len(text) > 200 else f"• {text}")

        # User query
        prompt_parts.append(f"\nUser: {query}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def generate_response(self, query: str, memory_results: Dict[str, Any],
                         temperature: float = 0.7, stream: bool = False) -> Dict[str, Any]:
        """
        SIMPLIFIED RESPONSE GENERATION: Direct query → response flow
        No complex enhancement loops - just generate a good response using available context

        Args:
            query: User query
            memory_results: Results from memory search
            temperature: Generation temperature
            stream: Whether to stream the response

        Returns:
            Response dictionary with answer and metadata
        """
        memory_chunks = memory_results.get('results', [])
        conversation_context = memory_results.get('conversation_context', [])

        logger.info(f"Generating simplified response with {len(memory_chunks)} memory chunks")

        # Build simple prompt with available context
        prompt = self.build_simple_prompt(query, memory_results)

        try:
            # Generate direct response
            response_text = self.model_loader.generate_text(
                prompt,
                max_tokens=self.response_max_tokens,
                temperature=temperature,
                stream=False  # Keep simple, no streaming complexity
            )

            # Clean response
            response_text = response_text.strip() if response_text else ""

            # Simple fallback for empty responses
            if not response_text or len(response_text) < 10:
                response_text = f"Hello! I'm Frankenstino AI, a conversational AI with memory capabilities. I understand you're asking about '{query}'. How can I help you today?"

            # Return simplified response data
            return {
                "query": query,
                "answer": response_text,
                "memory_chunks_used": len(memory_chunks),
                "conversation_messages_used": len(conversation_context),
                "learning_enabled": True
            }

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "query": query,
                "answer": f"I apologize, but I encountered an error processing your query. I'm Frankenstino AI and I'm here to help!",
                "error": str(e),
                "memory_chunks_used": 0,
                "learning_enabled": True
            }

    def stream_response(self, query: str, memory_results: Dict[str, Any],
                       temperature: float = 0.7) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a response token by token

        Args:
            query: User query
            memory_results: Memory search results
            temperature: Generation temperature

        Yields:
            Response chunks with token data
        """
        prompt = self.build_prompt(query, memory_results)

        try:
            response_generator = self.model_loader.generate_text(
                prompt,
                max_tokens=self.response_max_tokens,
                temperature=temperature,
                stream=True
            )

            full_response = ""
            for chunk in response_generator:
                # Handle llama-cpp-python streaming format
                # It returns simple strings or dicts, not OpenAI format
                if isinstance(chunk, str):
                    token = chunk
                elif isinstance(chunk, dict):
                    # Try different possible formats
                    if "choices" in chunk and chunk["choices"]:
                        token = chunk["choices"][0].get("text", "")
                    elif "text" in chunk:
                        token = chunk["text"]
                    else:
                        # Fallback: convert dict to string representation
                        token = str(chunk)
                else:
                    # Handle any other format by converting to string
                    token = str(chunk)

                full_response += token

                # Check if this is the final chunk
                finished = False
                if isinstance(chunk, dict):
                    finished = chunk.get("choices", [{}])[0].get("finish_reason") is not None

                yield {
                    "token": token,
                    "full_response": full_response,
                    "finished": finished
                }

        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            yield {
                "token": f"Error: {str(e)}",
                "full_response": f"I apologize, but I encountered an error: {str(e)}",
                "finished": True,
                "error": True
            }



    def estimate_cost(self, prompt: str, response_length: int = 256) -> Dict[str, Any]:
        """
        Estimate computational cost of a query

        Args:
            prompt: Input prompt
            response_length: Expected response length in tokens

        Returns:
            Cost estimation data
        """
        prompt_tokens = self.model_loader.estimate_tokens(prompt)
        total_tokens = prompt_tokens + response_length

        # Rough CPU time estimation (very approximate)
        # Based on Gemma-3-1B performance on typical hardware
        estimated_time_seconds = total_tokens * 0.01  # ~10ms per token

        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_length,
            "total_tokens": total_tokens,
            "estimated_time_seconds": estimated_time_seconds,
            "model_loaded": self.model_loader.is_loaded
        }
