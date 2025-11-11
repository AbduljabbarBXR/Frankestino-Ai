"""
Advanced Prompt Building for Memory-Augmented LLM
Handles different prompt strategies and templates
"""
import logging
from typing import List, Dict, Any, Optional
from ..config import settings

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Advanced prompt construction for different use cases"""

    def __init__(self):
        self.templates = {
            "qa_with_sources": self._qa_with_sources_template,
            "conversational": self._conversational_template,
            "technical": self._technical_template,
            "creative": self._creative_template
        }

    def build_qa_prompt(self, query: str, memory_chunks: List[Dict[str, Any]],
                       prompt_type: str = "qa_with_sources") -> str:
        """
        Build a question-answering prompt

        Args:
            query: User question
            memory_chunks: Relevant memory chunks
            prompt_type: Type of prompt template to use

        Returns:
            Formatted prompt
        """
        template_func = self.templates.get(prompt_type, self._qa_with_sources_template)
        return template_func(query, memory_chunks)

    def _qa_with_sources_template(self, query: str, memory_chunks: List[Dict[str, Any]]) -> str:
        """Standard Q&A template with source citations"""
        system_prompt = """You are Frankenstino AI, an intelligent assistant with access to a comprehensive knowledge base.

INSTRUCTIONS:
- Answer questions using the provided context when available
- Cite sources using [Source X] notation when referencing specific information
- If the context doesn't contain relevant information, clearly state this
- Be concise but comprehensive in your answers
- Maintain accuracy and avoid speculation"""

        context_parts = []
        for i, chunk in enumerate(memory_chunks, 1):
            text = chunk.get('text', '').strip()
            source = chunk.get('source', 'Unknown')
            score = chunk.get('score', 0)

            if text:
                context_parts.append(f"[Source {i}] {source} (relevance: {score:.3f}):\n{text}")

        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        return f"""{system_prompt}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    def _conversational_template(self, query: str, memory_chunks: List[Dict[str, Any]]) -> str:
        """Conversational template for natural dialogue"""
        system_prompt = """You are Frankenstino AI, a helpful and friendly assistant with access to personal knowledge.

Be conversational and natural in your responses. Use the provided context to inform your answers, but don't be overly formal. If you don't have relevant information, just say so casually."""

        context_parts = []
        for chunk in memory_chunks[:3]:  # Limit context for conversation
            text = chunk.get('text', '').strip()
            if text:
                context_parts.append(text)

        context = " ".join(context_parts) if context_parts else ""

        if context:
            return f"""{system_prompt}

Based on what I know: {context}

User: {query}
Frankenstino:"""
        else:
            return f"""{system_prompt}

User: {query}
Frankenstino:"""

    def _technical_template(self, query: str, memory_chunks: List[Dict[str, Any]]) -> str:
        """Technical template for code and technical questions"""
        system_prompt = """You are Frankenstino AI, a technical assistant specializing in programming and technical documentation.

Provide accurate, technical answers with code examples when appropriate. Cite sources for technical specifications."""

        context_parts = []
        for i, chunk in enumerate(memory_chunks, 1):
            text = chunk.get('text', '').strip()
            source = chunk.get('source', 'Unknown')

            if text:
                context_parts.append(f"From {source}:\n{text}")

        context = "\n\n".join(context_parts) if context_parts else "No technical context available."

        return f"""{system_prompt}

TECHNICAL CONTEXT:
{context}

TECHNICAL QUESTION: {query}

TECHNICAL ANSWER:"""

    def _creative_template(self, query: str, memory_chunks: List[Dict[str, Any]]) -> str:
        """Creative template for open-ended questions"""
        system_prompt = """You are Frankenstino AI, a creative and insightful assistant.

Draw upon the available knowledge to provide thoughtful, creative responses. Feel free to make connections and provide unique perspectives."""

        context_parts = []
        for chunk in memory_chunks[:5]:  # More context for creative responses
            text = chunk.get('text', '').strip()
            if text:
                context_parts.append(text)

        context = " ".join(context_parts) if context_parts else ""

        return f"""{system_prompt}

Drawing from available knowledge: {context}

Creative query: {query}

Insightful response:"""

    def truncate_context(self, memory_chunks: List[Dict[str, Any]],
                        max_tokens: int = 1500) -> List[Dict[str, Any]]:
        """
        Truncate memory chunks to fit within token limit

        Args:
            memory_chunks: Memory chunks to truncate
            max_tokens: Maximum token count for context

        Returns:
            Truncated list of chunks
        """
        # Simple truncation by chunk count (can be improved with token counting)
        max_chunks = min(len(memory_chunks), max_tokens // 100)  # Rough estimate
        return memory_chunks[:max_chunks]

    def format_memory_for_prompt(self, memory_chunks: List[Dict[str, Any]],
                                format_type: str = "numbered") -> str:
        """
        Format memory chunks for inclusion in prompts

        Args:
            memory_chunks: Memory chunks to format
            format_type: Formatting style ("numbered", "bulleted", "inline")

        Returns:
            Formatted context string
        """
        if not memory_chunks:
            return "No relevant context found."

        if format_type == "numbered":
            formatted = []
            for i, chunk in enumerate(memory_chunks, 1):
                text = chunk.get('text', '').strip()
                source = chunk.get('source', 'Unknown')
                if text:
                    formatted.append(f"[{i}] {source}: {text}")
            return "\n".join(formatted)

        elif format_type == "bulleted":
            formatted = []
            for chunk in memory_chunks:
                text = chunk.get('text', '').strip()
                source = chunk.get('source', 'Unknown')
                if text:
                    formatted.append(f"• {source}: {text}")
            return "\n".join(formatted)

        elif format_type == "inline":
            texts = [chunk.get('text', '').strip() for chunk in memory_chunks if chunk.get('text', '').strip()]
            return " ".join(texts)

        else:
            # Default to numbered
            return self.format_memory_for_prompt(memory_chunks, "numbered")

    def estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt"""
        # Rough estimation: ~4 characters per token
        return len(prompt) // 4

    def optimize_prompt(self, query: str, memory_chunks: List[Dict[str, Any]],
                       max_tokens: int = 2000) -> str:
        """
        Optimize prompt by selecting most relevant chunks and fitting within token limit

        Args:
            query: User query
            memory_chunks: Available memory chunks
            max_tokens: Maximum total prompt tokens

        Returns:
            Optimized prompt
        """
        # Sort by relevance score
        sorted_chunks = sorted(memory_chunks, key=lambda x: x.get('score', 0), reverse=True)

        # Reserve tokens for system prompt and query
        reserved_tokens = 500
        available_tokens = max_tokens - reserved_tokens

        # Select chunks that fit
        selected_chunks = []
        total_tokens = 0

        for chunk in sorted_chunks:
            chunk_text = chunk.get('text', '')
            chunk_tokens = len(chunk_text) // 4

            if total_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break

        # Build optimized prompt
        return self.build_qa_prompt(query, selected_chunks)
