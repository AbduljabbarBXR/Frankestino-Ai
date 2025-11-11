"""
Memory Transformer Reasoning Layer
Implements brain-inspired cognitive architecture for Frankenstino AI
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence awareness"""

    def __init__(self, embed_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Precompute RoPE parameters
        self._build_rope_cache()

    def _build_rope_cache(self):
        """Build the RoPE rotation matrices"""
        # RoPE frequencies (standard implementation)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim))

        # Position indices
        positions = torch.arange(self.max_seq_len).float()

        # Compute angles
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)

        # Create rotation matrices
        self.cos_cache = angles.cos()
        self.sin_cache = angles.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input embeddings

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]

        Returns:
            Rotated embeddings with positional information
        """
        batch_size, seq_len, embed_dim = x.shape

        # Get RoPE parameters for this sequence
        cos = self.cos_cache[:seq_len].to(x.device)
        sin = self.sin_cache[:seq_len].to(x.device)

        # Apply rotation to even/odd dimensions
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # RoPE rotation: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_odd*cos + x_even*sin]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_odd * cos + x_even * sin

        # Interleave back
        rotated = torch.zeros_like(x)
        rotated[..., ::2] = rotated_even
        rotated[..., 1::2] = rotated_odd

        return rotated


class SelfAttention(nn.Module):
    """Multi-head self-attention layer"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Self-attention forward pass

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Attention mask [batch_size, seq_len, seq_len]

        Returns:
            Attention output [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """Feed-forward network for concept shaping"""

    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 4

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward forward pass"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Complete transformer block with attention, feed-forward, and normalization"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Self-attention layer
        self.self_attn = SelfAttention(embed_dim, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, dropout=dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transformer block forward pass

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Attention mask

        Returns:
            Processed tensor [batch_size, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        attn_output = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class MemoryTransformer(nn.Module):
    """
    Memory Transformer - Brain-inspired cognitive architecture
    Combines embedding, positional encoding, transformer layers, and tied output projection
    """

    def __init__(self, embed_dim: int = 384, num_layers: int = 6, num_heads: int = 8,
                 vocab_size: int = 32000, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embeddings (will be tied to output projection)
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding (RoPE)
        self.position_embeddings = RotaryPositionEmbedding(embed_dim, max_seq_len)

        # Transformer layers (the "thinking engine")
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Layer normalization
        self.final_norm = nn.LayerNorm(embed_dim)

        # Output projection with tied weights (decoder.weight = embedding.weight.T)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        # Tie weights: output_projection.weight = token_embeddings.weight.T
        self._tie_weights()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(f"MemoryTransformer initialized: {num_layers} layers, {embed_dim}D embeddings, vocab={vocab_size}")

    def _tie_weights(self):
        """Tie output projection weights to token embeddings (weight sharing)"""
        self.output_projection.weight = self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor, memory_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through memory transformer

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            memory_context: Additional memory embeddings [batch_size, memory_seq_len, embed_dim]

        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        embeddings = self.token_embeddings(input_ids)  # [batch_size, seq_len, embed_dim]

        # Add memory context if provided (concatenate along sequence dimension)
        if memory_context is not None:
            embeddings = torch.cat([memory_context, embeddings], dim=1)
            seq_len = embeddings.shape[1]

        # Apply positional encoding (RoPE)
        embeddings = self.position_embeddings(embeddings)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final layer normalization
        hidden_states = self.final_norm(hidden_states)

        # Output projection (tied weights give us language generation)
        logits = self.output_projection(hidden_states)

        return logits

    def generate(self, input_ids: torch.Tensor, memory_context: Optional[torch.Tensor] = None,
                max_new_tokens: int = 50, temperature: float = 0.7, top_k: int = 50) -> torch.Tensor:
        """
        Autoregressive text generation

        Args:
            input_ids: Starting token sequence [batch_size, seq_len]
            memory_context: Memory embeddings to condition generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated token sequence [batch_size, seq_len + generated]
        """
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()

            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = self(generated, memory_context)  # [batch_size, current_seq_len, vocab_size]

                # Focus on last token
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if EOS token (assuming 2 is EOS like many models)
                if next_token.item() == 2:
                    break

            return generated

    def get_memory_embedding(self, memory_texts: List[str], embedder) -> torch.Tensor:
        """
        Convert memory text chunks to embeddings for context

        Args:
            memory_texts: List of memory text chunks
            embedder: Sentence transformer embedder

        Returns:
            Memory embeddings [len(memory_texts), embed_dim]
        """
        if not memory_texts:
            return None

        # Use sentence transformer to get embeddings
        embeddings = embedder.encode_texts(memory_texts)
        return torch.tensor(embeddings, dtype=torch.float32)


class MemoryChatInterface:
    """
    High-level interface for memory-based conversations
    Integrates transformer reasoning with neural mesh memory
    """

    def __init__(self, transformer: MemoryTransformer, memory_manager, model_loader):
        self.transformer = transformer
        self.memory_manager = memory_manager
        self.model_loader = model_loader
        self.embedder = memory_manager.embedder

        # Personality parameters (output logic layer)
        self.personality = {
            'tone': 'analytical',  # analytical, creative, helpful
            'confidence_threshold': 0.6,
            'max_memory_chunks': 5,
            'temperature': 0.7
        }

    def chat(self, user_message: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a chat message using memory-informed LLM generation

        Args:
            user_message: User's input message
            conversation_history: Previous conversation turns

        Returns:
            Response with generated text and metadata
        """
        try:
            # 1. Retrieve relevant memory context
            memory_results = self.memory_manager.hybrid_search(
                user_message, max_results=self.personality['max_memory_chunks']
            )

            # 2. Build memory context for LLM prompt
            memory_texts = [result['text'] for result in memory_results.get('results', [])]
            memory_context = ""
            if memory_texts:
                memory_context = "\n\nRelevant memory context:\n" + "\n".join(f"- {text}" for text in memory_texts)

            # 3. Create LLM prompt with memory context
            system_prompt = f"""You are Frankenstino AI, a helpful assistant with access to memory.

{memory_context}

Based on the above memory context (if any), provide a helpful and relevant response to the user's message. If memory context is provided, incorporate relevant information naturally into your response."""

            full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"

            # 4. Generate response using LLM
            response_text = self.model_loader.generate_text(
                full_prompt,
                max_tokens=200,
                temperature=self.personality['temperature']
            )

            # Clean up response (remove any prompt echo)
            if response_text.startswith(full_prompt):
                response_text = response_text[len(full_prompt):].strip()
            if response_text.startswith("Assistant:"):
                response_text = response_text[10:].strip()

            # Apply personality filter
            response_text = self._apply_personality_filter(response_text)

            # 5. Store interaction in memory for learning (async)
            # Note: We don't await here to avoid blocking the response
            # The storage happens in the background
            import asyncio
            asyncio.create_task(self._store_interaction(user_message, response_text, memory_results))

            return {
                'response': response_text,
                'memory_chunks_used': len(memory_texts),
                'confidence': self._estimate_confidence(memory_results),
                'method': 'memory_llm',
                'personality': self.personality['tone']
            }

        except Exception as e:
            logger.error(f"Memory chat failed: {e}")
            return {
                'response': "I apologize, but I'm having trouble accessing my memory right now.",
                'error': str(e),
                'method': 'memory_error'
            }

    def _clean_generated_response(self, generated_text: str, user_message: str) -> str:
        """Clean up the generated response"""
        # Remove the original user message if it was echoed
        if generated_text.startswith(user_message):
            generated_text = generated_text[len(user_message):].strip()

        # Stop at sentence boundaries or reasonable length
        sentences = generated_text.split('.')
        if len(sentences) > 3:
            generated_text = '.'.join(sentences[:3]) + '.'

        return generated_text.strip()

    def _apply_personality_filter(self, response: str) -> str:
        """Apply personality-based filtering to response"""
        if self.personality['tone'] == 'analytical':
            # Add analytical markers
            if not any(word in response.lower() for word in ['analyze', 'consider', 'based on']):
                response = f"Based on my memory analysis: {response}"

        return response

    def _estimate_confidence(self, memory_results: Dict[str, Any]) -> float:
        """Estimate confidence in the response"""
        results = memory_results.get('results', [])
        if not results:
            return 0.3

        # Average score as confidence proxy
        scores = [r.get('score', 0.5) for r in results]
        avg_score = sum(scores) / len(scores)

        return min(avg_score, 1.0)

    async def _store_interaction(self, user_message: str, response: str, memory_results: Dict[str, Any]):
        """Store the interaction for learning and memory reinforcement"""
        try:
            # Create conversation data
            conversation_data = [
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': response}
            ]

            # Store in memory (async)
            success = await self.memory_manager.store_conversation(
                conversation_id=f"memory_chat_{int(torch.rand(1).item() * 1000000)}",
                messages=conversation_data,
                category="memory_chat"
            )

            if success:
                # Apply Hebbian learning to reinforce used memory connections
                used_chunks = memory_results.get('results', [])
                if len(used_chunks) > 1:
                    # Get node IDs for Hebbian reinforcement
                    activated_nodes = []
                    for chunk in used_chunks:
                        # Find corresponding mesh node
                        for node_id, node in self.memory_manager.neural_mesh.nodes.items():
                            if (node.metadata.get('text_preview', '').startswith(chunk['text'][:50])):
                                activated_nodes.append(node_id)
                                break

                    if len(activated_nodes) > 1:
                        self.memory_manager.neural_mesh.reinforce_hebbian_connections(
                            activated_nodes, reward=0.1
                        )

        except Exception as e:
            logger.debug(f"Failed to store interaction: {e}")
