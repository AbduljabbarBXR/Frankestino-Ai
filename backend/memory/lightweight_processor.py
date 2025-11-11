"""
Lightweight Memory Processor - CPU-optimized for memory curation tasks
Replaces heavy transformer with efficient summarization and processing
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import re

logger = logging.getLogger(__name__)

class LightweightMemoryProcessor:
    """
    CPU-optimized processor for memory summarization and curation
    Uses DistilBERT (67M parameters) for efficient processing
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 cache_dir: str = "models/lightweight"):
        """
        Initialize lightweight processor

        Args:
            model_name: HuggingFace model name
            cache_dir: Local cache directory
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model components
        self.model = None
        self.tokenizer = None
        self.is_initialized = False

        # Processing stats
        self.stats = {
            'summaries_processed': 0,
            'relationships_extracted': 0,
            'avg_processing_time': 0,
            'memory_usage_mb': 0
        }

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize DistilBERT model and tokenizer"""
        try:
            logger.info(f"Initializing lightweight processor with {self.model_name}")

            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )

            # Load model
            self.model = DistilBertModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )

            # Set to evaluation mode
            self.model.eval()

            # Move to CPU explicitly
            self.model.to('cpu')

            self.is_initialized = True
            logger.info("Lightweight processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize lightweight processor: {e}")
            self.is_initialized = False

    def is_ready(self) -> bool:
        """Check if processor is ready"""
        return self.is_initialized and self.model is not None

    def summarize_text(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """
        Generate summary using lightweight model

        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens

        Returns:
            Summary dictionary with metadata
        """
        if not self.is_ready():
            return self._rule_based_summary(text)

        try:
            start_time = time.time()

            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]

            # Simple extractive summarization based on sentence embeddings
            sentences = text.split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]

            if len(sentences) <= 3:
                summary = text
            else:
                # Score sentences by similarity to document embedding
                sentence_embeddings = []
                for sentence in sentences[:10]:  # Limit to first 10 sentences
                    sent_inputs = self.tokenizer(
                        sentence,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                        padding=True
                    )
                    with torch.no_grad():
                        sent_outputs = self.model(**sent_inputs)
                        sent_embedding = sent_outputs.last_hidden_state.mean(dim=1)
                        sentence_embeddings.append(sent_embedding)

                # Calculate similarities
                similarities = []
                for sent_emb in sentence_embeddings:
                    sim = torch.cosine_similarity(embeddings, sent_emb, dim=1).item()
                    similarities.append(sim)

                # Select top 3 sentences
                top_indices = sorted(range(len(similarities)),
                                   key=lambda i: similarities[i],
                                   reverse=True)[:3]
                top_sentences = [sentences[i] for i in sorted(top_indices)]
                summary = ' '.join(top_sentences)

            processing_time = time.time() - start_time

            # Update stats
            self.stats['summaries_processed'] += 1
            self._update_avg_time(processing_time)

            return {
                'summary': summary,
                'method': 'lightweight_model',
                'confidence': 0.7,
                'processing_time': processing_time,
                'original_length': len(text),
                'summary_length': len(summary)
            }

        except Exception as e:
            logger.error(f"Lightweight summarization failed: {e}")
            return self._rule_based_summary(text)

    def _rule_based_summary(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based summarization"""
        sentences = text.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

        if len(sentences) <= 2:
            summary = text
        else:
            # Take first and last sentences
            summary = sentences[0] + ' ' + sentences[-1]

        return {
            'summary': summary,
            'method': 'rule_based_fallback',
            'confidence': 0.5,
            'processing_time': 0.01,
            'original_length': len(text),
            'summary_length': len(summary)
        }

    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract semantic relationships using lightweight processing

        Args:
            text: Input text

        Returns:
            List of relationship dictionaries
        """
        if not self.is_ready():
            return []

        try:
            # Simple rule-based relationship extraction
            # Look for common patterns
            relationships = []

            # Geographic patterns
            geo_patterns = [
                (r'([A-Z][a-z]+) is the capital of ([A-Z][a-z]+)', 'capital_of'),
                (r'([A-Z][a-z]+) is located in ([A-Z][a-z]+)', 'located_in'),
                (r'([A-Z][a-z]+) is in ([A-Z][a-z]+)', 'located_in')
            ]

            for pattern, rel_type in geo_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    entity1, entity2 = match
                    relationships.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'relationship': rel_type,
                        'relationship_type': 'geographic',
                        'confidence': 0.8,
                        'source': 'lightweight_processor'
                    })

            # Causal patterns
            causal_patterns = [
                (r'([A-Z][a-z]+) causes ([a-z]+)', 'causes'),
                (r'([A-Z][a-z]+) leads to ([a-z]+)', 'leads_to'),
                (r'because of ([A-Z][a-z]+)', 'caused_by')
            ]

            for pattern, rel_type in causal_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        entity1, entity2 = match
                    else:
                        entity1, entity2 = match, "effect"
                    relationships.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'relationship': rel_type,
                        'relationship_type': 'causal',
                        'confidence': 0.7,
                        'source': 'lightweight_processor'
                    })

            self.stats['relationships_extracted'] += len(relationships)

            return relationships

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

    def assess_quality(self, text: str) -> Dict[str, Any]:
        """
        Assess text quality using lightweight heuristics

        Args:
            text: Input text

        Returns:
            Quality assessment
        """
        try:
            # Quality heuristics
            word_count = len(text.split())
            sentence_count = len(text.split('.'))
            avg_sentence_length = word_count / max(sentence_count, 1)

            # Length score
            length_score = min(word_count / 100, 1.0)  # Prefer substantial content

            # Readability score (simple heuristic)
            readability_score = 1.0 if 10 <= avg_sentence_length <= 25 else 0.7

            # Combine scores
            quality_score = (length_score + readability_score) / 2

            return {
                'quality_score': quality_score,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'method': 'lightweight_heuristics'
            }

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                'quality_score': 0.5,
                'method': 'error_fallback'
            }

    def _update_avg_time(self, processing_time: float):
        """Update average processing time"""
        count = self.stats['summaries_processed']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (current_avg * (count - 1) + processing_time) / count

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            **self.stats,
            'is_ready': self.is_ready(),
            'model_name': self.model_name,
            'memory_usage_mb': self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage"""
        if not self.is_ready():
            return 0

        # Rough estimation for DistilBERT
        # Model parameters + KV cache
        param_count = sum(p.numel() for p in self.model.parameters())
        param_memory = param_count * 4 / (1024 * 1024)  # 4 bytes per float32

        return param_memory
