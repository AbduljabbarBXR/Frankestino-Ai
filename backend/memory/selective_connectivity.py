"""
Selective Neural Connectivity for Frankenstino AI
Implements intelligent word association strategies to replace full connectivity
"""

import logging
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import re
import numpy as np

logger = logging.getLogger(__name__)


class ConnectivityStrategy(Enum):
    """Connection strategies for neural mesh learning"""
    FULL_CONNECTIVITY = "full"  # Current approach (for comparison)
    SLIDING_WINDOW = "window"  # Proximity-based connections
    SYNTAX_AWARE = "syntax"    # Dependency parsing (future)
    ATTENTION_BASED = "attention"  # Transformer attention patterns (future)


class SelectiveConnectivity:
    """
    Intelligent connectivity strategies for word association learning
    Replaces the inefficient full-connectivity approach with selective, meaningful connections
    """

    def __init__(self, strategy: ConnectivityStrategy = ConnectivityStrategy.SLIDING_WINDOW,
                 window_size: int = 3, min_weight: float = 0.1):
        """
        Initialize selective connectivity

        Args:
            strategy: Connection strategy to use
            window_size: Size of sliding window for proximity connections
            min_weight: Minimum connection weight
        """
        self.strategy = strategy
        self.window_size = window_size
        self.min_weight = min_weight

        # Strategy-specific parameters
        self.attenuation_factor = 0.8  # How much weight decreases with distance
        self.max_connections_per_word = 50  # Limit connections to prevent explosion

        # Syntax parsing (for future SYNTAX_AWARE strategy)
        self.nlp_parser = None
        if strategy == ConnectivityStrategy.SYNTAX_AWARE:
            self._init_syntax_parser()

        logger.info(f"Initialized selective connectivity with strategy: {strategy.value}")

    def _init_syntax_parser(self):
        """Initialize syntax parsing components (spaCy/Stanza)"""
        try:
            import spacy
            self.nlp_parser = spacy.load("en_core_web_sm")
            logger.info("Syntax parser initialized")
        except ImportError:
            logger.warning("spaCy not available, falling back to basic parsing")
            self.nlp_parser = None

    def connect_words_in_text(self, words: List[str], context: str = None) -> List[Tuple[str, str, float]]:
        """
        Generate selective connections between words based on chosen strategy

        Args:
            words: List of words to connect
            context: Optional context information

        Returns:
            List of (word_a, word_b, weight) tuples
        """
        if not words or len(words) < 2:
            return []

        if self.strategy == ConnectivityStrategy.FULL_CONNECTIVITY:
            return self._full_connectivity(words)
        elif self.strategy == ConnectivityStrategy.SLIDING_WINDOW:
            return self._sliding_window_connectivity(words)
        elif self.strategy == ConnectivityStrategy.SYNTAX_AWARE:
            return self._syntax_aware_connectivity(words)
        elif self.strategy == ConnectivityStrategy.ATTENTION_BASED:
            return self._attention_based_connectivity(words)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}")
            return []

    def _full_connectivity(self, words: List[str]) -> List[Tuple[str, str, float]]:
        """
        Original approach: connect all word pairs
        Used for comparison and fallback
        """
        connections = []
        for i, word_a in enumerate(words):
            for j, word_b in enumerate(words[i+1:], i+1):
                connections.append((word_a, word_b, self.min_weight))

        logger.debug(f"Full connectivity: {len(connections)} connections for {len(words)} words")
        return connections

    def _sliding_window_connectivity(self, words: List[str]) -> List[Tuple[str, str, float]]:
        """
        Connect words within sliding window with distance-based attenuation
        Much more efficient than full connectivity
        """
        connections = []
        connection_counts = {}  # Track connections per word to prevent explosion

        for i, word_a in enumerate(words):
            if word_a not in connection_counts:
                connection_counts[word_a] = 0

            # Connect to words within window
            start_idx = max(0, i - self.window_size)
            end_idx = min(len(words), i + self.window_size + 1)

            for j in range(start_idx, end_idx):
                if i == j:
                    continue

                word_b = words[j]
                if word_b not in connection_counts:
                    connection_counts[word_b] = 0

                # Check connection limits
                if (connection_counts[word_a] >= self.max_connections_per_word or
                    connection_counts[word_b] >= self.max_connections_per_word):
                    continue

                # Distance-based weight with attenuation
                distance = abs(i - j)
                weight = max(self.min_weight, self.attenuation_factor ** distance)

                connections.append((word_a, word_b, weight))
                connection_counts[word_a] += 1
                connection_counts[word_b] += 1

        logger.debug(f"Sliding window: {len(connections)} connections for {len(words)} words")
        return connections

    def _syntax_aware_connectivity(self, words: List[str]) -> List[Tuple[str, str, float]]:
        """
        Connect words based on syntactic dependencies
        Requires spaCy for proper parsing
        """
        if not self.nlp_parser:
            logger.warning("Syntax parser not available, falling back to sliding window")
            return self._sliding_window_connectivity(words)

        try:
            # Join words back to text for parsing
            text = " ".join(words)

            # Parse dependencies
            doc = self.nlp_parser(text)
            connections = []

            for token in doc:
                # Connect to syntactic head
                if token.head != token:  # Not root
                    head_word = token.head.text.lower()
                    dep_word = token.text.lower()

                    # Weight based on dependency type
                    weight = self._get_dependency_weight(token.dep_)
                    if weight >= self.min_weight:
                        connections.append((dep_word, head_word, weight))

                # Connect to children (limited to prevent explosion)
                children = list(token.children)[:3]  # Limit children connections
                for child in children:
                    child_word = child.text.lower()
                    weight = self._get_dependency_weight(child.dep_)
                    if weight >= self.min_weight:
                        connections.append((token.text.lower(), child_word, weight))

            logger.debug(f"Syntax-aware: {len(connections)} connections for {len(words)} words")
            return connections

        except Exception as e:
            logger.error(f"Syntax parsing failed: {e}")
            return self._sliding_window_connectivity(words)

    def _attention_based_connectivity(self, words: List[str]) -> List[Tuple[str, str, float]]:
        """
        Use attention patterns for connections (placeholder for future implementation)
        Would use transformer attention weights to determine connection strengths
        """
        # For now, fall back to sliding window
        logger.info("Attention-based connectivity not yet implemented, using sliding window")
        return self._sliding_window_connectivity(words)

    def _get_dependency_weight(self, dep_type: str) -> float:
        """
        Get connection weight based on dependency type
        More important dependencies get higher weights
        """
        weight_map = {
            'nsubj': 0.9,      # Subject
            'dobj': 0.9,       # Direct object
            'iobj': 0.8,       # Indirect object
            'nsubjpass': 0.8,  # Passive subject
            'ccomp': 0.7,      # Clausal complement
            'xcomp': 0.7,      # Open clausal complement
            'amod': 0.6,       # Adjectival modifier
            'compound': 0.6,   # Compound
            'prep': 0.5,       # Prepositional modifier
            'pobj': 0.5,       # Object of preposition
            'advmod': 0.4,     # Adverbial modifier
            'det': 0.3,        # Determiner
            'aux': 0.3,        # Auxiliary
            'punct': 0.1,      # Punctuation
        }

        return weight_map.get(dep_type, self.min_weight)

    def get_connection_stats(self, words: List[str]) -> Dict[str, Any]:
        """
        Get statistics about connections that would be created
        Useful for analysis and optimization
        """
        connections = self.connect_words_in_text(words)

        if not connections:
            return {"total_connections": 0, "avg_weight": 0, "unique_words": len(set(words))}

        weights = [w for _, _, w in connections]
        unique_words = set()
        for word_a, word_b, _ in connections:
            unique_words.add(word_a)
            unique_words.add(word_b)

        return {
            "total_connections": len(connections),
            "avg_weight": np.mean(weights),
            "max_weight": max(weights),
            "min_weight": min(weights),
            "unique_words": len(unique_words),
            "connections_per_word": len(connections) / len(unique_words) if unique_words else 0,
            "strategy": self.strategy.value
        }

    def optimize_parameters(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        Optimize connectivity parameters based on sample texts
        Uses simple heuristics to find good defaults
        """
        if not sample_texts:
            return {}

        # Analyze different parameter combinations
        best_params = {}
        best_score = 0

        for window_size in [2, 3, 4, 5]:
            for attenuation in [0.7, 0.8, 0.9]:
                self.window_size = window_size
                self.attenuation_factor = attenuation

                total_connections = 0
                total_words = 0

                for text in sample_texts[:5]:  # Limit sample size
                    words = self._tokenize_sample_text(text)
                    stats = self.get_connection_stats(words)
                    total_connections += stats["total_connections"]
                    total_words += stats["unique_words"]

                # Score: balance between connectivity and efficiency
                avg_connections_per_word = total_connections / max(total_words, 1)
                score = 1.0 / (1.0 + avg_connections_per_word)  # Lower connections = higher score

                if score > best_score:
                    best_score = score
                    best_params = {
                        "window_size": window_size,
                        "attenuation_factor": attenuation,
                        "avg_connections_per_word": avg_connections_per_word
                    }

        # Restore best parameters
        if best_params:
            self.window_size = best_params["window_size"]
            self.attenuation_factor = best_params["attenuation_factor"]

        logger.info(f"Optimized parameters: {best_params}")
        return best_params

    def _tokenize_sample_text(self, text: str) -> List[str]:
        """Simple tokenization for parameter optimization"""
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter short words and stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [word for word in words if len(word) > 1 and word not in stop_words]