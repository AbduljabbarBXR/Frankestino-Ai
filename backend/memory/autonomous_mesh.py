"""
Autonomous Word Association Network
Extends NeuralMesh for word-level autonomous learning and semantic association
"""
import logging
import asyncio
import threading
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import re
import time
import numpy as np

from .neural_mesh import NeuralMesh, MeshNode, MeshEdge
from .selective_connectivity import SelectiveConnectivity, ConnectivityStrategy
from ..ingestion.embedder import DocumentEmbedder
from ..utils.cache import get_cache

logger = logging.getLogger(__name__)


class WordNeuron(MeshNode):
    """Specialized neuron for word-level learning"""

    def __init__(self, word: str, **kwargs):
        super().__init__(**kwargs)
        self.word = word
        self.frequency = 0
        self.contexts = []  # List of context windows where word appears
        self.pos_tag = None  # Part of speech
        self.semantic_domain = None  # Domain/category (technical, medical, etc.)
        self.last_activated = 0.0
        self.activation_count = 0

    def activate(self, context: str = None):
        """Activate this word neuron"""
        self.last_activated = time.time()
        self.activation_count += 1
        if context:
            self.contexts.append(context)
            # Keep only recent contexts
            if len(self.contexts) > 10:
                self.contexts = self.contexts[-10:]


class AutonomousMesh(NeuralMesh):
    """
    Autonomous word association network that learns semantic relationships
    through continuous scanning and self-organization
    """

    def __init__(self, storage_path: Optional[str] = None,
                 connectivity_strategy: ConnectivityStrategy = ConnectivityStrategy.SLIDING_WINDOW):
        super().__init__(storage_path)
        self.word_neurons: Dict[str, str] = {}  # word -> neuron_id mapping
        self.neuron_words: Dict[str, str] = {}  # neuron_id -> word mapping

        # Selective connectivity system
        self.connectivity_strategy = connectivity_strategy
        from ..config import settings
        self.connectivity = SelectiveConnectivity(
            strategy=connectivity_strategy,
            window_size=settings.connectivity_window_size,
            min_weight=settings.connectivity_min_weight
        )

        # Scanning configuration
        self.scanning_active = False
        self.scan_interval = 1.0  # seconds between scans
        self.similarity_threshold = 0.75
        self.max_scan_batch = 100  # neurons to scan per batch

        # Learning parameters
        self.hebbian_boost = 0.1
        self.decay_rate = 0.99  # daily decay factor
        self.min_connection_weight = 0.05

        # Background scanning
        self.scan_thread = None
        self.scan_event = threading.Event()

        # Word processing
        self.embedder = DocumentEmbedder()
        self.cache = get_cache()

        # Statistics
        self.learning_stats = {
            "total_words_processed": 0,
            "connections_formed": 0,
            "scans_performed": 0,
            "predictions_made": 0,
            "uptime_seconds": 0
        }

        logger.info("Autonomous Mesh initialized")

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()

    def normalize_word(self, word: str) -> str:
        """Normalize word for consistent processing"""
        # Convert to lowercase
        word = word.lower()

        # Remove punctuation
        word = re.sub(r'[^\w\s]', '', word)

        # Basic stemming (remove common suffixes)
        if len(word) > 3:
            if word.endswith(('ing', 'ed', 'er', 'est', 'ly', 's')):
                # Simple stemming - could be enhanced with proper stemmer
                pass

        return word.strip()

    def create_word_neuron(self, word: str, context: str = None) -> str:
        """Create a neuron for a word if it doesn't exist"""
        normalized_word = self.normalize_word(word)

        if not normalized_word or len(normalized_word) < 2:
            return None

        # Check if word neuron already exists
        if normalized_word in self.word_neurons:
            neuron_id = self.word_neurons[normalized_word]
            # Activate existing neuron
            if neuron_id in self.nodes:
                word_neuron = self.nodes[neuron_id]
                if hasattr(word_neuron, 'activate'):
                    word_neuron.activate(context)
            return neuron_id

        # Create new word neuron
        neuron_id = f"word_{normalized_word}_{int(self._get_timestamp() * 1000)}"

        # Get word embedding
        embedding = self._get_word_embedding(normalized_word)

        # Create word neuron
        word_neuron = WordNeuron(
            word=normalized_word,
            id=neuron_id,
            content_hash=normalized_word,  # Use word as hash for uniqueness
            embedding=embedding,
            metadata={
                "type": "word",
                "word": normalized_word,
                "created_from_context": context
            }
        )

        # Add to mesh
        self.nodes[neuron_id] = word_neuron
        self.word_neurons[normalized_word] = neuron_id
        self.neuron_words[neuron_id] = normalized_word

        # Activate the new neuron
        word_neuron.activate(context)

        logger.debug(f"Created word neuron for '{normalized_word}' with ID {neuron_id}")
        return neuron_id

    def _get_word_embedding(self, word: str) -> Optional[List[float]]:
        """Get embedding for a word with caching"""
        cache_key = f"word_embedding_{word}"

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            # Get embedding from embedder
            embeddings = self.embedder.encode_texts([word])
            if embeddings is not None and len(embeddings) > 0:
                embedding = embeddings[0].tolist()  # Convert to list for JSON serialization
                # Cache the embedding
                self.cache.set(cache_key, embedding, ttl_seconds=3600)  # 1 hour
                return embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding for word '{word}': {e}")

        return None

    def process_text_for_learning(self, text: str, context: str = None) -> Dict[str, Any]:
        """Process text to create word neurons and learn associations"""
        if not text or not text.strip():
            return {"words_processed": 0, "neurons_created": 0}

        # Tokenize text into words
        words = self._tokenize_text(text)
        words_processed = 0
        neurons_created = 0

        # Create neurons for each word
        neuron_ids = []
        for word in words:
            neuron_id = self.create_word_neuron(word, context)
            if neuron_id:
                neuron_ids.append(neuron_id)
                words_processed += 1
                if neuron_id not in self.nodes:
                    neurons_created += 1

        # Learn associations between words in this text
        if len(neuron_ids) > 1:
            self._learn_text_associations(neuron_ids, words, context)

        # Update statistics
        self.learning_stats["total_words_processed"] += words_processed

        # Trigger autonomous scanning if active
        if self.scanning_active:
            asyncio.create_task(self.perform_semantic_scan())

        # Get connection stats for more accurate reporting
        connection_stats = self.connectivity.get_connection_stats(words)

        return {
            "words_processed": words_processed,
            "neurons_created": neurons_created,
            "neurons_activated": len(neuron_ids),
            "associations_learned": connection_stats.get("total_connections", 0),
            "connectivity_strategy": self.connectivity_strategy.value,
            "avg_connection_weight": connection_stats.get("avg_weight", 0)
        }

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenization - could be enhanced with proper NLP
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out very short words and common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
        return [word for word in words if len(word) > 1 and word not in stop_words]

    def _learn_text_associations(self, neuron_ids: List[str], words: List[str], context: str = None):
        """Learn selective associations between words using intelligent connectivity"""
        if len(neuron_ids) < 2 or len(words) < 2:
            return

        # Get selective connections based on chosen strategy
        connections = self.connectivity.connect_words_in_text(words, context)

        # Create connections based on selective strategy
        connections_created = 0
        for word_a, word_b, weight in connections:
            neuron_a_id = self.word_neurons.get(word_a)
            neuron_b_id = self.word_neurons.get(word_b)

            if neuron_a_id and neuron_b_id:
                # Scale weight by hebbian boost and apply minimum threshold
                final_weight = max(self.min_connection_weight, weight * self.hebbian_boost)
                self.reinforce_connection(neuron_a_id, neuron_b_id, final_weight)
                connections_created += 1

                # Add semantic context if provided
                if context:
                    edge_key = (neuron_a_id, neuron_b_id)
                    reverse_key = (neuron_b_id, neuron_a_id)

                    edge = self.edges.get(edge_key) or self.edges.get(reverse_key)
                    if edge and hasattr(edge, 'semantic_context'):
                        if not edge.semantic_context:
                            edge.semantic_context = {}
                        edge.semantic_context['selective_connection_context'] = context
                        edge.semantic_context['connectivity_strategy'] = self.connectivity_strategy.value

        logger.debug(f"Created {connections_created} selective connections from {len(words)} words")

    async def start_autonomous_scanning(self):
        """Start the continuous semantic scanning process"""
        if self.scanning_active:
            logger.warning("Autonomous scanning already active")
            return

        self.scanning_active = True
        logger.info("Starting autonomous semantic scanning")

        while self.scanning_active:
            try:
                await self.perform_semantic_scan()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Error in autonomous scanning: {e}")
                await asyncio.sleep(self.scan_interval)

    def stop_autonomous_scanning(self):
        """Stop the autonomous scanning process"""
        self.scanning_active = False
        logger.info("Stopped autonomous semantic scanning")

    async def perform_semantic_scan(self):
        """Perform one iteration of semantic scanning"""
        if not self.nodes:
            return

        # Get random sample of neurons to scan
        neuron_ids = list(self.nodes.keys())
        if len(neuron_ids) > self.max_scan_batch:
            neuron_ids = np.random.choice(neuron_ids, self.max_scan_batch, replace=False)

        connections_formed = 0

        # Compare each neuron with others
        for i, neuron_a_id in enumerate(neuron_ids):
            neuron_a = self.nodes.get(neuron_a_id)
            if not neuron_a or not hasattr(neuron_a, 'embedding') or not neuron_a.embedding:
                continue

            # Compare with subsequent neurons to avoid duplicates
            for neuron_b_id in neuron_ids[i+1:]:
                neuron_b = self.nodes.get(neuron_b_id)
                if not neuron_b or not hasattr(neuron_b, 'embedding') or not neuron_b.embedding:
                    continue

                # Calculate semantic similarity
                similarity = self._cosine_similarity(neuron_a.embedding, neuron_b.embedding)

                if similarity >= self.similarity_threshold:
                    # Create autonomous connection
                    self._create_autonomous_connection(neuron_a_id, neuron_b_id, similarity)
                    connections_formed += 1

        # Update statistics
        self.learning_stats["scans_performed"] += 1
        self.learning_stats["connections_formed"] += connections_formed

        if connections_formed > 0:
            logger.debug(f"Semantic scan formed {connections_formed} new connections")

    def _create_autonomous_connection(self, neuron_a_id: str, neuron_b_id: str, similarity: float):
        """Create an autonomous connection between two neurons"""
        # Determine connection type based on similarity
        if similarity > 0.9:
            connection_type = "semantic"
            relationship_label = "highly_similar"
        elif similarity > 0.8:
            connection_type = "semantic"
            relationship_label = "similar"
        else:
            connection_type = "associative"
            relationship_label = "related"

        # Create edge with semantic context
        semantic_context = {
            "autonomous_connection": True,
            "similarity_score": similarity,
            "discovered_at": self._get_timestamp()
        }

        self._add_edge(
            source_id=neuron_a_id,
            target_id=neuron_b_id,
            weight=similarity,
            connection_type=connection_type,
            relationship_label=relationship_label,
            confidence=similarity,
            semantic_context=semantic_context
        )

    def predict_associations(self, word: str, context: List[str] = None,
                           top_k: int = 10) -> List[Tuple[str, float]]:
        """Predict associated words for a given word"""
        normalized_word = self.normalize_word(word)

        if normalized_word not in self.word_neurons:
            return []

        neuron_id = self.word_neurons[normalized_word]
        associations = []

        # Get direct connections
        for neighbor_id in self.adjacency_list.get(neuron_id, []):
            edge = self._get_edge(neuron_id, neighbor_id)
            if edge:
                neighbor_word = self.neuron_words.get(neighbor_id)
                if neighbor_word:
                    associations.append((neighbor_word, edge.weight))

        # Sort by strength and return top_k
        associations.sort(key=lambda x: x[1], reverse=True)

        # Update statistics
        self.learning_stats["predictions_made"] += 1

        return associations[:top_k]

    def _get_edge(self, source_id: str, target_id: str) -> Optional[MeshEdge]:
        """Get edge between two nodes"""
        edge_key = (source_id, target_id)
        reverse_key = (target_id, source_id)

        return self.edges.get(edge_key) or self.edges.get(reverse_key)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        stats = self.learning_stats.copy()
        stats.update({
            "total_neurons": len(self.nodes),
            "total_connections": len(self.edges),
            "unique_words": len(self.word_neurons),
            "scanning_active": self.scanning_active,
            "similarity_threshold": self.similarity_threshold
        })
        return stats

    def get_network_visualization_data(self) -> Dict[str, Any]:
        """Get data for frontend visualization"""
        nodes = []
        edges = []

        # Convert neurons to visualization nodes
        for neuron_id, neuron in self.nodes.items():
            if hasattr(neuron, 'word'):
                # Word neuron
                nodes.append({
                    "id": neuron_id,
                    "label": neuron.word,
                    "type": "word",
                    "activation": getattr(neuron, 'activation_level', 0.0),
                    "frequency": getattr(neuron, 'frequency', 0)
                })
            else:
                # Regular memory node
                nodes.append({
                    "id": neuron_id,
                    "label": neuron_id[:20] + "..." if len(neuron_id) > 20 else neuron_id,
                    "type": "memory",
                    "activation": neuron.activation_level
                })

        # Convert edges to visualization edges
        for edge_key, edge in self.edges.items():
            edges.append({
                "from": edge.source_id,
                "to": edge.target_id,
                "weight": edge.weight,
                "type": edge.connection_type,
                "label": getattr(edge, 'relationship_label', '')
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.get_learning_stats()
        }

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_autonomous_scanning()
