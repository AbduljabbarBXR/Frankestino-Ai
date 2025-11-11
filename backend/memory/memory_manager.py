"""
Unified Memory Manager
Combines hierarchical tree, neural mesh, and vector database with brain-inspired tiered storage
"""
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import numpy as np
import time
import json
from datetime import datetime, timedelta

from .hierarchy import HierarchicalMemory, MemoryNode
from .vector_store import OptimizedVectorStore as VectorStore
from .neural_mesh import NeuralMesh, MeshNode
from .autonomous_mesh import AutonomousMesh, WordNeuron
from .scanning_engine import SemanticScanner
from .connection_manager import ConnectionManager
from .association_engine import AssociationEngine
from ..ingestion.document_loader import DocumentProcessor, Document
from ..ingestion.embedder import DocumentEmbedder
from ..ingestion.text_processor import TextProcessor
from ..utils.cache import get_cache, get_memory_monitor
from ..config import settings
from .transformer_reasoning import MemoryTransformer, MemoryChatInterface

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified interface for the hybrid memory system with brain-inspired tiered storage"""

    # Memory tier definitions (brain-inspired)
    TIERS = {
        'active': {  # Tier 1: Working memory (fast access, recent)
            'max_age_hours': 24,  # Items accessed within 24 hours
            'min_access_count': 1,
            'compression': 'none',
            'storage_priority': 'high'
        },
        'short_term': {  # Tier 2: Recent memory (fast access, this week)
            'max_age_hours': 168,  # 7 days
            'min_access_count': 0,
            'compression': 'light',
            'storage_priority': 'medium'
        },
        'long_term': {  # Tier 3: Consolidated memory (compressed, this month+)
            'max_age_hours': 720,  # 30 days
            'min_access_count': 0,
            'compression': 'summary',
            'storage_priority': 'low'
        },
        'archived': {  # Tier 4: Deep memory (highly compressed, rare access)
            'max_age_hours': float('inf'),  # No age limit
            'min_access_count': 0,
            'compression': 'minimal',
            'storage_priority': 'minimal'
        }
    }

    def __init__(self, curator=None):
        """Initialize all memory components"""
        logger.info("Initializing Frankenstino Memory System...")

        # Optional curator for processing
        self.curator = curator

        # Initialize components
        self.hierarchy = HierarchicalMemory()
        self.vector_store = VectorStore(
            embedding_dim=settings.embedding_dim,
            index_type=settings.faiss_index_type,
            nlist=settings.nlist,
            use_gpu=settings.use_gpu_faiss
        )

        # Initialize autonomous word association network with selective connectivity
        from .selective_connectivity import ConnectivityStrategy

        # Map string config to enum (default to sliding window)
        strategy_map = {
            "sliding_window": ConnectivityStrategy.SLIDING_WINDOW,
            "syntax_aware": ConnectivityStrategy.SYNTAX_AWARE,
            "attention_based": ConnectivityStrategy.ATTENTION_BASED,
            "full": ConnectivityStrategy.FULL_CONNECTIVITY
        }

        connectivity_strategy = strategy_map.get(settings.connectivity_strategy, ConnectivityStrategy.SLIDING_WINDOW)

        self.neural_mesh = AutonomousMesh(
            connectivity_strategy=connectivity_strategy
        )
        self.scanner = SemanticScanner(self.neural_mesh)
        self.connection_manager = ConnectionManager(self.neural_mesh)
        self.association_engine = AssociationEngine(self.neural_mesh, self.connection_manager)

        self.document_processor = DocumentProcessor()
        self.embedder = DocumentEmbedder()
        self.text_processor = TextProcessor()

        # Memory tier tracking
        self.memory_tiers = {}  # node_id -> tier mapping
        self.access_patterns = {}  # node_id -> access history
        self.consolidation_cache = {}  # Track consolidated memories

        # Advanced caching for performance
        self.smart_cache = get_cache()  # Use global SmartCache instance
        self.search_cache = {}  # Local search results cache
        self.mesh_traversal_cache = {}  # Mesh traversal results cache
        self.embedding_cache = {}  # text -> embedding cache (keep separate for now)

        # Load existing tier data if available
        self._load_memory_tiers()

        # Initialize memory transformer for direct chat (lazy loading)
        self.memory_transformer = None
        self.memory_chat_interface = None

        logger.info("Memory system initialized successfully")

    def ingest_document(self, file_path: Path, category: str = None,
                       auto_categorize: bool = True) -> Dict[str, Any]:
        """
        Ingest a document into the memory system

        Args:
            file_path: Path to the document
            category: Target category (will create if doesn't exist)
            auto_categorize: Whether to auto-determine category

        Returns:
            Ingestion results
        """
        logger.info(f"Ingesting document: {file_path}")

        # Load document
        doc = self.document_processor.load_document(file_path)
        if not doc:
            raise ValueError(f"Failed to load document: {file_path}")

        # Determine category
        if not category and auto_categorize:
            category = self._auto_categorize_document(doc)
        elif not category:
            category = "general"

        # Ensure category exists in hierarchy
        if not self.hierarchy.find_node_by_name(category):
            self.hierarchy.add_node(category, f"Category for {category} documents")

        category_node = self.hierarchy.find_node_by_name(category)
        if not category_node:
            raise ValueError(f"Failed to create/find category: {category}")

        # Process document into chunks using semantic chunking
        if settings.semantic_chunking:
            text_chunks = self.text_processor.semantic_chunk(doc.content)
            logger.info(f"Using semantic chunking for document: {file_path}")
        else:
            text_chunks = self.text_processor.hybrid_chunk(doc.content)
            logger.info(f"Using hybrid chunking for document: {file_path}")

        # Convert to the expected format for embedding
        chunks = []
        for chunk_dict in text_chunks:
            chunks.append({
                'text': chunk_dict['text'],
                'metadata': {
                    **chunk_dict.get('metadata', {}),
                    **doc.metadata,
                    'document_id': doc.id
                }
            })

        logger.info(f"Document chunked into {len(chunks)} intelligent chunks (avg {sum(len(c['text']) for c in chunks)/len(chunks):.0f} chars each)")

        # Generate embeddings
        chunks_with_embeddings = self.embedder.encode_chunks(chunks)

        # Store in vector database
        vectors = np.array([chunk['embedding'] for chunk in chunks_with_embeddings])

        # Create proper metadata dictionaries for vector store
        metadata = [{
            'text': chunk['text'],
            'source': doc.id,
            'type': 'document',
            'chunk_index': i,
            'word_weights': chunk.get('word_weights', {}),
            'important_terms': chunk.get('important_terms', []),
            'filtered_text': chunk.get('filtered_text', chunk['text'])
        } for i, chunk in enumerate(chunks_with_embeddings)]

        sources = [doc.id] * len(chunks_with_embeddings)

        vector_ids = self.vector_store.add_vectors(vectors, metadata, sources)

        # Attach original document to hierarchy (not chunks)
        self.hierarchy.attach_document(category_node.id, doc.id)

        # Add chunks to neural mesh
        import hashlib
        for i, chunk in enumerate(chunks_with_embeddings):
            content_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
            mesh_node_id = f"{doc.id}_chunk_{i}"

            self.neural_mesh.add_node(
                node_id=mesh_node_id,
                content_hash=content_hash,
                embedding=chunk['embedding'],
                metadata={
                    'document_id': doc.id,
                    'chunk_index': i,
                    'vector_id': vector_ids[i],
                    'category': category,
                    'content_type': 'document',
                    'text_preview': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                }
            )

            # ===== AUTONOMOUS WORD ASSOCIATION LEARNING =====
            # Process chunk text to create word neurons and learn associations
            try:
                learning_result = self.neural_mesh.process_text_for_learning(
                    chunk['text'],
                    context=f"document_{doc.id}_chunk_{i}"
                )
                logger.debug(f"Autonomous learning from chunk {i}: {learning_result}")
            except Exception as e:
                logger.warning(f"Failed to process chunk {i} for autonomous learning: {e}")

        result = {
            'document_id': doc.id,
            'category': category,
            'chunks_created': len(chunks),
            'vectors_stored': len(vector_ids),
            'mesh_nodes_added': len(chunks_with_embeddings),
            'file_path': str(file_path)
        }

        logger.info(f"Document ingested successfully: {result}")
        return result

    def ingest_directory(self, directory_path: Path, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Ingest all documents from a directory

        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories

        Returns:
            List of ingestion results
        """
        logger.info(f"Ingesting directory: {directory_path}")

        documents = self.document_processor.load_directory(directory_path, recursive)
        results = []

        for doc in documents:
            try:
                # Determine category from file path
                category = self._extract_category_from_path(doc.source_path)
                result = self.ingest_document(doc.source_path, category, auto_categorize=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to ingest {doc.source_path}: {e}")
                results.append({
                    'document_id': doc.id,
                    'error': str(e),
                    'file_path': str(doc.source_path)
                })

        logger.info(f"Directory ingestion complete: {len(results)} documents processed")
        return results

    def search_memory(self, query: str, category: str = None,
                     max_results: int = 5) -> Dict[str, Any]:
        """
        Search the memory system with enhanced word-level weighting

        Args:
            query: Search query
            category: Limit search to category (optional)
            max_results: Maximum results to return

        Returns:
            Search results with hierarchical filtering and term boosting
        """
        logger.info(f"Searching memory for: '{query}' (category: {category})")

        # Extract important terms from query for boosting
        query_terms = self.text_processor.extract_important_terms(query, max_terms=5)
        logger.debug(f"Extracted query terms for boosting: {query_terms}")

        # Generate query embedding
        query_embedding = self.embedder.encode_texts([query])[0]

        # Get relevant document IDs from hierarchy
        relevant_doc_ids = set()
        if category:
            # Search specific category
            category_node = self.hierarchy.find_node_by_name(category)
            if category_node:
                relevant_doc_ids = self.hierarchy.get_all_documents(category_node.id)
        else:
            # Search all categories
            relevant_doc_ids = self.hierarchy.get_all_documents(self.hierarchy.root_id)

        # Vector search
        vector_results = self.vector_store.search(query_embedding, top_k=max_results * 3)  # Get more for reranking

        # Filter by relevant documents and apply term boosting
        filtered_results = []
        for result_tuple in vector_results:
            result_id, score, metadata = result_tuple
            doc_id = metadata['source']
            if doc_id in relevant_doc_ids or not category:
                # Apply term boosting based on query terms
                boosted_score = self._apply_term_boosting(score, metadata, query_terms)

                # Convert back to expected format for compatibility
                result_dict = {
                    'id': result_id,
                    'text': metadata['text'],
                    'source': doc_id,
                    'score': boosted_score,
                    'original_score': score,  # Keep original for comparison
                    **metadata  # Include all metadata
                }
                filtered_results.append(result_dict)

        # Re-sort by boosted score and limit results
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = filtered_results[:max_results]

        # Get category information
        categories = []
        if final_results:
            for result in final_results:
                doc_id = result['source']
                # Find which category this document belongs to
                category_name = self._find_document_category(doc_id)
                if category_name:
                    categories.append(category_name)

        return {
            'query': query,
            'results': final_results,
            'total_found': len(final_results),
            'categories': list(set(categories)),
            'search_type': 'word_weighted' if query_terms else 'standard',
            'query_terms': query_terms
        }

    def _apply_term_boosting(self, base_score: float, metadata: Dict[str, Any],
                           query_terms: List[str], boost_factor: float = 0.2) -> float:
        """
        Apply term-based boosting to search results

        Args:
            base_score: Original similarity score
            metadata: Chunk metadata containing text and word weights
            query_terms: Important terms extracted from query
            boost_factor: How much to boost per matching term

        Returns:
            Boosted score
        """
        if not query_terms:
            return base_score

        boosted_score = base_score
        chunk_text = metadata.get('text', '').lower()

        # Check for direct term matches in chunk text
        term_matches = 0
        for term in query_terms:
            if term.lower() in chunk_text:
                term_matches += 1

        # Apply boosting based on term matches
        if term_matches > 0:
            # Boost by boost_factor for each matching term, but cap at reasonable level
            boost_amount = min(term_matches * boost_factor, boost_factor * 3)
            boosted_score = base_score * (1 + boost_amount)

            logger.debug(f"Boosted score from {base_score:.3f} to {boosted_score:.3f} "
                        f"({term_matches} term matches)")

        # Additional boosting based on word importance scores
        word_weights = metadata.get('word_weights', {})
        if word_weights:
            # Calculate average importance of query terms in this chunk
            query_term_weights = []
            for term in query_terms:
                term_lower = term.lower()
                if term_lower in word_weights:
                    query_term_weights.append(word_weights[term_lower])

            if query_term_weights:
                avg_importance = sum(query_term_weights) / len(query_term_weights)
                # Additional small boost for high-importance terms
                importance_boost = min(avg_importance * 0.1, 0.05)  # Cap at 5% additional boost
                boosted_score *= (1 + importance_boost)

        return boosted_score

    def hybrid_search(self, query: str, category: str = None,
                     max_results: int = 5, use_mesh: bool = True) -> Dict[str, Any]:
        """
        Perform hybrid search combining hierarchy, vectors, and neural mesh

        Args:
            query: Search query
            category: Optional category filter
            max_results: Maximum results to return
            use_mesh: Whether to use neural mesh traversal

        Returns:
            Enhanced search results
        """
        logger.info(f"Performing hybrid search for: '{query}' (category: {category})")

        # Start with standard search
        base_results = self.search_memory(query, category, max_results * 2)

        if not base_results['results'] or not use_mesh:
            return base_results

        # Enhance with neural mesh traversal
        enhanced_results = []
        seen_chunks = set()

        for result in base_results['results']:
            chunk_text = result.get('text', '')
            chunk_source = result.get('source', '')

            # Find corresponding mesh node
            mesh_node_id = None
            for node_id, node in self.neural_mesh.nodes.items():
                if (node.metadata.get('document_id') == chunk_source and
                    node.metadata.get('text_preview', '').startswith(chunk_text[:50])):
                    mesh_node_id = node_id
                    break

            if mesh_node_id:
                # Traverse mesh from this node
                mesh_results = self.neural_mesh.traverse_mesh(
                    mesh_node_id,
                    max_depth=2,
                    min_weight=0.5
                )

                # Extract additional relevant chunks
                for mesh_result in mesh_results[1:]:  # Skip the starting node
                    mesh_node = self.neural_mesh.nodes.get(mesh_result['node_id'])
                    if mesh_node and mesh_node.id not in seen_chunks:
                        # Convert mesh node back to search result format
                        vector_result = {
                            'id': mesh_node.metadata.get('vector_id', mesh_node.id),
                            'text': mesh_node.metadata.get('text_preview', ''),
                            'source': mesh_node.metadata.get('document_id', ''),
                            'score': mesh_result.get('total_weight', 0.5),
                            'mesh_enhanced': True
                        }
                        enhanced_results.append(vector_result)
                        seen_chunks.add(mesh_node.id)

                        if len(enhanced_results) >= max_results:
                            break

                if len(enhanced_results) >= max_results:
                    break

        # Combine and deduplicate results
        all_results = base_results['results'] + enhanced_results
        seen_sources = set()
        unique_results = []

        for result in all_results:
            source_key = (result['source'], result.get('id', ''))
            if source_key not in seen_sources:
                unique_results.append(result)
                seen_sources.add(source_key)

        # Sort by score and limit results
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = unique_results[:max_results]

        return {
            'query': query,
            'results': final_results,
            'total_found': len(final_results),
            'categories': base_results.get('categories', []),
            'search_type': 'hybrid_mesh' if use_mesh else 'hybrid',
            'mesh_enhanced': len(enhanced_results) > 0
        }

    def reinforce_memory(self, query: str, selected_chunks: List[Dict[str, Any]],
                        reinforcement: float = 0.1):
        """
        Reinforce memory connections based on usage (digital neuroplasticity)

        Args:
            query: The query that led to these results
            selected_chunks: Chunks that were used/selected
            reinforcement: Amount of reinforcement
        """
        logger.debug(f"Reinforcing memory for query: '{query}' with {len(selected_chunks)} chunks")

        # Create query embedding for mesh connections
        query_embedding = self.embedder.encode_texts([query])[0]

        # Find mesh nodes corresponding to selected chunks
        mesh_nodes = []
        for chunk in selected_chunks:
            chunk_source = chunk.get('source', '')
            chunk_text = chunk.get('text', '')

            for node_id, node in self.neural_mesh.nodes.items():
                if (node.metadata.get('document_id') == chunk_source and
                    chunk_text.startswith(node.metadata.get('text_preview', '')[:50])):
                    mesh_nodes.append(node_id)
                    break

        # Reinforce connections between selected nodes
        for i, node_id1 in enumerate(mesh_nodes):
            for node_id2 in mesh_nodes[i+1:]:
                self.neural_mesh.reinforce_connection(node_id1, node_id2, reinforcement)

        # Create connections from query concept to selected nodes (if we had query nodes)
        # For now, just reinforce the selected nodes themselves
        for node_id in mesh_nodes:
            if node_id in self.neural_mesh.nodes:
                node = self.neural_mesh.nodes[node_id]
                node.activation_level = min(1.0, node.activation_level + reinforcement)
                node.last_accessed = self._get_timestamp()

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        hierarchy_stats = self.hierarchy.get_stats()
        vector_stats = self.vector_store.get_stats()
        mesh_stats = self.neural_mesh.get_mesh_stats()

        return {
            'hierarchy': hierarchy_stats,
            'vectors': vector_stats,
            'neural_mesh': mesh_stats,
            'total_memory_items': (hierarchy_stats['total_documents'] +
                                 vector_stats['total_vectors'] +
                                 mesh_stats['total_nodes'])
        }

    def cleanup_memory(self, days_old: float = 30, min_activation: float = 0.1):
        """
        Clean up old or weak memory connections

        Args:
            days_old: Remove connections older than this many days
            min_activation: Minimum activation level to keep
        """
        logger.info(f"Cleaning up memory (older than {days_old} days, activation < {min_activation})")

        current_time = self._get_timestamp()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)

        # Decay neural mesh weights
        self.neural_mesh.decay_weights(days_old / 30.0)  # Convert to monthly decay

        # Remove weak mesh connections
        self.neural_mesh.cleanup_weak_connections()

        # Could also implement hierarchy and vector cleanup here
        logger.info("Memory cleanup completed")

    async def store_conversation(self, conversation_id: str, messages: List[Dict[str, Any]],
                                category: str = None) -> bool:
        """
        Store a conversation in memory for persistence

        Args:
            conversation_id: Unique conversation identifier
            messages: List of message objects with user/assistant content
            category: Optional category for the conversation

        Returns:
            Success status
        """
        try:
            logger.info(f"Storing conversation: {conversation_id} with {len(messages)} messages")

            # Validate inputs
            if not conversation_id or not isinstance(conversation_id, str):
                logger.error("Invalid conversation_id provided")
                return False

            if not messages or not isinstance(messages, list):
                logger.error("Invalid messages provided")
                return False

            # Create conversation category if it doesn't exist
            conv_category = category or "general"  # Store in general category like other conversations
            logger.info(f"Looking for conversation category: {conv_category}")

            existing_node = self.hierarchy.find_node_by_name(conv_category)
            logger.info(f"Existing category node: {existing_node is not None}")

            if not existing_node:
                logger.info(f"Creating new conversation category: {conv_category}")
                new_node_id = self.hierarchy.add_node(conv_category, f"Category for {conv_category} content")
                logger.info(f"Created category with ID: {new_node_id}")

            category_node = self.hierarchy.find_node_by_name(conv_category)
            logger.info(f"Final category node: {category_node is not None}, ID: {category_node.id if category_node else None}")

            if not category_node:
                logger.error(f"Failed to create/find conversation category: {conv_category}")
                return False

            # Process conversation messages with error handling
            try:
                conversation_text = ""
                for msg in messages:
                    if not isinstance(msg, dict):
                        logger.warning(f"Skipping invalid message format: {msg}")
                        continue
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    conversation_text += f"{role}: {content}\n\n"
            except Exception as e:
                logger.error(f"Error processing conversation messages: {e}")
                return False

            # Split into chunks and embed with error handling
            try:
                chunks = self.text_processor.hybrid_chunk(conversation_text, max_chunk_size=512)
                chunks_with_embeddings = self.embedder.encode_chunks(chunks)
            except Exception as e:
                logger.error(f"Error processing conversation text: {e}")
                return False

            # Store in vector database with error handling
            try:
                vectors = np.array([chunk['embedding'] for chunk in chunks_with_embeddings])

                # Create proper metadata dictionaries for vector store
                metadata = [{
                    'text': chunk['text'],
                    'source': conversation_id,
                    'type': 'conversation',
                    'chunk_index': i
                } for i, chunk in enumerate(chunks_with_embeddings)]

                sources = [conversation_id] * len(chunks_with_embeddings)

                vector_ids = self.vector_store.add_vectors(vectors, metadata, sources)
            except Exception as e:
                logger.error(f"Error storing vectors: {e}")
                return False

            # Attach conversation to hierarchy (not chunks)
            try:
                self.hierarchy.attach_document(category_node.id, conversation_id)
            except Exception as e:
                logger.error(f"Error attaching to hierarchy: {e}")
                return False

            # Add to neural mesh with error handling
            try:
                import hashlib
                node_mappings = {}  # Track entity -> node_id mappings for semantic edges

                for i, chunk in enumerate(chunks_with_embeddings):
                    content_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
                    mesh_node_id = f"{conversation_id}_chunk_{i}"

                    self.neural_mesh.add_node(
                        node_id=mesh_node_id,
                        content_hash=content_hash,
                        embedding=chunk['embedding'],
                        metadata={
                            'conversation_id': conversation_id,
                            'chunk_index': i,
                            'vector_id': vector_ids[i],
                            'category': conv_category,
                            'content_type': 'conversation',
                            'message_count': len(messages),
                            'messages': messages,  # Store full messages for context retrieval
                            'text_preview': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                        }
                    )

                    # ===== AUTONOMOUS WORD ASSOCIATION LEARNING =====
                    # Process conversation chunk text to create word neurons and learn associations
                    try:
                        learning_result = self.neural_mesh.process_text_for_learning(
                            chunk['text'],
                            context=f"conversation_{conversation_id}_chunk_{i}"
                        )
                        logger.debug(f"Autonomous learning from conversation chunk {i}: {learning_result}")
                    except Exception as e:
                        logger.warning(f"Failed to process conversation chunk {i} for autonomous learning: {e}")

                    # Build node mappings for semantic relationship extraction
                    # Use chunk text as potential entity names for relationship mapping
                    chunk_text = chunk['text'][:50].strip()  # Use first 50 chars as entity identifier
                    if chunk_text:
                        node_mappings[chunk_text] = mesh_node_id

            except Exception as e:
                logger.error(f"Error adding to neural mesh: {e}")
                return False

            # ===== SCAFFOLDING & SUBSTRATE MODEL: Extract and create semantic relationships =====
            try:
                if self.curator and self.curator.is_ready():
                    logger.info(f"Extracting semantic relationships from conversation: {conversation_id}")

                    # Extract relationships from the full conversation text
                    relationships = await self.curator.extract_relationships(conversation_text, f"conversation_{conversation_id}")

                    if relationships:
                        logger.info(f"Found {len(relationships)} relationships in conversation")

                        # Create semantic edges in neural mesh
                        edges_created = await self.curator.create_semantic_edges(
                            relationships, self.neural_mesh, node_mappings
                        )

                        logger.info(f"Created {edges_created} semantic edges for conversation {conversation_id}")

                        # Store relationship metadata
                        for node_id in self.neural_mesh.nodes:
                            if node_id.startswith(f"{conversation_id}_chunk_"):
                                node = self.neural_mesh.nodes[node_id]
                                node.metadata['relationships_extracted'] = len(relationships)
                                node.metadata['semantic_edges_created'] = edges_created

                    # Apply Hebbian learning to reinforce connections between related conversation chunks
                    if len(chunks_with_embeddings) > 1:
                        chunk_node_ids = [f"{conversation_id}_chunk_{i}" for i in range(len(chunks_with_embeddings))]
                        self.neural_mesh.reinforce_hebbian_connections(chunk_node_ids, reward=0.2)
                        logger.debug(f"Applied Hebbian learning to {len(chunk_node_ids)} conversation chunks")

                else:
                    logger.debug("Memory curator not available for relationship extraction")

            except Exception as e:
                logger.error(f"Error in semantic relationship processing: {e}")
                # Don't fail the entire storage operation for relationship extraction errors

            logger.info(f"Conversation stored successfully: {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store conversation {conversation_id}: {e}")
            return False

    def get_conversation_messages(self, conversation_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve full conversation messages for context integration

        Args:
            conversation_id: Conversation ID to retrieve
            max_messages: Maximum messages to return (most recent)

        Returns:
            List of message objects with role and content
        """
        try:
            logger.info(f"Retrieving messages for conversation: {conversation_id}")

            # Find conversation chunks in neural mesh
            conversation_messages = []
            for node_id, node in self.neural_mesh.nodes.items():
                if (node.metadata.get('content_type') == 'conversation' and
                    node.metadata.get('conversation_id') == conversation_id):
                    # Get messages from metadata
                    messages = node.metadata.get('messages', [])
                    if messages:
                        conversation_messages.extend(messages)
                        break  # All chunks should have the same messages

            # Remove duplicates and sort by conversation order
            seen_messages = set()
            unique_messages = []
            for msg in conversation_messages:
                msg_key = (msg.get('role', ''), msg.get('content', ''))
                if msg_key not in seen_messages:
                    unique_messages.append(msg)
                    seen_messages.add(msg_key)

            logger.info(f"Retrieved {len(unique_messages)} messages for conversation {conversation_id}")
            return unique_messages[-max_messages:]  # Return most recent messages

        except Exception as e:
            logger.error(f"Failed to retrieve conversation messages for {conversation_id}: {e}")
            return []

    def get_conversation_history(self, conversation_id: str = None,
                               category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history from memory

        Args:
            conversation_id: Specific conversation ID (optional)
            category: Filter by category (optional)
            limit: Maximum conversations to return

        Returns:
            List of conversation summaries
        """
        try:
            conversations = []
            logger.info(f"Getting conversation history: category={category}, conversation_id={conversation_id}, limit={limit}")

            # First, get conversation IDs from hierarchy
            # Conversations are stored in the "general" category, so look there
            conv_category = category or "general"  # Changed from "conversations" to "general"
            logger.info(f"Looking for conversations in category: {conv_category}")

            category_node = self.hierarchy.find_node_by_name(conv_category)
            logger.info(f"Category node found: {category_node is not None}")

            conversation_ids = []
            if category_node:
                # Filter to only conversation IDs (those starting with "conv_")
                all_docs = category_node.document_ids
                conversation_ids = [doc_id for doc_id in all_docs if doc_id.startswith('conv_')]
                logger.info(f"Found {len(conversation_ids)} conversation IDs in category {conv_category}: {conversation_ids}")
            else:
                # Fallback: search all categories for conversation IDs
                logger.info("Category not found, searching all categories for conversations...")
                for node in self.hierarchy.nodes.values():
                    conv_docs = [doc_id for doc_id in node.document_ids if doc_id.startswith('conv_')]
                    if conv_docs:
                        conversation_ids.extend(conv_docs)
                        logger.info(f"Found conversations in category {node.name}: {conv_docs}")
                logger.info(f"Total conversation IDs found: {len(conversation_ids)}")

            # If specific conversation requested, filter to just that one
            if conversation_id:
                conversation_ids = [cid for cid in conversation_ids if cid == conversation_id]
                logger.info(f"Filtered to specific conversation: {conversation_ids}")

            # Get details for each conversation from neural mesh
            logger.info(f"Searching neural mesh for {len(conversation_ids)} conversations...")
            for conv_id in conversation_ids:
                # Find conversation chunks in neural mesh
                conv_chunks = []
                message_count = 0
                latest_accessed = 0
                preview_text = ""

                for node_id, node in self.neural_mesh.nodes.items():
                    if (node.metadata.get('content_type') == 'conversation' and
                        node.metadata.get('conversation_id') == conv_id):
                        conv_chunks.append(node)
                        message_count = max(message_count, node.metadata.get('message_count', 0))
                        latest_accessed = max(latest_accessed, node.last_accessed)
                        if not preview_text and node.metadata.get('text_preview'):
                            preview_text = node.metadata.get('text_preview', '')

                logger.info(f"Conversation {conv_id}: found {len(conv_chunks)} chunks")

                if conv_chunks:  # Only add if we found chunks
                    conversations.append({
                        'conversation_id': conv_id,
                        'category': conv_category,
                        'message_count': message_count,
                        'preview': preview_text,
                        'created_at': min(chunk.created_at for chunk in conv_chunks),
                        'last_accessed': latest_accessed,
                        'activation_level': max(chunk.activation_level for chunk in conv_chunks),
                        'chunks': len(conv_chunks)
                    })

            logger.info(f"Returning {len(conversations)} conversations")
            # Sort by last accessed, most recent first
            conversations.sort(key=lambda x: x['last_accessed'], reverse=True)
            return conversations[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation from memory

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            Success status
        """
        try:
            logger.info(f"Deleting conversation: {conversation_id}")

            # Remove from neural mesh
            nodes_to_remove = []
            for node_id, node in self.neural_mesh.nodes.items():
                if (node.metadata.get('content_type') == 'conversation' and
                    node.metadata.get('conversation_id') == conversation_id):
                    nodes_to_remove.append(node_id)

            for node_id in nodes_to_remove:
                # Remove connections
                for other_id in list(self.neural_mesh.adjacency_list[node_id]):
                    if other_id in self.neural_mesh.adjacency_list:
                        self.neural_mesh.adjacency_list[other_id].discard(node_id)
                    edge_keys = [(node_id, other_id), (other_id, node_id)]
                    for edge_key in edge_keys:
                        self.neural_mesh.edges.pop(edge_key, None)

                # Remove node
                del self.neural_mesh.nodes[node_id]
                self.neural_mesh.adjacency_list.pop(node_id, None)

            # Remove from vector store (would need additional implementation)
            # For now, just mark as deleted in mesh
            self.neural_mesh._save_mesh()

            logger.info(f"Conversation deleted: {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False

    def create_category(self, name: str, description: str = "",
                       parent_category: str = None) -> str:
        """
        Create a new category in the hierarchy

        Args:
            name: Category name
            description: Category description
            parent_category: Parent category name

        Returns:
            New category ID
        """
        parent_id = None
        if parent_category:
            parent_node = self.hierarchy.find_node_by_name(parent_category)
            if parent_node:
                parent_id = parent_node.id

        return self.hierarchy.add_node(name, description, parent_id)

    def list_categories(self) -> List[Dict[str, Any]]:
        """List all categories"""
        categories = []
        for node in self.hierarchy.nodes.values():
            if node.id != self.hierarchy.root_id:  # Skip root
                categories.append({
                    'id': node.id,
                    'name': node.name,
                    'description': node.description,
                    'document_count': len(node.document_ids),
                    'document_ids': node.document_ids,  # Include document IDs for filtering
                    'parent_id': node.parent_id
                })

        return categories

    def _auto_categorize_document(self, document: Document) -> str:
        """Automatically categorize a document based on content"""
        # Simple categorization based on file type and keywords
        file_type = document.metadata.get('file_type', '').lower()

        if file_type in ['.py', '.js', '.html', '.css']:
            return 'technical'
        elif file_type in ['.md', '.txt']:
            # Check for keywords
            content_lower = document.content.lower()
            if any(word in content_lower for word in ['project', 'code', 'development']):
                return 'projects'
            elif any(word in content_lower for word in ['personal', 'note', 'diary']):
                return 'personal'
        elif file_type == '.pdf':
            return 'technical'  # Assume PDFs are technical documents

        return 'general'

    def _extract_category_from_path(self, file_path: Path) -> str:
        """Extract category from file path"""
        # Use parent directory name as category
        parent_name = file_path.parent.name.lower()
        if parent_name in ['documents', 'data', 'files']:
            return 'general'
        return parent_name

    def _find_document_category(self, document_id: str) -> Optional[str]:
        """Find which category a document belongs to"""
        for node in self.hierarchy.nodes.values():
            if document_id in node.document_ids:
                return node.name
        return None

    # ===== BRAIN-INSPIRED MEMORY TIER MANAGEMENT =====

    def _load_memory_tiers(self):
        """Load memory tier data from disk"""
        try:
            tier_file = settings.data_dir / "memory_tiers.json"
            if tier_file.exists():
                with open(tier_file, 'r') as f:
                    data = json.load(f)
                    self.memory_tiers = data.get('tiers', {})
                    self.access_patterns = data.get('access_patterns', {})
                    self.consolidation_cache = data.get('consolidation_cache', {})
                logger.info(f"Loaded memory tier data for {len(self.memory_tiers)} nodes")
        except Exception as e:
            logger.warning(f"Failed to load memory tier data: {e}")

    def _save_memory_tiers(self):
        """Save memory tier data to disk"""
        try:
            tier_file = settings.data_dir / "memory_tiers.json"
            data = {
                'tiers': self.memory_tiers,
                'access_patterns': self.access_patterns,
                'consolidation_cache': self.consolidation_cache
            }
            with open(tier_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory tier data: {e}")

    def track_memory_access(self, node_id: str):
        """Track access to a memory node for tier management"""
        current_time = self._get_timestamp()

        if node_id not in self.access_patterns:
            self.access_patterns[node_id] = []

        self.access_patterns[node_id].append({
            'timestamp': current_time,
            'type': 'access'
        })

        # Keep only recent access history (last 100 accesses)
        if len(self.access_patterns[node_id]) > 100:
            self.access_patterns[node_id] = self.access_patterns[node_id][-100:]

        # Update node last accessed time
        if node_id in self.neural_mesh.nodes:
            self.neural_mesh.nodes[node_id].last_accessed = current_time

    def determine_memory_tier(self, node_id: str) -> str:
        """Determine which tier a memory node should be in"""
        if node_id not in self.neural_mesh.nodes:
            return 'archived'

        node = self.neural_mesh.nodes[node_id]
        current_time = self._get_timestamp()

        # Get access history
        access_history = self.access_patterns.get(node_id, [])
        recent_accesses = [a for a in access_history
                          if current_time - a['timestamp'] < (30 * 24 * 60 * 60)]  # Last 30 days

        # Calculate metrics
        hours_since_last_access = (current_time - node.last_accessed) / 3600
        access_count_recent = len(recent_accesses)
        activation_level = node.activation_level

        # Determine tier based on brain-inspired rules
        if hours_since_last_access <= 24 and (access_count_recent >= 1 or activation_level > 0.5):
            return 'active'
        elif hours_since_last_access <= 168:  # 7 days
            return 'short_term'
        elif hours_since_last_access <= 720:  # 30 days
            return 'long_term'
        else:
            return 'archived'

    def migrate_memory_tiers(self):
        """Automatically migrate memories between tiers based on usage patterns"""
        logger.info("Running automatic memory tier migration...")

        migrated_count = 0
        for node_id in list(self.neural_mesh.nodes.keys()):
            current_tier = self.memory_tiers.get(node_id, 'active')
            new_tier = self.determine_memory_tier(node_id)

            if current_tier != new_tier:
                logger.debug(f"Migrating {node_id} from {current_tier} to {new_tier}")
                self.memory_tiers[node_id] = new_tier
                migrated_count += 1

                # Apply tier-specific compression/optimization
                self._apply_tier_compression(node_id, new_tier)

        if migrated_count > 0:
            self._save_memory_tiers()
            logger.info(f"Migrated {migrated_count} memories between tiers")

        return migrated_count

    def _apply_tier_compression(self, node_id: str, tier: str):
        """Apply compression based on memory tier"""
        if node_id not in self.neural_mesh.nodes:
            return

        node = self.neural_mesh.nodes[node_id]
        tier_config = self.TIERS[tier]

        if tier_config['compression'] == 'summary':
            # Create summarized version for long-term storage
            if 'full_text' not in node.metadata:
                node.metadata['full_text'] = node.metadata.get('text_preview', '')
                # Generate summary (simplified for now)
                summary = node.metadata['full_text'][:200] + "..." if len(node.metadata['full_text']) > 200 else node.metadata['full_text']
                node.metadata['text_preview'] = f"[SUMMARY] {summary}"

        elif tier_config['compression'] == 'minimal':
            # Minimal representation for archived memories
            if 'full_text' not in node.metadata:
                node.metadata['full_text'] = node.metadata.get('text_preview', '')
                node.metadata['text_preview'] = f"[ARCHIVED] {node.metadata.get('category', 'unknown')} memory"

    def consolidate_similar_memories(self):
        """Automatically consolidate similar memories (brain-inspired memory consolidation)"""
        logger.info("Running memory consolidation...")

        consolidated_count = 0
        nodes_to_process = list(self.neural_mesh.nodes.keys())

        # Group nodes by content similarity (simplified clustering)
        content_groups = {}
        for node_id in nodes_to_process:
            if node_id in self.neural_mesh.nodes:
                node = self.neural_mesh.nodes[node_id]
                content_hash = node.content_hash

                if content_hash not in content_groups:
                    content_groups[content_hash] = []
                content_groups[content_hash].append(node_id)

        # Consolidate groups with multiple similar nodes
        for content_hash, node_ids in content_groups.items():
            if len(node_ids) > 1:
                # Keep the most active node, consolidate others
                nodes_with_activation = [(nid, self.neural_mesh.nodes[nid].activation_level)
                                       for nid in node_ids if nid in self.neural_mesh.nodes]
                nodes_with_activation.sort(key=lambda x: x[1], reverse=True)

                primary_node = nodes_with_activation[0][0]
                secondary_nodes = [n[0] for n in nodes_with_activation[1:]]

                # Strengthen the primary node
                if primary_node in self.neural_mesh.nodes:
                    self.neural_mesh.nodes[primary_node].activation_level = min(1.0,
                        self.neural_mesh.nodes[primary_node].activation_level + 0.1)

                # Mark secondary nodes for consolidation
                for secondary_id in secondary_nodes:
                    if secondary_id not in self.consolidation_cache:
                        self.consolidation_cache[secondary_id] = {
                            'consolidated_into': primary_node,
                            'timestamp': self._get_timestamp()
                        }
                        consolidated_count += 1

        if consolidated_count > 0:
            self._save_memory_tiers()
            logger.info(f"Consolidated {consolidated_count} similar memories")

        return consolidated_count

    def get_memory_tier_stats(self) -> Dict[str, Any]:
        """Get statistics about memory tier distribution"""
        tier_counts = {'active': 0, 'short_term': 0, 'long_term': 0, 'archived': 0}

        for tier in self.memory_tiers.values():
            if tier in tier_counts:
                tier_counts[tier] += 1

        return {
            'tier_distribution': tier_counts,
            'total_tracked': len(self.memory_tiers),
            'consolidation_cache_size': len(self.consolidation_cache),
            'access_patterns_tracked': len(self.access_patterns)
        }

    def run_memory_maintenance(self):
        """Run complete memory maintenance cycle (brain-inspired)"""
        logger.info("Running brain-inspired memory maintenance...")

        # 1. Migrate memories between tiers
        migrations = self.migrate_memory_tiers()

        # 2. Consolidate similar memories
        consolidations = self.consolidate_similar_memories()

        # 3. Clean up old consolidation records
        cleanup_count = self._cleanup_old_consolidations()

        # 4. Update neural mesh connections based on usage
        self.neural_mesh.decay_weights(0.01)  # Gentle decay

        stats = {
            'migrations': migrations,
            'consolidations': consolidations,
            'cleanups': cleanup_count,
            'tier_stats': self.get_memory_tier_stats()
        }

        logger.info(f"Memory maintenance complete: {stats}")
        return stats

    def _cleanup_old_consolidations(self) -> int:
        """Clean up old consolidation records"""
        current_time = self._get_timestamp()
        cutoff_time = current_time - (90 * 24 * 60 * 60)  # 90 days

        old_records = [k for k, v in self.consolidation_cache.items()
                      if v.get('timestamp', 0) < cutoff_time]

        for key in old_records:
            del self.consolidation_cache[key]

        return len(old_records)

    # ===== ADVANCED CACHING AND PERFORMANCE OPTIMIZATION =====

    def _get_cache_key(self, query: str, category: str = None, max_results: int = 5) -> str:
        """Generate a cache key for search queries"""
        import hashlib
        key_data = f"{query}|{category or 'all'}|{max_results}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_cached_search_results(self, query: str, category: str = None,
                                 max_results: int = 5) -> Optional[Dict[str, Any]]:
        """Get cached search results if available and fresh"""
        cache_key = self._get_cache_key(query, category, max_results)
        current_time = self._get_timestamp()

        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            cache_age = current_time - cached_result.get('timestamp', 0)

            # Cache valid for 1 hour
            if cache_age < 3600:
                logger.debug(f"Using cached search results for: '{query}'")
                return cached_result['data']

            # Remove expired cache entry
            del self.search_cache[cache_key]

        return None

    def cache_search_results(self, query: str, results: Dict[str, Any],
                           category: str = None, max_results: int = 5):
        """Cache search results for future use"""
        cache_key = self._get_cache_key(query, category, max_results)

        self.search_cache[cache_key] = {
            'data': results,
            'timestamp': self._get_timestamp(),
            'query': query,
            'category': category,
            'max_results': max_results
        }

        # Limit cache size to prevent memory bloat
        if len(self.search_cache) > 100:
            # Remove oldest entries (simple LRU approximation)
            oldest_key = min(self.search_cache.keys(),
                           key=lambda k: self.search_cache[k]['timestamp'])
            del self.search_cache[oldest_key]

    def get_cached_mesh_traversal(self, node_id: str, max_depth: int = 2,
                                 min_weight: float = 0.5) -> Optional[List[Dict[str, Any]]]:
        """Get cached mesh traversal results"""
        cache_key = f"{node_id}_{max_depth}_{min_weight}"

        if cache_key in self.mesh_traversal_cache:
            cached_result = self.mesh_traversal_cache[cache_key]
            cache_age = self._get_timestamp() - cached_result['timestamp']

            # Cache valid for 30 minutes
            if cache_age < 1800:
                logger.debug(f"Using cached mesh traversal for node: {node_id}")
                return cached_result['results']

            # Remove expired cache entry
            del self.mesh_traversal_cache[cache_key]

        return None

    def cache_mesh_traversal(self, node_id: str, results: List[Dict[str, Any]],
                           max_depth: int = 2, min_weight: float = 0.5):
        """Cache mesh traversal results"""
        cache_key = f"{node_id}_{max_depth}_{min_weight}"

        self.mesh_traversal_cache[cache_key] = {
            'results': results,
            'timestamp': self._get_timestamp(),
            'node_id': node_id,
            'max_depth': max_depth,
            'min_weight': min_weight
        }

        # Limit cache size
        if len(self.mesh_traversal_cache) > 50:
            oldest_key = min(self.mesh_traversal_cache.keys(),
                           key=lambda k: self.mesh_traversal_cache[k]['timestamp'])
            del self.mesh_traversal_cache[oldest_key]

    def hybrid_search_cached(self, query: str, category: str = None,
                           max_results: int = 5, use_mesh: bool = True) -> Dict[str, Any]:
        """
        Perform hybrid search with SmartCache integration

        Args:
            query: Search query
            category: Optional category filter
            max_results: Maximum results to return
            use_mesh: Whether to use neural mesh traversal

        Returns:
            Enhanced search results with intelligent caching
        """
        # Create cache key for SmartCache
        cache_key = f"search_{query}_{category or 'all'}_{max_results}_{use_mesh}"

        # Check SmartCache first
        cached_result = self.smart_cache.get(cache_key)
        if cached_result:
            logger.debug(f"SmartCache hit for query: '{query}'")
            return cached_result

        # Perform fresh search
        logger.info(f"Performing fresh hybrid search for: '{query}' (category: {category})")

        # Get final results for logging
        final_results = self._perform_hybrid_search(query, category, max_results, use_mesh)

        # Cache the results (TTL: 1 hour for search results)
        self.smart_cache.set(cache_key, final_results, ttl_seconds=3600)

        # Add retrieval logging as per Phase 1 requirements
        logger.info("hybrid_search_cached: query='%s', returned=%d chunks, top_scores=%s",
                   query, len(final_results.get('results', [])),
                   [round(r.get('score', 0), 3) for r in final_results.get('results', [])[:5]])

        return final_results

    def _perform_hybrid_search(self, query: str, category: str = None,
                             max_results: int = 5, use_mesh: bool = True) -> Dict[str, Any]:
        """
        Internal method to perform the actual hybrid search logic
        """

        # Start with standard search
        base_results = self.search_memory(query, category, max_results * 2)

        # Cache the base results
        self.cache_search_results(query, base_results, category, max_results * 2)

        if not base_results['results'] or not use_mesh:
            return base_results

        # Enhance with neural mesh traversal (with caching)
        enhanced_results = []
        seen_chunks = set()

        for result in base_results['results']:
            chunk_text = result['text']
            chunk_source = result['source']

            # Find corresponding mesh node
            mesh_node_id = None
            for node_id, node in self.neural_mesh.nodes.items():
                if (node.metadata.get('document_id') == chunk_source and
                    node.metadata.get('text_preview', '').startswith(chunk_text[:50])):
                    mesh_node_id = node_id
                    break

            if mesh_node_id:
                # Check for cached mesh traversal
                cached_traversal = self.get_cached_mesh_traversal(mesh_node_id, 2, 0.5)

                if cached_traversal:
                    mesh_results = cached_traversal
                else:
                    # Perform fresh traversal
                    mesh_results = self.neural_mesh.traverse_mesh(
                        mesh_node_id,
                        max_depth=2,
                        min_weight=0.5
                    )
                    # Cache the results
                    self.cache_mesh_traversal(mesh_node_id, mesh_results, 2, 0.5)

                # Extract additional relevant chunks
                for mesh_result in mesh_results[1:]:  # Skip the starting node
                    mesh_node = self.neural_mesh.nodes.get(mesh_result['node_id'])
                    if mesh_node and mesh_node.id not in seen_chunks:
                        # Convert mesh node back to search result format
                        vector_result = {
                            'id': mesh_node.metadata.get('vector_id', mesh_node.id),
                            'text': mesh_node.metadata.get('text_preview', ''),
                            'source': mesh_node.metadata.get('document_id', ''),
                            'score': mesh_result.get('total_weight', 0.5),
                            'mesh_enhanced': True
                        }
                        enhanced_results.append(vector_result)
                        seen_chunks.add(mesh_node.id)

                        if len(enhanced_results) >= max_results:
                            break

                if len(enhanced_results) >= max_results:
                    break

        # Combine and deduplicate results
        all_results = base_results['results'] + enhanced_results
        seen_sources = set()
        unique_results = []

        for result in all_results:
            source_key = (result['source'], result.get('id', ''))
            if source_key not in seen_sources:
                unique_results.append(result)
                seen_sources.add(source_key)

        # Sort by score and limit results
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = unique_results[:max_results]

        enhanced_response = {
            'query': query,
            'results': final_results,
            'total_found': len(final_results),
            'categories': base_results.get('categories', []),
            'search_type': 'hybrid_mesh_cached' if use_mesh else 'hybrid_cached',
            'mesh_enhanced': len(enhanced_results) > 0,
            'cached': False  # This is a fresh result
        }

        return enhanced_response

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance and caching statistics"""
        current_time = self._get_timestamp()

        # Calculate cache hit rates and performance metrics
        search_cache_stats = {
            'size': len(self.search_cache),
            'avg_age_seconds': 0,
            'oldest_entry_seconds': 0
        }

        if self.search_cache:
            ages = [current_time - entry['timestamp'] for entry in self.search_cache.values()]
            search_cache_stats['avg_age_seconds'] = sum(ages) / len(ages)
            search_cache_stats['oldest_entry_seconds'] = max(ages)

        mesh_cache_stats = {
            'size': len(self.mesh_traversal_cache),
            'avg_age_seconds': 0,
            'oldest_entry_seconds': 0
        }

        if self.mesh_traversal_cache:
            ages = [current_time - entry['timestamp'] for entry in self.mesh_traversal_cache.values()]
            mesh_cache_stats['avg_age_seconds'] = sum(ages) / len(ages)
            mesh_cache_stats['oldest_entry_seconds'] = max(ages)

        embedding_cache_stats = {
            'size': len(self.embedder.embedding_cache) if hasattr(self.embedder, 'embedding_cache') else 0
        }

        return {
            'caching': {
                'search_cache': search_cache_stats,
                'mesh_traversal_cache': mesh_cache_stats,
                'embedding_cache': embedding_cache_stats
            },
            'memory_tiers': self.get_memory_tier_stats(),
            'performance': {
                'total_cache_size': (search_cache_stats['size'] +
                                   mesh_cache_stats['size'] +
                                   embedding_cache_stats['size']),
                'cache_efficiency': 'high' if search_cache_stats['size'] > 10 else 'building'
            }
        }

    def add_knowledge_chunk(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a single knowledge chunk to memory (for learning pipeline)

        This is a simplified method for adding individual knowledge pieces
        without going through the full document ingestion pipeline.

        Args:
            text: The knowledge text to add
            metadata: Optional metadata (source, confidence, etc.)

        Returns:
            Success status
        """
        try:
            metadata = metadata or {}
            chunk_id = f"knowledge_{int(self._get_timestamp())}_{hash(text) % 10000}"

            logger.info(f"Adding knowledge chunk: {chunk_id} ({len(text)} chars)")

            # Create a simple chunk structure
            chunk = {
                'text': text,
                'metadata': {
                    'chunk_id': chunk_id,
                    'source': metadata.get('source', 'learning_pipeline'),
                    'confidence': metadata.get('confidence', 0.5),
                    'added_at': self._get_timestamp(),
                    **metadata
                }
            }

            # Generate embedding
            chunks_with_embeddings = self.embedder.encode_chunks([chunk])

            # Store in vector database
            vectors = np.array([chunks_with_embeddings[0]['embedding']])

            # Create metadata for vector store
            vector_metadata = [{
                'text': text,
                'source': chunk_id,
                'type': 'knowledge_chunk',
                'chunk_index': 0
            }]

            sources = [chunk_id]

            vector_ids = self.vector_store.add_vectors(vectors, vector_metadata, sources)

            # Add to neural mesh
            import hashlib
            content_hash = hashlib.md5(text.encode()).hexdigest()

            self.neural_mesh.add_node(
                node_id=chunk_id,
                content_hash=content_hash,
                embedding=chunks_with_embeddings[0]['embedding'],
                metadata={
                    'vector_id': vector_ids[0],
                    'category': metadata.get('category', 'learned'),
                    'content_type': 'knowledge_chunk',
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                    'confidence': metadata.get('confidence', 0.5),
                    'source': metadata.get('source', 'learning_pipeline')
                }
            )

            logger.info(f"Knowledge chunk added successfully: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add knowledge chunk: {e}")
            return False

    def optimize_performance(self):
        """Run performance optimization routines"""
        logger.info("Running performance optimization...")

        # Clean up expired cache entries
        current_time = self._get_timestamp()
        search_expired = []
        mesh_expired = []

        # Clean search cache (1 hour expiry)
        for key, entry in self.search_cache.items():
            if current_time - entry['timestamp'] > 3600:
                search_expired.append(key)

        # Clean mesh cache (30 minutes expiry)
        for key, entry in self.mesh_traversal_cache.items():
            if current_time - entry['timestamp'] > 1800:
                mesh_expired.append(key)

        # Remove expired entries
        for key in search_expired:
            del self.search_cache[key]

        for key in mesh_expired:
            del self.mesh_traversal_cache[key]

        # Run memory maintenance
        maintenance_stats = self.run_memory_maintenance()

        stats = {
            'cache_cleanup': {
                'search_cache_removed': len(search_expired),
                'mesh_cache_removed': len(mesh_expired)
            },
            'memory_maintenance': maintenance_stats,
            'final_stats': self.get_performance_stats()
        }

        logger.info(f"Performance optimization complete: {stats}")
        return stats

    # ===== MEMORY TRANSFORMER CHAT SYSTEM =====

    def initialize_memory_chat(self, model_loader):
        """
        Initialize the memory transformer for direct chat capabilities

        Args:
            model_loader: GGUF model loader for tokenizer access
        """
        if self.memory_transformer is None:
            try:
                logger.info("Initializing Memory Transformer for chat...")

                # Get vocabulary size from model loader
                vocab_size = model_loader.model.n_vocab() if model_loader.is_loaded else 32000

                # Initialize transformer with brain-inspired architecture
                self.memory_transformer = MemoryTransformer(
                    embed_dim=384,  # Match sentence-transformers
                    num_layers=6,   # Small but capable brain
                    num_heads=8,
                    vocab_size=vocab_size,
                    max_seq_len=2048,
                    dropout=0.1
                )

                # Initialize chat interface
                self.memory_chat_interface = MemoryChatInterface(
                    self.memory_transformer,
                    self,
                    model_loader
                )

                logger.info("Memory Transformer initialized for direct chat")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize memory transformer: {e}")
                return False
        return True

    def chat_with_memory(self, user_message: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Direct chat with memory system using transformer reasoning

        Args:
            user_message: User's message
            conversation_history: Previous conversation context

        Returns:
            Memory-generated response
        """
        if self.memory_chat_interface is None:
            return {
                'response': 'Memory chat system not initialized. Please initialize first.',
                'method': 'memory_error'
            }

        try:
            return self.memory_chat_interface.chat(user_message, conversation_history)
        except Exception as e:
            logger.error(f"Memory chat failed: {e}")
            return {
                'response': 'I apologize, but I encountered an error accessing my memory.',
                'error': str(e),
                'method': 'memory_error'
            }

    def get_memory_chat_stats(self) -> Dict[str, Any]:
        """Get memory chat system statistics"""
        if self.memory_transformer is None:
            return {'status': 'not_initialized'}

        return {
            'status': 'ready',
            'transformer_layers': self.memory_transformer.num_layers if hasattr(self.memory_transformer, 'num_layers') else 6,
            'embedding_dim': self.memory_transformer.embed_dim if hasattr(self.memory_transformer, 'embed_dim') else 384,
            'vocab_size': self.memory_transformer.vocab_size if hasattr(self.memory_transformer, 'vocab_size') else 32000,
            'memory_nodes': len(self.neural_mesh.nodes),
            'personality': self.memory_chat_interface.personality if self.memory_chat_interface else {}
        }
