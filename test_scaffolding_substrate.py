#!/usr/bin/env python3
"""
Test script for Scaffolding & Substrate Model - Phase 1
Tests semantic relationship extraction and edge creation
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
backend_path = str(Path(__file__).parent / "backend")
sys.path.insert(0, backend_path)

from backend.memory.memory_manager import MemoryManager
from backend.llm.memory_curator import MemoryCurator
from backend.memory.neural_mesh import NeuralMesh
from backend.memory.pattern_recognition import PatternRecognitionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_semantic_relationships():
    """Test semantic relationship extraction and edge creation"""
    logger.info("Testing Scaffolding & Substrate Model - Phase 1")

    # Initialize components
    memory_manager = MemoryManager()
    neural_mesh = memory_manager.neural_mesh
    curator = MemoryCurator()

    # Wait for curator to initialize
    await asyncio.sleep(1)

    if not curator.is_ready():
        logger.warning("Memory Curator not ready - some tests will be skipped")
        curator = None

    # Test 1: Direct relationship extraction
    test_text = """
    Paris is the capital city of France. The Eiffel Tower is located in Paris and was built in 1889.
    France is a country in Europe, and its president lives in Paris.
    """

    logger.info("Test 1: Direct relationship extraction")
    if curator:
        relationships = await curator.extract_relationships(test_text)
        logger.info(f"Extracted {len(relationships)} relationships:")
        for rel in relationships:
            logger.info(f"  {rel['source']} --[{rel['relationship_label']}]--> {rel['target']} ({rel['relationship_type']}, confidence: {rel['confidence']:.2f})")

        # Test 2: Create semantic edges
        logger.info("Test 2: Creating semantic edges")
        node_mappings = {
            "paris": "test_node_paris",
            "france": "test_node_france",
            "eiffel tower": "test_node_eiffel",
            "europe": "test_node_europe"
        }

        # Add test nodes to mesh
        for entity, node_id in node_mappings.items():
            neural_mesh.add_node(node_id, f"test_{entity}", metadata={'entity': entity})

        edges_created = await curator.create_semantic_edges(relationships, neural_mesh, node_mappings)
        logger.info(f"Created {edges_created} semantic edges in neural mesh")

        # Test 3: Query semantic relationships
        logger.info("Test 3: Querying semantic relationships")
        for node_id in node_mappings.values():
            relationships = neural_mesh.get_semantic_relationships(node_id)
            if relationships:
                logger.info(f"Node {node_id} has {len(relationships)} semantic relationships:")
                for rel in relationships:
                    logger.info(f"  {rel['direction']}: {rel['relationship_label']} ({rel['relationship_type']})")

    # Test 4: Conversation storage with semantic edges
    logger.info("Test 4: Conversation storage with semantic relationship extraction")

    conversation_id = "test_conv_scaffolding"
    messages = [
        {
            'role': 'user',
            'content': 'Tell me about the relationship between Paris and France.'
        },
        {
            'role': 'assistant',
            'content': 'Paris is the capital city of France. France is a European country, and Paris is located in the northern part of France along the Seine River.'
        }
    ]

    success = await memory_manager.store_conversation(conversation_id, messages)
    logger.info(f"Conversation storage {'successful' if success else 'failed'}")

    # Test 5: Pattern Recognition Engine
    logger.info("Test 5: Pattern Recognition Engine")
    pattern_engine = PatternRecognitionEngine(neural_mesh, memory_manager)

    # Simulate a successful query
    mock_memory_results = {
        'results': [
            {'text': 'Paris is the capital of France', 'score': 0.9},
            {'text': 'France is in Europe', 'score': 0.8}
        ],
        'total_found': 2,
        'categories': ['geography'],
        'search_type': 'hybrid_mesh'
    }

    analysis = pattern_engine.analyze_query_success(
        query="What is the capital of France?",
        memory_results=mock_memory_results,
        response_quality=0.9,
        execution_time=0.8
    )

    logger.info(f"Pattern analysis result: {analysis}")

    # Test 6: Neural mesh statistics
    logger.info("Test 6: Neural mesh statistics after testing")
    mesh_stats = neural_mesh.get_mesh_stats()
    logger.info(f"Neural mesh stats: {mesh_stats}")

    # Test 7: Find related nodes
    logger.info("Test 7: Finding semantically related nodes")
    if node_mappings:
        test_node = list(node_mappings.values())[0]
        related = neural_mesh.find_related_nodes(test_node, min_confidence=0.3)
        logger.info(f"Found {len(related)} related nodes for {test_node}")

    logger.info("Scaffolding & Substrate Model - Phase 1 testing completed!")


async def test_hebbian_learning():
    """Test Hebbian learning functionality"""
    logger.info("Testing Hebbian learning")

    neural_mesh = NeuralMesh()

    # Create test nodes
    nodes = []
    for i in range(5):
        node_id = f"hebb_test_{i}"
        neural_mesh.add_node(node_id, f"test_content_{i}")
        nodes.append(node_id)

    # Simulate co-activation (nodes that fire together)
    activated_nodes = nodes[:3]  # First 3 nodes fire together

    logger.info(f"Applying Hebbian learning to nodes: {activated_nodes}")
    neural_mesh.reinforce_hebbian_connections(activated_nodes, reward=0.2)

    # Check connections
    connections_found = 0
    for i, node_a in enumerate(activated_nodes):
        for node_b in activated_nodes[i+1:]:
            edge_key = (node_a, node_b)
            reverse_key = (node_b, node_a)
            if edge_key in neural_mesh.edges or reverse_key in neural_mesh.edges:
                connections_found += 1

    logger.info(f"Hebbian learning created {connections_found} connections between co-activated nodes")

    # Test mesh traversal with learned connections
    if activated_nodes:
        traversal_results = neural_mesh.traverse_mesh(activated_nodes[0], max_depth=2)
        logger.info(f"Mesh traversal found {len(traversal_results)} reachable nodes")


if __name__ == "__main__":
    async def main():
        await test_semantic_relationships()
        await test_hebbian_learning()

    asyncio.run(main())
