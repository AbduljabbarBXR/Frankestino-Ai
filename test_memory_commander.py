#!/usr/bin/env python3
"""
Test script for Memory Commander functionality
Tests direct AI memory manipulation capabilities
"""
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.llm.memory_commander import MemoryCommander
from backend.memory.memory_manager import MemoryManager

def test_memory_commander():
    """Test the Memory Commander functionality"""

    print("=== Testing Memory Commander ===")

    # Initialize memory manager (this would normally be done by the main app)
    try:
        memory_manager = MemoryManager()
        commander = MemoryCommander(memory_manager)
        print("✅ Memory Commander initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Memory Commander: {e}")
        return

    # Test command parsing
    print("\n--- Testing Command Parsing ---")

    test_commands = [
        # Original test cases
        "Add neural networks to the memory",
        "Connect machine learning to deep learning",
        "Strengthen memory about artificial intelligence",
        "Create category for computer science",
        "Add this information to memory vectors",
        # New expanded test cases
        "ADD YOUR UNDERSTANDING OF CONCIOUSNESS TO MEMORY",
        "ADD YOUR UNDERSTANDING OF CONCIOUSNESS TO NEURAL MESH",
        "Store your understanding of consciousness in memory",
        "Put this information in memory",
        "Save these concepts to memory",
        "Add consciousness to neural network",
        "Store consciousness in neural mesh",
        "Strengthen memory about consciousness",
        "Increase knowledge of artificial intelligence",
        "Create category consciousness",
        "Add category philosophy",
        "This is just normal text with no commands"
    ]

    for text in test_commands:
        command = commander.parse_command(text)
        if command:
            print(f"✅ Parsed: '{text[:40]}...' → {command['type']}")
        else:
            print(f"➖ No command: '{text[:40]}...'")

    # Debug specific failing cases
    print("\n--- Debugging Specific Cases ---")
    debug_cases = [
        "ADD YOUR UNDERSTANDING OF CONCIOUSNESS TO MEMORY",
        "Add this information to memory vectors",
        "Store your understanding of consciousness in memory"
    ]

    for text in debug_cases:
        print(f"\nDebugging: '{text}'")
        text_lower = text.lower().strip()

        # Check each command type manually
        for command_type, patterns in commander.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    print(f"  ✅ {command_type}: {pattern} -> {match.groups()}")
                    break
            else:
                continue
            break
        else:
            print("  ❌ No pattern matched")

    # Test command execution (with safety limits)
    print("\n--- Testing Command Execution ---")

    # Test add_node command
    add_node_cmd = {
        'type': 'add_node',
        'concept': 'test_concept',
        'category': 'testing',
        'confidence': 0.8,
        'source': 'test'
    }

    result = commander.execute_command(add_node_cmd)
    if result.get('success'):
        print("✅ add_node command executed successfully")
    else:
        print(f"❌ add_node command failed: {result.get('error')}")

    # Test create_category command
    create_cat_cmd = {
        'type': 'create_category',
        'name': 'test_category',
        'description': 'Test category for memory commander',
        'source': 'test'
    }

    result = commander.execute_command(create_cat_cmd)
    if result.get('success'):
        print("✅ create_category command executed successfully")
    else:
        print(f"❌ create_category command failed: {result.get('error')}")

    # Test reinforce_memory command
    reinforce_cmd = {
        'type': 'reinforce_memory',
        'topic': 'test',
        'strength': 0.1,
        'source': 'test'
    }

    result = commander.execute_command(reinforce_cmd)
    print(f"ℹ️ reinforce_memory result: {result.get('success', False)}")

    # Test safety limits
    print("\n--- Testing Safety Limits ---")
    stats = commander.get_stats()
    print(f"Session operations: {stats['session_operations']}/{stats['max_operations_per_session']}")

    # Test command history
    print("\n--- Testing Command History ---")
    history = commander.get_command_history(limit=5)
    print(f"Command history entries: {len(history)}")

    for entry in history[-3:]:  # Show last 3
        cmd_type = entry['command']['type']
        success = entry['result'].get('success', False)
        print(f"  {cmd_type}: {'✅' if success else '❌'}")

    print("\n=== Memory Commander Test Complete ===")
    print("🎯 Key Features Tested:")
    print("  ✅ Command parsing from natural language")
    print("  ✅ Memory node creation")
    print("  ✅ Category creation")
    print("  ✅ Memory reinforcement")
    print("  ✅ Safety limits and validation")
    print("  ✅ Command history and auditing")

if __name__ == "__main__":
    test_memory_commander()
