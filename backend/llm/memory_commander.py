"""
Memory Commander - Direct AI Memory Manipulation Interface
Enables the AI to directly modify its own memory structures through natural language commands
"""
import logging
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import hashlib

from ..memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class MemoryCommander:
    """
    Direct memory manipulation interface for AI self-modification

    Enables commands like:
    - "Add X to neural network"
    - "Connect A to B in mesh"
    - "Strengthen memory about Y"
    - "Create relationship between X and Y"
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.command_history = []
        self.max_history = 1000

        # Command patterns for natural language parsing - EXPANDED for better detection
        self.command_patterns = {
            'add_node': [
                r'add\s+(\w+(?:\s+\w+)*)\s+to\s+(?:neural\s+)?(?:network|mesh)',
                r'add\s+(?:your\s+)?(?:understanding\s+of\s+)?(\w+(?:\s+\w+)*)\s+to\s+(?:neural\s+)?(?:network|mesh)',
                r'create\s+node\s+(?:\w+\s+)?(\w+(?:\s+\w+)*)',
                r'add\s+(\w+(?:\s+\w+)*)\s+as\s+(?:a\s+)?node',
                r'store\s+(\w+(?:\s+\w+)*)\s+in\s+(?:neural\s+)?(?:network|mesh)',
            ],
            'connect_nodes': [
                r'connect\s+(\w+(?:\s+\w+)*)\s+(?:and|to)\s+(\w+(?:\s+\w+)*)',
                r'link\s+(\w+(?:\s+\w+)*)\s+(?:and|to)\s+(\w+(?:\s+\w+)*)',
                r'create\s+connection\s+between\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)',
            ],
            'reinforce_memory': [
                r'strengthen\s+(?:memory\s+)?(?:about\s+)?(\w+(?:\s+\w+)*)',
                r'reinforce\s+(?:memory\s+)?(?:about\s+)?(\w+(?:\s+\w+)*)',
                r'boost\s+(?:memory\s+)?(?:about\s+)?(\w+(?:\s+\w+)*)',
                r'increase\s+(?:memory\s+)?(?:about\s+)?(\w+(?:\s+\w+)*)',
            ],
            'create_category': [
                r'create\s+category\s+(\w+(?:\s+\w+)*)',
                r'make\s+(?:a\s+)?category\s+(?:\w+\s+)?(\w+(?:\s+\w+)*)',
                r'add\s+category\s+(\w+(?:\s+\w+)*)',
            ],
            'add_vectors': [
                r'add\s+.*?\s+to\s+(?:memory|vectors?)',
                r'store\s+.*?\s+in\s+(?:memory|vectors?)',
                r'put\s+.*?\s+in\s+(?:memory|vectors?)',
                r'save\s+.*?\s+to\s+(?:memory|vectors?)',
            ]
        }

        # Safety limits
        self.max_operations_per_session = 10
        self.session_operations = 0
        self.last_reset = datetime.now()

        logger.info("Memory Commander initialized")

    def parse_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse natural language text for memory manipulation commands

        Args:
            text: Natural language text that may contain commands

        Returns:
            Parsed command dictionary or None if no command found
        """
        text_lower = text.lower().strip()

        # Check each command type
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    command = self._build_command(command_type, match, text)
                    if command:
                        logger.info(f"Parsed command: {command_type} from '{text[:50]}...'")
                        return command

        return None

    def _build_command(self, command_type: str, match: re.Match,
                      original_text: str) -> Optional[Dict[str, Any]]:
        """Build structured command from regex match"""

        if command_type == 'add_node':
            concept = match.group(1)
            return {
                'type': 'add_node',
                'concept': concept,
                'category': self._extract_category(original_text),
                'confidence': 0.8,
                'source': 'ai_command'
            }

        elif command_type == 'connect_nodes':
            node1 = match.group(1)
            node2 = match.group(2)
            return {
                'type': 'connect_nodes',
                'node1': node1,
                'node2': node2,
                'weight': self._extract_weight(original_text),
                'source': 'ai_command'
            }

        elif command_type == 'reinforce_memory':
            topic = match.group(1)
            return {
                'type': 'reinforce_memory',
                'topic': topic,
                'strength': 0.2,
                'source': 'ai_command'
            }

        elif command_type == 'create_category':
            category_name = match.group(1)
            return {
                'type': 'create_category',
                'name': category_name,
                'description': f"Category for {category_name} content",
                'source': 'ai_command'
            }

        elif command_type == 'add_vectors':
            return {
                'type': 'add_vectors',
                'content': original_text,
                'source': 'ai_command'
            }

        return None

    def _extract_category(self, text: str) -> str:
        """Extract category from command context"""
        # Simple category extraction - can be enhanced
        categories = ['technical', 'personal', 'projects', 'learning', 'general']
        text_lower = text.lower()

        for category in categories:
            if category in text_lower:
                return category

        return 'learned'  # Default category for AI-learned content

    def _extract_weight(self, text: str) -> float:
        """Extract connection weight from command context"""
        # Look for strength indicators
        if 'strongly' in text.lower() or 'very' in text.lower():
            return 0.9
        elif 'weakly' in text.lower():
            return 0.3
        else:
            return 0.6  # Default medium strength

    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a parsed memory command

        Args:
            command: Structured command dictionary

        Returns:
            Execution result
        """
        try:
            # Check safety limits
            if not self._check_safety_limits():
                return {
                    'success': False,
                    'error': 'Safety limit exceeded',
                    'command': command
                }

            command_type = command['type']
            logger.info(f"Executing command: {command_type}")

            # Execute based on command type
            if command_type == 'add_node':
                result = self._execute_add_node(command)
            elif command_type == 'connect_nodes':
                result = self._execute_connect_nodes(command)
            elif command_type == 'reinforce_memory':
                result = self._execute_reinforce_memory(command)
            elif command_type == 'create_category':
                result = self._execute_create_category(command)
            elif command_type == 'add_vectors':
                result = self._execute_add_vectors(command)
            else:
                result = {
                    'success': False,
                    'error': f'Unknown command type: {command_type}'
                }

            # Log execution
            self._log_execution(command, result)

            # Update operation count
            if result.get('success', False):
                self.session_operations += 1

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'command': command
            }

    def _execute_add_node(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute add_node command"""
        try:
            concept = command['concept']
            category = command.get('category', 'learned')

            # Create content for the node
            content = f"Concept: {concept}"
            if 'description' in command:
                content += f"\nDescription: {command['description']}"

            # Add to memory as knowledge chunk
            success = self.memory_manager.add_knowledge_chunk(
                text=content,
                metadata={
                    'source': 'ai_memory_command',
                    'category': category,
                    'concept': concept,
                    'confidence': command.get('confidence', 0.5)
                }
            )

            return {
                'success': success,
                'action': 'add_node',
                'concept': concept,
                'category': category
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'action': 'add_node'
            }

    def _execute_connect_nodes(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute connect_nodes command"""
        try:
            node1 = command['node1']
            node2 = command['node2']
            weight = command.get('weight', 0.6)

            # Find nodes in neural mesh and reinforce connection
            # This is a simplified implementation - in practice would need
            # more sophisticated node matching
            self.memory_manager.reinforce_memory(
                query=f"{node1} {node2}",
                selected_chunks=[],  # Would need to find actual chunks
                reinforcement=weight * 0.1
            )

            return {
                'success': True,
                'action': 'connect_nodes',
                'node1': node1,
                'node2': node2,
                'weight': weight
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'action': 'connect_nodes'
            }

    def _execute_reinforce_memory(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reinforce_memory command"""
        try:
            topic = command['topic']
            strength = command.get('strength', 0.2)

            # Search for related memories and reinforce them
            search_results = self.memory_manager.search_memory(topic, max_results=5)

            if search_results['results']:
                # Reinforce the found memories
                self.memory_manager.reinforce_memory(
                    query=topic,
                    selected_chunks=search_results['results'],
                    reinforcement=strength
                )

                return {
                    'success': True,
                    'action': 'reinforce_memory',
                    'topic': topic,
                    'chunks_reinforced': len(search_results['results']),
                    'strength': strength
                }
            else:
                return {
                    'success': False,
                    'error': f'No memories found for topic: {topic}',
                    'action': 'reinforce_memory'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'action': 'reinforce_memory'
            }

    def _execute_create_category(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute create_category command"""
        try:
            category_name = command['name']
            description = command.get('description', f"Category for {category_name}")

            # Create new category
            category_id = self.memory_manager.create_category(
                name=category_name,
                description=description
            )

            return {
                'success': bool(category_id),
                'action': 'create_category',
                'category_name': category_name,
                'category_id': category_id,
                'description': description
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'action': 'create_category'
            }

    def _execute_add_vectors(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute add_vectors command"""
        try:
            content = command['content']

            # Add content as knowledge chunk
            success = self.memory_manager.add_knowledge_chunk(
                text=content,
                metadata={
                    'source': 'ai_memory_command',
                    'category': 'learned',
                    'confidence': 0.7
                }
            )

            return {
                'success': success,
                'action': 'add_vectors',
                'content_length': len(content)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'action': 'add_vectors'
            }

    def _check_safety_limits(self) -> bool:
        """Check if operation is within safety limits"""
        # Reset counter if it's been more than an hour
        now = datetime.now()
        if (now - self.last_reset).seconds > 3600:
            self.session_operations = 0
            self.last_reset = now

        return self.session_operations < self.max_operations_per_session

    def _log_execution(self, command: Dict[str, Any], result: Dict[str, Any]):
        """Log command execution for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'result': result,
            'session_operations': self.session_operations
        }

        self.command_history.append(log_entry)

        # Keep history within limits
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]

        logger.info(f"Memory command executed: {command['type']} - Success: {result.get('success', False)}")

    def get_command_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent command execution history"""
        return self.command_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get commander statistics"""
        successful_commands = [h for h in self.command_history if h['result'].get('success', False)]
        failed_commands = [h for h in self.command_history if not h['result'].get('success', False)]

        command_counts = {}
        for entry in self.command_history:
            cmd_type = entry['command']['type']
            command_counts[cmd_type] = command_counts.get(cmd_type, 0) + 1

        return {
            'total_commands': len(self.command_history),
            'successful_commands': len(successful_commands),
            'failed_commands': len(failed_commands),
            'command_types': command_counts,
            'session_operations': self.session_operations,
            'max_operations_per_session': self.max_operations_per_session
        }

    def reset_session_limits(self):
        """Reset session operation limits (admin function)"""
        self.session_operations = 0
        self.last_reset = datetime.now()
        logger.info("Session operation limits reset")
