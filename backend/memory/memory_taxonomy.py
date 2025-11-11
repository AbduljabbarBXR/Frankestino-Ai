"""
Memory Taxonomy - Formal Memory Type System
Implements Stable/Conversational/Functional memory types with policies and access controls
"""
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta
import time

from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Enumeration of memory types"""
    STABLE = "stable"
    CONVERSATIONAL = "conversational"
    FUNCTIONAL = "functional"


class MemoryTaxonomy:
    """
    Formal memory taxonomy system with type-specific policies and access controls.
    Manages Stable, Conversational, and Functional memory types.
    """

    # Memory type definitions with policies
    TYPE_POLICIES = {
        MemoryType.STABLE: {
            'description': 'Long-term canonical knowledge',
            'retention_policy': {
                'max_age_days': None,  # Never expires
                'consolidation_trigger': 'manual_review',
                'auto_cleanup': False,
                'backup_priority': 'high'
            },
            'ingestion_policy': {
                'allowed_sources': ['curator', 'manual_review'],
                'requires_validation': True,
                'confidence_threshold': 0.9,
                'auto_approve': False
            },
            'access_policy': {
                'read_access': ['frontend', 'curator', 'admin'],
                'write_access': ['curator', 'manual_review'],
                'query_priority': 'high'
            },
            'consolidation_policy': {
                'deduplication': True,
                'summarization': True,
                'cross_referencing': True
            }
        },

        MemoryType.CONVERSATIONAL: {
            'description': 'Session and persona memory',
            'retention_policy': {
                'max_age_hours': 24,  # 1 day default
                'consolidation_trigger': 'usage_based',
                'auto_cleanup': True,
                'backup_priority': 'medium'
            },
            'ingestion_policy': {
                'allowed_sources': ['frontend', 'user_direct'],
                'requires_validation': False,
                'confidence_threshold': 0.5,
                'auto_approve': True
            },
            'access_policy': {
                'read_access': ['frontend', 'curator'],
                'write_access': ['frontend', 'user_direct'],
                'query_priority': 'medium'
            },
            'consolidation_policy': {
                'deduplication': True,
                'summarization': False,
                'cross_referencing': False
            }
        },

        MemoryType.FUNCTIONAL: {
            'description': 'Procedural knowledge and tools',
            'retention_policy': {
                'max_age_days': 90,  # 3 months
                'consolidation_trigger': 'version_control',
                'auto_cleanup': True,
                'backup_priority': 'medium'
            },
            'ingestion_policy': {
                'allowed_sources': ['curator', 'manual', 'code_analysis'],
                'requires_validation': True,
                'confidence_threshold': 0.8,
                'auto_approve': False
            },
            'access_policy': {
                'read_access': ['frontend', 'curator', 'admin'],
                'write_access': ['curator', 'manual', 'code_analysis'],
                'query_priority': 'high'
            },
            'consolidation_policy': {
                'deduplication': True,
                'summarization': False,
                'cross_referencing': True
            }
        }
    }

    def __init__(self):
        """Initialize memory taxonomy system"""
        self.memory_managers = {}  # type -> MemoryManager mapping
        self.type_stats = {mem_type: {} for mem_type in MemoryType}
        self.access_log = []  # Track access patterns

        logger.info("Memory Taxonomy initialized")

    def register_memory_manager(self, memory_type: MemoryType, manager: MemoryManager):
        """Register a memory manager for a specific type"""
        self.memory_managers[memory_type] = manager
        logger.info(f"Registered memory manager for type: {memory_type.value}")

    def get_memory_manager(self, memory_type: MemoryType) -> Optional[MemoryManager]:
        """Get memory manager for a specific type"""
        return self.memory_managers.get(memory_type)

    def get_type_policy(self, memory_type: MemoryType) -> Dict[str, Any]:
        """Get policy configuration for a memory type"""
        return self.TYPE_POLICIES.get(memory_type, {})

    def check_access_permission(self, memory_type: MemoryType, operation: str,
                               source: str) -> bool:
        """
        Check if a source has permission for an operation on a memory type

        Args:
            memory_type: Type of memory
            operation: 'read' or 'write'
            source: Source requesting access ('frontend', 'curator', 'admin', etc.)

        Returns:
            True if access is allowed
        """
        policy = self.get_type_policy(memory_type)
        access_policy = policy.get('access_policy', {})

        if operation == 'read':
            allowed_sources = access_policy.get('read_access', [])
        elif operation == 'write':
            allowed_sources = access_policy.get('write_access', [])
        else:
            return False

        has_access = source in allowed_sources

        # Log access attempt
        self.access_log.append({
            'timestamp': time.time(),
            'memory_type': memory_type.value,
            'operation': operation,
            'source': source,
            'granted': has_access
        })

        # Keep only recent access logs (last 1000)
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]

        if not has_access:
            logger.warning(f"Access denied: {source} attempted {operation} on {memory_type.value}")

        return has_access

    def validate_ingestion_request(self, memory_type: MemoryType, source: str,
                                 content_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate an ingestion request against type policies

        Args:
            memory_type: Target memory type
            source: Source of the content
            content_metadata: Optional metadata about the content

        Returns:
            Validation result with approval status
        """
        policy = self.get_type_policy(memory_type)
        ingestion_policy = policy.get('ingestion_policy', {})

        validation_result = {
            'approved': False,
            'reason': '',
            'confidence_required': ingestion_policy.get('confidence_threshold', 0.5),
            'requires_validation': ingestion_policy.get('requires_validation', False)
        }

        # Check source permission
        if source not in ingestion_policy.get('allowed_sources', []):
            validation_result['reason'] = f"Source '{source}' not allowed for {memory_type.value} memory"
            return validation_result

        # Check auto-approval
        if ingestion_policy.get('auto_approve', False):
            validation_result['approved'] = True
            validation_result['reason'] = 'Auto-approved'
            return validation_result

        # Requires manual validation
        validation_result['approved'] = False
        validation_result['reason'] = 'Requires manual validation'
        return validation_result

    def route_ingestion_request(self, content: Any, memory_type: MemoryType,
                               source: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route an ingestion request to the appropriate memory manager

        Args:
            content: Content to ingest
            memory_type: Target memory type
            source: Source of the content
            metadata: Additional metadata

        Returns:
            Ingestion result
        """
        # Validate the request
        validation = self.validate_ingestion_request(memory_type, source, metadata)
        if not validation['approved']:
            return {
                'success': False,
                'error': validation['reason'],
                'validation_required': True
            }

        # Get the appropriate memory manager
        manager = self.get_memory_manager(memory_type)
        if not manager:
            return {
                'success': False,
                'error': f"No memory manager registered for type: {memory_type.value}"
            }

        # Route based on content type
        try:
            if isinstance(content, str) and content.startswith('/'):  # File path
                result = manager.ingest_document(content, metadata.get('category'))
            elif isinstance(content, dict) and 'messages' in content:  # Conversation
                result = manager.store_conversation(
                    content.get('conversation_id', f"conv_{int(time.time())}"),
                    content['messages'],
                    metadata.get('category')
                )
                result = {'success': result, 'message': 'Conversation stored' if result else 'Failed to store conversation'}
            else:
                return {
                    'success': False,
                    'error': f"Unsupported content type for {memory_type.value} memory"
                }

            # Update type statistics
            self._update_type_stats(memory_type, 'ingestion', result)

            return {
                'success': True,
                'memory_type': memory_type.value,
                'result': result
            }

        except Exception as e:
            logger.error(f"Ingestion routing failed for {memory_type.value}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def query_memory_type(self, query: str, memory_type: MemoryType,
                         source: str, **kwargs) -> Dict[str, Any]:
        """
        Query a specific memory type

        Args:
            query: Search query
            memory_type: Memory type to query
            source: Query source
            **kwargs: Additional query parameters

        Returns:
            Query results
        """
        # Check read access
        if not self.check_access_permission(memory_type, 'read', source):
            return {
                'success': False,
                'error': f"Access denied: {source} cannot read {memory_type.value} memory"
            }

        manager = self.get_memory_manager(memory_type)
        if not manager:
            return {
                'success': False,
                'error': f"No memory manager for type: {memory_type.value}"
            }

        try:
            # Use hybrid search for the specific type
            results = manager.hybrid_search_cached(
                query,
                category=kwargs.get('category'),
                max_results=kwargs.get('max_results', 5),
                use_mesh=kwargs.get('use_mesh', True)
            )

            # Update type statistics
            self._update_type_stats(memory_type, 'query', {
                'query': query,
                'results_count': len(results.get('results', []))
            })

            return {
                'success': True,
                'memory_type': memory_type.value,
                'results': results
            }

        except Exception as e:
            logger.error(f"Query failed for {memory_type.value}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_memory_type_stats(self, memory_type: MemoryType = None) -> Dict[str, Any]:
        """Get statistics for memory types"""
        if memory_type:
            manager = self.get_memory_manager(memory_type)
            if manager:
                base_stats = manager.get_memory_stats()
                type_stats = self.type_stats.get(memory_type, {})

                return {
                    'memory_type': memory_type.value,
                    'policy': self.get_type_policy(memory_type),
                    'stats': base_stats,
                    'usage': type_stats
                }
            else:
                return {'error': f"No manager for type: {memory_type.value}"}

        # Return stats for all types
        all_stats = {}
        for mem_type in MemoryType:
            manager = self.get_memory_manager(mem_type)
            if manager:
                all_stats[mem_type.value] = {
                    'policy': self.get_type_policy(mem_type),
                    'stats': manager.get_memory_stats(),
                    'usage': self.type_stats.get(mem_type, {})
                }

        return all_stats

    def apply_retention_policies(self):
        """Apply retention policies to all memory types"""
        logger.info("Applying memory retention policies...")

        for memory_type in MemoryType:
            policy = self.get_type_policy(memory_type)
            retention = policy.get('retention_policy', {})

            if retention.get('auto_cleanup', False):
                manager = self.get_memory_manager(memory_type)
                if manager:
                    try:
                        # Apply cleanup based on retention policy
                        max_age = retention.get('max_age_days')
                        if max_age:
                            days_old = max_age
                        else:
                            max_age_hours = retention.get('max_age_hours', 24)
                            days_old = max_age_hours / 24

                        manager.cleanup_memory(days_old=days_old)
                        logger.info(f"Applied retention policy to {memory_type.value} memory")

                    except Exception as e:
                        logger.error(f"Failed to apply retention policy for {memory_type.value}: {e}")

    def get_access_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent access log entries"""
        return self.access_log[-limit:] if self.access_log else []

    def _update_type_stats(self, memory_type: MemoryType, operation: str, data: Dict[str, Any]):
        """Update usage statistics for a memory type"""
        if memory_type not in self.type_stats:
            self.type_stats[memory_type] = {}

        stats = self.type_stats[memory_type]

        # Initialize counters if needed
        if 'operations' not in stats:
            stats['operations'] = {}
        if operation not in stats['operations']:
            stats['operations'][operation] = 0

        stats['operations'][operation] += 1
        stats['last_activity'] = time.time()

        # Store additional operation-specific data
        if operation == 'ingestion' and 'chunks_created' in data:
            stats['total_chunks'] = stats.get('total_chunks', 0) + data['chunks_created']
        elif operation == 'query' and 'results_count' in data:
            stats['total_queries'] = stats.get('total_queries', 0) + 1
            stats['total_results_returned'] = stats.get('total_results_returned', 0) + data['results_count']

    def __repr__(self):
        """String representation of the taxonomy"""
        registered_types = [t.value for t in self.memory_managers.keys()]
        return f"MemoryTaxonomy(registered_types={registered_types})"
