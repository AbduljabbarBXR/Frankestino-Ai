"""
Hierarchical Memory Tree Implementation
Organizes documents by topic hierarchy for efficient filtering
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """A node in the hierarchical memory tree"""
    id: str
    name: str
    description: str = ""
    parent_id: Optional[str] = None
    children: List[str] = None  # List of child node IDs
    document_ids: List[str] = None  # Documents attached to this node
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.document_ids is None:
            self.document_ids = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryNode':
        """Create from dictionary"""
        return cls(**data)


class HierarchicalMemory:
    """Hierarchical tree structure for organizing memory by topics"""

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize hierarchical memory

        Args:
            storage_path: Path to save/load hierarchy
        """
        self.storage_path = storage_path or settings.data_dir / "hierarchy" / "memory_tree.json"
        self.nodes: Dict[str, MemoryNode] = {}
        self.root_id: Optional[str] = None

        # Create storage directory
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing hierarchy
        self._load_hierarchy()

    def _load_hierarchy(self):
        """Load hierarchy from disk"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.root_id = data.get('root_id')
                self.nodes = {}

                for node_data in data.get('nodes', []):
                    node = MemoryNode.from_dict(node_data)
                    self.nodes[node.id] = node

                logger.info(f"Loaded hierarchy with {len(self.nodes)} nodes")
            except Exception as e:
                logger.warning(f"Failed to load hierarchy: {e}")
                self._create_default_hierarchy()
        else:
            self._create_default_hierarchy()

    def _create_default_hierarchy(self):
        """Create a basic default hierarchy"""
        # Create root node
        root = MemoryNode(
            id="root",
            name="Knowledge Base",
            description="Root of the knowledge hierarchy"
        )
        self.nodes["root"] = root
        self.root_id = "root"

        # Create some basic categories
        categories = [
            ("general", "General Knowledge", "General topics and miscellaneous information"),
            ("technical", "Technical", "Technical documentation and specifications"),
            ("personal", "Personal", "Personal notes and experiences"),
            ("projects", "Projects", "Project-related information and documentation")
        ]

        for cat_id, name, desc in categories:
            node = MemoryNode(
                id=cat_id,
                name=name,
                description=desc,
                parent_id="root"
            )
            self.nodes[cat_id] = node
            root.children.append(cat_id)

        self._save_hierarchy()
        logger.info("Created default hierarchy")

    def _save_hierarchy(self):
        """Save hierarchy to disk"""
        try:
            data = {
                'root_id': self.root_id,
                'nodes': [node.to_dict() for node in self.nodes.values()]
            }

            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug("Hierarchy saved to disk")
        except Exception as e:
            logger.error(f"Failed to save hierarchy: {e}")

    def add_node(self, name: str, description: str = "", parent_id: str = None,
                node_id: str = None) -> str:
        """
        Add a new node to the hierarchy

        Args:
            name: Node name
            description: Node description
            parent_id: Parent node ID (defaults to root)
            node_id: Custom node ID (auto-generated if None)

        Returns:
            New node ID
        """
        if parent_id is None:
            parent_id = self.root_id

        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} does not exist")

        if node_id is None:
            # Generate unique ID
            base_id = name.lower().replace(' ', '_').replace('-', '_')
            node_id = base_id
            counter = 1
            while node_id in self.nodes:
                node_id = f"{base_id}_{counter}"
                counter += 1

        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")

        # Create new node
        node = MemoryNode(
            id=node_id,
            name=name,
            description=description,
            parent_id=parent_id
        )

        self.nodes[node_id] = node

        # Add to parent's children
        if parent_id:
            self.nodes[parent_id].children.append(node_id)

        self._save_hierarchy()
        logger.info(f"Added node: {node_id}")
        return node_id

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[MemoryNode]:
        """Get all child nodes of a given node"""
        if node_id not in self.nodes:
            return []

        children = []
        for child_id in self.nodes[node_id].children:
            if child_id in self.nodes:
                children.append(self.nodes[child_id])

        return children

    def get_path_to_root(self, node_id: str) -> List[MemoryNode]:
        """Get the path from a node to the root"""
        if node_id not in self.nodes:
            return []

        path = []
        current_id = node_id

        while current_id:
            node = self.nodes[current_id]
            path.append(node)
            current_id = node.parent_id

        return path[::-1]  # Reverse to get root-to-node order

    def find_node_by_name(self, name: str, case_sensitive: bool = False) -> Optional[MemoryNode]:
        """Find a node by name"""
        for node in self.nodes.values():
            node_name = node.name if case_sensitive else node.name.lower()
            search_name = name if case_sensitive else name.lower()

            if node_name == search_name:
                return node
        return None

    def attach_document(self, node_id: str, document_id: str):
        """Attach a document to a node"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")

        if document_id not in self.nodes[node_id].document_ids:
            self.nodes[node_id].document_ids.append(document_id)
            self._save_hierarchy()
            logger.debug(f"Attached document {document_id} to node {node_id}")

    def detach_document(self, node_id: str, document_id: str):
        """Detach a document from a node"""
        if node_id in self.nodes and document_id in self.nodes[node_id].document_ids:
            self.nodes[node_id].document_ids.remove(document_id)
            self._save_hierarchy()
            logger.debug(f"Detached document {document_id} from node {node_id}")

    def get_all_documents(self, node_id: str) -> Set[str]:
        """Get all documents in a node and its subtree"""
        if node_id not in self.nodes:
            return set()

        documents = set(self.nodes[node_id].document_ids)

        # Recursively get documents from children
        for child_id in self.nodes[node_id].children:
            documents.update(self.get_all_documents(child_id))

        return documents

    def search_nodes(self, query: str, max_results: int = 10) -> List[MemoryNode]:
        """Search nodes by name or description"""
        query_lower = query.lower()
        results = []

        for node in self.nodes.values():
            # Search in name and description
            if (query_lower in node.name.lower() or
                query_lower in node.description.lower()):
                results.append(node)

                if len(results) >= max_results:
                    break

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get hierarchy statistics"""
        total_nodes = len(self.nodes)
        total_documents = sum(len(node.document_ids) for node in self.nodes.values())

        # Calculate depth
        max_depth = 0
        if self.root_id:
            max_depth = self._calculate_max_depth(self.root_id)

        return {
            "total_nodes": total_nodes,
            "total_documents": total_documents,
            "max_depth": max_depth,
            "root_id": self.root_id
        }

    def _calculate_max_depth(self, node_id: str, current_depth: int = 0) -> int:
        """Calculate maximum depth from a node"""
        if node_id not in self.nodes or not self.nodes[node_id].children:
            return current_depth

        max_child_depth = current_depth
        for child_id in self.nodes[node_id].children:
            child_depth = self._calculate_max_depth(child_id, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    def __len__(self) -> int:
        """Return number of nodes"""
        return len(self.nodes)
