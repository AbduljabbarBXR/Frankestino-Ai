"""
Memory Curator - Backend LLM for Memory Processing
Handles summarization, validation, and deduplication of memory content
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json

from .model_loader import GGUFModelLoader
from ..memory.lightweight_processor import LightweightMemoryProcessor
from ..config import settings

logger = logging.getLogger(__name__)


class MemoryCurator:
    """
    Backend curator model for processing and validating memory content.
    Uses Qwen2.5-7B for advanced reasoning tasks: summarization, validation, and quality assessment.
    Part of unified single-model architecture.
    """

    def __init__(self):
        """Initialize the memory curator with Qwen model and lightweight processor"""
        self.model_loader = None
        self.lightweight_processor = None
        self.is_initialized = False

        # Processing queues
        self.pending_summaries = asyncio.Queue()
        self.pending_validations = asyncio.Queue()
        self.processing_stats = {
            'summaries_processed': 0,
            'validations_performed': 0,
            'average_summary_time': 0,
            'average_validation_time': 0
        }

        # Review queue for human-in-the-loop validation
        self.review_queue: List[Dict[str, Any]] = []
        self.review_stats = {
            'total_reviews': 0,
            'accepted': 0,
            'rejected': 0,
            'modified': 0,
            'pending_reviews': 0
        }

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize both lightweight processor and LLM curator model"""
        try:
            logger.info("Initializing Memory Curator with lightweight processor and Qwen...")

            # Initialize lightweight processor first (always available)
            self.lightweight_processor = LightweightMemoryProcessor()
            logger.info("Lightweight processor initialized")

            # Initialize LLM curator (optional, for advanced tasks)
            model_path = Path(settings.backend_model_path)
            if not model_path.exists():
                logger.warning(f"Backend model not found at {model_path}, LLM curator will be disabled")
                self.model_loader = None
            else:
                # Initialize model loader with Qwen optimizations
                self.model_loader = GGUFModelLoader(
                    model_path=str(model_path),
                    model_type="qwen"
                )

                if self.model_loader.load_model():
                    logger.info("LLM Curator initialized successfully with Qwen")
                else:
                    logger.error("Failed to load Qwen backend model")
                    self.model_loader = None

            # Curator is ready if at least lightweight processor works
            self.is_initialized = self.lightweight_processor.is_ready()
            logger.info(f"Memory Curator initialization complete. Lightweight: {self.lightweight_processor.is_ready()}, LLM: {self.model_loader is not None and self.model_loader.is_loaded}")

        except Exception as e:
            logger.error(f"Memory Curator initialization failed: {e}")
            self.is_initialized = False

    def is_ready(self) -> bool:
        """Check if curator is ready for processing"""
        return self.is_initialized and self.model_loader and self.model_loader.is_loaded

    async def summarize_chunk(self, text: str, context: str = None) -> Dict[str, Any]:
        """
        Summarize a memory chunk using lightweight processor first, LLM as fallback

        Args:
            text: The text chunk to summarize
            context: Optional context about the chunk

        Returns:
            Dictionary with summary and metadata
        """
        # Try lightweight processor first
        if self.lightweight_processor and self.lightweight_processor.is_ready():
            try:
                result = self.lightweight_processor.summarize_text(text)
                if result['confidence'] > 0.6:  # Use lightweight result if confident enough
                    # Update stats
                    self.processing_stats['summaries_processed'] += 1
                    self.processing_stats['average_summary_time'] = (
                        (self.processing_stats['average_summary_time'] *
                         (self.processing_stats['summaries_processed'] - 1) +
                         result['processing_time']) / self.processing_stats['summaries_processed']
                    )
                    return result
            except Exception as e:
                logger.debug(f"Lightweight summarization failed, falling back to LLM: {e}")

        # Fallback to LLM if available
        if self.model_loader and self.model_loader.is_loaded:
            try:
                start_time = time.time()

                # Create summarization prompt
                if context:
                    prompt = f"""Summarize the following text in 1-3 concise sentences.
Focus on the key facts, concepts, and information that would be most useful for future reference.

Context: {context}

Text to summarize:
{text}

Summary:"""
                else:
                    prompt = f"""Summarize the following text in 1-3 concise sentences.
Focus on the key facts and information:

{text}

Summary:"""

                # Generate summary
                summary = self.model_loader.generate_text(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.3  # Lower temperature for consistent summaries
                ).strip()

                processing_time = time.time() - start_time

                # Update stats
                self.processing_stats['summaries_processed'] += 1
                self.processing_stats['average_summary_time'] = (
                    (self.processing_stats['average_summary_time'] *
                     (self.processing_stats['summaries_processed'] - 1) +
                     processing_time) / self.processing_stats['summaries_processed']
                )

                logger.debug(f"Generated LLM summary in {processing_time:.2f}s: {summary[:100]}...")

                return {
                    'summary': summary,
                    'method': 'llm_generated',
                    'confidence': 0.8,  # Estimated confidence
                    'processing_time': processing_time,
                    'original_length': len(text),
                    'summary_length': len(summary)
                }

            except Exception as e:
                logger.error(f"LLM summary generation failed: {e}")

        # Final fallback
        logger.warning("All summarization methods failed, using truncation fallback")
        return {
            'summary': text[:200] + "..." if len(text) > 200 else text,
            'method': 'truncation_fallback',
            'confidence': 0.0,
            'processing_time': 0.01,
            'original_length': len(text),
            'summary_length': len(text[:200] + "..." if len(text) > 200 else text)
        }

    # ===== SCAFFOLDING & SUBSTRATE MODEL METHODS =====

    async def extract_relationships(self, text: str, context: str = None) -> List[Dict[str, Any]]:
        """
        Extract semantic relationships from text for neural mesh scaffolding

        Args:
            text: Text to analyze for relationships
            context: Optional context about the text

        Returns:
            List of relationship dictionaries with confidence scores
        """
        if not self.is_ready():
            return []

        try:
            # Create relationship extraction prompt
            if context:
                prompt = f"""Extract semantic relationships from the following text.
Identify entities and their relationships. Format as: ENTITY1 --RELATIONSHIP--> ENTITY2

Context: {context}

Text:
{text}

Extract relationships (one per line, format: "entity1 --relationship--> entity2"):
"""
            else:
                prompt = f"""Extract semantic relationships from the following text.
Identify entities and their relationships. Format as: ENTITY1 --RELATIONSHIP--> ENTITY2

Text:
{text}

Extract relationships (one per line, format: "entity1 --relationship--> entity2"):
"""

            # Generate relationship extraction
            response = self.model_loader.generate_text(
                prompt=prompt,
                max_tokens=200,
                temperature=0.2  # Low temperature for consistent extraction
            ).strip()

            relationships = []
            lines = response.split('\n')

            for line in lines:
                line = line.strip()
                if '--' in line and '-->' in line:
                    try:
                        # Parse format: "entity1 --relationship--> entity2"
                        parts = line.split('--')
                        if len(parts) >= 3:
                            entity1 = parts[0].strip()
                            relationship = parts[1].strip()
                            entity2 = parts[2].replace('>', '').strip()

                            if entity1 and relationship and entity2:
                                # Determine relationship type
                                relationship_type = self._categorize_relationship(relationship)

                                relationships.append({
                                    'entity1': entity1,
                                    'entity2': entity2,
                                    'relationship': relationship,
                                    'relationship_type': relationship_type,
                                    'confidence': 0.7,  # Base confidence for LLM extraction
                                    'source_text': text[:100] + "..." if len(text) > 100 else text
                                })
                    except Exception as e:
                        logger.debug(f"Failed to parse relationship line: {line} - {e}")
                        continue

            logger.debug(f"Extracted {len(relationships)} relationships from text")
            return relationships

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

    def _categorize_relationship(self, relationship: str) -> str:
        """
        Categorize a relationship string into predefined types

        Args:
            relationship: Relationship description

        Returns:
            Relationship category
        """
        relationship_lower = relationship.lower()

        # Geographic relationships
        if any(word in relationship_lower for word in ['capital', 'located', 'in', 'city', 'country', 'continent']):
            return 'geographic'

        # Causal relationships
        elif any(word in relationship_lower for word in ['causes', 'leads', 'results', 'because', 'due to']):
            return 'causal'

        # Definitional relationships
        elif any(word in relationship_lower for word in ['is', 'are', 'means', 'defined as', 'refers to']):
            return 'definitional'

        # Temporal relationships
        elif any(word in relationship_lower for word in ['before', 'after', 'during', 'when', 'time']):
            return 'temporal'

        # Associative relationships (default)
        else:
            return 'associative'

    async def detect_duplicates(self, new_chunk: Dict[str, Any],
                               existing_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect duplicate or very similar chunks

        Args:
            new_chunk: New chunk to check
            existing_chunks: Existing chunks to compare against

        Returns:
            List of potential duplicates with similarity scores
        """
        if not self.is_ready() or not existing_chunks:
            return []

        try:
            duplicates = []
            new_text = new_chunk.get('text', '')

            for existing in existing_chunks[:10]:  # Limit comparisons
                existing_text = existing.get('text', '')

                # Simple similarity check prompt
                prompt = f"""Compare these two texts and rate their similarity from 0-1.
0 = completely different, 1 = nearly identical or duplicate.

Text 1:
{new_text}

Text 2:
{existing_text}

Similarity score (just the number):"""

                response = self.model_loader.generate_text(
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.1
                ).strip()

                try:
                    similarity = float(response)
                    if similarity > 0.8:  # High similarity threshold
                        duplicates.append({
                            'existing_chunk': existing,
                            'similarity_score': similarity,
                            'reason': 'high_text_similarity'
                        })
                except ValueError:
                    continue

            return duplicates

        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return []

    async def assess_memory_quality(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the overall quality of a memory chunk

        Args:
            chunk: Memory chunk to assess

        Returns:
            Quality assessment
        """
        if not self.is_ready():
            return {
                'quality_score': 0.5,
                'method': 'no_assessment',
                'reason': 'Curator model not available'
            }

        try:
            text = chunk.get('text', '')

            prompt = f"""Assess the quality of this text as a memory chunk.
Consider: informativeness, clarity, usefulness for future reference.

Rate from 0-1 where:
0 = useless/low quality
1 = highly informative and well-structured

Text:
{text}

Quality score and brief reason:"""

            response = self.model_loader.generate_text(
                prompt=prompt,
                max_tokens=100,
                temperature=0.2
            ).strip()

            # Parse response
            try:
                lines = response.split('\n', 1)
                score = float(lines[0].strip())
                reason = lines[1].strip() if len(lines) > 1 else "No reason provided"
            except:
                score = 0.5
                reason = f"Unparseable response: {response}"

            return {
                'quality_score': score,
                'reason': reason,
                'method': 'llm_assessment'
            }

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                'quality_score': 0.5,
                'method': 'error_fallback',
                'error': str(e)
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get curator processing statistics"""
        return {
            **self.processing_stats,
            'is_ready': self.is_ready(),
            'model_loaded': self.is_ready()
        }

    async def process_memory_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of memory chunks through the curator pipeline

        Args:
            chunks: List of memory chunks to process

        Returns:
            Processed chunks with curator enhancements
        """
        processed_chunks = []

        for chunk in chunks:
            try:
                # Summarize
                summary_result = await self.summarize_chunk(chunk.get('text', ''))
                chunk['summary'] = summary_result['summary']
                chunk['summary_metadata'] = summary_result

                # Validate
                validation_result = await self.validate_memory_chunk(chunk)
                chunk['validation'] = validation_result

                # Assess quality
                quality_result = await self.assess_memory_quality(chunk)
                chunk['quality'] = quality_result

                processed_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Failed to process chunk: {e}")
                chunk['processing_error'] = str(e)
                processed_chunks.append(chunk)

        logger.info(f"Processed {len(processed_chunks)} chunks through curator pipeline")
        return processed_chunks

    def submit_for_review(self, chunk: Dict[str, Any], reason: str = "quality_check",
                         priority: str = "normal") -> str:
        """
        Submit a memory chunk for human review

        Args:
            chunk: Memory chunk to review
            reason: Reason for review (quality_check, validation_failed, etc.)
            priority: Review priority (low, normal, high, urgent)

        Returns:
            Review ID for tracking
        """
        review_id = f"review_{int(time.time())}_{len(self.review_queue)}"

        review_item = {
            'review_id': review_id,
            'chunk': chunk,
            'reason': reason,
            'priority': priority,
            'submitted_at': time.time(),
            'status': 'pending',
            'reviewer': None,
            'decision': None,
            'feedback': None,
            'reviewed_at': None
        }

        self.review_queue.append(review_item)
        self.review_stats['pending_reviews'] = len(self.review_queue)

        logger.info(f"Submitted chunk for review: {review_id} (reason: {reason}, priority: {priority})")
        return review_id

    def get_pending_reviews(self, limit: int = 50, priority_filter: str = None) -> List[Dict[str, Any]]:
        """
        Get pending reviews for human review

        Args:
            limit: Maximum number of reviews to return
            priority_filter: Filter by priority (low, normal, high, urgent)

        Returns:
            List of pending review items
        """
        pending_reviews = [r for r in self.review_queue if r['status'] == 'pending']

        # Filter by priority if specified
        if priority_filter:
            pending_reviews = [r for r in pending_reviews if r['priority'] == priority_filter]

        # Sort by priority and submission time
        priority_order = {'urgent': 0, 'high': 1, 'normal': 2, 'low': 3}
        pending_reviews.sort(key=lambda x: (priority_order.get(x['priority'], 2), x['submitted_at']))

        return pending_reviews[:limit]

    def process_review_decision(self, review_id: str, decision: str,
                               reviewer: str = "unknown", feedback: str = None) -> bool:
        """
        Process a human review decision

        Args:
            review_id: ID of the review to process
            decision: Decision (accept, reject, modify)
            reviewer: Name/ID of the reviewer
            feedback: Optional feedback or modifications

        Returns:
            True if decision was processed successfully
        """
        # Find the review item
        review_item = None
        for item in self.review_queue:
            if item['review_id'] == review_id:
                review_item = item
                break

        if not review_item:
            logger.error(f"Review item not found: {review_id}")
            return False

        # Validate decision
        valid_decisions = ['accept', 'reject', 'modify']
        if decision not in valid_decisions:
            logger.error(f"Invalid decision: {decision}")
            return False

        # Update review item
        review_item['status'] = 'completed'
        review_item['decision'] = decision
        review_item['reviewer'] = reviewer
        review_item['feedback'] = feedback
        review_item['reviewed_at'] = time.time()

        # Update statistics
        self.review_stats['total_reviews'] += 1
        self.review_stats[decision + 'ed'] += 1  # accepted, rejected, modified
        self.review_stats['pending_reviews'] = len([r for r in self.review_queue if r['status'] == 'pending'])

        logger.info(f"Processed review decision: {review_id} -> {decision} by {reviewer}")

        # Handle the decision
        if decision == 'accept':
            self._handle_accept_decision(review_item)
        elif decision == 'reject':
            self._handle_reject_decision(review_item)
        elif decision == 'modify':
            self._handle_modify_decision(review_item, feedback)

        return True

    def _handle_accept_decision(self, review_item: Dict[str, Any]):
        """Handle acceptance of a memory chunk"""
        chunk = review_item['chunk']

        # Mark chunk as human-approved
        chunk['human_approved'] = True
        chunk['approved_at'] = review_item['reviewed_at']
        chunk['approved_by'] = review_item['reviewer']

        # Could trigger promotion to stable memory here
        # For now, just log the acceptance
        logger.info(f"Memory chunk accepted: {chunk.get('id', 'unknown')}")

    def _handle_reject_decision(self, review_item: Dict[str, Any]):
        """Handle rejection of a memory chunk"""
        chunk = review_item['chunk']

        # Mark chunk as rejected
        chunk['human_rejected'] = True
        chunk['rejected_at'] = review_item['reviewed_at']
        chunk['rejected_by'] = review_item['reviewer']
        chunk['rejection_reason'] = review_item.get('feedback', 'No reason provided')

        # Could trigger removal from memory here
        logger.info(f"Memory chunk rejected: {chunk.get('id', 'unknown')} - {review_item.get('feedback', 'No reason')}")

    def _handle_modify_decision(self, review_item: Dict[str, Any], feedback: str):
        """Handle modification of a memory chunk"""
        chunk = review_item['chunk']

        # Store modification feedback
        chunk['human_modified'] = True
        chunk['modified_at'] = review_item['reviewed_at']
        chunk['modified_by'] = review_item['reviewer']
        chunk['modification_feedback'] = feedback

        # Parse feedback to extract suggested changes
        # This could be enhanced with LLM parsing of the feedback
        if feedback:
            # Simple parsing - look for common patterns
            if 'summary:' in feedback.lower():
                # Extract new summary
                try:
                    summary_part = feedback.lower().split('summary:')[1].strip()
                    if summary_part:
                        chunk['original_summary'] = chunk.get('summary', '')
                        chunk['summary'] = summary_part[:500]  # Limit length
                        logger.info(f"Updated summary for chunk: {chunk.get('id', 'unknown')}")
                except:
                    logger.warning(f"Could not parse summary from feedback: {feedback}")

        logger.info(f"Memory chunk modified: {chunk.get('id', 'unknown')} - {feedback[:100]}...")

    def get_review_stats(self) -> Dict[str, Any]:
        """Get review statistics"""
        return {
            **self.review_stats,
            'queue_length': len(self.review_queue),
            'pending_reviews': len([r for r in self.review_queue if r['status'] == 'pending']),
            'completed_reviews': len([r for r in self.review_queue if r['status'] == 'completed'])
        }

    def get_review_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get review history"""
        # Return completed reviews, most recent first
        completed_reviews = [r for r in self.review_queue if r['status'] == 'completed']
        completed_reviews.sort(key=lambda x: x['reviewed_at'], reverse=True)
        return completed_reviews[:limit]

    def cleanup_old_reviews(self, max_age_days: int = 30):
        """Clean up old completed reviews"""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        # Keep pending reviews and recent completed ones
        self.review_queue = [
            r for r in self.review_queue
            if r['status'] == 'pending' or (r.get('reviewed_at', 0) > cutoff_time)
        ]

        logger.info(f"Cleaned up old reviews, {len(self.review_queue)} reviews remaining")

    async def extract_relationships(self, text: str, context: str = None) -> List[Dict[str, Any]]:
        """
        Extract semantic relationships from text using LLM analysis
        Part of Scaffolding & Substrate Model - Phase 1

        Args:
            text: Text to analyze for relationships
            context: Optional context about the text

        Returns:
            List of extracted relationships with confidence scores
        """
        if not self.is_ready():
            logger.warning("Memory Curator not ready for relationship extraction")
            return []

        try:
            # Create relationship extraction prompt
            if context:
                prompt = f"""Extract semantic relationships from the following text.
Identify entities/concepts and their relationships. For each relationship, provide:

1. Source entity/concept
2. Target entity/concept
3. Relationship label (e.g., "is_capital_of", "contains", "provides", "is_located_in")
4. Relationship type (geographic, causal, definitional, temporal, associative)
5. Confidence score (0.0-1.0)

Context: {context}

Text:
{text}

Format: source|target|label|type|confidence
One relationship per line. Only output relationships, no other text:"""
            else:
                prompt = f"""Extract semantic relationships from the following text.
Identify entities/concepts and their relationships. For each relationship, provide:

1. Source entity/concept
2. Target entity/concept
3. Relationship label (e.g., "is_capital_of", "contains", "provides", "is_located_in")
4. Relationship type (geographic, causal, definitional, temporal, associative)
5. Confidence score (0.0-1.0)

Text:
{text}

Format: source|target|label|type|confidence
One relationship per line. Only output relationships, no other text:"""

            # Generate relationship extraction
            response = self.model_loader.generate_text(
                prompt=prompt,
                max_tokens=300,
                temperature=0.2  # Balanced creativity and consistency
            ).strip()

            relationships = []
            lines = response.split('\n')

            for line in lines:
                line = line.strip()
                if not line or '|' not in line:
                    continue

                try:
                    parts = line.split('|', 4)
                    if len(parts) == 5:
                        source, target, label, rel_type, confidence_str = parts

                        # Clean and validate
                        source = source.strip()
                        target = target.strip()
                        label = label.strip()
                        rel_type = rel_type.strip()
                        confidence = float(confidence_str.strip())

                        # Validate relationship type
                        valid_types = ['geographic', 'causal', 'definitional', 'temporal', 'associative']
                        if rel_type not in valid_types:
                            rel_type = 'associative'  # Default fallback

                        # Only include high-confidence relationships
                        if confidence >= 0.3:
                            relationships.append({
                                'source': source,
                                'target': target,
                                'relationship_label': label,
                                'relationship_type': rel_type,
                                'confidence': confidence,
                                'extracted_from': text[:100] + "..." if len(text) > 100 else text
                            })

                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse relationship line: {line} - {e}")
                    continue

            logger.debug(f"Extracted {len(relationships)} relationships from text")
            return relationships

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

    async def create_semantic_edges(self, relationships: List[Dict[str, Any]],
                                   neural_mesh, node_mappings: Dict[str, str]) -> int:
        """
        Create semantic edges in the neural mesh from extracted relationships
        Part of Scaffolding & Substrate Model - Phase 1

        Args:
            relationships: List of extracted relationships
            neural_mesh: NeuralMesh instance to add edges to
            node_mappings: Mapping from entity names to node IDs

        Returns:
            Number of edges created
        """
        edges_created = 0

        for rel in relationships:
            try:
                source_name = rel['source']
                target_name = rel['target']

                # Find or create node mappings
                source_node_id = node_mappings.get(source_name)
                target_node_id = node_mappings.get(target_name)

                if not source_node_id or not target_node_id:
                    logger.debug(f"Missing node mapping for relationship: {source_name} -> {target_name}")
                    continue

                # Create semantic edge
                success = neural_mesh.add_semantic_edge(
                    source_id=source_node_id,
                    target_id=target_node_id,
                    relationship_label=rel['relationship_label'],
                    relationship_type=rel['relationship_type'],
                    confidence=rel['confidence'],
                    weight=min(0.8, rel['confidence'] * 0.8),  # Scale weight by confidence
                    semantic_context={
                        'extraction_source': 'llm_curator',
                        'original_text': rel['extracted_from'],
                        'extraction_timestamp': time.time()
                    }
                )

                if success:
                    edges_created += 1
                    logger.debug(f"Created semantic edge: {source_name} --[{rel['relationship_label']}]--> {target_name}")
                else:
                    logger.debug(f"Failed to create semantic edge: {source_name} -> {target_name}")

            except Exception as e:
                logger.error(f"Failed to create semantic edge for relationship {rel}: {e}")
                continue

        logger.info(f"Created {edges_created} semantic edges from {len(relationships)} relationships")
        return edges_created

    async def process_relationships_batch(self, chunks: List[Dict[str, Any]],
                                        neural_mesh, node_mappings: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a batch of chunks for relationship extraction and edge creation
        Part of Scaffolding & Substrate Model - Phase 1

        Args:
            chunks: List of memory chunks to process
            neural_mesh: NeuralMesh instance
            node_mappings: Mapping from entity names to node IDs

        Returns:
            Processing statistics
        """
        total_relationships = 0
        total_edges_created = 0
        processed_chunks = 0

        for chunk in chunks:
            try:
                text = chunk.get('text', '')
                if not text:
                    continue

                # Extract relationships from this chunk
                relationships = await self.extract_relationships(text, chunk.get('context'))

                if relationships:
                    # Create semantic edges
                    edges_created = await self.create_semantic_edges(
                        relationships, neural_mesh, node_mappings
                    )

                    total_relationships += len(relationships)
                    total_edges_created += edges_created

                    # Store relationship metadata in chunk
                    chunk['relationships_extracted'] = len(relationships)
                    chunk['semantic_edges_created'] = edges_created
                    chunk['extracted_relationships'] = relationships

                processed_chunks += 1

            except Exception as e:
                logger.error(f"Failed to process relationships for chunk: {e}")
                continue

        stats = {
            'chunks_processed': processed_chunks,
            'total_relationships_extracted': total_relationships,
            'total_semantic_edges_created': total_edges_created,
            'average_relationships_per_chunk': total_relationships / max(processed_chunks, 1)
        }

        logger.info(f"Relationship processing complete: {stats}")
        return stats

    def auto_submit_low_confidence_chunks(self, confidence_threshold: float = 0.7):
        """
        Automatically submit low-confidence chunks for human review

        Args:
            confidence_threshold: Threshold below which chunks are submitted for review
        """
        # This would be called periodically to check recently processed chunks
        # For now, it's a placeholder for the auto-review submission logic
        logger.info(f"Auto-review submission with threshold {confidence_threshold} - not yet implemented")

    def __del__(self):
        """Cleanup on deletion"""
        if self.model_loader:
            try:
                self.model_loader.unload_model()
            except:
                pass
