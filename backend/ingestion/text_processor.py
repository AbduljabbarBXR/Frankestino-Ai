"""
Text Processing and Chunking Utilities
Handles text preprocessing and intelligent chunking
"""
import re
import logging
from typing import List, Dict, Any, Optional
from ..config import settings

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text preprocessing and chunking"""

    def __init__(self):
        self.sentence_endings = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r'\b\w+\b')

        # Enhanced word processing for weighting
        self.stop_words = self._load_stop_words()
        self.common_words = self._load_common_words()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove control characters but keep newlines
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split on sentence endings
        sentences = self.sentence_endings.split(text)

        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def smart_chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 64) -> List[str]:
        """
        Smart chunking with proper sentence boundary detection

        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks

        Returns:
            List of text chunks
        """
        try:
            # Try to use NLTK for better sentence tokenization
            import nltk
            # Download punkt if not available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

            sentences = nltk.sent_tokenize(text)
        except ImportError:
            # Fallback to regex-based sentence splitting
            logger.warning("NLTK not available, using regex sentence splitting")
            sentences = self.split_into_sentences(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) > chunk_size and current_chunk:
                # Current chunk is full, save it
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap from current chunk
                if overlap > 0:
                    # Take overlap from the end of current chunk
                    words = current_chunk.split()
                    overlap_text = ""
                    for word in reversed(words):
                        if len(overlap_text + " " + word) <= overlap:
                            overlap_text = word + " " + overlap_text
                        else:
                            break
                    current_chunk = overlap_text.strip()
                else:
                    current_chunk = ""

            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.debug(f"Smart chunking produced {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation)

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    def intelligent_chunk(self, text: str, max_chunk_size: int = None,
                         overlap: int = None) -> List[Dict[str, Any]]:
        """
        Intelligently chunk text with sentence awareness

        Args:
            text: Input text
            max_chunk_size: Maximum characters per chunk
            overlap: Characters to overlap

        Returns:
            List of chunk dictionaries
        """
        max_chunk_size = max_chunk_size or settings.max_chunk_size
        overlap = overlap or settings.chunk_overlap

        text = self.clean_text(text)
        if not text:
            return []

        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = ""
        current_sentences = []

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'sentences': current_sentences.copy(),
                    'metadata': {
                        'chunk_type': 'sentence_aware',
                        'sentence_count': len(current_sentences),
                        'char_count': len(current_chunk)
                    }
                })

                # Start new chunk with overlap
                if overlap > 0 and current_sentences:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_text = ""

                    for sent in reversed(current_sentences):
                        if len(overlap_text + " " + sent) <= overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_text = sent + " " + overlap_text
                        else:
                            break

                    current_chunk = overlap_text.strip()
                    current_sentences = overlap_sentences
                else:
                    current_chunk = ""
                    current_sentences = []

            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

            current_sentences.append(sentence)

        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'sentences': current_sentences,
                'metadata': {
                    'chunk_type': 'sentence_aware',
                    'sentence_count': len(current_sentences),
                    'char_count': len(current_chunk)
                }
            })

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def fixed_size_chunk(self, text: str, chunk_size: int = None,
                        overlap: int = None) -> List[Dict[str, Any]]:
        """
        Split text into fixed-size chunks

        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        chunk_size = chunk_size or settings.max_chunk_size
        overlap = overlap or settings.chunk_overlap

        text = self.clean_text(text)
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at word boundaries
            if end < len(text):
                # Look for word boundary within last 50 characters
                word_boundary = text.rfind(' ', max(start, end - 50), end)
                if word_boundary > start:
                    end = word_boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chunk_type': 'fixed_size',
                        'start_pos': start,
                        'end_pos': end,
                        'char_count': len(chunk_text)
                    }
                })

            start = end - overlap

            # Prevent infinite loop
            if start >= len(text) - 1:
                break

        logger.debug(f"Split text into {len(chunks)} fixed-size chunks")
        return chunks

    def semantic_chunk(self, text: str, max_chunk_size: int = None,
                       overlap: int = None) -> List[Dict[str, Any]]:
        """
        Semantic chunking prioritizing paragraph and sentence boundaries

        Strategy:
        1. Split on paragraph boundaries (double newlines)
        2. For long paragraphs, split on sentence boundaries
        3. Maintain semantic coherence and context

        Args:
            text: Input text
            max_chunk_size: Maximum characters per chunk
            overlap: Characters to overlap

        Returns:
            List of chunk dictionaries with semantic metadata
        """
        max_chunk_size = max_chunk_size or 1024  # Reduced from 2048 for better semantics
        overlap = overlap or settings.chunk_overlap

        text = self.clean_text(text)
        if not text:
            return []

        # Step 1: Split into paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""
        current_paragraphs = []
        current_sentences = []

        for paragraph in paragraphs:
            # Get sentences in this paragraph
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                sentences = nltk.sent_tokenize(paragraph)
            except ImportError:
                sentences = self.split_into_sentences(paragraph)

            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            if len(potential_chunk) > max_chunk_size and current_chunk:
                # Current chunk is full, save it
                chunks.append({
                    'text': current_chunk.strip(),
                    'sentences': current_sentences.copy(),
                    'paragraphs': current_paragraphs.copy(),
                    'metadata': {
                        'chunk_type': 'semantic_paragraph',
                        'paragraph_count': len(current_paragraphs),
                        'sentence_count': len(current_sentences),
                        'char_count': len(current_chunk),
                        'semantic_boundary': 'paragraph'
                    }
                })

                # Start new chunk with overlap
                if overlap > 0 and current_sentences:
                    # Keep last sentence for overlap
                    overlap_sentences = []
                    overlap_text = ""

                    for sent in reversed(current_sentences[-2:]):  # Keep last 1-2 sentences
                        if len(overlap_text + " " + sent) <= overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_text = sent + " " + overlap_text
                        else:
                            break

                    current_chunk = overlap_text.strip()
                    current_sentences = overlap_sentences
                    current_paragraphs = []  # Reset paragraphs for new chunk
                else:
                    current_chunk = ""
                    current_sentences = []
                    current_paragraphs = []

            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

            current_paragraphs.append(paragraph)
            current_sentences.extend(sentences)

        # Handle final chunk
        if current_chunk:
            # Check if final chunk is too large - split by sentences if needed
            if len(current_chunk) > max_chunk_size * 1.2:
                # Split oversized final chunk by sentences
                sentence_chunks = self._split_oversized_chunk_by_sentences(
                    current_chunk, current_sentences, max_chunk_size, overlap
                )
                chunks.extend(sentence_chunks)
            else:
                chunks.append({
                    'text': current_chunk.strip(),
                    'sentences': current_sentences,
                    'paragraphs': current_paragraphs,
                    'metadata': {
                        'chunk_type': 'semantic_paragraph',
                        'paragraph_count': len(current_paragraphs),
                        'sentence_count': len(current_sentences),
                        'char_count': len(current_chunk),
                        'semantic_boundary': 'paragraph'
                    }
                })

        # Final validation: ensure no chunks are too small or meaningless
        validated_chunks = []
        for chunk in chunks:
            if len(chunk['text']) < 50:  # Too small
                if validated_chunks:  # Merge with previous chunk
                    validated_chunks[-1]['text'] += " " + chunk['text']
                    validated_chunks[-1]['sentences'].extend(chunk.get('sentences', []))
                    validated_chunks[-1]['paragraphs'].extend(chunk.get('paragraphs', []))
                    validated_chunks[-1]['metadata']['char_count'] = len(validated_chunks[-1]['text'])
                    validated_chunks[-1]['metadata']['sentence_count'] = len(validated_chunks[-1]['sentences'])
                    validated_chunks[-1]['metadata']['paragraph_count'] = len(validated_chunks[-1]['paragraphs'])
                continue

            # Enhance chunk with word weighting information
            enhanced_chunk = self.enhance_chunk_with_word_weights(chunk)
            validated_chunks.append(enhanced_chunk)

        logger.info(f"Semantic chunking produced {len(validated_chunks)} enhanced chunks from {len(paragraphs)} paragraphs")
        return validated_chunks

    def _split_oversized_chunk_by_sentences(self, text: str, sentences: List[str],
                                          max_chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Split an oversized chunk by sentence boundaries"""
        chunks = []
        current_chunk = ""
        current_sentences = []

        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'sentences': current_sentences.copy(),
                    'paragraphs': [],  # Sentence-level chunks don't track paragraphs
                    'metadata': {
                        'chunk_type': 'semantic_sentence',
                        'paragraph_count': 0,
                        'sentence_count': len(current_sentences),
                        'char_count': len(current_chunk),
                        'semantic_boundary': 'sentence'
                    }
                })

                # Start new chunk with overlap
                if overlap > 0 and current_sentences:
                    overlap_sentences = []
                    overlap_text = ""

                    for sent in reversed(current_sentences[-1:]):  # Keep last sentence
                        if len(overlap_text + " " + sent) <= overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_text = sent + " " + overlap_text
                        else:
                            break

                    current_chunk = overlap_text.strip()
                    current_sentences = overlap_sentences
                else:
                    current_chunk = ""
                    current_sentences = []

            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

            current_sentences.append(sentence)

        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'sentences': current_sentences,
                'paragraphs': [],
                'metadata': {
                    'chunk_type': 'semantic_sentence',
                    'paragraph_count': 0,
                    'sentence_count': len(current_sentences),
                    'char_count': len(current_chunk),
                    'semantic_boundary': 'sentence'
                }
            })

        return chunks

    def hybrid_chunk(self, text: str, max_chunk_size: int = None,
                    overlap: int = None) -> List[Dict[str, Any]]:
        """
        Hybrid chunking: sentence-aware with fallback to fixed-size

        Args:
            text: Input text
            max_chunk_size: Maximum chunk size
            overlap: Overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        max_chunk_size = max_chunk_size or settings.max_chunk_size
        overlap = overlap or settings.chunk_overlap

        # Try intelligent chunking first
        chunks = self.intelligent_chunk(text, max_chunk_size, overlap)

        # If we only got one chunk and it's too long, fall back to fixed-size
        if len(chunks) == 1 and len(chunks[0]['text']) > max_chunk_size * 1.5:
            logger.info("Intelligent chunking produced oversized chunk, falling back to fixed-size")
            chunks = self.fixed_size_chunk(text, max_chunk_size, overlap)

        return chunks

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract potential keywords from text

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of keywords
        """
        # Simple keyword extraction based on frequency
        words = self.word_pattern.findall(text.lower())

        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = [word for word in words if len(word) > 2 and word not in stop_words]

        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [word for word, freq in sorted_words[:max_keywords]]

    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about the text

        Args:
            text: Input text

        Returns:
            Dictionary with text statistics
        """
        sentences = self.split_into_sentences(text)
        words = self.word_pattern.findall(text)

        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'estimated_tokens': self.estimate_tokens(text)
        }

    def _load_stop_words(self) -> set:
        """Load comprehensive stop words list"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'whose', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'will', 'would', 'should', 'could', 'ought', 'i\'m', 'you\'re',
            'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re', 'i\'ve', 'you\'ve',
            'we\'ve', 'they\'ve', 'i\'d', 'you\'d', 'he\'d', 'she\'d', 'we\'d',
            'they\'d', 'i\'ll', 'you\'ll', 'he\'ll', 'she\'ll', 'we\'ll', 'they\'ll',
            'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t',
            'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t',
            'shan\'t', 'shouldn\'t', 'can\'t', 'cannot', 'couldn\'t', 'mustn\'t',
            'let\'s', 'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s',
            'when\'s', 'where\'s', 'why\'s', 'how\'s', 'a\'s', 'an\'s', 'the\'s'
        }

    def _load_common_words(self) -> set:
        """Load very common words that should have reduced weight"""
        return {
            'time', 'way', 'year', 'work', 'government', 'day', 'man', 'world',
            'life', 'part', 'house', 'course', 'case', 'system', 'place', 'end',
            'group', 'company', 'party', 'information', 'school', 'fact', 'money',
            'point', 'example', 'state', 'business', 'night', 'area', 'water',
            'thing', 'person', 'law', 'hand', 'car', 'power', 'question', 'service',
            'building', 'head', 'home', 'interest', 'member', 'country', 'price',
            'market', 'need', 'use', 'last', 'call', 'future', 'action', 'live',
            'word', 'issue', 'side', 'kind', 'head', 'far', 'black', 'long',
            'both', 'little', 'own', 'good', 'new', 'old', 'high', 'right',
            'small', 'large', 'next', 'early', 'young', 'important', 'public',
            'bad', 'same', 'able', 'human', 'local', 'sure', 'change', 'line',
            'based', 'around', 'god', 'sense', 'turn', 'move', 'level', 'face',
            'door', 'road', 'white', 'war', 'history', 'party', 'result', 'open',
            'change', 'morning', 'reason', 'research', 'girl', 'guy', 'food',
            'moment', 'air', 'teacher', 'force', 'education', 'foot', 'boy',
            'age', 'policy', 'love', 'process', 'music', 'god', 'sense', 'turn'
        }

    def calculate_word_importance(self, text: str) -> Dict[str, float]:
        """
        Calculate word importance scores using TF-IDF style weighting

        Args:
            text: Input text

        Returns:
            Dictionary mapping words to importance scores
        """
        # Tokenize and normalize
        words = self.word_pattern.findall(text.lower())

        # Filter out stop words and very short words
        filtered_words = [
            word for word in words
            if len(word) > 2 and word not in self.stop_words
        ]

        if not filtered_words:
            return {}

        # Calculate term frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        total_words = len(filtered_words)

        # Calculate TF (Term Frequency)
        tf_scores = {}
        for word, freq in word_freq.items():
            tf_scores[word] = freq / total_words

        # Simple IDF approximation (rarer words get higher weight)
        # In a real system, this would be calculated across the entire corpus
        idf_scores = {}
        for word in word_freq.keys():
            # Penalize very common words
            if word in self.common_words:
                idf_scores[word] = 0.5  # Reduce weight for common words
            else:
                # Simple heuristic: longer, less frequent words get higher weight
                idf_scores[word] = min(2.0, 1.0 + (len(word) - 3) * 0.1)

        # Calculate TF-IDF style scores
        importance_scores = {}
        for word in word_freq.keys():
            tf_idf = tf_scores[word] * idf_scores[word]

            # Additional heuristics
            # Capitalized words (potential proper nouns) get slight boost
            if word[0].isupper():
                tf_idf *= 1.2

            # Numbers get reduced weight
            if any(char.isdigit() for char in word):
                tf_idf *= 0.7

            importance_scores[word] = tf_idf

        return importance_scores

    def extract_important_terms(self, text: str, max_terms: int = 10) -> List[str]:
        """
        Extract the most important terms from text

        Args:
            text: Input text
            max_terms: Maximum number of terms to return

        Returns:
            List of important terms sorted by importance
        """
        importance_scores = self.calculate_word_importance(text)

        # Sort by importance score
        sorted_terms = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [term for term, score in sorted_terms[:max_terms]]

    def filter_text_for_embedding(self, text: str) -> str:
        """
        Filter text to improve embedding quality by removing noise

        Args:
            text: Input text

        Returns:
            Filtered text optimized for embeddings
        """
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Optional: Remove very common phrases that don't add meaning
        # This is a simple implementation - could be enhanced with NLP

        return text

    def enhance_chunk_with_word_weights(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a chunk dictionary with word-level weighting information

        Args:
            chunk: Chunk dictionary from chunking methods

        Returns:
            Enhanced chunk with word weighting metadata
        """
        text = chunk['text']

        # Calculate word importance
        word_weights = self.calculate_word_importance(text)

        # Extract important terms
        important_terms = self.extract_important_terms(text, max_terms=5)

        # Filter text for better embeddings
        filtered_text = self.filter_text_for_embedding(text)

        # Update chunk
        chunk['word_weights'] = word_weights
        chunk['important_terms'] = important_terms
        chunk['filtered_text'] = filtered_text
        chunk['metadata']['has_word_weighting'] = True
        chunk['metadata']['important_term_count'] = len(important_terms)

        return chunk
