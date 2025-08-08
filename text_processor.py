import re
import nltk
import logging
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


class AdvancedTextProcessor:
    """Advanced text processing with semantic chunking and TF-IDF similarity"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.stop_words = set()
            self.lemmatizer = None
            logger.warning("NLTK components not available, using basic processing")
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning and normalization"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\"\']+', ' ', text)
        
        # Normalize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = text.replace(''', "'").replace(''', "'")
        text = re.sub(r'[–—]', '-', text)
        
        # Fix sentence spacing
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'Page \d+.*?\n', '', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()
    
    def semantic_chunk_text(self, text: str, target_chunk_size: int = 1000, 
                          overlap_ratio: float = 0.15) -> List[str]:
        """Advanced semantic chunking with sentence boundary preservation"""
        if not text.strip():
            return []
        
        # Clean the text first
        text = self.clean_text(text)
        
        # Split into sentences
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to basic sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = ""
        overlap_size = int(target_chunk_size * overlap_ratio)
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # If adding this sentence would exceed target size
            if len(current_chunk) + len(sentence) > target_chunk_size and current_chunk:
                # Add the current chunk
                chunks.append(current_chunk.strip())
                
                # Create overlap for next chunk
                overlap_text = ""
                overlap_length = 0
                
                # Look backwards to create overlap
                j = i - 1
                while j >= 0 and overlap_length < overlap_size:
                    if overlap_length + len(sentences[j]) <= overlap_size:
                        overlap_text = sentences[j] + " " + overlap_text
                        overlap_length += len(sentences[j])
                        j -= 1
                    else:
                        break
                
                current_chunk = overlap_text.strip()
            
            current_chunk += " " + sentence if current_chunk else sentence
            i += 1
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 100]
        
        logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        return chunks
    
    def preprocess_for_similarity(self, text: str) -> str:
        """Preprocess text for TF-IDF similarity calculation"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep word boundaries
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
            # Remove stopwords and short words
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            
            # Lemmatize if available
            if self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return ' '.join(tokens)
        except Exception:
            # Fallback to basic processing
            words = text.split()
            words = [word for word in words if len(word) > 2]
            return ' '.join(words)
    
    def calculate_tfidf_similarity(self, query: str, chunks: List[str]) -> List[Tuple[float, str]]:
        """Calculate TF-IDF cosine similarity between query and chunks"""
        if not chunks or not query:
            return []
        
        # Preprocess query and chunks
        processed_query = self.preprocess_for_similarity(query)
        processed_chunks = [self.preprocess_for_similarity(chunk) for chunk in chunks]
        
        # Create corpus (query + chunks)
        corpus = [processed_query] + processed_chunks
        
        try:
            # Calculate TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # Include bigrams
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity between query and chunks
            query_vector = tfidf_matrix[0:1]
            chunk_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, chunk_vectors)[0]
            
            # Combine with original chunks and sort by similarity
            scored_chunks = [(float(sim), chunk) for sim, chunk in zip(similarities, chunks)]
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            
            return scored_chunks
            
        except Exception as e:
            logger.warning(f"TF-IDF calculation failed, falling back to keyword matching: {e}")
            return self.fallback_similarity(query, chunks)
    
    def fallback_similarity(self, query: str, chunks: List[str]) -> List[Tuple[float, str]]:
        """Fallback similarity calculation using keyword matching"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            
            if not query_words or not chunk_words:
                score = 0.0
            else:
                # Jaccard similarity
                intersection = len(query_words & chunk_words)
                union = len(query_words | chunk_words)
                score = intersection / union if union > 0 else 0.0
                
                # Boost for exact phrase matches
                if query.lower() in chunk.lower():
                    score += 0.3
            
            scored_chunks.append((score, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return scored_chunks
    
    def find_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 8, 
                           min_score: float = 0.03) -> List[str]:
        """Find most relevant chunks using TF-IDF similarity"""
        if not chunks or not query:
            return []
        
        # Calculate similarities
        scored_chunks = self.calculate_tfidf_similarity(query, chunks)
        
        # Filter by minimum score and return top k
        relevant_chunks = [
            chunk for score, chunk in scored_chunks[:top_k] 
            if score >= min_score
        ]
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks for query (top_k={top_k}, min_score={min_score})")
        return relevant_chunks
    
    def create_context(self, relevant_chunks: List[str], max_context_length: int = 4000) -> str:
        """Create optimized context from relevant chunks"""
        if not relevant_chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(relevant_chunks):
            chunk_with_marker = f"\n[Context {i+1}]\n{chunk}\n"
            
            if current_length + len(chunk_with_marker) > max_context_length:
                break
            
            context_parts.append(chunk_with_marker)
            current_length += len(chunk_with_marker)
        
        return "".join(context_parts)
