"""RAG retrieval system using TF-IDF."""
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import math


class Chunk:
    """Represents a document chunk."""
    
    def __init__(self, chunk_id: str, content: str, source: str, score: float = 0.0):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.score = score
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
        }


class TFIDFRetriever:
    """Simple TF-IDF retriever for document search."""
    
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[Chunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.chunk_vectors: List[Dict[str, float]] = []
        self._load_documents()
        self._build_index()
    
    def _load_documents(self):
        """Load and chunk documents from docs directory."""
        chunk_id_counter = {}
        
        for doc_file in self.docs_dir.glob("*.md"):
            content = doc_file.read_text(encoding="utf-8")
            source = doc_file.stem
            
            # Simple paragraph-level chunking
            paragraphs = re.split(r"\n\s*\n", content)
            
            for idx, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) < 10:  # Skip very short paragraphs
                    continue
                
                if source not in chunk_id_counter:
                    chunk_id_counter[source] = 0
                else:
                    chunk_id_counter[source] += 1
                
                chunk_id = f"{source}::chunk{chunk_id_counter[source]}"
                self.chunks.append(Chunk(chunk_id, para, source))
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        # Remove punctuation, keep alphanumeric and spaces
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Filter very short tokens
    
    def _build_index(self):
        """Build TF-IDF index."""
        # Build vocabulary
        all_tokens = set()
        chunk_tokens = []
        
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.content)
            chunk_tokens.append(tokens)
            all_tokens.update(tokens)
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        
        # Calculate IDF
        num_chunks = len(self.chunks)
        for token in self.vocab:
            doc_freq = sum(1 for tokens in chunk_tokens if token in tokens)
            self.idf[token] = math.log(num_chunks / (1 + doc_freq))
        
        # Build TF vectors for each chunk
        for tokens in chunk_tokens:
            tf = Counter(tokens)
            max_tf = max(tf.values()) if tf else 1
            
            vector = {}
            for token, count in tf.items():
                if token in self.vocab:
                    tf_score = count / max_tf
                    vector[token] = tf_score * self.idf.get(token, 0)
            
            self.chunk_vectors.append(vector)
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
        
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """
        Retrieve top-k chunks for a query.
        
        Returns chunks sorted by relevance score.
        """
        query_tokens = self._tokenize(query)
        
        # Build query vector
        query_tf = Counter(query_tokens)
        max_tf = max(query_tf.values()) if query_tf else 1
        
        query_vector = {}
        for token, count in query_tf.items():
            if token in self.vocab:
                tf_score = count / max_tf
                query_vector[token] = tf_score * self.idf.get(token, 0)
        
        # Calculate similarity scores
        scored_chunks = []
        for idx, chunk_vector in enumerate(self.chunk_vectors):
            score = self._cosine_similarity(query_vector, chunk_vector)
            chunk = self.chunks[idx]
            chunk.score = score
            scored_chunks.append((score, chunk))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k
        return [chunk for _, chunk in scored_chunks[:top_k]]

