# agi_core/memory.py
"""
Memory system with long-term and working memory
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict

from utils.similarity import SimilaritySearch
from utils.embeddings import EmbeddingModel
from utils.logger import logger

@dataclass
class MemoryItem:
    """Individual memory item"""
    id: str
    content: Any
    embedding: np.ndarray
    timestamp: datetime
    importance: float = 1.0
    metadata: Dict[str, Any] = None
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

class LongTermMemory:
    """Vector-based long-term memory with FAISS"""
    
    def __init__(self, config, embedding_model: EmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self.dimension = embedding_model.get_dimension()
        
        # Initialize similarity search
        self.similarity_search = SimilaritySearch(
            dimension=self.dimension,
            use_gpu=config.device == "cuda"
        )
        
        # Memory storage
        self.memories: Dict[str, MemoryItem] = {}
        self.counter = 0
        
        # Importance decay
        self.importance_decay = 0.99
        
    def store(self, content: Any, importance: float = 1.0, 
              metadata: Optional[Dict] = None) -> str:
        """Store a memory item"""
        
        # Generate embedding
        if isinstance(content, str):
            embedding = self.embedding_model.encode(content)
        else:
            # For multimodal content
            embedding = np.random.randn(1, self.dimension)  # Placeholder
            
        # Create memory item
        memory_id = f"mem_{self.counter:08d}"
        item = MemoryItem(
            id=memory_id,
            content=content,
            embedding=embedding,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )
        
        # Store
        self.memories[memory_id] = item
        self.similarity_search.add(embedding, item)
        self.counter += 1
        
        logger.debug(f"Stored memory: {memory_id}", 
                    content_length=len(str(content)),
                    importance=importance)
        
        return memory_id
    
    def retrieve(self, query: str, k: int = 5, 
                min_similarity: float = 0.5) -> List[MemoryItem]:
        """Retrieve similar memories"""
        
        # Encode query
        query_emb = self.embedding_model.encode(query)
        
        # Search
        distances, items = self.similarity_search.search(query_emb, k=k)
        
        # Filter by similarity
        results = []
        for dist, item in zip(distances, items):
            if dist >= min_similarity:
                # Update access statistics
                item.access_count += 1
                item.last_accessed = datetime.now()
                
                # Increase importance based on access
                item.importance *= 1.1
                item.importance = min(item.importance, 10.0)
                
                results.append(item)
                
        return results
    
    def consolidate(self, threshold: float = 0.9):
        """Consolidate similar memories"""
        logger.info("Starting memory consolidation")
        
        # Find similar memories
        to_remove = set()
        
        for mem_id, memory in list(self.memories.items()):
            if mem_id in to_remove:
                continue
                
            # Find similar memories
            distances, items = self.similarity_search.search(
                memory.embedding, 
                k=len(self.memories)
            )
            
            # Merge similar memories
            for dist, other_mem in zip(distances, items):
                if (dist >= threshold and 
                    other_mem.id != mem_id and
                    other_mem.id not in to_remove):
                    
                    # Merge content
                    if isinstance(memory.content, str) and isinstance(other_mem.content, str):
                        memory.content = f"{memory.content}. {other_mem.content}"
                    
                    # Use max importance
                    memory.importance = max(memory.importance, other_mem.importance)
                    
                    # Mark for removal
                    to_remove.add(other_mem.id)
                    
        # Remove duplicates
        for mem_id in to_remove:
            del self.memories[mem_id]
            
        logger.info(f"Consolidated memories, removed {len(to_remove)} duplicates")
        
    def decay_importance(self):
        """Decay importance of old memories"""
        for memory in self.memories.values():
            memory.importance *= self.importance_decay
            
    def save(self, path: Optional[str] = None):
        """Save memory to disk"""
        if path is None:
            path = self.config.memory_path / "long_term_memory.json"
            
        data = {
            'memories': [mem.to_dict() for mem in self.memories.values()],
            'counter': self.counter
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Save FAISS index
        index_path = path.replace('.json', '.faiss')
        self.similarity_search.save(index_path)
        
    def load(self, path: Optional[str] = None):
        """Load memory from disk"""
        if path is None:
            path = self.config.memory_path / "long_term_memory.json"
            
        if not path.exists():
            return
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Load memories
        self.memories = {}
        for mem_data in data['memories']:
            # Convert timestamps
            mem_data['timestamp'] = datetime.fromisoformat(mem_data['timestamp'])
            mem_data['last_accessed'] = datetime.fromisoformat(mem_data['last_accessed'])
            
            # Create memory item
            item = MemoryItem(**mem_data)
            self.memories[item.id] = item
            
            # Add to similarity search
            self.similarity_search.add(item.embedding, item)
            
        self.counter = data['counter']

class WorkingMemory:
    """Working memory with attention mechanism"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memories: List[MemoryItem] = []
        self.attention_weights: List[float] = []
        
    def add(self, memory: MemoryItem, attention_weight: float = 1.0):
        """Add memory to working memory"""
        self.memories.append(memory)
        self.attention_weights.append(attention_weight)
        
        # Maintain capacity
        if len(self.memories) > self.capacity:
            # Remove least important/attended memory
            min_idx = np.argmin(self.attention_weights)
            self.memories.pop(min_idx)
            self.attention_weights.pop(min_idx)
            
    def get_context(self, k: int = 10) -> List[MemoryItem]:
        """Get most relevant memories"""
        if not self.memories:
            return []
            
        # Sort by attention weight
        sorted_indices = np.argsort(self.attention_weights)[::-1]
        
        return [self.memories[i] for i in sorted_indices[:k]]
    
    def update_attention(self, relevance_scores: List[float]):
        """Update attention weights"""
        if len(relevance_scores) == len(self.attention_weights):
            # Update with new scores
            self.attention_weights = [
                self.attention_weights[i] * 0.5 + relevance_scores[i] * 0.5
                for i in range(len(self.attention_weights))
            ]
            
    def clear(self):
        """Clear working memory"""
        self.memories.clear()
        self.attention_weights.clear()

class EpisodicMemory:
    """Episodic memory for storing experiences"""
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes: List[Dict] = []
        
    def record_episode(self, episode: Dict):
        """Record an episode"""
        episode['timestamp'] = datetime.now().isoformat()
        self.episodes.append(episode)
        
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
            
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get recent episodes"""
        return self.episodes[-n:] if self.episodes else []
    
    def get_by_pattern(self, pattern: Dict) -> List[Dict]:
        """Get episodes matching pattern"""
        results = []
        for episode in self.episodes:
            match = True
            for key, value in pattern.items():
                if key not in episode or episode[key] != value:
                    match = False
                    break
            if match:
                results.append(episode)
        return results