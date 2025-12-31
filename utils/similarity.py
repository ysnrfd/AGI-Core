# agi_core/utils/similarity.py
"""
Similarity search and retrieval utilities
"""

import numpy as np
import faiss
import torch.nn as nn
from typing import List, Tuple, Optional, Any
import torch

class SimilaritySearch:
    """FAISS-based similarity search"""
    
    def __init__(self, dimension: int, use_gpu: bool = False):
        self.dimension = dimension
        self.use_gpu = use_gpu
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
        self.embeddings = []
        self.metadata = []
        
    def add(self, embeddings: np.ndarray, metadata: List[Any]):
        """Add embeddings with metadata"""
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")
            
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype(np.float32))
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
        
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[Any]]:
        """Search for similar embeddings"""
        if len(self.embeddings) == 0:
            return np.array([]), []
            
        # Normalize query
        query_norm = query / np.linalg.norm(query)
        
        # Search
        distances, indices = self.index.search(
            query_norm.reshape(1, -1).astype(np.float32), 
            min(k, len(self.embeddings))
        )
        
        # Retrieve metadata
        results = [self.metadata[idx] for idx in indices[0] if idx != -1]
        
        return distances[0][:len(results)], results
        
    def search_by_text(self, text: str, encoder, k: int = 10) -> Tuple[np.ndarray, List[Any]]:
        """Search by text query"""
        query_emb = encoder.encode(text)
        return self.search(query_emb, k)
        
    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, path)
        
    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(path)
        
class AttentionMechanism:
    """Attention mechanism for working memory"""
    
    def __init__(self, hidden_dim: int = 512):
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention forward pass"""
        Q = self.query_proj(query)
        K = self.key_proj(keys)
        V = self.value_proj(values)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights