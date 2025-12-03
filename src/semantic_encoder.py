"""
Task 2.2: Enhanced Semantic Encoding with Neural Networks
CNN-based encoder for hierarchical NDN names to extract semantic features
"""

import re
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using fallback hash-based encoding")


class SemanticEncoder:
    """
    CNN-based semantic encoder for NDN hierarchical names
    Parses names like /edu/university/department/type/content and extracts semantic features
    """
    
    def __init__(self, embedding_dim: int = 64, use_cnn: bool = True):
        """
        Initialize semantic encoder
        
        Args:
            embedding_dim: Dimension of output embeddings (64-128 recommended)
            use_cnn: Whether to use CNN (requires PyTorch) or fallback to hash-based
        """
        self.embedding_dim = embedding_dim
        self.use_cnn = use_cnn and TORCH_AVAILABLE
        
        if self.use_cnn:
            self._init_cnn_encoder()
        else:
            logger.info("Using hash-based semantic encoding (PyTorch not available)")
            self._init_hash_encoder()
    
    def _init_cnn_encoder(self):
        """Initialize CNN-based encoder"""
        # Parse hierarchical name into components
        # Each component becomes a token
        # Use 1D convolutions to extract features from token sequences
        
        # Vocabulary size: estimate based on common NDN name patterns
        self.vocab_size = 1000
        self.token_embedding_dim = 32
        
        # Token embedding layer
        self.token_embed = nn.Embedding(self.vocab_size, self.token_embedding_dim)
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=self.token_embedding_dim,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        # Pooling and final projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, self.embedding_dim)
        self.relu = nn.ReLU()
        
        # Hash function for tokenization (fallback)
        self.token_hash = {}
        self.next_token_id = 1
    
    def _init_hash_encoder(self):
        """Initialize hash-based fallback encoder"""
        # Simple hash-based encoding when CNN is not available
        pass
    
    def _tokenize_name(self, name: str) -> List[int]:
        """
        Tokenize hierarchical NDN name into token IDs
        Example: /edu/university/cs/course/content -> [token_edu, token_university, token_cs, ...]
        """
        # Remove leading/trailing slashes and split
        parts = [p for p in name.strip('/').split('/') if p]
        
        token_ids = []
        for part in parts:
            # Hash the part to get a token ID
            if self.use_cnn:
                # Use hash to map to vocabulary
                token_hash = hash(part) % self.vocab_size
                if part not in self.token_hash:
                    self.token_hash[part] = token_hash
                token_ids.append(self.token_hash[part])
            else:
                # For hash-based encoding, just use hash directly
                token_ids.append(hash(part))
        
        return token_ids
    
    def encode_cnn(self, name: str) -> np.ndarray:
        """
        Encode name using CNN-based neural network
        
        Args:
            name: NDN hierarchical name (e.g., /edu/university/department/type/content)
        
        Returns:
            Semantic embedding vector of dimension embedding_dim
        """
        if not self.use_cnn:
            return self.encode_hash(name)
        
        try:
            # Tokenize
            tokens = self._tokenize_name(name)
            if not tokens:
                # Empty name, return zero vector
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            # Convert to tensor
            token_tensor = torch.tensor([tokens], dtype=torch.long)
            
            # Embed tokens
            embedded = self.token_embed(token_tensor)  # [1, seq_len, token_embedding_dim]
            
            # Transpose for conv1d: [batch, channels, seq_len]
            embedded = embedded.transpose(1, 2)
            
            # Apply CNN layers
            x = self.relu(self.conv1(embedded))
            x = self.relu(self.conv2(x))
            
            # Global pooling
            x = self.pool(x)  # [1, 128, 1]
            x = x.squeeze(-1)  # [1, 128]
            
            # Final projection
            x = self.fc(x)  # [1, embedding_dim]
            
            # Convert to numpy
            embedding = x.detach().numpy().flatten()
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
        
        except Exception as e:
            logger.warning(f"CNN encoding failed for {name}: {e}, falling back to hash")
            return self.encode_hash(name)
    
    def encode_hash(self, name: str) -> np.ndarray:
        """
        Fallback hash-based encoding (used when CNN is not available)
        
        Args:
            name: NDN hierarchical name
        
        Returns:
            Semantic embedding vector
        """
        # Parse hierarchical structure
        parts = [p for p in name.strip('/').split('/') if p]
        
        # Create embedding from parts
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        for i, part in enumerate(parts):
            # Hash each part and add to embedding
            part_hash = hash(part)
            # Distribute hash across embedding dimensions
            for j in range(self.embedding_dim):
                # Use different hash seeds for each dimension
                hash_val = hash((part, j, i)) % 1000
                embedding[j] += (hash_val / 1000.0) * (1.0 / (i + 1))  # Weight by position
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def encode(self, name: str) -> np.ndarray:
        """
        Main encoding method - automatically chooses CNN or hash-based
        
        Args:
            name: NDN hierarchical name
        
        Returns:
            Semantic embedding vector
        """
        if self.use_cnn:
            return self.encode_cnn(name)
        else:
            return self.encode_hash(name)
    
    def encode_batch(self, names: List[str]) -> np.ndarray:
        """
        Encode a batch of names
        
        Args:
            names: List of NDN names
        
        Returns:
            Array of embeddings [batch_size, embedding_dim]
        """
        embeddings = []
        for name in names:
            embeddings.append(self.encode(name))
        return np.array(embeddings, dtype=np.float32)


# Global encoder instance (lazy initialization)
_global_encoder: Optional[SemanticEncoder] = None


def get_semantic_encoder(embedding_dim: int = 64, use_cnn: bool = True) -> SemanticEncoder:
    """
    Get or create global semantic encoder instance
    
    Args:
        embedding_dim: Dimension of embeddings
        use_cnn: Whether to use CNN (requires PyTorch)
    
    Returns:
        SemanticEncoder instance
    """
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = SemanticEncoder(embedding_dim=embedding_dim, use_cnn=use_cnn)
    return _global_encoder


def encode_name(name: str, embedding_dim: int = 64) -> np.ndarray:
    """
    Convenience function to encode a single name
    
    Args:
        name: NDN hierarchical name
        embedding_dim: Dimension of embedding
    
    Returns:
        Semantic embedding vector
    """
    encoder = get_semantic_encoder(embedding_dim=embedding_dim)
    return encoder.encode(name)

