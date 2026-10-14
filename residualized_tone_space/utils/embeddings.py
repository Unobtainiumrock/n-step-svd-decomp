"""
Utility functions for embedding processing and management.
"""

import numpy as np
import h5py
from typing import List, Dict, Any, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingProcessor:
    """
    Utility class for processing and managing embeddings.
    """
    
    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize the embedding processor.
        
        Args:
            embedding_dim: Dimension of embeddings (default: 1536 for OpenAI text-embedding-3-small)
        """
        self.embedding_dim = embedding_dim
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validate embedding array format and dimensions.
        
        Args:
            embeddings: Embedding array to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(embeddings, np.ndarray):
            logger.error("Embeddings must be numpy array")
            return False
        
        if embeddings.ndim != 2:
            logger.error("Embeddings must be 2D array", ndim=embeddings.ndim)
            return False
        
        if embeddings.shape[1] != self.embedding_dim:
            logger.error(
                "Invalid embedding dimension",
                expected=self.embedding_dim,
                actual=embeddings.shape[1]
            )
            return False
        
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            logger.error("Embeddings contain NaN or infinite values")
            return False
        
        logger.info("Embeddings validation passed", shape=embeddings.shape)
        return True
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        logger.info("Normalizing embeddings", shape=embeddings.shape)
        
        # Calculate L2 norms
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        
        # Normalize
        normalized_embeddings = embeddings / norms
        
        logger.info("Embeddings normalized")
        return normalized_embeddings
    
    def save_embeddings_hdf5(self, embeddings: np.ndarray, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save embeddings to HDF5 file.
        
        Args:
            embeddings: Embeddings to save
            filepath: Path to save file
            metadata: Optional metadata to save
        """
        logger.info("Saving embeddings to HDF5", filepath=filepath, shape=embeddings.shape)
        
        with h5py.File(filepath, 'w') as f:
            # Save embeddings
            f.create_dataset('embeddings', data=embeddings, compression='gzip')
            
            # Save metadata if provided
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        f.attrs[key] = value
                    elif isinstance(value, list):
                        f.create_dataset(f'metadata/{key}', data=value)
        
        logger.info("Embeddings saved successfully")
    
    def load_embeddings_hdf5(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings from HDF5 file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        logger.info("Loading embeddings from HDF5", filepath=filepath)
        
        with h5py.File(filepath, 'r') as f:
            # Load embeddings
            embeddings = f['embeddings'][:]
            
            # Load metadata
            metadata = {}
            for key, value in f.attrs.items():
                metadata[key] = value
            
            # Load metadata datasets
            if 'metadata' in f:
                for key in f['metadata'].keys():
                    metadata[key] = f[f'metadata/{key}'][:]
        
        logger.info("Embeddings loaded successfully", shape=embeddings.shape)
        return embeddings, metadata
    
    def calculate_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Dictionary of statistics
        """
        logger.info("Calculating embedding statistics", shape=embeddings.shape)
        
        stats = {
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'median': float(np.median(embeddings)),
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
        }
        
        logger.info("Embedding statistics calculated", stats=stats)
        return stats
    
    def find_similar_embeddings(
        self, 
        query_embedding: np.ndarray, 
        embeddings: np.ndarray, 
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query embedding
            embeddings: Embedding database
            top_k: Number of top results to return
            
        Returns:
            Tuple of (indices, similarities)
        """
        logger.info("Finding similar embeddings", top_k=top_k)
        
        # Calculate cosine similarities
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        logger.info("Similar embeddings found", n_results=len(top_indices))
        return top_indices, top_similarities
