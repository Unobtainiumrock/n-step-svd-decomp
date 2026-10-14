"""
Residual vector calculation and tone scoring functionality.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import structlog

logger = structlog.get_logger(__name__)


class ResidualToneCalculator:
    """
    Calculate residual vectors and tone scores for content.
    """
    
    def __init__(self, topic_projection_matrix: np.ndarray):
        """
        Initialize the residual tone calculator.
        
        Args:
            topic_projection_matrix: Projection matrix for topic subspace
        """
        self.topic_projection_matrix = topic_projection_matrix
        self.tone_axis_vectors: Optional[Dict[str, np.ndarray]] = None
        logger.info("ResidualToneCalculator initialized", matrix_shape=topic_projection_matrix.shape)
    
    @classmethod
    def from_saved_data(cls, models_dir: str = "data/models") -> 'ResidualToneCalculator':
        """
        Load calculator from saved residual analysis data.
        
        Args:
            models_dir: Directory containing saved model files
            
        Returns:
            Initialized ResidualToneCalculator with loaded data
        """
        import os
        
        # Load projection matrix
        projection_matrix_path = os.path.join(models_dir, "residual_projection_matrix.npy")
        if not os.path.exists(projection_matrix_path):
            raise FileNotFoundError(f"Projection matrix not found at {projection_matrix_path}")
        
        projection_matrix = np.load(projection_matrix_path)
        calculator = cls(projection_matrix)
        
        # Load tone axis data if available
        tone_axis_path = os.path.join(models_dir, "residual_tone_axis_data.npy")
        if os.path.exists(tone_axis_path):
            tone_axis_data = np.load(tone_axis_path, allow_pickle=True).item()
            calculator.load_tone_axis_vectors(
                tone_axis_data['names'], 
                tone_axis_data['vectors']
            )
            logger.info("Loaded tone axis vectors", 
                       n_tone_axes=len(tone_axis_data['names']))
        
        logger.info("ResidualToneCalculator loaded from saved data")
        return calculator
    
    def load_tone_axis_vectors(self, names: List[str], vectors: np.ndarray) -> None:
        """
        Load tone axis vector vectors.
        
        Args:
            names: List of tone axis names
            vectors: Array of tone axis vector vectors (n_tone_axes, 1536)
        """
        self.tone_axis_vectors = {}
        for name, vector in zip(names, vectors):
            self.tone_axis_vectors[name] = vector
        
        logger.info("Loaded tone axis vectors", n_tone_axes=len(names))
    
    def add_tone_axis_vector(self, name: str, preference_vector: np.ndarray) -> None:
        """
        Add a new tone axis vector vector.
        
        Args:
            name: Tone axis name
            preference_vector: Preference vector in residual space (1536,)
        """
        if self.tone_axis_vectors is None:
            self.tone_axis_vectors = {}
        
        # Ensure the preference vector is in residual space (orthogonal to topic space)
        preference_vector = preference_vector - (preference_vector @ self.topic_projection_matrix)
        preference_vector = preference_vector / np.linalg.norm(preference_vector)
        
        self.tone_axis_vectors[name] = preference_vector
        logger.info("Added tone axis vector", name=name)
    
    def calculate_residual_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate residual vectors by removing topic components.
        
        Args:
            embeddings: Input embeddings of shape (n_samples, 1536)
            
        Returns:
            Residual vectors of shape (n_samples, 1536)
        """
        logger.info("Calculating residual vectors", n_samples=embeddings.shape[0])
        
        # Project embeddings onto topic subspace
        topic_projections = embeddings @ self.topic_projection_matrix
        
        # Calculate residuals
        residual_vectors = embeddings - topic_projections
        
        logger.info("Residual vectors calculated", shape=residual_vectors.shape)
        
        return residual_vectors
    
    def get_residual_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Calculate residual embedding for a single embedding.
        
        Args:
            embedding: Input embedding of shape (1536,)
            
        Returns:
            Residual embedding of shape (1536,)
        """
        # Ensure embedding is 2D for matrix multiplication
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Project onto topic subspace
        topic_projection = embedding @ self.topic_projection_matrix
        
        # Calculate residual
        residual = embedding - topic_projection
        
        # Return as 1D if input was 1D
        if residual.shape[0] == 1:
            return residual[0]
        
        return residual
    
    def calculate_tone_score(self, residual_embedding: np.ndarray, tone_axis_vector: np.ndarray) -> float:
        """
        Calculate tone score for a residual embedding and tone axis.
        
        Args:
            residual_embedding: Residual embedding of shape (1536,)
            tone_axis_vector: Tone axis vector vector of shape (1536,)
            
        Returns:
            Tone score (float)
        """
        # Calculate dot product in residual space
        tone_score = np.dot(residual_embedding, tone_axis_vector)
        
        return float(tone_score)
    
    def calculate_tone_score_for_axis(self, embedding: np.ndarray, tone_axis_name: str) -> float:
        """
        Calculate tone score for an embedding and named tone axis.
        
        Args:
            embedding: Input embedding of shape (1536,)
            tone_axis_name: Name of the tone axis
            
        Returns:
            Tone score (float)
        """
        if self.tone_axis_vectors is None or tone_axis_name not in self.tone_axis_vectors:
            raise ValueError(f"Tone axis '{tone_axis_name}' not found")
        
        residual_embedding = self.get_residual_embedding(embedding)
        tone_axis_vector = self.tone_axis_vectors[tone_axis_name]
        
        return self.calculate_tone_score(residual_embedding, tone_axis_vector)
    
    def calculate_tone_scores_batch(self, residual_embeddings: np.ndarray, tone_axis_vector: np.ndarray) -> np.ndarray:
        """
        Calculate tone scores for a batch of residual embeddings.
        
        Args:
            residual_embeddings: Residual embeddings of shape (n_samples, 1536)
            tone_axis_vector: Tone axis vector vector of shape (1536,)
            
        Returns:
            Tone scores of shape (n_samples,)
        """
        logger.info("Calculating tone scores for batch", n_samples=residual_embeddings.shape[0])
        
        # Calculate dot products
        tone_scores = np.dot(residual_embeddings, tone_axis_vector)
        
        logger.info("Tone scores calculated", shape=tone_scores.shape)
        
        return tone_scores
    
    def calculate_tone_scores_all_axes(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate tone scores for all loaded tone axes.
        
        Args:
            embeddings: Input embeddings of shape (n_samples, 1536)
            
        Returns:
            Dictionary mapping tone axis names to tone scores
        """
        if self.tone_axis_vectors is None:
            raise ValueError("No tone axis vectors loaded")
        
        # Calculate residual vectors once
        residual_vectors = self.calculate_residual_vectors(embeddings)
        
        # Calculate scores for all tone axes
        all_scores = {}
        for name, tone_axis_vector in self.tone_axis_vectors.items():
            scores = self.calculate_tone_scores_batch(residual_vectors, tone_axis_vector)
            all_scores[name] = scores
        
        logger.info("Calculated tone scores for all tone axes", 
                   n_tone_axes=len(self.tone_axis_vectors),
                   n_samples=embeddings.shape[0])
        
        return all_scores
    
    def get_topic_projection(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get the topic projection of embeddings.
        
        Args:
            embeddings: Input embeddings of shape (n_samples, 1536) or (1536,)
            
        Returns:
            Topic projections of same shape as input
        """
        # Handle both single embedding and batch
        single_embedding = embeddings.ndim == 1
        if single_embedding:
            embeddings = embeddings.reshape(1, -1)
        
        # Project onto topic subspace
        topic_projections = embeddings @ self.topic_projection_matrix
        
        # Return as 1D if input was 1D
        if single_embedding:
            return topic_projections[0]
        
        return topic_projections
    
    def validate_residual_orthogonality(self, residual_vectors: np.ndarray, 
                                      topic_vectors: np.ndarray) -> Dict[str, float]:
        """
        Validate that residual vectors are orthogonal to topic space.
        
        Args:
            residual_vectors: Residual vectors of shape (n_samples, 1536)
            topic_vectors: Topic basis vectors of shape (n_components, 1536)
            
        Returns:
            Dictionary of validation metrics
        """
        # Check orthogonality
        orthogonality_scores = np.abs(residual_vectors @ topic_vectors.T)
        mean_orthogonality = np.mean(orthogonality_scores)
        max_orthogonality = np.max(orthogonality_scores)
        
        # Check reconstruction
        topic_projections = self.get_topic_projection(residual_vectors + 
                                                    self.get_topic_projection(residual_vectors))
        reconstruction_errors = np.linalg.norm(residual_vectors - 
                                             (residual_vectors + topic_projections - 
                                              self.get_topic_projection(residual_vectors)), axis=1)
        mean_reconstruction_error = np.mean(reconstruction_errors)
        
        validation_metrics = {
            'mean_orthogonality': float(mean_orthogonality),
            'max_orthogonality': float(max_orthogonality),
            'mean_reconstruction_error': float(mean_reconstruction_error),
            'orthogonality_passed': mean_orthogonality < 1e-10,
            'reconstruction_passed': mean_reconstruction_error < 1e-10
        }
        
        logger.info("Residual validation completed", **validation_metrics)
        
        return validation_metrics
    
    def get_tone_axis_names(self) -> List[str]:
        """Get list of loaded tone axis names."""
        if self.tone_axis_vectors is None:
            return []
        return list(self.tone_axis_vectors.keys())
    
    def save_tone_axis_vectors(self, filepath: str) -> None:
        """
        Save tone axis vectors to file.
        
        Args:
            filepath: Path to save tone axis data
        """
        if self.tone_axis_vectors is None:
            raise ValueError("No tone axis vectors to save")
        
        tone_axis_data = {
            'names': list(self.tone_axis_vectors.keys()),
            'vectors': np.array(list(self.tone_axis_vectors.values()))
        }
        
        np.save(filepath, tone_axis_data, allow_pickle=True)
        logger.info("Tone axis vectors saved", filepath=filepath)
