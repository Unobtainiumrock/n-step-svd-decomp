"""
Tone axis vector learning in residual space.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
import structlog

logger = structlog.get_logger(__name__)


class ToneAxisLearner:
    """
    Learn tone-axis-specific preference vectors in residual space.
    """
    
    def __init__(self, regularization: float = 1.0, random_state: int = 42):
        """
        Initialize the tone axis vector learner.
        
        Args:
            regularization: Regularization strength for logistic regression
            random_state: Random state for reproducibility
        """
        self.regularization = regularization
        self.random_state = random_state
        self.tone_axis_vectors: Dict[str, np.ndarray] = {}
        self.models: Dict[str, LogisticRegression] = {}
        
    def learn_tone_axis_residual_vectors(
        self, 
        tone_axis_id: str,
        tone_axis_choices: List[Dict[str, Any]], 
        residual_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Learn tone-axis-specific preference vectors in residual space.
        
        Args:
            tone_axis_id: Unique identifier for the tone axis
            tone_axis_choices: List of choice dictionaries with 'chosen_idx' and 'rejected_idx'
            residual_vectors: Residual vectors of shape (n_samples, 1536)
            
        Returns:
            Tone axis vector vector of shape (1536,)
        """
        logger.info(
            "Learning tone axis vectors",
            tone_axis_id=tone_axis_id,
            n_choices=len(tone_axis_choices)
        )
        
        # Prepare training data
        X = []  # Difference vectors in residual space
        y = []  # Choice labels (0 or 1)
        
        for choice in tone_axis_choices:
            chosen_idx = choice['chosen_idx']
            rejected_idx = choice['rejected_idx']
            
            # Validate indices
            if chosen_idx >= len(residual_vectors) or rejected_idx >= len(residual_vectors):
                logger.warning("Invalid indices in choice", chosen_idx=chosen_idx, rejected_idx=rejected_idx)
                continue
            
            chosen_residual = residual_vectors[chosen_idx]
            rejected_residual = residual_vectors[rejected_idx]
            
            # Difference vector
            diff_vector = chosen_residual - rejected_residual
            X.append(diff_vector)
            y.append(1)  # Chosen is preferred
        
        if not X:
            raise ValueError("No valid choices found for tone axis")
        
        X = np.array(X)
        y = np.array(y)
        
        # Train logistic regression
        lr = LogisticRegression(
            C=1.0/self.regularization,
            random_state=self.random_state,
            max_iter=1000
        )
        lr.fit(X, y)
        
        # Store model and preference vector
        self.models[tone_axis_id] = lr
        self.tone_axis_vectors[tone_axis_id] = lr.coef_[0]
        
        logger.info(
            "Tone axis vectors learned",
            tone_axis_id=tone_axis_id,
            vector_norm=np.linalg.norm(lr.coef_[0])
        )
        
        return lr.coef_[0]
    
    def optimize_cross_tone_axis_consensus(
        self, 
        tone_axis_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Optimize preference vectors across tone axis types.
        
        Args:
            tone_axis_weights: Optional weights for each tone axis
            
        Returns:
            Consensus preference vector
        """
        if not self.tone_axis_vectors:
            raise ValueError("No tone axis vectors found. Learn preferences first.")
        
        logger.info("Optimizing cross-tone-axis consensus", n_tone_axes=len(self.tone_axis_vectors))
        
        # Get tone axis IDs and vectors
        tone_axis_ids = list(self.tone_axis_vectors.keys())
        tone_axis_vectors = np.array([self.tone_axis_vectors[sid] for sid in tone_axis_ids])
        
        # Use equal weights if not provided
        if tone_axis_weights is None:
            weights = np.ones(len(tone_axis_ids)) / len(tone_axis_ids)
        else:
            weights = np.array([tone_axis_weights.get(sid, 0.0) for sid in tone_axis_ids])
            weights = weights / weights.sum()  # Normalize
        
        # Weighted combination of tone axis vectors
        consensus_vector = np.average(tone_axis_vectors, weights=weights, axis=0)
        
        # Normalize
        consensus_vector = consensus_vector / np.linalg.norm(consensus_vector)
        
        logger.info("Cross-tone-axis consensus optimized", vector_norm=np.linalg.norm(consensus_vector))
        
        return consensus_vector
    
    def predict_preference(
        self, 
        tone_axis_id: str, 
        residual_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Predict preferences for a tone axis.
        
        Args:
            tone_axis_id: Tone axis identifier
            residual_vectors: Residual vectors to score
            
        Returns:
            Preference scores
        """
        if tone_axis_id not in self.models:
            raise ValueError(f"No model found for tone axis: {tone_axis_id}")
        
        model = self.models[tone_axis_id]
        return model.predict_proba(residual_vectors)[:, 1]  # Probability of preference
    
    def get_tone_axis_vector(self, tone_axis_id: str) -> np.ndarray:
        """
        Get the preference vector for a tone axis.
        
        Args:
            tone_axis_id: Tone axis identifier
            
        Returns:
            Preference vector
        """
        if tone_axis_id not in self.tone_axis_vectors:
            raise ValueError(f"No vector found for tone axis: {tone_axis_id}")
        
        return self.tone_axis_vectors[tone_axis_id]
    
    def get_all_tone_axis_vectors(self) -> Dict[str, np.ndarray]:
        """
        Get all tone axis vector vectors.
        
        Returns:
            Dictionary mapping tone axis IDs to preference vectors
        """
        return self.tone_axis_vectors.copy()
