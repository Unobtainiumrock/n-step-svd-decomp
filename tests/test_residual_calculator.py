"""
Tests for residual calculator functionality.
"""

import pytest
import numpy as np
from residualized_tone_space.core.residual_calculator import ResidualToneCalculator


class TestResidualToneCalculator:
    """Test cases for ResidualToneCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple projection matrix (identity for testing)
        self.projection_matrix = np.eye(1536)
        self.calculator = ResidualToneCalculator(self.projection_matrix)
        self.embeddings = np.random.randn(100, 1536).astype(np.float32)
        self.single_embedding = np.random.randn(1536).astype(np.float32)
        self.tone_axis_vector = np.random.randn(1536).astype(np.float32)
    
    def test_initialization(self):
        """Test calculator initialization."""
        assert self.calculator.topic_projection_matrix.shape == (1536, 1536)
        assert np.array_equal(self.calculator.topic_projection_matrix, self.projection_matrix)
    
    def test_calculate_residual_vectors(self):
        """Test residual vector calculation."""
        residual_vectors = self.calculator.calculate_residual_vectors(self.embeddings)
        
        assert residual_vectors.shape == self.embeddings.shape
        # With identity projection matrix, residuals should be zero
        assert np.allclose(residual_vectors, 0.0, atol=1e-10)
    
    def test_get_residual_embedding_single(self):
        """Test single embedding residual calculation."""
        residual = self.calculator.get_residual_embedding(self.single_embedding)
        
        assert residual.shape == (1536,)
        # With identity projection matrix, residual should be zero
        assert np.allclose(residual, 0.0, atol=1e-10)
    
    def test_get_residual_embedding_batch(self):
        """Test batch embedding residual calculation."""
        residual = self.calculator.get_residual_embedding(self.embeddings)
        
        assert residual.shape == self.embeddings.shape
        # With identity projection matrix, residuals should be zero
        assert np.allclose(residual, 0.0, atol=1e-10)
    
    def test_calculate_tone_score(self):
        """Test tone score calculation."""
        # Create a non-zero residual for testing
        residual = np.random.randn(1536).astype(np.float32)
        tone_score = self.calculator.calculate_tone_score(residual, self.tone_axis_vector)
        
        assert isinstance(tone_score, float)
        expected_score = np.dot(residual, self.tone_axis_vector)
        assert abs(tone_score - expected_score) < 1e-10
    
    def test_calculate_tone_scores_batch(self):
        """Test batch tone score calculation."""
        # Create non-zero residuals for testing
        residuals = np.random.randn(10, 1536).astype(np.float32)
        tone_scores = self.calculator.calculate_tone_scores_batch(residuals, self.tone_axis_vector)
        
        assert tone_scores.shape == (10,)
        expected_scores = np.dot(residuals, self.tone_axis_vector)
        assert np.allclose(tone_scores, expected_scores, atol=1e-10)
    
    def test_get_topic_projection(self):
        """Test topic projection calculation."""
        projection = self.calculator.get_topic_projection(self.single_embedding)
        
        assert projection.shape == (1536,)
        # With identity projection matrix, projection should equal original
        assert np.allclose(projection, self.single_embedding, atol=1e-10)
    
    def test_get_topic_projection_batch(self):
        """Test batch topic projection calculation."""
        projection = self.calculator.get_topic_projection(self.embeddings)
        
        assert projection.shape == self.embeddings.shape
        # With identity projection matrix, projection should equal original
        assert np.allclose(projection, self.embeddings, atol=1e-10)
    
    def test_non_identity_projection(self):
        """Test with non-identity projection matrix."""
        # Create a simple projection matrix (project onto first 100 dimensions)
        projection_matrix = np.zeros((1536, 1536))
        projection_matrix[:100, :100] = np.eye(100)
        
        calculator = ResidualToneCalculator(projection_matrix)
        
        # Test with single embedding
        residual = calculator.get_residual_embedding(self.single_embedding)
        projection = calculator.get_topic_projection(self.single_embedding)
        
        # Check that residual + projection equals original
        reconstruction = residual + projection
        assert np.allclose(reconstruction, self.single_embedding, atol=1e-10)
        
        # Check that residual is orthogonal to projection
        dot_product = np.dot(residual, projection)
        assert abs(dot_product) < 1e-10
