"""
Tests for topic subspace extraction functionality.
"""

import pytest
import numpy as np
from residualized_tone_space.core.topic_subspace import TopicSubspaceExtractor


class TestTopicSubspaceExtractor:
    """Test cases for TopicSubspaceExtractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.embeddings = np.random.randn(1000, 1536).astype(np.float32)
        self.extractor = TopicSubspaceExtractor(method="svd", n_components=200)
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert self.extractor.method == "svd"
        assert self.extractor.n_components == 200
        assert self.extractor.random_state == 42
        assert self.extractor.topic_vectors is None
    
    def test_svd_extraction(self):
        """Test SVD-based topic extraction."""
        extractor = TopicSubspaceExtractor(method="svd", n_components=200)
        extractor.fit(self.embeddings)
        
        assert extractor.topic_vectors is not None
        assert extractor.topic_vectors.shape == (200, 1536)
        assert extractor.explained_variance is not None
        assert len(extractor.explained_variance) == 200
    
    def test_pca_extraction(self):
        """Test PCA-based topic extraction."""
        extractor = TopicSubspaceExtractor(method="pca", n_components=200)
        extractor.fit(self.embeddings)
        
        assert extractor.topic_vectors is not None
        assert extractor.topic_vectors.shape == (200, 1536)
        assert extractor.explained_variance is not None
        assert len(extractor.explained_variance) == 200
    
    def test_clustering_extraction(self):
        """Test clustering-based topic extraction."""
        extractor = TopicSubspaceExtractor(method="clustering", n_components=200)
        extractor.fit(self.embeddings)
        
        assert extractor.topic_vectors is not None
        assert extractor.topic_vectors.shape == (200, 1536)
        assert extractor.explained_variance is not None
        assert len(extractor.explained_variance) == 1000  # Clustering returns cluster labels
    
    def test_projection_matrix_creation(self):
        """Test projection matrix creation."""
        self.extractor.fit(self.embeddings)
        projection_matrix, orthonormal_basis = self.extractor.create_topic_projection_matrix()
        
        assert projection_matrix.shape == (1536, 1536)
        assert orthonormal_basis.shape[0] == 1536
        assert orthonormal_basis.shape[1] <= 200  # Should be <= n_components
    
    def test_invalid_method(self):
        """Test error handling for invalid method."""
        extractor = TopicSubspaceExtractor(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            extractor.fit(self.embeddings)
    
    def test_projection_matrix_without_fit(self):
        """Test error when creating projection matrix without fitting."""
        with pytest.raises(ValueError, match="Must fit the extractor first"):
            self.extractor.create_topic_projection_matrix()
