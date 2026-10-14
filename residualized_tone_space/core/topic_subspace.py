"""
Topic subspace extraction using PCA, SVD, and clustering methods.
"""

import numpy as np
from typing import Tuple, Optional, Union
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from scipy.linalg import qr
import structlog

logger = structlog.get_logger(__name__)


class TopicSubspaceExtractor:
    """
    Extract topic subspace from embeddings using various methods.
    """
    
    def __init__(self, method: str = "svd", n_components: int = 200, random_state: int = 42):
        """
        Initialize the topic subspace extractor.
        
        Args:
            method: Method to use ('pca', 'svd', 'clustering')
            n_components: Number of components/clusters
            random_state: Random state for reproducibility
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.topic_vectors: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None
        self.model = None
        
    def extract_topic_subspace_pca(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, PCA]:
        """
        Extract topic subspace using PCA on embeddings.
        
        Args:
            embeddings: Input embeddings of shape (n_samples, 1536)
            
        Returns:
            Tuple of (topic_vectors, explained_variance, pca_model)
        """
        logger.info("Extracting topic subspace using PCA", n_components=self.n_components)
        
        # Apply PCA to embeddings
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        pca.fit(embeddings)
        
        # Get topic basis vectors (principal components)
        topic_vectors = pca.components_  # Shape: (n_components, 1536)
        
        # Get explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        
        logger.info(
            "PCA extraction complete",
            explained_variance_ratio=explained_variance[:10].tolist(),
            total_variance_explained=explained_variance.sum()
        )
        
        return topic_vectors, explained_variance, pca
    
    def extract_topic_subspace_svd(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, TruncatedSVD]:
        """
        Extract topic subspace using SVD on embeddings.
        
        Args:
            embeddings: Input embeddings of shape (n_samples, 1536)
            
        Returns:
            Tuple of (topic_vectors, explained_variance, svd_model)
        """
        logger.info("Extracting topic subspace using SVD", n_components=self.n_components)
        
        # Apply SVD to embeddings
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        svd.fit(embeddings)
        
        # Get topic basis vectors (right singular vectors - principal components)
        topic_vectors = svd.components_  # Shape: (n_components, 1536)
        
        # Get explained variance ratio
        explained_variance = svd.explained_variance_ratio_
        
        logger.info(
            "SVD extraction complete",
            explained_variance_ratio=explained_variance[:10].tolist(),
            total_variance_explained=explained_variance.sum()
        )
        
        return topic_vectors, explained_variance, svd
    
    def extract_topic_subspace_clustering(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, KMeans]:
        """
        Extract topic subspace using K-means clustering.
        
        Args:
            embeddings: Input embeddings of shape (n_samples, 1536)
            
        Returns:
            Tuple of (topic_vectors, cluster_labels, kmeans_model)
        """
        logger.info("Extracting topic subspace using K-means clustering", n_clusters=self.n_components)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=10)
        kmeans.fit(embeddings)
        
        # Get topic basis vectors (cluster centroids)
        topic_vectors = kmeans.cluster_centers_  # Shape: (n_clusters, 1536)
        
        # Get cluster assignments
        cluster_labels = kmeans.labels_
        
        logger.info("K-means clustering complete", n_clusters=self.n_components)
        
        return topic_vectors, cluster_labels, kmeans
    
    def fit(self, embeddings: np.ndarray) -> 'TopicSubspaceExtractor':
        """
        Fit the topic subspace extractor to the embeddings.
        
        Args:
            embeddings: Input embeddings of shape (n_samples, 1536)
            
        Returns:
            Self for method chaining
        """
        if self.method == "pca":
            self.topic_vectors, self.explained_variance, self.model = self.extract_topic_subspace_pca(embeddings)
        elif self.method == "svd":
            self.topic_vectors, self.explained_variance, self.model = self.extract_topic_subspace_svd(embeddings)
        elif self.method == "clustering":
            self.topic_vectors, self.explained_variance, self.model = self.extract_topic_subspace_clustering(embeddings)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def create_topic_projection_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create projection matrix for topic subspace.
        
        Returns:
            Tuple of (projection_matrix, orthonormal_basis)
        """
        if self.topic_vectors is None:
            raise ValueError("Must fit the extractor first")
        
        logger.info("Creating topic projection matrix")
        
        # Orthonormalize topic vectors
        Q, R = qr(self.topic_vectors.T, mode='economic')
        
        # Create projection matrix
        topic_projection_matrix = Q @ Q.T  # Shape: (1536, 1536)
        
        logger.info("Topic projection matrix created", shape=topic_projection_matrix.shape)
        
        return topic_projection_matrix, Q
