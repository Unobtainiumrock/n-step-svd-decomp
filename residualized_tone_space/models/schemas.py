"""
Pydantic schemas for data validation and serialization.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import numpy as np


class EmbeddingData(BaseModel):
    """Schema for embedding data."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    
    class Config:
        arbitrary_types_allowed = True


class TopicSubspaceData(BaseModel):
    """Schema for topic subspace data."""
    topic_vectors: List[List[float]] = Field(..., description="Topic basis vectors")
    explained_variance: List[float] = Field(..., description="Explained variance ratios")
    method: str = Field(..., description="Extraction method used")
    n_components: int = Field(..., description="Number of components")
    
    class Config:
        arbitrary_types_allowed = True


class ResidualData(BaseModel):
    """Schema for residual data."""
    residual_vectors: List[List[float]] = Field(..., description="Residual vectors")
    original_embeddings: List[List[float]] = Field(..., description="Original embeddings")
    topic_projections: List[List[float]] = Field(..., description="Topic projections")
    
    class Config:
        arbitrary_types_allowed = True


class ToneAxisChoice(BaseModel):
    """Schema for tone axis choice data."""
    tone_axis_id: str = Field(..., description="Tone axis identifier")
    chosen_idx: int = Field(..., description="Index of chosen content")
    rejected_idx: int = Field(..., description="Index of rejected content")
    timestamp: Optional[str] = Field(None, description="Choice timestamp")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ToneAxisPreference(BaseModel):
    """Schema for tone axis vector vector."""
    tone_axis_id: str = Field(..., description="Tone axis identifier")
    preference_vector: List[float] = Field(..., description="Preference vector in residual space")
    confidence: Optional[float] = Field(None, description="Model confidence")
    n_training_samples: int = Field(..., description="Number of training samples")


class ToneScore(BaseModel):
    """Schema for tone score."""
    content_idx: int = Field(..., description="Content index")
    tone_axis_id: str = Field(..., description="Tone axis identifier")
    tone_score: float = Field(..., description="Tone score")
    confidence: Optional[float] = Field(None, description="Score confidence")


class ContentPair(BaseModel):
    """Schema for content pair generation."""
    content_a_idx: int = Field(..., description="Index of content A")
    content_b_idx: int = Field(..., description="Index of content B")
    expected_information_gain: Optional[float] = Field(None, description="Expected information gain")
    topic_similarity: Optional[float] = Field(None, description="Topic similarity between pair")


class TopicToneSeparationMetrics(BaseModel):
    """Schema for topic-tone separation validation metrics."""
    topic_coherence: float = Field(..., description="Topic coherence score")
    tone_separation: float = Field(..., description="Tone separation score")
    cross_topic_consistency: float = Field(..., description="Cross-topic consistency score")
    reconstruction_error: float = Field(..., description="Reconstruction error")


class SystemPerformanceMetrics(BaseModel):
    """Schema for system performance metrics."""
    latency_ms: float = Field(..., description="Average latency in milliseconds")
    throughput_per_second: float = Field(..., description="Throughput per second")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
