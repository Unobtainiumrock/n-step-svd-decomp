"""
Core functionality for the Residualized Tone Space system.
"""

from .topic_subspace import TopicSubspaceExtractor
from .residual_calculator import ResidualToneCalculator
from .tone_axis_learning import ToneAxisLearner

__all__ = [
    "TopicSubspaceExtractor",
    "ResidualToneCalculator", 
    "ToneAxisLearner",
]
