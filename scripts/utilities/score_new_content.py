#!/usr/bin/env python3
"""
Score new content using the residualized tone space
"""
import numpy as np
from residualized_tone_space.core import ResidualToneCalculator

def score_new_content(new_embedding):
    """Score a single new embedding."""
    
    # Load optimal projection matrix
    projection_matrix = np.load("data/reports/optimal_160_projection_matrix.npy")
    calculator = ResidualToneCalculator(projection_matrix)
    
    # Load tone axes
    tone_axes = np.load("data/reports/optimal_130_tone_axes.npy")
    tone_axis_names = [f"Tone_Axis_{i+1}" for i in range(len(tone_axes))]
    calculator.load_tone_axis_vectors(tone_axis_names, tone_axes)
    
    # Score the content
    all_scores = calculator.calculate_tone_scores_all_axes(new_embedding.reshape(1, -1))
    
    # Get top positive and negative scores
    results = {}
    for axis_name, scores in all_scores.items():
        results[axis_name] = float(scores[0])
    
    # Sort by absolute strength
    sorted_scores = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"🎵 Top 10 Tone Patterns:")
    for i, (axis, score) in enumerate(sorted_scores[:10]):
        direction = "+" if score > 0 else "-"
        print(f"   {i+1:2d}. {axis}: {direction}{abs(score):.3f}")
    
    return results

if __name__ == "__main__":
    # Example: Score a random embedding from your dataset
    import h5py
    with h5py.File('data/embeddings/ugc-posts_ALL_embeddings.h5', 'r') as f:
        sample_embedding = f['embeddings'][0]  # First embedding
        sample_id = f['ids'][0].decode('utf-8')
    
    print(f"�� Scoring content: {sample_id}")
    scores = score_new_content(sample_embedding)
