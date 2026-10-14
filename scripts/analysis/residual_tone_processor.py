#!/usr/bin/env python3
"""
Residual Tone Processor - Use optimal components for real content analysis
"""
import numpy as np
import h5py
import json
from pathlib import Path
from residualized_tone_space.core import ResidualToneCalculator

def create_production_calculator():
    """Create ResidualToneCalculator with optimal pre-computed components."""
    
    # Load optimal projection matrix (160 topic dimensions)
    projection_matrix = np.load("data/reports/optimal_160_projection_matrix.npy")
    
    # Initialize calculator
    calculator = ResidualToneCalculator(projection_matrix)
    
    # Load optimal tone axes (130 tone dimensions)
    tone_axes = np.load("data/reports/optimal_130_tone_axes.npy")
    
    # Create tone axis names and load them
    tone_axis_names = [f"Tone_Axis_{i+1}" for i in range(len(tone_axes))]
    calculator.load_tone_axis_vectors(tone_axis_names, tone_axes)
    
    print(f"✅ Calculator initialized with:")
    print(f"   📊 Topic dimensions: {projection_matrix.shape}")
    print(f"   🎵 Tone axes: {len(tone_axes)}")
    
    return calculator

def process_content(calculator, embeddings, content_ids=None):
    """Process content through residualized tone space."""
    
    print(f"\n🔄 Processing {len(embeddings):,} embeddings...")
    
    # Step 1: Calculate residual vectors (removes topic information)
    residual_vectors = calculator.calculate_residual_vectors(embeddings)
    print(f"   ✅ Residual vectors calculated: {residual_vectors.shape}")
    
    # Step 2: Calculate tone scores for all axes
    all_tone_scores = calculator.calculate_tone_scores_all_axes(embeddings)
    print(f"   ✅ Tone scores calculated for {len(all_tone_scores)} axes")
    
    # Step 3: Get topic projections for analysis
    topic_projections = calculator.get_topic_projection(embeddings)
    print(f"   ✅ Topic projections calculated: {topic_projections.shape}")
    
    return {
        'residual_vectors': residual_vectors,
        'tone_scores': all_tone_scores,
        'topic_projections': topic_projections,
        'content_ids': content_ids
    }

def analyze_tone_patterns(tone_scores):
    """Analyze tone patterns across content."""
    
    print("\n📊 Tone Pattern Analysis:")
    
    analysis = {}
    for axis_name, scores in tone_scores.items():
        stats = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'range': float(np.max(scores) - np.min(scores))
        }
        
        # Determine dominant direction
        stats['dominant_direction'] = 'positive' if stats['mean'] > 0 else 'negative'
        stats['strength'] = abs(stats['mean'])
        
        analysis[axis_name] = stats
        
        print(f"   {axis_name}: {stats['dominant_direction']} "
              f"(μ={stats['mean']:.3f}, σ={stats['std']:.3f}, "
              f"range=[{stats['min']:.3f}, {stats['max']:.3f}])")
    
    return analysis

def find_extreme_content(tone_scores, content_ids=None, top_k=5):
    """Find content with extreme tone scores."""
    
    print(f"\n🎯 Finding top {top_k} extreme content for each tone axis:")
    
    extremes = {}
    
    for axis_name, scores in tone_scores.items():
        # Find highest and lowest scores
        high_indices = np.argsort(scores)[-top_k:][::-1]  # Top k highest
        low_indices = np.argsort(scores)[:top_k]  # Top k lowest
        
        extremes[axis_name] = {
            'highest': {
                'indices': high_indices.tolist(),
                'scores': scores[high_indices].tolist(),
                'content_ids': [content_ids[i] if content_ids else f"content_{i}" for i in high_indices]
            },
            'lowest': {
                'indices': low_indices.tolist(), 
                'scores': scores[low_indices].tolist(),
                'content_ids': [content_ids[i] if content_ids else f"content_{i}" for i in low_indices]
            }
        }
        
        print(f"\n   {axis_name}:")
        print(f"     Highest: {scores[high_indices[0]]:.3f} (ID: {extremes[axis_name]['highest']['content_ids'][0]})")
        print(f"     Lowest:  {scores[low_indices[0]]:.3f} (ID: {extremes[axis_name]['lowest']['content_ids'][0]})")
    
    return extremes

def save_analysis_results(results, analysis, extremes, output_file="data/production/tone_analysis.json"):
    """Save analysis results to file."""
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON serialization
    output_data = {
        'metadata': {
            'n_embeddings': len(results['content_ids']) if results['content_ids'] else results['residual_vectors'].shape[0],
            'n_tone_axes': len(results['tone_scores']),
            'topic_dimensions': results['topic_projections'].shape[1],
            'residual_dimensions': results['residual_vectors'].shape[1]
        },
        'tone_analysis': analysis,
        'extreme_content': extremes,
        'summary': {
            'strongest_tone_axis': max(analysis.keys(), key=lambda x: analysis[x]['strength']),
            'most_variable_axis': max(analysis.keys(), key=lambda x: analysis[x]['std']),
            'widest_range_axis': max(analysis.keys(), key=lambda x: analysis[x]['range'])
        }
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")
    
    return output_file

def main():
    """Main processing pipeline."""
    
    print("🎵 Residual Tone Space Processor")
    print("=" * 40)
    
    # Step 1: Create calculator with optimal components
    calculator = create_production_calculator()
    
    # Step 2: Load your embeddings
    print("\n📂 Loading embeddings...")
    with h5py.File('data/embeddings/ugc-posts_ALL_embeddings.h5', 'r') as f:
        embeddings = f['embeddings'][:]
        ids = [id.decode('utf-8') for id in f['ids'][:]]
    
    print(f"   ✅ Loaded {len(embeddings):,} embeddings")
    
    # Optional: Process subset for faster testing
    # embeddings = embeddings[:5000]  # First 5k for testing
    # ids = ids[:5000]
    
    # Step 3: Process content
    results = process_content(calculator, embeddings, ids)
    
    # Step 4: Analyze tone patterns
    analysis = analyze_tone_patterns(results['tone_scores'])
    
    # Step 5: Find extreme content
    extremes = find_extreme_content(results['tone_scores'], results['content_ids'])
    
    # Step 6: Save results
    output_file = save_analysis_results(results, analysis, extremes)
    
    # Step 7: Summary
    print("\n🎉 Processing Complete!")
    print(f"   📊 Processed: {len(embeddings):,} embeddings")
    print(f"   🎵 Analyzed: {len(results['tone_scores'])} tone axes")
    print(f"   📁 Results: {output_file}")
    
    # Show key insights
    strongest_axis = max(analysis.keys(), key=lambda x: analysis[x]['strength'])
    most_variable = max(analysis.keys(), key=lambda x: analysis[x]['std'])
    
    print(f"\n🔍 Key Insights:")
    print(f"   💪 Strongest tone axis: {strongest_axis} (strength: {analysis[strongest_axis]['strength']:.3f})")
    print(f"   📈 Most variable axis: {most_variable} (std: {analysis[most_variable]['std']:.3f})")
    
    return results, analysis, extremes

if __name__ == "__main__":
    results, analysis, extremes = main()
