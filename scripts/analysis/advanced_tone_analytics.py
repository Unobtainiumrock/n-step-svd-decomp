#!/usr/bin/env python3
"""
Advanced analytics for tone patterns
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_tone_patterns():
    """Generate advanced analytics from tone analysis."""
    
    # Load analysis results
    with open('data/production/tone_analysis.json', 'r') as f:
        data = json.load(f)
    
    tone_analysis = data['tone_analysis']
    
    # Extract metrics for analysis
    axes = list(tone_analysis.keys())
    means = [tone_analysis[axis]['mean'] for axis in axes]
    stds = [tone_analysis[axis]['std'] for axis in axes]
    ranges = [tone_analysis[axis]['range'] for axis in axes]
    strengths = [tone_analysis[axis]['strength'] for axis in axes]
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Standard deviation distribution
    ax1.hist(stds, bins=20, alpha=0.7, color='skyblue')
    ax1.set_title('Distribution of Tone Axis Variability (Std Dev)')
    ax1.set_xlabel('Standard Deviation')
    ax1.set_ylabel('Number of Tone Axes')
    
    # 2. Range vs Standard deviation
    ax2.scatter(stds, ranges, alpha=0.6, color='coral')
    ax2.set_title('Tone Axis Range vs Variability')
    ax2.set_xlabel('Standard Deviation')
    ax2.set_ylabel('Range')
    
    # 3. Top 20 most variable axes
    top_20_indices = np.argsort(stds)[-20:]
    top_20_axes = [axes[i] for i in top_20_indices]
    top_20_stds = [stds[i] for i in top_20_indices]
    
    ax3.barh(range(20), top_20_stds)
    ax3.set_yticks(range(20))
    ax3.set_yticklabels([f"Axis_{i+1}" for i in range(111, 131)])  # Top 20
    ax3.set_title('Top 20 Most Variable Tone Axes')
    ax3.set_xlabel('Standard Deviation')
    
    # 4. Strength distribution
    ax4.hist(strengths, bins=20, alpha=0.7, color='lightgreen')
    ax4.set_title('Distribution of Tone Axis Strengths')
    ax4.set_xlabel('Absolute Mean Score')
    ax4.set_ylabel('Number of Tone Axes')
    
    plt.tight_layout()
    plt.savefig('data/production/tone_analytics_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate insights
    insights = {
        'total_axes': len(axes),
        'most_variable_axis': axes[np.argmax(stds)],
        'max_variability': max(stds),
        'least_variable_axis': axes[np.argmin(stds)],
        'min_variability': min(stds),
        'widest_range_axis': axes[np.argmax(ranges)],
        'max_range': max(ranges),
        'strongest_axis': axes[np.argmax(strengths)],
        'max_strength': max(strengths),
        'variability_stats': {
            'mean_std': np.mean(stds),
            'std_of_stds': np.std(stds),
            'median_std': np.median(stds)
        }
    }
    
    # Save insights
    with open('data/production/tone_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print("📊 Advanced Analytics Generated:")
    print(f"   🎯 Most variable axis: {insights['most_variable_axis']} (σ={insights['max_variability']:.4f})")
    print(f"   📏 Widest range axis: {insights['widest_range_axis']} (range={insights['max_range']:.4f})")
    print(f"   💪 Strongest axis: {insights['strongest_axis']} (strength={insights['max_strength']:.6f})")
    print(f"   📈 Average variability: {insights['variability_stats']['mean_std']:.4f}")
    print(f"   📊 Visualization saved: data/production/tone_analytics_overview.png")
    
    return insights

if __name__ == "__main__":
    analyze_tone_patterns()
