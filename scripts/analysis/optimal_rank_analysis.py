#!/usr/bin/env python3
"""
Optimal Rank Analysis for Residualized Tone Space
Finds optimal number of components using elbow method and singular value analysis
"""
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import qr
import pandas as pd
import time
import os
from scipy.signal import find_peaks
from kneed import KneeLocator

def analyze_optimal_ranks():
    print("🔍 Optimal Rank Analysis for Residualized Tone Space")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading embeddings...")
    with h5py.File('data/embeddings/ugc-posts_COMPLETE_embeddings.h5', 'r') as f:
        embeddings = f['embeddings'][:]
        ids = [id.decode('utf-8') for id in f['ids'][:]]
        metadata = []
        if 'metadata' in f:
            metadata_json = [md.decode('utf-8') for md in f['metadata'][:]]
            metadata = [json.loads(md) for md in metadata_json]
    
    print(f"   ✅ Dataset: {len(embeddings):,} embeddings, {embeddings.shape[1]} dimensions")
    
    # Step 1: Analyze Topic Space Optimal Rank
    print(f"\n🔍 Step 1: Finding Optimal Rank for Topic Space...")
    topic_ranks, topic_variances, topic_singular_values, optimal_topic_rank = analyze_topic_space_ranks(embeddings)
    
    # Step 2: Calculate Residuals with Optimal Topic Rank
    print(f"\n🎯 Step 2: Computing Residuals with Optimal Topic Rank ({optimal_topic_rank})...")
    residual_vectors = compute_residuals_with_rank(embeddings, optimal_topic_rank)
    
    # Step 3: Analyze Tone Space Optimal Rank
    print(f"\n🎵 Step 3: Finding Optimal Rank for Tone Space...")
    tone_ranks, tone_variances, tone_singular_values, optimal_tone_rank = analyze_tone_space_ranks(residual_vectors)
    
    # Step 4: Compare Different Rank Combinations
    print(f"\n📈 Step 4: Comparing Rank Combinations...")
    rank_comparison = compare_rank_combinations(embeddings, topic_ranks, tone_ranks)
    
    # Step 5: Create Comprehensive Visualization
    print(f"\n🎨 Step 5: Creating Optimal Rank Visualization...")
    create_optimal_rank_visualization(
        topic_ranks, topic_variances, topic_singular_values, optimal_topic_rank,
        tone_ranks, tone_variances, tone_singular_values, optimal_tone_rank,
        rank_comparison
    )
    
    # Step 6: Run Analysis with Optimal Ranks
    print(f"\n🚀 Step 6: Running Analysis with Optimal Ranks...")
    optimal_results = run_optimal_analysis(embeddings, optimal_topic_rank, optimal_tone_rank)
    
    # Step 7: Save Results
    print(f"\n💾 Step 7: Saving Optimal Rank Results...")
    save_optimal_results(optimal_results, optimal_topic_rank, optimal_tone_rank, 
                        topic_ranks, topic_variances, tone_ranks, tone_variances)
    
    print(f"\n🎉 Optimal Rank Analysis Complete!")
    print(f"   📊 Optimal Topic Rank: {optimal_topic_rank}")
    print(f"   📊 Optimal Tone Rank: {optimal_tone_rank}")
    print(f"   📊 Topic Variance Captured: {topic_variances[topic_ranks.index(optimal_topic_rank)]:.1%}")
    print(f"   📊 Tone Variance Captured: {tone_variances[tone_ranks.index(optimal_tone_rank)]:.1%}")

def analyze_topic_space_ranks(embeddings, max_rank=500, step=10):
    """Analyze different ranks for topic space using elbow method."""
    
    ranks_to_test = list(range(10, min(max_rank, embeddings.shape[1]), step))
    explained_variances = []
    singular_values_list = []
    
    print(f"   🔄 Testing ranks: {ranks_to_test[0]} to {ranks_to_test[-1]} (step {step})")
    
    for rank in ranks_to_test:
        print(f"      Testing topic rank {rank}...", end=" ")
        
        start_time = time.time()
        svd = TruncatedSVD(n_components=rank, random_state=42)
        svd.fit(embeddings)
        
        explained_variance = np.sum(svd.explained_variance_ratio_)
        explained_variances.append(explained_variance)
        singular_values_list.append(svd.singular_values_)
        
        print(f"Variance: {explained_variance:.3f} ({time.time()-start_time:.1f}s)")
    
    # Find elbow using KneeLocator
    try:
        knee_locator = KneeLocator(ranks_to_test, explained_variances, 
                                 curve='concave', direction='increasing')
        optimal_rank = knee_locator.elbow
        
        if optimal_rank is None:
            # Fallback: find point where improvement drops below 1%
            improvements = np.diff(explained_variances)
            threshold_idx = np.where(improvements < 0.01)[0]
            optimal_rank = ranks_to_test[threshold_idx[0]] if len(threshold_idx) > 0 else ranks_to_test[-1]
            
    except Exception as e:
        print(f"      Warning: Elbow detection failed ({e}), using 90% variance threshold")
        # Fallback: 90% variance
        target_variance = 0.90
        optimal_idx = np.argmax(np.array(explained_variances) >= target_variance)
        optimal_rank = ranks_to_test[optimal_idx] if optimal_idx > 0 else ranks_to_test[-1]
    
    print(f"   ✅ Optimal topic rank: {optimal_rank}")
    
    return ranks_to_test, explained_variances, singular_values_list, optimal_rank

def analyze_tone_space_ranks(residual_vectors, max_rank=300, step=10):
    """Analyze different ranks for tone space using elbow method."""
    
    ranks_to_test = list(range(10, min(max_rank, residual_vectors.shape[1]), step))
    explained_variances = []
    singular_values_list = []
    
    print(f"   🔄 Testing ranks: {ranks_to_test[0]} to {ranks_to_test[-1]} (step {step})")
    
    for rank in ranks_to_test:
        print(f"      Testing tone rank {rank}...", end=" ")
        
        start_time = time.time()
        svd = TruncatedSVD(n_components=rank, random_state=42)
        svd.fit(residual_vectors)
        
        explained_variance = np.sum(svd.explained_variance_ratio_)
        explained_variances.append(explained_variance)
        singular_values_list.append(svd.singular_values_)
        
        print(f"Variance: {explained_variance:.3f} ({time.time()-start_time:.1f}s)")
    
    # Find elbow using KneeLocator
    try:
        knee_locator = KneeLocator(ranks_to_test, explained_variances, 
                                 curve='concave', direction='increasing')
        optimal_rank = knee_locator.elbow
        
        if optimal_rank is None:
            # Fallback: find point where improvement drops below 0.5%
            improvements = np.diff(explained_variances)
            threshold_idx = np.where(improvements < 0.005)[0]
            optimal_rank = ranks_to_test[threshold_idx[0]] if len(threshold_idx) > 0 else ranks_to_test[-1]
            
    except Exception as e:
        print(f"      Warning: Elbow detection failed ({e}), using 70% variance threshold")
        # Fallback: 70% variance for tone space
        target_variance = 0.70
        optimal_idx = np.argmax(np.array(explained_variances) >= target_variance)
        optimal_rank = ranks_to_test[optimal_idx] if optimal_idx > 0 else ranks_to_test[-1]
    
    print(f"   ✅ Optimal tone rank: {optimal_rank}")
    
    return ranks_to_test, explained_variances, singular_values_list, optimal_rank

def compute_residuals_with_rank(embeddings, topic_rank):
    """Compute residual vectors using specified topic rank."""
    
    print(f"      Computing residuals with topic rank {topic_rank}...")
    
    # Topic discovery with optimal rank
    topic_svd = TruncatedSVD(n_components=topic_rank, random_state=42)
    topic_svd.fit(embeddings)
    topic_vectors = topic_svd.components_
    
    # Create projection matrix
    Q, R = qr(topic_vectors.T, mode='economic')
    topic_projection_matrix = Q @ Q.T
    
    # Calculate residuals
    topic_projections = embeddings @ topic_projection_matrix
    residual_vectors = embeddings - topic_projections
    
    print(f"      ✅ Residuals computed: {residual_vectors.shape}")
    
    return residual_vectors

def compare_rank_combinations(embeddings, topic_ranks, tone_ranks):
    """Compare different combinations of topic and tone ranks."""
    
    # Test a few key combinations
    combinations_to_test = [
        (50, 50), (100, 100), (200, 200), (300, 200), (400, 300)
    ]
    
    # Add the extremes from our tested ranges
    combinations_to_test.extend([
        (topic_ranks[0], tone_ranks[0]),  # Min-min
        (topic_ranks[-1], tone_ranks[-1]),  # Max-max
    ])
    
    results = []
    
    for topic_rank, tone_rank in combinations_to_test:
        if topic_rank <= max(topic_ranks) and tone_rank <= max(tone_ranks):
            print(f"      Testing combination: Topic={topic_rank}, Tone={tone_rank}")
            
            # Compute residuals
            residual_vectors = compute_residuals_with_rank(embeddings, topic_rank)
            
            # Compute tone variance
            tone_svd = TruncatedSVD(n_components=min(tone_rank, residual_vectors.shape[1]-1), random_state=42)
            tone_svd.fit(residual_vectors)
            tone_variance = np.sum(tone_svd.explained_variance_ratio_)
            
            # Compute topic variance
            topic_svd = TruncatedSVD(n_components=topic_rank, random_state=42)
            topic_svd.fit(embeddings)
            topic_variance = np.sum(topic_svd.explained_variance_ratio_)
            
            results.append({
                'topic_rank': topic_rank,
                'tone_rank': tone_rank,
                'topic_variance': topic_variance,
                'tone_variance': tone_variance,
                'total_params': topic_rank + tone_rank,
                'efficiency': (topic_variance + tone_variance * 0.5) / (topic_rank + tone_rank)  # Weighted efficiency
            })
    
    return results

def run_optimal_analysis(embeddings, optimal_topic_rank, optimal_tone_rank):
    """Run the full analysis with optimal ranks."""
    
    print(f"      Running full analysis with Topic={optimal_topic_rank}, Tone={optimal_tone_rank}")
    
    start_time = time.time()
    
    # Topic discovery
    topic_svd = TruncatedSVD(n_components=optimal_topic_rank, random_state=42)
    topic_svd.fit(embeddings)
    topic_vectors = topic_svd.components_
    topic_variance = np.sum(topic_svd.explained_variance_ratio_)
    
    # Create projection matrix
    Q, R = qr(topic_vectors.T, mode='economic')
    topic_projection_matrix = Q @ Q.T
    
    # Calculate residuals
    topic_projections = embeddings @ topic_projection_matrix
    residual_vectors = embeddings - topic_projections
    
    # Tone discovery
    tone_svd = TruncatedSVD(n_components=optimal_tone_rank, random_state=42)
    tone_svd.fit(residual_vectors)
    tone_axes = tone_svd.components_
    tone_variance = np.sum(tone_svd.explained_variance_ratio_)
    
    # Validation
    orthogonality_check = np.mean(np.abs(residual_vectors @ Q))
    reconstruction_error = np.mean(np.linalg.norm(embeddings - (topic_projections + residual_vectors), axis=1))
    
    total_time = time.time() - start_time
    
    print(f"      ✅ Analysis completed in {total_time:.2f}s")
    print(f"         Topic variance: {topic_variance:.1%}")
    print(f"         Tone variance: {tone_variance:.1%}")
    print(f"         Orthogonality: {orthogonality_check:.6f}")
    print(f"         Reconstruction error: {reconstruction_error:.6f}")
    
    return {
        'topic_vectors': topic_vectors,
        'tone_axes': tone_axes,
        'topic_projection_matrix': topic_projection_matrix,
        'residual_vectors': residual_vectors,
        'topic_projections': topic_projections,
        'topic_variance': topic_variance,
        'tone_variance': tone_variance,
        'orthogonality_check': orthogonality_check,
        'reconstruction_error': reconstruction_error,
        'processing_time': total_time
    }

def create_optimal_rank_visualization(topic_ranks, topic_variances, topic_singular_values, optimal_topic_rank,
                                    tone_ranks, tone_variances, tone_singular_values, optimal_tone_rank,
                                    rank_comparison):
    """Create comprehensive visualization of optimal rank analysis."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Topic Space Elbow Plot
    plt.subplot(2, 4, 1)
    plt.plot(topic_ranks, topic_variances, 'b-', linewidth=2, label='Topic Variance')
    plt.axvline(x=optimal_topic_rank, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_topic_rank}')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Topic Space: Elbow Method', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Topic Singular Values
    plt.subplot(2, 4, 2)
    # Use singular values from optimal rank
    optimal_idx = topic_ranks.index(optimal_topic_rank)
    singular_vals = topic_singular_values[optimal_idx]
    plt.plot(range(1, len(singular_vals)+1), singular_vals, 'b-', linewidth=2)
    plt.xlabel('Component Index')
    plt.ylabel('Singular Value')
    plt.title(f'Topic Singular Values (k={optimal_topic_rank})', fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 3. Tone Space Elbow Plot
    plt.subplot(2, 4, 3)
    plt.plot(tone_ranks, tone_variances, 'g-', linewidth=2, label='Tone Variance')
    plt.axvline(x=optimal_tone_rank, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_tone_rank}')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Tone Space: Elbow Method', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Tone Singular Values
    plt.subplot(2, 4, 4)
    optimal_tone_idx = tone_ranks.index(optimal_tone_rank)
    tone_singular_vals = tone_singular_values[optimal_tone_idx]
    plt.plot(range(1, len(tone_singular_vals)+1), tone_singular_vals, 'g-', linewidth=2)
    plt.xlabel('Component Index')
    plt.ylabel('Singular Value')
    plt.title(f'Tone Singular Values (k={optimal_tone_rank})', fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 5. Rank Combination Comparison
    plt.subplot(2, 4, 5)
    if rank_comparison:
        comparison_df = pd.DataFrame(rank_comparison)
        plt.scatter(comparison_df['total_params'], comparison_df['topic_variance'], 
                   c='blue', alpha=0.7, s=100, label='Topic Variance')
        plt.scatter(comparison_df['total_params'], comparison_df['tone_variance'], 
                   c='green', alpha=0.7, s=100, label='Tone Variance')
        plt.xlabel('Total Parameters (Topic + Tone Rank)')
        plt.ylabel('Explained Variance')
        plt.title('Rank Combinations Comparison', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No comparison data available', ha='center', va='center')
        plt.title('Rank Combinations Comparison', fontweight='bold')
    
    # 6. Efficiency Analysis
    plt.subplot(2, 4, 6)
    if rank_comparison:
        comparison_df = pd.DataFrame(rank_comparison)
        plt.scatter(comparison_df['total_params'], comparison_df['efficiency'], 
                   c='purple', alpha=0.7, s=100)
        plt.xlabel('Total Parameters')
        plt.ylabel('Efficiency Score')
        plt.title('Parameter Efficiency', fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No efficiency data available', ha='center', va='center')
        plt.title('Parameter Efficiency', fontweight='bold')
    
    # 7. Variance Improvements
    plt.subplot(2, 4, 7)
    topic_improvements = np.diff(topic_variances)
    tone_improvements = np.diff(tone_variances)
    
    plt.plot(topic_ranks[1:], topic_improvements, 'b-', linewidth=2, label='Topic Improvement')
    plt.plot(tone_ranks[1:], tone_improvements, 'g-', linewidth=2, label='Tone Improvement')
    plt.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='1% Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Improvement')
    plt.title('Marginal Variance Gains', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Summary Statistics
    plt.subplot(2, 4, 8)
    plt.axis('off')
    
    summary_text = f"""
    Optimal Rank Analysis Results:
    
    Topic Space:
    • Optimal rank: {optimal_topic_rank}
    • Variance captured: {topic_variances[topic_ranks.index(optimal_topic_rank)]:.1%}
    • Efficiency: {topic_variances[topic_ranks.index(optimal_topic_rank)]/optimal_topic_rank:.4f}
    
    Tone Space:
    • Optimal rank: {optimal_tone_rank}
    • Variance captured: {tone_variances[tone_ranks.index(optimal_tone_rank)]:.1%}
    • Efficiency: {tone_variances[tone_ranks.index(optimal_tone_rank)]/optimal_tone_rank:.4f}
    
    Total Parameters: {optimal_topic_rank + optimal_tone_rank}
    
    Best Efficiency Combination:
    • {max(rank_comparison, key=lambda x: x['efficiency']) if rank_comparison else 'N/A'}
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('data/models/optimal_rank_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("      ✅ Visualization saved to data/models/optimal_rank_analysis.png")

def save_optimal_results(results, optimal_topic_rank, optimal_tone_rank, 
                        topic_ranks, topic_variances, tone_ranks, tone_variances):
    """Save optimal analysis results."""
    
    os.makedirs('data/models', exist_ok=True)
    
    # Save optimal model components
    np.save('data/models/optimal_topic_vectors.npy', results['topic_vectors'])
    np.save('data/models/optimal_tone_axes.npy', results['tone_axes'])
    np.save('data/models/optimal_projection_matrix.npy', results['topic_projection_matrix'])
    np.save('data/models/optimal_residual_vectors.npy', results['residual_vectors'])
    np.save('data/models/optimal_topic_projections.npy', results['topic_projections'])
    
    # Save analysis metadata
    analysis_metadata = {
        'optimal_topic_rank': optimal_topic_rank,
        'optimal_tone_rank': optimal_tone_rank,
        'topic_variance': results['topic_variance'],
        'tone_variance': results['tone_variance'],
        'orthogonality_check': results['orthogonality_check'],
        'reconstruction_error': results['reconstruction_error'],
        'processing_time': results['processing_time'],
        'topic_ranks_tested': topic_ranks,
        'topic_variances': topic_variances,
        'tone_ranks_tested': tone_ranks,
        'tone_variances': tone_variances
    }
    
    np.save('data/models/optimal_analysis_metadata.npy', analysis_metadata, allow_pickle=True)
    
    print(f"      ✅ Optimal results saved with ranks: Topic={optimal_topic_rank}, Tone={optimal_tone_rank}")

if __name__ == "__main__":
    analyze_optimal_ranks() 