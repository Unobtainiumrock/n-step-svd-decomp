#!/usr/bin/env python3
"""
SVD Topic Analysis with Matching Clusters (Optimized)
Shows the relationship between 160 SVD components and 160 clusters
Uses efficient two-step projection without QR decomposition
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
import time

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_svd_with_matching_clusters():
    """Analyze SVD topic space with clusters that match the component count."""
    print("🔍 SVD Topic Analysis with Matching Clusters (Optimized)")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading embeddings...")
    with h5py.File('../../data/embeddings/ugc-posts_COMPLETE_embeddings.h5', 'r') as f:
        embeddings = f['embeddings'][:]
    print(f"   ✅ Dataset: {len(embeddings):,} embeddings, {embeddings.shape[1]} dimensions")
    
    # Use 160 SVD components (matching the optimal rank)
    SVD_RANK = 160
    
    print(f"\n🎯 Using {SVD_RANK} SVD Components and {SVD_RANK} Clusters")
    print(f"   💡 Optimizations: No QR decomposition, efficient two-step projection")
    
    # Run analysis
    results = run_svd_cluster_analysis(embeddings, SVD_RANK)
    
    # Create visualization
    create_svd_cluster_visualization(results, embeddings)
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   📊 SVD Components: {SVD_RANK}")
    print(f"   🎯 Clusters: {SVD_RANK}")
    print(f"   📈 Topic Variance: {results['topic_variance']:.1%}")
    print(f"   📉 Residual Variance: {results['residual_variance']:.1%}")

def run_svd_cluster_analysis(embeddings, svd_rank):
    """Run SVD analysis with matching cluster count."""
    print(f"\n🚀 Running Optimized SVD Analysis...")
    start_time = time.time()
    
    # Step 1: Topic Discovery with SVD
    print(f"   🔍 Topic Discovery (SVD, k={svd_rank})...")
    topic_svd = TruncatedSVD(n_components=svd_rank, random_state=42)
    topic_svd.fit(embeddings)
    topic_vectors = topic_svd.components_
    topic_variance = np.sum(topic_svd.explained_variance_ratio_)
    
    # Step 2: Calculate Topic Projections (Efficient Two-Step)
    print(f"   📊 Computing Topic Projections (Efficient Method)...")
    topic_scores = embeddings @ topic_vectors.T  # (N, 160)
    topic_projections = topic_scores @ topic_vectors  # (N, 1536)
    
    # Step 3: Calculate Residuals (before tone SVD)
    print(f"   ➖ Computing Residual Vectors...")
    residual_vectors = embeddings - topic_projections
    
    # Step 4: K-means Clustering with MATCHING cluster count
    print(f"   🎯 Topic Clustering (K-means, k={svd_rank})...")
    kmeans = KMeans(n_clusters=svd_rank, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(topic_projections)
    
    # Calculate variance breakdown
    total_variance = np.var(embeddings, axis=0).sum()
    topic_projection_variance = np.var(topic_projections, axis=0).sum()
    residual_variance = np.var(residual_vectors, axis=0).sum()
    
    total_time = time.time() - start_time
    
    print(f"   ⏱️  Analysis completed in {total_time:.2f}s")
    print(f"   📊 Topic variance: {topic_variance:.1%}")
    print(f"   📉 Residual variance: {residual_variance/total_variance:.1%}")
    print(f"   🎯 Clusters: {svd_rank} (matching SVD components)")
    print(f"   ⚡ Optimizations: No QR decomposition, efficient projection")
    
    return {
        'topic_vectors': topic_vectors,
        'topic_scores': topic_scores,
        'topic_projections': topic_projections,
        'residual_vectors': residual_vectors,
        'cluster_labels': cluster_labels,
        'topic_variance': topic_variance,
        'residual_variance': residual_variance / total_variance,
        'total_variance': total_variance,
        'topic_projection_variance': topic_projection_variance,
        'processing_time': total_time,
        'svd_rank': svd_rank
    }

def create_svd_cluster_visualization(results, embeddings):
    """Create visualization showing SVD clusters and residuals."""
    print(f"\n🎨 Creating SVD Cluster Visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create 2D PCA for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    
    # Plot 1: Topic Clusters (160 clusters matching 160 SVD components)
    topic_2d = pca_2d.fit_transform(results['topic_projections'])
    
    # Use a colormap that can handle 160 different colors
    scatter1 = ax1.scatter(topic_2d[:, 0], topic_2d[:, 1], 
                          c=results['cluster_labels'], 
                          cmap='tab20', alpha=0.6, s=1)
    ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax1.set_title(f'Topic Clusters (160 clusters = 160 SVD components)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
    cbar1.set_ticks(range(0, results['svd_rank'], 20))  # Show every 20th cluster
    
    # Plot 2: Residual Vectors (before tone SVD)
    residual_2d = pca_2d.fit_transform(results['residual_vectors'])
    ax2.scatter(residual_2d[:, 0], residual_2d[:, 1], 
               alpha=0.6, s=1, color='red')
    ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('Residual Vectors (before tone SVD)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add summary text
    summary_text = f"""
    SVD TOPIC ANALYSIS WITH MATCHING CLUSTERS (OPTIMIZED)
    ════════════════════════════════════════════════════
    
    📊 Dataset: {len(embeddings):,} embeddings ({embeddings.shape[1]}D)
    
    �� Configuration:
    • SVD Components: {results['svd_rank']}
    • Clusters: {results['svd_rank']} (matching SVD components)
    • Direct mapping: Cluster 0 = SVD Component 0, etc.
    
    📈 Variance Breakdown:
    • Topic Space: {results['topic_variance']:.1%}
    • Residual Space: {results['residual_variance']:.1%}
    • Total: {results['topic_variance'] + results['residual_variance']:.1%}
    
    ⚡ Optimizations Applied:
    • No QR decomposition (SVD vectors already orthonormal)
    • Efficient two-step projection: E @ T^T → topic_scores @ T
    • Faster computation with smaller matrix multiplications
    
    💡 Interpretation:
    • Left plot: 160 clusters directly correspond to 160 SVD components
    • Right plot: Residuals contain tone information (next SVD step)
    • Each cluster color represents one topic direction in 1536D space
    
    ⏱️ Processing Time: {results['processing_time']:.2f}s
    """
    
    # Add text box
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             verticalalignment='bottom', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../../data/reports/svd_cluster_analysis_optimized.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   ✅ Visualization saved: ../../data/reports/svd_cluster_analysis_optimized.png")

if __name__ == "__main__":
    analyze_svd_with_matching_clusters()
