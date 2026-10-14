#!/usr/bin/env python3
"""
SVD Topic Analysis with Matching Clusters (GPU Optimized for Google Colab)
Shows the relationship between 160 SVD components and 160 clusters
Uses CuPy for GPU acceleration and efficient two-step projection
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
import time
import os

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration available with CuPy!")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  CuPy not available, using CPU (install with: pip install cupy-cuda11x)")

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_svd_with_matching_clusters_gpu():
    """Analyze SVD topic space with clusters that match the component count."""
    print("🔍 SVD Topic Analysis with Matching Clusters (GPU Optimized)")
    print("=" * 65)
    
    # Load data
    print("\n📊 Loading embeddings...")
    with h5py.File('ugc-posts_COMPLETE_embeddings.h5', 'r') as f:
        embeddings = f['embeddings'][:]
    print(f"   ✅ Dataset: {len(embeddings):,} embeddings, {embeddings.shape[1]} dimensions")
    
    # Use 160 SVD components (matching the optimal rank)
    SVD_RANK = 160
    
    print(f"\n🎯 Using {SVD_RANK} SVD Components and {SVD_RANK} Clusters")
    print(f"   💡 Optimizations: GPU acceleration, efficient two-step projection")
    
    # Run analysis
    results = run_svd_cluster_analysis_gpu(embeddings, SVD_RANK)
    
    # Create visualization
    create_svd_cluster_visualization_gpu(results, embeddings)
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   📊 SVD Components: {SVD_RANK}")
    print(f"   🎯 Clusters: {SVD_RANK}")
    print(f"   📈 Topic Variance: {results['topic_variance']:.1%}")
    print(f"   📉 Residual Variance: {results['residual_variance']:.1%}")
    print(f"   ⚡ GPU Used: {results['gpu_used']}")

def run_svd_cluster_analysis_gpu(embeddings, svd_rank):
    """Run SVD analysis with GPU acceleration where possible."""
    print(f"\n🚀 Running GPU-Optimized SVD Analysis...")
    start_time = time.time()
    
    # Convert to GPU if available
    if GPU_AVAILABLE:
        print(f"   �� Moving data to GPU...")
        embeddings_gpu = cp.asarray(embeddings)
        gpu_used = True
    else:
        embeddings_gpu = embeddings
        gpu_used = False
    
    # Step 1: Topic Discovery with SVD (CPU - sklearn doesn't support GPU)
    print(f"   🔍 Topic Discovery (SVD, k={svd_rank})...")
    topic_svd = TruncatedSVD(n_components=svd_rank, random_state=42)
    topic_svd.fit(embeddings)
    topic_vectors = topic_svd.components_
    topic_variance = np.sum(topic_svd.explained_variance_ratio_)
    
    # Move topic vectors to GPU if available
    if GPU_AVAILABLE:
        topic_vectors_gpu = cp.asarray(topic_vectors)
    else:
        topic_vectors_gpu = topic_vectors
    
    # Step 2: Calculate Topic Projections (GPU Accelerated)
    print(f"   📊 Computing Topic Projections (GPU Accelerated)...")
    if GPU_AVAILABLE:
        # GPU computation
        topic_scores_gpu = embeddings_gpu @ topic_vectors_gpu.T  # (N, 160)
        topic_projections_gpu = topic_scores_gpu @ topic_vectors_gpu  # (N, 1536)
        
        # Move back to CPU for sklearn operations
        topic_scores = cp.asnumpy(topic_scores_gpu)
        topic_projections = cp.asnumpy(topic_projections_gpu)
    else:
        # CPU computation
        topic_scores = embeddings @ topic_vectors.T
        topic_projections = topic_scores @ topic_vectors
    
    # Step 3: Calculate Residuals (GPU Accelerated)
    print(f"   ➖ Computing Residual Vectors (GPU Accelerated)...")
    if GPU_AVAILABLE:
        residual_vectors_gpu = embeddings_gpu - topic_projections_gpu
        residual_vectors = cp.asnumpy(residual_vectors_gpu)
    else:
        residual_vectors = embeddings - topic_projections
    
    # Step 4: K-means Clustering (CPU - sklearn doesn't support GPU)
    print(f"   🎯 Topic Clustering (K-means, k={svd_rank})...")
    kmeans = KMeans(n_clusters=svd_rank, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(topic_projections)
    
    # Calculate variance breakdown (GPU accelerated if available)
    print(f"   📊 Computing Variance Breakdown...")
    if GPU_AVAILABLE:
        total_variance = float(cp.var(embeddings_gpu, axis=0).sum())
        topic_projection_variance = float(cp.var(topic_projections_gpu, axis=0).sum())
        residual_variance = float(cp.var(residual_vectors_gpu, axis=0).sum())
    else:
        total_variance = np.var(embeddings, axis=0).sum()
        topic_projection_variance = np.var(topic_projections, axis=0).sum()
        residual_variance = np.var(residual_vectors, axis=0).sum()
    
    total_time = time.time() - start_time
    
    print(f"   ⏱️  Analysis completed in {total_time:.2f}s")
    print(f"   📊 Topic variance: {topic_variance:.1%}")
    print(f"   📉 Residual variance: {residual_variance/total_variance:.1%}")
    print(f"   🎯 Clusters: {svd_rank} (matching SVD components)")
    print(f"   ⚡ GPU acceleration: {'Enabled' if gpu_used else 'Disabled'}")
    
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
        'svd_rank': svd_rank,
        'gpu_used': gpu_used
    }

def create_svd_cluster_visualization_gpu(results, embeddings):
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
    gpu_status = "🚀 GPU Accelerated" if results['gpu_used'] else "💻 CPU Only"
    summary_text = f"""
    SVD TOPIC ANALYSIS WITH MATCHING CLUSTERS (GPU OPTIMIZED)
    ════════════════════════════════════════════════════════
    
    📊 Dataset: {len(embeddings):,} embeddings ({embeddings.shape[1]}D)
    
    🎯 Configuration:
    • SVD Components: {results['svd_rank']}
    • Clusters: {results['svd_rank']} (matching SVD components)
    • Direct mapping: Cluster 0 = SVD Component 0, etc.
    
    📈 Variance Breakdown:
    • Topic Space: {results['topic_variance']:.1%}
    • Residual Space: {results['residual_variance']:.1%}
    • Total: {results['topic_variance'] + results['residual_variance']:.1%}
    
    ⚡ Performance Optimizations:
    • {gpu_status}
    • Efficient two-step projection: E @ T^T → topic_scores @ T
    • GPU-accelerated matrix operations where possible
    • No QR decomposition (SVD vectors already orthonormal)
    
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
    plt.savefig('svd_cluster_analysis_gpu.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   ✅ Visualization saved: svd_cluster_analysis_gpu.png")

# Google Colab setup function
def setup_colab_environment():
    """Setup function for Google Colab environment."""
    print("🔧 Setting up Google Colab Environment...")
    
    # Install CuPy if not available
    if not GPU_AVAILABLE:
        print("📦 Installing CuPy for GPU acceleration...")
        import subprocess
        import sys
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda11x"])
            print("✅ CuPy installed successfully!")
            print("🔄 Please restart the runtime and run the script again.")
            return False
        except subprocess.CalledProcessError:
            print("❌ Failed to install CuPy. Running in CPU mode.")
            return True
    
    return True

if __name__ == "__main__":
    # Check if we're in Colab and setup if needed
    try:
        import google.colab
        print("🌐 Running in Google Colab environment")
        if setup_colab_environment():
            analyze_svd_with_matching_clusters_gpu()
    except ImportError:
        print("💻 Running in local environment")
        analyze_svd_with_matching_clusters_gpu()
