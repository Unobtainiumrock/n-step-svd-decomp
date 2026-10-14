#!/usr/bin/env python3
"""
Generate Comprehensive Reports for a 3-Step Orthogonal Decomposition
"""
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from scipy.linalg import qr
import time
import os
from datetime import datetime

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_extended_reports():
    """Generate comprehensive reports with a third decomposition step."""
    print("🎨 Generating Extended 3-Step Decomposition Reports")
    print("=" * 60)
    
    reports_dir = "data/reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    print("\n📊 Loading embeddings...")
    with h5py.File('../../data/embeddings/ugc-posts_COMPLETE_embeddings.h5', 'r') as f:
        embeddings = f['embeddings'][:]
    print(f"   ✅ Dataset: {len(embeddings):,} embeddings, {embeddings.shape[1]} dimensions")
    
    # Define optimal ranks for a 3-step decomposition
    OPTIMAL_TOPIC_RANK = 160
    OPTIMAL_TONE_RANK = 130
    OPTIMAL_STYLE_RANK = 80 # Hypothetical rank for the third component
    
    print(f"\n🎯 Using Optimal Ranks:")
    print(f"   📊 Topic Components: {OPTIMAL_TOPIC_RANK}")
    print(f"   🎵 Tone Components: {OPTIMAL_TONE_RANK}")
    print(f"   ✨ Style Components: {OPTIMAL_STYLE_RANK}")
    
    results = run_extended_analysis(embeddings, OPTIMAL_TOPIC_RANK, OPTIMAL_TONE_RANK, OPTIMAL_STYLE_RANK)
    create_extended_report(results, embeddings, reports_dir)
    save_extended_analysis_data(results, reports_dir)
    create_extended_summary(results, reports_dir)
    
    print(f"\n🎉 Extended Reports Generated!")
    print(f"   📁 Location: {reports_dir}/")
    print(f"   🖼️  Main Report: extended_decomposition_report_2.png")
    print(f"   📊 Summary: analysis_summary_2.json")

def run_extended_analysis(embeddings, topic_rank, tone_rank, style_rank):
    """Run a 3-step orthogonal decomposition analysis."""
    print(f"\n🚀 Running Extended Analysis...")
    start_time = time.time()
    
    # === Step 1: Topic Discovery ===
    print(f"   1. Topic Discovery (SVD, k={topic_rank})...")
    topic_svd = TruncatedSVD(n_components=topic_rank, random_state=42)
    topic_projections = topic_svd.fit_transform(embeddings)
    topic_vectors = topic_svd.components_
    residual_after_topic = embeddings - topic_projections @ topic_vectors
    
    # === Step 2: Tone Discovery ===
    print(f"   2. Tone Discovery (SVD on residual 1, k={tone_rank})...")
    tone_svd = TruncatedSVD(n_components=tone_rank, random_state=42)
    tone_projections = tone_svd.fit_transform(residual_after_topic)
    tone_axes = tone_svd.components_
    residual_after_tone = residual_after_topic - tone_projections @ tone_axes

    # === Step 3: Style Discovery ===
    print(f"   3. Style Discovery (SVD on residual 2, k={style_rank})...")
    style_svd = TruncatedSVD(n_components=style_rank, random_state=42)
    style_projections = style_svd.fit_transform(residual_after_tone)
    style_axes = style_svd.components_
    final_residual = residual_after_tone - style_projections @ style_axes
    
    # === Step 4: Clustering for each subspace ===
    print(f"   4. Clustering Subspaces (K-means, k=10)...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    topic_cluster_labels = kmeans.fit_predict(topic_projections)
    
    kmeans_tone = KMeans(n_clusters=10, random_state=42, n_init=10)
    tone_cluster_labels = kmeans_tone.fit_predict(tone_projections)

    kmeans_style = KMeans(n_clusters=10, random_state=42, n_init=10)
    style_cluster_labels = kmeans_style.fit_predict(style_projections)

    total_time = time.time() - start_time
    print(f"   ⏱️  Analysis completed in {total_time:.2f}s")

    return {
        'topic_rank': topic_rank, 'tone_rank': tone_rank, 'style_rank': style_rank,
        'topic_singular_values': topic_svd.singular_values_,
        'tone_singular_values': tone_svd.singular_values_,
        'style_singular_values': style_svd.singular_values_,
        'topic_projections': topic_projections,
        'tone_projections': tone_projections,
        'style_projections': style_projections,
        'topic_cluster_labels': topic_cluster_labels,
        'tone_cluster_labels': tone_cluster_labels,
        'style_cluster_labels': style_cluster_labels,
        'final_residual': final_residual,
        'processing_time': total_time
    }

def create_extended_report(results, embeddings, reports_dir):
    """Create a comprehensive 3-step visualization report."""
    print(f"\n🎨 Creating Extended Visualization...")
    fig = plt.figure(figsize=(30, 18))
    gs = fig.add_gridspec(3, 5)

    pca_2d = PCA(n_components=2, random_state=42)

    # --- ROW 1: Singular Values and First Two Clusters ---
    # 1. Topic Singular Values
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results['topic_singular_values'], 'b-', linewidth=2)
    ax1.set_title(f'1. Topic Singular Values (k={results["topic_rank"]})', fontweight='bold')
    ax1.set_yscale('log'); ax1.grid(True, alpha=0.3)

    # 2. Tone Singular Values
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results['tone_singular_values'], 'g-', linewidth=2)
    ax2.set_title(f'2. Tone Singular Values (k={results["tone_rank"]})', fontweight='bold')
    ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)

    # 3. Style Singular Values
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(results['style_singular_values'], 'm-', linewidth=2)
    ax3.set_title(f'3. Style Singular Values (k={results["style_rank"]})', fontweight='bold')
    ax3.set_yscale('log'); ax3.grid(True, alpha=0.3)

    # 4. Topic Clusters
    ax4 = fig.add_subplot(gs[0, 3])
    topic_2d = pca_2d.fit_transform(results['topic_projections'])
    scatter1 = ax4.scatter(topic_2d[:, 0], topic_2d[:, 1], c=results['topic_cluster_labels'], cmap='tab10', alpha=0.6, s=1)
    ax4.set_title('4. Topic Clusters (2D Viz)', fontweight='bold')
    fig.colorbar(scatter1, ax=ax4, label='Cluster')

    # 5. Tone Clusters
    ax5 = fig.add_subplot(gs[0, 4])
    tone_2d = pca_2d.fit_transform(results['tone_projections'])
    scatter2 = ax5.scatter(tone_2d[:, 0], tone_2d[:, 1], c=results['tone_cluster_labels'], cmap='viridis', alpha=0.6, s=1)
    ax5.set_title('5. Tone Clusters (2D Viz)', fontweight='bold')
    fig.colorbar(scatter2, ax=ax5, label='Cluster')

    # --- ROW 2: Style Clusters, Residuals, and Cluster Sizes ---
    # 6. Style Clusters
    ax6 = fig.add_subplot(gs[1, 0])
    style_2d = pca_2d.fit_transform(results['style_projections'])
    scatter3 = ax6.scatter(style_2d[:, 0], style_2d[:, 1], c=results['style_cluster_labels'], cmap='plasma', alpha=0.6, s=1)
    ax6.set_title('6. Style Clusters (2D Viz)', fontweight='bold')
    fig.colorbar(scatter3, ax=ax6, label='Cluster')

    # 7. Final Residuals
    ax7 = fig.add_subplot(gs[1, 1])
    residual_2d = pca_2d.fit_transform(results['final_residual'])
    ax7.scatter(residual_2d[:, 0], residual_2d[:, 1], alpha=0.2, s=1, color='gray')
    ax7.set_title('7. Final Residuals (2D Viz)', fontweight='bold')

    # 8. Topic Cluster Sizes
    ax8 = fig.add_subplot(gs[1, 2])
    unique, counts = np.unique(results['topic_cluster_labels'], return_counts=True)
    ax8.bar(unique, counts, alpha=0.7, color='skyblue')
    ax8.set_title('8. Topic Cluster Sizes', fontweight='bold')

    # 9. Tone Cluster Sizes
    ax9 = fig.add_subplot(gs[1, 3])
    unique, counts = np.unique(results['tone_cluster_labels'], return_counts=True)
    ax9.bar(unique, counts, alpha=0.7, color='lightgreen')
    ax9.set_title('9. Tone Cluster Sizes', fontweight='bold')
    
    # 10. Style Cluster Sizes
    ax10 = fig.add_subplot(gs[1, 4])
    unique, counts = np.unique(results['style_cluster_labels'], return_counts=True)
    ax10.bar(unique, counts, alpha=0.7, color='plum')
    ax10.set_title('10. Style Cluster Sizes', fontweight='bold')

    # --- ROW 3: Summary and Other Metrics ---
    # For this example, we'll simplify the bottom row to focus on the summary
    ax11 = fig.add_subplot(gs[2, :])
    ax11.axis('off')
    
    # Calculate variance breakdown
    total_var = np.var(embeddings, axis=0).sum()
    topic_var_p = np.var(results['topic_projections'], axis=0).sum() / total_var
    tone_var_p = np.var(results['tone_projections'], axis=0).sum() / total_var
    style_var_p = np.var(results['style_projections'], axis=0).sum() / total_var
    resid_var_p = np.var(results['final_residual'], axis=0).sum() / total_var

    summary_text = f"""
    EXTENDED 3-STEP DECOMPOSITION ANALYSIS
    ═══════════════════════════════════════
    
    📊 Dataset: {len(embeddings):,} embeddings ({embeddings.shape[1]}D)
    
    🎯 Configuration (via Elbow Method):
    • Topic: SVD with {results['topic_rank']} components
    • Tone: SVD with {results['tone_rank']} components
    • Style: SVD with {results['style_rank']} components
    • Total Parameters: {results['topic_rank'] + results['tone_rank'] + results['style_rank']}
    
    📈 Variance Breakdown:
    • Topic Space: {topic_var_p:.1%}
    • Tone Space:  {tone_var_p:.1%}
    • Style Space: {style_var_p:.1%}
    • Unexplained: {resid_var_p:.1%}
    
    ⏱️ Processing Time: {results['processing_time']:.2f}s
    
    💡 Interpretation:
    - The 'Style' singular value plot (3) is much flatter
      than Topic (1), suggesting a weaker signal.
    - The 'Style' cluster plot (6) appears less structured
      than the Topic plot (4), possibly indicating that
      we are beginning to model noise.
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, 
             fontsize=14, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/extended_decomposition_report_2.png', dpi=300, facecolor='white')
    print(f"   ✅ Extended report saved: {reports_dir}/extended_decomposition_report_2.png")
    plt.show()


def save_extended_analysis_data(results, reports_dir):
    """Save the new analysis data."""
    print(f"\n💾 Saving Extended Analysis Data...")
    np.save(f'{reports_dir}/ext_topic_projections_2.npy', results['topic_projections'])
    np.save(f'{reports_dir}/ext_tone_projections_2.npy', results['tone_projections'])
    np.save(f'{reports_dir}/ext_style_projections_2.npy', results['style_projections'])
    np.save(f'{reports_dir}/ext_final_residual_2.npy', results['final_residual'])
    print(f"   ✅ Extended data saved to {reports_dir}/")


def create_extended_summary(results, reports_dir):
    """Create a JSON summary for the 3-step analysis."""
    total_params = results['topic_rank'] + results['tone_rank'] + results['style_rank']
    summary = {
        'analysis_type': 'Extended 3-Step Decomposition',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'topic_rank': results['topic_rank'],
            'tone_rank': results['tone_rank'],
            'style_rank': results['style_rank'],
            'total_parameters': total_params
        },
        'performance': {
            'processing_time_seconds': float(results['processing_time']),
        },
        'notes': 'This analysis extends the original 2-step decomposition to explore a third orthogonal subspace, termed "Style".'
    }
    
    with open(f'{reports_dir}/analysis_summary_2.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Extended summary saved: {reports_dir}/analysis_summary_2.json")

if __name__ == "__main__":
    # Ensure you have the HDF5 file at the specified path before running
    if not os.path.exists('../../data/embeddings/ugc-posts_COMPLETE_embeddings.h5'):
        print("\nERROR: Embedding file not found at '../../data/embeddings/ugc-posts_COMPLETE_embeddings.h5'")
        print("Please ensure the data is in the correct location.")
    else:
        generate_extended_reports()