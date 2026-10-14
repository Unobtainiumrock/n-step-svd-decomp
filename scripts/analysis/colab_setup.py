"""
Google Colab Setup for SVD Cluster Analysis
Run this first in your Colab notebook
"""

# Install required packages
!pip install cupy-cuda11x
!pip install h5py
!pip install seaborn

# Check GPU availability
import torch
print(f"🚀 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Upload your files here:
print("\n📁 Please upload the following files:")
print("1. ugc-posts_COMPLETE_embeddings.h5")
print("2. svd_cluster_analysis_gpu.py")
print("\nThen run the analysis script!")
