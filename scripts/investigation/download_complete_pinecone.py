#!/usr/bin/env python3
"""
Complete Pinecone Download Script - Resource check and full download
"""

import os
import sys
import time
import argparse
import shutil
import numpy as np
import h5py
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pinecone import Pinecone
except ImportError:
    print("❌ Error: pinecone-client not installed")
    print("Install with: pip install pinecone-client")
    sys.exit(1)

def check_resources_and_estimate(total_vectors: int):
    """Check resources and estimate requirements."""
    
    print(f"📊 RESOURCE ESTIMATION")
    print(f"=" * 40)
    
    # Calculate requirements
    storage_gb = (total_vectors * 1536 * 4) / (1024**3) * 1.2  # With compression
    memory_gb = storage_gb * 2  # Processing overhead
    time_minutes = total_vectors / 1000 / 60  # ~1000 vectors/minute
    
    # Check available resources
    disk_usage = shutil.disk_usage(".")
    available_gb = disk_usage.free / (1024**3)
    
    print(f"  ├─ Vectors to download: {total_vectors:,}")
    print(f"  ├─ Storage needed: {storage_gb:.2f} GB")
    print(f"  ├─ Memory needed: {memory_gb:.2f} GB") 
    print(f"  ├─ Available disk: {available_gb:.1f} GB")
    print(f"  └─ Estimated time: {time_minutes:.1f} minutes")
    
    # Check if we have enough resources
    storage_ok = available_gb >= storage_gb
    
    print(f"\n✅ RESOURCE CHECK:")
    print(f"  └─ Storage: {'✅ PASS' if storage_ok else '❌ FAIL'}")
    
    return storage_ok, storage_gb, time_minutes

def get_pinecone_stats(api_key: str, index_name: str):
    """Get Pinecone index statistics."""
    
    print(f"🔗 PINECONE INDEX CHECK")
    print(f"=" * 40)
    
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        total_vectors = stats.get('total_vector_count', 0)
        dimension = stats.get('dimension', 1536)
        
        print(f"  ├─ Index name: {index_name}")
        print(f"  ├─ Total vectors: {total_vectors:,}")
        print(f"  ├─ Dimension: {dimension}")
        print(f"  └─ Status: {'✅ Active' if total_vectors > 0 else '❌ Empty'}")
        
        return total_vectors, dimension, stats
        
    except Exception as e:
        print(f"  └─ Error: {e}")
        return 0, 1536, {}

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description="Download ALL embeddings from Pinecone")
    parser.add_argument("--api-key", help="Pinecone API key (or use PINECONE_API_KEY env)")
    parser.add_argument("--index-name", required=True, help="Pinecone index name")
    parser.add_argument("--dry-run", action="store_true", help="Check only, don't download")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ No Pinecone API key provided")
        print("Set PINECONE_API_KEY env var or use --api-key")
        return
    
    print("🎯 COMPLETE PINECONE DOWNLOAD")
    print("=" * 60)
    
    # Step 1: Check Pinecone
    total_vectors, dimension, stats = get_pinecone_stats(api_key, args.index_name)
    
    if total_vectors == 0:
        print("❌ No vectors found or connection failed")
        return
    
    # Step 2: Check resources
    storage_ok, storage_needed, time_needed = check_resources_and_estimate(total_vectors)
    
    # Step 3: Show comparison with current dataset
    print(f"\n📊 DATASET COMPARISON")
    print(f"=" * 40)
    print(f"  ├─ Current local: 58,354 posts")
    print(f"  ├─ Pinecone total: {total_vectors:,} posts")
    print(f"  ├─ Difference: {total_vectors - 58354:,} additional posts")
    print(f"  └─ Increase: {(total_vectors - 58354) / 58354 * 100:.1f}% more data")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN COMPLETE")
        print(f"  ├─ Ready for download: {'✅ YES' if storage_ok else '❌ NO'}")
        print(f"  ├─ Storage needed: {storage_needed:.2f} GB")
        print(f"  └─ Estimated time: {time_needed:.1f} minutes")
        return
    
    if not storage_ok:
        print(f"\n❌ INSUFFICIENT STORAGE")
        print(f"  └─ Need {storage_needed:.2f} GB, have {available_gb:.1f} GB")
        return
    
    # Proceed with actual download
    print(f"\n⬇️  STARTING COMPLETE DOWNLOAD")
    print(f"=" * 40)
    
    try:
        from residualized_tone_space.utils.pinecone_client import PineconeEmbeddingDownloader
        
        # Initialize downloader
        downloader = PineconeEmbeddingDownloader(api_key)
        
        # Download ALL embeddings (no max_vectors limit)
        print(f"  └─ Downloading all {total_vectors:,} vectors...")
        start_time = time.time()
        
        embeddings, metadata, ids = downloader.download_embeddings_batch(
            index_name=args.index_name,
            batch_size=1000,
            max_vectors=None,  # No limit - download everything
            include_metadata=True
        )
        
        download_time = time.time() - start_time
        
        if len(embeddings) == 0:
            print("❌ Download failed - no embeddings retrieved")
            return
        
        # Save to new file
        output_file = f"data/embeddings/{args.index_name}_COMPLETE_embeddings.h5"
        print(f"\n💾 Saving to: {output_file}")
        
        downloader.save_embeddings_locally(embeddings, metadata, ids, output_file)
        
        # Success summary
        success_rate = len(embeddings) / total_vectors * 100
        print(f"\n🎉 DOWNLOAD COMPLETE!")
        print(f"  ├─ Downloaded: {len(embeddings):,}/{total_vectors:,} vectors ({success_rate:.1f}%)")
        print(f"  ├─ File size: {embeddings.nbytes / (1024**2):.1f} MB")
        print(f"  ├─ Download time: {download_time:.1f} seconds")
        print(f"  └─ Output: {output_file}")
        
        if len(embeddings) < total_vectors:
            missing = total_vectors - len(embeddings)
            print(f"\n⚠️  INCOMPLETE DOWNLOAD")
            print(f"  ├─ Missing: {missing:,} vectors ({missing/total_vectors*100:.1f}%)")
            print(f"  └─ This may be due to API limits or network issues")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    main()
