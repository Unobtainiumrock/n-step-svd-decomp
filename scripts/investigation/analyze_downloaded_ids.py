#!/usr/bin/env python3
"""
Analyze Downloaded IDs - Check what types of content we have
"""

import h5py
import json
from collections import Counter
import numpy as np

def analyze_downloaded_content():
    """Analyze what content we downloaded from the latest file."""
    
    print("🔍 DOWNLOADED CONTENT ANALYSIS")
    print("=" * 50)
    
    # Check if we have a new complete file
    files_to_check = [
        'data/embeddings/ugc-posts_COMPLETE_embeddings.h5',
        'data/embeddings/ugc-posts_ALL_embeddings.h5'
    ]
    
    for file_path in files_to_check:
        try:
            print(f"\n📁 Analyzing: {file_path}")
            
            with h5py.File(file_path, 'r') as f:
                embeddings_shape = f['embeddings'].shape
                ids = [id.decode('utf-8') for id in f['ids'][:]]
                
                print(f"  ├─ Total vectors: {embeddings_shape[0]:,}")
                print(f"  ├─ Embedding dimensions: {embeddings_shape[1]}")
                print(f"  └─ IDs count: {len(ids):,}")
                
                # Analyze ID patterns
                analyze_id_patterns(ids, file_path)
                
                # Check metadata if available
                if 'metadata' in f:
                    analyze_metadata_sample(f, file_path)
                    
        except FileNotFoundError:
            print(f"  └─ File not found: {file_path}")
        except Exception as e:
            print(f"  └─ Error reading {file_path}: {e}")

def analyze_id_patterns(ids, file_path):
    """Analyze the patterns in downloaded IDs."""
    
    print(f"\n📊 ID PATTERN ANALYSIS")
    print("=" * 30)
    
    # Separate video vs non-video IDs
    video_ids = [id for id in ids if id.endswith('_video')]
    pure_ids = [id for id in ids if not id.endswith('_video')]
    
    print(f"  ├─ Pure IDs (no suffix): {len(pure_ids):,}")
    print(f"  ├─ Video IDs (_video suffix): {len(video_ids):,}")
    print(f"  └─ Total: {len(ids):,}")
    
    # Check ID number ranges for pure IDs
    if pure_ids:
        try:
            pure_numbers = [int(id) for id in pure_ids if id.isdigit()]
            if pure_numbers:
                print(f"\n📈 PURE ID STATISTICS:")
                print(f"  ├─ Numeric IDs: {len(pure_numbers):,}")
                print(f"  ├─ Min ID: {min(pure_numbers):,}")
                print(f"  ├─ Max ID: {max(pure_numbers):,}")
                print(f"  └─ Range span: {max(pure_numbers) - min(pure_numbers):,}")
                
        except Exception as e:
            print(f"  └─ Error analyzing pure ID numbers: {e}")
    
    # Sample IDs for inspection
    print(f"\n📋 SAMPLE IDs:")
    sample_pure = pure_ids[:5] if pure_ids else []
    sample_video = video_ids[:5] if video_ids else []
    
    if sample_pure:
        print(f"  ├─ Pure IDs: {sample_pure}")
    if sample_video:
        print(f"  └─ Video IDs: {sample_video}")

def analyze_metadata_sample(f, file_path):
    """Analyze a sample of metadata to understand content types."""
    
    print(f"\n📝 METADATA SAMPLE ANALYSIS")
    print("=" * 30)
    
    try:
        # Sample first 10 metadata entries
        sample_size = min(10, len(f['metadata']))
        content_types = []
        post_types = []
        
        for i in range(sample_size):
            metadata_json = f['metadata'][i].decode('utf-8')
            metadata = json.loads(metadata_json)
            
            if 'embedded_content' in metadata:
                content_types.append(metadata['embedded_content'])
            if 'post_type' in metadata:
                post_types.append(metadata['post_type'])
        
        # Count content types
        if content_types:
            content_counts = Counter(content_types)
            print(f"  ├─ Content types found:")
            for content_type, count in content_counts.items():
                print(f"  │   └─ {content_type}: {count}")
        
        if post_types:
            post_counts = Counter(post_types)
            print(f"  ├─ Post types found:")
            for post_type, count in post_counts.items():
                print(f"  │   └─ {post_type}: {count}")
        
        # Show sample metadata structure
        if sample_size > 0:
            metadata_json = f['metadata'][0].decode('utf-8')
            sample_metadata = json.loads(metadata_json)
            print(f"  └─ Sample metadata keys: {list(sample_metadata.keys())}")
            
    except Exception as e:
        print(f"  └─ Error analyzing metadata: {e}")

def compare_with_supabase_expectations():
    """Compare with expected Supabase counts."""
    
    print(f"\n📊 SUPABASE COMPARISON")
    print("=" * 30)
    
    expected_pure_ids = 84200  # Your Supabase UGC posts
    expected_video_ids = 15578  # Your Supabase video transcripts
    expected_total = expected_pure_ids + expected_video_ids
    
    print(f"  ├─ Expected pure IDs: {expected_pure_ids:,}")
    print(f"  ├─ Expected video IDs: {expected_video_ids:,}")
    print(f"  └─ Expected total: {expected_total:,}")

def main():
    """Main analysis function."""
    
    # Analyze downloaded content
    analyze_downloaded_content()
    
    # Compare with expectations
    compare_with_supabase_expectations()
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
