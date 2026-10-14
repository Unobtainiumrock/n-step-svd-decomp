#!/usr/bin/env python3
"""
Check Pinecone Database Row Count
Query the Pinecone database to see how many rows are available vs our local H5 file
"""

import os
import h5py
from residualized_tone_space.utils.pinecone_client import PineconeEmbeddingDownloader

def check_local_embeddings():
    """Check our local H5 embeddings file."""
    print("📁 LOCAL EMBEDDINGS FILE")
    print("=" * 40)
    
    try:
        with h5py.File('data/embeddings/ugc-posts_ALL_embeddings.h5', 'r') as f:
            embeddings_shape = f['embeddings'].shape
            print(f"  └─ Embeddings shape: {embeddings_shape}")
            print(f"  └─ Total posts: {embeddings_shape[0]:,}")
            print(f"  └─ Embedding dimensions: {embeddings_shape[1]}")
            
            # Check what other data is available
            print(f"  └─ Available keys: {list(f.keys())}")
            
            if 'ids' in f:
                ids_count = len(f['ids'][:])
                print(f"  └─ IDs count: {ids_count:,}")
                
                # Sample a few IDs to see the format
                sample_ids = [id.decode('utf-8') for id in f['ids'][:5]]
                print(f"  └─ Sample IDs: {sample_ids}")
            
            if 'metadata' in f:
                metadata_count = len(f['metadata'][:])
                print(f"  └─ Metadata count: {metadata_count:,}")
            
            # Check file attributes
            if f.attrs:
                print(f"  └─ File attributes:")
                for key, value in f.attrs.items():
                    print(f"      ├─ {key}: {value}")
                    
    except Exception as e:
        print(f"  └─ Error reading local file: {e}")

def check_pinecone_database():
    """Check the Pinecone database directly."""
    print("\n🔗 PINECONE DATABASE")
    print("=" * 40)
    
    # Check if we have Pinecone API key
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("  └─ No PINECONE_API_KEY found in environment")
        print("  └─ Cannot query Pinecone database directly")
        return
    
    try:
        # Initialize Pinecone client
        downloader = PineconeEmbeddingDownloader(api_key=api_key)
        
        # List available indexes
        print("  └─ Available indexes:")
        indexes = downloader.list_indexes()
        
        for idx, index in enumerate(indexes, 1):
            print(f"      ├─ {idx}. {index['name']}")
            print(f"      │   ├─ Dimension: {index['dimension']}")
            print(f"      │   ├─ Metric: {index['metric']}")
            print(f"      │   └─ Status: {index['status']}")
            
            # Get stats for each index
            try:
                stats = downloader.get_index_stats(index['name'])
                total_vectors = stats.get('total_vector_count', 'Unknown')
                print(f"      │   └─ Total vectors: {total_vectors:,}" if isinstance(total_vectors, int) else f"      │   └─ Total vectors: {total_vectors}")
            except Exception as e:
                print(f"      │   └─ Stats error: {e}")
        
    except Exception as e:
        print(f"  └─ Error connecting to Pinecone: {e}")
        print("  └─ Check API key and network connection")

def analyze_discrepancy():
    """Analyze the potential reasons for the discrepancy."""
    print("\n🔍 DISCREPANCY ANALYSIS")
    print("=" * 40)
    
    expected_count = 96280
    actual_count = 58354
    difference = expected_count - actual_count
    percentage_retained = (actual_count / expected_count) * 100
    
    print(f"  └─ Expected (Pinecone): {expected_count:,} posts")
    print(f"  └─ Actual (Local H5): {actual_count:,} posts")
    print(f"  └─ Difference: {difference:,} posts ({100-percentage_retained:.1f}% reduction)")
    print(f"  └─ Retention rate: {percentage_retained:.1f}%")
    
    print(f"\n  🤔 Possible explanations:")
    print(f"      ├─ Data quality filtering (duplicates, empty content)")
    print(f"      ├─ Embedding generation failures")
    print(f"      ├─ API rate limits or timeouts")
    print(f"      ├─ Intentional sampling for analysis")
    print(f"      ├─ Content type filtering (text-only vs multimedia)")
    print(f"      └─ Processing pipeline constraints")

def main():
    """Main function to check both local and remote data sources."""
    
    print("🔍 DATA SOURCE INVESTIGATION")
    print("=" * 60)
    print("Comparing local H5 embeddings file vs Pinecone database")
    print("=" * 60)
    
    # Check local file
    check_local_embeddings()
    
    # Check Pinecone database
    check_pinecone_database()
    
    # Analyze discrepancy
    analyze_discrepancy()
    
    print(f"\n✅ Investigation complete!")

if __name__ == "__main__":
    main() 