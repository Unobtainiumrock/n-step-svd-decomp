"""
Pinecone integration for downloading and managing embeddings.
"""

import pinecone
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = structlog.get_logger(__name__)


class PineconeEmbeddingDownloader:
    """
    Download embeddings from Pinecone for local processing.
    """
    
    def __init__(self, api_key: str, environment: str = None):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (optional for newer versions)
        """
        self.api_key = api_key
        self.environment = environment
        
        # Initialize Pinecone client
        try:
            self.pc = pinecone.Pinecone(api_key=api_key)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Pinecone client", error=str(e))
            raise
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all available indexes.
        
        Returns:
            List of index information
        """
        try:
            indexes = self.pc.list_indexes()
            index_list = []
            
            for index in indexes:
                index_info = {
                    'name': index.name,
                    'dimension': index.dimension,
                    'metric': index.metric,
                    'host': index.host,
                    'status': index.status
                }
                index_list.append(index_info)
            
            logger.info("Retrieved index list", count=len(index_list))
            return index_list
            
        except Exception as e:
            logger.error("Failed to list indexes", error=str(e))
            raise
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Index statistics
        """
        try:
            index = self.pc.Index(index_name)
            stats = index.describe_index_stats()
            
            logger.info("Retrieved index stats", index=index_name, stats=stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to get index stats", index=index_name, error=str(e))
            raise
    
    def download_embeddings_batch(
        self, 
        index_name: str, 
        batch_size: int = 1000,
        max_vectors: Optional[int] = None,
        include_metadata: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
        """
        Download embeddings from Pinecone in batches.
        
        Args:
            index_name: Name of the Pinecone index
            batch_size: Number of vectors to fetch per batch
            max_vectors: Maximum number of vectors to download (None for all)
            include_metadata: Whether to include metadata
            
        Returns:
            Tuple of (embeddings_array, metadata_list, ids_list)
        """
        logger.info(
            "Starting embedding download",
            index=index_name,
            batch_size=batch_size,
            max_vectors=max_vectors
        )
        
        try:
            index = self.pc.Index(index_name)
            
            # Get index stats to determine total vectors
            stats = index.describe_index_stats()
            total_vectors = stats['total_vector_count']
            
            if max_vectors:
                total_vectors = min(total_vectors, max_vectors)
            
            logger.info("Index contains vectors", total=total_vectors)
            
            # Download vectors in batches
            all_embeddings = []
            all_metadata = []
            all_ids = []
            
            # Use query with empty vector to get all vectors
            # This is a workaround since Pinecone doesn't have a direct "get all" method
            downloaded_count = 0
            
            # REAL IMPLEMENTATION: Download all vectors using multiple strategies
            logger.info("Starting complete download", target_vectors=total_vectors)
            
            # Strategy 1: Try to list all vector IDs first (most efficient)
            try:
                logger.info("Attempting to list all vector IDs...")
                
                # Try using list_vectors if available (newer Pinecone)
                if hasattr(index, 'list_vectors'):
                    all_vector_ids = []
                    for ids_batch in index.list_vectors():
                        all_vector_ids.extend(ids_batch)
                    
                    logger.info("Found vector IDs via list_vectors", count=len(all_vector_ids))
                    
                    # Download by IDs using fetch
                    return self._fetch_vectors_by_ids(index, all_vector_ids, include_metadata)
                else:
                    logger.warning("list_vectors not available, trying alternative method")
                    
            except Exception as e:
                logger.warning("Vector ID listing failed", error=str(e))
            
            # Strategy 2: Use multiple random queries to get comprehensive coverage
            logger.info("Using multiple query strategy for complete download")
            return self._download_via_multiple_queries(index, total_vectors, include_metadata)
            
        except Exception as e:
            logger.error("Failed to download embeddings", error=str(e))
            raise
    
    def download_embeddings_by_ids(
        self, 
        index_name: str, 
        vector_ids: List[str],
        include_metadata: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
        """
        Download specific embeddings by their IDs.
        
        Args:
            index_name: Name of the Pinecone index
            vector_ids: List of vector IDs to download
            include_metadata: Whether to include metadata
            
        Returns:
            Tuple of (embeddings_array, metadata_list, ids_list)
        """
        logger.info(
            "Downloading embeddings by IDs",
            index=index_name,
            count=len(vector_ids)
        )
        
        try:
            index = self.pc.Index(index_name)
            
            all_embeddings = []
            all_metadata = []
            all_ids = []
            
            # Fetch vectors by ID
            fetch_response = index.fetch(ids=vector_ids)
            
            for vector_id, vector_data in fetch_response['vectors'].items():
                if 'values' in vector_data:
                    all_embeddings.append(vector_data['values'])
                    all_ids.append(vector_id)
                    
                    if include_metadata and 'metadata' in vector_data:
                        all_metadata.append(vector_data['metadata'])
                    else:
                        all_metadata.append({})
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            logger.info(
                "Successfully downloaded embeddings",
                count=len(embeddings_array),
                shape=embeddings_array.shape
            )
            
            return embeddings_array, all_metadata, all_ids
            
        except Exception as e:
            logger.error("Failed to download embeddings by IDs", error=str(e))
            raise
    
    def download_embeddings_sample(
        self, 
        index_name: str, 
        sample_size: int = 10000,
        include_metadata: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
        """
        Download a random sample of embeddings for analysis.
        
        Args:
            index_name: Name of the Pinecone index
            sample_size: Number of vectors to sample
            include_metadata: Whether to include metadata
            
        Returns:
            Tuple of (embeddings_array, metadata_list, ids_list)
        """
        logger.info(
            "Downloading embedding sample",
            index=index_name,
            sample_size=sample_size
        )
        
        try:
            index = self.pc.Index(index_name)
            
            # Get index stats
            stats = index.describe_index_stats()
            total_vectors = stats['total_vector_count']
            
            if sample_size >= total_vectors:
                logger.warning(
                    "Sample size larger than total vectors, downloading all",
                    sample_size=sample_size,
                    total_vectors=total_vectors
                )
                return self.download_embeddings_batch(
                    index_name, 
                    batch_size=1000,
                    max_vectors=total_vectors,
                    include_metadata=include_metadata
                )
            
            # REAL IMPLEMENTATION: Use multiple random queries for sampling
            logger.info("Starting random sampling", sample_size=sample_size, total_vectors=total_vectors)
            
            all_embeddings = []
            all_metadata = []
            all_ids = []
            seen_ids = set()
            
            dimension = index.describe_index_stats()['dimension']
            iterations_needed = max(10, sample_size // 1000)  # Estimate iterations needed
            
            for iteration in range(iterations_needed):
                if len(all_embeddings) >= sample_size:
                    break
                    
                # Generate random query vector
                query_vector = np.random.normal(0, 1, dimension).tolist()
                
                try:
                    # Query for similar vectors
                    remaining_needed = sample_size - len(all_embeddings)
                    top_k = min(10000, remaining_needed * 2)  # Get more than needed for diversity
                    
                    result = index.query(
                        vector=query_vector,
                        top_k=top_k,
                        include_metadata=include_metadata,
                        include_values=True
                    )
                    
                    new_vectors = 0
                    for match in result['matches']:
                        if len(all_embeddings) >= sample_size:
                            break
                            
                        if match['id'] not in seen_ids:
                            seen_ids.add(match['id'])
                            all_embeddings.append(match['values'])
                            all_ids.append(match['id'])
                            
                            if include_metadata and 'metadata' in match:
                                all_metadata.append(match['metadata'])
                            else:
                                all_metadata.append({})
                            new_vectors += 1
                    
                    logger.info("Sampling progress", 
                              iteration=iteration+1, 
                              collected=len(all_embeddings), 
                              target=sample_size,
                              new_this_round=new_vectors)
                    
                except Exception as e:
                    logger.warning("Sampling iteration failed", iteration=iteration+1, error=str(e))
                    continue
            
            # Trim to exact sample size if we got more
            if len(all_embeddings) > sample_size:
                all_embeddings = all_embeddings[:sample_size]
                all_metadata = all_metadata[:sample_size]
                all_ids = all_ids[:sample_size]
            
            logger.info("Sampling complete", final_count=len(all_embeddings), target=sample_size)
            return np.array(all_embeddings), all_metadata, all_ids
            
        except Exception as e:
            logger.error("Failed to download embedding sample", error=str(e))
            raise
    
    def save_embeddings_locally(
        self, 
        embeddings: np.ndarray, 
        metadata: List[Dict[str, Any]], 
        ids: List[str],
        filepath: str
    ) -> None:
        """
        Save downloaded embeddings to local HDF5 file.
        
        Args:
            embeddings: Embeddings array
            metadata: Metadata list
            ids: Vector IDs
            filepath: Path to save file
        """
        logger.info("Saving embeddings locally", filepath=filepath, count=len(embeddings))
        
        try:
            import h5py
            
            with h5py.File(filepath, 'w') as f:
                # Save embeddings
                f.create_dataset('embeddings', data=embeddings, compression='gzip')
                
                # Save IDs
                f.create_dataset('ids', data=[id.encode('utf-8') for id in ids])
                
                # Save metadata (simplified - in practice, you might need more complex handling)
                if metadata:
                    # Convert metadata to JSON strings
                    import json
                    metadata_json = [json.dumps(md) for md in metadata]
                    f.create_dataset('metadata', data=[md.encode('utf-8') for md in metadata_json])
                
                # Save attributes
                f.attrs['total_vectors'] = len(embeddings)
                f.attrs['embedding_dim'] = embeddings.shape[1]
                f.attrs['download_timestamp'] = time.time()
            
            logger.info("Embeddings saved successfully", filepath=filepath)
            
        except Exception as e:
            logger.error("Failed to save embeddings", error=str(e))
            raise
    
    def load_embeddings_locally(self, filepath: str) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
        """
        Load embeddings from local HDF5 file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Tuple of (embeddings_array, metadata_list, ids_list)
        """
        logger.info("Loading embeddings from local file", filepath=filepath)
        
        try:
            import h5py
            import json
            
            with h5py.File(filepath, 'r') as f:
                # Load embeddings
                embeddings = f['embeddings'][:]
                
                # Load IDs
                ids = [id.decode('utf-8') for id in f['ids'][:]]
                
                # Load metadata
                metadata = []
                if 'metadata' in f:
                    metadata_json = [md.decode('utf-8') for md in f['metadata'][:]]
                    metadata = [json.loads(md) for md in metadata_json]
                
                # Load attributes
                attrs = dict(f.attrs)
            
            logger.info(
                "Embeddings loaded successfully",
                count=len(embeddings),
                shape=embeddings.shape,
                attrs=attrs
            )
            
            return embeddings, metadata, ids
            
        except Exception as e:
            logger.error("Failed to load embeddings", error=str(e))
            raise
    
    def _fetch_vectors_by_ids(self, index, vector_ids: List[str], include_metadata: bool):
        """Fetch vectors by their IDs in batches."""
        
        all_embeddings = []
        all_metadata = []
        all_ids = []
        
        batch_size = 100  # Pinecone fetch limit
        total_batches = (len(vector_ids) + batch_size - 1) // batch_size
        
        logger.info("Starting fetch by IDs", total_ids=len(vector_ids), batches=total_batches)
        
        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                result = index.fetch(ids=batch_ids, include_metadata=include_metadata)
                
                for vector_id, vector_data in result['vectors'].items():
                    all_embeddings.append(vector_data['values'])
                    all_ids.append(vector_id)
                    
                    if include_metadata and 'metadata' in vector_data:
                        all_metadata.append(vector_data['metadata'])
                    else:
                        all_metadata.append({})
                
                # Progress logging every 100 batches
                if batch_num % 100 == 0 or batch_num == total_batches:
                    logger.info("Fetch progress", batch=batch_num, total_batches=total_batches, vectors=len(all_embeddings))
                
            except Exception as e:
                logger.warning("Batch fetch failed", batch=batch_num, error=str(e))
                continue
        
        logger.info("Fetch by IDs complete", downloaded=len(all_embeddings), requested=len(vector_ids))
        return np.array(all_embeddings), all_metadata, all_ids
    
    def _download_via_multiple_queries(self, index, total_vectors: int, include_metadata: bool):
        """Download using multiple random queries to get comprehensive coverage."""
        
        all_embeddings = []
        all_metadata = []
        all_ids = []
        seen_ids = set()
        
        dimension = index.describe_index_stats()['dimension']
        max_iterations = 50  # Reduced iterations for efficiency
        batch_size = 5000  # Smaller batches to avoid timeouts
        
        logger.info("Starting multiple query download", target=total_vectors, max_iterations=max_iterations)
        
        for iteration in range(max_iterations):
            # Generate random query vector
            query_vector = np.random.normal(0, 1, dimension).tolist()
            
            try:
                result = index.query(
                    vector=query_vector,
                    top_k=batch_size,
                    include_metadata=include_metadata,
                    include_values=True
                )
                
                new_vectors = 0
                for match in result['matches']:
                    if match['id'] not in seen_ids:
                        seen_ids.add(match['id'])
                        all_embeddings.append(match['values'])
                        all_ids.append(match['id'])
                        
                        if include_metadata and 'metadata' in match:
                            all_metadata.append(match['metadata'])
                        else:
                            all_metadata.append({})
                        new_vectors += 1
                
                total_unique = len(all_embeddings)
                
                # Log progress every 10 iterations
                if iteration % 10 == 0 or new_vectors == 0:
                    logger.info("Query progress", 
                              iteration=iteration+1, 
                              new_vectors=new_vectors, 
                              total_unique=total_unique,
                              coverage_percent=round(total_unique/total_vectors*100, 1))
                
                # Stop if we're not finding many new vectors
                if new_vectors < 50 and iteration > 20:
                    logger.info("Diminishing returns, stopping query method", 
                              final_count=total_unique, 
                              coverage_percent=round(total_unique/total_vectors*100, 1))
                    break
                
                # Stop if we got everything
                if total_unique >= total_vectors * 0.99:  # 99% coverage
                    logger.info("Near complete coverage achieved", 
                              final_count=total_unique,
                              coverage_percent=round(total_unique/total_vectors*100, 1))
                    break
                    
            except Exception as e:
                logger.warning("Query iteration failed", iteration=iteration+1, error=str(e))
                continue
        
        logger.info("Multiple query download complete", 
                   downloaded=len(all_embeddings), 
                   target=total_vectors,
                   coverage_percent=round(len(all_embeddings)/total_vectors*100, 1))
        
        return np.array(all_embeddings), all_metadata, all_ids
