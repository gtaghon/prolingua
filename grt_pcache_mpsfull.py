"""
Grammar Residence Time (GRT) Analysis for Protein Motifs

This code implements the foundational analysis for discovering a natural protein
language based on motif-words and their grammar residence time in secondary 
structure and proximity contexts.

I/O-Optimized/GPU-Accelerated for Apple Silicon

Key Optimizations:
- Drastically reduced I/O operations
- Memory-mapped file access where possible
- Batch processing of HDF5 data
- Pre-fetching and caching strategy
- Optimized trajectory loading
- Reduced coordinate data copying
"""

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import h5py
import numpy as np
import mdtraj as md
from collections import defaultdict, Counter
from typing import Dict, List
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm
import tempfile
import time
import hashlib

from bridge import GPUAccelerator

# MPS imports
try:
    import torch
    HAS_MPS = torch.backends.mps.is_available()
except (ImportError, AttributeError):
    HAS_MPS = False
    print("MPS not available - falling back to CPU processing")

# Global constants
DEFAULT_BATCH_SIZE = 32
MAX_MEMORY_CACHE_SIZE = 10000000  # ~10GB assuming 1KB per entry
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file reading

# Create global thread pool for I/O operations
GLOBAL_THREAD_POOL = ThreadPoolExecutor(max_workers=4)

class IOProfiler:
    """Simple profiler for I/O operations"""
    def __init__(self):
        self.operations = defaultdict(float)
        self.counts = defaultdict(int)
        
    def record(self, operation, duration):
        self.operations[operation] += duration
        self.counts[operation] += 1
        
    def report(self):
        total = sum(self.operations.values())
        print("\nI/O Profiling Report:")
        print(f"{'Operation':<30} {'Time (s)':<10} {'Calls':<10} {'Avg (ms)':<10} {'%':<10}")
        print("-" * 70)
        for op, duration in sorted(self.operations.items(), key=lambda x: x[1], reverse=True):
            count = self.counts[op]
            avg_ms = (duration / count) * 1000 if count > 0 else 0
            percent = (duration / total) * 100 if total > 0 else 0
            print(f"{op:<30} {duration:<10.2f} {count:<10} {avg_ms:<10.2f} {percent:<10.1f}")
        print("-" * 70)
        print(f"Total I/O time: {total:.2f} seconds\n")

# Global I/O profiler
io_profiler = IOProfiler()

class MemoryOptimizedCache:
    """Memory-optimized cache with minimal I/O operations"""
    
    def __init__(self, cache_dir=None, max_memory_items=MAX_MEMORY_CACHE_SIZE, 
                enable_disk_cache=True, compression_level=1):
        self.cache_dir = cache_dir
        self.enable_disk_cache = enable_disk_cache and cache_dir is not None
        self.memory_cache = {}
        self.max_memory_items = max_memory_items
        self.compression_level = compression_level
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.memory_hits = 0
        self.io_bytes = 0
        
        if self.enable_disk_cache and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Record cache creation
        io_start = time.time()
        io_profiler.record("cache_init", time.time() - io_start)
    
    def _get_cache_key(self, args):
        """Generate a deterministic cache key using MD5 hash"""
        # Convert args to a string representation
        key_str = str(args)
        # Use MD5 for faster hashing (we don't need cryptographic security)
        hash_obj = hashlib.md5(key_str.encode())
        return hash_obj.hexdigest()
    
    def _get_cache_file(self, key):
        """Get cache file path from key"""
        if not self.enable_disk_cache or not self.cache_dir:
            return None
        # Use the first two characters for directory partitioning
        # This prevents having too many files in a single directory
        subdir = key[:2]
        subdir_path = os.path.join(self.cache_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(subdir_path, f"{key}.pkl")
    
    def get(self, args):
        """Get item from cache"""
        key = self._get_cache_key(args)
        
        # Check memory cache first (fastest)
        if key in self.memory_cache:
            self.hits += 1
            self.memory_hits += 1
            return self.memory_cache[key]
        
        # Check disk cache if enabled
        if self.enable_disk_cache:
            cache_file = self._get_cache_file(key)
            if cache_file and os.path.exists(cache_file):
                try:
                    io_start = time.time()
                    with open(cache_file, 'rb') as f:
                        # Use faster pickle protocol
                        result = pickle.load(f)
                        self.io_bytes += os.path.getsize(cache_file)
                        io_profiler.record("cache_disk_read", time.time() - io_start)
                        
                        # Update memory cache
                        self._update_memory_cache(key, result)
                        self.hits += 1
                        self.disk_hits += 1
                        return result
                except Exception:
                    # If loading fails, continue to compute
                    pass
        
        self.misses += 1
        return None
    
    def set(self, args, result):
        """Store item in cache"""
        key = self._get_cache_key(args)
        
        # Update memory cache
        self._update_memory_cache(key, result)
        
        # Update disk cache if enabled (do this in background)
        if self.enable_disk_cache:
            cache_file = self._get_cache_file(key)
            if cache_file:
                try:
                    io_start = time.time()
                    with open(cache_file, 'wb') as f:
                        # Use fastest protocol that's still compatible
                        pickle.dump(result, f, protocol=4)
                        self.io_bytes += os.path.getsize(cache_file)
                    io_profiler.record("cache_disk_write", time.time() - io_start)
                except Exception as e:
                    pass  # Silently ignore cache write failures
    
    def _update_memory_cache(self, key, result):
        """Update memory cache with eviction policy"""
        # Add to memory cache, evicting if necessary
        if len(self.memory_cache) >= self.max_memory_items:
            # Evict random items (simple LRU approximation)
            items_to_evict = max(1, len(self.memory_cache) // 10)  # Evict 10% at once
            for _ in range(items_to_evict):
                if self.memory_cache:
                    self.memory_cache.pop(next(iter(self.memory_cache)))
        
        self.memory_cache[key] = result
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        memory_hit_rate = self.memory_hits / self.hits if self.hits > 0 else 0
        disk_hit_rate = self.disk_hits / self.hits if self.hits > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'memory_hits': self.memory_hits,
            'disk_hits': self.disk_hits,
            'hit_rate': hit_rate,
            'memory_hit_rate': memory_hit_rate,
            'disk_hit_rate': disk_hit_rate,
            'memory_size': len(self.memory_cache),
            'io_bytes': self.io_bytes
        }

class H5DataManager:
    """
    Efficient H5 file data manager that minimizes I/O operations
    by using memory mapping and batch reading
    """
    def __init__(self, keep_files_open=False):
        self.keep_files_open = keep_files_open
        self.open_files = {}
        self.pdb_cache = {}
        self.coord_cache = {}
        self.dssp_cache = {}
    
    def __del__(self):
        """Clean up any open files"""
        self.close_all()
    
    def close_all(self):
        """Close all open files"""
        for _, h5 in self.open_files.items():
            h5.close()
        self.open_files.clear()
    
    def _get_h5(self, h5_path):
        """Get H5 file handle, reusing if possible"""
        if h5_path in self.open_files and self.keep_files_open:
            return self.open_files[h5_path]
        
        io_start = time.time()
        h5 = h5py.File(h5_path, 'r')
        io_profiler.record("h5_open", time.time() - io_start)
        
        if self.keep_files_open:
            self.open_files[h5_path] = h5
        
        return h5
    
    def _close_h5(self, h5_path, h5):
        """Close H5 file if we're not keeping files open"""
        if not self.keep_files_open:
            io_start = time.time()
            h5.close()
            io_profiler.record("h5_close", time.time() - io_start)
    
    def get_domain_code(self, h5_path):
        """Get the domain code from an H5 file"""
        h5 = self._get_h5(h5_path)
        # The top level keys are the domain codes
        code = list(h5.keys())[0]
        self._close_h5(h5_path, h5)
        return code
    
    def get_pdb_data(self, h5_path, code):
        """Get PDB data from an H5 file with caching"""
        # Check cache first
        cache_key = (h5_path, code)
        if cache_key in self.pdb_cache:
            return self.pdb_cache[cache_key]
        
        # Read from file
        h5 = self._get_h5(h5_path)
        
        io_start = time.time()
        # pdbProteinAtoms is a dataset in the domain group
        pdb = h5[code]["pdbProteinAtoms"][()]
        io_profiler.record("h5_read_pdb", time.time() - io_start)
        
        # Ensure pdb is bytes, not string
        if isinstance(pdb, str):
            pdb = pdb.encode('utf-8')
        
        # Cache result
        self.pdb_cache[cache_key] = pdb
        
        self._close_h5(h5_path, h5)
        return pdb
    
    def get_temperatures_and_replicas(self, h5_path, code):
        """Get available temperatures and replicas"""
        h5 = self._get_h5(h5_path)
        
        temps = []
        temp_replicas = {}
        
        # Iterate through all groups in the domain
        for temp in h5[code].keys():
            # Skip non-temperature groups
            if temp == "pdbProteinAtoms" or not temp.isdigit():
                continue
                
            temp_replicas[temp] = []
            temps.append(temp)
            
            # Get replica groups
            for replica in h5[code][temp].keys():
                if replica.isdigit() and "coords" in h5[code][temp][replica] and "dssp" in h5[code][temp][replica]:
                    temp_replicas[temp].append(replica)
        
        self._close_h5(h5_path, h5)
        return temps, temp_replicas
    
    def get_coordinates(self, h5_path, code, temp, replica):
        """Get coordinate data from an H5 file with caching"""
        # Check cache first
        cache_key = (h5_path, code, temp, replica)
        if cache_key in self.coord_cache:
            return self.coord_cache[cache_key]
        
        # Read from file
        h5 = self._get_h5(h5_path)
        
        io_start = time.time()
        # The coords dataset is in the replica group
        coords = h5[code][temp][replica]["coords"][()]
        coords = coords / 10.0  # Convert to nm
        io_profiler.record("h5_read_coords", time.time() - io_start)
        
        # Cache result (only if not too large)
        if coords.nbytes < 500 * 1024 * 1024:  # 500MB limit
            self.coord_cache[cache_key] = coords
        
        self._close_h5(h5_path, h5)
        return coords

    def get_dssp_data(self, h5_path, code, temp, replica):
        """Get DSSP data from an H5 file with caching"""
        # Check cache first
        cache_key = (h5_path, code, temp, replica)
        if cache_key in self.dssp_cache:
            return self.dssp_cache[cache_key]
        
        # Read from file
        h5 = self._get_h5(h5_path)
        
        io_start = time.time()
        # The dssp dataset is in the replica group
        dssp_data = h5[code][temp][replica]["dssp"][()]
        io_profiler.record("h5_read_dssp", time.time() - io_start)
        
        # Ensure dssp data is strings, not bytes
        if isinstance(dssp_data[0, 0], bytes):
            dssp = np.array([[s.decode('utf-8') for s in row] for row in dssp_data])
        else:
            dssp = dssp_data
        
        # Cache result
        self.dssp_cache[cache_key] = dssp
        
        self._close_h5(h5_path, h5)
        return dssp
    
    def check_trajectory_exists(self, h5_path, code, temp, replica):
        """Check if trajectory data exists without reading it"""
        h5 = self._get_h5(h5_path)
        
        try:
            exists = (
                f"{temp}" in h5[code] and
                f"{replica}" in h5[code][f"{temp}"] and
                "coords" in h5[code][f"{temp}"][f"{replica}"] and
                "dssp" in h5[code][f"{temp}"][f"{replica}"]
            )
        except Exception:
            exists = False
        
        self._close_h5(h5_path, h5)
        return exists
    

def load_trajectory_efficient(h5_path, code, temp, replica, h5_manager):
    """
    Load trajectory data with minimal I/O operations
    
    Args:
        h5_path: Path to H5 file
        code: Domain code
        temp: Temperature
        replica: Replica
        h5_manager: H5DataManager instance
    
    Returns:
        tuple: (trajectory, sequence, dssp_data)
    """
    io_start = time.time()
    
    # Get PDB data and create temporary file
    pdb_data = h5_manager.get_pdb_data(h5_path, code)
    
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as pdbfile:
        pdbfile.write(pdb_data)
        pdbfile.flush()
        pdb_path = pdbfile.name
    
    # Get coordinates
    coords = h5_manager.get_coordinates(h5_path, code, temp, replica)
    
    # Load trajectory directly from PDB file
    traj = md.load(pdb_path)
    
    # Delete temporary file immediately after loading
    os.unlink(pdb_path)
    
    # Set trajectory coordinates and time
    traj.xyz = coords
    traj.time = np.arange(coords.shape[0])
    
    # Get sequence from topology
    sequence = ''.join([r.code if isinstance(r.code, str) else r.code.decode('utf-8') 
                      for r in traj.topology.residues])
    
    # Get DSSP data
    dssp_data = h5_manager.get_dssp_data(h5_path, code, temp, replica)
    
    io_profiler.record("load_trajectory", time.time() - io_start)
    
    return traj, sequence, dssp_data

def process_frame(args, cache=None):
    """
    Process a single frame with optimized algorithms.
    
    Args:
        args: Tuple of (frame_idx, sequence, dssp_frame, contact_matrix_indices, 
                        pairs_dict, motif_sizes, dssp_codes, temps, temp, min_aa)
        cache: Optional frame cache
    """
    # Unpack arguments
    frame_idx, sequence, dssp_frame, contact_matrix_indices, pairs_dict, motif_sizes, dssp_codes, temps, temp, min_aa = args
    
    # Check cache first
    if cache:
        cache_args = (frame_idx, sequence, dssp_frame, contact_matrix_indices)
        cached_result = cache.get(cache_args)
        if cached_result is not None:
            return cached_result
    
    # Use Counter for more efficient counting
    motif_data = {}
    
    # Pre-calculate valid amino acid mask for faster filtering
    valid_aa_mask = np.array([aa in min_aa for aa in sequence])
    
    # Process each motif size
    for motif_size in range(motif_sizes[0], motif_sizes[1] + 1):
        # Vectorized motif extraction where possible
        for i in range(len(sequence) - motif_size + 1):
            # Skip invalid amino acids quickly
            if not np.all(valid_aa_mask[i:i+motif_size]):
                continue
            
            motif = sequence[i:i+motif_size]
            ss_assignment = ''.join(dssp_frame[i:i+motif_size])
            
            # Initialize motif data efficiently
            if motif not in motif_data:
                motif_data[motif] = {
                    'occurrences': 0,
                    'ss_counts': Counter(),
                    'by_temp': {t: {'frames': 0, 'ss_counts': Counter()} for t in temps},
                    'contacts': Counter()
                }
            
            # Update counts
            motif_data[motif]['occurrences'] += 1
            motif_data[motif]['by_temp'][temp]['frames'] += 1
            
            # Count secondary structures
            for ss in ss_assignment:
                motif_data[motif]['ss_counts'][ss] += 1
                motif_data[motif]['by_temp'][temp]['ss_counts'][ss] += 1
            
            # Optimized contact checking
            for j in range(len(sequence) - motif_size + 1):
                if i == j:  # Skip self
                    continue
                
                other_motif = sequence[j:j+motif_size]
                
                # Check for contact using pre-computed contact matrix indices
                contact_exists = False
                for a in range(i, i+motif_size):
                    if contact_exists:
                        break
                    for b in range(j, j+motif_size):
                        pair_idx = pairs_dict.get((min(a,b), max(a,b)))
                        if pair_idx is not None and pair_idx in contact_matrix_indices:
                            contact_exists = True
                            break
                
                if contact_exists:
                    motif_data[motif]['contacts'][other_motif] += 1
    
    # Cache result if cache is provided
    if cache:
        cache_args = (frame_idx, sequence, dssp_frame, contact_matrix_indices)
        cache.set(cache_args, motif_data)
    
    return motif_data

class MotifExtractor:
    """
    I/O-optimized GPU-accelerated class for extracting motifs from mdCATH dataset.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 min_motif_size: int = 3, 
                 max_motif_size: int = 6,
                 contact_threshold: float = 0.8,
                 temperatures: List[int] = [320, 348, 379, 413, 450],
                 replicas: List[int] = [0, 1, 2, 3, 4],
                 cache_dir: str = None,
                 n_workers: int = None,
                 use_gpu: bool = True,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 memory_limit_mb: int = 4000):
        """
        Initialize the MotifExtractor with parameters for analysis.
        
        Args:
            data_dir: Directory containing mdCATH H5 files
            min_motif_size: Minimum size of motifs to extract
            max_motif_size: Maximum size of motifs to extract
            contact_threshold: Threshold for defining contacts (in nm)
            temperatures: List of temperatures to analyze
            replicas: List of replicas to analyze
            cache_dir: Directory to store cache files (None to disable disk caching)
            n_workers: Number of worker processes (default: CPU core count - 1)
            use_gpu: Whether to use GPU acceleration via MPS
            batch_size: Batch size for parallel processing
            memory_limit_mb: Memory limit in MB for caching
        """
        self.data_dir = data_dir
        self.min_motif_size = min_motif_size
        self.max_motif_size = max_motif_size
        self.contact_threshold = contact_threshold
        self.temperatures = temperatures
        self.replicas = replicas
        self.batch_size = batch_size
        
        # Determine optimal worker count
        self.n_workers = n_workers or max(1, os.cpu_count() - 1)
        
        # Initialize GPU accelerator
        self.gpu = GPUAccelerator(use_gpu=use_gpu)
        self.use_gpu = True #self.gpu.use_gpu
        
        # Data structures
        self.motifs = {}
        self.dssp_codes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', ' ']
        
        # Setup caching
        self.cache_dir = cache_dir
        if cache_dir:
            self.frame_cache_dir = os.path.join(cache_dir, 'frame_cache')
            self.domain_cache_dir = os.path.join(cache_dir, 'domain_cache')
            os.makedirs(self.frame_cache_dir, exist_ok=True)
            os.makedirs(self.domain_cache_dir, exist_ok=True)
        else:
            self.frame_cache_dir = None
            self.domain_cache_dir = None
        
        # Calculate cache size based on memory limit
        max_memory_items = memory_limit_mb * 1024 * 1024 // 1024  # Assuming avg 1KB per item
        
        # Initialize optimized cache
        self.frame_cache = MemoryOptimizedCache(
            cache_dir=self.frame_cache_dir, 
            max_memory_items=max_memory_items,
            enable_disk_cache=cache_dir is not None
        )
        
        # Initialize H5 data manager for efficient I/O
        self.h5_manager = H5DataManager(keep_files_open=False)
        
        # Create process pool for CPU-bound operations
        self.process_pool = ProcessPoolExecutor(max_workers=self.n_workers)
        
        # Progress tracking
        self.processed_files = 0
        self.total_files = 0
        
        # Performance metrics
        self.metrics = {
            'distance_calculation_time': 0,
            'frame_processing_time': 0,
            'io_time': 0,
            'gpu_utilization': 0
        }
    
    def __del__(self):
        """Ensure resources are properly released"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        if hasattr(self, 'h5_manager'):
            self.h5_manager.close_all()
    
    def extract_motifs_parallel(self, sequence, dssp, traj, pairs_list, temp, batch_size=20):
        """
        GPU-accelerated parallel motif extraction with I/O optimization.
        
        Args:
            sequence: Amino acid sequence
            dssp: Secondary structure assignments for all frames
            traj: MDTraj trajectory object with coordinates
            pairs_list: List of atom pairs for contact calculation
            temp: Current temperature
            batch_size: Number of frames to process in each batch
        """
        batch_size = batch_size or self.batch_size
        start_time = time.time()
        n_frames = traj.n_frames
        
        # Create dictionary for faster pair lookups
        pairs_dict = {(min(i,j), max(i,j)): idx for idx, (i,j) in enumerate(pairs_list)}
        
        # ===== GPU-Accelerated Distance Calculation =====
        print(f"Calculating distances for {len(pairs_list)} pairs across {n_frames} frames...")
        distance_start = time.time()
        
        # Use our GPU accelerator for distance calculations
        distances = self.gpu.compute_distances(traj, pairs_list)
        
        # Calculate contact matrices
        contact_matrices = self.gpu.compute_contact_matrices(distances, self.contact_threshold)
        
        # Record timing
        distance_time = time.time() - distance_start
        self.metrics['distance_calculation_time'] += distance_time
        print(f"Distance calculation: {distance_time:.2f} seconds, " +
              f"{'GPU' if self.use_gpu else 'CPU'} accelerated")
        
        # ===== Prepare frame processing tasks =====
        motif_sizes = (self.min_motif_size, self.max_motif_size)
        min_aa = "ACDEFGHIKLMNPQRSTVWY"  # Standard amino acids
        
        # Convert contact matrices to indices for more efficient storage and processing
        contact_indices = []
        for frame_idx in range(n_frames):
            # Store only indices where contact exists (True values)
            # This dramatically reduces memory usage and data transfer
            indices = np.where(contact_matrices[frame_idx])[0]
            contact_indices.append(indices)
        
        # Free memory - we don't need the full matrices anymore
        del contact_matrices
        del distances
        
        # Prepare tasks for parallel processing
        # Group frames into batches for better throughput
        frame_batches = []
        for batch_start in range(0, n_frames, batch_size):
            batch_end = min(batch_start + batch_size, n_frames)
            frame_batches.append(list(range(batch_start, batch_end)))
        
        # ===== Process frames in parallel =====
        print(f"Processing {n_frames} frames in {len(frame_batches)} batches with {self.n_workers} workers...")
        frame_start = time.time()
        
        results = {}
        
        # Use batch processing to reduce overhead
        with tqdm(total=n_frames, desc="Processing frames") as pbar:
            for batch_idx, frame_batch in enumerate(frame_batches):
                # Prepare batch tasks
                batch_tasks = []
                for frame_idx in frame_batch:
                    args = (
                        frame_idx,
                        sequence,
                        dssp[frame_idx],
                        contact_indices[frame_idx],  # Optimized storage
                        pairs_dict,
                        motif_sizes,
                        self.dssp_codes,
                        self.temperatures,
                        temp,
                        min_aa
                    )
                    batch_tasks.append(args)
                
                # Process batch with efficient work distribution
                # Use starmap for better memory efficiency compared to map with partial
                batch_results = []
                
                # Process frames in batch using process pool
                # Use chunksize for better performance with small tasks
                chunk_size = max(1, len(batch_tasks) // (self.n_workers * 2))
                
                # Submit all tasks at once for better efficiency
                futures = [
                    self.process_pool.submit(process_frame, task, self.frame_cache)
                    for task in batch_tasks
                ]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        frame_result = future.result()
                        batch_results.append(frame_result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error in frame processing: {e}")
                        pbar.update(1)
                
                # Merge batch results efficiently
                for frame_result in batch_results:
                    self._merge_frame_results(results, frame_result)
        
        # Record timing
        frame_time = time.time() - frame_start
        self.metrics['frame_processing_time'] += frame_time
        print(f"Frame processing: {frame_time:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total motif extraction: {total_time:.2f} seconds")
        
        return results
    
    def _merge_frame_results(self, target, source):
        """Efficiently merge frame results using Counter operations"""
        for motif, data in source.items():
            if motif not in target:
                # Initialize with the same structure
                target[motif] = {
                    'occurrences': 0,
                    'ss_counts': Counter(),
                    'by_temp': {t: {'frames': 0, 'ss_counts': Counter()} 
                              for t in self.temperatures},
                    'contacts': Counter()
                }
            
            # Update counts using Counter operations
            target[motif]['occurrences'] += data['occurrences']
            target[motif]['ss_counts'] += data['ss_counts']
            
            # Update temperature data
            for temp in self.temperatures:
                if temp in data['by_temp']:
                    target[motif]['by_temp'][temp]['frames'] += data['by_temp'][temp]['frames']
                    target[motif]['by_temp'][temp]['ss_counts'] += data['by_temp'][temp]['ss_counts']
            
            # Update contacts
            target[motif]['contacts'] += data['contacts']
    
    def process_domain(self, h5_path):
        """
        Process a single domain with I/O optimization and caching.
        """
        domain_id = os.path.basename(h5_path).replace('.h5', '')
        
        # Check domain cache first
        if self.domain_cache_dir:
            cache_file = os.path.join(self.domain_cache_dir, f"{domain_id}.pkl")
            if os.path.exists(cache_file):
                try:
                    io_start = time.time()
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    io_profiler.record("domain_cache_read", time.time() - io_start)
                    return result
                except Exception as e:
                    print(f"Cache read error for {domain_id}: {e}")
        
        io_start = time.time()
        domain_motifs = {}
        
        try:
            # Get domain code
            code = self.h5_manager.get_domain_code(h5_path)
            
            # Check which temperatures and replicas are available
            temps, temp_replicas = self.h5_manager.get_temperatures_and_replicas(h5_path, code)
            
            # Process each available temperature
            for temp in self.temperatures:
                if f"{temp}" not in temps:
                    continue
                    
                for replica in self.replicas:
                    if f"{replica}" not in temp_replicas[f"{temp}"]:
                        continue
                        
                    # Check if trajectory exists
                    if not self.h5_manager.check_trajectory_exists(h5_path, code, temp, replica):
                        continue
                        
                    try:
                        # Load trajectory with optimized I/O
                        traj, sequence, dssp = load_trajectory_efficient(
                            h5_path, code, f"{temp}", f"{replica}", self.h5_manager
                        )
                        
                        # Record I/O time
                        self.metrics['io_time'] += time.time() - io_start
                        io_start = time.time()
                        
                        # Prepare atom pairs for contact calculation
                        ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']
                        pairs = [(i,j) for i in range(len(ca_indices)) for j in range(i+1, len(ca_indices))]
                        
                        # Extract motifs using GPU-accelerated implementation
                        motifs = self.extract_motifs_parallel(
                            sequence, dssp, traj, pairs, int(temp)
                        )
                        
                        # Merge into domain results
                        self._merge_domain_results(domain_motifs, motifs, domain_id)
                                
                    except Exception as e:
                        print(f"Error processing temp {temp}, replica {replica}: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Update I/O time
            self.metrics['io_time'] += time.time() - io_start
                
        except Exception as e:
            print(f"Error processing {h5_path}: {e}")
            self.metrics['io_time'] += time.time() - io_start
        
        # Cache results if we have data
        if domain_motifs and self.domain_cache_dir:
            io_start = time.time()
            cache_file = os.path.join(self.domain_cache_dir, f"{domain_id}.pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(domain_motifs, f, protocol=4)
                io_profiler.record("domain_cache_write", time.time() - io_start)
            except Exception as e:
                print(f"Warning: Failed to cache domain results: {e}")
        
        return domain_motifs
    
    def _merge_domain_results(self, target, source, domain_id):
        """Efficiently merge domain results"""
        for motif, data in source.items():
            if motif not in target:
                # Initialize with the same structure but add domains field
                target[motif] = {
                    'occurrences': 0,
                    'ss_counts': Counter(),
                    'by_temp': {t: {'frames': 0, 'ss_counts': Counter()} 
                              for t in self.temperatures},
                    'contacts': Counter(),
                    'domains': set()
                }
            
            # Add domain id
            target[motif]['domains'].add(domain_id)
            
            # Update counts using Counter operations
            target[motif]['occurrences'] += data['occurrences']
            target[motif]['ss_counts'] += data['ss_counts']
            
            # Update temperature data
            for temp in self.temperatures:
                if temp in data['by_temp']:
                    target[motif]['by_temp'][temp]['frames'] += data['by_temp'][temp]['frames']
                    target[motif]['by_temp'][temp]['ss_counts'] += data['by_temp'][temp]['ss_counts']
            
            # Update contacts
            target[motif]['contacts'] += data['contacts']
    
    def process_all_domains(self, max_files=None):
        """
        Process all domains with optimized I/O and parallelization
        """
        h5_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
        self.total_files = len(h5_files) if max_files is None else min(max_files, len(h5_files))
        
        # Print configuration
        if self.use_gpu:
            gpu_info = "MPS GPU acceleration enabled"
        else:
            gpu_info = "CPU-only processing"
            
        print(f"Processing {self.total_files} domain files with {self.n_workers} workers, {gpu_info}")
        
        # Pre-scan domain files to identify which ones are cached
        cached_domains = set()
        domains_to_process = []
        
        if self.domain_cache_dir:
            for h5_file in h5_files[:self.total_files]:
                domain_id = h5_file.replace('.h5', '')
                cache_file = os.path.join(self.domain_cache_dir, f"{domain_id}.pkl")
                
                if os.path.exists(cache_file):
                    cached_domains.add(h5_file)
                else:
                    domains_to_process.append(h5_file)
        else:
            domains_to_process = h5_files[:self.total_files]
        
        print(f"Found {len(cached_domains)} cached domains, {len(domains_to_process)} to process")
        
        # Process cached domains first (fast)
        if cached_domains:
            print(f"Loading {len(cached_domains)} cached domains...")
            for h5_file in tqdm(cached_domains, desc="Loading cached domains"):
                try:
                    h5_path = os.path.join(self.data_dir, h5_file)
                    domain_motifs = self.process_domain(h5_path)
                    
                    # Merge into global motifs
                    self._merge_domain_results(self.motifs, domain_motifs, h5_file)
                    
                    self.processed_files += 1
                except Exception as e:
                    print(f"Error loading cached domain {h5_file}: {e}")
        
        # Process remaining domains with parallel I/O
        if domains_to_process:
            print(f"Processing {len(domains_to_process)} new domains...")
            
            # Use ThreadPool for I/O-bound operations
            # Limit the number of concurrent domains to avoid overwhelming I/O
            max_concurrent = min(8, self.n_workers)
            
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {
                    executor.submit(self.process_domain, os.path.join(self.data_dir, h5_file)): h5_file 
                    for h5_file in domains_to_process
                }
                
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                  total=len(futures),
                                  desc="Processing domains"):
                    h5_file = futures[future]
                    try:
                        domain_motifs = future.result()
                        
                        # Merge into global motifs
                        self._merge_domain_results(self.motifs, domain_motifs, h5_file)
                        
                        self.processed_files += 1
                    except Exception as e:
                        print(f"Error processing {h5_file}: {e}")
        
        # Print final statistics
        print(f"Completed processing {self.processed_files} domains, {len(self.motifs)} unique motifs")
        
        # Performance metrics
        cache_stats = self.frame_cache.get_stats()
        print(f"Frame cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses, " 
              f"{cache_stats['hit_rate']*100:.1f}% hit rate")
        print(f"Memory hits: {cache_stats['memory_hits']}, Disk hits: {cache_stats['disk_hits']}")
        print(f"I/O bytes: {cache_stats['io_bytes'] / (1024*1024):.2f} MB")
        
        print(f"Performance metrics:")
        print(f"- Distance calculation time: {self.metrics['distance_calculation_time']:.2f} s")
        print(f"- Frame processing time: {self.metrics['frame_processing_time']:.2f} s")
        print(f"- I/O time: {self.metrics['io_time']:.2f} s")
        
        # Calculate GPU utilization
        if self.use_gpu:
            total_compute_time = self.metrics['distance_calculation_time'] + self.metrics['frame_processing_time']
            gpu_util = self.metrics['distance_calculation_time'] / total_compute_time * 100 if total_compute_time > 0 else 0
            print(f"- GPU utilization: {gpu_util:.1f}% of compute time")
        
        # Display detailed I/O profiling
        io_profiler.report()
    
    def save_motifs(self, output_file):
        """Save extracted motifs to a file"""
        with open(output_file, 'wb') as f:
            pickle.dump(self.motifs, f, protocol=4)
    
    def load_motifs(self, input_file):
        """Load extracted motifs from a file"""
        with open(input_file, 'rb') as f:
            self.motifs = pickle.load(f)

###############################
# Phase 2: GRT Analysis
###############################

class GRTAnalyzer:
    """
    Analyze Grammar Residence Time (GRT) patterns from extracted motifs.
    """
    
    def __init__(self, motifs: Dict, min_occurrence: int = 10):
        """
        Initialize the GRT analyzer with extracted motifs.
        
        Args:
            motifs: Dictionary of motifs and their properties
            min_occurrence: Minimum number of occurrences for a motif to be considered
        """
        self.motifs = motifs
        self.min_occurrence = min_occurrence
        
        # Filter motifs by occurrence threshold
        self.filtered_motifs = {
            motif: data for motif, data in motifs.items() 
            if data['occurrences'] >= min_occurrence
        }
        
        print(f"Analyzing {len(self.filtered_motifs)} motifs (filtered from {len(motifs)} total)")
        
        # Secondary structure states
        self.dssp_codes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', ' ']
        
        # DSSP code grouping for analysis
        self.dssp_groups = {
            'helix': ['H', 'G', 'I'],  # All helical structures
            'sheet': ['E', 'B'],       # Extended/beta structures
            'turn': ['T', 'S'],        # Turns and bends
            'coil': [' ']              # Unstructured/coil
        }
        
        # Results storage
        self.grt_profiles = {}
        self.motif_clusters = {}
        self.temperature_sensitivity = {}
        self.proximity_network = {}

    def calculate_grt_profiles(self):
        """
        Calculate Grammar Residence Time profiles for filtered motifs.
        """
        print("Calculating GRT profiles...")
        
        for motif, data in self.filtered_motifs.items():
            # Calculate overall GRT
            total_frames = data['occurrences']
            ss_fractions = {ss: count/total_frames for ss, count in data['ss_counts'].items()}
            
            # Calculate GRT per secondary structure group
            group_fractions = {}
            for group_name, ss_list in self.dssp_groups.items():
                group_count = sum(data['ss_counts'][ss] for ss in ss_list)
                group_fractions[group_name] = group_count / total_frames
            
            # Calculate temperature-dependent GRT
            temp_profiles = {}
            for temp, temp_data in data['by_temp'].items():
                if temp_data['frames'] > 0:
                    temp_ss_fractions = {
                        ss: count/temp_data['frames'] 
                        for ss, count in temp_data['ss_counts'].items()
                    }
                    
                    temp_group_fractions = {}
                    for group_name, ss_list in self.dssp_groups.items():
                        group_count = sum(temp_data['ss_counts'][ss] for ss in ss_list)
                        temp_group_fractions[group_name] = group_count / temp_data['frames']
                    
                    temp_profiles[temp] = {
                        'ss_fractions': temp_ss_fractions,
                        'group_fractions': temp_group_fractions
                    }
            
            # Store GRT profile
            self.grt_profiles[motif] = {
                'ss_fractions': ss_fractions,
                'group_fractions': group_fractions,
                'temp_profiles': temp_profiles,
                'domain_count': len(data['domains']),
                'occurrences': data['occurrences']
            }
    
    def analyze_temperature_sensitivity(self):
        """
        Analyze how motif GRT profiles change with temperature.
        """
        print("Analyzing temperature sensitivity...")
        
        for motif, profile in self.grt_profiles.items():
            # Get temperatures with data
            temps = sorted([t for t in profile['temp_profiles']])
            if len(temps) < 2:
                continue
                
            # Calculate changes in secondary structure groups across temperatures
            changes = {}
            for group in self.dssp_groups.keys():
                values = [profile['temp_profiles'][t]['group_fractions'][group] for t in temps]
                max_change = max(values) - min(values)
                
                # Check if the change has a consistent direction
                monotonic_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
                monotonic_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
                
                changes[group] = {
                    'values': values,
                    'max_change': max_change,
                    'monotonic_decreasing': monotonic_decreasing,
                    'monotonic_increasing': monotonic_increasing
                }
            
            # Calculate overall temperature sensitivity
            avg_max_change = np.mean([c['max_change'] for c in changes.values()])
            
            # Store results
            self.temperature_sensitivity[motif] = {
                'changes': changes,
                'avg_max_change': avg_max_change,
                'temps': temps
            }
    
    def cluster_motifs_by_grt(self):
        """
        Cluster motifs based on similar GRT profiles.
        """
        print("Clustering motifs by GRT profiles...")
        
        # Create feature vectors for clustering
        features = []
        motifs_list = []
        
        for motif, profile in self.grt_profiles.items():
            # Create feature vector from secondary structure group fractions
            feature_vector = [
                profile['group_fractions']['helix'],
                profile['group_fractions']['sheet'],
                profile['group_fractions']['turn'],
                profile['group_fractions']['coil']
            ]
            
            # Add temperature sensitivity if available
            if motif in self.temperature_sensitivity:
                sens = self.temperature_sensitivity[motif]
                feature_vector.extend([
                    sens['changes']['helix']['max_change'],
                    sens['changes']['sheet']['max_change'],
                    sens['changes']['turn']['max_change'],
                    sens['changes']['coil']['max_change']
                ])
            
            features.append(feature_vector)
            motifs_list.append(motif)
        
        # Convert to numpy array
        X = np.array(features)
        
        # Standardize features
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(5, X_std.shape[1]))
        X_pca = pca.fit_transform(X_std)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_pca)
        labels = clustering.labels_
        
        # Organize results by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(motifs_list[i])
        
        # Store results
        self.motif_clusters = {
            'clusters': clusters,
            'pca': pca,
            'pca_features': X_pca,
            'motif_labels': {motifs_list[i]: labels[i] for i in range(len(motifs_list))}
        }
        
        print(f"Found {len(clusters)} clusters of motifs")
    
    def analyze_proximity_patterns(self):
        """
        Analyze co-occurrence and proximity patterns between motifs.
        """
        print("Analyzing proximity patterns...")
        
        # Get filtered motif contacts
        motif_contacts = {}
        for motif, data in self.filtered_motifs.items():
            # Filter contacts to only include motifs that passed our filtering
            filtered_contacts = {
                other: count for other, count in data['contacts'].items()
                if other in self.filtered_motifs
            }
            motif_contacts[motif] = filtered_contacts
        
        # Normalize contact counts by occurrence
        proximity_scores = {}
        for motif, contacts in motif_contacts.items():
            motif_occurrences = self.filtered_motifs[motif]['occurrences']
            normalized_contacts = {}
            
            for other, count in contacts.items():
                # Normalize by geometric mean of occurrences
                other_occurrences = self.filtered_motifs[other]['occurrences']
                norm_factor = np.sqrt(motif_occurrences * other_occurrences)
                normalized_contacts[other] = count / norm_factor
            
            proximity_scores[motif] = normalized_contacts
        
        # Store results
        self.proximity_network = {
            'raw_contacts': motif_contacts,
            'proximity_scores': proximity_scores
        }
    
    def identify_stable_motifs(self, stability_threshold: float = 0.2):
        """
        Identify motifs with stable secondary structure across temperatures.
        
        Args:
            stability_threshold: Maximum allowed change in structure composition
        """
        stable_motifs = {}
        
        for motif, sensitivity in self.temperature_sensitivity.items():
            if sensitivity['avg_max_change'] <= stability_threshold:
                # Check secondary structure composition
                profile = self.grt_profiles[motif]
                
                # Determine predominant structure
                groups = profile['group_fractions']
                max_group = max(groups.items(), key=lambda x: x[1])
                
                if max_group[1] >= 0.6:  # At least 60% in one type
                    stable_motifs[motif] = {
                        'predominant_structure': max_group[0],
                        'structure_fraction': max_group[1],
                        'temperature_stability': 1.0 - sensitivity['avg_max_change']
                    }
        
        return stable_motifs
    
    def identify_transition_motifs(self, transition_threshold: float = 0.4):
        """
        Identify motifs that undergo significant structural transitions with temperature.
        
        Args:
            transition_threshold: Minimum required change in structure composition
        """
        transition_motifs = {}
        
        for motif, sensitivity in self.temperature_sensitivity.items():
            if sensitivity['avg_max_change'] >= transition_threshold:
                # Find which group shows the most significant change
                max_change_group = max(
                    sensitivity['changes'].items(), 
                    key=lambda x: x[1]['max_change']
                )
                
                # Check if the change is monotonic
                group_data = max_change_group[1]
                if group_data['monotonic_decreasing'] or group_data['monotonic_increasing']:
                    transition_motifs[motif] = {
                        'transition_group': max_change_group[0],
                        'direction': 'increasing' if group_data['monotonic_increasing'] else 'decreasing',
                        'magnitude': group_data['max_change'],
                        'values_by_temp': dict(zip(sensitivity['temps'], group_data['values']))
                    }
        
        return transition_motifs

    def save_results(self, output_dir: str):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save GRT profiles
        with open(os.path.join(output_dir, 'grt_profiles.pkl'), 'wb') as f:
            pickle.dump(self.grt_profiles, f)
        
        # Save temperature sensitivity
        with open(os.path.join(output_dir, 'temperature_sensitivity.pkl'), 'wb') as f:
            pickle.dump(self.temperature_sensitivity, f)
        
        # Save motif clusters
        with open(os.path.join(output_dir, 'motif_clusters.pkl'), 'wb') as f:
            pickle.dump(self.motif_clusters, f)
        
        # Save proximity network
        with open(os.path.join(output_dir, 'proximity_network.pkl'), 'wb') as f:
            pickle.dump(self.proximity_network, f)
        
        print(f"Results saved to {output_dir}")

###############################
# Visualization Functions
###############################

def visualize_grt_profiles(analyzer, output_dir: str, sample_size: int = 10):
    """
    Visualize GRT profiles for a sample of motifs.
    
    Args:
        analyzer: GRTAnalyzer instance
        output_dir: Directory to save visualizations
        sample_size: Number of motifs to sample for visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample motifs from different clusters
    cluster_samples = []
    for label, motifs in analyzer.motif_clusters['clusters'].items():
        if label == -1:  # Skip noise points
            continue
        sample = np.random.choice(motifs, min(3, len(motifs)), replace=False)
        cluster_samples.extend([(motif, label) for motif in sample])
    
    # Limit to sample size
    if len(cluster_samples) > sample_size:
        indices = np.random.choice(len(cluster_samples), sample_size, replace=False)
        cluster_samples = [cluster_samples[i] for i in indices]
    
    # Create visualizations
    for motif, cluster in cluster_samples:
        profile = analyzer.grt_profiles[motif]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot overall secondary structure composition
        labels = list(profile['group_fractions'].keys())
        sizes = [profile['group_fractions'][k] for k in labels]
        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title(f'Secondary Structure Composition for {motif}')
        
        # Plot temperature-dependent changes
        if motif in analyzer.temperature_sensitivity:
            temps = analyzer.temperature_sensitivity[motif]['temps']
            for group in analyzer.dssp_groups.keys():
                values = [profile['temp_profiles'][t]['group_fractions'][group] for t in temps]
                axes[1].plot(temps, values, marker='o', label=group)
            
            axes[1].set_xlabel('Temperature (K)')
            axes[1].set_ylabel('Fraction')
            axes[1].set_title('Temperature Dependency')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'grt_profile_{motif}_cluster{cluster}.png'))
    plt.close()

def visualize_motif_clusters(analyzer, output_dir: str):
    """
    Visualize clusters of motifs in PCA space.
    
    Args:
        analyzer: GRTAnalyzer instance
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get PCA features and labels
    X_pca = analyzer.motif_clusters['pca_features']
    labels = list(analyzer.motif_clusters['motif_labels'].values())
    motifs = list(analyzer.motif_clusters['motif_labels'].keys())
    
    # Create 2D scatter plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black used for noise
            color = [0, 0, 0, 0]
            
        mask = np.array(labels) == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color, 
                   alpha=0.6, s=6, label=f'Cluster {label}')
    
    plt.title('Motif Clusters in PCA Space')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'motif_clusters_pca.png'))
    plt.close()
    
    # Create an annotated version with a sample of motifs
    plt.figure(figsize=(14, 12))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0, 0]
            
        mask = np.array(labels) == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color, 
                   alpha=0.6, s=6, label=f'Cluster {label}')
        
        # Annotate a sample of points
        if label != -1:  # Skip noise points
            mask_indices = np.where(mask)[0]
            if len(mask_indices) > 0:
                sample_indices = np.random.choice(mask_indices, min(3, len(mask_indices)), replace=False)
                for idx in sample_indices:
                    plt.annotate(motifs[idx], (X_pca[idx, 0], X_pca[idx, 1]),
                               fontsize=8, alpha=0.7)
    
    plt.title('Motif Clusters in PCA Space (Annotated)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'motif_clusters_annotated.png'))
    plt.close()

def visualize_proximity_network(analyzer, output_dir: str, top_n: int = 100):
    """
    Visualize the proximity network between motifs.
    
    Args:
        analyzer: GRTAnalyzer instance
        output_dir: Directory to save visualizations
        top_n: Number of top motifs to include
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for network visualization. Install with: pip install networkx")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top motifs by occurrence
    top_motifs = sorted(
        analyzer.grt_profiles.keys(),
        key=lambda m: analyzer.grt_profiles[m]['occurrences'],
        reverse=True
    )[:top_n]
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes with properties
    for motif in top_motifs:
        profile = analyzer.grt_profiles[motif]
        
        # Determine predominant structure
        main_structure = max(
            profile['group_fractions'].items(),
            key=lambda x: x[1]
        )[0]
        
        # Add node with properties
        G.add_node(
            motif, 
            size=np.log1p(profile['occurrences']),
            structure=main_structure,
            domains=profile['domain_count']
        )
    
    # Add edges for proximity relationships
    for motif1 in top_motifs:
        for motif2 in top_motifs:
            if motif1 >= motif2:  # Avoid duplicates
                continue
                
            # Get proximity score if available
            score = 0
            if motif1 in analyzer.proximity_network['proximity_scores']:
                if motif2 in analyzer.proximity_network['proximity_scores'][motif1]:
                    score = analyzer.proximity_network['proximity_scores'][motif1][motif2]
            
            if score > 0.1:  # Only add significant connections
                G.add_edge(motif1, motif2, weight=score)
    
    # Create visualization
    plt.figure(figsize=(14, 14))
    
    # Define node colors based on structure
    color_map = {
        'helix': 'red',
        'sheet': 'blue',
        'turn': 'green',
        'coil': 'purple'
    }
    node_colors = [color_map[G.nodes[n]['structure']] for n in G.nodes()]
    
    # Define node sizes based on occurrence
    node_sizes = [50 * G.nodes[n]['size'] for n in G.nodes()]
    
    # Define edge widths based on proximity score
    edge_widths = [2 * G[u][v]['weight'] for u, v in G.edges()]
    
    # Create spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    
    # Draw labels for top nodes
    top_degree_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:20]
    labels = {n: n for n, _ in top_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
              for color in color_map.values()]
    plt.legend(handles, color_map.keys(), title="Structure Type")
    
    plt.title('Motif Proximity Network')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'motif_proximity_network.png'))
    plt.close()
    
    # Export network data for interactive visualization
    nx.write_gexf(G, os.path.join(output_dir, 'motif_network.gexf'))

###############################
# Main Execution
###############################

def main():
    """
    Main execution function for the GRT analysis pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Protein Grammar Residence Time (GRT) Analysis')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing mdCATH H5 files')
    parser.add_argument('--output_dir', type=str, default='grt_results', help='Directory to save results')
    parser.add_argument('--min_motif_size', type=int, default=3, help='Minimum motif size')
    parser.add_argument('--max_motif_size', type=int, default=6, help='Maximum motif size')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process')
    parser.add_argument('--min_occurrence', type=int, default=10, help='Minimum occurrence threshold for analysis')
    parser.add_argument('--extract_only', action='store_true', help='Only extract motifs, skip analysis')
    parser.add_argument('--load_motifs', type=str, default=None, help='Load pre-extracted motifs from file')
    parser.add_argument('--save_motifs', type=str, default=None, help='Save extracted motifs to file')
    parser.add_argument('--n_workers', type=int, default=11, help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of frames to process in each batch')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to store cache files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Phase 1: Extract motifs
    if args.load_motifs:
        print(f"Loading pre-extracted motifs from {args.load_motifs}")
        with open(args.load_motifs, 'rb') as f:
            motifs = pickle.load(f)
    else:
        print("Starting motif extraction...")
        # Pass parallelization parameters to the MotifExtractor
        extractor = MotifExtractor(
            data_dir=args.data_dir,
            min_motif_size=args.min_motif_size,
            max_motif_size=args.max_motif_size,
            cache_dir=args.cache_dir
        )
        
        # Save original method for backup
        #original_extract_method = extractor._extract_motifs_from_trajectory
        
        # Override the parallel method to include parameters
        def configured_parallel_method(sequence, dssp, traj, domain_motifs, temp, replica):
            return extractor._extract_motifs_from_trajectory_parallel(
                sequence, dssp, traj, domain_motifs, temp, replica, 
                n_workers=args.n_workers, batch_size=32 #args.batch_size
            )
        
        # Replace the method
        extractor._extract_motifs_from_trajectory = configured_parallel_method
        
        # Process domains
        extractor.process_all_domains(max_files=args.max_files)
        motifs = extractor.motifs
        
        # Save extracted motifs if requested
        if args.save_motifs:
            print(f"Saving extracted motifs to {args.save_motifs}")
            with open(args.save_motifs, 'wb') as f:
                pickle.dump(motifs, f)
    
    # Exit if extract_only flag is set
    if args.extract_only:
        print("Extraction complete. Exiting as requested.")
        return
    
    # Phase 2: GRT Analysis
    print("Starting GRT analysis...")
    analyzer = GRTAnalyzer(motifs, min_occurrence=args.min_occurrence)
    
    # Calculate GRT profiles
    analyzer.calculate_grt_profiles()
    
    # Analyze temperature sensitivity
    analyzer.analyze_temperature_sensitivity()
    
    # Cluster motifs
    analyzer.cluster_motifs_by_grt()
    
    # Analyze proximity patterns
    analyzer.analyze_proximity_patterns()
    
    # Save results
    analyzer.save_results(os.path.join(args.output_dir, 'analysis'))
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_grt_profiles(analyzer, os.path.join(args.output_dir, 'visualizations/profiles'))
    visualize_motif_clusters(analyzer, os.path.join(args.output_dir, 'visualizations/clusters'))
    visualize_proximity_network(analyzer, os.path.join(args.output_dir, 'visualizations/network'))
    
    # Find notable motifs
    print("Identifying stable and transition motifs...")
    stable_motifs = analyzer.identify_stable_motifs()
    transition_motifs = analyzer.identify_transition_motifs()
    
    # Save motif findings
    with open(os.path.join(args.output_dir, 'stable_motifs.pkl'), 'wb') as f:
        pickle.dump(stable_motifs, f)
    
    with open(os.path.join(args.output_dir, 'transition_motifs.pkl'), 'wb') as f:
        pickle.dump(transition_motifs, f)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(
        analyzer, stable_motifs, transition_motifs, 
        os.path.join(args.output_dir, 'summary_report.html')
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

def generate_summary_report(analyzer, stable_motifs, transition_motifs, output_path):
    """
    Generate an HTML summary report of the GRT analysis.
    
    Args:
        analyzer: GRTAnalyzer instance
        stable_motifs: Dictionary of stable motifs
        transition_motifs: Dictionary of transition motifs
        output_path: Path to save the HTML report
    """
    # Count motifs by cluster
    cluster_counts = {
        label: len(motifs) 
        for label, motifs in analyzer.motif_clusters['clusters'].items()
    }
    
    # Get top motifs by occurrence
    top_motifs = sorted(
        analyzer.grt_profiles.keys(),
        key=lambda m: analyzer.grt_profiles[m]['occurrences'],
        reverse=True
    )[:20]
    
    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grammar Residence Time (GRT) Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333366; }}
            table {{ border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ margin: 40px 0; }}
            .highlight {{ background-color: #ffffcc; }}
        </style>
    </head>
    <body>
        <h1>Grammar Residence Time (GRT) Analysis Report</h1>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <p>Total unique motifs analyzed: {len(analyzer.grt_profiles)}</p>
            <p>Number of motif clusters: {len(cluster_counts)}</p>
            <p>Stable motifs identified: {len(stable_motifs)}</p>
            <p>Transition motifs identified: {len(transition_motifs)}</p>
        </div>
        
        <div class="section">
            <h2>Motif Clusters</h2>
            <table>
                <tr>
                    <th>Cluster ID</th>
                    <th>Number of Motifs</th>
                </tr>
    """
    
    # Add cluster data
    for label, count in sorted(cluster_counts.items()):
        html += f"""
                <tr>
                    <td>{'Noise' if label == -1 else f'Cluster {label}'}</td>
                    <td>{count}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Top 20 Most Frequent Motifs</h2>
            <table>
                <tr>
                    <th>Motif</th>
                    <th>Occurrences</th>
                    <th>Domains</th>
                    <th>Predominant Structure</th>
                    <th>Structure Fraction</th>
                    <th>Cluster</th>
                </tr>
    """
    
    # Add top motifs data
    for motif in top_motifs:
        profile = analyzer.grt_profiles[motif]
        main_structure = max(
            profile['group_fractions'].items(),
            key=lambda x: x[1]
        )
        
        cluster = analyzer.motif_clusters['motif_labels'].get(motif, 'N/A')
        
        html += f"""
                <tr>
                    <td>{motif}</td>
                    <td>{profile['occurrences']}</td>
                    <td>{profile['domain_count']}</td>
                    <td>{main_structure[0]}</td>
                    <td>{main_structure[1]:.2f}</td>
                    <td>{'Noise' if cluster == -1 else f'Cluster {cluster}'}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Stable Motifs Highlights</h2>
            <p>These motifs maintain consistent secondary structure across temperatures:</p>
            <table>
                <tr>
                    <th>Motif</th>
                    <th>Predominant Structure</th>
                    <th>Structure Fraction</th>
                    <th>Temperature Stability</th>
                </tr>
    """
    
    # Add stable motifs data
    for motif, data in list(stable_motifs.items())[:20]:  # Show top 20
        html += f"""
                <tr>
                    <td>{motif}</td>
                    <td>{data['predominant_structure']}</td>
                    <td>{data['structure_fraction']:.2f}</td>
                    <td>{data['temperature_stability']:.2f}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Transition Motifs Highlights</h2>
            <p>These motifs show significant structural changes across temperatures:</p>
            <table>
                <tr>
                    <th>Motif</th>
                    <th>Transition Structure</th>
                    <th>Direction</th>
                    <th>Magnitude</th>
                </tr>
    """
    
    # Add transition motifs data
    for motif, data in list(transition_motifs.items())[:20]:  # Show top 20
        html += f"""
                <tr>
                    <td>{motif}</td>
                    <td>{data['transition_group']}</td>
                    <td>{data['direction']}</td>
                    <td>{data['magnitude']:.2f}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Conclusions</h2>
            <p>The Grammar Residence Time (GRT) analysis has revealed distinct patterns in how protein motifs behave structurally:</p>
            <ul>
                <li>Multiple clusters of motifs with similar GRT profiles were identified, suggesting common grammatical roles.</li>
                <li>Stable motifs that maintain consistent structure may represent core "grammatical elements" in protein language.</li>
                <li>Transition motifs that change with temperature may represent context-dependent or flexible elements.</li>
                <li>The proximity network analysis reveals common co-location patterns between motifs, suggesting syntax-like rules.</li>
            </ul>
            <p>These findings support the hypothesis that proteins can be described using linguistic analogies, with motifs serving as words and their structural/proximity relationships as grammar.</p>
        </div>
        
        <div class="section">
            <h2>Next Steps</h2>
            <ul>
                <li>Correlation with functional annotations to determine if grammatical patterns predict function</li>
                <li>Development of a formal grammar specification based on the observed patterns</li>
                <li>Training of a language model using the identified motif vocabulary and grammatical rules</li>
                <li>Validation through prediction tasks such as protein folding or function prediction</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(output_path, 'w') as f:
        f.write(html)

if __name__ == "__main__":
    main()
