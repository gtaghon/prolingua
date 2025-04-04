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


from concurrent.futures import ThreadPoolExecutor
import os
import h5py
import numpy as np
import mdtraj as md
from collections import defaultdict, Counter
import pickle
import tempfile
import time
import hashlib


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

class GPUAccelerator:
    """
    Handles GPU acceleration for computationally intensive operations
    using Metal Performance Shaders on Apple Silicon
    """
    
    def __init__(self, use_gpu=True):
        """Initialize GPU accelerator"""
        self.use_gpu = use_gpu and HAS_MPS
        self.device = None
        
        if self.use_gpu:
            try:
                self.device = torch.device("mps")
                print(f"Using GPU acceleration with Metal Performance Shaders")
            except Exception as e:
                print(f"Failed to initialize MPS device: {e}")
                self.use_gpu = False
    
    def compute_distances(self, traj, pairs):
        """
        Calculate distances between atom pairs with GPU acceleration
        
        Args:
            traj: MDTraj trajectory object
            pairs: List of atom index pairs to calculate distances for
        
        Returns:
            Distances matrix with shape (n_frames, n_pairs)
        """
        if not self.use_gpu or self.device is None:
            # Fall back to mdtraj implementation
            return md.compute_distances(traj, pairs)
        
        try:
            # Convert inputs to torch tensors
            start_time = time.time()

            # Extract xyz coord array from trajectory
            xyz = traj.xyz
            
            # Get dimensions
            n_frames = xyz.shape[0]
            n_pairs = len(pairs)
            
            # Convert coordinates to tensor and move to GPU
            # Using float32 for better performance on Apple Silicon
            xyz_tensor = torch.tensor(xyz, dtype=torch.float32, device=self.device)
            
            # Prepare pair indices tensor
            pair_indices = torch.tensor(pairs, dtype=torch.long, device=self.device)
            
            # Allocate output tensor
            distances = torch.zeros((n_frames, n_pairs), dtype=torch.float32, device=self.device)
            
            # Batch processing to handle large trajectories
            batch_size = 5000  # Adjust based on available GPU memory
            
            for i in range(0, n_pairs, batch_size):
                end_idx = min(i + batch_size, n_pairs)
                batch_pairs = pair_indices[i:end_idx]
                
                # Extract coordinates for atom pairs
                atom1 = xyz_tensor[:, batch_pairs[:, 0]]
                atom2 = xyz_tensor[:, batch_pairs[:, 1]]
                
                # Calculate Euclidean distances
                # This is fully vectorized and runs efficiently on MPS
                batch_distances = torch.sqrt(torch.sum((atom1 - atom2) ** 2, dim=2))
                
                # Store in output tensor
                distances[:, i:end_idx] = batch_distances
            
            # Transfer back to CPU and convert to numpy
            cpu_distances = distances.cpu().numpy()
            
            print(f"GPU distance calculation: {time.time() - start_time:.3f} seconds")
            return cpu_distances
            
        except Exception as e:
            print(f"GPU acceleration failed: {e}. Falling back to CPU implementation.")
            return md.compute_distances(traj, pairs)
    
    def compute_contact_matrices(self, distances, threshold):
        """
        Calculate contact matrices from distances with GPU acceleration
        
        Args:
            distances: Distance matrix with shape (n_frames, n_pairs)
            threshold: Distance threshold for contacts
        
        Returns:
            Boolean contact matrix with shape (n_frames, n_pairs)
        """
        if not self.use_gpu or self.device is None:
            # CPU implementation
            return distances < threshold
        
        try:
            # Convert to tensor and move to GPU
            distances_tensor = torch.tensor(distances, dtype=torch.float32, device=self.device)
            
            # Calculate contacts
            contacts_tensor = distances_tensor < threshold
            
            # Transfer back to CPU
            cpu_contacts = contacts_tensor.cpu().numpy()
            
            return cpu_contacts
            
        except Exception as e:
            print(f"GPU contact calculation failed: {e}. Falling back to CPU.")
            return distances < threshold

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

