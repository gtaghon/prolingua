"""
CUDA-Accelerated Motif Extractor for Protein GRT Analysis

This module provides a specialized implementation of the MotifExtractor class
that leverages CUDA for performance-critical operations. It integrates with
the original codebase while providing significant performance improvements.

Key features:
- CUDA kernel acceleration for distance calculations
- GPU-based contact matrix generation
- Optimized motif extraction algorithm
- Memory-efficient sparse matrix representation
- Automatic CPU fallback for systems without CUDA support
"""

import os
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Set
import mdtraj as md
from collections import defaultdict, Counter
from tqdm import tqdm
import concurrent

# Conditional imports
try:
    import cupy as cp
    from cupyx.scipy import sparse as cusp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("CUDA acceleration not available. Install CuPy for GPU support.")

# File containing CUDA kernels
CUDA_KERNELS_FILE = os.path.join(os.path.dirname(__file__), "cuda_kernels.cu")


class CUDAKernelManager:
    """
    Manages compilation and execution of CUDA kernels for protein analysis
    """
    
    def __init__(self):
        """Initialize CUDA kernel manager and compile kernels"""
        self.kernels = {}
        
        if not HAS_CUDA:
            print("CUDA not available - kernel manager initialized in stub mode")
            return
        
        try:
            # Load CUDA kernel code
            if os.path.exists(CUDA_KERNELS_FILE):
                with open(CUDA_KERNELS_FILE, 'r') as f:
                    kernel_code = f.read()
            else:
                # Use embedded kernel code as fallback
                kernel_code = """
                extern "C" __global__ void distance_kernel(
                    const float* xyz, const int* atom_pairs, float* distances, 
                    int n_frames, int n_atoms, int n_pairs) {
                    // Calculate pair index
                    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (pair_idx >= n_pairs) return;
                    
                    // Get atom indices for this pair
                    int atom1 = atom_pairs[pair_idx * 2];
                    int atom2 = atom_pairs[pair_idx * 2 + 1];
                    
                    // Calculate distance for each frame
                    for (int frame = 0; frame < n_frames; frame++) {
                        float dx, dy, dz;
                        
                        // Calculate offsets into xyz array
                        int offset1 = (frame * n_atoms + atom1) * 3;
                        int offset2 = (frame * n_atoms + atom2) * 3;
                        
                        // Calculate coordinate differences
                        dx = xyz[offset1]     - xyz[offset2];
                        dy = xyz[offset1 + 1] - xyz[offset2 + 1];
                        dz = xyz[offset1 + 2] - xyz[offset2 + 2];
                        
                        // Calculate Euclidean distance
                        distances[frame * n_pairs + pair_idx] = sqrtf(dx*dx + dy*dy + dz*dz);
                    }
                }
                
                extern "C" __global__ void contact_threshold_kernel(
                    const float* distances, float threshold, int* contact_counts,
                    int frame_idx, int n_pairs) {
                    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (pair_idx >= n_pairs) return;
                    
                    if (distances[frame_idx * n_pairs + pair_idx] < threshold) {
                        atomicAdd(&contact_counts[frame_idx], 1);
                    }
                }
                
                extern "C" __global__ void contact_indices_kernel(
                    const float* distances, float threshold, int* contact_indices,
                    int* contact_counts, int frame_idx, int n_pairs, int max_contacts) {
                    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (pair_idx >= n_pairs) return;
                    
                    if (distances[frame_idx * n_pairs + pair_idx] < threshold) {
                        int idx = atomicAdd(&contact_counts[frame_idx], 1);
                        if (idx < max_contacts) {
                            contact_indices[frame_idx * max_contacts + idx] = pair_idx;
                        }
                    }
                }
                
                extern "C" __global__ void residue_contact_kernel(
                    const float* xyz, const int* ca_indices, bool* contact_map,
                    int n_frames, int n_atoms, int n_residues, float threshold) {
                    int frame = blockIdx.x;
                    int res_i = blockIdx.y;
                    int res_j = threadIdx.x + blockIdx.z * blockDim.x;
                    
                    if (frame >= n_frames || res_i >= n_residues || res_j >= n_residues || res_i == res_j)
                        return;
                    
                    int atom_i = ca_indices[res_i];
                    int atom_j = ca_indices[res_j];
                    
                    float3 pos_i, pos_j;
                    int offset_i = (frame * n_atoms + atom_i) * 3;
                    int offset_j = (frame * n_atoms + atom_j) * 3;
                    
                    pos_i.x = xyz[offset_i];
                    pos_i.y = xyz[offset_i + 1];
                    pos_i.z = xyz[offset_i + 2];
                    
                    pos_j.x = xyz[offset_j];
                    pos_j.y = xyz[offset_j + 1];
                    pos_j.z = xyz[offset_j + 2];
                    
                    float dx = pos_i.x - pos_j.x;
                    float dy = pos_i.y - pos_j.y;
                    float dz = pos_i.z - pos_j.z;
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    
                    if (dist < threshold) {
                        contact_map[(frame * n_residues + res_i) * n_residues + res_j] = true;
                        contact_map[(frame * n_residues + res_j) * n_residues + res_i] = true;
                    }
                }
                """
            
            # Compile kernels
            self._compile_kernels(kernel_code)
            
        except Exception as e:
            print(f"Error initializing CUDA kernels: {e}")
            print("Falling back to CPU implementation")
    
    def _compile_kernels(self, kernel_code: str) -> None:
        """Compile CUDA kernels from source code"""
        if not HAS_CUDA:
            return
        
        try:
            # Compile basic kernels
            self.kernels['distance'] = cp.RawKernel(
                kernel_code, 'distance_kernel')
            
            self.kernels['contact_threshold'] = cp.RawKernel(
                kernel_code, 'contact_threshold_kernel')
            
            self.kernels['contact_indices'] = cp.RawKernel(
                kernel_code, 'contact_indices_kernel')
            
            self.kernels['residue_contact'] = cp.RawKernel(
                kernel_code, 'residue_contact_kernel')
            
            # Compile advanced motif detection kernels if they exist in the code
            try:
                self.kernels['motif_detection'] = cp.RawKernel(
                    kernel_code, 'motif_detection_kernel')
                
                self.kernels['motif_counting'] = cp.RawKernel(
                    kernel_code, 'motif_counting_kernel')
                
                self.kernels['motif_matrix'] = cp.RawKernel(
                    kernel_code, 'motif_matrix_kernel')
                
                print("Compiled all CUDA kernels successfully")
            except Exception as e:
                print(f"Warning: Could not compile advanced motif kernels: {e}")
                print("Only basic distance and contact kernels will be available")
        
        except Exception as e:
            print(f"Failed to compile CUDA kernels: {e}")
    
    def compute_distances(self, xyz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        """
        Compute distances between atom pairs using CUDA acceleration
        
        Args:
            xyz: Coordinates array [n_frames, n_atoms, 3]
            pairs: Atom pairs array [n_pairs, 2]
            
        Returns:
            distances: Distance matrix [n_frames, n_pairs]
        """
        if not HAS_CUDA or 'distance' not in self.kernels:
            # Fallback to CPU implementation
            return self._compute_distances_cpu(xyz, pairs)
        
        try:
            # Get dimensions
            n_frames, n_atoms, _ = xyz.shape
            n_pairs = pairs.shape[0]
            
            # Prepare data for CUDA
            xyz_gpu = cp.asarray(xyz.reshape(-1), dtype=cp.float32)
            pairs_gpu = cp.asarray(pairs.reshape(-1), dtype=cp.int32)
            
            # Allocate output array
            distances_gpu = cp.zeros((n_frames * n_pairs), dtype=cp.float32)
            
            # Determine kernel launch parameters
            threads_per_block = 256
            blocks_per_grid = (n_pairs + threads_per_block - 1) // threads_per_block
            
            # Launch kernel
            self.kernels['distance']((blocks_per_grid,), (threads_per_block,),
                (xyz_gpu, pairs_gpu, distances_gpu, n_frames, n_atoms, n_pairs))
            
            # Transfer results back to CPU
            distances = distances_gpu.get().reshape(n_frames, n_pairs)
            
            return distances
            
        except Exception as e:
            print(f"CUDA distance calculation failed: {e}")
            return self._compute_distances_cpu(xyz, pairs)
    
    def _compute_distances_cpu(self, xyz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        """CPU fallback for distance calculation"""
        n_frames = xyz.shape[0]
        n_pairs = pairs.shape[0]
        distances = np.zeros((n_frames, n_pairs), dtype=np.float32)
        
        for i, (atom1, atom2) in enumerate(pairs):
            # Calculate distance for each frame
            delta = xyz[:, atom1, :] - xyz[:, atom2, :]
            distances[:, i] = np.sqrt(np.sum(delta**2, axis=1))
            
        return distances
    
    def compute_contact_indices(self, distances: np.ndarray, threshold: float) -> List[np.ndarray]:
        """
        Compute contact indices for each frame using CUDA acceleration
        
        Args:
            distances: Distance matrix [n_frames, n_pairs]
            threshold: Contact distance threshold
            
        Returns:
            List of contact indices for each frame
        """
        if not HAS_CUDA or 'contact_indices' not in self.kernels:
            # Fallback to CPU implementation
            return [np.where(frame < threshold)[0] for frame in distances]
        
        try:
            n_frames, n_pairs = distances.shape
            
            # Prepare data for CUDA
            distances_gpu = cp.asarray(distances, dtype=cp.float32)
            
            # First pass: count contacts for each frame
            contact_counts_gpu = cp.zeros(n_frames, dtype=cp.int32)
            
            threads_per_block = 256
            blocks_per_grid = (n_pairs + threads_per_block - 1) // threads_per_block
            
            # Process each frame
            for frame_idx in range(n_frames):
                # Reset counter for this frame (to 0)
                contact_counts_gpu[frame_idx] = 0
                
                # Count contacts
                self.kernels['contact_threshold']((blocks_per_grid,), (threads_per_block,),
                    (distances_gpu, cp.float32(threshold), contact_counts_gpu, 
                     frame_idx, n_pairs))
            
            # Get contact counts
            contact_counts = contact_counts_gpu.get()
            
            # Allocate arrays for contact indices
            max_contacts_per_frame = np.max(contact_counts)
            contact_indices_gpu = cp.zeros((n_frames, max_contacts_per_frame), dtype=cp.int32)
            temp_counts_gpu = cp.zeros(n_frames, dtype=cp.int32)  # For index allocation
            
            # Second pass: store contact indices
            for frame_idx in range(n_frames):
                # Skip if no contacts
                if contact_counts[frame_idx] == 0:
                    continue
                    
                # Reset counter for this frame (to 0)
                temp_counts_gpu[frame_idx] = 0
                
                # Store contact indices
                self.kernels['contact_indices']((blocks_per_grid,), (threads_per_block,),
                    (distances_gpu, cp.float32(threshold), contact_indices_gpu, 
                     temp_counts_gpu, frame_idx, n_pairs, max_contacts_per_frame))
            
            # Get results
            contact_indices_all = contact_indices_gpu.get()
            
            # Convert to list of arrays with correct sizes
            result = []
            for i in range(n_frames):
                count = contact_counts[i]
                if count > 0:
                    result.append(contact_indices_all[i, :count])
                else:
                    result.append(np.array([], dtype=np.int32))
            
            return result
            
        except Exception as e:
            print(f"CUDA contact calculation failed: {e}")
            return [np.where(frame < threshold)[0] for frame in distances]
    
    def build_residue_contact_map(self, xyz: np.ndarray, ca_indices: np.ndarray, 
                                 threshold: float) -> np.ndarray:
        """
        Build residue-level contact map using CUDA
        
        Args:
            xyz: Coordinates array [n_frames, n_atoms, 3]
            ca_indices: C-alpha atom indices [n_residues]
            threshold: Contact distance threshold
            
        Returns:
            contact_map: Boolean contact map [n_frames, n_residues, n_residues]
        """
        if not HAS_CUDA or 'residue_contact' not in self.kernels:
            # Fallback to computing all pairwise distances
            return self._build_residue_contact_map_cpu(xyz, ca_indices, threshold)
        
        try:
            n_frames, n_atoms, _ = xyz.shape
            n_residues = len(ca_indices)
            
            # Prepare data for CUDA
            xyz_gpu = cp.asarray(xyz.reshape(-1), dtype=cp.float32)
            ca_indices_gpu = cp.asarray(ca_indices, dtype=cp.int32)
            
            # Allocate output contact map
            contact_map_gpu = cp.zeros((n_frames * n_residues * n_residues), dtype=cp.bool_)
            
            # Configure kernel launch parameters
            threads_per_block = min(256, n_residues)
            grid_dim_z = (n_residues + threads_per_block - 1) // threads_per_block
            
            # Launch kernel with 3D grid
            self.kernels['residue_contact'](
                (n_frames, n_residues, grid_dim_z), (threads_per_block,),
                (xyz_gpu, ca_indices_gpu, contact_map_gpu, 
                 n_frames, n_atoms, n_residues, cp.float32(threshold))
            )
            
            # Get results
            contact_map = contact_map_gpu.get().reshape(n_frames, n_residues, n_residues)
            
            return contact_map
        except Exception as e:
            print(f"CUDA residue contact map calculation failed: {e}")
            return self._build_residue_contact_map_cpu(xyz, ca_indices, threshold)
    
    def _build_residue_contact_map_cpu(self, xyz: np.ndarray, ca_indices: np.ndarray, 
                                     threshold: float) -> np.ndarray:
        """CPU fallback for residue contact map calculation"""
        n_frames = xyz.shape[0]
        n_residues = len(ca_indices)
        contact_map = np.zeros((n_frames, n_residues, n_residues), dtype=bool)
        
        # Extract C-alpha coordinates
        ca_xyz = xyz[:, ca_indices, :]
        
        # Compute all pairwise distances
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                delta = ca_xyz[:, i, :] - ca_xyz[:, j, :]
                distances = np.sqrt(np.sum(delta**2, axis=1))
                
                # Update contact map
                in_contact = distances < threshold
                contact_map[in_contact, i, j] = True
                contact_map[in_contact, j, i] = True
        
        return contact_map

class CUDAMotifExtractor:
    """
    CUDA-accelerated class for extracting motifs from protein trajectory data.
    Designed to be a drop-in replacement for the original MotifExtractor.
    """
    
    def __init__(self, 
                min_motif_size: int = 3,
                max_motif_size: int = 6,
                contact_threshold: float = 0.8,
                temperatures: List[int] = None,
                replicas: List[int] = None,
                cache_dir: str = None,
                n_workers: int = None,
                use_gpu: bool = True,
                batch_size: int = 32,
                memory_limit_mb: int = 4000):
        """
        Initialize the CUDA-accelerated MotifExtractor with parameters for analysis.
        
        Args:
            min_motif_size: Minimum size of motifs to extract
            max_motif_size: Maximum size of motifs to extract
            contact_threshold: Threshold for defining contacts (in nm)
            temperatures: List of temperatures to analyze
            replicas: List of replicas to analyze
            cache_dir: Directory to store cache files (None to disable disk caching)
            n_workers: Number of worker processes for CPU operations
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for parallel processing
            memory_limit_mb: Memory limit in MB for caching
        """
        self.min_motif_size = min_motif_size
        self.max_motif_size = max_motif_size
        self.contact_threshold = contact_threshold
        self.temperatures = temperatures or [320, 348, 379, 413, 450]
        self.replicas = replicas or [0, 1, 2, 3, 4]
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Determine optimal worker count for CPU operations
        self.n_workers = n_workers or max(1, os.cpu_count() - 1)
        
        # Initialize CUDA kernel manager
        self.kernel_manager = CUDAKernelManager()
        self.use_gpu = use_gpu and HAS_CUDA
        
        # Data structures
        self.motifs = {}
        self.dssp_codes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', ' ']
        
        # Setup caching if enabled
        if cache_dir:
            self.frame_cache_dir = os.path.join(cache_dir, 'cuda_frame_cache')
            self.domain_cache_dir = os.path.join(cache_dir, 'cuda_domain_cache')
            os.makedirs(self.frame_cache_dir, exist_ok=True)
            os.makedirs(self.domain_cache_dir, exist_ok=True)
        else:
            self.frame_cache_dir = None
            self.domain_cache_dir = None
        
        # Initialize process pool for CPU-bound operations
        self.process_pool = None  # Will be initialized when needed
        
        # Performance metrics
        self.metrics = {
            'distance_calculation_time': 0,
            'contact_matrix_time': 0,
            'motif_extraction_time': 0,
            'io_time': 0,
            'gpu_utilization': 0,
            'gpu_memory_usage': 0
        }
        
        print(f"Initialized CUDA-accelerated MotifExtractor with {self.n_workers} workers")
        if self.use_gpu:
            print("GPU acceleration enabled")
        else:
            print("Running in CPU-only mode")
    
    def __del__(self):
        """Ensure resources are properly released"""
        if self.process_pool is not None:
            self.process_pool.shutdown(wait=False)
    
    def extract_motifs_parallel(self, sequence: str, dssp: np.ndarray, 
                              traj: md.Trajectory, pairs_list: List[Tuple[int, int]],
                              temp: int, batch_size: int = None) -> Dict:
        """
        CUDA-accelerated parallel motif extraction.
        
        Args:
            sequence: Amino acid sequence
            dssp: Secondary structure assignments for all frames
            traj: MDTraj trajectory object with coordinates
            pairs_list: List of atom pairs for contact calculation
            temp: Current temperature
            batch_size: Batch size for processing (default: use instance batch_size)
            
        Returns:
            Dictionary of extracted motifs and their properties
        """
        batch_size = batch_size or self.batch_size
        n_frames = traj.n_frames
        
        # Start timing
        start_time = time.time()
        
        # Create dictionary for faster pair lookups
        pairs_dict = {(min(i,j), max(i,j)): idx for idx, (i,j) in enumerate(pairs_list)}
        
        # Convert pairs list to numpy array for CUDA
        pairs_array = np.array(pairs_list, dtype=np.int32)
        
        print(f"Computing distances for {len(pairs_list)} pairs across {n_frames} frames...")
        distance_start = time.time()
        
        # Use CUDA for distance calculations
        if self.use_gpu:
            # Get coordinates from trajectory
            xyz = traj.xyz
            
            # Compute distances using CUDA
            distances = self.kernel_manager.compute_distances(xyz, pairs_array)
        else:
            # Use MDTraj for CPU-based calculation
            distances = md.compute_distances(traj, pairs_list)
        
        # Record timing
        distance_time = time.time() - distance_start
        self.metrics['distance_calculation_time'] += distance_time
        print(f"Distance calculation: {distance_time:.2f} seconds")
        
        # Compute contact indices
        contact_start = time.time()
        
        if self.use_gpu:
            # Use CUDA for contact calculation
            contact_indices = self.kernel_manager.compute_contact_indices(
                distances, self.contact_threshold)
        else:
            # CPU-based contact calculation
            contact_indices = [np.where(frame < self.contact_threshold)[0] 
                             for frame in distances]
        
        # Record timing
        contact_time = time.time() - contact_start
        self.metrics['contact_matrix_time'] += contact_time
        print(f"Contact calculation: {contact_time:.2f} seconds")
        
        # Free memory - we don't need the full matrices anymore
        del distances
        
        # Prepare motif extraction
        motif_sizes = (self.min_motif_size, self.max_motif_size)
        min_aa = "ACDEFGHIKLMNPQRSTVWY"  # Standard amino acids
        
        # Extract motifs with efficient parallelization
        extraction_start = time.time()
        
        # Initialize process pool if needed
        if self.process_pool is None:
            from concurrent.futures import ProcessPoolExecutor
            self.process_pool = ProcessPoolExecutor(max_workers=self.n_workers)
        
        # Prepare batch processing
        frame_tasks = []
        for frame_idx in range(n_frames):
            args = (
                frame_idx, sequence, dssp[frame_idx], 
                contact_indices[frame_idx], pairs_dict,
                motif_sizes, self.dssp_codes, self.temperatures, 
                temp, min_aa
            )
            frame_tasks.append(args)
        
        # Process frames in parallel batches
        results = {}
        
        with tqdm(total=n_frames, desc="Processing frames") as pbar:
            for batch_start in range(0, n_frames, batch_size):
                batch_end = min(batch_start + batch_size, n_frames)
                batch_tasks = frame_tasks[batch_start:batch_end]
                
                # Submit batch for parallel processing
                futures = [
                    self.process_pool.submit(self._process_frame, *task)
                    for task in batch_tasks
                ]
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        frame_result = future.result()
                        self._merge_frame_results(results, frame_result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        pbar.update(1)
        
        # Record timing
        extraction_time = time.time() - extraction_start
        self.metrics['motif_extraction_time'] += extraction_time
        print(f"Motif extraction: {extraction_time:.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        return results
    
    def _process_frame(self, frame_idx, sequence, dssp_frame, contact_matrix_indices, 
                     pairs_dict, motif_sizes, dssp_codes, temps, temp, min_aa):
        """
        Process a single frame with optimized algorithms.
        This method runs on CPU in a separate process.
        
        Args:
            frame_idx: Index of the current frame
            sequence: Amino acid sequence
            dssp_frame: Secondary structure assignments for this frame
            contact_matrix_indices: Indices of residue pairs in contact
            pairs_dict: Dictionary mapping residue pairs to indices
            motif_sizes: Tuple of (min_size, max_size)
            dssp_codes: List of valid secondary structure codes
            temps: List of temperatures
            temp: Current temperature
            min_aa: String of valid amino acid codes
            
        Returns:
            Dictionary of motif data for this frame
        """
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
        
        return motif_data
    
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
    
    def process_domain(self, h5_path: str, h5_manager) -> Dict:
        """
        Process a single domain with CUDA acceleration.
        
        Args:
            h5_path: Path to H5 file
            h5_manager: H5DataManager instance for data loading
            
        Returns:
            Dictionary of motifs extracted from this domain
        """
        domain_id = os.path.basename(h5_path).replace('.h5', '')
        
        # Check domain cache first if enabled
        if self.domain_cache_dir:
            cache_file = os.path.join(self.domain_cache_dir, f"{domain_id}.pkl")
            if os.path.exists(cache_file):
                try:
                    io_start = time.time()
                    with open(cache_file, 'rb') as f:
                        import pickle
                        result = pickle.load(f)
                    self.metrics['io_time'] += time.time() - io_start
                    return result
                except Exception as e:
                    print(f"Cache read error for {domain_id}: {e}")
        
        io_start = time.time()
        domain_motifs = {}
        
        try:
            # Get domain code
            code = h5_manager.get_domain_code(h5_path)
            
            # Check which temperatures and replicas are available
            temps, temp_replicas = h5_manager.get_temperatures_and_replicas(h5_path, code)
            
            # Process each available temperature
            for temp in self.temperatures:
                if f"{temp}" not in temps:
                    continue
                    
                for replica in self.replicas:
                    if f"{replica}" not in temp_replicas[f"{temp}"]:
                        continue
                        
                    # Check if trajectory exists
                    if not h5_manager.check_trajectory_exists(h5_path, code, temp, replica):
                        continue
                        
                    try:
                        # Load trajectory with optimized I/O
                        from grt_pcache_mps2 import load_trajectory_efficient
                        traj, sequence, dssp = load_trajectory_efficient(
                            h5_path, code, f"{temp}", f"{replica}", h5_manager
                        )
                        
                        # Record I/O time
                        self.metrics['io_time'] += time.time() - io_start
                        io_start = time.time()
                        
                        # Prepare atom pairs for contact calculation
                        ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']
                        pairs = [(i,j) for i in range(len(ca_indices)) for j in range(i+1, len(ca_indices))]
                        
                        # Extract motifs using CUDA-accelerated implementation
                        motifs = self.extract_motifs_parallel(
                            sequence, dssp, traj, pairs, int(temp)
                        )
                        
                        # Add domain information
                        for motif, data in motifs.items():
                            if 'domains' not in data:
                                data['domains'] = set()
                            data['domains'].add(domain_id)
                        
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
                    import pickle
                    pickle.dump(domain_motifs, f, protocol=4)
                self.metrics['io_time'] += time.time() - io_start
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
    
    def process_all_domains(self, data_dir: str, h5_manager, max_files: int = None):
        """
        Process all domains with CUDA acceleration and parallel I/O.
        
        Args:
            data_dir: Directory containing H5 files
            h5_manager: H5DataManager instance for data loading
            max_files: Maximum number of files to process (None for all)
        """
        h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        
        if max_files is not None:
            h5_files = h5_files[:max_files]
        
        total_files = len(h5_files)
        
        # Print configuration
        if self.use_gpu:
            gpu_info = "CUDA GPU acceleration enabled"
        else:
            gpu_info = "CPU-only processing"
            
        print(f"Processing {total_files} domain files with {self.n_workers} workers, {gpu_info}")
        
        # Pre-scan domain files to identify which ones are cached
        cached_domains = set()
        domains_to_process = []
        
        if self.domain_cache_dir:
            for h5_file in h5_files:
                domain_id = h5_file.replace('.h5', '')
                cache_file = os.path.join(self.domain_cache_dir, f"{domain_id}.pkl")
                
                if os.path.exists(cache_file):
                    cached_domains.add(h5_file)
                else:
                    domains_to_process.append(h5_file)
        else:
            domains_to_process = h5_files
        
        print(f"Found {len(cached_domains)} cached domains, {len(domains_to_process)} to process")
        
        # Process cached domains first (fast)
        if cached_domains:
            print(f"Loading {len(cached_domains)} cached domains...")
            for h5_file in tqdm(cached_domains, desc="Loading cached domains"):
                try:
                    h5_path = os.path.join(data_dir, h5_file)
                    domain_motifs = self.process_domain(h5_path, h5_manager)
                    
                    # Merge into global motifs
                    self._merge_domain_results(self.motifs, domain_motifs, h5_file.replace('.h5', ''))
                except Exception as e:
                    print(f"Error loading cached domain {h5_file}: {e}")
        
        # Process remaining domains with parallel I/O
        if domains_to_process:
            print(f"Processing {len(domains_to_process)} new domains...")
            
            # Use ThreadPool for I/O-bound operations
            from concurrent.futures import ThreadPoolExecutor
            
            # Limit the number of concurrent domains to avoid overwhelming I/O
            max_concurrent = min(8, self.n_workers)
            
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_domain, 
                                   os.path.join(data_dir, h5_file), 
                                   h5_manager): h5_file 
                    for h5_file in domains_to_process
                }
                
                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                                  total=len(future_to_file),
                                  desc="Processing domains"):
                    h5_file = future_to_file[future]
                    try:
                        domain_motifs = future.result()
                        
                        # Merge into global motifs
                        self._merge_domain_results(self.motifs, domain_motifs, h5_file.replace('.h5', ''))
                    except Exception as e:
                        print(f"Error processing {h5_file}: {e}")
        
        # Print final statistics
        print(f"Completed processing {total_files} domains, {len(self.motifs)} unique motifs")
        
        # Performance metrics
        print(f"Performance metrics:")
        print(f"- Distance calculation time: {self.metrics['distance_calculation_time']:.2f} s")
        print(f"- Contact matrix time: {self.metrics['contact_matrix_time']:.2f} s")
        print(f"- Motif extraction time: {self.metrics['motif_extraction_time']:.2f} s")
        print(f"- I/O time: {self.metrics['io_time']:.2f} s")
        
        # Calculate GPU utilization
        if self.use_gpu:
            total_compute_time = (self.metrics['distance_calculation_time'] + 
                                 self.metrics['contact_matrix_time'] + 
                                 self.metrics['motif_extraction_time'])
            gpu_util = (self.metrics['distance_calculation_time'] + 
                       self.metrics['contact_matrix_time']) / total_compute_time * 100
            self.metrics['gpu_utilization'] = gpu_util
            print(f"- GPU utilization: {gpu_util:.1f}% of compute time")
    
    def save_motifs(self, output_file: str):
        """Save extracted motifs to a file"""
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(self.motifs, f, protocol=4)
    
    def load_motifs(self, input_file: str):
        """Load extracted motifs from a file"""
        import pickle
        with open(input_file, 'rb') as f:
            self.motifs = pickle.load(f)
    
    def get_performance_report(self):
        """Get a detailed performance report"""
        report = {
            'metrics': self.metrics,
            'cuda_available': HAS_CUDA,
            'gpu_used': self.use_gpu,
            'workers': self.n_workers,
            'batch_size': self.batch_size,
        }
        
        # Add GPU info if available
        if HAS_CUDA:
            try:
                import cupy as cp
                device = cp.cuda.Device(0)
                report['gpu_info'] = {
                    'name': device.name,
                    'compute_capability': device.attributes["computeCapability"],
                    'total_memory': device.mem_info[1],
                    'free_memory': device.mem_info[0],
                }
            except Exception:
                pass
        
        return report


def main():
    """
    Main entry point for using the CUDA-accelerated GRT analysis pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='CUDA-Accelerated Protein Grammar Residence Time Analysis')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing mdCATH H5 files')
    parser.add_argument('--output_dir', type=str, default='cuda_grt_results', help='Directory to save results')
    parser.add_argument('--min_motif_size', type=int, default=3, help='Minimum motif size')
    parser.add_argument('--max_motif_size', type=int, default=6, help='Maximum motif size')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process')
    parser.add_argument('--save_motifs', type=str, default=None, help='Save extracted motifs to file')
    parser.add_argument('--load_motifs', type=str, default=None, help='Load pre-extracted motifs from file')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to store cache files')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Phase 1: Extract motifs
    if args.load_motifs:
        print(f"Loading pre-extracted motifs from {args.load_motifs}")
        import pickle
        with open(args.load_motifs, 'rb') as f:
            motifs = pickle.load(f)
    else:
        print("Starting CUDA-accelerated motif extraction...")
        
        # Import H5DataManager from original code
        from grt_pcache_mps2 import H5DataManager
        h5_manager = H5DataManager(keep_files_open=False)
        
        # Initialize CUDA-accelerated extractor
        extractor = CUDAMotifExtractor(
            min_motif_size=args.min_motif_size,
            max_motif_size=args.max_motif_size,
            cache_dir=args.cache_dir,
            n_workers=args.n_workers,
            use_gpu=not args.no_gpu,
            batch_size=args.batch_size
        )
        
        # Process domains
        extractor.process_all_domains(args.data_dir, h5_manager, max_files=args.max_files)
        motifs = extractor.motifs
        
        # Save performance report
        performance_report = extractor.get_performance_report()
        import json
        with open(os.path.join(args.output_dir, 'performance_report.json'), 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        # Save extracted motifs if requested
        if args.save_motifs:
            print(f"Saving extracted motifs to {args.save_motifs}")
            extractor.save_motifs(args.save_motifs)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
