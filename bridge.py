import numpy as np
import tempfile
import ctypes
from collections import Counter
import timeit
import time

# Import PyObjC components for Metal integration
try:
    import objc
    from Foundation import NSURL
    import Metal
    import MetalPerformanceShaders as MPS
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    print("Warning: PyObjC and Metal frameworks not available. Using CPU fallback.")

class GPUAccelerator:
    """
    MPS-based GPU acceleration for motif extraction and analysis.
    """
    
    def __init__(self, use_gpu=True, debug=False):
        """
        Initialize the GPU accelerator with Metal Performance Shaders.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            debug: Enable debug messages
        """
        self.debug = debug
        self.use_gpu = use_gpu and HAS_METAL
        self.device = None
        self.command_queue = None
        self.library = None
        self.function_generate_motifs = None
        self.function_process_ss = None
        self.function_process_contacts = None
        self.pipeline_generate_motifs = None
        self.pipeline_process_ss = None
        self.pipeline_process_contacts = None
        
        # Constants
        self.MAX_MOTIF_SIZE = 10
        self.NUM_DSSP_CODES = 8
        self.NUM_AMINO_ACIDS = 20
        self.HASH_TABLE_SIZE = 1000003  # Prime number for hash table
        
        # Performance metrics
        self.metrics = {
            'kernel_execution_time': 0.0,
            'data_transfer_time': 0.0,
            'compilation_time': 0.0
        }
        
        if self.use_gpu:
            self._init_metal()
    
    def _init_metal(self):
        """Initialize Metal device, command queue and compile kernels."""
        try:
            start_time = timeit.default_timer()
            
            # Get default Metal device
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device is None:
                if self.debug:
                    print("No Metal device found. Falling back to CPU.")
                self.use_gpu = False
                return
                
            # Create command queue
            self.command_queue = self.device.newCommandQueue()
            
            # Load and compile Metal library
            metal_source_path = self._prepare_metal_source()
            
            # Create URL to the Metal source file
            source_url = NSURL.fileURLWithPath_(metal_source_path)
            
            # Load options
            compile_options = Metal.MTLCompileOptions.alloc().init()
            compile_options.setFastMathEnabled_(True)
            
            # Prepare for capturing errors
            error = objc.objc_object(c_void_p=None)
            
            # Load library
            self.library = self.device.newLibraryWithURL_options_error_(
                source_url, compile_options, error
            )
            
            if self.library is None:
                if self.debug:
                    print(f"Failed to compile Metal library: {error}")
                self.use_gpu = False
                return
            
            # Get kernel functions
            self.function_generate_motifs = self.library.newFunctionWithName_("generate_motifs")
            self.function_process_ss = self.library.newFunctionWithName_("process_ss_assignments")
            self.function_process_contacts = self.library.newFunctionWithName_("process_motif_contacts")
            
            # Create pipeline states
            self.pipeline_generate_motifs = self.device.newComputePipelineStateWithFunction_error_(
                self.function_generate_motifs, error
            )
            self.pipeline_process_ss = self.device.newComputePipelineStateWithFunction_error_(
                self.function_process_ss, error
            )
            self.pipeline_process_contacts = self.device.newComputePipelineStateWithFunction_error_(
                self.function_process_contacts, error
            )
            
            # Check for errors
            if (self.pipeline_generate_motifs is None or 
                self.pipeline_process_ss is None or 
                self.pipeline_process_contacts is None):
                if self.debug:
                    print(f"Failed to create compute pipeline: {error}")
                self.use_gpu = False
                return
                
            # Record compilation time
            self.metrics['compilation_time'] = timeit.default_timer() - start_time
            
            if self.debug:
                print(f"Metal setup complete in {self.metrics['compilation_time']:.3f} seconds")
                print(f"Using device: {self.device.name()}")
        
        except Exception as e:
            if self.debug:
                print(f"Error initializing Metal: {e}")
            self.use_gpu = False
    
    def _prepare_metal_source(self):
        """
        Write the Metal source code to a temporary file.
        
        Returns:
            str: Path to the Metal source file
        """
        # Metal source code for the kernels
        with open('motif.metal', 'r') as f:
            metal_source = f.read()
        
        # Create a temporary file to hold the Metal source
        with tempfile.NamedTemporaryFile(suffix=".metal", delete=False) as f:
            f.write(metal_source.encode('utf-8'))
            return f.name
    
    def compute_distances(self, traj, pairs_list):
        """
        GPU-accelerated distance calculation for atom pairs.
        
        Args:
            traj: MDTraj trajectory
            pairs_list: List of atom pairs to calculate distances for
            
        Returns:
            np.ndarray: Distance matrix
        """
        if not self.use_gpu:
            # Fall back to CPU computation
            return traj.compute_distances(pairs_list)
        
        start_time = timeit.default_timer()
        
        try:
            # Prepare data
            n_frames = traj.n_frames
            n_pairs = len(pairs_list)
            
            # Convert trajectory coordinates to the right format
            # MPS requires coordinates in a flattened array
            coords = traj.xyz.astype(np.float32)
            
            # Convert pairs list to numpy array
            pairs = np.array(pairs_list, dtype=np.uint32)
            
            # Create output array
            distances = np.zeros((n_frames, n_pairs), dtype=np.float32)
            
            # Create Metal buffers
            transfer_start = timeit.default_timer()
            
            coords_buffer = self.device.newBufferWithBytes_length_options_(
                coords.tobytes(), 
                coords.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            pairs_buffer = self.device.newBufferWithBytes_length_options_(
                pairs.tobytes(),
                pairs.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            distances_buffer = self.device.newBufferWithLength_options_(
                distances.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            params_buffer = self.device.newBufferWithBytes_length_options_(
                np.array([n_frames, n_pairs, traj.n_atoms], dtype=np.uint32).tobytes(),
                12,  # 3 uint32 values
                Metal.MTLResourceStorageModeShared
            )
            
            # Record data transfer time
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Create command buffer and encoder
            command_buffer = self.command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()
            
            # Set compute pipeline state
            compute_encoder.setComputePipelineState_(self.pipeline_distances)
            
            # Set buffers
            compute_encoder.setBuffer_offset_atIndex_(coords_buffer, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(pairs_buffer, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(distances_buffer, 0, 2)
            compute_encoder.setBuffer_offset_atIndex_(params_buffer, 0, 3)
            
            # Calculate grid and threadgroup sizes
            threads_per_threadgroup = min(
                self.pipeline_distances.maxTotalThreadsPerThreadgroup(),
                256
            )
            
            threadgroups = (n_frames * n_pairs + threads_per_threadgroup - 1) // threads_per_threadgroup
            
            # Dispatch threads
            compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSize.Make(threadgroups, 1, 1),
                Metal.MTLSize.Make(threads_per_threadgroup, 1, 1)
            )
            
            # End encoding and commit
            compute_encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            
            # Read back results
            transfer_start = timeit.default_timer()
            
            result_data = distances_buffer.contents()
            ctypes.memmove(
                distances.ctypes.data,
                result_data,
                distances.nbytes
            )
            
            # Record data transfer time
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Record total execution time
            self.metrics['kernel_execution_time'] += timeit.default_timer() - start_time
            
            return distances
            
        except Exception as e:
            if self.debug:
                print(f"Error in compute_distances: {e}")
            
            # Fall back to CPU
            return traj.compute_distances(pairs_list)
    
    def compute_contact_matrices(self, distances, threshold):
        """
        Compute contact matrices from distances.
        
        Args:
            distances: Distance matrix from compute_distances
            threshold: Contact distance threshold
            
        Returns:
            np.ndarray: Boolean contact matrices
        """
        if not self.use_gpu:
            # Fall back to CPU computation
            return distances < threshold
        
        start_time = timeit.default_timer()
        
        try:
            # Prepare data
            n_frames, n_pairs = distances.shape
            
            # Convert to correct data types
            distances = distances.astype(np.float32)
            threshold = np.float32(threshold)
            
            # Create output array (using uint8 to save memory)
            contacts = np.zeros((n_frames, n_pairs), dtype=np.uint8)
            
            # Create Metal buffers
            transfer_start = timeit.default_timer()
            
            distances_buffer = self.device.newBufferWithBytes_length_options_(
                distances.tobytes(),
                distances.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            contacts_buffer = self.device.newBufferWithLength_options_(
                contacts.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            params_buffer = self.device.newBufferWithBytes_length_options_(
                np.array([n_frames, n_pairs, np.float32(threshold)], dtype=np.float32).tobytes(),
                12,  # 3 values: 2 uint32 and 1 float32
                Metal.MTLResourceStorageModeShared
            )
            
            # Record data transfer time
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Create command buffer and encoder
            command_buffer = self.command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()
            
            # Set compute pipeline state
            compute_encoder.setComputePipelineState_(self.pipeline_contacts)
            
            # Set buffers
            compute_encoder.setBuffer_offset_atIndex_(distances_buffer, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(contacts_buffer, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)
            
            # Calculate grid and threadgroup sizes
            threads_per_threadgroup = min(
                self.pipeline_contacts.maxTotalThreadsPerThreadgroup(),
                256
            )
            
            threadgroups = (n_frames * n_pairs + threads_per_threadgroup - 1) // threads_per_threadgroup
            
            # Dispatch threads
            compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSize.Make(threadgroups, 1, 1),
                Metal.MTLSize.Make(threads_per_threadgroup, 1, 1)
            )
            
            # End encoding and commit
            compute_encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            
            # Read back results
            transfer_start = timeit.default_timer()
            
            result_data = contacts_buffer.contents()
            ctypes.memmove(
                contacts.ctypes.data,
                result_data,
                contacts.nbytes
            )
            
            # Record data transfer time
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Record total execution time
            self.metrics['kernel_execution_time'] += timeit.default_timer() - start_time
            
            # Convert to boolean array
            return contacts.astype(bool)
            
        except Exception as e:
            if self.debug:
                print(f"Error in compute_contact_matrices: {e}")
            
            # Fall back to CPU
            return distances < threshold
    
    def extract_motifs(self, sequence, dssp_frames, contact_matrices, 
                      min_motif_size, max_motif_size, temp_index, temperatures):
        """
        Extract motifs using GPU acceleration.
        
        Args:
            sequence: Amino acid sequence (string)
            dssp_frames: DSSP secondary structure assignments for all frames
            contact_matrices: Contact matrices for all frames
            min_motif_size: Minimum motif size
            max_motif_size: Maximum motif size
            temp_index: Index of current temperature in temperatures list
            temperatures: List of temperatures
            
        Returns:
            dict: Motif data
        """
        if not self.use_gpu:
            # Fall back to CPU implementation
            return self._extract_motifs_cpu(
                sequence, dssp_frames, contact_matrices,
                min_motif_size, max_motif_size, temp_index, temperatures
            )
        
        start_time = timeit.default_timer()
        
        try:
            # Convert string data to numerical representations
            seq_array, valid_mask, dssp_arrays = self._prepare_sequence_data(sequence, dssp_frames)
            
            # Get dimensions
            n_frames = len(dssp_frames)
            seq_len = len(sequence)
            
            # Prepare contact matrices for GPU
            # We use a compact representation to save memory
            contact_indices = self._prepare_contact_matrices(contact_matrices, seq_len)
            
            # Set maximum motifs per frame (for buffer allocation)
            max_motifs_per_frame = seq_len * (max_motif_size - min_motif_size + 1)
            
            # Prepare buffers for first kernel (generate_motifs)
            transfer_start = timeit.default_timer()
            
            # Create constant parameters
            motif_params = np.array([
                seq_len, min_motif_size, max_motif_size
            ], dtype=np.uint32)
            
            # Create buffers
            seq_buffer = self.device.newBufferWithBytes_length_options_(
                seq_array.tobytes(),
                seq_array.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            valid_mask_buffer = self.device.newBufferWithBytes_length_options_(
                valid_mask.tobytes(),
                valid_mask.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            motif_params_buffer = self.device.newBufferWithBytes_length_options_(
                motif_params.tobytes(),
                motif_params.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            # Allocate output buffers for generated motifs
            motifs_size = n_frames * max_motifs_per_frame * 4 * 4  # 4 uint32 fields per Motif struct
            motifs_buffer = self.device.newBufferWithLength_options_(
                motifs_size,
                Metal.MTLResourceStorageModeShared
            )
            
            # Initialize frameInfo buffer with zeros
            frame_info = np.zeros((n_frames, 3), dtype=np.uint32)  # frame_idx, seq_len, valid_motif_count
            frame_info_buffer = self.device.newBufferWithBytes_length_options_(
                frame_info.tobytes(),
                frame_info.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            # Record data transfer time
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Execute first kernel: generate_motifs
            command_buffer = self.command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()
            
            # Set pipeline and buffers
            compute_encoder.setComputePipelineState_(self.pipeline_generate_motifs)
            compute_encoder.setBuffer_offset_atIndex_(seq_buffer, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(valid_mask_buffer, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(motif_params_buffer, 0, 2)
            compute_encoder.setBuffer_offset_atIndex_(motifs_buffer, 0, 3)
            compute_encoder.setBuffer_offset_atIndex_(frame_info_buffer, 0, 4)
            
            # Calculate dispatch sizes
            thread_execution_width = self.pipeline_generate_motifs.threadExecutionWidth()
            max_threads = self.pipeline_generate_motifs.maxTotalThreadsPerThreadgroup()
            
            # We dispatch one thread per position in each frame
            threads_per_threadgroup = min(max_threads, seq_len)
            threadgroups_per_grid = n_frames
            
            # Dispatch
            compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSize.Make(threadgroups_per_grid, 1, 1),
                Metal.MTLSize.Make(threads_per_threadgroup, 1, 1)
            )
            
            compute_encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            
            # Read back frame info to see how many motifs were generated
            transfer_start = timeit.default_timer()
            new_frame_info = np.zeros((n_frames, 3), dtype=np.uint32)
            ctypes.memmove(
                new_frame_info.ctypes.data,
                frame_info_buffer.contents(),
                new_frame_info.nbytes
            )
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Prepare second kernel: process_ss_assignments
            transfer_start = timeit.default_timer()
            
            # Create buffers for DSSP data
            dssp_buffer = self.device.newBufferWithBytes_length_options_(
                dssp_arrays.tobytes(),
                dssp_arrays.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            # Hash table for SS counts (motif_hash -> [counts for each SS type])
            ss_counts_size = self.HASH_TABLE_SIZE * self.NUM_DSSP_CODES * 4  # uint32 for each counter
            ss_counts_buffer = self.device.newBufferWithLength_options_(
                ss_counts_size,
                Metal.MTLResourceStorageModeShared
            )
            
            # Initialize ss_counts buffer with zeros
            ss_counts_ptr = ss_counts_buffer.contents()
            ctypes.memset(ss_counts_ptr, 0, ss_counts_size)
            
            # Hash table for motif occurrences
            motif_occurrences_size = self.HASH_TABLE_SIZE * 4  # uint32 counter for each hash
            motif_occurrences_buffer = self.device.newBufferWithLength_options_(
                motif_occurrences_size,
                Metal.MTLResourceStorageModeShared
            )
            
            # Initialize occurrences buffer with zeros
            motif_occ_ptr = motif_occurrences_buffer.contents()
            ctypes.memset(motif_occ_ptr, 0, motif_occurrences_size)
            
            # Hash table for temperature-specific SS counts
            temp_ss_counts_size = len(temperatures) * self.HASH_TABLE_SIZE * self.NUM_DSSP_CODES * 4
            temp_ss_counts_buffer = self.device.newBufferWithLength_options_(
                temp_ss_counts_size,
                Metal.MTLResourceStorageModeShared
            )
            
            # Initialize temp_ss_counts buffer with zeros
            temp_ss_ptr = temp_ss_counts_buffer.contents()
            ctypes.memset(temp_ss_ptr, 0, temp_ss_counts_size)
            
            # Parameters for process_ss kernel
            ss_params = np.array([
                seq_len, temp_index, max_motifs_per_frame
            ], dtype=np.uint32)
            
            ss_params_buffer = self.device.newBufferWithBytes_length_options_(
                ss_params.tobytes(),
                ss_params.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            # Record data transfer time
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Execute second kernel: process_ss_assignments
            command_buffer = self.command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()
            
            # Set pipeline and buffers
            compute_encoder.setComputePipelineState_(self.pipeline_process_ss)
            compute_encoder.setBuffer_offset_atIndex_(motifs_buffer, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(frame_info_buffer, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(dssp_buffer, 0, 2)
            compute_encoder.setBuffer_offset_atIndex_(ss_counts_buffer, 0, 3)
            compute_encoder.setBuffer_offset_atIndex_(motif_occurrences_buffer, 0, 4)
            compute_encoder.setBuffer_offset_atIndex_(temp_ss_counts_buffer, 0, 5)
            compute_encoder.setBuffer_offset_atIndex_(ss_params_buffer, 0, 6)
            
            # Calculate total number of motifs for dispatching threads
            total_motifs = sum(new_frame_info[:, 2])
            
            # We dispatch one thread per motif
            threads_per_threadgroup = min(
                self.pipeline_process_ss.maxTotalThreadsPerThreadgroup(), 
                256
            )
            
            threadgroups = (n_frames * max_motifs_per_frame + threads_per_threadgroup - 1) // threads_per_threadgroup
            
            # Dispatch
            compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSize.Make(threadgroups, 1, 1),
                Metal.MTLSize.Make(threads_per_threadgroup, 1, 1)
            )
            
            compute_encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            
            # Prepare third kernel: process_motif_contacts
            transfer_start = timeit.default_timer()
            
            # Create compact contact matrices buffer
            contact_buffer = self.device.newBufferWithBytes_length_options_(
                contact_indices.tobytes(),
                contact_indices.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            # Create motif contacts hash table
            # This uses a hash table for (motif1, motif2) pairs
            contacts_size = self.HASH_TABLE_SIZE * 4  # uint32 counter for each hash
            motif_contacts_buffer = self.device.newBufferWithLength_options_(
                contacts_size,
                Metal.MTLResourceStorageModeShared
            )
            
            # Initialize contacts buffer with zeros
            contacts_ptr = motif_contacts_buffer.contents()
            ctypes.memset(contacts_ptr, 0, contacts_size)
            
            # Parameters for contacts kernel
            contact_params = np.array([
                max_motifs_per_frame, seq_len
            ], dtype=np.uint32)
            
            contact_params_buffer = self.device.newBufferWithBytes_length_options_(
                contact_params.tobytes(),
                contact_params.nbytes,
                Metal.MTLResourceStorageModeShared
            )
            
            # Record data transfer time
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Execute third kernel: process_motif_contacts
            command_buffer = self.command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()
            
            # Set pipeline and buffers
            compute_encoder.setComputePipelineState_(self.pipeline_process_contacts)
            compute_encoder.setBuffer_offset_atIndex_(motifs_buffer, 0, 0)
            compute_encoder.setBuffer_offset_atIndex_(frame_info_buffer, 0, 1)
            compute_encoder.setBuffer_offset_atIndex_(contact_buffer, 0, 2)
            compute_encoder.setBuffer_offset_atIndex_(motif_contacts_buffer, 0, 3)
            compute_encoder.setBuffer_offset_atIndex_(contact_params_buffer, 0, 4)
            
            # For contacts, we need to check each pair of motifs in each frame
            # This is more expensive, so we'll use a different dispatch strategy
            threads_per_threadgroup = min(
                self.pipeline_process_contacts.maxTotalThreadsPerThreadgroup(),
                256
            )
            
            # We dispatch n_frames * max_motifs_per_frame^2 threads (one for each potential motif pair)
            threadgroups = (n_frames * max_motifs_per_frame * max_motifs_per_frame + 
                         threads_per_threadgroup - 1) // threads_per_threadgroup
            
            # Dispatch
            compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSize.Make(threadgroups, 1, 1),
                Metal.MTLSize.Make(threads_per_threadgroup, 1, 1)
            )
            
            compute_encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()
            
            # Read back results
            transfer_start = timeit.default_timer()
            
            # Read SS counts
            ss_counts_np = np.zeros(self.HASH_TABLE_SIZE * self.NUM_DSSP_CODES, dtype=np.uint32)
            ctypes.memmove(
                ss_counts_np.ctypes.data,
                ss_counts_buffer.contents(),
                ss_counts_np.nbytes
            )
            
            # Read motif occurrences
            motif_occurrences_np = np.zeros(self.HASH_TABLE_SIZE, dtype=np.uint32)
            ctypes.memmove(
                motif_occurrences_np.ctypes.data,
                motif_occurrences_buffer.contents(),
                motif_occurrences_np.nbytes
            )
            
            # Read temperature-specific SS counts
            temp_ss_counts_np = np.zeros(len(temperatures) * self.HASH_TABLE_SIZE * self.NUM_DSSP_CODES, 
                                     dtype=np.uint32)
            ctypes.memmove(
                temp_ss_counts_np.ctypes.data,
                temp_ss_counts_buffer.contents(),
                temp_ss_counts_np.nbytes
            )
            
            # Read motif contacts
            motif_contacts_np = np.zeros(self.HASH_TABLE_SIZE, dtype=np.uint32)
            ctypes.memmove(
                motif_contacts_np.ctypes.data,
                motif_contacts_buffer.contents(),
                motif_contacts_np.nbytes
            )
            
            # Record data transfer time
            self.metrics['data_transfer_time'] += timeit.default_timer() - transfer_start
            
            # Convert results to Python data structures
            # We need to decode the hash tables back to meaningful data
            motif_data = self._process_kernel_results(
                motif_occurrences_np,
                ss_counts_np,
                temp_ss_counts_np,
                motif_contacts_np,
                sequence,
                temperatures,
                temp_index,
                min_motif_size,
                max_motif_size
            )
            
            # Record kernel execution time
            self.metrics['kernel_execution_time'] += timeit.default_timer() - start_time
            
            return motif_data
            
        except Exception as e:
            if self.debug:
                print(f"Error in extract_motifs: {e}")
                import traceback
                traceback.print_exc()
            
            # Fall back to CPU implementation
            return self._extract_motifs_cpu(
                sequence, dssp_frames, contact_matrices,
                min_motif_size, max_motif_size, temp_index, temperatures
            )
    
    def _process_kernel_results(self, motif_occurrences, ss_counts, temp_ss_counts, 
                              motif_contacts, sequence, temperatures, temp_index,
                              min_motif_size, max_motif_size):
        """
        Process raw kernel results into Python dictionary format.
        
        Args:
            motif_occurrences: Array of motif occurrence counts
            ss_counts: Array of secondary structure counts
            temp_ss_counts: Array of temperature-specific SS counts
            motif_contacts: Array of motif contact counts
            sequence: Original amino acid sequence
            temperatures: List of temperatures
            temp_index: Current temperature index
            min_motif_size: Minimum motif size
            max_motif_size: Maximum motif size
            
        Returns:
            dict: Processed motif data matching the original format
        """
        # Initialize result dictionary
        motif_data = {}
        
        # Decode motif hash back to sequence
        # This is approximate - we'll need to regenerate all possible motifs
        # and check which ones have non-zero counts
        for size in range(min_motif_size, max_motif_size + 1):
            for i in range(len(sequence) - size + 1):
                motif = sequence[i:i+size]
                
                # Skip invalid amino acids
                if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in motif):
                    continue
                
                # Compute hash for this motif
                motif_hash = self._compute_hash(motif)
                
                # Check if this motif has any occurrences
                if motif_occurrences[motif_hash] > 0:
                    # Create motif entry if it doesn't exist
                    if motif not in motif_data:
                        motif_data[motif] = {
                            'occurrences': int(motif_occurrences[motif_hash]),
                            'ss_counts': Counter(),
                            'by_temp': {t: {'frames': 0, 'ss_counts': Counter()} 
                                    for t in temperatures},
                            'contacts': Counter()
                        }
                    
                    # Add secondary structure counts
                    ss_offset = motif_hash * self.NUM_DSSP_CODES
                    for j in range(self.NUM_DSSP_CODES):
                        count = ss_counts[ss_offset + j]
                        if count > 0:
                            motif_data[motif]['ss_counts'][self._idx_to_dssp(j)] = int(count)
                    
                    # Add temperature-specific counts
                    temp_offset = temp_index * self.HASH_TABLE_SIZE * self.NUM_DSSP_CODES + motif_hash * self.NUM_DSSP_CODES
                    temp_count = 0
                    for j in range(self.NUM_DSSP_CODES):
                        count = temp_ss_counts[temp_offset + j]
                        if count > 0:
                            temp_count += count
                            motif_data[motif]['by_temp'][temperatures[temp_index]]['ss_counts'][self._idx_to_dssp(j)] = int(count)
                    
                    # Set the frame count for this temperature
                    motif_data[motif]['by_temp'][temperatures[temp_index]]['frames'] = temp_count // size
        
        # Process contacts (this is more complex as we need to decode both motifs)
        # For simplicity, we'll look for motifs that we already found
        for motif1 in motif_data:
            hash1 = self._compute_hash(motif1)
            
            for motif2 in motif_data:
                # Compute the hash for this pair
                pair_hash = (hash1 * 17 + self._compute_hash(motif2)) % self.HASH_TABLE_SIZE
                
                # Check if there are any contacts
                count = motif_contacts[pair_hash]
                if count > 0:
                    motif_data[motif1]['contacts'][motif2] = int(count)
        
        return motif_data
    
    def _compute_hash(self, motif):
        """
        Compute hash for a motif using the same algorithm as the Metal shader.
        
        Args:
            motif: Amino acid sequence
            
        Returns:
            int: Hash value
        """
        # Map amino acids to indices
        aa_to_idx = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        
        # Compute hash
        hash_val = 17
        for aa in motif:
            aa_idx = aa_to_idx.get(aa, 0)
            hash_val = hash_val * 31 + aa_idx
        
        return hash_val % self.HASH_TABLE_SIZE
    
    def _idx_to_dssp(self, idx):
        """Convert DSSP index to DSSP code."""
        dssp_codes = ['H', 'B', 'E', 'G', 'I', 'T', 'S', ' ']
        return dssp_codes[idx] if 0 <= idx < len(dssp_codes) else ' '
    
    def _prepare_sequence_data(self, sequence, dssp_frames):
        """
        Convert sequence and DSSP data to numeric arrays for GPU processing.
        
        Args:
            sequence: Amino acid sequence string
            dssp_frames: List of DSSP assignment strings for each frame
            
        Returns:
            tuple: (sequence array, valid mask, DSSP arrays)
        """
        # Create amino acid mapping
        aa_to_idx = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        
        # Create DSSP code mapping
        dssp_to_idx = {code: idx for idx, code in enumerate(['H', 'B', 'E', 'G', 'I', 'T', 'S', ' '])}
        
        # Convert sequence to integer array
        seq_array = np.array([aa_to_idx.get(aa, 255) for aa in sequence], dtype=np.uint32)
        
        # Create valid amino acid mask
        valid_mask = np.array([1 if aa in aa_to_idx else 0 for aa in sequence], dtype=np.uint32)
        
        # Convert DSSP frames to integer arrays
        dssp_arrays = np.zeros((len(dssp_frames), len(sequence)), dtype=np.uint32)
        
        for i, frame in enumerate(dssp_frames):
            for j, code in enumerate(frame):
                dssp_arrays[i, j] = dssp_to_idx.get(code, 7)  # Default to space (' ') if not found
        
        return seq_array, valid_mask, dssp_arrays
    
    def _prepare_contact_matrices(self, contact_matrices, seq_len):
        """
        Convert contact matrices to a compact representation for GPU processing.
        
        Args:
            contact_matrices: List of boolean contact matrices for each frame
            seq_len: Length of the sequence
            
        Returns:
            np.ndarray: Compact representation of contact matrices
        """
        n_frames = len(contact_matrices)
        
        # For a sequence of length N, we only need N*(N-1)/2 entries
        # (lower triangular part of the matrix, excluding diagonal)
        triangle_size = (seq_len * (seq_len - 1)) // 2
        
        # Initialize output array (1 for contact, 0 for no contact)
        contact_indices = np.zeros((n_frames, triangle_size), dtype=np.uint32)
        
        # Convert each frame
        for frame_idx, matrix in enumerate(contact_matrices):
            idx = 0
            for i in range(seq_len):
                for j in range(i):  # j < i (lower triangle)
                    if matrix[i, j]:
                        contact_indices[frame_idx, idx] = 1
                    idx += 1
        
        return contact_indices
    
    def _extract_motifs_cpu(self, sequence, dssp_frames, contact_matrices,
                          min_motif_size, max_motif_size, temp_index, temperatures):
        """
        CPU fallback implementation for motif extraction.
        This matches the functionality of the GPU version but runs on CPU.
        """
        print("Using CPU fallback for motif extraction")
        
        # Create result structure
        motif_data = {}
        
        # Get current temperature
        temp = temperatures[temp_index]
        
        # Process each frame
        for frame_idx, (dssp_frame, contact_matrix) in enumerate(zip(dssp_frames, contact_matrices)):
            # Generate all valid motifs for this frame
            for size in range(min_motif_size, max_motif_size + 1):
                for i in range(len(sequence) - size + 1):
                    # Extract motif
                    motif = sequence[i:i+size]
                    
                    # Skip invalid amino acids
                    if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in motif):
                        continue
                    
                    # Extract secondary structure for this motif
                    ss_assignment = dssp_frame[i:i+size]
                    
                    # Initialize motif data if needed
                    if motif not in motif_data:
                        motif_data[motif] = {
                            'occurrences': 0,
                            'ss_counts': Counter(),
                            'by_temp': {t: {'frames': 0, 'ss_counts': Counter()} 
                                      for t in temperatures},
                            'contacts': Counter()
                        }
                    
                    # Update counts
                    motif_data[motif]['occurrences'] += 1
                    motif_data[motif]['by_temp'][temp]['frames'] += 1
                    
                    # Count secondary structures
                    for ss in ss_assignment:
                        motif_data[motif]['ss_counts'][ss] += 1
                        motif_data[motif]['by_temp'][temp]['ss_counts'][ss] += 1
                    
                    # Check contacts with other motifs
                    for j in range(len(sequence) - size + 1):
                        if i == j:  # Skip self
                            continue
                        
                        other_motif = sequence[j:j+size]
                        
                        # Skip invalid amino acids in other motif
                        if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in other_motif):
                            continue
                        
                        # Check for contact
                        has_contact = False
                        for a in range(i, i+size):
                            if has_contact:
                                break
                            for b in range(j, j+size):
                                if contact_matrix[a, b]:
                                    has_contact = True
                                    break
                        
                        if has_contact:
                            motif_data[motif]['contacts'][other_motif] += 1
        
        return motif_data

class GPUAcceleratedMotifExtractor:
    """
    Integration class to use MPS-based GPU acceleration with the MotifExtractor.
    This class extends the functionality of the MotifExtractor by optimizing
    the motif extraction pipeline for GPU execution.
    """
    
    def __init__(self, extractor, debug=False):
        """
        Initialize the GPU-accelerated extractor.
        
        Args:
            extractor: The parent MotifExtractor instance
            debug: Enable debug output
        """
        self.extractor = extractor
        self.gpu = extractor.gpu
        self.debug = debug
        
        # Alias important parameters for convenience
        self.min_motif_size = extractor.min_motif_size
        self.max_motif_size = extractor.max_motif_size
        self.temperatures = extractor.temperatures
        self.contact_threshold = extractor.contact_threshold
        self.frame_cache = extractor.frame_cache
        
        # Track performance metrics
        self.metrics = {
            'total_extraction_time': 0.0,
            'distance_calculation_time': 0.0,
            'motif_extraction_time': 0.0,
            'data_preparation_time': 0.0,
            'result_processing_time': 0.0,
            'frames_processed': 0
        }
    
    def extract_motifs_parallel(self, sequence, dssp, traj, pairs_list, temp, batch_size=None):
        """
        GPU-accelerated parallel motif extraction that replaces the original method.
        
        Args:
            sequence: Amino acid sequence
            dssp: Secondary structure assignments for all frames
            traj: MDTraj trajectory object with coordinates
            pairs_list: List of atom pairs for contact calculation
            temp: Current temperature
            batch_size: Batch size for processing (ignored in GPU version)
            
        Returns:
            dict: Extracted motif data
        """
        n_frames = traj.n_frames
        temp_index = self.temperatures.index(temp) if temp in self.temperatures else 0
        
        # Start timer for full extraction process
        start_time = time.time()
        
        # ===== Step 1: Calculate distances using GPU =====
        print(f"Calculating distances for {len(pairs_list)} pairs across {n_frames} frames...")
        distance_start = time.time()
        
        # Use GPU accelerator for distance calculations
        distances = self.gpu.compute_distances(traj, pairs_list)
        
        # Calculate contact matrices
        contact_matrices = self.gpu.compute_contact_matrices(distances, self.contact_threshold)
        
        # Record timing
        distance_time = time.time() - distance_start
        self.metrics['distance_calculation_time'] += distance_time
        self.metrics['frames_processed'] += n_frames
        
        print(f"Distance calculation: {distance_time:.2f} seconds, " +
              f"{'GPU' if self.gpu.use_gpu else 'CPU'} accelerated")
        
        # Free memory for distances as we only need contact matrices now
        del distances
        
        # ===== Step 2: Extract motifs using GPU =====
        motif_start = time.time()
        
        # Use our GPU accelerator for motif extraction
        motifs = self.gpu.extract_motifs(
            sequence, 
            dssp, 
            contact_matrices, 
            self.min_motif_size, 
            self.max_motif_size,
            temp_index,
            self.temperatures
        )
        
        # Record timing
        motif_time = time.time() - motif_start
        self.metrics['motif_extraction_time'] += motif_time
        
        print(f"Motif extraction: {motif_time:.2f} seconds")
        
        # Total time
        total_time = time.time() - start_time
        self.metrics['total_extraction_time'] += total_time
        
        print(f"Total extraction time: {total_time:.2f} seconds for {n_frames} frames")
        
        # Calculate frames per second
        fps = n_frames / total_time
        print(f"Performance: {fps:.2f} frames/second")
        
        return motifs
    
    def print_performance_report(self):
        """Print detailed performance metrics for the GPU acceleration."""
        if self.metrics['frames_processed'] == 0:
            print("No frames processed yet.")
            return
        
        print("\n===== GPU Acceleration Performance Report =====")
        print(f"Total frames processed: {self.metrics['frames_processed']}")
        print(f"Total extraction time: {self.metrics['total_extraction_time']:.2f} seconds")
        
        fps = self.metrics['frames_processed'] / self.metrics['total_extraction_time']
        print(f"Overall performance: {fps:.2f} frames/second")
        
        # Show breakdown
        print("\nTime breakdown:")
        print(f"- Distance calculation: {self.metrics['distance_calculation_time']:.2f}s " +
              f"({100 * self.metrics['distance_calculation_time'] / self.metrics['total_extraction_time']:.1f}%)")
        print(f"- Motif extraction: {self.metrics['motif_extraction_time']:.2f}s " +
              f"({100 * self.metrics['motif_extraction_time'] / self.metrics['total_extraction_time']:.1f}%)")
        
        # Add GPU accelerator metrics
        if hasattr(self.gpu, 'metrics'):
            print("\nGPU accelerator metrics:")
            print(f"- Kernel execution time: {self.gpu.metrics.get('kernel_execution_time', 0):.2f}s")
            print(f"- Data transfer time: {self.gpu.metrics.get('data_transfer_time', 0):.2f}s")
            print(f"- Compilation time: {self.gpu.metrics.get('compilation_time', 0):.2f}s")

# Function to enhance the original MotifExtractor with GPU acceleration
def enhance_with_gpu_acceleration(extractor):
    """
    Enhance an existing MotifExtractor with GPU acceleration.
    
    Args:
        extractor: The MotifExtractor instance to enhance
        
    Returns:
        The enhanced extractor
    """
    # Check if the extractor already has a gpu attribute
    if not hasattr(extractor, 'gpu') or extractor.gpu is None:
        # Create a new GPUAccelerator instance
        extractor.gpu = GPUAccelerator(use_gpu=extractor.use_gpu, debug=True)
        
        # Check if GPU initialization succeeded
        if not extractor.gpu.use_gpu:
            print("Warning: GPU acceleration failed to initialize. Falling back to CPU.")
    
    # Create the GPU accelerator wrapper
    gpu_extractor = GPUAcceleratedMotifExtractor(extractor)
    
    # Replace the extract_motifs_parallel method with the GPU-accelerated version
    original_method = extractor.extract_motifs_parallel
    
    # Store the original method for fallback
    extractor._original_extract_motifs_parallel = original_method
    
    # Replace with GPU-accelerated version
    extractor.extract_motifs_parallel = gpu_extractor.extract_motifs_parallel
    
    # Add performance reporting method
    extractor.print_gpu_performance = gpu_extractor.print_performance_report
    
    return extractor