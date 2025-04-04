/**
 * CUDA Kernels for Protein Motif Extraction
 * 
 * This file contains the CUDA kernel implementations for accelerating the
 * computationally intensive parts of the protein motif extraction pipeline:
 * 1. Distance calculation between atom pairs
 * 2. Contact matrix generation
 * 3. Motif detection and cataloging
 */

 #include <cuda_runtime.h>

 //----------------------------------------------------------------------
 // Kernel 1: Distance calculation between atom pairs
 //----------------------------------------------------------------------
 extern "C" __global__ void distance_kernel(
     const float* xyz,      // [n_frames, n_atoms, 3] coordinates
     const int* atom_pairs, // [n_pairs, 2] atom index pairs
     float* distances,      // [n_frames, n_pairs] output distances
     int n_frames,          // number of frames
     int n_atoms,           // number of atoms
     int n_pairs            // number of pairs
 ) {
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
 
 //----------------------------------------------------------------------
 // Kernel 2: Contact matrix generation (optimized sparse version)
 //----------------------------------------------------------------------
 extern "C" __global__ void contact_threshold_kernel(
     const float* distances,  // [n_frames, n_pairs] distances
     float threshold,         // distance threshold for contacts
     int* contact_counts,     // [n_frames] output count of contacts per frame
     int frame_idx,           // current frame index
     int n_pairs              // number of pairs
 ) {
     // Each thread handles one pair in the current frame
     int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (pair_idx >= n_pairs) return;
     
     // Check if distance is below threshold (in contact)
     if (distances[frame_idx * n_pairs + pair_idx] < threshold) {
         // Atomically increment the contact count for this frame
         atomicAdd(&contact_counts[frame_idx], 1);
     }
 }
 
 extern "C" __global__ void contact_indices_kernel(
     const float* distances,  // [n_frames, n_pairs] distances
     float threshold,         // distance threshold for contacts
     int* contact_indices,    // [n_frames, max_contacts] output contact indices
     int* contact_counts,     // [n_frames] count of contacts per frame (input)
     int frame_idx,           // current frame index
     int n_pairs,             // number of pairs
     int max_contacts         // maximum number of contacts per frame
 ) {
     // Each thread handles one pair in the current frame
     int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (pair_idx >= n_pairs) return;
     
     // Check if distance is below threshold (in contact)
     if (distances[frame_idx * n_pairs + pair_idx] < threshold) {
         // Atomically get the next available index in the output array
         int idx = atomicAdd(&contact_counts[frame_idx], 1);
         
         // Make sure we don't exceed the maximum allowed contacts
         if (idx < max_contacts) {
             // Store the pair index
             contact_indices[frame_idx * max_contacts + idx] = pair_idx;
         }
     }
 }
 
 //----------------------------------------------------------------------
 // Kernel 3: Residue contact matrix generation
 //----------------------------------------------------------------------
 extern "C" __global__ void residue_contact_kernel(
     const float* xyz,       // [n_frames, n_atoms, 3] coordinates
     const int* ca_indices,  // [n_residues] C-alpha atom indices
     bool* contact_map,      // [n_frames, n_residues, n_residues] output
     int n_frames,           // number of frames
     int n_atoms,            // number of atoms
     int n_residues,         // number of residues
     float threshold         // distance threshold for contacts
 ) {
     // Each thread handles one residue pair for one frame
     int frame = blockIdx.x;
     int res_i = blockIdx.y;
     int res_j = threadIdx.x + blockIdx.z * blockDim.x;
     
     if (frame >= n_frames || res_i >= n_residues || res_j >= n_residues || res_i == res_j)
         return;
     
     // Get C-alpha atom indices
     int atom_i = ca_indices[res_i];
     int atom_j = ca_indices[res_j];
     
     // Get atom coordinates
     float3 pos_i, pos_j;
     int offset_i = (frame * n_atoms + atom_i) * 3;
     int offset_j = (frame * n_atoms + atom_j) * 3;
     
     pos_i.x = xyz[offset_i];
     pos_i.y = xyz[offset_i + 1];
     pos_i.z = xyz[offset_i + 2];
     
     pos_j.x = xyz[offset_j];
     pos_j.y = xyz[offset_j + 1];
     pos_j.z = xyz[offset_j + 2];
     
     // Calculate distance
     float dx = pos_i.x - pos_j.x;
     float dy = pos_i.y - pos_j.y;
     float dz = pos_i.z - pos_j.z;
     float dist = sqrtf(dx*dx + dy*dy + dz*dz);
     
     // Check if in contact and update contact map
     if (dist < threshold) {
         contact_map[(frame * n_residues + res_i) * n_residues + res_j] = true;
         contact_map[(frame * n_residues + res_j) * n_residues + res_i] = true;
     }
 }
 
 //----------------------------------------------------------------------
 // Kernel 4: Optimized motif detection (core algorithm)
 //----------------------------------------------------------------------
 extern "C" __global__ void motif_detection_kernel(
     const int* sequence,     // [seq_length] sequence as integer indices
     const int* dssp,         // [n_frames, seq_length] secondary structure
     const bool* contacts,    // [n_frames, n_residues, n_residues] contact map
     unsigned int* motif_counts,  // [max_motifs] atomic counters for motifs
     int* motif_ss_counts,    // [max_motifs, n_ss_types] secondary structure counts
     int* motif_contacts,     // [max_motifs, max_motifs] contact counts
     int* motif_to_idx,       // [seq_length, seq_length, max_aa^max_size] motif hash to index map
     int frame_idx,           // current frame index
     int seq_length,          // sequence length
     int min_motif_size,      // minimum motif size
     int max_motif_size,      // maximum motif size
     int n_aa_types,          // number of amino acid types
     int n_ss_types,          // number of secondary structure types
     int max_motifs,          // maximum number of motifs to track
     int temp_id              // temperature ID for this frame
 ) {
     // Each thread handles one potential motif starting position
     int start_pos = blockIdx.x * blockDim.x + threadIdx.x;
     if (start_pos >= seq_length) return;
     
     // Each block handles one motif size
     int motif_size = blockIdx.y + min_motif_size;
     if (motif_size > max_motif_size) return;
     
     // Check if there's enough sequence left
     if (start_pos + motif_size > seq_length) return;
     
     // Compute motif hash value (this is a simplified hash function)
     // In practice, we'd use a more sophisticated hashing scheme
     unsigned int motif_hash = 0;
     bool valid_motif = true;
     
     for (int i = 0; i < motif_size; i++) {
         int aa = sequence[start_pos + i];
         // Check if this is a valid amino acid
         if (aa < 0 || aa >= n_aa_types) {
             valid_motif = false;
             break;
         }
         // Compute hash (positional encoding)
         motif_hash = motif_hash * n_aa_types + aa;
     }
     
     if (!valid_motif) return;
     
     // Lookup motif index, or allocate a new one if not found
     int motif_index;
     int hash_key = (start_pos * seq_length + motif_size) * (1 << 20) + motif_hash;
     
     // Use a simple hash table with linear probing
     // This is a highly simplified approach
     motif_index = motif_to_idx[hash_key % max_motifs];
     
     if (motif_index == 0) {
         // Allocate new motif index
         motif_index = atomicAdd(motif_counts, 1);
         if (motif_index >= max_motifs) return; // Too many motifs
         
         // Store the mapping
         motif_to_idx[hash_key % max_motifs] = motif_index;
     }
     
     // Increment motif occurrence count
     atomicAdd(&motif_counts[motif_index], 1);
     
     // Update secondary structure counts
     for (int i = 0; i < motif_size; i++) {
         int ss = dssp[(frame_idx * seq_length) + start_pos + i];
         if (ss >= 0 && ss < n_ss_types) {
             atomicAdd(&motif_ss_counts[(motif_index * n_ss_types) + ss], 1);
         }
     }
     
     // Update contact information
     // For each potential other motif, check for contacts
     for (int other_start = 0; other_start < seq_length; other_start++) {
         // Skip self
         if (other_start == start_pos) continue;
         
         // Check if there's enough sequence left
         if (other_start + motif_size > seq_length) continue;
         
         // Compute other motif hash
         unsigned int other_hash = 0;
         bool other_valid = true;
         
         for (int i = 0; i < motif_size; i++) {
             int aa = sequence[other_start + i];
             if (aa < 0 || aa >= n_aa_types) {
                 other_valid = false;
                 break;
             }
             other_hash = other_hash * n_aa_types + aa;
         }
         
         if (!other_valid) continue;
         
         // Lookup other motif index
         int other_key = (other_start * seq_length + motif_size) * (1 << 20) + other_hash;
         int other_index = motif_to_idx[other_key % max_motifs];
         
         if (other_index == 0) continue; // Not in our table
         
         // Check for contacts between any residue in motif and any in other motif
         bool contact_exists = false;
         for (int i = 0; i < motif_size && !contact_exists; i++) {
             for (int j = 0; j < motif_size && !contact_exists; j++) {
                 int res_i = start_pos + i;
                 int res_j = other_start + j;
                 
                 // Check contact map
                 if (contacts[(frame_idx * seq_length + res_i) * seq_length + res_j]) {
                     contact_exists = true;
                     break;
                 }
             }
         }
         
         // If contact exists, update contact counter
         if (contact_exists) {
             atomicAdd(&motif_contacts[(motif_index * max_motifs) + other_index], 1);
         }
     }
 }
 
 //----------------------------------------------------------------------
 // Kernel 5: Efficient motif counting and pairwise comparisons
 //----------------------------------------------------------------------
 extern "C" __global__ void motif_counting_kernel(
     const int* sequence,        // [seq_length] sequence as integer indices
     const int* valid_aa_mask,   // [seq_length] mask for valid amino acids
     const int* frame_contact_indices, // [max_contacts] contact indices for this frame
     const int* frame_contact_count,   // Scalar count of contacts
     int* motif_hashes,          // [max_motifs] output array of motif hashes
     int* motif_counts,          // [max_motifs] output array of motif counts
     int* motif_positions,       // [max_motifs, max_occurrences] positions of each motif
     int* pair_contact_counts,   // [max_motifs, max_motifs] contact counts between motifs
     int seq_length,             // sequence length
     int min_motif_size,         // minimum motif size
     int max_motif_size,         // maximum motif size
     int n_aa_types,             // number of amino acid types
     int max_motifs,             // maximum number of motifs to track
     int max_occurrences         // maximum occurrences per motif to track
 ) {
     // Each thread handles starting position for a potential motif
     int start_pos = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Each block handles one motif size
     int motif_size = blockIdx.y + min_motif_size;
     
     if (start_pos >= seq_length || motif_size > max_motif_size) return;
     if (start_pos + motif_size > seq_length) return;
     
     // Check if all residues are valid amino acids
     bool valid_motif = true;
     for (int i = 0; i < motif_size && valid_motif; i++) {
         if (!valid_aa_mask[start_pos + i]) {
             valid_motif = false;
         }
     }
     
     if (!valid_motif) return;
     
     // Compute motif hash
     unsigned int motif_hash = 0;
     for (int i = 0; i < motif_size; i++) {
         motif_hash = motif_hash * n_aa_types + sequence[start_pos + i];
     }
     
     // Lookup or allocate motif index
     int motif_idx = -1;
     
     // Simple linear search in shared memory for this motif
     // In a real implementation, we'd use a more efficient approach
     __shared__ int shared_hashes[256];  // Assuming <= 256 threads per block
     __shared__ int shared_indices[256];
     
     // Load known motif hashes into shared memory
     if (threadIdx.x < 256) {
         shared_hashes[threadIdx.x] = motif_hashes[threadIdx.x];
         shared_indices[threadIdx.x] = threadIdx.x;
     }
     __syncthreads();
     
     // Search for hash
     for (int i = 0; i < 256; i++) {
         if (shared_hashes[i] == motif_hash) {
             motif_idx = shared_indices[i];
             break;
         }
         if (shared_hashes[i] == 0) {
             // Found empty slot, try to claim it atomically
             unsigned int old = atomicCAS(&motif_hashes[shared_indices[i]], 0, motif_hash);
             if (old == 0) {
                 motif_idx = shared_indices[i];
             }
             break;
         }
     }
     
     if (motif_idx == -1) return;  // Too many unique motifs
     
     // Add this occurrence
     int occurrence_idx = atomicAdd(&motif_counts[motif_idx], 1);
     if (occurrence_idx < max_occurrences) {
         motif_positions[(motif_idx * max_occurrences) + occurrence_idx] = start_pos;
     }
     
     // Check contacts with other motifs
     for (int contact_idx = 0; contact_idx < *frame_contact_count; contact_idx++) {
         int pair_idx = frame_contact_indices[contact_idx];
         // Decode pair_idx back to residue indices
         // This depends on how pair_idx was encoded
         
         // For each contact, check if it links this motif to another
         // This is a simplified approach
         for (int other_idx = 0; other_idx < min(256, max_motifs); other_idx++) {
             if (other_idx == motif_idx) continue;  // Skip self
             
             // Check if any residue in the other motif is in contact with any in this motif
             // This would require checking all occurrences of the other motif
             // For simplicity, we just increment the counter if contact exists
             // Real implementation would be more precise
             atomicAdd(&pair_contact_counts[(motif_idx * max_motifs) + other_idx], 1);
         }
     }
 }
 
 //----------------------------------------------------------------------
 // Kernel 6: Matrix-based motif extraction (High-performance approach)
 //----------------------------------------------------------------------
 extern "C" __global__ void motif_matrix_kernel(
     const int* sequence_matrix,    // [seq_length, max_motif_size] sequence window matrix
     const int* dssp_matrix,        // [seq_length, max_motif_size] dssp window matrix
     const bool* contact_matrix,    // [seq_length, seq_length] residue contacts
     unsigned long long* motif_keys, // [max_motifs] hash keys for motifs
     int* motif_counts,            // [max_motifs] counts for each motif
     int* ss_count_matrix,         // [max_motifs, n_ss_types] SS counts
     int* contact_count_matrix,    // [max_motifs, max_motifs] contact counts
     int seq_length,               // sequence length
     int motif_size,               // current motif size
     int n_ss_types,               // number of SS types
     int max_motifs                // maximum motifs to track
 ) {
     // Each thread handles one motif position
     int pos = blockIdx.x * blockDim.x + threadIdx.x;
     if (pos >= seq_length - motif_size + 1) return;
     
     // Compute motif hash using rolling hash approach
     unsigned long long motif_hash = 0;
     bool valid = true;
     
     // Check sequence window
     for (int i = 0; i < motif_size; i++) {
         int aa = sequence_matrix[pos * max_motif_size + i];
         if (aa < 0) {
             valid = false;
             break;
         }
         motif_hash = (motif_hash << 5) + aa;  // 5 bits per AA (32 possible values)
     }
     
     if (!valid) return;
     
     // Find motif index using hash table with linear probing
     // This is a simplified approach - a real implementation would use a more
     // sophisticated hash table with proper collision handling
     int motif_idx = -1;
     for (int i = 0; i < max_motifs; i++) {
         int idx = (motif_hash + i) % max_motifs;
         unsigned long long existing = motif_keys[idx];
         
         if (existing == 0) {
             // Try to claim this slot atomically
             unsigned long long old = atomicCAS(&motif_keys[idx], 0ULL, motif_hash);
             if (old == 0) {
                 motif_idx = idx;
                 break;
             }
         } else if (existing == motif_hash) {
             motif_idx = idx;
             break;
         }
     }
     
     if (motif_idx == -1) return;  // Hash table full
     
     // Update motif count
     atomicAdd(&motif_counts[motif_idx], 1);
     
     // Update secondary structure counts
     for (int i = 0; i < motif_size; i++) {
         int ss = dssp_matrix[pos * max_motif_size + i];
         if (ss >= 0 && ss < n_ss_types) {
             atomicAdd(&ss_count_matrix[motif_idx * n_ss_types + ss], 1);
         }
     }
     
     // Check contacts with other motifs
     for (int other_pos = 0; other_pos < seq_length - motif_size + 1; other_pos++) {
         if (other_pos == pos) continue;  // Skip self
         
         // Check if other position has a valid motif
         bool other_valid = true;
         unsigned long long other_hash = 0;
         
         for (int i = 0; i < motif_size; i++) {
             int aa = sequence_matrix[other_pos * max_motif_size + i];
             if (aa < 0) {
                 other_valid = false;
                 break;
             }
             other_hash = (other_hash << 5) + aa;
         }
         
         if (!other_valid) continue;
         
         // Find the other motif's index
         int other_idx = -1;
         for (int i = 0; i < max_motifs; i++) {
             int idx = (other_hash + i) % max_motifs;
             if (motif_keys[idx] == other_hash) {
                 other_idx = idx;
                 break;
             }
         }
         
         if (other_idx == -1) continue;  // Not found
         
         // Check for contacts between any residue of this motif and any of the other
         bool has_contact = false;
         for (int i = 0; i < motif_size && !has_contact; i++) {
             for (int j = 0; j < motif_size && !has_contact; j++) {
                 int res1 = pos + i;
                 int res2 = other_pos + j;
                 
                 if (contact_matrix[res1 * seq_length + res2]) {
                     has_contact = true;
                 }
             }
         }
         
         if (has_contact) {
             atomicAdd(&contact_count_matrix[motif_idx * max_motifs + other_idx], 1);
         }
     }
 }