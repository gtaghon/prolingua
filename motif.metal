#include <metal_stdlib>
using namespace metal;

// Struct for encoding motif information
struct Motif {
    uint hash;     // Hash of the motif sequence
    uint position; // Position in the sequence
    uint size;     // Size of the motif
    uint valid;    // Whether the motif is valid (contains valid amino acids)
};

// Struct for frame information
struct FrameInfo {
    uint frame_idx;
    uint sequence_length;
    uint valid_motif_count;
};

// Constants used in the shader
constant uint MAX_MOTIF_SIZE = 10;      // Maximum supported motif size
constant uint NUM_DSSP_CODES = 8;       // Number of DSSP secondary structure codes
constant uint NUM_AMINO_ACIDS = 20;     // Number of standard amino acids
constant uint HASH_TABLE_SIZE = 1000003; // Prime number for hash table size

// Utility function to compute motif hash
uint compute_motif_hash(
    device const uint* sequence,
    uint position,
    uint size,
    device uint* is_valid
) {
    uint hash = 17; // Initial hash value
    
    for (uint i = 0; i < size; i++) {
        uint aa_idx = sequence[position + i];
        
        // Check if the amino acid is valid (within the standard 20)
        if (aa_idx >= NUM_AMINO_ACIDS) {
            *is_valid = 0;
            return 0;
        }
        
        // MurmurHash-inspired mixing function
        hash = hash * 31 + aa_idx;
    }
    
    *is_valid = 1;
    return hash % HASH_TABLE_SIZE;
}

// Kernel for generating motifs from a sequence
kernel void generate_motifs(
    device const uint* sequence [[buffer(0)]],
    device const uint* sequence_valid_mask [[buffer(1)]],
    constant uint* motif_params [[buffer(2)]],
    device Motif* motifs [[buffer(3)]],
    device FrameInfo* frame_info [[buffer(4)]],
    uint thread_id [[thread_position_in_grid]],
    uint thread_group_id [[threadgroup_position_in_grid]]
) {
    // Extract parameters
    uint seq_len = motif_params[0];
    uint min_motif_size = motif_params[1];
    uint max_motif_size = motif_params[2];
    
    // Calculate position in sequence
    uint frame_idx = thread_group_id;
    uint pos = thread_id % seq_len;
    
    // Early exit if out of bounds
    if (pos >= seq_len) {
        return;
    }
    
    // Get frame info for this group
    device FrameInfo& info = frame_info[frame_idx];
    info.frame_idx = frame_idx;
    info.sequence_length = seq_len;
    
    // Quick check if position has invalid amino acid
    if (sequence_valid_mask[pos] == 0) {
        return;
    }
    
    // Process each motif size
    for (uint motif_size = min_motif_size; motif_size <= max_motif_size && motif_size <= MAX_MOTIF_SIZE; motif_size++) {
        // Skip if motif would exceed sequence boundary
        if (pos + motif_size > seq_len) {
            continue;
        }
        
        // Quick check for valid segment (using precomputed mask)
        bool valid_segment = true;
        for (uint i = 0; i < motif_size; i++) {
            if (sequence_valid_mask[pos + i] == 0) {
                valid_segment = false;
                break;
            }
        }
        
        if (!valid_segment) {
            continue;
        }
        
        // Compute motif hash
        uint is_valid = 1;
        uint hash = compute_motif_hash(sequence, pos, motif_size, &is_valid);
        
        if (is_valid) {
            // Add to motifs array using atomic operation to avoid race conditions
            uint idx = atomic_fetch_add_explicit(&info.valid_motif_count, 1, memory_order_relaxed);
            
            // Store motif information
            device Motif& motif = motifs[frame_idx * seq_len + idx];
            motif.hash = hash;
            motif.position = pos;
            motif.size = motif_size;
            motif.valid = 1;
        }
    }
}

// Kernel for processing secondary structure assignments
kernel void process_ss_assignments(
    device const Motif* motifs [[buffer(0)]],
    device const FrameInfo* frame_info [[buffer(1)]],
    device const uint* dssp_data [[buffer(2)]],
    device atomic_uint* ss_counts [[buffer(3)]],
    device atomic_uint* motif_occurrences [[buffer(4)]],
    device atomic_uint* temp_ss_counts [[buffer(5)]],
    constant uint* params [[buffer(6)]],
    uint thread_id [[thread_position_in_grid]]
) {
    // Extract parameters
    uint seq_len = params[0];
    uint temp_index = params[1];
    uint max_motifs_per_frame = params[2];
    
    // Calculate frame and motif indices
    uint frame_idx = thread_id / max_motifs_per_frame;
    uint motif_idx = thread_id % max_motifs_per_frame;
    
    // Get frame info and check bounds
    uint valid_motif_count = frame_info[frame_idx].valid_motif_count;
    if (motif_idx >= valid_motif_count) {
        return;
    }
    
    // Get motif information
    const Motif& motif = motifs[frame_idx * max_motifs_per_frame + motif_idx];
    if (!motif.valid) {
        return;
    }
    
    // Update motif occurrence counter
    atomic_fetch_add_explicit(&motif_occurrences[motif.hash], 1, memory_order_relaxed);
    
    // Process secondary structure assignments for each position in the motif
    uint pos = motif.position;
    for (uint i = 0; i < motif.size; i++) {
        uint dssp_code = dssp_data[frame_idx * seq_len + pos + i];
        
        if (dssp_code < NUM_DSSP_CODES) {
            // Update global SS counts
            atomic_fetch_add_explicit(
                &ss_counts[motif.hash * NUM_DSSP_CODES + dssp_code], 
                1, 
                memory_order_relaxed
            );
            
            // Update temperature-specific SS counts
            atomic_fetch_add_explicit(
                &temp_ss_counts[temp_index * HASH_TABLE_SIZE * NUM_DSSP_CODES + motif.hash * NUM_DSSP_CODES + dssp_code],
                1,
                memory_order_relaxed
            );
        }
    }
}

// Kernel for processing contacts between motifs
kernel void process_motif_contacts(
    device const Motif* motifs [[buffer(0)]],
    device const FrameInfo* frame_info [[buffer(1)]],
    device const uint* contact_matrix [[buffer(2)]],
    device atomic_uint* motif_contacts [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint thread_id [[thread_position_in_grid]]
) {
    // Extract parameters
    uint max_motifs_per_frame = params[0];
    uint seq_len = params[1];
    
    // Calculate frame and motif pair indices
    uint frame_idx = thread_id / (max_motifs_per_frame * max_motifs_per_frame);
    uint remaining = thread_id % (max_motifs_per_frame * max_motifs_per_frame);
    uint motif1_idx = remaining / max_motifs_per_frame;
    uint motif2_idx = remaining % max_motifs_per_frame;
    
    // Skip if same motif or out of bounds
    uint valid_motif_count = frame_info[frame_idx].valid_motif_count;
    if (motif1_idx == motif2_idx || motif1_idx >= valid_motif_count || motif2_idx >= valid_motif_count) {
        return;
    }
    
    // Get motif information
    const Motif& motif1 = motifs[frame_idx * max_motifs_per_frame + motif1_idx];
    const Motif& motif2 = motifs[frame_idx * max_motifs_per_frame + motif2_idx];
    
    if (!motif1.valid || !motif2.valid) {
        return;
    }
    
    // Check for contact between motifs
    bool has_contact = false;
    for (uint i = 0; i < motif1.size && !has_contact; i++) {
        for (uint j = 0; j < motif2.size && !has_contact; j++) {
            uint pos1 = motif1.position + i;
            uint pos2 = motif2.position + j;
            
            // Look up in contact matrix (assuming linearized)
            // For a symmetric matrix, always use the lower triangle
            uint min_pos = min(pos1, pos2);
            uint max_pos = max(pos1, pos2);
            
            // Skip diagonal
            if (min_pos == max_pos) {
                continue;
            }
            
            // Calculate 1D index for contact matrix
            // Using the formula for lower triangle of matrix: idx = n*(n-1)/2 + m
            // where n is the row and m is the column (for m < n)
            uint contact_idx = (max_pos * (max_pos - 1)) / 2 + min_pos;
            
            // Check if contact exists
            if (contact_matrix[frame_idx * ((seq_len * (seq_len - 1)) / 2) + contact_idx] == 1) {
                has_contact = true;
                break;
            }
        }
    }
    
    // Record contact if found
    if (has_contact) {
        // Store contact in hash table (motif1.hash -> motif2.hash)
        uint slot = (motif1.hash * 17 + motif2.hash) % HASH_TABLE_SIZE;
        atomic_fetch_add_explicit(&motif_contacts[slot], 1, memory_order_relaxed);
        
        // Also store reverse contact (motif2.hash -> motif1.hash)
        slot = (motif2.hash * 17 + motif1.hash) % HASH_TABLE_SIZE;
        atomic_fetch_add_explicit(&motif_contacts[slot], 1, memory_order_relaxed);
    }
}