import os
import argparse
import time
import numpy as np
from collections import Counter
import pickle
import mdtraj as md
from tqdm import tqdm

# Import our accelerated components
from bridge import GPUAccelerator
from bridge import enhance_with_gpu_acceleration
from grt_pcache_mpsfull import MotifExtractor, H5DataManager, io_profiler

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GPU-accelerated motif extraction')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing H5 files')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to store cache files')
    parser.add_argument('--output', type=str, default='motifs.pkl', help='Output file for extracted motifs')
    parser.add_argument('--min_motif_size', type=int, default=3, help='Minimum motif size')
    parser.add_argument('--max_motif_size', type=int, default=6, help='Maximum motif size')
    parser.add_argument('--contact_threshold', type=float, default=0.8, help='Threshold for contacts (nm)')
    parser.add_argument('--cpu_only', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for frame processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()

def main():
    """Main function for motif extraction."""
    # Parse arguments
    args = parse_args()
    
    print(f"=== GPU-Accelerated Motif Extraction ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Motif size range: {args.min_motif_size}-{args.max_motif_size}")
    print(f"Contact threshold: {args.contact_threshold} nm")
    print(f"GPU acceleration: {'Disabled' if args.cpu_only else 'Enabled'}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize motif extractor
    extractor = MotifExtractor(
        data_dir=args.data_dir,
        min_motif_size=args.min_motif_size,
        max_motif_size=args.max_motif_size,
        contact_threshold=args.contact_threshold,
        cache_dir=args.cache_dir,
        n_workers=args.n_workers,
        use_gpu=not args.cpu_only,
        batch_size=args.batch_size,
    )
    
    # Enhance with GPU acceleration
    if not args.cpu_only:
        extractor = enhance_with_gpu_acceleration(extractor)
        print("GPU acceleration enabled with Metal Performance Shaders")
    
    # Process all domains
    processing_start = time.time()
    extractor.process_all_domains(max_files=args.max_files)
    processing_time = time.time() - processing_start
    
    # Print performance report
    print(f"\n=== Performance Summary ===")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Files processed: {extractor.processed_files}")
    print(f"Unique motifs found: {len(extractor.motifs)}")
    
    if hasattr(extractor, 'print_gpu_performance'):
        extractor.print_gpu_performance()
    
    # Save motifs
    print(f"\nSaving motifs to {args.output}...")
    extractor.save_motifs(args.output)
    print("Done!")
    
    # Display detailed I/O profiling
    io_profiler.report()
    
    # Print top motifs by occurrence
    print("\n=== Top 10 Most Common Motifs ===")
    top_motifs = sorted(extractor.motifs.items(), 
                         key=lambda x: x[1]['occurrences'], 
                         reverse=True)[:10]
    
    for motif, data in top_motifs:
        print(f"Motif: {motif}, Occurrences: {data['occurrences']}, " +
              f"Domains: {len(data['domains'])}")

def benchmark_performance(data_dir, n_files=5):
    """
    Run a performance benchmark comparing CPU vs GPU implementations.
    
    Args:
        data_dir: Directory containing H5 files
        n_files: Number of files to use for benchmarking
    """
    print("=== Performance Benchmark: CPU vs GPU ===")
    
    # List of configurations to test
    configs = [
        {"name": "CPU Implementation", "use_gpu": False},
        {"name": "GPU Implementation", "use_gpu": True}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        # Initialize extractor
        extractor = MotifExtractor(
            data_dir=data_dir,
            min_motif_size=3,
            max_motif_size=6,
            contact_threshold=0.8,
            cache_dir=None,  # Disable caching for benchmarking
            use_gpu=config['use_gpu']
        )
        
        if config['use_gpu']:
            extractor = enhance_with_gpu_acceleration(extractor)
        
        # Process domains
        start_time = time.time()
        extractor.process_all_domains(max_files=n_files)
        total_time = time.time() - start_time
        
        # Record results
        results.append({
            "config": config['name'],
            "time": total_time,
            "motifs": len(extractor.motifs),
            "fps": extractor.processed_files / total_time
        })
        
        print(f"Completed in {total_time:.2f} seconds")
        print(f"Found {len(extractor.motifs)} unique motifs")
        
        # Display GPU performance if available
        if hasattr(extractor, 'print_gpu_performance'):
            extractor.print_gpu_performance()
    
    # Print comparison
    print("\n=== Benchmark Results ===")
    print(f"{'Configuration':<20} {'Time (s)':<10} {'Motifs':<10} {'Files/sec':<10}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['config']:<20} {r['time']:<10.2f} {r['motifs']:<10} {r['fps']:<10.2f}")
    
    # Calculate speedup
    if len(results) == 2:
        speedup = results[0]['time'] / results[1]['time']
        print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")

# Entry point
if __name__ == "__main__":
    main()