###############################
# Main Execution
###############################

import os
import pickle
from motif_extractor_cuda import CUDAMotifExtractor
from grt_analysis import GRTAnalyzer
from grt_visualize import visualize_grt_profiles, visualize_motif_clusters, visualize_proximity_network


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
        extractor = CUDAMotifExtractor(
            data_dir=args.data_dir,
            min_motif_size=args.min_motif_size,
            max_motif_size=args.max_motif_size,
            cache_dir=args.cache_dir,
            memory_limit_mb=12000,
        )
        
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