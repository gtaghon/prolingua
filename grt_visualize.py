###############################
# Visualization Functions
###############################


import numpy as np
import matplotlib.pyplot as plt
import os


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