###############################
# Phase 2: GRT Analysis
###############################

import os
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from typing import Dict, DefaultDict


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
        clusters = DefaultDict(list)
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