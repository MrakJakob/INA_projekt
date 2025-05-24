import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_graph(filepath="./graphs/bipartite_playlists_tracks.graphml"):
    """Load the bipartite graph from GraphML file."""
    print(f"Loading graph from {filepath}...")
    G = nx.read_graphml(filepath)
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def extract_playlist_nodes(G):
    """Extract all playlist nodes from the graph."""
    playlist_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'playlist']
    print(f"Found {len(playlist_nodes)} playlist nodes")
    return playlist_nodes

def extract_track_nodes(G):
    """Extract all track nodes from the graph."""
    track_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'track']
    print(f"Found {len(track_nodes)} track nodes")
    return track_nodes

def compute_basic_features(G, playlist_nodes):
    """Compute basic features for each playlist based on track connections."""
    features = {}
    
    for pid in tqdm(playlist_nodes, desc="Computing basic features"):
        # Get playlist attributes
        playlist_attrs = G.nodes[pid]
        
        # Get tracks connected to this playlist
        track_neighbors = list(G.neighbors(pid))
        
        # Extract track attributes for analysis
        track_durations = []
        
        for track_id in track_neighbors:
            track_attrs = G.nodes.get(track_id, {})
            
            if 'duration_ms' in track_attrs:
                track_durations.append(float(track_attrs['duration_ms']))
          
        
        # Basic playlist features
        features[pid] = {
            'name': playlist_attrs.get('name', ''),
            'followers': int(playlist_attrs.get('followers', 0)),
            'num_tracks': len(track_neighbors),
            'collaborative': playlist_attrs.get('collaborative', 'false') == 'true',
            
            # Duration features
            'total_duration_ms': sum(track_durations) if track_durations else 0,
            'avg_track_duration': np.mean(track_durations) if track_durations else 0,
            'median_track_duration': np.median(track_durations) if track_durations else 0,
            'std_track_duration': np.std(track_durations) if len(track_durations) > 1 else 0,
            'min_track_duration': min(track_durations) if track_durations else 0,
            'max_track_duration': max(track_durations) if track_durations else 0,
            
        }
        
    return features

def compute_network_features(G, playlist_nodes, track_nodes):
    """Compute network-based features for each playlist."""
    print("Computing network features...")
    
    # Pre-compute track popularity (how many playlists contain each track)
    print("Computing track popularity across playlists...")
    track_playlist_count = {}
    for track in tqdm(track_nodes, desc="Computing track playlist frequency"):
        track_playlist_count[track] = len(list(G.neighbors(track)))
    
    # Compute features for each playlist
    features = {}
    for pid in tqdm(playlist_nodes, desc="Computing network features for playlists"):
        # Get tracks connected to this playlist
        track_neighbors = list(G.neighbors(pid))
        
        if not track_neighbors:
            continue
            
        # Calculate track popularity metrics for this playlist
        track_popularities = [track_playlist_count[track] for track in track_neighbors]
        
        # Network features
        features[pid] = {
            'degree': len(track_neighbors),
            'track_playlist_freq_mean': np.mean(track_popularities),
            'track_playlist_freq_median': np.median(track_popularities),
            'track_playlist_freq_std': np.std(track_popularities) if len(track_popularities) > 1 else 0,
            'track_playlist_freq_max': max(track_popularities),
            'track_playlist_freq_min': min(track_popularities),
            'track_playlist_freq_range': max(track_popularities) - min(track_popularities),
        }
        
        # Calculate track uniqueness/rarity metrics
        rare_tracks = sum(1 for pop in track_popularities if pop <= 5)  # Tracks in 5 or fewer playlists
        common_tracks = sum(1 for pop in track_popularities if pop >= 50)  # Tracks in 50+ playlists
        
        features[pid].update({
            'rare_tracks_count': rare_tracks,
            'rare_tracks_ratio': rare_tracks / len(track_neighbors) if track_neighbors else 0,
            'common_tracks_count': common_tracks,
            'common_tracks_ratio': common_tracks / len(track_neighbors) if track_neighbors else 0,
        })
        
        # Calculate diversity metrics - Herfindahl-Hirschman Index (HHI)
        # Lower HHI means more diverse playlist (tracks with varied popularity)
        popularity_sum = sum(track_popularities)
        if popularity_sum > 0:
            normalized_popularities = [p/popularity_sum for p in track_popularities]
            features[pid]['track_diversity_hhi'] = sum([p**2 for p in normalized_popularities])
        else:
            features[pid]['track_diversity_hhi'] = 1.0  # Max concentration (least diverse)
        
        # Calculate entropy-based diversity
        if track_popularities:
            # Normalize to probabilities
            probs = np.array(track_popularities) / sum(track_popularities)
            # Calculate Shannon entropy
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            features[pid]['track_diversity_entropy'] = entropy
        else:
            features[pid]['track_diversity_entropy'] = 0
            
    return features

def compute_second_order_features(G, playlist_nodes):
    """Compute second-order network features (playlist-playlist similarity based on shared tracks)."""
    print("Computing second-order features...")
    
    # Create a dictionary to store features
    features = {}
    
    # For each playlist, compute similarity with other playlists
    for pid in tqdm(playlist_nodes, desc="Computing playlist similarity features"):
        # Get this playlist's tracks
        pid_tracks = set(G.neighbors(pid))
        
        if not pid_tracks:
            continue
            
        # Find playlists that share at least one track
        similar_playlists = []
        shared_track_counts = defaultdict(int)
        
        for track in pid_tracks:
            for neighbor_pid in G.neighbors(track):
                if neighbor_pid != pid and neighbor_pid.startswith('p_'):
                    similar_playlists.append(neighbor_pid)
                    shared_track_counts[neighbor_pid] += 1
        
        # Count unique similar playlists
        unique_similar_playlists = set(similar_playlists)
        
        # Calculate Jaccard similarity with similar playlists
        jaccard_similarities = []
        cosine_similarities = []
        overlap_coefficients = []
        
        for other_pid in unique_similar_playlists:
            other_tracks = set(G.neighbors(other_pid))
            if other_tracks:
                intersection = len(pid_tracks.intersection(other_tracks))
                union = len(pid_tracks.union(other_tracks))
                
                # Jaccard similarity
                jaccard_similarities.append(intersection / union)
                
                # Cosine similarity
                cosine_sim = intersection / np.sqrt(len(pid_tracks) * len(other_tracks))
                cosine_similarities.append(cosine_sim)
                
                # Overlap coefficient (Szymkiewicz–Simpson coefficient)
                overlap_coeff = intersection / min(len(pid_tracks), len(other_tracks))
                overlap_coefficients.append(overlap_coeff)
        
        # Calculate statistics for shared track counts
        shared_counts = list(shared_track_counts.values())
        
        # Store features
        features[pid] = {
            'num_similar_playlists': len(unique_similar_playlists),
            'avg_jaccard_similarity': np.mean(jaccard_similarities) if jaccard_similarities else 0,
            'max_jaccard_similarity': max(jaccard_similarities) if jaccard_similarities else 0,
            'std_jaccard_similarity': np.std(jaccard_similarities) if len(jaccard_similarities) > 1 else 0,
            
            'avg_cosine_similarity': np.mean(cosine_similarities) if cosine_similarities else 0,
            'max_cosine_similarity': max(cosine_similarities) if cosine_similarities else 0,
            
            'avg_overlap_coefficient': np.mean(overlap_coefficients) if overlap_coefficients else 0,
            'max_overlap_coefficient': max(overlap_coefficients) if overlap_coefficients else 0,
            
            'connectivity': len(similar_playlists),  # Total connections to other playlists through tracks
            'avg_shared_tracks': np.mean(shared_counts) if shared_counts else 0,
            'max_shared_tracks': max(shared_counts) if shared_counts else 0,
            'std_shared_tracks': np.std(shared_counts) if len(shared_counts) > 1 else 0,
        }
    
    return features



def calculate_entropy(counts):
    """Calculate Shannon entropy for a list of counts."""
    if not counts or sum(counts) == 0:
        return 0
    
    total = sum(counts)
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log(p) for p in probs if p > 0)

def merge_features(basic_features, network_features, second_order_features):
    """Merge all feature dictionaries into a single DataFrame."""
    print("Merging all features...")
    
    all_features = {}
    
    # Collect all playlist IDs
    all_pids = set(basic_features.keys())
    all_pids.update(network_features.keys())
    all_pids.update(second_order_features.keys())
    
    # Merge features for each playlist
    for pid in all_pids:
        all_features[pid] = {
            **basic_features.get(pid, {}),
            **network_features.get(pid, {}),
            **second_order_features.get(pid, {}),
        }
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_features, orient='index')
    
    # Reset index and rename it to 'playlist_id'
    df.index.name = 'playlist_id'
    df.reset_index(inplace=True)
    
    return df

def analyze_features(df):
    """Perform basic analysis on the extracted features."""
    print("\n=== Feature Analysis ===")
    
    # Basic statistics
    print(f"Total playlists: {len(df)}")
    print(f"Total features: {len(df.columns) - 1}")  # -1 for playlist_id column
    
    # Feature categories
    basic_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                 ['num_tracks', 'followers', 'duration'])]
    network_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                   ['degree', 'freq', 'diversity', 'rare', 'common'])]
    similarity_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                      ['jaccard', 'cosine', 'overlap', 'similar', 'shared'])]
    
    print(f"Basic features: {len(basic_cols)}")
    print(f"Network features: {len(network_cols)}")
    print(f"Similarity features: {len(similarity_cols)}")
    
    # Top correlations with number of tracks
    if 'num_tracks' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['num_tracks'].abs().sort_values(ascending=False)
        print(f"\nTop 10 features correlated with num_tracks:")
        for feature, corr in correlations.head(10).items():
            if feature != 'num_tracks':
                print(f"  {feature}: {corr:.3f}")

def main():
    # Load the graph
    G = load_graph()
    
    # Extract playlist and track nodes
    playlist_nodes = extract_playlist_nodes(G)
    track_nodes = extract_track_nodes(G)
    
    # Compute features
    basic_features = compute_basic_features(G, playlist_nodes)
    network_features = compute_network_features(G, playlist_nodes, track_nodes)
    second_order_features = compute_second_order_features(G, playlist_nodes)
    
    # Merge features
    df = merge_features(basic_features, network_features, second_order_features)
    
    # Analyze features
    analyze_features(df)
    
    # Save features to CSV
    output_file = 'spotify_playlists_tracks_features.csv'
    df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to {output_file}")
    print(f"DataFrame shape: {df.shape}")

if __name__ == "__main__":
    main()