import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
#import community as community_louvain  # python-louvain package
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_graph(filepath="./graphs/bipartite_artist_playlist.graphml"):
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

def extract_artist_nodes(G):
    """Extract all artist nodes from the graph."""
    artist_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'artist']
    print(f"Found {len(artist_nodes)} artist nodes")
    return artist_nodes

def compute_basic_features(G, playlist_nodes):
    """Compute basic features for each playlist."""
    features = {}
    
    for pid in tqdm(playlist_nodes, desc="Computing basic features"):
        # Get playlist attributes
        playlist_attrs = G.nodes[pid]
        
        # Get artists connected to this playlist
        artist_neighbors = [n for n in G.neighbors(pid)]
        
        # Basic features
        features[pid] = {
            'name': playlist_attrs.get('name', ''),
            'followers': int(playlist_attrs.get('followers', 0)),
            'num_tracks': int(playlist_attrs.get('num_tracks', 0)),
            'num_artists': len(artist_neighbors),
            'num_albums': int(playlist_attrs.get('num_albums', 0)),
            'collaborative': playlist_attrs.get('collaborative', 'false') == 'true',
            'duration_ms': int(playlist_attrs.get('duration_ms', 0)),
            'avg_duration_per_track': int(playlist_attrs.get('duration_ms', 0)) / max(1, int(playlist_attrs.get('num_tracks', 1)))
        }
        
    return features

def compute_network_features(G, playlist_nodes, artist_nodes):
    """Compute network-based features for each playlist."""
    print("Computing network features...")
    
    # Pre-compute some global metrics
    print("Computing global network metrics...")
    
    # Compute artist popularity (degree centrality in the bipartite graph)
    artist_popularity = {}
    for artist in tqdm(artist_nodes, desc="Computing artist popularity"):
        artist_popularity[artist] = len(list(G.neighbors(artist)))
    
    # Compute features for each playlist
    features = {}
    for pid in tqdm(playlist_nodes, desc="Computing network features for playlists"):
        # Get artists connected to this playlist
        artist_neighbors = list(G.neighbors(pid))
        
        if not artist_neighbors:
            continue
            
        # Calculate artist popularity metrics for this playlist
        artist_popularities = [artist_popularity[artist] for artist in artist_neighbors]
        
        # Network features
        features[pid] = {
            'degree': len(artist_neighbors),
            'artist_popularity_mean': np.mean(artist_popularities),
            'artist_popularity_median': np.median(artist_popularities),
            'artist_popularity_std': np.std(artist_popularities) if len(artist_popularities) > 1 else 0,
            'artist_popularity_max': max(artist_popularities),
            'artist_popularity_min': min(artist_popularities),
            'artist_popularity_range': max(artist_popularities) - min(artist_popularities),
        }
        
        # Calculate diversity metrics - Herfindahl-Hirschman Index (HHI)
        # Lower HHI means more diverse playlist
        popularity_sum = sum(artist_popularities)
        if popularity_sum > 0:
            normalized_popularities = [p/popularity_sum for p in artist_popularities]
            features[pid]['artist_diversity_hhi'] = sum([p**2 for p in normalized_popularities])
        else:
            features[pid]['artist_diversity_hhi'] = 1.0  # Max concentration (least diverse)
            
    return features

def compute_second_order_features(G, playlist_nodes):
    """Compute second-order network features (playlist-playlist similarity)."""
    print("Computing second-order features...")
    
    # Create a dictionary to store features
    features = {}
    
    # For each playlist, compute similarity with other playlists
    for pid in tqdm(playlist_nodes, desc="Computing playlist similarity features"):
        # Get this playlist's artists
        pid_artists = set(G.neighbors(pid))
        
        if not pid_artists:
            continue
            
        # Find playlists that share at least one artist
        similar_playlists = []
        for artist in pid_artists:
            for neighbor_pid in G.neighbors(artist):
                if neighbor_pid != pid and neighbor_pid.startswith('p_'):
                    similar_playlists.append(neighbor_pid)
        
        # Count unique similar playlists
        unique_similar_playlists = set(similar_playlists)
        
        # Calculate average Jaccard similarity with similar playlists
        jaccard_similarities = []
        for other_pid in unique_similar_playlists:
            other_artists = set(G.neighbors(other_pid))
            if other_artists:
                intersection = len(pid_artists.intersection(other_artists))
                union = len(pid_artists.union(other_artists))
                jaccard_similarities.append(intersection / union)
        
        # Store features
        features[pid] = {
            'num_similar_playlists': len(unique_similar_playlists),
            'avg_jaccard_similarity': np.mean(jaccard_similarities) if jaccard_similarities else 0,
            'max_jaccard_similarity': max(jaccard_similarities) if jaccard_similarities else 0,
            'connectivity': len(similar_playlists)  # Total connections to other playlists through artists
        }
    
    return features

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
            **second_order_features.get(pid, {})
        }
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(all_features, orient='index')
    
    # Reset index and rename it to 'playlist_id'
    df.index.name = 'playlist_id'
    df.reset_index(inplace=True)
    
    return df


def main():
    # Load the graph
    G = load_graph()
    
    # Extract playlist and artist nodes
    playlist_nodes = extract_playlist_nodes(G)
    artist_nodes = extract_artist_nodes(G)
    
    # Compute features
    basic_features = compute_basic_features(G, playlist_nodes)
    network_features = compute_network_features(G, playlist_nodes, artist_nodes)
    second_order_features = compute_second_order_features(G, playlist_nodes)
    
    # Merge features
    df = merge_features(basic_features, network_features, second_order_features)
    
    # Save features to CSV
    df.to_csv('spotify_playlist_features.csv', index=False)
    print(f"Features saved to spotify_playlist_features.csv")


if __name__ == "__main__":
    main()