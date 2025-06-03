import networkx as nx
import pandas as pd
import math
import numpy as np
from tqdm import tqdm  # For progress bars
from network_analysis_features.utils_copy import get_followers, get_train_test, project_graph, stratified_by_followers

def save_graph_to_graphml(G, output_file):
    """
    Save the graph to GraphML format with proper attribute formatting for nodes and edges.
    
    Parameters:
    - G: NetworkX graph (can be weighted)
    - output_file: Path to save the GraphML file
    """
    G_out = nx.Graph()

    # Add nodes with stringified attributes
    for node, data in G.nodes(data=True):
        node_data = {k: str(v) for k, v in data.items()}
        G_out.add_node(node, **node_data)

    # Add edges with stringified attributes (e.g., weight)
    for u, v, edata in G.edges(data=True):
        edge_data = {k: str(v) for k, v in edata.items()}
        G_out.add_edge(u, v, **edge_data)

    nx.write_graphml(G_out, output_file)
    print(f"Graph saved to {output_file}")


def compute_network_features_popularity(G, playlist_nodes, track_nodes):
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



graph_name = "uniformly_sampled_playlist_tracks_15000_balanced.graphml"
projection_graph_name = "uniformly_sampled_playlist_tracks_15000_balanced_projection.graphml"
# === Load Graph ===
print("Loading graph...")
dir = "./network_analysis_features/graphs"
graph_path = f"{dir}/{graph_name}"
G = nx.read_graphml(graph_path)
print(f"Graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges")

# === Separate playlist and track nodes ===
print("Separating playlist and track nodes...")
playlist_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "playlist"]
track_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "track"]
num_playlists = len(playlist_nodes)
print(f"Found {num_playlists} playlists and {len(track_nodes)} tracks")

# === Projected Playlist–Playlist Graph ===
print("\nCreating playlist-playlist projection...")
from networkx.algorithms import bipartite

# Load the saved projection if it exists
projection_path = f"{dir}/{projection_graph_name}"
P_proj = nx.read_graphml(projection_path)
if P_proj is None:
    print("Projection not found, exiting...")
    exit()
# P_proj = project_graph(G)
# nx.write_graphml(P_proj, projection_path)


print("Calculating pagerank scores...")
pagerank_scores = nx.pagerank(P_proj, weight="weight")

# calculate popularity features
popularity_features = compute_network_features_popularity(G, playlist_nodes, track_nodes)

# === Feature Extraction ===
print("\nExtracting features from playlists:")
features = []

for p in tqdm(playlist_nodes, desc="Processing playlists"):
    pdata = G.nodes[p]
    neighbors = list(G.neighbors(p))
    track_neighbors = [t for t in neighbors if G.nodes[t].get("type") == "track"]

    if not track_neighbors:
        continue

    artist_names = set()
    album_names = set()
    durations = []
    track_degrees = []
    track_idfs = []

    for t in track_neighbors:
        tdata = G.nodes[t]
        if "artist_name" in tdata:
            artist_names.add(tdata["artist_name"])
        if "album_name" in tdata:
            album_names.add(tdata["album_name"])
        if "duration_ms" in tdata:
            try:
                durations.append(int(tdata["duration_ms"]))
            except ValueError:
                continue
        track_deg = G.degree(t)
        track_degrees.append(track_deg)
        if track_deg > 0:
            track_idfs.append(math.log(num_playlists / track_deg))

    avg_duration = sum(durations) / len(durations) if durations else 0
    avg_track_deg = sum(track_degrees) / len(track_degrees) if track_degrees else 0
    avg_track_idf = sum(track_idfs) / len(track_idfs) if track_idfs else 0

    features.append({
        "playlist_id": p,
        "name": pdata.get("name", ""),
        "followers": int(pdata.get("followers", "0")),
        "num_tracks": len(track_neighbors),
        "unique_artists": len(artist_names),
        "unique_albums": len(album_names),
        "avg_track_duration_ms": avg_duration,
        "avg_track_degree": avg_track_deg,
        "avg_track_idf": avg_track_idf,
        "collaborative": 1 if pdata.get("collaborative") == "true" else 0,
        "pagerank": pagerank_scores.get(p, 0),
        "track_diversity_hhi": popularity_features.get(p).get('track_diversity_hhi', 1.0),
        "track_diversity_entropy": popularity_features.get(p).get('track_diversity_entropy', 0),
        "track_playlist_freq_mean": popularity_features.get(p).get('track_playlist_freq_mean', 0),
        "track_playlist_freq_median": popularity_features.get(p).get('track_playlist_freq_median', 0),
        "track_playlist_freq_std": popularity_features.get(p).get('track_playlist_freq_std', 0),
        "track_playlist_freq_max": popularity_features.get(p).get('track_playlist_freq_max', 0),
        "track_playlist_freq_min": popularity_features.get(p).get('track_playlist_freq_min', 0),
        "track_playlist_freq_range": popularity_features.get(p).get('track_playlist_freq_range', 0),
        "rare_tracks_count": popularity_features.get(p).get('rare_tracks_count', 0),
        "rare_tracks_ratio": popularity_features.get(p).get('rare_tracks_ratio', 0),
        "common_tracks_count": popularity_features.get(p).get('common_tracks_count', 0),
        "common_tracks_ratio": popularity_features.get(p).get('common_tracks_ratio', 0),
        "degree": popularity_features.get(p).get('degree', 0),  
    })

# === Save to CSV ===
print("\nSaving features to CSV...")
df_features = pd.DataFrame(features)
df_features.to_csv(f"network_analysis_features/features/playlist_graph_features_15000_balanced.csv", index=False)
print("Successfully saved features to playlist_graph_features_15000_balanced.csv")