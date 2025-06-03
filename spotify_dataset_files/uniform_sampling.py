import json
import os
import glob
import random
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

def create_bipartite_graph_from_json(data_dir, sample_size=1000, max_files=None, random_state=42):
    """
    Create a bipartite graph from Spotify Million Playlist Dataset with uniform sampling.
    
    Parameters:
    - data_dir: Directory containing the mpd.slice.*.json files
    - sample_size: Number of playlists to sample uniformly
    - max_files: Maximum number of files to process (for testing)
    - random_state: Random seed for reproducibility
    
    Returns:
    - NetworkX graph in bipartite format
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Find all slice files
    slice_files = sorted(glob.glob(os.path.join(data_dir, "mpd.slice.*.json")))
    
    if max_files is not None:
        slice_files = slice_files[:max_files]
    
    print(f"Found {len(slice_files)} slice files, sampling {sample_size} playlists")
    
    # First pass: count total playlists and create sampling plan
    total_playlists = 0
    file_playlist_counts = []
    
    for file_path in tqdm(slice_files, desc="Counting playlists"): 
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            num_playlists = len(data['playlists'])
            file_playlist_counts.append((file_path, num_playlists))
            total_playlists += num_playlists
    
    # Determine which playlists to sample
    target_indices = sorted(random.sample(range(total_playlists), min(sample_size, total_playlists)))
    
    # Second pass: collect the sampled playlists
    sampled_playlists = []
    current_index = 0
    target_ptr = 0
    
    for file_path, num_playlists in tqdm(file_playlist_counts, desc="Sampling playlists"):
        if target_ptr >= len(target_indices):
            break
            
        # Check if any targets are in this file
        file_start = current_index
        file_end = current_index + num_playlists - 1
        
        # Find all targets in this file's range
        file_targets = []
        while (target_ptr < len(target_indices) and 
               file_start <= target_indices[target_ptr] <= file_end):
            file_targets.append(target_indices[target_ptr] - file_start)
            target_ptr += 1
        
        if file_targets:
            # Load the file and extract targets
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for idx in file_targets:
                    sampled_playlists.append(data['playlists'][idx])
        
        current_index += num_playlists
    
    print(f"Successfully sampled {len(sampled_playlists)} playlists")
    
    # Now build the bipartite graph
    G = nx.Graph()
    
    # Add attribute keys (as in your example)
    G.graph['node_attributes'] = {
        'type': {'type': 'string', 'id': 'd0'},
        'name': {'type': 'string', 'id': 'd1'},
        'followers': {'type': 'long', 'id': 'd2'},
        'duration_ms': {'type': 'long', 'id': 'd3'},
        'num_tracks': {'type': 'long', 'id': 'd4'},
        'num_artists': {'type': 'long', 'id': 'd5'},
        'num_albums': {'type': 'long', 'id': 'd6'},
        'collaborative': {'type': 'string', 'id': 'd7'},
        #'artist_name': {'type': 'string', 'id': 'd8'},
        'artist_uri': {'type': 'string', 'id': 'd9'},
        # 'album_name': {'type': 'string', 'id': 'd10'},
        'album_uri': {'type': 'string', 'id': 'd11'}
    }
    
    # Track seen tracks 
    seen_tracks = set()
    
    # Process each sampled playlist
    for playlist in tqdm(sampled_playlists, desc="Building graph"):
        # Add playlist node
        pid = f"pl_{playlist['pid']}"
        
        # Calculate duration if not present
        duration_ms = playlist.get('duration_ms', 0)
        if duration_ms == 0 and 'tracks' in playlist:
            duration_ms = sum(track['duration_ms'] for track in playlist['tracks'])
        
        G.add_node(
            pid,
            type="playlist",
            name=playlist.get('name', ''),
            followers=playlist.get('num_followers', 0),
            duration_ms=duration_ms,
            num_tracks=playlist.get('num_tracks', 0),
            num_artists=playlist.get('num_artists', 0),
            num_albums=playlist.get('num_albums', 0),
            collaborative=str(playlist.get('collaborative', 'false')).lower()
        )
        
        # Process tracks if available
        if 'tracks' in playlist:
            for track in playlist['tracks']:
                track_uri = track.get('track_uri', '')
                artist_uri = track.get('artist_uri', '')
                
                # Add track node if not already present
                if track_uri and track_uri not in seen_tracks:
                    G.add_node(
                        track_uri,
                        type="track",
                        name=track.get('track_name', ''),
                        # artist_name=track.get('artist_name', ''),
                        artist_uri=artist_uri,
                        # album_name=track.get('album_name', ''),
                        album_uri=track.get('album_uri', ''),
                        duration_ms=track.get('duration_ms', 0)
                    )
                    seen_tracks.add(track_uri)
                
                # Create edges
                if track_uri:
                    G.add_edge(pid, track_uri)
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def save_graph_to_graphml(G, output_file):
    """
    Save the graph to GraphML format with proper attribute formatting.
    
    Parameters:
    - G: NetworkX graph
    - output_file: Path to save the GraphML file
    """
    # Create a new graph with stringified attributes to ensure GraphML compatibility
    G_out = nx.Graph()
    
    # Add nodes with properly formatted attributes
    for node, data in G.nodes(data=True):
        node_data = {}
        for key, value in data.items():
            # Convert all attributes to strings
            node_data[key] = str(value)
        G_out.add_node(node, **node_data)
    
    # Add edges (no attributes in this case)
    for u, v in G.edges():
        G_out.add_edge(u, v)
    
    # Write to GraphML
    nx.write_graphml(G_out, output_file)
    print(f"Graph saved to {output_file}")

def create_and_save_bipartite_graph(data_dir, output_file, sample_size=1000, max_files=None):
    G = create_bipartite_graph_from_json(data_dir, sample_size, max_files)
    save_graph_to_graphml(G, output_file)

# Example usage:
if __name__ == "__main__":
    data_dir = "./spotify_dataset_files/data"
    size = 10000
    # size
    output_file = "./graphs/uniformly_sampled_playlist_tracks_{}.graphml".format(size)
    create_and_save_bipartite_graph(data_dir, output_file, sample_size=size)