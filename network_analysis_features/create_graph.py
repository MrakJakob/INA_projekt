import json
import os
import glob
import random
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

def save_graph_to_graphml(G, output_file):
    G_out = nx.Graph()
    
    for node, data in G.nodes(data=True):
        node_data = {}
        for key, value in data.items():
            # Convert all attributes to strings
            node_data[key] = str(value)
        G_out.add_node(node, **node_data)

    for u, v in G.edges():
        G_out.add_edge(u, v)

    nx.write_graphml(G_out, output_file)
    print(f"Graph saved to {output_file}")


def create_bipartite_graph_from_json(data_dir, sample_size=1000, random_state=42):
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # open json one json file that has path data_dir
    file_path = os.path.join(data_dir, "balanced_spotify_playlists_10_followers.json")
    
    # First pass: count total playlists and create sampling plan
    total_playlists = 0
    file_playlist_counts = []
    

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
    
    print(f"Sampling {len(target_indices)} playlists from {total_playlists} total playlists")
    for file_path, num_playlists in file_playlist_counts:
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
    
    # build the bipartite graph
    G = nx.Graph()
    
    # attribute keys
    G.graph['node_attributes'] = {
        'type': {'type': 'string', 'id': 'd0'},
        'name': {'type': 'string', 'id': 'd1'},
        'followers': {'type': 'long', 'id': 'd2'},
        'duration_ms': {'type': 'long', 'id': 'd3'},
        'num_tracks': {'type': 'long', 'id': 'd4'},
        'num_artists': {'type': 'long', 'id': 'd5'},
        'num_albums': {'type': 'long', 'id': 'd6'},
        'collaborative': {'type': 'string', 'id': 'd7'},
        # 'artist_name': {'type': 'string', 'id': 'd8'},
        'artist_uri': {'type': 'string', 'id': 'd9'},
        # 'album_name': {'type': 'string', 'id': 'd10'},
        'album_uri': {'type': 'string', 'id': 'd11'}
    }
    
    # seen tracks 
    seen_tracks = set()
    
    for playlist in tqdm(sampled_playlists, desc="Building graph"):
        pid = f"pl_{playlist['pid']}"
        
        G.add_node(
            pid,
            type="playlist",
            name=playlist.get('name', ''),
            followers=playlist.get('num_followers', 0),
            # num_tracks=playlist.get('num_tracks', 0),
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
                        # name=track.get('track_name', ''),
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


if __name__ == "__main__":
    data_dir = "spotify_dataset_files"
    num_playlists = 1000
    output_file = "./graphs/{}_playlists_balanced.graphml".format(num_playlists)
    
    G = create_bipartite_graph_from_json(data_dir, num_playlists)
    save_graph_to_graphml(G, output_file)