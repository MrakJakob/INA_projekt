import json
import os
import glob
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm


def load_playlist_slice(filepath):
    """Load a single slice file of the Million Playlist Dataset."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["playlists"]


def process_dataset(data_directory, max_slices=None):
    """Process the entire dataset or a subset of slices."""

    slice_files = sorted(glob.glob(os.path.join(data_directory, "mpd.slice.*.json")))

    if max_slices:
        slice_files = slice_files[:max_slices]

    # Create empty bipartite graph
    G = nx.Graph()

    # Track all playlists for later analysis
    all_playlists = []

    # Process each slice file
    for slice_file in tqdm(slice_files, desc="Processing slice files"):
        playlists = load_playlist_slice(slice_file)
        all_playlists.extend(playlists)

        # Process each playlist
        for playlist in playlists:
            # Add playlist node with attributes
            pid = f"p_{playlist['pid']}"

            G.add_node(
                pid,
                type="playlist",
                name=playlist["name"],
                followers=playlist["num_followers"],
                duration_ms=playlist.get("duration_ms", 0),
                num_tracks=playlist["num_tracks"],
                num_artists=playlist.get("num_artists", 0),
                num_albums=playlist.get("num_albums", 0),
                collaborative=playlist["collaborative"],
            )

            # Extract unique artists from playlist
            artists = set()
            for track in playlist["tracks"]:
                artist_uri = track["artist_uri"]
                artists.add(artist_uri)

                # Add artist node if not already present
                if not G.has_node(artist_uri):
                    G.add_node(
                        artist_uri,
                        type="artist",
                        name=track["artist_name"],
                        
                    )

            # Create edges between playlist and all its artists
            for artist_uri in artists:
                G.add_edge(pid, artist_uri)

    return G, all_playlists


data_dir = "./data"
# change max_slices to None to process all slices
G, playlists = process_dataset(data_dir, max_slices=10)
# Save the graph to a file
nx.write_graphml_lxml(G, "./graphs/bipartite_artist_playlist.graphml")

print(
    f"Network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
)
print(f"Processed {len(playlists)} playlists")
