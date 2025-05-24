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


def create_graph_playlist_tracks(data_directory, max_slices=None):
    """Process the entire dataset or a subset of slices to create playlist-track bipartite graph."""

    # Read from balanced_spotify_playlists.json
    slice_files = sorted(
        glob.glob(os.path.join(data_directory, "balanced_spotify_playlists.json"))
    )

    # Create empty bipartite graph
    G = nx.Graph()

    # Track all playlists for later analysis
    all_playlists = []

    # Track statistics
    total_tracks = 0
    unique_tracks = set()

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

            # Process each track in the playlist
            for track in playlist["tracks"]:
                track_uri = track["track_uri"]
                total_tracks += 1
                unique_tracks.add(track_uri)

                # Add track node if not already present
                if not G.has_node(track_uri):
                    G.add_node(
                        track_uri,
                        type="track",
                        name=track["track_name"],
                        artist_name=track["artist_name"],
                        artist_uri=track["artist_uri"],
                        album_name=track["album_name"],
                        album_uri=track["album_uri"],
                        duration_ms=track.get("duration_ms", 0),
                    )

                # Create edge between playlist and track
                G.add_edge(pid, track_uri)

    print(f"Total track occurrences: {total_tracks}")
    print(f"Unique tracks: {len(unique_tracks)}")

    return G, all_playlists


def create_graph_playlist_artists(data_directory, max_slices=None):
    """Process the entire dataset or a subset of slices."""

    # slice_files = sorted(glob.glob(os.path.join(data_directory, "mpd.slice.*.json")))

    # if max_slices:
    #     slice_files = slice_files[:max_slices]
    # read from balanced_spotify_playlists.json
    slice_files = sorted(
        glob.glob(os.path.join(data_directory, "balanced_spotify_playlists.json"))
    )

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


def create_bipartite_projection(
    bipartite_graph,
    output_dir="./graphs/projections/",
    graph_name="bipartite",
    weighted=True,
):
    """
    Create both projections of a bipartite graph (playlist-artist or playlist-track).

    Parameters:
    - bipartite_graph: NetworkX bipartite graph
    - output_dir: Directory to save projection graphs
    - graph_name: Base name for output files
    - weighted: Whether to create weighted projections
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Separate nodes by type
    playlist_nodes = [
        n for n, d in bipartite_graph.nodes(data=True) if d.get("type") == "playlist"
    ]
    content_nodes = [
        n
        for n, d in bipartite_graph.nodes(data=True)
        if d.get("type") in ["artist", "track"]
    ]

    content_type = (
        "artist"
        if any(d.get("type") == "artist" for n, d in bipartite_graph.nodes(data=True))
        else "track"
    )

    print(f"Creating projections for {graph_name}...")
    print(f"Playlist nodes: {len(playlist_nodes)}")
    print(f"{content_type.title()} nodes: {len(content_nodes)}")

    # Verify it's bipartite
    if not nx.is_bipartite(bipartite_graph):
        print("Warning: Graph is not bipartite!")
        return None, None

    # Create projections
    if weighted:
        print("Creating weighted projections...")

        # Playlist projection (playlists connected if they share artists/tracks)
        playlist_projection = nx.bipartite.weighted_projected_graph(
            bipartite_graph, playlist_nodes
        )

        # # Content projection (artists/tracks connected if they appear in same playlists)
        # content_projection = nx.bipartite.weighted_projected_graph(
        #     bipartite_graph, content_nodes
        # )
    else:
        print("Creating unweighted projections...")

        # Playlist projection
        playlist_projection = nx.bipartite.projected_graph(
            bipartite_graph, playlist_nodes
        )

        # # Content projection
        # content_projection = nx.bipartite.projected_graph(
        #     bipartite_graph, content_nodes
        # )

    # Add metadata to projection graphs
    playlist_projection.graph["projection_type"] = "playlist"
    playlist_projection.graph["original_graph"] = graph_name
    playlist_projection.graph["weighted"] = weighted

    # content_projection.graph["projection_type"] = content_type
    # content_projection.graph["original_graph"] = graph_name
    # content_projection.graph["weighted"] = weighted

    # Analyze projections
    # analyze_projection(playlist_projection, f"Playlist-Playlist ({content_type} similarity)")
    # analyze_projection(content_projection, f"{content_type.title()}-{content_type.title()} (playlist co-occurrence)")

    # Save projections
    weight_suffix = "_weighted" if weighted else "_unweighted"

    playlist_file = os.path.join(
        output_dir, f"{graph_name}_playlist_projection{weight_suffix}.graphml"
    )
    content_file = os.path.join(
        output_dir, f"{graph_name}_{content_type}_projection{weight_suffix}.graphml"
    )

    print(f"Saving playlist projection to: {playlist_file}")
    nx.write_graphml_lxml(playlist_projection, playlist_file)

    # print(f"Saving {content_type} projection to: {content_file}")
    # nx.write_graphml_lxml(content_projection, content_file)

    return playlist_projection # content_projection


def analyze_bipartite_graph(G):
    """Analyze the bipartite graph structure and provide statistics."""

    # Separate nodes by type
    playlist_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "playlist"]
    track_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "track"]

    print(f"\nGraph Analysis:")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print(f"Playlist nodes: {len(playlist_nodes)}")
    print(f"Track nodes: {len(track_nodes)}")

    # Calculate degree statistics
    playlist_degrees = [G.degree(n) for n in playlist_nodes]
    track_degrees = [G.degree(n) for n in track_nodes]

    print(f"\nPlaylist degree stats (tracks per playlist):")
    print(f"  Mean: {np.mean(playlist_degrees):.2f}")
    print(f"  Median: {np.median(playlist_degrees):.2f}")
    print(f"  Min: {min(playlist_degrees)}")
    print(f"  Max: {max(playlist_degrees)}")

    print(f"\nTrack degree stats (playlists per track):")
    print(f"  Mean: {np.mean(track_degrees):.2f}")
    print(f"  Median: {np.median(track_degrees):.2f}")
    print(f"  Min: {min(track_degrees)}")
    print(f"  Max: {max(track_degrees)}")

    # Check if graph is bipartite
    is_bipartite = nx.is_bipartite(G)
    print(f"\nIs bipartite: {is_bipartite}")

    return {
        "playlist_nodes": len(playlist_nodes),
        "track_nodes": len(track_nodes),                                                                                                                             
        "playlist_degrees": playlist_degrees,
        "track_degrees": track_degrees,
        "is_bipartite": is_bipartite,                                                                                                                  
    }


if __name__ == "__main__":
    data_dir = "./spotify_dataset_files/balanced_spotify_playlist_followers"
    # # change max_slices to None to process all slices
    G_artist, playlists = create_graph_playlist_artists(data_dir, max_slices=None)
    # Save the graph to a file
    nx.write_graphml_lxml(G_artist, "./graphs/bipartite_artist_playlist.graphml")

    print(
        f"Network loaded with {G_artist.number_of_nodes()} nodes and {G_artist.number_of_edges()} edges"
    )
    print(f"Processed {len(playlists)} playlists")

    # playlist_proj_artist, artist_proj = create_bipartite_projection(
    #     G_artist, graph_name="artist_playlist", weighted=True
    # )

    # Create basic playlist-track bipartite graph
    print("Creating playlist-track bipartite graph...")
    G_tracks, playlists = create_graph_playlist_tracks(data_dir, max_slices=None)

    # Save the graph
    print("Saving graph...")
    nx.write_graphml_lxml(G_tracks, "./graphs/bipartite_playlist_track.graphml")

    print(
        f"Network loaded with {G_tracks.number_of_nodes()} nodes and {G_tracks.number_of_edges()} edges"
    )
    print(f"Processed {len(playlists)} playlists")

    # playlist_proj_track, track_proj = create_bipartite_projection(
    #     G_tracks, graph_name="playlist_track", weighted=True
    # )

    # Analyze the graph
    stats = analyze_bipartite_graph(G_tracks)
