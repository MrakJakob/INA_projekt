import networkx as nx
import json
import os
from networkx.algorithms import bipartite

with open("analysis/config.json") as f:
    config = json.load(f)

G = nx.read_graphml(config["input_graphml"])
output_dir = config["output_dir"]

playlist_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "playlist"]
track_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "track"]

# Project playlist-to-playlist graph
P = bipartite.weighted_projected_graph(G, playlist_nodes)
nx.write_graphml(P, f"{output_dir}/projected_playlist_playlist.graphml")
print("Saved projected playlist-playlist graph.")