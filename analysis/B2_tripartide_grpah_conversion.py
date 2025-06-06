import networkx as nx
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

G = nx.read_graphml(config["input_graphml"])
output_dir = config["output_dir"]

tripartite_graph = nx.Graph()

for n, d in G.nodes(data=True):
    node_type = d.get("type")
    d_cleaned = {k: v for k, v in d.items() if k != "type"}
    tripartite_graph.add_node(n, type=node_type, **d_cleaned)

    if node_type == "track":
        artist_uri = d.get("artist_uri")
        if artist_uri:
            tripartite_graph.add_node(artist_uri, type="artist")
            tripartite_graph.add_edge(n, artist_uri)

# Copy over original edges from playlist to tracks
for u, v in G.edges():
    if G.nodes[u].get("type") == "playlist" and G.nodes[v].get("type") == "track":
        tripartite_graph.add_edge(u, v)
    elif G.nodes[v].get("type") == "playlist" and G.nodes[u].get("type") == "track":
        tripartite_graph.add_edge(v, u)

nx.write_graphml(tripartite_graph, f"{output_dir}/tripartite_graph.graphml")
print("Saved tripartite playlist-track-artist graph.")
