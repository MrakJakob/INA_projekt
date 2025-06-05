import networkx as nx

#Prepare the model
# Assume G is your original graph
G = nx.read_graphml("F:/INA-Project/INA_projekt/matej/graphs/5K_balanced/5000_playlists_balanced.graphml")
bipartite_edges = [
    (u, v) for u, v in G.edges()
    if G.nodes[u]["type"] != G.nodes[v]["type"]
]

# Write to file
with open("bipartite_edges.txt", "w") as f:
    for u, v in bipartite_edges:
        f.write(f"{u} {v}\n")