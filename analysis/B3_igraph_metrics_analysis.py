import igraph as ig
import pandas as pd
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
input_path = f"{output_dir}/converted_edgelist.csv"

# Load graph from edgelist (skip header manually)
print("Loading graph with igraph...")
with open(input_path, "r") as f:
    lines = f.readlines()[1:]  # Skip header

# Convert comma-separated values to space-separated format for igraph
temp_path = f"{output_dir}/temp_edgelist.txt"
with open(temp_path, "w") as f:
    for line in lines:
        f.write(" ".join(line.strip().split(",")) + "\n")

g = ig.Graph.Read_Edgelist(temp_path, directed=False)

# Compute useful centrality metrics (for small graphs)
print("Calculating closeness centrality...")
closeness = g.closeness()
print("Calculating eigenvector centrality...")
eigenvector = g.eigenvector_centrality()
print("Calculating clustering coefficient...")
clustering = g.transitivity_local_undirected(mode="zero")

# Create DataFrame and save
vertex_ids = list(range(g.vcount()))
df = pd.DataFrame({
    "id": vertex_ids,
    "closeness_igraph": closeness,
    "eigenvector_igraph": eigenvector,
    "clustering_igraph": clustering
})
df.to_csv(f"{output_dir}/igraph_metrics.csv", index=False)
print("Saved igraph metrics.")