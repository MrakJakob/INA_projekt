import networkx as nx
import pandas as pd
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]

# Read original graphml for full ID context
graphml_path = config["input_graphml"]
G = nx.read_graphml(graphml_path)

# Compute simple and fast metrics
degree = dict(G.degree())
degree_centrality = nx.degree_centrality(G)

# Build dataframe
df = pd.DataFrame({
    "original_id": list(degree.keys()),
    "degree": list(degree.values()),
    "degree_centrality_nx": [degree_centrality[n] for n in degree.keys()]
})
df.to_csv(f"{output_dir}/basic_graph_features.csv", index=False)
print("Saved basic NetworkX features.")