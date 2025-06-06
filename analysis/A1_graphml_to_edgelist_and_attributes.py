import networkx as nx
import pandas as pd
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

graphml_path = config["input_graphml"]
output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

G = nx.read_graphml(graphml_path)

# Create integer ID mapping
id_map = {node: i for i, node in enumerate(G.nodes())}
reverse_id_map = {i: node for node, i in id_map.items()}

# Save edgelist with integer IDs
int_edges = [(id_map[u], id_map[v]) for u, v in G.edges()]
int_edge_df = pd.DataFrame(int_edges, columns=["source", "target"])
int_edge_df.to_csv(f"{output_dir}/converted_edgelist.csv", index=False)

# Save node attributes with original ID mapping
rows = []
for node, data in G.nodes(data=True):
    row = {"id": id_map[node], "original_id": node}
    row.update(data)
    rows.append(row)

pd.DataFrame(rows).to_csv(f"{output_dir}/node_attributes.csv", index=False)
print("Saved integer edgelist and node attributes with mapping.")
