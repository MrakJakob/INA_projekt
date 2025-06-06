import cudf
import cugraph
import pandas as pd
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
edgelist_path = os.path.join(output_dir, "converted_edgelist.csv")
node_attributes_path = os.path.join(output_dir, "node_attributes.csv")
community_output_path = os.path.join(output_dir, "community_labels.csv")

print("Loading edgelist and node attributes...")
edges = cudf.read_csv(edgelist_path)
full_nodes_df = pd.read_csv(node_attributes_path)
full_nodes_df["id"] = full_nodes_df["id"].astype(str)
nodes_df = cudf.DataFrame.from_pandas(full_nodes_df)

print("Building cuGraph object...")
G = cugraph.Graph()
G.from_cudf_edgelist(edges, source="source", destination="target", renumber=False)

print("Running Louvain...")
louvain_df, _ = cugraph.louvain(G)
louvain_df = louvain_df.rename(columns={"partition": "louvain"})

print("Merging results with full node attributes...")
louvain_df["vertex"] = louvain_df["vertex"].astype(str)
final_df = louvain_df.merge(nodes_df, left_on="vertex", right_on="id", how="left")

final_df.to_csv(community_output_path, index=False)
print(f"Saved community assignments to {community_output_path}")
