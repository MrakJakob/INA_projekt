import pandas as pd
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]

attributes = pd.read_csv(f"{output_dir}/node_attributes.csv")
basic = pd.read_csv(f"{output_dir}/basic_graph_features.csv")
cugraph_metrics = pd.read_csv(f"{output_dir}/cugraph_metrics.csv")

# Join on original_id for basic, and id for cugraph
combined = attributes.merge(basic, on="original_id", how="left")
combined = combined.merge(cugraph_metrics, on="id", how="left")

# Optionally include igraph metrics if they exist
igraph_path = f"{output_dir}/igraph_metrics.csv"
if os.path.exists(igraph_path):
    igraph_df = pd.read_csv(igraph_path)
    combined = combined.merge(igraph_df, on="id", how="left")
    print("Merged igraph metrics.")

combined.to_csv(f"{output_dir}/all_node_features.csv", index=False)
print("Saved all combined features to all_node_features.csv")