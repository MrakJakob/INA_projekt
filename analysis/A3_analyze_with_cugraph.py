import cudf
import cugraph
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]

df = cudf.read_csv(f"{output_dir}/converted_edgelist.csv")
G = cugraph.Graph()
G.from_cudf_edgelist(df, source="source", destination="target", renumber=False)

# Compute features
print("Analyzing graph with cuGraph...")
print("Degree centrality...")
degree_df = G.degree()
print("PageRank...")
pagerank_df = cugraph.pagerank(G)
print("Betweenness centrality...")
betweenness_df = cugraph.betweenness_centrality(G, k=512)
print("Eigenvector centrality...")
eigenvector_df = cugraph.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
print("Katz centrality...")
katz_df = cugraph.katz_centrality(G)

metrics = degree_df.merge(pagerank_df, on="vertex")
metrics = metrics.merge(betweenness_df, on="vertex")
metrics = metrics.merge(eigenvector_df, on="vertex")
metrics = metrics.merge(katz_df, on="vertex")
metrics = metrics.rename(columns={"vertex": "id"})
metrics.to_pandas().to_csv(f"{output_dir}/cugraph_metrics.csv", index=False)
print("Saved cuGraph metrics.")