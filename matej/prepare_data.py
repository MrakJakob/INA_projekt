import sys
import numpy as np
import pandas as pd
import networkx as nx
from utils import project_graph, get_train_test, stratified_by_followers, \
                    get_followers, balance_buckets


if __name__ == "__main__":

    gname = sys.argv[1]
    gdir = f"graphs/{gname}"

    # Uncomment if you want to generate the necessary files for the 5K playlists graph
    #gdir = f"./matej/graphs/5K_playlists/balanced"
    #gname = "5000_playlists_balanced"

    print(f"Preparing data in {gdir}...")

    G = nx.read_graphml(f"{gdir}/{gname}.graphml")
    
    projection = project_graph(G)
    nx.write_graphml(projection, f"{gdir}/{gname}_projection.graphml")

    max_followers = np.max(get_followers(G)[1])
    node_ids, buckets, edges = stratified_by_followers(G, bucket_edges=[0, 10, max_followers])
    # next line is not necessary for the balanced dataset - 
    #  it can stay, shouldn't do anything if its already balanced
    node_ids, buckets, edges = balance_buckets(node_ids, buckets, edges, ref_bucket=1)
    #node_ids, buckets, edges = stratified_by_followers(G, num_buckets=3)
    tr_nodes, tr_buckets, ts_nodes, ts_buckets = get_train_test(node_ids, buckets, ratio=0.7)

    for i in range(len(edges) - 1):
        print(f"Bucket {i} [{edges[i]:.1f} - {edges[i+1]:.1f}]: {sum(b == i for b in buckets)} nodes")

    train_df = pd.DataFrame({"nodes": tr_nodes, "buckets": tr_buckets})
    test_df = pd.DataFrame({"nodes": ts_nodes, "buckets": ts_buckets})
    train_df.to_csv(f"{gdir}/{gname}_train.csv")
    test_df.to_csv(f"{gdir}/{gname}_test.csv")
    np.save(f"{gdir}/{gname}_edges", edges)
