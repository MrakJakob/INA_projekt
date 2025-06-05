import sys
import numpy as np
import pandas as pd
import networkx as nx
from utils import project_graph, get_train_test, stratified_by_followers, \
                    get_followers, balance_buckets, project_graph_thresholded, \
                    project_graph_thresholded_fast, get_balanced_train_test


if __name__ == "__main__":

    gname = sys.argv[1]
    gdir = f"graphs/{gname}"

    # Uncomment if you want to generate the necessary files for the 5K playlists graph
    #gdir = f"./graphs/5K_playlists/balanced"
    #gname = "5000_playlists_balanced"

    print(f"Preparing data in {gdir}...")

    G = nx.read_graphml(f"{gdir}/{gname}.graphml")
    
    n_edges = G.number_of_edges()
    print(f"Loaded graph with {len(G)} nodes, {n_edges} edges")
    print("Computing projection...")

    # SWITCH TO "THRESHOLDED" FOR SMALLER PROJECTION GRAPH
    #projection = project_graph(G)
    #projection = project_graph_thresholded_fast(G, 2)

    #nx.write_graphml(projection, f"{gdir}/{gname}_projection.graphml")

    max_followers = np.max(get_followers(G)[1])
    #node_ids, buckets, edges = stratified_by_followers(G, bucket_edges=[0, 10, max_followers])
    edges = [0, 10, max_followers + 1]
    
    # UNCOMMENT NEXT LINE IF DATASET IS ALREADY BALANCED
    #node_ids, buckets, edges = balance_buckets(node_ids, buckets, edges, ref_bucket=1, upsample_factor=2)
    #tr_nodes, tr_buckets, ts_nodes, ts_buckets = get_train_test(node_ids, buckets, ratio=0.7)

    tr_nodes, tr_buckets, ts_nodes, ts_buckets = get_balanced_train_test(G,
                                                    edges,
                                                    ref_bucket=1, upsample_factor=1,
                                                    ratio=0.7)


    print("Train:")
    for i in range(len(edges) - 1):
        print(f"Bucket {i} [{edges[i]:.1f} - {edges[i+1]:.1f}]: {sum(b == i for b in tr_buckets)} nodes")
    print("Test:")
    for i in range(len(edges) - 1):
        print(f"Bucket {i} [{edges[i]:.1f} - {edges[i+1]:.1f}]: {sum(b == i for b in ts_buckets)} nodes")

    train_df = pd.DataFrame({"nodes": tr_nodes, "buckets": tr_buckets})
    test_df = pd.DataFrame({"nodes": ts_nodes, "buckets": ts_buckets})
    train_df.to_csv(f"{gdir}/{gname}_train.csv")
    test_df.to_csv(f"{gdir}/{gname}_test.csv")
    np.save(f"{gdir}/{gname}_edges", edges)
