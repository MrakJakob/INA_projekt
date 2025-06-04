import sys
import numpy as np
import pandas as pd
import networkx as nx
from utils import get_playlists_tracks, get_degree_dist, plot_distributions
from networkx.algorithms import bipartite


if __name__ == "__main__":

    gname = sys.argv[1]
    gdir = f"graphs/{gname}"

    print(f"Showing info for {gdir}...")

    G = nx.read_graphml(f"{gdir}/{gname}.graphml")
    projection = nx.read_graphml(f"{gdir}/{gname}_projection.graphml")
    train_df = pd.read_csv(f"{gdir}/{gname}_train.csv")
    tr_nodes, tr_buckets = np.array(train_df["nodes"]), np.array(train_df["buckets"])
    test_df = pd.read_csv(f"{gdir}/{gname}_test.csv")
    ts_nodes, ts_buckets = np.array(test_df["nodes"]), np.array(test_df["buckets"])
    edges = np.load(f"{gdir}/{gname}_edges.npy")
    
    playlists, tracks = get_playlists_tracks(G)

    print(f"nodes: {len(G)}, playlists: {len(playlists)}, tracks: {len(tracks)}")
    print(f"train/test playlists: {len(tr_nodes)}/{len(ts_nodes)}, {len(tr_nodes)+len(ts_nodes)} total")
    print(f"bucket edges: ", edges)
    print(f"samples per bucket (train): ", [np.sum(tr_buckets == i) for i in range(len(edges)-1)])
    print(f"samples per bucket (test): ", [np.sum(ts_buckets == i) for i in range(len(edges)-1)])

    print("CC in G:", [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    print("CC in projection:", [len(c) for c in sorted(nx.connected_components(projection), key=len, reverse=True)])
    print("edges in projection: ", len(projection.edges()))


    plot_distributions([get_degree_dist(G, tracks)], loglog=True)
    plot_distributions([get_degree_dist(G, playlists)], loglog=True)

    playlist_projection = bipartite.weighted_projected_graph(G, playlists)
    plot_distributions([get_degree_dist(playlist_projection)], loglog=True)
