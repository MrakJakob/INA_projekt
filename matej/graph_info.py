import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from utils import get_playlists_tracks, get_degree_dist, plot_distributions, get_followers, \
                    plot_follower_distribution
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt


if __name__ == "__main__":

    gname = sys.argv[1]
    gdir = f"graphs/{gname}"

    print(f"Showing info for {gdir}...")

    G = nx.read_graphml(f"{gdir}/{gname}.graphml")
    proj_path = f"{gdir}/{gname}_projection.graphml"
    projection = nx.read_graphml(proj_path) if os.path.isfile(proj_path) else None
    if projection is not None:
        train_df = pd.read_csv(f"{gdir}/{gname}_train.csv")
        tr_nodes, tr_buckets = np.array(train_df["nodes"]), np.array(train_df["buckets"])
        test_df = pd.read_csv(f"{gdir}/{gname}_test.csv")
        ts_nodes, ts_buckets = np.array(test_df["nodes"]), np.array(test_df["buckets"])
        edges = np.load(f"{gdir}/{gname}_edges.npy")
    
    playlists, tracks = get_playlists_tracks(G)

    print(f"nodes: {len(G)}, playlists: {len(playlists)}, tracks: {len(tracks)}")

    if projection is not None:
        print(f"train/test playlists: {len(tr_nodes)}/{len(ts_nodes)}, {len(tr_nodes)+len(ts_nodes)} total")
        print(f"bucket edges: ", edges)
        print(f"samples per bucket (train): ", [np.sum(tr_buckets == i) for i in range(len(edges)-1)])
        print(f"samples per bucket (test): ", [np.sum(ts_buckets == i) for i in range(len(edges)-1)])

    print("CC in G:", [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])

    if projection is not None:
        print(f"Projection with {len(projection)} nodes and {projection.number_of_edges()} edges")
        print("CC in projection:", [len(c) for c in sorted(nx.connected_components(projection), key=len, reverse=True)])

    plt.figure(figsize=(12, 9))
    plt.subplot(2, 2, 1)
    plot_distributions([get_degree_dist(G, tracks)], loglog=True, title="Track node degree",
                    compute_exponent=True, colors=["black"], styles=["."], labels=["tracks"])
    plt.subplot(2, 2, 2)
    plot_distributions([get_degree_dist(G, playlists)], title="Playlist node degree",
                    loglog=False, colors=["black"], styles=["."], labels=["playlists"])

    plt.subplot(2, 2, 3)
    if projection is not None:
        plot_distributions([get_degree_dist(projection)], loglog=True, title="Degree in playlist projection",
                        compute_exponent=True, colors=["black"], styles=["."],
                        labels=["playlists"], fit_mins=[0], fit_maxes=[200])

    plt.subplot(2, 2, 4)
    _, followers = get_followers(G)
    flcounts = np.unique(followers, return_counts=True)
    # plot_distributions([flcounts], loglog=True, compute_exponent=True,
    #                 colors=["blue"], styles=["."], labels=["followers"],
    #                 fit_mins=[0], fit_maxes=[100])
    plot_follower_distribution(flcounts, 10, 90)
    #plt.show()
    plt.tight_layout()
    os.makedirs(f"results/{gname}", exist_ok=True)
    plt.savefig(f"results/{gname}/distributions.png")