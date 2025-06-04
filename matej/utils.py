from itertools import combinations
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

def load_file(G, file_path):
    print(f"📥 Loading: {file_path}")
    new_graph = nx.read_graphml(file_path)

    for node, attrs in new_graph.nodes(data=True):
        G.add_node(node, **attrs)
    for u, v, attrs in new_graph.edges(data=True):
        G.add_edge(u, v, **attrs)

def load_folder(G, folder_path):
    files = sorted(f for f in os.listdir(folder_path) if f.endswith('.graphml'))
    for file in files:
        file_path = os.path.join(folder_path, file)
        load_file(G, file_path)

def plot_histograms(data, labels=None, colors=None, loglog=False, xlabel="degree"):
    for i, (hist, edges) in enumerate(data):
        plt.stairs(hist, edges, label=labels[i] if labels else "",
                    color=colors[i] if colors else None)
    plt.xlabel(xlabel)
    plt.ylabel("frequency")
    if labels:
        plt.legend()
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    plt.show()

def plot_distributions(data, labels=None, colors=None, styles=None, loglog=False, xlabel="degree"):
    for i, (x, y) in enumerate(data):
        plt.plot(x, y, styles[i] if styles else "-",
                    label=labels[i] if labels else "",
                    color=colors[i] if colors else None)
    plt.xlabel(xlabel)
    plt.ylabel("frequency")
    if labels:
        plt.legend()
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    plt.show()

def get_degree_dist(G, nodes=None):
    nodes = G.nodes() if nodes is None else nodes
    ddist = np.unique([G.degree(n) for n in nodes], return_counts=True)
    return ddist

def get_followers(G):
    playlists = [(n, float(d["followers"])) for n, d in G.nodes(data=True) if d.get("type") == "playlist"]
    nodes, followers = zip(*playlists)
    return nodes, followers

def subsample_graph_uniform(G, k, remove_isolates=True, giant_component_only=False):
    sampled_nodes = random.sample(G.nodes, k)
    sampled_graph = G.subgraph(sampled_nodes).copy()
    if remove_isolates:
        sampled_graph.remove_nodes_from(list(nx.isolates(sampled_graph)))
    if giant_component_only:
        cc = max(nx.connected_components(sampled_graph), key=len)
        sampled_graph = sampled_graph.subgraph(cc).copy()
    return sampled_graph

def project_graph(G, onto="playlist"):
    proj_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == onto}
    projection = bipartite.weighted_projected_graph(G, proj_nodes)
    return projection

def project_graph_thresholded(G, threshold):
    playlists, tracks = get_playlists_tracks(G)
    tracks = set(tracks)

    proj = nx.Graph()
    proj.add_nodes_from((n, G.nodes[n]) for n in playlists)

    for u, v in combinations(playlists, 2):
        neighbors_u = set(G.neighbors(u)) & tracks
        neighbors_v = set(G.neighbors(v)) & tracks
        common = neighbors_u & neighbors_v
        if len(common) >= threshold:
            proj.add_edge(u, v, weight=len(common))

    return proj

def stratified_by_followers(G, num_buckets=None, bucket_edges=None):

    playlists = [(n, float(d["followers"])) for n, d in G.nodes(data=True) if d.get("type") == "playlist"]
    playlists.sort(key=lambda x: x[1])
    node_ids, follower_counts = zip(*playlists)

    if bucket_edges is None:
        assert num_buckets is not None
        quantiles = np.linspace(0, 1, num_buckets + 1)
        bucket_edges = np.quantile(follower_counts, quantiles)
    else:
        bucket_edges = np.array(bucket_edges)
        num_buckets = len(bucket_edges) - 1

    bucket_edges[-1] += 1

    node_ids_out = []
    bucket_indices = []

    bucket_indices = np.digitize(follower_counts, bucket_edges) - 1
    node_ids_out = node_ids

    return np.array(node_ids_out), np.array(bucket_indices), bucket_edges.tolist()

def balance_buckets(node_ids, buckets, edges, ref_bucket):
    num_buckets = len(edges) - 1
    k = np.sum(buckets == ref_bucket)
    node_bins = []
    bucket_bins = []
    for bi in range(num_buckets):
        samp = np.random.permutation(k)
        node_bins.append(node_ids[buckets == bi][samp])
        bucket_bins.append(np.array([bi] * k))
    node_ids = np.concatenate(node_bins)
    buckets = np.concatenate(bucket_bins)
    samp = np.random.permutation(len(node_ids))
    return node_ids[samp], buckets[samp], edges

def get_train_test(node_ids, buckets, ratio=0.7, shuffle=True):
    split = round(len(node_ids) * ratio)
    idx = np.random.permutation(len(node_ids)) if shuffle else np.arange(0, len(node_ids))
    train = idx[:split]
    test = idx[split:]
    return node_ids[train], buckets[train], node_ids[test], buckets[test]

def read_graph(name):
    return nx.read_graphml(f"graphs/{name}/{name}.graphml")

def get_playlists_tracks(G):
    playlists = [n for n, d in G.nodes(data=True) if d['type'] == 'playlist']
    tracks = [n for n, d in G.nodes(data=True) if d['type'] == 'track']
    return playlists, tracks


if __name__ == "__main__":

    G = nx.Graph()
    #load_file(G, "../playlist_graph/playlist_graph_0-99.graphml")
    load_folder(G, "../playlist_graph")
    print(len(G))
    sampled_graph = subsample_graph_uniform(G, 50000, giant_component_only=True)
    nx.write_graphml(sampled_graph, "graphs/test_mini/test_mini.graphml")

    # sampled_mid = subsample_graph_uniform(G, 200000)
    # nx.write_graphml(sampled_mid, "graphs/mid/mid.graphml")

    # sampled_test = subsample_graph_uniform(G, 100000, giant_component_only=True)
    # nx.write_graphml(sampled_test, "graphs/test/test.graphml")


    # G = nx.read_graphml("graphs/mid/mid.graphml")

    # tracks = {n for n, d in G.nodes(data=True) if d.get("type") == "track"}
    # playlists = {n for n, d in G.nodes(data=True) if d.get("type") == "playlist"}

    # print(len(tracks), len(playlists))
    # print(len([n for n in tracks if len(list(G.neighbors(n))) == 0]))
    # print(len([n for n in playlists if len(list(G.neighbors(n))) == 0]))


    # plot_distributions([get_degree_dist(G, tracks)], loglog=True)
    # plot_distributions([get_degree_dist(G, playlists)], loglog=True)

    # playlist_projection = bipartite.weighted_projected_graph(G, playlists)
    # plot_distributions([get_degree_dist(playlist_projection)], loglog=True)

    # max_followers = np.max(get_followers(G)[1])
    # node_ids, bucket_indices, buckets = stratified_by_followers(G, bucket_edges=[0, 20, max_followers])
    # for i in range(2):
    #     print(f"Bucket {i} [{buckets[i]:.1f} - {buckets[i+1]:.1f}]: {sum(b == i for b in bucket_indices)} nodes")
