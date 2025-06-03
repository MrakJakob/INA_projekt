import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import accuracy_score, recall_score, f1_score

from neural_network import NeuralClassifier
from utils import project_graph, get_train_test, stratified_by_followers, get_followers
from models import NeighborMean, TrackDegree, Majority, Spectral


if __name__ == "__main__":
    gname = sys.argv[1]
    gdir = f"graphs/{gname}"
    #gname = "1000_playlists_balanced"
    #gdir = "./matej/graphs/1K_playlists/balanced"
    fdir = "./matej/graphs/1K_playlists/balanced/features"

    print(f"Training and evaluating models on data in {gdir}...")

    G = nx.read_graphml(f"{gdir}/{gname}.graphml")
    projection = nx.read_graphml(f"{gdir}/{gname}_projection.graphml")
    train_df = pd.read_csv(f"{gdir}/{gname}_train.csv")
    tr_nodes, tr_buckets = np.array(train_df["nodes"]), np.array(train_df["buckets"])
    test_df = pd.read_csv(f"{gdir}/{gname}_test.csv")
    ts_nodes, ts_buckets = np.array(test_df["nodes"]), np.array(test_df["buckets"])
    edges = np.load(f"{gdir}/{gname}_edges.npy")
    features_df = pd.read_csv(f"{fdir}/{gname}_features.csv")

    print("CC in G:", [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    print("CC in projection:", [len(c) for c in sorted(nx.connected_components(projection), key=len, reverse=True)])
    print(f"total nodes: ", len(G))
    print(f"train/test playlists: {len(tr_nodes)}/{len(ts_nodes)}")


    models = {
        "Neighbor Mean": NeighborMean(),
        "Track Degree": TrackDegree(),
        "Majority": Majority(),
        "Spectral": Spectral(),
        # "Neural Network": NeuralClassifier()
    }

    all_scores = {}
    for mname, model in models.items():
        model.init_data(G, projection, ts_nodes, edges, features_df)
        model.train(tr_nodes, tr_buckets)
        pred = model.predict(ts_nodes)

        # followers = get_followers(G)[1]
        # for tr, pr, f in zip(ts_buckets, pred, followers):
        #     print(tr, pr, f)

        scores = {
            "ca": accuracy_score(ts_buckets, pred),
            "recall": recall_score(ts_buckets, pred, average="binary"),
            "f1": f1_score(ts_buckets, pred, average="binary")
        }

        print(f"{mname}: {scores}")


    
    

