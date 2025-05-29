import sys
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import accuracy_score, recall_score, f1_score
from utils import project_graph, get_train_test, stratified_by_followers, get_followers
from models import NeighborMean, TrackDegree


if __name__ == "__main__":

    gname = sys.argv[1]
    gdir = f"graphs/{gname}"

    print(f"Training and evaluating models on data in {gdir}...")

    G = nx.read_graphml(f"{gdir}/{gname}.graphml")
    projection = nx.read_graphml(f"{gdir}/{gname}_projection.graphml")
    train_df = pd.read_csv(f"{gdir}/{gname}_train.csv")
    tr_nodes, tr_buckets = np.array(train_df["nodes"]), np.array(train_df["buckets"])
    test_df = pd.read_csv(f"{gdir}/{gname}_test.csv")
    ts_nodes, ts_buckets = np.array(test_df["nodes"]), np.array(test_df["buckets"])
    edges = np.load(f"{gdir}/{gname}_edges.npy")

    model = NeighborMean()

    models = {
        "Neighbor Mean": NeighborMean(),
        "Track Degree": TrackDegree()
    }

    all_scores = {}
    for mname, model in models.items():
        model.init_data(G, projection, ts_nodes, edges)
        model.train(tr_nodes, tr_buckets)
        pred = model.predict(ts_nodes)

        scores = {
            "ca": accuracy_score(ts_buckets, pred),
            "recall": recall_score(ts_buckets, pred, average="binary"),
            "f1": f1_score(ts_buckets, pred, average="binary")
        }

        print(f"{mname}: {scores}")


    
    

