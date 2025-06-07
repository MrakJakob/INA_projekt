import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, roc_curve
from models import NeighborMean, TrackDegree, Majority, Spectral, NameEmbedding, SimilarNeighbor, Node2VecModel
from gnn import GraphSAGEBasic
from neural_network import NeuralClassifier



if __name__ == "__main__":
    gname = sys.argv[1]
    gdir = f"graphs/{gname}"

    print(f"Training and evaluating models on data in {gdir}...")

    G = nx.read_graphml(f"{gdir}/{gname}.graphml")

    G.graph["dir"] = gdir
    G.graph["features_file"] = f"{gdir}/features/{gname}_features.csv"
    projection = nx.read_graphml(f"{gdir}/{gname}_projection.graphml")
    train_df = pd.read_csv(f"{gdir}/{gname}_train.csv")
    tr_nodes, tr_buckets = np.array(train_df["nodes"]), np.array(train_df["buckets"])
    test_df = pd.read_csv(f"{gdir}/{gname}_test.csv")
    ts_nodes, ts_buckets = np.array(test_df["nodes"]), np.array(test_df["buckets"])
    edges = np.load(f"{gdir}/{gname}_edges.npy")
    #features_df = pd.read_csv(f"{fdir}/{gname}_features.csv")

    print("CC in G:", [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    print("CC in projection:", [len(c) for c in sorted(nx.connected_components(projection), key=len, reverse=True)])
    print(f"total nodes: ", len(G))
    print(f"train/test playlists: {len(tr_nodes)}/{len(ts_nodes)}")


    models = {
        #"Node descriptors + NN": NeuralClassifier()
        "GraphSAGE (random)": GraphSAGEBasic(epochs=30, node_ft=None),
        "GraphSAGE (degree)": GraphSAGEBasic(epochs=30, node_ft="degree"),
        "GraphSAGE (name)": GraphSAGEBasic(epochs=30, node_ft="name", ft_dim=384, hidden_dim=32),
        #"Node2Vec + LR": Node2VecModel(dimensions=128, walk_length=40, num_walks=10, p=1, q=2),
        "Spectral embedding + LR": Spectral(),
        "Name embeddings + LR": NameEmbedding(),
        "Track Degree + LR": TrackDegree(),
        "Neighbor Mean": NeighborMean(),
        #"Similar Neighbor": SimilarNeighbor(),
        #"Similar Neighbor": SimilarNeighbor(),
        "Majority": Majority(),
    }

    if len(sys.argv) > 2:
        mname = sys.argv[2]
        models = {mname: models[mname]}

    print("\nModels to train: ")
    for mname in models.keys():
        print(mname)
    print("")

    all_scores = []
    for mname, model in models.items():
        model.init_data(G, projection, ts_nodes, edges)
        model.train(tr_nodes, tr_buckets)
        pred = model.predict(ts_nodes)
        prob = model.predict_proba(ts_nodes) if hasattr(model, "predict_proba") else None

        os.makedirs(f"predictions/{gname}", exist_ok=True)
        pred_df = pd.DataFrame({"true": ts_buckets, "pred": pred})

        scores = {
            "model": mname,
            "ca": accuracy_score(ts_buckets, pred),
            "precision": precision_score(ts_buckets, pred, average="binary"),
            "recall": recall_score(ts_buckets, pred, average="binary"),
            "f1": f1_score(ts_buckets, pred, average="binary")
        }

        if prob is not None:
            prob = prob[:, 1] if len(prob.shape) > 1 else prob
            fpr, tpr, thresholds = roc_curve(ts_buckets, prob)
            youden_index = tpr - fpr
            optimal_threshold = thresholds[np.argmax(youden_index)]

            pred_df["prob"] = prob
            auc = roc_auc_score(ts_buckets, prob)
            scores["auc"] = auc
            
            # # Get predictions with optimal threshold
            # optimal_predictions = (prob[:, 1] >= optimal_threshold).astype(int)
            # optimal_accuracy = accuracy_score(ts_buckets, optimal_predictions)
            # optimal_f1 = f1_score(ts_buckets, optimal_predictions, average="weighted")

            # print(f"{mname} - Optimal Threshold: {optimal_threshold:.4f}, "
            #         f"Accuracy: {optimal_accuracy:.4f}, F1 Score: {optimal_f1:.4f}")

        pred_df.to_csv(f"predictions/{gname}/{mname}.csv")


        all_scores.append(scores)
        print(f"{mname}: {scores}")

    os.makedirs(f"results/{gname}", exist_ok=True)
    results_df = pd.DataFrame(all_scores)
    results_df.to_csv(f"results/{gname}/results.csv")

    
    

