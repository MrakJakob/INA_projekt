import optuna
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from models import Node2VecModel

# Define the objective function for Optuna
def objective(trial):
    # Load the graph
    graph_path = "F:/INA-Project/INA_projekt/matej/graphs/5K_balanced/5K_balanced.graphml"
    G = nx.read_graphml(graph_path)

    # Load training and testing data
    train_df = pd.read_csv("F:/INA-Project/INA_projekt/matej/graphs/5K_balanced/5K_balanced_train.csv")
    test_df = pd.read_csv("F:/INA-Project/INA_projekt/matej/graphs/5K_balanced/5K_balanced_test.csv")
    tr_nodes, tr_buckets = np.array(train_df["nodes"]), np.array(train_df["buckets"])
    ts_nodes, ts_buckets = np.array(test_df["nodes"]), np.array(test_df["buckets"])
    edges = np.load("F:/INA-Project/INA_projekt/matej/graphs/5K_balanced/5k_balanced_edges.npy")

    # Suggest hyperparameters
    dimensions = trial.suggest_int("dimensions", 32, 256, step=32)
    walk_length = trial.suggest_int("walk_length", 10, 80, step=10)
    num_walks = trial.suggest_int("num_walks", 5, 20, step=5)
    p = trial.suggest_float("p", 0.25, 4.0, log=True)
    q = trial.suggest_float("q", 0.25, 4.0, log=True)

    # Initialize the Node2Vec model with suggested parameters
    node2vec_model = Node2VecModel(dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q)

    # Initialize the model with graph data
    node2vec_model.init_data(G, projection=None, test_nodes=ts_nodes, edges=edges)

    # Train the model
    node2vec_model.train(tr_nodes, tr_buckets)

    # Predict using the model
    predictions = node2vec_model.predict(ts_nodes)

    # Evaluate the model using F1 score
    f1 = f1_score(ts_buckets, predictions, average="weighted")
    return f1

# Run the Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    # Print the best parameters and score
    print("Best parameters:", study.best_params)
    print("Best F1 score:", study.best_value)

    # Save the study results
    study.trials_dataframe().to_csv("optuna_node2vec_results.csv", index=False)