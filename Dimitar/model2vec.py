import networkx as nx
import numpy as np
from sklearn.metrics import classification_report
from matej.models import Node2VecModel

# Load the graph
graph_path = "F:/INA-Project/INA_projekt/matej/graphs/5K_balanced/5000_playlists_balanced.graphml"
G = nx.read_graphml(graph_path)

# Initialize the Node2VecModel
node2vec_model = Node2VecModel(dimensions=128, walk_length=40, num_walks=10, p=1, q=2)

# Prepare test nodes and edges (bucket thresholds)
test_nodes = np.array([node for node in G.nodes if G.nodes[node].get("type") == "playlist"])
edges = np.array([10, 100, 1000])  # Define bucket thresholds based on followers

# Initialize the model with graph data
node2vec_model.init_data(G, projection=None, test_nodes=test_nodes, edges=edges)

# Prepare training data using graph attributes
train_nodes = test_nodes[:int(len(test_nodes) * 0.8)]  # Use 80% of nodes for training
train_buckets = np.digitize([G.nodes[node]["followers"] for node in train_nodes], edges) - 1

# Train the model
node2vec_model.train(train_nodes, train_buckets)

# Predict using the model
predictions = node2vec_model.predict(test_nodes)

# Evaluate the predictions using actual labels
true_labels = np.digitize([G.nodes[node]["followers"] for node in test_nodes], edges) - 1
print(classification_report(true_labels, predictions))

