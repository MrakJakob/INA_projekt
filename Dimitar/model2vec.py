import networkx as nx
from collections import defaultdict
from node2vec import Node2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matej.models import Node2VecModel



import networkx as nx
import numpy as np
from sklearn.metrics import classification_report
from matej.models import Node2VecModel

# Load the graph
graph_path = "F:/INA-Project/INA_projekt/matej/graphs/5K_balanced/5000_playlists_balanced.graphml"
G = nx.read_graphml(graph_path)

# Initialize the Node2VecModel
node2vec_model = Node2VecModel(dimensions=128, walk_length=40, num_walks=10, p=1, q=2)

# Prepare test nodes and edges (example placeholders)
test_nodes = np.array([node for node in G.nodes if G.nodes[node].get("type") == "playlist"])
edges = np.array([1, 10, 100])  # Example edges for bucket digitization

# Initialize the model with graph data
node2vec_model.init_data(G, projection=None, test_nodes=test_nodes, edges=edges)

# Prepare training data (example placeholders)
train_nodes = test_nodes[:int(len(test_nodes) * 0.8)]  # Use 80% of nodes for training
train_buckets = np.random.randint(0, len(edges), size=len(train_nodes))  # Random buckets for training

# Train the model
node2vec_model.train(train_nodes, train_buckets)

# Predict using the model
predictions = node2vec_model.predict(test_nodes)

# Evaluate the predictions
true_labels = np.random.randint(0, len(edges), size=len(test_nodes))  # Example true labels
print(classification_report(true_labels, predictions))

