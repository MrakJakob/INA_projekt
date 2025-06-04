import numpy as np
import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.manifold import SpectralEmbedding
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer
from utils import get_playlists_tracks

class BaseModel():

    def __init__(self):
        # set hyperparams here
        self.name = "Base Model"

    def init_data(self, G: nx.Graph, projection: nx.Graph, 
                    test_nodes: np.ndarray[str], edges: np.ndarray[int]):
        # preprocess any type of graph you need here
        # if uses labels (follower counts) at inference, 
        # make sure to remove test nodes in training !!
        pass

    def train(self, train_nodes: np.ndarray[str], train_buckets: np.ndarray[int]):
        # train the model here - return nothing
        pass

    def predict(self, test_nodes: np.ndarray[str]) -> np.ndarray[int]:
        # return predictions here
        pass    

class Majority(BaseModel):

    def init_data(self, G, projection, test_nodes, edges):
        pass

    def train(self, train_nodes, train_buckets):
        vals, counts = np.unique(train_buckets, return_counts=True)
        self.majority = vals[np.argmax(counts)]

    def predict(self, test_nodes):
        return (np.ones_like(test_nodes) * self.majority).astype(int)

class NeighborMean(BaseModel):

    def init_data(self, G, projection, test_nodes, edges):
        self.proj = projection
        # not needed here, but as demo:
        self.proj_train = self.proj.copy()
        self.proj_train.remove_nodes_from(test_nodes)
        self.edges = np.array(edges).astype(int)

    def predict(self, test_nodes):
        predictions = []
        for n in test_nodes:
            neighbors = list(self.proj.neighbors(n))
            #predictions.append(self.proj.nodes[n]["followers"])
            if len(neighbors) == 0:
                predictions.append(self.edges[0])
            else:
                nb_followers = np.mean([int(self.proj.nodes[nb]["followers"]) for nb in neighbors])
                #nb_followers = self.proj.nodes[n]["followers"]
                predictions.append(nb_followers)
        return np.digitize(predictions, self.edges) - 1

class TrackDegree(BaseModel):

    def __init__(self, agg="sum"):
        self.agg = np.sum if agg == "sum" else np.mean
    
    def init_data(self, G, projection, test_nodes, edges):
        self.G = G
        self.fitter = LogisticRegression()

    def _track_degs(self, nodes):
        avg_degs = []
        for n in nodes:
            track_degs = np.array([self.G.degree(nb) for nb in self.G.neighbors(n)])
            avg_degs.append(np.sum(track_degs) if len(track_degs) > 0 else 0)
        return np.array(avg_degs)
        #return np.array([self.G.degree[n] < 10 for n in nodes])


    def train(self, train_nodes, train_buckets):        
        avg_degs = self._track_degs(train_nodes)
        self.fitter.fit(avg_degs.reshape(-1, 1), train_buckets)

    def predict(self, test_nodes):
        avg_degs = self._track_degs(test_nodes)
        return self.fitter.predict(avg_degs.reshape(-1, 1))

class Spectral(BaseModel):

    def init_data(self, G, projection, test_nodes, edges):
        adj_matrix = nx.to_numpy_array(projection)
        embedder = SpectralEmbedding(n_components=16, affinity='precomputed')
        self.x = embedder.fit_transform(adj_matrix)
        self.node_to_index = {node: i for i, node in enumerate(list(projection.nodes()))}

        self.fitter = LogisticRegression()

    def train(self, train_nodes, train_buckets):
        train_idx = [self.node_to_index[n] for n in train_nodes]
        self.fitter.fit(self.x[train_idx], train_buckets)

    def predict(self, test_nodes):
        test_idx = [self.node_to_index[n] for n in test_nodes]
        return self.fitter.predict(self.x[test_idx])    

class NameEmbedding(BaseModel):

    def init_data(self, G, projection, test_nodes, edges):
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        playlists, tracks = get_playlists_tracks(G)
        pl_names = [G.nodes[n]['name'] for n in playlists]
        self.pl_emb = self.text_embedder.encode(pl_names)
        self.node_to_index = {n: i for i, n in enumerate(playlists)}

        self.x = self.pl_emb
        self.fitter = LogisticRegression()

    def train(self, train_nodes, train_buckets):
        train_idx = [self.node_to_index[n] for n in train_nodes]
        self.fitter.fit(self.x[train_idx], train_buckets)

    def predict(self, test_nodes):
        test_idx = [self.node_to_index[n] for n in test_nodes]
        return self.fitter.predict(self.x[test_idx])    

                
    
class PlaylistClassifier(nn.Module):
    """Improved neural network with better architecture for balanced data"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.4):
        super(PlaylistClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralClassifier(BaseModel):
    """Enhanced neural network classifier inheriting from BaseModel"""

    def __init__(self, hidden_dims=[256, 128, 64], dropout_rate=0.4, 
                 num_epochs=200, learning_rate=0.001, batch_size=64):
        super().__init__()
        # Set hyperparams here
        self.name = "Enhanced Neural Network Model"
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = None
        
        # Data storage
        self.G = None
        self.projection = None
        self.test_nodes = None
        self.edges = None

    def init_data(self, G: nx.Graph, projection: nx.Graph, 
                  test_nodes: np.ndarray, edges: np.ndarray):
        """Initialize and preprocess graph data"""
        self.G = G
        self.projection = projection
        self.test_nodes = test_nodes
        self.edges = np.array(edges).astype(int)
        
        # Create training projection (remove test nodes)
        self.proj_train = self.projection.copy()
        self.proj_train.remove_nodes_from(test_nodes)
        
        # Initialize feature extractor
        self.feature_extractor = GraphFeatureExtractor(self.G, self.projection, self.proj_train)

    def train(self, train_nodes: np.ndarray, train_buckets: np.ndarray):
        """Train the neural network model"""
        # Extract features for training nodes
        train_features = self.feature_extractor.extract_features(train_nodes)
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Convert to tensors
        X_train = torch.FloatTensor(train_features_scaled)
        y_train = torch.LongTensor(train_buckets)
        
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train the model
        self._train_neural_network(train_loader)

    def predict(self, test_nodes: np.ndarray) -> np.ndarray:
        """Make predictions for test nodes"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        # Extract features for test nodes
        test_features = self.feature_extractor.extract_features(test_nodes)
        
        # Scale features using fitted scaler
        test_features_scaled = self.scaler.transform(test_features)
        
        # Convert to tensor and predict
        X_test = torch.FloatTensor(test_features_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predictions = torch.max(outputs, 1)
            
        return predictions.cpu().numpy()

    def _train_neural_network(self, train_loader: DataLoader):
        """Internal method to train the neural network"""
        # Get input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]

        # Initialize model
        self.model = PlaylistClassifier(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Use standard CrossEntropy for balanced data
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        # Learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=self.num_epochs,
            pct_start=0.1,
        )

        print(f"Training on {self.device}")
        print(
            f"Model architecture: {input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> 2"
        )

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_labels = []

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_predictions.extend(predicted.cpu().numpy())
                train_labels.extend(batch_labels.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            train_acc = accuracy_score(train_labels, train_predictions)

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_acc*100:.2f}%"
                )
    






