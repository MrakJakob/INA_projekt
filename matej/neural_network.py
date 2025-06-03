import numpy as np
import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
)
from typing import Dict, List, Optional
from models import BaseModel


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

class PlaylistDataset(Dataset):
    """Custom Dataset for playlist features"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NeuralClassifier(BaseModel):
    """Enhanced neural network classifier inheriting from BaseModel"""

    def __init__(
        self,
        hidden_dims=[256, 128, 64, 32],
        dropout_rate=0.3,
        num_epochs=200,
        learning_rate=0.001,
        batch_size=64,
    ):
        super().__init__()
        # Set hyperparams here
        self.name = "Enhanced Neural Network Model"
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = None

        # Model components
        self.model = None
        self.scaler = StandardScaler()


        # Data storage
        self.G = None
        self.projection = None
        self.test_nodes = None
        self.edges = None

    def init_data(
        self,
        G: nx.Graph,
        projection: nx.Graph,
        test_nodes: np.ndarray,
        edges: np.ndarray,
        features_df: pd.DataFrame = None,
    ):
        """Initialize and preprocess graph data"""
        self.G = G
        self.projection = projection
        self.test_nodes = test_nodes
        self.edges = np.array(edges).astype(int)
        self.features = features_df

        # Create training projection (remove test nodes)
        self.proj_train = self.projection.copy()
        self.proj_train.remove_nodes_from(test_nodes)

    def filter_features(self):
        return [
            col
            for col in self.features.columns
            if col
            not in [
                "playlist_id",
                "name",
                "followers",
                # "betweenness_centrality",
                "avg_track_duration_ms",
                # "closeness_centrality",
                # "num_tracks",
                # "unique_albums",
                # "collaborative",
                # "clustering_coeff",
                # "unique_artists",
                # "rare_tracks_count",
                # "common_tracks_count",
                # "track_diversity_hhi",
            ]
        ]

    def prepare_features(
        self, nodes: np.ndarray, buckets: np.ndarray
    ):
        train_labels = pd.DataFrame({"nodes": nodes, "buckets": buckets})

        train_data = train_labels.merge(
            self.features, left_on="nodes", right_on="playlist_id", how="inner"
        )

        feature_cols = self.filter_features()

        X = train_data[feature_cols].values
        y = train_data["buckets"].values

        
        X_scaled = self.scaler.fit_transform(X)
        
        dataset = PlaylistDataset(X_scaled, y)

        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        return loader

    def prepare_features_test(self, nodes: np.ndarray):
        test = pd.DataFrame({"nodes": nodes})
        test_data = test.merge(
            self.features, left_on="nodes", right_on="playlist_id", how="inner"
        )
        feature_cols = self.filter_features()
        X = test_data[feature_cols].values
        X_scaled = self.scaler.transform(X)

        dataset = PlaylistDataset(X_scaled, np.zeros(len(X_scaled)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        return loader

    def train(self, train_nodes: np.ndarray, train_buckets: np.ndarray):
        """Train the neural network model"""
        # Extract features for training nodes
        train_loader = self.prepare_features(train_nodes, train_buckets)
        # Train the model
        self._train_neural_network(train_loader)

    def predict(self, test_nodes: np.ndarray) -> np.ndarray:
        """Make predictions for test nodes"""
        if self.model is None:
            raise ValueError("Model must be trained first!")

        # Extract features for test nodes
        test_loader = self.prepare_features_test(test_nodes)

        self.model.eval()
        val_predictions = []

        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())

        return np.array(val_predictions)
        

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
        self.criterion = nn.CrossEntropyLoss()
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
                loss = self.criterion(outputs, batch_labels)
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
