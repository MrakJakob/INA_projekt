from pathlib import Path
from matplotlib import pyplot as plt
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_curve,
)
from typing import Dict, List, Optional
from models import BaseModel


class PlaylistClassifier(nn.Module):
    """Improved neural network with better architecture for balanced data"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.4):
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
        hidden_dims=[128, 64, 32],
        dropout_rate=0.4,
        num_epochs=200,
        learning_rate=0.001,
        batch_size=64,
        imbalance_strategy=None  # weighted loss
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
        #features_df: pd.DataFrame = None,
    ):
        """Initialize and preprocess graph data"""
        self.G = G
        self.projection = projection
        self.test_nodes = test_nodes
        self.edges = np.array(edges).astype(int)
        self.dir = Path(G.graph.get("dir", ""))
        features_file_path = Path(G.graph.get("features_file", ""))
        self.features = pd.read_csv(features_file_path)

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
                # "avg_track_duration_ms",
                # "closeness_centrality",
                # "num_tracks",
                # "unique_albums",
                "collaborative",
                # "clustering_coeff",
                "unique_artists",
                "rare_tracks_count",
                "common_tracks_count",
                "track_diversity_hhi",
            ]
        ]

    def prepare_features(self, nodes: np.ndarray, buckets: np.ndarray):
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

    def predict(self, test_nodes: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
        """Make predictions for test nodes"""
        if self.model is None:
            raise ValueError("Model must be trained first!")

        # Extract features for test nodes
        test_loader = self.prepare_features_test(test_nodes)
        self.test_labels = test_labels

        self.model.eval()
        val_predictions = []
        probabilities = []

        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())

        all_probabilities = np.array(probabilities)
    
        accuracy = accuracy_score(self.test_labels, val_predictions)
        
        f1 = f1_score(self.test_labels, val_predictions, average="binary")
        precision, recall, f1, support = precision_recall_fscore_support(
            self.test_labels, val_predictions, average=None
        )
        weighted_f1 = f1_score(self.test_labels, val_predictions, average="weighted")
        macro_f1 = f1_score(self.test_labels, val_predictions, average="macro")

        results = {
            "accuracy": accuracy,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "predictions": val_predictions,
            "true_labels": self.test_labels,
            "probabilities": np.array(all_probabilities),
            "classification_report": classification_report(self.test_labels, val_predictions),
            "confusion_matrix": confusion_matrix(self.test_labels, val_predictions),
            # "auc_score": auc_score,
            # "feature_importances": feature_importances,
            # "optimal_threshold": optimal_threshold,
            # "optimal_accuracy": optimal_accuracy,
            # "optimal_f1": optimal_f1,

        }

        # Plot detailed results
        self.plot_detailed_results(results)

        return np.array(val_predictions), all_probabilities

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
        history = {"train_loss": [], "train_acc": [], "train_f1": []}

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
            f1 = f1_score(
                train_labels, train_predictions, average="weighted"
            )

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc * 100)
            history["train_f1"].append(f1)
            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_acc*100:.2f}%"
                )
        self.save_training_history(history)

    def save_training_history(self, history: Dict):
        """Plot training history"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        # Loss plot
        ax1.plot(history["train_loss"])
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(history["train_acc"])
        ax2.set_title("Training Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True)

        # F1 Score plot
        ax3.plot(history["train_f1"])
        ax3.set_title("Training F1 Score")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1 Score")
        ax3.grid(True)

        # save figure to dir 

        plt.savefig(f"{self.dir}/images/neural_network_training_history.png", dpi=300)

    
    def plot_detailed_results(self, results: Dict):
        """Plot comprehensive results for imbalanced classification"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Confusion Matrix
        cm = results["confusion_matrix"]
        labels = ["Low Followers", "High Followers"]

        # Plot with matplotlib
        im = ax1.imshow(cm, cmap="Blues")

        # Title and labels
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(labels)
        ax1.set_yticklabels(labels)
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

        # Annotate the cells with the numeric values
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        # Optional: Add colorbar
        plt.colorbar(im, ax=ax1)
        ax1.set_title("Confusion Matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")

        # Per-class metrics
        classes = ["Low Followers", "High Followers"]
        metrics = ["Precision", "Recall", "F1-Score"]

        precision = results["precision_per_class"]
        recall = results["recall_per_class"]
        f1 = results["f1_per_class"]

        x = np.arange(len(classes))
        width = 0.25

        ax2.bar(x - width, precision, width, label="Precision", alpha=0.8)
        ax2.bar(x, recall, width, label="Recall", alpha=0.8)
        ax2.bar(x + width, f1, width, label="F1-Score", alpha=0.8)

        ax2.set_xlabel("Classes")
        ax2.set_ylabel("Score")
        ax2.set_title("Per-Class Metrics")
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes)
        ax2.legend()
        ax2.set_ylim(0, 1)

        # Probability distribution for class 1 (high followers)
        probs_class_1 = results["probabilities"][:, 1]
        true_labels = results["true_labels"]

        ax3.hist(
            probs_class_1[np.array(true_labels) == 0],
            bins=30,
            alpha=0.7,
            label="True Low Followers",
            density=True,
        )
        ax3.hist(
            probs_class_1[np.array(true_labels) == 1],
            bins=30,
            alpha=0.7,
            label="True High Followers",
            density=True,
        )
        ax3.set_xlabel("Predicted Probability (High Followers)")
        ax3.set_ylabel("Density")
        ax3.set_title("Probability Distribution")
        ax3.legend()

        # ROC-like plot (prediction confidence vs accuracy)
        sorted_indices = np.argsort(probs_class_1)[::-1]
        sorted_probs = probs_class_1[sorted_indices]
        sorted_labels = np.array(true_labels)[sorted_indices]

        cumulative_precision = np.cumsum(sorted_labels) / np.arange(
            1, len(sorted_labels) + 1
        )

        ax4.plot(np.arange(len(sorted_labels)), cumulative_precision)
        ax4.set_xlabel("Number of Predictions (sorted by confidence)")
        ax4.set_ylabel("Precision")
        ax4.set_title("Precision vs Confidence")
        ax4.grid(True)

        # save figure to file
        plt.savefig(f"{self.dir}/images/neural_network_detailed_results.png", dpi=300)