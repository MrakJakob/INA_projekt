import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import warnings

warnings.filterwarnings("ignore")

metadata = None


class PlaylistDataset(Dataset):
    """Custom Dataset for playlist features"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PlaylistClassifier(nn.Module):
    """Neural Network for playlist classification"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ):
        super(PlaylistClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



DATASET = "2000" 

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(
        self,
        alpha=1,
        gamma=2,
        reduction="mean",
        imbalance_strategy: str = "weighted_loss",
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.3,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.imbalance_strategy = imbalance_strategy
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.class_weights = None

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def train_model(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Dict:
        """Train the neural network with class imbalance handling"""

        # Get input dimension from first batch
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]

        # Initialize model
        self.model = PlaylistClassifier(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Choose loss function based on strategy
        if self.imbalance_strategy == "focal_loss":
            criterion = FocalLoss(alpha=1, gamma=2)
        elif self.imbalance_strategy == "weighted_loss":
            class_weights_tensor = torch.FloatTensor(self.class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        # Training history
        history = {"train_loss": [], "train_acc": [], "train_f1": []}

        print(f"Training on {self.device}")
        print(
            f"Model architecture: {input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> 2"
        )
        print(f"Using {self.imbalance_strategy} strategy")

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            all_predictions = []
            all_labels = []

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

            avg_loss = train_loss / len(train_loader)
            accuracy = accuracy_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions, average="weighted")

            history["train_loss"].append(avg_loss)
            history["train_acc"].append(accuracy * 100)
            history["train_f1"].append(f1)

            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy*100:.2f}%, F1: {f1:.4f}"
                )

        return history

    def evaluate_model(self, test_loader: DataLoader, metadata) -> Dict:
        """Evaluate the trained model with comprehensive metrics for imbalanced data"""
        if self.model is None:
            raise ValueError("Model must be trained first!")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        auc_score = 0.0

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                probs_class_1 = np.array(all_probabilities)[:, 1]

            # Calculate AUC
            auc_score = roc_auc_score(all_labels, probs_class_1)
        all_probabilities = np.array(all_probabilities)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities[:, 1])
        youden_index = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_index)]
        print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")
        # adjust threshold using Youden's J statistic
        # all_predictions = all_probabilities[:, 1] >= optimal_threshold

        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        weighted_f1 = f1_score(all_labels, all_predictions, average="weighted")
        macro_f1 = f1_score(all_labels, all_predictions, average="macro")

        # calculate feature importances
        feature_names = metadata["feature_columns"]
        baseline_auc = auc_score
        X_list = []
        y_list = []

        for X_batch, y_batch in test_loader:
            X_list.append(X_batch)
            y_list.append(y_batch)

        X_test = torch.cat(X_list, dim=0)
        y_test = torch.cat(y_list, dim=0)
        feature_importances = permutation_importance(
            self.model, X_test, y_test, baseline_auc, feature_names
        )

        results = {
            "accuracy": accuracy,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "predictions": all_predictions,
            "true_labels": all_labels,
            "probabilities": np.array(all_probabilities),
            "classification_report": classification_report(all_labels, all_predictions),
            "confusion_matrix": confusion_matrix(all_labels, all_predictions),
            "auc_score": auc_score,
            "feature_importances": feature_importances,
        }

        return results

    def plot_training_history(self, history: Dict):
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

        # save figure to file
        plt.savefig(f"playlist_classification_training_history_{DATASET}.png", dpi=300)
        plt.tight_layout()
        plt.show()

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
        plt.savefig(f"playlist_classification_results_{DATASET}.png", dpi=300)
        plt.tight_layout()
        plt.show()

    def predict_playlist_quality(
        self, playlist_features: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict playlist quality for new playlists

        Args:
            playlist_features: Array of playlist features (scaled)
        Returns:
            Predictions (0 for low quality, 1 for high quality)
        """
        if self.model is None:
            raise ValueError("Model must be trained first!")

        self.model.eval()
        features_scaled = self.scaler.transform(playlist_features)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def prepare_data(
        self, train_labels_path: str, test_labels_path: str, features_path: str
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Prepare data for training with class imbalance handling

        Args:
            train_labels_path: Path to training labels CSV
            test_labels_path: Path to test labels CSV
            features_path: Path to features CSV
        """
        # Load data
        train_labels = pd.read_csv(train_labels_path)
        test_labels = pd.read_csv(test_labels_path)
        features_df = pd.read_csv(features_path)

        print(
            f"Loaded {len(train_labels)} training samples, {len(test_labels)} test samples"
        )
        print(f"Features shape: {features_df.shape}")

        # Merge labels with features
        train_data = train_labels.merge(
            features_df, left_on="nodes", right_on="playlist_id", how="inner"
        )
        test_data = test_labels.merge(
            features_df, left_on="nodes", right_on="playlist_id", how="inner"
        )

        print(f"After merging - Train: {len(train_data)}, Test: {len(test_data)}")

        # Select feature columns (exclude identifiers and target, and columns that did not contribute to the model)
        feature_cols = [
            col
            for col in features_df.columns
            if col
            not in [
                "playlist_id",
                "name",
                "followers",
                "betweenness_centrality",
                #"avg_track_duration_ms",
                "closeness_centrality",
                #"num_tracks",
                #"unique_albums",
                #"collaborative",
                #"clustering_coeff",
                #"unique_artists",
                'rare_tracks_count',
                'common_tracks_count',
                'track_diversity_hhi',
            ]
        ]

        # Handle missing values
        train_data[feature_cols] = train_data[feature_cols].fillna(
            train_data[feature_cols].median()
        )
        test_data[feature_cols] = test_data[feature_cols].fillna(
            train_data[feature_cols].median()
        )

        # Prepare features and labels
        X_train = train_data[feature_cols].values
        y_train = train_data["buckets"].values
        X_test = test_data[feature_cols].values
        y_test = test_data["buckets"].values

        # Scale features BEFORE resampling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Handle class imbalance
        X_train_resampled, y_train_resampled = (
            X_train_scaled,
            y_train,
        )  # self.handle_class_imbalance(X_train_scaled, y_train)

        # Calculate class weights for loss function
        self.class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        print(f"Class weights: {self.class_weights}")

        # Create datasets
        train_dataset = PlaylistDataset(X_train_resampled, y_train_resampled)
        test_dataset = PlaylistDataset(X_test_scaled, y_test)

        # Create dataloaders
        if self.imbalance_strategy == "weighted_loss":
            # Use weighted sampler
            sampler = self.create_weighted_sampler(y_train_resampled)
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Store metadata
        metadata = {
            "feature_columns": feature_cols,
            "num_features": len(feature_cols),
            "train_size": len(X_train_resampled),
            "test_size": len(test_data),
            "train_class_distribution_original": np.bincount(y_train),
            "train_class_distribution": np.bincount(y_train_resampled),
            "test_class_distribution": np.bincount(y_test),
            "class_weights": self.class_weights,
        }

        return train_loader, test_loader, metadata

    def create_weighted_sampler(self, y_train: np.ndarray) -> WeightedRandomSampler:
        """Create weighted sampler for DataLoader"""
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]

        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )


import numpy as np
from sklearn.metrics import roc_auc_score


def permutation_importance(model, X_test, y_test, baseline_metric, feature_names):
    importances = []
    X_test_np = X_test.cpu().numpy().copy()
    # assuming numpy array or pandas DataFrame

    for i, feature in enumerate(feature_names):
        X_permuted = X_test_np.copy()
        np.random.shuffle(X_permuted[:, i])  # shuffle feature i
        with torch.no_grad():
            outputs = model(torch.tensor(X_permuted).float())
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        metric_value = roc_auc_score(y_test, probs)
        importance = baseline_metric - metric_value
        importances.append((feature, importance))

    importances.sort(key=lambda x: x[1], reverse=True)
    return importances


# Example usage
def main():
    """Example of how to use the PlaylistPopularityPredictor"""

    # Initialize predictor
    # predictor = PlaylistPopularityPredictor(hidden_dims=[128, 64, 32], dropout_rate=0.3)
    predictor = FocalLoss(
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3,
        imbalance_strategy="weighted_loss",  # Change to 'weighted_loss', 'focal_loss'
    )

    dir = "network_analysis_features"
    graph_dir = "network_analysis_features/graphs"
    feature_dir = "network_analysis_features/features"
    # Prepare data (you'll need to provide the actual file paths)
    train_loader, test_loader, metadata = predictor.prepare_data(
        train_labels_path=f"{graph_dir}/uniformly_sampled_playlist_tracks_{DATASET}_train.csv",
        test_labels_path=f"{graph_dir}/uniformly_sampled_playlist_tracks_{DATASET}_test.csv",
        features_path=f"{feature_dir}/playlist_graph_features_{DATASET}.csv",
    )

    print("Data preparation complete!")
    print(f"Features used: {metadata['feature_columns']}")
    print(f"Training class distribution: {metadata['train_class_distribution']}")
    print(f"Test class distribution: {metadata['test_class_distribution']}")

    # Train model
    print("\nStarting training...")
    history = predictor.train_model(train_loader, num_epochs=200, learning_rate=0.001)

    # Evaluate model
    print("\nEvaluating model...")
    results = predictor.evaluate_model(test_loader, metadata)

    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results["classification_report"])
    print("\nAUC Score:", results["auc_score"])

    print("\nFeature Importances:")
    print(results["feature_importances"])

    # Plot results
    predictor.plot_training_history(history)
    predictor.plot_detailed_results(results)

    return predictor, results




if __name__ == "__main__":
    # Uncomment to run
    predictor, results = main()
    pass
