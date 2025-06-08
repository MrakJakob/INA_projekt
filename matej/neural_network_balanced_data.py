from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, Tuple


class PlaylistDataset(Dataset):
    """Custom Dataset for playlist features"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ImprovedPlaylistClassifier(nn.Module):
    """Improved neural network with better architecture for balanced data"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.4):
        super(ImprovedPlaylistClassifier, self).__init__()

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


class EnhancedNeuralClassifier:
    """Enhanced neural network classifier with better training strategies"""

    def __init__(self, hidden_dims=[256, 128, 64], dropout_rate=0.4):
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        num_epochs: int = 200,
        learning_rate: float = 0.001,
    ) -> Dict:
        """Enhanced training with validation, early stopping, and learning rate scheduling"""

        # Get input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]

        # Initialize model
        self.model = ImprovedPlaylistClassifier(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        # Use standard CrossEntropy for balanced data

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.01
        )

        # Learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,
        )

        history = {"train_loss": [], "train_acc": [], "train_f1": []}
        best_val_acc = 0
        patience = 20
        patience_counter = 0

        print(f"Training on {self.device}")
        print(
            f"Model architecture: {input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> 2"
        )

        for epoch in range(num_epochs):
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
            f1 = f1_score(train_labels, train_predictions, average="weighted")
            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc * 100)
            history["train_f1"].append(f1)

            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                # history["val_loss"].append(val_loss)
                # history["val_acc"].append(val_acc * 100)

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), "best_model.pth")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Load best model
                    self.model.load_state_dict(torch.load("best_model.pth"))
                    break

            if (epoch + 1) % 20 == 0:
                if val_loader:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                        f"Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%"
                    )
                else:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                        f"Train Acc: {train_acc*100:.2f}%"
                    )

        return history

    def _validate(self, val_loader, criterion):
        """Validation helper function"""
        self.model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_predictions)

        return avg_val_loss, val_acc

    def evaluate_model(self, test_loader: DataLoader, metadata) -> Dict:
        """Evaluate the trained neural network model"""
        if self.model is None:
            raise ValueError("Model must be trained first!")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_weighted = f1_score(all_labels, all_predictions, average="weighted")
        f1_macro = f1_score(all_labels, all_predictions, average="macro")
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
        recall = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted"
        )

        # Calculate optimal threshold using Youden's J statistic
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities[:, 1])
        youden_index = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_index)]

        # Get predictions with optimal threshold
        optimal_predictions = (all_probabilities[:, 1] >= optimal_threshold).astype(int)
        optimal_accuracy = accuracy_score(all_labels, optimal_predictions)
        optimal_f1 = f1_score(all_labels, optimal_predictions, average="weighted")

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

        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )

        results = {
            "accuracy": accuracy,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support,
            "optimal_threshold_accuracy": optimal_accuracy,
            "weighted_f1": f1_weighted,
            "macro_f1": f1_macro,
            "optimal_f1": optimal_f1,
            "auc_score": auc_score,
            "optimal_threshold": optimal_threshold,
            "predictions": all_predictions,
            "optimal_predictions": optimal_predictions,
            "true_labels": all_labels,
            "probabilities": all_probabilities,
            "classification_report": classification_report(all_labels, all_predictions),
            "optimal_classification_report": classification_report(
                all_labels, optimal_predictions
            ),
            "confusion_matrix": confusion_matrix(all_labels, all_predictions),
            "optimal_confusion_matrix": confusion_matrix(
                all_labels, optimal_predictions
            ),
            "feature_importances": feature_importances,
        }

        print(
            f"Standard Threshold (0.5) - Accuracy: {accuracy:.4f}, F1: {f1_weighted:.4f}, Recall: {recall[1]:.4f}, Precision: {precision[1]:.4f}"
        )

        # print(
        #     f"Optimal Threshold ({optimal_threshold:.4f}) - Accuracy: {optimal_accuracy:.4f}, F1: {optimal_f1:.4f}"
        # )
        print(f"AUC Score: {auc_score:.4f}")

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

        plt.tight_layout()
        plt.show()


class EnsembleClassifier:
    """Ensemble of multiple classifiers for better performance"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.ensemble = None

    def train_model(self, X_train, y_train):
        """Train ensemble of different classifiers"""

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Define individual classifiers
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
        )

        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
        )

        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        svm = SVC(C=1.0, kernel="rbf", probability=True, random_state=42)

        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ("rf", rf),
                ("gb", gb),
                ("xgb", xgb_clf),
                ("lr", lr),
                ("svm", svm),
            ],
            voting="soft",  # Use probability voting
        )

        print("Training ensemble classifier...")
        self.ensemble.fit(X_train_scaled, y_train)

        return {"message": "Ensemble training completed"}

    def evaluate_model(self, X_test, y_test):
        """Evaluate the ensemble model"""
        X_test_scaled = self.scaler.transform(X_test)

        # Predictions
        predictions = self.ensemble.predict(X_test_scaled)
        probabilities = self.ensemble.predict_proba(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")
        auc = roc_auc_score(y_test, probabilities[:, 1])

        results = {
            "accuracy": accuracy,
            "weighted_f1": f1,
            "auc_score": auc,
            "predictions": predictions,
            "probabilities": probabilities,
            "classification_report": classification_report(y_test, predictions),
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }

        # Individual classifier performance
        print("\nIndividual Classifier Performance:")
        for name, clf in self.ensemble.named_estimators_.items():
            pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, pred)
            print(f"{name}: {acc:.4f}")

        return results


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


# Usage example:
def compare_classifiers(
    train_loader, test_loader, X_train, y_train, X_test, y_test, metadata
):
    """Compare different classifier approaches"""

    print("=== Enhanced Neural Network ===")
    nn_classifier = EnhancedNeuralClassifier(
        hidden_dims=[128, 64, 32], dropout_rate=0.4
    )
    nn_history = nn_classifier.train_model(
        train_loader, num_epochs=200, learning_rate=0.001
    )
    nn_results = nn_classifier.evaluate_model(test_loader, metadata)

    print(f"Neural Network AUC: {nn_results['auc_score']:.4f}")
    print(f"Neural Network Accuracy: {nn_results['accuracy']:.4f}")

    print(nn_results["feature_importances"])

    nn_classifier.plot_training_history(nn_history)
    nn_classifier.plot_detailed_results(nn_results)

    # print("\n=== Ensemble Classifier ===")
    # ensemble = EnsembleClassifier()
    # ensemble.train_model(X_train, y_train)
    # ensemble_results = ensemble.evaluate_model(X_test, y_test)

    # print(f"Ensemble Accuracy: {ensemble_results['accuracy']:.4f}")
    # print(f"Ensemble AUC: {ensemble_results['auc_score']:.4f}")

    # return nn_classifier, ensemble, nn_results, ensemble_results


def prepare_data(
    train_labels_path: str, test_labels_path: str, features_path: str
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

    scaler = StandardScaler()
    # Scale features BEFORE resampling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create datasets
    train_dataset = PlaylistDataset(X_train_scaled, y_train)
    test_dataset = PlaylistDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Store metadata
    metadata = {
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "train_size": len(X_train_scaled),
        "test_size": len(test_data),
        "train_class_distribution": np.bincount(y_train),
        "test_class_distribution": np.bincount(y_test),
    }

    return train_loader, test_loader, X_train, y_train, X_test, y_test, metadata


if __name__ == "__main__":

    graph_dir = "matej/graphs/5K_playlists/balanced"
    feature_dir = "matej/graphs/5K_playlists/balanced/features"

    train_loader, test_loader, X_train, y_train, X_test, y_test, metadata = (
        prepare_data(
            train_labels_path=f"{graph_dir}/5000_playlists_balanced_train.csv",
            test_labels_path=f"{graph_dir}/5000_playlists_balanced_test.csv",
            features_path=f"{feature_dir}/5000_playlists_balanced_features.csv",
        )
    )

    # Compare classifiers
    compare_classifiers(
        train_loader, test_loader, X_train, y_train, X_test, y_test, metadata
    )
