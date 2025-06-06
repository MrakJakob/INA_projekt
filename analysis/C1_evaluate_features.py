import pandas as pd
import numpy as np
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
features_path = f"{output_dir}/all_node_features.csv"
results_path = f"{output_dir}/feature_classification_metrics.csv"

# Load node data
df = pd.read_csv(features_path)

# Filter for playlists only
df = df[df["type"] == "playlist"].copy()
df["label"] = (df["followers"] > 10).astype(int)

# Features to evaluate
exclude_cols = {"id", "original_id", "name", "followers", "type", "label", "num_tracks", "num_albums", "num_artists", "duration_ms", "collaborative"}
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != "object"]

results = []

for feature in feature_cols:
    scores = df[[feature, "label"]].dropna()
    y_true = scores["label"].values
    y_score = scores[feature].values

    thresholds = np.linspace(y_score.min(), y_score.max(), num=100)
    for t in thresholds:
        y_pred = (y_score > t).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        TPR = recall
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        specificity = TNR
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0.0

        results.append({
            "feature": feature,
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TPR": TPR,
            "FPR": FPR,
            "TNR": TNR,
            "FNR": FNR,
            "specificity": specificity,
            "accuracy": accuracy
        })

results_df = pd.DataFrame(results)
results_df.to_csv(results_path, index=False)
print("Saved evaluation metrics to feature_classification_metrics.csv")