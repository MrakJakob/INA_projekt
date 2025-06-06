import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import precision_recall_curve, average_precision_score

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
features_path = f"{output_dir}/all_node_features.csv"
plot_dir = os.path.join(output_dir, "feature_plots")

# Load node data
df = pd.read_csv(features_path)
df = df[df["type"] == "playlist"].copy()
df["label"] = (df["followers"] > 10).astype(int)

exclude_cols = {"id", "original_id", "name", "followers", "type", "label", "num_tracks", "num_albums", "num_artists", "duration_ms", "collaborative"}
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != "object"]

plt.figure(figsize=(10, 8))

for feature in feature_cols:
    scores = df[[feature, "label"]].dropna()
    if scores["label"].nunique() < 2:
        continue
    precision, recall, _ = precision_recall_curve(scores["label"], scores[feature])
    avg_prec = average_precision_score(scores["label"], scores[feature])
    plt.plot(recall, precision, label=f"{feature} (AP = {avg_prec:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for Features Predicting Playlist Follower > 10")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{plot_dir}/pr_curves.png")
print("Saved PR curves to pr_curves.png")