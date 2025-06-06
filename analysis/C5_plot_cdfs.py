import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
features_path = f"{output_dir}/all_node_features.csv"
plot_dir = os.path.join(output_dir, "feature_plots")
os.makedirs(plot_dir, exist_ok=True)

# Load playlist-only features
df = pd.read_csv(features_path)
df = df[df["type"] == "playlist"]

exclude_cols = {
    "id", "original_id", "name", "followers", "type", "label",
    "num_tracks", "num_albums", "num_artists", "duration_ms", "collaborative"
}
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != "object"]

# One plot per feature
for feature in feature_cols:
    # raw = df[feature].dropna()
    # values = ((raw - raw.min()) / (raw.max() - raw.min())).sort_values()
    values = df[feature].dropna().sort_values()
    cdf_y = (values.rank(method="first") - 1) / (len(values) - 1)

    plt.figure(figsize=(8, 5))
    plt.plot(values, cdf_y, label="CDF")
    plt.title(f"Cumulative Distribution for {feature}")
    plt.xlabel(feature)
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"cdf_{feature}.png"))
    plt.close()

# Combined CDF plot
plt.figure(figsize=(10, 6))
for feature in feature_cols:
    raw = df[feature].dropna()
    values = ((raw - raw.min()) / (raw.max() - raw.min())).sort_values()
    cdf_y = (values.rank(method="first") - 1) / (len(values) - 1)
    plt.plot(values, cdf_y, label=feature)

plt.title("CDFs of All Playlist Features")
plt.xlabel("Feature Value")
plt.ylabel("Cumulative Probability")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "cdf_all_features.png"))
plt.close()

print("Saved individual and combined CDF plots.")
