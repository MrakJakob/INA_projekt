import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import spearmanr, kendalltau

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
features_path = f"{output_dir}/all_node_features.csv"
plot_dir = os.path.join(output_dir, "feature_plots")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(features_path)
df = df[df["type"] == "playlist"]

exclude_cols = {
    "id", "original_id", "name", "followers", "type", "label",
    "num_tracks", "num_albums", "num_artists", "duration_ms", "collaborative"
}
feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != "object"]

correlations = []

for feature in feature_cols:
    x = df[feature]
    y = df["followers"]

    # Plot scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x, y=y, alpha=0.3)
    plt.xlabel(feature)
    plt.ylabel("Followers")
    plt.yscale("log")
    plt.title(f"{feature} vs Followers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"scatter_{feature}.png"))
    plt.close()

    # Compute correlations
    valid = df[[feature, "followers"]].dropna()
    rho, _ = spearmanr(valid[feature], valid["followers"])
    tau, _ = kendalltau(valid[feature], valid["followers"])

    correlations.append({
        "feature": feature,
        "spearman_rho": rho,
        "kendall_tau": tau
    })

# Save correlation results
cor_df = pd.DataFrame(correlations)
cor_df.to_csv(f"{output_dir}/feature_follower_correlations.csv", index=False)

print("Saved scatter plots and correlation results.")

# Optional: sort and visualize correlations
cor_sorted = cor_df.sort_values(by="spearman_rho", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=cor_sorted, x="spearman_rho", y="feature", color="steelblue")
plt.title("Spearman Correlation with Followers")
plt.xlabel("Spearman ρ")
plt.ylabel("Feature")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "spearman_correlation_barplot.png"))
plt.close()

print("Saved Spearman correlation bar plot.")