import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
community_path = os.path.join(output_dir, "community_labels.csv")
plot_path = os.path.join(output_dir, "feature_plots")
os.makedirs(plot_path, exist_ok=True)

df = pd.read_csv(community_path)

df = df[df["type"] == "playlist"]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="louvain", y="followers")
plt.yscale("log")
plt.title("Playlist Followers by Community (log scale)")
plt.xlabel("Louvain Community")
plt.ylabel("Number of Followers (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "followers_by_community_log.png"))
plt.close()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="louvain", order=df['louvain'].value_counts().index)
plt.title("Playlist Count per Louvain Community")
plt.xlabel("Louvain Community")
plt.ylabel("Count of Playlists")
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "playlist_count_per_community.png"))
plt.close()

print("Saved community-based plots to feature_plots/")
