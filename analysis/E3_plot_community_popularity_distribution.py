import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
community_path = os.path.join(output_dir, "community_labels.csv")
plot_path = os.path.join(output_dir, "feature_plots")
os.makedirs(plot_path, exist_ok=True)

df = pd.read_csv(community_path)
df = df[df["type"] == "playlist"].copy()
df["is_popular"] = (df["followers"] > 10).astype(int)

counts = df.groupby(["louvain", "is_popular"]).size().unstack(fill_value=0)
percentages = counts.div(counts.sum(axis=1), axis=0)

ax = percentages[[0, 1]].rename(columns={0: "Unpopular", 1: "Popular"}).plot(
    kind="bar",
    stacked=True,
    color=["red", "green"],
    edgecolor="black",
    figsize=(14, 6)
)

community_sizes = counts.sum(axis=1)
for idx, size in enumerate(community_sizes):
    ax.text(idx, 1.02, str(size), ha='center', va='bottom', fontsize=8, rotation=90)

plt.ylabel("Percentage within Community")
plt.xlabel("Louvain Community")
plt.title("Popularity Composition per Community (Followers > 10)")
plt.legend(title="Playlist Type", loc="lower right")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "community_popularity_distribution.png"))
plt.close()

print("Saved popularity distribution plot to feature_plots/")
