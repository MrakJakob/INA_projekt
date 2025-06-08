import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# --- Load the GraphML file ---
G = nx.read_graphml("g_uniform45k/uniformly_sampled_playlist_tracks_45000.graphml")  # replace with your filename

# Extract playlist follower counts
follower_counts = []
for node, data in G.nodes(data=True):
    if data.get('type') == 'playlist':
        try:
            followers = int(data.get('followers', 0))
            follower_counts.append(followers)
        except ValueError:
            continue

# Frequency count
frequency = Counter(follower_counts)
x = np.array(sorted(frequency.keys()))
y = np.array([frequency[k] for k in x])

# Filter out 0s for log-log plot
mask = (x > 0) & (y > 0)
x, y = x[mask], y[mask]

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, edgecolors='k', s=30)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Followers")
plt.ylabel("Playlists")
plt.title("Distribution of Playlists by Follower Count (log-log)")
plt.tight_layout()
plt.savefig("powerlaw_clean_plot.png", dpi=300)
plt.show()