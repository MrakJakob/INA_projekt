
# ANALYSIS ON BASIC NETWORK FEATURES

This part of project analyzes structural graph features of playlists in the Spotify Million Playlist Dataset to predict playlist popularity (e.g., having more than 10 followers). The analysis combines GPU-accelerated, CPU-based, and graph-theoretic methods on various graph representations.

---

## 📊 GOALS

- Predict playlist popularity using structural features from graph representations.
- Compare metrics across:
  - Bipartite playlist–track graphs
  - Projected playlist–playlist graphs
  - Tripartite playlist–track–artist graphs
- Evaluate classification performance (AUC, F1, precision, recall, etc.).
- Visualize feature behavior (scatter, ROC, PR, CDF, threshold performance).

---

## 🧠 STRUCTURE

### `config.json`

Defines paths:
```json
{
  "input_graphml": "uniformly_sampled_playlist_tracks_45000.graphml",
  "output_dir": "graph_features_output"
}
```

---

## 🧩 SCRIPTS OVERVIEW

### A1. `graphml_to_edgelist_and_attributes.py`
- Converts `.graphml` to:
  - `converted_edgelist.csv` for cuGraph
  - `node_attributes.csv` for node metadata
  - `int_mapping.csv` for string→int ID mapping

### A2. `basic_graph_analysis.py`
- Uses NetworkX to compute:
  - Degree
  - Degree centrality
- Fast CPU-based analysis of the original graph

### A3. `analyze_with_cugraph.py`
- Uses cuGraph (GPU) to compute:
  - Degree
  - PageRank
  - Betweenness (approximate)
  - Katz centrality
- Uses renumbered graph and maps back to original IDs

### A4. `combine_all_features.py`
- Merges:
  - Node attributes
  - NetworkX features
  - cuGraph features
  - Optional: igraph features if file exists
- Produces: `all_node_features.csv`

---

### B1. `project_playlist_graph.py`
- Projects playlist–playlist graph based on shared tracks
- Saves as GraphML: `projected_playlist_playlist.graphml`

### B2. `tripartite_graph_conversion.py`
- Adds artist nodes to form a playlist–track–artist tripartite graph
- Saves as GraphML

### B3. `igraph_metrics_analysis.py`
- Uses `igraph` (CPU) for:
  - Closeness centrality
  - Eigenvector centrality
  - Local clustering coefficient
- Only suitable for small graphs (e.g., <10k nodes)

---

## 🧪 METRICS & EVALUATION

### C1. `evaluate_features_binary_classification.py`
- Binary classification: followers > 10
- Computes:
  - Precision, Recall, F1
  - AUC, Accuracy
  - TPR, FPR, TNR, FNR, Specificity
- Saves: `feature_classification_metrics.csv`

### C2. `plot_feature_threshold_metrics.py`
- Plots all metrics vs threshold for each feature
- Saved to: `feature_plots/threshold_metrics_<feature>.png`

### C3. `plot_feature_pr_curves.py`
- Precision-Recall curves for top features
- Baseline precision line included

### C4. `plot_feature_roc_curves.py`
- ROC curves for all numeric features

---

## 📈 FEATURE BEHAVIOR

### D1. `plot_feature_cdfs.py`
- Normalized CDF for each feature and combined
- Shows how features are distributed

### D2. `analyze_feature_vs_followers.py`
- Scatter plots: feature vs followers (log-log)
- Calculates Spearman & Kendall correlations
- Saves correlation bar chart

---

## ⚙️ INSTALLING cuGRAPH IN DOCKER (Recommended)

1. Install Docker Desktop
2. Pull the base image:
```bash
docker pull rapidsai/base:25.06a-cuda11.8-py3.10-amd64
```

3. Launch container:
```bash
docker run --gpus all -it --rm ^
  -v C:/path/to/your/project:/workspace ^
  rapidsai/base:25.06a-cuda11.8-py3.10-amd64 --entrypoint /bin/bash
```

4. Inside container:
```bash
cd /workspace
python3 analysis/A3_analyze_with_cugraph.py
```

---

## ✅ RECOMMENDATIONS

- Use full pipeline for <25k nodes.
- For large-scale graphs, skip `igraph` and only run cuGraph + basic.
- Always define your graph in `config.json`.

---

## 📁 OUTPUT STRUCTURE

```
graph_features_output/
│
├── all_node_features.csv
├── feature_classification_metrics.csv
├── feature_follower_correlations.csv
├── converted_edgelist.csv
├── node_attributes.csv
│
└── feature_plots/
    ├── cdf_<feature>.png
    ├── scatter_<feature>.png
    ├── threshold_metrics_<feature>.png
    ├── pr_curves.png
    ├── roc_curves.png
    └── spearman_correlation_barplot.png
```
