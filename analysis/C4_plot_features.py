import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

with open("analysis/config.json") as f:
    config = json.load(f)

output_dir = config["output_dir"]
metrics_path = f"{output_dir}/feature_classification_metrics.csv"
plot_dir = os.path.join(output_dir, "feature_plots")
os.makedirs(plot_dir, exist_ok=True)

# Load metrics
df = pd.read_csv(metrics_path)

# Get all unique features
features = df["feature"].unique()

# Plot metrics vs threshold for each feature
for feature in features:
    subset = df[df["feature"] == feature]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset, x="threshold", y="precision", label="Precision")
    sns.lineplot(data=subset, x="threshold", y="recall", label="Recall")
    sns.lineplot(data=subset, x="threshold", y="f1", label="F1 Score")
    sns.lineplot(data=subset, x="threshold", y="TPR", label="TPR")
    sns.lineplot(data=subset, x="threshold", y="FPR", label="FPR")
    sns.lineplot(data=subset, x="threshold", y="TNR", label="TNR")
    sns.lineplot(data=subset, x="threshold", y="FNR", label="FNR")
    sns.lineplot(data=subset, x="threshold", y="accuracy", label="Accuracy")
    plt.title(f"Metrics vs Threshold for Feature: {feature}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"threshold_metrics_{feature}.png"))
    plt.close()

print("Saved threshold-wise metric plots for all features")
