import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Load BiNE embeddings (assumes format: ID dim1 dim2 ... dimN)
embeddings = {}
with open("output_playlist_embeddings.txt", "r") as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split()
        node = parts[0]
        vector = list(map(float, parts[1:]))
        embeddings[node] = vector

# Extract playlist nodes only
playlist_ids = [n for n in embeddings if n.startswith("pl_")]
X = [embeddings[n] for n in playlist_ids]
y = [int(G.nodes[n]["followers"]) for n in playlist_ids]

# Binary target
y_binary = (np.array(y) >= 100).astype(int)

# Classify
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, stratify=y_binary)

clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
