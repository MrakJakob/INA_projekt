import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


if __name__ == "__main__":

    gname = sys.argv[1]
    gdir = f"graphs/{gname}"

    fnames = os.listdir(f"predictions/{gname}")

    all_scores = []
    for fname in fnames:

        pred_df = pd.read_csv(f"predictions/{gname}/{fname}")
        mname = os.path.splitext(fname)[0]
        pred = np.array(pred_df["pred"])
        ts_buckets = np.array(pred_df["true"])

        scores = {
            "model": mname,
            "ca": accuracy_score(ts_buckets, pred),
            "precision": precision_score(ts_buckets, pred, average="binary"),
            "recall": recall_score(ts_buckets, pred, average="binary"),
            "f1": f1_score(ts_buckets, pred, average="binary")
        }

        all_scores.append(scores)
        print(f"{mname}: {scores}")

    os.makedirs(f"results/{gname}", exist_ok=True)
    results_df = pd.DataFrame(all_scores)
    results_df.to_csv(f"results/{gname}/results.csv")



    
    

