In `matej/`, working on e.g. graph `mini`:

`prepare_data.py mini` to compute and save the projection and train/test splits.
Comment out certain lines if you want different preprocessing.

`graph_info.py mini` for network and train/test stats.

Add your model in `eval_models.py` like:
```
    models = {
        "Neighbor Mean": NeighborMean(),
        "Track Degree": TrackDegree(),
        "Majority": Majority(),
        "Spectral": Spectral(),
    }
```
Then `eval_models.py mini` to train, compute predictions and compute score for all models.

Do `eval_models.py mini "Model Name"` to run a single model only.

Predictions will be saved in `predictions/mini` for all evaluated models.

`compute_scores.py mini` to just compute scores from all predictions in `predictions/mini`.
Scores will be saved in  `results/mini`
