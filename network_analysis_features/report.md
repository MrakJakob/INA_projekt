Feature Importances (playlist_graph_features_5000_balanced.csv):
Test Accuracy: 0.6150

Classification Report:
              precision    recall  f1-score   support

           0       0.60      0.62      0.61       487
           1       0.63      0.61      0.62       513

    accuracy                           0.61      1000
   macro avg       0.61      0.62      0.61      1000
weighted avg       0.62      0.61      0.62      1000

AUC Score: 0.6590174958271791

[('num_tracks', 0.07695201956522602), ('avg_track_idf', 0.030616696887095562), ('avg_track_degree', 0.010371010803302938), ('clustering_coeff', 0.0065003942665241565), ('pagerank', 0.003654470422005174), ('collaborative', 6.804599909526754e-05), ('unique_artists', 0.0), ('unique_albums', 0.0), ('closeness_centrality', -0.0008685871649236621), ('avg_track_duration_ms', -0.0010006764572851878), ('betweenness_centrality', -0.0037945651260252733)]



Test Accuracy: 0.6340

Classification Report:
              precision    recall  f1-score   support

           0       0.62      0.66      0.64       487
           1       0.65      0.61      0.63       513

    accuracy                           0.63      1000
   macro avg       0.63      0.63      0.63      1000
weighted avg       0.64      0.63      0.63      1000


AUC Score: 0.6666786747841542

Feature Importances:
[('degree_x', 0.02327973710228126), ('track_playlist_freq_std_y', 0.014513811336463567), ('track_playlist_freq_median_y', 0.011379692672246478), ('common_tracks_count_y', 0.010723248916267525), ('track_diversity_entropy_x', 0.008381666006220367), ('pagerank', 0.008161517185617528), ('rare_tracks_ratio_y', 0.007797270955165914), ('track_playlist_freq_std_x', 0.006448359090745592), ('rare_tracks_count_x', 0.004647141467632232), ('common_tracks_ratio_y', 0.00447102241115005), ('track_playlist_freq_max_y', 0.003682489362809349), ('avg_track_idf', 0.0035263838354728794), ('track_playlist_freq_max_x', 0.0032702106624079574), ('avg_track_degree', 0.0032101700749708195), ('track_playlist_freq_median_x', 0.0028179049037149184), ('track_playlist_freq_mean_y', 0.0023135639692434262), ('degree_y', 0.0017211635065305098), ('common_tracks_count_x', 0.0012768631594959334), ('track_playlist_freq_range_x', 0.00031221105467316157), ('track_diversity_hhi_y', -0.00027618670221052355), ('track_diversity_entropy_y', -0.0003722516421099664), ('common_tracks_ratio_x', -0.0007965384599988301), ('track_playlist_freq_mean_x', -0.0008405682241193757), ('rare_tracks_ratio_x', -0.0026698047880365783)]


test.py:
Optimal Threshold (0.4980) - Accuracy: 0.6340, F1: 0.6337
AUC Score: 0.6537
Neural Network AUC: 0.6537
Neural Network Accuracy: 0.6320

=== Ensemble Classifier ===
Training ensemble classifier...

Individual Classifier Performance:
rf: 0.6070
gb: 0.6020
xgb: 0.5960
lr: 0.6300
Training ensemble classifier...

Individual Classifier Performance:
rf: 0.6070
gb: 0.6020
xgb: 0.5960
lr: 0.6300
gb: 0.6020
xgb: 0.5960
lr: 0.6300
lr: 0.6300
svm: 0.6290
Ensemble Accuracy: 0.6190
Ensemble AUC: 0.6547



Loaded 4000 training samples, 1000 test samples
Features shape: (5000, 40)
After merging - Train: 4000, Test: 1000
=== Enhanced Neural Network ===
Training on cpu
Model architecture: 24 -> 512 -> 256 -> 128 -> 64 -> 32 -> 2
Epoch [20/1000], Train Loss: 0.6601, Train Acc: 61.22%
Epoch [40/1000], Train Loss: 0.6418, Train Acc: 63.73%
Epoch [60/1000], Train Loss: 0.6416, Train Acc: 63.20%
Epoch [80/1000], Train Loss: 0.6358, Train Acc: 64.28%
Epoch [100/1000], Train Loss: 0.6325, Train Acc: 64.15%
Epoch [120/1000], Train Loss: 0.6329, Train Acc: 63.70%
Epoch [140/1000], Train Loss: 0.6326, Train Acc: 63.50%
Epoch [160/1000], Train Loss: 0.6312, Train Acc: 64.20%
Epoch [180/1000], Train Loss: 0.6264, Train Acc: 64.92%
Epoch [200/1000], Train Loss: 0.6238, Train Acc: 65.05%
Epoch [220/1000], Train Loss: 0.6237, Train Acc: 65.08%
Epoch [240/1000], Train Loss: 0.6210, Train Acc: 65.62%
Epoch [260/1000], Train Loss: 0.6123, Train Acc: 66.15%
Epoch [280/1000], Train Loss: 0.6104, Train Acc: 66.03%
Epoch [300/1000], Train Loss: 0.6020, Train Acc: 66.72%
Epoch [320/1000], Train Loss: 0.6048, Train Acc: 66.38%
Epoch [340/1000], Train Loss: 0.5996, Train Acc: 66.10%
Epoch [360/1000], Train Loss: 0.5998, Train Acc: 67.03%
Epoch [380/1000], Train Loss: 0.5913, Train Acc: 67.90%
Epoch [400/1000], Train Loss: 0.5860, Train Acc: 68.58%
Epoch [420/1000], Train Loss: 0.5865, Train Acc: 68.25%
Epoch [440/1000], Train Loss: 0.5842, Train Acc: 68.35%
Epoch [460/1000], Train Loss: 0.5714, Train Acc: 69.58%
Epoch [480/1000], Train Loss: 0.5719, Train Acc: 68.80%
Epoch [500/1000], Train Loss: 0.5631, Train Acc: 69.90%
Epoch [520/1000], Train Loss: 0.5658, Train Acc: 69.62%
Epoch [540/1000], Train Loss: 0.5579, Train Acc: 70.45%
Epoch [560/1000], Train Loss: 0.5606, Train Acc: 70.15%
Epoch [580/1000], Train Loss: 0.5569, Train Acc: 70.65%
Epoch [600/1000], Train Loss: 0.5445, Train Acc: 70.80%
Epoch [620/1000], Train Loss: 0.5308, Train Acc: 72.65%
Epoch [640/1000], Train Loss: 0.5388, Train Acc: 71.05%
Epoch [660/1000], Train Loss: 0.5323, Train Acc: 71.60%
Epoch [680/1000], Train Loss: 0.5357, Train Acc: 71.05%
Epoch [700/1000], Train Loss: 0.5226, Train Acc: 71.83%
Epoch [720/1000], Train Loss: 0.5207, Train Acc: 72.97%
Epoch [740/1000], Train Loss: 0.5230, Train Acc: 72.90%
Epoch [760/1000], Train Loss: 0.5156, Train Acc: 72.80%
Epoch [780/1000], Train Loss: 0.5188, Train Acc: 73.30%
Epoch [800/1000], Train Loss: 0.5104, Train Acc: 72.80%
Epoch [820/1000], Train Loss: 0.5066, Train Acc: 72.82%
Epoch [840/1000], Train Loss: 0.5148, Train Acc: 72.88%
Epoch [860/1000], Train Loss: 0.4994, Train Acc: 73.42%
Epoch [880/1000], Train Loss: 0.5109, Train Acc: 72.88%
Epoch [900/1000], Train Loss: 0.4986, Train Acc: 74.22%
Epoch [920/1000], Train Loss: 0.4947, Train Acc: 74.25%
Epoch [940/1000], Train Loss: 0.5037, Train Acc: 73.40%
Epoch [960/1000], Train Loss: 0.4987, Train Acc: 74.45%
Epoch [980/1000], Train Loss: 0.4997, Train Acc: 74.48%
Epoch [1000/1000], Train Loss: 0.4977, Train Acc: 74.58%
Standard Threshold (0.5) - Accuracy: 0.6020, F1: 0.6015
Optimal Threshold (0.4781) - Accuracy: 0.6100, F1: 0.6081
AUC Score: 0.6267
Neural Network AUC: 0.6267
Neural Network Accuracy: 0.6020

=== Ensemble Classifier ===
Training ensemble classifier...

Individual Classifier Performance:
rf: 0.6070
gb: 0.6020
xgb: 0.5960
lr: 0.6300
svm: 0.6290
Ensemble Accuracy: 0.6190
Ensemble AUC: 0.6547




### Preparing data for the report

#### Training and testing on unbalanced data (uniformly sampled)
Loaded 1400 training samples, 600 test samples
Features shape: (2000, 14)
After merging - Train: 1400, Test: 600
Class weights: [ 0.51584377 16.27906977]
Data preparation complete!
Features used: ['num_tracks', 'unique_artists', 'unique_albums', 'avg_track_duration_ms', 'avg_track_degree', 'avg_track_idf', 'collaborative', 'pagerank', 'clustering_coeff', 'betweenness_centrality', 'closeness_centrality']
Training class distribution: [1357   43]
Test class distribution: [585  15]

Starting training...
Training on cpu
Model architecture: 11 -> 128 -> 64 -> 32 -> 2
Using weighted_loss strategy
Epoch [10/200], Loss: 0.1254, Accuracy: 53.64%, F1: 0.3865
Epoch [20/200], Loss: 0.1284, Accuracy: 49.79%, F1: 0.3532
Epoch [30/200], Loss: 0.1100, Accuracy: 55.50%, F1: 0.4370
Epoch [40/200], Loss: 0.1020, Accuracy: 56.21%, F1: 0.4511
Epoch [50/200], Loss: 0.1006, Accuracy: 57.86%, F1: 0.4831
Epoch [60/200], Loss: 0.1025, Accuracy: 58.71%, F1: 0.5121
Epoch [70/200], Loss: 0.0954, Accuracy: 62.57%, F1: 0.5684
Epoch [80/200], Loss: 0.0906, Accuracy: 63.29%, F1: 0.5727
Epoch [90/200], Loss: 0.0858, Accuracy: 64.57%, F1: 0.5967
Epoch [100/200], Loss: 0.0827, Accuracy: 66.79%, F1: 0.6269
Epoch [110/200], Loss: 0.0804, Accuracy: 68.36%, F1: 0.6427
Epoch [120/200], Loss: 0.0894, Accuracy: 65.50%, F1: 0.6120
Epoch [130/200], Loss: 0.0865, Accuracy: 65.64%, F1: 0.6112
Epoch [140/200], Loss: 0.0864, Accuracy: 65.14%, F1: 0.6032
Epoch [150/200], Loss: 0.0937, Accuracy: 64.86%, F1: 0.5989
Epoch [160/200], Loss: 0.0895, Accuracy: 65.07%, F1: 0.6054
Epoch [170/200], Loss: 0.0947, Accuracy: 64.14%, F1: 0.5975
Epoch [180/200], Loss: 0.0866, Accuracy: 65.71%, F1: 0.6093
Epoch [190/200], Loss: 0.0910, Accuracy: 66.29%, F1: 0.6186
Epoch [200/200], Loss: 0.0975, Accuracy: 64.50%, F1: 0.5999

Evaluating model...
Optimal threshold (Youden's J): 0.9114

Test Accuracy: 0.3317

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.32      0.48       585
           1       0.03      0.73      0.05        15

    accuracy                           0.33       600
   macro avg       0.50      0.53      0.27       600
weighted avg       0.96      0.33      0.47       600


AUC Score: 0.5901994301994302

Feature Importances:
[('avg_track_idf', 0.23225071225071225), ('num_tracks', 0.22917378917378922), ('collaborative', 0.029743589743589705), ('unique_artists', 0.0), ('unique_albums', 0.0), ('pagerank', -0.0011396011396009875), ('avg_track_duration_ms', -0.002849002849002913), ('clustering_coeff', -0.010484330484330506), ('closeness_centrality', -0.018803418803418848), ('betweenness_centrality', -0.10962962962962963), ('avg_track_degree', -0.11498575498575503)]


### Training and testing on balanced data
Loaded 1600 training samples, 400 test samples
Features shape: (2000, 14)
After merging - Train: 1600, Test: 400
Class weights: [0.99750623 1.00250627]
Data preparation complete!
Features used: ['num_tracks', 'unique_artists', 'unique_albums', 'avg_track_duration_ms', 'avg_track_degree', 'avg_track_idf', 'collaborative', 'pagerank', 'clustering_coeff', 'betweenness_centrality', 'closeness_centrality']
Training class distribution: [802 798]
Test class distribution: [212 188]

Starting training...
Training on cpu
Model architecture: 11 -> 128 -> 64 -> 32 -> 2
Using weighted_loss strategy
Epoch [10/200], Loss: 0.6644, Accuracy: 61.81%, F1: 0.6178
Epoch [20/200], Loss: 0.6378, Accuracy: 64.44%, F1: 0.6443
Epoch [30/200], Loss: 0.6332, Accuracy: 64.44%, F1: 0.6444
Epoch [40/200], Loss: 0.6289, Accuracy: 65.00%, F1: 0.6501
Epoch [50/200], Loss: 0.6303, Accuracy: 64.25%, F1: 0.6426
Epoch [60/200], Loss: 0.6362, Accuracy: 63.19%, F1: 0.6320
Epoch [70/200], Loss: 0.6408, Accuracy: 62.50%, F1: 0.6249
Epoch [80/200], Loss: 0.6324, Accuracy: 65.69%, F1: 0.6568
Epoch [90/200], Loss: 0.6253, Accuracy: 64.44%, F1: 0.6444
Epoch [100/200], Loss: 0.6292, Accuracy: 64.50%, F1: 0.6450
Epoch [110/200], Loss: 0.6325, Accuracy: 65.44%, F1: 0.6544
Epoch [120/200], Loss: 0.6154, Accuracy: 66.44%, F1: 0.6644
Epoch [130/200], Loss: 0.6299, Accuracy: 64.12%, F1: 0.6412
Epoch [140/200], Loss: 0.6153, Accuracy: 66.88%, F1: 0.6688
Epoch [150/200], Loss: 0.6253, Accuracy: 64.62%, F1: 0.6463
Epoch [160/200], Loss: 0.6283, Accuracy: 64.75%, F1: 0.6475
Epoch [170/200], Loss: 0.6235, Accuracy: 65.62%, F1: 0.6564
Epoch [180/200], Loss: 0.6290, Accuracy: 65.81%, F1: 0.6581
Epoch [190/200], Loss: 0.6346, Accuracy: 64.12%, F1: 0.6413
Epoch [200/200], Loss: 0.6278, Accuracy: 63.06%, F1: 0.6306

Evaluating model...
Optimal threshold (Youden's J): 0.4824

Test Accuracy: 0.6550

Classification Report:
              precision    recall  f1-score   support

           0       0.68      0.67      0.67       212
           1       0.63      0.64      0.64       188

    accuracy                           0.66       400

Feature Importances:
[('num_tracks', 0.09298474508229637), ('avg_track_degree', 0.04614110798875959), ('betweenness_centrality', 0.04511240465676436), ('clustering_coeff', 0.021276595744680882), ('closeness_centrality', 0.017864311521477383), ('pagerank', 0.012043356081894774), ('avg_track_duration_ms', 0.010061220393416281), ('avg_track_idf', 0.005720594138900137), ('unique_artists', 0.0), ('unique_albums', 0.0), ('collaborative', -0.0031613809714973184)]