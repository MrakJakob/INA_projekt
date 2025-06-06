import numpy as np
import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from typing import Dict, List, Optional

from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.loader.neighbor_loader import NeighborLoader
from sentence_transformers import SentenceTransformer

from utils import get_followers, get_playlists_tracks
from models import BaseModel

def get_playlist_name_ft(G):
    text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    playlists, _ = get_playlists_tracks(G)
    pl_names = [G.nodes[n]['name'] for n in playlists]
    pl_emb = torch.from_numpy(text_embedder.encode(pl_names))
    # dim = 384
    return pl_emb

def get_playlist_followers_ft(G, repeat=1):
    _, followers = get_followers(G)
    playlist_ft = torch.tensor(followers).unsqueeze(1).float().repeat(1, repeat)
    playlist_ft = (playlist_ft - playlist_ft.mean(dim=0)) / playlist_ft.std(dim=0)
    return playlist_ft

def get_node_degree_ft(G, repeat=1):
    playlists, tracks = get_playlists_tracks(G)
    playlist_dg = [G.degree(n) for n in playlists]
    playlist_ft = torch.tensor(playlist_dg).unsqueeze(1).float().repeat(1, repeat)
    playlist_ft = (playlist_ft - playlist_ft.mean(dim=0)) / playlist_ft.std(dim=0)
    track_dg = [G.degree(n) for n in tracks]
    track_ft = torch.tensor(track_dg).unsqueeze(1).float().repeat(1, repeat)
    track_ft = (track_ft - track_ft.mean(dim=0)) / track_ft.std(dim=0)
    return playlist_ft, track_ft

def prepare_torch_data(G, d=16, playlist_ft=None, track_ft=None):
    
    playlists, tracks = get_playlists_tracks(G)
    nodes = playlists + tracks
    num_pl = len(playlists)

    G_ = nx.Graph()
    G_.add_edges_from(G.edges())

    id_map = {id_: i for i, id_ in enumerate(nodes)}
    G_ = nx.relabel_nodes(G_, id_map, copy=True)
    node_type = torch.zeros(len(nodes), dtype=torch.long)
    node_type[num_pl:] = 1

    node_features = torch.randn(len(nodes), d)
    if playlist_ft is not None:
        assert playlist_ft.shape[1] <= node_features.shape[1]
        node_features[:num_pl, :playlist_ft.shape[1]] = playlist_ft
    if track_ft is not None: 
        assert track_ft.shape[1] <= node_features.shape[1]
        node_features[num_pl:, :track_ft.shape[1]] = track_ft

    data = from_networkx(G_)
    data.x = node_features
    data.node_type = node_type
    return data, id_map

class _GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return self.head(x)

class GraphSAGEBasic(BaseModel):

    def __init__(self, node_ft=None, ft_dim=16, hidden_dim=16, repeat_ft=False,
                    epochs=30, lr=0.01):
        self.node_ft = node_ft
        self.ft_dim = ft_dim
        self.hidden_dim = hidden_dim
        self.repeat_ft = repeat_ft
        self.epochs = epochs
        self.lr = lr

    def init_data(self, G, projection, test_nodes, edges):
        
        if self.node_ft == "name":
            pl_ft = get_playlist_name_ft(G)
            tr_ft = None
        elif self.node_ft == "followers":
            pl_ft = get_playlist_followers_ft(G, repeat=self.ft_dim if self.repeat_ft else 1)
            tr_ft = None
        elif self.node_ft == "degree":
            pl_ft, tr_ft = get_node_degree_ft(G, repeat=self.ft_dim if self.repeat_ft else 1)
        else:
            pl_ft, tr_ft = None, None

        self.data, self.id_map = prepare_torch_data(G, d=self.ft_dim, 
                                            playlist_ft=pl_ft,
                                            track_ft=tr_ft)

        self.model = _GraphSAGE(in_channels=self.data.num_node_features,
                            hidden_channels=self.hidden_dim,
                            out_channels=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, train_nodes, train_buckets):
        tr_indices = torch.tensor([self.id_map[id_] for id_ in train_nodes])
        train_mask = torch.zeros((len(self.data.x), ), dtype=bool)
        train_mask[tr_indices] = True
        y = torch.from_numpy(train_buckets).unsqueeze(1).float()

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(out[tr_indices], y)
            loss.backward()
            self.optimizer.step()

        # train_loader = NeighborLoader(
        #     self.data, input_nodes=train_mask,
        #     num_neighbors=[5, 5], batch_size=32, shuffle=True
        # )

        # for epoch in range(self.epochs):
        #     self.model.train()
        #     total_loss = 0
        #     for batch in train_loader:
        #         self.optimizer.zero_grad()
        #         out = self.model(batch.x, batch.edge_index)
        #         loss = self.criterion(out[batch.batch], batch.y[batch.batch])
        #         loss.backward()
        #         self.optimizer.step()
        #         total_loss += loss.item()
        #     print(f'Epoch {epoch}, Loss: {total_loss:.4f}')

    def predict(self, test_nodes):
        ts_indices = torch.tensor([self.id_map[id_] for id_ in test_nodes])
        test_mask = torch.zeros((len(self.data.x), ), dtype=bool)
        test_mask[ts_indices] = True

        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            probs = torch.sigmoid(out)
            pred = (probs[ts_indices] > 0.5).long().squeeze().detach().numpy()
        return pred

if __name__ == "__main__":
    gdir = "graphs/test_mini"
    gname = "test_mini"
    G = nx.read_graphml(f"{gdir}/{gname}.graphml")
    train_df = pd.read_csv(f"{gdir}/{gname}_train.csv")
    tr_nodes, tr_buckets = np.array(train_df["nodes"]), np.array(train_df["buckets"])
    test_df = pd.read_csv(f"{gdir}/{gname}_test.csv")
    ts_nodes, ts_buckets = np.array(test_df["nodes"]), np.array(test_df["buckets"])
    edges = np.load(f"{gdir}/{gname}_edges.npy")

    gs = GraphSAGEBasic(node_ft="degree", ft_dim=16, epochs=60)

    gs.init_data(G, None, ts_nodes, edges)
    gs.train(tr_nodes, tr_buckets)

    pred = gs.predict(ts_nodes)
    print(accuracy_score(ts_buckets, pred))
    pred = gs.predict(tr_nodes)
    print(accuracy_score(tr_buckets, pred))