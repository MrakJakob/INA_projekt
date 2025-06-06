import numpy as np
import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.manifold import SpectralEmbedding
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer
from utils import get_playlists_tracks
from heapq import nlargest
from node2vec import Node2Vec



class BaseModel:
    def __init__(self):
        # set hyperparams here
        self.name = "Base Model"

    def init_data(
        self,
        G: nx.Graph,
        projection: nx.Graph,
        test_nodes: np.ndarray[str],
        edges: np.ndarray[int],
        features_df: pd.DataFrame = None,
    ):
        # preprocess any type of graph you need here
        # if uses labels (follower counts) at inference,
        # make sure to remove test nodes in training !!
        pass

    def train(self, train_nodes: np.ndarray[str], train_buckets: np.ndarray[int]):
        # train the model here - return nothing
        pass

    def predict(self, test_nodes: np.ndarray[str]) -> np.ndarray[int]:
        # return predictions here
        pass


class Majority(BaseModel):

    def init_data(self, G, projection, test_nodes, edges, features_df=None):
        pass

    def train(self, train_nodes, train_buckets):
        vals, counts = np.unique(train_buckets, return_counts=True)
        self.majority = vals[np.argmax(counts)]

    def predict(self, test_nodes):
        return (np.ones_like(test_nodes) * self.majority).astype(int)


class NeighborMean(BaseModel):

    def init_data(self, G, projection, test_nodes, edges, features_df=None):
        self.proj = projection
        # not needed here, but as demo:
        self.proj_train = self.proj.copy()
        self.proj_train.remove_nodes_from(test_nodes)
        self.edges = np.array(edges).astype(int)

    def predict(self, test_nodes):
        predictions = []
        for n in test_nodes:
            neighbors = list(self.proj.neighbors(n))
            #predictions.append(self.proj.nodes[n]["followers"])
            if len(neighbors) == 0:
                predictions.append(self.edges[0])
            else:
                nb_followers = np.mean(
                    [int(self.proj.nodes[nb]["followers"]) for nb in neighbors]
                )
                # nb_followers = self.proj.nodes[n]["followers"]
                predictions.append(nb_followers)
        return np.digitize(predictions, self.edges) - 1

    
class SimilarNeighbor(BaseModel):

    def init_data(self, G, projection, test_nodes, edges, features_df=None):
        self.G = G
        self.edges = np.array(edges).astype(int)

    def predict(self, test_nodes):
        predictions = []
        playlists, _ = get_playlists_tracks(self.G)

        for n in test_nodes:
            potentials = ((n, other) for other in playlists if other != n)

            #aa = nx.adamic_adar_index(self.G, potentials)
            #aa = nx.jaccard_coefficient(self.G, potentials)
            aa = [(u, v, len(set(self.G.neighbors(u)) & set(self.G.neighbors(v))))            
                    for u, v in potentials]

            scored = [(v, score) for u, v, score in aa]
            top = nlargest(20, scored, key=lambda x: x[1])
            top_nodes = [n for n, s in top]


            nb_followers = np.max(
                [int(self.G.nodes[n]["followers"]) for n in top_nodes]
            )
            predictions.append(nb_followers)

        return np.digitize(predictions, self.edges) - 1


class TrackDegree(BaseModel):

    def __init__(self, agg="sum"):
        self.agg = np.sum if agg == "sum" else np.mean

    def init_data(self, G, projection, test_nodes, edges, features_df=None):
        self.G = G
        self.fitter = LogisticRegression()

    def _track_degs(self, nodes):
        avg_degs = []
        for n in nodes:
            track_degs = np.array([self.G.degree(nb) for nb in self.G.neighbors(n)])
            avg_degs.append(np.sum(track_degs) if len(track_degs) > 0 else 0)
        return np.array(avg_degs)
        #return np.array([self.G.degree[n] < 10 for n in nodes])


    def train(self, train_nodes, train_buckets):
        avg_degs = self._track_degs(train_nodes)
        self.fitter.fit(avg_degs.reshape(-1, 1), train_buckets)

    def predict(self, test_nodes):
        avg_degs = self._track_degs(test_nodes)
        return self.fitter.predict(avg_degs.reshape(-1, 1))


class Spectral(BaseModel):

    def init_data(self, G, projection, test_nodes, edges, features_df=None):
        adj_matrix = nx.to_numpy_array(projection)
        embedder = SpectralEmbedding(n_components=16, affinity="precomputed")
        self.x = embedder.fit_transform(adj_matrix)
        self.node_to_index = {
            node: i for i, node in enumerate(list(projection.nodes()))
        }

        self.fitter = LogisticRegression()

    def train(self, train_nodes, train_buckets):
        train_idx = [self.node_to_index[n] for n in train_nodes]
        self.fitter.fit(self.x[train_idx], train_buckets)

    def predict(self, test_nodes):
        test_idx = [self.node_to_index[n] for n in test_nodes]
        return self.fitter.predict(self.x[test_idx])

class NameEmbedding(BaseModel):

    def init_data(self, G, projection, test_nodes, edges):
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        playlists, tracks = get_playlists_tracks(G)
        pl_names = [G.nodes[n]['name'] for n in playlists]
        self.pl_emb = self.text_embedder.encode(pl_names)
        self.node_to_index = {n: i for i, n in enumerate(playlists)}

        self.x = self.pl_emb
        self.fitter = LogisticRegression()

    def train(self, train_nodes, train_buckets):
        train_idx = [self.node_to_index[n] for n in train_nodes]
        self.fitter.fit(self.x[train_idx], train_buckets)

    def predict(self, test_nodes):
        test_idx = [self.node_to_index[n] for n in test_nodes]
        return self.fitter.predict(self.x[test_idx])

class Node2VecModel(BaseModel):
    def __init__(self, dimensions=128, walk_length=40, num_walks=10, p=1, q=2):
        super().__init__()
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.name = "Node2Vec Model"
        self.fitter = LogisticRegression(max_iter=1000)

    def init_data(self, G, projection, test_nodes, edges, features_df=None):
        print("Running Node2Vec...")
        node2vec = Node2Vec(
            G,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=2,
            p=self.p,
            q=self.q
        )
        self.model = node2vec.fit(window=5, min_count=1)

        # Extract embeddings for all nodes
        self.node_embeddings = {}
        for node in G.nodes():
            if str(node) in self.model.wv:
                self.node_embeddings[node] = self.model.wv[str(node)]

        # Map nodes to indices for training and testing
        self.node_to_index = {node: i for i, node in enumerate(G.nodes())}

    def train(self, train_nodes, train_buckets):
        # Prepare feature matrix (X) and labels (y) for training
        X_train = []
        y_train = []
        for node, bucket in zip(train_nodes, train_buckets):
            if node in self.node_embeddings:
                X_train.append(self.node_embeddings[node])
                y_train.append(bucket)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train the logistic regression model
        self.fitter.fit(X_train, y_train)

    def predict(self, test_nodes):
        # Prepare feature matrix (X) for testing
        X_test = []
        for node in test_nodes:
            if node in self.node_embeddings:
                X_test.append(self.node_embeddings[node])
            else:
                X_test.append(np.zeros(self.dimensions))  # Fallback for missing embeddings
        X_test = np.array(X_test)

        # Predict using the trained logistic regression model
        return self.fitter.predict(X_test)