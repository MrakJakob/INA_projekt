import numpy as np
import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression

class BaseModel():

    def __init__(self):
        # set hyperparams here
        self.name = "Base Model"

    def init_data(self, G: nx.Graph, projection: nx.Graph, 
                    test_nodes: np.ndarray[str], edges: np.ndarray[int]):
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


class NeighborMean(BaseModel):

    def __init__(self):
        self.name = "Neighbor Mean"

    def init_data(self, G, projection, test_nodes, edges):
        self.proj = projection
        # not needed here, but as demo:
        self.proj_train = self.proj.copy()
        self.proj_train.remove_nodes_from(test_nodes)
        self.edges = np.array(edges).astype(int)

    def predict(self, test_nodes):
        predictions = []
        for n in test_nodes:
            neighbors = list(self.proj.neighbors(n))
            if len(neighbors) == 0:
                predictions.append(self.edges[0])
            else:
                nb_followers = np.mean([int(self.proj.nodes[nb]["followers"]) for nb in neighbors])
                #nb_followers = self.proj.nodes[n]["followers"]
                predictions.append(nb_followers)
        return np.digitize(predictions, self.edges) - 1

class TrackDegree(BaseModel):

    def __init__(self, agg="sum"):
        self.agg = np.sum if agg == "sum" else np.mean
    
    def init_data(self, G, projection, test_nodes, edges):
        self.G = G
        self.fitter = LogisticRegression()

    def _track_degs(self, nodes):
        avg_degs = []
        for n in nodes:
            track_degs = np.array([self.G.degree(nb) for nb in self.G.neighbors(n)])
            avg_degs.append(np.mean(track_degs) if len(track_degs) > 0 else 0)
        return np.array(avg_degs)

    def train(self, train_nodes, train_buckets):        
        avg_degs = self._track_degs(train_nodes)
        self.fitter.fit(avg_degs.reshape(-1, 1), train_buckets)

    def predict(self, test_nodes):
        avg_degs = self._track_degs(test_nodes)
        return self.fitter.predict(avg_degs.reshape(-1, 1))
        
    

    






