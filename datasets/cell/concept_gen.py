import torch

import numpy as np
from scipy.cluster.hierarchy import fcluster
import torch.nn as nn
import torch.nn.functional as F


class LinkageConcepts:

    def __init__(self, n_concepts, data_path, n_feats, **kwargs):
        # Convert to n_clusters flat clusters

        self.n_concepts = n_concepts
        self.linkage_path = data_path
        Z = torch.load(data_path)
        clusters = fcluster(Z, n_concepts, criterion='maxclust')

        # subtract 1 from clusters to match python array indexing
        clusters = np.array(clusters) - 1
        N = len(clusters)
        clusters = clusters.reshape(N, 1)
        clusters = np.array(clusters, dtype=np.float32)
        clusters = torch.from_numpy(clusters)
        clusters = torch.nn.functional.one_hot(clusters.to(torch.int64), n_concepts).squeeze().transpose(0, 1)
        self.mask = clusters
        print("Generated concepts from linkage path", data_path)
        # Sanity check
        assert clusters.shape[1] == n_feats, f"Clusters shape {clusters.shape} does not match n_feats {n_feats}"

class PCAConcepts:

  def __init__(self, n_concepts, data_path, n_feats, pca_top_k,**kwargs):
    pca_components = torch.load(data_path)
    
    concepts = np.zeros((n_concepts, n_feats))

    for i in range(n_concepts):
      indices = np.argpartition(pca_components[i], -pca_top_k)[-pca_top_k:]
      concepts[i, indices] = 1
    
    self.mask = concepts