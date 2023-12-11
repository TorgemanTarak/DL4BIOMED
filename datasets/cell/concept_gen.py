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

        mask = []
        for i in range(n_concepts):
            mask.append(np.where(clusters == i)[0])
        self.mask = mask

        print("Generated concepts from linkage path", data_path)
        

class PCAConcepts:

  def __init__(self, n_concepts, data_path, n_feats, top_k,**kwargs):
    pca_components = torch.load(data_path)
    
    mask = []

    for i in range(n_concepts):
      indices = np.argpartition(np.abs(pca_components[i]), -top_k)[-top_k:]
      mask.append(indices)
    
    self.mask = mask

class NoConcepts:
    def __init__(self, n_feats, **kwargs):
        self.mask = np.arange(n_feats).reshape(1, n_feats)

class RandomConcepts:
    
    def __init__(self, n_concepts, data_path, n_feats, top_k, **kwargs):
      mask = []
      for i in range(n_concepts):
        mask.append(
            np.random.choice(n_feats, size=(top_k), replace=False)
        )

        self.mask = mask