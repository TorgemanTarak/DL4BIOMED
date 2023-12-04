import torch

import numpy as np
from scipy.cluster.hierarchy import fcluster
import torch.nn as nn
import torch.nn.functional as F


class LinkageConcepts:

    def __init__(self, n_clusters, linkage_path, n_feats):
        # Convert to n_clusters flat clusters

        self.n_clusters = n_clusters
        self.linkage_path = linkage_path
        Z = np.load(linkage_path)
        clusters = fcluster(Z, n_clusters, criterion='maxclust')

        # subtract 1 from clusters to match python array indexing
        clusters = np.array(clusters) - 1
        N = len(clusters)
        clusters = clusters.reshape(N, 1)
        clusters = np.array(clusters, dtype=np.float32)
        clusters = torch.from_numpy(clusters)
        clusters = torch.nn.functional.one_hot(clusters.to(torch.int64), n_clusters).squeeze().transpose(0, 1)
        self.mask = clusters
        print("Generated concepts from linkage path", linkage_path)
        # Sanity check
        assert clusters.shape[1] == n_feats, f"Clusters shape {clusters.shape} does not match n_feats {n_feats}"

