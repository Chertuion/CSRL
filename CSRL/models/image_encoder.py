import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch

from utils.get_subgraph import relabel, split_batch
from utils.mask import clear_masks, set_masks
import dgl
from models.conv import GNN_node, GNN_node_Virtualnode
from models.gnn import GNN, LeGNN
import math

from dgl.nn import NNConv
from utils.util import batch_gamma_positional_encoding
from torch_geometric.nn import (GlobalAttention, LEConv, Set2Set,
                                global_add_pool, global_max_pool,
                                global_mean_pool)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** -0.5  # Scale factor for attention

    def forward(self, x):
        # x is the image features of shape [batch_size, feature_dim]
        query = self.query_layer(x)  # Shape: [batch_size, feature_dim]
        key = self.key_layer(x).T  # Shape: [feature_dim, batch_size]
        value = self.value_layer(x)  # Shape: [batch_size, feature_dim]

        # Compute the attention scores
        attention_scores = torch.mm(query, key)  # Shape: [batch_size, batch_size]
        attention_scores *= self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: [batch_size, batch_size]

        # Apply attention weights to the values
        attended_features = torch.mm(attention_weights, value)  # Shape: [batch_size, feature_dim]
        return attended_features





class ImageEncoder(nn.Module):

    def __init__(self, emb_dim=128):
        super(ImageEncoder, self).__init__()

        self.image_feature = nn.Linear(emb_dim, emb_dim)
        self.attention = SelfAttention(emb_dim) 
        self.fp_projection = torch.nn.Sequential(nn.Linear(167, 2 * emb_dim), nn.ReLU(),
                                           nn.Linear(2 * emb_dim, emb_dim))
        self.combine_projection = torch.nn.Sequential(nn.Linear(emb_dim * 2, 2 * emb_dim), nn.ReLU(),
                                           nn.Linear(2 * emb_dim, emb_dim))
        self.spu_mlp = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Dropout(0.5),
                                           nn.Linear(2 * emb_dim, 2))
        self.cq = nn.Linear(2, 2)
        self.spu_fw = torch.nn.Sequential(self.spu_mlp, self.cq)


    def forward(self, batch, goal="pred"):
        # obtain the graph embeddings from the featurizer GNN encoder
        # h_image = self.image_feature(batch.image_embedding)
        h_fp = self.fp_projection(batch)
        # h_combine = torch.cat((h_fp, h_image), dim=1)
        # Apply self-attention to the image features

        # h_combine = self.attention(h_fp)  
        # h_combine = self.combine_projection(h_combine)

        p_image = self.spu_fw(h_fp)

        if goal == "pred":
            return p_image
        return h_fp, p_image
