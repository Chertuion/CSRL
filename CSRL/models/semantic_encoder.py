import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch

from utils.get_subgraph import relabel, split_batch
from utils.mask import clear_masks, set_masks
# import dgl
# from models.conv import GNN_node, GNN_node_Virtualnode
# from models.gnn import GNN, LeGNN
# import math
import os
# from dgl.nn import NNConv
# from utils.util import batch_gamma_positional_encoding
# from torch_geometric.nn import (GlobalAttention, LEConv, Set2Set,
#                                 global_add_pool, global_max_pool,
#                                 global_mean_pool)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_encoder import GraphEncoder
from models.image_encoder import ImageEncoder
from torch.nn import Dropout
from torch.autograd import Function
from utils.util import mmc_2

#梯度反转
class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by the number of heads"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # Query, Key, Value transformations for all heads
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Scale factor for more stable gradients
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        # Final linear transformation for concatenated heads
        self.out = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Transform and split into heads: (batch_size, num_heads, head_dim)
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)

        # Compute attention scores per head
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attention = F.softmax(attention_scores, dim=-1)

        # Apply attention to values and concatenate heads
        attended_values = torch.matmul(attention, V)  # (batch_size, num_heads, head_dim)
        attended_values = attended_values.contiguous().view(batch_size, self.feature_dim)

        # Final linear transformation
        attended_values = self.out(attended_values)

        return attended_values

class SimpleSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        # Query, Key, Value transformations
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        # Scale factor (sqrt(feature_dim)) for more stable gradients
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))

    def forward(self, x):
        # x shape: (batch_size, feature_dim)
        # Transform input to get Query, Key and Value
        Q = self.query(x)  # shape: (batch_size, feature_dim)
        K = self.key(x)    # shape: (batch_size, feature_dim)
        V = self.value(x)  # shape: (batch_size, feature_dim)

        # Compute attention scores
        # shape: (batch_size, batch_size)
        attention_scores = torch.matmul(Q, K.transpose(1, 0)) / self.scale.to(x.device)
        # Apply softmax to get probabilities (along the second dimension)
        attention = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum of values
        attended_values = torch.matmul(attention, V)  # shape: (batch_size, feature_dim)

        return attended_values

class SemanticEncoder(nn.Module):

    def __init__(self, args):
        super(SemanticEncoder, self).__init__()
        directory = "/data/home/wxl22/CSRL/checkpoints"
        graphEncoder_path = os.path.join(directory, args.dataset) +"_" + str(args.num_space)+"_" + "graphEncoder.pt"
        imageEncoder_path = os.path.join(directory, args.dataset) +"_" + str(args.num_space)+"_" + "imageEncoder.pt"
        self.args = args
        self.graphEncoder = GraphEncoder(args=args,
                        ratio=args.r,
                        input_dim=39,
                        edge_dim=-1,
                        out_dim=2,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node,
                        c_dim=args.classifier_emb_dim,
                        c_in=args.classifier_input_feat,
                        c_rep=args.contrast_rep,
                        c_pool=args.contrast_pooling,
                        s_rep=args.spurious_rep)
        self.imageEncoder = ImageEncoder()

        # graphEncoder.load_state_dict(torch.load(graphEncoder_path))
        # self.graphEncoder = graphEncoder.eval()

        # imageEncoder.load_state_dict(torch.load(imageEncoder_path))
        # self.imageEncoder = imageEncoder.eval()

        # self.semantic_extraction = torch.nn.Sequential(nn.Linear(args.emb_dim * 2, 2 * args.emb_dim), nn.ReLU(), nn.Linear(2 * args.emb_dim, args.emb_dim))
        # self.semantic_extraction = MultiHeadSelfAttention(args.emb_dim * 2, 4)

        self.semantic_projection = torch.nn.Sequential(nn.Linear(args.emb_dim * 2, 2 * args.emb_dim), nn.ReLU(), nn.Linear(2 * args.emb_dim, args.emb_dim))

        self.spu_mlp = torch.nn.Sequential(nn.Linear(args.emb_dim, 2 * args.emb_dim), nn.ReLU(),nn.Linear(2 * args.emb_dim, args.num_classes))
        self.cq = nn.Linear(args.num_classes, args.num_classes)

        self.semantic_predictor = torch.nn.Sequential(nn.Dropout(args.dropout), self.spu_mlp, self.cq)
        # self.semantic_predictor.apply(xavier_init)
        # self.semantic_predictor = Classifier(args.dropout, args.emb_dim, 2 * args.emb_dim, args.num_classes)
        self.GRL=GRL()
        self.domain_classifier = nn.Sequential(
            nn.Linear(args.emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, args.num_classes)  
        )
        self.graph_domain_classifier = nn.Sequential(
            nn.Linear(args.hid_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, args.num_classes)  
        )


    def forward(self, gs, af, bf,fnf,fef,molf,batch,labels = None, alpha = 0, goal="feat"):
            # obtain the graph embeddings from the featurizer GNN encoder
            h_graph, p_graph, atom_weight = self.graphEncoder(gs, af, bf,fnf,fef,molf,batch,goal="feat", wt="train")
            h_image, p_image = self.imageEncoder(molf, goal="feat")
            SUC_loss = 0
            if goal == "dann":
                SUC_loss = mmc_2(self, h_graph, h_image, p_graph, p_image, labels, self.args.T)

            h_concatenated = torch.cat((h_image, h_graph),dim=1) 
            # h_semantic = self.semantic_extraction(h_concatenated)
            h_semantic = self.semantic_projection(h_concatenated)
            p_semantic = self.semantic_predictor(h_semantic)




            if goal == "pred":
                return p_semantic
            elif goal == "dann":
                h_graph_grl = self.GRL.apply(h_graph, alpha)
                p_graph_domain = self.domain_classifier(h_graph_grl)
                h_image_grl = self.GRL.apply(h_image, alpha)
                p_image_domain = self.domain_classifier(h_image_grl)
                h_semantic = self.GRL.apply(h_semantic, alpha)
                p_semantic_domain = self.domain_classifier(h_semantic)

                return p_semantic,p_graph_domain,p_image_domain,p_semantic_domain,SUC_loss,p_graph, p_image
            else:
                return h_semantic, p_semantic, h_image,h_graph, atom_weight