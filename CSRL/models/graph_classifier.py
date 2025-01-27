import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphClassifier(nn.Module):
    def __init__(self, num_class=2, emb_dim=128,):
            super(GraphClassifier, self).__init__()
            self.num_class = num_class
            self.graph_pred_linear = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                                            nn.Linear(2 * emb_dim, self.num_class))
            self.spu_mlp = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                                nn.Linear(2 * emb_dim, self.num_class))
            self.cq = nn.Linear(self.num_class, self.num_class)
            self.spu_fw = torch.nn.Sequential(self.spu_mlp, self.cq)

    def forward(self, h_semantic):
        # h_concat = torch.cat((h_semantic,h_graph,h_image), dim=1)
        return self.spu_fw(h_semantic)

