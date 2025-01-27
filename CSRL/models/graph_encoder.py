import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch
from torch_geometric.nn import (ASAPooling, global_add_pool, global_max_pool,
                                global_mean_pool)
from utils.get_subgraph import relabel, split_batch
from utils.mask import clear_masks, set_masks
import dgl
from models.conv import GNN_node, GNN_node_Virtualnode
from models.gnn import GNN, LeGNN
import math
from torch_geometric.nn import GlobalAttention
from dgl.nn import NNConv


class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.bn = nn.BatchNorm1d(out_channel)
        #self.bn=nn.LayerNorm(in_channel)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        size = x.size()
        x = x.view(-1, x.size()[-1], 1)
        x = self.bn(x)
        x = x.view(size)
        if self.act is not None:
            x = self.act(x)
        return x

class DGL_MPNNLayer(nn.Module):
    """message passing layer"""
    def __init__(self,
                 hid_dim,
                 edge_func,
                 resdual,
                 ):
        super(DGL_MPNNLayer, self).__init__()
        self.hidden_dim=hid_dim
        self.node_conv=NNConv(self.hidden_dim,self.hidden_dim,edge_func,'sum',resdual)
        
    def forward(self,g,nf,initial_ef):
        unm=self.node_conv(g,nf,initial_ef)
        return unm

class AttentionAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, Mdim):
        super(AttentionAgger, self).__init__()
        self.model_dim = Mdim
        self.WQ = torch.nn.Linear(Qdim, Mdim)
        self.WK = torch.nn.Linear(Qdim, Mdim)

    def forward(self, Q, K, V, mask=None):
       
        Q, K = self.WQ(Q), self.WK(K)
     
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        if mask is not None:
           
            Attn = torch.masked_fill(Attn, mask.bool(), -(1 << 32))
        Attn = torch.softmax(Attn, dim=-1)
     
        Attn.diagonal().fill_(0)
 
        Attn = torch.sum(Attn, dim=1)
        # Attn /= torch.sum(Attn)
        return Attn

class Readout(nn.Module):
    def __init__(self,args,ntype:str,use_attention:bool):
        super(Readout,self).__init__()
        self.ntype=ntype
        self.use_attention=use_attention
        self.linear = nn.Linear(args.hid_dim, 1)
    def forward(self,g,nf):
        if self.use_attention:
            g.nodes[self.ntype].data['nw']=self.linear(nf)
            weights=dgl.softmax_nodes(g,'nw',ntype=self.ntype)
            with g.local_scope():
                g.nodes[self.ntype].data['w'] = weights
                g.nodes[self.ntype].data['feat'] = nf
                weighted_mean_rd = dgl.readout_nodes(g, 'feat','w',op='sum', ntype=self.ntype)
                max_rd = dgl.readout_nodes(g, 'feat', op='max', ntype=self.ntype)
                ###
                return torch.cat([weighted_mean_rd,max_rd],dim=1), weights
        else:
            with g.local_scope():
                g.nodes[self.ntype].data['feat'] = nf
                mean_rd = dgl.readout_nodes(g, 'feat',op='mean', ntype=self.ntype)
                max_rd = dgl.readout_nodes(g, 'feat', op='max', ntype=self.ntype)
                return torch.cat([mean_rd,max_rd],dim=1)
                #return mean_rd

class LocalAugmentation(nn.Module):
    def __init__(self,args):
        super(LocalAugmentation,self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(args.hid_dim, args.hid_dim,bias=False) for _ in range(3)])
        self.W_o=nn.Linear(args.hid_dim, args.hid_dim)
        self.heads=args.heads
        self.d_k=args.hid_dim//args.heads
    def forward(self,fine_messages,coarse_messages,motif_features):
        batch_size=fine_messages.shape[0]
        hid_dim=fine_messages.shape[-1]
        Q=motif_features
        K=[]
        K.append(fine_messages.unsqueeze(1))
        K.append(coarse_messages.unsqueeze(1))
        K=torch.cat(K,dim=1)
        Q=Q.view(batch_size, -1, 1,hid_dim).transpose(1, 2)
        K=K.view(batch_size, -1, 1,hid_dim).transpose(1, 2)
        V=K
        Q, K, V = [l(x).view(batch_size, -1,self.heads,self.d_k).transpose(1, 2)
                                      for l, x in zip(self.linear_layers, (Q,K,V))]
        #print(Q[0],K.transpose(-2, -1)[0])
        message_interaction=torch.matmul( Q,K.transpose(-2, -1))/self.d_k
        #print(message_interaction[0])
        att_score=torch.nn.functional.softmax(message_interaction,dim=-1)
        motif_messages=torch.matmul(att_score, V).transpose(1, 2).contiguous().view(batch_size, -1, hid_dim)
        motif_messages=self.W_o(motif_messages)
        return motif_messages.squeeze(1)

class GraphEncoder(nn.Module):

    def __init__(self,
                 args,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 do = "do",
                 sigma_len=3):
        super(GraphEncoder, self).__init__()
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
#################
        #define encoders to generate the atom and substructure embeddings
        self.atom_encoder = nn.Sequential(
            LinearBn(args.atom_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )
        self.motif_encoder = nn.Sequential(
            LinearBn(args.ss_node_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )

        self.step=args.step
        self.agg_op=args.agg_op
        self.mol_FP=args.mol_FP
        #define the message passing layer
        self.motif_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.ss_edge_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        self.atom_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.bond_in_dim,args.hid_dim*args.hid_dim),args.resdual)

        #define the update function
        self.motif_update=nn.GRUCell(args.hid_dim,args.hid_dim)
        self.atom_update=nn.GRUCell(args.hid_dim,args.hid_dim)

        #define the readout layer
        self.atom_readout=Readout(args,ntype='atom',use_attention=args.attention)
        self.motif_readout=Readout(args,ntype='func_group',use_attention=args.attention)
        self.LA=LocalAugmentation(args)
###################



        self.ratio = ratio
        self.edge_att = nn.Sequential(nn.Linear(args.hid_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1))
        # self.edge_att = AttentionAgger(emb_dim * 2, emb_dim * 2, emb_dim * 2)
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool
        self.do = do

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=39,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)
        self.causal_rep_projection = nn.Sequential(nn.Linear(emb_dim + args.hid_dim * 2 + args.mol_in_dim, emb_dim * 4), nn.ReLU(),nn.Linear(emb_dim * 4, emb_dim))
        self.combine_rep_projection =  nn.Sequential(nn.Linear(args.hid_dim * 4, emb_dim * 2), nn.ReLU(),nn.Linear(emb_dim * 2, emb_dim))


        self.log_sigmas = nn.Parameter(torch.zeros(sigma_len))
        self.log_sigmas.requires_grad_(True)

        self.spu_mlp = torch.nn.Sequential(nn.Linear(args.hid_dim * 4, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, 2))
        self.combine_mlp = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, 2))
        self.cq = nn.Linear(2, 2)
        self.spu_fw = torch.nn.Sequential(self.spu_mlp, self.cq)
        self.combine_fw = torch.nn.Sequential(self.combine_mlp, self.cq)



    def forward(self,g, af, bf, fnf, fef,mf, batch, goal="pred", wt="pretrain", return_spu=True, debug=False):
        with g.local_scope():
            #generate atom and substructure embeddings
            ufnf=self.motif_encoder(fnf)
            uaf=self.atom_encoder(af)

            #message passing and uodate
            for i in range(self.step):
                ufnm=self.motif_mp_layer(g[('func_group', 'interacts', 'func_group')],ufnf,fef)
                uam=self.atom_mp_layer(g[('atom', 'interacts', 'atom')],uaf,bf)
                g.nodes['atom'].data['_uam']=uam
                if self.agg_op=='sum':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.sum('uam','agg_uam'),\
                            etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op=='max':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.max('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op=='mean':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.mean('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))
                augment_ufnm=g.nodes['func_group'].data['agg_uam']
                #local augmentation
                ufnm=self.LA(augment_ufnm,ufnm,ufnf)
                #ufnm=torch.cat([ufnm,augment_ufnm],dim=1)

                ufnf=self.motif_update(ufnm,ufnf)
                uaf=self.atom_update(uam,uaf)
            #readout
            motif_readout, motif_nf=self.motif_readout(g,ufnf)
            atom_readout, atom_weight=self.atom_readout(g,uaf)
            ##############################
            atom_representation=atom_readout
            motif_representation=motif_readout

            ##############################
            if self.mol_FP=='atom':
                atom_representation=torch.cat([atom_readout,mf],dim=1)
            elif self.mol_FP=='ss':
                motif_representation=torch.cat([motif_readout,mf],dim=1)
            elif self.mol_FP=='both':
                atom_representation=torch.cat([atom_readout,mf],dim=1)
                motif_representation=torch.cat([motif_readout,mf],dim=1)
            #############################

        h_graph = torch.cat([atom_representation, motif_representation], dim=1)

        if wt == "train":
            h_graph = self.combine_rep_projection(h_graph)
            p_graph = self.combine_fw(h_graph)
        # elif wt == "gnn":
        #     p_graph,h_graph = self.classifier(batch, get_rep=True)
        else:
            p_graph = self.spu_fw(h_graph)
        if goal == "pred":
            return p_graph
        return h_graph, p_graph, atom_weight
