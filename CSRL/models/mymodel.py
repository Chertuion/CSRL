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
        #将Q，K映射到同一维度
        Q, K = self.WQ(Q), self.WK(K)
        # 把点成的结果除以一个常数，
        # 这个值一般是采用上文提到的矩阵的第一个维度的开方，
        # 当然也可以选择其他的值，然后把得到的结果做一个softmax的计算。
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)
        if mask is not None:
            #  -(1 << 32)表示二进制下第32位为1，其他位为0的数取反后再加1，即为一个非常小的负数。
            Attn = torch.masked_fill(Attn, mask.bool(), -(1 << 32))
        Attn = torch.softmax(Attn, dim=-1)
        # 将注意力权重矩阵的对角线上的值设置为0
        Attn.diagonal().fill_(0)
        # 对每个边的权重进行求和，得到每条边相对于整体的重要性程度
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
                return torch.cat([weighted_mean_rd,max_rd],dim=1), nf
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

class mymodel(nn.Module):

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
        super(mymodel, self).__init__()
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
                              input_dim=self.c_input_dim,
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
        self.mol_rep_projection =  nn.Sequential(nn.Linear(args.hid_dim * 2 + args.mol_in_dim * 1, emb_dim * 4), nn.ReLU(),nn.Linear(emb_dim * 4, emb_dim))

        self.mx_classifier = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(),
                                            nn.Linear(emb_dim * 4, emb_dim * 2), nn.Linear(emb_dim * 2, out_dim), 
                                            nn.Linear(out_dim, out_dim))
        self.log_sigmas = nn.Parameter(torch.zeros(sigma_len))
        self.log_sigmas.requires_grad_(True)



    def forward(self,g, af, bf, fnf, fef,mf, batch, return_data="pred", return_spu=True, debug=False):
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
            atom_readout, atom_nf=self.atom_readout(g,uaf)
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

        mol_representation = torch.cat([atom_representation, motif_representation], dim=1)


        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        device = h.device
        # seperate the input graphs into \hat{G_c} and \hat{G_s}
        # using edge-level attetion
        # row--> start node; col--> end node
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([atom_nf[row], atom_nf[col]], dim=-1)

        edge_indices, num_nodes, cum_nodes, num_edges, cum_edges = split_batch(batch)
        mask = torch.zeros(batch.edge_index.shape[1]).to(device)
        mask_arr = []
        for N, C in zip(num_edges, cum_edges):
            mask = torch.zeros(batch.edge_index.shape[1])
            mask[C:C + N] = 1
            for i in range(N):
                mask_arr.append(mask)
        graph_edge_mask = torch.stack(mask_arr, dim=0).int().to(device)


        # pred_edge_weight = self.edge_att(edge_rep, edge_rep, edge_rep, mask=graph_edge_mask).view(-1)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)

        # set_masks(pred_edge_weight, self.classifier)
        # graph_pred, graph_rep = self.classifier(batch, get_rep = True)
        # clear_masks(self.classifier)

        graph_rep = self.mol_rep_projection(motif_representation)


        causal_edge_index = torch.LongTensor([[], []]).to(device)
        causal_edge_weight = torch.tensor([]).to(device)
        causal_edge_attr = torch.tensor([]).to(device)
        spu_edge_index = torch.LongTensor([[], []]).to(device)
        spu_edge_weight = torch.tensor([]).to(device)
        spu_edge_attr = torch.tensor([]).to(device)


        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
            n_reserve = int(self.ratio * N)

            edge_attr = batch.edge_attr[C:C + N]
            single_mask = pred_edge_weight[C:C + N]
            single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
            rank = np.argpartition(-single_mask_detach, n_reserve)
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
            if debug:
                print(n_reserve)
                print(idx_reserve)
                print(idx_drop)
            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            spu_edge_index = torch.cat([spu_edge_index, edge_index[:, idx_drop]], dim=1)

            causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
            spu_edge_weight = torch.cat([spu_edge_weight, -1 * single_mask[idx_drop]])

            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            spu_edge_attr = torch.cat([spu_edge_attr, edge_attr[idx_drop]])

        if self.c_in.lower() == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        else:
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)



        # obtain \hat{G_c}
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        # obtain predictions with the classifier based on \hat{G_c}
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)
        causal_rep = torch.cat([causal_rep, motif_representation], dim=1)
        causal_rep = self.causal_rep_projection(causal_rep)


        # whether to return the \hat{G_s} for further use
        spu_graph = DataBatch.Batch(batch=spu_batch,
                                        edge_index=spu_edge_index,
                                        x=spu_x,
                                        edge_attr=spu_edge_attr)
        set_masks(spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
        clear_masks(self.classifier)
        causal_pred_ex = (causal_pred, spu_pred)

        if self.do.lower() == "do":
            if return_data.lower() == "rep":
                causal_rep_expanded = causal_rep.unsqueeze(1).expand(-1, len(spu_rep), -1)
                spu_rep_expanded = spu_rep.unsqueeze(0).expand(len(causal_rep), -1, -1)
                mx_rep = torch.cat((causal_rep_expanded, spu_rep_expanded), dim=2)
                # flatten the first two dimensions
                mx_rep = mx_rep.view(-1, mx_rep.shape[-1])
                mx_pred = self.mx_classifier(mx_rep)

            if return_data.lower() == "pred":
                mx_rep = torch.cat((causal_rep, spu_rep), dim = -1)
                mx_pred = self.mx_classifier(mx_rep)
        else:
            mx_pred = causal_pred


        if return_data.lower() == "pred":
            return mx_pred
        elif return_data.lower() == "rep":
            return causal_pred_ex, causal_rep, spu_rep, graph_rep, mx_pred
        elif return_data.lower() == "feat":
            causal_h, _, __, ___ = relabel(h, causal_edge_index, batch.batch)
            if self.c_pool.lower() == "add":
                casual_rep_from_feat = global_add_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "max":
                casual_rep_from_feat = global_max_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "mean":
                casual_rep_from_feat = global_mean_pool(causal_h, batch=causal_batch)
            else:
                raise Exception("Not implemented contrastive feature pooling")

            return causal_pred, casual_rep_from_feat
        else:
            return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight