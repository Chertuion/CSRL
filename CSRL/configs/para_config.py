import argparse

#模型参数
def init_args():
    parser = argparse.ArgumentParser('graph Mutual Information for OOD')
    # base config
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--root', default='/data/home/wxl22/CSRL/data',type=str, help='root for datasets')
    parser.add_argument('--data_config', default='/data/home/wxl22/CSRL/configs', type=str, help='root for data config')
    parser.add_argument('--seed', default='[0,1,3]', help=' random seed')
    parser.add_argument('--dataset', default='tox21', type=str, help='name for datasets')
    parser.add_argument('--dataset_type', default='tox21', type=str, help='name for datasets')
    # parser.add_argument('--cuda', default=0, type=int, help='select cuda id')
    parser.add_argument('--log_dir', default='/data/home/wxl22/CSRL/logs', type=str, help='root for logs')
    parser.add_argument('--input_dim', default=39, type=int)
    parser.add_argument('--edge_dim', default=-1, type=int)
    parser.add_argument('--num_classes', default=2, type=int)

    #mol represent config
    parser.add_argument('--mol_FP',type=str,choices=['atom','ss','both','none'],default='none',help='cat mol FingerPrint to Motif or Atom representation')
    parser.add_argument('--atom_in_dim', type=int, default=37, help='atom feature init dim')
    parser.add_argument('--bond_in_dim', type=int, default=13, help='bond feature init dim')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate in MLP')
    parser.add_argument('--hid_dim', type=int, default=96, help='node, edge, fg hidden dims in Net')
    parser.add_argument('--mol_in_dim', type=int, default=167, help='molecule fingerprint init dim')
    parser.add_argument('--ss_node_in_dim', type=int, default=50, help='func group node feature init dim')
    parser.add_argument('--ss_edge_in_dim', type=int, default=37, help='func group edge feature init dim')
    parser.add_argument('--step',type=int,default=4,help='message passing steps')
    parser.add_argument('--agg_op',type=str,choices=['max','mean','sum'],default='mean',help='aggregations in local augmentation')
    parser.add_argument('--resdual',type=bool,default=False,help='resdual choice in message passing')
    parser.add_argument('--attention',type=bool,default=True,help='whether to use global attention pooling')
    parser.add_argument('--heads',type=int,default=4,help='Multi-head num')

    #semetic inference
    parser.add_argument('--epoch_ast', default=120, type=int)
    parser.add_argument('--num_space', default=50, type=int)
    parser.add_argument('--fp_dim', default=2048, type=int)
    parser.add_argument('--semetic_dim', default=128, type=int)
    parser.add_argument('--dist', default='gaussian', type=str, help='the prior distribution of ELBO')

    #latent diffusion
    parser.add_argument('--latent_dim', default=128, type=int)

    # training config
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=30, type=int, help='training iterations')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for the predictor')

    # model config
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--r', default=0.5, type=float, help='selected ratio')
    # emb dim of classifier
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=128, type=int)
    # inputs of classifier
    # raw:  raw feat
    # feat: hidden feat from featurizer
    parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='attention', type=str)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric',default='rocauc',type=str,help='specify a particular eval metric, e.g., mat for MatthewsCoef')

    # contrasting summary from the classifier or featurizer
    # rep:  classifier rep
    # feat: featurizer rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-c_rep', '--contrast_rep', default='rep', type=str)
    # pooling method for the last two options in c_rep
    parser.add_argument('-c_pool', '--contrast_pooling', default='add', type=str)

    # spurious rep for maximizing I(G_S;Y)
    # rep:  classifier rep
    # conv: featurizer rep + 1L GNNConv
    parser.add_argument('-s_rep', '--spurious_rep', default='rep', type=str)
    # strength of the hinge reg, \beta in the paper
    parser.add_argument('--sl_coe', default= 10, type=float)
    parser.add_argument('--ml_coe', default = 0.1, type=float)
    parser.add_argument('--uni_coe', default = 1, type=float)
    parser.add_argument('--T', default=4.1, type=float)
    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--save_model', default='true', type=bool)  # save pred to ./pred if not empty
    parser.add_argument('--note', default='no', type=str)
    # parser.add_argument('--irm_opt', default='vrex', type=str)
    parser.add_argument('--drop_early', default=0, type=int)

    args = parser.parse_args()
    return args