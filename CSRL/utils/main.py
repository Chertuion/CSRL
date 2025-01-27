import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from drugood.datasets import build_dataset
from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix, average_precision_score, matthews_corrcoef
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wasserstein_distance

from datasets.drugood_dataset import DrugOOD
from models.losses import get_contrast_loss, get_irm_loss
from utils.logger import Logger
from utils.util import args_print, set_seed




@torch.no_grad()
def eval_model(model, device, loader, evaluator, eval_metric='acc', save_pred=False):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            is_labeled = batch.y == batch.y
            if eval_metric == 'acc':
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'rocauc':
                pred = F.softmax(pred, dim=-1)[:, 1].unsqueeze(-1)
                if len(batch.y.size()) == len(batch.y.size()):
                    y_true.append(batch.y.view(-1, 1).detach().cpu())
                    y_pred.append(pred.detach().view(-1, 1).cpu())
                else:
                    y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                    y_pred.append(pred.unsqueeze(-1).detach().cpu())
            elif eval_metric == 'mat':
                y_true.append(batch.y.unsqueeze(-1).detach().cpu())
                y_pred.append(pred.argmax(-1).unsqueeze(-1).detach().cpu())
            elif eval_metric == 'ap':
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            else:
                if is_labeled.size() != pred.size():
                    with torch.no_grad():
                        pred, rep = model(batch, return_data="rep", debug=True)
                        print(rep.size())
                    print(batch)
                    print(global_mean_pool(batch.x, batch.batch).size())
                    print(pred.shape)
                    print(batch.y.size())
                    print(sum(is_labeled))
                    print(batch.y)
                batch.y = batch.y[is_labeled]
                pred = pred[is_labeled]
                y_true.append(batch.y.view(pred.shape).unsqueeze(-1).detach().cpu())
                y_pred.append(pred.detach().unsqueeze(-1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    if eval_metric == 'mat':
        res_metric = matthews_corrcoef(y_true, y_pred)
    else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        res_metric = evaluator.eval(input_dict)[eval_metric]

    if save_pred:
        return res_metric, y_pred
    else:
        return res_metric


def evaluate(pred, gt, metric='auc'):
    if isinstance(metric, str):
        metric = [metric]
    allowed_metric = ['auc', 'acc', 'mcc', 'f1', 'pr_auc', 'recall']
    invalid_metric = set(metric) - set(allowed_metric)
    if len(invalid_metric) != 0:
        raise ValueError(f'Invalid Value {invalid_metric}')
    result = {}
    for M in metric:
        if M == 'auc':
            all_prob = pred[:, 0] + pred[:, 1]
            assert torch.all(torch.abs(all_prob - 1) < 1e-2), \
                "Input should be a binary distribution"
            score = pred[:, 1]
            result[M] = roc_auc_score(gt, score)
        elif M == 'acc':
            pred = pred.argmax(dim=-1)
            total, correct = len(pred), torch.sum(pred.long() == gt.long())
            result[M] = (correct / total).item()
        elif M == 'recall':
            result[M] = recall_score(gt, pred, average='weighted', zero_division=0)
        elif M == 'f1':
            result[M] = f1_score(gt, pred, average='weighted', zero_division=0)
        elif M == 'pr_auc':
            result[M] = average_precision_score(gt, pred)
        elif M == 'mcc':
            result[M] = matthews_corrcoef(gt, pred)
    return result

def eval_one_epoch(model, loader, device, verbose=True):
    model = model.eval()
    result_all, gt_all = [], []
    features = []
    fp_list = []
    for data in (tqdm(loader) if verbose else loader):
        with torch.no_grad():
            data = data.to(device)
            result, feature = model(data)
            result = torch.softmax(result, dim=-1)
            result_all.append(result.detach().cpu())
            gt_all.append(data.y.cpu())
            fp_list.append(data.fp)
            features.append(feature)
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    features = torch.cat(features, dim=0)
    fps = torch.cat(fp_list,dim=0)
    return features, gt_all,fps

def main():
    parser = argparse.ArgumentParser(description='Causality Inspired Invariant Graph LeArning')
    parser.add_argument('--device', default=0, type=int, help='cuda device')
    parser.add_argument('--root', default='/data_1/wxl22/CIGA_vis/data', type=str, help='directory for datasets.')
    parser.add_argument('--dataset', default='label_EC50_core', type=str)
    parser.add_argument('--bias', default='0.33', type=str, help='select bias extend')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')

    # training config
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=50, type=int, help='training iterations')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for the predictor')
    parser.add_argument('--seed', nargs='?', default='[2019,2020,2021,2022]', help='random seed')
    parser.add_argument('--pretrain', default=200, type=int, help='pretrain epoch before early stopping')

    # model config
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--r', default=0.75, type=float, help='selected ratio')
    # emb dim of classifier
    parser.add_argument('-c_dim', '--classifier_emb_dim', default=128, type=int)
    # inputs of classifier
    # raw:  raw feat
    # feat: hidden feat from featurizer
    parser.add_argument('-c_in', '--classifier_input_feat', default='raw', type=str)
    parser.add_argument('--model', default='gin', type=str)
    parser.add_argument('--pooling', default='attention', type=str)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--early_stopping', default=5, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--virtual_node', action='store_true')
    parser.add_argument('--eval_metric',
                        default='auc',
                        type=str,
                        help='specify a particular eval metric, e.g., mat for MatthewsCoef')

    # Invariant Learning baselines config
    parser.add_argument('--num_envs', default=1, type=int, help='num of envs need to be partitioned')
    parser.add_argument('--irm_p', default=1, type=float, help='penalty weight')
    parser.add_argument('--irm_opt', default='irm', type=str, help='algorithms to use')


    # Invariant Graph Learning config
    parser.add_argument('--erm', action='store_true')  # whether to use normal GNN arch
    parser.add_argument('--ginv_opt', default='ginv', type=str)  # which interpretable GNN archs to use
    parser.add_argument('--dir', default=0, type=float)
    parser.add_argument('--contrast_t', default=1.0, type=float, help='temperature prameter in contrast loss')
    # strength of the contrastive reg, \alpha in the paper
    parser.add_argument('--contrast', default=4, type=float)
    parser.add_argument('--not_norm', action='store_true')  # whether not using normalization for the constrast loss
    parser.add_argument('-c_sam', '--contrast_sampling', default='mul', type=str)
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
    parser.add_argument('--spu_coe', default=1, type=float)

    # misc
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--save_model', action='store_true')  # save pred to ./pred if not empty

    args = parser.parse_args()
    erm_model = None  # used to obtain pesudo labels for CNC sampling in contrastive loss

    args.seed = eval(args.seed)

    device = torch.device("cuda")

    def ce_loss(a, b, reduction='mean'):
        return F.cross_entropy(a, b, reduction=reduction)

    criterion = ce_loss
    eval_metric = 'acc' if len(args.eval_metric) == 0 else args.eval_metric
    edge_dim = -1.

    ### automatic dataloading and splitting

    # drugood_lbap_core_ic50_assay.json
    config_path = os.path.join("CIGA/configs", args.dataset + ".py")
    cfg = Config.fromfile(config_path)
    root = args.root
    train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=args.dataset, mode="train")
    val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=args.dataset, mode="ood_val")
    test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=args.dataset, mode="ood_test")
    if args.eval_metric == 'auc':
        evaluator = Evaluator('ogbg-molhiv')
        eval_metric = args.eval_metric = 'rocauc'
    else:
        evaluator = Evaluator('ogbg-ppa')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    input_dim = 39
    edge_dim = 10
    num_classes = 2

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

    all_seed_info = {
        'train_acc':[],
        'test_acc':[],
        'val_acc':[],
    }
    experiment_name = f'{args.dataset}-{args.bias}_{args.ginv_opt}_erm{args.erm}_dir{args.dir}_coes{args.contrast}-{args.spu_coe}_seed{args.seed}_{datetime_now}'
    # experiment_name = f'{datetime_now[4::]}'
    exp_dir = os.path.join('/data_1/wxl22/CIGA_vis/logs', experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    logger.info(f"Using criterion {criterion}")

    logger.info(f"# Train: {len(train_loader.dataset)}  #Val: {len(valid_loader.dataset)} #Test: {len(test_loader.dataset)} ")
    best_weights = None

    all_info = {
        'test_auc': [],
        'train_auc': [],
        'val_auc': [],
        'test_acc': [],
        'train_acc': [],
        'val_acc': [],
        'test_mcc': [],
        'train_mcc': [],
        'val_mcc': [],
        'test_f1': [],
        'train_f1': [],
        'val_f1': [],
        'test_recall': [],
        'train_recall': [],
        'val_recall': [],
        'test_pr_auc': [],
        'train_pr_auc': [],
        'val_pr_auc': [],        
        }
    for seed in args.seed:
        set_seed(seed)
        # models and optimizers

        if args.erm:
            model = GNNERM(input_dim=input_dim,
                           edge_dim=edge_dim,
                           out_dim=num_classes,
                           gnn_type=args.model,
                           num_layers=args.num_layers,
                           emb_dim=args.emb_dim,
                           drop_ratio=args.dropout,
                           graph_pooling=args.pooling,
                           virtual_node=args.virtual_node).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        elif args.ginv_opt.lower() in ['asap']:
            model = GNNPooling(pooling=args.ginv_opt,
                               ratio=args.r,
                               input_dim=input_dim,
                               edge_dim=edge_dim,
                               out_dim=num_classes,
                               gnn_type=args.model,
                               num_layers=args.num_layers,
                               emb_dim=args.emb_dim,
                               drop_ratio=args.dropout,
                               graph_pooling=args.pooling,
                               virtual_node=args.virtual_node).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        elif args.ginv_opt.lower() == 'gib':
            model = GIB(ratio=args.r,
                        input_dim=input_dim,
                        edge_dim=edge_dim,
                        out_dim=num_classes,
                        gnn_type=args.model,
                        num_layers=args.num_layers,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.dropout,
                        graph_pooling=args.pooling,
                        virtual_node=args.virtual_node).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        else:
            model = CIGA(ratio=args.r,
                         input_dim=input_dim,
                         edge_dim=edge_dim,
                         out_dim=num_classes,
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
                         s_rep=args.spurious_rep).to(device)
            model_optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
        print(model)
    # model.load_state_dict(torch.load("/data_1/wxl22/erm_model/label_EC50_core.pt"))
    # model.load_state_dict(torch.load("/data_1/wxl22/MIOOD/logs/sota_label_ec50_scaffold-4-1-20231225-141325/label_ec50_scaffold_2022.pt"))
    train_features, ty, tfps = eval_one_epoch(model, train_loader, device)
    valid_features, vy, vfps = eval_one_epoch(model, valid_loader, device)
    # train_features = torch.cat(train_features, dim=0)
    # valid_features = torch.cat(valid_features, dim=0)
    # # 假设 X 是你的特征向量矩阵，其中每一行是一个数据点的特征

    # class SimpleNN(nn.Module):
    #     def __init__(self):
    #         super(SimpleNN, self).__init__()
    #         emb = 167
    #         self.fc = nn.Linear(emb, emb * 2)
    #         self.pro = nn.Linear(emb * 2, emb)
    #         self.pred = nn.Linear(emb, 2)

    #     def forward(self, x):
    #         fc = self.fc(x)
    #         pro = self.pro(fc)
    #         x = self.pred(pro)
    #         return x, pro

    # # 实例化并使用模型
    # model = SimpleNN().to("cuda:0")
    # model.load_state_dict(torch.load("/data_1/wxl22/fp_model/label_EC50_core.pt"))
    # _, tfps = model(tfps)
    # _, vfps = model(vfps)

    # # 创建 t-SNE 实例
    tsne = TSNE(n_components=2, verbose=1, perplexity=35, n_iter=750)
    # # 进行降维
    # train_results = tsne.fit_transform(tfps.detach().cpu())
    # valid_results = tsne.fit_transform(vfps.detach().cpu())
    train_results = tsne.fit_transform(train_features.cpu())
    valid_results = tsne.fit_transform(valid_features.cpu())



    train_colors = [[59/255, 144/255, 217/255],[230/255, 185/255, 117/255]]
    valid_colors = [[174/255, 78/255, 137/255],[63/255, 45/255, 107/255]]
    train_l = ['y = 0 (train)', 'y = 1 (train)']
    valid_l = ['y = 0 (valid)', 'y = 1 (valid)']

    # 绘制训练集点
    for label, color, idx in zip([0, 1], train_colors, train_l):
        indices = [i for i, l in enumerate(ty) if l == label]
        plt.scatter(train_results[indices, 0], train_results[indices, 1], color=color, label=idx, s=2)

    # 绘制验证集点
    for label, color, idx in zip([0, 1], valid_colors, valid_l):
        indices = [i for i, l in enumerate(vy) if l == label]
        plt.scatter(valid_results[indices, 0], valid_results[indices, 1], color=color, label=idx, s=2)



    # # 合并训练集和验证集的结果和标签
    # all_results = np.vstack((train_results, valid_results))
    # all_labels = np.hstack((ty, vy))  # 假设ty是训练集的标签，vy是验证集的标签

    # # 提取每个类别的坐标
    # coords_0 = all_results[all_labels == 0]
    # coords_1 = all_results[all_labels == 1]

    train_0 = train_results[ty == 0]
    train_1 = train_results[ty == 1]
    valid_0 = valid_results[vy == 0]
    valid_1 = valid_results[vy == 1]

    # 计算Wasserstein distance
    distance_x = wasserstein_distance(train_0[:, 0], valid_0[:, 0])
    distance_y = wasserstein_distance(train_1[:, 1], valid_1[:, 1])

    # 计算平均距离
    # average_distance = (distance_x + distance_y) / 2
    # 添加图例
    plt.legend()
    # 可选：添加标题和轴标签
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # 添加文本显示Wasserstein distance
    plt.text(0, 1, f'Wasserstein distance: D(Y=0):{distance_x:.4f}, D(Y=1):{distance_y:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig('/data_1/wxl22/CIGA_vis/tsne_fusion_EMD.png', dpi=900)
    plt.show()



    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制训练集点
    # for label, color, idx in zip([0, 1], train_colors, train_l):
    #     indices = [i for i, l in enumerate(ty) if l == label]
    #     ax.scatter(train_results[indices, 0], train_results[indices, 1], train_results[indices, 2], color=color, label=idx, s=2)

    # # 绘制验证集点
    # for label, color, idx in zip([0, 1], valid_colors, valid_l):
    #     indices = [i for i, l in enumerate(vy) if l == label]
    #     ax.scatter(valid_results[indices, 0], valid_results[indices, 1], valid_results[indices, 2], color=color, label=idx, s=2)

    # # 添加图例
    # plt.legend()

    # # 可选：调整视角以获得侧视图
    # ax.view_init(elev=0., azim=270)

    # plt.savefig('/data_1/wxl22/CIGA_vis/tsne_fp_3d.png', dpi=900)
    # plt.show()


if __name__ == "__main__":
    main()
