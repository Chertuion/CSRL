#导包
import argparse
import os
import os.path as osp
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
# from sklearn.decomposition import PCA
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
# from datasets.drugood_dataset import DrugOOD
# from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
# from drugood.datasets import build_dataloader, build_dataset
# from drugood.models import build_backbone
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from mmcv import Config
# from joblib import dump, load
# from models.mymodel import mymodel
# from models.semetic_inference import DomainDiscriminator
# from models.losses import KLDist, MeanLoss, DeviationLoss, discrete_gaussian
# from torch.nn.functional import cross_entropy
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split

# from torch_geometric.nn import global_mean_pool
# from torch_geometric.data import DataLoader
from tqdm import tqdm
from utils.logger import Logger
from utils.util import args_print, set_seed, gamma_positional_encoding_vector, eval_one_epoch,get_prior
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix, average_precision_score, matthews_corrcoef
from sklearn.neighbors import KernelDensity
import json
# from diffusers import DiffusionPipeline
# from torch.utils.data import DataLoader
from models.graph_classifier import GraphClassifier
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import io
import warnings
from configs.para_config import init_args

from sklearn import metrics
from sklearn.metrics import mutual_info_score
import warnings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from models.semantic_encoder import SemanticEncoder
from models.graph_classifier import GraphClassifier
from models.graph_encoder import GraphEncoder
from models.image_encoder import ImageEncoder
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim
from utils.drawer import draw_tsne, draw_tsne_3d
from sklearn.mixture import GaussianMixture
from models.discriminators import Discriminator1, Discriminator2
import matplotlib.pyplot as plt
from datasets.mymodel_dataset import GraphDataset_Classification, GraphDataLoader_Classification
from datasets.mymodel_dataset import DILIDataset
import dgl
from configs.para_config import init_args
import statistics

def filter_data(gs, ls, pyg):
    filtered_train_gs = []
    filtered_train_ls = []
    filtered_train_pygs = []
    # singal_pyg = torch_geometric.data.Batch.to_data_list(pyg[0].Data)
    for g, l, p in zip(gs, ls, pyg):
        if g is not None:
            filtered_train_gs.append(g)
            filtered_train_ls.append(l)
            filtered_train_pygs.append(p)
    return filtered_train_gs, filtered_train_ls, filtered_train_pygs



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
            score = pred[:, 1]
            result[M] = roc_auc_score(gt, score)
        elif M == 'acc':
            pred = torch.tensor(pred)
            gt = torch.tensor(gt)     
            pred = pred.clone().detach()
            gt = gt.clone().detach()
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

def eval_graphEncoder_epoch(model, dl, device, goal="pred"):
    model = model.eval()
    traYAll = []
    result_all, gt_all = [], []
    for step, (gs, labels, pygs) in enumerate(dl):
        labels = labels.to(device).long()
        pygs = pygs.to(device)
        gs = gs.to(device)
        af = gs.nodes['atom'].data['feat']
        bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
        fnf = gs.nodes['func_group'].data['feat']
        fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
        molf=gs.nodes['molecule'].data['feat']

        traYAll += labels.detach().cpu().numpy().tolist()
        with torch.no_grad():
            result = model(gs, af, bf,fnf,fef,molf,pygs, goal)
            result = torch.softmax(result, dim=-1)
            result_all.append(result.detach().cpu())
    result_all = torch.cat(result_all, dim=0)
    # return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'acc', 'mcc', 'f1', 'recall', 'pr_auc'])
    return evaluate(pred=result_all, gt=traYAll, metric=['auc'])

def eval_semanticEncoder_epoch(model, dl, device, goal="pred"):
    model = model.eval()
    traYAll = []
    result_all, gt_all = [], []
    for step, (gs, labels, pygs) in enumerate(dl):
        labels = labels.to(device).long()
        pygs = pygs.to(device)
        gs = gs.to(device)
        af = gs.nodes['atom'].data['feat']
        bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
        fnf = gs.nodes['func_group'].data['feat']
        fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
        molf=gs.nodes['molecule'].data['feat']

        traYAll += labels.detach().cpu().numpy().tolist()
        with torch.no_grad():
            result = model(gs, af, bf,fnf,fef,molf,pygs, goal="pred")
            result = torch.softmax(result, dim=-1)
            result_all.append(result.detach().cpu())
    result_all = torch.cat(result_all, dim=0)
    # return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'acc', 'mcc', 'f1', 'recall', 'pr_auc'])
    return evaluate(pred=result_all, gt=traYAll, metric=['auc'])

def eval_imageEncoder_epoch(model, dl, device, verbose=True):
    model = model.eval()
    traYAll = []
    result_all, gt_all = [], []
    for step, (gs, labels, data) in enumerate(dl):
        traYAll += labels.detach().cpu().numpy().tolist()
        molf=gs.nodes['molecule'].data['feat']
        molf = molf.to(device)
        with torch.no_grad():
            data = data.to(device)
            result = model(molf)
            result = torch.softmax(result, dim=-1)
            result_all.append(result.detach().cpu())
    result_all = torch.cat(result_all, dim=0)

    return evaluate(pred=result_all, gt=traYAll, metric=['auc'])


def eval_one_epoch(encoder, classifier, loader, device, verbose=True):
    encoder = encoder.eval()
    classifier = classifier.eval()
    result_all, gt_all = [], []
    for data in (tqdm(loader) if verbose else loader):
        with torch.no_grad():
            data = data.to(device)
            encoded = encoder(data)
            result = classifier(encoded)
            result = torch.softmax(result, dim=-1)
            result_all.append(result.detach().cpu())
            gt_all.append(data.y.cpu())
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'acc', 'mcc', 'f1', 'recall', 'pr_auc'])

def eval_one_epoch_full(model, loader, device, verbose=True):
    model = model.eval()
    result_all, gt_all = [], []
    for data in (tqdm(loader) if verbose else loader):
        with torch.no_grad():
            data = data.to(device)
            result = model(data, goal="pred")
            result = torch.softmax(result, dim=-1)
            result_all.append(result.detach().cpu())
            gt_all.append(data.y.cpu())
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'acc', 'mcc', 'f1', 'recall', 'pr_auc'])

def get_dis_info_loss(domain_dis):

    domain_dis = torch.softmax(domain_dis, dim=-1)
    variances = domain_dis.var(dim=0)  
    average_variance = variances.mean() 

    entropies = -torch.sum(domain_dis * torch.log(domain_dis), dim=1) 
    average_entropy = entropies.mean() 

    n_classes = 2 
    max_entropy = math.log(n_classes)

    return average_variance.item(), max_entropy - average_entropy.item()


if __name__ ==  "__main__":

    args = init_args()
    args.seed = eval(args.seed)
    all_seed_info = {
            'train_auc': [],
            'val_auc': [],
            'test_auc': [],
        }

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")


    experiment_name = f'{args.dataset}-{args.ml_coe}-{args.sl_coe}-{args.uni_coe}-{args.T}-{datetime_now}'
    exp_dir = os.path.join(args.log_dir, experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    for seed in args.seed:
        set_seed(seed)

        data_config_path = os.path.join(args.data_config, args.dataset+'.py')
        # dataset_config = Config.fromfile(data_config_path)

        root = os.path.join(args.root, args.dataset)
        if not os.path.exists(root):
            os.mkdir(root)
        if args.dataset_type.lower() == 'admeood':
                train_dataset = DILIDataset(root = root, mode="train", name=args.dataset)
                train_gs, train_ls, train_pyg = filter_data(train_dataset.gt_data_gs, train_dataset.gt_data_ls, train_dataset.pygGraphs)
                valid_dataset = DILIDataset(root = root, mode="ood_val", name=args.dataset)
                valid_gs, valid_ls, valid_pyg = filter_data(valid_dataset.gt_data_gs, valid_dataset.gt_data_ls,valid_dataset.pygGraphs)
                test_dataset = DILIDataset(root = root, mode="ood_test", name=args.dataset)
                test_gs, test_ls, test_pyg = filter_data(test_dataset.gt_data_gs, test_dataset.gt_data_ls,valid_dataset.pygGraphs)
        elif args.dataset_type.lower() in ['bbbp', 'clintox', 'tox21']:

                train_dataset = DILIDataset(root = root, mode="train_seed1", name=args.dataset)
                train_gs, train_ls, train_pyg = filter_data(train_dataset.gt_data_gs, train_dataset.gt_data_ls, train_dataset.pygGraphs)
                valid_dataset = DILIDataset(root = root, mode="val_seed1", name=args.dataset)
                valid_gs, valid_ls, valid_pyg = filter_data(valid_dataset.gt_data_gs, valid_dataset.gt_data_ls,valid_dataset.pygGraphs)
                test_dataset = DILIDataset(root = root, mode="test_seed1", name=args.dataset)
                test_gs, test_ls, test_pyg = filter_data(test_dataset.gt_data_gs, test_dataset.gt_data_ls,valid_dataset.pygGraphs)



        if args.dataset_type.lower() == 'admeood':
            train_ds = GraphDataset_Classification(train_gs, train_ls, train_pyg)
            train_dl = GraphDataLoader_Classification(train_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=True, drop_last = True)
            valid_ds = GraphDataset_Classification(valid_gs, valid_ls, valid_pyg)
            valid_dl = GraphDataLoader_Classification(valid_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=True, drop_last = True)
            test_ds = GraphDataset_Classification(test_gs, test_ls, test_pyg)
            test_dl = GraphDataLoader_Classification(test_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=True, drop_last = True)
        elif args.dataset_type.lower() in ['bbbp', 'tox21']:
            train_ds = GraphDataset_Classification(train_gs, train_ls, train_pyg)
            train_dl = GraphDataLoader_Classification(train_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=True, drop_last = True)
            valid_ds = GraphDataset_Classification(valid_gs, valid_ls, valid_pyg)
            valid_dl = GraphDataLoader_Classification(valid_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=True, drop_last = True)
            test_ds = GraphDataset_Classification(test_gs, test_ls, test_pyg)
            test_dl = GraphDataLoader_Classification(test_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=True, drop_last = True)
        elif args.dataset_type.lower() in ['clintox']:
            train_ds = GraphDataset_Classification(train_gs, train_ls, train_pyg)
            train_dl = GraphDataLoader_Classification(train_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=False, drop_last = True)
            valid_ds = GraphDataset_Classification(valid_gs, valid_ls, valid_pyg)
            valid_dl = GraphDataLoader_Classification(valid_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=False, drop_last = True)
            test_ds = GraphDataset_Classification(test_gs, test_ls, test_pyg)
            test_dl = GraphDataLoader_Classification(test_ds, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=False, drop_last = True)


        

            print(f"训练数据集的样本数量: {len(train_ds)}")

            # train_gs = dgl.batch(train_gs).to(args.device)
            # train_labels=torch.tensor(train_ls).to(args.device)

            # val_gs = dgl.batch(valid_gs).to(args.device)
            # val_labels=torch.tensor(valid_ls).to(args.device)

            # test_gs=dgl.batch(test_gs).to(args.device)
            # test_labels=torch.tensor(test_ls).to(args.device)



        input_dim = 39
        edge_dim = 10
        num_classes = 2


        device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
        torch.cuda.set_device(device)

        def ce_loss(a, b, reduction='mean'):
            return F.cross_entropy(a, b, reduction=reduction)

        criterion = ce_loss
        edge_dim = -1


        #save path of encoder and classifier
        directory = "/data/home/wxl22/CSRL/checkpoints"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # domainClassifier_path = os.path.join(directory, args.dataset) +"_" + str(seed)+"_" + "domainClassifier.pt"
        dis_loss_path = os.path.join(directory, args.dataset) +"_" + str(seed)+"_" + "loss_info.json"
        # graphEncoder_path = os.path.join(directory, args.dataset) +"_" + str(seed)+"_" + "graphEncoder.pt"
        # imageEncoder_path = os.path.join(directory, args.dataset) +"_" + str(seed)+"_" + "imageEncoder.pt" 3

        semanticEncoder_path = os.path.join(directory, args.dataset) +"_" + str(seed)+"_" + "semanticEncoder.pt"

        logger.info(f"Using criterion {criterion}")
        logger.info(f"#Train: {len(train_dl.dataset)}  #Val: {len(valid_dl.dataset)} #Test: {len(test_dl.dataset)} ")
        best_encoder_weights = []
        best_classifier_weights = []
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



        semanticEncoder = SemanticEncoder(args).to(device)
        discriminator_image = Discriminator1().to(device)
        discriminator_graph = Discriminator2().to(device)

        # dann = DomainDiscriminator().to(device)
 
        eps = 1e-8
        if True:
            #train encoder and classifier
            criterion = nn.CrossEntropyLoss()
            dis_image_optimizer = AdamW(discriminator_image.parameters(), lr=args.lr)
            dis_graph_optimizer = AdamW(discriminator_graph.parameters(), lr=args.lr)
            semantic_optimizer = optim.Adam(semanticEncoder.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-4)


            num_batch = (len(train_dl.dataset) // args.batch_size) + int((len(train_dl.dataset) % args.batch_size) > 0)
            env_idx = (torch.sigmoid(torch.randn(len(train_dl.dataset))) > 0.5).long()
            print(f"num env 0: {sum(env_idx == 0)} num env 1: {sum(env_idx == 1)}")

            valid_curv, train_curv, test_curv ={}, {}, {}

            loss_dis = []
            for epoch in range(args.epochs):
                all_loss, n_bw, all_semantic_loss, all_dis_image_loss, all_dis_graph_loss = 0, 0, 0, 0, 0
                all_losses = {}
                semanticEncoder.train()
                discriminator_image.train()
                discriminator_graph.train()
                for step, (gs, labels, data) in tqdm(enumerate(train_dl), total=num_batch, desc=f'Epoch[{epoch}] >> ', disable=args.no_tqdm, ncols=60):
                    n_bw += 1
                    data = data.to(device)
                    gs = gs.to(args.device)
                    data = data.to(args.device)
                    labels = labels.to(args.device).long()
                    af=gs.nodes['atom'].data['feat']
                    bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
                    fnf = gs.nodes['func_group'].data['feat']
                    fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                    molf=gs.nodes['molecule'].data['feat']
                    p = float(n_bw + epoch * len(train_dl)) / (50 * len(train_dl))
                    alpha = torch.tensor(2. / (1. + np.exp(-10 * p)) - 1)
                    p_semantic,p_graph_domain,p_image_domain,p_semantic_domain,SUC_loss,p_graph, p_image = semanticEncoder(gs, af, bf,fnf,fef,molf,data,labels,goal="dann")  
                    semantic_loss = criterion(p_semantic, labels)

                 
                    domain_labels_graph = torch.zeros(args.batch_size, dtype=torch.long).to(device)
                    domain_labels_image = torch.ones(args.batch_size, dtype=torch.long).to(device)
             
                    p_graph_domain_loss = criterion(p_graph_domain, domain_labels_graph)
                    p_image_domain_loss = criterion(p_image_domain, domain_labels_image)
                    av, ae = get_dis_info_loss(p_semantic_domain)
                    mydann_loss = ae + p_graph_domain_loss + p_image_domain_loss

              
                    #for lbap_core_ec50_size lbap_core_ec50_scaffold lbap_core_ec50_assay
                    #label_ec50_core label_ec50_scaffold
                    # gen_semantic_loss =  (10 * alpha) * semantic_loss + mydann_loss * 0.1 + SUC_loss


                    # gen_semantic_loss =  semantic_loss + mydann_loss * (1-alpha) + SUC_loss * (10 * alpha)
                    #for lbap_core_ec50_assay lbap_core_ec50_scaffold

                    # gen_semantic_loss =  (args.sl_coe * alpha) * semantic_loss + mydann_loss * args.ml_coe + SUC_loss * args.uni_coe
                    gen_semantic_loss =  (args.sl_coe * alpha) * semantic_loss + mydann_loss * args.ml_coe + SUC_loss * args.uni_coe



                    semantic_optimizer.zero_grad()
                    gen_semantic_loss.backward()
                    semantic_optimizer.step()



                    all_semantic_loss += gen_semantic_loss.item()

                all_semantic_loss /= n_bw
                loss_dis.append((all_semantic_loss, all_dis_image_loss, all_dis_graph_loss))

                val_perf = eval_semanticEncoder_epoch(semanticEncoder, valid_dl, device, goal="pred")
                val_perf = val_perf
                train_perf = eval_semanticEncoder_epoch(semanticEncoder, train_dl, device, goal="pred")
                test_perf = eval_semanticEncoder_epoch(semanticEncoder, test_dl, device, goal="pred")
                best_encoder_weights.append(deepcopy(semanticEncoder.state_dict()))
                for k, v in val_perf.items():
                    if k not in valid_curv:
                        valid_curv[k], train_curv[k], test_curv[k] = [], [], []
                    valid_curv[k].append(val_perf[k])
                    train_curv[k].append(train_perf[k])
                    test_curv[k].append(test_perf[k])
                logger.info('[INFO] epoch: {}, train: {}, valid: {}, test: {}'.format(epoch, train_perf['auc'], val_perf['auc'], test_perf['auc']))

            best_result = {}

            for k, v in valid_curv.items():
                if k == 'auc':
                    v = v[args.drop_early:]
                    train_curv[k] = train_curv[k][args.drop_early:]
                    test_curv[k] = test_curv[k][args.drop_early:]
                    pos = int(np.argmax(v))
                best_result[k] = [pos, v[pos], train_curv[k][pos], test_curv[k][pos]]
                best_encoder_model = best_encoder_weights[pos]
            print("Saving best weights..")
            torch.save(best_encoder_model, semanticEncoder_path)
            original_perf = best_result['auc']
            all_seed_info['train_auc'].append(original_perf[2])
            all_seed_info['val_auc'].append(original_perf[1])
            all_seed_info['test_auc'].append(original_perf[3])
            logger.info('[INFO] best result: {}'.format(original_perf))

        
            with open(dis_loss_path, 'w') as f:
                json.dump({'loss_curve': loss_dis}, f, indent=4)
    #calculate ae av
    results = {}
    for key, values in all_seed_info.items():
        mean_value = statistics.mean(values)
        var_value = statistics.variance(values)
        results[key] = {'mean': mean_value, 'variance': var_value}
    logger.info(results)