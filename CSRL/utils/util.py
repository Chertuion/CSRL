import torch
import numpy as np
import torch.nn.functional as F
import random
from texttable import Texttable
import torch
import numpy as np
# import dgl
# from drugood.utils import smile2graph
from functools import reduce
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix, average_precision_score, matthews_corrcoef,accuracy_score
from tqdm import tqdm
from models.losses import KLDist, MeanLoss, DeviationLoss, discrete_gaussian
# from CSRL.models.losses import discrete_gaussian

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_partition(len_dataset, device, seed, p=[0.5,0.5]):
    '''
        group the graph randomly

        Input:   len_dataset   -> [int]
                 the number of data to be groupped
                 
                 device        -> [torch.device]
                
                 p             -> [list]
                 probabilities of the random assignment for each group
        Output: 
                 vec           -> [torch.LongTensor]
                 group assignment for each data
    '''
    assert abs(np.sum(p) - 1) < 1e-4
    
    vec = torch.tensor([]).to(device)
    for idx, idx_p in enumerate(p):
        vec = torch.cat([vec, torch.ones(int(len_dataset * idx_p)).to(device) * idx])
        
    vec = torch.cat([vec, torch.ones(len_dataset - len(vec)).to(device) * idx])
    perm = torch.randperm(len_dataset, generator=torch.Generator().manual_seed(seed))
    return vec.long()[perm]


def args_print(args, logger):
    print('\n')
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())


def PrintGraph(graph):

    if graph.name:
        print("Name: %s" % graph.name)
    print("# Nodes:%6d      | # Edges:%6d |  Class: %2d" \
          % (graph.num_nodes, graph.num_edges, graph.y))

    print("# Node features: %3d| # Edge feature(s): %3d" \
          % (graph.num_node_features, graph.num_edge_features))

def evaluate(pred, gt, metric='auc'):
    if isinstance(metric, str):
        metric = [metric]
    allowed_metric = ['auc', 'accuracy']
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
        else:
            pred = pred.argmax(dim=-1)
            total, correct = len(pred), torch.sum(pred.long() == gt.long())
            result[M] = (correct / total).item()
    return result


def gamma_positional_encoding(p, L):
    """ Generate gamma positional encoding for a given position p and maximum frequency L.

    Args:
    p: Position in the sequence (scalar).
    L: Maximum frequency power (scalar).

    Returns:
    A numpy array containing the gamma positional encoding for the position p.
    """
    # Initialize the gamma positional encoding vector
    gamma_pe = []
    p = p.cpu().numpy()
    # Calculate the gamma positional encoding values
    for i in range(L):
        gamma_pe.append(np.sin(2 ** i * np.pi * p))
        gamma_pe.append(np.cos(2 ** i * np.pi * p))

    return np.array(gamma_pe)

#使用gamma将特征扩展至高维度
def gamma_positional_encoding_vector(input_vector, L):
    """ Generate gamma positional encoding for an input vector of positions.

    Args:
    input_vector: A numpy array of positions.
    L: Maximum frequency power (scalar).

    Returns:
    A numpy array of shape (len(input_vector), 2*L) containing the gamma positional encodings.
    """
    # Initialize the gamma positional encoding matrix
    gamma_pe_matrix = np.zeros((len(input_vector), 2*L))

    # Calculate the gamma positional encoding for each position in the input vector
    for idx, p in enumerate(input_vector):
        gamma_pe_matrix[idx, :] = gamma_positional_encoding(p, L)

    return gamma_pe_matrix

def batch_gamma_positional_encoding(batch_input_vector, L):
    """ Generate gamma positional encoding for a batch of input vectors.

    Args:
    batch_input_vector: A 2D numpy array of shape (batch_size, feature_dim) where each row is a position vector.
    L: Maximum frequency power (scalar).

    Returns:
    A numpy array of shape (batch_size, feature_dim, 2*L) containing the gamma positional encodings for the batch.
    """
    # Initialize the gamma positional encoding 3D array
    batch_size, feature_dim = batch_input_vector.shape
    gamma_pe_batch = np.zeros((batch_size, feature_dim, 2*L))

    # Calculate the gamma positional encoding for each position vector in the batch
    for batch_idx, input_vector in enumerate(batch_input_vector):
        for feature_idx, p in enumerate(input_vector):
            gamma_pe_batch[batch_idx, feature_idx, :] = torch.Tensor(gamma_positional_encoding(p, L))

    return gamma_pe_batch


def get_prior(num_domain, dtype='uniform'):
    assert dtype in ['uniform', 'gaussian'], 'Invalid distribution type'
    if dtype == 'uniform':
        prior = torch.ones(num_domain) / num_domain
    else:
        prior = discrete_gaussian(num_domain)
    return prior

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
    for data in (tqdm(loader) if verbose else loader):
        with torch.no_grad():
            data = data.to(device)
            result = model(data)
            result = torch.softmax(result, dim=-1)
            result_all.append(result.detach().cpu())
            gt_all.append(data.y.cpu())
    result_all = torch.cat(result_all, dim=0)
    gt_all = torch.cat(gt_all, dim=0)
    return evaluate(pred=result_all, gt=gt_all, metric=['auc', 'acc', 'mcc', 'f1', 'recall', 'pr_auc'])




def UniSMMConLoss(self, feature_a, feature_b, predict_a, predict_b, labels, temperature=0.5):
        feature_a_ = feature_a.detach()
        feature_b_ = feature_b.detach()

        a_pre = predict_a.eq(labels)  # a True or not
        a_pre_ = ~a_pre
        b_pre = predict_b.eq(labels)  # b True or not
        b_pre_ = ~b_pre

        a_b_pre = torch.gt(a_pre | b_pre, 0)  # For mask ((P: TT, nP: TF & FT)=T, (N: FF)=F)
        a_b_pre_ = torch.gt(a_pre & b_pre, 0) # For computing nP, ((P: TT)=T, (nP: TF & FT, N: FF)=F)

        a_ = a_pre_ | a_b_pre_  # For locating nP not gradient of a
        b_ = b_pre_ | a_b_pre_  # For locating nP not gradient of b

        if True not in a_b_pre:
            a_b_pre = ~a_b_pre
            a_ = ~a_
            b_ = ~b_
        mask = a_b_pre.float()

        feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
        for i in range(feature_a.shape[0]):
            if not a_[i]:
                feature_a_f[i] = feature_a_[i].clone()
        feature_a_f = torch.stack(feature_a_f)

        feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] # feature_b  # [[0,1]])
        for i in range(feature_b.shape[0]):
            if not b_[i]:
                feature_b_f[i] = feature_b_[i].clone()
        feature_b_f = torch.stack(feature_b_f)

        # compute logits
        logits = torch.div(torch.matmul(feature_a_f, feature_b_f.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)

        # compute log_prob
        exp_logits = torch.exp(logits-logits_max.detach())[0]
        mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum() + 1e-8)# + 1e-6

        return mean_log_pos


def mmc_2(self, f0, f1, p0, p1, l, T):
    f0 = f0 / f0.norm(dim=-1, keepdim=True)
    f1 = f1 / f1.norm(dim=-1, keepdim=True)
    p0 = torch.argmax(F.softmax(p0, dim=1), dim=1)
    p1 = torch.argmax(F.softmax(p1, dim=1), dim=1)

    return UniSMMConLoss(self, f0, f1, p0, p1, l, T)