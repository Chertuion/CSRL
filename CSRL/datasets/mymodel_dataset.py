import os.path as osp
import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit.Chem import Draw
import dgl.backend as F
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Batch
import dgl
import sys
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
# import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets.MolGraph_Construction import smiles_to_Molgraph,ATOM_FEATURIZER, BOND_FEATURIZER, SmileToGraph
from concurrent.futures import ThreadPoolExecutor
# import concurrent
# from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 获取当前代码文件绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到sys.path中
sys.path.append(os.path.join(current_dir, ".."))

def process_smiles(smiles):
    return smiles, smiles_to_Molgraph(smiles)

class DILIDataset(InMemoryDataset):

    def __init__(self, root, name, mode, transform=None, pre_transform=None, pre_filter=None):
        """
        init from original DILIDataset dataset in the form of dgl.heterograph.DGLHeteroGraph
        """
        super(DILIDataset, self).__init__(root, transform, pre_transform, pre_filter)
        gt_data_gs, gt_data_ls, data_list = self.process_dataset(root, name, mode)
        self.load_data(gt_data_gs, gt_data_ls, data_list)

    def get_task_pos_weights(labels, masks):
        num_pos = F.sum(labels, dim=0)
        num_indices = F.sum(masks, dim=0)
        task_pos_weights = (num_indices - num_pos) / num_pos
        return task_pos_weights

    def get_data(sub_data):
        gs, ys, ms = [], [], []
        for i in range(len(sub_data)):
            gs.append(sub_data[i][1])
            ys.append(sub_data[i][2])
            ms.append(sub_data[i][3])
        ys = torch.stack(ys)
        ms = torch.stack(ms)
        task_weights = DILIDataset.get_task_pos_weights(ys, ms)
        return gs, ys, ms, task_weights

    def process_dataset(self, root, name, mode):
        if name.lower() in ["lbap_core_ec50_size","lbap_core_ec50_assay","lbap_core_ec50_scaffold","lbap_core_ic50_size","lbap_core_ic50_assay","lbap_core_ic50_scaffold","label_ec50_core","label_ec50_scaffold","label_ki_core","label_ki_scaffold", "bbbp", 'lbap_refined_ec50_assay', 'lbap_general_ec50_assay', 'clintox', 'tox21']:
            data_path = osp.join(root, name + "_" + mode + ".pt")
            if not osp.exists(data_path):
                keys = []
                results = {}
                compound_id_list = []
                smiles_list = []
                labels_list = []
                data_list = []
                data_pyg_list = []
                graph_list = []
                dataset_path = root + "_" + mode + '.csv'
                data = pd.read_csv(dataset_path)
                # data = data[:10]
                label = data['cls_label']

                # 假设 data 是一个 pandas DataFrame
                smiles = data["smiles"].tolist()
                keys = list(range(len(smiles)))
                labels = label[:len(data)]  # 假设 label 是一个列表

                smile_to_graph = SmileToGraph(keys)
                compound_graph_json = smile_to_graph(smiles)

                for i in range(len(keys)):
                    pyg_data = {}
                    pyg_data['input'] = compound_graph_json[keys[i]]
                    data_list.append(pyg_data)

                for index, row in tqdm(data.iterrows(), desc="Transforming smiles", total=len(data)):
                    smiles = row["smiles"]
                    smiles_list.append(smiles)
                    labels_list.append(label[index])
                    results[index] = smiles
                    smiles2graph = smiles_to_Molgraph(smiles)
                    graph_list.append(smiles2graph)

                # gt_data_gs, gt_data_ls, gt_data_mask, gt_data_tw = DILIDataset.get_data(graph_list)
                gt_data_gs = graph_list
                gt_data_ls = labels

                for step, data in tqdm(enumerate(data_list), total=len(data_list), desc="Converting"):
                    if step == 687:
                        print("stop")
                    graph = data['input']
                    if graph is None:
                        continue
                    gs = gt_data_gs
                    ls = gt_data_ls
                    node_num = graph.num_nodes()
                    edge_num = graph.num_edges()
                    edge_index = graph.edges()
                    edge_attr = graph.edata['x']
                    node_attr = graph.ndata['x']

                    smile = smiles_list[keys[step]]
                    # 将SMILES转换为分子
                    mol = Chem.MolFromSmiles(smile)
                    # 计算分子的Morgan指纹（一种常见的分子指纹）
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
                    # 将分子指纹转换为PyTorch张量
                    fp_tensor = torch.Tensor([fp])

                    new_data = Data(edge_index = torch.stack(list(edge_index), dim=0),
                                    node_num = node_num,
                                    edge_num = edge_num,
                                    x = node_attr,
                                    fp = fp_tensor,
                                    smile = smile,
                                    edge_attr=edge_attr)
                    data_pyg_list.append(new_data)
                # for i in range(len(compound_id_list)):
                #     data = {}
                #     data["input"] = graph_list[i]
                #     data["smile"] = smiles_list[i]
                #     data["gt_label"] = labels_list[i]
                #     data_list.append(data)
                # 将它们放入一个字典中
                data_to_save = {
                    'graphs': gt_data_gs,
                    'labels': gt_data_ls,
                    'pygGraphs': data_pyg_list
                }
                torch.save(data_to_save, data_path)
            # 加载数据
            loaded_data = torch.load(data_path, weights_only=False)
            # 从字典中获取数据
            gt_data_gs = loaded_data['graphs']
            gt_data_ls = loaded_data['labels']
            pygGraphs = loaded_data['pygGraphs']
            return gt_data_gs, gt_data_ls, pygGraphs



    def load_data(self, gt_data_gs, gt_data_ls, pygGraphs):
        self.gt_data_gs, self.gt_data_ls, self.pygGraphs = gt_data_gs, gt_data_ls,pygGraphs




class GraphDataset_Classification(Dataset):
    def __init__(self, g_list, y_tensor, g_pyg):
        label_tensor = torch.tensor(y_tensor, dtype=torch.long)
        self.g_list = g_list
        self.y_tensor = label_tensor
        self.g_pyg = g_pyg
        self.len = len(g_list)

    def __getitem__(self, idx):
        return self.g_list[idx], self.y_tensor[idx], self.g_pyg[idx]

    def __len__(self):
        return self.len


class GraphDataLoader_Classification(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader_Classification, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_ys = torch.stack([item[1] for item in batch])
        batched_pyg = Batch.from_data_list([item[2] for item in batch])
        #batched_ws = torch.stack([item[2] for item in batch])
        return (batched_gs, batched_ys, batched_pyg)


if __name__ == "__main__":
    print("test")