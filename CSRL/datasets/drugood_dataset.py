import os.path as osp
import pickle as pkl
import random

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops, remove_self_loops
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from rdkit.Chem import Draw
from PIL import Image


class   DrugOOD(InMemoryDataset):

    def __init__(self, root, dataset, name, mode, transform=None, pre_transform=None, pre_filter=None):
        """
        init from original DrugOOD dataset in the form of dgl.heterograph.DGLHeteroGraph
        """
        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data(root, dataset, name, mode)

    def load_data(self, root, dataset, name, mode):
        data_path = osp.join(root, name + "_" + mode + ".pt")
        if not osp.exists(data_path):
            data_list = []
            fp_list = []
            image_path_list = []
            image_embedding = []
            image_path = osp.join(root, "images")
            # for data in dataset:
            # 加载预训练的 ViT 模型
            model = models.vit_b_16(pretrained=True)

            # 修改最后一个全连接层以输出 256 维的向量
            model.heads.head = torch.nn.Linear(model.heads.head.in_features, 128)

            # 确保模型处于评估模式
            model.eval()

            # 图片预处理
            transform = Compose([
                Resize((224, 224)),  # 调整图片大小以匹配模型输入
                ToTensor(),         # 将图片转换为 Tensor
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
            ])
            for step, data in tqdm(enumerate(dataset), total=len(dataset), desc="Converting"):
                graph = data['input']
                y = data['gt_label']
                group = data['group']
                smile = dataset.data_infos[step]["smiles"]
                # 将SMILES转换为分子
                mol = Chem.MolFromSmiles(smile)
                # 计算分子的Morgan指纹（一种常见的分子指纹）
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
                # 将分子指纹转换为PyTorch张量
                fp_tensor = torch.Tensor([fp])

                mol_image = Draw.MolToImage(mol, size=(300, 300))

                save_path = osp.join(image_path, str(step) + ".png")
                mol_image.save(save_path)
                image_path_list.append(save_path)



                # 使用 PIL 加载图片
                img = Image.open(save_path)
                # 预处理图片并添加批次维度
                input_tensor = transform(img).unsqueeze(0)
                # 提取特征
                # 使用模型对图片进行编码
                with torch.no_grad():
                    encoded_image = model(input_tensor)
                # image_embedding.append(encoded_image)

                edge_index = graph.edges()
                edge_attr = graph.edata['x']  #.long()
                node_attr = graph.ndata['x']  #.long()
                new_data = Data(edge_index=torch.stack(list(edge_index), dim=0),
                                edge_attr=edge_attr,
                                x=node_attr,
                                y=y,
                                smile = smile,
                                image_embedding = encoded_image,
                                fp=fp_tensor,
                                group=group)
                data_list.append(new_data)
                # fp_list.append(fp_tensor)
            torch.save(self.collate(data_list), data_path)

        self.data, self.slices = torch.load(data_path)
