import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        # 输入是两个128特征向量的拼接，所以输入大小是256
        self.fc1 = nn.Linear(128, 256)# 第一个全连接层
        self.fc2 = nn.Linear(256, 128)  # 第二个全连接层
        self.fc3 = nn.Linear(128, 1)    # 第三个全连接层，输出一个值表示判别结果

    def forward(self, vec1, fake="false"):
        # 通过三个全连接层
        x = F.relu(self.fc1(vec1))
        x = F.dropout(x, 0.5)  # 可以添加dropout减少过拟合
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 使用sigmoid来将输出归一化到[0, 1]
        return x

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        # 输入是两个128特征向量的拼接，所以输入大小是256
        self.fake_fc1 = nn.Linear(128, 256)
        self.fc1 = nn.Linear(384+128, 256)
        self.fc2 = nn.Linear(256, 128)  # 第二个全连接层
        self.fc3 = nn.Linear(128, 1)    # 第三个全连接层，输出一个值表示判别结果

    def forward(self, vec1, wt="real"):
        # 通过三个全连接层
        if wt == "fake":
            x = F.relu(self.fake_fc1(vec1))
        else:
            x = F.relu(self.fc1(vec1))
        x = F.dropout(x, 0.5)  # 可以添加dropout减少过拟合
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # 使用sigmoid来将输出归一化到[0, 1]
        return x
