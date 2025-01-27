import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, 1)   

    def forward(self, vec1, fake="false"):
        
        x = F.relu(self.fc1(vec1))
        x = F.dropout(x, 0.5)  
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  
        return x

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
 
        self.fake_fc1 = nn.Linear(128, 256)
        self.fc1 = nn.Linear(384+128, 256)
        self.fc2 = nn.Linear(256, 128) 
        self.fc3 = nn.Linear(128, 1)   

    def forward(self, vec1, wt="real"):

        if wt == "fake":
            x = F.relu(self.fake_fc1(vec1))
        else:
            x = F.relu(self.fc1(vec1))
        x = F.dropout(x, 0.5) 
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
