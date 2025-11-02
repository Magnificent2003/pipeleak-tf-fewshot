import torch
import torch.nn as nn
import torch.nn.functional as F

class Gatenet(nn.Module):
    def __init__(self, in_dim=5, hid=16, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(hid, 1)
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        return torch.sigmoid(self.fc2(h)).squeeze(-1)
