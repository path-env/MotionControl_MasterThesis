from main.data_extractor import PhysioNet
import torch
import torchvision
from torch import nn, optim
# import torch.nn.functional as F
torch.manual_seed(1)

class DRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(64,64,bias=True)
        self.L2 = nn.Linear(64,64,bias=True)
        self.L3 = nn.Linear(64,64,bias=True)
        self.L4 = nn.Linear(64,64,bias=True)
        # self.LSTM5 = nn.LSTM(64,64,bias=True)
        # self.LSTM6 = nn.LSTM(64,64, bias=True)
        self.L7 = nn.Linear(64, 3)
        # self.seq = nn.Sequential(nn.Linear(1,1),nn.Sigmoid())
    
    def forward(self, x):
        x = torch.sigmoid(self.L1(x))
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        # x = self.LSTM5(x)
        # x = self.LSTM6(x)
        x = self.L7(x)
        y = torch.argmax (x)
        return y
