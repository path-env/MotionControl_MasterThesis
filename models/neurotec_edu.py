import torch
import tensorboardX

import torch.nn as nn
import numpy as np
torch.manual_seed(100)

class Neurotech_net(nn.Module):
    def __init__(self):
        super(Neurotech_net,self).__init__()
        self.L1 = nn.Linear(226,500,bias=True)
        self.L2 = nn.Linear(500,1000,bias=True)
        self.L3 = nn.Linear(1000,100,bias=True)
        self.L4 = nn.Linear(100,10,bias=True)
        self.L5 = nn.Linear(10, 1)
        # self.seq = nn.Sequential(nn.Linear(1,1),nn.Sigmoid())
    
    def forward(self, x):
        x = torch.celu(self.L1(x))
        x = torch.relu(self.L2(x))
        x = torch.relu(self.L3(x))
        x = torch.relu(self.L4(x))
        y = torch.sigmoid(self.L5(x))
        return y


if __name__ =='__main__':
    model = Neurotech_net()
    print(model)