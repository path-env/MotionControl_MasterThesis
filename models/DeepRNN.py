import torch
from torch import nn
torch.manual_seed(1)

class SEQnet(nn.Module):
    def __init__(self, n_class, n_chan, n_T, sf, dt = 0.379) -> None:
        super().__init__()
        # inputs in shape (batch x Colorchn x n_ch x n_TS)
        hid_size = 35
        self.rnn = nn.LSTM(n_chan, hid_size, num_layers = 2, batch_first = True, dropout = dt)
        self.fc1 = nn.Linear(hid_size,100)
        self.fc2 = nn.Linear(100, n_class)
    
    def forward(self, x):
        x = x[:,-1,:,:]
        x = torch.transpose(x,1,2)
        x,_ = self.rnn(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return x
