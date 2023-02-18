import torch
import torch.nn as nn
from data.params import ATTNnetParams

class ATTNnet(nn.Module):
    def __init__(self, n_class, n_s, cCh, n_chan, n_T, n_layers = 3, dt = 0.379, n_hid = 500, nCfg = ATTNnetParams()) -> None:
        super(ATTNnet, self).__init__()
        self.hid_size = n_hid
        self.n_feat = n_T*cCh
        self.num_layers = n_layers
        if (n_class) == 2:
            n_class = 1
        self.n_classes = n_class
        self.lstm = nn.LSTM(input_size = self.n_feat, hidden_size = self.hid_size, 
                num_layers = self.num_layers, batch_first = True, dropout= dt)
        self.bn2 = nn.BatchNorm2d(n_chan)
        self.bn1 = nn.BatchNorm1d(n_chan)
        self.do = nn.Dropout(dt)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(n_chan, n_class)
        self.W = nn.Parameter(torch.randn(nCfg.train_bs ,n_chan,self.hid_size), requires_grad= True)
        self.b = nn.Parameter(torch.randn(n_chan,1), requires_grad=True)

    def forward(self, x):
        # x is always batch x Cch x EEGCH x ts  
        # x = x[:,-1,:,:] # chuck out color channel
        # x = torch.transpose(x,1,2) # transpose so 'ts' lstm cells and each cell gets 'EEGCH' features

        # x with STAT batch x EEGch x Segments x features
        # 
        x = torch.transpose(x,1,2) # batch x Segments x EEGch x features
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=2)
        # x = nn.functional.relu(self.bn1(x))
        bs = x.shape[0]
        # h_0 = torch.zeros(self.num_layers, bs, self.hid_size).to('cuda')
        # c_0 = torch.zeros(self.num_layers, bs, self.hid_size).to('cuda')
        out,h = self.lstm(x) #,(h_0, c_0)) # h : batch x ts x EEGCH   
        # TO use only the last 't'
        # out = out[:,-1,:] # batch x EEGCH
        u = torch.tanh(self.fc1(self.W*out + self.b))
        a = torch.softmax(u, dim =2)
        v = torch.sum(a*out, 2)
        # v = self.bn(v)
        if self.n_classes == 1:
            x = torch.flatten(self.fc2(v))
        else:
            x = self.fc2(v)
        # softmax for multiclass classification
        return x