import torch
import torch.nn as nn


class ATTNnet(nn.Module):
    def __init__(self, n_class, n_feat, n_layers, dt = 0.379) -> None:
        super(ATTNnet, self).__init__()
        self.hid_size = 500
        self.n_feat = n_feat
        self.num_layers = n_layers
        self.n_classes = n_class
        self.lstm = nn.LSTM(input_size = self.n_feat, hidden_size = self.hid_size, 
                num_layers = self.num_layers, batch_first = True, dropout= dt)
        self.dropout = nn.Dropout(0.37)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, n_class)

    def forward(self, x):
        # x is always batch x Cch x EEGCH x ts  
        # x = x[:,-1,:,:] # chuck out color channel
        # x = torch.transpose(x,1,2) # transpose so 'ts' lstm cells and each cell gets 'EEGCH' features

        # x with STAT batch x EEGch x Segments x features
        x = torch.transpose(x,1,2) # batch x EEGch x Segments x features
        x = torch.flatten(x, start_dim=2)
        bs = x.shape[0]
        # h_0 = torch.zeros(self.num_layers, bs, self.hid_size).to('cuda')
        # c_0 = torch.zeros(self.num_layers, bs, self.hid_size).to('cuda')
        out,h = self.lstm(x) #,(h_0, c_0)) # h : batch x ts x EEGCH   
        # TO use only the last 't'
        out = out[:,-1,:] # batch x EEGCH
        u = torch.tanh(self.fc1(out))
        # a = torch.softmax(u, dim =1)
        v = u*out
        v = self.dropout(v)
        if self.n_classes == 1:
            x = torch.flatten(self.fc2(v.sum(dim=1)))
        else:
            
            x = self.fc2(v)
        # softmax for multiclass classification
        return x