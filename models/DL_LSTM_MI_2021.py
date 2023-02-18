import torch
import torch.nn as nn
import torch.nn.functional as F

class CasCnnRnnnet(nn.Module):
    def __init__(self, n_class, n_seg, n_row_img, n_row_col ,n_layers = 2, dt = 0.379) -> None:
        super(CasCnnRnnnet, self).__init__()
        self.hid_size = 300
        if (n_class) == 2:
            n_class = 1
        self.n_classes = n_class
        self.spac_feat = (n_row_img*n_row_col)*3
        self.temp_feat = 10
        self.num_layers = n_layers
        self.n_seg = n_seg

        self.spac = nn.Sequential(
                            nn.Conv2d(1,32,(3,3), padding='same', bias=False),
                            nn.ReLU(),
                            nn.Conv2d(32,64, (3,3), padding='same', bias=False),
                            nn.ReLU(),            
                            nn.Conv2d(64,128,(3,3), padding='same', bias=False),
                            nn.ReLU(),
                            nn.Flatten(start_dim=1),
                            # nn.Flatten(start_dim=0),
                            nn.Linear(128*n_row_img*n_row_col, self.spac_feat)
                        )

        self.cnns = nn.ModuleList([self.spac for _ in range(self.n_seg)])

        self.lstm = nn.LSTM(input_size = self.spac_feat , hidden_size = self.hid_size, 
                num_layers = self.num_layers, batch_first = True, dropout= dt)

        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(n_seg, n_class)

    def forward(self, x):
        # x IMG data: bs x n_seg x step x *IMG
        sp_feat = []
        for i,S in enumerate(self.cnns):
            sp_feat.append(S(x[:,i:i+1,:,:]))
        
        sp_feat = torch.stack(sp_feat)
        sp_feat = torch.transpose(sp_feat,0,1)
        out, h = self.lstm(sp_feat)
        # x = x[:,-1,:]
        u = torch.tanh(self.fc1(out))
        a = torch.softmax(u, dim =2)
        v = torch.sum(a*out, 2)
        x = nn.functional.relu(self.fc2(v))
        return x