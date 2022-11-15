# load_ext tensorboard
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from utils.train_net import train_and_validate
from data.params import BCI3Params, EEGNetParams, PhysionetParams
from main.extraction.data_extractor import DataContainer
from utils.profile_net import profiler

class EEGnet(nn.Module):
    def __init__(self, n_classes, n_chan, n_T, sf, F1 = 10, F2 = 22, D =4, dt = 0.379) -> None:
        super(EEGnet, self).__init__()
        if (n_classes) == 2:
            n_classes = 1
        self.n_classes = n_classes
        self.C = n_chan
        self.T = n_T
        self.sf = sf
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.dt = dt
        self.filterlength = int(np.round(self.sf/2))-1

        """        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1,self.F1,(1,self.filterlength)),
        #     nn.relu(),
        #     nn.BatchNorm1d(self.F1) )

        # self.depthwiseConv2D = nn.Sequential(
        #     nn.Conv2d(self.F1,self.D*self.F1,kernel_size=(self.C,1),groups=self.F1),
        #     nn.relu(),
        #     nn.BatchNorm1d(self.F1),
        #     nn.ELU(),
        #     nn.AvgPool2d((1,4)),
        #     nn.Dropout(0.5) )

        # self.separableConv2D = nn.Sequential(
        #     nn.Conv2d(16,self.F2, kernel_size=(1,self.C)),
        #     nn.relu(),
        #     nn.BatchNorm1d(self.F2),
        #     nn.ELU(),
        #     nn.AvgPool2d((1,8)),
        #     nn.Dropout(0.5),
        #     nn.Flatten(start_dim=0) )

        # self.classifier =nn.Sequential(
        #     nn.Linear(self.F2*(self.T//2), self.F2*(self.T//2)),
        #     nn.Linear(self.F2*(self.T//2), 2),
        #     nn.Softmax() )
        """

        # Block1
        self.dropout = nn.Dropout2d(0.25)
        self.conv2d1 = nn.Conv2d(1, self.F1, kernel_size=(1,self.filterlength), padding='same', bias=False) # kernel = sfreq//2
        # self.conv2d2 = nn.Conv2d(self.F1, self.F1*2, (1,int(self.filterlength/2)), padding='same', bias=False) # kernel = sfreq//2

        # self.conv2d3 = nn.Conv2d(self.F1*2, self.F1*3, (1,int(self.filterlength/3)), padding='same', bias=False) # kernel = sfreq//2

        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthconv2d1 = nn.Conv2d(self.F1,self.D*self.F1, #padding='same',
                                    kernel_size=(self.C,1),groups=self.F1, bias=False)
        self.batchnorm11 = nn.BatchNorm2d(self.D*self.F1)
        self.averagePool1 = nn.AvgPool2d((1,4))
        self.dropout1 = nn.Dropout(self.dt)

        #Block2
        self.sepconv2d1 = nn.Conv2d(self.D*self.F1, self.F2, (1,self.filterlength), padding='same', bias=False)
                            # groups=self.D*self.F1) # kernel = freq of interest//2
        # self.batchnorm2 = nn.BatchNorm2d(self.D*self.F1)
        # self.sepconv2d2= nn.Conv2d(self.D*self.F1, self.F2, kernel_size=(1,1), bias=False, padding='same')
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.averagePool2 = nn.AvgPool2d((1,8))
        self.dropout2 = nn.Dropout(self.dt)

        # FC
        self.fc1 = nn.Linear((self.F2*(self.T//32)), self.F2)
        # self.fc1 = nn.Linear(self.F2*1*1, self.F2)
        self.fc2 = nn.Linear(self.F2, n_classes)

    def forward(self,x):
        # x = self.conv(x)
        # x = self.depthwiseConv2D(x)
        # x = self.separableConv2D(x)
        # x = self.classifier(x)

        # x with RAW/TF batch x cCH x EEGch x ts
        x = self.conv2d1(x) # bs x F1 x EEGch x ts
        # x = self.dropout(x)
        # x = self.conv2d2(x)
        # x = self.dropout(x)
        # x = self.conv2d3(x)
        x = torch.relu(self.batchnorm1(x)) # bs x F1 x EEGch x ts
        x = self.depthconv2d1(x) # bs x F1 x EEGch x ts
        x = torch.relu(self.batchnorm11(x))
        # x = self.dropout1(x)
        x = self.averagePool1(x)
        x = self.dropout1(x)

        # x = nn.functional.relu(self.sepconv2d1(x))
        x = self.sepconv2d1(x)
        # x = torch.relu(self.batchnorm2(x))
        # x = self.sepconv2d2(x)
        x = torch.relu(self.batchnorm3(x))
        # x = self.dropout2(x)
        x = self.averagePool2(x)
        x = self.dropout2(x)

        x = torch.flatten(x,1)
        x = torch.relu(self.fc1(x))
        # x = self.dropout2(x)
        if self.n_classes == 1:
            x = torch.flatten(self.fc2(x))
        else:
            x = self.fc2(x)
        # x = torch.sigmoid(self.fc2(x))
        # x = nn.functional.softmax(self.fc2(x)).float()
        return x
        
if __name__ =='__main__':
    dCfg = BCI3Params()
    nCfg = EEGNetParams()
    nCfg.lr = 0.15448 

    filename = 'BCI3IVa_locl_ssp_car_ica_RAW_[3]_1.npz'
    # filename = 'Physionet_locl_ssp_car_ica_RAW_EEGnet_CNN_[3, 4, 7, 8, 11, 12]_1.npz'
    datafile = np.load(f'data/train/{filename}')
    train_data = datafile['arr_0']
    if len(train_data.shape) <4:
        train_data = np.expand_dims(train_data, axis=1)
    labels = datafile['arr_1']
    # print(labels)
    n_classes = np.unique(labels).shape[0]

    bs, lr = 1, 0.001
    data = DataContainer(train_data, labels, nCfg)

    _,_,n_chan,n_T = data.x.shape
    model =  EEGnet(n_classes,n_chan,n_T ,dCfg.sfreq)
    # print(model)
    # Loss function
    if n_classes == 2:
        loss = torch.nn.BCEWithLogitsLoss() # No sigmoid required at the last layer & in train loop, Binary classification
    else:
        loss = torch.nn.CrossEntropyLoss() # No softmax required at the last layer, Multiclass classification

    # Observe that all parameters are being optimized
    optimizer = optim.ASGD(model.parameters(), lr=nCfg.lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    LRscheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    tb_comment = f'batch_size={nCfg.train_bs}, lr={nCfg.lr}'
    tb_info = (f'runs/{model._get_name()}/{dCfg.name}/{filename}', tb_comment)
    # tb_info = ('trial','comment')
    # train and test
    train_and_validate(data,model,loss,optimizer, LRscheduler, tb_info, dCfg, nCfg,epochs = 100)
    # profiler(model, optimizer, loss, data, LRscheduler, tb_info[0])
    torch.save(model,f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()+'_modified.pt'}")