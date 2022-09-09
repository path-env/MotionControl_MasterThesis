# load_ext tensorboard
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from models.train_net import train_and_validate
from data.params import BCI3Params, EEGNetParams
from main.extraction.data_extractor import data_container
from models.profile_net import profiler

class EEGnet(nn.Module):
    def __init__(self, n_classes, n_chan, sf) -> None:
        super(EEGnet, self).__init__()
        if (n_classes) == 2:
            n_classes = 1
        self.F1 = 30
        self.F2 = 20
        self.D = 3
        self.C = n_chan
        self.sf = sf
        self.filterlength = int(np.round(self.sf/2))

        '''        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1,self.F1,(1,self.filterlength)),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.F1) )

        # self.depthwiseConv2D = nn.Sequential(
        #     nn.Conv2d(self.F1,self.D*self.F1,kernel_size=(self.C,1),groups=self.F1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.F1),
        #     nn.ELU(),
        #     nn.AvgPool2d((1,4)),
        #     nn.Dropout(0.5) )

        # self.separableConv2D = nn.Sequential(
        #     nn.Conv2d(16,self.F2, kernel_size=(1,self.C)),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.F2),
        #     nn.ELU(),
        #     nn.AvgPool2d((1,8)),
        #     nn.Dropout(0.5),
        #     nn.Flatten(start_dim=0) )

        # self.classifier =nn.Sequential(
        #     nn.Linear(self.F2*(self.T//2), self.F2*(self.T//2)),
        #     nn.Linear(self.F2*(self.T//2), 2),
        #     nn.Softmax() )'''

        # Block1
        self.conv2d1 = nn.Conv2d(1,self.F1,(1,self.filterlength), padding='same', bias=False) # kernel = sfreq//2
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthconv2d1 = nn.Conv2d(self.F1,self.D*self.F1,
                                    kernel_size=(self.C,1),groups=self.F1, bias=False)
        self.batchnorm11 = nn.BatchNorm2d(self.D*self.F1)
        self.averagePool1 = nn.AvgPool2d((1,4))
        self.dropout1 = nn.Dropout(0.5)

        #Block2
        self.sepconv2d1 = nn.Conv2d(self.D*self.F1, self.D*self.F1, (1,14), padding='same', bias=False,
                            groups=self.D*self.F1) # kernel = freq of interest//2
        self.sepconv2d2 = nn.Conv2d(self.D*self.F1,self.F2, kernel_size=(1,1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F2)
        self.averagePool2 = nn.AvgPool2d((1,8))
        self.dropout2 = nn.Dropout(0.5)

        # FC
        self.fc1 = nn.LazyLinear(self.F2)
        # self.fc1 = nn.Linear(self.F2*1*1, self.F2)
        self.fc2 = nn.Linear(self.F2, n_classes)

    def forward(self,x):
        # x = self.conv(x)
        # x = self.depthwiseConv2D(x)
        # x = self.separableConv2D(x)
        # x = self.classifier(x)
        x = nn.functional.relu(self.conv2d1(x))
        x = self.batchnorm1(x)
        x = nn.functional.relu(self.depthconv2d1(x))
        x = nn.functional.elu(self.batchnorm11(x))
        x = self.averagePool1(x)
        x = self.dropout1(x)
        x = nn.functional.relu(self.sepconv2d1(x))
        x = nn.functional.relu(self.sepconv2d2(x))
        x = nn.functional.relu(self.batchnorm2(x))
        x = self.averagePool2(x)
        x = self.dropout2(x)
        x = torch.flatten(x,1)
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # x = nn.functional.softmax(self.fc2(x)).float()
        return x
        
if __name__ =='__main__':
    dCfg = BCI3Params()
    nCfg = EEGNetParams()

    filename = 'Train_locl_ssp_car_ica_TF_EEGnet_CNN_[3]_1.npz'
    datafile = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/feature_extraction/{filename}')
    train_data = datafile['arr_0']
    if len(train_data.shape) <4:
        train_data = np.expand_dims(train_data, axis=1)
    labels = datafile['arr_1']
    # print(labels)
    n_classes = np.unique(labels).shape[0]

    bs, lr = 1, 0.001
    data = data_container(train_data, labels, nCfg)

    _,_,n_chan,_ = data.x.shape
    model =  EEGnet(n_classes,n_chan,dCfg.sfreq)
    # print(model)
    # Loss function
    loss = torch.nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=nCfg.lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    LRscheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    tb_comment = f'batch_size={nCfg.train_bs}, lr={nCfg.lr}'
    tb_info = (f'log/{model._get_name()}/{dCfg.name}/{filename}', tb_comment)
    # tb_info = ('trial','comment')
    # train and test
    train_and_validate(data,model,loss,optimizer, LRscheduler, tb_info, dCfg, nCfg,epochs = 10)
    # profiler(model, optimizer, loss, data, LRscheduler, tb_info[0])
    # torch.save(model,f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()+'_modified.pt'}")