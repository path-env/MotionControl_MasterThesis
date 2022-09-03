# load_ext tensorboard
from random import sample
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from models.train_net import train_and_validate
from data.params import BCI3Params
from main.extraction.data_extractor import data_container


class EEGnet(nn.Module):
    def __init__(self, n_classes, n_chan, n_T) -> None:
        super(EEGnet, self).__init__()
        if (n_classes) == 2:
            n_classes = 1
        self.F1 = 30
        self.F2 = 20
        self.D = 3
        self.C = n_chan
        self.T = n_T
        self.filterlength = int(np.round(self.T/4))

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
        self.conv2d1 = nn.Conv2d(1,self.F1,(1,self.filterlength), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthconv2d1 = nn.Conv2d(self.F1,self.D*self.F1,
                                    kernel_size=(self.C,1),groups=self.F1)
        self.batchnorm11 = nn.BatchNorm2d(self.D*self.F1)
        self.averagePool1 = nn.AvgPool2d((1,4))
        self.dropout1 = nn.Dropout(0.25)

        #Block2
        self.sepconv2d2 = nn.Conv2d(self.D*self.F1,self.F2, kernel_size=(1,1))
        self.batchnorm2 = nn.BatchNorm2d(self.F2)
        self.averagePool2 = nn.AvgPool2d((1,8))
        self.dropout2 = nn.Dropout(0.25)

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
        x = nn.functional.relu(self.sepconv2d2(x))
        x = nn.functional.relu(self.batchnorm2(x))
        x = self.averagePool2(x)
        x = self.dropout2(x)
        x = torch.flatten(x,1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x)).float()
        # x = nn.functional.softmax(self.fc2(x)).float()
        return x
        
if __name__ =='__main__':
    filename = 'Train_RAW_[3]_1.npz'
    datafile = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/feature_extraction/{filename}')
    train_data = datafile['arr_0']
    labels = datafile['arr_1']
    labels[labels==3] = 0
    # print(labels)
    n_classes = np.unique(labels).shape[0]

    bs, lr = 3, 0.001
    data = data_container(train_data, labels)
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(0.25 * dataset_size))
    # if shuffle_dataset :
    np.random.seed(41)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Train and validate shuffle split
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(data, batch_size=bs, sampler = train_sampler)
    feat, label = next(iter(train_loader))

    _,_,n_chan,n_T = feat.shape
    model =  EEGnet(n_classes,n_chan,n_T).double()
    # print(model)
    # Loss function
    loss = torch.nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    LRscheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    dCfg = BCI3Params()

    tb_comment = f'batch_size={bs}, lr={lr}'
    tb_info = (f'runs/{model._get_name()}/{dCfg.name}/{filename}', tb_comment)
    # train and test
    train_and_validate(model, train_loader, train_loader, loss,optimizer, LRscheduler, tb_info,epochs = 100)