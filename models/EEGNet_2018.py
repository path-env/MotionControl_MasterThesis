# load_ext tensorboard
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
import torch, torchvision
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from main.extraction.data_extractor import data_container


class EEGnet(nn.Module):
    def __init__(self, classes) -> None:
        super(EEGnet, self).__init__()
        self.F1 = 16
        self.F2 = 16
        self.D = 1
        self.C = 25
        self.T = 151
        self.filterlength = int(np.round(self.T/2))

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
        #     nn.Softmax() )
        # Block1
        self.conv2d1 = nn.Conv2d(1,self.F1,(1,64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthconv2d1 = nn.Conv2d(self.F1,self.D*self.F1,
                                    kernel_size=(self.C,1),groups=self.F1)
        self.batchnorm11 = nn.BatchNorm1d(1)
        self.averagePool1 = nn.AvgPool2d((1,4))
        self.dropout1 = nn.Dropout(0.5)

        #Block2
        self.sepconv2d2 = nn.Conv2d(self.F1,self.F2, kernel_size=(1,1))
        self.batchnorm2 = nn.BatchNorm1d(1)
        self.averagePool2 = nn.AvgPool2d((1,8))
        self.dropout2 = nn.Dropout(0.5)

        # FC
        self.fc = nn.Flatten(1)

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
        x = nn.functional.elu(self.batchnorm2(x))
        x = self.averagePool2(x)
        x = self.dropout2(x)
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=0)
        return x
        
if __name__ =='__main__':
    model =  EEGnet(['right','left']).double()
    # print(model)

    data = np.load('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/feature_extraction/Train_TF_3_3.npz')
    
    train_data = data['arr_0']
    labels = data['arr_1']
    data = data_container(train_data, labels)
    train_loader = torch.utils.data.DataLoader(data, batch_size=1)
    data, label = next(iter(train_loader))

    tb = SummaryWriter()
    tb.add_graph(model, data.double())
    tb.close()