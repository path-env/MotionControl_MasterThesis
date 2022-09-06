# load_ext tensorboard
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
from matplotlib import cm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

from models.train_net import train_and_validate
from data.params import BCI3Params, EEGNetParams, TFNetParams
from main.extraction.data_extractor import data_container

import torch, torchvision
from torchvision import transforms, models
from main.extraction.data_extractor import data_container


def TFnet(n_classes, n_chan, n_T):
    if (n_classes) == 2:
        n_classes = 1
    alexnet = models.alexnet(weights='DEFAULT')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # alexnet.eval();
    alexnet

    # # Freesze model parameters
    # for param in alexnet.parameters():
    #     param.requires_grad = False

    # change final layer of alexnet model
    alexnet.classifier[6] = nn.Linear(4096,n_classes)
    alexnet.classifier.add_module('7',nn.Sigmoid())
    return alexnet

def matrix2Image(train_data):
        # self.train_data = scaler.fit_transform(self.train_data)
        cmap = np.uint8(cm.gist_earth(train_data)*255)[:,:,:,:3]
        cmap = np.swapaxes(cmap, 1,3)
        cmap = np.swapaxes(cmap, 2,3)

        preprocess = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(227),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_data = torch.from_numpy(cmap.astype(np.float32))
        # input_image = Image.fromarray(cmap)
        input_tensor = preprocess(train_data)
        input_tensor-=input_tensor.min()
        input_tensor/=input_tensor.max()
        train_data = input_tensor #.unsqueeze(0) # create a mini-batch as expected by the model
    
        # # Plot the proceessed image
        # import cv2
        # test = self.train_data[0,:,:,:].swap_axes(0,2).numpy()
        # cv2.imshow('Test', test)
        # cv2.waitKey(0)
        return train_data

if __name__ =='__main__':
    dCfg = BCI3Params()
    nCfg = TFNetParams()

    filename = 'Train_locl_ssp_car_ica_TF_EEGnet_CNN_[3]_1.npz'
    datafile = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/feature_extraction/{filename}')
    train_data = datafile['arr_0']
    train_data = matrix2Image(train_data)
    
    labels = datafile['arr_1']

    # print(labels)
    n_classes = np.unique(labels).shape[0]

    bs, lr = 1, 0.001
    data = data_container(train_data, labels)

    # _,n_chan,n_T = train_loader.dataset.__getitem__(0)[0].shape
    model =  TFnet(n_classes,None,None).float()
    # print(model)
    # Loss function
    loss = torch.nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 7 epochs
    LRscheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    

    tb_comment = f'batch_size={bs}, lr={lr}'
    tb_info = (f'runs/{model._get_name()}/{dCfg.name}/{filename}', tb_comment)
    # tb_info = ('trial','comment')
    # train and test
    train_and_validate(data,model,loss,optimizer, LRscheduler, tb_info, dCfg, nCfg,epochs = 100)
    torch.save(model,f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()+'_modified.pt'}")