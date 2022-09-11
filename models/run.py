import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.train_net import train_and_validate, test_net
from data.params import BCI3Params, EEGNetParams, PhysionetParams
from main.extraction.data_extractor import data_container
from models.profile_net import profiler

from models.EEGNet_2018 import EEGnet

if __name__ =='__main__':
    dCfg = PhysionetParams()
    nCfg = EEGNetParams()

    filename = 'Physionet__RAW_[3, 4, 7, 8, 11, 12]_1.npz'
    # filename = 'Physionet_locl_ssp_car_ica_RAW_EEGnet_CNN_[3, 4, 7, 8, 11, 12]_1.npz'

    train = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data/train/{filename}')
    train_x = train['arr_0']
    train_y = train['arr_1']
    
    n_classes = np.unique(train_y).shape[0]
    bs, lr = 1, 0.001
    data = data_container(train_x, train_y, nCfg)

    _,_,n_chan,n_T = data.x.shape
    model =  EEGnet(n_classes, n_chan, n_T, dCfg.sfreq, F1 = 14, F2 = 30, D =4, dt = 0.3665)

    tb_comment = f'batch_size={nCfg.train_bs}, lr={nCfg.lr}'
    tb_info = (f'logs/{model._get_name()}/{dCfg.name}/{filename}', tb_comment)

    tb = SummaryWriter(tb_info[0], filename_suffix= tb_info[1])

    if n_classes == 2:
        loss = torch.nn.BCEWithLogitsLoss() # No sigmoid required at the last layer & in train loop, Binary classification
    else:
        loss = torch.nn.CrossEntropyLoss() # No softmax required at the last layer, Multiclass classification

    optimizer = optim.RAdam(model.parameters(), lr=nCfg.lr)
    LRscheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    _, f1_train, f1_val = train_and_validate(data,model,loss,optimizer, LRscheduler, tb , dCfg, nCfg,epochs = 100)
    # profiler(model, optimizer, loss, data, LRscheduler, tb_info[0])
    if f1_train > 0.7:
        print('Saving the model.....')
        torch.save(model,f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()+'_modified.pt'}")
    
    ## Testing 
    test = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data/test/{filename}')
    test_x = test['arr_0']
    test_y = test['arr_1']
    
    data = data_container(test_x, test_y, nCfg)
    test_loader = DataLoader(data, batch_size=1, pin_memory=True, num_workers= nCfg.num_wrkrs)
    _, f1_test = test_net(model, test_loader, tb, dCfg)
    print(f'Test F1 Score : {f1_test}')