import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, RepeatedKFold
from torch.utils.tensorboard import SummaryWriter
from models.DL_LSTM_MI_2021 import CasCnnRnnnet

from utils.train_net import train_and_validate, test_net
from data.params import ATTNnetParams, BCI3Params, EEGNetParams, OCIParams, PhysionetParams, CasCnnRnnnetParams
from main.extraction.data_extractor import DataDispenser, DataContainer
from utils.profile_net import profiler

from models.EEGNet_2018 import EEGnet
from models.DeepRNN import SEQnet
from models.DLSTM_MI_2019 import ATTNnet

import brainflow as bf
import matplotlib.pyplot as plt

if __name__ =='__main__':
    dCfg = OCIParams()
    # dCfg = OCIParams()
    nCfg = EEGNetParams()

    # filename = 'Physionet_16locl_ssp_car_ica_RAW_[3, 4, 7, 8, 11, 12]_1.npz' # ep x cCH x EEGch x Ts
    # filename = 'Physionet_16locl_STAT_[3, 4, 7, 8, 11, 12]_1.npz'# ep x EEGch x n_seg x n_stats
    # filename = 'Physionet_16locl_IMG_[3, 4, 7, 8, 11, 12]_1.npz' # ep x (n_seg.step_size) x n_row_img x n_col_img
    # filename = 'OBCI_3cls_EEGnet_onlinetrn.npz' # ep x cCH x EEGch x Ts
    filename = 'Physionet_locl_ssp_car_ica_RAW_[3, 4, 7, 8, 11, 12]_1.npz'

    filename = 'OCIParams_locl_ssp_car_ica_RAW_[3, 4, 7, 8, 11, 12]_1.npz'
    filename = 'OCIParams_locl_ssp_car_ica_TF_LDA_ML.npz'

    train = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data/train/{filename}', allow_pickle=True)
    train_x = np.float64(train['arr_0'])
    train_y = train['arr_1']

    
    # for i in range(train_x.shape[0]):
    #     # train_x[i,:,:,:] = (train_x[i,:,:,:] - train_x[i,:,:,:].min()) / (train_x[i,:,:,:].max() - train_x[i,:,:,:].min())
    #     for j in range(16):
    #         bf.DataFilter.perform_bandpass(train_x[i,0,j,:], 80, 1, 49,4, 1, ripple=0.5)
    #         # train_x[i,0,j,:]-=np.mean(train_x[i,0,j,:])
    #         # train_x[i,0,j,:] = train_x[i,0,j,:]/np.std(train_x[i,0,j,:]) 
    #     # train_x[i,:,:,:] = train_x[i,:,:,:]/np.std(train_x[i,:,:,:])    
    #     train_x[i,:,:,:] = (train_x[i,:,:,:] - train_x[i,:,:,:].min()) / (train_x[i,:,:,:].max() - train_x[i,:,:,:].min())

    # train_x = train_x[:,:,:,200:] 
    n_classes = len(set(train_y))
    bs, lr = 1, 0.001
    crsval = ShuffleSplit(n_splits=1, test_size = nCfg.val_split, random_state=42)
    datacont = DataContainer(train_x, train_y)
    data = DataDispenser(datacont, nCfg)
    traincrossvalacc, valcrossvalacc = [],[]
    traincrossvalf1, valcrossvalf1 = [],[]
    testcrossval = []
    for fold, (data.train_idx, data.val_idx) in enumerate(crsval.split(train_x)):
        print(f'   ############### Fold - {fold} ######################')
        n_s,cCh,n_chan,n_T = datacont.x.shape
        # model = CasCnnRnnnet(n_classes, n_seg = cCh, n_row_img = n_chan, n_row_col= n_T, n_layers = 3, dt = 0.55)
        # model = ATTNnet(n_classes, cCh*n_T, cCh, n_layers = 3, dt = 0.55)
        model =  EEGnet(n_classes = n_classes, n_chan = n_chan, n_T = n_T,
                sf = dCfg.sfreq, F1 = 24, F2 = 16, D =4, dt = 0.5)

        tb_comment = f'batch_size={nCfg.train_bs}, lr={nCfg.lr}'
        tb_info = (f'logs/{model._get_name()}/{dCfg.name}/{filename}', tb_comment)

        tb = SummaryWriter(tb_info[0], filename_suffix= tb_info[1])

        if n_classes == 2:
            loss = torch.nn.BCEWithLogitsLoss() # No sigmoid required at the last layer & in train loop, Binary classification
        else:
            loss = torch.nn.CrossEntropyLoss() # No softmax required at the last layer, Multiclass classification

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        LRscheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        acc_train, acc_val, f1_train, f1_val = train_and_validate(data,model,loss,optimizer, LRscheduler, tb , 
                                                                    dCfg, nCfg,epochs = 100)
        traincrossvalacc.append(acc_train), valcrossvalacc.append(acc_val)
        traincrossvalf1.append(f1_train), valcrossvalf1.append(f1_val)
        # profiler(model, optimizer, loss, data, LRscheduler, tb_info[0])
        if f1_val > 0.7:
            print('Saving the model.....')
            # torch.save(model,f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()+'_modified.pt'}")
            model_scripted = torch.jit.script(model) # Export to TorchScript
            model_scripted.save(f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()}_md_script.pt") 

        # Testing 
        test = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data/test/{filename}')
        test_x = test['arr_0']
        test_y = test['arr_1']
        
        datacont = DataContainer(test_x, test_y)
        test_loader = DataLoader(datacont, batch_size=1, pin_memory=True, num_workers= nCfg.num_wrkrs)
        _, f1_test = test_net(model, test_loader, tb, dCfg)
        print(f'Test F1 Score : {f1_test}\n')
        testcrossval.append(f1_test)
    
    print(f"Cross Validation Results:")
    print(f"Training  : Accuracy : {np.mean(traincrossvalacc)}, F1 Score : {np.mean(traincrossvalf1)}")
    print(f"Validation: Accuracy : {np.mean(valcrossvalacc)}, F1 Score : {np.mean(valcrossvalf1)}")
    print(f"Tests: {np.mean(testcrossval)}")