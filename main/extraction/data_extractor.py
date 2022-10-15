import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
import numpy as np
from sklearn.model_selection import train_test_split

from data.params import EEGNetParams

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

# from mne.datasets import eegbci
# from data_shredder.physionet import extractPhysioNet
from data.fake_data import extractFakeData

class DataContainer(Dataset):
    def __init__(self, feat, label, check_net = False) -> None:
        super().__init__()
        self.channels = 64
        if check_net:
            train, test = extractFakeData()            
            self.x = train[0]
            self.y = train[1]

            self.test_x = test[0]
            self.test_y = test[1]
        else:
            if len(feat.shape) <3:
                print('Input feature shape is wrong')
                sys.exit(1)
            self.x = torch.tensor(feat)#.unsqueeze(1)
            if not label.__contains__(0):
                label = label-1
            self.y = torch.tensor(label)
    
    def __getitem__(self, index):
        return self.x[index,:], self.y[index]
    
    def __getall__(self):
        return self.x, self.y
    
    def __len__(self):
        return len(self.y)

class DataDispenser():
    def __init__(self, dataset, nCfg) -> None:
        self.nCfg = nCfg
        self.dataset = dataset
        self.train_idx = []
        self.val_idx = []

    def get_loaders(self, shuffle = True):
        # ds_size = len(data)
        # idx = list(range(ds_size))
        # if shuffle :
        #     # np.random.seed(random_seed)
        #     np.random.shuffle(idx)

        # val_split = int(np.floor(self.nCfg.val_split * ds_size))
        # test_split = int(np.floor(self.nCfg.test_split  * ds_size))
        # val_idx, test_idx,train_idx = idx[:val_split], idx[val_split:val_split+test_split], idx[val_split+test_split:]

        # val_split = int(np.floor(self.nCfg.val_split * ds_size))
        test_split = int(np.floor(self.nCfg.test_split  * len(self.val_idx)))
        val_idx, test_idx = self.val_idx[test_split:], self.val_idx[:test_split]
        # val_idx, test_idx,train_idx = idx[:val_split], idx[val_split:val_split+test_split], idx[val_split+test_split:]

        # Train and validate shuffle split
        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(self.dataset, batch_size= self.nCfg.train_bs, sampler=train_sampler, pin_memory=True, num_workers= self.nCfg.num_wrkrs)
        validation_loader = DataLoader(self.dataset, batch_size= self.nCfg.val_bs, sampler=valid_sampler, pin_memory=True, num_workers= self.nCfg.num_wrkrs)
        test_loader = DataLoader(self.dataset, batch_size=1, sampler=test_sampler, pin_memory=True, num_workers= self.nCfg.num_wrkrs)
        return train_loader, validation_loader, test_loader

    def __test_train_split__(self, feat, label):
        # Train - Test split
        self.train_x, self.test_x,self.train_y,self.test_y =train_test_split(feat, label, test_size= 0.25, stratify= label,random_state= 42)
        # print(random_split(feat,[3, 7], generator=torch.Generator().manual_seed(42)))
        # self.feat_shape = feat.shape
        # self.train_x = torch.tensor(feat)
        # self.train_y = torch.tensor(feat)

        # self.test_x = torch.tensor(feat)
        # self.test_y = torch.tensor(feat)      
    
if __name__ == "__main__":
    # dd= eegbci.load_data(1, [4, 10, 14], "/media/mangaldeep/HDD3/DataSets/MNEPhysioNet")
    nCfg = EEGNetParams()
    feat = np.random.randn(30,64,227)
    labels = np.concatenate([-np.ones(15), np.ones(15)])
    data = DataContainer(feat, labels, nCfg, check_net = False)
    tr, val,test = data.get_loaders(data)
    validation_split = 0.25
    shuffle_dataset = True
    random_seed = 42
    batch_size = 5
    # Input data & labels
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                    sampler=valid_sampler)
    # dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())

