import sys
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

# from mne.datasets import eegbci
# from data_shredder.physionet import extractPhysioNet
from data.fake_data import extractFakeData

class data_container(Dataset):
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
            device = 'cuda' if torch.cuda.is_available() else "cpu"
            if len(feat.shape) <3:
                print('Input feature shape is wrong')
                sys.exit(1)
            self.x = torch.tensor(feat, device=device, dtype= torch.float16)#.half()#.unsqueeze(1)
            label[label==3] = 0
            self.y = torch.tensor(label, device=device, dtype= torch.float16)#.half()
            
            # feat, label = extractPhysioNet(16,8)
            # self.__test_train_split__(feat, label)
    
    def __getitem__(self, index):
        return self.x[index,:], self.y[index]
    
    def __len__(self):
        return len(self.y)
    
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
    feat = np.random.randn(30,64,227)
    labels = np.concatenate([-np.ones(15), np.ones(15)])
    data = data_container(feat, labels, check_net = False)
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

