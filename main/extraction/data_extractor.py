from tabnanny import check
import torch
from torch.utils.data import Dataset, DataLoader

# from mne.datasets import eegbci
from data_shredder.physionet import extractPhysioNet
from data.fake_data import extractFakeData

class data_container(Dataset):
    def __init__(self, check_net = False) -> None:
        super().__init__()
        self.channels = 64
        if check_net:
            train, test = extractFakeData()            
            self.train_x = train[0]
            self.train_y = train[1]
            self.feat_shape = self.train_x.shape

            self.test_x = test[0]
            self.test_y = test[1]
        else:
            feat, label = extractPhysioNet(16,8)
            self.__test_train_split__(feat, label)
    
    def __getitem__(self, index):
        return self.x[index,:]
    
    def __len__(self):
        return self.len
    
    def __test_train_split__(self, feat, label):
        self.feat_shape = feat.shape
        self.train_x = torch.tensor(feat)
        self.train_y = torch.tensor(feat)

        self.test_x = torch.tensor(feat)
        self.test_y = torch.tensor(feat)      
    
    # def __getattribute__(self, index):
    #     return self.y[index]

if __name__ == "__main__":
    # dd= eegbci.load_data(1, [4, 10, 14], "/media/mangaldeep/HDD3/DataSets/MNEPhysioNet")
    data = data_container(check_net = True)
    trainloader = DataLoader(dataset=data, batch_size=1)
    # dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())

