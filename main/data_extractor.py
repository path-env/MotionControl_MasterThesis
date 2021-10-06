from numpy.core.numeric import full
import torch
import mne
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

folderpath = "/media/mangaldeep/HDD3/DataSets/PhysioNet/S001/"


def PhysioNet(folderPath):
    gt = []
    full_data = np.empty((1,64))
    clss = np.array(['T0', 'T1', 'T2'])
    for fileno in range(1,15):
        file = folderPath+ "S001R" + str(fileno).zfill(2) +".edf"
        data = mne.io.read_raw_edf(file)
        raw_data = data.get_data()
        print(raw_data.shape)
        full_data = np.vstack((full_data,raw_data.T))
        # info = data.info
        # channels = data.ch_names
        i = 0
        for t in data.times:
            if i<len(data.annotations.onset) and data.annotations.onset[i] == t:
                tag = data.annotations.description[i]
                i+=1
            gt.append(np.where(tag==clss)[0][0])
        full_data = full_data[1:,:]
    return full_data, gt

class data_container(Dataset):
    def __init__(self) -> None:
        super().__init__()
        full_data , gt = PhysioNet(folderpath)
        self.x = torch.tensor(full_data, dtype=torch.float32)
        self.y = torch.tensor(gt).view(-1,1)
        self.len = self.x.shape[0]
        self.channels = 64

        #Test train split
        train_size = 0.8*self.len
        test_size = self.len - train_size
        # train_set, test
    
    def __getitem__(self, index):
        return self.x[index,:]
    
    def __len__(self):
        return self.len
    
    # def __getattribute__(self, index):
    #     return self.y[index]

if __name__ == "__main__":
    PhysioNet_data = data_container()
    trainloader = DataLoader(dataset=PhysioNet_data, batch_size=1)
    # dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())